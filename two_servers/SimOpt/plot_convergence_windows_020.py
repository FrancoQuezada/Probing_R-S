"""Export all five-instance convergence plots for a stored iteration window."""
import argparse
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from analyze_i0252_statistical_019 import COLORS, moments, save_figure
from i0252_statistical_storage_019 import (
    DEFAULT_OUTPUT, METHODS, LABELS, atomic_json, config, load_complete, sha,
)
from probing_statistical_instances_020 import BASE_OUTPUT, INSTANCES
from plot_individual_methods_020 import METRICS

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ALL_INSTANCES = ("I0252", *INSTANCES)


def load_window(output, instance, start, end):
    cfg = config(output)
    if cfg["science"]["scenario"] != instance:
        raise ValueError(f"Wrong instance: {output}")
    if not 0 <= start < end <= cfg["science"]["max_iters"]:
        raise ValueError(f"Invalid iteration window for {instance}")
    results = load_complete(output, cfg)
    counts = {m: sum(meta["method"] == m for _, meta, _, _ in results) for m in METHODS}
    analysis = json.loads((output / "summaries/analysis_manifest.json").read_text())
    if analysis["fingerprint"] != cfg["fingerprint"] or analysis["completed"] != counts:
        raise ValueError(f"Stale statistics for {instance}; run analysis first")
    source = output / "summaries/iteration_statistics.csv"
    stats = pd.read_csv(source)
    window = stats[stats.iteration.between(start, end) & stats.metric.isin(METRICS)].copy()
    traces = pd.DataFrame([row for _, _, rows, _ in results for row in rows])
    for column in ("iteration", "replication", *METRICS):
        traces[column] = pd.to_numeric(traces[column])
    traces = traces[traces.iteration.between(start, end)]
    verified = 0
    for method in METHODS:
        if counts[method] < 2:
            raise ValueError(f"At least two replications needed: {instance}/{method}")
        raw = traces[traces.method == method].groupby("iteration", sort=True)
        for metric in METRICS:
            group = window[(window.method == method) & (window.metric == metric)].sort_values("iteration")
            if group.iteration.tolist() != list(range(start, end + 1)):
                raise ValueError(f"Missing/duplicate statistics: {instance}/{method}/{metric}")
            for row in group.itertuples(index=False):
                values = raw.get_group(row.iteration)[metric]
                expected = moments(values)
                if len(values) != counts[method] or row.n != counts[method]:
                    raise ValueError(f"Replication count mismatch: {instance}/{method}/{metric}")
                for key, value in expected.items():
                    if not np.isclose(getattr(row, key), value, rtol=1e-11, atol=1e-11):
                        raise ValueError(f"Stored {key} differs from raw results: {instance}/{method}/{metric}/{row.iteration}")
                verified += 1
    if len(window) != verified:
        raise ValueError(f"Unexpected statistics rows for {instance}")
    return dict(instance=instance, output=output, stats=window, traces=traces,
                counts=counts, verified=verified, source=source,
                fingerprint=cfg["fingerprint"])


def limits_for(data, metric):
    group = data["stats"][data["stats"].metric == metric]
    values = group[["mean", "ci95_low", "ci95_high"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite mean or confidence interval")
    lower, upper = float(values.min()), float(values.max())
    margin = max(.06 * (upper - lower), .001)
    limits = (lower - margin, upper + margin)
    if metric in ("z3", "z4"):
        limits = (max(0., limits[0]), min(1., limits[1]))
    return limits


def convergence(ax, data, metric, methods, start, end):
    for method in methods:
        group = data["stats"][(data["stats"].method == method) & (data["stats"].metric == metric)].sort_values("iteration")
        x, mean, low, high = [group[k].to_numpy(dtype=float) for k in ("iteration", "mean", "ci95_low", "ci95_high")]
        ax.plot(x, mean, color=COLORS[method], lw=1.9,
                label=f"{LABELS[method]}, R={data['counts'][method]}")
        ax.fill_between(x, low, high, color=COLORS[method], alpha=.18, linewidth=0)
    ax.set(xlabel="Iteration", ylabel=METRICS[metric][1], xlim=(start, end), ylim=limits_for(data, metric))
    if (start, end) == (2, 25):
        ax.set_xticks([2, 5, 10, 15, 20, 25])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best", fontsize=8.5)


def export_instance(data, start, end):
    output, instance = data["output"], data["instance"]
    suffix = f"_iter{start}_{end}"
    files = []

    def save(stem, figure):
        save_figure(output, stem, figure)
        files.extend(f"{stem}.{fmt}" for fmt in ("png", "pdf"))

    for metric, (stem, _) in METRICS.items():
        for method in (None, *METHODS):
            fig, ax = plt.subplots(figsize=(9.5, 5.8))
            convergence(ax, data, metric, METHODS if method is None else (method,), start, end)
            name = "All methods" if method is None else LABELS[method]
            ax.set_title(f"{instance} · {name} · iterations {start}–{end}\nReplication means and pointwise 95% t confidence intervals")
            fig.tight_layout()
            selection = "all_methods" if method is None else method
            folder = "" if method is None else f"{method}/"
            save(f"{folder}{stem}_{instance}_{selection}{suffix}", fig)

    # In the trajectory figure, endpoints are positions at the window's end,
    # not the independently evaluated decisions at iteration 100.
    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    surface_path = output / f"config/{instance}_reference_surface.csv"
    if surface_path.exists():
        surface = pd.read_csv(surface_path).pivot(index="z4", columns="z3", values="objective")
        contour = ax.contourf(surface.columns.to_numpy(), surface.index.to_numpy(), surface.to_numpy(),
                              levels=18, cmap="Greys", alpha=.55)
        fig.colorbar(contour, ax=ax, label=f"Archived {instance} direct-CRN reference objective", shrink=.83)
    for method in METHODS:
        means = data["stats"][data["stats"].method == method].pivot(index="iteration", columns="metric", values="mean").sort_index()
        ends = data["traces"][(data["traces"].method == method) & (data["traces"].iteration == end)]
        ax.plot(means.z3, means.z4, color=COLORS[method], lw=2.2,
                label=f"{LABELS[method]}, R={data['counts'][method]}")
        ax.scatter(ends.z3, ends.z4, s=15, color=COLORS[method], alpha=.35, linewidth=0)
        ax.scatter(means.z3.iloc[0], means.z4.iloc[0], marker="o", s=65,
                   facecolors="white", edgecolors=COLORS[method], linewidth=1.5, zorder=6)
        ax.scatter(means.z3.iloc[-1], means.z4.iloc[-1], marker="X", s=115,
                   color=COLORS[method], edgecolor="white", linewidth=.7, zorder=7)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([], [], color=".3", marker="o", markerfacecolor="white", linestyle="None", label=f"Mean at iteration {start}"),
        Line2D([], [], color=".3", marker="X", linestyle="None", label=f"Mean at iteration {end}"),
        Line2D([], [], color=".5", marker=".", linestyle="None", label=f"Individual positions at iteration {end}"),
    ])
    ax.legend(handles=handles, loc="best", fontsize=7.8)
    ax.set(xlabel="$z_3$", ylabel="$z_4$", xlim=(0, 1), ylim=(0, 1), aspect="equal",
           title=f"{instance} · mean trajectories · iterations {start}–{end}")
    fig.tight_layout()
    save(f"z3_z4_trajectories_{instance}_all_methods{suffix}", fig)

    manifest = dict(instance=instance, data_directory=str(output),
                    fingerprint=data["fingerprint"], replications=data["counts"],
                    iteration_start=start, iteration_end=end,
                    statistics_source=str(data["source"]), statistics_sha256=sha(data["source"]),
                    stored_statistics_checked_against_raw=data["verified"],
                    confidence_intervals="Unchanged stored pointwise 95% t confidence intervals for replication means",
                    vertical_axes="Fit all methods' means and CIs in this window; shared across individual and combined figures; z bounded by [0,1]",
                    vertical_limits={metric: limits_for(data, metric) for metric in METRICS},
                    trajectory_endpoints=f"Observed per-replication positions at iteration {end}; no joint confidence region",
                    trajectory_surface_sha256=sha(surface_path) if surface_path.exists() else None,
                    new_simulations=0, figures=len(files)//2, script_sha256=sha(Path(__file__)),
                    files={name: sha(output / "plots" / name) for name in files})
    atomic_json(output / "plots" / f"all_plots{suffix}_manifest.json", manifest)
    print(f"{instance}: {len(files)//2} figures (PNG/PDF), iterations {start}–{end}", flush=True)
    return manifest


def export_comparison(base, datasets, start, end):
    files = []
    for metric, (stem, _) in METRICS.items():
        fig, axes = plt.subplots(3, 2, figsize=(14, 14.2))
        for ax, instance in zip(axes.flat, ALL_INSTANCES):
            convergence(ax, datasets[instance], metric, METHODS, start, end)
            ax.set_title(instance)
        axes.flat[-1].set_axis_off()
        axes.flat[-1].text(.05, .7,
            f"Iterations {start}–{end}, inclusive\n\nMeans and pointwise 95% t confidence intervals\n\nOriginal objective scales; no normalization\nAxes fit each instance's selected window",
            transform=axes.flat[-1].transAxes, fontsize=12, va="top", linespacing=1.7)
        fig.suptitle(f"All five instances · iterations {start}–{end}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .97))
        name = f"{stem}_comparison_all_instances_iter{start}_{end}"
        save_figure(base, name, fig)
        files.extend(f"plots/{name}.{fmt}" for fmt in ("png", "pdf"))
    print("All-instance comparison: 3 figures (PNG/PDF)", flush=True)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--i0252-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=25)
    args = parser.parse_args()
    base, original = args.base.resolve(), args.i0252_output.resolve()
    start, end = args.start, args.end
    suffix = f"_iter{start}_{end}"
    outputs = {instance: original if instance == "I0252" else base / instance for instance in ALL_INSTANCES}
    # Record pre-existing figures/statistics, excluding this export's own files
    # on a repeat invocation. This command never writes numerical results.
    preserved = {}
    for output in (*outputs.values(), base / "I0252", base):
        for folder in ("plots", "summaries", "config"):
            for path in sorted((output / folder).rglob("*")):
                if path.is_file() and not (suffix in path.name and
                        ("all_methods" in path.name or "all_instances" in path.name or
                         any(f"_{m}{suffix}" in path.name for m in METHODS) or
                         path.name.startswith(("all_plots", "convergence_window")))):
                    preserved[str(path)] = sha(path)
    atomic_json(base / "validation" / f"convergence_window{suffix}_preserved_before.json", preserved)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True,
                         "grid.alpha": .18, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    datasets, manifests, exported = {}, {}, []
    for instance, output in outputs.items():
        data = load_window(output, instance, start, end)
        manifest = export_instance(data, start, end)
        datasets[instance], manifests[instance] = data, manifest
        destination = base / instance
        if instance == "I0252":
            for name in (*manifest["files"], f"all_plots{suffix}_manifest.json"):
                target = destination / "plots" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output / "plots" / name, target)
                if sha(target) != sha(output / "plots" / name):
                    raise ValueError(f"I0252 copy mismatch: {name}")
        exported.extend(f"{instance}/plots/{name}" for name in manifest["files"])
        exported.append(f"{instance}/plots/all_plots{suffix}_manifest.json")
    exported.extend(export_comparison(base, datasets, start, end))
    for name, expected in preserved.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Pre-existing file changed: {name}")
    images = [name for name in exported if Path(name).suffix in (".png", ".pdf")]
    verification = dict(passed=True, instances=list(ALL_INSTANCES), methods=list(METHODS),
                        iteration_start=start, iteration_end=end, iteration_count=end-start+1,
                        replications={key: value["counts"] for key, value in datasets.items()},
                        figures=len(images)//2, exported_image_files=len(images),
                        stored_statistics_checked_against_raw=sum(data["verified"] for data in datasets.values()),
                        preexisting_figures_statistics_configs_unchanged=len(preserved),
                        i0252_original_results_directory=str(original), i0252_visual_copy=str(base / "I0252" / "plots"),
                        new_simulations=0, script_sha256=sha(Path(__file__)),
                        files={name: sha(base / name) for name in exported})
    report = base / "validation" / f"convergence_window{suffix}_verification.json"
    atomic_json(report, verification)
    archive = base / f"all_instances_plots{suffix}.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name in exported:
            bundle.write(base / name, name)
        bundle.write(report, str(report.relative_to(base)))
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    print(f"Verified {len(images)//2} figures / {len(images)} PNG+PDF files; original files unchanged; no simulations.", flush=True)
    print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
