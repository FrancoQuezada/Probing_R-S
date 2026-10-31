"""Export objective/z3/z4 mean-CI plots with exactly one method per figure."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyze_i0252_statistical_019 import COLORS, save_figure
from i0252_statistical_storage_019 import DEFAULT_OUTPUT, METHODS, LABELS, config, load_complete, atomic_json, sha
from probing_statistical_instances_020 import BASE_OUTPUT, INSTANCES, canonical_instance

METRICS = {
    "objective_crn": ("objective_convergence_mean_ci95", "Objective J(z), independent direct CRN"),
    "z3": ("z3_convergence_mean_ci95", "$z_3$"),
    "z4": ("z4_convergence_mean_ci95", "$z_4$"),
}


def plot_individual(output, methods=METHODS, start=0, end=None):
    cfg = config(output)
    instance = cfg["science"]["scenario"]
    final_iteration = cfg["science"]["max_iters"]
    end = final_iteration if end is None else end
    if not 0 <= start < end <= final_iteration:
        raise ValueError(f"Iteration window must lie within [0,{final_iteration}]")
    results = load_complete(output, cfg)
    counts = {m: sum(meta["method"] == m for _, meta, _, _ in results) for m in METHODS}
    del results
    analyzed = json.loads((output / "summaries/analysis_manifest.json").read_text())
    if analyzed["fingerprint"] != cfg["fingerprint"] or analyzed["completed"] != counts:
        raise ValueError(f"Stale statistics for {instance}; regenerate analysis before plotting")
    source = output / "summaries/iteration_statistics.csv"
    stats = pd.read_csv(source)
    window = stats[stats.iteration.between(start, end)]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True, "grid.alpha": .18,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    created = []
    suffix = "" if (start, end) == (0, final_iteration) else f"_iter{start}_{end}"
    for metric, (stem, ylabel) in METRICS.items():
        # Keep the same vertical scale across methods within each instance.
        reference = window[window.metric == metric][["mean", "ci95_low", "ci95_high"]].to_numpy(dtype=float)
        finite = reference[np.isfinite(reference)]
        if not finite.size:
            raise ValueError(f"No finite stored statistics for {instance}/{metric}")
        lower, upper = finite.min(), finite.max()
        padding = .05 * (upper - lower) if upper > lower else max(abs(lower) * .05, .01)
        limits = (0, 1) if metric in ("z3", "z4") else (lower-padding, upper+padding)
        for method in methods:
            group = window[(window.method == method) & (window.metric == metric)].sort_values("iteration")
            if group.iteration.tolist() != list(range(start, end + 1)):
                raise ValueError(f"Missing or duplicate iterations for {instance}/{method}/{metric}")
            n = group.n.unique()
            if len(n) != 1 or int(n[0]) != counts[method] or counts[method] < 2:
                raise ValueError(f"Inconsistent replication counts or undefined CI for {instance}/{method}")
            x, mean, low, high = [group[k].to_numpy(dtype=float) for k in ("iteration", "mean", "ci95_low", "ci95_high")]
            if not np.isfinite(np.concatenate((mean, low, high))).all():
                raise ValueError(f"Nonfinite mean or CI for {instance}/{method}/{metric}")
            fig, ax = plt.subplots(figsize=(9.5, 5.8))
            ax.plot(x, mean, color=COLORS[method], lw=1.9, label=f"Mean, R={counts[method]}")
            ax.fill_between(x, low, high, color=COLORS[method], alpha=.18, linewidth=0, label="95% confidence interval")
            ax.set(xlabel="Iteration", ylabel=ylabel, xlim=(start, end), ylim=limits,
                   title=f"{instance} · {LABELS[method]}\nIndependent replication mean and pointwise 95% t confidence interval")
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            name = f"{stem}_{instance}_{method}{suffix}"
            save_figure(output, f"{method}/{name}", fig)
            created.extend([f"{method}/{name}.png", f"{method}/{name}.pdf"])
    manifest = {
        "instance": instance, "methods": list(methods), "replications": {m: counts[m] for m in methods},
        "iteration_start": start, "iteration_end": end,
        "statistics_source": str(source.relative_to(output)), "statistics_sha256": sha(source),
        "script_sha256": sha(Path(__file__)),
        "confidence_intervals": "Unchanged stored pointwise 95% t confidence intervals for replication means",
        "vertical_scales": "Shared across methods within each instance; z3/z4 fixed to [0,1]",
        "new_simulations": 0, "figures": len(created)//2,
        "files": {name: sha(output / "plots" / name) for name in created},
    }
    selection = "all" if set(methods) == set(METHODS) else "_".join(methods)
    atomic_json(output / "plots" / f"individual_methods_{selection}{suffix}_manifest.json", manifest)
    print(f"{instance}: {len(created)//2} individual figures saved as PNG/PDF in {output/'plots'}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=canonical_instance, choices=("I0252", *INSTANCES), nargs="+", default=["I0252", *INSTANCES])
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--base", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--i0252-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    if len(args.instances) != len(set(args.instances)) or len(args.methods) != len(set(args.methods)):
        parser.error("Select each instance and method only once")
    for instance in args.instances:
        output = args.i0252_output if instance == "I0252" else args.base / instance
        if config(output)["science"]["scenario"] != instance:
            raise ValueError(f"Incorrect instance in {output}")
        plot_individual(output.resolve(), args.methods, args.start, args.end)


if __name__ == "__main__":
    main()
