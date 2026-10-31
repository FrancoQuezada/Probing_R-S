"""Rebuild statistics and figures from complete stored replications. No Julia calls."""
import argparse
from io import BytesIO
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t

from i0252_statistical_storage_019 import DEFAULT_OUTPUT, METHODS, LABELS, atomic_bytes, atomic_json, config, load_complete, sha

COLORS = dict(zip(METHODS, ("#0072B2", "#009E73", "#D55E00", "#CC79A7")))


def moments(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n)
    half = student_t.ppf(.975, n - 1) * se if n > 1 else np.nan
    return dict(n=n, mean=mean, sd=sd, se=se, ci95_low=mean-half, ci95_high=mean+half)


def save_csv(path, frame):
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode())


def save_figure(output, stem, fig):
    for fmt in ("png", "pdf"):
        buffer = BytesIO()
        metadata = {"CreationDate": None, "ModDate": None, "Creator": "I0252 statistical campaign"} if fmt == "pdf" else {"Software": "I0252 statistical campaign"}
        fig.savefig(buffer, format=fmt, dpi=300, bbox_inches="tight", metadata=metadata)
        atomic_bytes(output / "plots" / f"{stem}.{fmt}", buffer.getvalue())
    plt.close(fig)


def analyze(output):
    c = config(output)
    instance = c["science"]["scenario"]
    results = load_complete(output, c)
    if not results:
        raise ValueError("No complete replications to analyze")
    # Each row is one independent algorithm replication, not one Monte Carlo path.
    traces = pd.DataFrame([row for _, _, rows, _ in results for row in rows])
    finals = pd.DataFrame([f for _, _, _, f in results])
    for frame in (traces, finals):
        for col in frame.columns:
            if col != "method":
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["method"] = pd.Categorical(frame["method"], categories=METHODS, ordered=True)
    traces = traces.sort_values(["method", "replication", "iteration"]).reset_index(drop=True)
    finals = finals.sort_values(["method", "replication"]).reset_index(drop=True)
    grouped = traces.groupby(["method", "iteration"], observed=True, sort=True)
    rows = []
    for (method, iteration), group in grouped:
        for metric in ("objective_crn", "z3", "z4", "cumulative_optimization_seconds"):
            rows.append(dict(method=str(method), iteration=int(iteration), metric=metric, **moments(group[metric])))
    stats = pd.DataFrame(rows)
    final_stats = []
    for method, group in finals.groupby("method", observed=True, sort=True):
        for metric in ("objective", "z3", "z4", "optimization_seconds", "diagnostic_seconds", "evaluation_se",
                       "convergence_evaluation_seconds", "final_evaluation_seconds"):
            final_stats.append(dict(method=str(method), metric=metric, **moments(group[metric])))
    final_stats = pd.DataFrame(final_stats)
    # Paired differences use replication as the independent statistical unit.
    paired = []
    wide = finals.pivot(index="replication", columns="method", values="objective")
    if "green" in wide:
        for method in METHODS:
            if method == "green" or method not in wide:
                continue
            delta = (wide[method] - wide["green"]).dropna()
            if len(delta):
                paired.append(dict(method=method, reference="green", metric="final_objective_difference", **moments(delta)))
    policies = finals.groupby(["method", "selected_policy"], observed=True).size().reset_index(name="replications")
    stability = []
    for (method, rep), g in traces.groupby(["method", "replication"], observed=True):
        positions = g[["z3", "z4"]].to_numpy()
        differences = np.diff(positions, axis=0)
        tail = positions[-min(20, len(positions)):]
        stability.append(dict(method=str(method), replication=int(rep),
                              total_path_length=float(np.linalg.norm(differences, axis=1).sum()),
                              final_step_norm=float(np.linalg.norm(differences[-1])),
                              tail_rms_distance=float(np.sqrt(np.mean(np.sum((tail-tail.mean(axis=0))**2, axis=1))))))
    stability = pd.DataFrame(stability)
    stability_summary = pd.DataFrame([dict(method=str(method), metric=k, **moments(g[k]))
        for method, g in stability.groupby("method", observed=True)
        for k in ("total_path_length", "final_step_norm", "tail_rms_distance")])
    for name, frame in (("trajectories_all", traces), ("final_results_all", finals), ("iteration_statistics", stats),
                        ("final_statistics", final_stats), ("paired_final_vs_green", pd.DataFrame(paired)),
                        ("selected_policy_counts", policies), ("trajectory_stability", stability), ("stability_statistics", stability_summary)):
        save_csv(output / "summaries" / f"{name}.csv", frame)
    counts = {m: int(sum(meta["method"] == m for _, meta, _, _ in results)) for m in METHODS}
    count_label = ", ".join(f"{m}: R={n}" for m, n in counts.items())
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True, "grid.alpha": .18,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    labels = {"objective_crn": "Objective J(z), independent direct CRN", "z3": "$z_3$", "z4": "$z_4$"}
    stems = {"objective_crn": "objective_convergence_mean_ci95", "z3": "z3_convergence_mean_ci95", "z4": "z4_convergence_mean_ci95"}
    for metric, ylabel in labels.items():
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for method in METHODS:
            g = stats[(stats.method == method) & (stats.metric == metric)]
            if g.empty:
                continue
            x, y, low, high = [g[k].to_numpy(dtype=float) for k in ("iteration", "mean", "ci95_low", "ci95_high")]
            ax.plot(x, y, color=COLORS[method], lw=1.9, label=f"{LABELS[method]}, R={counts[method]}")
            ax.fill_between(x, low, high, color=COLORS[method], alpha=.18, linewidth=0)
        ax.set(xlabel="Iteration", ylabel=ylabel, xlim=(0, c["science"]["max_iters"]),
               title=f"{instance} · independent replication means and pointwise 95% t confidence intervals")
        if metric in ("z3", "z4"):
            ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        save_figure(output, stems[metric], fig)
    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    surface_path = output / f"config/{instance}_reference_surface.csv"
    surface_reused = surface_path.exists()
    if surface_reused:
        s = pd.read_csv(surface_path)
        grid = s.pivot(index="z4", columns="z3", values="objective")
        contour = ax.contourf(grid.columns.to_numpy(), grid.index.to_numpy(), grid.to_numpy(), levels=18, cmap="Greys", alpha=.55)
        reference_source = "011" if instance == "I0252" else "010"
        fig.colorbar(contour, ax=ax, label=f"Archived {instance} direct-CRN reference objective ({reference_source})", shrink=.83)
    for method in METHODS:
        if not counts[method]:
            continue
        g = traces[traces.method == method].groupby("iteration", sort=True)[["z3", "z4"]].mean()
        ends = finals[finals.method == method]
        ax.plot(g.z3, g.z4, lw=2.2, color=COLORS[method], label=f"{LABELS[method]}, R={counts[method]}")
        ax.scatter(ends.z3, ends.z4, s=15, color=COLORS[method], alpha=.4, linewidth=0)
        ax.scatter(g.z3.iloc[-1], g.z4.iloc[-1], marker="X", s=115, color=COLORS[method], edgecolor="white", linewidth=.7, zorder=6)
    ax.scatter(*c["science"]["Z0"], marker="*", s=180, color="#F0E442", edgecolor="black", label="Common initial point", zorder=7)
    ax.set(xlabel="$z_3$", ylabel="$z_4$", xlim=(0, 1), ylim=(0, 1), aspect="equal",
           title=f"{instance} · mean optimization trajectories and individual final decisions")
    ax.legend(loc="lower right", fontsize=8.3)
    fig.tight_layout()
    save_figure(output, "z3_z4_trajectories", fig)
    provenance = {
        "fingerprint": c["fingerprint"], "completed": counts, "statistical_unit": "independent algorithm replication",
        "analysis_sources": {name: sha(Path(__file__).with_name(name)) for name in
                             ("analyze_i0252_statistical_019.py", "i0252_statistical_storage_019.py")},
        "analysis_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                              "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "confidence_interval": "pointwise two-sided t(.975,R-1), sample sd ddof=1; undefined for R=1",
        "crn_paths_per_point": c["science"]["trace_paths"], "reference_surface_reused": surface_reused,
        "new_simulations_started_by_analysis": 0,
        "input_completion_manifests": {str(p.relative_to(output)): sha(p / "complete.toml") for p, _, _, _ in results},
        "notes": ["Marginal CIs can extend beyond the parameter domain; z axes are fixed to [0,1].",
                  "CRN convergence uses max of 144 policy sample means and can have policy-selection optimism.",
                  "Final performance evaluates one policy selected independently; MC SE excludes selection adjustment.",
                  "CI across replications includes optimizer and evaluation randomness; no simultaneous coverage claim.",
                  "Final scatter is observed dispersion; no artificial two-dimensional confidence band.",
                  "Optimization times are concurrent-process wall times; interpret hardware comparisons descriptively."],
    }
    atomic_json(output / "summaries/analysis_manifest.json", provenance)
    lines = [f"{instance} — resultados de la campaña", "", count_label, "", "Media final (IC95% t entre replicaciones):", ""]
    for row in final_stats[final_stats.metric == "objective"].itertuples():
        lines.append(f"{LABELS[row.method]}: {row.mean:.6f} [{row.ci95_low:.6f}, {row.ci95_high:.6f}], R={row.n}")
    lines.extend(["", "Diferencias finales pareadas respecto de Green (unidad: replicación):", ""])
    for row in paired:
        lines.append(f"{LABELS[row['method']]}: {row['mean']:.6f} [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}], R={row['n']}")
    lines.extend(["", "Las evaluaciones CRN de convergencia maximizan medias de políticas; la evaluación final usa selección y holdout independientes.",
                  "Los intervalos son puntuales, sin corrección por comparaciones múltiples. Tiempos con procesos concurrentes.", ""])
    atomic_bytes(output / "summaries/RESULTADOS.txt", "\n".join(lines).encode())
    print("Analysis rebuilt from stored complete replications:", counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    analyze(parser.parse_args().output.resolve())
