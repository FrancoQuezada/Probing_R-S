"""Cross-instance table and three multipanel plots; stored data only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyze_i0252_statistical_019 import COLORS, analyze, save_csv, save_figure
from i0252_statistical_storage_019 import METHODS, LABELS, config, load_complete, atomic_json, sha
from probing_statistical_instances_020 import BASE_OUTPUT, INSTANCES


def comparison(base, refresh=False):
    for folder in ("summaries", "plots"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    data = {}
    rows = []
    manifest = {}
    for instance in INSTANCES:
        output = base / instance
        if not (output / "config/experiment.toml").exists():
            continue
        cfg = config(output)
        if cfg["science"]["scenario"] != instance:
            raise ValueError(f"Wrong instance in {output}")
        results = load_complete(output, cfg)
        if not results:
            continue
        if refresh:
            analyze(output)
        stats = pd.read_csv(output / "summaries/iteration_statistics.csv")
        finals = pd.read_csv(output / "summaries/final_statistics.csv")
        analyzed = json.loads((output / "summaries/analysis_manifest.json").read_text())
        counts = {m: sum(meta["method"] == m for _, meta, _, _ in results) for m in METHODS}
        if analyzed["completed"] != counts or analyzed["fingerprint"] != cfg["fingerprint"]:
            raise ValueError(f"Stale summaries for {instance}; use --refresh")
        data[instance] = (stats, counts, cfg["science"]["max_iters"])
        manifest[instance] = {"fingerprint": cfg["fingerprint"], "completed": counts,
                              "statistics_sha256": sha(output / "summaries/iteration_statistics.csv"),
                              "final_statistics_sha256": sha(output / "summaries/final_statistics.csv")}
        for method in METHODS:
            group = finals[finals.method == method].set_index("metric")
            if group.empty:
                continue
            objective = group.loc["objective"]
            rows.append(dict(instance=instance, method=method, replications=int(objective["n"]),
                             z3_final_mean=group.loc["z3", "mean"], z4_final_mean=group.loc["z4", "mean"],
                             objective_final_mean=objective["mean"], objective_final_sd=objective["sd"],
                             objective_final_ci95_low=objective["ci95_low"], objective_final_ci95_high=objective["ci95_high"],
                             optimization_seconds_mean=group.loc["optimization_seconds", "mean"]))
    if not data:
        raise ValueError("No complete replications available for comparison")
    save_csv(base / "summaries/final_comparison.csv", pd.DataFrame(rows))
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True, "grid.alpha": .18,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    for metric, stem, ylabel in (("objective_crn", "objective_convergence_mean_ci95_comparison", "Objective J(z), independent direct CRN"),
                                 ("z3", "z3_convergence_mean_ci95_comparison", "$z_3$"),
                                 ("z4", "z4_convergence_mean_ci95_comparison", "$z_4$")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True, sharey=metric != "objective_crn")
        for instance, ax in zip(INSTANCES, axes.flat):
            ax.set(xlabel="Iteration", ylabel=ylabel, xlim=(0, 100), title=instance)
            if metric in ("z3", "z4"):
                ax.set_ylim(0, 1)
            if instance not in data:
                ax.text(.5, .5, "No complete replications", transform=ax.transAxes, ha="center")
                continue
            stats, counts, iterations = data[instance]
            ax.set_xlim(0, iterations)
            for method in METHODS:
                g = stats[(stats.method == method) & (stats.metric == metric)]
                if g.empty:
                    continue
                x, y, low, high = [g[k].to_numpy(dtype=float) for k in ("iteration", "mean", "ci95_low", "ci95_high")]
                ax.plot(x, y, color=COLORS[method], lw=1.9, label=f"{LABELS[method]}, R={counts[method]}")
                ax.fill_between(x, low, high, color=COLORS[method], alpha=.18, linewidth=0)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle("Independent replication means and pointwise 95% t confidence intervals", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .965))
        save_figure(base, stem, fig)
    atomic_json(base / "summaries/comparison_manifest.json", {
        "instances": manifest, "statistical_unit": "independent algorithm replication",
        "confidence_protocol": "Same pointwise Student-t 95% intervals as experiment 019",
        "objective_normalization": "none", "new_simulations": 0,
        "source_sha256": sha(Path(__file__)),
    })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="Rebuild individual statistics and plots first, without simulations")
    args = parser.parse_args()
    comparison(args.base.resolve(), args.refresh)
