"""Stored-results analysis for 021; never runs historical simulations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from run_021 import BASE, ROOT, METHOD, INSTANCES, historical, validate_new, verify_history
from i0252_statistical_storage_019 import METHODS as OLD_METHODS, LABELS as OLD_LABELS, config, load_complete, atomic_json, atomic_bytes, sha
from analyze_i0252_statistical_019 import COLORS as OLD_COLORS, moments, save_csv, save_figure

METHODS = (*OLD_METHODS, METHOD)
LABELS = {**OLD_LABELS, METHOD: "Green Variance + Softmax (κ=2, β=0.5)"}
SHORT = dict(zip(METHODS, ("Fresh", "Green", "Green Softmax", "Green Variance", "GV+S")))
COLORS = {**OLD_COLORS, METHOD: "#6A3D9A"}
METRICS = {"objective_crn": ("objective_convergence_mean_ci95", "Objective J(z), independent direct CRN"),
           "z3": ("z3_convergence_mean_ci95", "$z_3$"), "z4": ("z4_convergence_mean_ci95", "$z_4$")}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True, "grid.alpha": .18,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def numeric(frame, exclude=("method", "scenario")):
    for key in frame:
        if key not in exclude:
            frame[key] = pd.to_numeric(frame[key], errors="coerce")
    return frame


def load(instance, verify_banks=True):
    out = BASE / instance
    old = load_complete(historical(instance))
    assert {m: sum(meta["method"] == m for _, meta, _, _ in old) for m in OLD_METHODS} == dict.fromkeys(OLD_METHODS, 50)
    new = [(out / "raw" / METHOD / f"rep_{r:04d}", *validate_new(instance, r, hashes=verify_banks)) for r in range(1, 51)]
    results = old + new
    for method in METHODS:
        assert sorted(meta["replication"] for _, meta, _, _ in results if meta["method"] == method) == list(range(1, 51))
    seeds = [new[r-1][1]["seeds"] for r in range(1, 51)]
    flat = [v for s in seeds for v in [*s["search"], s["trace"], s["selection"], s["holdout"]]]
    assert len(flat) == len(set(flat)) == 5150
    tr = numeric(pd.DataFrame([row for _, _, rows, _ in results for row in rows]))
    finals = numeric(pd.DataFrame([f for _, _, _, f in results]))
    diag = [];runtime = []
    for path, meta, rows, final in results:
        method = meta["method"];rep = meta["replication"]
        d = pd.read_csv(path / "iteration_diagnostics.csv")
        d = d[d.iteration > 0].copy();assert len(d) == 100
        kept = d.retained_banks if "retained_banks" in d else d.history_banks
        generated = d.generated_banks if "generated_banks" in d else d.iteration
        # Fresh has no historical rejection mechanism; use explicit unavailability.
        rejected = generated-kept if method != "fresh" else pd.Series(np.nan, index=d.index)
        diag.append(dict(method=method, replication=rep, accepted_banks_mean=kept.mean(),
                         rejected_banks_mean=rejected.mean(), ess_mean=pd.to_numeric(pd.DataFrame(rows).ess, errors="coerce").mean(),
                         rejection_available=method != "fresh"))
        runtime.append(dict(method=method, replication=rep,
                            end_to_end_seconds=meta.get("last_attempt_seconds", np.nan)))
    diagnostics = pd.DataFrame(diag)
    finals = finals.merge(diagnostics, on=["method", "replication"]).merge(pd.DataFrame(runtime), on=["method", "replication"])
    return tr.sort_values(["method", "replication", "iteration"]), finals, results


def statistics(instance, tr, finals):
    out = BASE / instance
    stats = pd.DataFrame([dict(method=m, iteration=int(t), metric=metric, **moments(g[metric]))
                          for (m, t), g in tr.groupby(["method", "iteration"])
                          for metric in (*METRICS, "objective_internal", "cumulative_optimization_seconds")])
    # Numerical regression against the original 019/020 statistical tables.
    archived = pd.read_csv(historical(instance) / "summaries/iteration_statistics.csv")
    joined = archived.merge(stats, on=["method", "iteration", "metric"], suffixes=("_old", "_021"), validate="one_to_one")
    assert len(joined) == len(archived)
    for key in ("n", "mean", "sd", "se", "ci95_low", "ci95_high"):
        np.testing.assert_allclose(joined[key+"_old"], joined[key+"_021"], rtol=1e-11, atol=1e-11)
    # Internal estimates have no value at t=100. Keep those missing, not fabricated.
    fs = pd.DataFrame([dict(method=m, metric=metric, **moments(g[metric]))
                       for m, g in finals.groupby("method")
                       for metric in ("objective", "z3", "z4", "optimization_seconds", "end_to_end_seconds",
                                      "diagnostic_seconds", "evaluation_se", "accepted_banks_mean", "rejected_banks_mean", "ess_mean")])
    table = []
    for method in METHODS:
        g = finals[finals.method == method];s = moments(g.objective)
        table.append(dict(instance=instance, method=method, replications=50,
                          objective_final_mean=s["mean"], ci95_low=s["ci95_low"], ci95_high=s["ci95_high"],
                          objective_final_sd=s["sd"], objective_final_se=s["se"],
                          z3_final_mean=g.z3.mean(), z4_final_mean=g.z4.mean(),
                          optimization_seconds_mean=g.optimization_seconds.mean(),
                          end_to_end_seconds_mean=g.end_to_end_seconds.mean(),
                          accepted_banks_mean=g.accepted_banks_mean.mean(), rejected_banks_mean=g.rejected_banks_mean.mean(),
                          ess_mean=g.ess_mean.mean(), diagnostic_seconds_mean=g.diagnostic_seconds.mean(),
                          unavailable="historical bank rejection: not applicable (Fresh)" if method == "fresh" else "none"))
    stability = []
    for (method, rep), group in tr.groupby(["method", "replication"]):
        z = group[["z3", "z4"]].to_numpy();tail = z[-20:]
        stability.append(dict(method=method, replication=rep,
                              total_path_length=np.linalg.norm(np.diff(z, axis=0), axis=1).sum(),
                              final_step_norm=np.linalg.norm(z[-1]-z[-2]),
                              tail_rms_distance=np.sqrt(np.mean(np.sum((tail-tail.mean(axis=0))**2, axis=1))),
                              tail_objective_sd=group.objective_crn.iloc[-20:].std(ddof=1)))
    stability = pd.DataFrame(stability)
    pairs = []
    rng = np.random.default_rng(int(hashlib.sha256(("021/"+instance).encode()).hexdigest()[:8], 16))
    boot = rng.integers(0, 50, size=(5000, 50))
    new = finals[finals.method == METHOD].set_index("replication").sort_index()
    stnew = stability[stability.method == METHOD].set_index("replication").sort_index()
    for reference in OLD_METHODS:
        old = finals[finals.method == reference].set_index("replication").sort_index()
        delta = new.objective-old.objective
        sd_ratio = np.std(new.objective.to_numpy()[boot], axis=1, ddof=1)/np.std(old.objective.to_numpy()[boot], axis=1, ddof=1)
        sto = stability[stability.method == reference].set_index("replication").sort_index()
        row = dict(instance=instance, method=METHOD, reference=reference, metric="final_holdout_objective_difference",
                   **moments(delta), p_raw=1. if np.all(delta.to_numpy() == 0) else float(ttest_1samp(delta, 0).pvalue),
                   sd_ratio=new.objective.std(ddof=1)/old.objective.std(ddof=1),
                   sd_ratio_boot_ci95_low=np.quantile(sd_ratio, .025), sd_ratio_boot_ci95_high=np.quantile(sd_ratio, .975),
                   optimization_time_ratio=new.optimization_seconds.mean()/old.optimization_seconds.mean(),
                   end_to_end_time_ratio=new.end_to_end_seconds.mean()/old.end_to_end_seconds.mean())
        row.update({"tail_rms_difference_"+k: v for k, v in moments(stnew.tail_rms_distance-sto.tail_rms_distance).items()})
        row.update({"path_length_difference_"+k: v for k, v in moments(stnew.total_path_length-sto.total_path_length).items()})
        pairs.append(row)
    for name, frame in (("trajectories_all", tr), ("final_results_all", finals), ("iteration_statistics", stats),
                        ("final_statistics", fs), ("final_comparison", pd.DataFrame(table)),
                        ("trajectory_stability", stability), ("paired_final_differences", pd.DataFrame(pairs))):
        save_csv(out / "summaries" / f"{name}.csv", frame)
    return stats, pd.DataFrame(table), pd.DataFrame(pairs), stability


def line_stats(ax, stats, metric, methods=METHODS, start=0, end=100):
    for method in methods:
        g = stats[(stats.method == method) & (stats.metric == metric) & stats.iteration.between(start, end)]
        assert len(g) == end-start+1
        x, mean, low, high = [g[k].to_numpy(float) for k in ("iteration", "mean", "ci95_low", "ci95_high")]
        ax.plot(x, mean, color=COLORS[method], lw=1.9, label=f"{SHORT[method]}, R=50")
        ax.fill_between(x, low, high, color=COLORS[method], alpha=.18, linewidth=0)
    ax.set(xlabel="Iteration", ylabel=METRICS[metric][1], xlim=(start, end))
    if metric in ("z3", "z4"):
        ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)


def trajectory(ax, instance, tr, finals, methods=METHODS, background=True):
    if background:
        surface = pd.read_csv(BASE / instance / f"config/{instance}_reference_surface.csv")
        grid = surface.pivot(index="z4", columns="z3", values="objective")
        ax.contourf(grid.columns, grid.index, grid.to_numpy(), levels=18, cmap="Greys", alpha=.55)
    for method in methods:
        g = tr[tr.method == method].groupby("iteration")[["z3", "z4"]].mean()
        ends = finals[finals.method == method]
        ax.plot(g.z3, g.z4, lw=2.2, color=COLORS[method], label=SHORT[method])
        ax.scatter(ends.z3, ends.z4, s=15, color=COLORS[method], alpha=.4, linewidth=0)
        ax.scatter(g.z3.iloc[-1], g.z4.iloc[-1], marker="X", s=100, color=COLORS[method], edgecolor="white", zorder=6)
    ax.scatter(.12, .10, marker="*", s=150, color="#F0E442", edgecolor="black", zorder=7)
    ax.set(xlabel="$z_3$", ylabel="$z_4$", xlim=(0, 1), ylim=(0, 1), aspect="equal", title=instance)
    ax.legend(loc="best", fontsize=7)


def instance_figures(instance, tr, finals, stats):
    out = BASE / instance
    for metric, (stem, ylabel) in METRICS.items():
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        line_stats(ax, stats, metric);ax.set_title(f"{instance} · means and pointwise 95% t CI")
        fig.tight_layout();save_figure(out, stem, fig)

        for method in METHODS:
            fig, ax = plt.subplots(figsize=(9.5, 5.8))
            line_stats(ax, stats, metric, (method,));ax.set_title(f"{instance} · {LABELS[method]} · mean and 95% t CI")
            if metric == "objective_crn":
                ref = stats[stats.metric == metric][["ci95_low", "ci95_high"]].to_numpy(float)
                low, high = np.nanmin(ref), np.nanmax(ref);pad = .05*(high-low)
                ax.set_ylim(low-pad, high+pad)
            fig.tight_layout();save_figure(out, f"{method}/{stem}_{instance}_{method}", fig)
    for methods, stem in ((METHODS, "z3_z4_trajectories"), ((METHOD,), f"{METHOD}/z3_z4_trajectories")):
        fig, ax = plt.subplots(figsize=(8.4, 7.3));trajectory(ax, instance, tr, finals, methods)
        fig.tight_layout();save_figure(out, stem, fig)


def individual_i0252(tr):
    out = BASE / "I0252"
    for rep in (1, 2, 3):
        g = tr[(tr.method == METHOD) & (tr.replication == rep)]
        for metric, (_, ylabel) in METRICS.items():
            fig, ax = plt.subplots(figsize=(9.5, 5.8))
            ax.plot(g.iteration, g[metric], c=COLORS[METHOD], lw=1.8)
            ax.set(xlabel="Iteration", ylabel=ylabel, title=f"I0252 · GV+S · rep{rep:02d}", xlim=(0, 100))
            if metric in ("z3", "z4"):
                ax.set_ylim(0, 1)
            fig.tight_layout();save_figure(out, f"rep{rep:02d}/{metric}", fig)
        fig, ax = plt.subplots(figsize=(8.4, 7.3))
        trajectory(ax, "I0252", g, g.tail(1), (METHOD,))
        ax.set_title(f"I0252 · GV+S · rep{rep:02d}")
        fig.tight_layout();save_figure(out, f"rep{rep:02d}/z3_z4_trajectory", fig)
        path = out / "raw" / METHOD / f"rep_{rep:04d}"
        d = pd.read_csv(path / "iteration_diagnostics.csv")
        banks = pd.read_csv(path / "bank_diagnostics.csv")
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
        axes[0, 0].plot(d.iteration, d.retained_banks, label="Accepted", color=COLORS[METHOD])
        axes[0, 0].plot(d.iteration, d.rejected_banks, label="Rejected", color="#D55E00")
        axes[0, 0].set(ylabel="Banks");axes[0, 0].legend()
        axes[0, 1].plot(d.iteration, d.filter_current_variance, label="Current-bank sample variance")
        axes[0, 1].plot(d.iteration, d.filter_threshold, label="Threshold = 2 × current variance")
        axes[0, 1].set(ylabel="Variance of provisional-policy LR reward");axes[0, 1].legend(fontsize=8)
        axes[1, 0].plot(d.iteration, d.ess_fraction, color=COLORS[METHOD], label="ESS / accepted observations")
        axes[1, 0].set(ylabel="ESS fraction", ylim=(0, 1.05));axes[1, 0].legend(fontsize=8)
        heat = banks.pivot(index="bank_iteration", columns="iteration", values="retained").astype(float)
        axes[1, 1].imshow(heat.to_numpy(), origin="lower", aspect="auto", extent=(.5, 100.5, .5, 100.5),
                          cmap="Purples", vmin=0, vmax=1, interpolation="nearest")
        axes[1, 1].set(ylabel="Bank generation iteration", title="Accepted=purple; rejected=white; future=blank")
        for ax in axes.flat:
            ax.set_xlabel("Iteration")
        fig.suptitle(f"I0252 · GV+S · rep{rep:02d} · internal bank diagnostics")
        fig.tight_layout();save_figure(out, f"rep{rep:02d}/green_variance_diagnostics", fig)


def compare017():
    import toml
    old = ROOT / "resultados/017probing_green_five_method_comparison"
    science = toml.load(old / "configs/experiment.toml")["science"]
    assert science["new_paths_per_iteration"] == 500 and science["iterations"] == 100 and science["trace_paths"] == 1000
    assert science["model_sha256"] == config(BASE / "I0252")["science"]["model_sha256"]
    ids = ("Green", "GreenSoftmax050", "Exp50", "Rolling5", METHOD)
    traces = pd.read_csv(old / "tables/objective_traces.csv")
    coords = pd.read_csv(old / "tables/coordinates.csv")
    traces = traces[traces.method.isin(ids)].copy()
    traces["search_seed_replication"] = traces.rep
    traces["source_experiment"] = "017"
    for rep in (1, 2, 3):
        path = BASE / "I0252/raw" / METHOD / f"rep_{rep:04d}/trace_crn017.csv"
        new = pd.read_csv(path).rename(columns={"replication": "rep"})
        new["method"] = METHOD;new["search_seed_replication"] = 19_000_000+rep;new["source_experiment"] = "021"
        oldseed = traces[traces.rep == rep].evaluation_seed.unique()
        assert len(oldseed) == 1 and np.all(new.evaluation_seed == oldseed[0])
        traces = pd.concat([traces, new], ignore_index=True)
    out = BASE / "comparisons/017"
    save_csv(out / "summaries/trajectories_017_plus_021.csv", traces)
    summary = []
    with plt.rc_context():
        from analyze_green_five_comparison_017 import COLORS as C17, LABELS as L17
        from analyze_landscape_010 import background
        colors = {**C17, METHOD: "#009E73"};labels = {**L17, METHOD: "Green Variance + Softmax"}
        surface = pd.read_csv(old / "configs/I0252_surface.csv")
        for rep in (1, 2, 3):
            for field, ylabel, stem in (("objective", "Objective J(z), direct CRN N=1000", "A_objective"),
                                        ("z3", "$z_3$", "B_z3"), ("z4", "$z_4$", "C_z4")):
                fig, ax = plt.subplots(figsize=(9.5, 5.8))
                for method in ids:
                    g = traces[(traces.rep == rep) & (traces.method == method)].sort_values("iteration")
                    assert g.iteration.tolist() == list(range(101))
                    if method != METHOD:
                        z = coords[(coords.rep == rep) & (coords.method == method)].sort_values("iteration")
                        np.testing.assert_allclose(g[["z3", "z4"]], z[["z3", "z4"]], rtol=0, atol=1e-14)
                    ax.plot(g.iteration, g[field], c=colors[method], label=labels[method], lw=1.8)
                ax.set(xlabel="Iteration", ylabel=ylabel, xlim=(0, 100), title=f"I0252 · rep{rep:02d} · 500 new paths/iteration")
                if field != "objective":
                    ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                fig.text(.5, .005, "Shared evaluation CRN; historical and new search streams differ (descriptive comparison)", ha="center", fontsize=8)
                fig.tight_layout(rect=(0, .025, 1, 1));save_figure(out, f"rep{rep:02d}/{stem}", fig)
            fig, ax = plt.subplots(figsize=(9, 7))
            image = background(ax, surface, f"I0252: trajectories, rep{rep:02d} — five methods")
            for method in ids:
                g = traces[(traces.rep == rep) & (traces.method == method)].sort_values("iteration")
                ax.plot(g.z3, g.z4, c=colors[method], lw=1.6, label=labels[method])
                ax.scatter(g.z3.iloc[-1], g.z4.iloc[-1], c=colors[method], s=60, edgecolor="white", zorder=10)
                zz = g[["z3", "z4"]].to_numpy()
                summary.append(dict(rep=rep, method=method, objective_crn017_final=g.objective.iloc[-1],
                                    mean_objective_over_iterations=g.objective.mean(), z3_final=zz[-1, 0], z4_final=zz[-1, 1],
                                    path_length=np.linalg.norm(np.diff(zz, axis=0), axis=1).sum(),
                                    z3_at10=zz[10,0], z4_at10=zz[10,1], objective_at10=g.objective.iloc[10],
                                    min_objective_after10=g.objective.iloc[10:].min(),
                                    tail_objective_range=g.objective.iloc[-20:].max()-g.objective.iloc[-20:].min(),
                                    tail_decision_rms=np.sqrt(np.mean(np.sum((zz[-20:]-zz[-20:].mean(axis=0))**2,axis=1))),
                                    search_seed_replication=int(g.search_seed_replication.iloc[0])))
            ax.scatter(.12, .10, marker="*", s=130, color="gold", edgecolor="black", zorder=11)
            fig.colorbar(image, ax=ax, label="J(z)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.28, 1), fontsize=8)
            fig.tight_layout();save_figure(out, f"rep{rep:02d}/D_trajectories", fig)
    save_csv(out / "summaries/individual_comparison.csv", pd.DataFrame(summary))
    atomic_json(out / "summaries/provenance.json", dict(historical_source=str(old), historical_budget=500,
        evaluation_paths=1000, same_evaluation_crn=True, paired_search_streams=False,
        historical_optimizations_rerun=0, historical_observations_added_to_statistical_sample=0,
        source_files={name: sha(old / "tables" / name) for name in ("coordinates.csv", "objective_traces.csv")}))
    return pd.DataFrame(summary)


def grouped_figures(data, paired):
    out = BASE / "comparisons/020"
    for start, end in ((0, 100), (2, 25)):
        suffix = "" if start == 0 else "_all_instances_iter2_25"
        for metric, (stem, _) in METRICS.items():
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=metric != "objective_crn")
            for instance, ax in zip(INSTANCES, axes.flat):
                line_stats(ax, data[instance][2], metric, start=start, end=end);ax.set_title(instance)
            axes.flat[-1].axis("off")
            fig.suptitle("Independent replication means and pointwise 95% t confidence intervals (R=50)")
            fig.tight_layout(rect=(0, 0, 1, .965));save_figure(out, stem+"_comparison"+suffix, fig)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for instance, ax in zip(INSTANCES, axes.flat):
        tr, finals, _ = data[instance];trajectory(ax, instance, tr, finals)
    axes.flat[-1].axis("off");fig.tight_layout();save_figure(out, "z3_z4_trajectories_all_instances", fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, reference in zip(axes, ("green_variance", "green_softmax")):
        g = paired[paired.reference == reference].set_index("instance").loc[list(INSTANCES)]
        y = np.arange(5)
        ax.errorbar(g["mean"], y, xerr=np.array([g["mean"]-g.ci95_low, g.ci95_high-g["mean"]]),
                    fmt="o", c=COLORS[METHOD], capsize=4)
        ax.axvline(0, c="black", lw=1);ax.set_yticks(y, INSTANCES)
        ax.set(xlabel="GV+S − reference: paired final objective difference", title=f"Reference: {SHORT[reference]}")
    fig.tight_layout();save_figure(out, "paired_final_objective_differences", fig)


def verify_numeric(instance, reps=range(1, 51)):
    reps = list(reps)
    import toml
    alpha = np.asarray(toml.load(BASE / instance / f"config/{instance}.toml")["model"]["ALPHA"])
    for rep in reps:
        path = BASE / instance / "raw" / METHOD / f"rep_{rep:04d}"
        d = pd.read_csv(path / "iteration_diagnostics.csv")
        b = pd.read_csv(path / "bank_diagnostics.csv")
        p = pd.read_csv(path / "policy_estimates.csv")
        assert len(d) == 100 and len(b) == 5050 and len(p) == 14400
        assert d.iteration.tolist() == list(range(1, 101))
        assert np.all(d.new_paths == 500) and np.all(d.cumulative_new_paths == 500*d.iteration)
        assert np.all(d.retained_banks+d.rejected_banks == d.iteration)
        assert np.isfinite(d[["g3", "g4", "ess", "filter_current_variance", "filter_threshold", "internal_lse_surrogate"]]).all().all()
        assert np.isfinite(p[["gross_mean", "gross_g3", "gross_g4", "softmax_weight"]]).all().all()
        expected_keep = (b.filter_sample_variance <= b.filter_threshold) | b.is_current
        assert np.array_equal(b.retained, expected_keep)
        assert np.all(b.filter_sample_variance >= 0) and np.all(b.bank_lr_ess > 0)
        for k, group in b.groupby("iteration"):
            row = d.iloc[k-1]
            assert group.bank_iteration.tolist() == list(range(1, k+1))
            assert group.retained.sum() == row.retained_banks
            assert group[group.is_current].retained.all() and len(group[group.is_current]) == 1
            assert np.isclose(row.filter_threshold, 2*row.filter_current_variance, rtol=1e-13)
            np.testing.assert_allclose(group.filter_threshold, row.filter_threshold, rtol=1e-13)
            assert [int(x) for x in str(row.retained_bank_indices).split(";")] == group[group.retained].bank_iteration.tolist()
        v = p.gross_mean.to_numpy().reshape(100, 144)
        w = p.softmax_weight.to_numpy().reshape(100, 144)
        expected = np.exp(.5*(v-v.max(axis=1)[:, None]));expected /= expected.sum(axis=1)[:, None]
        np.testing.assert_allclose(w, expected, rtol=1e-12, atol=1e-14)
        assert np.all(w >= 0);np.testing.assert_allclose(w.sum(axis=1), 1., atol=1e-14)
        gradients = np.column_stack([(p[key].to_numpy().reshape(100, 144)*w).sum(axis=1) for key in ("gross_g3", "gross_g4")])-alpha
        np.testing.assert_allclose(gradients, d[["g3", "g4"]], rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(v.max(axis=1)-d.probing_cost, d.objective_internal, rtol=1e-12, atol=1e-10)
        tr = pd.read_csv(path / "trajectory.csv")
        np.testing.assert_allclose(tr.iloc[:-1][["g3", "g4"]], gradients, rtol=1e-10, atol=1e-9)
    result = dict(instance=instance, complete_replications=50, iterations_per_replication=100,
                  paths_per_iteration=500, numeric_banks_checked=252500, policy_weights_checked=720000,
                  passed=True, checks=["all retained indices match original variance rule", "current bank always retained",
                  "stable softmax from accepted policy estimates", "gradient equals gross G'weights-alpha",
                  "no NaN/Inf in gradients, weights, variances or ESS", "independent seeds match 020"])
    if len(reps) == 50:
        atomic_json(BASE / instance / "validation/production_checks.json", result)
    return result


def holm(paired):
    paired = paired.copy()
    order = np.argsort(paired.p_raw.to_numpy());adjusted = np.empty(len(order));previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order)-rank)*paired.iloc[index].p_raw))
        adjusted[index] = previous
    paired["p_holm_20"] = adjusted
    return paired


def markdown_table(frame, columns, names=None):
    names = names or columns
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in names) + " |"]
    for _, row in frame.iterrows():
        values = []
        for key in columns:
            value = row[key]
            if isinstance(value, (float, np.floating)):
                values.append("No disponible / no aplica" if np.isnan(value) else f"{value:.5f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(tables, paired, individual, checks, preservation, figure_count):
    lines = ["# RESUMEN EXPERIMENTAL 021 — Green Variance + Softmax", "",
        "**Resultado principal:** se completaron y validaron 250 replicaciones nuevas y 135 figuras. "
        "En ninguna de las cinco instancias se encontró una mejora del objetivo final de GV+S frente a Green Variance o Green Softmax con respaldo tras el ajuste Holm de los 20 contrastes. "
        "Frente a Fresh, las mejoras finales de I0252 y L030 sí conservan significación tras ese ajuste. "
        "El comportamiento depende de la instancia; L026 conserva una dispersión alta y soluciones ocasionalmente muy desfavorables.", "",
        "## A. Implementación", "",
        "Se implementó **Green → filtro de varianza original (κ=2) → estimación por política con bancos aceptados → Softmax (β=0.5) → actualización original**. "
        "El nuevo módulo llama directamente a `ProbingGreenBankVariants.estimate(..., Filter2)` de las fuentes congeladas originales; no redefine su varianza, política provisional ni regla de aceptación. "
        "El filtro primero identifica la política de mayor media con Green completo; para esa misma política calcula la varianza muestral corregida de `Y_i × LR` dentro de cada banco. "
        "Conserva los bancos cuya varianza es ≤2 veces la del banco actual y siempre conserva el actual. Recalcula después las medias y gradientes con Green exclusivamente sobre los aceptados, con pesos iguales por banco.", "",
        "Softmax reutiliza `R.softmax_weights(values; beta=.5)`, que resta el máximo antes de exponenciar. El gradiente es `R.combine(G,w,cfg) = Gᵀw − alpha`; `G` contiene beneficios brutos. "
        "El costo se resta una sola vez. La actualización llama a `R.projected_step` sin cambios. Se conservan LR ordinarios, clamp log-LR ±35 y safeguard de probabilidades 1e−12.", "",
        "El objetivo de evaluación sigue siendo `max_i E[Y_i] − alpha'z`, con una política ex ante común. No se usa un máximo realizado por trayectoria. "
        "El gradiente Softmax corresponde al surrogate log-sum-exp de las medias estimadas: no es el gradiente completo del promedio de beneficios ponderado por Softmax. "
        "El campo interno histórico `objective_internal` conserva el máximo duro de las medias menos costo; `internal_lse_surrogate` guarda separadamente el surrogate. "
        "Las varianzas de contribuciones en los diagnósticos del 016 corresponden a su política dura aceptada, y se etiquetan como tales, no como varianza del gradiente de mezcla.", "",
        "Código: [módulo Julia](code/021_green_variance_softmax.jl), [ejecutor](code/run_021.py), [análisis](code/analyze_021.py). "
        "Cada instancia conserva las fuentes numéricas y entorno Julia originales, con hashes, en `reproduction/`.", "",
        "## B. Experimentos", "",
        "Se completaron **250 replicaciones nuevas: 50 por instancia**, con 100 actualizaciones y 500 trayectorias nuevas por actualización: **12.500.000 trayectorias de búsqueda**. "
        "Se preservan los 25.000 bancos materializados y sus matrices de beneficios de 144 políticas, las semillas, decisiones t=0…100, estimadores, pesos y diagnósticos. "
        "Todas las configuraciones se leyeron directamente de las campañas históricas, sin cambiar costos, distribuciones, llegadas, servicios ni políticas.", "",
        "Parámetros comunes verificados: inicio `(z3,z4)=(0.12,0.10)`, paso base 0.035, clipping por componente 15, 45 trabajos por trayectoria y 144 políticas. "
        "Se utilizaron 12 procesos Julia persistentes con BLAS=1; cada réplica es secuencial y su semilla no depende del orden de ejecución. Julia y dependencias son las versiones congeladas en 019/020.", "",
        "Semillas estadísticas: `X.seed(instancia, \"compare\", 19000000+r, stream, iteración)`, para r=1…50. Se verificó igualdad de los streams de búsqueda, traza, selección y holdout con la réplica histórica correspondiente. "
        "Los streams de cada réplica y las diferentes réplicas son independientes; el CRN compartido entre métodos permite diferencias pareadas.", "",
        "Evaluación de curvas: banco independiente de 1.000 primitivas CRN por réplica, rematerializado en cada decisión, evaluando las 144 medias y su máximo. "
        "Esa evaluación puede presentar optimismo por selección de la máxima media. Evaluación final principal: selección de una política con 10.000 trayectorias y medición de esa única política con otras 50.000 independientes. "
        "El informe prioriza este holdout para calidad final. Ambas evaluaciones reutilizan exactamente las funciones del protocolo histórico y nunca influyen en la búsqueda.", "",
        "Estadística: unidad independiente = réplica del algoritmo, n=50; desviación muestral con ddof=1; error estándar = sd/√50; IC95% puntual Student-t con 49 grados de libertad. "
        "Se reutilizó `moments` del análisis 019, utilizado por 020. No se mezcla la varianza de bancos con la variabilidad entre réplicas. "
        "La dispersión entre réplicas incorpora la aleatoriedad de optimización, selección y evaluación; no es únicamente variabilidad de la decisión final. "
        "Las diferencias de objetivo final usan t pareada por réplica; además se informa p ajustada por Holm en los 20 contrastes de las cinco instancias frente a cuatro referencias. "
        "Los IC95% siguen siendo puntuales, sin ajuste simultáneo, como en 020. Las razones de desviaciones estándar incluyen bootstrap pareado percentil con 5.000 remuestreos y semilla fija; son análisis exploratorios.", "",
        "**Procedencia de I0252:** el directorio 020 contiene gráficos de I0252, pero sus datos originales son los de `019probing_i0252_four_methods_statistical`. "
        "Se leyeron allí las mismas 50 réplicas de cada método una sola vez. Para L000/L025/L026/L030 se leyeron las carpetas originales de 020. No se repitió ninguna optimización histórica ni se agregó 019 como una segunda muestra.", "",
        "## C. Resultados I0252 — comparación con el experimento 017", "",
        "El 017 archiva N=500, T=100, Softmax β=0.5, Exponential ρ=0.5 y Rolling W=5. Sus tablas originales `coordinates.csv` y `objective_traces.csv` alimentan las nuevas figuras A–D para rep01–rep03. "
        "Se conservaron sus colores y el fondo original de la superficie I0252. No se incorporaron presupuestos N=250 ni otros regímenes.", "",
        "Las búsquedas de 017 utilizan índices de semilla r=1,2,3, mientras que las tres primeras réplicas estadísticas de 021 usan 19000000+r. "
        "Para hacer comparables las curvas se evaluaron adicionalmente **solo las nuevas decisiones** con los mismos bancos CRN de 1.000 trayectorias del 017. "
        "Estas evaluaciones adicionales se guardan en `trace_crn017.csv` y no sustituyen ni duplican observaciones estadísticas. Las comparaciones individuales siguen siendo descriptivas: no hay pareamiento de las búsquedas.", ""]
    for rep in (1, 2, 3):
        frame = individual[individual.rep == rep].copy()
        lines += [f"### rep{rep:02d}", "", markdown_table(frame,
                  ["method", "objective_crn017_final", "z3_final", "z4_final", "path_length"],
                  ["Método", "Objetivo CRN final (017)", "z3 final", "z4 final", "Longitud de trayectoria"]), ""]
        new = frame[frame.method == METHOD].iloc[0]
        best = frame.loc[frame.objective_crn017_final.idxmax()]
        lines += [f"GV+S termina en ({new.z3_final:.4f}, {new.z4_final:.4f}), con objetivo CRN {new.objective_crn017_final:.4f}; "
                  f"el mayor valor de esta comparación individual es {best['method']} ({best.objective_crn017_final:.4f}). "
                  f"Su longitud acumulada de trayectoria es {new.path_length:.4f}. "
                  f"En t=10 había alcanzado ({new.z3_at10:.4f}, {new.z4_at10:.4f}), con objetivo {new.objective_at10:.4f}. "
                  f"En las últimas 20 decisiones, el RMS respecto de su media es {new.tail_decision_rms:.5f} y el rango del objetivo CRN es {new.tail_objective_range:.5f}. "
                  "La longitud y el movimiento de cola son descriptores, no pruebas de convergencia ni superioridad. "
                  f"[Figuras A–D](comparisons/017/plots/rep{rep:02d}/A_objective.png); "
                  f"[diagnósticos del nuevo método](I0252/plots/rep{rep:02d}/green_variance_diagnostics.png).", ""]
    lines += ["## D. Resultados estadísticos — referencia 020", "",
              "Las tablas siguientes usan la evaluación final independiente con selección y holdout. Tiempo = segundos medios de búsqueda calentada, excluyendo diagnósticos, persistencia y evaluaciones, siguiendo 020. "
              "También se conservan tiempos completos por réplica en los CSV. La carga concurrente, instrumentación y sesiones distintas impiden interpretar sus razones como benchmarks de costo intrínseco. "
              "Bancos y ESS son medias sobre las 100 iteraciones y las 50 réplicas. Fresh usa un banco actual; el rechazo de históricos no aplica y se indica explícitamente.", ""]
    for instance in INSTANCES:
        table = tables[tables.instance == instance].copy();table["method"] = table.method.map(SHORT)
        table["IC95%"] = table.apply(lambda r: f"[{r.ci95_low:.5f}, {r.ci95_high:.5f}]", axis=1)
        lines += [f"### {instance}", "", markdown_table(table,
                  ["method", "objective_final_mean", "IC95%", "objective_final_sd", "z3_final_mean", "z4_final_mean", "optimization_seconds_mean", "accepted_banks_mean", "rejected_banks_mean", "ess_mean"],
                  ["Método", "Objetivo final medio", "IC95%", "SD", "z3", "z4", "Tiempo (s)", "Bancos aceptados", "Bancos rechazados", "ESS"]), "",
                  f"[Tabla completa]({instance}/summaries/final_comparison.csv), [curvas comparativas]({instance}/plots/objective_convergence_mean_ci95.png), "
                  f"[trayectorias]({instance}/plots/z3_z4_trajectories.png).", ""]
    lines += ["### Diferencias finales pareadas", "", "Valores positivos favorecen GV+S. `p Holm` corrige los 20 contrastes; los IC mostrados son puntuales.", ""]
    view = paired.copy();view["reference"] = view.reference.map(SHORT)
    lines += [markdown_table(view, ["instance", "reference", "mean", "ci95_low", "ci95_high", "p_holm_20", "sd_ratio", "optimization_time_ratio"],
                            ["Instancia", "Referencia", "Diferencia", "IC95% inferior", "IC95% superior", "p Holm", "SD nueva / referencia", "Tiempo nueva / referencia"]), "",
              "[Figuras agrupadas de las cinco instancias](comparisons/020/plots/objective_convergence_mean_ci95_comparison.png), "
              "incluyendo objetivo/z3/z4, ventana 2–25, trayectorias y diferencias finales.", "",
              "## E. Análisis científico", ""]
    questions = (("1. ¿Mejora a Green Variance?", "green_variance"), ("2. ¿Mejora a Green Softmax?", "green_softmax"))
    for question, reference in questions:
        lines += [f"### {question}", ""]
        for row in paired[paired.reference == reference].itertuples():
            finding = ("mejora estadísticamente respaldada tras Holm" if row.p_holm_20 < .05 and row.mean > 0 else
                       "deterioro estadísticamente respaldado tras Holm" if row.p_holm_20 < .05 else
                       "diferencia sin respaldo tras el ajuste Holm; no demuestra equivalencia")
            lines += [f"- **{row.instance}:** Δ={row.mean:.5f}, IC95% [{row.ci95_low:.5f}, {row.ci95_high:.5f}], p Holm={row.p_holm_20:.5g}: {finding}."]
        lines += [""]
    lines += ["### 3. ¿Presenta una convergencia más estable?", "",
              "Se cuantificó la distancia RMS de las decisiones a su media en las últimas 20 posiciones, además de longitud total y último paso. "
              "Un valor menor describe menos movimiento, no garantiza convergencia al óptimo. Los siguientes contrastes son exploratorios y puntuales:", ""]
    for row in paired[paired.reference.isin(("green_variance", "green_softmax"))].itertuples():
        lines += [f"- {row.instance}, RMS de cola GV+S − {SHORT[row.reference]}: {row.tail_rms_difference_mean:.5f}, "
                  f"IC95% [{row.tail_rms_difference_ci95_low:.5f}, {row.tail_rms_difference_ci95_high:.5f}]."]
    lines += ["", "### 4. ¿Reduce la variabilidad entre replicaciones?", "",
              "Se compara la desviación estándar del objetivo final independiente entre las 50 réplicas, no la varianza interna del filtro. "
              "Las razones <1 favorecen menor dispersión de GV+S. IC bootstrap exploratorio, sin ajuste múltiple:", ""]
    for row in paired[paired.reference.isin(("green_variance", "green_softmax"))].itertuples():
        lines += [f"- {row.instance} frente a {SHORT[row.reference]}: razón SD={row.sd_ratio:.4f}, "
                  f"IC95% bootstrap [{row.sd_ratio_boot_ci95_low:.4f}, {row.sd_ratio_boot_ci95_high:.4f}]."]
    lines += ["", "### 5. ¿Produce mejores soluciones finales bajo evaluación independiente?", "",
              "La respuesta depende de la instancia y referencia: los contrastes anteriores utilizan el holdout independiente, no el estimador interno ni solo las curvas suavizadas. "
              "Se consideran respaldadas las mejoras cuyo contraste pareado conserva p Holm<0.05. Un intervalo que incluye cero o un contraste no significativo no acredita ausencia de efecto.", "",
              "### 6. ¿Cambia su comportamiento entre instancias?", ""]
    for instance in INSTANCES:
        g = tables[tables.instance == instance];best = g.loc[g.objective_final_mean.idxmax()]
        new = g[g.method == METHOD].iloc[0]
        lines += [f"- {instance}: GV+S tiene media final {new.objective_final_mean:.4f}, z final medio ({new.z3_final_mean:.4f}, {new.z4_final_mean:.4f}); "
                  f"el mayor promedio descriptivo entre los cinco métodos es {SHORT[best['method']]} ({best.objective_final_mean:.4f})."]
    l026 = pd.read_csv(BASE / "L026/summaries/final_results_all.csv")
    l026 = l026[l026.method == METHOD].sort_values("objective")
    worst = l026.head(2)
    lines += ["", f"**L026 requiere especial atención:** el objetivo final de GV+S tiene mediana {l026.objective.median():.5f}, "
              f"media {l026.objective.mean():.5f} y SD {l026.objective.std(ddof=1):.5f}. "
              + "Las dos soluciones finales más bajas son "
              + "; ".join(f"rep{int(r.replication):02d}: J={r.objective:.5f}, z=({r.z3:.5f},{r.z4:.5f})" for r in worst.itertuples())
              + ". Estas observaciones se conservaron en todas las estadísticas. Los métodos Green históricos también presentan resultados extremos en esta instancia; "
              "Fresh tiene una SD final de 1.70988. Las bandas amplias y los IC bootstrap de razones SD reflejan esta sensibilidad a réplicas poco frecuentes. "
              "La combinación propuesta no resuelve de forma general la inestabilidad de esta instancia."]
    lines += ["", "### 7. ¿Justifica el costo computacional adicional?", "",
              "El filtrado sigue evaluando Green completo para la política provisional y las varianzas de todos los bancos; Softmax añade la combinación de 144 gradientes. "
              "La persistencia completa de bancos del 021 añade E/S que los históricos no realizaban. Se informan por separado tiempo de optimización y tiempo completo, sin atribuir toda diferencia observada al cálculo Softmax.", ""]
    for row in paired[paired.reference == "green_variance"].itertuples():
        verdict = "hay una mejora de calidad respaldada que puede justificar el costo según el uso" if row.p_holm_20 < .05 and row.mean > 0 else "estos datos no justifican el costo por una mejora de calidad respaldada frente a GV"
        lines += [f"- {row.instance}: Δ final={row.mean:.4f}; razón de tiempo de búsqueda={row.optimization_time_ratio:.3f}; "
                  f"razón de tiempo completo={row.end_to_end_time_ratio:.3f}. {verdict}."]
    lines += ["", "La relevancia práctica depende de la escala económica y frecuencia de uso, para las que no se especificó un umbral mínimo de mejora. "
              "No se declara superioridad general ni se usa la apariencia visual de suavidad como evidencia.", "",
              "## F. Verificación, artefactos y problemas", "",
              f"- Replicaciones nuevas completas: **250/250**; 25.000 actualizaciones y 25.250 estados de decisión; 12.500.000 trayectorias nuevas de búsqueda.",
              "- Pruebas previas: filtro original idéntico para observaciones y decisiones comunes; likelihood ratios Bernoulli reconstruidos; scores comprobados por diferencias finitas; gradiente y costo; pesos estables; repetición determinista; continuación por checkpoint sin regenerar bancos previos.",
              "- Verificación de producción: hashes de los 25.000 bancos y todos los CSV/checkpoints publicados; 1.262.500 decisiones de aceptación por banco; 3.600.000 pesos de política; todas las identidades de Softmax y gradiente, tamaños, semillas y finitud.",
              f"- Conservación: **{preservation['checked_files']} archivos históricos y fuentes originales comprobados por SHA256, sin cambios**.",
              "- Reanudación de la campaña completa: 500 manifiestos publicados idénticos, cero nuevas simulaciones y cero replicaciones de producción parciales; evidencia en `validation/completed_campaign_resume.json`.",
              f"- Figuras: **{figure_count} figuras, {2*figure_count} archivos PNG/PDF**, con gráficos individuales de métodos, comparativos por instancia, agrupados, rep01–rep03 y diagnósticos.",
              "- Datos: `INSTANCIA/raw/green_variance_softmax/rep_XXXX/` contiene trayectoria, evaluación final, diagnósticos, estimaciones por política, semillas, bancos, checkpoints y manifiestos de integridad. `summaries/` contiene media, SD, SE, IC95%, tablas finales, estabilidad y diferencias pareadas.",
              "- Los tiempos históricos son descriptivos por sesiones y contención distintas. El rechazo de bancos históricos no aplica a Fresh. No se inventaron indicadores faltantes.",
              "- Problema detectado en pruebas previas: la copia congelada 019 de I0252 no define la constante `INSTANCE` introducida después en 020. El módulo nuevo utiliza el identificador original común `V.ID`; se corrigió antes de iniciar producción y todas las pruebas posteriores pasaron. No se editó ninguna fuente histórica.",
              "- El entorno Python emite avisos de Axes3D no disponible y versiones antiguas de aceleradores opcionales numexpr/bottleneck. Solo se usan gráficos 2D y se validaron los resultados numéricos; no se modificaron paquetes ni el entorno histórico.",
              "- Se detectó que I0252 del 020 remite a datos del 019 y que las semillas 017/020 son distintas; ambas diferencias se trataron con procedencia explícita, sin mezclar muestras.", "",
              "## G. Reproducción y reanudación", "", "Desde este directorio:", "", "```bash",
              "python3 code/run_021.py test", "python3 code/run_021.py run --workers 12",
              "python3 code/analyze_021.py", "python3 code/run_021.py verify-history", "```", "",
              "`run` valida y omite las réplicas completas. Una interrupción conserva bancos y estado por iteración, búsqueda, traza y evaluación final; al reanudar solo se calcula lo pendiente. "
              "La configuración y el código numérico quedan congelados al generar el primer banco de producción. El análisis solo lee datos guardados y escribe dentro de 021.", ""]
    atomic_bytes(BASE / "RESUMEN_EXPERIMENTAL_021.md", "\n".join(lines).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-bank-hashes", action="store_true", help="For analysis reruns after a full verified audit only")
    args = parser.parse_args()
    data = {};tables = [];pairs = [];checks = [];global_seeds = []
    for instance in INSTANCES:
        import toml
        audit = toml.load(BASE / instance / "validation/persisted_banks_audit.toml")
        assert audit["passed"] and audit["saved_banks_checked"] == 5000 and audit["new_simulations"] == 0
        print(f"LOAD {instance}", flush=True)
        tr, finals, results = load(instance, verify_banks=not args.skip_bank_hashes)
        for _, meta, _, _ in results:
            if meta["method"] == METHOD:
                s=meta["seeds"]
                global_seeds.extend([*s["search"], s["trace"], s["selection"], s["holdout"]])
        print(f"VERIFY {instance}", flush=True);checks.append(verify_numeric(instance))
        stats, table, pair, stability = statistics(instance, tr, finals)
        instance_figures(instance, tr, finals, stats)
        if instance == "I0252":
            individual_i0252(tr)
        data[instance] = tr, finals, stats;tables.append(table);pairs.append(pair)
        atomic_json(BASE / instance / "summaries/analysis_manifest.json", dict(
            fingerprint=config(BASE / instance)["fingerprint"], completed=dict.fromkeys(METHODS, 50),
            historical_source=str(historical(instance)), statistical_unit="independent replication",
            confidence_protocol="Same moments() as 019/020, pointwise Student-t 95%, ddof=1",
            source_manifests={str(path): sha(path / "complete.toml") for path, _, _, _ in results},
            historical_reruns=0))
        print(f"ANALYZED {instance}", flush=True)
    assert len(global_seeds) == len(set(global_seeds)) == 25750
    table = pd.concat(tables, ignore_index=True);paired = holm(pd.concat(pairs, ignore_index=True))
    save_csv(BASE / "comparisons/020/summaries/final_comparison.csv", table)
    save_csv(BASE / "comparisons/020/summaries/paired_final_differences.csv", paired)
    for instance in INSTANCES:
        save_csv(BASE / instance / "summaries/paired_final_differences.csv", paired[paired.instance == instance])
    individual = compare017();grouped_figures(data, paired)
    preservation = verify_history()
    figures = sorted(BASE.rglob("*.png"))
    assert len(figures) == 135, f"Expected 135 figures; found {len(figures)}"
    for path in figures:
        assert path.with_suffix(".pdf").is_file()
    report(table, paired, individual, checks, preservation, len(figures))
    manifest = dict(complete=True, new_replications=dict.fromkeys(INSTANCES, 50), total_new_replications=250,
                    figures=len(figures), png_files=len(figures), pdf_files=len(figures),
                    historical_preservation=preservation, production_checks=checks,
                    unique_statistical_streams=25750,
                    bank_audits={i: sha(BASE/i/"validation/persisted_banks_audit.toml") for i in INSTANCES},
                    source_hashes={p.name: sha(p) for p in Path(__file__).parent.glob("*") if p.is_file()},
                    artifacts={str(p.relative_to(BASE)): sha(p) for p in BASE.rglob("*")
                               if p.is_file() and p.suffix in (".csv", ".png", ".pdf", ".md")})
    atomic_json(BASE / "COMPLETE.json", manifest)
    print(table.to_string(index=False));print(paired[["instance", "reference", "mean", "ci95_low", "ci95_high", "p_holm_20"]].to_string(index=False))


if __name__ == "__main__":
    main()
