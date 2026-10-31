"""Verify stored campaign data, summaries and preserved history without simulation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import toml

from i0252_statistical_storage_019 import DEFAULT_OUTPUT, ROOT, METHODS, atomic_json, config, csv_rows, load_complete, sha


def verify(output):
    c = config(output)
    science = c["science"]
    results = load_complete(output, c)
    ledger = json.loads((output / "config/requests.json").read_text())
    counts = {m: sum(meta["method"] == m for _, meta, _, _ in results) for m in METHODS}
    for method in METHODS:
        ids = sorted(meta["replication"] for _, meta, _, _ in results if meta["method"] == method)
        assert ids == list(range(1, len(ids) + 1)), (method, ids)
        assert counts[method] >= ledger["targets"][method], (method, counts[method], ledger["targets"][method])
    instance = science["scenario"]
    alpha = np.asarray(toml.load(output / f"config/{instance}.toml")["model"]["ALPHA"])
    policies = {int(p["policy"]): p for p in csv_rows(output / "config/policies.csv")}
    same_points = {}
    checked_updates = 0
    for path, meta, rows, final in results:
        assert meta["science"] == science
        for left, right in zip(rows, rows[1:]):
            k = int(right["iteration"])
            z = np.array([float(left["z3"]), float(left["z4"])])
            gradient = np.array([float(left["g3"]), float(left["g4"])])
            actual = np.array([float(right["z3"]), float(right["z4"])])
            expected = np.clip(z + science["BASE_STEP"] / np.sqrt(k) * np.clip(gradient, -science["GRAD_CLIP"], science["GRAD_CLIP"]), 0, 1)
            assert np.allclose(actual, expected, rtol=0, atol=5e-15), (path, k)
            checked_updates += 1
        assert rows[-1]["objective_internal"] == rows[-1]["g3"] == rows[-1]["g4"] == ""
        for row in rows:
            assert int(row["evaluation_paths"]) == science["trace_paths"]
            assert int(row["cumulative_new_search_paths"]) == int(row["iteration"]) * science["n_paths"]
            key = (meta["replication"], float(row["z3"]), float(row["z4"]))
            value = (float(row["objective_crn"]), int(row["crn_selected_policy"]), float(row["crn_selected_policy_se"]))
            if key in same_points:
                assert value == same_points[key], ("CRN inconsistency at same point", path)
            same_points[key] = value
        z = np.array([float(final["z3"]), float(final["z4"])])
        assert abs(float(final["probing_cost"]) - alpha @ z) < 1e-12
        assert abs(float(final["objective"]) - (float(final["gross"]) - alpha @ z)) < 1e-12
        p = policies[int(final["selected_policy"])]
        for k in ("av", "aq", "as", "tau"):
            assert float(final["policy_" + k]) == float(p[k])
        assert int(final["selection_seed"]) == meta["seeds"]["selection"]
        assert int(final["holdout_seed"]) == meta["seeds"]["holdout"]
        assert float(final["evaluation_se"]) >= 0
    # Reconstruct every requested per-iteration statistic from the raw records.
    grouped = {}
    for _, meta, rows, _ in results:
        for row in rows:
            for metric in ("objective_crn", "z3", "z4", "cumulative_optimization_seconds"):
                grouped.setdefault((meta["method"], int(row["iteration"]), metric), []).append(float(row[metric]))
    statistics = csv_rows(output / "summaries/iteration_statistics.csv")
    assert len(statistics) == len(grouped)
    for row in statistics:
        values = np.asarray(grouped[(row["method"], int(row["iteration"]), row["metric"])])
        assert len(values) == int(row["n"])
        assert np.isclose(float(row["mean"]), values.mean(), atol=1e-12, rtol=1e-13)
        if len(values) > 1:
            sd = values.std(ddof=1)
            half = student_t.ppf(.975, len(values) - 1) * sd / np.sqrt(len(values))
            for name, expected in (("sd", sd), ("se", sd / np.sqrt(len(values))),
                                   ("ci95_low", values.mean() - half), ("ci95_high", values.mean() + half)):
                assert np.isclose(float(row[name]), expected, atol=1e-12, rtol=1e-13)
    for row in csv_rows(output / "summaries/final_statistics.csv"):
        assert int(row["n"]) == counts[row["method"]]
    analysis = json.loads((output / "summaries/analysis_manifest.json").read_text())
    assert analysis["completed"] == counts and analysis["fingerprint"] == c["fingerprint"]
    expected_stems = ("objective_convergence_mean_ci95", "z3_convergence_mean_ci95", "z4_convergence_mean_ci95", "z3_z4_trajectories")
    figure_hashes = {}
    for stem in expected_stems:
        for extension in ("png", "pdf"):
            p = output / "plots" / f"{stem}.{extension}"
            assert p.stat().st_size > 1000
            figure_hashes[p.name] = sha(p)
    preserved = json.loads((output / "validation/preserved_before.json").read_text())
    changed = [name for name, expected in preserved.items() if sha(ROOT / name) != expected]
    assert not changed, changed
    for name, expected in science["numerical_sources"].items():
        assert sha(output / "reproduction" / name) == expected
    assert sha(output / f"config/{instance}.toml") == science["instance_file_sha256"]
    assert sha(output / "config/policies.csv") == science["policy_grid_sha256"]
    report = {
        "passed": True, "verified_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": c["fingerprint"],
        "completed": counts, "targets": ledger["targets"], "total_complete_optimizations": len(results),
        "trajectory_rows": sum(len(rows) for _, _, rows, _ in results), "projected_updates_checked": checked_updates,
        "new_optimization_paths": len(results) * science["max_iters"] * science["n_paths"],
        "convergence_nominal_path_evaluations": len(results) * (science["max_iters"] + 1) * science["trace_paths"],
        "final_selector_path_evaluations": len(results) * science["selection_paths"],
        "final_holdout_path_evaluations": len(results) * science["holdout_paths"],
        "historical_files_checked_unchanged": len(preserved), "historical_replications_imported": 0,
        "same_point_crn_consistency": True, "gradients_steps_costs_policies_verified": True,
        "iteration_statistics_reconstructed": len(statistics), "figure_sha256": figure_hashes,
        "production_partial_directories": [str(p.relative_to(output)) for m in METHODS for p in (output / "raw" / m).glob(".*.partial")],
        "simulations_started_by_verification": 0,
    }
    atomic_json(output / "validation/final_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify(parser.parse_args().output.resolve())
