"""Functional, incremental and reproducibility checks before production simulation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import toml

from i0252_statistical_storage_019 import ROOT, DEFAULT_OUTPUT, METHODS, atomic_json, config, csv_rows, load_complete, sha
from run_i0252_statistical_019 import initialize
from probing_statistical_instances_020 import canonical_instance

HISTORICAL = ("010probing_instance_landscape_search", "011probing_high_contrast_interior", "013probing_green_simulation_efficiency",
              "013probing_green_softmax010", "016probing_green_bank_variants", "018probing_green_n250_vs_n500",
              "019probing_i0252_four_methods_statistical")


def preserve_manifest(output):
    path = output / "validation/preserved_before.json"
    if not path.exists():
        files = [p for name in HISTORICAL for p in (ROOT / "resultados" / name).rglob("*") if p.is_file()]
        files += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".jl", ".py", ".toml", ".txt", ".md")
                  and "019" not in p.name and "020" not in p.name]
        atomic_json(path, {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)})
    return json.loads(path.read_text())


def run(command, log):
    with open(log, "w") as f:
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def all_hashes(output):
    return {str(p.relative_to(output)): sha(p) for method in METHODS for rep in (1, 2)
            for p in (output / "raw" / method / f"rep_{rep:04d}").rglob("*") if p.is_file()}


def validate(output, instance="I0252"):
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation").mkdir(exist_ok=True)
    initialize(output, instance=instance)
    preserved = preserve_manifest(output)
    print(f"Historical protection manifest: {len(preserved)} files", flush=True)
    env = dict(os.environ, GKSwstype="100", JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    audit_log = output / "validation/algorithm_audit.log"
    with open(audit_log, "w") as f:
        subprocess.run(["julia", "--startup-file=no", "--project=" + str(output / "reproduction"),
                        str(output / "reproduction/019probing_i0252_statistical.jl"), str(output), "audit", str(ROOT)],
                       cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("Original algorithm regression passed", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    testroot = output / "validation" / f"functional_{stamp}"
    testroot.mkdir()
    incremental = testroot / "incremental"
    isolated = testroot / "isolated"
    runner = [sys.executable, str(ROOT / "run_i0252_statistical_019.py")]

    def execute(folder, logname, *options):
        run(runner + ["run", "--instance", instance, "--quick-test", "--output", str(folder), "--workers", "2", "--no-analysis", *options], testroot / logname)

    execute(incremental, "target2.log", "--target", "2")
    assert len(load_complete(incremental)) == 8
    before = all_hashes(incremental)
    execute(incremental, "add3.log", "--add", "3", "--request-id", "validation_add_three")
    assert len(load_complete(incremental)) == 20
    assert before == all_hashes(incremental)
    after = {str(p.relative_to(incremental)): sha(p) for p, _, _, _ in load_complete(incremental) for p in p.rglob("*") if p.is_file()}
    execute(incremental, "repeat_add3.log", "--add", "3", "--request-id", "validation_add_three")
    execute(incremental, "repeat_target5.log", "--target", "5")
    assert len(load_complete(incremental)) == 20
    assert after == {str(p.relative_to(incremental)): sha(p) for p, _, _, _ in load_complete(incremental) for p in p.rglob("*") if p.is_file()}
    print("Incremental 2 + 3, idempotent extension and preservation passed", flush=True)
    execute(isolated, "isolated.log", "--target", "1", "--methods", "fresh", "green", "green_softmax", "green_variance")
    for method in METHODS:
        a = incremental / "raw" / method / "rep_0001"
        b = isolated / "raw" / method / "rep_0001"
        ra, rb = csv_rows(a / "trajectory.csv"), csv_rows(b / "trajectory.csv")
        for x, y in zip(ra, rb):
            assert {k: v for k, v in x.items() if "seconds" not in k} == {k: v for k, v in y.items() if "seconds" not in k}
        fa, fb = csv_rows(a / "final.csv")[0], csv_rows(b / "final.csv")[0]
        assert {k: v for k, v in fa.items() if "seconds" not in k} == {k: v for k, v in fb.items() if "seconds" not in k}
    print("Standalone versus campaign numerical reproducibility passed", flush=True)
    # Simulate interruption after a finished search, retaining the actual checkpoint.
    a = isolated / "raw/fresh/rep_0001"
    search_hash = sha(a / "search.jls")
    partial = a.parent / ".rep_0001.partial"
    a.rename(partial)
    for name in ("complete.toml", "convergence.jls", "final.jls", "trajectory.csv", "final.csv", "metadata.toml"):
        (partial / name).unlink()
    assert len(load_complete(isolated)) == 3
    execute(isolated, "resume_partial.log", "--resume", "--methods", "fresh")
    assert len(load_complete(isolated)) == 4 and sha(a / "search.jls") == search_hash
    # The loader rejects damage rather than silently mixing/omitting a published record.
    damaged = testroot / "damaged"
    shutil.copytree(isolated / "config", damaged / "config")
    shutil.copytree(isolated / "raw/fresh/rep_0001", damaged / "raw/fresh/rep_0001")
    with open(damaged / "raw/fresh/rep_0001/trajectory.csv", "a") as f:
        f.write("corruption\n")
    try:
        load_complete(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt replication accepted")
    # Config changes must fail without touching old data.
    badconfig = testroot / "badconfig"
    shutil.copytree(isolated / "config", badconfig / "config")
    c = toml.load(badconfig / "config/experiment.toml")
    c["science"]["n_paths"] = 21
    (badconfig / "config/experiment.toml").write_text(toml.dumps(c))
    try:
        config(badconfig)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed config accepted")
    run(runner + ["analyze", "--instance", instance, "--output", str(incremental)], testroot / "analysis.log")
    plot_hashes = {p.name: sha(p) for p in (incremental / "plots").iterdir()}
    summary_hashes = {p.name: sha(p) for p in (incremental / "summaries").iterdir()}
    run(runner + ["analyze", "--instance", instance, "--output", str(incremental)], testroot / "analysis_repeat.log")
    assert plot_hashes == {p.name: sha(p) for p in (incremental / "plots").iterdir()}
    assert summary_hashes == {p.name: sha(p) for p in (incremental / "summaries").iterdir()}
    assert len(plot_hashes) == 8
    stats = csv_rows(incremental / "summaries/iteration_statistics.csv")
    values = [float(csv_rows(incremental / "raw/fresh" / f"rep_{r:04d}" / "trajectory.csv")[-1]["objective_crn"]) for r in range(1, 6)]
    row = next(s for s in stats if s["method"] == "fresh" and s["metric"] == "objective_crn" and s["iteration"] == "4")
    assert int(row["n"]) == 5
    assert np.isclose(float(row["mean"]), np.mean(values), atol=1e-13)
    assert np.isclose(float(row["sd"]), np.std(values, ddof=1), atol=1e-13)
    # t(.975,4) is independently fixed here, rather than reusing analysis.moments.
    expected_half = 2.7764451051977987 * np.std(values, ddof=1) / np.sqrt(5)
    assert np.isclose(float(row["ci95_high"]), np.mean(values) + expected_half, atol=1e-11)
    assert before == all_hashes(incremental)
    changed = [p for p, h in preserved.items() if sha(ROOT / p) != h]
    assert not changed, changed
    atomic_json(output / "validation/functional_tests.json", {
        "passed": True, "instance": instance, "test_output": str(testroot), "algorithm_regression": True,
        "two_plus_three_replications_per_method": True, "idempotent_extension": True,
        "first_two_replications_unchanged": True, "isolated_campaign_numerical_reproducibility": True,
        "resume_uses_search_checkpoint": True, "incomplete_results_excluded": True,
        "corruption_rejected": True, "changed_configuration_rejected": True,
        "student_t_interval_verified": True, "analysis_and_plots_byte_reproducible": True,
        "historical_files_checked_unchanged": len(preserved), "production_replications_imported": 0,
        "quick_protocol": {"iterations": 4, "new_paths": 20, "trace": 30, "selector": 40, "holdout": 80},
    })
    print("All functional validations passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--instance", type=canonical_instance, default="I0252")
    args = parser.parse_args()
    validate(args.output.resolve(), args.instance)
