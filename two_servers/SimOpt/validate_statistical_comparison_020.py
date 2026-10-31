"""Validate the extension, then verify completed campaigns without simulating I0252."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import toml

from i0252_statistical_storage_019 import ROOT, DEFAULT_OUTPUT, METHODS, config, load_complete, sha, atomic_json
from probing_statistical_instances_020 import BASE_OUTPUT, INSTANCES, canonical_instance, original_instance
from run_i0252_statistical_019 import initialize


def preserved_i0252(base):
    baseline_path = base / "validation/preserved_i0252_before.json"
    if not baseline_path.exists():
        files = [p for p in DEFAULT_OUTPUT.rglob("*") if p.is_file()]
        atomic_json(baseline_path, {str(p.relative_to(ROOT / "resultados")): sha(p) for p in files})
    before = json.loads(baseline_path.read_text())
    checked = 0
    code_changes = {}
    for name, expected in before.items():
        actual = sha(ROOT / "resultados" / name)
        if name.startswith("../"):
            if actual != expected:
                code_changes[name] = {"before": expected, "after": actual}
        else:
            if actual != expected:
                raise ValueError(f"I0252 artifact changed: {name}")
            checked += 1
    results = load_complete(DEFAULT_OUTPUT)
    assert len(results) == 200
    assert all(sum(meta["method"] == m for _, meta, _, _ in results) == 50 for m in METHODS)
    # Initialization of an existing campaign is read-only and retains its frozen code.
    initialize(DEFAULT_OUTPUT, instance="I0252")
    archived = DEFAULT_OUTPUT / "reproduction"
    baseline = config(DEFAULT_OUTPUT)["science"]["numerical_sources"]
    for name, expected in baseline.items():
        if name != "019probing_i0252_statistical.jl":
            assert sha(ROOT / name) == expected, ("Original algorithm source changed", name)
    old = (archived / "019probing_i0252_statistical.jl").read_text()
    new = (ROOT / "019probing_i0252_statistical.jl").read_text()
    for first, following in (("search", "seeds"), ("direct_trace", "endpoint"), ("endpoint", "run_rep")):
        assert old.split(f"function {first}(", 1)[1].split(f"function {following}(", 1)[0] == \
               new.split(f"function {first}(", 1)[1].split(f"function {following}(", 1)[0]
    return {"artifact_files_unchanged": checked, "complete_replications_read": len(results),
            "i0252_optimizations_executed": 0, "generalized_code_changes": code_changes}


def configurations(base):
    report = {}
    for instance in INSTANCES:
        original = original_instance(instance)
        c = initialize(base / instance, instance=instance)
        d = c["science"]
        md = toml.load(base / instance / f"config/{instance}.toml")["model"]
        assert md == original["model"]
        assert md != toml.load(DEFAULT_OUTPUT / "config/I0252.toml")["model"]
        assert d["model_sha256"] == original["model_sha256"]
        for name, binding in d["instance_bindings"].items():
            adapted = (base / instance / "reproduction" / name).read_bytes()
            old, new = binding["old"].encode(), binding["new"].encode()
            assert adapted.count(new) == 1
            restored = adapted.replace(new, old)
            assert hashlib.sha256(restored).hexdigest() == d["original_numerical_sources"][name]
        report[instance] = {"model_sha256": d["model_sha256"], "model": md,
                            "configuration_cross_checks": original["cross_checks"],
                            "reference_grid_best": original["reference_grid_best"],
                            "initial_point": d["Z0"], "policy_grid_sha256": d["policy_grid_sha256"],
                            "only_wrapper_id_declarations_changed": True}
    assert canonical_instance("L30") == canonical_instance("L030") == "L030"
    assert not (base / "L30").exists()
    # Explicitly check all intended RNG stream IDs, including independence from I0252.
    seeds = []
    for instance in (*INSTANCES, "I0252"):
        for rep in range(1, 51):
            for stream, iterations in (("search", range(1, 101)), ("trace", [0]), ("selection", [0]), ("holdout", [0])):
                for iteration in iterations:
                    message = f"Probing010-v1/{instance}/compare/{19000000+rep}/{stream}/{iteration}"
                    seeds.append(int(hashlib.sha256(message.encode()).hexdigest()[:15], 16) + 1)
    assert len(seeds) == len(set(seeds)) == 5 * 50 * 103
    atomic_json(base / "validation/configuration_audit.json", {"passed": True, "instances": report,
                "canonical_alias": {"L30": "L030"}, "distinct_rng_seeds_checked": len(seeds)})
    return report


def functional(base):
    (base / "validation").mkdir(parents=True, exist_ok=True)
    compatibility = preserved_i0252(base)
    configurations(base)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_output = base / "validation" / f"extension_{run_name}"
    run_output.mkdir()

    def one(instance):
        path = base / "validation" / f"functional_{instance}.log"
        with open(path, "w") as log:
            subprocess.run([sys.executable, str(ROOT / "validate_i0252_statistical_019.py"), "--instance", instance,
                            "--output", str(base / instance)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Validated {instance}: exact historical replay, Filter2, storage, 2+3, isolated replay, resume, t intervals and plots", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, INSTANCES))
    quick = {instance: Path(json.loads((base / instance / "validation/functional_tests.json").read_text())["test_output"]) / "incremental"
             for instance in INSTANCES}
    # Extending one method affects neither other methods nor other instances.
    output = quick["L026"]
    old = {str(p): sha(p) for m in METHODS for p in (output / "raw" / m).glob("rep_*/*") if p.is_file()}
    command = [sys.executable, str(ROOT / "run_i0252_statistical_019.py"), "run", "--instance", "L026", "--quick-test",
               "--output", str(output), "--add", "1", "--request-id", "variance_only_increment", "--methods", "green_variance", "--no-analysis"]
    with open(run_output / "method_only_increment.log", "w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert all(sha(Path(p)) == expected for p, expected in old.items())
    results = load_complete(output)
    assert {m: sum(meta["method"] == m for _, meta, _, _ in results) for m in METHODS} == \
           {"fresh": 5, "green": 5, "green_softmax": 5, "green_variance": 6}
    assert all(len(load_complete(quick[i])) == 20 for i in INSTANCES if i != "L026")
    # A complete foreign result must be rejected, not blended into another instance.
    foreign = run_output / "foreign_instance"
    shutil.copytree(quick["L025"] / "config", foreign / "config")
    shutil.copytree(quick["L000"] / "raw/fresh/rep_0001", foreign / "raw/fresh/rep_0001")
    try:
        load_complete(foreign)
    except ValueError:
        pass
    else:
        raise AssertionError("Foreign instance result accepted")
    try:
        initialize(quick["L000"], quick=True, instance="L025")
    except ValueError:
        pass
    else:
        raise AssertionError("Instance mismatch accepted")
    # Exercise the cross-instance table and figures on separate small test data.
    combined = run_output / "combined_quick"
    combined.mkdir()
    for instance in INSTANCES:
        (combined / instance).symlink_to(quick[instance], target_is_directory=True)
    from analyze_statistical_comparison_020 import comparison
    comparison(combined, refresh=True)
    table = __import__("pandas").read_csv(combined / "summaries/final_comparison.csv")
    assert len(table) == 16
    assert int(table[(table.instance == "L026") & (table.method == "green_variance")].replications.iloc[0]) == 6
    assert len(list((combined / "plots").glob("*.png"))) == 3
    compatibility = preserved_i0252(base)
    atomic_json(base / "validation/extension_tests.json", {"passed": True, "instances_validated": list(INSTANCES), "test_output": str(run_output),
                "i0252_compatibility": compatibility, "one_method_incremental_extension": True,
                "foreign_instance_rejected": True, "cross_instance_table_and_panels": True,
                "statistical_criterion_unchanged": True, "production_optimizations_run_during_validation": 0})
    print("All extension validations passed; no I0252 optimizations executed", flush=True)


def final_verify(base):
    from verify_i0252_statistical_019 import verify
    reports = {}
    all_streams = []
    for instance in INSTANCES:
        assert config(base / instance)["science"]["scenario"] == instance
        verify(base / instance)
        report = json.loads((base / instance / "validation/final_verification.json").read_text())
        reports[instance] = report
        seen = set()
        for _, meta, _, _ in load_complete(base / instance):
            if meta["replication"] in seen:
                continue
            seen.add(meta["replication"])
            stream = meta["seeds"]
            all_streams.extend([*stream["search"], stream["trace"], stream["selection"], stream["holdout"]])
    assert len(all_streams) == len(set(all_streams))
    compatibility = preserved_i0252(base)
    import pandas as pd
    table = pd.read_csv(base / "summaries/final_comparison.csv")
    assert len(table) == 16
    for row in table.itertuples():
        assert row.replications == reports[row.instance]["completed"][row.method]
    assert len(list((base / "plots").glob("*.png"))) == 3
    assert len(list((base / "plots").glob("*.pdf"))) == 3
    report = {"passed": True, "completed": {i: r["completed"] for i, r in reports.items()},
              "total_complete_optimizations": sum(r["total_complete_optimizations"] for r in reports.values()),
              "trajectory_rows": sum(r["trajectory_rows"] for r in reports.values()),
              "new_optimization_paths": sum(r["new_optimization_paths"] for r in reports.values()),
              "i0252_compatibility": compatibility, "cross_instance_seed_collisions": 0,
              "comparison_rows": len(table), "individual_figures": 16, "comparison_figures": 3,
              "new_simulations_started_by_verification": 0}
    atomic_json(base / "validation/final_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()
    (final_verify if args.final_only else functional)(args.base.resolve())
