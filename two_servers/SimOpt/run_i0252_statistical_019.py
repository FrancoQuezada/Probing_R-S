"""Run, extend, resume or analyze the original four-method protocol by instance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid

import toml

from i0252_statistical_storage_019 import (
    ROOT, DEFAULT_OUTPUT, METHODS, atomic_bytes, atomic_json, config, fingerprint, load_complete, sha, validate_rep,
)
from probing_statistical_instances_020 import canonical_instance, default_output, original_instance

NUMERICAL_SOURCES = (
    "019probing_i0252_statistical.jl", "016probing_green_bank_variants.jl",
    "012probing_anytime_budget_analysis.jl", "probing_anytime_012_runtime.jl",
    "011probing_high_contrast_interior.jl", "probing_interior_011_diagnostics.jl",
    "010probing_instance_landscape_search.jl", "probing_regime_model.jl",
    "probing_experiments_common.jl", "003probing_four_searches_softmax.jl",
)
PYTHON_SOURCES = ("run_i0252_statistical_019.py", "i0252_statistical_storage_019.py",
                  "analyze_i0252_statistical_019.py", "validate_i0252_statistical_019.py", "verify_i0252_statistical_019.py",
                  "probing_statistical_instances_020.py")
SOURCE = ROOT / "resultados/011probing_high_contrast_interior"
ENVIRONMENT = ROOT / "resultados/003probing_softmax_20261021/entorno"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def initialize(output, quick=False, instance="I0252"):
    instance = canonical_instance(instance)
    path = output / "config/experiment.toml"
    if path.exists():
        c = config(output)
        if c["science"]["scenario"] != instance:
            raise ValueError("Instance mismatch; results from different instances cannot be mixed")
        if c["science"]["quick_test"] != quick:
            raise ValueError("Quick/full protocol mismatch; use a different output directory")
        version = subprocess.check_output(["julia", "--startup-file=no", "-e", "print(VERSION)"], text=True)
        if version != c["science"]["julia_version"]:
            raise ValueError(f"Julia version mismatch: campaign requires {c['science']['julia_version']}, found {version}")
        for name, expected in c["science"]["numerical_sources"].items():
            if sha(output / "reproduction" / name) != expected:
                raise ValueError(f"Frozen numerical source changed: {name}")
        # Execute the frozen numerical code. Later original-source edits do not affect this campaign.
        if sha(output / f"config/{instance}.toml") != c["science"]["instance_file_sha256"]:
            raise ValueError(f"Changed {instance} instance configuration")
        for name, expected in c["science"]["environment"].items():
            if sha(output / "reproduction" / name) != expected:
                raise ValueError(f"Changed Julia environment: {name}")
        return c
    for folder in ("config", "raw", "summaries", "plots", "logs", "validation", "reproduction"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    reference = original_instance(instance) if instance != "I0252" else None
    instance_bytes = (toml.dumps({"scenario": instance, "model": reference["model"]}).encode()
                      if reference else (SOURCE / "configs/main_instance_parameters.toml").read_bytes())
    policies = reference["policies"] if reference else SOURCE / "configs/policies.csv"
    surface = reference["surface"] if reference else SOURCE / "high/I0252_surface.csv"
    # Bind only the two historical wrapper IDs in the frozen copies. All function
    # bodies, estimators and reductions remain byte-for-byte identical to 019.
    bindings = ({name: {"old": 'const ID="I0252"', "new": f'const ID="{instance}"'}
                 for name in ("012probing_anytime_budget_analysis.jl", "016probing_green_bank_variants.jl")}
                if reference else {})
    numerical_bytes = {}
    for name in NUMERICAL_SOURCES:
        data = (ROOT / name).read_bytes()
        if name in bindings:
            old, new = bindings[name]["old"].encode(), bindings[name]["new"].encode()
            if data.count(old) != 1:
                raise ValueError(f"Unexpected wrapper ID declaration: {name}")
            data = data.replace(old, new)
        numerical_bytes[name] = data
        (output / "reproduction" / name).write_bytes(data)
    science = {
        "protocol": "I0252-four-methods-v1", "quick_test": quick, "scenario": instance, "methods": list(METHODS),
        "max_iters": 4 if quick else 100, "n_paths": 20 if quick else 500, "jobs": 45,
        "trace_paths": 30 if quick else 1000, "selection_paths": 40 if quick else 10000,
        "holdout_paths": 80 if quick else 50000, "Z0": [.12, .10], "BASE_STEP": .035, "GRAD_CLIP": 15.,
        "policies": 144, "beta_softmax": .5, "kappa": 2., "lr_log_clip": 35., "probability_clamp": 1e-12,
        "model_sha256": reference["model_sha256"] if reference else "1ecdc3dc6161f7783a79a4e9d575f1f753a6fbb1ff4fb0d9337af56d5da54fd7",
        "instance_source": str(reference["source"].relative_to(ROOT)) if reference else "resultados/011probing_high_contrast_interior/configs/main_instance_parameters.toml",
        "instance_file_sha256": hashlib.sha256(instance_bytes).hexdigest(),
        "policy_grid_sha256": sha(policies),
        "objective": "max_i E_z[Y_i] - alpha'z; common ex-ante routing policy",
        "seed_protocol": f"Original Probing010-v1/{instance}/compare/(19000000+rep)/stream/iteration; SHA256 first 15 hex + 1",
        "optimization_stream_shared_across_methods": True,
        "convergence_evaluation": "Direct simulation, same primitive bank within rep, rematerialized at each z; hard maximum of policy sample means",
        "final_evaluation": "Independent selector followed by selected-policy holdout; shared CRN across methods within rep",
        "confidence_protocol": "Pointwise two-sided Student-t 95% CI of independent replication means; ddof=1; no multiplicity correction",
        "timing_protocol": "Per-process warmed wall time, BLAS=1; concurrent worker contention possible; diagnostics and evaluations separate",
        "numerical_sources": {name: sha(output / "reproduction" / name) for name in NUMERICAL_SOURCES},
        "original_numerical_sources": {name: sha(ROOT / name) for name in NUMERICAL_SOURCES},
        "instance_bindings": bindings,
        "environment": {name: sha(ENVIRONMENT / name) for name in ("Project.toml", "Manifest.toml")},
        "julia_version": subprocess.check_output(["julia", "--startup-file=no", "-e", "print(VERSION)"], text=True),
    }
    for name in PYTHON_SOURCES:
        shutil.copyfile(ROOT / name, output / "reproduction" / name)
    for name in ("Project.toml", "Manifest.toml"):
        shutil.copyfile(ENVIRONMENT / name, output / "reproduction" / name)
    atomic_bytes(output / f"config/{instance}.toml", instance_bytes)
    shutil.copyfile(policies, output / "config/policies.csv")
    if surface is not None and surface.exists():
        shutil.copyfile(surface, output / f"config/{instance}_reference_surface.csv")
    if reference:
        atomic_bytes(output / "config/original_model_entry.toml", toml.dumps({instance: reference["original_entry"]}).encode())
        science["instance_model_table"] = instance
        science["historical_models_sha256"] = reference["source_sha256"]
        science["reference_grid_best"] = reference["reference_grid_best"] or []
        science["reference_surface_sha256"] = sha(surface) if surface else "unavailable"
    c = {"fingerprint": fingerprint(science), "created_utc": utcnow(), "science": science}
    atomic_bytes(path, toml.dumps(c).encode())
    atomic_json(output / "config/provenance.json", {
        "created_utc": utcnow(), "git_commit": None, "git_note": "No .git directory in supplied workspace; numerical source SHA256 identifies code",
        "historical_replications_imported": 0, "source_root": str(ROOT),
        "differences": ([
            "011/012: identical I0252 model, policy grid, optimizer and evaluator; new seed replication namespace +19000000",
            "013 simulation-efficiency study: retain N=500 and T=100, exclude N=100/250 and shorter horizons",
            "013 Softmax010: beta=0.1 differs; reuse GreenSoftmax050 from 011/012, beta=0.5",
            "016: Filter2 exactly retained, including provisional full-Green policy, per-bank objective contribution variance, <=2*current variance, forced current bank, equal weights",
            "018: N=250 searches excluded; use original N=500 protocol",
            "No historical data imported; all campaign replications generated anew",
        ] if reference is None else [
            "Identical 019 optimization/evaluation/statistical protocol; exact original instance parameters from 010, cross-checked against 011",
            "Only wrapper instance-ID constants are bound in frozen copies; every algorithm function body remains unchanged",
            "Instance ID is part of the original seed hash; no streams shared between different instances",
            "No historical replications imported; archived grid optima are descriptive references only",
        ]),
        "configuration_cross_checks": reference["cross_checks"] if reference else [],
    })
    return c


def plan(args, output, completed):
    path = output / "config/requests.json"
    ledger = json.loads(path.read_text()) if path.exists() else {"targets": {m: 0 for m in METHODS}, "requests": {}}
    methods = args.methods or list(METHODS)
    if args.target is not None:
        if args.target < 1:
            raise ValueError("--target must be positive")
        for m in methods:
            ledger["targets"][m] = max(ledger["targets"][m], args.target)
    elif args.add is not None:
        if args.add < 1:
            raise ValueError("--add must be positive")
        request_id = args.request_id or str(uuid.uuid4())
        specification = {"add": args.add, "methods": sorted(methods)}
        if request_id in ledger["requests"]:
            if ledger["requests"][request_id]["specification"] != specification:
                raise ValueError("Request ID already used with a different extension")
            print(f"Resuming extension {request_id}; no new IDs allocated", flush=True)
        else:
            for m in methods:
                highest = max((r for mm, r in completed if mm == m), default=0)
                ledger["targets"][m] = max(highest, ledger["targets"][m]) + args.add
            ledger["requests"][request_id] = {"specification": specification, "targets": dict(ledger["targets"]), "created_utc": utcnow()}
            print(f"Extension ID: {request_id}", flush=True)
    elif not args.resume:
        for m in methods:
            ledger["targets"][m] = max(ledger["targets"][m], 50)
    atomic_json(path, ledger)
    # Rotate methods each replication to balance resident workers without changing RNG.
    pending = []
    for r in range(1, max(ledger["targets"].values(), default=0) + 1):
        rotated = methods[(r - 1) % len(methods):] + methods[:(r - 1) % len(methods)]
        for m in rotated:
            if r <= ledger["targets"][m] and (m, r) not in completed:
                pending.append((m, r))
    return ledger, pending


def simulate(output, cfg, pending, workers, counts, targets):
    tasks = queue.Queue()
    for item in pending:
        tasks.put(item)
    failures = []
    mutex = threading.Lock()
    start = time.monotonic()
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    env = dict(os.environ, GKSwstype="100", JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = ["julia", "--startup-file=no", "--project=" + str(output / "reproduction"),
               str(output / "reproduction/019probing_i0252_statistical.jl"), str(output)]

    def worker(index):
        logpath = output / "logs" / f"{session}_worker{index}.log"
        with open(logpath, "x") as log:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env, cwd=ROOT)
            current = None
            try:
                for line in proc.stdout:
                    log.write(line); log.flush()
                    if line.rstrip() == "READY":
                        break
                else:
                    raise RuntimeError(f"Julia worker startup failed; see {logpath}")
                while True:
                    try:
                        current = tasks.get_nowait()
                    except queue.Empty:
                        break
                    m, r = current
                    with mutex:
                        print(f"RUNNING {m} rep_{r:04d}; complete={counts[m]}/{targets[m]}; elapsed={time.monotonic()-start:.1f}s", flush=True)
                    proc.stdin.write(f"{m}\t{r}\n"); proc.stdin.flush()
                    for line in proc.stdout:
                        log.write(line); log.flush()
                        if line.startswith("SEARCH\t"):
                            with mutex:
                                print(f"EVALUATING {m} rep_{r:04d}; optimization={line.strip().split(chr(9))[-1]}s", flush=True)
                        if line.startswith(("COMPLETE\t", "FAILED\t")):
                            fields = line.rstrip().split("\t")
                            if fields[1:] != [m, str(r)]:
                                raise RuntimeError("Worker/task identity mismatch")
                            if fields[0] == "COMPLETE":
                                validate_rep(output / "raw" / m / f"rep_{r:04d}", cfg, m, r)
                                with mutex:
                                    counts[m] += 1
                                    print(f"COMPLETE {m} rep_{r:04d}; complete={counts[m]}/{targets[m]}; elapsed={time.monotonic()-start:.1f}s", flush=True)
                            else:
                                with mutex:
                                    failures.append([m, r, str(logpath)])
                                    print(f"FAILED {m} rep_{r:04d}; see {logpath}", flush=True)
                            break
                    else:
                        raise RuntimeError("Julia worker exited during a replication")
                    tasks.task_done()
                    current = None
            except Exception as exc:
                with mutex:
                    failures.append([*(current or ("worker", index)), str(exc), str(logpath)])
                    print(f"WORKER ERROR: {exc}", flush=True)
            finally:
                if proc.stdin and not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except BrokenPipeError:
                        pass
                if proc.stdout:
                    for line in proc.stdout:
                        log.write(line)
                code = proc.wait()
                if code:
                    with mutex:
                        failures.append(["worker_exit", index, code, str(logpath)])

    with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
        list(executor.map(worker, range(1, min(workers, len(pending)) + 1)))
    if not tasks.empty():
        failures.append(["unscheduled_tasks", tasks.qsize()])
    return failures, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "analyze", "status", "validate"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--instance", type=canonical_instance, default="I0252", help="I0252, L000, L025, L026 or L030 (L30 is an alias)")
    parser.add_argument("--quick-test", action="store_true", help="Separate reduced validation campaign, never production data")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target", type=int)
    group.add_argument("--add", type=int)
    group.add_argument("--resume", action="store_true")
    parser.add_argument("--request-id", help="Idempotency key for --add; reuse this ID after interruption")
    parser.add_argument("--methods", choices=METHODS, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args()
    selected_output = default_output(args.instance)
    output = (args.output or (selected_output / "quick_test" if args.quick_test else selected_output)).resolve()
    if args.command == "analyze":
        if config(output)["science"]["scenario"] != args.instance:
            parser.error("Instance mismatch between --instance and --output")
        subprocess.run([sys.executable, str(ROOT / "analyze_i0252_statistical_019.py"), "--output", str(output)], check=True)
        return
    if args.command == "status":
        if config(output)["science"]["scenario"] != args.instance:
            parser.error("Instance mismatch between --instance and --output")
        results = load_complete(output)
        print(json.dumps({m: sum(v[1]["method"] == m for v in results) for m in METHODS}, indent=2))
        return
    if args.command == "validate":
        subprocess.run([sys.executable, str(ROOT / "validate_i0252_statistical_019.py"), "--instance", args.instance, "--output", str(output)], check=True)
        return
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.methods and len(args.methods) != len(set(args.methods)):
        parser.error("Duplicate methods")
    if args.request_id and args.add is None:
        parser.error("--request-id requires --add")
    output.mkdir(parents=True, exist_ok=True)
    with open(output / ".campaign.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Campaign already has an active runner")
        cfg = initialize(output, args.quick_test, args.instance)
        results = load_complete(output, cfg)
        completed = {(v[1]["method"], v[1]["replication"]) for v in results}
        counts = {m: sum(mm == m for mm, r in completed) for m in METHODS}
        ledger, pending = plan(args, output, completed)
        print(f"Pending optimizations: {len(pending)}; complete={counts}; targets={ledger['targets']}", flush=True)
        failures, seconds = simulate(output, cfg, pending, args.workers, counts, ledger["targets"]) if pending else ([], 0.)
        # Re-read and verify the durable state; never infer completion from process return alone.
        results = load_complete(output, cfg)
        counts = {m: sum(v[1]["method"] == m for v in results) for m in METHODS}
        state = {"updated_utc": utcnow(), "fingerprint": cfg["fingerprint"], "completed": counts,
                 "targets": ledger["targets"], "last_run_seconds": seconds, "failures": failures,
                 "complete": all(counts[m] >= ledger["targets"][m] for m in METHODS)}
        atomic_json(output / "STATUS.json", state)
        if not args.no_analysis and results:
            subprocess.run([sys.executable, str(ROOT / "analyze_i0252_statistical_019.py"), "--output", str(output)], check=True)
        print(json.dumps(state, indent=2), flush=True)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
