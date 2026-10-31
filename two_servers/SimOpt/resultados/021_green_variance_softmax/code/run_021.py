"""Frozen-source, resumable 021 campaign. Writes only beneath experiment 021."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
from i0252_statistical_storage_019 import atomic_bytes, atomic_json, config, fingerprint, load_complete, sha, validate_rep

METHOD = "green_variance_softmax"
INSTANCES = ("I0252", "L000", "L025", "L026", "L030")
OLD020 = ROOT / "resultados/020probing_four_instances_statistical"
OLD019 = ROOT / "resultados/019probing_i0252_four_methods_statistical"
SOURCE = Path(__file__).with_name("021_green_variance_softmax.jl")


def historical(instance):
    return OLD019 if instance == "I0252" else OLD020 / instance


def immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen artifact differs: {path}")
    else:
        atomic_bytes(path, data)


def initialize(instance):
    import copy
    import toml
    old = historical(instance)
    c = config(old)
    d = c["science"]
    expected = dict(scenario=instance, max_iters=100, n_paths=500, jobs=45, trace_paths=1000,
                    selection_paths=10000, holdout_paths=50000, Z0=[.12, .10], BASE_STEP=.035,
                    GRAD_CLIP=15., policies=144, beta_softmax=.5, kappa=2.)
    for k, value in expected.items():
        if d[k] != value:
            raise ValueError(f"Historical protocol mismatch: {instance}/{k}")
    out = BASE / instance
    for folder in ("config", "reproduction", "raw", "summaries", "plots", "logs", "validation"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for name, digest in {**d["numerical_sources"], **d["environment"]}.items():
        path = old / "reproduction" / name
        if sha(path) != digest:
            raise ValueError(f"Corrupted historical frozen source: {path}")
        immutable(out / "reproduction" / name, path.read_bytes())
    for path in sorted((old / "config").glob("*")):
        if path.suffix in (".toml", ".csv") and path.name != "experiment.toml":
            immutable(out / "config" / path.name, path.read_bytes())
    # Before the first production bank, test-driven changes may refresh 021 only.
    frozen = any((out / "raw").rglob("bank_*.jls"))
    write_new_source = immutable if frozen else atomic_bytes
    write_new_source(out / "reproduction" / SOURCE.name, SOURCE.read_bytes())
    science = copy.deepcopy(d)
    science.update(protocol="021-green-variance-softmax-v1", methods=[METHOD], replications=50,
                   historical_output=str(old), historical_fingerprint=c["fingerprint"],
                   seed_pairing="Same 019/020 seed per instance/replication; no historical reruns",
                   bank_persistence="Every materialized bank and 144-policy reward matrix; iteration checkpoint",
                   filter_then_softmax=True)
    science["numerical_sources"][SOURCE.name] = sha(SOURCE)
    new = dict(fingerprint=fingerprint(science), science=science)
    write_new_source(out / "config/experiment.toml", toml.dumps(new).encode())
    return new


def validate_new(instance, rep, hashes=True):
    import toml
    out = BASE / instance
    c = config(out)
    path = out / "raw" / METHOD / f"rep_{rep:04d}"
    meta, rows, final = validate_rep(path, c, METHOD, rep)
    extra = toml.load(path / "supplement_complete.toml")
    if extra["fingerprint"] != c["fingerprint"] or extra["status"] != "complete":
        raise ValueError(f"Invalid supplement: {path}")
    if hashes:
        for name, digest in extra["files"].items():
            if sha(path / name) != digest:
                raise ValueError(f"Corrupted bank or supplement: {path/name}")
    if len(list((path / "banks").glob("bank_*.jls"))) != 100:
        raise ValueError(f"Incomplete banks: {path}")
    # Every statistical comparison is genuinely paired at search and evaluation streams.
    oldmeta, _, _ = validate_rep(historical(instance) / "raw/green_variance" / f"rep_{rep:04d}",
                                config(historical(instance)), "green_variance", rep)
    if meta["seeds"] != oldmeta["seeds"]:
        raise ValueError("Historical seed pairing mismatch")
    return meta, rows, final


def snapshot():
    path = BASE / "validation/historical_before.json"
    if path.exists():
        return
    folders = [ROOT / "resultados" / name for name in (
        "016probing_green_bank_variants", "017probing_green_five_method_comparison",
        "019probing_i0252_four_methods_statistical", "020probing_four_instances_statistical")]
    files = [p for folder in folders for p in sorted(folder.rglob("*")) if p.is_file()]
    files += sorted(p for p in ROOT.iterdir() if p.is_file())
    atomic_json(path, {str(p.relative_to(ROOT)): sha(p) for p in files})


def verify_history():
    before = json.loads((BASE / "validation/historical_before.json").read_text())
    changed = [name for name, digest in before.items() if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    result = dict(passed=not changed, checked_files=len(before), changed=changed)
    atomic_json(BASE / "validation/historical_preservation.json", result)
    if changed:
        raise ValueError(f"Historical files changed: {changed}")
    return result


def command(instance):
    out = BASE / instance
    return ["julia", "--startup-file=no", f"--project={out/'reproduction'}",
            str(out / "reproduction" / SOURCE.name), str(out)]


def run_tests():
    def one(instance):
        path = BASE / "logs" / f"tests_{instance}.log"
        with open(path, "w") as log:
            result = subprocess.run(command(instance) + ["test"], stdout=log, stderr=subprocess.STDOUT, env=environment())
        if result.returncode:
            raise RuntimeError(f"Consistency tests failed: {path}")
        print(f"TEST PASSED {instance}", flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(one, INSTANCES))


def environment():
    return dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")


def run(workers):
    # Validate and skip published results; incomplete supplements are repaired by Julia.
    pending = {i: queue.Queue() for i in INSTANCES}
    counts = {i: 0 for i in INSTANCES}
    for instance in INSTANCES:
        for rep in range(1, 51):
            path = BASE / instance / "raw" / METHOD / f"rep_{rep:04d}"
            if (path / "supplement_complete.toml").exists():
                validate_new(instance, rep)
                counts[instance] += 1
            else:
                pending[instance].put(rep)
    failures = [];mutex = threading.Lock();started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def publish():
        atomic_json(BASE / "STATUS.json", dict(completed=counts, total=sum(counts.values()),
                    target=250, failures=failures, elapsed_seconds=time.monotonic()-started))

    def worker(item):
        instance, index = item
        if pending[instance].empty():
            return
        logpath = BASE / "logs" / f"{stamp}_{instance}_worker{index}.log"
        with open(logpath, "x") as log:
            proc = subprocess.Popen(command(instance), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment())
            try:
                for line in proc.stdout:
                    log.write(line);log.flush()
                    if line.strip() == "READY":
                        break
                else:
                    raise RuntimeError(f"Worker startup failed: {logpath}")
                while True:
                    try:
                        rep = pending[instance].get_nowait()
                    except queue.Empty:
                        break
                    print(f"RUNNING {instance} rep{rep:02d}", flush=True)
                    proc.stdin.write(f"{rep}\n");proc.stdin.flush()
                    for line in proc.stdout:
                        log.write(line);log.flush()
                        if line.startswith("COMPLETE\t"):
                            if int(line.split("\t")[1]) != rep:
                                raise ValueError("Worker identity mismatch")
                            validate_new(instance, rep)
                            with mutex:
                                counts[instance] += 1;publish()
                                print(f"COMPLETE {instance} rep{rep:02d}: {counts[instance]}/50; total {sum(counts.values())}/250; {time.monotonic()-started:.0f}s", flush=True)
                            break
                        if line.startswith("FAILED\t"):
                            with mutex:
                                failures.append([instance, rep, str(logpath)]);publish()
                            print(f"FAILED {instance} rep{rep:02d}: {logpath}", flush=True)
                            break
                    else:
                        raise RuntimeError(f"Worker exited: {logpath}")
            except Exception as exc:
                with mutex:
                    failures.append([instance, index, str(exc)]);publish()
                print(str(exc), flush=True)
            finally:
                if proc.stdin:
                    try:
                        proc.stdin.close()
                    except BrokenPipeError:
                        pass
                for line in proc.stdout:
                    log.write(line)
                proc.wait()
    publish()
    tasks = [(instance, j) for index, instance in enumerate(INSTANCES)
             for j in range(workers // 5 + (index < workers % 5))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, tasks))
    publish()
    if failures or sum(counts.values()) != 250:
        raise RuntimeError(f"Campaign incomplete: counts={counts}, failures={failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "test", "run", "status", "verify-history"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.command == "status":
        print((BASE / "STATUS.json").read_text());return
    if args.command == "verify-history":
        print(verify_history());return
    with open(BASE / ".campaign.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot()
        for instance in INSTANCES:
            initialize(instance)
        if args.command == "test":
            run_tests()
        if args.command == "run":
            for instance in INSTANCES:
                import toml
                if not toml.load(BASE / instance / "validation/algorithm_tests.toml")["passed"]:
                    raise ValueError(f"Run consistency tests first: {instance}")
            if args.workers < 5:
                raise ValueError("At least five workers, one per instance, required")
            run(args.workers)


if __name__ == "__main__":
    main()
