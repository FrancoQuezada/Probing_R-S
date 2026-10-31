"""Coordinate the existing statistical runner over the four original instances."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from i0252_statistical_storage_019 import ROOT, METHODS, config, load_complete, atomic_json
from probing_statistical_instances_020 import BASE_OUTPUT, INSTANCES, canonical_instance
from run_i0252_statistical_019 import initialize


def run_selected(args, command, extra=(), jobs=None):
    instances = args.instances
    concurrency = min(len(instances), jobs or args.workers)
    allocations = [max(1, args.workers // concurrency + (i < args.workers % concurrency)) for i in range(len(instances))]
    lock = threading.Lock()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    (args.base / "logs").mkdir(parents=True, exist_ok=True)

    def run_one(item):
        index, instance = item
        logfile = args.base / "logs" / f"{stamp}_{command}_{instance}.log"
        cmd = [sys.executable, str(ROOT / "run_i0252_statistical_019.py"), command,
               "--instance", instance, "--output", str(args.base / instance)]
        if command == "run":
            cmd.extend(["--workers", str(allocations[index])])
        cmd.extend(extra)
        with open(logfile, "x") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line); log.flush()
                with lock:
                    print(f"[{instance}] {line.rstrip()}", flush=True)
            code = proc.wait()
        return instance, code

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        statuses = dict(pool.map(run_one, enumerate(instances)))
    if any(statuses.values()):
        raise RuntimeError(f"Campaign command failures: {statuses}; completed replications are retained")


def status(base, instances):
    counts = {}
    for instance in instances:
        path = base / instance
        results = load_complete(path) if (path / "config/experiment.toml").exists() else []
        counts[instance] = {method: sum(meta["method"] == method for _, meta, _, _ in results) for method in METHODS}
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "validate", "run", "analyze", "status", "verify"))
    parser.add_argument("--base", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--instances", type=canonical_instance, choices=INSTANCES, nargs="+", default=list(INSTANCES))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target", type=int)
    group.add_argument("--add", type=int)
    group.add_argument("--resume", action="store_true")
    parser.add_argument("--request-id")
    parser.add_argument("--methods", choices=METHODS, nargs="+")
    parser.add_argument("--workers", type=int, default=12, help="Total Julia worker budget across selected instances")
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args()
    args.base = args.base.resolve()
    if args.workers < 1 or len(args.instances) != len(set(args.instances)):
        parser.error("Positive worker count and unique canonical instance IDs required")
    if args.command == "status":
        print(json.dumps(status(args.base, args.instances), indent=2)); return
    if args.command == "preflight":
        for instance in args.instances:
            cfg = initialize(args.base / instance, instance=instance)
            print(instance, cfg["science"]["model_sha256"], "reference grid point", cfg["science"]["reference_grid_best"])
        return
    if args.command == "validate":
        subprocess.run([sys.executable, str(ROOT / "validate_statistical_comparison_020.py"), "--base", str(args.base)], cwd=ROOT, check=True)
        return
    if args.command == "verify":
        subprocess.run([sys.executable, str(ROOT / "validate_statistical_comparison_020.py"), "--base", str(args.base), "--final-only"], cwd=ROOT, check=True)
        return
    if args.command == "analyze":
        run_selected(args, "analyze")
        from analyze_statistical_comparison_020 import comparison
        comparison(args.base)
        return
    extra = []
    for option, value in (("--target", args.target), ("--add", args.add), ("--request-id", args.request_id)):
        if value is not None:
            extra.extend([option, str(value)])
    if args.resume:
        extra.append("--resume")
    if args.methods:
        extra.extend(["--methods", *args.methods])
    if args.no_analysis:
        extra.append("--no-analysis")
    run_selected(args, "run", extra)
    counts = status(args.base, args.instances)
    atomic_json(args.base / "STATUS.json", {"completed": counts, "total_complete_optimizations": sum(sum(v.values()) for v in counts.values())})
    if not args.no_analysis:
        from analyze_statistical_comparison_020 import comparison
        comparison(args.base)
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
