"""Additive DE runner: single run, selected subset, 250 statistical + 3 original-017 runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import toml

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
from i0252_statistical_storage_019 import atomic_bytes, atomic_json, config, fingerprint, sha, csv_rows

INSTANCES = ("I0252", "L000", "L025", "L026", "L030")
METHOD = "differential_evolution"
GVSMETHOD = "green_variance_softmax"
SOURCE = Path(__file__).with_name("022_differential_evolution.jl")
ENV = BASE / "environment"
OLD021 = ROOT / "resultados/021_green_variance_softmax"
OLD017 = ROOT / "resultados/017probing_green_five_method_comparison"


def historical(instance):
    return ROOT / "resultados" / ("019probing_i0252_four_methods_statistical" if instance == "I0252"
                                 else f"020probing_four_instances_statistical/{instance}")


def output(cohort, instance):
    return BASE / instance if cohort == "statistical" else BASE / "individual_017" / instance


def expected_seeds(cohort, instance, rep):
    r = 19_000_000+rep if cohort == "statistical" else rep
    seed = lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:15], 16)+1
    streams = {s: seed(f"Probing010-v1/{instance}/compare/{r}/{s}/0") for s in ("trace", "selection", "holdout")}
    streams["training"] = seed(f"Probing010-v1/{instance}/compare/{r}/search/1")
    streams["optimizer"] = seed(f"DE022-v1/{instance}/{cohort}/{rep}/optimizer")
    return streams


def immutable(path, data):
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Frozen artifact changed: {path}")
    if not path.exists():
        atomic_bytes(path, data)


def snapshot():
    destination = BASE / "validation/historical_before.json"
    if destination.exists():
        return
    files = [p for name in ("016probing_green_bank_variants", "017probing_green_five_method_comparison",
             "019probing_i0252_four_methods_statistical", "020probing_four_instances_statistical", "021_green_variance_softmax")
             for p in sorted((ROOT / "resultados" / name).rglob("*")) if p.is_file()]
    files.extend(p for p in ROOT.iterdir() if p.is_file())
    atomic_json(destination, {str(p.relative_to(ROOT)): sha(p) for p in files})


def verify_history():
    before = json.loads((BASE / "validation/historical_before.json").read_text())
    changed = [name for name, digest in before.items() if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    result = dict(passed=not changed, checked_files=len(before), changed=changed)
    atomic_json(BASE / "validation/historical_preservation.json", result)
    if changed:
        raise ValueError(f"Historical files changed: {changed}")
    return result


def initialize(cohort, instance):
    old = historical(instance);previous = config(old);d = previous["science"]
    version=subprocess.check_output(["julia","--startup-file=no","-e","print(VERSION)"],text=True)
    if version!=d["julia_version"]:
        raise ValueError(f"Frozen experiment requires Julia {d['julia_version']}, found {version}")
    gvs = config(OLD021 / instance)["science"]
    for key, expected in dict(scenario=instance, max_iters=100, n_paths=500, trace_paths=1000,
                              selection_paths=10000, holdout_paths=50000, jobs=45, policies=144).items():
        if d[key] != expected or d[key] != gvs[key]:
            raise ValueError(f"Historical protocol mismatch: {instance}/{key}")
    if d["model_sha256"] != gvs["model_sha256"] or d["policy_grid_sha256"] != gvs["policy_grid_sha256"]:
        raise ValueError("Historical model or policy-grid mismatch")
    if cohort == "017":
        ref = toml.load(OLD017 / "configs/experiment.toml")["science"]
        assert instance == "I0252" and ref["model_sha256"] == d["model_sha256"]
        assert ref["iterations"] == d["max_iters"] and ref["new_paths_per_iteration"] == d["n_paths"]
    out = output(cohort, instance)
    for folder in ("config", "reproduction", "raw", "summaries", "plots", "logs", "validation"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for name, digest in d["numerical_sources"].items():
        source = old / "reproduction" / name
        if sha(source) != digest:
            raise ValueError(f"Historical source corrupted: {source}")
        immutable(out / "reproduction" / name, source.read_bytes())
    for path in (old / "config").glob("*"):
        if path.suffix in (".toml", ".csv") and path.name != "experiment.toml":
            immutable(out / "config" / path.name, path.read_bytes())
    for name in ("Project.toml", "Manifest.toml"):
        immutable(out / "reproduction" / name, (ENV/name).read_bytes())
    # Test-stage 022 changes may refresh new files only before any production run.
    frozen = any((out / "raw").rglob("state.toml")) or any((out / "raw").rglob("complete.toml"))
    write = immutable if frozen else atomic_bytes
    write(out / "reproduction" / SOURCE.name, SOURCE.read_bytes())
    science = copy.deepcopy(d)
    science.update(protocol="022-DE-direct-fixed-CRN-v1", scenario=instance, methods=[METHOD], cohort=cohort,
                   historical_output=str(old), historical_fingerprint=previous["fingerprint"],
                   objective="max of 144 expected gross-policy rewards minus active probing cost; original direct simulator",
                   seed_protocol="DE022 SHA256 optimizer task RNG; original Probing010 training/search1 and independent validation streams",
                   timing_protocol="Warmed single-process wall time; BLAS=1; optimization separate from validation, storage and plotting")
    active = toml.load(out / f"config/{instance}.toml")["model"]["ACTIVE_CLASSES"]
    assert active == [3, 4]
    budget = d["max_iters"]*d["n_paths"]
    science["de"] = dict(package="BlackBoxOptim", version="0.6.12", method="adaptive_de_rand_1_bin_radiuslimited",
                         dimension=len(active), bounds=[[0., 1.]]*len(active), population_size=10*len(active),
                         sampler_radius=8, mutation="library adaptive bimodal Cauchy defaults", crossover="library adaptive bimodal Cauchy defaults",
                         paths_per_evaluation=d["n_paths"], max_objective_evaluations=budget//d["n_paths"],
                         path_budget=budget, max_policy_path_evaluations=budget*d["policies"],
                         initial_point_in_budget=True, callback_margin_evaluations=1, fixed_training_crn=True,
                         library_MaxFuncEvals=budget//d["n_paths"], library_MaxSteps=10000,
                         library_MaxStepsWithoutProgress=10000, library_MaxTime=0., library_TargetFitness="none",
                         library_MinDeltaFitnessTolerance=1e-50, library_FitnessTolerance=1e-8)
    science["numerical_sources"][SOURCE.name] = sha(SOURCE)
    science["environment"] = {name: sha(ENV/name) for name in ("Project.toml", "Manifest.toml")}
    science["historical_environment"] = d["environment"]
    c = dict(fingerprint=fingerprint(science), science=science)
    write(out / "config/experiment.toml", toml.dumps(c).encode())
    return c


def validate_rep(cohort, instance, rep, hashes=True):
    out = output(cohort, instance);c = config(out);d = c["science"]
    path = out / "raw" / METHOD / f"rep_{rep:04d}"
    done = toml.load(path / "complete.toml");meta = toml.load(path / "metadata.toml")
    assert done["status"] == meta["status"] == "complete"
    assert done["fingerprint"] == meta["fingerprint"] == c["fingerprint"]
    assert (meta["scenario"], meta["cohort"], meta["replication"], meta["method"]) == (instance, cohort, rep, METHOD)
    assert meta["seeds"] == expected_seeds(cohort, instance, rep)
    assert len(set(meta["seeds"].values())) == 5
    if hashes:
        for name, digest in done["files"].items():
            assert sha(path/name) == digest, str(path/name)
    rows = csv_rows(path / "trajectory.csv");f = csv_rows(path / "final.csv");assert len(f) == 1;f = f[0]
    count = int(f["objective_evaluations"]);n = d["de"]["paths_per_evaluation"];cap = d["de"]["max_objective_evaluations"]
    assert cap-1 <= count <= cap
    assert int(f["library_objective_evaluations"])+1 == count
    assert len(rows) == count+1 and [int(r["iteration"]) for r in rows] == list(range(count+1))
    assert [float(rows[0][key]) for key in ("z3", "z4")] == d["Z0"]
    assert all(0 <= float(r[k]) <= 1 for r in rows for k in ("z3", "z4"))
    assert all(math.isfinite(float(r[k])) for r in rows for k in ("z3", "z4", "objective_crn", "cumulative_optimization_seconds"))
    assert [float(f[k]) for k in ("z3", "z4")] == [float(rows[-1][k]) for k in ("z3", "z4")]
    assert int(f["new_search_paths"]) == n and int(f["path_evaluations"]) == n*count <= d["de"]["path_budget"]
    assert int(f["policy_path_evaluations"]) == n*count*d["policies"]
    assert int(f["selection_paths"]) == d["selection_paths"] and int(f["holdout_paths"]) == d["holdout_paths"]
    assert all(int(r["evaluation_seed"]) == meta["seeds"]["trace"] and int(r["evaluation_paths"]) == 1000 for r in rows)
    if cohort == "statistical":
        oldmeta = toml.load(historical(instance) / "raw/fresh" / f"rep_{rep:04d}/metadata.toml")
        assert all(meta["seeds"][k] == oldmeta["seeds"][k] for k in ("trace", "selection", "holdout"))
        assert meta["seeds"]["training"] == oldmeta["seeds"]["search"][0]
    return path, meta, rows, f


def command(cohort, instance):
    out = output(cohort, instance)
    return ["julia", "--startup-file=no", f"--project={ENV}", str(out / "reproduction" / SOURCE.name), str(out)]


def environment():
    return dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")


def test(selected):
    def one(item):
        cohort, instance = item
        with open(BASE / "logs" / f"tests_{cohort}_{instance}.log", "w") as log:
            result = subprocess.run(command(cohort, instance)+["test"], env=environment(), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Tests failed: {cohort}/{instance}; see logs")
        print(f"TEST PASSED {cohort}/{instance}", flush=True)
    with ThreadPoolExecutor(max_workers=min(5, len(selected))) as pool:
        list(pool.map(one, selected))


def run(selected, reps, workers):
    pending = queue.Queue();counts = {}
    for cohort, instance in selected:
        ids = reps if reps is not None else list(range(1, 51)) if cohort == "statistical" else [1, 2, 3]
        if cohort == "017" and (instance != "I0252" or any(r not in (1, 2, 3) for r in ids)):
            raise ValueError("017 contains only I0252 replications 1, 2, 3")
        out = output(cohort, instance)
        assert toml.load(out / "validation/preflight_tests.toml")["passed"]
        key=f"{cohort}/{instance}";counts[key] = 0
        for rep in ids:
            if (out / "raw" / METHOD / f"rep_{rep:04d}/complete.toml").exists():
                validate_rep(cohort, instance, rep);counts[key] += 1
            else:
                pending.put((cohort, instance, rep))
    failures = [];mutex = threading.Lock();start = time.monotonic();initial_pending = pending.qsize()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    def publish():
        all_counts = {f"{cohort}/{instance}": len(list((output(cohort, instance)/"raw"/METHOD).glob("rep_*/complete.toml")))
                      for cohort, instance in [("statistical",i) for i in INSTANCES]+[("017","I0252")]}
        atomic_json(BASE / "STATUS.json", dict(completed=all_counts, total=sum(all_counts.values()),
                    statistical_total=sum(v for k,v in all_counts.items() if k.startswith("statistical/")),
                    individual_017_total=all_counts["017/I0252"], failures=failures, target=253,
                    elapsed_seconds=time.monotonic()-start))
    def worker(index):
        current=None;proc=None;log=None;identity=None
        try:
            while True:
                try:
                    cohort, instance, rep = pending.get_nowait()
                except queue.Empty:
                    break
                if identity != (cohort, instance):
                    if proc is not None:
                        proc.stdin.close();proc.stdout.read();proc.wait();log.close()
                    logfile=BASE/"logs"/f"{stamp}_worker{index}_{cohort}_{instance}.log"
                    log=open(logfile,"a")
                    proc=subprocess.Popen(command(cohort,instance),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT,text=True,bufsize=1,env=environment())
                    for line in proc.stdout:
                        log.write(line);log.flush()
                        if line.strip()=="READY":
                            break
                    else:
                        raise RuntimeError(f"Worker startup failed: {logfile}")
                    identity=cohort,instance
                current=cohort,instance,rep
                print(f"RUNNING {cohort}/{instance} rep{rep:02d}",flush=True)
                proc.stdin.write(f"{rep}\n");proc.stdin.flush()
                for line in proc.stdout:
                    log.write(line);log.flush()
                    if line.startswith("COMPLETE\t"):
                        assert int(line.split("\t")[1])==rep
                        validate_rep(cohort,instance,rep)
                        with mutex:
                            counts[f"{cohort}/{instance}"]+=1;publish()
                            print(f"COMPLETE {cohort}/{instance} rep{rep:02d}; {time.monotonic()-start:.0f}s",flush=True)
                        break
                    if line.startswith("FAILED\t"):
                        with mutex:
                            failures.append([*current,str(logfile)]);publish()
                        print(f"FAILED {current}; {logfile}",flush=True);break
                else:
                    raise RuntimeError(f"Worker exited during {current}")
                pending.task_done();current=None
        except Exception as exc:
            with mutex:
                failures.append([str(current),str(exc)]);publish()
            print(f"WORKER ERROR {exc}",flush=True)
        finally:
            if proc is not None:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                for line in proc.stdout:
                    log.write(line)
                proc.wait();log.close()
    publish()
    with ThreadPoolExecutor(max_workers=min(workers,max(1,initial_pending))) as pool:
        list(pool.map(worker,range(min(workers,initial_pending))))
    publish()
    if failures or not pending.empty():
        raise RuntimeError(f"Campaign failures: {failures}; unfinished queue={pending.qsize()}")


def main():
    global BASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","test","run","status","verify-history"))
    parser.add_argument("--cohort",choices=("statistical","017","both"),default="statistical")
    parser.add_argument("--instances",choices=INSTANCES,nargs="+",default=list(INSTANCES))
    parser.add_argument("--replications",type=int,nargs="+")
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--output-root",type=Path,help="Separate destination for independently reproducing selected runs; uses the frozen 022 environment")
    args=parser.parse_args()
    if args.workers<1 or len(args.instances)!=len(set(args.instances)):
        parser.error("Positive worker count and unique instance IDs required")
    if args.output_root is not None:
        selected_root=args.output_root.resolve()
        protected=[ROOT/"resultados"/name for name in ("016probing_green_bank_variants","017probing_green_five_method_comparison",
                   "019probing_i0252_four_methods_statistical","020probing_four_instances_statistical","021_green_variance_softmax")]
        if any(selected_root==p or p in selected_root.parents for p in protected):
            parser.error("The DE output namespace must be separate from all historical experiments")
        BASE=selected_root
        for folder in ("logs","validation"):
            (BASE/folder).mkdir(parents=True,exist_ok=True)
    if args.command=="status":
        print((BASE/"STATUS.json").read_text());return
    if args.command=="verify-history":
        print(verify_history());return
    if args.replications and (len(set(args.replications))!=len(args.replications) or any(not 1<=r<=50 for r in args.replications)):
        parser.error("Unique replication IDs in 1..50 required")
    selected=([("statistical",i) for i in args.instances] if args.cohort in ("both","statistical") else [])
    if args.cohort in ("both","017"):
        selected.append(("017","I0252"))
    with open(BASE/".campaign.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot()
        for cohort,instance in selected:
            initialize(cohort,instance)
        if args.command=="test":
            test(selected)
        if args.command=="run":
            run(selected,args.replications,args.workers)


if __name__=="__main__":
    main()
