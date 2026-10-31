"""Shared, simulation-free integrity checks for the I0252 replication campaign."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import tempfile

import toml

METHODS = ("fresh", "green", "green_softmax", "green_variance")
LABELS = dict(zip(METHODS, ("Fresh", "Green", "Green Softmax (β=0.5)", "Green Variance (κ=2)")))
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "resultados/019probing_i0252_four_methods_statistical"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def config(output):
    p = Path(output) / "config/experiment.toml"
    c = toml.load(p)
    if fingerprint(c["science"]) != c["fingerprint"]:
        raise ValueError(f"Invalid configuration fingerprint: {p}")
    return c


def csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def validate_rep(path, cfg, method=None, rep=None):
    """Only atomically published, hash-verified complete directories are data."""
    path = Path(path)
    done = toml.load(path / "complete.toml")
    meta = toml.load(path / "metadata.toml")
    if done.get("status") != "complete" or meta.get("status") != "complete":
        raise ValueError(f"Incomplete replication: {path}")
    if done["fingerprint"] != cfg["fingerprint"] or meta["fingerprint"] != cfg["fingerprint"]:
        raise ValueError(f"Incompatible replication: {path}")
    if meta["science"]["scenario"] != cfg["science"]["scenario"]:
        raise ValueError(f"Incorrect instance identity: {path}")
    method = method or path.parent.name
    rep = rep or int(path.name.removeprefix("rep_"))
    if meta["method"] != method or meta["replication"] != rep:
        raise ValueError(f"Incorrect replication identity: {path}")
    required = {"trajectory.csv", "final.csv", "iteration_diagnostics.csv", "search_seeds.csv", "metadata.toml",
                "search.jls", "convergence.jls", "final.jls"}
    if set(done["files"]) != required:
        raise ValueError(f"Unexpected completion manifest: {path}")
    for name, expected in done["files"].items():
        if sha(path / name) != expected:
            raise ValueError(f"Damaged result: {path / name}")
    rows = csv_rows(path / "trajectory.csv")
    science = cfg["science"]
    K = science["max_iters"]
    if len(rows) != K + 1 or [int(r["iteration"]) for r in rows] != list(range(K + 1)):
        raise ValueError(f"Incorrect trajectory length/indexing: {path}")
    previous_time = -1.0
    for row in rows:
        if row["method"] != method or int(row["replication"]) != rep:
            raise ValueError(f"Incorrect trajectory identity: {path}")
        for k in ("z3", "z4", "objective_crn", "cumulative_optimization_seconds", "crn_selected_policy_se"):
            if not math.isfinite(float(row[k])):
                raise ValueError(f"Nonfinite {k}: {path}")
        if not all(0 <= float(row[k]) <= 1 for k in ("z3", "z4")):
            raise ValueError(f"Decision outside domain: {path}")
        seconds = float(row["cumulative_optimization_seconds"])
        if seconds < previous_time:
            raise ValueError(f"Nonmonotonic optimization time: {path}")
        previous_time = seconds
        if int(row["evaluation_seed"]) != meta["seeds"]["trace"]:
            raise ValueError(f"Incorrect CRN association: {path}")
    if [float(rows[0][k]) for k in ("z3", "z4")] != science["Z0"]:
        raise ValueError(f"Incorrect initial decision: {path}")
    final = csv_rows(path / "final.csv")
    if len(final) != 1:
        raise ValueError(f"Incorrect endpoint count: {path}")
    f = final[0]
    for k in ("z3", "z4"):
        if float(f[k]) != float(rows[-1][k]):
            raise ValueError(f"Endpoint/trajectory mismatch: {path}")
    for key, expected in (("iterations", K), ("new_search_paths", K * science["n_paths"]),
                          ("selection_paths", science["selection_paths"]), ("holdout_paths", science["holdout_paths"])):
        if int(f[key]) != expected:
            raise ValueError(f"Incorrect {key}: {path}")
    for key in ("objective", "evaluation_se", "optimization_seconds", "diagnostic_seconds"):
        if not math.isfinite(float(f[key])):
            raise ValueError(f"Nonfinite final {key}: {path}")
    seeds = [int(r["seed"]) for r in csv_rows(path / "search_seeds.csv")]
    external = [meta["seeds"][k] for k in ("trace", "selection", "holdout")]
    if len(seeds) != K or len(set(seeds + external)) != K + 3:
        raise ValueError(f"Overlapping/missing RNG streams: {path}")
    return meta, rows, f


def load_complete(output, cfg=None):
    output = Path(output)
    cfg = cfg or config(output)
    results = []
    seen_streams = {}
    for method in METHODS:
        for path in sorted((output / "raw" / method).glob("rep_*")):
            rep = int(path.name.removeprefix("rep_"))
            meta, rows, final = validate_rep(path, cfg, method, rep)
            key = rep
            streams = meta["seeds"]
            if key in seen_streams and seen_streams[key] != streams:
                raise ValueError(f"CRN pairing differs between methods: replication {rep}")
            seen_streams[key] = streams
            results.append((path, meta, rows, final))
    # Different replications have independent streams; method sharing is deliberate.
    all_seeds = [int(v) for s in seen_streams.values() for v in [*s["search"], s["trace"], s["selection"], s["holdout"]]]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("RNG collision between replications")
    return results
