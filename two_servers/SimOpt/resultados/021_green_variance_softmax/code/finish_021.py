"""Wait for the campaign, audit saved banks, analyze, and verify resume idempotency."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import time

from run_021 import BASE, INSTANCES, METHOD, atomic_json, sha


def main():
    while True:
        status = json.loads((BASE / "STATUS.json").read_text())
        if status["failures"]:
            raise RuntimeError(status)
        if status["total"] == 250:
            break
        time.sleep(5)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", JULIA_NUM_THREADS="1")

    def audit(instance):
        out = BASE / instance
        with open(BASE / "logs" / f"saved_bank_audit_{instance}.log", "w") as log:
            subprocess.run(["julia", "--startup-file=no", f"--project={out/'reproduction'}",
                            str(BASE / "code/audit_saved_banks_021.jl"), str(out)],
                           stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        print(f"SAVED BANK AUDIT PASSED {instance}", flush=True)

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(audit, INSTANCES))
    with open(BASE / "logs/analysis.log", "w") as log:
        subprocess.run([sys.executable, str(BASE / "code/analyze_021.py")],
                       stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    print("ANALYSIS COMPLETE", flush=True)
    files = [p for i in INSTANCES for p in (BASE / i / "raw" / METHOD).rglob("*complete.toml")]
    before = {str(p.relative_to(BASE)): sha(p) for p in files}
    with open(BASE / "logs/completed_campaign_resume.log", "w") as log:
        subprocess.run([sys.executable, str(BASE / "code/run_021.py"), "run", "--workers", "12"],
                       stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    after = {str(p.relative_to(BASE)): sha(p) for p in files}
    assert before == after
    assert not list(BASE.glob("*/raw/green_variance_softmax/.rep_*.partial"))
    atomic_json(BASE / "validation/completed_campaign_resume.json", dict(passed=True,
        published_manifests_checked=len(files), all_manifests_identical=before==after,
        new_simulations_on_resume=0, partial_production_replications=0))
    completion = json.loads((BASE / "COMPLETE.json").read_text())
    completion["completed_campaign_resume"] = json.loads((BASE / "validation/completed_campaign_resume.json").read_text())
    completion["source_hashes"] = {p.name: sha(p) for p in (BASE / "code").glob("*") if p.is_file()}
    atomic_json(BASE / "COMPLETE.json", completion)
    print("COMPLETED CAMPAIGN RESUME PASSED", flush=True)


if __name__ == "__main__":
    main()
