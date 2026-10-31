"""Close a complete DE campaign with audits, analysis, preservation and resume checks."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import time

from run_022 import BASE, INSTANCES, METHOD, output, sha, atomic_json, environment


def main():
    while True:
        status=json.loads((BASE/"STATUS.json").read_text())
        if status["failures"]:
            raise RuntimeError(status)
        if status["total"]==253:
            break
        time.sleep(5)
    atomic_json(BASE/"validation/production_execution.json",status)
    selected=[("statistical",i) for i in INSTANCES]+[("017","I0252")]
    def audit(item):
        cohort,instance=item;out=output(cohort,instance)
        with open(BASE/"logs"/f"saved_audit_{cohort}_{instance}.log","w") as log:
            subprocess.run(["julia","--startup-file=no",f"--project={BASE/'environment'}",
                str(BASE/"code/audit_saved_de_022.jl"),str(out)],stdout=log,stderr=subprocess.STDOUT,
                env=environment(),check=True)
        print(f"SAVED DE AUDIT PASSED {cohort}/{instance}",flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(audit,selected))
    with open(BASE/"logs/analysis.log","w") as log:
        subprocess.run([sys.executable,str(BASE/"code/analyze_022.py")],stdout=log,stderr=subprocess.STDOUT,
                       env=environment(),check=True)
    print("ANALYSIS COMPLETE",flush=True)
    manifests=[p for cohort,instance in selected for p in (output(cohort,instance)/"raw"/METHOD).glob("rep_*/complete.toml")]
    before={str(p.relative_to(BASE)):sha(p) for p in manifests}
    with open(BASE/"logs/completed_campaign_resume.log","w") as log:
        subprocess.run([sys.executable,str(BASE/"code/run_022.py"),"run","--cohort","both","--workers","12"],
                       stdout=log,stderr=subprocess.STDOUT,env=environment(),check=True)
    after={str(p.relative_to(BASE)):sha(p) for p in manifests}
    assert len(manifests)==253 and before==after
    resume=dict(passed=True,published_manifests_checked=253,all_manifests_identical=True,new_simulations=0)
    atomic_json(BASE/"validation/completed_campaign_resume.json",resume)
    completion=json.loads((BASE/"COMPLETE.json").read_text())
    from PIL import Image
    for path in BASE.rglob("*.png"):
        with Image.open(path) as im:
            assert min(im.size)>500;im.verify()
        pdf=path.with_suffix(".pdf").read_bytes()
        assert pdf.startswith(b"%PDF-") and b"%%EOF" in pdf[-1024:]
    for name,digest in completion["artifacts"].items():
        assert sha(BASE/name)==digest,name
    completion["completed_campaign_resume"]=resume
    completion["final_artifact_audit"]=dict(passed=True,artifact_hashes_checked=len(completion["artifacts"]),
        pngs_opened=completion["png_files"],pdfs_checked=completion["pdf_files"])
    atomic_json(BASE/"validation/final_artifact_audit.json",completion["final_artifact_audit"])
    atomic_json(BASE/"COMPLETE.json",completion)
    print("FINAL ARTIFACT AND RESUME AUDITS PASSED",flush=True)


if __name__=="__main__":
    main()
