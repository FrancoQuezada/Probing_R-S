#!/usr/bin/env python3
"""Sequential 009 execution with exact reuse of valid completed tasks."""
from pathlib import Path
import subprocess,os,time,json,hashlib

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"resultados/009probing_parameter_regimes"
ENV=os.environ.copy();ENV.update(GKSwstype="100",OPENBLAS_NUM_THREADS="1",JULIA_NUM_THREADS="8")
JULIA=["julia","--startup-file=no","--project=resultados/003probing_softmax_20261021/entorno"]
ENTRY="009probing_parameter_regimes.jl"

def execute(args,log):
    log.parent.mkdir(parents=True,exist_ok=True);cmd=JULIA+args;start=time.perf_counter()
    print("RUN",cmd,flush=True)
    with log.open("a") as f:subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (OUT/"logs/resources.jsonl").open("a") as f:
        f.write(json.dumps(dict(command=cmd,wall_seconds=time.perf_counter()-start,threads=8,blas_threads=1))+"\n")

def hashes():
    return {str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("characterization","pilot","confirmation") for p in (OUT/folder).glob("S*/*.csv")}

def main():
    (OUT/"logs").mkdir(parents=True,exist_ok=True)
    execute(["tests/validate_parameter_regimes.jl"],OUT/"validation/final_model_regression.log")
    execute([ENTRY,"--quick-test","--all-scenarios","--resume"],OUT/"validation/quick_full_workflow.log")
    execute(["export_parameter_regime_routing.jl","--quick-test"],OUT/"validation/routing_decisions_quick.log")
    quick=json.loads((OUT/"prueba_validada/COMPLETE.json").read_text());assert quick["quick"] and quick["complete"]
    execute([ENTRY,"--characterize","--resume"],OUT/"logs/characterization_resume.log")
    execute([ENTRY,"--pilot","--resume"],OUT/"logs/pilot.log")
    execute([ENTRY,"--confirm","--resume"],OUT/"logs/confirmation.log")
    execute(["export_parameter_regime_routing.jl"],OUT/"logs/routing_decisions.log")
    execute([ENTRY,"--plots","--report"],OUT/"logs/analysis.log")
    before=hashes()
    execute([ENTRY,"--characterize","--pilot","--confirm","--resume"],OUT/"validation/resume.log")
    after=hashes();assert before==after,"Resume changed raw numerical data"
    (OUT/"validation/resume_test.json").write_text(json.dumps(dict(passed=True,unchanged_csv_files=len(before)),indent=2))
    complete=json.loads((OUT/"COMPLETE.json").read_text());assert complete["complete"] and not complete["quick"]
    print("009 COMPLETE",json.dumps(complete),flush=True)

if __name__=="__main__":main()
