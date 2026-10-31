#!/usr/bin/env python3
"""Sequential, restartable execution of the authorized research experiments."""
import os, sys, subprocess, shutil, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NAMES=["005probing_softmax_multiseed","006probing_softmax_sample_efficiency",
       "007probing_green_softmax","008probing_probability_softmax"]
ENV=os.environ.copy()
ENV.update(GKSwstype="100",OPENBLAS_NUM_THREADS="1",JULIA_NUM_THREADS="4")
JULIA=["julia","--startup-file=no","--project=resultados/003probing_softmax_20261021/entorno"]

def execute(cmd,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    print("RUN", " ".join(cmd),flush=True)
    with log.open("a") as f:
        subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    tests=ROOT/"resultados/validacion_probing_005_008"
    tests.mkdir(exist_ok=True)
    execute(JULIA+["tests/validate_probing_005_008.jl"],tests/"algorithm_tests.log")
    # Run previous tests as regression checks, without rewriting any old outputs.
    for name in ["validate_four_searches_softmax.jl","validate_softmax_sensitivity.jl"]:
        execute(JULIA+["tests/"+name],tests/name.replace(".jl",".log"))
    for stage,name in enumerate(NAMES,5):
        out=ROOT/"resultados"/name
        val=out/"validacion";val.mkdir(parents=True,exist_ok=True)
        for f in tests.glob("*.log"):shutil.copyfile(f,val/f.name)
        quick=out/"prueba_validada"
        cmd=JULIA+[name+".jl","--quick-test","--output",str(quick)]
        if (quick/"configuration.toml").exists():cmd.append("--resume")
        execute(cmd,val/"quick_console.log")
        assert json.loads((quick/"COMPLETE.json").read_text())["quick"]
        cmd=JULIA+[name+".jl"]
        if (out/"configuration.toml").exists():cmd.append("--resume")
        execute(cmd,out/"console.log")
        c=json.loads((out/"COMPLETE.json").read_text())
        assert not c["quick"] and c["completed_replications"]==c["expected_replications"]
        # A second invocation must reuse every complete checkpoint and produce
        # identical numerical CSVs, without generating new system trajectories.
        import hashlib
        before={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/"replications").glob("*.csv")}
        execute(JULIA+[name+".jl","--resume"],val/"resume_console.log")
        after={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/"replications").glob("*.csv")}
        assert before==after,"Resume changed numerical results"
        (val/"resume_test.json").write_text(json.dumps(dict(passed=True,unchanged_csv_files=len(before)),indent=2))
    execute(["python3",str(ROOT/"summarize_probing_005_008.py")],ROOT/"resultados/informe_probing_005_008.log")
    print("ALL FOUR STAGES COMPLETE",flush=True)

if __name__=="__main__":main()
