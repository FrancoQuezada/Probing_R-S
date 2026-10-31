"""Complete 009b without modifying/re-running the previous experiments."""
from pathlib import Path
import subprocess,os,time,json,hashlib
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'resultados/009b_probing_regime_validation'
ENV=os.environ.copy();ENV.update(OPENBLAS_NUM_THREADS='1',JULIA_NUM_THREADS='8',GKSwstype='100')
JULIA=['julia','--startup-file=no','--project=resultados/003probing_softmax_20261021/entorno']
ENTRY='009b_probing_regime_validation.jl'

def run(command,log):
    target=OUT/log;target.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    print('RUN',command,flush=True)
    with target.open('a') as f:subprocess.run(command,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (OUT/'logs/resources.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,wall_seconds=time.perf_counter()-start,threads=8,blas_threads=1))+'\n')

def hashes():
    return {str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['diagnostics','confirmation'] for f in (OUT/folder).glob('S*/*.csv')}

def main():
    run(JULIA+['tests/validate_parameter_regimes.jl'],'validation/original_regression.log')
    run(JULIA+[ENTRY,'--quick-test','--diagnostics','--confirm','--resume'],'validation/quick_workflow.log')
    run(JULIA+['tests/validate_009b.jl',str(OUT/'prueba_validada')],'validation/final_regression.log')
    run(['python3','tests/validate_009b_statistics.py'],'validation/final_statistics.log')
    run(['python3','analyze_009b.py',str(OUT/'prueba_validada')],'validation/quick_analysis.log')
    quick=json.loads((OUT/'prueba_validada/COMPLETE.json').read_text())
    assert quick['quick'] and quick['quick_workflow_complete']
    run(JULIA+[ENTRY,'--diagnostics','--resume'],'logs/diagnostics.log')
    run(JULIA+[ENTRY,'--confirm','--resume'],'logs/confirmation.log')
    run(JULIA+[ENTRY,'--plots','--report'],'logs/analysis.log')
    before=hashes()
    run(JULIA+[ENTRY,'--diagnostics','--confirm','--resume'],'validation/resume.log')
    assert hashes()==before,'Resume altered raw results'
    (OUT/'validation/resume_test.json').write_text(json.dumps(dict(passed=True,unchanged_csv_files=len(before)),indent=2))
    complete=json.loads((OUT/'COMPLETE.json').read_text());assert complete['complete'] and not complete['quick']
    print('009b COMPLETE',json.dumps(complete),flush=True)

if __name__=='__main__':main()
