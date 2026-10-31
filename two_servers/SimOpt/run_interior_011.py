"""Execute or resume 011 without re-running valid completed simulations."""
from pathlib import Path
import subprocess,os,time,json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'resultados/011probing_high_contrast_interior'
ENV=os.environ.copy();ENV.update(JULIA_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',GKSwstype='100')
JULIA=['julia','--startup-file=no','--project=resultados/003probing_softmax_20261021/entorno']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def execute(command,log):
    (OUT/'logs').mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    with (OUT/'logs'/log).open('a') as f:subprocess.run(command,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (OUT/'logs/resources.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,wall_seconds=time.perf_counter()-start,threads=8,blas_threads=1))+'\n')

def main():
    execute(JULIA+['011probing_high_contrast_interior.jl','--calibrate','--screen','--refine','--resume'],'geometry_workflow.log')
    execute(JULIA+['refinar_compromisos_011.jl'],'lower_delay_finalize.log')
    execute(JULIA+['validar_fronteras_011.jl'],'boundary_validation.log')
    execute(JULIA+['011probing_high_contrast_interior.jl','--compare','--plots','--resume'],'full_workflow.log')
    complete=json.loads((OUT/'COMPLETE.json').read_text());assert complete['complete'] and not complete['quick']
    manifest=OUT/'validation/preserved_before.json'
    if manifest.exists():
        before=json.loads(manifest.read_text());changed=[name for name,h in before.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
        assert not changed,changed
        (OUT/'validation/previous_files_verified.json').write_text(json.dumps(dict(passed=True,unchanged_files=len(before)),indent=2)+'\n')
    for file in ['analyze_interior_011.py','analyze_landscape_010.py','run_interior_011.py','refinar_compromisos_011.jl','validar_fronteras_011.jl']:
        shutil.copy2(ROOT/file,OUT/'reproduction'/file)
    print(json.dumps(complete,indent=2))

if __name__=='__main__':main()
