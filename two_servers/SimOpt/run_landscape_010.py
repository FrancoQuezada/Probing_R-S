"""Reproduce/resume the complete 010 experiment and independent scientific audits."""
from pathlib import Path
import os,subprocess,time,json,hashlib
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'resultados/010probing_instance_landscape_search'
ENV=os.environ.copy();ENV.update(JULIA_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',GKSwstype='100')
JULIA=['julia','--startup-file=no','--project=resultados/003probing_softmax_20261021/entorno']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def run(command,name):
    (OUT/'logs').mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    with (OUT/'logs'/name).open('a') as log:
        subprocess.run(command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (OUT/'logs/resources.jsonl').open('a') as log:
        log.write(json.dumps(dict(command=command,wall_seconds=time.perf_counter()-start,threads=8,blas_threads=1))+'\n')

def main():
    run(JULIA+['010probing_instance_landscape_search.jl','--resume'],'workflow_resume.log')
    run(JULIA+['diagnostico_economico_010.jl'],'economic_attribution_resume.log')
    run(JULIA+['validar_contraste_trayectorias_010.jl'],'trajectory_contrast.log')
    run(['python3','interpretar_landscape_010.py',str(OUT)],'scientific_interpretation.log')
    complete=json.loads((OUT/'COMPLETE.json').read_text());assert complete['complete'] and not complete['quick']
    preserved=OUT/'validation/preserved_before.json'
    if preserved.exists():
        before=json.loads(preserved.read_text());changed=[name for name,value in before.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=value]
        assert not changed,changed
        (OUT/'validation/previous_files_verified.json').write_text(json.dumps(dict(passed=True,unchanged_files=len(before)),indent=2)+'\n')
    print(json.dumps(complete,indent=2))

if __name__=='__main__':main()
