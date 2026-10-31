"""One entrypoint for the additional-method search and all figure families."""
from pathlib import Path
import os,subprocess,sys
root=Path(__file__).resolve().parent
env=dict(os.environ,GKSwstype='100',JULIA_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
julia=['julia','--startup-file=no','--project=resultados/003probing_softmax_20261021/entorno']
args=sys.argv[1:]
subprocess.run(julia+['013probing_green_softmax010.jl']+args,cwd=root,env=env,check=True)
quick='--quick-test' in args
if not quick:
    subprocess.run(julia+['audit_green_softmax010_013.jl'],cwd=root,env=env,check=True)
out=root/'resultados/013probing_green_softmax010'
if quick:out=out/'quick_test'
subprocess.run([sys.executable,'plot_remaining_families_013.py',str(out)],cwd=root,env=env,check=True)
