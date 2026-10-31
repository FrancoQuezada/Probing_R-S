"""Preserve old artifacts, then run the additive Julia experiment."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
root=Path(__file__).resolve().parent
out=root/'resultados/016probing_green_bank_variants'
if '--help' in sys.argv:
    print('python3 run_green_bank_variants_016.py [--quick-test] [--resume] [--tests-only] [--no-plots] [--plots-only]')
    raise SystemExit(0)
manifest=out/'validation/preserved_before.json'
if not manifest.exists():
    manifest.parent.mkdir(parents=True,exist_ok=True)
    files=[p for folder in (root/'resultados').iterdir() if folder.is_dir() and folder!=out for p in folder.rglob('*') if p.is_file()]
    files += [p for p in root.iterdir() if p.is_file() and p.suffix in ('.jl','.py','.toml','.txt','.md')]
    manifest.write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2)+'\n')
env=dict(os.environ,GKSwstype='100',JULIA_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
subprocess.run(['julia','--startup-file=no','--project=resultados/003probing_softmax_20261021/entorno',
    '016probing_green_bank_variants.jl']+sys.argv[1:],cwd=root,env=env,check=True)
