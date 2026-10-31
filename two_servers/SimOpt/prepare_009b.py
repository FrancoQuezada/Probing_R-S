"""Freeze structural locations using only archived 009 diagnostics/coordinates."""
from pathlib import Path
import argparse, hashlib, json, warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'resultados/009probing_parameter_regimes'
SCENARIOS=['S000','S001','S010','S101','S111']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def prepare(out):
    out=Path(out).resolve();(out/'configs').mkdir(parents=True,exist_ok=True)
    used=[SOURCE/'structural_diagnostics.csv',SOURCE/'configs/configuration.toml',SOURCE/'configs/policies.csv',SOURCE/'scenarios.csv']
    structural=pd.read_csv(used[0]);rows=[]
    for scenario in ['S001','S010','S111']:
        grid=structural[structural.scenario==scenario].copy();chosen=[]
        for role,column in [('initial',None),('selection_uncertainty','selection_uncertainty'),('gradient_difference','relative_gradient_gap')]:
            eligible=grid[~grid.point.isin(chosen)]
            r=(eligible[eligible.point=='Z0'].iloc[0] if column is None else eligible.sort_values([column,'point'],ascending=[False,True],kind='stable').iloc[0])
            chosen.append(r.point)
            rows.append(dict(scenario=scenario,point=f'P{len(chosen)-1}',role=role,z3=r.z3,z4=r.z4,source_point=r.point,
                nearest_old_point=r.point,old_regime=r.regime,selection_basis='fixed initial point' if column is None else f'descending {column}; ties by point ID; exclude duplicates'))
        files=sorted((SOURCE/'pilot'/scenario).glob('*_coordinates.csv'));used.extend(files)
        coordinates=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
        tail=coordinates[coordinates.iteration.between(81,100)].copy()
        center=tail[['z3','z4']].median().to_numpy();tail['distance']=np.sum((tail[['z3','z4']].to_numpy()-center)**2,axis=1)
        existing=np.array([[r['z3'],r['z4']] for r in rows if r['scenario']==scenario])
        tail=tail[np.min(np.linalg.norm(tail[['z3','z4']].to_numpy()[:,None,:]-existing[None,:,:],axis=2),axis=1)>1e-12]
        r=tail.sort_values(['distance','method','rep','iteration'],kind='stable').iloc[0]
        near=grid.iloc[np.argmin(np.sum((grid[['z3','z4']].to_numpy()-r[['z3','z4']].to_numpy(float))**2,axis=1))]
        rows.append(dict(scenario=scenario,point='P3',role='visited_region',z3=r.z3,z4=r.z4,
            source_point=f'pilot/{r.method}/rep={r.rep}/iteration={r.iteration}',nearest_old_point=near.point,old_regime='new_location',
            selection_basis='archived iterate nearest coordinate-wise median of iterations 81:100, all five methods and all five pilot replications equally; no objective values used'))
    content=pd.DataFrame(rows).to_csv(index=False,float_format='%.17g')
    path=out/'selected_points.csv'
    if path.exists() and path.read_text()!=content:raise RuntimeError('Refusing to alter frozen points')
    path.write_text(content)
    point_toml='\n'.join('[[points]]\n'+'\n'.join(f'{key} = {json.dumps(value,ensure_ascii=False)}' for key,value in row.items())+'\n' for row in rows)
    target_points=out/'configs/selected_points.toml'
    if target_points.exists() and target_points.read_text()!=point_toml:raise RuntimeError('Frozen TOML points changed')
    target_points.write_text(point_toml)
    manifest=dict(criteria='Z0; highest old selection uncertainty; highest remaining relative gradient distance; actual iterate nearest pooled tail median. No winner/performance criterion.',
        confirmation_scenarios=SCENARIOS,points_sha256=sha(path),inputs={str(f.relative_to(ROOT)):sha(f) for f in used})
    target=out/'configs/point_selection.json';serialized=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    if target.exists() and target.read_text()!=serialized:raise RuntimeError('Frozen selection evidence changed')
    target.write_text(serialized)
    for name,source in [('source_009_configuration.toml',used[1]),('policies.csv',used[2])]:
        dest=out/'configs'/name
        if dest.exists() and dest.read_bytes()!=source.read_bytes():raise RuntimeError('Source configuration changed')
        dest.write_bytes(source.read_bytes())
    model=pd.read_csv(SOURCE/'scenarios.csv');model[model.scenario.isin(SCENARIOS)].to_csv(out/'configs/scenarios.csv',index=False)
    preserved=out/'configs/preserved_files.json'
    if not preserved.exists():
        paths=[f for f in ROOT.glob('*.jl') if not f.name.startswith(('009b','probing_validation'))]
        paths += [f for f in (ROOT/'resultados').rglob('*') if f.is_file() and not f.is_relative_to(ROOT/'resultados/009b_probing_regime_validation')]
        preserved.write_text(json.dumps({str(f.relative_to(ROOT)):sha(f) for f in sorted(set(paths))},indent=2,sort_keys=True)+'\n')
    print(pd.DataFrame(rows)[['scenario','point','role','z3','z4','source_point']].to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');prepare(parser.parse_args().output)
