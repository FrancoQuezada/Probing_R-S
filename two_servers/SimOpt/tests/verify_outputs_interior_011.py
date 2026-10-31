"""Verify completed 011 numerical artifacts, seeds, provenance and plot files."""
from pathlib import Path
import sys,json,hashlib,re
try:import tomllib
except ModuleNotFoundError:import tomli as tomllib
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'resultados/011probing_high_contrast_interior'
if len(sys.argv)>1:OUT=Path(sys.argv[1]).resolve()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    meta=json.loads((OUT/'COMPLETE.json').read_text())
    assert meta['complete'] and not meta['quick']
    cfg=tomllib.loads((OUT/'configs/experiment.toml').read_text())['science']
    ids=pd.read_csv(OUT/'configs/selected_instances.csv').scenario.tolist()
    methods=meta['methods'];reps=cfg['reps'];steps=cfg['max_iters']
    t=pd.read_csv(OUT/'tables/traces_all.csv');z=pd.read_csv(OUT/'tables/coordinates_all.csv')
    f=pd.read_csv(OUT/'tables/finals_all.csv');s=pd.read_csv(OUT/'tables/seeds_all.csv',dtype={'seed':'int64'})
    keys=['scenario','rep','method','iteration']
    expected=len(ids)*reps*len(methods)*(steps+1)
    assert len(t)==len(z)==expected and not t.duplicated(keys).any()
    assert np.isfinite(t.objective).all()
    tz=t.merge(z,on=keys,suffixes=('_t','_z'),validate='one_to_one')
    assert np.array_equal(tz.z3_t,tz.z3_z) and np.array_equal(tz.z4_t,tz.z4_z)
    assert z[['z3','z4']].ge(0).all().all() and z[['z3','z4']].le(1).all().all()
    for (case,rep,method),g in t.groupby(['scenario','rep','method']):
        assert np.array_equal(g.iteration.to_numpy(),np.arange(steps+1))
        assert np.allclose(g[['z3','z4']].iloc[0],cfg['Z0'],atol=0,rtol=0)
    assert not f.duplicated(['scenario','rep','method']).any()
    for _,g in f.groupby(['scenario','rep']):assert set(methods+['GridBest','Initial'])<=set(g.method)
    assert np.isfinite(f.objective_holdout).all() and f.evaluation_se.ge(0).all()
    assert np.allclose(f.gross-f.probing_cost,f.objective_holdout,rtol=1e-12,atol=1e-11)
    assert f.selection_paths.eq(cfg['selection_paths']).all() and f.holdout_paths.eq(cfg['final_paths']).all()
    assert not s.seed.duplicated().any() and s.seed.gt(0).all()
    geometry_seeds=set()
    paths=list((OUT/'high').glob('*_surface.csv'))+list((OUT/'refine').glob('*_surface.csv'))
    paths+=list((OUT/'screen').glob('*_surface.csv'))+[OUT/'calibration/slopes.csv']
    for path in paths:
        a=pd.read_csv(path,dtype={'seed':'int64'})
        if 'seed' in a:geometry_seeds.update(a.seed.tolist())
        if {'gross','cost','objective'}<=set(a.columns):
            assert np.allclose(a.gross-a.cost,a.objective,rtol=1e-12,atol=1e-11)
    assert not(set(s.seed)&geometry_seeds),'Selection and comparison banks overlap'
    c=pd.read_csv(OUT/'tables/contrast_per_iteration.csv');cs=pd.read_csv(OUT/'tables/contrast_summary.csv')
    for row in cs.itertuples():
        q=t[(t.scenario==row.scenario)&(t.rep==row.rep)]
        if not row.with_fixed:q=q[q.method!='Fixed']
        p=q.pivot(index='iteration',columns='method',values='objective')
        vals=p.max(axis=1)-p.min(axis=1)
        saved=c[(c.scenario==row.scenario)&(c.rep==row.rep)&(c.with_fixed==row.with_fixed)].sort_values('iteration')
        assert np.allclose(vals,saved.contrast)
        assert row.tail_points==81 and np.isclose(row.C_tail,vals.loc[20:100].mean())
        assert np.isclose(row.C_mean50,vals.loc[50:100].mean())
        assert np.isclose(row.C_mean80,vals.loc[80:100].mean())
        assert np.isclose(row.C_final,vals.iloc[-1])
    assert len(pd.read_csv(OUT/'configs/policies.csv'))==144
    for source,h in cfg['sources'].items():assert sha(ROOT/source)==h,(source,'changed scientific source')
    for source,h in meta['analysis_sha256'].items():assert sha(ROOT/source)==h
    for source,h in meta.get('additional_sources_sha256',{}).items():assert sha(ROOT/source)==h
    old=ROOT/'resultados/010probing_instance_landscape_search'
    reuse=pd.read_csv(OUT/'configs/reused_comparisons.csv')
    identical_controls=0
    for case in ['L000','L025','L026']:
        for path in (OUT/'comparisons'/case).glob('rep_*.csv'):
            original=old/'comparisons'/case/path.name
            assert sha(path)==sha(original),path
            identical_controls+=1
    pngs=list((OUT/'figures').glob('*.png'));pdfs=list((OUT/'figures').glob('*.pdf'))
    assert len(pngs)==meta['png_files'] and len(pngs)==len(pdfs)
    for path in pngs:
        with Image.open(path) as im:
            assert min(im.size)>200;im.verify()
    for path in pdfs:
        raw=path.read_bytes();assert raw.startswith(b'%PDF-') and b'%%EOF' in raw[-64:]
    for target in re.findall(r'(?:href|src)="([^"]+)"',(OUT/'index.html').read_text()):
        if not target.startswith(('http:','https:','#')):assert (OUT/target).is_file(),target
    for case in ids:
        for stem in ['surface','z3_trajectory_rep01','z4_trajectory_rep01','oracle_trajectories_rep01','mean_convergence_full','mean_convergence_zoom','mean_contrast','actual_saturation','independent_temporal_contrast']:
            assert (OUT/'figures'/f'{case}_{stem}.png').is_file()
        for rep in range(1,reps+1):
            for stem in ['convergence_rep%02d_full','convergence_rep%02d_zoom','contrast_rep%02d']:
                assert (OUT/'figures'/f'{case}_{stem%rep}.png').is_file()
    previous=json.loads((OUT/'validation/previous_files_verified.json').read_text())
    assert previous['passed']
    result=dict(passed=True,trace_rows=expected,method_searches=len(ids)*reps*len(methods),all_101_points=True,
        seeds=len(s),selection_banks_disjoint=True,control_csv_byte_identical=identical_controls,
        unchanged_previous_files=previous['unchanged_files'],png_verified=len(pngs),pdf_verified=len(pdfs),
        scientific_sources_unchanged=True)
    (OUT/'validation/output_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
