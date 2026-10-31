"""Verify 012 estimands, budgets, censoring, statistical inputs and final artifacts."""
from pathlib import Path
import hashlib,json,re,sys
try:import tomllib
except ModuleNotFoundError:import tomli as tomllib
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'resultados/012probing_anytime_budget'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def main():
 o=OUT;c=json.loads((o/'COMPLETE.json').read_text());config=tomllib.loads((o/'configs/experiment.toml').read_text());s=config['science']
 assert c['complete'] and not c['quick'] and c['objective_rows']==3535
 assert c['original_repetitions']==5 and c['new_repetitions']==0
 assert c['exact_original_replays']==35
 for f,h in s['sources'].items():assert sha(ROOT/f)==h,f
 for f,h in c['analysis_sha256'].items():assert sha(ROOT/f)==h and sha(o/'reproduction'/f)==h,f
 source=ROOT/'resultados/011probing_high_contrast_interior'
 assert tomllib.loads((o/'configs/I0252.toml').read_text())==tomllib.loads((source/'configs/main_instance_parameters.toml').read_text())
 for f,h in tomllib.loads((o/'configs/input_sha256.toml').read_text()).items():assert sha(source/f)==h,f
 d=pd.read_csv(o/'tables/anytime_objectives_budget_runtime.csv');a=pd.read_csv(o/'tables/accumulated_quality.csv')
 refs=pd.read_csv(o/'tables/reference_by_rep.csv');events=pd.read_csv(o/'tables/hitting_times.csv',dtype={'epsilon_name':str})
 targets=pd.read_csv(o/'tables/target_summary.csv',dtype={'epsilon_name':str})
 assert len(d)==3535 and not d.duplicated(['cohort','rep','method','iteration']).any()
 assert np.isfinite(d.objective).all() and d.cohort.eq('original_011').all()
 phases=['generation_seconds','materialization_seconds','policy_seconds','gradient_seconds','bootstrap_seconds','update_seconds','other_seconds']
 assert np.allclose(d[phases].sum(axis=1),d.search_seconds,rtol=1e-7,atol=1e-9)
 assert (d[phases]>=0).all().all()
 for (rep,m),g in d.groupby(['rep','method']):
  g=g.sort_values('iteration');assert np.array_equal(g.iteration,np.arange(101))
  assert np.allclose(g[['z3','z4']].iloc[0],[.12,.10])
  assert g.iteration.iloc[0]==0 and g.cumulative_trajectories.iloc[0]==0 and g.cumulative_search_seconds.iloc[0]==0
  expected=np.minimum(g.iteration,1)*500 if m=='Fixed' else g.iteration*500
  assert np.array_equal(g.cumulative_trajectories,expected)
  assert g.new_trajectories.sum()==(500 if m=='Fixed' else 50000)
  assert g.new_policy_path_evaluations.sum()==144*g.new_trajectories.sum()
  if m in ['Green','GreenSoftmax050']:
   assert np.array_equal(g.history_banks.iloc[1:],np.arange(1,101))
   assert np.array_equal(g.lr_path_evaluations.iloc[1:],500*(np.arange(1,101)+1))
  if m=='PSoft300':assert g.bootstrap_path_draws.sum()==500*300*100
  else:assert g.bootstrap_path_draws.eq(0).all()
  assert g.cumulative_search_seconds.is_monotonic_increasing
  assert np.allclose(g.search_seconds.cumsum(),g.cumulative_search_seconds)
  original=pd.read_csv(source/'comparisons/I0252'/f'rep_{rep:02d}_traces.csv');q=original[original.method==m].sort_values('iteration')
  assert np.allclose(q.objective,g.objective,rtol=0,atol=1e-10)
  assert np.allclose(q[['z3','z4']],g[['z3','z4']],rtol=0,atol=1e-14)
  rr=refs[refs.rep==rep].iloc[0];assert np.allclose(g.reference_objective,rr.objective,rtol=0,atol=1e-12)
  assert g.evaluation_seed.eq(rr.seed).all() and g.n_paths.eq(1000).all()
  aa=a[(a.rep==rep)&(a.method==m)].sort_values('iteration')
  expectedA=g.objective.cumsum().to_numpy()/np.arange(1,102)
  assert np.allclose(aa.accumulated_objective,expectedA)
  assert np.allclose(aa.accumulated_deficit,rr.objective-expectedA)
  for ev in events[(events.rep==rep)&(events.method==m)].itertuples():
   if ev.achieved:
    start=int(ev.start_iteration);end=int(ev.confirmation_iteration)
    assert end==start+(4 if ev.kind=='sustained5' else 0)
    assert (g.reference_objective.iloc[start:end+1]-g.objective.iloc[start:end+1]<=ev.epsilon+1e-11).all()
    assert ev.hitting_simulation_budget==g.cumulative_trajectories.iloc[start]
    assert ev.confirmation_simulation_budget==g.cumulative_trajectories.iloc[end]
    assert np.isclose(ev.confirmation_search_seconds,g.cumulative_search_seconds.iloc[end])
   else:
    assert pd.isna(ev.start_iteration) and pd.isna(ev.confirmation_iteration)
    assert pd.isna(ev.hitting_simulation_budget) and pd.isna(ev.confirmation_search_seconds)
    assert ev.observed_iteration==100 and ev.censor_iteration==100
 assert (a.accumulated_deficit<0).any() or (d.deficit<0).any()
 for row in targets.itertuples():
  g=events[(events.cohort==row.cohort)&(events.method==row.method)&(events.epsilon_name==row.epsilon_name)&(events.kind==row.kind)]
  assert row.repetitions==5 and np.isclose(row.success_proportion,g.achieved.mean())
  contribution=np.where(g.achieved,g.confirmation_iteration,100)
  assert np.isclose(row.restricted_mean_hitting_iteration,contribution.mean())
  if g.achieved.sum()<3:assert pd.isna(row.median_confirmation_iteration)
  else:assert row.median_confirmation_iteration==g.loc[g.achieved,'confirmation_iteration'].sort_values().iloc[2]
 plan=pd.read_csv(o/'configs/validation_plan.csv',dtype={'epsilon_name':str});v=pd.read_csv(o/'tables/crossing_validation.csv',dtype={'epsilon_name':str})
 assert not v.duplicated(['rep','method','epsilon_name','kind','iteration']).any()
 expected_rows=0
 for p in plan[plan.achieved].itertuples():
  q=v[(v.rep==p.rep)&(v.method==p.method)&(v.epsilon_name==p.epsilon_name)&(v.kind==p.kind)]
  wanted=[int(x) for x in str(p.validation_iterations).split(';')];expected_rows+=len(wanted)
  assert sorted(q.iteration.tolist())==sorted(wanted)
 assert len(v)==expected_rows and v.n_paths.eq(5000).all()
 assert v.paired_mc_se.ge(0).all() and np.allclose(v.reference_objective-v.objective,v.deficit,atol=1e-10)
 assert (v.meets_threshold==(v.deficit<=v.epsilon)).all()
 val_seeds=set(v.evaluation_seed.astype('int64'));main_seeds=set(refs.seed.astype('int64'))
 assert len(val_seeds)==5 and not val_seeds&main_seeds
 for rep in range(1,6):
  seeds=pd.read_csv(source/'comparisons/I0252'/f'rep_{rep:02d}_seeds.csv')
  searchseeds=set(seeds[seeds.stream.isin(['fixed','search','bootstrap'])].seed.astype('int64'))
  assert not searchseeds&(val_seeds|main_seeds)
 ledger=pd.read_csv(o/'tables/external_evaluation_budget.csv')
 assert ledger[ledger.phase=='independent_crossing_validation'].new_unique_exogenous_paths.sum()==25000
 assert ledger[ledger.phase=='main_reference_and_reconstruction'].new_unique_exogenous_paths.sum()==0
 cachecheck=tomllib.loads((o/'validation/point_cache_checks.toml').read_text());assert cachecheck['passed'] and cachecheck['exact_independent_recomputations']==10
 for name in ['quick_test','quick_recomputed']:
  q=json.loads((o/name/'COMPLETE.json').read_text());assert q['complete'] and q['quick']
 assert json.loads((o/'quick_test/COMPLETE.json').read_text())['new_repetitions']==1
 assert tomllib.loads((o/'quick_recomputed/configs/experiment.toml').read_text())['science']['recompute_evaluations']
 png=list((o/'figures').glob('*.png'));pdf=list((o/'figures').glob('*.pdf'));assert len(png)==len(pdf)==c['png_files']
 for f in png:
  with Image.open(f) as im:assert min(im.size)>200;im.verify()
 for f in pdf:
  raw=f.read_bytes();assert raw.startswith(b'%PDF-') and b'%%EOF' in raw[-64:]
 for target in re.findall(r'(?:href|src)="([^"]+)"',(o/'index.html').read_text()):
  if not target.startswith(('https:','http:','#')):assert (o/target).is_file(),target
 before=json.loads((o/'validation/preserved_before.json').read_text())
 changed=[name for name,h in before.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
 assert not changed,changed
 result=dict(passed=True,original_files_unchanged=len(before),exact_original_replays=35,decision_points=3535,
  model_unchanged=True,original_objectives_reused=True,fixed_budget=500,other_method_budget=50000,
  bootstrap_not_counted_as_system_paths=True,reference_comparable=True,censoring_preserved=True,
  validation_records=len(v),validation_rows_unique_and_complete=True,point_cache_exact_checks=10,
  png_verified=len(png),pdf_verified=len(pdf),quick_extra_and_recomputed_modes_passed=True)
 (o/'validation/output_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
