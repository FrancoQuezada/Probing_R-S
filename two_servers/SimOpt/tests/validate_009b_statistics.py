import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze_009b as a
import numpy as np
import pandas as pd
from scipy import stats
from types import SimpleNamespace

assert np.allclose(a.holm([.01,.04,.03]),[.03,.06,.06])
rows=[]
for s in a.IDS:
    for aa,bb in a.PRIMARY+[(a.METHODS[3],a.METHODS[0])]:
        for estimand in ['holdout','direct']:
            for rep,x in enumerate([.5,1.,1.5,2.,2.5]):
                rows.append(dict(scenario=s,estimand=estimand,method_a=aa,method_b=bb,rep=rep,difference=x,evaluation_paired_se=.1))
c=a.compare(pd.DataFrame(rows));p=c[c.family=='primary_holdout_15']
assert len(p)==15 and len(c[c.family=='secondary_direct_hardmax'])==20
assert np.allclose(p.p_holm_family,a.holm(p.p_value))
expected=stats.ttest_1samp([.5,1.,1.5,2.,2.5],0)
assert np.allclose(p.p_value,expected.pvalue)
assert np.allclose(p.se,np.std([.5,1.,1.5,2.,2.5],ddof=1)/np.sqrt(5))
r=SimpleNamespace(delta_G3=1.,delta_G4=2.,covariance_33=.04,covariance_34=0.,covariance_44=.09,n_reference=30000)
t,p,rank,critical=a.vector_test(r)
assert np.isclose(t,1/.04+4/.09) and rank==2 and 0<=p<1e-10
assert np.isclose(critical,2*29999/29998*stats.f.ppf(.95,2,29998))
base=dict(n_equivalence_groups=2,modal_count=50,second_count=40,selection_reps=100,modal_group=1,best_A_group=1,best_B_group=1,
    modal_reference_status='certified_inside',standardized_gap_A=4.,relative_gap_low=0.,relative_gap_high=.1,cosine_low=.98,cosine_high=.999,local6_distance_low=.05)
cases=[base,dict(base,modal_count=95,second_count=5),dict(base,relative_gap_low=.3,relative_gap_high=.8),
    dict(base,relative_gap_high=.8,cosine_low=.7,cosine_high=.8),dict(base,relative_gap_high=1.,cosine_low=.5),dict(base,modal_count=85),dict(base,n_equivalence_groups=1)]
assert list(a.classify_points(pd.DataFrame(cases)).regime)==['R2','R1','R3','R3','indeterminate_gradient','indeterminate_selection','R1']
old=pd.read_csv(a.OLD/'structural_diagnostics.csv')
sel=pd.read_csv(a.OLD/'policy_selection_diagnostics.csv');sel=sel[sel.n_paths==500]
old=old.merge(sel[['scenario','point','selection_reps','modal_group_probability','second_group_probability']],on=['scenario','point'])
old['modal_count']=np.round(old.modal_group_probability*old.selection_reps).astype(int)
old['second_count']=np.round(old.second_group_probability*old.selection_reps).astype(int)
policies=pd.read_csv(a.OLD/'policy_value_diagnostics.csv').set_index(['scenario','point','policy'])
old['modal_reference_status']=[policies.loc[(r.scenario,r.point,r.modal_i),'reference_status'] for r in old.itertuples()]
old['local6_distance_low']=old.gradient_distance_low
assert list(a.classify_points(old).regime)==list(old.regime)
print('PASS: Holm principal exactly 15; separate secondary families; paired t/SE; vector test; all prior 80 classifications and seven regime cases.')
