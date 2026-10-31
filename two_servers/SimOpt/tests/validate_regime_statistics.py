#!/usr/bin/env python3
"""Analytic fixtures for paired inference and full factorial coding."""
import sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import analyze_parameter_regimes as A
import numpy as np
import pandas as pd
from scipy import stats

np.testing.assert_allclose(A.holm([.01,.04,.03]),[.03,.06,.06])
rows=[]
for s in A.SCENARIOS:
    x,y,z=[2*int(v)-1 for v in s[1:]]
    for r,noise in enumerate([-.2,-.1,0,.1,.2],1):
        rows.append(dict(phase="pilot",scenario=s,estimand="holdout",method_a=A.METHOD_NAMES[1],
            method_b="Fresh",rep=r,difference=5+x-2*y+.5*z+.3*x*y+noise,evaluation_paired_se=.01))
f=A.factorial(pd.DataFrame(rows)).set_index("factor")
expected={"A":2.,"B":-4.,"C":1.,"AB":.6,"AC":0.,"BC":0.,"ABC":0.}
for name,value in expected.items():
    np.testing.assert_allclose(f.loc[name,"effect"],value,atol=1e-13)
    np.testing.assert_allclose(f.loc[name,"se"],.05,atol=1e-13)
    np.testing.assert_allclose(f.loc[name,"df"],32.,atol=1e-10)
c=A.compare(pd.DataFrame(rows))
for r in c.itertuples():
    x=pd.DataFrame(rows).query("scenario == @r.scenario").difference.to_numpy()
    np.testing.assert_allclose(r.p_value,stats.ttest_1samp(x,0).pvalue)
    np.testing.assert_allclose(r.se,stats.sem(x))
    assert r.p_holm_global>=r.p_holm_scenario-1e-15
print("Analytic Holm, paired t and seven factorial effects: PASS")
