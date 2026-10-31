"""Hand-calculated edge cases for hitting/censoring/accumulated-quality definitions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from analyze_anytime_012 import first_run,empirical_median,restricted_mean

assert first_run([8,9.6,8],10,.5)==(1,1)
assert first_run([9.6,8,8],10,.5)==(0,0)
assert first_run([8,8,8],10,.5)==(None,None)
assert first_run([9.6]*5+[8],10,.5,5)==(0,4)
assert first_run([9.6]*4+[8],10,.5,5)==(None,None)
assert first_run([8,9.6,9.7,9.6,9.8,9.6],10,.5,5)==(1,5)
x=np.zeros(101);x[-1]=10
assert first_run(x,10,.5)==(100,100)
assert first_run(x,10,.5,5)==(None,None)
assert empirical_median([1,2,3,100,100],[1,1,1,0,0])==3
assert np.isnan(empirical_median([1,2,100,100,100],[1,1,0,0,0]))
assert empirical_median([0,0,0,10,10],[1,1,1,1,1])==0
assert restricted_mean([1,4,np.nan],[1,1,0],100)==35
assert restricted_mean([np.nan,np.nan],[0,0],100)==100
v=np.array([1.,3.,8.]);A=np.cumsum(v)/np.arange(1,4)
assert np.array_equal(A,[1,2,4]) and np.array_equal(3-A,[2,1,-1])
print('14 hand-calculated hitting/censoring/accumulated-quality checks passed')
