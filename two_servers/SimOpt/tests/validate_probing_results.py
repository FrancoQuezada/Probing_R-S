#!/usr/bin/env python3
"""Independent audit of exported full experiments; no simulations are run."""
from pathlib import Path
import sys, json, hashlib, warnings
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore",UserWarning)
    import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NAMES=["005probing_softmax_multiseed","006probing_softmax_sample_efficiency",
       "007probing_green_softmax","008probing_probability_softmax"]

def audit(out):
    cfg=tomllib.loads((out/"configuration.toml").read_text());stage=cfg["stage"]
    finals=pd.read_csv(out/"final_evaluations.csv");diag=pd.read_csv(out/"iteration_diagnostics.csv")
    coords=pd.read_csv(out/"coordinates.csv");traces=pd.read_csv(out/"convergence_traces.csv")
    paired=pd.read_csv(out/"paired_replications.csv")
    diag_lookup=diag.set_index(["rep","n_paths","method","iteration"])
    assert diag_lookup.index.is_unique
    reps=cfg["n_reps"]
    assert not cfg["quick"]
    assert len(list((out/"checkpoints").glob("*_complete.jls")))==reps*len(cfg["n_paths"])
    methods=1+len(cfg["betas"])+(stage==8)
    assert len(finals)==reps*len(cfg["n_paths"])*methods*(2 if stage==6 else 1)
    assert (finals.search_paths==finals.n_paths*finals.iteration).all()
    assert (traces.search_paths==traces.n_paths*traces.iteration).all()
    assert ((coords[["z3","z4"]]>=0)&(coords[["z3","z4"]]<=1)).all().all()
    assert np.isfinite(diag[["g3","g4","gradient_norm"]]).all().all()
    assert (finals.objective_direct>=finals.objective_holdout-1e-10).all()
    assert np.allclose(coords[coords.iteration==0][["z3","z4"]],cfg["model"]["Z0"])
    assert all(len(x)==reps for _,x in finals.groupby(["n_paths","analysis","method"]))
    for r in paired.itertuples():
        f=finals[(finals.rep==r.rep)&(finals.n_paths==r.n_paths)&(finals.analysis==r.analysis)]
        column="objective_"+r.estimand
        av=f[f.method==r.method_a][column].iloc[0];bv=f[f.method==r.method_b][column].iloc[0]
        assert np.isclose(r.difference,av-bv,atol=1e-11,rtol=1e-11)
    for file,sha in cfg["source_sha256"].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha,file
    if stage in (7,8):
        for path in sorted((out/"replications").glob("*_weights.csv")):
            w=pd.read_csv(path)
            sums=w.groupby(["method","iteration"])[["weight","soft_weight"]].sum()
            assert np.allclose(sums,1.,atol=1e-12)
            assert ((w[["weight","soft_weight"]]>=0)&(w[["weight","soft_weight"]]<=1)).all().all()
            for (method,k),g in w.groupby(["method","iteration"],sort=False):
                rep=int(g.rep.iloc[0]);n=int(g.n_paths.iloc[0])
                d=diag_lookup.loc[(rep,n,method,k)]
                values=g.value.to_numpy();soft=g.soft_weight.to_numpy()
                beta=float(method.split("beta=")[1]) if "beta=" in method else .1
                exp=np.exp(beta*(values-values.max()));exp/=exp.sum()
                assert np.allclose(soft,exp,atol=2e-14)
                G=g[["G3","G4"]].to_numpy();weight=g.weight.to_numpy()
                gradient=G.T@weight-np.asarray(cfg["model"]["ALPHA"])
                assert np.allclose(gradient,[d.g3,d.g4],atol=1e-9,rtol=1e-10)
                assert np.isclose(d.K_eff,1/np.sum(weight**2),atol=1e-10)
                assert np.isclose(d.soft_K_eff,1/np.sum(soft**2),atol=1e-10)
                if stage==7:assert d.banks==k
    if stage==6:
        expected=np.where(finals.analysis=="fixed_iterations",cfg["max_iters"],np.maximum(1,cfg["budget"]//finals.n_paths))
        assert (finals.iteration==expected).all()
    if stage==8 and (out/"psoft_same_bank_policy_weights.csv").exists():
        sw=pd.read_csv(out/"psoft_same_bank_policy_weights.csv")
        columns=[c for c in sw.columns if c.startswith("soft_weight")]+["weight"]
        assert np.isfinite(sw[columns]).all().all()
        assert np.allclose(sw.groupby(["rep","iteration"])[columns].sum(),1.,atol=1e-12)
        for beta in cfg["betas"]:
            col="soft_weight" if beta==.1 else "soft_weight_beta_"+str(beta).replace(".","p")
            exponential=np.exp(beta*(sw.value-sw.groupby(["rep","iteration"]).value.transform("max")))
            expected=exponential/exponential.groupby([sw.rep,sw.iteration]).transform("sum")
            assert np.allclose(sw[col],expected,atol=1e-12)
    result=dict(passed=True,stage=stage,replications=reps*len(cfg["n_paths"]),
                methods=methods,endpoint_rows=len(finals),gradient_rows=len(diag),
                coordinates=len(coords),paired_rows=len(paired))
    (out/"validacion").mkdir(exist_ok=True)
    (out/"validacion/result_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return pd.read_csv(out/"seeds.csv")

if __name__=="__main__":
    names=sys.argv[1:] or NAMES
    seeds=[audit(Path(name) if Path(name).is_absolute() else ROOT/"resultados"/name) for name in names]
    values=pd.concat(seeds).seed
    assert values.nunique()==len(values),"Seed streams overlap between stages/repetitions"
    print("Exported results and seed separation validated.",flush=True)
