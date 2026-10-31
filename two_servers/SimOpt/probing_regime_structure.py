#!/usr/bin/env python3
"""Structural selection is independent of all algorithm performance results."""
import sys, json, hashlib, warnings, math
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore",UserWarning)
    import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parent
SCENARIOS=[f"S{a}{b}{c}" for a in range(2) for b in range(2) for c in range(2)]

def parts(out,root,suffix):
    files=sorted((out/root).glob(f"S*/*_{suffix}.csv"))
    return pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else pd.DataFrame()

def wilson(k,n):
    if n==0:return np.nan,np.nan
    z=stats.norm.ppf(.975);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)

def entropy(counts):
    p=np.asarray(counts,float);p=p[p>0]/p.sum()
    return -np.sum(p*np.log(p))

def classify(out,cfg,fingerprint):
    summaries=parts(out,"characterization","summary")
    if summaries.empty:raise RuntimeError("No structural characterization found")
    policies=parts(out,"characterization","policy_values")
    pairs=parts(out,"characterization","gradient_pairs")
    routing=parts(out,"characterization","routing_pairs")
    draws=parts(out,"characterization","selection_draws")
    policies.to_csv(out/"policy_value_diagnostics.csv",index=False)
    pairs.to_csv(out/"policy_gradient_diagnostics.csv",index=False)
    routing.to_csv(out/"routing_disagreement.csv",index=False)
    parts(out,"characterization","routing_policies").to_csv(out/"routing_policy_diagnostics.csv",index=False)
    selection=[]; frequencies=[]
    for (s,p,n),g in draws.groupby(["scenario","point","n_paths"],sort=True):
        total=len(g);outside=int(g.outside_nominal_epsilon.sum())
        certain=int((g.reference_status=="certified_outside").sum());ambiguous=int((g.reference_status=="reference_ambiguous").sum())
        counts=g.policy.value_counts();groups=g.equivalence_group.value_counts()
        modal=int(groups.max());lo,hi=wilson(modal,total);elo,ehi=wilson(outside,total)
        selection.append(dict(scenario=s,point=p,n_paths=n,selection_reps=total,
            nominal_selection_error=outside/total,error_ci95_low=elo,error_ci95_high=ehi,
            certified_outside_frequency=certain/total,reference_ambiguous_frequency=ambiguous/total,
            policy_entropy=entropy(counts),equivalence_group_entropy=entropy(groups),
            n_policies_selected=len(counts),n_groups_selected=len(groups),modal_group_probability=modal/total,
            modal_wilson_low=lo,modal_wilson_high=hi,second_group_probability=groups.iloc[1]/total if len(groups)>1 else 0.))
        pg=policies[(policies.scenario==s)&(policies.point==p)].set_index("policy")
        for policy in range(1,145):
            count=int(counts.get(policy,0));flo,fhi=wilson(count,total)
            frequencies.append(dict(scenario=s,point=p,n_paths=n,policy=policy,frequency=count/total,
                count=count,ci95_low=flo,ci95_high=fhi,equivalence_group=int(pg.loc[policy,"equivalence_group"])))
    sel=pd.DataFrame(selection);sel.to_csv(out/"policy_selection_diagnostics.csv",index=False)
    pd.DataFrame(frequencies).to_csv(out/"policy_selection_frequencies.csv",index=False)
    structural=[]
    for r in summaries.itertuples():
        s=sel[(sel.scenario==r.scenario)&(sel.point==r.point)&(sel.n_paths==500)].iloc[0]
        i,j=sorted([r.modal_i,r.modal_j])
        pair=pairs[(pairs.scenario==r.scenario)&(pairs.point==r.point)&(pairs.policy_i==i)&(pairs.policy_j==j)].iloc[0]
        route=routing[(routing.scenario==r.scenario)&(routing.point==r.point)&(routing.policy_i==i)&(routing.policy_j==j)].iloc[0]
        policy=policies[(policies.scenario==r.scenario)&(policies.point==r.point)&(policies.policy==r.modal_i)].iloc[0]
        stable=(s.modal_group_probability>=cfg["stable_modal_min"] and s.modal_wilson_low>=cfg["stable_wilson_min"]
            and r.modal_group==r.best_A_group==r.best_B_group and policy.reference_status=="certified_inside"
            and r.standardized_gap_A>=3)
        uncertain=(s.modal_group_probability<=cfg["uncertain_modal_max"] and s.modal_wilson_high<=cfg["uncertain_wilson_max"]
            and s.second_group_probability>=.05)
        if r.n_equivalence_groups==1 or stable:regime="R1"
        elif uncertain and pair.relative_gap_high<=cfg["relative_gradient_boundary"] and pair.cosine_low>=cfg["similar_cosine_min"]:regime="R2"
        elif uncertain and (pair.relative_gap_low>cfg["relative_gradient_boundary"] or (pair.cosine_high<cfg["different_cosine_max"] and pair.distance_low>0)):regime="R3"
        elif uncertain:regime="indeterminate_gradient"
        else:regime="indeterminate_selection"
        row=r._asdict();row.update(regime=regime,selection_uncertainty=1-s.modal_group_probability,
            selection_entropy=s.equivalence_group_entropy,nominal_selection_error=s.nominal_selection_error,
            certified_selection_error=s.certified_outside_frequency,reference_ambiguity=s.reference_ambiguous_frequency,
            modal_wilson_low=s.modal_wilson_low,modal_wilson_high=s.modal_wilson_high,
            gradient_distance=pair.gradient_distance,gradient_distance_low=pair.distance_low,gradient_distance_high=pair.distance_high,
            relative_gradient_gap=pair.relative_gradient_gap,relative_gap_low=pair.relative_gap_low,relative_gap_high=pair.relative_gap_high,
            gradient_cosine=pair.cosine,cosine_low=pair.cosine_low,cosine_high=pair.cosine_high,
            routing_disagreement=route.disagreement,routing_disagreement_se=route.disagreement_se,
            projected_step1_difference=pair.projected_step1_difference,projected_step100_difference=pair.projected_step100_difference)
        structural.append(row)
    structural=pd.DataFrame(structural).sort_values(["scenario","point"])
    structural.to_csv(out/"structural_diagnostics.csv",index=False)
    rows=[]
    for scenario,g in structural.groupby("scenario",sort=True):
        count=g.regime.value_counts()
        label="R3" if count.get("R3",0)>0 else "R2" if count.get("R2",0)>0 else "R1" if count.get("R1",0)==len(g) else "indeterminate"
        rows.append(dict(scenario=scenario,regime=label,n_points=len(g),R1_points=int(count.get("R1",0)),
            R2_points=int(count.get("R2",0)),R3_points=int(count.get("R3",0)),
            indeterminate_points=int(sum(v for k,v in count.items() if k.startswith("indeterminate"))),
            selection_uncertainty=g.selection_uncertainty.mean(),selection_error=g.nominal_selection_error.mean(),
            certified_selection_error=g.certified_selection_error.mean(),reference_ambiguity=g.reference_ambiguity.mean(),
            mean_gap=g.gap_A.mean(),mean_standardized_gap=g.standardized_gap_A.mean(),
            gradient_distance=g.gradient_distance.mean(),relative_gradient_gap=g.relative_gradient_gap.mean(),
            gradient_cosine=g.gradient_cosine.mean(),routing_disagreement=g.routing_disagreement.mean()))
    regimes=pd.DataFrame(rows);regimes.to_csv(out/"tables/structural_scenario_summary.csv",index=False)
    complete=set(regimes.scenario)==set(SCENARIOS) and (regimes.n_points==10).all()
    selected=["S000"];represented={}
    if complete:
        for regime in ("R1","R2","R3"):
            candidates=regimes[regimes.regime==regime].copy()
            if candidates.empty:continue
            candidates=candidates.sort_values([regime+"_points","selection_uncertainty","scenario"],
                ascending=[False,regime=="R1",True],kind="stable")
            choice=candidates.scenario.iloc[0];represented[regime]=choice
            if choice not in selected:selected.append(choice)
        digest=hashlib.sha256((out/"structural_diagnostics.csv").read_bytes()).hexdigest()
        text="fingerprint = "+json.dumps(fingerprint)+"\nscenarios = "+json.dumps(sorted(selected))+"\n"
        text+="selection_basis = \"structural characterization only; before pilot\"\nstructural_sha256 = "+json.dumps(digest)+"\n"
        text+="missing_regimes = "+json.dumps([r for r in ("R1","R2","R3") if r not in represented])+"\n"
        for regime,s in represented.items():text+=f"representative_{regime} = {json.dumps(s)}\n"
        path=out/"configs/selected_scenarios.toml"
        if path.exists() and path.read_text()!=text:raise RuntimeError("Refusing to modify frozen structural selection")
        path.write_text(text)
    (out/"configs/characterization_status.json").write_text(json.dumps(dict(complete=bool(complete),
        quick=cfg["quick"],scenario_count=len(regimes),points=len(structural),selected=sorted(selected) if complete else [],
        fingerprint=fingerprint),indent=2))
    print("STRUCTURAL CLASSIFICATION",regimes[["scenario","regime","R1_points","R2_points","R3_points"]].to_string(index=False),flush=True)
    if complete:print("FROZEN CONFIRMATION SCENARIOS",sorted(selected),flush=True)
    return structural,regimes

