"""Simulation-free comparison of external DE with untouched historical LR runs."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import toml

from run_022 import BASE, ROOT, INSTANCES, METHOD, GVSMETHOD, OLD021, OLD017, historical, output, validate_rep, verify_history, expected_seeds
from i0252_statistical_storage_019 import METHODS as LR4, config, load_complete, validate_rep as validate_historical, atomic_json, atomic_bytes, sha
from analyze_i0252_statistical_019 import moments, save_csv, save_figure

REGISTRY = {
    "fresh": ("Fresh", "#0072B2"), "green": ("Green", "#009E73"),
    "green_softmax": ("Green Softmax", "#D55E00"), "green_variance": ("Green Variance", "#CC79A7"),
    GVSMETHOD: ("Green Variance + Softmax", "#6A3D9A"), METHOD: ("Differential Evolution", "#E69F00"),
}
METHODS = tuple(REGISTRY)
LABELS = {key: pair[0] for key, pair in REGISTRY.items()}
COLORS = {key: pair[1] for key, pair in REGISTRY.items()}
METRICS = {"objective_crn": "Objective J(z), independent direct CRN", "z3": "$z_3$", "z4": "$z_4$"}
plt.rcParams.update({"font.size":10,"axes.titlesize":12,"axes.grid":True,"grid.alpha":.18,
                     "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})


def numeric(frame):
    for col in frame:
        if col not in ("method", "instance", "cohort", "stop_reason", "accounting_source"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def load_instance(instance):
    old = load_complete(historical(instance))
    assert {m:sum(meta["method"]==m for _,meta,_,_ in old) for m in LR4} == dict.fromkeys(LR4,50)
    cfg = config(OLD021 / instance)
    gvs = []
    for rep in range(1,51):
        path = OLD021 / instance / "raw" / GVSMETHOD / f"rep_{rep:04d}"
        gvs.append((path,*validate_historical(path,cfg,GVSMETHOD,rep)))
    de = [validate_rep("statistical",instance,rep) for rep in range(1,51)]
    results = old+gvs+de
    traces=[];finals=[];costs=[]
    for path,meta,rows,final in results:
        method=meta["method"];rep=meta["replication"]
        tr=numeric(pd.DataFrame(rows));tr["instance"]=instance
        d=numeric(pd.read_csv(path/"iteration_diagnostics.csv"));d=d[d.iteration>0]
        if method==METHOD:
            consumed=int(final["path_evaluations"]);unique=int(final["new_search_paths"])
            routing=int(final["policy_path_evaluations"]);updates=int(final["objective_evaluations"])
            score_paths=0;score_source="DE uses no LR or score evaluations"
            tr["budget"]=tr.cumulative_path_evaluations
        else:
            consumed=int(final["new_search_paths"]);unique=consumed;updates=int(final["iterations"])
            routing=consumed*144;tr["budget"]=tr.cumulative_new_search_paths
            if "lr_path_evaluations" in d:
                score_paths=int(d.lr_path_evaluations.sum());score_source="Recorded original 012/019/020 counter"
            elif method in ("green_variance",GVSMETHOD):
                # Frozen V.search: local_data(N) + full Green(kN) + bank_kernel(kN)
                # + accepted Green(retained*N). Optional diagnostics use cached kernels.
                score_paths=int((500*(1+2*d.iteration+d.retained_banks)).sum())
                score_source="Exact static reconstruction from original call graph and recorded retained-bank counts"
            else:
                score_paths=np.nan;score_source="Unavailable"
        assert consumed<=50000 and routing==144*consumed
        trace_unique=len(tr[["z3","z4"]].drop_duplicates())
        row=dict(final);row.update(instance=instance,method=method,replication=rep,
                                  unique_training_paths=unique,path_evaluations=consumed,policy_path_evaluations=routing,
                                  objective_or_gradient_updates=updates,score_path_evaluations=score_paths,
                                  end_to_end_seconds=meta.get("last_attempt_seconds",np.nan),
                                  trace_unique_decisions=trace_unique,trace_path_evaluations=1000*trace_unique,
                                  trace_policy_path_evaluations=144000*trace_unique,
                                  final_policy_path_evaluations=144*10000+50000)
        costs.append(dict(instance=instance,method=method,replication=rep,unique_training_paths=unique,
                          path_evaluations=consumed,policy_path_evaluations=routing,
                          score_path_evaluations=score_paths,accounting_source=score_source))
        traces.append(tr);finals.append(row)
    tr=pd.concat(traces,ignore_index=True).sort_values(["method","replication","budget"])
    fin=numeric(pd.DataFrame(finals)).sort_values(["method","replication"])
    assert fin.groupby("method").size().to_dict()==dict.fromkeys(METHODS,50)
    assert all(sorted(g.replication)==list(range(1,51)) for _,g in fin.groupby("method"))
    for _,g in tr.groupby(["method","replication"]):
        assert np.all(np.diff(g.budget)>=0) and np.all(np.diff(g.cumulative_optimization_seconds)>=0)
    return tr,fin,pd.DataFrame(costs),results


def sampled_statistics(tr,axis,grid):
    rows=[];sampled=[]
    for method in METHODS:
        group=tr[tr.method==method]
        arrays=[];finishtimes=[]
        for rep,g in group.groupby("replication"):
            g=g.sort_values(axis)
            indices=np.searchsorted(g[axis].to_numpy(float),grid,side="right")-1
            indices=np.maximum(indices,0)
            values=g.iloc[indices][list(METRICS)].to_numpy(float)
            arrays.append(values);finishtimes.append(float(g[axis].iloc[-1]))
            for k,x in enumerate(grid):
                sampled.append(dict(method=method,replication=int(rep),axis=axis,coordinate=x,
                                    **dict(zip(METRICS,values[k]))))
        values=np.stack(arrays);assert values.shape[0]==50
        for k,x in enumerate(grid):
            for j,metric in enumerate(METRICS):
                rows.append(dict(method=method,metric=metric,axis=axis,coordinate=x,
                                 completed_fraction=float(np.mean(np.asarray(finishtimes)<=x)),**moments(values[:,k,j])))
    return pd.DataFrame(rows),pd.DataFrame(sampled)


def final_statistics(instance,fin):
    tables=[];details=[]
    for method in METHODS:
        g=fin[fin.method==method];s=moments(g.objective)
        tables.append(dict(instance=instance,method=method,replications=50,
                           objective_final_mean=s["mean"],objective_final_sd=s["sd"],objective_final_se=s["se"],
                           ci95_low=s["ci95_low"],ci95_high=s["ci95_high"],objective_min=g.objective.min(),
                           objective_median=g.objective.median(),objective_max=g.objective.max(),
                           z3_mean=g.z3.mean(),z4_mean=g.z4.mean(),optimization_seconds_mean=g.optimization_seconds.mean(),
                           end_to_end_seconds_mean=g.end_to_end_seconds.mean(),
                           independent_validation_seconds_mean=(g.convergence_evaluation_seconds+g.final_evaluation_seconds).mean(),
                           unique_training_paths_mean=g.unique_training_paths.mean(),path_evaluations_mean=g.path_evaluations.mean(),
                           policy_path_evaluations_mean=g.policy_path_evaluations.mean(),
                           score_path_evaluations_mean=g.score_path_evaluations.mean(),
                           objective_or_gradient_updates_mean=g.objective_or_gradient_updates.mean(),
                           trace_policy_path_evaluations_mean=g.trace_policy_path_evaluations.mean(),
                           final_policy_path_evaluations_mean=g.final_policy_path_evaluations.mean()))
        for metric in ("objective","z3","z4","optimization_seconds","end_to_end_seconds","path_evaluations","unique_training_paths"):
            details.append(dict(method=method,metric=metric,**moments(g[metric])))
    paired=[];new=fin[fin.method==METHOD].set_index("replication").sort_index()
    for reference in METHODS:
        if reference==METHOD:
            continue
        old=fin[fin.method==reference].set_index("replication").sort_index()
        delta=new.objective-old.objective
        p=1. if np.all(delta.to_numpy()==0) else float(ttest_1samp(delta,0).pvalue)
        paired.append(dict(instance=instance,method=METHOD,reference=reference,**moments(delta),p_raw=p,
                           optimization_time_ratio=new.optimization_seconds.mean()/old.optimization_seconds.mean(),
                           path_evaluation_ratio=new.path_evaluations.mean()/old.path_evaluations.mean(),
                           sd_ratio=new.objective.std(ddof=1)/old.objective.std(ddof=1),
                           wins=int(np.sum(delta>0)),losses=int(np.sum(delta<0)),ties=int(np.sum(delta==0))))
    return pd.DataFrame(tables),pd.DataFrame(details),pd.DataFrame(paired)


def quality_target(instance,tr):
    surface=pd.read_csv(output("statistical",instance)/f"config/{instance}_reference_surface.csv")
    best=surface.objective.max();target=best-.01*max(1.,abs(best))
    hits=[]
    for (method,rep),g in tr.groupby(["method","replication"]):
        reached=g[g.objective_crn>=target]
        first=reached.iloc[0] if len(reached) else None
        hits.append(dict(instance=instance,method=method,replication=rep,reference_grid_objective=best,
                          target=target,reached=first is not None,
                          first_path_evaluations=np.nan if first is None else first.budget,
                          first_optimization_seconds=np.nan if first is None else first.cumulative_optimization_seconds))
    hits=pd.DataFrame(hits)
    summary=pd.DataFrame([dict(instance=instance,method=m,target=target,reached_count=int(g.reached.sum()),replications=50,
                                reached_fraction=g.reached.mean(),conditional_mean_paths=g.first_path_evaluations.mean(),
                                conditional_median_paths=g.first_path_evaluations.median(),
                                conditional_mean_seconds=g.first_optimization_seconds.mean()) for m,g in hits.groupby("method")])
    return hits,summary


def lines(ax,stats,metric,methods=METHODS,xlabel="Materialized paths evaluated (all 144 policies per path)"):
    for method in methods:
        g=stats[(stats.method==method)&(stats.metric==metric)].sort_values("coordinate")
        x,mean,low,high=[g[key].to_numpy(float) for key in ("coordinate","mean","ci95_low","ci95_high")]
        ax.plot(x,mean,c=COLORS[method],lw=1.8,label=LABELS[method])
        ax.fill_between(x,low,high,color=COLORS[method],alpha=.15,lw=0)
    ax.set(xlabel=xlabel,ylabel=METRICS[metric])
    if metric in ("z3","z4"):
        ax.set_ylim(0,1)
    ax.legend(fontsize=7.5)


def trajectories(ax,instance,tr,fin,budgetstats):
    surface=pd.read_csv(output("statistical",instance)/f"config/{instance}_reference_surface.csv")
    grid=surface.pivot(index="z4",columns="z3",values="objective")
    ax.contourf(grid.columns,grid.index,grid.to_numpy(),levels=18,cmap="Greys",alpha=.55)
    for method in METHODS:
        saved=budgetstats[(budgetstats.method==method)&budgetstats.metric.isin(("z3","z4"))]
        coords=saved.pivot(index="coordinate",columns="metric",values="mean").sort_index()[["z3","z4"]].to_numpy(float)
        ends=fin[fin.method==method]
        ax.plot(coords[:,0],coords[:,1],color=COLORS[method],label=LABELS[method],lw=1.9)
        ax.scatter(ends.z3,ends.z4,s=12,c=COLORS[method],alpha=.35,lw=0)
        ax.scatter(*coords[-1],marker="X",s=85,c=COLORS[method],edgecolor="white",zorder=6)
    ax.scatter(.12,.10,marker="*",s=120,c="gold",edgecolor="black",zorder=7)
    ax.set(xlabel="$z_3$",ylabel="$z_4$",xlim=(0,1),ylim=(0,1),aspect="equal",title=instance)
    ax.legend(fontsize=7)


def attainment(ax,hits):
    grid=np.arange(0,50001,500)
    for method in METHODS:
        g=hits[hits.method==method]
        fractions=[np.mean(g.first_path_evaluations.to_numpy()<=x) for x in grid]
        ax.plot(grid,fractions,color=COLORS[method],label=LABELS[method],lw=1.8)
    ax.set(xlabel="Materialized paths evaluated",ylabel="Fraction reaching reference-grid target",ylim=(0,1.02))
    ax.legend(fontsize=7)


def individual_figures(instance,tr,fin,budgetstats,timestats,hits):
    out=output("statistical",instance)
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(10,6));lines(ax,budgetstats,metric)
        ax.set_title(f"{instance} · independent CRN · mean and pointwise 95% t CI, R=50")
        fig.tight_layout();save_figure(out,f"{metric}_vs_simulation_effort",fig)
        fig,ax=plt.subplots(figsize=(10,6));lines(ax,budgetstats,metric,(METHOD,))
        ax.set_title(f"{instance} · Differential Evolution · mean and pointwise 95% t CI, R=50")
        fig.tight_layout();save_figure(out,f"{METHOD}/{metric}_vs_simulation_effort",fig)
    fig,ax=plt.subplots(figsize=(10,6));lines(ax,timestats,"objective_crn",xlabel="Optimization-only wall-time budget (s)")
    ax.set_title(f"{instance} · offline independent CRN; last decision retained after termination")
    fig.tight_layout();save_figure(out,"objective_vs_computation_time",fig)
    fig,ax=plt.subplots(figsize=(9,8));trajectories(ax,instance,tr,fin,budgetstats)
    fig.tight_layout();save_figure(out,"z3_z4_trajectories",fig)
    # DE internal SAA is labeled separately from offline independent validation.
    new=tr[tr.method==METHOD];grid=np.arange(500,50001,500)
    internal=[];independent=[]
    for _,g in new.groupby("replication"):
        ix=np.searchsorted(g.budget.to_numpy(),grid,side="right")-1
        internal.append(g.iloc[ix].objective_internal.to_numpy(float));independent.append(g.iloc[ix].objective_crn.to_numpy(float))
    fig,ax=plt.subplots(figsize=(10,6))
    for values,label,color in ((internal,"Internal fixed training SAA (N=500)","#E69F00"),(independent,"Independent direct CRN (N=1000)","#0072B2")):
        stats=[moments(np.asarray(values)[:,k]) for k in range(len(grid))]
        ax.plot(grid,[r["mean"] for r in stats],label=label,color=color)
        ax.fill_between(grid,[r["ci95_low"] for r in stats],[r["ci95_high"] for r in stats],color=color,alpha=.15)
    ax.set(xlabel="Materialized paths evaluated",ylabel="Objective estimate",title=f"{instance} · DE training and independent validation")
    ax.legend();fig.tight_layout();save_figure(out,"de_training_vs_independent_objective",fig)
    fig,ax=plt.subplots(figsize=(10,6));attainment(ax,hits);ax.set_title(f"{instance} · exploratory target attainment")
    fig.tight_layout();save_figure(out,"reference_quality_attainment",fig)


def compare_017():
    tr=pd.read_csv(OLD017/"tables/objective_traces.csv")
    coords=pd.read_csv(OLD017/"tables/coordinates.csv")
    fin=pd.read_csv(OLD017/"tables/final_results.csv").rename(columns={"rep":"replication"})
    diag=pd.read_csv(OLD017/"tables/iteration_diagnostics.csv")
    original_methods=toml.load(OLD017/"configs/experiment.toml")["science"]["methods"]
    mapping={"Green":"green","GreenSoftmax050":"green_softmax","Filter2":"green_variance",
             "Exp50":"green_exponential","Rolling5":"green_rolling"}
    labels={**LABELS,"green_exponential":"Green Exponential (ρ=0.5)","green_rolling":"Green Rolling (W=5)"}
    colors={"green":"#222222","green_softmax":"#7B3294","green_variance":"#009E73",
            "green_exponential":"#D55E00","green_rolling":"#0072B2",METHOD:COLORS[METHOD]}
    methods=[mapping[m] for m in original_methods]+[METHOD]
    tr=tr.rename(columns={"rep":"replication","objective":"objective_crn"})
    tr["method"]=tr.method.map(mapping);tr["budget"]=500*tr.iteration
    diag["method"]=diag.method.map(mapping)
    timing=[]
    for (m,r),g in tr.groupby(["method","replication"]):
        d=diag[(diag.method==m)&(diag.rep==r)].sort_values("iteration")
        timing.extend([dict(method=m,replication=r,iteration=0,cumulative_optimization_seconds=0.)]+
                      [dict(method=m,replication=r,iteration=int(row.iteration),cumulative_optimization_seconds=row.cumulative_search_seconds) for row in d.itertuples()])
    tr=tr.merge(pd.DataFrame(timing),on=["method","replication","iteration"],validate="one_to_one")
    fin["method"]=fin.method.map(mapping)
    for rep in (1,2,3):
        path,meta,rows,f=validate_rep("017","I0252",rep)
        new=numeric(pd.DataFrame(rows));new["budget"]=new.cumulative_path_evaluations
        trace_seeds=tr[tr.replication==rep].evaluation_seed.unique()
        assert len(trace_seeds)==1 and np.all(new.evaluation_seed==trace_seeds[0])
        tr=pd.concat([tr,new],ignore_index=True)
        f=numeric(pd.DataFrame([f]));fin=pd.concat([fin,f],ignore_index=True)
        old=fin[(fin.replication==rep)&(fin.method=="green")].iloc[0]
        assert int(old.selection_seed)==meta["seeds"]["selection"] and int(old.holdout_seed)==meta["seeds"]["holdout"]
    out=BASE/"comparisons/017"
    save_csv(out/"summaries/trajectories_all.csv",tr)
    save_csv(out/"summaries/final_comparison.csv",fin)
    with plt.rc_context():
        from analyze_landscape_010 import background
        surface=pd.read_csv(OLD017/"configs/I0252_surface.csv")
        for rep in (1,2,3):
            for metric in METRICS:
                fig,ax=plt.subplots(figsize=(10,6))
                for method in methods:
                    g=tr[(tr.replication==rep)&(tr.method==method)].sort_values("budget")
                    ax.plot(g.budget,g[metric],c=colors[method],label=labels[method],lw=1.8)
                ax.set(xlabel="Materialized paths evaluated (144 policies per path)",ylabel=METRICS[metric],
                       xlim=(0,50000),title=f"I0252 · original 017 rep{rep:02d} + DE · independent shared CRN")
                if metric in ("z3","z4"):
                    ax.set_ylim(0,1)
                ax.legend(fontsize=8);fig.tight_layout();save_figure(out,f"rep{rep:02d}/{metric}_vs_simulation_effort",fig)
            fig,ax=plt.subplots(figsize=(9,7));image=background(ax,surface,f"I0252 · 017 rep{rep:02d} + Differential Evolution")
            for method in methods:
                g=tr[(tr.replication==rep)&(tr.method==method)].sort_values("budget")
                ax.plot(g.z3,g.z4,c=colors[method],lw=1.7,label=labels[method])
                ax.scatter(g.z3.iloc[-1],g.z4.iloc[-1],s=55,c=colors[method],edgecolor="white",zorder=6)
            fig.colorbar(image,ax=ax,label="Archived J(z)");ax.legend(loc="upper left",bbox_to_anchor=(1.28,1),fontsize=8)
            fig.tight_layout();save_figure(out,f"rep{rep:02d}/z3_z4_trajectories",fig)
            fig,ax=plt.subplots(figsize=(10,6))
            for method in methods:
                g=tr[(tr.replication==rep)&(tr.method==method)].sort_values("cumulative_optimization_seconds")
                ax.plot(g.cumulative_optimization_seconds,g.objective_crn,c=colors[method],label=labels[method],lw=1.8)
            ax.set(xlabel="Optimization-only wall time (s)",ylabel=METRICS["objective_crn"],title=f"I0252 · 017 rep{rep:02d} + DE")
            ax.legend(fontsize=8);fig.tight_layout();save_figure(out,f"rep{rep:02d}/objective_vs_computation_time",fig)
    atomic_json(out/"summaries/provenance.json",dict(original_source=str(OLD017),paired_validation_seeds=True,
        original_search_seed_replications=[1,2,3],statistical_search_namespace_included=False,
        historical_optimizers_executed=0,historical_files={str(p):sha(p) for p in (OLD017/"tables").glob("*.csv")}))
    return fin


def grouped_figures(data,paired,tables):
    out=BASE/"comparisons/statistical"
    for metric in METRICS:
        for window in ((0,50000),(1000,12500)):
            fig,axes=plt.subplots(2,3,figsize=(18,10),sharex=True,sharey=metric in ("z3","z4"))
            for instance,ax in zip(INSTANCES,axes.flat):
                stats=data[instance][2];stats=stats[stats.coordinate.between(*window)]
                lines(ax,stats,metric);ax.set_title(instance);ax.set_xlim(*window)
            axes.flat[-1].axis("off")
            fig.suptitle(f"Independent CRN means and pointwise 95% t CI · R=50 · {len(METHODS)} algorithms")
            fig.tight_layout(rect=(0,0,1,.965))
            suffix="" if window[0]==0 else "_budget1000_12500"
            save_figure(out,f"{metric}_vs_simulation_effort_all_instances{suffix}",fig)
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for instance,ax in zip(INSTANCES,axes.flat):
        lines(ax,data[instance][3],"objective_crn",xlabel="Optimization-only wall-time budget (s)");ax.set_title(instance)
    axes.flat[-1].axis("off");fig.tight_layout();save_figure(out,"objective_vs_time_all_instances",fig)
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for instance,ax in zip(INSTANCES,axes.flat):
        tr,fin,budgetstats,_,_=data[instance];trajectories(ax,instance,tr,fin,budgetstats)
    axes.flat[-1].axis("off");fig.tight_layout();save_figure(out,"z3_z4_trajectories_all_instances",fig)
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for instance,ax in zip(INSTANCES,axes.flat):
        attainment(ax,data[instance][4]);ax.set_title(instance)
    axes.flat[-1].axis("off");fig.tight_layout();save_figure(out,"target_attainment_all_instances",fig)
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for reference,ax in zip([m for m in METHODS if m!=METHOD],axes.flat):
        g=paired[paired.reference==reference].set_index("instance").loc[list(INSTANCES)]
        mean=g["mean"].to_numpy();ax.errorbar(mean,np.arange(len(INSTANCES)),
            xerr=np.array([mean-g.ci95_low,g.ci95_high-mean]),fmt="o",capsize=4,color=COLORS[METHOD])
        ax.axvline(0,color="black",lw=1);ax.set_yticks(np.arange(len(INSTANCES)),INSTANCES)
        ax.set(xlabel="DE − reference: final holdout objective",title=LABELS[reference])
    axes.flat[-1].axis("off");fig.tight_layout();save_figure(out,"paired_final_objective_differences",fig)
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for instance,ax in zip(INSTANCES,axes.flat):
        for method in METHODS:
            row=tables[(tables.instance==instance)&(tables.method==method)].iloc[0]
            ax.errorbar(row.optimization_seconds_mean,row.objective_final_mean,
                yerr=np.array([[row.objective_final_mean-row.ci95_low],[row.ci95_high-row.objective_final_mean]]),
                fmt="o",capsize=3,color=COLORS[method],label=LABELS[method])
        ax.set(xlabel="Mean optimization-only wall time (s)",ylabel="Final independent objective, mean/95% t CI",title=instance)
        ax.legend(fontsize=7)
    axes.flat[-1].axis("off");fig.tight_layout();save_figure(out,"final_quality_vs_mean_computation_time",fig)


def holm(frame):
    result=frame.copy();assert np.isfinite(result.p_raw).all()
    order=np.argsort(result.p_raw.to_numpy());adjusted=np.empty(len(order));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(order)-rank)*result.iloc[index].p_raw));adjusted[index]=previous
    result["p_holm_25"]=adjusted
    return result


def historical_regression(instance,stats):
    sources=[historical(instance)/"summaries/iteration_statistics.csv",OLD021/instance/"summaries/iteration_statistics.csv"]
    checked=0
    for source in sources:
        old=pd.read_csv(source);old=old[old.metric.isin(METRICS)]
        if source.parts[-4]=="021_green_variance_softmax":
            old=old[old.method==GVSMETHOD]
        old=old.copy();old["coordinate"]=old.iteration*500
        merged=old.merge(stats,on=["method","metric","coordinate"],suffixes=("_old","_022"),validate="one_to_one")
        assert len(merged)==len(old)
        for key in ("n","mean","sd","se","ci95_low","ci95_high"):
            np.testing.assert_allclose(merged[key+"_old"],merged[key+"_022"],rtol=1e-11,atol=1e-11)
        checked+=len(merged)
    result=dict(passed=True,statistical_rows_checked=checked,historical_optimizers_executed=0)
    atomic_json(output("statistical",instance)/"validation/historical_statistics_regression.json",result)
    return result


def verify_numeric(cohort,instance,rep):
    path,meta,rows,final=validate_rep(cohort,instance,rep)
    h=pd.read_csv(path/"objective_history.csv");tr=numeric(pd.DataFrame(rows))
    d=config(output(cohort,instance))["science"];n=d["de"]["paths_per_evaluation"]
    assert h.objective_evaluation.tolist()==list(range(1,len(h)+1))
    assert h.new_primitive_paths.sum()==n and h.paths_evaluated.sum()==n*len(h)
    assert h.policy_path_evaluations.sum()==144*n*len(h)
    assert np.array_equal(h.cumulative_path_evaluations,n*h.objective_evaluation)
    assert np.array_equal(h.cumulative_policy_path_evaluations,144*n*h.objective_evaluation)
    assert np.isfinite(h[["candidate_objective","candidate_z3","candidate_z4","objective_internal","z3","z4"]]).all().all()
    assert h[["candidate_z3","candidate_z4","z3","z4"]].ge(0).all().all() and h[["candidate_z3","candidate_z4","z3","z4"]].le(1).all().all()
    best=h.candidate_objective.cummax();np.testing.assert_allclose(best,h.objective_internal,rtol=1e-13,atol=1e-12)
    np.testing.assert_allclose(h[["z3","z4"]],tr.iloc[1:][["z3","z4"]].to_numpy(float),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(h.objective_internal,tr.iloc[1:].objective_internal.to_numpy(float),rtol=1e-13,atol=1e-12)
    return meta


def markdown(frame,keys,names=None):
    names=names or keys
    lines=["| "+" | ".join(names)+" |","| "+" | ".join("---" for _ in names)+" |"]
    for _,row in frame.iterrows():
        values=[]
        for key in keys:
            x=row[key]
            values.append(("No disponible" if not np.isfinite(x) else f"{x:.5f}") if isinstance(x,(float,np.floating)) else str(x))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def report(tables,paired,targets,fin017,preservation,figures):
    positive=paired[(paired["mean"]>0)&(paired.p_holm_25<.05)]
    negative=paired[(paired["mean"]<0)&(paired.p_holm_25<.05)]
    lines=["# Experimento 022 — Differential Evolution como benchmark externo", "",
        "**Ejecución completa:** 250 búsquedas DE estadísticas (50 por instancia) y tres búsquedas DE con las semillas originales del 017. "
        f"Se generaron {figures} figuras PNG/PDF, tablas comparativas y resultados individuales. Ningún optimizador LR se volvió a ejecutar.","",
        f"En el objetivo final independiente, DE presenta {len(positive)} contrastes favorables y {len(negative)} desfavorables con p Holm<0.05 entre los 25 contrastes frente a cinco referencias. "
        "La conclusión depende de la instancia y del presupuesto concreto. Las tablas e intervalos siguientes incluyen todos los resultados.","",
        "## 1. Implementación y archivos", "",
        "El benchmark utiliza **BlackBoxOptim.jl 0.6.12**, variante `adaptive_de_rand_1_bin_radiuslimited`, en Julia 1.11.7. "
        "Se reutilizan sin modificar `primitive_bank`, `materialize_bank`, `rewards`, `cost`, las 144 políticas, `direct_trace` y `endpoint` de las fuentes congeladas del estudio. "
        "La función de DE simula directamente y devuelve el negativo de `max_i mean(Y_i) − alpha'z`. No llama a pesos LR, scores, gradientes ni estimadores Green. "
        "El problema es el routing entre los servidores fast/standard con señales de probing, con z3,z4 en [0,1]². No se usa el modelo de producción/retrabajo ni el objetivo de máximo realizado por trayectoria.","",
        "Archivos nuevos: [módulo Julia](code/022_differential_evolution.jl), [ejecutor](code/run_022.py), "
        "[análisis](code/analyze_022.py), [auditoría de resultados](code/audit_saved_de_022.jl), [cierre reproducible](code/finish_022.py), "
        "[protocolo previo](AUDITORIA_Y_PROTOCOLO_022.md), `environment/Project.toml` y `Manifest.toml`, configuraciones, logs, CSV, checkpoints y figuras. "
        "**Archivos históricos modificados: ninguno.** Las versiones de las dependencias históricas se conservaron en el entorno aislado; la representación de weakdeps de ColorVectorSpace cambió en su copia del Manifest, sin cambiar su versión ni código.","",
        "## 2. Parámetros y parada", "",
        "Población 20 (10 por dimensión; valor no predeterminado fijado antes de los resultados), radio de muestreo 8, dominio [0,1]². "
        "Los parámetros adaptativos F y CR son los de la biblioteca: mezcla equiprobable de Cauchy(0.65,0.1)/Cauchy(1.0,0.1) para F y "
        "Cauchy(0.1,0.1)/Cauchy(0.95,0.1) para CR; se rechazan valores ≤0 y se truncan valores ≥1. Se guardan sus valores finales por individuo. "
        "No se hizo ajuste por instancia ni selección de parámetros basada en los resultados.","",
        "Máximo nominal 100 consultas de 500 trayectorias, incluyendo una consulta inicial en z0=(0.12,0.10). La parada usa el callback público de la biblioteca, con intervalo 0 y margen de una consulta, "
        "porque un paso puede evaluar trial y target antes de comprobar el límite. El consumo efectivo es de 99–100 consultas; siempre ≤50.000 evaluaciones de trayectorias. "
        "`MaxFuncEvals=100`, `MaxTime=0`, sin `TargetFitness`; se mantienen tolerancias predeterminadas 1e−50 y 1e−8. "
        "El valor nominal predeterminado `MaxSteps=10000` queda **desactivado a 0** por la biblioteca al fijar MaxFuncEvals; el archivo `validation/effective_library_settings.toml` registra los valores efectivos. "
        "El guard del wrapper impide una llamada adicional al simulador si hubiera una sobrepasada inesperada. No se proporcionan valores ficticios al optimizador.","",
        "Se utiliza `FunctionBasedProblem` de BlackBoxOptim para evitar la evaluación aleatoria de comprobación que `bbsetup(function)` realiza fuera de `f_calls`. "
        "La consulta inicial se suma al contador de biblioteca; los contadores efectivos se verifican en cada réplica.","",
        "## 3. CRN, semillas y evaluación", "",
        "DE optimiza una SAA determinista con un banco fijo de 500 primitivas por réplica. Cada candidato conserva llegadas, clases, complejidades, valores y servicios, "
        "y rematerializa señales mediante `u_signal < p(signal | clase, complejidad, z)` con la función original. El banco usa el primer stream de búsqueda histórico. "
        "Los otros streams son distintos: RNG interno de DE, traza CRN, selección de política y holdout. DE se siembra en una tarea Julia nueva; el simulador conserva sus MersenneTwister explícitos. "
        "No se usan los parámetros obsoletos `RngSeed`/`RandomizeRngSeed` de esta versión de BlackBoxOptim.","",
        "La cohorte estadística usa los índices originales 19000000+r; la cohorte 017 usa r=1,2,3. Se ejecutaron por separado para preservar el pareamiento. "
        "I0252 del 020 remite a la única muestra de 50 réplicas del 019; no se añade como una segunda muestra. Se leen las réplicas GV+S del 021 sin repetirlas.","",
        "Cada incumbente se evalúa **después de la búsqueda**, con 1.000 primitivas CRN independientes y compartidas con la referencia correspondiente. "
        "La curva representa el máximo de las 144 medias muestrales y admite optimismo por selección. La evaluación final principal selecciona una política con 10.000 trayectorias "
        "y mide esa política con otras 50.000 independientes. La validación nunca realimenta DE. Sus tiempos y conteos se guardan separados.","",
        "## 4. Presupuesto y comparabilidad", "",
        "El presupuesto se derivó de los archivos originales: 100 actualizaciones×500 trayectorias=50.000 trayectorias y 7.200.000 evaluaciones política-trayectoria nuevas. "
        "DE consume como máximo el mismo trabajo de routing, con una posible holgura de hasta 1% por la parada conservadora. Los archivos registran el consumo real y no descuentan candidatos repetidos.","",
        "**Esta igualdad no equivale a igual información estadística:** DE genera 500 primitivas únicas y las rematerializa repetidamente; LR genera 50.000 primitivas nuevas. "
        "Los métodos Green reutilizan además bancos con trabajo de scores y productos matriciales. Los contadores de scores se leen del histórico cuando existen; "
        "para GV/GV+S se reconstruyen exactamente de las llamadas originales `local_data + full Green + bank_kernel + accepted Green` y los conteos de bancos aceptados guardados. "
        "Se etiqueta esa reconstrucción en `computational_accounting.csv`. No es una estimación de tiempo CPU.","",
        "Las curvas por esfuerzo se agregan en presupuestos comunes mediante el último incumbente disponible, conservando la solución final tras la parada sin atribuirle trabajo adicional. "
        "Las curvas temporales usan el mismo criterio y el reloj de **solo optimización**. Todas las réplicas contribuyen; `completed_fraction` informa la proporción ya detenida. "
        "Los tiempos son descriptivos: los históricos proceden de otras sesiones, con contención e instrumentación distintas. La compilación, validación y figuras quedan fuera del tiempo de búsqueda.","",
        "La prueba de calidad usa un umbral exploratorio común: máximo de la superficie archivada menos 1% de su magnitud absoluta (mínimo denominador 1). "
        "Esa superficie no es un óptimo certificado; los cruces se miden en curvas CRN ruidosas. Se informan proporción de éxitos y esfuerzo condicionado al éxito, con los fracasos explícitos. "
        "No se presenta este umbral como prueba confirmatoria ni se usa para ajustar DE.","",
        "## 5. Resultados finales — 50 réplicas por método e instancia", "",
        "Media, SD, error estándar e IC95% Student-t con 49 grados de libertad; la unidad independiente es la réplica. "
        "La evaluación final es el holdout de la política seleccionada. La dispersión incluye optimización, selección y evaluación. "
        "Los intervalos son puntuales. Las diferencias DE−referencia se parean por semilla de réplica y se ajustan con Holm en una familia de 25 contrastes.","" ]
    for instance in INSTANCES:
        t=tables[tables.instance==instance].copy();t["method"]=t.method.map(LABELS)
        t["IC95%"]=t.apply(lambda r:f"[{r.ci95_low:.5f}, {r.ci95_high:.5f}]",axis=1)
        lines += [f"### {instance}","",markdown(t,["method","objective_final_mean","IC95%","objective_final_sd","objective_min","objective_max","z3_mean","z4_mean","optimization_seconds_mean"],
                  ["Método","Media final","IC95%","SD","Peor","Mejor","z3","z4","Tiempo búsqueda (s)"]),"",
                  f"[Resultados y costos completos]({instance}/summaries/final_comparison.csv) · "
                  f"[objetivo por esfuerzo]({instance}/plots/objective_crn_vs_simulation_effort.png) · "
                  f"[objetivo por tiempo]({instance}/plots/objective_vs_computation_time.png).",""]
    lines += ["### Diferencias pareadas del objetivo final", "", "Los valores positivos favorecen DE. IC95% puntual; significación confirmatoria según p Holm<0.05.",""]
    view=paired.copy();view["reference"]=view.reference.map(LABELS)
    lines += [markdown(view,["instance","reference","mean","ci95_low","ci95_high","p_holm_25","sd_ratio","optimization_time_ratio"],
              ["Instancia","Referencia","DE−referencia","IC inferior","IC superior","p Holm","Razón SD","Razón tiempo"]),"",
              "### Contabilidad de simulación",""]
    view=tables.copy();view["method"]=view.method.map(LABELS)
    lines += [markdown(view,["instance","method","unique_training_paths_mean","path_evaluations_mean","policy_path_evaluations_mean","score_path_evaluations_mean"],
              ["Instancia","Método","Primitivas únicas","Trayectorias evaluadas","Política×trayectoria","Pases de score"]),"",
              "Los conteos anteriores excluyen validación. Selección y holdout finales consumen por réplica 10.000×144+50.000=1.490.000 evaluaciones política-trayectoria. "
              "La validación de curvas evalúa solo decisiones distintas, debido a la caché original de `direct_trace`; sus conteos se guardan por réplica.","",
              "### Calidad alcanzada con el presupuesto", "",markdown(targets,["instance","method","target","reached_count","reached_fraction","conditional_mean_paths","conditional_mean_seconds"],
              ["Instancia","Método","Umbral exploratorio","Éxitos/50","Proporción","Trayectorias (éxitos)","Segundos (éxitos)"]),"",
              "## 6. rep01, rep02 y rep03 — comparación con 017", "",
              "Se incorporan Green, Green Softmax, Green Variance, Green Exponential y Green Rolling originales, más DE, sin mezclar las réplicas estadísticas del 019/021. "
              "Cada comparación conserva sus semillas originales de entrenamiento y validación. Las curvas se presentan por esfuerzo y tiempo; las trayectorias reutilizan la superficie original.",""]
    for rep in (1,2,3):
        g=fin017[fin017.replication==rep].copy();g["method"]=g.method.map(lambda m:LABELS.get(m,m))
        lines += [f"### rep{rep:02d}","",markdown(g,["method","objective","evaluation_se","z3","z4"],
                  ["Método","Objetivo final independiente","SE condicional","z3","z4"]),"",
                  f"[Curvas comparativas](comparisons/017/plots/rep{rep:02d}/objective_crn_vs_simulation_effort.png). "
                  "Con tres réplicas no se sostiene una conclusión estadística general; las filas muestran resultados individuales y errores MC condicionales.",""]
    lines += ["## 7. Interpretación científica", ""]
    for instance in INSTANCES:
        g=tables[tables.instance==instance];de=g[g.method==METHOD].iloc[0]
        best=g.loc[g.objective_final_mean.idxmax()]
        lines += [f"### {instance}","",f"DE: media {de.objective_final_mean:.5f}, SD {de.objective_final_sd:.5f}, "
                  f"rango [{de.objective_min:.5f}, {de.objective_max:.5f}], z medio ({de.z3_mean:.4f},{de.z4_mean:.4f}). "
                  f"El mayor promedio descriptivo entre los métodos es {LABELS[best['method']]} ({best.objective_final_mean:.5f})."]
        for reference in ("fresh","green_variance",GVSMETHOD):
            p=paired[(paired.instance==instance)&(paired.reference==reference)].iloc[0]
            statement=("ventaja de DE respaldada tras Holm" if p.p_holm_25<.05 and p["mean"]>0 else
                       "ventaja de la referencia respaldada tras Holm" if p.p_holm_25<.05 else
                       "diferencia no respaldada tras Holm; no demuestra equivalencia")
            lines += [f"- Frente a {LABELS[reference]}: Δ={p['mean']:.5f}, IC95% [{p.ci95_low:.5f},{p.ci95_high:.5f}], "
                      f"p Holm={p.p_holm_25:.5g}; {statement}. Razón de tiempo DE/referencia={p.optimization_time_ratio:.3f}, razón SD={p.sd_ratio:.3f}."]
        lines += [""]
    filters=paired[paired.reference.isin(("green_variance",GVSMETHOD))]
    both_better=[i for i in INSTANCES if np.all((filters[filters.instance==i]["mean"]<0)&(filters[filters.instance==i].p_holm_25<.05))]
    l026=tables[tables.instance=="L026"].set_index("method")
    lines += ["**Síntesis de calidad y costo:** con este presupuesto, Green Variance y GV+S alcanzan una media final superior a DE con respaldo tras Holm en "
              + ", ".join(both_better) + ". "
              + f"En L026, la SD de DE es {l026.loc[METHOD,'objective_final_sd']:.5f}, frente a {l026.loc['green_variance','objective_final_sd']:.5f} de GV "
              + f"y {l026.loc[GVSMETHOD,'objective_final_sd']:.5f} de GV+S: es una menor dispersión descriptiva entre réplicas. "
              + "La mejora descriptiva de su media frente a esos métodos en L026 no queda respaldada por los contrastes de medias ajustados. "
              + f"Las razones registradas de tiempo DE/filtro LR varían entre {filters.optimization_time_ratio.min():.3f} y {filters.optimization_time_ratio.max():.3f}; "
              + "este ahorro de wall time observado se interpreta junto con la pérdida de calidad en cuatro instancias y con las diferencias entre sesiones de medición. "
              + "No demuestra por sí solo una ventaja causal de tiempo de CPU.","",
              "**Respuesta al objetivo científico:** el benchmark permite juzgar calidad independiente y costo bajo el presupuesto de routing especificado, "
              "y se informan tanto victorias como derrotas de DE. Una diferencia no significativa no basta para afirmar que un método alcanza soluciones equivalentes con menor costo. "
              "La combinación de tiempo, conteos y calidad debe interpretarse por instancia. La muestra fija pequeña, el presupuesto reducido de consultas y la única configuración DE "
              "limitan la generalización a otras estrategias SAA o presupuestos. Se conservan soluciones desfavorables, sin selección de semillas ni exclusión de extremos.","",
              "Los gráficos `de_training_vs_independent_objective` exponen las diferencias entre entrenamiento y validación. "
              "Su brecha también incluye cambios de política seleccionada y ruido de evaluación; no se identifica automáticamente con sesgo puro por sobreajuste.","",
              "## 8. Verificación y problemas", "",
              "- Pruebas previas: 13.563 comprobaciones por cada una de seis configuraciones; identidad del objetivo frente al evaluador original, señales por umbrales y distribuciones marginales, CRN, límites, semillas, repetibilidad y contadores. No se ejecutaron optimizadores históricos.",
              "- Auditoría de producción: las 253 muestras guardadas se regeneraron con sus semillas para comprobar identidad; se reprodujeron tres candidatos directos por réplica y el prefijo N=500 del smoke test. Este trabajo de verificación está separado del presupuesto de optimización en `saved_de_audit.toml`.",
              "- Los CSV y metadatos se verificaron por hashes, tamaños, rangos, monotonía del incumbente de entrenamiento, ausencia de NaN/Inf en objetivos y decisiones, streams y presupuesto. ESS y gradientes se dejan vacíos porque DE no los utiliza.",
              "- Las estadísticas históricas se reprodujeron desde los datos originales y coincidieron con las tablas archivadas; los gráficos son actualizaciones separadas en 022.",
              f"- Conservación: {preservation['checked_files']} archivos históricos/fuentes comprobados por SHA256, sin cambios.",
              "- El entorno Python emite avisos sobre Axes3D y aceleradores opcionales antiguos; se utilizan figuras 2D y operaciones verificadas. No se modificaron los entornos históricos.",
              "- Se corrigió un error de sintaxis del módulo nuevo durante las primeras pruebas; ninguna réplica de producción usó esa versión. El log inicial se conserva en `validation/initial_parse_failure.log`.",
              "- Los límites de evaluaciones y parámetros RNG de la biblioteca requirieron el tratamiento explícito descrito arriba; no se modificó la implementación DE de la biblioteca.","",
              "## 9. Reproducción y reanudación", "", "Desde este directorio:","", "```bash",
              "julia --startup-file=no --project=environment -e 'using Pkg; Pkg.instantiate()'",
              "python3 code/run_022.py test --cohort both",
              "python3 code/run_022.py run --cohort statistical --instances L000 --replications 7 --workers 1",
              "python3 code/run_022.py run --cohort statistical --instances I0252 L025 --replications 1 2 3 --workers 4",
              "python3 code/run_022.py run --cohort both --workers 12",
              "python3 code/finish_022.py",
              "python3 code/run_022.py status", "```", "",
              "Las réplicas completas se validan y omiten. Búsqueda, validación de curvas y holdout tienen checkpoints separados. "
              "Una búsqueda DE interrumpida antes de guardar `search.jls` se reinicia determinísticamente; las réplicas completas permanecen intactas. "
              "El trabajo de intentos interrumpidos figura en los logs y no se incorpora como observación estadística. No se produjeron interrupciones en esta campaña.","",
              "Para reproducir una réplica en otro destino sin sobrescribir la campaña: "
              "`python3 code/run_022.py test --instances L000 --output-root ../022_de_replay_rep7` y después "
              "`python3 code/run_022.py run --instances L000 --replications 7 --workers 1 --output-root ../022_de_replay_rep7`.", "",
              "Documentación primaria: [BlackBoxOptim.jl](https://github.com/SciML/BlackBoxOptim.jl). "
              "Los archivos de implementación y licencia de la versión instalada están archivados en `validation/package_sources/`; "
              "los resultados numéricos de este informe proceden exclusivamente de los archivos del estudio.",""]
    atomic_bytes(BASE/"RESUMEN_EXPERIMENTAL_022.md","\n".join(lines).encode())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    analysis_started=time.perf_counter();plot_seconds={}
    data={};alltables=[];allpairs=[];alltargets=[];seeds=[];regressions=[]
    for instance in INSTANCES:
        out=output("statistical",instance)
        audit=toml.load(out/"validation/saved_de_audit.toml")
        assert audit["passed"] and audit["published_replications_checked"]==50
        print(f"LOAD {instance}",flush=True)
        tr,fin,costs,results=load_instance(instance)
        for rep in range(1,51):
            meta=verify_numeric("statistical",instance,rep);seeds.extend(meta["seeds"].values())
        print(f"VERIFIED {instance}",flush=True)
        budgetstats,_=sampled_statistics(tr,"budget",np.arange(0,50001,500))
        maximum=tr.cumulative_optimization_seconds.max()
        timestats,_=sampled_statistics(tr,"cumulative_optimization_seconds",np.linspace(0,maximum,201))
        regressions.append(historical_regression(instance,budgetstats))
        table,details,pairs=final_statistics(instance,fin)
        hits,targetstats=quality_target(instance,tr)
        for name,frame in (("trajectories_all",tr),("final_results_all",fin),("budget_statistics",budgetstats),
                           ("time_statistics",timestats),("final_comparison",table),("final_statistics",details),
                           ("paired_final_differences",pairs),("computational_accounting",costs),
                           ("quality_target_hits",hits),("quality_target_summary",targetstats)):
            save_csv(out/"summaries"/f"{name}.csv",frame)
        plot_started=time.perf_counter()
        individual_figures(instance,tr,fin,budgetstats,timestats,hits)
        plot_seconds[instance]=time.perf_counter()-plot_started
        data[instance]=(tr,fin,budgetstats,timestats,hits)
        alltables.append(table);allpairs.append(pairs);alltargets.append(targetstats)
        atomic_json(out/"summaries/analysis_manifest.json",dict(fingerprint=config(out)["fingerprint"],
            replications=dict.fromkeys(METHODS,50),statistical_unit="independent algorithm replication",
            historical_source=str(historical(instance)),historical_optimizers_executed=0,
            completion_manifests={str(path):sha(path/"complete.toml") for path,_,_,_ in results},
            confidence_protocol="original moments(), pointwise Student-t 95%, ddof=1",
            time_sampling="LOCF; final incumbent retained after termination; validation excluded from clock"))
        print(f"ANALYZED {instance}",flush=True)
    for rep in (1,2,3):
        meta=verify_numeric("017","I0252",rep);seeds.extend(meta["seeds"].values())
    assert len(seeds)==len(set(seeds))==1265
    tables=pd.concat(alltables,ignore_index=True);paired=holm(pd.concat(allpairs,ignore_index=True));targets=pd.concat(alltargets,ignore_index=True)
    for name,frame in (("final_comparison",tables),("paired_final_differences",paired),("quality_target_summary",targets)):
        save_csv(BASE/"comparisons/statistical/summaries"/f"{name}.csv",frame)
    for instance in INSTANCES:
        save_csv(output("statistical",instance)/"summaries/paired_final_differences.csv",paired[paired.instance==instance])
    plot_started=time.perf_counter();fin017=compare_017()
    comparison017_seconds=time.perf_counter()-plot_started
    plot_started=time.perf_counter();grouped_figures(data,paired,tables)
    grouped_plot_seconds=time.perf_counter()-plot_started
    preservation=verify_history()
    figures=sorted(BASE.rglob("*.png"));assert len(figures)==76, f"Expected 76 figures; found {len(figures)}"
    for path in figures:
        assert path.with_suffix(".pdf").is_file()
    report(tables,paired,targets,fin017,preservation,len(figures))
    analysis_timing=dict(instance_plot_seconds=plot_seconds,grouped_plot_seconds=grouped_plot_seconds,
                         comparison_017_load_and_plot_seconds=comparison017_seconds,
                         total_analysis_wall_seconds=time.perf_counter()-analysis_started,
                         included_in_optimization_time=False)
    atomic_json(BASE/"validation/analysis_timing.json",analysis_timing)
    atomic_json(BASE/"COMPLETE.json",dict(complete=True,new_statistical_replications=250,new_017_replications=3,
        total_new_replications=253,figures=len(figures),png_files=len(figures),pdf_files=len(figures),
        historical_preservation=preservation,unique_new_rng_streams=1265,regressions=regressions,analysis_timing=analysis_timing,
        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob("*") if p.is_file()},
        artifacts={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob("*") if p.is_file() and p.suffix in (".csv",".png",".pdf",".md")},
        source_package="BlackBoxOptim 0.6.12",variant="adaptive_de_rand_1_bin_radiuslimited"))
    print(tables.to_string(index=False));print(paired[["instance","reference","mean","ci95_low","ci95_high","p_holm_25"]].to_string(index=False))


if __name__=="__main__":
    main()
