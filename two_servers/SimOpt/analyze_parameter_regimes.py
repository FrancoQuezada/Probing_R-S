#!/usr/bin/env python3
from probing_regime_structure import *

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);result=np.empty(len(p))
    result[order]=np.minimum(1.,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return result

def phase_data(out,phase,suffix):
    frame=parts(out,phase,suffix)
    planpath=out/f"configs/{phase}_plan.toml"
    if not frame.empty and planpath.exists():
        plan=tomllib.loads(planpath.read_text())
        frame=frame[frame.scenario.isin(plan["scenarios"])&(frame.rep<=plan["n_reps"])].copy()
    return frame

def compare(paired):
    rows=[]
    for key,g in paired.groupby(["phase","scenario","estimand","method_a","method_b"],sort=True):
        phase,s,estimand,a,b=key;x=g.difference.to_numpy();n=len(x)
        sd=x.std(ddof=1);se=sd/np.sqrt(n);crit=stats.t.ppf(.975,n-1)
        p=2*stats.t.sf(abs(x.mean()/se),n-1) if se>0 else (1. if x.mean()==0 else 0.)
        rows.append(dict(phase=phase,scenario=s,estimand=estimand,method_a=a,method_b=b,n_reps=n,
            mean_difference=x.mean(),sd=sd,se=se,ci95_low=x.mean()-crit*se,ci95_high=x.mean()+crit*se,
            p_value=p,positive_proportion=np.mean(x>0),evaluation_mc_rms_se=np.sqrt(np.mean(g.evaluation_paired_se**2)),
            optimization_sd_approx=np.sqrt(max(0.,sd*sd-np.mean(g.evaluation_paired_se**2)))))
    result=pd.DataFrame(rows)
    if result.empty:return result
    result["p_holm_scenario"]=np.nan;result["p_holm_global"]=np.nan
    for _,g in result.groupby(["phase","scenario"]):result.loc[g.index,"p_holm_scenario"]=holm(g.p_value.fillna(1))
    for _,g in result.groupby("phase"):
        result.loc[g.index,"p_holm_global"]=holm(g.p_value.fillna(1))
        crit=stats.t.ppf(1-.05/(2*len(g)),g.n_reps-1)
        result.loc[g.index,"simultaneous_ci95_low"]=g.mean_difference-crit*g.se
        result.loc[g.index,"simultaneous_ci95_high"]=g.mean_difference+crit*g.se
    return result

def factorial(paired):
    rows=[]
    for (phase,a,b),g in paired[paired.estimand=="holdout"].groupby(["phase","method_a","method_b"]):
        cells=g.groupby("scenario").difference.agg(["mean","var","count"])
        if set(cells.index)!=set(SCENARIOS):continue
        for factors in ("A","B","C","AB","AC","BC","ABC"):
            sign=np.array([np.prod([2*int(s[1+"ABC".index(f)])-1 for f in factors]) for s in cells.index])
            effect=np.dot(sign,cells["mean"])/4
            contributions=cells["var"]/cells["count"]/16
            variance=contributions.sum();se=np.sqrt(variance)
            df=variance**2/np.sum(contributions**2/(cells["count"]-1)) if variance>0 else np.inf
            crit=stats.t.ppf(.975,df)
            p=2*stats.t.sf(abs(effect/se),df) if se>0 else (1. if effect==0 else 0.)
            rows.append(dict(phase=phase,method_a=a,method_b=b,factor=factors,effect=effect,se=se,df=df,
                ci95_low=effect-crit*se,ci95_high=effect+crit*se,p_value=p,
                interpretation="exploratory pilot" if phase=="pilot" else "full-factorial confirmation"))
    frame=pd.DataFrame(rows)
    if not frame.empty:
        frame["p_holm"]=np.nan
        for _,g in frame.groupby("phase"):frame.loc[g.index,"p_holm"]=holm(g.p_value)
    return frame

def scenario_table(out,comparisons,regimes):
    model=pd.read_csv(out/"scenarios.csv")
    base=model.merge(regimes,on="scenario",how="left")
    structural=pd.read_csv(out/"structural_diagnostics.csv")
    z0=structural[structural.point=="Z0"][["scenario","regime","selection_uncertainty","relative_gradient_gap","routing_disagreement"]]
    base=base.merge(z0.rename(columns={c:c+"_Z0" for c in z0.columns if c!="scenario"}),on="scenario",how="left")
    routing=pd.read_csv(out/"routing_policy_diagnostics.csv")
    winners=structural[["scenario","point","best_A"]].rename(columns={"best_A":"policy"})
    queue=routing.merge(winners,on=["scenario","point","policy"])
    queue=queue.groupby("scenario")[["fast_fraction","fast_utilization","total_wait","delay_penalty","routing_cost"]].mean().add_prefix("reference_winner_").reset_index()
    base=base.merge(queue,on="scenario",how="left")
    names={"Fresh + Softmax beta=0.5":"softmax_minus_fresh","Fresh + PSoft B=300":"psoft_minus_fresh",
           "Green + Softmax beta=0.5":"green_softmax_minus_green","Green":"green_minus_fresh"}
    rows=[]
    for phase in comparisons.phase.unique():
        for row in base.to_dict("records"):
            row=dict(row,phase=phase)
            c=comparisons[(comparisons.phase==phase)&(comparisons.scenario==row["scenario"])&(comparisons.estimand=="holdout")]
            for r in c.itertuples():
                prefix=names[r.method_a]
                for key,value in (("mean",r.mean_difference),("se",r.se),("ci95_low",r.ci95_low),("ci95_high",r.ci95_high),
                    ("p_holm_scenario",r.p_holm_scenario),("p_holm_global",r.p_holm_global),("positive_proportion",r.positive_proportion),("n_reps",r.n_reps)):
                    row[prefix+"_"+key]=value
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/"scenario_summary.csv",index=False)
    return frame

def audit(out,cfg,fingerprint,finals,paired):
    results={"fingerprint":fingerprint,"quick":cfg["quick"],"phases":{}}
    for file,sha in cfg["source_sha256"].items():
        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest()!=sha:raise RuntimeError(f"Changed source: {file}")
    if not finals.empty:
        assert np.isfinite(finals[["objective_holdout","objective_direct","evaluation_se"]]).all().all()
        assert (finals.objective_direct>=finals.objective_holdout-1e-9).all()
        assert (finals.search_paths==cfg["n_paths"]*cfg["max_iters"]).all()
        assert ((finals[["z3","z4"]]>=0)&(finals[["z3","z4"]]<=1)).all().all()
    for phase in ("pilot","confirmation"):
        planpath=out/f"configs/{phase}_plan.toml"
        if not planpath.exists():results["phases"][phase]={"complete":False};continue
        plan=tomllib.loads(planpath.read_text());f=finals[finals.phase==phase]
        expected=len(plan["scenarios"])*plan["n_reps"]*5
        complete=len(f)==expected and all(len(g)==plan["n_reps"] for _,g in f.groupby(["scenario","method"]))
        if not complete:raise RuntimeError(f"Incomplete {phase}: {len(f)}/{expected}")
        coords=phase_data(out,phase,"coordinates")
        assert ((coords[["z3","z4"]]>=0)&(coords[["z3","z4"]]<=1)).all().all()
        diag=phase_data(out,phase,"diagnostics")
        assert np.isfinite(diag[["g3","g4","gradient_norm"]]).all().all()
        assert (diag[diag.method.str.startswith("Green")].banks==diag[diag.method.str.startswith("Green")].iteration).all()
        lookup=f.set_index(["scenario","rep","method"])
        for r in paired[paired.phase==phase].itertuples():
            column="objective_"+r.estimand
            difference=lookup.loc[(r.scenario,r.rep,r.method_a),column]-lookup.loc[(r.scenario,r.rep,r.method_b),column]
            assert np.isclose(difference,r.difference,atol=1e-9,rtol=1e-10)
        results["phases"][phase]={"complete":True,"scenarios":plan["scenarios"],"n_reps":plan["n_reps"],"endpoint_rows":len(f)}
    seeds=parts(out,"characterization","seeds")
    for phase in ("pilot","confirmation"):
        extra=phase_data(out,phase,"seeds")
        if not extra.empty:seeds=pd.concat([seeds,extra],ignore_index=True)
    if not seeds.empty:
        seeds["iteration"]=seeds.get("iteration",pd.Series(0,index=seeds.index)).fillna(0).astype(int)
        keys=["scenario","phase","rep","stream","iteration"]
        seeds=seeds.drop_duplicates(keys)
        assert seeds.seed.nunique()==len(seeds),"Seed collision between independent streams"
        seeds.to_csv(out/"tables/seed_manifest.csv",index=False)
        results["unique_seed_streams"]=len(seeds)
    char=json.loads((out/"configs/characterization_status.json").read_text())
    results["characterization_complete"]=char["complete"]
    routing_manifest=out/"validation/routing_decisions.toml"
    if routing_manifest.exists():
        routing=tomllib.loads(routing_manifest.read_text())
        assert routing["passed"] and routing["fingerprint"]==fingerprint
        assert routing["source_sha256"]==hashlib.sha256((ROOT/"export_parameter_regime_routing.jl").read_bytes()).hexdigest()
        expected_rows=len(pd.read_csv(out/"routing_policy_diagnostics.csv"))*cfg["reference_paths"]
        assert routing["rows"]==expected_rows
        archives=list((out/"characterization").glob("S*/*_routing_decisions.csv.gz"))
        assert len(archives)==char["points"]
        archived_rows=0
        for archive in archives:
            point=archive.name.removesuffix("_routing_decisions.csv.gz")
            metadata=tomllib.loads((out/f"checkpoints/routing_decisions_{archive.parent.name}_{point}.toml").read_text())
            assert metadata["fingerprint"]==fingerprint and metadata["source_sha256"]==routing["source_sha256"]
            assert metadata["sha256"]==hashlib.sha256(archive.read_bytes()).hexdigest()
            archived_rows+=metadata["rows"]
        assert archived_rows==routing["rows"]
        results["routing_decisions_archived_rows"]=routing["rows"]
    selected=tomllib.loads((out/"configs/selected_scenarios.toml").read_text()) if (out/"configs/selected_scenarios.toml").exists() else {}
    if selected:
        assert hashlib.sha256((out/"structural_diagnostics.csv").read_bytes()).hexdigest()==selected["structural_sha256"],"Frozen structural evidence changed"
    cp=results["phases"]["confirmation"]
    results["complete"]=bool(char["complete"] and results["phases"]["pilot"].get("complete",False) and
        cp.get("complete",False) and set(selected.get("scenarios",[])).issubset(cp.get("scenarios",[])) and "S000" in cp.get("scenarios",[]))
    (out/"validation/result_audit.json").write_text(json.dumps(results,indent=2))
    return results

def analyze(out,cfg,fingerprint):
    if not (out/"structural_diagnostics.csv").exists():classify(out,cfg,fingerprint)
    regimes=pd.read_csv(out/"tables/structural_scenario_summary.csv")
    finals=pd.concat([phase_data(out,p,"finals") for p in ("pilot","confirmation")],ignore_index=True)
    paired=pd.concat([phase_data(out,p,"paired") for p in ("pilot","confirmation")],ignore_index=True)
    finals.to_csv(out/"final_objectives.csv",index=False);paired.to_csv(out/"paired_replications.csv",index=False)
    if not paired.empty:
        comp=compare(paired);comp.to_csv(out/"paired_comparisons.csv",index=False)
        fac=factorial(paired);fac.to_csv(out/"tables/factorial_effects.csv",index=False)
        scenarios=scenario_table(out,comp,regimes)
    else:comp=pd.DataFrame();fac=pd.DataFrame();scenarios=pd.DataFrame()
    coords=pd.concat([phase_data(out,p,"coordinates") for p in ("pilot","confirmation")],ignore_index=True)
    traces=pd.concat([phase_data(out,p,"traces") for p in ("pilot","confirmation")],ignore_index=True)
    diag=pd.concat([phase_data(out,p,"diagnostics") for p in ("pilot","confirmation")],ignore_index=True)
    costs=pd.concat([phase_data(out,p,"costs") for p in ("pilot","confirmation")],ignore_index=True)
    traces.to_csv(out/"convergence_checkpoints.csv",index=False);costs.to_csv(out/"tables/evaluation_costs.csv",index=False)
    if not coords.empty:
        aligned=diag.rename(columns={"iteration":"gradient_iteration"}).copy();aligned["iteration"]=aligned.gradient_iteration-1
        aligned=aligned.drop(columns=["z3","z4"])
        keys=["scenario","phase","rep","method","iteration"]
        full=coords.merge(aligned,on=keys,how="left")
        ev=traces[keys+["objective"]].rename(columns={"objective":"objective_independent"})
        full=full.merge(ev,on=keys,how="left");full.to_csv(out/"optimization_traces.csv",index=False)
    result=audit(out,cfg,fingerprint,finals,paired)
    make_plots(out,cfg,finals,traces,coords,comp,fac,scenarios)
    write_report(out,cfg,regimes,finals,comp,fac,scenarios,result)
    if result["complete"]:(out/"COMPLETE.json").write_text(json.dumps(result,indent=2))
    print("ANALYSIS COMPLETE",json.dumps(result),flush=True)

METHOD_NAMES=["Fresh","Fresh + Softmax beta=0.5","Fresh + PSoft B=300","Green","Green + Softmax beta=0.5"]
SHORT={METHOD_NAMES[0]:"Fresh",METHOD_NAMES[1]:"Softmax β=.50",METHOD_NAMES[2]:"PSoft B=300",
       METHOD_NAMES[3]:"Green",METHOD_NAMES[4]:"Green + Softmax β=.50"}
COLORS=dict(zip(METHOD_NAMES,["black","royalblue","purple","forestgreen","darkorange"]))

def make_plots(out,cfg,finals,traces,coords,comp,fac,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"figure.figsize":(10,6),"figure.dpi":130,"font.size":10,"axes.grid":True,"grid.alpha":.20})
    figures=out/"figures";figures.mkdir(exist_ok=True)
    context=(f"{'QUICK TEST; ' if cfg['quick'] else ''}009; n={cfg['n_paths']}; iterations={cfg['max_iters']}; jobs/path={cfg['jobs_per_path']}; βsoft=.50; PSoft B=300\n"
        f"Reference={cfg['reference_paths']} A + B; selection tests={cfg['selection_reps']}/n; evaluation trace/selection/holdout={cfg['trace_paths']}/{cfg['selection_paths']}/{cfg['final_paths']}; step=.035/sqrt(k), clip ±15")
    def save(fig,name,extra=""):
        fig.text(.01,.006,context+("\n"+extra if extra else ""),fontsize=7,va="bottom")
        fig.tight_layout(rect=(0,.075,1,1));fig.savefig(figures/(name+".png"));plt.close(fig)
    def band(ax,g,label,color):
        a=g.groupby("iteration").objective.agg(["mean","std"])
        ax.plot(a.index,a["mean"],label=label,color=color)
        ax.fill_between(a.index.to_numpy(float),(a["mean"]-a["std"]).to_numpy(float),
                        (a["mean"]+a["std"]).to_numpy(float),color=color,alpha=.12)
    if not traces.empty:
        for (phase,s),g in traces.groupby(["phase","scenario"]):
            fig,ax=plt.subplots()
            for m in METHOD_NAMES:band(ax,g[g.method==m],SHORT[m],COLORS[m])
            nr=g.rep.nunique();ax.set(xlabel="Iteration",ylabel="Independent direct hard-max objective",title=f"{s} — {phase}, {nr} repetitions; bands ±1 SD")
            ax.legend(fontsize=8);save(fig,f"convergence_{phase}_{s}","One common evaluation CRN bank per scenario/phase; conditional optimization variability.")
    if not finals.empty:
        for (phase,s),g in finals.groupby(["phase","scenario"]):
            fig,axes=plt.subplots(1,2,figsize=(13,6))
            for ax,metric,title in zip(axes,["objective_holdout","objective_direct"],["Independent selected-policy holdout","Direct sample hard max"]):
                ax.boxplot([g[g.method==m][metric] for m in METHOD_NAMES],tick_labels=[SHORT[m] for m in METHOD_NAMES],showmeans=True)
                ax.tick_params(axis="x",labelrotation=50,labelsize=8);ax.set(ylabel="Objective",title=title)
            fig.suptitle(f"{s} — {phase}, {g.rep.nunique()} independent repetitions")
            save(fig,f"final_objectives_{phase}_{s}")
    if not comp.empty:
        for phase,g in comp[comp.estimand=="holdout"].groupby("phase"):
            ns=g.scenario.nunique()
            fig,axes=plt.subplots(1,3,figsize=(14,3.5 if ns==1 else 5))
            fig.suptitle(f"{phase}: {g.n_reps.iloc[0]} repetitions per scenario, {ns} scenario"+("s" if ns!=1 else ""))
            for ax,a,b in zip(axes,[METHOD_NAMES[1],METHOD_NAMES[2],METHOD_NAMES[4]],
                                  [METHOD_NAMES[0],METHOD_NAMES[0],METHOD_NAMES[3]]):
                h=g[(g.method_a==a)&(g.method_b==b)].sort_values("scenario")
                pos=np.arange(len(h));err=np.vstack([h.mean_difference-h.ci95_low,h.ci95_high-h.mean_difference])
                ax.errorbar(h.mean_difference,pos,xerr=np.maximum(err,0),fmt="o",color=COLORS[a],capsize=3)
                for k,r in enumerate(h.itertuples()):
                    if r.p_holm_global<.05:ax.plot(r.mean_difference,k,"o",ms=10,mfc="none",mec="black")
                ax.set(yticks=pos,yticklabels=h.scenario,xlabel="Mean paired holdout difference",title=f"{SHORT[a]} − {SHORT[b]}")
                ax.axvline(0,color="black",lw=1)
            save(fig,f"paired_forest_{phase}","Marginal 95% t intervals; black rings: Holm adjusted across all scenarios <.05. Pilot is exploratory.")
    if not coords.empty:
        structural=pd.read_csv(out/"structural_diagnostics.csv")
        for (phase,s),g in coords.groupby(["phase","scenario"]):
            oracle=out/f"characterization/oracle_{s}.csv"
            if phase=="confirmation" and oracle.exists():
                surface=pd.read_csv(oracle);value="objective";subtitle=f"{cfg['oracle_grid_n']}×{cfg['oracle_grid_n']} own direct CRN oracle"
            else:
                surface=structural[(structural.scenario==s)&(structural.point!="Z0")];value="objective_A";subtitle="own 3×3 exploratory direct CRN reference"
            matrix=surface.pivot(index="z4",columns="z3",values=value)
            fig,ax=plt.subplots();im=ax.pcolormesh(matrix.columns,matrix.index,matrix,shading="auto",cmap="viridis")
            fig.colorbar(im,ax=ax,label="Scenario-specific J(z)")
            for m in METHOD_NAMES:
                h=g[g.method==m]
                for _,run in h.groupby("rep"):ax.plot(run.z3,run.z4,color=COLORS[m],alpha=.06,lw=.8)
                avg=h.groupby("iteration")[["z3","z4"]].mean()
                ax.plot(avg.z3,avg.z4,color=COLORS[m],lw=2,label=SHORT[m])
                ends=h[h.iteration==cfg["max_iters"]];ax.scatter(ends.z3,ends.z4,color=COLORS[m],s=16,alpha=.6)
            ax.set(xlabel="z3",ylabel="z4",xlim=(0,1),ylim=(0,1),title=f"{s} — {phase}; {subtitle}")
            ax.legend(fontsize=7,loc="best");save(fig,f"oracle_trajectories_{phase}_{s}","Bold: mean trajectory; faint: individual trajectories; dots: endpoints. No LR used for this surface.")
    selection=pd.read_csv(out/"policy_selection_diagnostics.csv")
    fig,axes=plt.subplots(2,4,figsize=(15,8),sharex=True,sharey=True)
    for ax,s in zip(axes.flat,SCENARIOS):
        g=selection[selection.scenario==s]
        if g.empty:continue
        a=g.groupby("n_paths").nominal_selection_error.agg(["mean","min","max"])
        ax.plot(a.index,a["mean"],"o-",label="Outside nominal ε-set")
        ax.fill_between(a.index.to_numpy(float),a["min"].to_numpy(float),a["max"].to_numpy(float),alpha=.12)
        c=g.groupby("n_paths").certified_outside_frequency.mean();ax.plot(c.index,c,"s--",label="Certified outside")
        z=g[g.point=="Z0"];ax.plot(z.n_paths,z.nominal_selection_error,"x:",label="Z0 nominal")
        ax.set(title=s,xlabel="Selection bank paths",ylabel="Selection frequency",ylim=(-.02,1.02))
    axes[0,0].legend(fontsize=7);save(fig,"selection_error_vs_sample_size","Band: range over fixed reference points, not a confidence interval. ε=.50; finite reference uncertainty reported separately.")
    structural=pd.read_csv(out/"structural_diagnostics.csv");z=structural[structural.point=="Z0"].sort_values("scenario")
    fig,axes=plt.subplots(1,2,figsize=(13,5));pos=np.arange(len(z))
    for ax,metric,lo,hi,label in [(axes[0],"gradient_distance","gradient_distance_low","gradient_distance_high","||Gi−Gj||"),
                                (axes[1],"gradient_cosine","cosine_low","cosine_high","Gradient cosine")]:
        vals=z[metric];err=np.vstack([vals-z[lo],z[hi]-vals])
        ax.errorbar(pos,vals,yerr=np.maximum(err,0),fmt="o",capsize=3)
        ax.set(xticks=pos,xticklabels=z.scenario,ylabel=label,title="Most-selected nonequivalent groups at Z0")
        ax.tick_params(axis="x",labelrotation=45)
    axes[1].set_ylim(-1.05,1.05);save(fig,"policy_gradient_comparison_Z0","Paired reference uncertainty: conservative norm bounds and joint-path bootstrap cosine intervals.")
    if not summary.empty:
        for phase,frame in summary.groupby("phase"):
            for metric,label in [("softmax_minus_fresh_mean","Softmax − Fresh"),("psoft_minus_fresh_mean","PSoft − Fresh")]:
                h=frame[np.isfinite(frame[metric])]
                if h.empty:continue
                limit=max(1e-6,np.abs(h[metric]).max());fig,ax=plt.subplots()
                scatter=ax.scatter(h.selection_uncertainty,h.relative_gradient_gap,c=h[metric],s=110,
                    cmap="coolwarm",norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),edgecolor="black")
                for j,r in enumerate(h.itertuples()):
                    right=r.selection_uncertainty>h.selection_uncertainty.median()
                    ax.annotate(r.scenario,(r.selection_uncertainty,r.relative_gradient_gap),
                        xytext=(-6 if right else 6,6 if j%2 else -12),ha="right" if right else "left",textcoords="offset points",fontsize=8)
                fig.colorbar(scatter,ax=ax,label=f"{phase} mean holdout improvement: {label}")
                ax.set(xlabel="Selection uncertainty: mean(1−modal group frequency), n=500",
                    ylabel="Mean relative difference between competitive gradients",title=f"Structural map — {phase}: {label}")
                save(fig,f"structural_map_{phase}_{metric}","Eight fixed scenarios in pilot; regimes: "+", ".join(sorted(h.regime.unique()))+". Associations are descriptive, not general causal evidence.")
    if not fac.empty:
        for phase,g in fac.groupby("phase"):
            contrasts=list(g.groupby(["method_a","method_b"],sort=False));fig,axes=plt.subplots(2,2,figsize=(14,9))
            fig.suptitle(f"Full 2×2×2 factorial effects — {phase}"+(" (exploratory)" if phase=="pilot" else ""))
            for ax,((a,b),h) in zip(axes.flat,contrasts):
                order=["A","B","C","AB","AC","BC","ABC"];h=h.set_index("factor").loc[order].reset_index();y=np.arange(7)
                err=np.vstack([h.effect-h.ci95_low,h.ci95_high-h.effect])
                ax.errorbar(h.effect,y,xerr=np.maximum(err,0),fmt="o",capsize=3)
                ax.axvline(0,color="black",lw=1);ax.set(yticks=y,yticklabels=order,xlabel="High−low factorial effect on paired difference",title=f"{SHORT[a]} − {SHORT[b]}")
            save(fig,f"factorial_effects_{phase}","A=initial information; B=congestion; C=fast-server cost. Welch 95% intervals; pilot interactions are exploratory.")
    if not finals.empty:
        for phase,g in finals.groupby("phase"):
            fig,axes=plt.subplots(1,2,figsize=(13,5))
            for m in METHOD_NAMES:
                h=g[g.method==m]
                axes[0].scatter(h.search_seconds,h.objective_holdout,label=SHORT[m],color=COLORS[m],alpha=.5,s=20)
                means=h.groupby("scenario").search_seconds.mean()
                axes[1].plot(means.index,means,"o-",label=SHORT[m],color=COLORS[m])
            axes[0].set(xlabel="Instrumented search wall seconds",ylabel="Independent holdout value",title=phase)
            axes[1].set(xlabel="Scenario",ylabel="Mean search wall seconds",title="Evaluation excluded; concurrent replications")
            axes[0].legend(fontsize=7);axes[1].tick_params(axis="x",labelrotation=45);save(fig,f"computational_cost_{phase}")

def write_report(out,cfg,regimes,finals,comp,fac,summary,result):
    structural=pd.read_csv(out/"structural_diagnostics.csv")
    sel=pd.read_csv(out/"policy_selection_diagnostics.csv")
    gradients=pd.read_csv(out/"policy_gradient_diagnostics.csv")
    routes=pd.read_csv(out/"routing_disagreement.csv")
    selected=tomllib.loads((out/"configs/selected_scenarios.toml").read_text()) if (out/"configs/selected_scenarios.toml").exists() else {}
    title="PRUEBA REDUCIDA" if cfg["quick"] else "EXPERIMENTO COMPLETO" if result["complete"] else "RESULTADOS PARCIALES"
    report=["009 — REGÍMENES PARAMÉTRICOS: "+title,
        "Objetivo: J(z)=max_i E_z[Y_i]−ALPHA'z, políticas elegidas ex ante. Se conserva el modelo de dos servidores y 144 políticas de 002–008. No se introduce la formulación general de dos etapas ni el máximo realizado por camino de SimOpt009.",
        f"Configuración: {cfg['n_paths']} caminos x {cfg['max_iters']} iteraciones por método; {cfg['jobs_per_path']} trabajos/camino. Softmax beta=.50 y PSoft B=300 fijados antes del experimento. La actualización, clipping y ausencia de early stopping se preservan.",
        "La configuración base reprodujo EXACTAMENTE cinco búsquedas guardadas de 100 iteraciones x500 caminos: Fresh, Softmax, Green, Green+Softmax y PSoft. También se verificaron marginales CRN, LR/scores por diferencias finitas, parámetros económicos y aislamiento de historias por escenario.",
        f"Fase I: {len(structural)} puntos caracterizados; referencias independientes A/B de {cfg['reference_paths']} caminos, y {cfg['selection_reps']} bancos independientes por n=50,100,500 y punto. Epsilon nominal=.50. Referencia A identifica candidatos; B estima gradientes/routing e incertidumbre de pertenencia al conjunto epsilon-óptimo.",
        "Se distinguen elección fuera del conjunto epsilon nominal, elección certificadamente fuera y pertenencia ambigua según IC simultáneos de diferencias emparejadas de referencia. Ninguna política ganadora finita se trata como óptima verdadera. Los grupos de rewards equivalentes en A/B no son errores de selección y no prueban identidad poblacional o de routing.",
        "La clasificación operacional usa frecuencias por grupo equivalente con n=500 y límites Wilson; R2 requiere similitud de gradientes apoyada por precisión y R3 diferencias apoyadas por límites de incertidumbre. Los casos no resueltos permanecen indeterminados. Etiquetar un escenario R3/R2 indica presencia en la grilla, no uniformidad en todo el dominio.",
        "Escenarios elegidos SOLO por estructura, antes del piloto: "+", ".join(selected.get("scenarios",[]))+". Regímenes sin representante de escenario: "+", ".join(selected.get("missing_regimes",[]))+".",
        "Clasificación de los puntos: "+", ".join(f"{label}={count}" for label,count in structural.regime.value_counts().items())+". No identificar un régimen con este presupuesto no demuestra que esté ausente.",
        "Búsquedas completas: "+"; ".join(f"{phase}: {g.scenario.nunique()} escenarios × {g.rep.nunique()} repeticiones × {g.method.nunique()} métodos" for phase,g in finals.groupby("phase"))+".",
        "Piloto y confirmación usan semillas independientes. La inferencia principal usa diferencias holdout emparejadas entre repeticiones; IC t95% marginales, Holm dentro de escenario y Holm adicional entre escenarios. Se incluyen IC Bonferroni simultáneos. Los resultados del piloto y sus efectos factoriales se identifican como exploratorios.",
        "El hard-max directo muestral puede tener sesgo optimista de selección. El holdout evalúa una política elegida con otra muestra, no necesariamente el máximo poblacional exacto. Son estimandos diferentes y se informan por separado.",
        "La dispersión entre repeticiones combina optimización, selección final de política y error Monte Carlo holdout. Se registran SE de evaluación y SE emparejados. optimization_sd_approx descuenta aproximadamente la varianza MC condicional, pero no separa toda la incertidumbre de selección ni corrige el sesgo del máximo muestral.",
        f"Evaluaciones: convergencia {cfg['trace_paths']}, selección final {cfg['selection_paths']}, holdout {cfg['final_paths']}. El banco de convergencia es común por escenario/fase a todas las repeticiones; sus bandas muestran variabilidad condicional de optimización. Los holdouts finales sí son independientes entre repeticiones y búsquedas.",
        "Tiempos de pared instrumentados con repeticiones concurrentes y BLAS=1; payloads idénticos compartidos se imputan a cada método. El kernel LR precalcula escalares repetidos sin cambiar el orden de suma ni resultados. Los tiempos incluyen diagnósticos comunes; no son benchmarks aislados. Evaluación fuera del presupuesto de búsqueda.",
        "P1. MAYOR INCERTIDUMBRE DE SELECCIÓN"]
    for r in regimes.sort_values("selection_uncertainty",ascending=False).itertuples():
        report.append(f"{r.scenario}: incertidumbre media 1−P(grupo modal)={r.selection_uncertainty:.4f}; error epsilon nominal={r.selection_error:.4f}; error certificado={r.certified_selection_error:.4f}; ambigüedad de referencia={r.reference_ambiguity:.4f}; régimen={r.regime}; puntos R1/R2/R3/indeterminados={r.R1_points}/{r.R2_points}/{r.R3_points}/{r.indeterminate_points}.")
    report.append("P2. DIFERENCIAS DE GRADIENTES")
    for r in regimes.itertuples():
        report.append(f"{r.scenario}: diferencia relativa media={r.relative_gradient_gap:.4f}; distancia media={r.gradient_distance:.4f}; coseno medio={r.gradient_cosine:.4f}; {r.R3_points} puntos R3 y {r.R2_points} puntos R2. No se interpreta falta de clasificación como igualdad de gradientes.")
    report.append("P3. VALORES CERCANOS Y ROUTING DIFERENTE")
    q=gradients.merge(routes,on=["scenario","point","policy_i","policy_j"],suffixes=("","_routing"))
    candidates=q[(q.gap_B.abs()<=cfg["epsilon"])&(q.disagreement>.01)].sort_values("disagreement",ascending=False)
    report.append(f"Se observaron {len(candidates)} pares/puntos con |gap_B|<=epsilon y desacuerdo de routing>1%; son descripciones de referencias finitas, no pruebas de equivalencia poblacional.")
    for r in candidates.head(10).itertuples():
        report.append(f"{r.scenario}, {r.point}, políticas {r.policy_i}/{r.policy_j}: gap_B={r.gap_B:.4f} (SE emparejado={r.gap_paired_se:.4f}), desacuerdo={r.disagreement:.4f} (SE por camino={r.disagreement_se:.4f}).")
    report.append("P4–P6. INFORMACIÓN, CONGESTIÓN Y COSTO DEL RÁPIDO")
    base=pd.read_csv(out/"scenarios.csv").merge(regimes,on="scenario")
    for factor,description in [("A","señales inicialmente menos informativas"),("B","mayor congestión: interarrival .60 frente a .80"),("C","routing rápido 1.40 frente a .70")]:
        high=base[base[factor]==1];low=base[base[factor]==0]
        report.append(f"Factor {factor}, {description}: contraste descriptivo alto−bajo en incertidumbre={high.selection_uncertainty.mean()-low.selection_uncertainty.mean():.4f}; gap={high.mean_gap.mean()-low.mean_gap.mean():.4f}; diferencia relativa de gradientes={high.relative_gradient_gap.mean()-low.relative_gradient_gap.mean():.4f}; desacuerdo de routing={high.routing_disagreement.mean()-low.routing_disagreement.mean():.4f}.")
    report.append("Los contrastes anteriores describen esta grilla factorial controlada del simulador. Las asociaciones entre estructura y ganancias no aíslan por sí solas causas ni se generalizan a otros problemas. La utilización y la espera son transitorias de 45 trabajos, no estimaciones de estado estacionario.")
    report.append("P7–P9. COMPARACIONES DE ALGORITMOS")
    if comp.empty:report.append("No hay comparaciones de búsqueda completas disponibles.")
    else:
        for phase in ("pilot","confirmation"):
            report.append("PILOTO EXPLORATORIO" if phase=="pilot" else "CONFIRMACIÓN CON SEMILLAS NUEVAS")
            for r in comp[(comp.phase==phase)&(comp.estimand=="holdout")].itertuples():
                verdict="mejora" if r.p_holm_global<.05 and r.mean_difference>0 else "deterioro" if r.p_holm_global<.05 and r.mean_difference<0 else "sin diferencia concluyente"
                report.append(f"{r.scenario}: {r.method_a}−{r.method_b}: delta={r.mean_difference:.5f}, SD={r.sd:.5f}, SE={r.se:.5f}, IC95% [{r.ci95_low:.5f},{r.ci95_high:.5f}], pHolm escenario={r.p_holm_scenario:.6g}, pHolm entre escenarios={r.p_holm_global:.6g}, proporción positiva={r.positive_proportion:.3f}; {verdict}."+(" Exploratorio, no confirmatorio." if phase=="pilot" else ""))
        confirmation=comp[(comp.phase=="confirmation")&(comp.estimand=="holdout")]
        for method,question in [(METHOD_NAMES[1],"P7. Softmax frente a Fresh"),(METHOD_NAMES[2],"P8. PSoft frente a Fresh"),(METHOD_NAMES[4],"P9. Green+Softmax frente a Green")]:
            positive=confirmation[(confirmation.method_a==method)&(confirmation.mean_difference>0)&(confirmation.p_holm_global<.05)]
            report.append(question+": "+("mejora confirmada en "+", ".join(positive.scenario)+" para este diseño." if len(positive) else "no se detectaron mejoras después del ajuste entre escenarios; esto no demuestra equivalencia."))
    report.append("EFECTOS FACTORIALES E INTERACCIONES")
    if fac.empty:report.append("No hay un factorial completo de búsquedas disponible para estimar los siete efectos.")
    else:
        for r in fac.itertuples():
            report.append(f"{r.phase}, {r.method_a}−{r.method_b}, {r.factor}: efecto alto−bajo={r.effect:.5f}, IC95% [{r.ci95_low:.5f},{r.ci95_high:.5f}], pHolm={r.p_holm:.6g}; {r.interpretation}.")
    report.append("P10–P11. RELACIÓN ESTRUCTURA/DESEMPEÑO Y REPRODUCIBILIDAD")
    correlations=[]
    if not summary.empty:
        for phase,frame in summary.groupby("phase"):
            for x in ("selection_uncertainty","selection_error","relative_gradient_gap","routing_disagreement"):
                for y in ("softmax_minus_fresh_mean","psoft_minus_fresh_mean","green_softmax_minus_green_mean"):
                    g=frame[[x,y]].dropna()
                    if len(g)<3:continue
                    rho=stats.spearmanr(g[x],g[y]).statistic if g[x].nunique()>1 and g[y].nunique()>1 else np.nan
                    correlations.append(dict(phase=phase,structural_metric=x,benefit=y,n_scenarios=len(g),spearman_rho=rho,interpretation="descriptive only"))
                    if phase=="pilot":report.append(f"Ocho puntos piloto: rho Spearman({x}, {y})={rho:.4f}; descriptivo, sin conclusión causal ni generalización.")
    pd.DataFrame(correlations).to_csv(out/"tables/structure_performance_correlations.csv",index=False)
    report.append("La incertidumbre de selección sola no prueba una ventaja del suavizado. Los gradientes de políticas, su precisión, el clipping y los pasos proyectados también intervienen. Las comparaciones confirmatorias repiten independientemente los métodos dentro de escenarios elegidos por estructura; no replican una ley general estructura/ganancia con solo ocho instancias.")
    report.append("La falta de mejoras significativas, si ocurre, es compatible con políticas competitivas de gradientes similares o con ventajas pequeñas respecto al ruido; no demuestra que Softmax/PSoft carezcan de beneficios en general. Cuando hay diferencias, se reportan tanto las positivas como las negativas y sus intervalos.")
    report.append("P12. SIGUIENTES INSTANCIAS, COMO EXPERIMENTO FUTURO")
    report.append("Regímenes no identificados operacionalmente: "+", ".join(selected.get("missing_regimes",[]))+". La presencia de puntos indeterminados refleja precisión insuficiente o criterios no satisfechos, no ausencia poblacional del régimen.")
    report.append("Cuando domina indeterminate_gradient, la prioridad es aumentar primero la precisión de referencia (por ejemplo --reference-paths 30000 en otro --output), manteniendo los mismos ocho modelos y criterios. Ello permite separar falta de precisión de ausencia de diferencias estructurales, antes de cambiar parámetros o afirmar gradientes similares.")
    report.append("Una segunda familia prospectiva podría cruzar contraste de servicios (por ejemplo STD complejo 8/12 y FAST complejo 2.4/1.8), composición de clases/complejidad y capacidad de mejora de señal BETA. Un estudio separado de ALPHA permitiría distinguir gradientes brutos de diferencias borradas por clipping/proyección. Los niveles y criterios deben fijarse antes de ver ganadores; no se ejecutaron ni ajustaron aquí.")
    report.extend(["REPRODUCCIÓN Y LÍMITES",f"Fingerprint completo: {result['fingerprint']}.",
        "Los checkpoints válidos y evaluaciones cacheadas se reutilizan sin repetir simulación. Una búsqueda interrumpida sin checkpoint de búsqueda completa debe reiniciarse; las búsquedas y evaluaciones por método ya guardadas se conservan.",
        "Todos los parámetros, semillas, versiones, fuentes, planes de repeticiones y resultados negativos quedan disponibles. Véanse configs/, characterization/, pilot/, confirmation/, figures/, tables/, logs/, checkpoints/ y validation/.",
        "Las decisiones por trabajo de las políticas principales se archivan en characterization/Sxxx/*_routing_decisions.csv.gz. Una fila corresponde a política/camino: el bit j−1 de fast_route_mask indica FAST=1/STD=0 para el trabajo j (índices desde 1). Se reconstruyeron las mismas referencias B y se verificaron rewards y desacuerdos frente a los diagnósticos congelados, sin modificar la clasificación.",
        "PSoft usa frecuencias bootstrap de selección condicionales a bancos finitos; no probabilidades bayesianas ni verdaderas de optimalidad. Los estimadores suavizados y la reutilización adaptativa de Green pueden tener sesgo respecto al gradiente hard-max original."])
    (out/"report.txt").write_text("\n\n".join(report)+"\n")
    readme=["REPRODUCCIÓN DEL EXPERIMENTO 009",f"Estado: {title}","",
        "Desde la raíz del proyecto:",
        "env GKSwstype=100 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=8 julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno 009probing_parameter_regimes.jl --resume",
        "julia --project=resultados/003probing_softmax_20261021/entorno export_parameter_regime_routing.jl",
        "O ejecutar python3 run_parameter_regimes.py, que incluye validación, archivo de routing y prueba de reanudación.",
        "","Fases individuales (añadir --resume si hay configuración existente):",
        "julia --project=resultados/003probing_softmax_20261021/entorno 009probing_parameter_regimes.jl --characterize --resume",
        "julia --project=resultados/003probing_softmax_20261021/entorno 009probing_parameter_regimes.jl --pilot --resume",
        "julia --project=resultados/003probing_softmax_20261021/entorno 009probing_parameter_regimes.jl --confirm --resume",
        "julia --project=resultados/003probing_softmax_20261021/entorno 009probing_parameter_regimes.jl --plots --report",
        "","Ampliación confirmatoria a los ocho escenarios, sin repetir los completos:",
        "env GKSwstype=100 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=8 julia --project=resultados/003probing_softmax_20261021/entorno 009probing_parameter_regimes.jl --confirm --all-scenarios --n-reps 20 --plots --report --resume",
        "","Prueba reducida:",
        "env GKSwstype=100 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=8 julia --project=resultados/003probing_softmax_20261021/entorno 009probing_parameter_regimes.jl --quick-test --resume",
        "","--scenario S000,S111 limita una fase explícita. --n-reps afecta las fases solicitadas.",
        "Cambiar n-paths, max-iters, reference-paths, epsilon u oracle-grid-n requiere otro --output para conservar las configuraciones originales.",
        "Caracterización futura con mayor precisión, sin cambiar los ocho modelos ni los criterios:",
        "env OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=8 julia --project=resultados/003probing_softmax_20261021/entorno 009probing_parameter_regimes.jl --characterize --reference-paths 30000 --output resultados/009probing_parameter_regimes_ref30000",
        "Dependencias: Julia 1.11.7, entorno anterior con Plots; Python numpy/pandas/scipy/matplotlib y tomli si Python<3.11.",
        "Pruebas de regresión: tests/validate_parameter_regimes.jl. Protocolo preespecificado: PROTOCOLO_009.txt.",
        "El tamaño de las pruebas reducidas no permite conclusiones científicas. El piloto es exploratorio; las conclusiones ajustadas entre escenarios usan solo confirmation."]
    (out/"README_reproduction.txt").write_text("\n".join(readme)+"\n")
    (out/"configs/analysis_environment.json").write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))

def main():
    out=Path(sys.argv[1]).resolve();root=tomllib.loads((out/"configs/configuration.toml").read_text())
    if "--classify" in sys.argv[2:]:classify(out,root["science"],root["fingerprint"])
    if "--analyze" in sys.argv[2:]:analyze(out,root["science"],root["fingerprint"])

if __name__=="__main__":main()
