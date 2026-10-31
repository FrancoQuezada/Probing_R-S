#!/usr/bin/env python3
"""Tables/figures for independent replications; never feeds back into a search.

The only tuning artifact is 005/selected_beta.toml, frozen before later stages.
Intervals use the replication as the independent unit. Same-bank max and
independently selected-policy evaluation are reported as distinct estimands.
"""
import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
import sys, json, hashlib, warnings, subprocess
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIGURE_CONTEXT = ""
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": .22, "font.size": 10})

def read_parts(out, suffix):
    files = sorted((out / "replications").glob(f"*_{suffix}.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p)
    adjusted = np.empty(len(p)); adjusted[order] = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted

def comparisons(paired):
    rows = []
    keys = ["n_paths", "analysis", "estimand", "method_a", "method_b"]
    for key, g in paired.groupby(keys, sort=False):
        x = g.difference.to_numpy(); n = len(x); se = x.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan
        t = x.mean()/se if se > 0 else (0. if x.mean() == 0 else np.inf)
        p = 2*stats.t.sf(abs(t), n-1) if n > 1 else np.nan
        crit = stats.t.ppf(.975, n-1)
        mc_var = np.mean(g.evaluation_paired_se**2)
        rows.append(dict(zip(keys, key), n_reps=n, mean_difference=x.mean(), sd_difference=x.std(ddof=1),
            se_across_reps=se, ci95_low=x.mean()-crit*se, ci95_high=x.mean()+crit*se, p_value=p,
            evaluation_mc_rms_se=np.sqrt(mc_var), optimization_sd_approx=np.sqrt(max(0., x.var(ddof=1)-mc_var)),
            wins=int(np.sum(x>0))))
    df = pd.DataFrame(rows)
    # One family includes every endpoint comparison and both reported estimands.
    df["p_holm"] = holm(df.p_value.fillna(1))
    crit = stats.t.ppf(1-.05/(2*len(df)), df.n_reps-1)
    df["simultaneous_ci95_low"] = df.mean_difference-crit*df.se_across_reps
    df["simultaneous_ci95_high"] = df.mean_difference+crit*df.se_across_reps
    return df

def save(fig, out, name):
    fig.text(.01,.008,FIGURE_CONTEXT,fontsize=7,va="bottom")
    fig.tight_layout(rect=(0,.055,1,1)); fig.savefig(out/f"{name}.png"); plt.close(fig)

def band(ax, frame, x="iteration", y="objective", label="", **kwargs):
    a = frame.groupby(x)[y].agg(["mean", "std"])
    line, = ax.plot(a.index, a["mean"], label=label, **kwargs)
    ax.fill_between(a.index.to_numpy(float), (a["mean"]-a["std"]).to_numpy(float),
                    (a["mean"]+a["std"]).to_numpy(float), color=line.get_color(), alpha=.13)

def draw_core(out, cfg, finals, traces, paired, coords, diag):
    stage = cfg["stage"]; names = list(finals.method.unique())
    fig, ax = plt.subplots()
    for name in names:
        g = traces[traces.method == name]
        if stage == 6:
            for n, gn in g.groupby("n_paths"):
                band(ax, gn, x="search_paths", label=f"{name}, n={n}")
        else:
            band(ax, g, label=name)
    ax.set(xlabel="Cumulative simulation replications" if stage == 6 else "Iteration",
           ylabel="Mean independent objective value J(z)", title="Direct hard-max evaluation; bands = ±1 SD across replications")
    ax.legend(fontsize=7, ncol=2); save(fig, out, "convergence")
    # Pair before aggregating: common evaluation noise largely cancels, while
    # the absolute objective bands above retain between-bank evaluation noise.
    base=traces[traces.method==names[0]][["rep","n_paths","iteration","objective"]].rename(columns={"objective":"baseline_objective"})
    delta=traces.merge(base,on=["rep","n_paths","iteration"])
    delta["difference"]=delta.objective-delta.baseline_objective
    fig,ax=plt.subplots()
    for name in names[1:]:
        g=delta[delta.method==name]
        if stage==6:
            for n,gn in g.groupby("n_paths"):band(ax,gn,x="search_paths",y="difference",label=f"{name}, n={n}")
        else:band(ax,g,y="difference",label=name)
    ax.axhline(0,color="black",lw=1)
    ax.set(xlabel="Cumulative simulation replications" if stage==6 else "Iteration",
        ylabel=f"Paired objective difference versus {names[0]}",title="Common evaluation CRN; bands = ±1 SD across replications")
    ax.legend(fontsize=7,ncol=2);save(fig,out,"paired_convergence")
    for analysis, g in finals.groupby("analysis"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, metric in zip(axes, ["objective_direct", "objective_holdout"]):
            labels = [f"{m}\nn={n}" for n in sorted(g.n_paths.unique()) for m in names]
            vals = [g[(g.n_paths == n)&(g.method == m)][metric] for n in sorted(g.n_paths.unique()) for m in names]
            ax.boxplot(vals, tick_labels=labels, showmeans=True)
            ax.tick_params(axis="x", labelrotation=70, labelsize=7)
            ax.set(ylabel="Objective", title=metric)
        save(fig, out, f"final_objectives_{analysis}")
        fig, ax = plt.subplots()
        gp = paired[(paired.analysis == analysis)&(paired.estimand == "holdout")]
        groups = list(gp.groupby(["n_paths", "method_a", "method_b"], sort=False))
        ax.boxplot([x.difference for _, x in groups], tick_labels=[f"{a} − {b}\nn={n}" for (n,a,b), _ in groups], showmeans=True)
        ax.axhline(0., color="black", lw=1); ax.tick_params(axis="x", labelrotation=55, labelsize=7)
        ax.set(ylabel="Paired holdout objective difference", title=analysis)
        save(fig, out, f"paired_differences_{analysis}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m in names:
        for ax, dim in zip(axes, ["z3", "z4"]):
            if stage == 6:
                for n, gn in coords[coords.method == m].groupby("n_paths"):
                    band(ax, gn, y=dim, label=f"{m}, n={n}")
            else:
                band(ax, coords[coords.method == m], y=dim, label=m)
            ax.set(xlabel="Iteration", ylabel=dim, ylim=(-.02,1.02))
    axes[0].legend(fontsize=7); save(fig, out, "probing_coordinates")
    surface_file = ROOT/"resultados/003probing_softmax_20261021/completo/probing_four_oracle_surface.csv"
    fig, ax = plt.subplots()
    if surface_file.exists():
        s = pd.read_csv(surface_file)
        surface = s.pivot(index="z4", columns="z3", values="J_oracle_crn")
        im = ax.pcolormesh(surface.columns, surface.index, surface, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label="Cached direct CRN hard-max surface (003, visual reference)")
    for m in names:
        g = finals[finals.method == m]
        ax.scatter(g.z3, g.z4, label=m, edgecolor="white", s=36)
    ax.set(xlabel="z3", ylabel="z4", xlim=(0,1), ylim=(0,1), title="Independent search endpoints")
    ax.legend(fontsize=7); save(fig, out, "oracle_endpoints")
    fig, ax = plt.subplots()
    for m in names:
        g = finals[finals.method == m]
        ax.scatter(g.search_seconds, g.objective_holdout, label=m, alpha=.65)
    ax.set(xlabel="Search wall seconds (concurrent replications)", ylabel="Independent selected-policy value")
    ax.legend(fontsize=7); save(fig, out, "time_vs_quality")
    fig, ax = plt.subplots()
    ax.boxplot([finals[finals.method == m].search_seconds for m in names], tick_labels=names, showmeans=True)
    ax.tick_params(axis="x", labelrotation=25); ax.set(ylabel="Search wall seconds", title="Evaluation costs reported separately")
    save(fig, out, "search_time")
    if stage == 6:
        for analysis, g in finals.groupby("analysis"):
            fig, ax = plt.subplots()
            for m in names:
                band(ax, g[g.method == m], x="n_paths", y="objective_holdout", label=m)
            ax.set(xlabel="N_PATHS", ylabel="Independent selected-policy value", title=analysis)
            ax.legend(fontsize=8); save(fig, out, f"quality_vs_n_paths_{analysis}")
        fig, ax = plt.subplots()
        for (analysis, a, b), g in paired[paired.estimand == "holdout"].groupby(["analysis","method_a","method_b"]):
            band(ax, g, x="n_paths", y="difference", label=f"{analysis}: {a} − {b}")
        ax.axhline(0,color="black",lw=1);ax.legend(fontsize=7)
        ax.set(xlabel="N_PATHS",ylabel="Paired holdout objective difference")
        save(fig,out,"paired_vs_n_paths")
    if stage in (7, 8):
        fig, ax = plt.subplots()
        if stage == 8:
            anchor = diag[diag.method.str.startswith("PSoft")]
            band(ax,anchor,y="K_eff",label="PSoft (PSoft bank)")
            band(ax,anchor,y="soft_K_eff",label="Softmax beta=0.1 (same bank)")
        else:
            for m,g in diag.groupby("method",sort=False):band(ax,g,y="K_eff",label=m)
        ax.set(xlabel="Iteration",ylabel="Effective number of policies");ax.legend(fontsize=8)
        save(fig,out,"effective_policies")
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for m,g in diag.groupby("method",sort=False):
            band(axes[0],g,y="hard_gap",label=m)
            band(axes[1],g,y="hard_cosine",label=m)
        axes[0].set(ylabel="||g − g_hard||, same point/history",xlabel="Iteration")
        axes[1].set(ylabel="Cosine with g_hard, same point/history",xlabel="Iteration")
        axes[0].legend(fontsize=7);save(fig,out,"gradient_comparisons")
        if stage==7:
            fig,ax=plt.subplots()
            for m,g in diag.groupby("method",sort=False):band(ax,g,y="ess",label=m)
            ax.set(xlabel="Iteration",ylabel="ESS across all likelihood ratios");ax.legend(fontsize=8)
            save(fig,out,"likelihood_ratio_ess")

def weight_analysis(out, cfg, diag):
    if cfg["stage"] not in (7,8):return []
    weights=read_parts(out,"weights")
    weights.groupby(["method","policy"])[["weight","soft_weight","value"]].mean().reset_index().to_csv(out/"mean_policy_weights.csv",index=False)
    notes=[]
    if cfg["stage"]==8:
        w=weights[weights.method.str.startswith("PSoft")].copy()
        # Reconstruct every tested beta on PSoft's SAME saved bank. This is
        # postprocessing of Z/G, with no resimulation or additional bootstrap.
        extra_columns={b:"soft_weight_beta_"+str(b).replace(".","p") for b in cfg["betas"] if b!=.1}
        for col in extra_columns.values():w[col]=np.nan
        same_bank=[]
        for (rep,k),g in w.groupby(["rep","iteration"],sort=False):
            values=g.value.to_numpy();G=g[["G3","G4"]].to_numpy();p=g.weight.to_numpy()
            gp=G.T@p-np.asarray(cfg["model"]["ALPHA"])
            for beta in cfg["betas"]:
                sw=np.exp(beta*(values-values.max()));sw/=sw.sum()
                if beta in extra_columns:w.loc[g.index,extra_columns[beta]]=sw
                gs=G.T@sw-np.asarray(cfg["model"]["ALPHA"])
                den=np.linalg.norm(gp)*np.linalg.norm(gs)
                same_bank.append(dict(rep=rep,iteration=k,beta=beta,l1_weight_distance=np.abs(p-sw).sum(),
                    K_eff_psoft=1/np.sum(p**2),K_eff_softmax=1/np.sum(sw**2),
                    gradient_difference=np.linalg.norm(gp-gs),cosine=np.dot(gp,gs)/den if den>1e-24 else np.nan))
        pd.DataFrame(same_bank).to_csv(out/"psoft_softmax_same_bank_comparisons.csv",index=False)
        w.to_csv(out/"psoft_same_bank_policy_weights.csv",index=False)
        for beta,g in pd.DataFrame(same_bank).groupby("beta"):
            notes.append(f"Mismos bancos PSoft, Softmax beta={beta:g}: L1 medio de pesos={g.l1_weight_distance.mean():.6f}; diferencia media de gradiente={g.gradient_difference.mean():.6f}; coseno medio={g.cosine.mean():.6f}. Comparación descriptiva sobre la trayectoria; las iteraciones no son independientes.")
        w["abs_gap"]=(w.weight-w.soft_weight).abs()
        distances=w.groupby(["rep","iteration"]).abs_gap.sum()
        notes.append(f"A. Distancia L1 media PSoft–Softmax beta=0.1 en los mismos bancos: {distances.mean():.6f} (máximo posible: 2).")
        top=w.groupby("policy")[["weight","soft_weight"]].mean()
        for col, name in [("weight","PSoft"),("soft_weight","Softmax beta=0.1")]:
            ids=top.nlargest(8,col)
            notes.append(f"B. Mayores pesos medios de {name} sobre los bancos de PSoft: "+", ".join(f"{i}: {r[col]:.4f}" for i,r in ids.iterrows())+".")
        a=diag[diag.method.str.startswith("PSoft")]
        notes.append(f"C. Comparación al mismo punto/banco: media ||gP−gFresh||={a.hard_gap.mean():.5f}; ||gP−gSoft||={a.soft_gap.mean():.5f}; cosenos medios={a.hard_cosine.mean():.5f}, {a.soft_cosine.mean():.5f}. Son diagnósticos descriptivos, no una prueba basada en iteraciones independientes.")
        # At iteration 1 every repetition is at the SAME deterministic Z0.
        fixed=a[a.iteration==1]
        vectors={"Fresh":fixed[["hard_g3","hard_g4"]].to_numpy(),"Softmax beta=0.1":fixed[["soft_g3","soft_g4"]].to_numpy(),"PSoft":fixed[["g3","g4"]].to_numpy()}
        gradients=[]
        for name in ("Fresh","Softmax beta=0.1"):
            diff=vectors["PSoft"]-vectors[name]
            for j,coordinate in enumerate(("z3","z4")):
                values=diff[:,j];se=values.std(ddof=1)/np.sqrt(len(values))
                crit=stats.t.ppf(.975,len(values)-1)
                pvalue=2*stats.t.sf(abs(values.mean()/se),len(values)-1) if se>0 else (1. if values.mean()==0 else 0.)
                gradients.append(dict(reference=name,coordinate=coordinate,n_reps=len(values),
                    mean_gradient_difference=values.mean(),se=se,ci95_low=values.mean()-crit*se,
                    ci95_high=values.mean()+crit*se,p_value=pvalue))
        gradients=pd.DataFrame(gradients);gradients["p_holm"]=holm(gradients.p_value)
        gradients.to_csv(out/"fixed_point_gradient_comparisons.csv",index=False)
        notes.append("C. Inferencia adicional sobre diferencias de gradientes en Z0: IC t por componente y ajuste Holm para las cuatro comparaciones PSoft–Fresh/Softmax, usando solo los 20 bancos independientes de k=1; fixed_point_gradient_comparisons.csv. Esto prueba diferencias de gradientes medios en Z0, no igualdad de todas sus distribuciones ni de los gradientes a lo largo de la búsqueda.")
        rows=[];rng=np.random.default_rng(2026102208)
        indices=rng.integers(0,len(fixed),size=(10000,len(fixed)))
        pv=np.var(vectors["PSoft"][indices],axis=1,ddof=1).sum(axis=1)
        for name,v in vectors.items():
            var=np.var(v,axis=0,ddof=1).sum()
            denom=np.var(v[indices],axis=1,ddof=1).sum(axis=1)
            ratio=np.divide(pv,denom,out=np.full_like(pv,np.nan),where=denom>0)
            rows.append(dict(method=name,n_reps=len(v),z3=cfg["model"]["Z0"][0],z4=cfg["model"]["Z0"][1],
                covariance_trace=var,psoft_variance_ratio=np.var(vectors["PSoft"],axis=0,ddof=1).sum()/var if var else np.nan,
                ratio_ci95_low=np.nanquantile(ratio,.025),ratio_ci95_high=np.nanquantile(ratio,.975),
                ratio_simultaneous95_low=np.nanquantile(ratio,.0125),ratio_simultaneous95_high=np.nanquantile(ratio,.9875)))
        pd.DataFrame(rows).to_csv(out/"fixed_point_gradient_variance.csv",index=False)
        notes.append("D. Varianza de gradientes: estimada entre repeticiones independientes en Z0, antes de actualizar; ver fixed_point_gradient_variance.csv. IC bootstrap marginales 95% y simultáneos aproximados 95% con Bonferroni para las dos referencias (Fresh y Softmax); se remuestrean las repeticiones emparejadas. Las fluctuaciones sobre trayectorias diferentes no estiman la varianza a punto fijo.")
        for k in [1,min(50,cfg["max_iters"]),cfg["max_iters"]]:
            x=w[w.iteration==k].groupby("policy")[["weight","soft_weight",*extra_columns.values()]].mean()
            ids=x.max(axis=1).nlargest(12).index
            fig,ax=plt.subplots();pos=np.arange(len(ids))
            series=[("weight",f"PSoft B={cfg['bootstrap_reps']}"),("soft_weight","Softmax beta=0.1, same bank")]
            series += [(col,f"Softmax beta={b:g}, same bank") for b,col in extra_columns.items()]
            width=.8/len(series)
            for j,(col,label) in enumerate(series):ax.bar(pos+(j-(len(series)-1)/2)*width,x.loc[ids,col],width,label=label)
            ax.set(xticks=pos,xticklabels=ids,xlabel="Policy index",ylabel="Mean weight",title=f"Iteration {k}, PSoft trajectory banks")
            ax.legend();save(fig,out,f"top_policy_weights_iteration_{k}")
        sens=read_parts(out,"bootstrap_sensitivity")
        s=sens.groupby("B")[["l1_error","squared_error","gradient_error","seconds"]].agg(["mean","std"])
        s.to_csv(out/"bootstrap_sensitivity_summary.csv")
        notes.append("Sensibilidad bootstrap: referencia independiente B=10000 y 10 remuestreos por B=100,300,1000 en las iteraciones prefijadas 1 y 50 (o final en prueba). Error respecto a esa referencia finita, sin nuevas simulaciones ni evaluate_policy().")
        notes.append("Error cuadrático medio de los pesos por B: "+", ".join(f"B={b}: {g.squared_error.mean():.6g}" for b,g in sens.groupby("B"))+".")
        anchors=a[["rep","iteration","gradient_norm"]]
        relative=sens[sens.B==300].merge(anchors,on=["rep","iteration"])
        relative["relative_gradient_error"]=relative.gradient_error/relative.gradient_norm.clip(lower=1e-12)
        notes.append(f"Para B=300: error L1 medio={relative.l1_error.mean():.5f}; error de gradiente medio={relative.gradient_error.mean():.5f}; error relativo medio respecto a la norma observada={relative.relative_gradient_error.mean():.5f}. La suficiencia depende de la precisión requerida; no se declara estable solo porque B=300 sea el valor inicial.")
        fig,ax=plt.subplots()
        band(ax,sens,x="B",y="l1_error",label="Error L1 respecto a B=10000")
        ax.set(xlabel="Bootstrap resamples B",ylabel="L1 weight error");ax.legend();save(fig,out,"bootstrap_sensitivity")
        ratio=a.bootstrap_seconds.sum()/a.iteration_seconds.sum()
        notes.append(f"H. Bootstrap durante búsqueda: {a.bootstrap_seconds.sum():.3f} s acumulados; {100*ratio:.2f}% del tiempo imputado a PSoft. Diagnóstico de sensibilidad y evaluaciones se contabilizan aparte.")
        notes.append("I. El remuestreo conjunto refleja incertidumbre de selección condicional a cada banco observado. Estas frecuencias no son probabilidades bayesianas ni las probabilidades verdaderas de optimalidad.")
        notes.append("J. Green + PSoft queda como hipótesis posterior. Su estudio requiere tratar el remuestreo de bancos adaptativos y ponderados; estos resultados Fresh no establecen su desempeño.")
    return notes

def main(directory):
    global FIGURE_CONTEXT
    out=Path(directory).resolve();cfg=tomllib.loads((out/"configuration.toml").read_text())
    FIGURE_CONTEXT=(f"{'QUICK TEST; ' if cfg['quick'] else ''}Stage {cfg['stage']:03d}; n={cfg['n_paths']}; reps={cfg['n_reps']}; jobs/path={cfg['jobs_per_path']}; base iterations={cfg['max_iters']}; budget={cfg['budget']}\n"
        f"Evaluation paths (trace/selection/final)={cfg['trace_paths']}/{cfg['selection_paths']}/{cfg['final_paths']}; ALPHA={cfg['model']['ALPHA']}; step=.035/sqrt(k); component clip=15")
    n_expected=cfg["n_reps"]*len(cfg["n_paths"])
    checks=list((out/"checkpoints").glob("*_complete.jls"))
    if len(checks)!=n_expected:raise RuntimeError(f"Incomplete stage: {len(checks)}/{n_expected}")
    finals=read_parts(out,"finals");traces=read_parts(out,"traces");paired=read_parts(out,"paired")
    coords=read_parts(out,"coordinates");diag=read_parts(out,"diagnostics");costs=read_parts(out,"costs")
    method_count=1+len(cfg["betas"])+(cfg["stage"]==8)
    expected_finals=n_expected*method_count*(2 if cfg["stage"]==6 else 1)
    if len(finals)!=expected_finals:raise RuntimeError(f"Missing endpoint records: {len(finals)}/{expected_finals}")
    if not (finals.groupby(["n_paths","analysis","method"]).size()==cfg["n_reps"]).all():raise RuntimeError("Incomplete endpoint groups")
    if not np.isfinite(finals[["objective_direct","objective_holdout","evaluation_se"]]).all().all():raise RuntimeError("Nonfinite final results")
    for frame,name in [(finals,"final_evaluations"),(traces,"convergence_traces"),(paired,"paired_replications"),
                       (coords,"coordinates"),(diag,"iteration_diagnostics"),(costs,"evaluation_costs")]:
        frame.to_csv(out/f"{name}.csv",index=False)
    comp=comparisons(paired);comp.to_csv(out/"paired_comparisons.csv",index=False)
    summary=finals.groupby(["n_paths","analysis","method"])[["objective_direct","objective_holdout","evaluation_se","search_seconds","bootstrap_seconds","total_variation","terminal_step_sd"]].agg(["mean","std"])
    summary.columns=["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/"summary.csv",index=False)
    draw_core(out,cfg,finals,traces,paired,coords,diag)
    notes=weight_analysis(out,cfg,diag)
    report=[f"ETAPA {cfg['stage']:03d} — "+("PRUEBA REDUCIDA COMPLETA; NO ES EL EXPERIMENTO FINAL" if cfg["quick"] else "EXPERIMENTO COMPLETO PARA LA CONFIGURACIÓN REGISTRADA"),
        f"{cfg['n_reps']} repeticiones independientes; n={cfg['n_paths']}; {cfg['max_iters']} iteraciones base; {cfg['jobs_per_path']} trabajos por trayectoria.",
        f"Objetivo preservado: J(z)=max_i E_z[Y_i]−ALPHA'z; 144 políticas ex ante; ALPHA={cfg['model']['ALPHA']}.",
        "Búsquedas emparejadas mediante primitivas CRN dentro de cada repetición; semillas independientes entre etapas y repeticiones. RNG bootstrap independiente del RNG de trayectorias.",
        f"Evaluación de convergencia: {cfg['trace_paths']} trayectorias comunes independientes de las búsquedas, en checkpoints prefijados. Bandas ±1 SD entre repeticiones, no intervalos de confianza.",
        f"Evaluación final: selección con {cfg['selection_paths']} trayectorias; evaluación con {cfg['final_paths']} trayectorias independientes, con CRN entre métodos.",
        "objective_direct = máximo muestral en evaluación; aproxima J y puede tener sesgo optimista de selección. objective_holdout = valor de una política elegida en otra muestra; evalúa esa política y puede estar por debajo de J si la selección no identifica la mejor. No se intercambian ambos estimandos.",
        "IC t del 95% sobre diferencias emparejadas entre repeticiones. Holm sobre TODA la familia de comparaciones de esta etapa, incluidos ambos estimandos y ambos presupuestos cuando corresponde. IC simultáneos conservadores Bonferroni también disponibles.",
        "evaluation_mc_rms_se cuantifica el error Monte Carlo de evaluación condicional a las políticas seleccionadas. optimization_sd_approx descuenta esa varianza de la variabilidad entre repeticiones; no elimina incertidumbre de selección ni sesgo de máximo muestral.",
        "Los tiempos son de pared durante repeticiones concurrentes en una misma máquina; incluyen contención y no son benchmarks aislados. No se incluye evaluación en el presupuesto de búsqueda.",
        "Los tiempos de búsqueda incluyen los diagnósticos comunes y el cálculo de gradientes por política incluso para el método hard. La implementación original permanece intacta; estos tiempos corresponden al ejecutor instrumentado, no a un benchmark mínimo de Fresh/Green.",
        f"Evaluaciones: {costs.evaluation_seconds.sum():.3f} s acumulados entre repeticiones/configuraciones; ver evaluation_costs.csv.",
        "",
        "RESULTADOS EMPAREJADOS (diferencia positiva favorece método A):"]
    for r in comp.itertuples():
        report.append(f"n={r.n_paths}, {r.analysis}, {r.estimand}: {r.method_a} − {r.method_b}: {r.mean_difference:.6f}, IC95% [{r.ci95_low:.6f}, {r.ci95_high:.6f}], pHolm={r.p_holm:.6g}; victorias {r.wins}/{r.n_reps}.")
    hold=comp[comp.estimand=="holdout"]
    significant_better=hold[(hold.p_holm<.05)&(hold.mean_difference>0)]
    significant_worse=hold[(hold.p_holm<.05)&(hold.mean_difference<0)]
    report.append(f"Conclusión de calidad holdout: {len(significant_better)} comparaciones favorecen significativamente al método A y {len(significant_worse)} al método B después de Holm; las restantes no permiten concluir diferencia. Las configuraciones concretas se identifican en la tabla anterior, sin extrapolar superioridad universal.")
    if cfg["stage"]==5:
        scores=finals[finals.beta>0].groupby("beta").objective_holdout.mean()
        selected=float(scores.idxmax())
        report.extend(["",f"Beta seleccionado exploratoriamente por mayor media holdout de 005: {selected:g}. Se fija antes de 006–008. No se presenta 005 como validación independiente del beta seleccionado.",
            "La comparación beta=0.10 contra Fresh estaba prefijada. Las demás comparaciones son simultáneas/exploratorias."])
        check=hold[hold.method_a=="Softmax beta=0.1"]
        if not check.empty:
            r=check.iloc[0]
            report.append("La mejora previa de beta=0.10 "+("persiste con evidencia estadística entre estas semillas." if r.p_holm<.05 and r.mean_difference>0 else "no se reproduce en este experimento; "+("la diferencia favorece a Fresh." if r.p_holm<.05 and r.mean_difference<0 else "no hay evidencia concluyente de mejora frente a Fresh.")))
        if not cfg["quick"]:
            content=f"complete = true\nquick = false\nselected_beta = {selected!r}\nn_reps = {cfg['n_reps']}\nselection_rule = \"highest mean independent selected-policy value; lower beta breaks exact ties\"\nsource_results_sha256 = \"{hashlib.sha256((out/'final_evaluations.csv').read_bytes()).hexdigest()}\"\n"
            target=out/"selected_beta.toml"
            if target.exists() and target.read_text()!=content:raise RuntimeError("Refusing to overwrite frozen beta")
            target.write_text(content)
    if cfg["stage"]==6:
        report.extend(["",f"Presupuesto fijo={cfg['budget']} trayectorias de búsqueda; iteraciones=floor(presupuesto/n), como mínimo una. Se ejecuta max(iteraciones base, iteraciones de presupuesto) y se reutilizan prefijos.",
            "La regla eta_k=BASE_STEP/sqrt(k) no cambia. Con presupuesto fijo cambian tanto el número de actualizaciones como los tamaños de paso; las diferencias no se atribuyen exclusivamente a n.",
            "Para concluir eficiencia se requiere una mejora de calidad a igual presupuesto o alcanzar calidad comparable con menor costo. Una curva más suave no basta."])
        # A fixed common numerical target, declared in the protocol before running.
        target=84.0
        rows=[]
        for (n,m),g in traces.groupby(["n_paths","method"]):
            a=g.groupby("iteration").agg(objective=("objective","mean"),paths=("search_paths","first"),seconds=("search_seconds","mean"))
            hit=a[a.objective>=target]
            rows.append(dict(n_paths=n,method=m,target=target,reached=not hit.empty,
                first_search_paths=hit.paths.iloc[0] if not hit.empty else np.nan,
                first_search_seconds=hit.seconds.iloc[0] if not hit.empty else np.nan))
        pd.DataFrame(rows).to_csv(out/"time_to_target.csv",index=False)
        report.append(f"Tiempo hasta objetivo prefijado J={target:g}: time_to_target.csv; cruce de medias en checkpoints, descriptivo y sujeto al error de evaluación y sesgo del máximo muestral.")
    if cfg["stage"]==7:
        report.extend(["","Green original conserva el argmax único sobre valores agregados. Green + Softmax agrega valores y TODOS los gradientes por política antes de calcular un único vector de pesos. No hay pesos distintos por banco.",
            "Los LR usan el mismo clipping logL ∈ [−35,35] que el original. ESS y contador de clipping permiten diagnosticar degeneración; el clipping puede introducir sesgo.",
            "Además, la reutilización histórica es adaptativa: el punto actual depende de bancos anteriores. No se afirma insesgadez del estimador Green condicionado a ese punto; el PDF señala esta fuente adicional de sesgo en la página 9.",
            "La prueba de beta grande verifica convergencia al gradiente hard cuando hay una política ganadora única. Los resultados de calidad, estabilidad y costo deben juzgarse conjuntamente."])
        report.append("Beneficio adicional de Green + Softmax: "+("hay configuraciones con mejor calidad holdout después de Holm; ver betas y diferencias en la tabla." if len(significant_better) else "no se detectó una mejora holdout después de Holm para los betas estudiados; esto no prueba equivalencia."))
    if cfg["stage"]==8:
        for reference,label in [("Fresh","E. PSoft frente a Fresh"),("Softmax beta=0.1","F. PSoft frente a Softmax beta=0.1")]:
            c=hold[hold.method_a.str.startswith("PSoft")&(hold.method_b==reference)]
            if not c.empty:
                r=c.iloc[0]
                result="mejora" if r.p_holm<.05 and r.mean_difference>0 else "deterioro" if r.p_holm<.05 else "sin diferencia concluyente"
                report.append(f"{label}: {result}; diferencia={r.mean_difference:.6f}, pHolm={r.p_holm:.6g}.")
    report += [""]+notes+["","INTERPRETACIÓN:",
        "Las comparaciones con pHolm<0.05 aportan evidencia de diferencias para el estimando indicado y esta configuración. Un resultado no significativo no demuestra equivalencia.",
        "La estabilidad se resume con variación total de z, dispersión de pasos finales y dispersión entre puntos finales; ver summary.csv. La suavidad por sí sola no establece superioridad.",
        "PSoft combina gradientes con frecuencias bootstrap de selección condicionadas a una muestra finita. La combinación no es necesariamente insesgada para el gradiente del hard-max original.",
        "Reproducción y continuación exacta: reproduce.txt. Los checkpoints completos se reutilizan; no se regeneran sus simulaciones."]
    (out/"report.txt").write_text("\n\n".join(report)+"\n")
    (out/"analysis_environment.json").write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,matplotlib=matplotlib.__version__,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    if not cfg["quick"]:
        subprocess.run([sys.executable,str(ROOT/"tests/validate_probing_results.py"),str(out)],check=True,cwd=ROOT)
    (out/"COMPLETE.json").write_text(json.dumps(dict(stage=cfg["stage"],quick=cfg["quick"],completed_replications=n_expected,expected_replications=n_expected),indent=2))
    print(f"Analysis complete: {out}; {n_expected} replications; {len(comp)} paired comparisons",flush=True)

if __name__=="__main__":main(sys.argv[1])
