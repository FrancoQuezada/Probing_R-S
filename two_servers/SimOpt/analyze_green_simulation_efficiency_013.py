"""Scientific analysis of six bank-size configurations; no search is run here."""
from pathlib import Path
import sys,json,hashlib,shutil,re,platform
try:import tomllib
except ModuleNotFoundError:import tomli as tomllib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analyze_anytime_012 import first_run,empirical_median,restricted_mean

ROOT=Path(__file__).resolve().parent
METHODS=[f'{base}-{n}' for base in ['Fresh','Green'] for n in [100,250,500]]
COLORS=dict(zip(METHODS,['#6baed6','#2171b5','black','#a1d99b','#31a354','#006d2c']))
STYLES={m:('-' if m.startswith('Fresh') else '--') for m in METHODS}
EPS=[.5,1.,2.]
PAIRINGS=[('Green-100','Fresh-100'),('Green-250','Fresh-250'),('Green-500','Fresh-500'),
          ('Green-100','Fresh-500'),('Green-250','Fresh-500'),('Green-100','Fresh-250')]

def write(out,name,df):
    df.to_csv(out/'tables'/f'{name}.csv',index=False)

def build_metrics(out,science):
    tr=pd.read_csv(out/'tables/objectives_by_iteration.csv');bu=pd.read_csv(out/'tables/search_budget_runtime.csv')
    dg=pd.read_csv(out/'tables/gradients_and_diagnostics.csv')
    keys=['rep','method','iteration'];assert not tr.duplicated(keys).any() and not bu.duplicated(keys).any()
    data=tr.merge(bu.drop(columns=['scenario','cohort','algorithm','N_PATHS']),on=keys,validate='one_to_one')
    data=data.merge(dg.drop(columns=['algorithm','N_PATHS']),on=keys,how='left',validate='one_to_one')
    K=science['max_iters'];reps=science['rep_ids'];assert len(data)==len(reps)*6*(K+1)
    rows=[];events=[];iterations=[];budget_rows=[];individual=[];coverage=[]
    for (rep,m),g in data.groupby(['rep','method'],sort=True):
        g=g.sort_values('iteration');n=int(g.N_PATHS.iloc[0]);v=g.objective.to_numpy();ref=float(g.reference_objective.iloc[0])
        assert np.array_equal(g.iteration,np.arange(K+1))
        assert np.array_equal(g.cumulative_trajectories,np.arange(K+1)*n)
        assert g.new_trajectories.iloc[1:].eq(n).all() and g.new_trajectories.iloc[0]==0
        assert g.new_policy_path_evaluations.sum()==144*n*K
        assert g.cumulative_search_seconds.is_monotonic_increasing
        assert np.allclose(g.search_seconds.cumsum(),g.cumulative_search_seconds)
        if m.startswith('Green'):
            assert np.array_equal(g.history_banks.iloc[1:],np.arange(1,K+1))
            assert np.array_equal(g.available_history_paths,np.arange(K+1)*n)
        else:assert g.history_banks.iloc[1:].eq(1).all()
        A=np.cumsum(v)/np.arange(1,K+2)
        Q=np.r_[v[0],np.cumsum(v)[:-1]/np.arange(1,K+1)]
        tv=np.r_[0.,np.cumsum(g.step_norm.iloc[1:].to_numpy())]
        clip=np.r_[0.,np.cumsum(g.clipping.iloc[1:].gt(0))/np.arange(1,K+1)]
        component=np.r_[0.,np.cumsum(g.clipping.iloc[1:])/(2*np.arange(1,K+1))]
        proj=np.r_[0.,np.cumsum(g.projection.iloc[1:].gt(1e-12))/np.arange(1,K+1)]
        for j,r in enumerate(g.to_dict('records')):
            r.update(accumulated_objective=A[j],accumulated_deficit=ref-A[j],budget_area_objective=Q[j],
                budget_area_deficit=ref-Q[j],total_variation=tv[j],clipping_fraction=clip[j],
                clipped_components_fraction=component[j],projection_fraction=proj[j])
            rows.append(r)
            if j in science['horizons']:
                iterations.append(dict(rep=rep,method=m,N_PATHS=n,iteration=j,budget=n*j,objective=v[j],gap_to_reference=ref-v[j],
                    accumulated_objective=A[j],accumulated_deficit=ref-A[j],budget_area_objective=Q[j],
                    total_variation=tv[j],clipping_fraction=clip[j],projection_fraction=proj[j],runtime=r['cumulative_search_seconds']))
            if n*j in science['budgets']:
                budget_rows.append(dict(rep=rep,method=m,N_PATHS=n,budget=n*j,iteration=j,objective=v[j],gap_to_reference=ref-v[j],
                    accumulated_objective=A[j],budget_area_objective=Q[j],runtime_to_budget=r['cumulative_search_seconds'],
                    total_variation=tv[j],clipping_fraction=clip[j],projection_fraction=proj[j]))
        individual.append(dict(rep=rep,method=m,algorithm=g.algorithm.iloc[0],N_PATHS=n,total_simulations=n*K,final_objective=v[-1],
            accumulated_objective=A[-1],accumulated_deficit=ref-A[-1],budget_area_objective=Q[-1],runtime=float(g.cumulative_search_seconds.iloc[-1]),
            final_z3=float(g.z3.iloc[-1]),final_z4=float(g.z4.iloc[-1]),clipping_fraction=clip[-1],clipped_components_fraction=component[-1],
            projection_fraction=proj[-1],total_variation=tv[-1],generation_seconds=g.generation_seconds.sum(),policy_seconds=g.policy_seconds.sum(),
            gradient_seconds=g.gradient_seconds.sum(),reuse_seconds=g.reuse_seconds.sum(min_count=1),reference_objective=ref))
        for B in science['budgets']:
            coverage.append(dict(rep=rep,method=m,budget=B,reached=B%n==0 and B//n<=K,
                iteration=B//n if B%n==0 and B//n<=K else np.nan,max_budget=n*K))
        for eps in EPS:
            for kind,w in [('first',1),('sustained5',5)]:
                start,confirm=first_run(v,ref,eps,w);ok=start is not None
                events.append(dict(rep=rep,method=m,N_PATHS=n,epsilon=eps,epsilon_name=str(eps),kind=kind,achieved=ok,
                    start_iteration=start,confirmation_iteration=confirm,start_budget=n*start if ok else np.nan,
                    confirmation_budget=n*confirm if ok else np.nan,start_runtime=g.cumulative_search_seconds.iloc[start] if ok else np.nan,
                    confirmation_runtime=g.cumulative_search_seconds.iloc[confirm] if ok else np.nan,censor_iteration=K,censor_budget=n*K,
                    observed_iteration=confirm if ok else K,status='attained' if ok else 'censored'))
    data=pd.DataFrame(rows);ev=pd.DataFrame(events);it=pd.DataFrame(iterations);bd=pd.DataFrame(budget_rows);ind=pd.DataFrame(individual)
    for name,df in [('all_iterations',data),('hitting_times',ev),('equal_iteration_comparisons',it),('equal_budget_comparisons',bd),
                    ('main_individual',ind),('budget_coverage',pd.DataFrame(coverage))]:write(out,name,df)
    plan=pd.read_csv(out/'configs/validation_plan.csv',dtype={'epsilon_name':str})
    q=ev.merge(plan,on=['rep','method','epsilon_name','kind'],suffixes=('','_plan'),validate='one_to_one')
    assert q.achieved.eq(q.achieved_plan).all()
    assert np.allclose(q.start_iteration,q.start_iteration_plan,equal_nan=True) and np.allclose(q.confirmation_iteration,q.confirmation_iteration_plan,equal_nan=True)
    targets=[];common_cap=min(100*min(science['sample_sizes']),K*min(science['sample_sizes']))
    for key,g in ev.groupby(['method','epsilon_name','kind']):
        m,en,kind=key;ok=g.achieved.to_numpy();times=g.observed_iteration.to_numpy();at_cap=ok&(g.confirmation_budget.fillna(np.inf).to_numpy()<=common_cap)
        targets.append(dict(method=m,epsilon_name=en,kind=kind,repetitions=len(g),successes=int(ok.sum()),success_fraction=ok.mean(),
            median_start_iteration=empirical_median(g.start_iteration.fillna(K),ok),median_confirmation_iteration=empirical_median(g.confirmation_iteration.fillna(K),ok),
            median_confirmation_budget=empirical_median(g.confirmation_budget.fillna(g.censor_budget),ok),
            restricted_mean_confirmation_iteration=restricted_mean(times,ok,K),
            common_budget_cap=common_cap,success_fraction_at_common_cap=at_cap.mean(),
            restricted_mean_budget_common_cap=restricted_mean(g.confirmation_budget.fillna(common_cap).to_numpy(),at_cap,common_cap)))
    ts=pd.DataFrame(targets);write(out,'target_summary',ts)
    def summarize(df,keys,exclude):
        numeric=df.select_dtypes(include=np.number).columns.difference(keys+exclude)
        r=df.groupby(keys)[numeric].agg(['mean','std']);r.columns=['_'.join(c) for c in r.columns];return r.reset_index()
    summary=summarize(ind,['method'],['rep']);write(out,'main_summary',summary)
    display=ind.groupby(['method','N_PATHS','total_simulations']).agg(
        final_objective=('final_objective','mean'),final_objective_sd=('final_objective','std'),
        accumulated_objective=('accumulated_objective','mean'),accumulated_objective_sd=('accumulated_objective','std'),
        runtime=('runtime','mean'),runtime_sd=('runtime','std'),final_z3=('final_z3','mean'),final_z4=('final_z4','mean'),
        clipping_fraction=('clipping_fraction','mean'),projection_fraction=('projection_fraction','mean')).reset_index()
    display=display.rename(columns={'accumulated_objective':f'accumulated_objective_A{K}'})
    write(out,'main_table',display)
    write(out,'equal_iteration_summary',summarize(it,['method','iteration'],['rep']))
    write(out,'equal_budget_summary',summarize(bd,['method','budget'],['rep']))
    pairs=[]
    for mode,df,key in [('equal_iterations',it,'iteration'),('equal_budget',bd,'budget')]:
        for rep in reps:
            for a,b in PAIRINGS:
                aa=df[(df.rep==rep)&(df.method==a)];bb=df[(df.rep==rep)&(df.method==b)]
                for value in sorted(set(aa[key])&set(bb[key])):
                    x=aa[aa[key]==value].iloc[0];y=bb[bb[key]==value].iloc[0]
                    pairs.append(dict(rep=rep,comparison=mode,checkpoint=value,method_a=a,method_b=b,
                        a_budget=x.budget,b_budget=y.budget,a_iteration=x.iteration,b_iteration=y.iteration,
                        objective_difference=x.objective-y.objective,accumulated_objective_difference=x.accumulated_objective-y.accumulated_objective,
                        budget_area_difference=x.budget_area_objective-y.budget_area_objective,
                        a_runtime=x.runtime if mode=='equal_iterations' else x.runtime_to_budget,
                        b_runtime=y.runtime if mode=='equal_iterations' else y.runtime_to_budget,
                        within_half_unit=x.objective-y.objective>=-.5))
    for rep in reps:
        for a,b in PAIRINGS[3:]:
            x=ind[(ind.rep==rep)&(ind.method==a)].iloc[0];y=ind[(ind.rep==rep)&(ind.method==b)].iloc[0]
            pairs.append(dict(rep=rep,comparison='final_different_budgets',checkpoint=K,method_a=a,method_b=b,
                a_budget=x.total_simulations,b_budget=y.total_simulations,a_iteration=K,b_iteration=K,
                objective_difference=x.final_objective-y.final_objective,accumulated_objective_difference=x.accumulated_objective-y.accumulated_objective,
                budget_area_difference=x.budget_area_objective-y.budget_area_objective,a_runtime=x.runtime,b_runtime=y.runtime,
                within_half_unit=x.final_objective-y.final_objective>=-.5))
    pa=pd.DataFrame(pairs);write(out,'paired_comparisons_individual',pa)
    write(out,'paired_comparisons_summary',summarize(pa,['comparison','checkpoint','method_a','method_b'],['rep']))
    losses=[]
    for rep in reps:
        vals=ind[ind.rep==rep].set_index('method')
        for n in [100,250]:
            for metric in ['final_objective','accumulated_objective']:
                fresh=vals.loc[f'Fresh-{n}',metric]-vals.loc['Fresh-500',metric]
                green=vals.loc[f'Green-{n}',metric]-vals.loc['Green-500',metric]
                losses.append(dict(rep=rep,N_PATHS=n,metric=metric,fresh_change=fresh,green_change=green,
                    green_minus_fresh_change=green-fresh,comparison='same updates; different simulation budgets'))
    lo=pd.DataFrame(losses);write(out,'sample_size_loss_paired',lo)
    cv=pd.read_csv(out/'tables/crossing_validation.csv',dtype={'epsilon_name':str})
    assert not cv.duplicated(['rep','method','epsilon_name','kind','iteration']).any()
    for p in plan[plan.achieved].itertuples():
        q=cv[(cv.rep==p.rep)&(cv.method==p.method)&(cv.kind==p.kind)&(cv.epsilon_name==p.epsilon_name)]
        assert sorted(q.iteration)==sorted(int(x) for x in str(p.validation_iterations).split(';'))
    vs=cv[cv.role=='candidate'].groupby(['rep','method','epsilon_name','kind']).agg(
        points=('iteration','size'),all_candidates_pass=('meets_threshold','all'),largest_deficit=('deficit','max'),largest_paired_mc_se=('paired_mc_se','max')).reset_index()
    write(out,'crossing_validation_summary',vs)
    bv=pd.read_csv(out/'tables/budget_validation.csv')
    last=bv.loc[bv.groupby(['rep','method']).budget.idxmax()]
    cross=[]
    if not science['quick']:
        for rep in reps:
            for a,b in PAIRINGS[3:]:
                x=last[(last.rep==rep)&(last.method==a)].iloc[0];y=last[(last.rep==rep)&(last.method==b)].iloc[0]
                assert x.evaluation_seed==y.evaluation_seed and x.evaluation_paths==y.evaluation_paths
                delta=x.objective-y.objective
                cross.append(dict(rep=rep,method_a=a,method_b=b,a_budget=x.budget,b_budget=y.budget,
                    difference=delta,within_half_unit=delta>=-.5,within_one_unit=delta>=-1,within_two_units=delta>=-2,
                    evaluation_paths=x.evaluation_paths,evaluation_seed=x.evaluation_seed,
                    scope='different final search budgets; same independent validation bank'))
        write(out,'cross_budget_final_validation',pd.DataFrame(cross))
    work=pd.read_csv(out/'tables/external_point_workload.csv')
    workload=[('new_algorithm_searches',2*350*len(reps)*K,2*350*len(reps)*K),
        ('archived_500_controls',0,2*500*len(reps)*K),('CRN_prefix_audit',850*len(reps)*K,0),
        ('kernel_equivalence_warmup',14400,0),('Green500_instrumented_replays',500*len(reps)*K,0),
        ('Green_LR_diagnostic_reconstruction',850*len(reps)*K,0),('main_evaluation_primitive_rebuild',1000*len(reps),0),
        ('validation_primitive_rebuild',science['validation_paths']*len(reps),0),
        ('final_selector_holdout_primitive_rebuild',0 if science['quick'] else 60000*len(reps),0)]
    ww=pd.DataFrame(workload,columns=['stage','generated_or_rebuilt_primitive_paths','logical_search_paths_charged'])
    ww['scope']='first completed execution; resumes may read checkpoints';write(out,'workload_by_phase',ww)
    write(out,'external_point_workload_summary',work.groupby(['phase','origin']).agg(
        points=('evaluation_paths','size'),materialized_paths=('materialized_paths','sum'),policy_path_evaluations=('policy_path_evaluations','sum')).reset_index())
    for rep in reps:
        folder=out/'tables'/f'rep{rep:02d}';folder.mkdir(exist_ok=True)
        for name,df in [('main',ind),('iterations',it),('budgets',bd),('hitting_times',ev),('all_iterations',data)]:df[df.rep==rep].to_csv(folder/f'{name}.csv',index=False)
    return dict(data=data,events=ev,iterations=it,budgets=bd,individual=ind,summary=summary,targets=ts,pairs=pa,losses=lo,validation=vs)

def scientific_answers(out,items,science):
    if science['quick']:return ['Prueba reducida: no se extraen conclusiones científicas de dos prefijos de seis actualizaciones.']
    ind=items['individual'];av=ind.groupby('method').mean(numeric_only=True);lo=items['losses']
    bv=pd.read_csv(out/'tables/budget_validation.csv');cross=pd.read_csv(out/'tables/cross_budget_final_validation.csv')
    gd=pd.read_csv(out/'tables/green_lr_diagnostics.csv')
    txt=['RESPUESTAS CIENTÍFICAS — cinco semillas; resultados descriptivos, sin afirmación de superioridad robusta.']
    for p,base in [('P1','Fresh'),('P2','Green')]:
        vals='; '.join(f'N={n}: J_final(traza)={av.loc[f"{base}-{n}","final_objective"]:.4f}, A(100)={av.loc[f"{base}-{n}","accumulated_objective"]:.4f}' for n in [100,250,500])
        txt.append(f'{p}. {base}, a 100 actualizaciones con presupuestos distintos: {vals}.')
    for n in [100,250]:
        rows=lo[(lo.N_PATHS==n)&(lo.metric=='final_objective')]
        txt.append(f'P3. N={n}: pérdida Green relativa a 500={rows.green_change.mean():.4f}, pérdida Fresh={rows.fresh_change.mean():.4f}; diferencia de cambios={rows.green_minus_fresh_change.mean():.4f}. Positivo significa menor deterioro de Green; se comparan presupuestos distintos.')
    for p,n in [('P4',100),('P5',250)]:
        q=cross[(cross.method_a==f'Green-{n}')&(cross.method_b=='Fresh-500')].sort_values('rep')
        txt.append(f'{p}. Green-{n} usa {100*n:,} frente a 50,000 caminos de Fresh-500. En el mismo banco independiente de 5000 caminos, diferencia final media={q.difference.mean():.4f}; dentro de .5 en {int(q.within_half_unit.sum())}/5 semillas. Diferencias por rep={q.difference.round(4).tolist()}.')
    for p,B in [('P6',10000),('P7',25000)]:
        q=bv[bv.budget==B];rank=q.groupby('method').objective.mean().sort_values(ascending=False)
        main=items['budgets'];rank0=main[main.budget==B].groupby('method').objective.mean().sort_values(ascending=False)
        txt.append(f'{p}. B={B:,}: mayor media en traza1000={rank0.index[0]} ({rank0.iloc[0]:.4f}); mayor media en revalidación5000={rank.index[0]} ({rank.iloc[0]:.4f}). Es una clasificación descriptiva; comparar diferencias emparejadas, no mezclar niveles absolutos entre bancos.')
    txt.append('P8. A B=10,000, los tamaños100/250/500 ejecutan100/40/20 actualizaciones. A T=100 consumen10000/25000/50000. El efecto de más actualizaciones es parte del procedimiento; no se atribuye solo a la varianza del banco.')
    stats=gd.groupby('method').agg(mean_ess_fraction=('ess_fraction','mean'),min_ess=('ess','min'),max_truncation=('truncation_fraction','max'))
    txt.append('P9. Green conserva exactamente todos los bancos. Fracción ESS media y mínimo ESS descriptivo: '+stats.round(6).to_string()+'. La reutilización verificada no demuestra por sí sola ahorro manteniendo calidad.')
    for n in [100,250,500]:
        a=av.loc[f'Green-{n}'];b=av.loc[f'Fresh-{n}']
        txt.append(f'P10. N={n}: variación total media Green={a.total_variation:.4f}/Fresh={b.total_variation:.4f}; pasos clipped Green={a.clipping_fraction:.3f}/Fresh={b.clipping_fraction:.3f}; proyección Green={a.projection_fraction:.3f}/Fresh={b.projection_fraction:.3f}.')
    for n in [100,250]:
        txt.append(f'P11. Tiempos primarios medios: Green-{n}={av.loc[f"Green-{n}","runtime"]:.4f}s, Fresh-{n}={av.loc[f"Fresh-{n}","runtime"]:.4f}s, Fresh-500={av.loc["Fresh-500","runtime"]:.4f}s. Evaluar calidad a la vez; sesiones distintas y OS compartido impiden tratarlos como benchmarks aislados.')
    q=cross[(cross.method_a=='Green-250')&(cross.method_b=='Fresh-500')]
    if q.within_half_unit.all():
        txt.append('P12. Green-250 fue comparable con Fresh-500 al finalizar usando la mitad de simulaciones en las cinco semillas según el margen exploratorio .5. Sin embargo, Fresh-250 también debe compararse: una calidad similar con N=250 no identifica una ventaja específica de la historia de Green.')
    else:txt.append('P12. La comparabilidad final de Green-250 con Fresh-500 no fue uniforme en las cinco semillas; conservar los casos desfavorables.')
    ff=out/'tables/final_selected_holdout.csv'
    if ff.exists():
        f=pd.read_csv(ff);summary=f.groupby('method').objective_holdout.agg(['mean','std'])
        txt += ['Comprobación suplementaria de política seleccionada con holdout50000 (estimando distinto):',summary.round(6).to_string()]
        for n in [100,250]:
            a=f[f.method==f'Green-{n}'].set_index('rep').objective_holdout;b=f[f.method=='Fresh-500'].set_index('rep').objective_holdout
            txt.append(f'Green-{n} frente a Fresh-500: dentro de .5 en {int((a-b).ge(-.5).sum())}/5; diferencias={(a-b).round(4).tolist()}.')
        txt.append(f'Fresh-250: holdout medio={summary.loc["Fresh-250","mean"]:.6f}; Green-250={summary.loc["Green-250","mean"]:.6f}. Por ello, no se demuestra una superioridad propia de Green frente a Fresh con el mismo banco.')
    txt.append('La conclusión distingue calidad final, transitorio/acumulación, nuevas simulaciones, costo CPU y estabilidad. Los datos desfavorables no se eliminan ni se ajustan parámetros para favorecer un método.')
    return txt

def save(fig,out,name):
    fig.savefig(out/'figures'/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(out/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def lines(out,data,name,x='iteration',y='objective',mean=False,band=False,title='',ylabel=None,step=False):
    fig,ax=plt.subplots(figsize=(10,6))
    for m in METHODS:
        q=data[data.method==m].sort_values('iteration')
        if mean:
            q=q.groupby('iteration').agg(x=(x,'mean'),value=(y,'mean'),sd=(y,'std')).reset_index();xx=q.x.to_numpy();yy=q.value.to_numpy();sd=q.sd.to_numpy()
        else:xx=q[x].to_numpy();yy=q[y].to_numpy();sd=np.zeros(len(q))
        kwargs=dict(color=COLORS[m],ls=STYLES[m],label=m,lw=1.8)
        if step:ax.step(xx,yy,where='post',**kwargs)
        else:ax.plot(xx,yy,**kwargs)
        if band:ax.fill_between(xx,yy-sd,yy+sd,color=COLORS[m],alpha=.12,step='post' if step else None)
    xlabel={'iteration':'Iteration','cumulative_trajectories':'Cumulative new search trajectories','cumulative_search_seconds':'Cumulative measured search time (seconds)'}[x]
    ax.set(xlabel=xlabel,ylabel=ylabel or 'Independent objective estimate J(z)',title=title);ax.grid(alpha=.2)
    ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=9);save(fig,out,name)

def plots(out,items,science):
    d=items['data'];reps=sorted(d.rep.unique());K=science['max_iters']
    for rep in reps:
        r=d[d.rep==rep]
        lines(out,r,f'convergence_iteration_rep{rep:02d}',title=f'I0252: convergence by iteration, rep{rep:02d}')
        lines(out,r,f'convergence_budget_rep{rep:02d}',x='cumulative_trajectories',step=True,title=f'I0252: convergence by new simulation budget, rep{rep:02d}')
        fig,axes=plt.subplots(2,1,figsize=(11,9),sharex=True)
        for ax,dim in zip(axes,['z3','z4']):
            for m in METHODS:
                q=r[r.method==m].sort_values('iteration');ax.plot(q.iteration,q[dim],color=COLORS[m],ls=STYLES[m],label=m)
            ax.set(ylabel=f'Class-{dim[-1]} probing intensity {dim}',ylim=(-.025,1.025));ax.grid(alpha=.2)
        axes[0].set_title(f'I0252: probing trajectories, rep{rep:02d}');axes[-1].set_xlabel('Iteration');axes[0].legend(loc='upper left',bbox_to_anchor=(1,1));save(fig,out,f'probing_trajectories_rep{rep:02d}')
        # Individual views preserve unfavorable repetitions without overlapping methods.
        for m in METHODS:
            q=r[r.method==m].sort_values('iteration');fig,ax=plt.subplots(figsize=(9,5))
            ax.plot(q.iteration,q.objective,color=COLORS[m],label=m);ax.axhline(q.reference_objective.iloc[0],color='grey',ls=':',label='Frozen reference, same CRN bank')
            ax.set(xlabel='Iteration',ylabel='Independent objective estimate J(z)',title=f'I0252: {m}, rep{rep:02d}');ax.grid(alpha=.2);ax.legend();save(fig,out,f'{m}_convergence_rep{rep:02d}')
    lines(out,d,'average_convergence_iteration',mean=True,band=True,title='I0252: mean convergence ± between-repetition SD')
    lines(out,d,'average_convergence_budget',x='cumulative_trajectories',mean=True,band=True,step=True,title='I0252: mean convergence by simulation budget ± SD')
    lines(out,d,'accumulated_objective_iteration',y='accumulated_objective',mean=True,band=True,ylabel='Native-update average A(T)',title='I0252: accumulated quality; equal updates, unequal budgets')
    lines(out,d,'accumulated_objective_native_budget',x='cumulative_trajectories',y='accumulated_objective',mean=True,step=True,ylabel='Native-update average A(T)',title='I0252: native A(T) at actual search budget (different update grids)')
    lines(out,d,'budget_area_quality',x='cumulative_trajectories',y='budget_area_objective',mean=True,band=True,step=True,ylabel='Budget-area average Q(B)',title='I0252: last-available-decision quality integrated over new simulation budget')
    lines(out,d,'accumulated_deficit_iteration',y='accumulated_deficit',mean=True,band=True,ylabel='Reference minus A(T)',title='I0252: accumulated deficit to approximate reference (negative values retained)')
    lines(out,d[d.rep==1],'runtime_objective_rep01',x='cumulative_search_seconds',title='I0252: objective versus measured search time, predefined rep01')
    fig,axes=plt.subplots(2,3,figsize=(16,9));axes=axes.ravel()
    for ax,rep in zip(axes,reps):
        for m in METHODS:
            q=d[(d.rep==rep)&(d.method==m)].sort_values('iteration');ax.step(q.cumulative_trajectories,q.objective,where='post',color=COLORS[m],ls=STYLES[m],label=m)
        ax.set(title=f'rep{rep:02d}',xlabel='New search trajectories',ylabel='J(z)');ax.grid(alpha=.2)
    for ax in axes[len(reps):]:ax.set_visible(False)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='center left',bbox_to_anchor=(.99,.5));fig.suptitle('I0252: every independent repetition, simulation-budget comparison');fig.tight_layout();save(fig,out,'individual_repetitions_budget')
    bd=items['budgets'];bs=sorted(bd.budget.unique());fig,axes=plt.subplots(1,len(bs),figsize=(6*len(bs),5.5),squeeze=False)
    for ax,B in zip(axes[0],bs):
        ms=[m for m in METHODS if m in set(bd[bd.budget==B].method)]
        for j,m in enumerate(ms):
            q=bd[(bd.method==m)&(bd.budget==B)];ax.scatter(j+np.linspace(-.14,.14,len(q)),q.objective,color=COLORS[m]);ax.errorbar(j,q.objective.mean(),yerr=q.objective.std(),fmt='_',color=COLORS[m],capsize=5)
        ax.set_xticks(range(len(ms)),ms,rotation=30,ha='right');ax.set(title=f'B={B:,}: reached configurations only',ylabel='Independent objective J(z)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('I0252: exact equal-budget decisions; individual runs and mean ± SD');fig.tight_layout();save(fig,out,'budget_normalized_performance')
    e=items['events'];fig,axes=plt.subplots(2,3,figsize=(17,9))
    for ri,kind in enumerate(['first','sustained5']):
        for ci,eps in enumerate(EPS):
            ax=axes[ri,ci]
            for j,m in enumerate(METHODS):
                q=e[(e.kind==kind)&(e.epsilon==eps)&(e.method==m)].sort_values('rep');ok=q.achieved.to_numpy();x=j+np.linspace(-.15,.15,len(q))
                ax.scatter(x[ok],q.confirmation_budget.to_numpy()[ok],color=COLORS[m],s=32)
                ax.scatter(x[~ok],q.censor_budget.to_numpy()[~ok],color='red',marker='x',s=48)
            ax.set_xticks(range(6),METHODS,rotation=35,ha='right',fontsize=8);ax.set(title=f'{kind}; ε={eps}',ylabel='New trajectories at confirmation / censoring');ax.grid(axis='y',alpha=.2)
    fig.suptitle('I0252: budget to target; red crosses are censored, not arrivals');fig.tight_layout();save(fig,out,'time_to_target_budget')
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for ri,kind in enumerate(['first','sustained5']):
        for ci,eps in enumerate(EPS):
            ax=axes[ri,ci]
            for m in METHODS:
                q=e[(e.kind==kind)&(e.epsilon==eps)&(e.method==m)];n=int(q.N_PATHS.iloc[0]);grid=np.arange(K+1)*n
                y=[np.mean(q.achieved&(q.confirmation_budget<=B)) for B in grid];ax.step(grid,y,where='post',color=COLORS[m],ls=STYLES[m],label=m)
            ax.set(title=f'{kind}; ε={eps}',xlabel='New search trajectories',ylabel='Fraction attaining target',ylim=(-.03,1.03));ax.grid(alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='center left',bbox_to_anchor=(.99,.5));fig.suptitle('I0252: empirical attainment probability; each curve ends at its own budget horizon');fig.tight_layout();save(fig,out,'target_attainment_budget')
    ind=items['individual'];fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,metric,title in [(axes[0],'runtime','Primary search times (012 control clocks retained)'),(axes[1],'total_variation','Total path variation over 100 updates')]:
        for j,m in enumerate(METHODS):
            q=ind[ind.method==m][metric];ax.scatter(j+np.linspace(-.14,.14,len(q)),q,color=COLORS[m]);ax.scatter(j,q.mean(),marker='_',s=180,color=COLORS[m])
        ax.set_xticks(range(6),METHODS,rotation=30,ha='right');ax.set(title=title,ylabel='Seconds' if metric=='runtime' else 'Sum of step norms');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,out,'runtime_and_path_variation')
    phases=['generation_seconds','materialization_seconds','policy_seconds','gradient_seconds','update_seconds','other_seconds']
    means=d.groupby(['rep','method'])[phases].sum().groupby('method').mean().reindex(METHODS);fig,ax=plt.subplots(figsize=(12,6));bottom=np.zeros(6)
    for p in phases:
        v=means[p].to_numpy();ax.bar(np.arange(6),v,bottom=bottom,label=p.replace('_seconds',''));bottom+=v
    ax.set_xticks(range(6),METHODS,rotation=30,ha='right');ax.set(ylabel='Mean measured search seconds',title='I0252: search-time phases; gradient includes history reuse once');ax.legend(loc='upper left',bbox_to_anchor=(1,1));save(fig,out,'runtime_phase_breakdown')
    fig,ax=plt.subplots(figsize=(12,6));x=np.arange(6)
    for shift,col,label in [(-.24,'clipped_components_fraction','Clipped gradient components'),(0,'clipping_fraction','Updates with clipping'),(.24,'projection_fraction','Projected updates')]:
        v=ind.groupby('method')[col].mean().reindex(METHODS);ax.bar(x+shift,v,width=.24,label=label)
    ax.set_xticks(x,METHODS,rotation=30,ha='right');ax.set(ylim=(0,1.05),ylabel='Mean fraction',title='I0252: clipping and projection across every repetition');ax.legend();save(fig,out,'clipping_projection')
    gd=pd.read_csv(out/'tables/green_lr_diagnostics.csv')
    for metric,name,ylabel,log in [('ess','green_ess','Descriptive pooled LR ESS',True),('ess_fraction','green_ess_fraction','ESS / available historical paths',False),
                                 ('truncation_fraction','green_lr_truncation','Fraction of logLR values truncated at ±35',False),('historical_lr_mass','green_historical_mass','Relative LR mass of older banks (descriptive)',False)]:
        fig,ax=plt.subplots(figsize=(10,6))
        for m in METHODS[3:]:
            for rep in reps:
                q=gd[(gd.method==m)&(gd.rep==rep)].sort_values('iteration');ax.plot(q.iteration,q[metric],color=COLORS[m],alpha=.3,lw=.8)
            q=gd[gd.method==m].groupby('iteration')[metric].mean();ax.plot(q.index,q,color=COLORS[m],label=m,lw=2)
        if log:ax.set_yscale('log')
        ax.set(xlabel='Gradient iteration (pre-update decision)',ylabel=ylabel,title='I0252: individual runs and mean; ESS is not exact gradient precision');ax.grid(alpha=.2);ax.legend();save(fig,out,name)
    for rep in reps:
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for ax,m in zip(axes,METHODS[3:]):
            q=gd[(gd.rep==rep)&(gd.method==m)].sort_values('iteration');x=q.iteration.to_numpy()
            ax.fill_between(x,q.loglr_p01.to_numpy(),q.loglr_p99.to_numpy(),color=COLORS[m],alpha=.25,label='Within-run 1%–99% logLR')
            ax.plot(x,q.loglr_median,color=COLORS[m],label='Median logLR');ax.axhline(35,color='grey',ls=':');ax.axhline(-35,color='grey',ls=':')
            ax.set(title=m,xlabel='Gradient iteration',ylabel='Raw log likelihood ratio');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'I0252: unmodified LR dispersion, rep{rep:02d}; dotted lines are original truncation limits');fig.tight_layout();save(fig,out,f'green_loglr_dispersion_rep{rep:02d}')
    h=pd.read_csv(out/'tables/green_loglr_histograms.csv');hs=science['horizons'];fig,axes=plt.subplots(3,len(hs),figsize=(6*len(hs),12),squeeze=False)
    for ri,m in enumerate(METHODS[3:]):
        for ci,t in enumerate(hs):
            q=h[(h.rep==1)&(h.method==m)&(h.iteration==t)].sort_values('bin');ax=axes[ri,ci]
            ax.bar(q.bin,q.fraction,color=COLORS[m]);ax.set(title=f'{m}, t={t}',xlabel='Fixed raw-logLR interval index',ylabel='Fraction of historical/current paths');ax.grid(axis='y',alpha=.2)
    fig.suptitle('I0252: predefined rep01 logLR histograms (bin edges in CSV; unequal-width frequency bins)');fig.tight_layout();save(fig,out,'green_lr_distributions')
    prof=pd.read_csv(out/'tables/control_history_profiling.csv');fig,ax=plt.subplots(figsize=(10,6))
    for j,m in enumerate(METHODS[3:]):
        vals=prof.groupby('rep').reuse_seconds.sum() if m=='Green-500' else ind[ind.method==m].set_index('rep').reuse_seconds
        ax.scatter(j+np.linspace(-.12,.12,len(vals)),vals,color=COLORS[m]);ax.scatter(j,vals.mean(),marker='_',s=200,color=COLORS[m])
    ax.set_xticks(range(3),METHODS[3:]);ax.set(ylabel='Historical-estimator seconds',title='I0252: history computation (Green-500 from separate exact profiling replays)');ax.grid(axis='y',alpha=.2);save(fig,out,'green_history_runtime')
    f=out/'tables/final_selected_holdout.csv'
    if f.exists():
        final=pd.read_csv(f);write(out,'final_selected_holdout_summary',final.groupby('method').agg(mean=('objective_holdout','mean'),sd=('objective_holdout','std')).reset_index())
        fig,ax=plt.subplots(figsize=(12,6))
        for j,m in enumerate(METHODS):
            q=final[final.method==m].objective_holdout;ax.scatter(j+np.linspace(-.14,.14,len(q)),q,color=COLORS[m]);ax.scatter(j,q.mean(),marker='_',s=200,color=COLORS[m])
        ax.set_xticks(range(6),METHODS,rotation=30,ha='right');ax.set(ylabel='Independent selected-policy value',title='I0252: final 10000-path selector / 50000-path holdout; unequal search budgets');ax.grid(axis='y',alpha=.2);save(fig,out,'final_selected_holdout')
    bv=pd.read_csv(out/'tables/budget_validation_pairs.csv');fig,axes=plt.subplots(1,len(bs),figsize=(7*len(bs),6),squeeze=False)
    for ax,B in zip(axes[0],bs):
        chosen=[(a,b) for a,b in PAIRINGS if ((bv.budget==B)&(bv.method_a==a)&(bv.method_b==b)).any()]
        for j,(a,b) in enumerate(chosen):
            q=bv[(bv.budget==B)&(bv.method_a==a)&(bv.method_b==b)].sort_values('rep');xx=j+np.linspace(-.15,.15,len(q))
            ax.errorbar(xx,q.difference,yerr=q.paired_mc_se,fmt='o',color=COLORS[a],capsize=3)
        ax.axhline(0,color='black',lw=.8);ax.axhline(-.5,color='grey',ls=':');ax.set_xticks(range(len(chosen)),[f'{a} minus {b}' for a,b in chosen],rotation=40,ha='right',fontsize=8)
        ax.set(title=f'B={B:,}',ylabel='Paired independent objective difference ± MC SE');ax.grid(axis='y',alpha=.2)
    fig.suptitle('I0252: independent equal-budget recheck (5000 paths; 100 in quick test)');fig.tight_layout();save(fig,out,'equal_budget_independent_validation')

def report(out,items,config):
    s=config['science'];d=items['data'];ind=items['individual'];bd=items['budgets'];it=items['iterations'];pa=items['pairs'];lo=items['losses'];ev=items['events']
    text=['I0252 — eficiencia de simulación de Fresh y Green']+scientific_answers(out,items,s)+['',
        f'{4*len(s["rep_ids"])} búsquedas nuevas (Fresh/Green con N=100/250) y {2*len(s["rep_ids"])} controles N=500 reutilizados. La prueba reducida es un prefijo separado y no añade repeticiones.',
        f'Configuración: {s["max_iters"]} actualizaciones, 45 trabajos, 144 políticas, z0=(.12,.10), paso .035/sqrt(t), clipping ±15, proyección [0,1]^2.',
        'Se lee exactamente el modelo de 011; BETA de señal=4.0, alpha y parámetros operacionales permanecen intactos. No se usa Softmax.',
        'Objetivo J(z)=max_i E_z[Y_i]-alpha z, política elegida ex ante. No es E[max_i Y_i].',
        'Calidad principal: máximo de medias sobre el banco de traza de 1000 caminos independiente del entrenamiento, mismo banco y referencia (.70,.70) dentro de cada repetición.',
        'La referencia es competitiva y aproximada; no es un óptimo global certificado. Los valores máximos de medias pueden tener sesgo por selección; las SE de revalidación son descriptivas y condicionales a políticas elegidas.',
        'CRN: se verificó cada campo de los prefijos 100/250/500 en cada repetición/iteración y coincidencia con las semillas originales. Distintas decisiones materializan señales distintas.',
        'A igual presupuesto y distinto N cambian tanto el número de actualizaciones como el conjunto global de primitivas utilizado: el anidamiento es por iteración, no una afirmación de igualdad de todos los caminos a igual presupuesto.',
        'Las cinco repeticiones usan semillas distintas; dentro de cada repetición se comparte la secuencia de semillas entre las seis configuraciones.',
        'Solo se cuenta N*t como presupuesto de búsqueda. LR, historia, auditorías, replays y evaluación externa se guardan por separado.',
        'A(T) promedia J(z0),...,J(zT) sobre actualizaciones. Su comparación a T=100 entre tamaños usa presupuestos diferentes y no demuestra eficiencia de simulación.',
        'Q(B)=integral_0^B J(z_floor(b/N))db/B usa la última decisión disponible, con Q(0)=J(z0); no inventa objetivos intermedios ni es recompensa operacional.',
        'Comparabilidad exploratoria: diferencia al menos -.5 unidades; sensibilidad .5/1/2. No es una prueba de no inferioridad estadística.',
        'Tiempos: búsquedas nuevas secuenciales, BLAS1, mismas funciones y alcance de reloj que 012. Clocks primarios N=500 son exactamente los de 012.',
        'Cinco replays instrumentados Green-500 verifican todas las decisiones/gradientes y recuperan el costo histórico no desglosado en 012; sus tiempos se presentan aparte, no sustituyen controles ni agregan repeticiones/búsqueda.',
        'reuse_seconds es parte de gradient_seconds: no se suma dos veces. No se incluyen calentamiento, auditorías, evaluaciones externas, LR adicionales, persistencia o gráficos. Sesiones diferentes/OS compartido: comparación de reloj descriptiva.',
        '', 'Tabla principal: medias y desviaciones estándar entre repeticiones; presupuesto máximo diferente por tamaño.',items['summary'].to_string(index=False,float_format=lambda v:f'{v:.6f}'),
        '', 'P1. Fresh al reducir el tamaño del banco (mismas 100 actualizaciones, presupuestos desiguales):',
        ind[ind.algorithm=='Fresh'][['rep','method','total_simulations','final_objective','accumulated_objective','runtime','clipping_fraction','projection_fraction']].to_string(index=False),
        '', 'P2. Green al reducir el tamaño del banco:',ind[ind.algorithm=='Green'][['rep','method','total_simulations','final_objective','accumulated_objective','runtime','clipping_fraction','projection_fraction']].to_string(index=False),
        '', 'P3. Pérdida relativa a N=500: (Green_N-Green_500)-(Fresh_N-Fresh_500); positivo implica menor deterioro de Green en esa métrica. Actualizaciones iguales, presupuestos distintos.',
        lo.to_string(index=False),
        '', 'P4/P5. Green-100/250 frente a Fresh-500: resultados finales usan 10000/25000 frente a 50000 trayectorias; estas comparaciones de calidad/costo NO son comparaciones a igual presupuesto.',
        pa[pa.comparison=='final_different_budgets'].to_string(index=False),
        '', 'P6/P7. Comparación PRINCIPAL a igual presupuesto, sin extrapolar configuraciones no participantes:']
    for B in s['budgets']:
        g=bd[bd.budget==B];av=g.groupby('method').objective.agg(['mean','std']).sort_values('mean',ascending=False)
        text += [f'B={B}: clasificación descriptiva en el banco de traza n=1000:',av.to_string(),g.to_string(index=False)]
    validated=pd.read_csv(out/'tables/budget_validation.csv');vp=pd.read_csv(out/'tables/budget_validation_pairs.csv')
    text += ['', 'Revalidación independiente de esos puntos con banco común n=5000 (n=100 en prueba):',validated.to_string(index=False),
        'Diferencias emparejadas Green frente a Fresh en ese banco; las SE son MC condicionales, no SD entre repeticiones:']
    selected=[]
    for a,b in PAIRINGS:
        q=vp[(vp.method_a==a)&(vp.method_b==b)]
        if len(q):selected.append(q)
    vsel=pd.concat(selected,ignore_index=True);text.append(vsel.to_string(index=False))
    for (B,a,b),g in vsel.groupby(['budget','method_a','method_b']):
        text.append(f'B={B}: {a} menos {b}: media {g.difference.mean():.6f}, SD entre semillas {g.difference.std():.6f}; diferencias por rep={g.sort_values("rep").difference.tolist()}; fracción dentro de .5={g.difference.ge(-.5).mean():.2f}.')
    text += ['', 'P8. Igual número de iteraciones (20,50,100) usa N*t caminos distintos según tamaño; igual presupuesto usa t=B/N y distinto número de actualizaciones. No es un efecto aislado de varianza muestral.',
        it.to_string(index=False),
        '', 'P9. ESS=(sum L)^2/sum L^2 sobre todos los N*t caminos históricos/actuales, L=exp(clamp(logLR,-35,35)). No es precisión exacta del gradiente.',
        'El coeficiente de cada banco en el estimador es 1/t; la masa LR relativa es solo descriptiva. Todas las iteraciones reconstruidas coinciden EXACTAMENTE en ESS y recortes con los estimadores ejecutados.',
        '', 'P10. Estabilidad: variación total sum||delta z||, clipping por componentes/pasos, proyección y diagnósticos LR completos. Se conservan todas las repeticiones desfavorables.',
        ind[['rep','method','total_variation','clipping_fraction','projection_fraction','final_z3','final_z4']].to_string(index=False),
        '', 'P11. Ahorro de simulaciones y costo CPU se juzgan por separado. Consultar paired_comparisons_summary.csv y tiempos individuales; no inferir ahorro de CPU de la mera reutilización.',
        ind[['rep','method','total_simulations','runtime','gradient_seconds','reuse_seconds']].to_string(index=False),
        '', 'P12. Consistencia: cinco semillas emparejadas, no una inferencia robusta de superioridad. Las tablas individuales y gráficos incluyen cada repetición.',
        'Llegadas: primera y ventana de cinco decisiones. Se almacenan inicio/confirmación; censura mantiene tau faltante. La confirmación cuesta cuatro actualizaciones extra, de diferente presupuesto según N.',
        items['targets'].to_string(index=False),
        'La revalidación independiente solo examina candidatos fijados antes de sus nuevos valores; no genera nuevos tiempos completos de llegada.',
        'Bandas: SD entre repeticiones, no intervalos de confianza. No se rectifican déficits negativos ni se imputan llegadas a fallos.']
    gd=pd.read_csv(out/'tables/green_lr_diagnostics.csv');text += ['', 'Resumen de diagnóstico Green por repetición:',gd.groupby(['rep','method']).agg(min_ess=('ess','min'),median_ess=('ess','median'),minimum_ess_fraction=('ess_fraction','min'),max_truncation_fraction=('truncation_fraction','max')).to_string()]
    if (out/'tables/final_selected_holdout.csv').exists():
        f=pd.read_csv(out/'tables/final_selected_holdout.csv');text += ['', 'Evaluación suplementaria final: selección independiente n=10000 y holdout n=50000. Estimando de política seleccionada, separado del máximo directo de traza.',f.to_string(index=False)]
        ff=f.groupby('method').objective_holdout.agg(['mean','std']);text.append(ff.to_string())
        for a,b in PAIRINGS[3:]:
            aa=f[f.method==a].set_index('rep').objective_holdout;bb=f[f.method==b].set_index('rep').objective_holdout;dif=aa-bb
            text.append(f'Final holdout {a} menos {b}: media={dif.mean():.6f}, diferencias={dif.tolist()}, dentro de .5={dif.ge(-.5).mean():.2f}. Presupuestos distintos y 100 actualizaciones.')
    text += ['', 'Ejecución y validación: modelo/semillas/regla originales congelados por SHA256; todos los controles previos quedan intactos. Dependencias Julia existentes; advertencias opcionales pandas/Axes3D no afectan gráficos 2D.']
    (out/'informe_cientifico.txt').write_text('\n'.join(text)+'\n')

def verify(out,items,config):
    from PIL import Image
    s=config['science'];d=items['data'];reps=s['rep_ids'];K=s['max_iters'];old=ROOT/'resultados/012probing_anytime_budget'
    assert len(d)==6*len(reps)*(K+1) and set(d.method)==set(METHODS)
    assert np.isfinite(d.objective).all() and ((d[['z3','z4']]>=0)&(d[['z3','z4']]<=1)).all().all()
    cp=pd.read_csv(out/'validation/crn_prefix_checks.csv');assert len(cp)==len(reps)*K and cp.prefix100_exact.all() and cp.prefix250_exact.all()
    assert cp.seed.nunique()==len(cp) and cp.first_path_sha256.nunique()==len(cp)
    cr=pd.read_csv(out/'validation/control_reuse.csv');assert len(cr)==2*len(reps) and cr.exact_decisions.all() and cr.exact_gradients_and_diagnostics.all()
    oldtr=pd.read_csv(old/'tables/objectives_by_iteration.csv');oldtime=pd.read_csv(old/'tables/search_budget_runtime.csv')
    original_events=pd.read_csv(old/'tables/hitting_times.csv',dtype={'epsilon_name':str})
    for rep in reps:
        for base in ['Fresh','Green']:
            new=d[(d.rep==rep)&(d.method==f'{base}-500')].sort_values('iteration')
            orig=oldtr[(oldtr.rep==rep)&(oldtr.method==base)&(oldtr.iteration<=K)].sort_values('iteration')
            assert np.allclose(new[['z3','z4','objective','reference_objective']],orig[['z3','z4','objective','reference_objective']],rtol=1e-13,atol=1e-13)
            bt=oldtime[(oldtime.rep==rep)&(oldtime.method==base)&(oldtime.iteration<=K)].sort_values('iteration')
            for col in ['new_trajectories','cumulative_trajectories','generation_seconds','policy_seconds','gradient_seconds','search_seconds','cumulative_search_seconds']:
                assert np.allclose(new[col],bt[col],rtol=1e-13,atol=1e-13)
            if not s['quick']:
                for eps in EPS:
                    for kind in ['first','sustained5']:
                        a=items['events'][(items['events'].rep==rep)&(items['events'].method==f'{base}-500')&(items['events'].epsilon_name==str(eps))&(items['events'].kind==kind)].iloc[0]
                        b=original_events[(original_events.rep==rep)&(original_events.method==base)&(original_events.epsilon_name==str(eps))&(original_events.kind==kind)].iloc[0]
                        assert a.achieved==b.achieved
                        assert np.allclose([a.start_iteration,a.confirmation_iteration],[b.start_iteration,b.confirmation_iteration],equal_nan=True)
    bd=items['budgets']
    for B in s['budgets']:
        expected=[m for m in METHODS if B%int(m.split('-')[1])==0 and B//int(m.split('-')[1])<=K]
        q=bd[bd.budget==B];assert set(q.method)==set(expected) and len(q)==len(expected)*len(reps)
        assert all(int(r.iteration)*int(r.N_PATHS)==B for r in q.itertuples())
    gd=pd.read_csv(out/'tables/green_lr_diagnostics.csv');assert len(gd)==3*len(reps)*K
    assert np.allclose(gd.available_paths,gd.N_PATHS*gd.iteration) and np.allclose(gd.history_banks,gd.iteration)
    assert (gd.ess>=1-1e-8).all() and (gd.ess<=gd.available_paths+1e-6).all()
    dg=pd.read_csv(out/'tables/gradients_and_diagnostics.csv');q=dg[dg.method.str.startswith('Green')].merge(gd,on=['rep','method','iteration'],suffixes=('','_replay'),validate='one_to_one')
    assert np.array_equal(q.lr_clips,q.truncations) and np.allclose(q.ess,q.ess_replay,rtol=1e-13)
    hist=pd.read_csv(out/'tables/green_loglr_histograms.csv');assert np.allclose(hist.groupby(['rep','method','iteration']).fraction.sum(),1)
    plan=pd.read_csv(out/'configs/validation_plan.csv');checks=pd.read_csv(out/'tables/crossing_validation.csv')
    assert len(checks)==sum(len(str(x).split(';')) for x in plan.loc[plan.achieved,'validation_iterations'])
    bv=pd.read_csv(out/'tables/budget_validation.csv');assert len(bv)==len(bd)
    assert bv.evaluation_paths.eq(s['validation_paths']).all()
    seeds=pd.read_csv(out/'configs/search_seeds.csv')
    assert len(seeds)==len(reps)*6*K and seeds.groupby(['rep','iteration']).seed.nunique().eq(1).all()
    assert seeds.seed.nunique()==len(reps)*K
    training=set(seeds.seed);trace=set(d.evaluation_seed);validation=set(bv.evaluation_seed)
    assert len(trace)==len(reps) and len(validation)==len(reps)
    assert not(training&trace or training&validation or trace&validation)
    prof=pd.read_csv(out/'tables/control_history_profiling.csv')
    assert len(prof)==len(reps)*(K+1) and prof.reuse_seconds.ge(0).all()
    assert prof.reuse_seconds.le(prof.gradient_seconds+1e-12).all()
    if not s['quick']:
        final=pd.read_csv(out/'tables/final_selected_holdout.csv');assert len(final)==len(reps)*6
        assert final.selection_paths.eq(10000).all() and final.holdout_paths.eq(50000).all()
        assert not(training&set(final.selection_seed) or training&set(final.holdout_seed))
        for rep in reps:
            previous=pd.read_csv(ROOT/f'resultados/011probing_high_contrast_interior/comparisons/I0252/rep_{rep:02d}_finals.csv')
            for base in ['Fresh','Green']:
                a=final[(final.rep==rep)&(final.method==f'{base}-500')].iloc[0];b=previous[previous.method==base].iloc[0]
                assert np.allclose([a.objective_holdout,a.evaluation_se,a.objective_direct],[b.objective_holdout,b.evaluation_se,b.objective_direct],atol=1e-12,rtol=1e-13)
    pngs=sorted((out/'figures').glob('*.png'));assert len(pngs)==len(list((out/'figures').glob('*.pdf'))) and len(pngs)>25
    for p in pngs:
        with Image.open(p) as im:assert im.width>600 and im.height>400 and np.asarray(im.convert('RGB')).std()>10
        assert p.with_suffix('.pdf').read_bytes().startswith(b'%PDF-')
    before=json.loads((ROOT/'resultados/013probing_green_simulation_efficiency/validation/preserved_before.json').read_text())
    changed=[p for p,h in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];assert not changed,changed[:10]
    manifest=dict(complete=True,passed=True,quick=s['quick'],fingerprint=config['fingerprint'],new_searches=4*len(reps),archived_searches_reused=2*len(reps),
        exact_history_profiling_replays=len(reps),repetitions=len(reps),iterations=K,objective_points=len(d),
        new_search_trajectories_executed=2*(100+250)*len(reps)*K,logical_search_trajectories_all_methods=2*(100+250+500)*len(reps)*K,
        nested_bank_checks=len(cp),all_controls_exact=True,original_hitting_conventions_verified=True,
        separate_training_evaluation_seeds_verified=True,original_files_verified=len(before),all_original_files_unchanged=True,
        green_diagnostic_iterations=len(gd),candidate_revalidation_rows=len(checks),equal_budget_revalidation_points=len(bv),
        png_files=len(pngs),pdf_files=len(pngs),all_figures_nonblank=True,
        sources_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['013probing_green_simulation_efficiency.jl','probing_green_efficiency_013_runtime.jl','analyze_green_simulation_efficiency_013.py','run_green_simulation_efficiency_013.py','PROTOCOLO_green_simulation_efficiency_013.txt']})
    (out/'validation/output_verification.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

def index(out,items,manifest):
    reps=sorted(items['data'].rep.unique())
    page=['<!doctype html><meta charset="utf-8"><title>I0252: simulation efficiency</title>',
        '<style>body{font-family:system-ui;margin:2em;max-width:1600px}img{max-width:100%}td,th{padding:.5em;text-align:left}a{margin-right:.7em}</style>',
        '<h1>I0252: Fresh y Green con 100, 250 y 500 trayectorias</h1>',
        '<p>La comparación principal usa el mismo presupuesto de nuevas trayectorias. Se conservan los controles originales de 500; veinte búsquedas nuevas con tamaños 100 y 250. Cinco repeticiones independientes, métodos emparejados por primitivas CRN anidadas en cada iteración.</p>',
        '<p><a href="informe_cientifico.txt">Informe P1–P12</a><a href="README_execution.txt">Reproducción</a><a href="tables/main_table.csv">Tabla principal</a><a href="tables/main_individual.csv">Cada repetición</a><a href="tables/equal_budget_comparisons.csv">Igual presupuesto</a><a href="tables/budget_validation_pairs.csv">Revalidación independiente emparejada</a><a href="validation/output_verification.json">Verificación</a></p>',
        '<p>A(T) promedia las decisiones visitadas por actualización; Q(B) integra la última decisión disponible sobre presupuesto real. Las bandas muestran SD, no intervalos de confianza. El costo histórico y las evaluaciones no se cuentan como simulaciones nuevas de búsqueda.</p>',
        '<table><tr><th>Repetición</th><th>Por presupuesto</th><th>Por iteración</th><th>z3 / z4</th></tr>']
    for r in reps:
        page.append(f'<tr><td>rep{r:02d}</td><td><a href="figures/convergence_budget_rep{r:02d}.png">PNG</a></td><td><a href="figures/convergence_iteration_rep{r:02d}.png">PNG</a></td><td><a href="figures/probing_trajectories_rep{r:02d}.png">PNG</a></td></tr>')
    page.append('</table>')
    preferred=['convergence_budget_rep01','average_convergence_budget','budget_normalized_performance','equal_budget_independent_validation',
        'average_convergence_iteration','individual_repetitions_budget','budget_area_quality','accumulated_objective_iteration','time_to_target_budget',
        'runtime_objective_rep01','green_ess','green_lr_distributions']
    files={p.stem for p in (out/'figures').glob('*.png')}
    for name in preferred+sorted(files-set(preferred)):
        assert name in files
        page += [f'<h2>{name.replace("_"," ")}</h2><p><a href="figures/{name}.png">PNG</a><a href="figures/{name}.pdf">PDF</a></p><img loading="lazy" src="figures/{name}.png">']
    (out/'index.html').write_text('\n'.join(page)+'\n')
    links=re.findall(r'(?:href|src)="([^"]+)"','\n'.join(page));assert all((out/x).is_file() for x in links)
    manifest['all_gallery_links_valid']=True;manifest['gallery_links']=len(links)
    (out/'validation/output_verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)

def main(output):
    out=Path(output).resolve();config=tomllib.loads((out/'configs/experiment.toml').read_text());items=build_metrics(out,config['science'])
    plots(out,items,config['science']);report(out,items,config)
    (out/'README_execution.txt').write_text('''013probing_green_simulation_efficiency: execute from the original repository.
First execution with automatic SHA256 preservation snapshot, or resume:
python3 run_green_simulation_efficiency_013.py --resume
Full/resume:
env GKSwstype=100 JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno 013probing_green_simulation_efficiency.jl --resume
Reduced prefix test: add --quick-test (two repetitions/six updates, separate folder).
Scientific phases without figures: add --no-plots. Then run:
python3 analyze_green_simulation_efficiency_013.py resultados/013probing_green_simulation_efficiency
Plots only: add --plots-only. Completed searches/evaluations/diagnostics use checkpoints.
Exact original model and source hashes guard scientific resumes. Existing Julia environment is reused.
Four new configurations, five original paired seed streams; ten original controls are retained exactly.
Five additional Green-500 instrumented replays recover the history-clock split; these are neither new searches nor extra statistical replicates. Control primary times remain 012.
No concurrent search timings. Parallel work starts only during external evaluation and diagnostics.
Pointwise LR diagnostics reconstruct existing seed banks without changing any estimator or consuming search budget.
Evaluation budgets, profiling replays, auditing and history work are excluded from N*t search paths.
Main J(z) uses the original common independent 1000-path bank. Candidate/budget revalidation uses the common independent 5000-path bank. Supplemental final selected-policy value uses 10000/50000 selector/holdout.
The native A(T), budget-area Q(B), equal-iteration and equal-budget tables are distinct. Unreached budgets are missing/nonparticipants, not extrapolated points. Missing target arrivals remain censored.
Report: informe_cientifico.txt; all English PNG/PDF figures: index.html; exact model/seeds: configs/.
''')
    for f in ['013probing_green_simulation_efficiency.jl','probing_green_efficiency_013_runtime.jl','PROTOCOLO_green_simulation_efficiency_013.txt',
        'analyze_green_simulation_efficiency_013.py','run_green_simulation_efficiency_013.py','012probing_anytime_budget_analysis.jl','probing_anytime_012_runtime.jl','011probing_high_contrast_interior.jl',
        'probing_interior_011_diagnostics.jl','010probing_instance_landscape_search.jl','probing_regime_model.jl','probing_experiments_common.jl',
        '003probing_four_searches_softmax.jl','002probing_three_searches_n500_full45iters.jl','analyze_anytime_012.py','analyze_landscape_010.py']:
        shutil.copy2(ROOT/f,out/'reproduction'/f)
    for f in ['Project.toml','Manifest.toml','python_requirements.txt']:
        src=ROOT/'resultados/012probing_anytime_budget/reproduction'/f
        if src.exists():shutil.copy2(src,out/'reproduction'/f)
    manifest=verify(out,items,config);index(out,items,manifest)

if __name__=='__main__':main(sys.argv[1])
