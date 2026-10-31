"""013: unchanged 012 metrics/figures extended with Green + Softmax beta=0.10."""
from pathlib import Path
import sys,json,math,hashlib,shutil,platform,re
try:import tomllib
except ModuleNotFoundError:import tomli as tomllib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analyze_landscape_010 import METHODS,COLORS,STYLES
LABELS=dict(zip(METHODS,['Fixed','Fresh','Green','Fresh + Softmax β=0.10','Fresh + Softmax β=0.50','Fresh + PSoft B=300','Green + Softmax β=0.50']))
METHODS=list(METHODS)+['GreenSoftmax010']
COLORS=dict(COLORS,GreenSoftmax010='#008080')
STYLES=dict(STYLES,GreenSoftmax010='--')
LABELS=dict(LABELS,GreenSoftmax010='Green + Softmax β=0.10')
EPS=[.5,1.,2.]


def first_run(values,reference,epsilon,window=1):
    good=(reference-np.asarray(values))<=epsilon
    for start in range(len(good)-window+1):
        if good[start:start+window].all():return start,start+window-1
    return None,None


def empirical_median(times,event):
    times=np.asarray(times);event=np.asarray(event,dtype=bool)
    rank=int(math.ceil(len(times)*.5))-1
    vals=np.sort(times[event])
    return float(vals[rank]) if len(vals)>rank else np.nan


def restricted_mean(times,event,horizon):
    # Administrative censoring at a common iteration horizon, not tau imputation.
    return np.where(np.asarray(event,dtype=bool),np.minimum(times,horizon),horizon).mean()


def build_metrics(out,science):
    traces=pd.read_csv(out/'tables/objectives_by_iteration.csv')
    budget=pd.read_csv(out/'tables/search_budget_runtime.csv')
    keys=['cohort','rep','method','iteration']
    data=traces.merge(budget.drop(columns='scenario'),on=keys,validate='one_to_one',suffixes=('','_budget'))
    assert not data.duplicated(keys).any()
    K=science['max_iters'];n=science['n_paths']
    assert len(data)==data.rep.nunique()*len(METHODS)*(K+1)
    assert np.isfinite(data.objective).all()
    acc=[];events=[];individual=[];quality=[]
    horizons=sorted(set([min(t,K) for t in ([2,4,6] if science['quick'] else [20,50,100])]))
    for (cohort,rep,method),g in data.groupby(['cohort','rep','method'],sort=True):
        g=g.sort_values('iteration').copy();assert np.array_equal(g.iteration,np.arange(K+1))
        expected=np.minimum(g.iteration,1)*n if method=='Fixed' else g.iteration*n
        assert np.array_equal(g.cumulative_trajectories,expected)
        assert g.new_policy_path_evaluations.sum()==(n if method=='Fixed' else n*K)*144
        assert g.cumulative_search_seconds.is_monotonic_increasing
        assert np.allclose(g.search_seconds.cumsum(),g.cumulative_search_seconds)
        assert (g.bootstrap_path_draws.eq(0).all() if method!='PSoft300' else g.bootstrap_path_draws.iloc[1:].eq(n*300).all())
        if method in ['Green','GreenSoftmax050','GreenSoftmax010']:assert np.array_equal(g.history_banks.iloc[1:],np.arange(1,K+1))
        v=g.objective.to_numpy();ref=float(g.reference_objective.iloc[0]);A=np.cumsum(v)/np.arange(1,K+2);R=ref-A
        for j,row in enumerate(g.itertuples()):
            acc.append(dict(cohort=cohort,rep=rep,method=method,iteration=j,accumulated_objective=A[j],
                            accumulated_deficit=R[j],reference_objective=ref,point_deficit=ref-v[j],
                            cumulative_trajectories=row.cumulative_trajectories,cumulative_search_seconds=row.cumulative_search_seconds))
        r=dict(cohort=cohort,rep=rep,method=method,final_objective=v[-1],accumulated_deficit_final=R[-1],
               total_simulated_trajectories=int(g.cumulative_trajectories.iloc[-1]),total_search_seconds=g.cumulative_search_seconds.iloc[-1],
               gradient_seconds=g.gradient_seconds.sum(),bootstrap_seconds=g.bootstrap_seconds.sum(),
               final_z3=g.z3.iloc[-1],final_z4=g.z4.iloc[-1],reference_objective=ref)
        for T in horizons:
            r[f'mean_objective_T{T}']=A[T]
            quality.append(dict(cohort=cohort,rep=rep,method=method,horizon=T,mean_objective=A[T],mean_deficit=R[T]))
        individual.append(r)
        rel=.05*(ref-v[0]);tolerances=[(str(e),e) for e in EPS]+([('relative_5pct',rel)] if rel>0 else [])
        for ename,epsilon in tolerances:
            for kind,w in [('first',1),('sustained5',5)]:
                start,confirm=first_run(v,ref,epsilon,w);ok=start is not None
                row=dict(cohort=cohort,rep=rep,method=method,epsilon_name=ename,epsilon=epsilon,kind=kind,achieved=ok,
                    status=('initial' if start==0 and w==1 else 'initial_run_confirmed' if start==0 else 'attained') if ok else 'censored',
                    start_iteration=start,confirmation_iteration=confirm,detection_iteration=confirm,censor_iteration=K,
                    observed_iteration=confirm if ok else K,observed_event=ok,
                    hitting_simulation_budget=g.cumulative_trajectories.iloc[start] if ok else np.nan,
                    confirmation_simulation_budget=g.cumulative_trajectories.iloc[confirm] if ok else np.nan,
                    censor_simulation_budget=int(g.cumulative_trajectories.iloc[-1]),
                    hitting_search_seconds=g.cumulative_search_seconds.iloc[start] if ok else np.nan,
                    confirmation_search_seconds=g.cumulative_search_seconds.iloc[confirm] if ok else np.nan,
                    censor_search_seconds=g.cumulative_search_seconds.iloc[-1],
                    objective_at_start=v[start] if ok else np.nan,objective_at_confirmation=v[confirm] if ok else np.nan,
                    reference_objective=ref)
                events.append(row)
    data.to_csv(out/'tables/anytime_objectives_budget_runtime.csv',index=False)
    acc=pd.DataFrame(acc);events=pd.DataFrame(events);individual=pd.DataFrame(individual);quality=pd.DataFrame(quality)
    for name,t in [('accumulated_quality',acc),('hitting_times',events),('comparison_individual',individual),('quality_at_horizons',quality)]:t.to_csv(out/'tables'/f'{name}.csv',index=False)
    plan=pd.read_csv(out/'configs/validation_plan.csv',dtype={'epsilon_name':str})
    merged=events.merge(plan,on=['cohort','rep','method','epsilon_name','kind'],suffixes=('','_plan'),validate='one_to_one')
    assert (merged.achieved==merged.achieved_plan).all()
    assert np.allclose(merged.start_iteration,merged.start_iteration_plan,equal_nan=True)
    assert np.allclose(merged.confirmation_iteration,merged.confirmation_iteration_plan,equal_nan=True)
    summaries=[]
    for key,g in events.groupby(['cohort','method','epsilon_name','kind']):
        cohort,method,en,kind=key;e=g.achieved.to_numpy();t=g.observed_iteration.to_numpy()
        summaries.append(dict(cohort=cohort,method=method,epsilon_name=en,kind=kind,repetitions=len(g),
            success_proportion=e.mean(),successes=int(e.sum()),epsilon_mean=g.epsilon.mean(),
            median_start_iteration=empirical_median(g.start_iteration.fillna(K),e),
            median_confirmation_iteration=empirical_median(g.confirmation_iteration.fillna(K),e),
            median_hitting_simulation_budget=empirical_median(g.hitting_simulation_budget.fillna(g.censor_simulation_budget),e),
            median_confirmation_simulation_budget=empirical_median(g.confirmation_simulation_budget.fillna(g.censor_simulation_budget),e),
            restricted_mean_hitting_iteration=restricted_mean(t,e,K),restricted_horizon=K,
            mean_search_time_to_target_successes_only=g.loc[g.achieved,'confirmation_search_seconds'].mean(),
            censoring_note='administrative; missing tau retained; sustained clock is confirmation'))
    targets=pd.DataFrame(summaries)
    targets=targets.merge(individual.groupby(['cohort','method']).accumulated_deficit_final.mean().rename('mean_accumulated_deficit'),on=['cohort','method'])
    targets.to_csv(out/'tables/target_summary.csv',index=False)
    numeric=individual.select_dtypes(include=np.number).columns.difference(['rep'])
    summary=individual.groupby(['cohort','method'])[numeric].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();summary.to_csv(out/'tables/comparison_summary.csv',index=False)
    validation=pd.read_csv(out/'tables/crossing_validation.csv',dtype={'epsilon_name':str})
    assert not validation.duplicated(['cohort','rep','method','epsilon_name','kind','iteration']).any()
    for p in plan[plan.achieved].itertuples():
        q=validation[(validation.rep==p.rep)&(validation.method==p.method)&(validation.epsilon_name==p.epsilon_name)&(validation.kind==p.kind)]
        wanted=[int(t) for t in str(p.validation_iterations).split(';')]
        assert sorted(q.iteration.tolist())==sorted(wanted),(p.rep,p.method,p.epsilon_name,p.kind)
    validation_summary=[]
    for key,g in validation[validation.role=='candidate'].groupby(['cohort','rep','method','epsilon_name','kind']):
        co,rep,m,en,kind=key;validation_summary.append(dict(cohort=co,rep=rep,method=m,epsilon_name=en,kind=kind,
            points=len(g),all_candidates_meet_new_threshold=g.meets_threshold.all(),
            first_candidate_meets=g.sort_values('iteration').meets_threshold.iloc[0],
            max_paired_mc_se=g.paired_mc_se.max(),largest_deficit=g.deficit.max()))
    validation_summary=pd.DataFrame(validation_summary);validation_summary.to_csv(out/'tables/crossing_validation_summary.csv',index=False)
    # Pairwise tables retain every repetition; censored pairs get no invented tau difference.
    pairs=[]
    for co,rep in data[['cohort','rep']].drop_duplicates().itertuples(index=False,name=None):
        for a,b in [('GreenSoftmax010','Green'),('GreenSoftmax010','GreenSoftmax050'),('GreenSoftmax050','Green'),('Softmax010','Fresh'),('Softmax050','Fresh'),('PSoft300','Fresh')]:
            ia=individual[(individual.cohort==co)&(individual.rep==rep)&(individual.method==a)].iloc[0]
            ib=individual[(individual.cohort==co)&(individual.rep==rep)&(individual.method==b)].iloc[0]
            for en in [str(e) for e in EPS]:
                for kind in ['first','sustained5']:
                    ea=events[(events.cohort==co)&(events.rep==rep)&(events.method==a)&(events.epsilon_name==en)&(events.kind==kind)].iloc[0]
                    eb=events[(events.cohort==co)&(events.rep==rep)&(events.method==b)&(events.epsilon_name==en)&(events.kind==kind)].iloc[0]
                    both=ea.achieved and eb.achieved
                    order=('a_earlier' if ea.confirmation_iteration<eb.confirmation_iteration else 'b_earlier' if ea.confirmation_iteration>eb.confirmation_iteration else 'tie') if both else 'only_a_attained' if ea.achieved else 'only_b_attained' if eb.achieved else 'both_censored'
                    row=dict(cohort=co,rep=rep,method_a=a,method_b=b,epsilon_name=en,kind=kind,
                        a_achieved=ea.achieved,b_achieved=eb.achieved,a_confirmation_iteration=ea.confirmation_iteration,b_confirmation_iteration=eb.confirmation_iteration,
                        order=order,iteration_difference=ea.confirmation_iteration-eb.confirmation_iteration if both else np.nan,
                        simulation_budget_difference=ea.confirmation_simulation_budget-eb.confirmation_simulation_budget if both else np.nan,
                        a_search_time_to_target=ea.confirmation_search_seconds,b_search_time_to_target=eb.confirmation_search_seconds,
                        final_objective_difference=ia.final_objective-ib.final_objective,total_search_time_difference=ia.total_search_seconds-ib.total_search_seconds)
                    for T in horizons:row[f'mean_objective_difference_T{T}']=ia[f'mean_objective_T{T}']-ib[f'mean_objective_T{T}']
                    pairs.append(row)
    pairs=pd.DataFrame(pairs);pairs.to_csv(out/'tables/method_pair_comparisons.csv',index=False)
    return data,acc,events,individual,quality,targets,summary,validation,validation_summary,pairs,horizons


def savefig(fig,out,name):
    fig.savefig(out/'figures'/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def lineplot(data,out,name,mode='iteration',mean=False,band=False,exclude_fixed=False):
    fig,ax=plt.subplots(figsize=(10,6))
    for m in METHODS:
        if exclude_fixed and m=='Fixed':continue
        q=data[data.method==m].sort_values('iteration')
        if mean:
            q=q.groupby('iteration').agg(objective=('objective','mean'),sd=('objective','std'),
                cumulative_trajectories=('cumulative_trajectories','first'),cumulative_search_seconds=('cumulative_search_seconds','mean')).reset_index()
        xname={'iteration':'iteration','budget':'cumulative_trajectories','runtime':'cumulative_search_seconds'}[mode]
        x=q[xname].to_numpy();y=q.objective.to_numpy();sd=q.sd.to_numpy() if mean else np.zeros(len(q))
        if mode=='budget' and m=='Fixed':
            keep=q.iteration>0;xx=q.loc[keep,'cumulative_trajectories'].iloc[0];yy=q.loc[keep,'objective']
            ax.vlines(xx,yy.min(),yy.max(),color=COLORS[m],lw=2,label='Fixed (constant 500 trajectories)')
            ax.scatter(np.full(keep.sum(),xx),yy,color=COLORS[m],alpha=.25,s=12)
            ax.scatter(xx,y[-1],marker='D',s=65,color=COLORS[m],edgecolor='black',zorder=9)
            if band:ax.errorbar(xx,y[-1],yerr=sd[-1],fmt='none',color=COLORS[m],capsize=4)
        else:
            ax.plot(x,y,color=COLORS[m],ls=STYLES[m],label=LABELS[m],lw=1.7)
            if band:ax.fill_between(x,y-sd,y+sd,color=COLORS[m],alpha=.13)
    xlabel={'iteration':'Iteration','budget':'Cumulative simulated trajectories (search only)','runtime':'Cumulative search computation time (seconds)'}[mode]
    title=('Mean of original repetitions' if mean else 'Predefined illustrative repetition 1')+(' ± SD' if band else '')
    ax.set(xlabel=xlabel,ylabel='Independent objective value J(z)',title='I0252: anytime convergence\n'+title)
    ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
    if mode=='budget' and not exclude_fixed:fig.text(.12,.01,'Fixed diamond: final decision after 100 updates; vertical column consumes no additional system paths.',fontsize=8)
    savefig(fig,out,name)


def threshold_plots(events,out,K):
    for mode,name in [('iterations','time_to_target_iterations'),('simulations','time_to_target_simulations')]:
        fig,axes=plt.subplots(2,3,figsize=(17,9),squeeze=False)
        for ri,kind in enumerate(['first','sustained5']):
            for ci,epsilon in enumerate(EPS):
                ax=axes[ri,ci];q=events[(events.kind==kind)&(events.epsilon_name==str(epsilon))]
                for j,m in enumerate(METHODS):
                    g=q[q.method==m].sort_values('rep');hit=g.achieved.to_numpy();xx=j+np.linspace(-.16,.16,len(g))
                    yy=g.confirmation_iteration.to_numpy() if mode=='iterations' else g.confirmation_simulation_budget.to_numpy()
                    cens=np.full(len(g),K) if mode=='iterations' else g.censor_simulation_budget.to_numpy()
                    ax.scatter(xx[hit],yy[hit],color=COLORS[m],s=30,zorder=3)
                    ax.scatter(xx[~hit],cens[~hit],color='red',marker='x',s=45,zorder=4)
                ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=35,ha='right',fontsize=8)
                ax.set(title=f'{"First hit" if kind=="first" else "Five-decision confirmation"}; ε={epsilon}',
                       ylabel='Iteration' if mode=='iterations' else 'Search trajectories');ax.grid(axis='y',alpha=.2)
        fig.legend([Line2D([],[],marker='o',ls='',color='black'),Line2D([],[],marker='x',ls='',color='red')],
                   ['Observed attainment','Right-censored at method horizon'],loc='upper center',ncol=2)
        fig.tight_layout(rect=(0,0,1,.95));savefig(fig,out,name)
    for mode,name in [('iteration','target_attainment_probability'),('budget','target_attainment_probability_simulations')]:
        fig,axes=plt.subplots(2,3,figsize=(16,8),squeeze=False)
        for ri,kind in enumerate(['first','sustained5']):
            for ci,epsilon in enumerate(EPS):
                ax=axes[ri,ci];q=events[(events.kind==kind)&(events.epsilon_name==str(epsilon))]
                for m in METHODS:
                    g=q[q.method==m];ok=g.achieved.to_numpy()
                    if mode=='budget' and m=='Fixed':
                        ax.scatter(500,ok.mean(),marker='D',s=50,color=COLORS[m],label='Fixed (≤100 updates)',zorder=8);continue
                    grid=np.arange(K+1) if mode=='iteration' else np.arange(K+1)*500
                    t=g.confirmation_iteration.to_numpy() if mode=='iteration' else g.confirmation_simulation_budget.to_numpy()
                    f=np.array([np.mean(ok&(t<=v)) for v in grid])
                    ax.step(grid,f,where='post',color=COLORS[m],ls=STYLES[m],label=LABELS[m])
                ax.set(title=f'{"First hit" if kind=="first" else "Five-decision confirmation"}; ε={epsilon}',ylim=(-.03,1.03),
                    xlabel='Iteration' if mode=='iteration' else 'Cumulative search trajectories',ylabel='Fraction attaining target')
                ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='center left',bbox_to_anchor=(.99,.5),fontsize=8)
        fig.tight_layout();savefig(fig,out,name)


def render_figures(out,data,acc,events,quality,validation,validation_summary,K):
    # All main figures and summaries below use the original five-seed cohort.
    d=data[data.cohort=='original_011'];a=acc[acc.cohort=='original_011'];e=events[events.cohort=='original_011'];q=quality[quality.cohort=='original_011']
    rep1=d[d.rep==1]
    lineplot(rep1,out,'anytime_convergence_iteration')
    lineplot(rep1,out,'anytime_convergence_simulation_budget',mode='budget')
    lineplot(rep1,out,'anytime_convergence_simulation_budget_without_fixed',mode='budget',exclude_fixed=True)
    lineplot(d,out,'anytime_convergence_simulation_budget_mean',mode='budget',mean=True)
    lineplot(d,out,'anytime_convergence_simulation_budget_mean_sd',mode='budget',mean=True,band=True)
    lineplot(rep1,out,'anytime_convergence_runtime',mode='runtime')
    lineplot(d,out,'anytime_mean_convergence',mean=True,band=True)
    lineplot(d,out,'anytime_mean_convergence_without_fixed',mean=True,band=True,exclude_fixed=True)
    threshold_plots(e,out,K)
    for value,name,ylabel in [('accumulated_objective','accumulated_objective','Mean visited-decision objective A(T)'),
                              ('accumulated_deficit','accumulated_regret','Mean deficit to approximate reference R(T)')]:
        fig,ax=plt.subplots(figsize=(10,6))
        for m in METHODS:
            g=a[a.method==m].groupby('iteration')[value].agg(['mean','std']);x=g.index.to_numpy();y=g['mean'].to_numpy();sd=g['std'].to_numpy()
            ax.plot(x,y,color=COLORS[m],ls=STYLES[m],label=LABELS[m]);ax.fill_between(x,y-sd,y+sd,color=COLORS[m],alpha=.12)
        if value=='accumulated_deficit':ax.axhline(0,color='gray',lw=.8)
        ax.set(xlabel='Iteration horizon T',ylabel=ylabel,title='I0252: accumulated decision quality, mean ± SD');ax.grid(alpha=.2)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8);savefig(fig,out,name)
    reps=sorted(d.rep.unique());fig,axes=plt.subplots(2,3,figsize=(17,10));axes=axes.ravel()
    for ax,rep in zip(axes,reps):
        for m in METHODS:
            g=d[(d.rep==rep)&(d.method==m)].sort_values('iteration');ax.plot(g.iteration,g.objective,color=COLORS[m],ls=STYLES[m],label=LABELS[m])
        ax.set(title=f'Repetition {rep}'+(' (predefined example)' if rep==1 else ''),xlabel='Iteration',ylabel='Independent J(z)');ax.grid(alpha=.2)
    for ax in axes[len(reps):]:ax.set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='center left',bbox_to_anchor=(.99,.5),fontsize=8)
    fig.tight_layout();savefig(fig,out,'anytime_individual_repetitions')
    for m in METHODS:
        fig,ax=plt.subplots(figsize=(9,5.5))
        for j,rep in enumerate(reps):
            g=d[(d.rep==rep)&(d.method==m)].sort_values('iteration')
            ax.plot(g.iteration,g.objective,color=COLORS[m],ls=['-','--','-.',':',(0,(5,1,1,1,1,1))][j%5],label=f'Repetition {rep}',lw=1.6)
        ax.set(title=f'I0252: {LABELS[m]}, all original repetitions',xlabel='Iteration',ylabel='Independent objective value J(z)');ax.grid(alpha=.2);ax.legend()
        savefig(fig,out,f'anytime_{m}_individual_repetitions')
    fig,axes=plt.subplots(2,1,figsize=(11,9),sharex=True)
    for ax,dim in zip(axes,['z3','z4']):
        for m in METHODS:
            g=rep1[rep1.method==m].sort_values('iteration');ax.plot(g.iteration,g[dim],color=COLORS[m],ls=STYLES[m],label=LABELS[m])
        ax.set(ylabel=f'Class-{dim[-1]} probing intensity',ylim=(-.03,1.03));ax.grid(alpha=.2)
    axes[0].set_title('I0252: probing trajectories, predefined repetition 1');axes[-1].set_xlabel('Iteration')
    axes[0].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8);savefig(fig,out,'probing_trajectories')
    sf=pd.read_csv(out/'source_011/I0252_surface.csv');grid=np.sort(sf.z3.unique());zz=sf.pivot(index='z4',columns='z3',values='objective').to_numpy()
    fig,ax=plt.subplots(figsize=(9,7));im=ax.contourf(grid,grid,zz,levels=24,cmap='inferno');ax.contour(grid,grid,zz,levels=12,colors='k',linewidths=.35,alpha=.5)
    for m in METHODS:
        g=rep1[rep1.method==m].sort_values('iteration');ax.plot(g.z3,g.z4,color=COLORS[m],ls=STYLES[m],label=LABELS[m],lw=1.5)
        ax.scatter(g.z3.iloc[-1],g.z4.iloc[-1],color=COLORS[m],edgecolor='white',s=40,zorder=10)
    ax.scatter(.7,.7,marker='X',color='cyan',edgecolor='black',s=100,label='Frozen competitive reference',zorder=12)
    ax.scatter(.12,.1,marker='*',color='gold',edgecolor='black',s=100,label='Initial decision',zorder=12)
    ax.set(xlabel='Class-3 probing intensity z3',ylabel='Class-4 probing intensity z4',title='I0252: trajectories on estimated objective surface, repetition 1',aspect='equal')
    fig.colorbar(im,ax=ax,label='Direct estimated J(z), independent 5000-path grid')
    ax.legend(loc='upper left',bbox_to_anchor=(1.3,1),fontsize=8);savefig(fig,out,'trajectories_on_oracle')
    horizons=sorted(q.horizon.unique());fig,axes=plt.subplots(1,len(horizons),figsize=(5.5*len(horizons),5),squeeze=False)
    for ax,T in zip(axes[0],horizons):
        for j,m in enumerate(METHODS):
            g=q[(q.method==m)&(q.horizon==T)];ax.scatter(j+np.linspace(-.15,.15,len(g)),g.mean_objective,color=COLORS[m],s=30)
        ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=40,ha='right',fontsize=8);ax.set(title=f'T={T}',ylabel='A(T)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('I0252: accumulated objective across original repetitions');fig.tight_layout();savefig(fig,out,'accumulated_objective_distribution')
    phases=['generation_seconds','materialization_seconds','policy_seconds','gradient_seconds','bootstrap_seconds','update_seconds','other_seconds']
    totals=d.groupby(['rep','method'])[phases].sum().groupby('method').mean().reindex(METHODS)
    fig,ax=plt.subplots(figsize=(12,6));bottom=np.zeros(len(METHODS))
    for phase in phases:
        vals=totals[phase].to_numpy();ax.bar(range(len(METHODS)),vals,bottom=bottom,label=phase.replace('_seconds','').replace('_',' '));bottom+=vals
    ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=25,ha='right');ax.set(ylabel='Mean measured search time (seconds; separate sessions)',title='I0252: measured search costs (013 new method; 012 archived timings)')
    ax.legend(loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();savefig(fig,out,'search_time_breakdown')
    fig,ax=plt.subplots(figsize=(10,5.5))
    for m in ['Green','GreenSoftmax050','GreenSoftmax010']:
        for rep in reps:
            g=a[(a.method==m)&(a.rep==rep)].sort_values('iteration');ax.plot(g.iteration,g.accumulated_objective,color=COLORS[m],alpha=.4,lw=1,label=f'{LABELS[m]}: individual runs' if rep==reps[0] else None)
        g=a[a.method==m].groupby('iteration').accumulated_objective.mean();ax.plot(g.index,g,color=COLORS[m],lw=2.5,label=f'{LABELS[m]}: mean')
    ax.set(xlabel='Iteration horizon T',ylabel='Accumulated objective A(T)',title='I0252: Green versus Green + Softmax, every original repetition');ax.grid(alpha=.2);ax.legend()
    savefig(fig,out,'green_vs_green_softmax_accumulated')
    # Independent checks are conditional on candidates selected on the main bank.
    vs=validation_summary[(validation_summary.cohort=='original_011')&(validation_summary.epsilon_name=='0.5')]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,kind in zip(axes,['first','sustained5']):
        for j,m in enumerate(METHODS):
            g=vs[(vs.kind==kind)&(vs.method==m)]
            if len(g):ax.bar(j,g.all_candidates_meet_new_threshold.mean(),color=COLORS[m]);ax.text(j,1.02,f'n={len(g)}',ha='center',fontsize=8)
            else:ax.text(j,.03,'n=0',ha='center',fontsize=8,rotation=90)
        ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=35,ha='right',fontsize=8)
        ax.set(ylim=(0,1.16),ylabel='Fraction passing independent recheck',title='First candidate' if kind=='first' else 'Entire candidate five-decision run')
    fig.suptitle('ε=0.5: new independent-bank checks of selected candidates (conditional subset)');fig.tight_layout();savefig(fig,out,'crossing_validation_sensitivity')


def draw_repetition(out,data,rep,methods,suffix=''):
    """Five families, with identical data/colours in full and focused views."""
    d=data[data.rep==rep]
    for variable,tag in [('z3','z3_trajectory'),('z4','z4_trajectory'),
                         ('objective','convergence_full'),('objective','convergence_zoom')]:
        fig,ax=plt.subplots(figsize=(10,6))
        for m in methods:
            g=d[d.method==m].sort_values('iteration')
            ax.plot(g.iteration,g[variable],color=COLORS[m],ls=STYLES[m],label=LABELS[m],lw=1.6)
        if variable.startswith('z'):
            ax.set(ylim=(-.025,1.025),ylabel=f'Class-{variable[-1]} probing intensity {variable}')
            title=f'Class-{variable[-1]} probing trajectory'
        else:
            ax.set_ylabel('Independent objective estimate J(z)');title='Convergence (full scale)'
            if tag.endswith('zoom'):
                v=d[d.method.isin([m for m in methods if m!='Fixed'])].objective
                pad=max(.05,.05*(v.max()-v.min()));ax.set_ylim(v.min()-pad,v.max()+pad)
                title='Convergence (zoom: range determined by methods other than Fixed)'
        ax.set(xlabel='Iteration',xlim=(0,int(d.iteration.max())),title=f'I0252: {title}\nrep{rep:02d}, {len(methods)} methods')
        ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=9)
        # Match the familiar convergence_repNN_full/zoom naming.
        name=f'I0252_{tag}_rep{rep:02d}{suffix}'
        if tag.startswith('convergence_'):name=f'I0252_convergence_rep{rep:02d}_{tag.split("_")[-1]}{suffix}'
        savefig(fig,out,name)
    oracle(out,d,rep,methods,f'I0252_oracle_trajectories_rep{rep:02d}{suffix}')


def oracle(out,data,rep,methods,name):
    sf=pd.read_csv(out/'source_011/I0252_surface.csv');grid=np.sort(sf.z3.unique())
    zz=sf.pivot(index='z4',columns='z3',values='objective').reindex(index=grid,columns=grid).to_numpy()
    fig,ax=plt.subplots(figsize=(9,7));im=ax.contourf(grid,grid,zz,levels=24,cmap='inferno')
    ax.contour(grid,grid,zz,levels=12,colors='k',linewidths=.35,alpha=.5)
    for m in methods:
        g=data[data.method==m].sort_values('iteration')
        ax.plot(g.z3,g.z4,color=COLORS[m],ls=STYLES[m],label=LABELS[m],lw=1.8)
        ax.scatter(g.z3.iloc[-1],g.z4.iloc[-1],s=50,color=COLORS[m],edgecolor='white',zorder=10)
    ax.scatter(.12,.10,marker='*',color='gold',edgecolor='black',s=140,zorder=12,label='Initial decision')
    ax.scatter(.70,.70,marker='X',color='cyan',edgecolor='black',s=95,zorder=11,label='Frozen competitive reference')
    ax.set(xlabel='Class-3 probing intensity z3',ylabel='Class-4 probing intensity z4',
           title=f'I0252: trajectories on estimated objective surface, rep{rep:02d}',aspect='equal')
    fig.colorbar(im,ax=ax,label='Direct estimated J(z), independent 5000-path grid')
    ax.legend(loc='upper left',bbox_to_anchor=(1.3,1),fontsize=8)
    savefig(fig,out,name)


def supplemental_figures(out,items,quick):
    d,acc=items[:2]
    focused=[m for m in METHODS if m not in ('PSoft300','Softmax010')]
    for rep in sorted(d.rep.unique()):
        draw_repetition(out,d,rep,METHODS)
        draw_repetition(out,d,rep,focused,'_six_methods')
        r=d[d.rep==rep]
        for m in METHODS:
            g=r[r.method==m].sort_values('iteration')
            fig,ax=plt.subplots(figsize=(9,5.5))
            ax.plot(g.iteration,g.objective,color=COLORS[m],ls=STYLES[m],lw=1.7,label=LABELS[m])
            ax.axhline(g.reference_objective.iloc[0],color='grey',ls=':',label='Frozen reference (same CRN bank)')
            ax.set(xlabel='Iteration',ylabel='Independent objective estimate J(z)',title=f'I0252: {LABELS[m]}, rep{rep:02d}')
            ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
            savefig(fig,out,f'I0252_convergence_rep{rep:02d}_{m}')
            oracle(out,r,rep,[m],f'I0252_oracle_trajectories_rep{rep:02d}_{m}')
        print(f'All convergence / z3 / z4 / oracle views rendered: rep{rep:02d}',flush=True)
    # New method, every repetition on one plot, with mean/SD views for both intensities.
    for variable in ['z3','z4']:
        fig,ax=plt.subplots(figsize=(10,6))
        for m in METHODS:
            g=d[d.method==m].groupby('iteration')[variable].agg(['mean','std'])
            x=g.index.to_numpy();v=g['mean'].to_numpy();sd=g['std'].to_numpy()
            ax.plot(x,v,color=COLORS[m],ls=STYLES[m],label=LABELS[m]);ax.fill_between(x,v-sd,v+sd,color=COLORS[m],alpha=.10)
        ax.set(xlabel='Iteration',ylabel=f'Probing intensity {variable}',title=f'I0252: {variable}, mean ± between-repetition SD')
        ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
        savefig(fig,out,f'I0252_{variable}_mean_sd')
    if not quick:
        fs=pd.concat([pd.read_csv(p) for p in sorted((out/'comparisons/I0252').glob('rep_*_finals.csv'))])
        fs=fs[fs.method.isin(METHODS)];fs.to_csv(out/'tables/final_holdout_by_rep.csv',index=False)
        ss=fs.groupby('method').agg(mean_holdout=('objective_holdout','mean'),sd_holdout=('objective_holdout','std'),
                mean_direct_holdout=('objective_direct','mean'),mean_z3=('z3','mean'),mean_z4=('z4','mean'))
        ss.reindex(METHODS).to_csv(out/'tables/final_holdout_summary.csv')
        fig,ax=plt.subplots(figsize=(13,6))
        for j,m in enumerate(METHODS):
            g=fs[fs.method==m].sort_values('rep');ax.scatter(j+np.linspace(-.16,.16,len(g)),g.objective_holdout,color=COLORS[m],s=35)
            ax.scatter(j,g.objective_holdout.mean(),marker='_',s=220,color=COLORS[m],linewidths=3)
        ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=30,ha='right',fontsize=9)
        ax.set(ylabel='Independent selected-policy holdout value',title='I0252: final decisions; 10000-path selection / 50000-path holdout')
        ax.grid(axis='y',alpha=.2);savefig(fig,out,'I0252_final_holdout_comparison')


def report_and_index(out,config,items):
    d,acc,events,individual,quality,targets,summary,validation,vs,pairs,horizons=items
    science=config['science'];quick=science['quick'];new='GreenSoftmax010'
    lines=['I0252 — ampliación Green + Softmax β=0.10',
        'Se conservan los siete métodos anteriores y se añade GreenSoftmax010 en rep01–rep05 (dos prefijos en prueba).',
        'Las repeticiones son independientes entre sí; los métodos dentro de cada repetición comparten semillas CRN.',
        f'Configuración: 500 trayectorias de 45 trabajos/iteración, {science["max_iters"]} actualizaciones; 144 políticas.',
        'Inicio (0.12,0.10), paso .035/sqrt(t), clipping ±15, proyección [0,1]^2, modelo y alpha originales.',
        'β=0.10 es el parámetro de los pesos Softmax. El parámetro de señal BETA=4.0 no cambia.',
        '', 'Método y estimando:',
        'Green reutiliza cada banco histórico mediante likelihood ratios P_z/P_zgenerador; pesos LR recortados según el código original.',
        'Promedia valores y gradientes LR por banco; aplica softmax(values; beta=.10) y combina todos los gradientes de políticas.',
        'Pesos exactos: w_i=exp(0.10*(Vhat_i-max_j Vhat_j))/sum_j exp(0.10*(Vhat_j-max_k Vhat_k)).',
        'El beta multiplica los valores: .10 produce pesos más difusos que .50; no es el denominador de una temperatura.',
        'El gradiente combinado resta alpha una vez; después se aplica la actualización proyectada original.',
        'El objetivo de evaluación sigue siendo max_i E_z[Y_i] - alpha*z, aproximado por máximo de medias en un banco independiente.',
        'No es una media de máximos realizados. El objetivo evaluado tampoco se sustituye por log-sum-exp.',
        'Cada trayectoria de convergencia y su referencia (.70,.70) se evalúan con el mismo banco CRN de 1000 trayectorias, independiente de entrenamiento.',
        'La superficie oracle es la malla estimada original de 5000 trayectorias; no certifica el óptimo global.',
        'Las bandas describen la desviación estándar entre repeticiones, no intervalos de confianza.',
        '', 'Reutilización y tiempos:',
        f'{7*d.rep.nunique()} búsquedas antiguas reutilizadas sin ejecutar sus algoritmos; {d.rep.nunique()} búsquedas nuevas del método adicional.',
        'Los tiempos antiguos proceden de los replays exactos 012; los del método nuevo se miden en otra sesión secuencial con BLAS=1.',
        'La validación del lanzador ejecutó un prefijo de seis actualizaciones del antiguo beta=.50; no se reejecutaron sus cinco búsquedas completas.',
        'Las comparaciones de reloj entre sesiones son exploratorias; los presupuestos de trayectorias sí son directamente comparables.',
        'Los tiempos excluyen calentamiento, evaluación externa y generación de figuras. Bootstrap no cuenta como trayectorias nuevas del sistema.',
        'Fixed consume 500 trayectorias totales; GreenSoftmax010, como los otros métodos Fresh/Green, consume 50000 en 100 actualizaciones.',
        '', 'Comparación sobre las mismas evaluaciones de convergencia (promedios de repeticiones):']
    cols=['final_objective','accumulated_deficit_final',f'mean_objective_T{horizons[-1]}','total_simulated_trajectories','total_search_seconds','final_z3','final_z4']
    lines.append(individual.groupby('method')[cols].mean().reindex(METHODS).to_string(float_format=lambda x:f'{x:.6f}'))
    lines += ['', 'Green + Softmax β=0.10: resultados por repetición:',individual[individual.method==new].to_string(index=False,float_format=lambda x:f'{x:.6f}'),
        '', 'Tiempos a ε=0.5: inicio y confirmación (cinco decisiones consecutivas). No alcanzar el umbral mantiene tau faltante/censurada.']
    lines.append(events[(events.method.isin(['Green',new,'GreenSoftmax050']))&(events.epsilon_name=='0.5')][
        ['rep','method','kind','achieved','start_iteration','confirmation_iteration']].to_string(index=False))
    lines += ['', 'Las revalidaciones independientes usan 5000 trayectorias (100 en prueba); solo verifican candidatos preseleccionados y no reconstruyen un tiempo completo nuevo.',
        'Los umbrales se calculan contra la referencia en el mismo banco; no se mezclan con la estimación holdout de alta precisión.',
        'A(T) es la media de valores de decisiones visitadas, no una recompensa operacional acumulada; su déficit frente a la referencia puede ser negativo.',
        'Con cinco repeticiones, las diferencias observadas no demuestran una superioridad universal.']
    if not quick:
        fs=pd.read_csv(out/'tables/final_holdout_summary.csv')
        lines += ['', 'Evaluación final independiente: selección de política n=10000 y holdout n=50000 (distinta del estimando máximo directo en holdout).',
            fs.to_string(index=False,float_format=lambda x:f'{x:.6f}')]
        raw=pd.read_csv(out/'tables/final_holdout_by_rep.csv');newrows=raw[raw.method=='GreenSoftmax010']
        worst=newrows.loc[newrows.objective_holdout.idxmin()]
        lines += ['',f'Resultado a conservar: el nuevo método termina rep{int(worst.rep):02d} en (z3,z4)=({worst.z3:.6f},{worst.z4:.6f}), con valor holdout {worst.objective_holdout:.6f}.',
            'Su promedio final queda por debajo de Green+Softmax beta=.50 en estas cinco semillas. Esto es un resultado experimental; no se eliminan repeticiones desfavorables.',
            'Todas las decisiones y gradientes se verificaron finitos. Los diagnósticos de clipping y proyección se guardan; el método Green puede presentar alta variabilidad.']
    (out/'informe_cientifico.txt').write_text('\n'.join(lines)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>I0252: ocho métodos, cinco repeticiones</title>',
        '<style>body{font-family:system-ui;margin:2em;max-width:1600px}img{max-width:100%}td,th{padding:.6em;text-align:left}a{margin-right:.8em}</style>',
        '<h1>I0252: Green + Softmax β=0.10 añadido</h1>',
        '<p>Ocho métodos: siete resultados anteriores conservados y el nuevo método en las mismas cinco repeticiones independientes. '
        '100 iteraciones, 500 trayectorias por iteración. La prueba reducida se guarda por separado.</p>',
        '<p>Nuevo método: línea verde azulado discontinua. Las vistas de seis métodos mantienen Fixed, Fresh, Green, Fresh+Softmax β=0.50 y Green+Softmax β=0.50, y añaden Green+Softmax β=0.10.</p>',
        '<p><a href="informe_cientifico.txt">Informe y resultados</a><a href="README_execution.txt">Ejecución</a>'
        '<a href="tables/comparison_individual.csv">Resultados por repetición</a><a href="tables/final_holdout_by_rep.csv">Evaluación final</a>'
        '<a href="validation/output_verification.json">Verificación</a></p>']
    generated=set()
    for suffix,title in [('', 'Los ocho métodos'),('_six_methods','Vista de seis métodos')]:
        page += [f'<h2>{title}: todas las repeticiones</h2><table><tr><th>Repetición</th><th>Convergencia completa</th><th>Convergencia zoom</th><th>z3</th><th>z4</th><th>Trayectoria sobre oracle</th></tr>']
        for rep in sorted(d.rep.unique()):
            names=[f'I0252_convergence_rep{rep:02d}_full{suffix}',f'I0252_convergence_rep{rep:02d}_zoom{suffix}',
                f'I0252_z3_trajectory_rep{rep:02d}{suffix}',f'I0252_z4_trajectory_rep{rep:02d}{suffix}',f'I0252_oracle_trajectories_rep{rep:02d}{suffix}']
            page.append(f'<tr><td>rep{rep:02d}</td>'+''.join(f'<td><a href="figures/{n}.png">PNG</a><a href="figures/{n}.pdf">PDF</a></td>' for n in names)+'</tr>')
            generated.update(names)
        page.append('</table>')
    page += ['<h2>Curvas separadas por método y repetición</h2><p>Cada enlace abre una sola curva de convergencia o una sola trayectoria sobre oracle.</p><table><tr><th>Método</th>']
    page.append(''.join(f'<th>rep{r:02d}</th>' for r in sorted(d.rep.unique()))+'</tr>')
    for m in METHODS:
        row=f'<tr><td>{LABELS[m]}</td>'
        for rep in sorted(d.rep.unique()):
            names=[f'I0252_convergence_rep{rep:02d}_{m}',f'I0252_oracle_trajectories_rep{rep:02d}_{m}'];generated.update(names)
            row+='<td>'+''.join(f'<a href="figures/{n}.png">{label}</a>' for n,label in zip(names,['Convergencia','Oracle']))+'</td>'
        page.append(row+'</tr>')
    page.append('</table><h2>Convergencia, z3, z4 y oracle: rep01</h2>')
    preferred=['I0252_convergence_rep01_full','I0252_z3_trajectory_rep01','I0252_z4_trajectory_rep01','I0252_oracle_trajectories_rep01']
    for name in preferred:
        page += [f'<p><a href="figures/{name}.png">PNG</a><a href="figures/{name}.pdf">PDF</a></p><img loading="lazy" src="figures/{name}.png">']
    page.append('<h2>Figuras estadísticas, presupuestos y tiempos</h2>')
    for p in sorted((out/'figures').glob('*.png')):
        if p.stem in generated:continue
        page += [f'<h3>{p.stem}</h3><p><a href="figures/{p.stem}.png">PNG</a><a href="figures/{p.stem}.pdf">PDF</a></p><img loading="lazy" src="figures/{p.stem}.png">']
    (out/'index.html').write_text('\n'.join(page)+'\n')


def verify(out,config,items):
    from PIL import Image
    d=items[0];science=config['science'];reps=2 if science['quick'] else 5;K=science['max_iters']
    assert len(d)==reps*8*(K+1)
    assert set(d.method)==set(METHODS)
    assert ((d[['z3','z4']]>=0)&(d[['z3','z4']]<=1)).all().all()
    assert np.isfinite(d.objective).all()
    reuse=pd.read_csv(out/'validation/archived_search_reuse.csv');assert len(reuse)==7*reps and reuse.exact_original.all()
    root=Path(__file__).resolve().parent;donor=root/'resultados/012probing_anytime_budget'
    # Compare every retained scalar, not just final values; old timing/validation rows too.
    for name,keys in [('objectives_by_iteration',['rep','method','iteration']),('search_budget_runtime',['rep','method','iteration']),
                      ('crossing_validation',['rep','method','epsilon_name','kind','iteration'])]:
        if science['quick'] and name=='crossing_validation':continue
        a=pd.read_csv(out/'tables'/f'{name}.csv',dtype={'epsilon_name':str});a=a[a.method!='GreenSoftmax010'].sort_values(keys).reset_index(drop=True)
        b=pd.read_csv(donor/'tables'/f'{name}.csv',dtype={'epsilon_name':str});b=b[(b.rep<=reps)]
        if 'iteration' in b and name!='crossing_validation':b=b[b.iteration<=K]
        b=b.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-13,atol=1e-13)
    for name in ['accumulated_quality','hitting_times','comparison_individual','quality_at_horizons','target_summary','comparison_summary']:
        if science['quick']:continue
        a=pd.read_csv(out/'tables'/f'{name}.csv');a=a[a.method!='GreenSoftmax010'].sort_values(list(a.select_dtypes('object').columns)+['rep'] if 'rep' in a else list(a.select_dtypes('object').columns)).reset_index(drop=True)
        b=pd.read_csv(donor/'tables'/f'{name}.csv');b=b.sort_values(list(b.select_dtypes('object').columns)+['rep'] if 'rep' in b else list(b.select_dtypes('object').columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    if not science['quick']:
        source=root/'resultados/011probing_high_contrast_interior/comparisons/I0252'
        for rep in range(1,6):
            for name in ['finals','paired']:
                old=(source/f'rep_{rep:02d}_{name}.csv').read_bytes().splitlines()
                new=(out/f'comparisons/I0252/rep_{rep:02d}_{name}.csv').read_bytes().splitlines()
                assert new[:len(old)]==old,('Old final/paired rows changed',rep,name)
    pngs=sorted((out/'figures').glob('*.png'));assert len(pngs)==len(list((out/'figures').glob('*.pdf')))
    for p in pngs:
        with Image.open(p) as im:
            assert im.width>600 and im.height>400
            assert np.asarray(im.convert('RGB')).std()>10
        assert (p.with_suffix('.pdf')).read_bytes().startswith(b'%PDF-')
    preserved=root/'resultados/013probing_green_softmax010/validation/preserved_before.json'
    before=json.loads(preserved.read_text());changed=[p for p,h in before.items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed[:10]
    proof=dict(passed=True,objective_rows=len(d),methods=METHODS,repetitions=reps,iterations=K,
        archived_searches_reused=7*reps,new_searches=reps,old_values_budgets_timings_validation_unchanged=True,
        original_source_files_unchanged=True,prior_files_verified=len(before),all_previous_files_unchanged=True,
        png_files=len(pngs),pdf_files=len(pngs),all_raster_figures_nonblank=True,
        kernel_regression='kernel_regression.toml',crossing_rows=len(items[7]),
        new_method_search_paths_total=int(d[d.method=='GreenSoftmax010'].groupby('rep').cumulative_trajectories.max().sum()),
        source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in [
            '013probing_green_softmax010.jl','analyze_green_softmax010_013.py','plot_remaining_families_013.py',
            'audit_green_softmax010_013.jl','run_green_softmax010_013.py','probing_anytime_012_runtime.jl']})
    (out/'validation/output_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(dict(proof,complete=True,fingerprint=config['fingerprint']),indent=2)+'\n')
    print(json.dumps(proof,indent=2),flush=True)


def main(output):
    out=Path(output).resolve();config=tomllib.loads((out/'configs/experiment.toml').read_text());science=config['science']
    items=build_metrics(out,science)
    render_figures(out,items[0],items[1],items[2],items[4],items[7],items[8],science['max_iters'])
    supplemental_figures(out,items,science['quick'])
    report_and_index(out,config,items)
    (out/'README_execution.txt').write_text('''013: I0252, Green + Softmax beta=0.10 adicional.
Desde el repositorio, con el entorno Julia existente:
env GKSwstype=100 JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno 013probing_green_softmax010.jl --resume
Prueba reducida separada: añadir --quick-test (dos repeticiones, seis actualizaciones; 500 trayectorias de entrenamiento, 45 trabajos, evaluación de traza 1000 y validación 100).
Regenerar figuras: python3 analyze_green_softmax010_013.py resultados/013probing_green_softmax010
Los siete métodos anteriores se leen de 011/012 sin reejecutarlos. Solo el nuevo método se ejecuta para las cinco semillas originales.
Ningún archivo previo se sobrescribe. El estado se guarda por método y repetición, y por punto evaluado.
Los prefijos de prueba no son cinco repeticiones adicionales. Los tiempos nuevos se miden en sesión separada; los antiguos se conservan.
La evaluación final mantiene la selección independiente de política (10000) y holdout (50000).
Las vistas de ocho métodos y de seis métodos incluyen el método nuevo. Hay convergencia y oracle por curva/repetición para evitar superposición.
''')
    root=Path(__file__).resolve().parent
    for name in ['013probing_green_softmax010.jl','analyze_green_softmax010_013.py','analyze_anytime_012.py',
        '012probing_anytime_budget_analysis.jl','probing_anytime_012_runtime.jl','analyze_landscape_010.py',
        '011probing_high_contrast_interior.jl','probing_interior_011_diagnostics.jl','010probing_instance_landscape_search.jl',
        'probing_regime_model.jl','probing_experiments_common.jl','003probing_four_searches_softmax.jl','002probing_three_searches_n500_full45iters.jl']:
        shutil.copy2(root/name,out/'reproduction'/name)
    for name in ['Project.toml','Manifest.toml','python_requirements.txt']:
        src=root/'resultados/012probing_anytime_budget/reproduction'/name
        if src.exists():shutil.copy2(src,out/'reproduction'/name)
    verify(out,config,items)


if __name__=='__main__':main(sys.argv[1])

