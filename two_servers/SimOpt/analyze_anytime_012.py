"""012 analysis of archived direct objectives and independently timed exact replays."""
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
    assert len(data)==data.rep.nunique()*7*(K+1)
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
        if method in ['Green','GreenSoftmax050']:assert np.array_equal(g.history_banks.iloc[1:],np.arange(1,K+1))
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
        for a,b in [('GreenSoftmax050','Green'),('Softmax010','Fresh'),('Softmax050','Fresh'),('PSoft300','Fresh')]:
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
                ax.set_xticks(range(7),[LABELS[m] for m in METHODS],rotation=35,ha='right',fontsize=8)
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
        ax.set_xticks(range(7),[LABELS[m] for m in METHODS],rotation=40,ha='right',fontsize=8);ax.set(title=f'T={T}',ylabel='A(T)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('I0252: accumulated objective across original repetitions');fig.tight_layout();savefig(fig,out,'accumulated_objective_distribution')
    phases=['generation_seconds','materialization_seconds','policy_seconds','gradient_seconds','bootstrap_seconds','update_seconds','other_seconds']
    totals=d.groupby(['rep','method'])[phases].sum().groupby('method').mean().reindex(METHODS)
    fig,ax=plt.subplots(figsize=(12,6));bottom=np.zeros(7)
    for phase in phases:
        vals=totals[phase].to_numpy();ax.bar(range(7),vals,bottom=bottom,label=phase.replace('_seconds','').replace('_',' '));bottom+=vals
    ax.set_xticks(range(7),[LABELS[m] for m in METHODS],rotation=25,ha='right');ax.set(ylabel='Mean sequential replay search time (seconds)',title='I0252: measured search costs; bootstrap counted once')
    ax.legend(loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();savefig(fig,out,'search_time_breakdown')
    fig,ax=plt.subplots(figsize=(10,5.5))
    for m in ['Green','GreenSoftmax050']:
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
        ax.set_xticks(range(7),[LABELS[m] for m in METHODS],rotation=35,ha='right',fontsize=8)
        ax.set(ylim=(0,1.16),ylabel='Fraction passing independent recheck',title='First candidate' if kind=='first' else 'Entire candidate five-decision run')
    fig.suptitle('ε=0.5: new independent-bank checks of selected candidates (conditional subset)');fig.tight_layout();savefig(fig,out,'crossing_validation_sensitivity')


def write_report(out,config,items):
    data,acc,events,individual,quality,targets,summary,validation,vs,pairs,horizons=items
    d=config['science'];orig=individual[individual.cohort=='original_011'];e=events[events.cohort=='original_011'];tar=targets[targets.cohort=='original_011'];p=pairs[pairs.cohort=='original_011']
    refs=pd.read_csv(out/'tables/reference_by_rep.csv');reference=tomllib.loads((out/'configs/reference.toml').read_text())
    report=[]
    def add(s=''):report.append(str(s))
    def tab(t):add(t.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    add('012 — I0252: CONVERGENCIA ANYTIME Y PRESUPUESTO DE SIMULACIÓN')
    add('PRUEBA REDUCIDA: no utilizar para conclusiones científicas.' if d['quick'] else 'Estudio exploratorio de las cinco repeticiones originales, sin seleccionar semillas favorables.')
    add(f'Configuración fija: 500 trayectorias, 45 trabajos, {d["max_iters"]} actualizaciones, 144 políticas; .035/sqrt(t), clipping +/-15 y proyección originales. Siete métodos.')
    add((out/'configs/I0252.toml').read_text())
    add('Objetivo: max_i E_z[Y_i]-alpha·z, selección ex ante; todas las curvas principales son simulación directa independiente de la búsqueda. No son log-sum-exp, LR ni máximo realizado por trayectoria.')
    add('Se reutilizan 35 búsquedas originales en ejecución completa. Los tiempos 011 por iteración NO existen: se hicieron 35 replays secuenciales necesarios para medirlos, con semillas, decisiones y gradientes verificados exactamente. Estos replays no son nuevas repeticiones estadísticas ni tiempos recuperados del proceso concurrente de 011.')
    add('Las repeticiones adicionales, si se solicitan, comienzan en6, llevan cohorte new_012 y tienen tablas agregadas separadas. Las figuras principales utilizan sólo original_011. La repetición ilustrativa es1, fijada anteriormente.')
    add('REFERENCIA Y COMPATIBILIDAD ESTADÍSTICA')
    add('z_ref=(.70,.70), rol anchor en auditoría geométrica independiente de011, congelada antes de indicadores; no es un óptimo global certificado.')
    hp=reference['high_precision'];add(f'Auditoría alta precisión reutilizada: {hp["holdout_paths"]} caminos holdout, selector {hp["selection_paths"]}; política {hp["selected_policy"]}, valor holdout={hp["objective_holdout"]:.6f}, SE={hp["se"]:.6f}, hard-max directo={hp["objective_direct"]:.6f}. Estimandos distintos: valor de política seleccionada versus máximo de medias; se reportan separados.')
    add('Este valor NO define los umbrales de las trazas. Cada referencia siguiente se evalúa en el MISMO banco CRN de1000 caminos de su repetición y con el mismo máximo de medias usado en sus iterados:')
    tab(refs)
    add('El máximo muestral tiene sesgo optimista potencial; la evaluación adicional y sus SE emparejados son condicionales a políticas seleccionadas y no resuelven toda la incertidumbre de selección. La SD entre repeticiones combina búsqueda y banco de evaluación; NO es SE Monte Carlo ni IC.')
    add('CONVENCIONES')
    add('z^t es posterior a t actualizaciones. t0 no consume datos de búsqueda: B(0)=0. Fixed genera500 antes del primer paso y B(t)=500 para t>=1; los demás B(t)=500t. Las evaluaciones externas, calentamiento y replays instrumentales no se suman como presupuesto adicional del método original.')
    add('El contador de políticas es número de simulaciones política×trayectoria, no productos algebraicos LR. Green procesa historia creciente y además conserva la llamada local_data original: sus evaluaciones de score por paso son500(t+1), pero sólo500 son nuevas trayectorias y72000 son nuevas simulaciones política×trayectoria. Los300 bootstrap PSoft usan columnas existentes:150000 extracciones por paso, cero trayectorias nuevas por remuestreo.')
    add('El driver conjunto original genera primitivas compartidas y reutiliza payloads cuando z coincide. joint_011_driver_accounting.csv audita estas llamadas físicas; no se suman sus ahorros entre métodos como si fueran presupuestos individuales. Los tiempos nuevos corresponden a métodos aislados secuencialmente, manteniendo los estimadores, sin compartir cómputo entre algoritmos.')
    add('Primera llegada: primer J>=Jref-epsilon. Llegada sostenida: inicio de primera racha de cinco decisiones; se guardan inicio y confirmación. Las CDF y RMST sostenidas usan la CONFIRMACIÓN, para representar cuándo existe evidencia de la racha. Tau censurado es missing; observed_iteration=100 sólo registra final de observación. Una contribución100 al RMST no imputa tau=100.')
    add('El RMST por iteraciones hasta100 usa todas las ejecuciones. La mediana es de la distribución empírica incondicional: se informa sólo si la CDF llega a.5. Los tiempos de cómputo medios entre éxitos están etiquetados como CONDICIONALES y no representan el tiempo de todas las ejecuciones.')
    add('Una primera llegada, incluso una racha de cinco decisiones, permite salir después de la región competitiva. No equivale a convergencia permanente. Por ello A(T), R(T) y todas las trayectorias acompañan a los tiempos de llegada.')
    add('P1. Iteraciones para alcanzar calidad competitiva, frecuencias, medianas identificables y RMST:')
    tab(tar[tar.epsilon_name!='relative_5pct'][['method','epsilon_name','kind','success_proportion','median_start_iteration','median_confirmation_iteration','restricted_mean_hitting_iteration']])
    add('Tolerancia relativa opcional=.05*(Jref-Jinitial), estimada por repetición; se guarda por separado porque no fija el mismo umbral absoluto entre bancos.')
    add('P2. Presupuesto hasta umbral, sin contabilizar evaluación externa:')
    tab(tar[tar.epsilon_name!='relative_5pct'][['method','epsilon_name','kind','success_proportion','median_hitting_simulation_budget','median_confirmation_simulation_budget']])
    add('Los métodos distintos de Fixed terminan con50000 trayectorias; Fixed con500. La probabilidad de éxito Fixed a budget500 presupone como máximo100 actualizaciones; no es una garantía de tiempo de cómputo ni una curva que consume500t.')
    add('P3. Heterogeneidad entre semillas; diferencia Green+Softmax menos Green por repetición:')
    cols=['rep','method_a','method_b']+[f'mean_objective_difference_T{T}' for T in horizons]+['final_objective_difference']
    gg=p[(p.method_a=='GreenSoftmax050')&(p.epsilon_name=='0.5')&(p.kind=='first')]
    tab(gg[cols])
    add('P4/P5. Comparación de llegada Green+Softmax frente a Green (a=Green+Softmax):')
    for en in [str(v) for v in EPS]:
        for kind in ['first','sustained5']:
            g=p[(p.method_a=='GreenSoftmax050')&(p.epsilon_name==en)&(p.kind==kind)]
            orders={o:g.loc[g.order==o,'rep'].astype(int).tolist() for o in g.order.unique()}
            add(f'epsilon={en}, {kind}: {orders}')
    add('a_earlier/b_earlier significan que ambos alcanzaron y se puede ordenar su confirmación. Un solo éxito y dos censuras se identifican por separado; no se inventa una diferencia de tau. Las ventajas transitorias de rep1 no se extrapolan a las otras semillas.')
    faster=p[(p.method_a=='GreenSoftmax050')&(p.order=='b_earlier')]
    add('No se observó Green antes que Green+Softmax en las tolerancias y definiciones de llegada estudiadas; hubo empates y ventajas de Green+Softmax dependientes de semilla. No implica imposibilidad de una ventaja Green con otras semillas.' if faster.empty else 'Green se adelantó en estos casos: '+str(faster[['rep','epsilon_name','kind']].to_dict('records')))
    add('P6. Fresh, Softmax y PSoft: diferencias de A(T) emparejadas, preservando todos los resultados:')
    ff=p[(p.method_b=='Fresh')&(p.epsilon_name=='0.5')&(p.kind=='first')]
    for m in ['Softmax010','Softmax050','PSoft300']:
        g=ff[ff.method_a==m]
        for T in horizons:
            x=g[f'mean_objective_difference_T{T}'];add(f'{m}-Fresh, T{T}: media={x.mean():.6f}, SD={x.std():.6f}, positivos={int((x>0).sum())}/{len(x)}; no inferencia confirmatoria.')
    add('Presupuesto nuevo igual entre estos cuatro métodos. Una diferencia de trayectoria no implica diferencia importante de calidad; PSoft añade cómputo bootstrap, no simulaciones del sistema. Ver orden por tolerancia/semilla en method_pair_comparisons.csv.')
    add('P7. Fixed conserva500 trayectorias; puede efectuar100 actualizaciones con reutilización LR y perder calidad. Su bajo presupuesto no mide automáticamente eficiencia para alcanzar un objetivo. En el gráfico de simulación sólo aparece una columna vertical en500 y un diamante final, sin una línea horizontal ficticia entre actualizaciones.')
    tab(orig[orig.method=='Fixed'][['rep','final_objective','total_simulated_trajectories','total_search_seconds']])
    add('P8. Simulaciones frente a tiempo:')
    tab(orig.groupby('method')[['total_simulated_trajectories','total_search_seconds','gradient_seconds','bootstrap_seconds']].mean().reset_index())
    add('Tiempo medido después de calentamiento, métodos secuenciales, BLAS1, fases y sobrecoste registrados; bootstrap se suma exactamente una vez. No incluye evaluación externa. Son replays nuevos, no las paredes concurrentes originales. OS, GC, caché, orden y una medición por semilla limitan cualquier conclusión de benchmark; no se presenta superioridad universal de velocidad.')
    add('P9. Calidad media acumulada y déficit, media±SD de todas las repeticiones:')
    tab(quality[quality.cohort=='original_011'].groupby(['method','horizon'])[['mean_objective','mean_deficit']].agg(['mean','std']).reset_index())
    for T in horizons:
        g=orig.groupby('method')[f'mean_objective_T{T}'].mean().sort_values(ascending=False)
        add(f'T{T}: mayor media observada={g.index[0]} ({g.iloc[0]:.5f}), menor={g.index[-1]} ({g.iloc[-1]:.5f}); ranking descriptivo de estas cinco semillas.')
    add('A incluye t0 y R=Jref-A sin truncar negativos. No son necesariamente ingresos realmente obtenidos por un sistema que opera mientras optimiza.')
    add('P10. Finales similares y transitorios distintos:')
    final=orig.groupby('method').final_objective.agg(['mean','std']);tab(final.reset_index())
    nonfixed=final.drop('Fixed');add(f'Rango de medias finales sin Fixed en banco de traza={nonfixed["mean"].max()-nonfixed["mean"].min():.6f}. El nivel de este banco no se mezcla con finales holdout de011. El valor medio acumulado captura pérdidas previas que el final omite.')
    rows=data[(data.cohort=='original_011')&(data.rep==1)&data.method.isin(['Green','GreenSoftmax050'])&data.iteration.isin([t for t in [0,20,50,80,100] if t<=d['max_iters']])]
    tab(rows[['method','iteration','objective','deficit']])
    add('P11. Ventajas/limitaciones ilustradas: Fixed ahorra muestras pero puede tener error LR persistente; Fresh renueva información y sus decisiones fluctúan; Green reutiliza rewards y crece en trabajo de score/historia, con transitorios dependientes de semilla; Softmax cambia pesos de gradiente, no el objetivo de evaluación; PSoft agrega bootstrap y puede parecerse mucho a Fresh; Green+Softmax puede estabilizar algunas búsquedas sin mejorar siempre llegada, A(T) o finales. Las tablas emparejadas muestran los casos concretos; no se establece un mecanismo causal a partir de suavidad visual.')
    add('INCERTIDUMBRE EN CRUCES')
    tab(vs[vs.cohort=='original_011'].groupby(['method','epsilon_name','kind']).agg(candidates=('all_candidates_meet_new_threshold','size'),fraction_revalidated=('all_candidates_meet_new_threshold','mean')).reset_index())
    add(f'Los candidatos se congelaron antes del nuevo banco independiente de{d["validation_paths"]} caminos por repetición. Se validan el primer cruce y su decisión anterior, y todas las cinco decisiones de la racha seleccionada. No se reconstruye un nuevo tiempo completo con este subconjunto. Un cambio de clasificación revela sensibilidad Monte Carlo; un éxito repetido tampoco certifica el umbral poblacional.')
    add('P12. Para un artículo: panel de las cinco semillas y ejemplo rep1; media±SD; presupuesto con Fixed correctamente marcado; runtime y desglose de costes con limitaciones; CDF primera/sostenida y censura; A(T)/R(T) a20/50/100; y revalidación independiente de cruces. Incluir referencia/procedencia, semillas, 144 políticas ex ante y cinco repeticiones como estudio exploratorio. No elegir sólo semillas favorables ni presentar SD como IC.')
    if (out/'validation/validation_collector_fix.toml').exists():add('ERROR DETECTADO Y CORREGIDO: el recolector paralelo nuevo de012 capturaba una variable rows compartida y mezclaba/duplicaba filas de la validación adicional. Se corrigió el alcance local, se añadieron comprobaciones de repetición, cardinalidad y unicidad y se reconstruyeron las tablas desde los puntos simulados válidos. Modelos, gradientes, trayectorias, tiempos y bancos de evaluación no cambiaron; los derivados previos afectados se conservaron en validation/superseded_collector/.')
    add('REPRODUCCIÓN: ver README_execution.txt. Ningún archivo anterior se modifica; hashes en validation/preserved_before.json y configs/input_sha256.toml. Tablas de cohortes adicionales separadas. Las fuentes y dependencias se conservan en reproduction/.')
    (out/'informe_cientifico.txt').write_text('\n'.join(report)+'\n')


def main(output):
    out=Path(output).resolve();config=tomllib.loads((out/'configs/experiment.toml').read_text());science=config['science']
    items=build_metrics(out,science)
    render_figures(out,items[0],items[1],items[2],items[4],items[7],items[8],science['max_iters'])
    write_report(out,config,items)
    figures=sorted(p.stem for p in (out/'figures').glob('*.png'))
    preferred=['anytime_convergence_iteration','anytime_convergence_simulation_budget','anytime_convergence_runtime',
        'anytime_individual_repetitions','anytime_mean_convergence','target_attainment_probability',
        'time_to_target_iterations','time_to_target_simulations','accumulated_objective','accumulated_regret',
        'probing_trajectories','trajectories_on_oracle']
    index=['<!doctype html><meta charset="utf-8"><title>I0252: anytime and simulation budget</title>',
        '<style>body{font-family:system-ui;margin:2em;max-width:1600px}img{max-width:100%}a{margin-right:1em}td,th{padding:.4em}</style>',
        '<h1>I0252: anytime convergence and simulation budget</h1>',
        '<p>Original five repetitions, seven unchanged algorithms. Example repetition 1 was predefined. Bands show between-repetition SD, not confidence intervals.</p>',
        '<p>Reference and iterates use the same 1000-path CRN bank within each repetition. High-precision holdout is reported separately.</p>',
        '<p><a href="informe_cientifico.txt">Scientific report (P1–P12)</a><a href="README_execution.txt">Reproduction</a><a href="configs/I0252.toml">Model parameters</a><a href="configs/reference.toml">Frozen reference</a></p>',
        '<p><a href="tables/comparison_summary.csv">Comparison summary</a><a href="tables/comparison_individual.csv">Every repetition</a><a href="tables/target_summary.csv">Target summary</a><a href="tables/hitting_times.csv">Events and censoring</a><a href="tables/method_pair_comparisons.csv">Paired comparisons</a><a href="tables/crossing_validation_summary.csv">Independent rechecks</a></p>']
    for name in preferred+[f for f in figures if f not in preferred]:
        index += [f'<h2>{name.replace("_"," ")}</h2><p><a href="figures/{name}.png">PNG</a><a href="figures/{name}.pdf">PDF</a></p><img loading="lazy" src="figures/{name}.png">']
    (out/'index.html').write_text('\n'.join(index)+'\n')
    readme='''012 I0252 anytime/budget analysis
Run from the original repository; Julia1.11.7, BLAS1. Existing Julia Project/Manifest and Python requirements are copied to reproduction/.
Complete/resume:
env GKSwstype=100 JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno 012probing_anytime_budget_analysis.jl --reuse-011 --resume
Quick test (separate folder, first six original updates): add --quick-test.
Replot only: add --plots-only --resume, or python3 analyze_anytime_012.py resultados/012probing_anytime_budget.
Optional NEW evaluation banks: --recompute-evaluations --output resultados/012probing_anytime_budget_recomputed. Original outputs remain unchanged. Repeat this flag on resume for that folder.
Optional additional independent repetitions: --extra-reps N --resume. New ids begin at6, cohort new_012, all original five remain. Main figures stay original-only; cohort summaries and individual CSVs are separate and clearly labelled.
Frozen scientific source/budget fingerprint prevents mixing incompatible calculations. Report/plot Python sources are hashed separately.
Timing requires a controlled replay because011 has no per-update timing. Original decisions and gradients must match exactly. Timings are not extra statistical replicates. Warmup/evaluation are excluded from search budgets. Fixed consumes500 paths; other methods50000. PSoft bootstrap does not generate system paths.
Candidate validation plans are frozen before new evaluation banks. Point and method checkpoints survive interruption. Completed timing replays are not repeated by --resume; warmup remains outside measurement. Evaluation points/checks are reused.
Tables keep missing hitting times for censored runs. Sustained-run start and confirmation are both stored; CDF/RMST use confirmation. Conditional runtime-to-target means are labelled successful-only. Main objective estimates and reference use common banks, distinct from the high-precision selected-policy holdout estimand.
Figures: English PNG/PDF; index.html. Report: informe_cientifico.txt. Main data: anytime_objectives_budget_runtime.csv, accumulated_quality.csv, hitting_times.csv, target_summary.csv, comparison_individual.csv, comparison_summary.csv, method_pair_comparisons.csv, external_evaluation_budget.csv, crossing_validation*.csv.
Optional pandas accelerator/Axes3D warnings in this environment do not affect the 2D plots or numerical simulation.
'''
    (out/'README_execution.txt').write_text(readme)
    for name in ['analyze_anytime_012.py','analyze_landscape_010.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'reproduction'/name)
    for name in ['011probing_high_contrast_interior.jl','probing_interior_011_diagnostics.jl','010probing_instance_landscape_search.jl','probing_regime_model.jl','probing_experiments_common.jl','003probing_four_searches_softmax.jl','002probing_three_searches_n500_full45iters.jl','009probing_parameter_regimes.jl','009b_probing_regime_validation.jl','PROTOCOLO_011.txt']:
        shutil.copy2(Path(__file__).with_name(name),out/'reproduction'/name)

    from importlib.metadata import version
    (out/'reproduction/python_requirements.txt').write_text('\n'.join(f'{p}=={version(p)}' for p in ['numpy','pandas','scipy','matplotlib','tomli','Pillow'])+'\n')
    data=items[0];checks=pd.read_csv(out/'validation/exact_replays.csv');original=checks[checks.cohort=='original_011']
    assert original.exact_original.all()
    assert len(original)==7*(2 if science['quick'] else 5)
    assert all((out/'figures'/f'{name}.png').is_file() for name in preferred)
    complete=dict(complete=True,quick=science['quick'],fingerprint=config['fingerprint'],original_repetitions=int(data[data.cohort=='original_011'].rep.nunique()),
        new_repetitions=int(data[data.cohort=='new_012'].rep.nunique()),archived_searches_reused=len(original),
        exact_timing_replays=len(checks),exact_original_replays=int(original.exact_original.sum()),objective_rows=len(data),
        png_files=len(figures),pdf_files=len(list((out/'figures').glob('*.pdf'))),
        analysis_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['analyze_anytime_012.py','analyze_landscape_010.py']},
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    (out/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(complete,indent=2))

if __name__=='__main__':main(sys.argv[1])
