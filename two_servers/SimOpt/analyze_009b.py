"""Inference/reporting for 009b, independent of its frozen simulation sources."""
from pathlib import Path
import sys,json,hashlib,warnings,time,math,shutil
try:import tomllib
except ModuleNotFoundError:import tomli as tomllib
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter('ignore',UserWarning)
    import pandas as pd
from scipy import stats
from probing_regime_structure import wilson,entropy

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'resultados/009probing_parameter_regimes'
IDS=['S000','S001','S010','S101','S111']
METHODS=['Fresh','Fresh + Softmax beta=0.5','Fresh + PSoft B=300','Green','Green + Softmax beta=0.5']
SHORT=dict(zip(METHODS,['Fresh','Softmax β=.50','PSoft B=300','Green','Green+Softmax β=.50']))
COLORS=dict(zip(METHODS,['black','royalblue','purple','forestgreen','darkorange']))
PRIMARY=[(METHODS[1],METHODS[0]),(METHODS[2],METHODS[0]),(METHODS[4],METHODS[3])]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def parts(out,folder,suffix):
    files=sorted((out/folder).glob(f'S*/*_{suffix}.csv'))
    return pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else pd.DataFrame()

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);result=np.empty(len(p))
    result[order]=np.minimum(1.,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return result

def compare(paired):
    rows=[]
    for (scenario,estimand,a,b),g in paired.groupby(['scenario','estimand','method_a','method_b'],sort=True):
        x=g.difference.to_numpy();n=len(x);sd=x.std(ddof=1);se=sd/np.sqrt(n);c=stats.t.ppf(.975,n-1)
        p=2*stats.t.sf(abs(x.mean()/se),n-1) if se>0 else (1. if x.mean()==0 else 0.)
        primary=estimand=='holdout' and (a,b) in PRIMARY
        family='primary_holdout_15' if primary else 'secondary_green_fresh_holdout' if estimand=='holdout' else 'secondary_direct_hardmax'
        rows.append(dict(scenario=scenario,estimand=estimand,method_a=a,method_b=b,n_reps=n,
            mean_difference=x.mean(),sd=sd,se=se,ci95_low=x.mean()-c*se,ci95_high=x.mean()+c*se,
            p_value=p,positive_proportion=np.mean(x>0),family=family,
            evaluation_mc_rms_se=np.sqrt(np.mean(g.evaluation_paired_se**2)),
            optimization_sd_approx=np.sqrt(max(0.,sd**2-np.mean(g.evaluation_paired_se**2)))))
    result=pd.DataFrame(rows);result['p_holm_family']=np.nan;result['p_holm_scenario_secondary']=np.nan
    for family,g in result.groupby('family'):
        # Fix alpha denominator at all 15 prespecified primary comparisons even
        # for explicitly partial output; absent tests conservatively receive p=1.
        size=15 if family=='primary_holdout_15' else 5 if family=='secondary_green_fresh_holdout' else 20
        result.loc[g.index,'p_holm_family']=holm(np.r_[g.p_value,np.ones(max(0,size-len(g)))])[:len(g)]
        crit=stats.t.ppf(1-.05/(2*size),g.n_reps-1)
        result.loc[g.index,'simultaneous_ci95_low']=g.mean_difference-crit*g.se
        result.loc[g.index,'simultaneous_ci95_high']=g.mean_difference+crit*g.se
    for _,g in result[result.family=='primary_holdout_15'].groupby('scenario'):
        result.loc[g.index,'p_holm_scenario_secondary']=holm(g.p_value)
    return result

def vector_test(row):
    delta=np.array([row.delta_G3,row.delta_G4]);cov=np.array([[row.covariance_33,row.covariance_34],[row.covariance_34,row.covariance_44]])
    values,vectors=np.linalg.eigh(cov);threshold=max(1e-16,values.max()*1e-10);positive=values>threshold;rank=int(positive.sum())
    if rank==0:return 0.,1.,0,np.nan
    projected=vectors.T@delta
    if np.linalg.norm(projected[~positive])>1e-8:return np.inf,0.,rank,np.nan
    statistic=np.sum(projected[positive]**2/values[positive]);n=row.n_reference
    f=(n-rank)*statistic/(rank*(n-1));p=stats.f.sf(f,rank,n-rank)
    return statistic,p,rank,rank*(n-1)/(n-rank)*stats.f.ppf(.95,rank,n-rank)

def classify_points(frame):
    frame=frame.copy();labels=[]
    for r in frame.itertuples():
        lo,hi=wilson(r.modal_count,r.selection_reps);modal=r.modal_count/r.selection_reps
        stable=(modal>=.90 and lo>=.80 and r.modal_group==r.best_A_group==r.best_B_group
            and r.modal_reference_status=='certified_inside' and r.standardized_gap_A>=3)
        uncertain=modal<=.80 and hi<=.90 and r.second_count/r.selection_reps>=.05
        if r.n_equivalence_groups==1 or stable:label='R1'
        elif uncertain and r.relative_gap_high<=.20 and r.cosine_low>=.95:label='R2'
        elif uncertain and (r.relative_gap_low>.20 or (r.cosine_high<.90 and r.local6_distance_low>0)):label='R3'
        elif uncertain:label='indeterminate_gradient'
        else:label='indeterminate_selection'
        labels.append((label,1-modal,lo,hi))
    frame[['regime','selection_uncertainty','modal_wilson_low','modal_wilson_high']]=pd.DataFrame(labels,index=frame.index)
    return frame

def structural_tables(out,cfg):
    g=parts(out,'diagnostics','gradients');g['vector_statistic']=np.nan;g['vector_p_value']=np.nan
    policies=parts(out,'diagnostics','policies')
    reference_A=policies[['scenario','point','policy','mean_A']].drop_duplicates(['scenario','point','policy'])
    maxima=reference_A.groupby(['scenario','point']).mean_A.max();lookup=reference_A.set_index(['scenario','point','policy']).mean_A
    for suffix in ['i','j']:
        g[f'policy_{suffix}_gap_A']=[maxima.loc[(r.scenario,r.point)]-lookup.loc[(r.scenario,r.point,getattr(r,'policy_'+suffix))] for r in g.itertuples()]
    g['both_nominal_epsilon']=(g.policy_i_gap_A<=.5)&(g.policy_j_gap_A<=.5)
    g['vector_rank']=0;g['vector_ellipse95_critical']=np.nan
    for r in g.itertuples():g.loc[r.Index,['vector_statistic','vector_p_value','vector_rank','vector_ellipse95_critical']]=vector_test(r)
    primary=g[g.primary];g['vector_p_holm_final']=np.nan
    g.loc[primary.index,'vector_p_holm_final']=holm(primary.vector_p_value.fillna(1))
    g['significant_global_rectangle']=((g.G3_simultaneous_low>0)|(g.G3_simultaneous_high<0)|(g.G4_simultaneous_low>0)|(g.G4_simultaneous_high<0))
    g.to_csv(out/'gradient_differences.csv',index=False)
    components=parts(out,'diagnostics','components');components.to_csv(out/'gradient_confidence_intervals.csv',index=False)
    values=parts(out,'diagnostics','values');values.to_csv(out/'tables/paired_policy_values.csv',index=False)
    candidates=parts(out,'diagnostics','candidates');candidates.to_csv(out/'tables/competitive_policies.csv',index=False)
    steps=parts(out,'diagnostics','steps');steps.to_csv(out/'projected_step_differences.csv',index=False)
    routes=parts(out,'diagnostics','routing');routes.to_csv(out/'tables/routing_disagreement.csv',index=False)
    bootstrap=parts(out,'diagnostics','bootstrap');bootstrap.to_csv(out/'tables/bootstrap_selection.csv',index=False)
    policies.to_csv(out/'tables/policy_values.csv',index=False)
    classification=classify_points(parts(out,'diagnostics','classification'));classification.to_csv(out/'tables/regime_classification.csv',index=False)
    selected=pd.read_csv(out/'selected_points.csv');rows=[]
    old=pd.read_csv(OLD/'tables/structural_scenario_summary.csv')
    final=classification[classification.primary]
    for s in IDS:
        f=final[final.scenario==s];r=dict(scenario=s,precision_source='009b: four targeted points, final N='+str(max(cfg['reference_sizes'])) if len(f) else '009: original ten-point grid, N=3000',
            high_precision_available=bool(len(f)),reference_paths=max(cfg['reference_sizes']) if len(f) else 3000,n_points=len(f) if len(f) else 10)
        if len(f):
            gp=g[(g.scenario==s)&g.primary].merge(f[['scenario','point','modal_i','modal_j']],on=['scenario','point'])
            gp=gp[((gp.policy_i==gp.modal_i)&(gp.policy_j==gp.modal_j))|((gp.policy_i==gp.modal_j)&(gp.policy_j==gp.modal_i))]
            rp=routes[routes.scenario==s].merge(gp[['scenario','point','policy_i','policy_j']],on=['scenario','point','policy_i','policy_j'])
            sp=steps[(steps.scenario==s)&steps.primary&(steps.iteration==100)].merge(gp[['scenario','point','policy_i','policy_j']],on=['scenario','point','policy_i','policy_j'])
            vp=values[(values.scenario==s)&values.primary].merge(gp[['scenario','point','policy_i','policy_j']],on=['scenario','point','policy_i','policy_j'])
            counts=f.regime.value_counts();regime='R3' if counts.get('R3',0) else 'R2' if counts.get('R2',0) else 'R1' if counts.get('R1',0)==len(f) else 'indeterminate'
            r.update(regime=regime,mean_gap=f.gap_A.mean(),mean_standardized_gap=f.standardized_gap_A.mean(),
                gap_A_reference_paths=cfg['candidate_paths'],value_difference_B_reference_paths=max(cfg['reference_sizes']),
                mean_abs_modal_value_difference_B=vp.delta_value.abs().mean(),rms_modal_value_paired_se_B=np.sqrt(np.mean(vp.paired_se**2)),
                selection_uncertainty=pd.to_numeric(f.selection_uncertainty).mean(),selection_error=f.nominal_selection_error.mean(),
                gradient_distance=gp.distance.mean(),relative_gradient_gap=gp.relative_gradient_gap.mean(),gradient_cosine=gp.cosine.mean(),
                projected_step100_difference=sp.projected_step_distance.mean(),routing_disagreement=rp.disagreement.mean(),
                R1_points=(f.regime=='R1').sum(),R2_points=(f.regime=='R2').sum(),R3_points=(f.regime=='R3').sum(),
                indeterminate_points=f.regime.str.startswith('indeterminate').sum())
        else:
            r.update(old[old.scenario==s].iloc[0].to_dict())
            oldpoints=pd.read_csv(OLD/'structural_diagnostics.csv');r['projected_step100_difference']=oldpoints[oldpoints.scenario==s].projected_step100_difference.mean()
            oldpairs=pd.read_csv(OLD/'policy_gradient_diagnostics.csv');oldpoints=oldpoints[oldpoints.scenario==s].copy()
            oldpoints['policy_i']=oldpoints[['modal_i','modal_j']].min(axis=1);oldpoints['policy_j']=oldpoints[['modal_i','modal_j']].max(axis=1)
            vp=oldpairs.merge(oldpoints[['scenario','point','policy_i','policy_j']],on=['scenario','point','policy_i','policy_j'])
            r.update(gap_A_reference_paths=3000,value_difference_B_reference_paths=3000,
                mean_abs_modal_value_difference_B=vp.gap_B.abs().mean(),rms_modal_value_paired_se_B=np.sqrt(np.mean(vp.gap_paired_se**2)))
        rows.append(r)
    summary=pd.DataFrame(rows);summary.to_csv(out/'scenario_diagnostics.csv',index=False)
    return dict(gradients=g,components=components,values=values,candidates=candidates,steps=steps,routes=routes,classification=classification,scenario=summary)

def scenario_comparison(out,structural,comparisons):
    model=pd.read_csv(out/'configs/scenarios.csv');frame=model.merge(structural,on='scenario')
    for i,(a,b) in enumerate(PRIMARY):
        g=comparisons[(comparisons.method_a==a)&(comparisons.method_b==b)&(comparisons.estimand=='holdout')]
        columns=['scenario','mean_difference','ci95_low','ci95_high','se','p_holm_family','positive_proportion']
        prefix=['softmax_fresh','psoft_fresh','green_softmax_green'][i]
        frame=frame.merge(g[columns].rename(columns={c:prefix+'_'+c for c in columns if c!='scenario'}),on='scenario',how='left')
    frame.to_csv(out/'scenario_comparison.csv',index=False);return frame

def audit(out,cfg,fingerprint,finals,paired,traces,structure):
    for file,digest in cfg['source_sha256'].items():assert sha(ROOT/file)==digest,f'Changed scientific source: {file}'
    selected=json.loads((out/'configs/point_selection.json').read_text())
    assert sha(out/'selected_points.csv')==selected['points_sha256']==cfg['points_sha256']
    for file,digest in selected['inputs'].items():assert sha(ROOT/file)==digest
    plan=tomllib.loads((out/'configs/confirmation_plan.toml').read_text());n=plan['n_reps']
    expected=len(plan['scenarios'])*n*5
    assert len(finals)==expected and not finals.duplicated(['scenario','rep','method']).any()
    assert all(len(g)==n for _,g in finals.groupby(['scenario','method']))
    assert np.isfinite(finals[['objective_holdout','objective_direct','gross_benefit','probing_cost','evaluation_se']]).all().all()
    assert ((finals[['z3','z4']]>=0)&(finals[['z3','z4']]<=1)).all().all()
    assert np.allclose(finals.gross_benefit-finals.probing_cost,finals.objective_holdout,atol=1e-10)
    assert (finals.objective_direct>=finals.objective_holdout-1e-10).all()
    assert (finals.search_paths==cfg['n_paths']*cfg['max_iters']).all()
    diag=parts(out,'confirmation','diagnostics');assert np.isfinite(diag[['g3','g4','gradient_norm']]).all().all()
    green=diag[diag.method.str.startswith('Green')];assert (green.banks==green.iteration).all()
    assert len(traces)==expected*(cfg['max_iters']+1)
    assert all(len(g)==cfg['max_iters']+1 for _,g in traces.groupby(['scenario','rep','method']))
    assert np.isfinite(traces[['objective','mc_se']]).all().all()
    lookup=finals.set_index(['scenario','rep','method'])
    for r in paired.itertuples():
        key=(r.scenario,r.rep);column='objective_'+r.estimand
        assert np.isclose(lookup.loc[key+(r.method_a,),column]-lookup.loc[key+(r.method_b,),column],r.difference,atol=1e-9)
    g=structure['gradients'];assert np.isfinite(g[['delta_G3','delta_G4','covariance_33','covariance_34','covariance_44']]).all().all()
    assert (g.covariance_33*g.covariance_44-g.covariance_34**2>=-1e-10).all()
    assert len(structure['classification'])==12*len(cfg['reference_sizes'])
    seeds=pd.concat([parts(out,'confirmation','seeds'),parts(out,'diagnostics','seeds')],ignore_index=True)
    assert seeds.seed.nunique()==len(seeds),'Seed reused between independent streams'
    old=set(pd.read_csv(OLD/'tables/seed_manifest.csv').seed)
    assert not old.intersection(set(seeds.seed)),'Seed overlaps 009'
    for s in plan['scenarios']:
        trace=seeds[(seeds.scenario==s)&(seeds.stream=='trace')];assert trace.seed.nunique()==n
    seeds.to_csv(out/'tables/seed_manifest.csv',index=False)
    prior=json.loads((out/'configs/preserved_files.json').read_text())
    for file,digest in prior.items():assert sha(ROOT/file)==digest,f'Previous result changed: {file}'
    result=dict(complete=set(plan['scenarios'])==set(IDS) and n>=20 and len(finals)>=500 and not cfg['quick'],
        quick=cfg['quick'],quick_workflow_complete=cfg['quick'] and len(finals)==expected,
        fingerprint=fingerprint,scenarios=plan['scenarios'],n_reps=n,searches=len(finals),structural_points=12,
        reference_sizes=cfg['reference_sizes'],trace_rows=len(traces),independent_seed_streams=len(seeds),
        previous_files_verified=len(prior),primary_comparisons=15,trace_banks_independent_between_replications=True)
    (out/'validation/result_audit.json').write_text(json.dumps(result,indent=2))
    return result

def analyze(out):
    started=time.perf_counter();out=Path(out).resolve();root=tomllib.loads((out/'configs/configuration.toml').read_text());cfg=root['science']
    structure=structural_tables(out,cfg);finals=parts(out,'confirmation','finals');paired=parts(out,'confirmation','paired')
    traces=parts(out,'confirmation','traces');coords=parts(out,'confirmation','coordinates');diag=parts(out,'confirmation','diagnostics')
    finals['search_excluding_bootstrap_seconds']=finals.search_seconds-finals.bootstrap_seconds
    finals.to_csv(out/'final_objectives.csv',index=False);paired.to_csv(out/'tables/paired_replications.csv',index=False)
    finals[['scenario','rep','method','search_seconds','search_excluding_bootstrap_seconds','bootstrap_seconds','trace_evaluation_seconds','final_evaluation_seconds','search_paths']].to_csv(out/'tables/computational_costs.csv',index=False)
    comp=compare(paired);comp.to_csv(out/'paired_comparisons.csv',index=False)
    data=coords.merge(diag.rename(columns={'iteration':'gradient_iteration','z3':'gradient_z3','z4':'gradient_z4'}).assign(iteration=lambda f:f.gradient_iteration-1),
        on=['scenario','phase','rep','method','iteration'],how='left')
    data=data.merge(traces[['scenario','phase','rep','method','iteration','objective','mc_se']],on=['scenario','phase','rep','method','iteration'])
    data.to_csv(out/'optimization_traces.csv',index=False);parts(out,'confirmation','stability').to_csv(out/'tables/stability.csv',index=False)
    summary=scenario_comparison(out,structure['scenario'],comp)
    result=audit(out,cfg,root['fingerprint'],finals,paired,traces,structure)
    make_plots(out,cfg,structure,finals,traces,coords,comp,summary)
    write_report(out,cfg,structure,finals,comp,summary,result)
    (out/'configs/analysis_environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,analysis_sha256=sha(__file__)),indent=2))
    (out/'logs/analysis_resources.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started),indent=2))
    if result['complete'] or result['quick_workflow_complete']:(out/'COMPLETE.json').write_text(json.dumps(result,indent=2))
    print('009b ANALYSIS COMPLETE',json.dumps(result),flush=True)

def modal_pairs(structure):
    f=structure['classification'];f=f[f.primary]
    g=structure['gradients'];g=g[g.primary].merge(f[['scenario','point','modal_i','modal_j']],on=['scenario','point'])
    return g[((g.policy_i==g.modal_i)&(g.policy_j==g.modal_j))|((g.policy_i==g.modal_j)&(g.policy_j==g.modal_i))]

def make_plots(out,cfg,structure,finals,traces,coords,comp,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'figure.dpi':130,'figure.figsize':(10,6),'font.size':10,'axes.grid':True,'grid.alpha':.18})
    footer=(f"{'QUICK TEST; ' if cfg['quick'] else ''}009b; n={cfg['n_paths']}; iterations={cfg['max_iters']}; jobs=45; βsoft=.50; PSoft B=300\n"
        f"Reference prefixes={cfg['reference_sizes']}; trace/selector/holdout={cfg['trace_paths']}/{cfg['selection_paths']}/{cfg['final_paths']}; independent trace banks across replications")
    def save(fig,name,note=''):
        fig.text(.01,.005,footer+('\n'+note if note else ''),fontsize=7,va='bottom')
        fig.tight_layout(rect=(0,.095,1,1));fig.savefig(out/'figures'/f'{name}.png');plt.close(fig)
    for s in IDS:
        g=traces[traces.scenario==s];nr=g.rep.nunique()
        for mode in ['sd','ci95']:
            fig,ax=plt.subplots()
            for m in METHODS:
                h=g[g.method==m].groupby('iteration').objective.agg(['mean','std','count'])
                width=h['std'] if mode=='sd' else stats.t.ppf(.975,h['count']-1)*h['std']/np.sqrt(h['count'])
                ax.plot(h.index,h['mean'],color=COLORS[m],label=SHORT[m]);ax.fill_between(h.index.to_numpy(float),(h['mean']-width).to_numpy(float),(h['mean']+width).to_numpy(float),color=COLORS[m],alpha=.12)
            ax.set(xlabel='Iteration',ylabel='Direct sample hard-max objective',title=f'{s}: {nr} independent repetitions — '+('mean ± SD' if mode=='sd' else 'pointwise 95% confidence intervals'))
            ax.legend(fontsize=8);save(fig,f'convergence_{mode}_{s}','Bands across independent repetitions; final selected-policy holdout is the primary endpoint.')
        g=finals[finals.scenario==s];fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,column,title in zip(axes,['objective_holdout','objective_direct'],['Independent selected-policy holdout','Secondary direct sample maximum']):
            ax.boxplot([g[g.method==m][column] for m in METHODS],tick_labels=[SHORT[m] for m in METHODS],showmeans=True)
            ax.set(ylabel='Objective',title=title);ax.tick_params(axis='x',labelrotation=45,labelsize=8)
        fig.suptitle(f'{s}: {nr} independent repetitions');save(fig,f'final_objectives_{s}')
        h=coords[coords.scenario==s];fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,column in zip(axes,['z3','z4']):
            for m in METHODS:
                a=h[h.method==m].groupby('iteration')[column].agg(['mean','std'])
                ax.plot(a.index,a['mean'],label=SHORT[m],color=COLORS[m]);ax.fill_between(a.index.to_numpy(float),(a['mean']-a['std']).to_numpy(float),(a['mean']+a['std']).to_numpy(float),color=COLORS[m],alpha=.1)
            ax.set(xlabel='Iteration',ylabel=column,ylim=(-.02,1.02),title=f'{s}: {column}, mean ± SD')
        axes[0].legend(fontsize=7);save(fig,f'probing_trajectories_{s}')
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,(a,b) in zip(axes,PRIMARY):
        h=comp[(comp.method_a==a)&(comp.method_b==b)&(comp.estimand=='holdout')].sort_values('scenario');pos=np.arange(len(h))
        ax.errorbar(h.mean_difference,pos,xerr=np.maximum(0,np.vstack([h.mean_difference-h.ci95_low,h.ci95_high-h.mean_difference])),fmt='o',color=COLORS[a],capsize=3)
        for k,r in enumerate(h.itertuples()):
            if r.p_holm_family<.05:ax.plot(r.mean_difference,k,'o',ms=11,mfc='none',mec='black')
        ax.axvline(0,color='black',lw=1);ax.set(yticks=pos,yticklabels=h.scenario,xlabel='Paired holdout difference',title=f'{SHORT[a]} − {SHORT[b]}')
    fig.suptitle(f"009b confirmation: {finals.rep.nunique()} replications per scenario")
    save(fig,'paired_forest','Marginal 95% t intervals; black rings indicate significance after Holm over EXACTLY the 15 primary comparisons.')
    modal=modal_pairs(structure);labels=modal.scenario+'/'+modal.point+' ('+modal.policy_i.astype(str)+'/'+modal.policy_j.astype(str)+')'
    fig,axes=plt.subplots(1,2,figsize=(14,6));pos=np.arange(len(modal))
    for ax,coord in zip(axes,[3,4]):
        values=modal[f'delta_G{coord}']
        error=np.vstack([values-modal[f'G{coord}_simultaneous_low'],modal[f'G{coord}_simultaneous_high']-values])
        ax.errorbar(values,pos,xerr=np.maximum(0,error),fmt='o',capsize=3);ax.axvline(0,color='black',lw=1)
        ax.set(yticks=pos,yticklabels=labels,xlabel=f'Paired gradient difference, coordinate {coord}',title='Final reference: simultaneous component intervals')
    save(fig,'paired_gradient_components','Pairs are the two modal nonequivalent selection groups; intervals cover the full frozen pair family, including unplotted pairs.')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,s in zip(axes,['S001','S010','S111']):
        g=structure['gradients'];g=g[g.scenario==s].copy();g['paired_se_norm']=np.sqrt(g.covariance_33+g.covariance_44)
        for point,h in g.groupby('point'):
            curve=h.groupby('n_reference').paired_se_norm.mean();ax.plot(curve.index,curve,'o-',label=point)
        ax.set(xscale='log',yscale='log',xlabel='Reference paths (correlated prefixes)',ylabel='Mean paired gradient SE norm',title=s);ax.legend(fontsize=8)
    save(fig,'reference_precision','Same reference-B prefixes; only final N supports main inference. Each curve averages the fixed pair set, not repeated independent estimates.')
    for s in ['S001','S010','S111']:
        fig,axes=plt.subplots(2,2,figsize=(11,10));g=structure['gradients'];g=g[(g.scenario==s)&g.primary]
        for ax,(point,h) in zip(axes.flat,g.groupby('point')):
            ids=sorted(set(h.policy_i)|set(h.policy_j));matrix=pd.DataFrame(np.eye(len(ids)),index=ids,columns=ids)
            for r in h.itertuples():matrix.loc[r.policy_i,r.policy_j]=matrix.loc[r.policy_j,r.policy_i]=r.cosine
            im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm');ax.set(xticks=np.arange(len(ids)),yticks=np.arange(len(ids)),xticklabels=ids,yticklabels=ids,title=f'{s}/{point}: raw-gradient cosine')
            ax.tick_params(axis='both',labelsize=6);ax.tick_params(axis='x',labelrotation=90);fig.colorbar(im,ax=ax,fraction=.045)
        save(fig,f'gradient_cosines_{s}','Point estimates from final reference; component/paired uncertainty is reported separately. Cosine alignment alone does not establish similarity.')
        fig,axes=plt.subplots(1,2,figsize=(13,5));g=structure['steps'];h=g[(g.scenario==s)&g.primary].merge(modal[['scenario','point','policy_i','policy_j']],on=['scenario','point','policy_i','policy_j'])
        early=h[h.iteration==1];pos=np.arange(len(early));axes[0].bar(pos-.16,early.raw_gradient_distance,width=.32,label='Before clipping');axes[0].bar(pos+.16,early.clipped_gradient_distance,width=.32,label='After clipping')
        axes[0].set(xticks=pos,xticklabels=early.point,ylabel='Gradient distance',title=f'{s}: competitive modal pair');axes[0].legend(fontsize=8)
        for point,pp in h.groupby('point'):
            line=axes[1].plot(pp.iteration,pp.projected_step_distance,'o-',label=point)[0]
            axes[1].vlines(pp.iteration,pp.step_ci95_low,pp.step_ci95_high,color=line.get_color(),alpha=.75)
        axes[1].axhline(.01,color='gray',ls=':');axes[1].set(xlabel='Update iteration t',ylabel='Projected decision distance',title='Original clipping/projection; bootstrap 95% intervals');axes[1].legend(fontsize=8)
        save(fig,f'clipping_projection_{s}','Bootstrap intervals are descriptive; equality frequencies and boundary contacts are saved for all frozen pairs.')
    for metric,ylabel in [('relative_gradient_gap','Relative difference between raw gradients'),('projected_step100_difference','Projected-step difference at t=100')]:
        for benefit,label in [('softmax_fresh_mean_difference','Softmax − Fresh'),('psoft_fresh_mean_difference','PSoft − Fresh')]:
            limit=max(1e-6,summary[benefit].abs().max());fig,ax=plt.subplots()
            sc=ax.scatter(summary.selection_uncertainty,summary[metric],c=summary[benefit],s=110,cmap='coolwarm',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),edgecolor='black')
            for r in summary.itertuples():ax.annotate(r.scenario+(' *' if not r.high_precision_available else ''),(r.selection_uncertainty,getattr(r,metric)),xytext=(5,6),textcoords='offset points',fontsize=9)
            ax.margins(.18);ax.set(xlabel='Selection uncertainty: 1−modal group frequency',ylabel=ylabel,title=f'009b structure/performance — {label}');fig.colorbar(sc,ax=ax,label='Mean confirmatory holdout improvement')
            save(fig,f'structural_map_{metric}_{benefit}','* S000/S101: original 009 ten-point grid. Others: four targeted 009b points. Supports/precision differ; descriptive only.')
    provenance=[]
    oldcfg=tomllib.loads((OLD/'configs/configuration.toml').read_text());oldstruct=pd.read_csv(OLD/'structural_diagnostics.csv')
    for s in IDS:
        assert cfg['models'][s]==oldcfg['science']['models'][s]
        source=OLD/f'characterization/oracle_{s}.csv';dest=out/f'tables/oracle_{s}_source009.csv'
        if source.exists():
            shutil.copyfile(source,dest);surface=pd.read_csv(dest);title='11×11 direct CRN oracle reused exactly from 009'
            oracle_seed=int(hashlib.sha256(f'Probing009-v1/{s}/oracle/0/surface/0'.encode()).hexdigest()[:15],16)+1
            record=dict(scenario=s,source=str(source.relative_to(ROOT)),sha256=sha(source),seed=oracle_seed,paths=oldcfg['science']['oracle_paths'],configuration_sha256=sha(OLD/'configs/configuration.toml'))
        else:
            surface=oldstruct[(oldstruct.scenario==s)&(oldstruct.point!='Z0')][['z3','z4','objective_A']].rename(columns={'objective_A':'objective'})
            surface.to_csv(dest,index=False);title='3×3 exploratory direct CRN reference from 009'
            record=dict(scenario=s,source='resultados/009probing_parameter_regimes/structural_diagnostics.csv',sha256=sha(OLD/'structural_diagnostics.csv'),
                seed=int(oldstruct[oldstruct.scenario==s].reference_A_seed.iloc[0]),paths=3000,configuration_sha256=sha(OLD/'configs/configuration.toml'))
        record['model_sha256']=pd.read_csv(out/'configs/scenarios.csv').set_index('scenario').loc[s,'model_sha256']
        provenance.append(record);matrix=surface.pivot(index='z4',columns='z3',values='objective');fig,ax=plt.subplots()
        im=ax.pcolormesh(matrix.columns,matrix.index,matrix,shading='auto',cmap='viridis');fig.colorbar(im,ax=ax,label='Direct sample J(z)')
        g=coords[coords.scenario==s]
        for m in METHODS:
            h=g[g.method==m]
            for _,run in h.groupby('rep'):ax.plot(run.z3,run.z4,lw=.7,alpha=.035,color=COLORS[m])
            avg=h.groupby('iteration')[['z3','z4']].mean();ax.plot(avg.z3,avg.z4,lw=2,color=COLORS[m],label=SHORT[m])
            ends=h[h.iteration==cfg['max_iters']];ax.scatter(ends.z3,ends.z4,s=13,color=COLORS[m],alpha=.6)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='z3',ylabel='z4',title=s+': '+title);ax.legend(fontsize=7)
        save(fig,f'oracle_trajectories_{s}','Each surface uses its OWN verified model and original 009 reference seed; bold mean, faint individual trajectories, dots endpoints.')
    (out/'configs/oracle_provenance.json').write_text(json.dumps(provenance,indent=2))
    stability=pd.read_csv(out/'tables/stability.csv');fig,axes=plt.subplots(1,2,figsize=(13,5))
    for m in METHODS:
        h=stability[stability.method==m];axes[0].scatter(h.tail_temporal_sd,h.final_objective_holdout,color=COLORS[m],s=16,alpha=.5,label=SHORT[m])
        axes[1].scatter(h.total_variation,h.final_objective_holdout,color=COLORS[m],s=16,alpha=.5)
    axes[0].set(xlabel='Temporal SD over last 20 evaluated iterates',ylabel='Independent final holdout',title='Within-trajectory variation differs from between-replication variation')
    axes[1].set(xlabel='Total variation of probing trajectory',ylabel='Independent final holdout',title='Five scenarios pooled for visualization only');axes[0].legend(fontsize=7)
    save(fig,'stability','Temporal values share a CRN bank within replication; time points are not independent statistical observations.')

def write_report(out,cfg,structure,finals,comp,summary,result):
    classification=structure['classification'];finalclass=classification[classification.primary]
    gradients=structure['gradients'];finalgrad=gradients[gradients.primary]
    steps=structure['steps'];finalsteps=steps[steps.primary]
    values=structure['values'];finalvalues=values[values.primary]
    primary=comp[comp.family=='primary_holdout_15'];modal=modal_pairs(structure)
    sig=primary[primary.p_holm_family<.05];positive=sig[sig.mean_difference>0];negative=sig[sig.mean_difference<0]
    title='PRUEBA REDUCIDA' if cfg['quick'] else 'EXPERIMENTO COMPLETO' if result['complete'] else 'RESULTADO PARCIAL'
    report=[f'009b — VALIDACIÓN DE REGÍMENES: {title}',
        f"RESULTADO OBSERVADO: {len(finalgrad)} pares candidatos en 12 puntos con referencia final N={max(cfg['reference_sizes'])}; {int(finalgrad.significant_global_rectangle.sum())} diferencias de gradiente con al menos una componente cuyo IC simultáneo global excluye cero. En los pares de los dos grupos modales, {int(modal.significant_global_rectangle.sum())}/{len(modal)}. Clasificación final: "+', '.join(f'{k}={v}' for k,v in finalclass.regime.value_counts().items())+'.',
        f"CONFIRMACIÓN: {len(finals)} búsquedas, {len(result['scenarios'])} escenarios × {result['n_reps']} repeticiones nuevas × cinco métodos. Familia principal de 15 contrastes holdout: {len(positive)} mejoras y {len(negative)} deterioros después de Holm; {15-len(sig)} sin diferencia concluyente. No rechazo no demuestra equivalencia.",
        'Objetivo preservado: J(z)=max_i E_z[Y_i]−alpha^T z, política elegida ex ante. Softmax/PSoft cambian la combinación de gradientes; todos se evalúan por simulación directa del mismo objetivo. No se confunde con E[max_i Y_i] de SimOpt009 ni con su estimador por máximo realizado.',
        'La configuración se leyó y verificó del TOML científico guardado por 009. No se reconstruyeron manualmente niveles de parámetros ni se redefinieron constantes. Se importó search de 009 y sus generadores, scores, LR, pesos y projected_step originales. Las cinco trayectorias originales de 100×500 se reprodujeron exactamente. También pasaron las regresiones de configuraciones, marginales, LR, covarianza emparejada, bootstrap conjunto y namespaces.',
        f"Se verificaron SHA de {result['previous_files_verified']} archivos previos sin escribirlos. La diferencia emparejada ya se calculaba correctamente en 009: su SE usaba (Yi−Yj)S; no se detectó ni corrigió un supuesto error de independencia. 009b añade la covarianza vectorial completa, referencias más grandes, inferencia simultánea y pasos en t=1,25,50,100.",
        'DISEÑO Y PRECISIÓN',
        'Puntos: Z0; máxima incertidumbre de selección; máximo desvío relativo de gradientes restante; iterado más próximo a la mediana de las últimas 20 iteraciones piloto, todos los métodos/repeticiones igualmente. Solo se usaron diagnósticos y coordenadas, nunca ganancias. selected_points.csv, point_selection.json y sus SHA quedaron congelados antes de simular.',
        f"Candidatos seleccionados con {cfg['candidate_paths']} caminos A nuevos y {cfg['selection_reps']} bancos nuevos por n=50,100,500, independientes de B. Se incluyeron top5 A, representantes de sus dos primeros grupos y todas las políticas seleccionadas por los bancos. Las políticas raramente seleccionadas pueden no ser verdaderamente competitivas: no se identifica la ganadora finita como óptima poblacional.",
        f"Referencia B final: {max(cfg['reference_sizes'])} caminos por punto; niveles {cfg['reference_sizes']} son prefijos CORRELACIONADOS de la misma muestra. Se completaron todos, sin parar por resultados favorables. Solo el último nivel alimenta la inferencia principal. La referencia A y los bancos de ranking conservan su tamaño: aumentar B no elimina toda su incertidumbre.",
        'Diferencias de gradientes: mean((Yi−Yj)S), con el mismo score y camino. Covarianza guardada es cov((Yi−Yj)S)/N, de la MEDIA estimada; covariance_34 es la covarianza cruzada. IC t marginales y rectángulos Bonferroni globales sobre todas las componentes de todos los pares congelados de los 12 puntos. Los límites de norma derivan de esos rectángulos y toleran matrices singulares.',
        'Hotelling/F y p Holm vectoriales se incluyen como análisis adicional aproximado por CLT (no exacto bajo distribuciones de colas); su familia contiene los pares finales y se mantiene separada de la familia de componentes. Los IC de niveles intermedios y bootstrap de cosenos/pasos describen precisión; no autorizan inferencia secuencial. Las etiquetas R1/R2/R3 conservan los umbrales y límites locales de seis componentes de 009 y son diagnósticos operacionales, no pruebas de igualdad poblacional.',
        'Grupos equivalentes se refieren a rewards observados iguales en A/B hasta 1e−8, no a identidad poblacional. Epsilon=.50. Una referencia A empatada no define dirección de ranking: su frecuencia de inversión se deja sin definir y se guardan empates aparte. Las probabilidades bootstrap son frecuencias condicionales de selección, no probabilidades bayesianas de optimalidad.',
        'Los bancos independientes de ranking también contribuyen a elegir candidatos. Sus inversiones son frecuencias descriptivas de esos bancos, no probabilidades poblacionales sin sesgo después de selección. La referencia B inferencial permanece independiente de todos ellos.',
        f"Búsqueda por método: {cfg['n_paths']} caminos × {cfg['max_iters']} iteraciones, 45 trabajos, Z0=(.12,.10), eta=.035/sqrt(t), clip ±15 y proyección [0,1]^2; beta=.50, PSoft B=300, sin early stopping. Se preservan CRN, materialización en el z propio y guardas de historia Green por escenario/hash.",
        f"Convergencia: TODOS los iterados 0:{cfg['max_iters']}, {cfg['trace_paths']} caminos por escenario/repetición. El banco se comparte entre los cinco métodos e iterados de esa repetición y es independiente entre repeticiones. Se separan figuras media±SD de IC t95% puntuales. La dispersión entre repeticiones incluye optimización y error MC de evaluación; no se confunde con SD temporal.",
        f"Final: selector {cfg['selection_paths']} y holdout {cfg['final_paths']} independientes de búsqueda, convergencia y entre sí. CRN entre métodos dentro de repetición; independencia entre repeticiones. Holdout de una política seleccionada es la métrica principal. El hard-max muestral secundario tiene posible sesgo optimista y no estima exactamente la misma cantidad.",
        'Familia principal EXACTAMENTE 15 contrastes holdout: Softmax−Fresh, PSoft−Fresh y GreenSoftmax−Green × cinco escenarios. Holm global, IC t95% marginales y Bonferroni simultáneos, SD, SE y proporción positiva. Ajustes por escenario son secundarios. Green−Fresh holdout y los 20 contrastes hard-max constituyen familias secundarias separadas.',
        'P1. ¿LA MAYOR PRECISIÓN DISTINGUIÓ GRADIENTES?',
        f"Se detectaron {int(finalgrad.significant_global_rectangle.sum())}/{len(finalgrad)} diferencias por rectángulos globales finales; Hotelling/Holm adicional: {int((finalgrad.vector_p_holm_final<.05).sum())}/{len(finalgrad)}. Estos recuentos incluyen todos los pares candidatos, no solo los más cercanos al óptimo. No se interpreta una diferencia no detectada como igualdad."]
    for r in finalclass.itertuples():
        prefix=classification[(classification.scenario==r.scenario)&(classification.point==r.point)].sort_values('n_reference')
        progression=', '.join(f'{int(q.n_reference)}:{q.regime}' for q in prefix.itertuples())
        report.append(f"{r.scenario}/{r.point}: 009={r.old_regime}; nuevos prefijos [{progression}]; incertidumbre={float(r.selection_uncertainty):.3f}, límites finales de diferencia relativa [{r.relative_gap_low:.4f},{r.relative_gap_high:.4f}], coseno bootstrap [{r.cosine_low:.4f},{r.cosine_high:.4f}]."+(' P3 es ubicación nueva, sin comparación exacta previa.' if r.point=='P3' else ''))
    report.append('La comparación con 009 también cambia semillas y bancos A/ranking; no atribuye por sí sola cada cambio de etiqueta al tamaño B. La progresión de prefijos mantiene fijos candidatos y bancos A/ranking y permite observar la precisión, sin convertir niveles intermedios en pruebas confirmatorias. Una etiqueta R2 de escenario indica presencia en sus puntos estudiados, no uniformidad sobre todo z; los cuatro puntos nuevos tienen otro soporte que la grilla antigua.')
    report.append('P2. ¿VALORES CERCANOS CON GRADIENTES DIFERENTES?')
    close=finalgrad.merge(finalvalues,on=['scenario','point','policy_i','policy_j'],suffixes=('','_value'))
    close=close[(close.delta_value.abs()<=.5)&close.significant_global_rectangle&close.both_nominal_epsilon]
    report.append(f"Se observaron {len(close)} pares donde AMBAS políticas son epsilon-óptimas nominales según A, |delta valor B| estimado<=.50 y se detecta diferencia de gradiente por componentes globales. La cercanía y el conjunto epsilon son estimados: no prueban igualdad de valores ni que ambas políticas sean las mejores verdaderas.")
    for r in close.sort_values('distance',ascending=False).head(8).itertuples():
        report.append(f"{r.scenario}/{r.point}, {r.policy_i}/{r.policy_j}: delta valor={r.delta_value:.5f}, SE emparejado={r.paired_se:.5f}; deltaG=({r.delta_G3:.5f},{r.delta_G4:.5f}), norma={r.distance:.5f}, IC norma global=[{r.distance_low:.5f},{r.distance_high:.5f}].")
    report.append('P3. ¿SOBREVIVEN AL CLIPPING Y LA PROYECCIÓN?')
    h=finalsteps.merge(finalgrad[['scenario','point','policy_i','policy_j','significant_global_rectangle']],on=['scenario','point','policy_i','policy_j'])
    for t,g in h.groupby('iteration'):
        significant=g[g.significant_global_rectangle]
        report.append(f"t={t}: distancia bruta media={g.raw_gradient_distance.mean():.5f}, después de clip={g.clipped_gradient_distance.mean():.5f}, entre decisiones={g.projected_step_distance.mean():.6f}; entre {len(significant)} pares con deltaG detectado, {int((significant.projected_step_distance<=1e-12).sum())} pasos puntuales idénticos y {int((significant.projected_step_distance>.01).sum())} distancias>.01. Frecuencia bootstrap media de pasos idénticos={g.identical_update_bootstrap_frequency.mean():.3f}.")
    modalsteps=finalsteps.merge(modal[['scenario','point','policy_i','policy_j']],on=['scenario','point','policy_i','policy_j'])
    for t,g in modalsteps.groupby('iteration'):
        report.append(f"Solo pares de los dos grupos modales, t={t}: distancia de paso media={g.projected_step_distance.mean():.6f}, máximo={g.projected_step_distance.max():.6f}; {int((g.projected_step_distance>.01).sum())}/{len(g)} distancias>.01. Estos pares son más representativos de la incertidumbre de selección que el promedio sobre todos los candidatos raramente seleccionados.")
    report.append('Las frecuencias anteriores son remuestreos contrafactuales de gradientes de políticas, no frecuencias de elección del algoritmo. Los IC bootstrap y la escala .01 son descriptivos; no se presentan como prueba global de equivalencia de decisiones. Contactos de frontera por par y por trayectoria se guardan por separado.')
    for index,question in [(0,'P4. SOFTMAX FRENTE A FRESH'),(1,'P5. PSOFT FRENTE A FRESH'),(2,'P6. GREEN+SOFTMAX FRENTE A GREEN')]:
        report.append(question);a,b=PRIMARY[index]
        for r in primary[(primary.method_a==a)&(primary.method_b==b)].itertuples():
            verdict='mejora detectada' if r.p_holm_family<.05 and r.mean_difference>0 else 'deterioro detectado' if r.p_holm_family<.05 else 'sin diferencia concluyente'
            report.append(f"{r.scenario}: delta={r.mean_difference:.6f}, SD={r.sd:.6f}, SE={r.se:.6f}, IC95% [{r.ci95_low:.6f},{r.ci95_high:.6f}], p nominal={r.p_value:.6g}, p Holm15={r.p_holm_family:.6g}, proporción positiva={r.positive_proportion:.3f}: {verdict}.")
    report.append('P7. MULTIPLICIDAD Y ANÁLISIS ADICIONALES')
    report.append(f"Entre los 15 contrastes principales: {int((primary.p_value<.05).sum())} nominalmente significativos y {len(sig)} tras Holm. Los intervalos principales de forest plot son marginales; los simultáneos están en paired_comparisons.csv. Una diferencia puntual positiva no constituye superioridad.")
    for r in comp[comp.family=='secondary_green_fresh_holdout'].itertuples():
        report.append(f"Adicional Green−Fresh, {r.scenario}: delta={r.mean_difference:.6f}, IC95% [{r.ci95_low:.6f},{r.ci95_high:.6f}], p Holm5 secundario={r.p_holm_family:.6g}.")
    report.append('P8. ¿S101 REPRODUCE LA SEÑAL EXPLORATORIA DE 009?')
    old=pd.read_csv(OLD/'paired_comparisons.csv');old=old[(old.phase=='pilot')&(old.scenario=='S101')&(old.estimand=='holdout')&(old.method_a==METHODS[1])].iloc[0]
    new=primary[(primary.scenario=='S101')&(primary.method_a==METHODS[1])].iloc[0]
    report.append(f"Piloto009: delta={old.mean_difference:.6f}, IC95% [{old.ci95_low:.6f},{old.ci95_high:.6f}], cinco repeticiones, exploratorio. 009b NUEVO: delta={new.mean_difference:.6f}, IC95% [{new.ci95_low:.6f},{new.ci95_high:.6f}], {new.n_reps} repeticiones, p Holm15={new.p_holm_family:.6g}. "+('Se confirma una mejora en este escenario y diseño.' if new.mean_difference>0 and new.p_holm_family<.05 else 'La señal positiva previa no queda confirmada por el criterio principal.'))
    report.append('P9–P10. ESTRUCTURA, ACTUALIZACIONES Y DESEMPEÑO')
    correlations=[];oldstructure=pd.read_csv(OLD/'tables/structural_scenario_summary.csv')
    benefits=summary[[c for c in summary if c=='scenario' or c.endswith('_mean_difference')]]
    comparable=oldstructure[oldstructure.scenario.isin(IDS)].merge(benefits,on='scenario')
    for source,frame in [('009_common_grid_N3000',comparable),('009b_targeted_four_points',summary[summary.high_precision_available])]:
        for x in ['selection_uncertainty','relative_gradient_gap','routing_disagreement']+(['projected_step100_difference'] if 'projected_step100_difference' in frame else []):
            for y in ['softmax_fresh_mean_difference','psoft_fresh_mean_difference','green_softmax_green_mean_difference']:
                f=frame[[x,y]].dropna();rho=stats.spearmanr(f[x],f[y]).statistic if len(f)>=3 and f[x].nunique()>1 and f[y].nunique()>1 else np.nan
                correlations.append(dict(source=source,x=x,y=y,n=len(f),rho=rho,interpretation='descriptive only; no causal claim'))
    pd.DataFrame(correlations).to_csv(out/'tables/structure_performance_correlations.csv',index=False)
    report.append('La tabla de cinco escenarios identifica explícitamente la fuente estructural: S000/S101 conservan 009 (grilla10 puntos,N3000); S001/S010/S111 usan 009b (cuatro puntos elegidos,N final). Esos soportes no son intercambiables. Las correlaciones se separan: cinco escenarios con la grilla antigua común, y tres con diagnósticos nuevos comparables. No se calcula una correlación causal mezclando regiones/precisiones distintas.')
    for r in summary.itertuples():
        report.append(f"{r.scenario}: A={r.A}, B={r.B}, C={r.C}; incertidumbre={r.selection_uncertainty:.4f}, diferencia relativa={r.relative_gradient_gap:.4f}, paso100={r.projected_step100_difference:.6f}, desacuerdo routing={r.routing_disagreement:.4f}; fuente={r.precision_source}.")
    report.append('La incertidumbre de selección por sí sola no basta para atribuir una ganancia: importan diferencias de gradientes, su precisión, clipping/proyección y el rango de z visitado. Estas son interpretaciones del mecanismo y asociaciones descriptivas; cinco escenarios no identifican una ley general ni efectos factoriales separados. Diferencia entre significaciones por escenario tampoco prueba heterogeneidad estadística del efecto.')
    report.append('ESTABILIDAD Y COSTOS')
    stability=pd.read_csv(out/'tables/stability.csv')
    aggregate=stability.groupby(['scenario','method']).agg(final_holdout_mean=('final_objective_holdout','mean'),between_replication_sd=('final_objective_holdout','std'),
        best_visited_mean=('best_visited_objective','mean'),tail_mean=('tail_mean_objective','mean'),mean_temporal_sd=('tail_temporal_sd','mean'),
        mean_tail_mc_se=('tail_rms_mc_se','mean'),mean_total_variation=('total_variation','mean'),boundary_frequency=('boundary_frequency','mean'),search_seconds=('search_seconds','mean')).reset_index()
    aggregate.to_csv(out/'tables/stability_summary.csv',index=False)
    report.append('Mejor visitado=max sobre evaluaciones de los 101 iterados, potencialmente optimista por selección del mejor tiempo. Media/SD de las últimas20 corresponden a t81:100 y son variación temporal sobre un banco CRN; no equivalen a 20 repeticiones independientes. La dispersión final usa repeticiones independientes y se registra el SE MC por separado. Una trayectoria más suave no demuestra mejor convergencia.')
    report.append('Tiempos de búsqueda, bootstrap, convergencia y evaluación final están separados por método/repetición; diagnóstico y generación de gráficos tienen registros adicionales. search_seconds incluye bootstrap_seconds; search_excluding_bootstrap_seconds permite comparar tiempos sin doble conteo. Búsquedas concurrentes con ocho hilos y BLAS=1; search preserva la imputación de payloads compartidos de 009. Los tiempos no son benchmarks aislados. Las evaluaciones quedan fuera del presupuesto de búsqueda de 500×100 por método. Green mantiene el mismo presupuesto de nuevas trayectorias pero relee sus bancos históricos.')
    report.append('P11. ¿AMPLIAR LA FAMILIA DE INSTANCIAS?')
    report.append('La ampliación de tiempos de servicio, complejidades y capacidad de mejora de señales sigue siendo una pregunta experimental razonable para ampliar cobertura estructural, no una conclusión de superioridad del suavizado. Antes deben fijarse niveles y criterios sin observar ganadores. Si predominan etiquetas indeterminadas, separar primero limitaciones de precisión de A, normas/cosenos y tamaño de efectos; no fabricar R3 ni elegir exclusivamente casos positivos. No se ejecutó esa segunda familia aquí.')
    report.append('P12. ¿INVESTIGAR SOFTMAX ADAPTATIVO?')
    report.append('Este estudio solo prueba beta=.50 fija y PSoft B=300. No valida una regla adaptativa. La incertidumbre de selección y los cambios efectivos de pasos pueden motivar una hipótesis futura; deben medirse juntos y evaluarse con un protocolo independiente, nuevos presupuestos/semillas y comparación contra las reglas fijas. No basta un resultado no significativo o una correlación entre cinco escenarios para justificar eficacia adaptativa.')
    report.extend(['REPRODUCCIÓN Y LIMITACIONES',f"Fingerprint: {result['fingerprint']}. Semillas únicas auditadas: {result['independent_seed_streams']}. Preservación de resultados anteriores verificada por SHA.",
        'Los métodos y repeticiones completos se reanudan sin repetir búsquedas ni evaluaciones. Se guardan checkpoints por método y bancos/estadísticas de referencia; una búsqueda interrumpida antes de quedar completa se reinicia, pues los cinco métodos avanzan conjuntamente para CRN. Todos los resultados negativos se conservan.',
        'Las superficies son auxiliares de 009, con modelo, semilla, tamaño y SHA documentados en oracle_provenance.json; S000 tiene 11×11 y los demás su referencia3×3 propia. No sustituyen los nuevos bancos independientes de convergencia/holdout.',
        'Se distinguen resultados observados, inferencia por familias, interpretaciones del mecanismo e hipótesis futuras. Estimadores LR pueden tener alta varianza; bootstrap y CLT aportan aproximaciones, no garantías finitas exactas. No se extrapola de cinco escenarios a todos los problemas de probing.'])
    (out/'report.txt').write_text('\n\n'.join(report)+'\n')
    command='env GKSwstype=100 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=8 julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno 009b_probing_regime_validation.jl'
    readme=['009b — REPRODUCCIÓN',title,'Desde la raíz del repositorio:',command+' --resume','',
        'Flujo completo con regresiones y prueba de reanudación:','python3 run_009b.py','',
        'Fases:',command+' --diagnostics --resume',command+' --confirm --resume',command+' --plots --report','',
        'Prueba reducida, identificada expresamente y separada de resultados científicos:',command+' --quick-test --resume','',
        'Confirmación filtrada (se conservan los demás archivos):',command+' --confirm --scenario S101 --n-reps 20 --resume','',
        'Antes del informe conjunto, restaurar el plan de cinco escenarios con --confirm --resume; se reutilizan todos los completos.',
        'Cambiar presupuestos o fuentes exige otra subcarpeta --output, por ejemplo resultados/009b_probing_regime_validation/otra_configuracion. No modificar 002–009 ni sus resultados.',
        'Los cinco escenarios y los 12 puntos se congelan por prepare_009b.py. --scenario filtra confirmación; A conserva sus 12 puntos preespecificados.',
        'Todos los iterados se evalúan con 1000 caminos; bancos CRN dentro de repetición e independientes entre repeticiones.',
        'Referencias 3000/10000/30000 son prefijos correlacionados; inferencia principal solo al final. Los checkpoints A guardan Y y scores para no repetir evaluaciones válidas.',
        'Presupuestos completos: 20 repeticiones × cinco escenarios × cinco métodos=500 búsquedas,500 caminos ×100 iteraciones,45 trabajos; selector5000,holdout30000.',
        'Familia principal:15 contrastes holdout con Holm. Otros estimandos/contrastes se presentan en familias secundarias.',
        'Entorno: Julia1.11.7, proyecto existente de003; Python numpy,pandas,scipy,matplotlib,tomli paraPython<3.11.',
        'Validaciones: tests/validate_parameter_regimes.jl,tests/validate_009b.jl,tests/validate_009b_statistics.py.',
        'Protocolo: PROTOCOLO_009b.txt. Parámetros, semillas, versiones, costos, SHA y evidencia de preservación en configs/,tables/,logs/,validation/.',
        'Los tiempos medidos con repeticiones concurrentes no son benchmarks aislados. No interpretar no significación como equivalencia.']
    (out/'README_reproduction.txt').write_text('\n'.join(readme)+'\n')

if __name__=='__main__':analyze(sys.argv[1])
