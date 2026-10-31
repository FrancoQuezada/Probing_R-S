"""Homogeneous original-objective comparison; never launches Julia searches."""
from pathlib import Path
import argparse, hashlib, html, json, shutil, warnings
from html.parser import HTMLParser
warnings.filterwarnings('ignore',message='Pandas requires version')
import numpy as np
import pandas as pd
import tomli
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'resultados/017probing_green_five_method_comparison'
OLD=ROOT/'resultados/016probing_green_bank_variants'
SRC=ROOT/'resultados/011probing_high_contrast_interior'
METHODS=['Green','GreenSoftmax050','Exp50','Filter2','Rolling5']
LABELS={'Green':'Green Original','GreenSoftmax050':'Green + Softmax β=0.5',
        'Exp50':'Green Exponential ρ=0.5','Filter2':'Green Variance κ=2','Rolling5':'Green Rolling W=5'}
SHORT={'Green':'Original','GreenSoftmax050':'Softmax β=0.5','Exp50':'Exponential ρ=0.5','Filter2':'Variance κ=2','Rolling5':'Rolling W=5'}
COLORS=dict(zip(METHODS,['#222222','#7B3294','#D55E00','#009E73','#0072B2']))
MARKERS=dict(zip(METHODS,['o','^','D','s','P']))
plt.rcParams.update({'font.size':10,'figure.dpi':120,'savefig.dpi':180,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.titlepad':12})
Z95=1.959963984540054

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return pd.read_csv(OUT/'tables'/f'{name}.csv')
def save(fig,name,files):
    fig.tight_layout()
    for ext in ('png','pdf'):
        p=OUT/'figures'/f'{name}.{ext}';fig.savefig(p,bbox_inches='tight');files.append(p)
    plt.close(fig)
def legend(ax,small=False):
    ax.legend(fontsize=8 if small else 9,framealpha=.93)
def grid(ax):ax.grid(alpha=.18)
def select(df,rep,method):return df[(df.rep==rep)&(df.method==method)].sort_values('iteration')

def tables(coords,diag,tr,fin,runtime,pairs):
    merged=fin.merge(runtime,on=['rep','method'],validate='one_to_one')
    merged.to_csv(OUT/'tables/final_comparison.csv',index=False)
    runs=[]
    for rep in (1,2,3):
        for method in METHODS:
            z=select(coords,rep,method);d=select(diag,rep,method);t=select(tr,rep,method)
            steps=np.diff(z[['z3','z4']].to_numpy(),axis=0)
            f=merged[(merged.rep==rep)&(merged.method==method)].iloc[0]
            runs.append(dict(rep=rep,method=method,final_objective=f.objective,evaluation_se=f.evaluation_se,
                optimization_seconds=f.optimization_seconds,trace_mean_101=t.objective.mean(),
                trace_min_after10=t[t.iteration>=10].objective.min(),
                z3_total_variation=np.abs(steps[:,0]).sum(),z4_total_variation=np.abs(steps[:,1]).sum(),
                path_length=np.linalg.norm(steps,axis=1).sum(),mean_step_norm=d.step_norm.mean(),
                gradient_clipped_fraction=(d.clipping>0).mean(),projection_fraction=(d.projection>1e-12).mean(),
                mean_ess_fraction=d.ess_fraction.mean(),minimum_ess=d.ess.min(),
                mean_retained_banks=d.retained_banks.mean(),lr_clips=int(d.lr_clips.sum())))
    runs=pd.DataFrame(runs);runs.to_csv(OUT/'tables/trajectory_metrics.csv',index=False)
    rows=[]
    for method in METHODS:
        f=merged[merged.method==method];r=runs[runs.method==method]
        rows.append(dict(method=method,replications=3,objective_mean=f.objective.mean(),objective_sd=f.objective.std(ddof=1),
            conditional_mc_se_mean=f.evaluation_se.mean(),z3_mean=f.z3.mean(),z4_mean=f.z4.mean(),
            optimization_seconds_mean=f.optimization_seconds.mean(),optimization_seconds_sd=f.optimization_seconds.std(ddof=1),
            trace_mean_101=r.trace_mean_101.mean(),trace_mean_101_sd=r.trace_mean_101.std(ddof=1),
            path_length_mean=r.path_length.mean(),gradient_clip_fraction_mean=r.gradient_clipped_fraction.mean(),
            projection_fraction_mean=r.projection_fraction.mean(),ess_fraction_mean=r.mean_ess_fraction.mean(),
            retained_banks_mean=r.mean_retained_banks.mean(),lr_clips=int(r.lr_clips.sum()),
            logical_paths_per_run=50000,new_search_paths_this_study=150000 if method=='Exp50' else 0))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'tables/method_summary.csv',index=False)
    pp=[]
    for method in METHODS[1:]:
        p=pairs[pairs.method==method]
        pp.append(dict(method=method,difference_mean=p.difference.mean(),difference_sd=p.difference.std(ddof=1),
            conditional_paired_mc_se_mean=p.paired_mc_se.mean(),positive_reps=int((p.difference>0).sum()),reps=3))
    ps=pd.DataFrame(pp);ps.to_csv(OUT/'tables/paired_summary.csv',index=False)
    workload=tr.groupby(['method','source'],sort=False).agg(trace_rows=('iteration','size')).reset_index()
    workload['direct_points_reconstructed_this_study']=np.where(workload.source.str.startswith('017 new'),workload.trace_rows,0)
    workload.to_csv(OUT/'tables/trace_evaluation_workload.csv',index=False)
    return merged,runs,summary,ps

def trajectory_figures(coords):
    """Reuse the exact background and layout used by the original I0252/011
    figures. Only the five current saved trajectories are superimposed.
    contourf works on the original 21x21 nodes; no resampling or smoothing.
    """
    original_surface=SRC/'high/I0252_surface.csv'
    assert sha(original_surface)==sha(OUT/'configs/I0252_surface.csv')
    surface=pd.read_csv(original_surface)
    x=np.sort(surface.z3.unique());y=np.sort(surface.z4.unique())
    assert np.array_equal(x,y) and len(x)==21 and len(surface)==441
    matrix=surface.pivot(index='z4',columns='z3',values='objective').reindex(index=y,columns=x).to_numpy()
    for row in surface.itertuples():
        assert matrix[np.searchsorted(y,row.z4),np.searchsorted(x,row.z3)]==row.objective
    files=[];style={}
    # The imported module sets rcParams. Keep that original figure style local
    # so unrelated convergence, final-value and runtime figures are unaffected.
    with plt.rc_context():
        from analyze_landscape_010 import background, save as original_save
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
        for rep in (1,2,3):
            fig,ax=plt.subplots(figsize=(9,7))
            im=background(ax,surface,f'I0252: trayectorias, repetición {rep} — cinco métodos')
            for method in METHODS:
                z=select(coords,rep,method)
                assert np.array_equal(z.iteration.to_numpy(),np.arange(101))
                assert np.array_equal(z[['z3','z4']].iloc[0].to_numpy(),[.12,.10])
                ax.plot(z.z3,z.z4,color=COLORS[method],lw=1.6,alpha=.95,label=LABELS[method])
                end=ax.scatter(z.z3.iloc[-1],z.z4.iloc[-1],color=COLORS[method],marker=MARKERS[method],
                               s=65,edgecolor='white',lw=1.1,zorder=10)
                line=ax.lines[-1]
                assert np.array_equal(line.get_xdata(),z.z3.to_numpy())
                assert np.array_equal(line.get_ydata(),z.z4.to_numpy())
                assert np.array_equal(np.asarray(end.get_offsets())[0],z[['z3','z4']].iloc[-1].to_numpy())
            ax.scatter(.12,.10,marker='*',s=130,color='gold',edgecolor='black',zorder=11,label='Inicio común')
            fig.colorbar(im,ax=ax,label='J(z)')
            ax.legend(loc='upper left',bbox_to_anchor=(1.28,1),fontsize=8)
            assert im.cmap.name=='inferno' and ax.get_aspect()==1.
            assert ax.get_xlim()==(-.02,1.02) and ax.get_ylim()==(-.02,1.02)
            assert [line.get_label() for line in ax.lines]==[LABELS[m] for m in METHODS]
            style={'filled_levels':im.levels.tolist(),'line_levels':ax.collections[1].levels.tolist(),
                   'color_limits':list(im.get_clim()),'cmap':im.cmap.name,'xlim':list(ax.get_xlim()),
                   'ylim':list(ax.get_ylim()),'aspect':ax.get_aspect(),'colorbar_label':'J(z)'}
            stem=f'I0252_oracle_trajectories_rep{rep:02d}'
            original_save(fig,OUT,stem)
            files.extend(OUT/'figures'/f'{stem}.{ext}' for ext in ('png','pdf'))
    return files,style

def restore_trajectories_only():
    """A visualization-only path: no tables, numerical outputs or Julia calls."""
    before_path=OUT/'validation/trajectory_style_before.json'
    if before_path.exists():
        protected=json.loads(before_path.read_text())['protected_files']
    else:
        targets={f'figures/I0252_oracle_trajectories_rep{rep:02d}.{ext}' for rep in (1,2,3) for ext in ('png','pdf')}
        allowed=targets|{'reproduction/analyze_green_five_comparison_017.py'}
        protected={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and str(p.relative_to(OUT)) not in allowed}
    coords=read('coordinates');files,style=trajectory_figures(coords)
    for p in files:
        if p.suffix=='.png':
            a=np.asarray(Image.open(p).convert('RGB'));assert a.shape[0]>300 and a.shape[1]>500 and a.std()>10
        else:assert p.read_bytes().startswith(b'%PDF-')
    assert all((OUT/f).is_file() and sha(OUT/f)==h for f,h in protected.items())
    shutil.copyfile(Path(__file__),OUT/'reproduction'/Path(__file__).name)
    audit=dict(passed=True,original_renderer='analyze_landscape_010.py:background, imported by analyze_interior_011.py',
        original_renderer_sha256=sha(ROOT/'analyze_landscape_010.py'),original_surface=str(SRC/'high/I0252_surface.csv'),
        current_surface=str(OUT/'configs/I0252_surface.csv'),surface_sha256=sha(SRC/'high/I0252_surface.csv'),
        original_and_current_surfaces_identical=True,surface_points=441,surface_shape=[21,21],
        orientation='matrix[row=z4,column=z3]; x=z3, y=z4; checked every original grid node',
        rendering='original contourf(levels=24, cmap=inferno) + contour(levels=12, black, linewidth=.35, alpha=.5)',
        interpolation='No external interpolation, no resampling; original contourf on original grid',
        style=style,reps=[1,2,3],methods=METHODS,trajectory_points_verified=1515,
        numeric_outputs_unchanged=True,other_existing_output_files_unchanged=len(protected),
        simulations_executed=0,optimizations_executed=0,generated=[str(p.relative_to(OUT)) for p in files],
        command='python3 analyze_green_five_comparison_017.py --trajectories-only')
    (OUT/'validation/trajectory_style_restore.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

def plots(coords,diag,tr,fin,runtime,pairs,runs,summary):
    files,_=trajectory_figures(coords)
    for rep in (1,2,3):
        for dim in ('z3','z4'):
            fig,ax=plt.subplots(figsize=(9,4.9))
            for method in METHODS:
                z=select(coords,rep,method)
                ax.plot(z.iteration,z[dim],color=COLORS[method],label=LABELS[method],lw=1.7)
            ax.set(xlabel='Iteración completada',ylabel=dim,ylim=(-.02,1.02),title=f'I0252 · {dim} · rep{rep:02d}')
            legend(ax);grid(ax);save(fig,f'I0252_{dim}_trajectory_rep{rep:02d}',files)
        fig,ax=plt.subplots(figsize=(9.5,5.1))
        for method in METHODS:
            t=select(tr,rep,method);ax.plot(t.iteration,t.objective,color=COLORS[method],label=LABELS[method],lw=1.8)
        ax.set(xlabel='Iteración completada',ylabel='Objetivo original · evaluación directa CRN N=1000',
               title=f'I0252 · convergencia comparable · rep{rep:02d}')
        legend(ax);grid(ax);save(fig,f'I0252_convergence_rep{rep:02d}',files)
        # Archive truth: internal field is hard-max, including Softmax. LSE was
        # not recorded. Separate panels prevent confusion with direct curves.
        fig,axes=plt.subplots(5,1,figsize=(10,12),sharex=True)
        for ax,method in zip(axes,METHODS):
            d=select(diag,rep,method);t=select(tr,rep,method)
            ax.plot(d.iteration-1,d.internal_hard_max,c=COLORS[method],label='Estimador interno LR · máximo de medias')
            ax.plot(t.iteration,t.objective,c='gray',ls='--',lw=1.2,label='Evaluación directa independiente')
            ax.set(title=LABELS[method],ylabel='Objetivo');grid(ax)
            if method=='GreenSoftmax050':
                ax.text(.99,.04,'Gradiente suavizado β=0.5; valor interno archivado ≠ LSE',transform=ax.transAxes,ha='right',fontsize=8)
        axes[0].legend(fontsize=8);axes[-1].set_xlabel('Iterado evaluado t (objetivo interno de actualización k=t+1)')
        fig.suptitle(f'I0252 · valores internos y directos separados · rep{rep:02d}',y=1.005)
        save(fig,f'I0252_internal_vs_direct_rep{rep:02d}',files)
        f=fin[fin.rep==rep].set_index('method').loc[METHODS];x=np.arange(5)
        fig,ax=plt.subplots(figsize=(10,5))
        for j,method in enumerate(METHODS):
            ax.errorbar(j,f.loc[method,'objective'],yerr=Z95*f.loc[method,'evaluation_se'],fmt=MARKERS[method],
                        color=COLORS[method],capsize=4,ms=7)
        ax.set_xticks(x,[SHORT[m] for m in METHODS],rotation=20,ha='right')
        ax.set(ylabel='Objetivo final original · holdout N=50000',title=f'I0252 · rep{rep:02d} · IC 95% MC condicional')
        grid(ax);save(fig,f'I0252_final_objectives_rep{rep:02d}',files)
        fig,ax=plt.subplots(figsize=(9.5,4.9));p=pairs[pairs.rep==rep].set_index('method').loc[METHODS[1:]]
        for j,method in enumerate(METHODS[1:]):
            ax.errorbar(j,p.loc[method,'difference'],yerr=Z95*p.loc[method,'paired_mc_se'],fmt=MARKERS[method],
                        c=COLORS[method],capsize=4,ms=7)
        ax.axhline(0,c='black',lw=1);ax.set_xticks(np.arange(4),[SHORT[m] for m in METHODS[1:]],rotation=15,ha='right')
        ax.set(ylabel='Diferencia respecto de Green Original',title=f'I0252 · rep{rep:02d} · IC 95% MC pareado (CRN)')
        grid(ax);save(fig,f'I0252_paired_differences_rep{rep:02d}',files)
        fig,ax=plt.subplots(figsize=(10,4.9));r=runtime[runtime.rep==rep].set_index('method').loc[METHODS]
        for j,method in enumerate(METHODS):
            ax.bar(j,r.loc[method,'optimization_seconds'],color=COLORS[method],alpha=.85,
                   hatch='//' if method=='Exp50' else None)
        ax.set_xticks(x,[SHORT[m] for m in METHODS],rotation=20,ha='right')
        ax.set(ylabel='Optimización (s)',title=f'I0252 · rep{rep:02d} · tiempos archivados 012/016 y nuevos 017')
        ax.text(.02,.96,'Medición descriptiva: sesiones diferentes; sólo Exp50 ejecutado ahora',transform=ax.transAxes,va='top',fontsize=8)
        grid(ax);save(fig,f'I0252_computational_time_rep{rep:02d}',files)
        fig,axes=plt.subplots(2,2,figsize=(12,8.2))
        for ax,col,ylabel,log in zip(axes.ravel(),['step_norm','gradient_norm','ess','retained_banks'],
                                   ['Norma del paso','Norma del gradiente','ESS de pesos LR','Bancos retenidos'],[False,True,True,False]):
            for method in METHODS:
                d=select(diag,rep,method);ax.plot(d.iteration,d[col],c=COLORS[method],label=LABELS[method],lw=1.5)
            if log:ax.set_yscale('log')
            ax.set(xlabel='Actualización k',ylabel=ylabel);grid(ax)
        axes[0,0].legend(fontsize=8);fig.suptitle(f'I0252 · comportamiento de la búsqueda · rep{rep:02d}',y=1.01)
        save(fig,f'I0252_search_behavior_rep{rep:02d}',files)
    fig,ax=plt.subplots(figsize=(10,5.4))
    for method in METHODS:
        g=tr[tr.method==method].groupby('iteration').objective
        mu=g.mean();sd=g.std(ddof=1);x=mu.index.to_numpy()
        ax.plot(x,mu,c=COLORS[method],label=LABELS[method],lw=1.8)
        ax.fill_between(x,(mu-sd).to_numpy(),(mu+sd).to_numpy(),color=COLORS[method],alpha=.10)
    ax.set(xlabel='Iteración completada',ylabel='Objetivo original directo · media ± DE entre reps',
           title='I0252 · convergencia media · tres repeticiones (bandas ≠ IC)')
    legend(ax);grid(ax);save(fig,'I0252_convergence_mean_sd',files)
    s=summary.set_index('method').loc[METHODS];x=np.arange(5)
    fig,ax=plt.subplots(figsize=(10,5.2))
    for j,method in enumerate(METHODS):
        ax.errorbar(j,s.loc[method,'objective_mean'],yerr=s.loc[method,'objective_sd'],c=COLORS[method],
                    fmt=MARKERS[method],ms=8,capsize=5)
        vals=fin[fin.method==method].sort_values('rep').objective.to_numpy()
        ax.scatter(j+np.array([-.10,0,.10]),vals,c=COLORS[method],s=16,alpha=.45)
    ax.set_xticks(x,[SHORT[m] for m in METHODS],rotation=20,ha='right')
    ax.set(ylabel='Objetivo final · media ± DE entre tres reps',title='I0252 · variabilidad entre repeticiones (no IC de la media)')
    grid(ax);save(fig,'I0252_final_objectives_mean_sd',files)
    fig,axes=plt.subplots(1,3,figsize=(16,5.1),sharey=True)
    for ax,rep in zip(axes,(1,2,3)):
        p=pairs[pairs.rep==rep].set_index('method').loc[METHODS[1:]]
        for j,method in enumerate(METHODS[1:]):
            ax.errorbar(j,p.loc[method,'difference'],yerr=Z95*p.loc[method,'paired_mc_se'],fmt=MARKERS[method],c=COLORS[method],capsize=3)
        ax.axhline(0,c='black',lw=1);ax.set_xticks(np.arange(4),[SHORT[m] for m in METHODS[1:]],rotation=40,ha='right')
        ax.set(title=f'rep{rep:02d}',ylabel='Diferencia respecto de Green Original');grid(ax)
    fig.suptitle('I0252 · diferencias pareadas · IC 95% MC condicional por repetición',y=1.01)
    save(fig,'I0252_paired_differences_all_reps',files)
    fig,ax=plt.subplots(figsize=(10,5.1))
    for j,method in enumerate(METHODS):
        ax.bar(j,s.loc[method,'optimization_seconds_mean'],yerr=s.loc[method,'optimization_seconds_sd'],
               color=COLORS[method],capsize=4,hatch='//' if method=='Exp50' else None)
    ax.set_xticks(x,[SHORT[m] for m in METHODS],rotation=20,ha='right')
    ax.set(ylabel='Optimización (s) · media ± DE entre reps',title='Tiempos descriptivos · sesiones 012/016/017 · sin repetir búsquedas antiguas')
    grid(ax);save(fig,'I0252_computational_time_mean_sd',files)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,col,label in zip(axes,['path_length','gradient_clipped_fraction','projection_fraction'],
                           ['Longitud acumulada de la trayectoria','Fracción de pasos con clipping','Fracción de pasos con proyección']):
        for j,method in enumerate(METHODS):
            vals=runs[runs.method==method][col]
            ax.bar(j,vals.mean(),yerr=vals.std(ddof=1),capsize=3,color=COLORS[method])
        ax.set_xticks(x,[SHORT[m] for m in METHODS],rotation=45,ha='right');ax.set_ylabel(label);grid(ax)
    fig.suptitle('I0252 · comportamiento de las trayectorias · media ± DE entre reps',y=1.01)
    save(fig,'I0252_trajectory_behavior_summary',files)
    fig,ax=plt.subplots(figsize=(10,4.8))
    for j,method in enumerate(METHODS):
        ax.bar(j,s.loc[method,'trace_mean_101'],yerr=s.loc[method,'trace_mean_101_sd'],capsize=4,color=COLORS[method])
    ax.set_xticks(x,[SHORT[m] for m in METHODS],rotation=20,ha='right')
    ax.set(ylabel='Media de las 101 evaluaciones directas',title='I0252 · desempeño durante la búsqueda · media ± DE entre reps')
    grid(ax);save(fig,'I0252_transient_objective_summary',files)
    fig,ax=plt.subplots(figsize=(10,5.1))
    for method in METHODS:
        ax.scatter(s.loc[method,'optimization_seconds_mean'],s.loc[method,'trace_mean_101'],c=COLORS[method],
                   marker=MARKERS[method],s=80,label=LABELS[method])
    ax.set(xlabel='Tiempo medio de optimización (s; sesiones diferentes)',ylabel='Media de evaluaciones directas durante búsqueda',
           title='I0252 · calidad transitoria y coste observado · comparación descriptiva')
    legend(ax);grid(ax);save(fig,'I0252_runtime_vs_transient_quality',files)
    return files

def verify(coords,diag,tr,fin,runtime,pairs,files):
    assert set(coords.method)==set(METHODS) and set(coords.rep)=={1,2,3}
    assert len(coords)==len(tr)==1515 and len(diag)==1500 and len(fin)==15 and len(pairs)==12
    assert not coords.duplicated(['rep','method','iteration']).any()
    assert not tr.duplicated(['rep','method','iteration']).any()
    assert not diag.duplicated(['rep','method','iteration']).any()
    assert np.isfinite(coords[['z3','z4']]).all().all() and coords[['z3','z4']].ge(0).all().all() and coords[['z3','z4']].le(1).all().all()
    assert diag.logical_new_paths.eq(500).all() and diag.cumulative_logical_paths.le(50000).all()
    assert diag.new_paths_generated_this_study.sum()==150000
    assert runtime.new_paths_this_study.sum()==150000 and runtime.logical_search_paths.sum()==750000
    assert runtime.optimization_executed_this_study.sum()==3 and runtime[runtime.optimization_executed_this_study].method.eq('Exp50').all()
    assert fin.selection_paths.eq(10000).all() and fin.holdout_paths.eq(50000).all()
    assert np.allclose(fin.ci95_high-fin.objective,Z95*fin.evaluation_se)
    assert diag.internal_lse_surrogate.isna().all()
    assert diag[diag.method=='GreenSoftmax050'].gradient_target.eq('LSE beta=.5').all()
    for rep in (1,2,3):
        oldz=pd.read_csv(SRC/f'comparisons/I0252/rep_{rep:02d}_coordinates.csv')
        oldd=pd.read_csv(SRC/f'comparisons/I0252/rep_{rep:02d}_diagnostics.csv')
        oldf=pd.read_csv(SRC/f'comparisons/I0252/rep_{rep:02d}_finals.csv')
        oldt=pd.read_csv(SRC/f'comparisons/I0252/rep_{rep:02d}_traces.csv')
        for method in METHODS:
            c=select(coords,rep,method);d=select(diag,rep,method);t=select(tr,rep,method)
            assert np.array_equal(c.iteration,np.arange(101)) and np.array_equal(d.iteration,np.arange(1,101))
            assert np.allclose(d[['target_z3','target_z4']],c[['z3','z4']].to_numpy()[:-1],atol=1e-14,rtol=0)
            assert np.allclose(d[['next_z3','next_z4']],c[['z3','z4']].to_numpy()[1:],atol=1e-14,rtol=0)
            expected=np.clip(c[['z3','z4']].to_numpy()[:-1]+.035/np.sqrt(np.arange(1,101))[:,None]*np.clip(d[['g3','g4']].to_numpy(),-15,15),0,1)
            assert np.allclose(expected,c[['z3','z4']].to_numpy()[1:],atol=1e-13,rtol=0)
            f=fin[(fin.rep==rep)&(fin.method==method)].iloc[0]
            assert np.allclose([f.z3,f.z4],c[['z3','z4']].to_numpy()[-1],atol=1e-14,rtol=0)
            if method=='GreenSoftmax050':
                z=oldz[oldz.method==method].sort_values('iteration');dd=oldd[oldd.method==method].sort_values('iteration')
                tt=oldt[oldt.method==method].sort_values('iteration');ff=oldf[oldf.method==method].iloc[0]
                assert np.allclose(c[['z3','z4']],z[['z3','z4']],atol=1e-14,rtol=0)
                assert np.allclose(d[['g3','g4']],dd[['g3','g4']],atol=1e-12,rtol=0)
                assert np.allclose(d.internal_hard_max,dd.objective_internal,atol=1e-11,rtol=0)
                assert np.allclose(t.objective,tt.objective,atol=1e-12,rtol=0)
                assert abs(f.objective-ff.objective_holdout)<1e-11 and abs(f.evaluation_se-ff.evaluation_se)<1e-12
            elif method!='Exp50':
                z=select(pd.read_csv(OLD/'tables/coordinates.csv'),rep,method)
                tt=select(pd.read_csv(OLD/'tables/objective_traces.csv'),rep,method)
                ff=pd.read_csv(OLD/'tables/final_results.csv');ff=ff[(ff.rep==rep)&(ff.method==method)].iloc[0]
                assert np.allclose(c[['z3','z4']],z[['z3','z4']],atol=1e-14,rtol=0)
                assert np.allclose(t.objective,tt.objective,atol=1e-12,rtol=0)
                assert abs(f.objective-ff.objective)<1e-12 and abs(f.evaluation_se-ff.evaluation_se)<1e-12
        f=fin[fin.rep==rep].set_index('method');p=pairs[pairs.rep==rep].set_index('method')
        for method in METHODS[1:]:assert abs(p.loc[method,'difference']-(f.loc[method,'objective']-f.loc['Green','objective']))<1e-10
    banks=read('exp50_banks');assert len(banks)==15150 and banks.method.eq('Exp50').all() and banks.retained.all()
    keys=['rep','method','iteration'];g=banks.groupby(keys)
    assert np.allclose(g.bank_weight.sum(),1,atol=2e-14) and banks.bank_weight.ge(0).all()
    raw=.5**banks.bank_age;denom=raw.groupby([banks[k] for k in keys]).transform('sum')
    assert np.allclose(banks.bank_weight,raw/denom,atol=1e-14,rtol=1e-12)
    current=banks[banks.is_current];assert current.mean_lr.eq(1).all() and current.bank_lr_ess.eq(500).all()
    reference=diag[diag.method=='Exp50'].set_index(keys).sort_index()
    assert np.array_equal(g.size().sort_index(),reference.generated_banks)
    assert reference.retained_banks.eq(reference.generated_banks).all()
    for col,target,alpha in [('objective_contribution_mean','internal_hard_max',None),('gradient3_contribution_mean','g3',8.529251033547421),('gradient4_contribution_mean','g4',10.083855971068047)]:
        calc=(banks[col]*banks.bank_weight).groupby([banks[k] for k in keys]).sum().sort_index()
        cost=8.529251033547421*reference.target_z3+10.083855971068047*reference.target_z4
        calc-=cost if alpha is None else alpha
        assert np.allclose(calc,reference[target],atol=1e-8,rtol=1e-10)
    seeds=pd.read_csv(OUT/'configs/search_seeds.csv');assert len(seeds)==1500
    assert seeds.groupby(['rep','iteration']).seed.nunique().eq(1).all()
    assert len(seeds.groupby(['rep','iteration']).seed.first().unique())==300
    external=set(fin.selection_seed)|set(fin.holdout_seed)|set(tr.evaluation_seed)
    assert not set(seeds.seed).intersection(external)
    assert fin.groupby('rep').selection_seed.nunique().eq(1).all() and fin.groupby('rep').holdout_seed.nunique().eq(1).all()
    assert len(list((OUT/'checkpoints').glob('search_*.jls')))==3
    assert all((OUT/f'checkpoints/search_{r}_Exp50.jls').exists() for r in (1,2,3))
    science=tomli.loads((OUT/'validation/science_complete.toml').read_text());assert science['complete'] and science['new_optimizations']==3 and not science['rep4_or_rep5_executed']
    comp=tomli.loads((OUT/'validation/compatibility.toml').read_text());assert comp['passed'] and comp['final_protocol_identical']
    tests=tomli.loads((OUT/'validation/tests.toml').read_text());assert tests['passed']
    model=tomli.loads((OUT/'configs/experiment.toml').read_text())['science']
    assert all(sha(ROOT/f)==h for f,h in model['sources'].items())
    protected=json.loads((OUT/'validation/preserved_before.json').read_text())
    changed=[f for f,h in protected.items() if not (ROOT/f).is_file() or sha(ROOT/f)!=h];assert not changed,changed[:5]
    assert sha(OUT/'configs/I0252_surface.csv')==sha(OLD/'configs/I0252_surface.csv')
    png=[p for p in files if p.suffix=='.png'];pdf=[p for p in files if p.suffix=='.pdf']
    assert len(png)==len(pdf)==34
    for p in png:
        im=np.asarray(Image.open(p).convert('RGB'));assert im.shape[0]>300 and im.shape[1]>500 and im.std()>10,p
    assert all(p.read_bytes().startswith(b'%PDF-') for p in pdf)
    resume_path=OUT/'validation/resume_verification.json'
    resumed=False
    if resume_path.exists():
        proof=json.loads(resume_path.read_text());assert proof['passed'] and proof['all_hashes_unchanged'];resumed=True
    return dict(passed=True,new_optimizations=3,reused_optimizations=12,reps=[1,2,3],iterations=100,new_paths_per_iteration=500,
        new_production_search_paths=150000,logical_search_paths=750000,old_optimizations_rerun=0,
        new_final_evaluations=3,reused_final_evaluations=12,direct_trace_rows=1515,new_exp50_bank_diagnostic_rows=len(banks),
        basic_test_assertions=tests['assertions'],old_files_unchanged=len(protected),png=len(png),pdf=len(pdf),surface_recomputed=False,
        resume_verified_without_new_simulations=resumed,
        exp50_new_direct_trace_points=int(((tr.method=='Exp50')&tr.source.str.startswith('017 new')).sum()),
        archived_softmax_trace_points_reconstructed=int(((tr.method=='GreenSoftmax050')&tr.source.str.startswith('017 new')).sum()))

def report(fin,runs,summary,paired,diag,verified):
    base=summary[summary.method=='Green'].iloc[0];best=summary.loc[summary.objective_mean.idxmax()]
    new=summary[summary.method=='Exp50'].iloc[0];ps=paired[paired.method=='Exp50'].iloc[0]
    lines=['I0252 — comparación homogénea de cinco variantes Green (rep1, rep2, rep3)',
        '', 'Compatibilidad y origen de resultados:',
        'Antes de las búsquedas se verificó la compatibilidad de 011, 012 y 016 mediante hashes,',
        'configuración y archivos de resultados; se comprobó que Green016 reproduce exactamente Green011.',
        'Green Original, Variance κ=2 y Rolling W=5: optimizaciones y evaluaciones archivadas de016.',
        'Green + Softmax β=0.5: optimización y evaluación de I0252 en011; tiempo de012 (replay exacto ya archivado).',
        'Green Exponential ρ=0.5: ÚNICAS tres búsquedas nuevas de017. Ninguna optimización archivada se repitió.',
        'No se ejecutó rep4 ni rep5. No se probaron otros parámetros ni combinaciones.',
        'Standalone003 no es compatible: otro alpha, BETA del modelo, servicios, penalización y semillas;',
        'su Softmax es fresh-local. Se excluyeron sus resultados; sólo se reutilizó el entorno y la definición matemática.',
        '', 'Modelo, estimadores y parámetros:',
        'Modelo dos servidores I0252, 144 políticas umbral, z0=(0.12,0.10), 100 actualizaciones,',
        '500 trayectorias NUEVAS de 45 trabajos por actualización; 50000 por búsqueda.',
        'Model SHA256: 1ecdc3dc6161f7783a79a4e9d575f1f753a6fbb1ff4fb0d9337af56d5da54fd7.',
        "Objetivo original J(z)=max_i E_z[Y_i]-alpha'z; no E[max_i Y_i].",
        'alpha=(8.529251033547421,10.083855971068047), BETA_señal=4.0, eta=.035/sqrt(k),',
        'clipping del gradiente por componente ±15, proyección [0,1]^2, sin parada anticipada.',
        'Semillas originales X.seed(I0252,compare,rep,search,k), iguales entre métodos dentro de cada rep.',
        'Se conserva log-LR truncado a [-35,35] y probabilidades [1e-12,1-1e-12]. Esto puede introducir sesgo.',
        'Score evaluado en z actual; cociente LR con z actual y z generador de cada banco. No se usa SNIS.',
        'Green Original pondera uniformemente los k bancos. Rolling retiene los cinco más recientes con pesos iguales.',
        'Variance κ=2 selecciona política provisional con Green completo, mide por banco varianza muestral de Y*LR,',
        'retiene V_h <= 2*V_actual y siempre el actual, y vuelve a seleccionar política/gradiente con bancos retenidos.',
        'Esa selección adaptativa de bancos basada en recompensas puede introducir sesgo adicional.',
        'Exp50 conserva todos los bancos: a_h=.5^(k-h)/sum_{j=1}^k .5^(k-j), h=1,...,k.',
        'Corresponde a t=k-1,tau=h-1 en la fórmula solicitada. Mismos pesos en valores y gradiente de política común.',
        'El coste determinista y su derivada se sustraen una sola vez. No hay filtrado de observaciones/bancos en Exp50.',
        'Softmax: q_i=exp(.5*(v_i-max(v)))/sum_j exp(.5*(v_j-max(v))). β multiplica las diferencias;',
        'es temperatura inversa. No se divide por β ni se cambia la parametrización. Gradiente Gᵀq-alpha.',
        'El gradiente corresponde a un surrogate log-sum-exp, pero objective_internal archivado es max(v)-coste.',
        'Ese campo no es LSE; la LSE no fue archivada y no se inventó. La comparación principal usa objetivo original directo.',
        '', 'Protocolos de evaluación:',
        'Los cinco métodos usan el mismo protocolo final: selector directo independiente10000,',
        'política fijada antes de holdout directo independiente50000. CRN entre endpoints de una misma rep.',
        'Los protocolos anteriores ya son equivalentes; NO se reevaluaron los 12 endpoints archivados.',
        'Sólo se evaluaron los tres nuevos endpoints Exp50. No se usó LR en la evaluación final.',
        'SE=s/sqrt(50000); IC95% normal = media ±1.959963984540054 SE, condicional al endpoint y política seleccionada.',
        'Diferencias pareadas respecto de Green usan la covarianza inducida por CRN; no son SE de muestras independientes.',
        'Las curvas directas usan el mismo banco independiente N=1000 por rep y máximo de medias de las144 políticas.',
        'Por esa selección pueden ser optimistas. Sus puntos comparten CRN: no son réplicas independientes para inferencia.',
        'El campo k de diagnósticos evalúa target_z=z^(k-1); coordenadas y curvas directas t=0,...,100 representan z^t.',
        'La superficie archivada I0252 N=5000 se copió sin recalcular. No certifica un óptimo global.',
        '', 'Tabla comparativa (SD entre las tres reps; SE MC es otra medida):',
        'Método | objetivo final media ± SD | SE MC medio | tiempo medio(s) | media objetivo directo de101 iterados | longitud trayectoria media']
    for _,r in summary.iterrows():
        lines.append(f'{LABELS[r.method]} | {r.objective_mean:.6f} ± {r.objective_sd:.6f} | {r.conditional_mc_se_mean:.6f} | '
                     f'{r.optimization_seconds_mean:.4f} | {r.trace_mean_101:.6f} | {r.path_length_mean:.4f}')
    lines+=['','Resultados individuales:', 'rep | método | z3 | z4 | política | objetivo | SE MC | IC95% condicional | optimización(s)']
    for rep in (1,2,3):
        for method in METHODS:
            r=fin[(fin.rep==rep)&(fin.method==method)].iloc[0]
            lines.append(f'{rep:02d} | {LABELS[method]} | {r.z3:.8f} | {r.z4:.8f} | {int(r.selected_policy)} | '
                         f'{r.objective:.6f} | {r.evaluation_se:.6f} | [{r.ci95_low:.6f},{r.ci95_high:.6f}] | {r.optimization_seconds:.4f}')
    lines+=['','Diferencias pareadas respecto de Green Original:',
            'Método | diferencia media | SD de diferencias entre reps | SE MC pareado medio | diferencias positivas']
    for _,r in paired.iterrows():
        lines.append(f'{LABELS[r.method]} | {r.difference_mean:+.6f} | {r.difference_sd:.6f} | '
                     f'{r.conditional_paired_mc_se_mean:.6f} | {int(r.positive_reps)}/3')
    lines+=['','Interpretación:',
        f'Exp50: objetivo final {new.objective_mean:.6f}, frente a {base.objective_mean:.6f} de Green Original;',
        f'diferencia media {ps.difference_mean:+.6f}, positiva en {int(ps.positive_reps)}/3 reps.',
        f'Mayor media final observada entre los cinco: {LABELS[best.method]} ({best.objective_mean:.6f}).',
        f'Media de evaluaciones por iterado: Exp50 {new.trace_mean_101:.6f}; Green {base.trace_mean_101:.6f}.',
        'La media temporal de101 iterados mide comportamiento transitorio y no reemplaza el objetivo final.',
        'La longitud de trayectoria, clipping/proyección y ESS están en trajectory_metrics.csv; describen',
        'estabilidad y reutilización en puntos distintos, no prueban dominancia de varianza a z fijo.',
        'Con rho=.5, si LR=1 y bancos de igual tamaño, el ESS nominal de observaciones tiende a',
        'N*(1+rho)/(1-rho)=1500 aunque se retengan50000 observaciones. Los pesos temporales concentran',
        'la información en bancos recientes. LR diferentes pueden reducir aún más el ESS; no es precisión del gradiente.',
        'La SD entre tres reps mezcla variación de optimización y ruido de evaluación; se reporta separada de SE MC.',
        'Las diferencias/IC pareados describen esos endpoints. Tres semillas son insuficientes para afirmar superioridad poblacional.',
        '', 'Tiempos, límites y problemas:',
        'Se reutilizaron tiempos012/016; Exp50 se midió en017. Búsquedas secuenciales, BLAS=1,',
        'estimador calentado; evaluación externa, gráficos y diagnóstico adicional excluidos de tiempo de optimización.',
        'Sesiones y carga del SO distintas, orden fijo y diferencias de instrumentación/GC: tiempos descriptivos.',
        'No se repitieron búsquedas antiguas para mejorar comparabilidad de tiempos, conforme a la restricción.',
        f"Ineficiencia de consolidación: se reconstruyeron {verified['archived_softmax_trace_points_reconstructed']} puntos directos de la traza Softmax N=1000,",
        'porque no había caché por punto con SE/política en012. Sus objetivos ya existían en011 y se verificó',
        'su coincidencia (tolerancia1e-12). Podían haberse reutilizado sin esos metadatos; el trabajo adicional',
        'se explicita en trace_evaluation_workload.csv. Fue evaluación de decisiones archivadas con el mismo',
        'banco CRN, no repetición de optimización ni de holdouts finales. Los tiempos de optimización no lo incluyen.',
        'Inconsistencias detectadas y resueltas sin cambiar algoritmos:003 incompatible y campo interno Softmax',
        'mal interpretable como LSE. Se usaron resultados011 I0252 y curvas directas comunes.',
        f"Activaciones de recorte log-LR en los registros comparados: {int(diag.lr_clips.sum())}; "
        f"Exp50: {int(diag[diag.method=='Exp50'].lr_clips.sum())} (conteo de usos, no observaciones únicas).",
        'No se detectaron fallos numéricos en las tres búsquedas nuevas. Los avisos de Matplotlib sobre',
        'Axes3D y aceleradores opcionales antiguos de pandas no afectan estos gráficos 2D verificados.',
        '', 'Ejecución y reproducción desde SimOpt:',
        'python3 run_green_five_comparison_017.py --preflight-only',
        'python3 run_green_five_comparison_017.py --tests-only',
        'python3 run_green_five_comparison_017.py --resume',
        'Actualizar únicamente gráficos/tablas/resumen, sin Julia ni simulaciones:',
        'python3 analyze_green_five_comparison_017.py',
        'Verificar/reconstruir tablas y resumen sin volver a dibujar:',
        'python3 analyze_green_five_comparison_017.py --reuse-figures',
        'Julia1.11.7 y entorno resultados/003probing_softmax_20261021/entorno ya disponibles; no se modificó el entorno.',
        '', f"Verificación final: {verified['basic_test_assertions']} aserciones básicas + validaciones de todos los CSV y figuras; "
        f"{verified['old_files_unchanged']} archivos históricos sin cambios; {verified['png']} PNG y {verified['pdf']} PDF.",
        'Producción:150000 nuevas trayectorias de optimización;600000 trayectorias lógicas reutilizadas.',
        'Los prefijos de prueba y fixtures usan caminos aparte; nunca cuentan como presupuesto de las búsquedas productivas.']
    if verified['resume_verified_without_new_simulations']:
        lines+=['La ejecución real con --resume se verificó sin nuevas simulaciones: checkpoints y CSV científicos',
                'conservaron exactamente sus hashes; tampoco se volvieron a ejecutar pruebas ni calentamientos al estar completo.']
    (OUT/'RESUMEN_EXPERIMENTAL.txt').write_text('\n'.join(lines)+'\n')

def gallery(files,summary,verified):
    links=['RESUMEN_EXPERIMENTAL.txt','tables/final_comparison.csv','tables/method_summary.csv','tables/paired_vs_green.csv',
        'tables/coordinates.csv','tables/objective_traces.csv','tables/iteration_diagnostics.csv','tables/exp50_banks.csv',
        'tables/trajectory_metrics.csv','tables/provenance_and_runtime.csv','validation/compatibility.toml',
        'tables/trace_evaluation_workload.csv',
        'validation/compatibility.csv','validation/verification.json','reproduction/017probing_green_five_method_comparison.jl']
    if (OUT/'validation/resume_verification.json').exists():links.append('validation/resume_verification.json')
    rows=''.join(f'<tr><td>{html.escape(LABELS[r.method])}</td><td>{r.objective_mean:.6f}</td><td>{r.objective_sd:.6f}</td><td>{r.optimization_seconds_mean:.3f}</td></tr>' for _,r in summary.iterrows())
    content='<h1>I0252 · comparación de cinco variantes Green</h1><p>Rep1–rep3 · sólo Exponential ρ=0.5 ejecutado ahora; 12 búsquedas archivadas reutilizadas.</p>'
    content+='<p>'+' · '.join(f'<a href="{p}">{html.escape(p)}</a>' for p in links)+'</p>'
    content+='<table><tr><th>Método</th><th>Objetivo final medio</th><th>DE entre tres reps</th><th>Tiempo descriptivo medio(s)</th></tr>'+rows+'</table>'
    content+='<p>Curvas: objetivo original directo con CRN N=1000. Finales: selector10000 + holdout50000. IC por rep: ruido MC condicional. DE entre reps ≠ SE MC.</p>'
    order=['I0252_oracle_trajectories','I0252_z3_trajectory','I0252_z4_trajectory','I0252_convergence',
           'I0252_final_objectives','I0252_paired_differences','I0252_computational_time','I0252_search_behavior',
           'I0252_trajectory_behavior','I0252_transient','I0252_runtime','I0252_internal']
    png=[p for p in files if p.suffix=='.png']
    png.sort(key=lambda p:(next((i for i,x in enumerate(order) if p.stem.startswith(x)),99),p.name))
    for p in png:
        rel=p.relative_to(OUT).as_posix();pdf=p.with_suffix('.pdf').relative_to(OUT).as_posix()
        content+=f'<article><h2>{html.escape(p.stem)}</h2><a href="{rel}">PNG</a> · <a href="{pdf}">PDF</a><br><a href="{rel}"><img loading="lazy" src="{rel}"></a></article>'
    page='<!doctype html><html lang="es"><meta charset="utf-8"><title>I0252 cinco variantes Green</title><style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ddd}article{margin:40px 0}img{max-width:100%;height:auto}a{color:#075da8}</style>'+content+'</html>'
    (OUT/'index.html').write_text(page)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links += [v for k,v in attrs if k in ('href','src')]
    parser=Links();parser.feed(page)
    assert all((OUT/x).is_file() for x in parser.links)
    return len(parser.links)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reuse-figures',action='store_true')
    ap.add_argument('--trajectories-only',action='store_true',help='Restore original contourf style for rep1-rep3 only; leave all numeric outputs unchanged')
    args=ap.parse_args()
    if args.trajectories_only:
        restore_trajectories_only()
        return
    coords=read('coordinates');diag=read('iteration_diagnostics');tr=read('objective_traces');fin=read('final_results')
    runtime=read('provenance_and_runtime');pairs=read('paired_vs_green')
    merged,runs,summary,ps=tables(coords,diag,tr,fin,runtime,pairs)
    files=sorted([*OUT.glob('figures/*.png'),*OUT.glob('figures/*.pdf')]) if args.reuse_figures else plots(coords,diag,tr,fin,runtime,pairs,runs,summary)
    verified=verify(coords,diag,tr,fin,runtime,pairs,files)
    (OUT/'validation/verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    report(merged,runs,summary,ps,diag,verified)
    for file in ('017probing_green_five_method_comparison.jl','PROTOCOLO_green_five_comparison_017.txt','run_green_five_comparison_017.py','analyze_green_five_comparison_017.py'):
        shutil.copyfile(ROOT/file,OUT/'reproduction'/file)
    verified['gallery_links_verified']=gallery(files,summary,verified)
    (OUT/'validation/verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    (OUT/'COMPLETE.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps(verified,indent=2))
if __name__=='__main__':main()
