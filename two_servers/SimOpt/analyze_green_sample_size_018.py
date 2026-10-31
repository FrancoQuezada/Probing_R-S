"""Read-only scientific comparison of n500/n250; every figure has two panels."""
from pathlib import Path
from html.parser import HTMLParser
import argparse,hashlib,html,json,shutil,warnings
warnings.filterwarnings('ignore',message='Pandas requires version')
import numpy as np
import pandas as pd
import tomli
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from analyze_landscape_010 import background

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'resultados/018probing_green_n250_vs_n500'
OLD=ROOT/'resultados/017probing_green_five_method_comparison'
METHODS=['Green','GreenSoftmax050','Exp50','Filter2','Rolling5']
LABELS={'Green':'Green Original','GreenSoftmax050':'Green + Softmax β=0.5','Exp50':'Green Exponential ρ=0.5',
        'Filter2':'Green Variance κ=2','Rolling5':'Green Rolling W=5'}
SHORT={'Green':'Original','GreenSoftmax050':'Softmax β=0.5','Exp50':'Exponential ρ=0.5','Filter2':'Variance κ=2','Rolling5':'Rolling W=5'}
COLORS=dict(zip(METHODS,['#222222','#7B3294','#D55E00','#009E73','#0072B2']))
MARKERS=dict(zip(METHODS,['o','^','D','s','P']))
STYLES=dict(zip(METHODS,['-','-.','--','-',':']))
NS=[500,250]
TITLES=['n = 500 paths/iteration','n = 250 paths/iteration']
plt.rcParams.update({'font.size':10,'savefig.dpi':180,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
Z95=1.959963984540054

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return pd.read_csv(OUT/'tables'/f'{name}.csv')
def group(df,n,rep,method):return df[(df.n_paths==n)&(df.rep==rep)&(df.method==method)].sort_values('iteration')
def handles():return [Line2D([],[],color=COLORS[m],ls=STYLES[m],marker=MARKERS[m],ms=5,label=LABELS[m]) for m in METHODS]
def pair(title,ylabel='',xlabel='',height=5.8):
    fig,axes=plt.subplots(1,2,figsize=(14.5,height),sharex=True,sharey=True)
    fig.subplots_adjust(left=.075,right=.98,bottom=.25,top=.84,wspace=.14)
    fig.suptitle(title,fontsize=13,y=.97)
    for ax,t in zip(axes,TITLES):ax.set(title=t,xlabel=xlabel);ax.grid(alpha=.18)
    axes[0].set_ylabel(ylabel)
    return fig,axes
def finish(fig,axes,stem,files,audit,extra_handles=None):
    fig.legend(handles=handles()+(extra_handles or []),loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,fontsize=9,frameon=False)
    fig.canvas.draw()
    assert len(axes)==2
    assert [ax.get_title() for ax in axes]==TITLES
    assert np.allclose(axes[0].get_xlim(),axes[1].get_xlim())
    assert np.allclose(axes[0].get_ylim(),axes[1].get_ylim())
    assert axes[0].get_yscale()==axes[1].get_yscale()
    audit.append(dict(figure=stem,metric_panels=2,orientation='horizontal',left_n=500,right_n=250,
        panel_titles=TITLES,common_legend=True,methods_per_panel=METHODS,
        xlim=list(axes[0].get_xlim()),ylim=list(axes[0].get_ylim()),yscale=axes[0].get_yscale()))
    for ext in ('png','pdf'):
        p=OUT/'figures'/f'{stem}.{ext}';fig.savefig(p,bbox_inches='tight');files.append(p)
    plt.close(fig)

def build_tables(coords,diag,tr,fin,runtime,pairs,cross):
    full=fin.merge(runtime,on=['n_paths','rep','method'],validate='one_to_one')
    full.to_csv(OUT/'tables/final_comparison.csv',index=False)
    runrows=[]
    for n in NS:
        for rep in (1,2,3):
            for m in METHODS:
                z=group(coords,n,rep,m);d=group(diag,n,rep,m);t=group(tr,n,rep,m)
                f=full[(full.n_paths==n)&(full.rep==rep)&(full.method==m)].iloc[0]
                delta=np.diff(z[['z3','z4']].to_numpy(),axis=0)
                runrows.append(dict(n_paths=n,rep=rep,method=m,objective=f.objective,conditional_mc_se=f.evaluation_se,
                    optimization_seconds=f.optimization_seconds,trace_mean_101=t.objective.mean(),
                    trace_min_after10=t[t.iteration>=10].objective.min(),path_length=np.linalg.norm(delta,axis=1).sum(),
                    z3_total_variation=np.abs(delta[:,0]).sum(),z4_total_variation=np.abs(delta[:,1]).sum(),
                    clipping_fraction=(d.clipping>0).mean(),projection_fraction=(d.projection>1e-12).mean(),
                    mean_ess=d.ess.mean(),minimum_ess=d.ess.min(),mean_ess_fraction=d.ess_fraction.mean(),
                    mean_retained_banks=d.retained_banks.mean(),mean_objective_contribution_variance=d.objective_contribution_variance.mean(),
                    mean_gradient3_contribution_variance=d.gradient3_contribution_variance.mean(),
                    mean_gradient4_contribution_variance=d.gradient4_contribution_variance.mean(),
                    mean_gradient3_mean_variance_proxy=d.gradient3_mean_variance_proxy.mean(),
                    mean_gradient4_mean_variance_proxy=d.gradient4_mean_variance_proxy.mean(),
                    empirical_iteration_g3_sd=d.g3.std(ddof=1),empirical_iteration_g4_sd=d.g4.std(ddof=1),lr_clips=int(d.lr_clips.sum())))
    runs=pd.DataFrame(runrows);runs.to_csv(OUT/'tables/run_metrics.csv',index=False)
    summary=[]
    for n in NS:
        for m in METHODS:
            r=runs[(runs.n_paths==n)&(runs.method==m)]
            row=dict(n_paths=n,method=m,reps=3,logical_search_paths=100*n)
            for col in ['objective','optimization_seconds','trace_mean_101','path_length','clipping_fraction','projection_fraction',
                        'mean_ess','mean_ess_fraction','mean_retained_banks','mean_objective_contribution_variance',
                        'mean_gradient3_contribution_variance','mean_gradient4_contribution_variance',
                        'mean_gradient3_mean_variance_proxy','mean_gradient4_mean_variance_proxy']:
                row[col+'_mean']=r[col].mean();row[col+'_sd']=r[col].std(ddof=1)
            row['conditional_mc_se_mean']=r.conditional_mc_se.mean();row['lr_clips']=int(r.lr_clips.sum());summary.append(row)
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'tables/method_summary.csv',index=False)
    comparisons=[]
    for rep in (1,2,3):
        for m in METHODS:
            a=runs[(runs.n_paths==500)&(runs.rep==rep)&(runs.method==m)].iloc[0]
            b=runs[(runs.n_paths==250)&(runs.rep==rep)&(runs.method==m)].iloc[0]
            p=cross[(cross.rep==rep)&(cross.method==m)].iloc[0]
            row=dict(rep=rep,method=m,objective_n500=a.objective,objective_n250=b.objective,
                difference_n250_minus_n500=p.difference_n250_minus_n500,paired_mc_se=p.paired_mc_se,
                ci95_low=p.ci95_low,ci95_high=p.ci95_high,seconds_n500=a.optimization_seconds,seconds_n250=b.optimization_seconds,
                runtime_ratio_n250_over_n500=b.optimization_seconds/a.optimization_seconds,
                time_reduction_fraction=1-b.optimization_seconds/a.optimization_seconds,
                path_length_change=b.path_length-a.path_length,trace_mean_change=b.trace_mean_101-a.trace_mean_101)
            for col in ['mean_gradient3_contribution_variance','mean_gradient4_contribution_variance',
                        'mean_gradient3_mean_variance_proxy','mean_gradient4_mean_variance_proxy']:
                row[col+'_n500']=a[col];row[col+'_n250']=b[col]
            comparisons.append(row)
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(OUT/'tables/n250_vs_n500_individual.csv',index=False)
    sensitivity=[]
    for m in METHODS:
        c=comparisons[comparisons.method==m];a=summary[(summary.n_paths==500)&(summary.method==m)].iloc[0]
        b=summary[(summary.n_paths==250)&(summary.method==m)].iloc[0]
        sensitivity.append(dict(method=m,mean_objective_change=c.difference_n250_minus_n500.mean(),
            sd_objective_change=c.difference_n250_minus_n500.std(ddof=1),mean_conditional_paired_mc_se=c.paired_mc_se.mean(),
            conditional_mc_se_of_average=np.sqrt(np.sum(c.paired_mc_se**2))/3,
            positive_reps=int((c.difference_n250_minus_n500>0).sum()),
            runtime_ratio_of_means=b.optimization_seconds_mean/a.optimization_seconds_mean,
            time_reduction_percent=100*(1-b.optimization_seconds_mean/a.optimization_seconds_mean),
            mean_path_length_change=c.path_length_change.mean(),mean_trace_change=c.trace_mean_change.mean(),
            gradient3_contribution_variance_ratio=b.mean_gradient3_contribution_variance_mean/a.mean_gradient3_contribution_variance_mean,
            gradient4_contribution_variance_ratio=b.mean_gradient4_contribution_variance_mean/a.mean_gradient4_contribution_variance_mean))
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(OUT/'tables/sample_size_sensitivity.csv',index=False)
    spread=diag.groupby(['n_paths','method','iteration']).agg(g3_sd=('g3','std'),g4_sd=('g4','std')).reset_index()
    spread.to_csv(OUT/'tables/gradient_spread_across_three_reps.csv',index=False)
    spread.groupby(['n_paths','method'])[['g3_sd','g4_sd']].mean().reset_index().to_csv(OUT/'tables/gradient_spread_summary.csv',index=False)
    return full,runs,summary,comparisons,sensitivity,spread

def plot_series(df,rep,col,ylabel,stem,files,audit,log=False,limits=None):
    fig,axes=pair(f'I0252 — rep{rep:02d}',ylabel,'Iteración')
    for ax,n in zip(axes,NS):
        for m in METHODS:
            d=group(df,n,rep,m)
            ax.plot(d.iteration,d[col],c=COLORS[m],ls=STYLES[m],lw=1.5,label=LABELS[m],
                    marker=MARKERS[m],markevery=20,ms=3.2)
        ax.set_xlim(0,100)
        if log:ax.set_yscale('log')
        if limits:ax.set_ylim(*limits)
    finish(fig,axes,stem,files,audit)

def figures(coords,diag,tr,fin,runtime,pairs,runs,summary,spread):
    files=[];audit=[];surface=pd.read_csv(OUT/'configs/I0252_surface.csv')
    for rep in (1,2,3):
        fig,axes=pair(f'I0252 — rep{rep:02d} · trayectorias', 'z4','z3',height=7)
        ims=[]
        for ax,n,title in zip(axes,NS,TITLES):
            ax.grid(False);im=background(ax,surface,title,mark=True);ims.append(im)
            for m in METHODS:
                z=group(coords,n,rep,m)
                line,=ax.plot(z.z3,z.z4,c=COLORS[m],ls=STYLES[m],lw=1.5)
                assert np.array_equal(line.get_xdata(),z.z3.to_numpy()) and np.array_equal(line.get_ydata(),z.z4.to_numpy())
                ax.scatter(z.z3.iloc[-1],z.z4.iloc[-1],c=COLORS[m],marker=MARKERS[m],s=50,edgecolor='white',zorder=10)
            ax.scatter(.12,.10,c='gold',edgecolor='black',marker='*',s=130,zorder=12)
            ax.set(xlim=(0,1),ylim=(0,1),title=title)
        assert np.array_equal(ims[0].levels,ims[1].levels) and ims[0].get_clim()==ims[1].get_clim()
        fig.colorbar(ims[0],ax=axes,label='J(z) · superficie original archivada',fraction=.025,pad=.025)
        extra=[Line2D([],[],marker='*',color='gold',markeredgecolor='black',ls='',ms=10,label='Inicio común'),
               Line2D([],[],marker='X',color='cyan',markeredgecolor='black',ls='',ms=8,label='Máximo de grilla')]
        finish(fig,axes,f'I0252_oracle_trajectories_rep{rep:02d}',files,audit,extra)
        for dim in ('z3','z4'):
            plot_series(coords,rep,dim,dim,f'I0252_{dim}_trajectory_rep{rep:02d}',files,audit,limits=(0,1))
        plot_series(tr,rep,'objective','Objetivo original directo · CRN N=1000',f'I0252_convergence_rep{rep:02d}',files,audit)
        for col,ylabel,tag,log in [
            ('ess','ESS de coeficientes LR','ess',True),('ess_fraction','ESS / observaciones retenidas','ess_fraction',False),
            ('retained_banks','Bancos retenidos','retained_banks',False),
            ('objective_contribution_variance','Media ponderada var(Y_hard · LR)','objective_contribution_variance',True),
            ('gradient3_contribution_variance','Varianza de contribuciones del gradiente z3','gradient3_contribution_variance',True),
            ('gradient4_contribution_variance','Varianza de contribuciones del gradiente z4','gradient4_contribution_variance',True),
            ('gradient3_mean_variance_proxy','Proxy descriptivo var(media gradiente z3)','gradient3_mean_variance_proxy',True),
            ('gradient4_mean_variance_proxy','Proxy descriptivo var(media gradiente z4)','gradient4_mean_variance_proxy',True),
            ('step_norm','Norma del paso','step_norm',False),('gradient_norm','Norma del gradiente','gradient_norm',True)]:
            plot_series(diag,rep,col,ylabel,f'I0252_{tag}_rep{rep:02d}',files,audit,log=log)
        fig,axes=pair(f'I0252 — rep{rep:02d} · solución final','Objetivo holdout · IC95% MC condicional','Método')
        for ax,n in zip(axes,NS):
            f=fin[(fin.n_paths==n)&(fin.rep==rep)].set_index('method')
            for j,m in enumerate(METHODS):ax.errorbar(j,f.loc[m,'objective'],yerr=Z95*f.loc[m,'evaluation_se'],
                fmt=MARKERS[m],c=COLORS[m],ms=6,capsize=3)
            ax.set_xticks(np.arange(5),[SHORT[m] for m in METHODS],rotation=25,ha='right')
        finish(fig,axes,f'I0252_final_objectives_rep{rep:02d}',files,audit)
        fig,axes=pair(f'I0252 — rep{rep:02d} · diferencias respecto de Green','Diferencia · IC95% MC pareado','Método')
        for ax,n in zip(axes,NS):
            p=pairs[(pairs.n_paths==n)&(pairs.rep==rep)].set_index('method')
            for j,m in enumerate(METHODS):
                mu=0 if m=='Green' else p.loc[m,'difference'];se=0 if m=='Green' else p.loc[m,'paired_mc_se']
                ax.errorbar(j,mu,yerr=Z95*se,fmt=MARKERS[m],c=COLORS[m],ms=6,capsize=3)
            ax.axhline(0,c='gray',lw=1);ax.set_xticks(np.arange(5),[SHORT[m] for m in METHODS],rotation=25,ha='right')
        finish(fig,axes,f'I0252_paired_vs_green_rep{rep:02d}',files,audit)
        fig,axes=pair(f'I0252 — rep{rep:02d} · tiempo descriptivo','Tiempo de optimización (s)','Método')
        for ax,n in zip(axes,NS):
            r=runtime[(runtime.n_paths==n)&(runtime.rep==rep)].set_index('method')
            ax.bar(np.arange(5),[r.loc[m,'optimization_seconds'] for m in METHODS],color=[COLORS[m] for m in METHODS])
            ax.set_xticks(np.arange(5),[SHORT[m] for m in METHODS],rotation=25,ha='right')
        finish(fig,axes,f'I0252_runtime_rep{rep:02d}',files,audit)
    fig,axes=pair('I0252 · objetivo final por repetición','Objetivo holdout · IC95% MC condicional','Método')
    for ax,n in zip(axes,NS):
        for j,m in enumerate(METHODS):
            f=fin[(fin.n_paths==n)&(fin.method==m)].sort_values('rep')
            for offset,r in zip([-.15,0,.15],f.itertuples()):
                ax.errorbar(j+offset,r.objective,yerr=Z95*r.evaluation_se,fmt=MARKERS[m],color=COLORS[m],ms=4.5,capsize=2)
        ax.set_xticks(np.arange(5),[SHORT[m] for m in METHODS],rotation=25,ha='right')
        ax.text(.02,.03,'rep1 / rep2 / rep3: izquierda / centro / derecha',transform=ax.transAxes,fontsize=8)
    finish(fig,axes,'I0252_final_objectives_all_reps_mc',files,audit)
    for metric,sd,ylabel,title,stem in [
        ('objective_mean','objective_sd','Objetivo final · media ± DE entre reps','I0252 · objetivo final medio','I0252_final_objectives_mean_sd'),
        ('optimization_seconds_mean','optimization_seconds_sd','Tiempo (s) · media ± DE entre reps','I0252 · tiempo medio descriptivo','I0252_runtime_mean_sd'),
        ('trace_mean_101_mean','trace_mean_101_sd','Media de 101 objetivos directos · media ± DE','I0252 · calidad durante la búsqueda','I0252_transient_quality_mean_sd'),
        ('path_length_mean','path_length_sd','Longitud de trayectoria · media ± DE','I0252 · estabilidad de trayectorias','I0252_path_length_mean_sd'),
        ('clipping_fraction_mean','clipping_fraction_sd','Fracción de pasos con clipping · media ± DE','I0252 · clipping del gradiente','I0252_gradient_clipping_mean_sd'),
        ('projection_fraction_mean','projection_fraction_sd','Fracción de pasos proyectados · media ± DE','I0252 · proyección','I0252_projection_mean_sd')]:
        fig,axes=pair(title,ylabel,'Método')
        for ax,n in zip(axes,NS):
            s=summary[summary.n_paths==n].set_index('method')
            for j,m in enumerate(METHODS):ax.errorbar(j,s.loc[m,metric],yerr=s.loc[m,sd],fmt=MARKERS[m],c=COLORS[m],ms=6,capsize=3)
            ax.set_xticks(np.arange(5),[SHORT[m] for m in METHODS],rotation=25,ha='right')
        finish(fig,axes,stem,files,audit)
    fig,axes=pair('I0252 · tiempos por repetición y medias','Optimización (s)','Método')
    for ax,n in zip(axes,NS):
        for j,m in enumerate(METHODS):
            vals=runtime[(runtime.n_paths==n)&(runtime.method==m)].sort_values('rep').optimization_seconds.to_numpy()
            ax.scatter(j+np.array([-.15,0,.15]),vals,c=COLORS[m],marker=MARKERS[m],s=30)
            ax.plot([j-.22,j+.22],[vals.mean()]*2,c=COLORS[m],lw=2)
        ax.set_xticks(np.arange(5),[SHORT[m] for m in METHODS],rotation=25,ha='right')
    finish(fig,axes,'I0252_runtime_all_reps',files,audit)
    fig,axes=pair('I0252 · convergencia media de tres repeticiones','Objetivo directo · media ± DE entre reps','Iteración')
    for ax,n in zip(axes,NS):
        for m in METHODS:
            g=tr[(tr.n_paths==n)&(tr.method==m)].groupby('iteration').objective
            mu=g.mean();sd=g.std(ddof=1);x=mu.index.to_numpy()
            ax.plot(x,mu,c=COLORS[m],ls=STYLES[m],lw=1.7)
            ax.fill_between(x,(mu-sd).to_numpy(),(mu+sd).to_numpy(),color=COLORS[m],alpha=.10)
        ax.set_xlim(0,100)
    finish(fig,axes,'I0252_convergence_mean_sd',files,audit)
    for col,label in [('g3_sd','DE empírica de g3 entre tres reps'),('g4_sd','DE empírica de g4 entre tres reps')]:
        fig,axes=pair('I0252 · dispersión de gradientes entre repeticiones',label,'Actualización k')
        for ax,n in zip(axes,NS):
            for m in METHODS:
                d=spread[(spread.n_paths==n)&(spread.method==m)].sort_values('iteration')
                ax.plot(d.iteration,d[col],c=COLORS[m],ls=STYLES[m],lw=1.5)
            ax.set(xlim=(0,100),yscale='log')
        finish(fig,axes,f'I0252_gradient_spread_{col}',files,audit)
    (OUT/'validation/figure_layout.json').write_text(json.dumps(audit,indent=2)+'\n')
    return files,audit

def verify(coords,diag,tr,fin,runtime,pairs,cross,files,audit):
    assert len(coords)==len(tr)==3030 and len(diag)==3000 and len(fin)==len(runtime)==30
    assert len(pairs)==24 and len(cross)==15
    assert set(coords.n_paths)=={250,500} and set(coords.rep)=={1,2,3} and set(coords.method)==set(METHODS)
    for df in (coords,diag,tr):assert not df.duplicated(['n_paths','rep','method','iteration']).any()
    assert runtime[runtime.n_paths==250].new_search_paths_this_study.eq(25000).all()
    assert runtime[runtime.n_paths==500].new_search_paths_this_study.eq(0).all()
    assert runtime.new_search_paths_this_study.sum()==375000
    assert fin.selection_paths.eq(10000).all() and fin.holdout_paths.eq(50000).all()
    assert np.allclose(fin.ci95_high-fin.objective,Z95*fin.evaluation_se)
    assert diag.logical_new_paths.eq(diag.n_paths).all()
    assert diag.retained_observations.eq(diag.n_paths*diag.retained_banks).all()
    assert diag.ess.gt(0).all() and (diag.ess<=diag.retained_observations*(1+1e-12)).all()
    for col in ['objective_contribution_variance','gradient3_contribution_variance','gradient4_contribution_variance']:
        assert np.isfinite(diag[col]).all() and diag[col].ge(0).all()
    oldcoords=pd.read_csv(OLD/'tables/coordinates.csv');oldtr=pd.read_csv(OLD/'tables/objective_traces.csv');oldfin=pd.read_csv(OLD/'tables/final_results.csv')
    for n in NS:
        for rep in (1,2,3):
            for m in METHODS:
                z=group(coords,n,rep,m);d=group(diag,n,rep,m);t=group(tr,n,rep,m)
                assert np.array_equal(z.iteration,np.arange(101)) and np.array_equal(d.iteration,np.arange(1,101))
                assert np.allclose(z[['z3','z4']].iloc[0],[.12,.10],atol=0,rtol=0)
                assert np.allclose(d[['target_z3','target_z4']],z[['z3','z4']].to_numpy()[:-1],atol=1e-14,rtol=0)
                calc=np.clip(z[['z3','z4']].to_numpy()[:-1]+(.035/np.sqrt(np.arange(1,101)))[:,None]*np.clip(d[['g3','g4']].to_numpy(),-15,15),0,1)
                assert np.allclose(calc,z[['z3','z4']].to_numpy()[1:],atol=1e-13,rtol=0)
                if m=='Rolling5':assert np.array_equal(d.retained_banks,np.minimum(d.iteration,5))
                if m in ('Green','GreenSoftmax050','Exp50'):assert d.retained_banks.eq(d.generated_banks).all()
                f=fin[(fin.n_paths==n)&(fin.rep==rep)&(fin.method==m)].iloc[0]
                assert np.allclose([f.z3,f.z4],z[['z3','z4']].iloc[-1],atol=1e-14,rtol=0)
                if n==500:
                    q=oldcoords[(oldcoords.rep==rep)&(oldcoords.method==m)].sort_values('iteration')
                    tt=oldtr[(oldtr.rep==rep)&(oldtr.method==m)].sort_values('iteration')
                    ff=oldfin[(oldfin.rep==rep)&(oldfin.method==m)].iloc[0]
                    assert np.allclose(z[['z3','z4']],q[['z3','z4']],atol=1e-14,rtol=0)
                    assert np.allclose(t.objective,tt.objective,atol=1e-12,rtol=0)
                    assert abs(f.objective-ff.objective)<1e-12 and abs(f.evaluation_se-ff.evaluation_se)<1e-12
    for r in cross.itertuples():
        f=fin[(fin.rep==r.rep)&(fin.method==r.method)].set_index('n_paths')
        assert abs(r.difference_n250_minus_n500-(f.loc[250,'objective']-f.loc[500,'objective']))<1e-10
        assert abs(r.ci95_high-r.difference_n250_minus_n500-Z95*r.paired_mc_se)<1e-12
    seeds=pd.read_csv(OUT/'configs/search_seeds.csv');assert len(seeds)==3000
    assert seeds.groupby(['rep','iteration']).seed.nunique().eq(1).all()
    external=set(fin.selection_seed)|set(fin.holdout_seed)|set(tr.seed)
    assert not set(seeds.seed).intersection(external)
    assert tr.groupby('rep').seed.nunique().eq(1).all()
    banks=read('bank_diagnostics');assert banks.bank_paths.eq(banks.n_paths).all()
    assert banks[banks.method=='GreenSoftmax050'].bank_weight.gt(0).all()
    bankkeys=['n_paths','rep','method','iteration']
    assert np.allclose(banks.groupby(bankkeys).bank_weight.sum(),1,atol=2e-14)
    retained=banks.assign(active=banks.bank_weight>0).groupby(bankkeys).active.sum().sort_index()
    ref=diag.set_index(bankkeys).retained_banks.reindex(retained.index)
    assert np.array_equal(retained,ref)
    exp=banks[banks.method=='Exp50'].copy()
    raw=.5**(exp.iteration-exp.bank_iteration)
    den=raw.groupby([exp[k] for k in bankkeys]).transform('sum')
    assert np.allclose(exp.bank_weight,raw/den,atol=1e-14,rtol=1e-12)
    assert len(list((OUT/'checkpoints').glob('search_250_*.jls')))==15
    assert len(list((OUT/'checkpoints').glob('search_500_*.jls')))==0
    assert all(len(a['methods_per_panel'])==5 and a['metric_panels']==2 and a['orientation']=='horizontal' and a['left_n']==500 and a['right_n']==250 for a in audit)
    png=[p for p in files if p.suffix=='.png'];pdf=[p for p in files if p.suffix=='.pdf']
    assert len(png)==len(pdf)==len(audit)>0
    for p in png:
        a=np.asarray(Image.open(p).convert('RGB'));assert a.shape[1]>1400 and a.shape[1]>a.shape[0] and a.std()>10,p
    assert all(p.read_bytes().startswith(b'%PDF-') for p in pdf)
    surface=OUT/'configs/I0252_surface.csv';assert sha(surface)==sha(OLD/'configs/I0252_surface.csv')
    proof=json.loads((OUT/'validation/surface_style_check.json').read_text());assert proof['passed'] and proof['checked_before_full_searches']
    protected=json.loads((OUT/'validation/preserved_before.json').read_text())
    changed=[f for f,h in protected.items() if not (ROOT/f).is_file() or sha(ROOT/f)!=h];assert not changed,changed[:5]
    tests=tomli.loads((OUT/'validation/tests.toml').read_text());assert tests['passed']
    science=tomli.loads((OUT/'validation/science_complete.toml').read_text());assert science['complete']
    config=tomli.loads((OUT/'configs/experiment.toml').read_text())['science']
    assert all(sha(ROOT/f)==h for f,h in config['sources'].items())
    resume=OUT/'validation/resume_verification.json'
    resumed=False
    if resume.exists():
        v=json.loads(resume.read_text());assert v['passed'] and v['all_hashes_unchanged'];resumed=True
    return dict(passed=True,new_n250_searches=15,n500_searches_rerun=0,rep_ids=[1,2,3],iterations=100,
        paths_per_new_search=25000,new_production_search_paths=375000,independent_final_evaluations_n250=15,
        archived_final_evaluations_n500=15,direct_trace_points=3030,iteration_diagnostics=3000,
        paired_sample_size_comparisons=15,basic_test_assertions=tests['assertions'],png=len(png),pdf=len(pdf),
        all_figures_two_horizontal_panels=True,common_scales_and_legends=True,surface_recomputed=False,
        historical_files_unchanged=len(protected),resume_verified_without_new_simulation=resumed)

def report(full,runs,summary,comparisons,sensitivity,verified):
    s250=summary[summary.n_paths==250];best=s250.loc[s250.objective_mean.idxmax()]
    stable=s250.loc[s250.path_length_mean.idxmin()]
    least=sensitivity.loc[sensitivity.mean_objective_change.abs().idxmin()]
    lines=['I0252 — sensibilidad de cinco métodos Green: n500 frente a n250',
        'Reps1,2,3 exclusivamente. Se ejecutaron15 nuevas búsquedas n250; ninguna búsqueda n500 se repitió.',
        '', 'Compatibilidad e implementación:',
        'Único cambio del optimizador:250 nuevas trayectorias/iteración, en lugar de500;100 iteraciones.',
        'Presupuesto por búsqueda25000 frente a50000. Primitivas/semillas originales por rep/iteración, compartidas entre métodos y n.',
        'Modelo I0252 intacto,144 políticas,45 trabajos/path,z0=(.12,.10),eta=.035/sqrt(k),clip por componente±15,',
        'proyección[0,1]^2,sin parada anticipada. Mismos alpha, BETA_señal4 y reglas LR/clamps/log-LR±35. Sin SNIS.',
        'Green, Exponential rho.5, Variance kappa2, Rolling W5 llaman V.search de016 con n250.',
        'Softmax beta.5 llama A.timed_method de012 con n250; pesos exp(.5*(v_i-max(v))), mismo gradiente Gᵀq-alpha.',
        'No se reimplementaron ni modificaron algoritmos históricos. Green n250 reproduce exactamente el experimento013 archivado.',
        'Semillas iguales implican prefijos comunes de primitivas. Las señales finales dependen de z; no se exige igualdad de bancos materializados.',
        '', 'Evaluación y diagnósticos:',
        'Final: selector independiente10000 + holdout independiente50000, CRN entre métodos y ambos n dentro de cada rep.',
        'Política fijada antes del holdout, evaluación directa sin LR y objetivo original no suavizado.',
        'Objetivos/SE/trazas finales n500 se reutilizan exactamente de017. Para diferencias pareadas se reconstruyó sólo',
        'el vector de payoffs holdout de Softmax n500 con política/decisión/semilla originales: no reselección ni búsqueda.',
        'SE e IC95% normales son MC condicionales a decisiones y políticas seleccionadas; DE entre tres reps no es SE MC.',
        'Curvas: mismo banco directo independiente N1000 por rep para ambos n. Máximo de medias puede tener sesgo de selección.',
        'El objetivo interno archivado Softmax es máximo duro, no LSE; nunca se mezcla con objetivos suavizados en gráficos.',
        'Varianzas por banco: contribución objetiva Y_hard*LR para todos; gradiente Softmax usa mezcla real sum(q_iY_i)*LR*score.',
        'Momentos faltantes de Softmax reconstruidos en coordenadas FIJAS, con kernels originales; cada gradiente/ESS/política',
        'reconstruido coincide exactamente con la búsqueda. Cero actualizaciones del optimizador en esos diagnósticos.',
        'Ese trabajo diagnóstico regenera150000 primitivas n500 y75000 n250, separadas del presupuesto/timing de optimización.',
        'Media ponderada de varianzas intrabanco y sum(a_h²*s_h²/n_h) son diagnósticos, no estimaciones válidas de varianza',
        'incondicional del optimizador adaptativo. Estados/políticas difieren; no se comparan estimadores en un z fijo.',
        '', 'Tabla comparativa:', 'n | método | objetivo final media ± DE | SE MC medio | tiempo medio(s) | media de101 objetivos directos | longitud media']
    for n in NS:
        for m in METHODS:
            r=summary[(summary.n_paths==n)&(summary.method==m)].iloc[0]
            lines.append(f'{n} | {LABELS[m]} | {r.objective_mean:.6f} ± {r.objective_sd:.6f} | {r.conditional_mc_se_mean:.6f} | '
                         f'{r.optimization_seconds_mean:.4f} | {r.trace_mean_101_mean:.6f} | {r.path_length_mean:.4f}')
    lines+=['','Resultados individuales:', 'n | rep | método | z3 | z4 | política | objetivo | SE MC | IC95% MC']
    for n in NS:
        for rep in (1,2,3):
            for m in METHODS:
                f=full[(full.n_paths==n)&(full.rep==rep)&(full.method==m)].iloc[0]
                lines.append(f'{n} | {rep:02d} | {SHORT[m]} | {f.z3:.8f} | {f.z4:.8f} | {int(f.selected_policy)} | '
                             f'{f.objective:.6f} | {f.evaluation_se:.6f} | [{f.ci95_low:.6f},{f.ci95_high:.6f}]')
    lines+=['','Diferencias pareadas n250 menos n500:', 'rep | método | delta | SE MC pareado | IC95% MC | cambio tiempo(%) | cambio longitud']
    for r in comparisons.itertuples():
        lines.append(f'{r.rep:02d} | {SHORT[r.method]} | {r.difference_n250_minus_n500:+.6f} | {r.paired_mc_se:.6f} | '
                     f'[{r.ci95_low:.6f},{r.ci95_high:.6f}] | {100*r.time_reduction_fraction:+.2f} | {r.path_length_change:+.4f}')
    lines+=['','Respuestas experimentales (tres semillas, sin conclusión de superioridad estadística):',
        f'1. Mayor media final n250 observada: {LABELS[best.method]}, {best.objective_mean:.6f}.',
        '2. Cambio de calidad y tiempo por método (n250-n500):',
        'método | media delta objetivo | DE delta entre reps | SE MC pareado medio | reducción tiempo de medias(%) | cambio media temporal']
    for r in sensitivity.itertuples():
        lines.append(f'{SHORT[r.method]} | {r.mean_objective_change:+.6f} | {r.sd_objective_change:.6f} | '
                     f'{r.mean_conditional_paired_mc_se:.6f} | {r.time_reduction_percent:+.2f} | {r.mean_trace_change:+.6f}')
    lines+=[f'3. Menor longitud de trayectoria media con n250: {LABELS[stable.method]} ({stable.path_length_mean:.4f}).',
        'La estabilidad se contrasta también mediante variación total de z3/z4, clipping y proyección en run_metrics.csv.',
        '4. Cocientes n250/n500 de medias de varianzas de contribuciones del gradiente (estados visitados distintos):']
    for r in sensitivity.itertuples():lines.append(f'{SHORT[r.method]}: z3={r.gradient3_contribution_variance_ratio:.4g}, z4={r.gradient4_contribution_variance_ratio:.4g}.')
    lines+=['La DE empírica entre tres reps por iteración está en gradient_spread_across_three_reps.csv y en figuras separadas.',
        'No se interpreta como varianza del estimador a z fijo ni se confunde con variación temporal de gradientes.',
        '5. Reducciones efectivas de tiempo observadas indicadas arriba. El ahorro de paths es exactamente50%.',
        'Los tiempos proceden de sesiones012/016/017/018; orden fijo, SO no aislado e instrumentación/GC distintos.',
        'Comparación descriptiva: no se promete que el speedup sea causal ni que se reproduzca en otro hardware.',
        f'6. Menor cambio absoluto de media final observado: {LABELS[least.method]} ({least.mean_objective_change:+.6f}).',
        'La sensibilidad no se decide sólo por esa media: revisar diferencias individuales, dispersión, calidad temporal y coste.',
        '', 'Lectura conjunta de la sensibilidad:',
        'Variance kappa2 conserva casi toda su calidad final en las tres reps y tiene la menor longitud media de trayectoria.',
        'En esta muestra ofrece el comportamiento más estable frente a la reducción de n, considerando calidad final y transitoria.',
        'Exponential mantiene prácticamente la media final, pero sus trayectorias se alargan y aumenta la frecuencia de clipping.',
        'Rolling mejora la media final principalmente por la recuperación de rep2; su calidad temporal y longitud de trayectoria empeoran.',
        'Green Original mejora la media y el comportamiento transitorio con n250 en estas semillas. Esto no demuestra que menos paths',
        'sean intrínsecamente mejores: cambiar las realizaciones de los bancos altera la trayectoria adaptativa completa.',
        'Softmax muestra la mayor pérdida media, concentrada en rep3; su variabilidad entre repeticiones aumenta.',
        '', 'Promedio temporal de la DE empírica del gradiente entre tres reps (no varianza del estimador a z fijo):',
        'método | DE(g3) n500 -> n250 | DE(g4) n500 -> n250']
    spread_summary=pd.read_csv(OUT/'tables/gradient_spread_summary.csv')
    for m in METHODS:
        a=spread_summary[(spread_summary.n_paths==500)&(spread_summary.method==m)].iloc[0]
        b=spread_summary[(spread_summary.n_paths==250)&(spread_summary.method==m)].iloc[0]
        lines.append(f'{SHORT[m]} | {a.g3_sd:.6g} -> {b.g3_sd:.6g} | {a.g4_sd:.6g} -> {b.g4_sd:.6g}')
    soft=comparisons[(comparisons.rep==3)&(comparisons.method=='GreenSoftmax050')].iloc[0]
    lines += [f'Softmax rep3: cambio final {soft.difference_n250_minus_n500:+.6f}, SE MC pareado {soft.paired_mc_se:.6f},',
        f'IC95% MC condicional [{soft.ci95_low:.6f},{soft.ci95_high:.6f}]. Esto caracteriza ese par de soluciones, no superioridad poblacional.',
        '', 'Límites y comprobaciones:',
        'Con100 iteraciones, reducir n también reduce el presupuesto total: no es un experimento a igual presupuesto total.',
        'La filtración adaptativa de bancos y el recorte LR pueden introducir sesgo; ESS no determina precisión del gradiente.',
        'Tres repeticiones no permiten afirmar superioridad poblacional. IC MC condicionales no sustituyen incertidumbre de optimización.',
        'Todas las figuras tienen EXACTAMENTE dos paneles métricos horizontales: izquierda500, derecha250, cinco métodos,',
        'misma identidad visual, leyenda común y ejes compartidos. La barra de colores es un eje auxiliar, no otro panel de datos.',
        'Superficie original441 nodos: contourf inferno(24),contour negro(12),sin remuestreo/interpolación externa ni simulación.',
        'Se verificó orientación z3 horizontal,z4 vertical; ejes[0,1] y mismos niveles y rango de color en ambos paneles.',
        'Un error de sintaxis en una aserción inicial se corrigió antes de las pruebas y búsquedas. No se cambiaron algoritmos.',
        'Las dependencias existentes fueron suficientes. Aviso opcional Axes3D de Matplotlib no afecta figuras2D.',
        '', 'Comandos desde SimOpt:',
        'python3 run_green_sample_size_018.py --tests-only',
        'python3 run_green_sample_size_018.py --resume',
        'Actualizar sólo gráficos/tablas, sin simulación: python3 analyze_green_sample_size_018.py',
        'Verificar con figuras existentes: python3 analyze_green_sample_size_018.py --reuse-figures',
        '',f"Verificación: {verified['basic_test_assertions']} aserciones previas,15 búsquedas nuevas completas,0 búsquedas n500 repetidas,",
        f"{verified['historical_files_unchanged']} archivos históricos sin cambios;{verified['png']} PNG y{verified['pdf']} PDF válidos."]
    if verified['resume_verified_without_new_simulation']:
        lines+=['Se ejecutó --resume sobre el estudio completo: ningún nuevo checkpoint, ninguna simulación ni búsqueda adicional;',
                'los hashes de todos los checkpoints y CSV científicos permanecieron idénticos.']
    (OUT/'RESUMEN_EXPERIMENTAL.txt').write_text('\n'.join(lines)+'\n')

def gallery(files,summary):
    links=['RESUMEN_EXPERIMENTAL.txt','tables/final_comparison.csv','tables/method_summary.csv','tables/n250_vs_n500_individual.csv',
        'tables/sample_size_sensitivity.csv','tables/paired_n250_vs_n500.csv','tables/paired_vs_green.csv','tables/coordinates.csv',
        'tables/objective_traces.csv','tables/iteration_diagnostics.csv','tables/diagnostic_workload.csv',
        'tables/run_metrics.csv','tables/gradient_spread_summary.csv',
        'validation/figure_layout.json','validation/surface_style_check.json','validation/verification.json']
    page=['<!doctype html><html lang="es"><meta charset="utf-8"><title>I0252 n500 vs n250</title>',
          '<style>body{font:16px system-ui;max-width:1600px;margin:30px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}img{max-width:100%}article{margin:40px 0}a{color:#075da8}</style>',
          '<h1>I0252 · cinco métodos Green · n500 frente a n250</h1>',
          '<p>Rep1–rep3;100 iteraciones;15 nuevas búsquedas n250. Todos los gráficos: izquierda n500,derecha n250; leyenda y escalas comunes.</p>',
          '<p>'+' · '.join(f'<a href="{p}">{html.escape(p)}</a>' for p in links)+'</p>',
          '<table><tr><th>n</th><th>Método</th><th>Objetivo final medio</th><th>DE entre reps</th><th>Tiempo medio(s)</th></tr>']
    for r in summary.itertuples():page.append(f'<tr><td>{r.n_paths}</td><td>{html.escape(LABELS[r.method])}</td><td>{r.objective_mean:.6f}</td><td>{r.objective_sd:.6f}</td><td>{r.optimization_seconds_mean:.3f}</td></tr>')
    page+=['</table><p>IC por endpoint: ruido Monte Carlo condicional; DE entre tres reps: variación de optimización/evaluación. Tres semillas son exploratorias.</p>']
    png=sorted((p for p in files if p.suffix=='.png'),key=lambda p:(0 if 'oracle' in p.stem else 1 if 'convergence' in p.stem else 2,p.name))
    for p in png:
        rel=p.relative_to(OUT).as_posix();pdf=p.with_suffix('.pdf').relative_to(OUT).as_posix()
        page.append(f'<article><h2>{p.stem}</h2><a href="{rel}">PNG</a> · <a href="{pdf}">PDF</a><br><a href="{rel}"><img loading="lazy" src="{rel}"></a></article>')
    page.append('</html>');content='\n'.join(page);(OUT/'index.html').write_text(content)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links += [v for k,v in attrs if k in ('href','src')]
    parser=Links();parser.feed(content);assert all((OUT/f).is_file() for f in parser.links)
    return len(parser.links)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reuse-figures',action='store_true');args=ap.parse_args()
    coords=read('coordinates');diag=read('iteration_diagnostics');tr=read('objective_traces');fin=read('final_results')
    runtime=read('runtime');pairs=read('paired_vs_green');cross=read('paired_n250_vs_n500')
    full,runs,summary,comparisons,sensitivity,spread=build_tables(coords,diag,tr,fin,runtime,pairs,cross)
    if args.reuse_figures:
        files=sorted([*OUT.glob('figures/*.png'),*OUT.glob('figures/*.pdf')]);audit=json.loads((OUT/'validation/figure_layout.json').read_text())
    else:files,audit=figures(coords,diag,tr,fin,runtime,pairs,runs,summary,spread)
    verified=verify(coords,diag,tr,fin,runtime,pairs,cross,files,audit)
    (OUT/'validation/verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    report(full,runs,summary,comparisons,sensitivity,verified)
    for f in ('018probing_green_n250_vs_n500.jl','PROTOCOLO_green_n250_vs_n500_018.txt','run_green_sample_size_018.py','analyze_green_sample_size_018.py'):
        shutil.copyfile(ROOT/f,OUT/'reproduction'/f)
    verified['gallery_links_verified']=gallery(files,summary)
    (OUT/'validation/verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    (OUT/'COMPLETE.json').write_text(json.dumps(verified,indent=2)+'\n');print(json.dumps(verified,indent=2))
if __name__=='__main__':main()
