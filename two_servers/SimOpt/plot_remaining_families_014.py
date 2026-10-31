"""Complete the 011 plot families using the nine saved methods, without search."""
from pathlib import Path
import sys,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_green_softmax090_014 as B


def remaining(out,items):
    d=items[0];K=int(d.iteration.max());reps=sorted(d.rep.unique());C=B.COLORS;S=B.STYLES;L=B.LABELS;M=B.METHODS
    contrasts=[]
    for rep in reps:
        fig,ax=plt.subplots(figsize=(10,5.5));p=d[d.rep==rep].pivot(index='iteration',columns='method',values='objective')
        for include,color,label in [(True,'darkorange','Nine methods, including Fixed'),(False,'royalblue','Eight methods, excluding Fixed')]:
            v=p if include else p.drop(columns='Fixed');spread=v.max(axis=1)-v.min(axis=1)
            ax.plot(spread.index,spread,color=color,label=label)
            contrasts.extend(dict(rep=int(rep),iteration=int(t),with_fixed=include,contrast=float(x)) for t,x in spread.items())
        ax.axvline(min(20,K),ls=':',color='grey');ax.set(xlabel='Iteration',ylabel='Range across methods C(t)',title=f'I0252: between-method contrast, rep{rep:02d}')
        ax.legend();ax.grid(alpha=.2);B.savefig(fig,out,f'I0252_contrast_rep{rep:02d}')
    ct=pd.DataFrame(contrasts);ct.to_csv(out/'tables/contrast_per_iteration.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5.5))
    for inc,color,label in [(True,'darkorange','Including Fixed'),(False,'royalblue','Excluding Fixed')]:
        g=ct[ct.with_fixed==inc].groupby('iteration').contrast.agg(['mean','std']);x=g.index.to_numpy();y=g['mean'].to_numpy();sd=g['std'].to_numpy()
        ax.plot(x,y,color=color,label=label);ax.fill_between(x,y-sd,y+sd,color=color,alpha=.15)
    ax.set(xlabel='Iteration',ylabel='Range across methods C(t)',title='I0252: mean contrast ± between-repetition SD');ax.legend();ax.grid(alpha=.2)
    B.savefig(fig,out,'I0252_mean_contrast')
    for zoom in [False,True]:
        fig,ax=plt.subplots(figsize=(10,6));lo=np.inf;hi=-np.inf
        for m in M:
            if zoom and m=='Fixed':continue
            g=d[d.method==m].groupby('iteration').objective.agg(['mean','std']);x=g.index.to_numpy();y=g['mean'].to_numpy();sd=g['std'].to_numpy()
            ax.plot(x,y,color=C[m],ls=S[m],label=L[m]);ax.fill_between(x,y-sd,y+sd,color=C[m],alpha=.12)
            lo=min(lo,np.min(y-sd));hi=max(hi,np.max(y+sd))
        if zoom:
            pad=max(.05,.05*(hi-lo));ax.set_ylim(lo-pad,hi+pad)
        ax.set(xlabel='Iteration',ylabel='Independent objective estimate J(z)',title='I0252: mean ± between-repetition SD'+(' (excluding Fixed)' if zoom else ' (all nine methods)'))
        ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
        B.savefig(fig,out,'I0252_mean_convergence_'+('zoom' if zoom else 'full'))
    di=pd.concat([pd.read_csv(out/f'comparisons/I0252/rep_{r:02d}_diagnostics.csv') for r in reps])
    di['clipped_components_fraction']=di.clipping/2;di['clipping_iterations_fraction']=di.clipping>0;di['projection_fraction']=di.projection>1e-12
    cols=['clipped_components_fraction','clipping_iterations_fraction','projection_fraction']
    stats=di.groupby(['rep','method'])[cols].mean();stats.to_csv(out/'tables/actual_saturation.csv')
    g=stats.groupby('method').mean().reindex(M);fig,ax=plt.subplots(figsize=(13,6));x=np.arange(len(M))
    for shift,col,label in [(-.24,cols[0],'Clipped gradient components'),(0,cols[1],'Updates with clipping'),(.24,cols[2],'Projected updates')]:
        ax.bar(x+shift,g[col],width=.24,label=label)
    ax.set_xticks(x,[L[m] for m in M],rotation=30,ha='right',fontsize=9);ax.set(ylim=(0,1.05),ylabel='Mean fraction',title='I0252: observed clipping and projection during searches')
    ax.legend();B.savefig(fig,out,'I0252_actual_saturation')
    # Preserve the model-only surface and decomposition, with their original values.
    sf=pd.read_csv(out/'source_011/I0252_surface.csv');grid=np.sort(sf.z3.unique())
    for decomposition in [False,True]:
        fig,axes=plt.subplots(1,3 if decomposition else 1,figsize=(16,5) if decomposition else (8,7),squeeze=False)
        for ax,value,label in zip(axes[0],['gross','cost','objective'] if decomposition else ['objective'],
                                 ['Gross value V(z)','Probing cost alpha·z','Net objective J(z)'] if decomposition else ['Net objective J(z)']):
            zz=sf.pivot(index='z4',columns='z3',values=value).reindex(index=grid,columns=grid).to_numpy()
            im=ax.contourf(grid,grid,zz,levels=24,cmap='inferno');ax.contour(grid,grid,zz,levels=12,colors='k',linewidths=.35,alpha=.5)
            ax.set(xlabel='z3',ylabel='z4',title=f'I0252: {label}',aspect='equal');fig.colorbar(im,ax=ax,label='Original model units; separate scale')
        fig.tight_layout();B.savefig(fig,out,'I0252_gross_cost_net' if decomposition else 'I0252_surface')
    for m in M:
        for kind in ['convergence','oracle']:
            source=(f'I0252_convergence_rep01_{m}' if kind=='convergence' else f'I0252_oracle_trajectories_rep01_{m}')
            for ext in ['png','pdf']:shutil.copy2(out/'figures'/f'{source}.{ext}',out/'figures'/f'I0252_{m}_{kind}_individual.{ext}')
    f=out/'tables/final_holdout_by_rep.csv'
    if f.exists():
        fs=pd.read_csv(f);fig,ax=plt.subplots(figsize=(13,6));values=[fs[fs.method==m].objective_holdout.to_numpy() for m in M]
        ax.boxplot(values,positions=range(len(M)),showfliers=False)
        for j,m in enumerate(M):ax.scatter(j+np.linspace(-.12,.12,len(values[j])),values[j],color=C[m],s=35,zorder=5)
        ax.set_xticks(range(len(M)),[L[m] for m in M],rotation=30,ha='right',fontsize=9)
        ax.set(ylabel='Independent selected-policy holdout value',title='I0252: all final solutions (five repetitions)');ax.grid(axis='y',alpha=.2)
        B.savefig(fig,out,'I0252_final_objectives')
    f=out/'tables/temporal_contrast_validation.csv'
    if f.exists():
        a=pd.read_csv(f).sort_values('iteration');p=d[(d.rep==1)&(d.method!='Fixed')].pivot(index='iteration',columns='method',values='objective')
        fig,ax=plt.subplots(figsize=(11,6));ax.plot(p.index,p.max(axis=1)-p.min(axis=1),color='grey',label='Eight-method range (excluding Fixed), trace bank')
        ax.errorbar(a.iteration,a.independent_difference,yerr=a.paired_se,fmt='o',color='royalblue',capsize=4,
                    label='Pair fixed before independent evaluation: holdout ± MC SE')
        for r in a.itertuples():ax.annotate(f'{r.method_high} − {r.method_low}',(r.iteration,r.independent_difference),xytext=(4,8),textcoords='offset points',fontsize=7,rotation=20)
        ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Iteration',ylabel='Objective difference',title='I0252: nine-method temporal contrast and independent validation, rep01')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8);ax.grid(alpha=.2);B.savefig(fig,out,'I0252_independent_temporal_contrast')


def main(output):
    out=Path(output).resolve();config=B.tomllib.loads((out/'configs/experiment.toml').read_text());items=B.build_metrics(out,config['science'])
    remaining(out,items);B.report_and_index(out,config,items)
    report=out/'informe_cientifico.txt'
    if (out/'tables/temporal_contrast_validation.csv').exists():
        a=pd.read_csv(out/'tables/temporal_contrast_validation.csv')
        with report.open('a') as f:
            f.write('\nAuditoría de contraste temporal: se vuelve a seleccionar el par máximo/mínimo entre los nueve métodos (excluyendo Fixed) en rep01.\n')
            f.write('El plan se fija antes de calcular nuevos valores independientes. Se reutilizan pares antiguos solo si coinciden exactamente método, iteración y estimando.\n')
            f.write(a.to_string(index=False)+'\n')
    with (out/'README_execution.txt').open('a') as f:
        f.write('\nPara repetir el flujo completo con todas las familias gráficas:\npython3 run_green_softmax090_014.py --resume\n')
    root=Path(__file__).resolve().parent
    for name in ['plot_remaining_families_014.py','audit_green_softmax090_014.jl','run_green_softmax090_014.py','analyze_green_softmax090_014.py']:
        shutil.copy2(root/name,out/'reproduction'/name)
    B.verify(out,config,items)


if __name__=='__main__':main(sys.argv[1])
