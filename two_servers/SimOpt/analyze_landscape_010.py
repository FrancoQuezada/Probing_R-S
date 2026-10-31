"""Exploratory scientific figures/report for 010. Reads saved direct simulations."""
from pathlib import Path
import sys,json,hashlib,html,platform
try: import tomllib
except ModuleNotFoundError: import tomli as tomllib
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['Fixed','Fresh','Green','Softmax010','Softmax050','PSoft300','GreenSoftmax050']
LABELS=dict(zip(METHODS,['Fixed','Fresh','Green','Softmax β=0.10','Softmax β=0.50','PSoft B=300','Green+Softmax β=0.50']))
COLORS=dict(zip(METHODS,['darkorange','black','forestgreen','royalblue','deepskyblue','purple','firebrick']))
STYLES=dict(zip(METHODS,['--','-','-', '-.','--',':','-.']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})

def loadparts(out,suffix):
    files=sorted((out/'comparisons').glob(f'L*/rep_*_{suffix}.csv'))
    return pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else pd.DataFrame()

def save(fig,out,name):
    fig.savefig(out/'figures'/f'{name}.png',bbox_inches='tight')
    fig.savefig(out/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def background(ax,surface,title,mark=True):
    grid=np.sort(surface.z3.unique())
    zz=surface.pivot(index='z4',columns='z3',values='objective').reindex(index=grid,columns=grid).to_numpy()
    image=ax.contourf(grid,grid,zz,levels=24,cmap='inferno')
    ax.contour(grid,grid,zz,levels=12,colors='k',linewidths=.35,alpha=.5)
    if mark:
        r=surface.loc[surface.objective.idxmax()]
        ax.scatter([r.z3],[r.z4],marker='X',s=90,color='cyan',edgecolor='black',zorder=12,label='Máximo de grilla')
    ax.set(xlabel='z3',ylabel='z4',title=title,xlim=(-.02,1.02),ylim=(-.02,1.02),aspect='equal')
    return image

def pairsummary(pairs):
    rows=[]
    for (s,a,b),g in pairs.groupby(['scenario','method_a','method_b']):
        x=g.difference.to_numpy();sd=x.std(ddof=1);se=sd/np.sqrt(len(x));c=stats.t.ppf(.975,len(x)-1)
        rows.append(dict(scenario=s,method_a=a,method_b=b,reps=len(x),mean_difference=x.mean(),sd=sd,se=se,
                         ci95_low=x.mean()-c*se,ci95_high=x.mean()+c*se,positive_fraction=np.mean(x>0),
                         mean_paired_mc_se=g.paired_se.mean(),status='exploratory'))
    return pd.DataFrame(rows)

def main(out):
    out=Path(out).resolve();config=tomllib.loads((out/'configs/experiment.toml').read_text())
    science=config['science'];quick=science['quick'];selected=pd.read_csv(out/'configs/selected_instances.csv')
    ids=selected.scenario.tolist();models=tomllib.loads((out/'configs/models.toml').read_text())
    screen=pd.read_csv(out/'screen/metrics.csv');high=pd.read_csv(out/'high/metrics.csv')
    configurations=pd.read_csv(out/'configs/configurations.csv')
    traces=loadparts(out,'traces');coords=loadparts(out,'coordinates');finals=loadparts(out,'finals')
    diag=loadparts(out,'diagnostics');pairs=loadparts(out,'paired');gaps=loadparts(out,'grid_gaps')
    for key,df in [('traces',traces),('coordinates',coords),('finals',finals),('diagnostics',diag),('paired',pairs),('grid_gaps',gaps)]:
        df.to_csv(out/'tables'/f'{key}_all.csv',index=False,float_format='%.17g')
    summary=finals[finals.method.isin(METHODS)].groupby(['scenario','method'],sort=False).agg(
        mean_objective=('objective_holdout','mean'),sd=('objective_holdout','std'),mean_direct=('objective_direct','mean'),
        mean_gross=('gross','mean'),mean_cost=('probing_cost','mean'),mean_mc_se=('evaluation_se','mean'),
        mean_z3=('z3','mean'),mean_z4=('z4','mean'),reps=('rep','size')).reset_index()
    summary['se_between_reps']=summary.sd/np.sqrt(summary.reps)
    summary.to_csv(out/'tables/final_summary.csv',index=False)
    contrast=pairsummary(pairs);contrast.to_csv(out/'tables/paired_exploratory.csv',index=False)
    stability=diag.groupby(['scenario','rep','method']).agg(total_variation=('step_norm','sum'),
        tail_mean_step=('step_norm',lambda x:x.iloc[-min(20,len(x)):].mean()),clip_fraction=('clipping',lambda x:np.mean(x>0)),
        projection_fraction=('projection',lambda x:np.mean(x>1e-12))).reset_index()
    stability.to_csv(out/'tables/trajectory_stability.csv',index=False)
    average=traces.groupby(['scenario','method','iteration']).agg(mean=('objective','mean'),sd=('objective','std'),reps=('objective','size')).reset_index()
    average.to_csv(out/'tables/mean_convergence.csv',index=False)
    configured=screen.merge(configurations,on='scenario',validate='one_to_one')
    configured.to_csv(out/'tables/screen_with_parameters.csv',index=False)
    base=configured[configured.scenario=='L000'].iloc[0]
    oat=configured[configured.origin.str.startswith('one_factor')].copy()
    for k in ['R_search','R_initial','gross_range','R_global']:
        oat[f'delta_{k}_from_original']=oat[k]-base[k]
    oat.to_csv(out/'tables/one_factor_effects.csv',index=False)
    # Descriptive adjusted linear association, not a causal attribution of interactions.
    factors=['delay','std_complex','fast_complex','signal','beta_signal','alpha3','mean_interarrival']
    xx=configured[factors].to_numpy(float);xstd=xx.std(axis=0);xx=(xx-xx.mean(axis=0))/np.where(xstd>0,xstd,1)
    fit=np.linalg.lstsq(np.column_stack([np.ones(len(xx)),xx]),configured.R_search,rcond=None)[0]
    pd.DataFrame({'factor':factors,'standardized_linear_coefficient':fit[1:]}).to_csv(out/'tables/descriptive_parameter_associations.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].scatter(configured.R_global,configured.R_search,c=configured.delay,cmap='viridis')
    axes[0].set(xlabel='Contraste global',ylabel='Contraste en [0,0.75]²',title='Todas las configuraciones, incluidos resultados negativos')
    axes[1].scatter(configured.R_initial,configured.R_search,c=configured.corner_best.map({True:'gray',False:'royalblue'}))
    axes[1].set(xlabel='Mejora de grilla respecto de z0',ylabel='Contraste en [0,0.75]²',title='Gris: mejor punto en esquina')
    for s in ids:
        r=configured[configured.scenario==s].iloc[0];axes[1].annotate(s,(r.R_initial,r.R_search))
    fig.tight_layout();save(fig,out,'screening_all_configurations')
    surfaces={s:pd.read_csv(out/'high'/f'{s}_surface.csv') for s in ids}
    for s in ids:
        surface=surfaces[s];sf=coords[(coords.scenario==s)&(coords.rep==1)]
        fig,ax=plt.subplots(figsize=(7.5,6.5));im=background(ax,surface,f'{s}: objetivo original J(z)')
        fig.colorbar(im,ax=ax,label='J(z), unidades originales');ax.legend();save(fig,out,f'{s}_surface')
        fig,ax=plt.subplots(figsize=(9,7));im=background(ax,surface,f'{s}: trayectorias, repetición predefinida 1')
        for method in METHODS:
            q=sf[sf.method==method].sort_values('iteration')
            ax.plot(q.z3,q.z4,color=COLORS[method],ls=STYLES[method],lw=1.5,label=LABELS[method])
            ax.scatter(q.z3.iloc[-1],q.z4.iloc[-1],color=COLORS[method],s=45,edgecolor='white',zorder=10)
        ax.scatter(.12,.10,marker='*',color='gold',edgecolor='black',s=130,zorder=11,label='Inicio')
        fig.colorbar(im,ax=ax,label='J(z)');ax.legend(loc='upper left',bbox_to_anchor=(1.28,1),fontsize=8)
        save(fig,out,f'{s}_oracle_trajectories_rep01')
        for rep in sorted(traces.rep.unique()):
            t=traces[(traces.scenario==s)&(traces.rep==rep)]
            for zoom in [False,True]:
                fig,ax=plt.subplots(figsize=(10,6))
                for method in METHODS:
                    q=t[t.method==method].sort_values('iteration')
                    ax.plot(q.iteration,q.objective,color=COLORS[method],ls=STYLES[method],label=LABELS[method],lw=1.7)
                if zoom:
                    q=t[t.method!='Fixed'].objective;lo,hi=q.min(),q.max();pad=max(.05,.05*(hi-lo));ax.set_ylim(lo-pad,hi+pad)
                ax.set(xlabel='Iteration',ylabel='Objective value J(z)',title=f'{s}: repetición {rep}; evaluación directa CRN'+(' — zoom sin rango de Fixed' if zoom else ' — escala completa'))
                ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=9);ax.grid(alpha=.2)
                save(fig,out,f'{s}_convergence_rep{rep:02d}'+('_zoom' if zoom else '_full'))
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for dim,ax in zip(['z3','z4'],axes):
            for method in METHODS:
                q=sf[sf.method==method].sort_values('iteration')
                ax.plot(q.iteration,q[dim],color=COLORS[method],ls=STYLES[method],label=LABELS[method],lw=1.6)
            ax.set(xlabel='Iteration',ylabel=dim,ylim=(-.025,1.025),title=f'{s}: Class-{dim[-1]} probing trajectory');ax.grid(alpha=.2)
        axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8);fig.tight_layout();save(fig,out,f'{s}_probing_coordinates_rep01')
        for zoom in [False,True]:
            fig,ax=plt.subplots(figsize=(10,6))
            sub=average[average.scenario==s]
            for method in METHODS:
                q=sub[sub.method==method].sort_values('iteration');x=q.iteration.to_numpy();y=q['mean'].to_numpy();sd=q.sd.to_numpy()
                ax.plot(x,y,color=COLORS[method],ls=STYLES[method],label=LABELS[method]);ax.fill_between(x,y-sd,y+sd,color=COLORS[method],alpha=.12)
            if zoom:
                q=sub[sub.method!='Fixed'];lo=(q['mean']-q.sd).min();hi=(q['mean']+q.sd).max();pad=max(.05,.05*(hi-lo));ax.set_ylim(lo-pad,hi+pad)
            ax.set(xlabel='Iteration',ylabel='Objective value J(z)',title=f'{s}: media ± SD de {science["reps"]} repeticiones'+(' — zoom' if zoom else ''))
            ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=9);ax.grid(alpha=.2);save(fig,out,f'{s}_mean_convergence'+('_zoom' if zoom else '_full'))
        fig,ax=plt.subplots(figsize=(11,5.5));f=finals[(finals.scenario==s)&finals.method.isin(METHODS)]
        vals=[f[f.method==m].objective_holdout.to_numpy() for m in METHODS]
        ax.boxplot(vals,positions=range(len(METHODS)),widths=.5,showfliers=False)
        for j,(m,v) in enumerate(zip(METHODS,vals)):
            ax.scatter(j+np.linspace(-.12,.12,len(v)),v,color=COLORS[m],s=30,zorder=4)
            sm=summary[(summary.scenario==s)&(summary.method==m)].iloc[0]
            ax.errorbar(j,sm.mean_objective,yerr=sm.mean_mc_se,fmt='o',color=COLORS[m],capsize=3)
        ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=20,ha='right')
        ax.set(ylabel='Objetivo holdout de la política seleccionada',title=f'{s}: todas las soluciones finales; puntos = repeticiones; barras = SE MC medio')
        ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,out,f'{s}_final_objectives')
    fig,axes=plt.subplots(1,len(ids),figsize=(5*len(ids),4.8),squeeze=False)
    for ax,s in zip(axes[0],ids):
        im=background(ax,surfaces[s],s);fig.colorbar(im,ax=ax,shrink=.8,label='J(z), escala propia')
    fig.tight_layout();save(fig,out,'selected_surfaces_comparison')
    report=[]
    def add(text=''):report.append(str(text))
    add('010 — BÚSQUEDA EXPLORATORIA DE INSTANCIAS DE PROBING')
    add('PRUEBA REDUCIDA; NO CONCLUSIONES CIENTÍFICAS' if quick else 'Resultados exploratorios; cinco repeticiones, sin afirmación de superioridad universal.')
    add(f'Configuraciones: {len(screen)}; refinadas: {len(pd.read_csv(out/"refine/metrics.csv"))}; seleccionadas: {len(ids)}.')
    add(f'Presupuestos: screening {science["screen_grid"]}²×{science["screen_paths"]}, refinamiento {science["refine_grid"]}²×{science["refine_paths"]}, alta precisión {science["high_grid"]}²×{science["high_paths"]}.')
    add(f'Algoritmos: {science["n_paths"]} caminos × {science["max_iters"]} iteraciones × {science["reps"]} repeticiones × {len(METHODS)} métodos por instancia; {science["jobs"]} trabajos/camino.')
    add(f'Trazas directas: {science["trace_paths"]}; selector final: {science["selection_paths"]}; holdout: {science["final_paths"]}.')
    add('La política final se elige en un banco y evalúa en otro: su holdout estima el valor de esa política, una aproximación inferior a J si la selección es imperfecta. objective_direct conserva max_i media(Y_i)-costo, con posible optimismo.')
    add('Los SE de superficie/SNR son condicionales a políticas escogidas en la muestra; no corrigen selección de máximos. GridBest se reevalúa con selección/holdout independientes de la superficie.')
    add('La banda de convergencia promedio es SD entre repeticiones, no intervalo de confianza. La repetición ilustrativa es siempre 1; se conservan todas las restantes.')
    add()
    for s in ids:
        m=high[high.scenario==s].iloc[0];p=models[s];ranking=summary[summary.scenario==s].sort_values('mean_objective',ascending=False)
        add(f'INSTANCIA {s}: {selected[selected.scenario==s].iloc[0].reason}')
        add(json.dumps(p,ensure_ascii=False,sort_keys=True))
        add(f'Rglobal={m.R_global:.4f}; Rsearch={m.R_search:.4f}; mejora desde inicio={m.R_initial:.4f}; máximo-mediana={m.best_minus_search_median:.4f}.')
        add(f'Mejor punto de grilla=({m.best_z3:.2f},{m.best_z4:.2f}), Jgrilla={m.best_value:.4f}; distancia frontera={m.boundary_distance:.3f}; área competitiva={m.competitive_fraction:.3f}; componentes observadas={int(m.competitive_components)}.')
        add(f'Rango bruto={m.gross_range:.4f}; rango de costos={m.cost_range:.4f}; SNR inicial descriptivo={m.initial_snr:.2f}; SNR contraste búsqueda={m.search_snr:.2f}.')
        add('Resultados finales (media holdout, SD entre repeticiones, SE MC medio, bruto, costo):')
        add(ranking[['method','mean_objective','sd','mean_mc_se','mean_gross','mean_cost']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
        delay=p['DELAY_PENALTY'];std=p['STD_MEAN'][1];fast=p['FAST_MEAN'][1]
        add(f'Interpretación económica: penalización por tiempo en sistema {delay:g}, servicio complejo estándar/rápido {std:g}/{fast:g}; más separación modifica congestión y costo del routing incorrecto. La información de clases 3/4 responde a beta_signal={p["BETA"]:g}; alpha={p["ALPHA"]} define el compromiso neto. Ingreso común preservado en 8.')
        add()
    add('RESPUESTAS A LAS PREGUNTAS')
    top=configured.sort_values('R_search',ascending=False).head(6)
    add('P1. Mayor contraste preliminar:');add(top[['scenario','delay','std_complex','fast_complex','signal','beta_signal','alpha3','mean_interarrival','R_search','R_initial']].to_string(index=False))
    add('P2. Variaciones de un factor frente al original (no efecto causal general de interacciones):')
    add(oat[['scenario','origin','delta_R_search_from_original','delta_R_initial_from_original','delta_gross_range_from_original']].to_string(index=False))
    add('Asociaciones lineales ajustadas, sólo descriptivas: '+str(dict(zip(factors,np.round(fit[1:],4)))))
    interiors=high[high.boundary_distance>0]
    add('P3. Máximos de grilla interiores: '+(', '.join(interiors.scenario) if len(interiors) else 'ninguno en las instancias seleccionadas')+'. No certifican óptimos poblacionales.')
    add('P4. Geometrías: '+ '; '.join(f'{r.scenario}: área competitiva {r.competitive_fraction:.3f}, componentes {int(r.competitive_components)}, frontera {r.boundary_distance:.2f}' for r in high.itertuples())+'. Componentes pequeñas pueden deberse al ruido/discretización; no se fuerza categoría multimodal.')
    add('P5/P6. Diferencia de trayectoria y dispersión del valor final medio:')
    diversity=[]
    for s in ids:
        q=coords[(coords.scenario==s)&(coords.iteration==science['max_iters'])]
        centers=q.groupby('method')[['z3','z4']].mean();v=summary[summary.scenario==s];nf=v[v.method!='Fixed']
        dist=max(np.linalg.norm(a-b) for a in centers.to_numpy() for b in centers.to_numpy())
        add(f'{s}: máxima distancia entre decisiones finales medias {dist:.4f}; rango entre objetivos medios {v.mean_objective.max()-v.mean_objective.min():.4f}; sin Fixed {nf.mean_objective.max()-nf.mean_objective.min():.4f}.')
        diversity.append((s,nf.mean_objective.max()-nf.mean_objective.min()))
    add('P7/P8. Diferencias Softmax−Fresh emparejadas; IC t95% exploratorio sobre repeticiones:')
    q=contrast[(contrast.method_a.isin(['Softmax010','Softmax050']))&(contrast.method_b=='Fresh')]
    add(q[['scenario','method_a','mean_difference','ci95_low','ci95_high','positive_fraction']].to_string(index=False))
    add('Los signos muestran ejemplos a favor/en contra; los intervalos que incluyen cero no sustentan una diferencia concluyente. Se seleccionaron instancias por geometría, no por ganadores.')
    add('P9. Estabilidad (menor longitud total y menor paso en últimas 20 iteraciones; no implica mejor objetivo):')
    st=stability.groupby(['scenario','method'])[['total_variation','tail_mean_step','clip_fraction','projection_fraction']].mean().reset_index()
    add(st[st.method.isin(['Fresh','Green','GreenSoftmax050'])].to_string(index=False))
    add('P10. PSoft−Fresh y PSoft−Softmax:')
    add(contrast[(contrast.method_a=='PSoft300')&contrast.method_b.isin(['Fresh','Softmax010','Softmax050'])][['scenario','method_b','mean_difference','ci95_low','ci95_high']].to_string(index=False))
    recommendation=max(diversity,key=lambda x:x[1])[0]
    add(f'P11. Conservar L000 como control, la instancia de contraste y la localizada; para ilustrar separación de métodos sin Fixed, el mayor rango medio observado corresponde a {recommendation}. La selección de este ejemplo es exploratoria, posterior a comparar.')
    add(f'P12. Ejemplo principal provisional: {recommendation}; parámetros completos en configs/models.toml y arriba. Recomendación basada en diversidad visible, no en demostrar un ganador. Validar con nuevas semillas antes de inferencia confirmatoria.')
    add()
    add('PROBLEMAS Y LIMITACIONES')
    add('1. Cinco repeticiones sólo permiten diagnóstico exploratorio; no resuelven por sí solas diferencias pequeñas entre algoritmos.')
    add('2. El aumento de Rsearch puede provenir de empeorar regiones de bajo probing. R_initial, máximo-mediana, bruto/costo y comparación sin Fixed se informan para evitar concluir sólo a partir de una esquina desfavorable.')
    add('3. Los máximos locales y las componentes competitivas se observan en grillas finitas con ruido. No equivalen a óptimos locales/globales certificados.')
    add('4. Los métodos conservan sus estimadores LR y clipping originales; se guardan clipping, ESS y proyección, incluyendo trayectorias desfavorables.')
    add('5. No se cambió el ingreso común ni se reescalaron rewards. Sólo se variaron los parámetros solicitados. La tasa exponencial sería inversa de la media indicada.')
    add('Reanudar: env JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 GKSwstype=100 julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno 010probing_instance_landscape_search.jl --resume')
    (out/'informe_final.txt').write_text('\n'.join(report)+'\n')
    index=['<!doctype html><meta charset="utf-8"><title>010: paisajes y algoritmos</title>',
           '<style>body{font-family:system-ui;margin:2em;max-width:1400px}img{max-width:100%}a{margin-right:1em}table{border-collapse:collapse}td,th{padding:.4em;border:1px solid #ddd}</style>',
           '<h1>010: búsqueda exploratoria de instancias</h1>',
           '<p><a href="informe_final.txt">Informe completo P1–P12</a><a href="configs/models.toml">Parámetros completos</a><a href="tables/final_summary.csv">Valores finales</a></p>',
           '<p>Semilla ilustrativa fijada: repetición 1. Curvas promedio: media ± SD. Todas las repeticiones se conservan.</p>',
           '<img src="figures/selected_surfaces_comparison.png">']
    for s in ids:
        index.append(f'<h2>{s}</h2>')
        index.append(summary[summary.scenario==s][['method','mean_objective','sd','mean_mc_se']].to_html(index=False,float_format=lambda x:f'{x:.4f}'))
        for name in [f'{s}_surface',f'{s}_convergence_rep01_full',f'{s}_convergence_rep01_zoom',f'{s}_probing_coordinates_rep01',f'{s}_oracle_trajectories_rep01',f'{s}_mean_convergence_full',f'{s}_mean_convergence_zoom',f'{s}_final_objectives']:
            index.append(f'<p><a href="figures/{name}.png">PNG</a><a href="figures/{name}.pdf">PDF</a></p><img src="figures/{name}.png">')
    (out/'index.html').write_text('\n'.join(index))
    seeds=loadparts(out,'seeds');assert not seeds.seed.duplicated().any(),'Seed collision'
    seeds.to_csv(out/'tables/seeds_all.csv',index=False)
    expected=len(ids)*science['reps']*len(METHODS)*(science['max_iters']+1)
    assert len(traces)==expected and len(coords)==expected
    assert not traces.duplicated(['scenario','rep','method','iteration']).any()
    assert set(finals.method)>=set(METHODS)
    assert np.isfinite(traces.objective).all() and np.isfinite(finals.objective_holdout).all()
    assert np.allclose(finals.gross-finals.probing_cost,finals.objective_holdout)
    assert coords[['z3','z4']].to_numpy().min()>=0 and coords[['z3','z4']].to_numpy().max()<=1
    complete=dict(complete=True,quick=quick,fingerprint=config['fingerprint'],configurations=len(screen),selected=ids,
        reps=science['reps'],methods=METHODS,trace_rows=len(traces),png_files=len(list((out/'figures').glob('*.png'))),
        independent_seed_streams=len(seeds),environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__))
    (out/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(json.dumps(complete,indent=2))

if __name__=='__main__': main(sys.argv[1])
