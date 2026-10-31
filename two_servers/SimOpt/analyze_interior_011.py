"""Scientific analysis/figures for 011; no simulation or algorithm changes."""
from pathlib import Path
import sys,json,hashlib,platform,warnings
try:import tomllib
except ModuleNotFoundError:import tomli as tomllib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_landscape_010 import METHODS,LABELS,COLORS,STYLES,background,save,pairsummary

def parts(out,suffix,folder='comparisons'):
    files=sorted((out/folder).glob(f'*/rep_*_{suffix}.csv'))
    return pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else pd.DataFrame()

def files_table(paths):
    tables=[]
    for p in paths:
        try:tables.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:pass
    return pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()

def contrast_tables(traces):
    curves=[];summary=[]
    for (s,rep),g in traces.groupby(['scenario','rep']):
        for include in [True,False]:
            q=g if include else g[g.method!='Fixed'];p=q.pivot(index='iteration',columns='method',values='objective').sort_index()
            c=p.max(axis=1)-p.min(axis=1);kmax=p.index.max();tail=c[c.index>=min(20,kmax)]
            for t,v in c.items():curves.append(dict(scenario=s,rep=rep,with_fixed=include,iteration=t,contrast=v))
            summary.append(dict(scenario=s,rep=rep,with_fixed=include,C_all=p.to_numpy().max()-p.to_numpy().min(),
                C_tail=tail.mean(),C_max_tail=tail.max(),C_mean50=c[c.index>=min(50,kmax)].mean(),
                C_mean80=c[c.index>=min(80,kmax)].mean(),C_final=c.iloc[-1],tail_start=min(20,kmax),tail_points=len(tail)))
    return pd.DataFrame(curves),pd.DataFrame(summary)

def main(output):
    out=Path(output).resolve();config=tomllib.loads((out/'configs/experiment.toml').read_text());d=config['science']
    selected=pd.read_csv(out/'configs/selected_instances.csv');ids=selected.scenario.tolist()
    primary=tomllib.loads((out/'configs/main_instance.toml').read_text())['scenario']
    models=tomllib.loads((out/'configs/models.toml').read_text())
    (out/'configs/main_instance_parameters.toml').write_text(f'scenario = {json.dumps(primary)}\n\n[model]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in sorted(models[primary].items()))+'\n')
    candidates=pd.read_csv(out/'configs/alpha_candidates.csv');ops=pd.read_csv(out/'configs/operational_design.csv')
    slopes=pd.read_csv(out/'calibration/slopes.csv');screen=pd.read_csv(out/'screen/metrics.csv');high=pd.read_csv(out/'high/metrics.csv')
    refined=pd.read_csv(out/'refine/all_metrics.csv' if (out/'refine/all_metrics.csv').exists() else out/'refine/metrics.csv');pre=files_table([f for f in sorted((out/'diagnostics').glob('*_gradients.csv')) if f.name not in ['presearch_gradients.csv','all_presearch_gradients.csv']])
    traces=parts(out,'traces');coords=parts(out,'coordinates');finals=parts(out,'finals');diag=parts(out,'diagnostics');paired=parts(out,'paired');gaps=parts(out,'grid_gaps');seeds=parts(out,'seeds')
    for name,table in [('traces',traces),('coordinates',coords),('finals',finals),('diagnostics',diag),('paired',paired),('grid_gaps',gaps),('seeds',seeds)]:
        table.to_csv(out/'tables'/f'{name}_all.csv',index=False,float_format='%.17g')
    expected=len(ids)*d['reps']*len(METHODS)*(d['max_iters']+1)
    assert len(traces)==len(coords)==expected
    assert not traces.duplicated(['scenario','rep','method','iteration']).any()
    assert np.isfinite(traces.objective).all() and np.isfinite(finals.objective_holdout).all()
    assert np.allclose(finals.gross-finals.probing_cost,finals.objective_holdout)
    assert not seeds.seed.duplicated().any()
    c,cs=contrast_tables(traces);c.to_csv(out/'tables/contrast_per_iteration.csv',index=False);cs.to_csv(out/'tables/contrast_summary.csv',index=False)
    summary=finals[finals.method.isin(METHODS)].groupby(['scenario','method']).agg(mean_objective=('objective_holdout','mean'),
        sd=('objective_holdout','std'),mean_mc_se=('evaluation_se','mean'),mean_direct=('objective_direct','mean'),mean_gross=('gross','mean'),
        mean_cost=('probing_cost','mean'),mean_z3=('z3','mean'),mean_z4=('z4','mean'),reps=('rep','size')).reset_index()
    summary['between_rep_se']=summary.sd/np.sqrt(summary.reps);summary.to_csv(out/'tables/final_summary.csv',index=False)
    pairs=pairsummary(paired);pairs.to_csv(out/'tables/paired_exploratory.csv',index=False)
    stability=diag.groupby(['scenario','rep','method']).agg(total_variation=('step_norm','sum'),
        tail_step=('step_norm',lambda x:x.iloc[-min(20,len(x)):].mean()),clipped_components_fraction=('clipping',lambda x:x.mean()/2),
        clipping_iterations_fraction=('clipping',lambda x:np.mean(x>0)),projection_fraction=('projection',lambda x:np.mean(x>1e-12))).reset_index()
    stability.to_csv(out/'tables/trajectory_stability.csv',index=False)
    distances=[]
    for (s,rep),g in coords.groupby(['scenario','rep']):
        arrays={m:g[g.method==m].sort_values('iteration')[['z3','z4']].to_numpy() for m in METHODS}
        for i,a in enumerate(METHODS):
            for b in METHODS[i+1:]:
                dist=np.linalg.norm(arrays[a]-arrays[b],axis=1)
                distances.append(dict(scenario=s,rep=rep,method_a=a,method_b=b,mean_distance=dist.mean(),rms_distance=np.sqrt(np.mean(dist**2)),final_distance=dist[-1],max_distance=dist.max()))
    distance_table=pd.DataFrame(distances);distance_table.to_csv(out/'tables/trajectory_distances.csv',index=False)
    premean=pre.groupby(['scenario','method','point','iteration']).agg(clipped_fraction=('clipped_fraction','mean'),
        clipping_iteration_fraction=('any_clipping','mean'),projection_fraction=('projected','mean'),boundary_fraction=('boundary_contact','mean'),
        mean_step=('effective_step','mean'),mean_gradient_norm=('gradient_norm','mean')).reset_index()
    premean.to_csv(out/'tables/presearch_saturation_summary.csv',index=False)
    geom=files_table(sorted((out/'audits').glob('*_geometry.csv')));local=files_table(sorted((out/'audits').glob('*_local_differences.csv')))
    geom.to_csv(out/'tables/independent_geometry.csv',index=False);local.to_csv(out/'tables/independent_local_differences.csv',index=False)
    audit=pd.read_csv(out/'tables/trajectory_contrast_validation.csv',dtype={'selection_seed':'Int64','holdout_seed':'Int64'})
    avg=traces.groupby(['scenario','method','iteration']).agg(mean=('objective','mean'),sd=('objective','std')).reset_index()
    avg.to_csv(out/'tables/mean_convergence.csv',index=False)
    cn=cs.groupby(['scenario','with_fixed'])[['C_all','C_tail','C_max_tail','C_mean50','C_mean80','C_final']].mean().reset_index()
    cn.to_csv(out/'tables/contrast_mean_summary.csv',index=False)
    surfaces={s:pd.read_csv(out/'high'/f'{s}_surface.csv') for s in ids}
    # Calibration precision/sensitivity is displayed for all operational candidates.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for j,ax in zip([3,4],axes):
        ax.errorbar(slopes[f'slope{j}'],slopes[f'slope{j}_h10'],xerr=slopes[f'se{j}'],yerr=slopes[f'se{j}_h10'],fmt='.',alpha=.35,color='steelblue',elinewidth=.4)
        lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],color='black',lw=1)
        ax.set(xlabel=f'Pendiente V, z{j}, h=.05',ylabel='Pendiente V, h=.10',title=f'Calibración: componente {j}; barras = SE condicional')
    fig.tight_layout();save(fig,out,'calibration_step_sensitivity')
    sc=screen.merge(candidates,on='scenario',suffixes=('','_candidate'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].scatter(sc.R_search,sc.boundary_distance,c=sc.calibrated.map({True:'royalblue',False:'gray'}),s=12,alpha=.5)
    axes[0].axhline(.1,color='black',ls='--');axes[0].set(xlabel='R_search en [0,.85]²',ylabel='Distancia a frontera del mejor nodo',title='Todos los candidatos netos: azul calibrado; gris control')
    axes[1].scatter(sc.R_search,sc.epsilon_mc_fraction,c=sc.boundary_distance,cmap='viridis',s=12,alpha=.5)
    axes[1].set(xlabel='R_search',ylabel='Fracción competitiva con epsilon MC',title='Contraste y localización; screening de 500 caminos')
    fig.tight_layout();save(fig,out,'screening_interiority_localization')
    for s in ids:
        sf=surfaces[s];q0=coords[(coords.scenario==s)&(coords.rep==1)]
        fig,ax=plt.subplots(figsize=(7.5,6.5));im=background(ax,sf,f'{s}: objetivo J(z)');fig.colorbar(im,ax=ax,label='J(z)');ax.legend();save(fig,out,f'{s}_surface')
        fig,axes=plt.subplots(1,3,figsize=(15,4.8))
        for ax,value,title in zip(axes,['gross','cost','objective'],['Beneficio bruto V(z)','Costo alpha·z','Objetivo neto J(z)']):
            data=sf.copy();data['objective']=sf[value];im=background(ax,data,f'{s}: {title}',mark=value=='objective');fig.colorbar(im,ax=ax,shrink=.8,label='Unidades originales; escala propia')
        fig.tight_layout();save(fig,out,f'{s}_gross_cost_net')
        fig,ax=plt.subplots(figsize=(9,7));im=background(ax,sf,f'{s}: trayectorias, repetición predefinida 1')
        for m in METHODS:
            q=q0[q0.method==m].sort_values('iteration');ax.plot(q.z3,q.z4,color=COLORS[m],ls=STYLES[m],lw=1.5,label=LABELS[m]);ax.scatter(q.z3.iloc[-1],q.z4.iloc[-1],s=40,color=COLORS[m],edgecolor='white',zorder=10)
        ax.scatter(.12,.10,marker='*',s=130,color='gold',edgecolor='black',zorder=11,label='Inicio');fig.colorbar(im,ax=ax,label='J(z)')
        ax.legend(loc='upper left',bbox_to_anchor=(1.28,1),fontsize=8);save(fig,out,f'{s}_oracle_trajectories_rep01')
        for dim in ['z3','z4']:
            fig,ax=plt.subplots(figsize=(10,6))
            for m in METHODS:
                q=q0[q0.method==m].sort_values('iteration');ax.plot(q.iteration,q[dim],color=COLORS[m],ls=STYLES[m],label=LABELS[m])
            ax.set(xlabel='Iteration',ylabel=dim,ylim=(-.025,1.025),title=f'{s}: Class-{dim[-1]} probing trajectory, rep. 1')
            ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=9);ax.grid(alpha=.2);save(fig,out,f'{s}_{dim}_trajectory_rep01')
        for rep in sorted(traces.rep.unique()):
            t=traces[(traces.scenario==s)&(traces.rep==rep)]
            for zoom in [False,True]:
                fig,ax=plt.subplots(figsize=(10,6))
                for m in METHODS:
                    if zoom and m=='Fixed':continue
                    q=t[t.method==m].sort_values('iteration');ax.plot(q.iteration,q.objective,color=COLORS[m],ls=STYLES[m],label=LABELS[m],lw=1.6)
                if zoom:
                    q=t[t.method!='Fixed'].objective;pad=max(.05,.05*(q.max()-q.min()));ax.set_ylim(q.min()-pad,q.max()+pad)
                ax.set(xlabel='Iteration',ylabel='Objective value J(z)',title=f'{s}: rep. {rep}, CRN independiente n={d["trace_paths"]}'+(' — zoom sin rango Fixed' if zoom else ' — escala completa'))
                ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=9);ax.grid(alpha=.2);save(fig,out,f'{s}_convergence_rep{rep:02d}'+('_zoom' if zoom else '_full'))
            fig,ax=plt.subplots(figsize=(10,5))
            for inc,col,label in [(True,'darkorange','Con Fixed'),(False,'royalblue','Sin Fixed')]:
                q=c[(c.scenario==s)&(c.rep==rep)&(c.with_fixed==inc)].sort_values('iteration');ax.plot(q.iteration,q.contrast,color=col,label=label)
            ax.axvline(min(20,d['max_iters']),color='gray',ls=':');ax.set(xlabel='Iteration',ylabel='C(t): rango entre métodos',title=f'{s}: contraste por iteración, rep. {rep}');ax.legend();ax.grid(alpha=.2)
            save(fig,out,f'{s}_contrast_rep{rep:02d}')
        for zoom in [False,True]:
            fig,ax=plt.subplots(figsize=(10,6));sub=avg[avg.scenario==s]
            for m in METHODS:
                if zoom and m=='Fixed':continue
                q=sub[sub.method==m].sort_values('iteration');x=q.iteration.to_numpy();y=q['mean'].to_numpy();sd=q.sd.to_numpy()
                ax.plot(x,y,color=COLORS[m],ls=STYLES[m],label=LABELS[m]);ax.fill_between(x,y-sd,y+sd,color=COLORS[m],alpha=.12)
            if zoom:
                q=sub[sub.method!='Fixed'];lo=(q['mean']-q.sd).min();hi=(q['mean']+q.sd).max();pad=max(.05,.05*(hi-lo));ax.set_ylim(lo-pad,hi+pad)
            ax.set(xlabel='Iteration',ylabel='Objective value J(z)',title=f'{s}: media ± SD, {d["reps"]} repeticiones'+(' — zoom' if zoom else ''))
            ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=9);ax.grid(alpha=.2);save(fig,out,f'{s}_mean_convergence'+('_zoom' if zoom else '_full'))
        fig,ax=plt.subplots(figsize=(10,5))
        for inc,col,label in [(True,'darkorange','Con Fixed'),(False,'royalblue','Sin Fixed')]:
            q=c[(c.scenario==s)&(c.with_fixed==inc)].groupby('iteration').contrast.agg(['mean','std']);x=q.index.to_numpy();y=q['mean'].to_numpy();sd=q['std'].to_numpy()
            ax.plot(x,y,color=col,label=label);ax.fill_between(x,y-sd,y+sd,color=col,alpha=.15)
        ax.set(xlabel='Iteration',ylabel='C(t)',title=f'{s}: contraste medio ± SD entre repeticiones');ax.legend();ax.grid(alpha=.2);save(fig,out,f'{s}_mean_contrast')
        fig,ax=plt.subplots(figsize=(11,5.5));f=finals[(finals.scenario==s)&finals.method.isin(METHODS)]
        values=[f[f.method==m].objective_holdout.to_numpy() for m in METHODS];ax.boxplot(values,positions=range(7),showfliers=False)
        for j,m in enumerate(METHODS):ax.scatter(j+np.linspace(-.12,.12,len(values[j])),values[j],color=COLORS[m],s=30,zorder=5)
        ax.set_xticks(range(7),[LABELS[m] for m in METHODS],rotation=20,ha='right');ax.set(ylabel='Valor holdout de política seleccionada',title=f'{s}: todas las soluciones finales');ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,out,f'{s}_final_objectives')
    fig,axes=plt.subplots(1,len(ids),figsize=(5*len(ids),4.8),squeeze=False)
    for ax,s in zip(axes[0],ids):im=background(ax,surfaces[s],s);fig.colorbar(im,ax=ax,shrink=.8,label='J(z), escala propia')
    fig.tight_layout();save(fig,out,'selected_surfaces_comparison')
    # Show actual saturation separately from fixed-point presearch diagnostics.
    for s in ids:
        q=stability[stability.scenario==s].groupby('method')[['clipped_components_fraction','clipping_iterations_fraction','projection_fraction']].mean().reindex(METHODS)
        fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(METHODS))
        for shift,col,label in [(-.24,'clipped_components_fraction','Componentes clipped'),(0.,'clipping_iterations_fraction','Pasos con clipping'),(.24,'projection_fraction','Pasos proyectados')]:
            ax.bar(x+shift,q[col],width=.24,label=label)
        ax.set_xticks(x,[LABELS[m] for m in METHODS],rotation=20,ha='right');ax.set(ylim=(0,1.05),ylabel='Fracción media',title=f'{s}: saturación observada durante las búsquedas')
        ax.legend();fig.tight_layout();save(fig,out,f'{s}_actual_saturation')
        q=traces[(traces.scenario==s)&(traces.rep==1)&(traces.method!='Fixed')].pivot(index='iteration',columns='method',values='objective')
        a=audit[audit.scenario==s].sort_values('iteration')
        fig,ax=plt.subplots(figsize=(10,5));ax.plot(q.index,q.max(axis=1)-q.min(axis=1),color='gray',alpha=.7,label='C(t) sin Fixed, banco de traza')
        ax.errorbar(a.iteration,a.independent_difference,yerr=a.paired_se,fmt='o',color='royalblue',capsize=4,label='Par fijado: nuevo holdout ± SE (o endpoint reutilizado)')
        ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Iteration',ylabel='Diferencia de valor',title=f'{s}: contraste temporal y validación independiente, rep. 1');ax.legend();ax.grid(alpha=.2)
        save(fig,out,f'{s}_independent_temporal_contrast')
    for method in METHODS:
        curve=coords[(coords.scenario==primary)&(coords.rep==1)&(coords.method==method)].sort_values('iteration')
        fig,ax=plt.subplots(figsize=(8,7));im=background(ax,surfaces[primary],f'{primary}: {LABELS[method]}, rep. 1')
        ax.plot(curve.z3,curve.z4,color=COLORS[method],lw=2,label=LABELS[method]);ax.scatter(curve.z3.iloc[0],curve.z4.iloc[0],marker='*',s=120,color='gold',edgecolor='black',label='Inicio',zorder=12)
        ax.scatter(curve.z3.iloc[-1],curve.z4.iloc[-1],s=60,color=COLORS[method],edgecolor='white',label='Final',zorder=11)
        fig.colorbar(im,ax=ax,label='J(z)');ax.legend(loc='upper left',bbox_to_anchor=(1.28,1),fontsize=9);save(fig,out,f'{primary}_{method}_oracle_individual')
        q=traces[(traces.scenario==primary)&(traces.rep==1)&(traces.method==method)].sort_values('iteration')
        fig,ax=plt.subplots(figsize=(9,5));ax.plot(q.iteration,q.objective,color=COLORS[method],lw=2,label=LABELS[method])
        ax.set(xlabel='Iteration',ylabel='Objective value J(z)',title=f'{primary}: {LABELS[method]}, rep. 1; evaluación directa CRN n={d["trace_paths"]}')
        ax.grid(alpha=.2);ax.legend();save(fig,out,f'{primary}_{method}_convergence_individual')

    report=[]
    def add(s=''):report.append(str(s))
    add('011 — ALTO CONTRASTE Y COMPROMISOS INTERIORES')
    add('PRUEBA REDUCIDA: NO CONCLUSIONES CIENTÍFICAS' if d['quick'] else 'Estudio exploratorio; selección por geometría y saturación antes de comparar algoritmos.')
    add(f'Operacionales: {len(ops)}; candidatos netos: {len(candidates)}; pendientes precisas: {int(slopes.usable.sum())}/{len(slopes)} anclas. Refinadas: {len(refined)}. Finales: {ids}.')
    add(f'Calibración {d["calibration_paths"]} caminos, h=.05/.10; screening {d["screen_grid"]}²×{d["screen_paths"]}; refinamiento {d["refine_grid"]}²×{d["refine_paths"]}; alta precisión {d["high_grid"]}²×{d["high_paths"]}.')
    add(f'Por método: {d["n_paths"]} caminos×{d["max_iters"]} iteraciones×{d["reps"]} repeticiones; 45 trabajos. Pasos .035/sqrt(t), clip +/-15, proyección [0,1]², sin cambios.')
    reused=out/'configs/reused_comparisons.csv';nr=len(pd.read_csv(reused)) if reused.exists() else 0
    add(f'Comparaciones: {len(ids)*d["reps"]*7} búsquedas de métodos, {nr*7} reutilizadas exactamente de 010 y {(len(ids)*d["reps"]-nr)*7} nuevas. Se conservan fuentes, semillas y presupuestos.')
    add('Objetivo siempre max_i E_z[Y_i]-alpha·z, selección ex ante. NO E(max Y). Superficies/calibración/trazas por simulación directa, nunca LR. Los algoritmos mantienen sus estimadores originales.')
    if (out/'configs/refinement_extension_protocol.toml').exists():
        add('El refinamiento inicial de 10 casos mostró saturación elevada en los siete nuevos. Se añadieron tres estratos de penalización .84,1.0,1.2 por geometría y se conservaron los 13 resultados. La selección final y recomendación permanecieron anteriores a comparar algoritmos; ver refinement_extension_protocol.toml.')
    add('Las pendientes de V y las curvaturas son diferencias finitas del máximo de medias, con SE condicional a políticas seleccionadas; no prueban diferenciabilidad ni optimalidad. La geometría se valida con bancos distintos de calibración.')
    add('Finales: selector independiente y holdout separado. El holdout estima el valor de una política elegida, cuyo valor poblacional <=J(z). El máximo muestral directo es otro estimando y puede tener optimismo. No se exige desigualdad entre estimaciones de bancos distintos.')
    add('La banda de convergencia es SD, combina variación de búsqueda y de bancos de evaluación; no es IC. Las comparaciones CRN son dentro de repetición. Diferentes bancos pueden desplazar niveles absolutos frente a superficies/finales.')
    add('El SNR 0/0 es no definido (NaN). Todas las semillas, incluidas las desfavorables, se conservan. La repetición ilustrativa es siempre 1.')
    add()
    for s in ids:
        add(f'INSTANCIA {s}'+(' — EJEMPLO PRINCIPAL PRESELECCIONADO' if s==primary else ''))
        add(json.dumps(models[s],ensure_ascii=False,sort_keys=True))
        m=high[high.scenario==s].iloc[0];cand=candidates[candidates.scenario==s].iloc[0]
        add(f'Costo {cand.origin}; ancla=({cand.anchor3},{cand.anchor4}); factores=({cand.scale3},{cand.scale4}).')
        if cand.calibrated:
            match=slopes[(slopes.op==cand.op)&np.isclose(slopes.z3,cand.anchor3)&np.isclose(slopes.z4,cand.anchor4)].iloc[0]
            add(f'Pendientes brutas h=.05: ({match.slope3:.6f}, {match.slope4:.6f}), SE=({match.se3:.6f}, {match.se4:.6f}); h=.10: ({match.slope3_h10:.6f}, {match.slope4_h10:.6f}). Cambió índice de política={match.policy_switch}; puede representar empates/equivalencia, no prueba de no diferenciabilidad.')

        add(f'Rglobal={m.R_global:.4f}, Rsearch={m.R_search:.4f}, mejora inicial={m.R_initial:.4f}; mejor nodo=({m.best_z3:.2f},{m.best_z4:.2f}), J={m.best_value:.4f}, interioridad={m.boundary_distance:.3f}.')
        add(f'Áreas epsilon .5/1/2/MC={m.epsilon05_fraction:.3f}/{m.epsilon1_fraction:.3f}/{m.epsilon2_fraction:.3f}/{m.epsilon_mc_fraction:.3f}; eps_MC={m.epsilon_mc:.4f}; dispersión RMS={m.competitive_spatial_rms:.3f}; diámetro={m.competitive_diameter:.3f}; componentes={int(m.competitive_components)}.')
        add(f'Rango bruto={m.gross_range:.4f}, rango costos={m.cost_range:.4f}; pérdida vecinos .1/.2={m.loss10_mean:.4f}/{m.loss20_mean:.4f}. Segundo orden es descriptivo; cambio política={m.curvature_policy_switch}.')
        qgeom=geom[geom.scenario==s];validated=qgeom.loc[qgeom.objective_holdout.idxmax()]
        add(f'En el banco independiente, mejor punto entre los auditados=({validated.z3:.2f},{validated.z4:.2f}), rol={validated.role}; mejora frente al nodo de grilla={-validated.grid_reference_loss:.4f}, SE emparejado={validated.paired_se:.4f}. Este máximo sobre puntos auditados también es exploratorio, no una nueva certificación.')
        if validated.grid_reference_loss < -2*validated.paired_se:
            add('El nodo de alta precisión NO permanece como mejor punto en la auditoría independiente: la región interior se sostiene, pero su ubicación puntual está afectada por ruido de grilla. Los gaps conservan la referencia original congelada y pueden resultar negativos.')

        add(summary[summary.scenario==s][['method','mean_objective','sd','mean_mc_se','mean_gross','mean_cost']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
        add()
    add('RESPUESTAS P1–P12')
    add('P1. Parámetros físicos y costos de los casos refinados; identificar interacciones, no atribuir causalidad a un único factor confundido:')
    add(refined.merge(candidates,on='scenario',suffixes=('','_c'))[['scenario','delay','std_complex','fast_complex','beta_signal','signal','alpha3','alpha4','R_search','boundary_distance']].to_string(index=False))
    if primary=='I0252':add('En el ejemplo principal, frente a L025 se mantienen penalización .84, servicio complejo 12/1.8 y BETA_SIGNAL=4; cambian la calidad inicial de señales y ALPHA. El rango bruto pasa de {:.2f} a {:.2f}, sin escalar artificialmente el reward. No se puede atribuir ese cambio a ALPHA, que sólo resta el plano de costos.'.format(high.loc[high.scenario=='L025','gross_range'].iloc[0],high.loc[high.scenario==primary,'gross_range'].iloc[0]))
    add('P2. Frecuencias descriptivas de mejor nodo interior en screening (no probabilidades poblacionales, pues los costos se construyeron para ese fin):')
    for cal in [False,True]:
        q=sc[sc.calibrated==cal];add(f'calibrado={cal}: {int((q.boundary_distance>=.1-1e-9).sum())}/{len(q)}. Confirmar sólo con superficies/auditorías independientes seleccionadas.')
    add('La igualdad de pendiente y ALPHA es sólo condición necesaria bajo diferenciabilidad, no condición suficiente de máximo. Ver sensibilidad h=.05/.10 y cambios de política en calibration/slopes.csv.')
    add('P3. Localización de alta precisión (tolerancias diferentes se mantienen separadas):')
    add(high[['scenario','boundary_distance','epsilon05_fraction','epsilon1_fraction','epsilon2_fraction','epsilon_mc','epsilon_mc_fraction','competitive_spatial_rms','competitive_diameter']].to_string(index=False))
    add('Auditoría independiente de desplazamientos .1/.2 y fronteras: tables/independent_geometry.csv; pendientes netas y curvaturas en independent_local_differences.csv. Máximos de grilla no certifican óptimos globales.')
    add(local[local.scenario==primary].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    add('P4. Saturación previa en mejor nodo, Fresh y t20:')
    add(premean[(premean.scenario.isin(ids))&(premean.method=='Fresh')&(premean.point=='grid_best')&(premean.iteration==20)].to_string(index=False))
    add('Green en el diagnóstico previo usa un banco, no su historia futura. Clipping alto no descarta automáticamente una instancia; L026 conserva el estrés. Si el ejemplo principal tiene saturación alta, se informa y no se declara resuelta esa condición.')
    add('P5. Geometrías: las superficies tienen sus propias escalas claramente indicadas; comparar Rsearch, ubicación, áreas y contraste bruto/costo, no sólo colores. No se forzó multimodalidad ni se modificó REVENUE_SCALE.')
    add('P6. Distancias entre decisiones/trayectorias y variación total están en trajectory_distances.csv y trajectory_stability.csv. Una trayectoria más estable no garantiza mejor objetivo.')
    for case in ids:
        q=distance_table[(distance_table.scenario==case)&(distance_table.method_a!='Fixed')&(distance_table.method_b!='Fixed')]
        add(f'{case}, sin Fixed: distancia media entre trayectorias={q.mean_distance.mean():.4f}, distancia final media={q.final_distance.mean():.4f}, máxima distancia final={q.final_distance.max():.4f}.')
    add('P7/P8. Contraste temporal promedio entre repeticiones:')
    add(cn.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    add('C_tail usa t20:100, 81 puntos en ejecución completa. C(t) del banco finito incluye selección max-min y ruido; no es evidencia de superioridad sistemática. Auditoría independiente de pares en repetición 1:')
    add(audit.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    add('Se reportan t20,50,80,100 y picos, sin escoger semillas favorables. Un par con diferencia no distinguible del ruido o signo contrario se conserva. Esto no convierte el promedio temporal de máximos en un estimador insesgado poblacional.')
    for case in ids:
        q=audit[(audit.scenario==case)&audit.iteration.isin([20,50,80,100])]
        signs=', '.join(f't{int(r.iteration)}: {r.independent_difference:.3f} ± {r.paired_se:.3f} SE' for r in q.itertuples())
        add(f'{case}, pares alto-bajo fijados por traza sin Fixed: {signs}. Un signo negativo significa inversión del orden observado en la traza.')
    add('P9. Softmax/PSoft frente a Fresh, contrastes emparejados sobre repeticiones; IC t95% EXPLORATORIOS, sin inferencia confirmatoria ni afirmación universal:')
    q=pairs[pairs.method_a.isin(['Softmax010','Softmax050','PSoft300'])&(pairs.method_b=='Fresh')]
    add(q[['scenario','method_a','mean_difference','ci95_low','ci95_high','positive_fraction']].to_string(index=False))
    if primary=='I0252':
        q=cs[(cs.scenario==primary)&(~cs.with_fixed)]
        add('En I0252, C_tail sin Fixed por repetición: '+', '.join(f'{r.C_tail:.3f}' for r in q.itertuples())+'. La heterogeneidad entre semillas es relevante; un promedio grande no representa a todas las repeticiones.')
    add('P10. GreenSoftmax−Green:')
    add(pairs[(pairs.method_a=='GreenSoftmax050')&(pairs.method_b=='Green')][['scenario','mean_difference','ci95_low','ci95_high','positive_fraction']].to_string(index=False))
    add('P11. Evidencia por método: Fixed puede ofrecer trayectorias estables por banco único y también errores LR persistentes; Fresh explora con ruido nuevo; Green reutiliza datos y suele estabilizar decisiones, pero puede quedar en región desfavorable; Softmax suaviza pesos sin garantizar mejores valores; PSoft aproxima probabilidades de selección con bootstrap y puede acercarse a Fresh. Las tablas de variación, clipping, gaps y todas las repeticiones muestran qué rasgos aparecen realmente aquí; no son garantías generales.')
    add(stability.groupby(['scenario','method'])[['total_variation','tail_step','clipped_components_fraction','projection_fraction']].mean().to_string())
    if primary=='I0252':
        q=traces[(traces.scenario==primary)&(traces.rep==1)&traces.iteration.isin([0,20,50,80,100])]
        add('Ejemplo predefinido I0252, repetición 1, valores del banco de traza:')
        add(q.pivot(index='method',columns='iteration',values='objective').to_string(float_format=lambda x:f'{x:.3f}'))
        add('Aquí Green tiene un transitorio prolongado, mientras Green+Softmax y varios métodos Fresh alcanzan antes la región competitiva. La repetición 2 puede retrasar a Green+Softmax y las repeticiones 4–5 tienen menor contraste: no es una ventaja universal del suavizado. Consultar auditoría independiente para separar niveles de bancos distintos y confirmar cada diferencia temporal.')

    for case in ids:
        q=stability[(stability.scenario==case)&(stability.method!='Fixed')].groupby('method').total_variation.mean()
        f=summary[(summary.scenario==case)&(summary.method!='Fixed')]
        add(f'{case}, menor variación total media observada: {q.idxmin()} ({q.min():.3f}); mayor: {q.idxmax()} ({q.max():.3f}). Rango de medias finales entre métodos sin Fixed: {f.mean_objective.max()-f.mean_objective.min():.3f}. Estos resúmenes no establecen una relación causal entre estabilidad y calidad.')
    add(f'P12. Ejemplo principal fijado antes de comparar: {primary}. Parámetros completos arriba y en configs/models.toml. La selección fue geométrica/operacional, sin premiar al algoritmo ganador. Si la auditoría no distingue un máximo interior localizado o persiste saturación, es el mejor compromiso disponible, no una afirmación de éxito forzado.')
    mainmeta=tomllib.loads((out/'configs/main_instance.toml').read_text())
    boundarypath=out/'tables/independent_best_boundaries.csv'
    if boundarypath.exists():
        boundary=pd.read_csv(boundarypath);add('Validación independiente de los mejores nodos seleccionados en cada frontera:')
        add(boundary[boundary.scenario==primary][['edge','z3','z4','center_minus_edge','paired_se','descriptive_lower95']].to_string(index=False))
        add('Los intervalos son descriptivos por par, no simultáneos y no certifican el máximo sobre fronteras continuas.')
    actual=stability[(stability.scenario==primary)&(stability.method=='Fresh')].mean(numeric_only=True)
    pm=high[high.scenario==primary].iloc[0];reference=high[high.scenario=='L025'].iloc[0]
    add(f'Comprobación de compromiso principal: clipping Fresh observado en {actual.clipped_components_fraction:.1%} de componentes, {actual.clipping_iterations_fraction:.1%} de pasos, proyección {actual.projection_fraction:.1%}. Área epsilon=.5: {pm.epsilon05_fraction:.1%}, frente a {reference.epsilon05_fraction:.1%} de L025. Rsearch usa [0,.85]² en ambos; el 010 previo usaba [0,.75]².')
    add(f'Con tolerancia MC, las áreas competitivas son {pm.epsilon_mc_fraction:.1%} y {reference.epsilon_mc_fraction:.1%}, respectivamente; no se afirma una reducción de área robusta a cualquier tolerancia. Las curvaturas diagonales no certifican definitud del Hessiano ni diferenciabilidad.')
    fixedactual=stability[(stability.scenario==primary)&(stability.method=='Fixed')].mean(numeric_only=True)
    add(f'Limitación del ejemplo: Fixed mantiene clipping en {fixedactual.clipping_iterations_fraction:.1%} de pasos y proyección en {fixedactual.projection_fraction:.1%}. No se ha eliminado la saturación para todos los algoritmos. El criterio previo de usabilidad y la afirmación anterior se refieren a Fresh; las mediciones de los siete métodos se conservan en tablas y gráficos.')
    balanced=pm.boundary_distance>=.1-1e-9 and actual.clipping_iterations_fraction<.5 and actual.projection_fraction<.2 and mainmeta.get('beats_tested_best_edge_nodes_at_descriptive95',False)
    add('Se encontró un compromiso interior y una dinámica Fresh sin saturación dominante según los criterios exploratorios; las diferencias entre valores finales se juzgan por separado.' if balanced else 'No se han verificado conjuntamente todas las propiedades deseadas: presentar el ejemplo principal como mejor compromiso disponible, indicando qué condición permanece incierta o falla.')

    add()
    add('LIMITACIONES: cinco repeticiones, selección de instancias exploratoria, máximos de grilla, SE condicionales de hard-max y diferencias finitas. No se modificaron los métodos para evitar clipping. Algunas diferencias de decisión pueden producir valores prácticamente equivalentes. Se conservan resultados negativos y controles de frontera.')
    add('ENTORNO: Julia 1.11.7 y Project/Manifest conservados; versiones Python en COMPLETE.json y reproduction/python_requirements.txt. El entorno avisa sobre aceleradores opcionales numexpr/bottleneck antiguos y Axes3D no disponible; el cálculo usa operaciones compatibles y todos los gráficos son 2D. Estas advertencias no interrumpen simulación ni generación de archivos.')
    add('Reproducir/reanudar protocolo completo, incluida ampliación adaptativa documentada: python3 run_interior_011.py. Regenerar sólo figuras e informe: python3 analyze_interior_011.py resultados/011probing_high_contrast_interior. Comandos de fases en LEEME_ejecucion.txt.')
    (out/'informe_final.txt').write_text('\n'.join(report)+'\n')
    index=['<!doctype html><meta charset="utf-8"><title>011: alto contraste interior</title><style>body{font-family:system-ui;margin:2em;max-width:1500px}img{max-width:100%}a{margin-right:1em}</style>',
           '<h1>011: superficies interiores y comportamiento de algoritmos</h1>',f'<p>Ejemplo principal preseleccionado: <b>{primary}</b>. Todas las semillas se conservan; ilustración principal: repetición 1.</p>',
           '<p><a href="informe_final.txt">Informe P1–P12</a><a href="configs/models.toml">Parámetros completos</a><a href="configs/main_instance_parameters.toml">Parámetros del ejemplo principal</a><a href="tables/contrast_summary.csv">Contrastes temporales</a><a href="tables/final_summary.csv">Valores finales</a></p>',
           '<p>Trazas directas n=1000, superficies n=5000 y holdout n=50000 utilizan bancos diferentes. Bandas: SD entre repeticiones, no IC.</p><img src="figures/selected_surfaces_comparison.png">']
    for s in ids:
        index.append(f'<h2>{s}</h2>')
        for stem in [f'{s}_surface',f'{s}_gross_cost_net',f'{s}_convergence_rep01_full',f'{s}_convergence_rep01_zoom',f'{s}_contrast_rep01',f'{s}_z3_trajectory_rep01',f'{s}_z4_trajectory_rep01',f'{s}_oracle_trajectories_rep01',f'{s}_mean_convergence_full',f'{s}_mean_convergence_zoom',f'{s}_mean_contrast',f'{s}_final_objectives',f'{s}_actual_saturation',f'{s}_independent_temporal_contrast']:
            index.append(f'<p><a href="figures/{stem}.png">PNG</a><a href="figures/{stem}.pdf">PDF</a></p><img src="figures/{stem}.png">')
        for rep in sorted(traces.rep.unique()):
            index.append(f'<p>Rep. {rep}: <a href="figures/{s}_convergence_rep{rep:02d}_full.png">Completa</a><a href="figures/{s}_convergence_rep{rep:02d}_zoom.png">Zoom</a><a href="figures/{s}_contrast_rep{rep:02d}.png">C(t)</a></p>')
    index.append(f'<h2>{primary}: trayectorias separadas por método</h2><ul>')
    for method in METHODS:
        index.append(f'<li>{LABELS[method]}: <a href="figures/{primary}_{method}_oracle_individual.png">Trayectoria PNG</a><a href="figures/{primary}_{method}_oracle_individual.pdf">Trayectoria PDF</a><a href="figures/{primary}_{method}_convergence_individual.png">Convergencia PNG</a><a href="figures/{primary}_{method}_convergence_individual.pdf">Convergencia PDF</a></li>')
    index.append('</ul>')
    (out/'index.html').write_text('\n'.join(index))
    complete=dict(complete=True,quick=d['quick'],fingerprint=config['fingerprint'],operational_configurations=len(ops),net_configurations=len(candidates),
        precise_anchors=int(slopes.usable.sum()),refined=len(refined),selected=ids,main_instance=primary,reps=d['reps'],methods=METHODS,
        searches_total=len(ids)*d['reps']*7,searches_reused=nr*7,searches_new=(len(ids)*d['reps']-nr)*7,trace_rows=len(traces),png_files=len(list((out/'figures').glob('*.png'))),
        analysis_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('analyze_landscape_010.py')]},
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    complete['additional_sources_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__).with_name('refinar_compromisos_011.jl'),Path(__file__).with_name('validar_fronteras_011.jl'),Path(__file__).with_name('PROTOCOLO_011_extension.txt')] if f.exists()}
    (out/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(complete,indent=2))

if __name__=='__main__':main(sys.argv[1])
