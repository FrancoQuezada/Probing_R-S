"""Scientific tables, PNG/PDF figures and independent verification for 016."""
from pathlib import Path
import argparse, hashlib, json, html, warnings
warnings.filterwarnings('ignore', message='Pandas requires version')
import numpy as np
import pandas as pd
import tomli
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'resultados/016probing_green_bank_variants'
METHODS=['Green','Rolling5','Rolling10','Rolling20','Exp70','Exp90','Exp95','Filter1','Filter2','Filter5']
LABELS={'Green':'Green original',**{f'Rolling{w}':f'Green W={w}' for w in (5,10,20)},
        **{f'Exp{r}':f'Green ρ={r/100:g}' for r in (70,90,95)},
        **{f'Filter{k}':f'Green κ={k}' for k in (1,2,5)}}
FAMILIES={'rolling':['Green','Rolling5','Rolling10','Rolling20'],
          'exponential':['Green','Exp70','Exp90','Exp95'],
          'filtered':['Green','Filter1','Filter2','Filter5']}
COLORS=dict(zip(METHODS,['#222222','#0072B2','#56B4E9','#009E73','#D55E00','#E69F00','#CC79A7','#7B3294','#008837','#C51B7D']))
plt.rcParams.update({'font.size':10,'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':180,'pdf.fonttype':42})

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(out,name): return pd.read_csv(out/'tables'/f'{name}.csv')
def save(fig,out,name,files):
    fig.tight_layout()
    for ext in ('png','pdf'):
        path=out/'figures'/f'{name}.{ext}';fig.savefig(path,bbox_inches='tight');files.append(path)
    plt.close(fig)

def summaries(out,it,fin,tr):
    records=[]
    for method in METHODS:
        f=fin[fin.method==method];d=it[it.method==method]
        a=[g.objective.mean() for _,g in tr[tr.method==method].groupby('rep')]
        records.append(dict(method=method,replications=len(f),final_objective_mean=f.objective.mean(),
            final_objective_sd=f.objective.std(ddof=1),mean_conditional_mc_se=f.evaluation_se.mean(),
            final_z3_mean=f.z3.mean(),final_z4_mean=f.z4.mean(),
            optimization_seconds_mean=f.optimization_seconds.mean(),optimization_seconds_sd=f.optimization_seconds.std(ddof=1),
            diagnostic_seconds_mean=f.diagnostic_seconds.mean(),mean_trace_objective=np.mean(a),
            mean_ess_fraction=d.ess_fraction.mean(),minimum_ess=d.ess.min(),
            mean_retained_banks=d.retained_banks.mean(),final_retained_banks=d[d.iteration==d.iteration.max()].retained_banks.mean(),
            mean_retained_fraction=(d.retained_banks/d.generated_banks).mean(),
            mean_objective_contribution_variance=d.objective_contribution_variance.mean(),
            mean_gradient3_contribution_variance=d.gradient3_contribution_variance.mean(),
            mean_gradient4_contribution_variance=d.gradient4_contribution_variance.mean(),
            mean_objective_mean_variance_proxy=d.objective_mean_variance_proxy.mean(),
            mean_gradient3_mean_variance_proxy=d.gradient3_mean_variance_proxy.mean(),
            mean_gradient4_mean_variance_proxy=d.gradient4_mean_variance_proxy.mean(),
            gradient_clip_fraction=(d.clipping>0).mean(),projection_fraction=(d.projection>1e-12).mean(),
            lr_truncations=int(d.lr_clips.sum()),new_search_paths_per_run=int(f.new_search_paths.iloc[0])))
    s=pd.DataFrame(records);s.to_csv(out/'tables/method_summary.csv',index=False)
    pairs=read(out,'paired_final_vs_green');r=[]
    for method,g in pairs.groupby('method',sort=False):
        r.append(dict(method=method,difference_mean=g.difference.mean(),difference_sd=g.difference.std(ddof=1),
                      paired_mc_se_mean=g.paired_mc_se.mean(),positive_replications=int((g.difference>0).sum()),replications=len(g)))
    pd.DataFrame(r).to_csv(out/'tables/paired_summary.csv',index=False)
    return s

def figures(out,it,fin,coords,tr,banks):
    files=[];reps=sorted(fin.rep.unique());surface=pd.read_csv(out/'configs/I0252_surface.csv')
    grid=surface.pivot(index='z4',columns='z3',values='objective')
    # Each family has only four curves. Separate replication plots and a panel
    # view avoid the unreadable all-method overlay from earlier experiments.
    for family,methods in FAMILIES.items():
        for rep in reps:
            fig,ax=plt.subplots(figsize=(7.4,6))
            mesh=ax.pcolormesh(grid.columns,grid.index,grid.values,shading='auto',cmap='viridis',
                               vmin=surface.objective.quantile(.1),vmax=surface.objective.max())
            fig.colorbar(mesh,ax=ax,label='Superficie directa archivada (N=5000)')
            for method in methods:
                d=coords[(coords.rep==rep)&(coords.method==method)].sort_values('iteration')
                ax.plot(d.z3,d.z4,color=COLORS[method],lw=1.8,label=LABELS[method],alpha=.95)
                ax.scatter(d.z3.iloc[-1],d.z4.iloc[-1],color=COLORS[method],s=40,edgecolor='white',zorder=5)
            ax.scatter([.12],[.1],marker='*',s=160,c='gold',edgecolor='black',zorder=6,label='Inicio común')
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Probing clase 3, z₃',ylabel='Probing clase 4, z₄',
                   title=f'I0252 · {family} · rep{rep:02d}')
            ax.legend(loc='best',fontsize=8,framealpha=.9)
            save(fig,out,f'I0252_{family}_oracle_trajectories_rep{rep:02d}',files)
            for col,label in [('z3','Class-3 probing trajectory'),('z4','Class-4 probing trajectory')]:
                fig,ax=plt.subplots(figsize=(8,4.5))
                for method in methods:
                    d=coords[(coords.rep==rep)&(coords.method==method)].sort_values('iteration')
                    ax.plot(d.iteration,d[col],color=COLORS[method],label=LABELS[method])
                ax.set(xlabel='Iteración',ylabel=col,ylim=(-.02,1.02),title=f'{label} · {family} · rep{rep:02d}')
                ax.legend(fontsize=9);ax.grid(alpha=.2)
                save(fig,out,f'I0252_{family}_{col}_trajectory_rep{rep:02d}',files)
            for name,column,ylabel,log in [('convergence','objective','Objetivo directo (CRN independiente N='+str(int(tr.paths.iloc[0]))+')',False),
                  ('internal_objective','objective_internal','Objetivo interno LR (adaptativo)',False),
                  ('ess','ess','ESS de aₕ·LR/nₕ',True),('ess_fraction','ess_fraction','ESS / observaciones retenidas',False),
                  ('retained_banks','retained_banks','Bancos retenidos',False),
                  ('cumulative_time','cumulative_search_seconds','Tiempo acumulado de optimización (s)',False),
                  ('objective_variance','objective_contribution_variance','Media ponderada de varianzas de Y·LR',True),
                  ('gradient3_variance','gradient3_contribution_variance','Varianza de contribuciones Y·LR·score₃',True),
                  ('gradient4_variance','gradient4_contribution_variance','Varianza de contribuciones Y·LR·score₄',True)]:
                fig,ax=plt.subplots(figsize=(8,4.6));df=tr if name=='convergence' else it
                for method in methods:
                    d=df[(df.rep==rep)&(df.method==method)].sort_values('iteration')
                    ax.plot(d.iteration,d[column],color=COLORS[method],label=LABELS[method])
                ax.set(xlabel='Iteración',ylabel=ylabel,title=f'I0252 · {family} · rep{rep:02d}')
                if log: ax.set_yscale('log')
                ax.legend(fontsize=8);ax.grid(alpha=.2)
                save(fig,out,f'I0252_{family}_{name}_rep{rep:02d}',files)
        for column,label in [('objective','Objetivo directo independiente'),('ess_fraction','ESS / observaciones retenidas'),
                             ('retained_banks','Bancos retenidos')]:
            df=tr if column=='objective' else it
            fig,axes=plt.subplots(1,len(reps),figsize=(5*len(reps),4.5),squeeze=False)
            for ax,rep in zip(axes.ravel(),reps):
                for method in methods:
                    d=df[(df.rep==rep)&(df.method==method)].sort_values('iteration')
                    ax.plot(d.iteration,d[column],label=LABELS[method],color=COLORS[method])
                ax.set(title=f'rep{rep:02d}',xlabel='Iteración',ylabel=label);ax.grid(alpha=.2)
            axes.ravel()[-1].legend(fontsize=8)
            fig.suptitle(f'I0252 · {family}')
            save(fig,out,f'I0252_{family}_{column}_all_reps',files)
    # Final uncertainty bars are WITHIN-holdout MC intervals, not three-seed SD.
    fig,ax=plt.subplots(figsize=(12,5.3));x=np.arange(len(METHODS))
    for j,rep in enumerate(reps):
        f=fin[fin.rep==rep].set_index('method').loc[METHODS];offset=(j-(len(reps)-1)/2)*.20
        ax.errorbar(x+offset,f.objective,yerr=1.959963984540054*f.evaluation_se,fmt='o',ms=4,capsize=2,label=f'rep{rep:02d}')
    ax.set_xticks(x,[LABELS[m] for m in METHODS],rotation=35,ha='right')
    ax.set(ylabel='Objetivo holdout directo',title='I0252 · evaluación final independiente · barras IC 95% MC condicional')
    ax.legend();ax.grid(axis='y',alpha=.2)
    save(fig,out,'I0252_final_independent_objectives',files)
    pairs=read(out,'paired_final_vs_green');others=METHODS[1:];x=np.arange(len(others))
    fig,ax=plt.subplots(figsize=(12,5.2))
    for j,rep in enumerate(reps):
        f=pairs[pairs.rep==rep].set_index('method').loc[others];offset=(j-(len(reps)-1)/2)*.20
        ax.errorbar(x+offset,f.difference,yerr=1.959963984540054*f.paired_mc_se,fmt='o',capsize=2,ms=4,label=f'rep{rep:02d}')
    ax.axhline(0,color='black',lw=1);ax.set_xticks(x,[LABELS[m] for m in others],rotation=35,ha='right')
    ax.set(ylabel='Diferencia respecto a Green original',title='Diferencias finales pareadas · IC 95% MC condicional (CRN)')
    ax.legend();ax.grid(axis='y',alpha=.2);save(fig,out,'I0252_final_paired_differences',files)
    fig,ax=plt.subplots(figsize=(12,5.2));x=np.arange(len(METHODS));width=.8/len(reps)
    for j,rep in enumerate(reps):
        f=fin[fin.rep==rep].set_index('method').loc[METHODS]
        ax.bar(x+(j-(len(reps)-1)/2)*width,f.optimization_seconds,width=width,label=f'rep{rep:02d}')
    ax.set_xticks(x,[LABELS[m] for m in METHODS],rotation=35,ha='right')
    ax.set(ylabel='Tiempo de optimización (s)',title='I0252 · búsquedas secuenciales · BLAS=1 · diagnóstico adicional excluido')
    ax.legend();save(fig,out,'I0252_computational_time',files)
    fig,ax=plt.subplots(figsize=(12,5.2))
    f=fin.groupby('method').mean(numeric_only=True).loc[METHODS];x=np.arange(len(METHODS))
    ax.bar(x,f.optimization_seconds,label='Optimización')
    ax.bar(x,f.diagnostic_seconds,bottom=f.optimization_seconds,label='Diagnóstico adicional')
    ax.set_xticks(x,[LABELS[m] for m in METHODS],rotation=35,ha='right')
    ax.set(ylabel='Tiempo medio (s)',title='Tiempo instrumentado: optimización + diagnóstico separado');ax.legend()
    save(fig,out,'I0252_instrumentation_time',files)
    return files

def verify(out,it,fin,coords,tr,banks,files,science):
    K=int(science['max_iters']);reps=science['rep_ids'];nm=len(METHODS)
    assert len(it)==nm*len(reps)*K and len(coords)==nm*len(reps)*(K+1)==len(tr)
    assert len(fin)==nm*len(reps) and len(banks)==nm*len(reps)*K*(K+1)//2
    assert not it.duplicated(['rep','method','iteration']).any()
    assert not banks.duplicated(['rep','method','iteration','bank_iteration']).any()
    assert it.new_paths.eq(500).all() and fin.new_search_paths.eq(500*K).all()
    assert it.objective_policy.eq(it.gradient_policy).all() and it.objective_policy.eq(it.hard_policy).all()
    assert np.isfinite(it.select_dtypes('number').drop(columns=['filter_threshold','filter_current_variance'])).all().all()
    assert (it.ess>0).all() and (it.ess<=it.retained_observations*(1+1e-12)).all()
    assert it.reused_observations.eq(it.retained_observations-500).all()
    keys=['rep','method','iteration'];g=banks.groupby(keys,sort=False)
    assert np.allclose(g.bank_weight.sum(),1,atol=2e-14)
    assert banks.bank_weight.ge(0).all()
    assert np.allclose(g.retained.sum().sort_index(),it.set_index(keys).retained_banks.sort_index())
    current=banks[banks.is_current];assert current.retained.all()
    assert current.mean_lr.eq(1).all() and current.bank_lr_ess.eq(500).all()
    for col in ['objective_contribution_sample_variance','gradient3_contribution_sample_variance','gradient4_contribution_sample_variance']:
        q=banks[banks.retained][col];assert np.isfinite(q).all() and q.ge(0).all()
    filtered=banks[banks.method.str.startswith('Filter')]
    assert ((filtered.retained==(filtered.filter_sample_variance<=filtered.filter_threshold))|filtered.is_current).all()
    assert np.isfinite(filtered.filter_sample_variance).all()
    selected=banks[banks.retained].copy()
    for col,target,alpha in [('objective_contribution_mean','objective_internal',None),
                             ('gradient3_contribution_mean','g3',8.529251033547421),
                             ('gradient4_contribution_mean','g4',10.083855971068047)]:
        selected['_weighted']=selected.bank_weight*selected[col]
        calc=selected.groupby(keys)._weighted.sum().sort_index();ref=it.set_index(keys).sort_index()
        calc-=ref.probing_cost if alpha is None else alpha
        assert np.allclose(calc,ref[target],rtol=1e-10,atol=1e-8),target
    for method in METHODS:
        d=it[it.method==method]
        if method.startswith('Rolling'): assert np.array_equal(d.retained_banks,np.minimum(d.iteration,int(method[7:])))
        if method.startswith('Exp') or method=='Green': assert d.retained_banks.eq(d.iteration).all()
    exp=banks[banks.method.str.startswith('Exp')].copy()
    exp['_raw']=(exp.method.str[3:].astype(int)/100)**exp.bank_age
    norm=exp.groupby(keys)._raw.transform('sum')
    assert np.allclose(exp.bank_weight,exp._raw/norm,rtol=1e-12,atol=1e-14)
    seeds=pd.read_csv(out/'configs/search_seeds.csv')
    assert seeds.groupby(['rep','iteration']).seed.nunique().eq(1).all()
    training=set(seeds.seed);external=set(fin.selection_seed)|set(fin.holdout_seed)|set(tr.seed)
    assert not training.intersection(external)
    assert fin.holdout_paths.nunique()==fin.selection_paths.nunique()==1
    assert np.allclose(fin.ci95_high-fin.objective,1.959963984540054*fin.evaluation_se)
    if not science['quick']:
        src=ROOT/'resultados/011probing_high_contrast_interior/comparisons/I0252'
        for rep in reps:
            old=pd.read_csv(src/f'rep_{rep:02d}_coordinates.csv');old=old[old.method=='Green'].sort_values('iteration')
            new=coords[(coords.rep==rep)&(coords.method=='Green')].sort_values('iteration')
            assert np.allclose(old[['z3','z4']],new[['z3','z4']],atol=1e-14,rtol=0)
            oldf=pd.read_csv(src/f'rep_{rep:02d}_finals.csv');oldf=oldf[oldf.method=='Green'].iloc[0]
            nf=fin[(fin.rep==rep)&(fin.method=='Green')].iloc[0]
            assert abs(nf.objective-oldf.objective_holdout)<1e-10
    manifest=BASE/'validation/preserved_before.json';protected=json.loads(manifest.read_text())
    changed=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not changed,changed[:5]
    png=[p for p in files if p.suffix=='.png'];pdf=[p for p in files if p.suffix=='.pdf']
    for p in png:
        a=np.asarray(Image.open(p).convert('RGB'));assert a.shape[0]>300 and a.shape[1]>500 and a.std()>10,p
    for p in pdf: assert p.read_bytes().startswith(b'%PDF-'),p
    assert len(png)==len(pdf)>0
    assert sha(out/'configs/I0252_surface.csv')==sha(ROOT/'resultados/011probing_high_contrast_interior/high/I0252_surface.csv')
    tests=tomli.loads((out/'validation/deterministic_tests.toml').read_text());assert tests['passed']
    done=tomli.loads((out/'validation/science_complete.toml').read_text());assert done['complete']
    extra_path=out/'validation/additional_audit.toml';extra=0
    if extra_path.exists():
        audit=tomli.loads(extra_path.read_text());assert audit['passed'];extra=int(audit['assertions'])
    result=dict(passed=True,completed_searches=len(fin),independent_replications=reps,iterations_per_run=K,
        new_paths_per_iteration=500,total_new_search_paths=int(fin.new_search_paths.sum()),
        iteration_rows=len(it),bank_rows=len(banks),direct_trace_points=len(tr),
        deterministic_tests=tests['tests_passed'],additional_audit_assertions=extra,
        total_assertions=tests['tests_passed']+extra,original_controls_exact=True,
        surface_recomputed=False,png=len(png),pdf=len(pdf),previous_files_unchanged=len(protected))
    (out/'validation/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def report(out,summary,fin,it,science,verified):
    pairs=pd.read_csv(out/'tables/paired_summary.csv');base=summary[summary.method=='Green'].iloc[0]
    lines=['I0252 — variantes de selección/ponderación de bancos Green',
           f"Ejecución {'DE PRUEBA' if science['quick'] else 'COMPLETA'}: {verified['completed_searches']} búsquedas, "
           f"reps {science['rep_ids']}, {science['max_iters']} iteraciones, 500 trayectorias nuevas por actualización.",
           'No se modificaron fuentes previas ni parámetros originales. La superficie archivada fue reutilizada.',
           '', 'Estimador y ejecución:',
           "J(z)=max_i E_z[Y_i]-alpha'z. Se elige una política común y se usa en valor y gradiente.",
           'Modelo I0252: dos servidores y colas propias por política; z3,z4 modula las probabilidades de señal.',
           'Arribos, clases, complejidad, valores y servicios son primitivas exógenas compartidas mediante CRN.',
           'Recompensa: ingreso - coste de ruta rápida - penalización por tiempo total en el sistema.',
           'No se usa aquí el modelo antiguo de disrupciones de SimOpt009; la familia contiene 144 reglas umbral.',
           'L_hl(z)=exp(clamp(log(f_z(x_hl)/f_z_h(x_hl)),-35,35)); S_hl(z)=grad log f_z(x_hl).',
           'v_i=sum_h a_h mean_l(Y_ihl*L_hl); i*=argmax_i v_i; Jhat=v_i*-alpha\'z.',
           'ghat=sum_h a_h mean_l(Y_i*hl*L_hl*S_hl)-alpha, con el MISMO i* y a_h.',
           'z^k=proy_[0,1]^2(z^(k-1)+(.035/sqrt(k))*clip_componente(ghat,-15,15)).',
           'Las únicas modificaciones son ventanas W=5/10/20, pesos rho=.7/.9/.95 o filtro kappa=1/2/5.',
           'Los casos límite delegan al Green original y reproducen exactamente sus reducciones.',
           'Filtro: política provisional del Green completo, varianza muestral de Y*LR por banco,',
           'equivalente a la varianza de Y*LR-c(z), porque c(z) es determinista y se sustrae una sola vez.',
           'retener V_h <= kappa*V_actual y siempre el banco actual. Después se vuelve a seleccionar la política.',
           'La desigualdad no estricta conserva empates de varianza cero; datos no finitos causan error explícito.',
           'Semillas originales por rep/iteración comunes entre métodos. No se simula al reutilizar bancos.',
           'Convención de índices: iterations.csv fila k evalúa target_z=z^(k-1) y produce next_z=z^k;',
           'coordinates.csv y objective_traces.csv fila t representan z^t tras t actualizaciones (incluyen t=0).',
           '', 'Evaluación final:',
           f"Selector directo independiente N={science['selection_paths']}; holdout directo independiente N={science['holdout_paths']}.",
           'La política se fija antes del holdout. Se usa CRN entre endpoints de cada rep.',
           'SE e IC95% son MC condicionales a z y a la política seleccionada. SD de la siguiente tabla',
           'describe variación entre repeticiones de optimización/evaluación; no es el SE de holdout.',
           '', 'Método | objetivo final media ± SD entre reps | SE MC medio | A(media traza) | tiempo optimización medio (s) | ESS/retención medio']
    for _,r in summary.iterrows():
        sd='NA' if pd.isna(r.final_objective_sd) else f'{r.final_objective_sd:.6f}'
        lines.append(f'{LABELS[r.method]} | {r.final_objective_mean:.6f} ± {sd} | {r.mean_conditional_mc_se:.6f} | '
                     f'{r.mean_trace_objective:.6f} | {r.optimization_seconds_mean:.4f} | {r.mean_ess_fraction:.4f}')
    lines+=['','Comparación pareada con Green original (misma rep, mismo holdout):',
            'Método | diferencia media | SD entre reps | SE MC pareado medio | reps con diferencia positiva']
    for _,r in pairs.iterrows():
        lines.append(f'{LABELS[r.method]} | {r.difference_mean:+.6f} | {r.difference_sd:.6f} | '
                     f'{r.paired_mc_se_mean:.6f} | {int(r.positive_replications)}/{int(r.replications)}')
    lines+=['','Diagnósticos de varianza (promedios de iteraciones/reps):',
            'Método | media var(Y*LR) | media var(Y*LR*score3) | media var(Y*LR*score4) | fracción media de bancos retenidos']
    for _,r in summary.iterrows():
        lines.append(f'{LABELS[r.method]} | {r.mean_objective_contribution_variance:.6g} | '
                     f'{r.mean_gradient3_contribution_variance:.6g} | {r.mean_gradient4_contribution_variance:.6g} | '
                     f'{r.mean_retained_fraction:.4f}')
    lines+=['Estos diagnósticos recorren estados z y políticas diferentes según el método; no son una',
            'comparación de varianzas a z fijo ni una prueba de dominancia estadística del estimador.']
    clipped=it[it.lr_clips>0]
    if len(clipped):
        lines+=['','Activaciones del recorte de log-LR existente:']
        for _,r in clipped.iterrows():
            lines.append(f'{r.method}, rep{int(r.rep):02d}, actualización {int(r.iteration)}: {int(r.lr_clips)} contribuciones recortadas.')
    best=summary.loc[summary.final_objective_mean.idxmax()]
    fastest=summary.loc[summary.optimization_seconds_mean.idxmin()]
    best_trace=summary.loc[summary.mean_trace_objective.idxmax()]
    chosen=sorted(fin.selected_policy.unique())
    lines+=['','Políticas seleccionadas en la evaluación final: '+', '.join(str(int(v)) for v in chosen)+'.']
    policy_table=pd.read_csv(out/'configs/policies.csv')
    for policy in chosen:
        p=policy_table[policy_table.policy==policy].iloc[0]
        lines.append(f'Política {int(policy)}: av={p.av:g}, aq={p.aq:g}, as={p["as"]:g}, tau={p.tau:g}.')
    lines+=['','Lectura experimental:',f'Mayor media final observada: {LABELS[best.method]} ({best.final_objective_mean:.6f}).',
            f'Menor tiempo medio observado: {LABELS[fastest.method]} ({fastest.optimization_seconds_mean:.4f} s).',
            f'Green original: {base.final_objective_mean:.6f}, tiempo {base.optimization_seconds_mean:.4f} s.',
            f'Mayor media de las evaluaciones directas por iterado: {LABELS[best_trace.method]} '
            f'({best_trace.mean_trace_objective:.6f}, frente a {base.mean_trace_objective:.6f} de Green).',
            'Ese indicador promedia K+1 iterados, incluye el inicio y describe el comportamiento transitorio;',
            'usa las trazas CRN y no reemplaza el holdout final. En rep1 el Green original oscila lejos de la',
            'zona de valores altos durante buena parte de la búsqueda; los filtros evitan ese transitorio.',
            'Las diferencias pareadas e intervalos por rep están en paired_final_vs_green.csv; no se concluye',
            'superioridad poblacional con tres semillas. Una mejora de ESS puede acompañar un gradiente menos',
            'variable, pero ESS no mide directamente la precisión del gradiente ni corrige la selección adaptativa.',
            'Ventanas usan menos historia; exponenciales reducen el peso de bancos viejos; filtros pagan el coste',
            'del Green completo provisional más las varianzas y la estimación retenida. No se igualan esos costes.',
            '', 'Limitaciones e inconsistencias:',
            'Se conserva el recorte de log-LR a [-35,35], que puede introducir sesgo; no se usa SNIS.',
            f"Se observaron {int(it.lr_clips.sum())} activaciones de recorte LR, contadas por banco/objetivo temporal (pueden reutilizar paths).",
            'El filtro usa los mismos datos para seleccionar bancos y estimar; puede inducir sesgo de selección.',
            'Las varianzas por banco se evalúan en z actual, con la política final común. El filtro registra',
            'por separado las varianzas de su política provisional. Las medias ponderadas de varianzas y',
            'sum(a_h²*s_h²/n_h) son diagnósticos descriptivos; no son errores estándar válidos del proceso adaptativo.',
            'Las trazas directas N=1000 usan max de medias en el mismo banco, con sesgo de selección posible;',
            'no son el estimador final de holdout. La superficie tampoco certifica un óptimo global.',
            'La referencia antigua 2serversModel tiene N=5000, 45 iteraciones y otro modelo; no es I0252.',
            'No se detectaron errores numéricos en las búsquedas completadas. Se corrigieron errores de sintaxis',
            'en una aserción inicial y en el script auxiliar de auditoría antes de sus validaciones respectivas.',
            'Pandas emite avisos por aceleradores opcionales antiguos; Matplotlib advierte sobre Axes3D.',
            'Los gráficos usados son 2D y se verificaron como imágenes no vacías y PDFs válidos.',
            '', 'Tiempo:',
            'Medición secuencial, BLAS=1, calentamiento excluido, incluye selección de bancos y varianzas del filtro.',
            'Momentos adicionales de diagnóstico y evaluaciones externas excluidos, tiempos separados en CSV.',
            'SO no aislado, orden fijo por rep/familia; diagnóstico y GC pueden afectar mediciones posteriores.',
            'Son tiempos descriptivos, no un benchmark con orden aleatorizado y aislamiento de recursos.',
            '', 'Reproducir desde la raíz:',
            'python3 run_green_bank_variants_016.py --quick-test',
            'python3 run_green_bank_variants_016.py --resume',
            'python3 run_green_bank_variants_016.py --plots-only',
            'Entorno Julia existente: resultados/003probing_softmax_20261021/entorno (Julia 1.11.7).',
            'No fueron necesarias instalaciones adicionales. Checkpoints separados por método/rep y evaluación.',
            'Cada invocación vuelve a ejecutar pruebas/calentar kernels; --resume reutiliza las búsquedas y evaluaciones completas.',
            '', f"Verificación: {verified['total_assertions']} aserciones ({verified['deterministic_tests']} previas y {verified['additional_audit_assertions']} auxiliares); controles Green exactos; "
            f"{verified['previous_files_unchanged']} archivos previos sin cambios; {verified['png']} PNG y {verified['pdf']} PDF."]
    (out/'RESUMEN_EXPERIMENTAL.txt').write_text('\n'.join(lines)+'\n')

def gallery(out,files,summary,verified):
    links=['RESUMEN_EXPERIMENTAL.txt','tables/final_results.csv','tables/method_summary.csv',
           'tables/iterations.csv','tables/banks.csv','tables/coordinates.csv','tables/objective_traces.csv',
           'tables/paired_final_vs_green.csv','validation/verification.json','configs/experiment.toml']
    rows=''.join(f'<tr><td>{html.escape(LABELS[r.method])}</td><td>{r.final_objective_mean:.5f}</td>'
                 f'<td>{r.final_objective_sd:.5f}</td><td>{r.optimization_seconds_mean:.3f}</td></tr>' for _,r in summary.iterrows())
    intro='<h1>I0252 · variantes Green</h1><p>Original, ventanas, ponderación exponencial y filtro por varianza · '
    intro+=f"{verified['completed_searches']} búsquedas verificadas.</p>"
    content=intro+'<p>'+' · '.join(f'<a href="{p}">{html.escape(p)}</a>' for p in links)+'</p>'
    content+='<table><tr><th>Método</th><th>Objetivo final medio</th><th>SD entre reps</th><th>Tiempo medio (s)</th></tr>'+rows+'</table>'
    content+='<p>IC en gráficos finales: ruido MC condicional del holdout. SD: variación entre repeticiones. Tres semillas son exploratorias.</p>'
    for p in files:
        if p.suffix!='.png': continue
        rel=p.relative_to(out).as_posix();pdf=p.with_suffix('.pdf').relative_to(out).as_posix()
        content+=f'<article><h2>{html.escape(p.stem)}</h2><a href="{rel}">PNG</a> · <a href="{pdf}">PDF</a><br><a href="{rel}"><img loading="lazy" src="{rel}"></a></article>'
    page='<!doctype html><html lang="es"><meta charset="utf-8"><title>I0252 Green bank variants</title><style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}article{margin-top:45px}img{max-width:100%;height:auto}a{color:#075da8}</style>'+content+'</html>'
    (out/'index.html').write_text(page)
    assert all((out/p).is_file() for p in links)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=BASE)
    ap.add_argument('--reuse-figures',action='store_true',help='Verify existing figures and rebuild tables/report without replotting')
    args=ap.parse_args();out=args.output
    science=tomli.loads((out/'configs/experiment.toml').read_text())['science']
    it=read(out,'iterations');fin=read(out,'final_results');coords=read(out,'coordinates');tr=read(out,'objective_traces');banks=read(out,'banks')
    summary=summaries(out,it,fin,tr)
    files=sorted([*out.glob('figures/*.png'),*out.glob('figures/*.pdf')]) if args.reuse_figures else figures(out,it,fin,coords,tr,banks)
    verified=verify(out,it,fin,coords,tr,banks,files,science)
    report(out,summary,fin,it,science,verified);gallery(out,files,summary,verified)
    for name in ['016probing_green_bank_variants.jl','PROTOCOLO_green_bank_variants_016.txt','run_green_bank_variants_016.py','analyze_green_bank_variants_016.py','audit_green_bank_variants_016.jl']:
        import shutil
        shutil.copyfile(ROOT/name,out/'reproduction'/name)
    (out/'COMPLETE.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps(verified,indent=2))
if __name__=='__main__': main()
