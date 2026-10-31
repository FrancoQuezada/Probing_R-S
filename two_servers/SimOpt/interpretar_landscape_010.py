"""Scientific interpretation after the frozen simulation and automatic summaries."""
from pathlib import Path
import sys,json,hashlib
try:import tomllib
except ModuleNotFoundError:import tomli as tomllib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out):
    out=Path(out).resolve();high=pd.read_csv(out/'high/metrics.csv');summary=pd.read_csv(out/'tables/final_summary.csv')
    pairs=pd.read_csv(out/'tables/paired_exploratory.csv');stability=pd.read_csv(out/'tables/trajectory_stability.csv')
    gaps=pd.read_csv(out/'tables/grid_gaps_all.csv');base=high[high.scenario=='L000'].iloc[0]
    eligible=high[(high.boundary_distance>0)&(high.R_search>2*base.R_search)]
    if eligible.empty:eligible=high[high.scenario!='L000']
    def spread(s):
        q=summary[(summary.scenario==s)&(summary.method!='Fixed')]
        return q.mean_objective.max()-q.mean_objective.min()
    recommendation=max(eligible.scenario,key=spread)
    models=tomllib.loads((out/'configs/models.toml').read_text())
    rows=[]
    for (s,m),g in gaps.groupby(['scenario','method']):
        n=len(g);sd=g.grid_reference_gap.std(ddof=1);se=sd/np.sqrt(n);crit=stats.t.ppf(.975,n-1)
        rows.append(dict(scenario=s,method=m,mean_grid_reference_gap=g.grid_reference_gap.mean(),between_rep_se=se,
                         ci95_low=g.grid_reference_gap.mean()-crit*se,ci95_high=g.grid_reference_gap.mean()+crit*se,
                         mean_paired_mc_se=g.paired_se.mean()))
    regional=pd.DataFrame(rows);regional.to_csv(out/'tables/independent_region_audit.csv',index=False)
    text=[]
    def add(s=''):text.append(str(s))
    add('INTERPRETACIÓN CIENTÍFICA DE LAS INSTANCIAS 010')
    add(f'Ejemplo principal recomendado: {recommendation}. Se exige un máximo de grilla interior y contraste relevante mayor que el original, antes de valorar la separación entre métodos. No se elige por un ganador de Softmax/PSoft.')
    add(json.dumps(models[recommendation],ensure_ascii=False,sort_keys=True))
    add('El máximo de grilla interior observado no prueba un óptimo interior poblacional; la reevaluación independiente de vecinos cuantifica cuánto se distingue la región.')
    add('Los niveles absolutos de una traza de 1000 caminos pueden diferir de una superficie de 5000 o un holdout de 50000 porque usan bancos independientes. CRN hace comparables los métodos dentro de una repetición; no fuerza coincidencia numérica entre bancos. La SD de convergencia promedio contiene variación de las búsquedas y de sus bancos de evaluación; la estabilidad de decisiones se mide separadamente en z.')
    add('El valor poblacional de una política seleccionada es menor o igual que J(z). Su estimación Monte Carlo puede superar una estimación de grilla por azar; esa desigualdad no se exige entre medias de muestras independientes. Tampoco hay contradicción si un punto continuo supera el mejor nodo de una grilla finita.')
    add('L000 es control relativamente plano. L025 ofrece un compromiso interior y contraste intermedio. L026 es un caso de información valiosa y costos de tiempo altos, pero el mejor punto de la grilla fina es la esquina (1,1); se conserva como resultado de frontera, sin reemplazarlo después de ver los algoritmos. L030 combina contraste pronunciado y mejor región cercana a frontera.')
    add('No se encontró evidencia suficiente de múltiples regiones competitivas separadas entre las cuatro superficies finas: todas tienen una componente con el umbral predefinido. Los máximos locales adicionales de una grilla ruidosa no establecen multimodalidad poblacional.')
    add('No se presenta L026 como ejemplo principal de óptimo no trivial. Sus valores netos negativos, y los de L030, son consecuencias reales de cobrar tiempo en sistema; no se reescalaron los rewards para producirlas.')
    add('En metrics.csv, search_snr compara el mejor punto de toda la grilla con el peor punto de la región de búsqueda; R_search compara máximo y mínimo dentro de esa región. Ambos son indicadores distintos. initial_snr usa el par mejor punto de grilla/Z0.')
    add()
    ab=out/'tables/economic_attribution_gains.csv'
    if ab.exists():
        a=pd.read_csv(ab)
        add('ATRIBUCIÓN ECONÓMICA LOCAL, INDEPENDIENTE DE LAS BÚSQUEDAS')
        add('Se compara z0=(.12,.10) con (.90,1.00), fijado por la grilla refinada antes de evaluar algoritmos. En cada variante se selecciona política con 10000 caminos y evalúa con 50000 independientes; CRN dentro de cada comparación. No es una nueva búsqueda de óptimos ni una descomposición aditiva de efectos con interacciones.')
        add(a.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
        add('Restaurar ALPHA cambia únicamente el costo, con beneficio bruto exactamente igual. Restaurar penalización o señales altera mucho el beneficio bruto del probing. Esto confirma que el contraste grande de L026 no procede sólo de restar costos de probing ni de multiplicar ingresos comunes.')
        fig,ax=plt.subplots(figsize=(11,5.5))
        names=['L026','Penalización .42','Servicio 8/2.4','Señales originales','beta_signal 2.2','ALPHA original']
        x=np.arange(len(a));ax.bar(x-.18,a.gross_gain,width=.36,label='Mejora de beneficio bruto',color='steelblue')
        ax.bar(x+.18,a.net_gain,width=.36,label='Mejora neta',color='darkorange',yerr=a.paired_se,capsize=3)
        ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylabel('Mejora entre dos puntos fijos');ax.set_title('Atribución local: restaurar un factor original cada vez; barras de error = SE MC')
        ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/'figures'/f'economic_attribution.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    add()
    add('CONTRASTE EN REGIONES RELEVANTES, CON EVALUACIÓN INDEPENDIENTE')
    add(regional[regional.method.isin(['Initial','GridNeighbor1','GridNeighbor2','GridNeighbor3','GridNeighbor4'])].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    add('Gap positivo indica que la política seleccionada para el mejor punto de grilla supera la elegida en el vecino/inicio en este holdout. No es un certificado global; la selección de la política también puede ser imperfecta.')
    add()
    audit_path=out/'tables/trajectory_contrast_validation.csv'
    if audit_path.exists():
        audit=pd.read_csv(audit_path)
        add('P5/P6: DIFERENCIAS DURANTE EL RECORRIDO, VERIFICADAS CON NUEVO BANCO')
        add('En la repetición ilustrativa predefinida 1 se eligió la iteración con mayor rango entre métodos distintos de Fixed según las trazas de 1000 caminos. Las decisiones y el par máximo-mínimo quedaron fijados antes de un nuevo selector de 10000 y holdout de 50000. Este diagnóstico es exploratorio y evita validar una diferencia con el mismo banco que la identificó.')
        add(audit.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
        add('El rango de las trazas puede disminuir o cambiar de signo al evaluarlo de forma independiente; se conserva también ese resultado. El SNR emparejado se refiere al par fijado en la traza y al valor de sus políticas seleccionadas independientemente.')
        fig,ax=plt.subplots(figsize=(9,5))
        x=np.arange(len(audit))
        ax.bar(x-.18,audit.exploratory_trace_spread,width=.36,label='Rango identificado en la traza',color='silver')
        ax.bar(x+.18,audit.independent_difference,width=.36,yerr=audit.paired_se,capsize=4,label='Diferencia del par en nuevo holdout ± SE',color='steelblue')
        ax.set_xticks(x,[f'{r.scenario}, iter. {r.iteration}' for r in audit.itertuples()])
        ax.set_ylabel('Unidades originales del objetivo');ax.set_title('Separación durante la trayectoria: repetición predefinida 1')
        ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/'figures'/f'independent_trajectory_contrast.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
        add()
    add('P7/P8: signo y precisión de Softmax frente a Fresh')
    q=pairs[(pairs.method_a.isin(['Softmax010','Softmax050']))&(pairs.method_b=='Fresh')]
    add(q[['scenario','method_a','mean_difference','ci95_low','ci95_high','positive_fraction']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    positive=q[q.mean_difference>0];negative=q[q.mean_difference<0]
    add('Medias a favor de Softmax: '+', '.join(f'{r.scenario}/{r.method_a}' for r in positive.itertuples())+'.')
    add('Medias a favor de Fresh: '+', '.join(f'{r.scenario}/{r.method_a}' for r in negative.itertuples())+'.')
    clear=q[(q.ci95_low>0)|(q.ci95_high<0)]
    add('Contrastes cuyo IC exploratorio no cruza cero: '+(', '.join(f'{r.scenario}/{r.method_a}' for r in clear.itertuples()) if len(clear) else 'ninguno')+'. No son pruebas confirmatorias ajustadas por multiplicidad.')
    add()
    add('P9: estabilidad y mecanismo')
    st=stability.groupby(['scenario','method'])[['total_variation','tail_mean_step','clip_fraction','projection_fraction']].mean().reset_index()
    add(st[st.method.isin(['Fresh','Green','GreenSoftmax050'])].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    add('Green reutiliza historia y puede suavizar el ruido, pero sus LR pueden degenerar al alejarse de bancos anteriores. Menor longitud de trayectoria no equivale a mayor objetivo. El reward de gran magnitud en regímenes de demora elevada también aumenta la variabilidad del score-function; se mantuvieron clipping y pasos iguales, sin introducir centrado o controles de varianza.')
    add('P10: PSoft conserva bootstrap de caminos completos B=300; no es Softmax de los valores. Sus contrastes frente a Fresh y ambas temperaturas están en paired_exploratory.csv, y todas sus trayectorias desfavorables se conservan.')
    finals=pd.read_csv(out/'tables/finals_all.csv')
    failures=finals[finals.method.isin(['Fresh','Green','Softmax010','Softmax050','PSoft300','GreenSoftmax050'])].merge(
        gaps[['scenario','rep','method','grid_reference_gap','paired_se']],on=['scenario','rep','method'],validate='one_to_one')
    failures=failures.merge(high[['scenario','R_search']],on='scenario')
    failures=failures[failures.grid_reference_gap>np.maximum(5.,.10*failures.R_search)].sort_values('grid_reference_gap',ascending=False)
    add()
    add('EJECUCIONES DESFAVORABLES CONSERVADAS')
    if len(failures):
        add('Diagnóstico posterior: pérdida frente a la referencia de grilla superior a max(5, 10% de R_search), excluyendo Fixed. No se cambia la repetición ilustrativa 1 ni se reemplazan semillas.')
        add(failures[['scenario','rep','method','z3','z4','objective_holdout','grid_reference_gap','paired_se']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
        from analyze_landscape_010 import background,LABELS,COLORS,STYLES,save
        coordinates=pd.read_csv(out/'tables/coordinates_all.csv')
        for s,rep in failures[['scenario','rep']].drop_duplicates().itertuples(index=False,name=None):
            q=coordinates[(coordinates.scenario==s)&(coordinates.rep==rep)]
            sf=pd.read_csv(out/'high'/f'{s}_surface.csv')
            fig,ax=plt.subplots(figsize=(9,7));im=background(ax,sf,f'{s}: repetición desfavorable {rep}, conservada')
            for method in LABELS:
                curve=q[q.method==method].sort_values('iteration')
                ax.plot(curve.z3,curve.z4,color=COLORS[method],ls=STYLES[method],lw=1.5,label=LABELS[method])
                ax.scatter(curve.z3.iloc[-1],curve.z4.iloc[-1],color=COLORS[method],edgecolor='white',s=45,zorder=10)
            ax.scatter(.12,.10,marker='*',color='gold',edgecolor='black',s=120,label='Inicio',zorder=11)
            fig.colorbar(im,ax=ax,label='J(z)');ax.legend(loc='upper left',bbox_to_anchor=(1.28,1),fontsize=8)
            save(fig,out,f'{s}_unfavorable_rep{rep:02d}_oracle_trajectories')
            fig,axes=plt.subplots(1,2,figsize=(13,5))
            for dim,ax in zip(['z3','z4'],axes):
                for method in LABELS:
                    curve=q[q.method==method].sort_values('iteration')
                    ax.plot(curve.iteration,curve[dim],color=COLORS[method],ls=STYLES[method],label=LABELS[method])
                ax.set(xlabel='Iteration',ylabel=dim,ylim=(-.025,1.025),title=f'{s}: repetición desfavorable {rep}')
            axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8);fig.tight_layout()
            save(fig,out,f'{s}_unfavorable_rep{rep:02d}_probing_coordinates')
    else:
        add('No hubo pérdidas superiores a max(5, 10% de R_search) entre métodos distintos de Fixed. Este resultado negativo también se conserva.')
    add()
    add('LIMITACIONES OPERATIVAS Y DE ENTORNO')
    add('La primera prueba reducida detectó un error de concatenación en una métrica del nuevo controlador; fue corregido antes de la ejecución científica. También se corrigió el tipo heterogéneo de un diccionario de metadatos antes de correr el experimento. No se modificaron formulación, estimadores ni fuentes 002–009b.')
    add('La revisión final corrigió el SNR 0/0 de soluciones idénticas: es no definido (NaN), no evidencia infinita. Se archivaron fuente/diagnósticos previos y se migró sólo ese indicador; 191 archivos de objetivos, SE, decisiones y búsquedas permanecieron idénticos bit a bit. La compatibilidad entre fingerprints está documentada en validation/snr_revision.toml; las superficies conservan el fingerprint de generación original.')
    add('Python informó versiones antiguas de aceleradores opcionales de pandas y falta de Axes3D. Las operaciones de pandas y todos los gráficos 2D requeridos terminaron; no se necesitaron esos componentes opcionales. Las imágenes se verificaron por separado.')
    add('Las conclusiones se limitan a las 56 configuraciones exploradas y cinco repeticiones por método. Mantener separadas la variación Monte Carlo, la variabilidad de optimización y la selección de instancias.')
    (out/'interpretacion_cientifica.txt').write_text('\n'.join(text)+'\n')
    report=(out/'informe_final.txt').read_text().split('\nINTERPRETACIÓN CIENTÍFICA DE LAS INSTANCIAS 010')[0]
    lines=[]
    for line in report.splitlines():
        if line.startswith('P11.'):
            line=f'P11. Para el artículo: L000 como control plano; {recommendation} como ejemplo principal con mejor punto interior; L026 y L030 como contrastes de alto costo temporal y frontera. Se conservan resultados desfavorables.'
        elif line.startswith('P12.'):
            line=f'P12. Ejemplo principal provisional: {recommendation}; requiere contraste relevante y mejor punto de grilla interior, además de trayectorias informativas. Parámetros completos en configs/models.toml. El caso L026 de esquina se conserva como diagnóstico de frontera, no como ejemplo de compromiso interior.'
        lines.append(line)
    (out/'informe_final.txt').write_text('\n'.join(lines).rstrip()+'\n\n'+'\n'.join(text)+'\n')
    index=out/'index.html';body=index.read_text()
    marker='<p id="scientific-review">'
    body=body.split(marker)[0]
    body+=f'{marker}<a href="interpretacion_cientifica.txt">Interpretación científica y atribución económica</a> Ejemplo principal: {recommendation}.</p><img src="figures/economic_attribution.png">'
    body+='<h2>Verificación independiente de diferencias durante el recorrido</h2><img src="figures/independent_trajectory_contrast.png">'
    body+='<h2>Todas las repeticiones conservadas</h2>'
    for s in high.scenario:
        body+=f'<h3>{s}</h3><p>'
        for rep in range(1,6):
            body+=f'<a href="figures/{s}_convergence_rep{rep:02d}_full.png">Rep. {rep}: completa</a><a href="figures/{s}_convergence_rep{rep:02d}_zoom.png">zoom</a> '
        body+='</p>'
    for s,rep in failures[['scenario','rep']].drop_duplicates().itertuples(index=False,name=None):
        body+=f'<h2>Ejecución desfavorable conservada: {s}, repetición {rep}</h2>'
        body+=f'<p>Diagnóstico posterior; la repetición ilustrativa principal sigue siendo 1.</p><img src="figures/{s}_convergence_rep{rep:02d}_full.png">'
        body+=f'<img src="figures/{s}_unfavorable_rep{rep:02d}_oracle_trajectories.png"><img src="figures/{s}_unfavorable_rep{rep:02d}_probing_coordinates.png">'
    index.write_text(body)
    complete=json.loads((out/'COMPLETE.json').read_text());complete['main_illustrative_instance']=recommendation
    complete['scientific_interpretation_complete']=True;complete['png_files']=len(list((out/'figures').glob('*.png')))
    complete['additional_analysis_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('diagnostico_economico_010.jl'),Path(__file__).with_name('validar_contraste_trayectorias_010.jl')]}
    (out/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n')
    print('Scientific interpretation complete; primary example:',recommendation)

if __name__=='__main__':main(sys.argv[1])
