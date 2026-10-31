#!/usr/bin/env python3
"""Final synthesis, generated only after all four complete full experiments."""
from pathlib import Path
import json, warnings, subprocess
with warnings.catch_warnings():
    warnings.simplefilter("ignore",UserWarning)
    import pandas as pd
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT=Path(__file__).resolve().parent
NAMES=["005probing_softmax_multiseed","006probing_softmax_sample_efficiency",
       "007probing_green_softmax","008probing_probability_softmax"]

def conclusion(row):
    if row.p_holm<.05:
        return "evidencia de mejora" if row.mean_difference>0 else "evidencia de deterioro"
    return "sin evidencia concluyente de diferencia (no demuestra equivalencia)"

def describe(r):
    return (f"n={r.n_paths}, {r.analysis}, {r.method_a} frente a {r.method_b}: "
            f"Δ={r.mean_difference:.4f}, IC95% [{r.ci95_low:.4f}, {r.ci95_high:.4f}], "
            f"pHolm={r.p_holm:.4g}: {conclusion(r)}.")

def main():
    subprocess.run(["python3",str(ROOT/"tests/validate_probing_results.py")],check=True,cwd=ROOT)
    dirs={s:ROOT/"resultados"/n for s,n in enumerate(NAMES,5)}
    tables={};finals={};configs={}
    for s,out in dirs.items():
        c=json.loads((out/"COMPLETE.json").read_text())
        assert not c["quick"] and c["completed_replications"]==c["expected_replications"]
        configs[s]=tomllib.loads((out/"configuration.toml").read_text())
        assert configs[s]["n_reps"]==20 and configs[s]["max_iters"]==100
        tables[s]=pd.read_csv(out/"paired_comparisons.csv")
        finals[s]=pd.read_csv(out/"final_evaluations.csv")
    selected=tomllib.loads((dirs[5]/"selected_beta.toml").read_text())["selected_beta"]
    rows=[]
    for s,t in tables.items():
        q=t.copy();q.insert(0,"stage",s);rows.append(q)
    pd.concat(rows,ignore_index=True).to_csv(ROOT/"resultados/comparaciones_probing_005_008.csv",index=False)
    strategies=[]
    for s in (5,7,8):
        f=finals[s]
        f=f[(f.beta==0)|(f.beta==.1)|(f.beta==selected)]
        for method,g in f.groupby("method",sort=False):
            family=("Green + Softmax" if method.startswith("Green +") else "Green" if method=="Green" else
                    "Fresh + PSoft" if method.startswith("PSoft") else "Fresh" if method=="Fresh" else "Fresh + Softmax")
            strategies.append(dict(strategy=family,stage=s,method=method,n_paths=500,iterations=100,n_reps=len(g),
                objective_holdout_mean=g.objective_holdout.mean(),objective_holdout_sd=g.objective_holdout.std(),
                objective_direct_mean=g.objective_direct.mean(),search_seconds_mean=g.search_seconds.mean(),
                bootstrap_seconds_mean=g.bootstrap_seconds.mean(),total_variation_mean=g.total_variation.mean()))
    strategy_table=pd.DataFrame(strategies)
    strategy_table.to_csv(ROOT/"resultados/resumen_cinco_estrategias_005_008.csv",index=False)
    report=["INFORME FINAL: EXPERIMENTOS 005–008",
        "Las cuatro etapas completas contienen 20 repeticiones independientes por configuración. Las pruebas reducidas se guardan aparte. Las conclusiones se refieren al modelo de 144 políticas ex ante de 002–004.",
        "Objetivo: J(z)=max_i E_z[Y_i]−(10,4.5)'z para z=(z3,z4)∈[0,1]². Fresh, Green, Softmax y Fixed originales siguen disponibles sin cambios. PSoft es adicional. No se introduce una etapa externa de información ni el máximo realizado por trayectoria de SimOpt009.",
        f"005 exploró beta=.05,.1,.2,.5. Eligió beta={selected:g} por media de valor holdout; esta selección es exploratoria. 006, 007 y 008 fijaron beta antes de ejecutar con semillas nuevas. beta=.1 se conserva como comparación prefijada.",
        "La estimación directa del hard-max y el valor de política con selección/evaluación independientes son dos estimandos distintos. La primera puede ser optimista; el segundo evalúa una decisión ex ante concreta y puede quedar por debajo de J. Las conclusiones inferenciales siguientes usan holdout; las tablas también muestran direct.",
        "IC t sobre diferencias emparejadas entre repeticiones; pHolm ajusta todos los contrastes de cada etapa, incluidos ambos estimandos y presupuestos. Los IC95% que se citan son marginales; los IC simultáneos Bonferroni figuran en CSV. Las bandas gráficas son ±1 SD entre repeticiones.",
        "RESULTADOS OBSERVADOS: 005, ROBUSTEZ"]
    for r in tables[5][tables[5].estimand=="holdout"].itertuples():report.append(describe(r))
    report.extend(["P1. ¿SOFTMAX MEJORA LA EFICIENCIA CON POCAS SIMULACIONES?",
        "006 separa 100 iteraciones de un presupuesto de 50000 trayectorias. En el primero cambia el presupuesto al cambiar n; en el segundo cambian el número de pasos y sus tamaños decrecientes. No se atribuye todo a n."])
    for r in tables[6][tables[6].estimand=="holdout"].itertuples():report.append(describe(r))
    target=pd.read_csv(dirs[6]/"time_to_target.csv")
    report.append("Costos hasta J=84.0 (objetivo prefijado), descriptivos, en checkpoints:")
    for r in target.itertuples():
        report.append(f"n={r.n_paths}, {r.method}: "+(f"{r.first_search_paths:.0f} trayectorias; {r.first_search_seconds:.3f} s de búsqueda." if r.reached else "objetivo no alcanzado en los checkpoints evaluados."))
    report.append("Una mejora de calidad con igual presupuesto apoya eficiencia estadística para esa configuración. Los tiempos con cuatro repeticiones concurrentes son descriptivos, incluyen contención y no establecen por sí solos una ventaja de tiempo aislada.")
    report.extend(["P2. ¿GREEN Y SOFTMAX PRODUCEN BENEFICIOS COMPLEMENTARIOS?",
        "007 agrega primero los valores y gradientes individuales de todas las políticas sobre todos los bancos históricos; después calcula los pesos. Green conserva su estimador original. No se cambian los LR ni su clipping."])
    for r in tables[7][tables[7].estimand=="holdout"].itertuples():report.append(describe(r))
    report.append("La evidencia anterior mide el beneficio adicional del suavizado dentro de Green. Las diferencias de medias Green–Fresh entre etapas distintas no constituyen una comparación emparejada ni aíslan causalmente la reutilización histórica.")
    report.extend(["P3. ¿PSOFT MEJORA EL DESEMPEÑO FRENTE A SOFTMAX?",
        "008 comparte primitivas de búsqueda entre Fresh, Softmax y PSoft. Todos consumen 50000 trayectorias por búsqueda; el bootstrap B=300 trabaja sobre Y y no genera trayectorias adicionales."])
    for r in tables[8][tables[8].estimand=="holdout"].itertuples():report.append(describe(r))
    psoft= tables[8][(tables[8].estimand=="holdout")&tables[8].method_a.str.startswith("PSoft")]
    strong=psoft[(psoft.p_holm<.05)&(psoft.mean_difference>0)&psoft.method_b.isin(["Fresh",f"Softmax beta={selected}"])]
    if not strong.empty:
        report.append("J. Hay motivación empírica inicial para estudiar Green + PSoft: PSoft mejora al menos una referencia fuerte en esta configuración. Esto no demuestra que el beneficio se traslade a Green; el remuestreo de historia adaptativa requiere un diseño nuevo.")
    else:
        report.append("J. Estos resultados no ofrecen una justificación empírica fuerte para esperar que Green + PSoft mejore la calidad: PSoft no supera con evidencia ajustada a Fresh ni al Softmax seleccionado en 005. Puede investigarse por interés en incertidumbre de selección, como hipótesis adicional y sin promesa de mejora.")
    same=pd.read_csv(dirs[8]/"psoft_softmax_same_bank_comparisons.csv")
    for beta,g in same.groupby("beta"):
        report.append(f"A–C. Mismos puntos/bancos, PSoft frente a Softmax beta={beta:g}: L1 medio de pesos={g.l1_weight_distance.mean():.5f}; ||gP−gS|| medio={g.gradient_difference.mean():.5f}; coseno medio={g.cosine.mean():.5f}.")
    weights=pd.read_csv(dirs[8]/"mean_policy_weights.csv")
    pweights=weights[weights.method.str.startswith("PSoft")]
    policies=pd.read_csv(dirs[8]/"policies.csv")
    for col,label in [("weight","PSoft"),("soft_weight","Softmax .1 en bancos de PSoft")]:
        top=pweights.nlargest(5,col).merge(policies,on="policy")
        report.append("Políticas de mayor peso medio de "+label+": "+"; ".join(f"{r['policy']} (av={r['av']:g}, aq={r['aq']:g}, as={r['as']:g}, tau={r['tau']:g}, peso={r[col]:.4f})" for r in top.to_dict("records"))+".")
    gc=pd.read_csv(dirs[8]/"fixed_point_gradient_comparisons.csv")
    for r in gc.itertuples():
        report.append(f"C. Diferencia media de gradiente PSoft−{r.reference} en {r.coordinate}, mismo Z0 y bancos emparejados: {r.mean_gradient_difference:.5f}, IC95% [{r.ci95_low:.5f}, {r.ci95_high:.5f}], pHolm={r.p_holm:.6g}.")
    variance=pd.read_csv(dirs[8]/"fixed_point_gradient_variance.csv")
    report.append("D. Cocientes de traza de covarianza PSoft/referencia en Z0, con bancos independientes entre repeticiones:")
    for r in variance.itertuples():report.append(f"Referencia {r.method}: cociente={r.psoft_variance_ratio:.4f}, IC bootstrap simultáneo aproximado95% [{r.ratio_simultaneous95_low:.4f}, {r.ratio_simultaneous95_high:.4f}]. No se generaliza automáticamente a todos los puntos.")
    vr=variance[variance.method=="Fresh"].iloc[0]
    report.append(f"D. En Z0 PSoft no redujo la varianza frente a Fresh: la traza de covarianza observada fue {(vr.psoft_variance_ratio-1)*100:.2f}% mayor. La inferencia bootstrap anterior es aproximada y corresponde a ese punto fijo.")
    report.append("G. Estabilidad de búsqueda y dispersión final en 008:")
    for m,g in finals[8].groupby("method",sort=False):
        report.append(f"{m}: variación total media de z={g.total_variation.mean():.4f}; SD del objetivo holdout={g.objective_holdout.std():.4f}; SD(z3)={g.z3.std():.4f}, SD(z4)={g.z4.std():.4f}; búsqueda media={g.search_seconds.mean():.3f} s; bootstrap medio={g.bootstrap_seconds.mean():.3f} s.")
    sens=pd.concat([pd.read_csv(f) for f in (dirs[8]/"replications").glob("*_bootstrap_sensitivity.csv")],ignore_index=True)
    gd=pd.read_csv(dirs[8]/"iteration_diagnostics.csv")
    anchors=gd[gd.method.str.startswith("PSoft")][["rep","iteration","gradient_norm"]]
    b300=sens[sens.B==300].merge(anchors,on=["rep","iteration"])
    rel=b300.gradient_error/b300.gradient_norm.clip(lower=1e-12)
    report.append(f"H. Sensibilidad de B: B=300 obtuvo error L1 medio de pesos={b300.l1_error.mean():.5f}, error relativo medio del gradiente={100*rel.mean():.3f}%, percentil95={100*rel.quantile(.95):.3f}% y máximo={100*rel.max():.3f}% frente a B=10000. Esta precisión observada resulta razonable como configuración inicial del estudio; B=1000 reduce más el error MC. No es garantía para todos los bancos ni implica menor varianza de búsqueda.")
    report.append("COMPARACIÓN DESCRIPTIVA DE LAS CINCO FAMILIAS (n=500, 100 iteraciones; no emparejar entre etapas):")
    for r in strategy_table.itertuples():
        report.append(f"Etapa {r.stage:03d}, {r.method}: objetivo holdout medio={r.objective_holdout_mean:.5f}, SD={r.objective_holdout_sd:.5f}; variación total media={r.total_variation_mean:.5f}; búsqueda media={r.search_seconds_mean:.3f} s.")
    report.extend(["A–C, H–J. El informe 008 documenta distancias entre pesos, políticas dominantes, diferencias y cosenos de gradientes sobre los MISMOS bancos, sensibilidad de B y costo bootstrap. Las series temporales no se tratan como muestras independientes para pruebas de significación.",
        "INTERPRETACIONES METODOLÓGICAS Y LÍMITES",
        "Las frecuencias bootstrap son incertidumbre de selección condicional a un banco finito, no probabilidades verdaderas ni bayesianas de optimalidad. PSoft no es necesariamente insesgado para el gradiente de J. Más B reduce error Monte Carlo de los pesos, sin garantizar menor varianza de la búsqueda.",
        "Las diferencias de pesos no implican automáticamente mejores puntos finales. Se priorizan las comparaciones emparejadas con ajuste múltiple; no se concluye superioridad a partir de suavidad visual, un solo recorrido o un p marginal sin ajustar.",
        "Interpretación del suavizado: con valores y gradientes poblacionales, los pesos Softmax a beta fijo producen el gradiente de un log-sum-exp. Ese gradiente puede desplazar la búsqueda respecto al hard-max, aunque reduzca discontinuidades de selección. Esto es un mecanismo posible para diferencias de calidad; las evaluaciones y los objetivos reportados siguen usando el hard-max original.",
        "Green + PSoft puede estudiarse como extensión adicional si los diagnósticos de selección y la relación calidad/costo lo justifican; los resultados Fresh no garantizan complementariedad con bancos históricos adaptativos.",
        "El error MC de evaluación se informa por separado mediante SE condicionales y SE de diferencias con CRN. La variabilidad entre repeticiones combina optimización, selección final y evaluación; el descuento de varianza es aproximado y no corrige sesgo de máximo muestral.",
        "ARTEFACTOS Y REPRODUCCIÓN",
        "Cada directorio 005–008 contiene configuración, semillas, políticas, CSV por repetición y agregados, PNG, logs, pruebas, informe, checkpoints y reproduce.txt. Comparaciones consolidadas: comparaciones_probing_005_008.csv. Resumen descriptivo de las cinco familias: resumen_cinco_estrategias_005_008.csv.",
        "Continuar/reproducir la cadena: python3 run_probing_005_008.py. Cada etapa completa se reutiliza sin regenerar trayectorias. Los hashes de las fuentes originales se verifican antes de cerrar cada etapa."])
    destination=ROOT/"resultados/informe_probing_005_008.txt"
    destination.write_text("\n\n".join(report)+"\n")
    print(destination)

if __name__=="__main__":main()
