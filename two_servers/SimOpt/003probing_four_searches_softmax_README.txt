003probing_four_searches_softmax.jl

Implementa Fresh + Softmax como cuarto método, usando exclusivamente la ecuación
(2) de «Notes on Softmax», Probing_R_S.pdf, página 25. Las páginas 24–26 se
revisaron para distinguir esta propuesta del método de probabilidades de ser
argmax de la ecuación (3), que NO se implementa.

La adaptación al modelo actual no introduce eta ni una decisión adicional:
    J(z) = max_i E_z[Y_i] - alpha'z, z=(z3,z4) en [0,1]^2.

En cada iteración Softmax genera exactamente un banco nuevo como Fresh:
    Z_i = mean_l Y[i,l]
    w_i = exp(beta_soft*(Z_i-max Z)) / sum_j exp(beta_soft*(Z_j-max Z))
    G_i = mean_l Y[i,l] score_l(z)
    g = sum_i w_i G_i - alpha.

El score proviene de loglr_and_score(path,z,z): LR=1. Se reutiliza la matriz Y,
sin simulaciones adicionales para las políticas. El coste se resta una vez,
fuera de la combinación. Para beta constante, esta combinación es el gradiente
de un surrogate log-sum-exp respecto de las medias; NO es la derivada de una
media ponderada sum_i w_i Z_i con pesos dependientes de z.

SOFTMAX_BETA=0.5 es configurable al comienzo del archivo. Es independiente de
BETA=2.20, que sigue controlando la calidad de las señales. No se introduce un
calendario para beta. Con argmax único, beta grande recupera el gradiente Fresh.
En empates, el límite mezcla las políticas empatadas y no tiene por qué coincidir
con la primera política devuelta por argmax. Los gradientes estimados con pesos
calculados sobre la misma muestra no se afirman insesgados para un objetivo
poblacional a tamaño finito.

Los parámetros de los cuatro métodos permanecen en n=500, 45 trabajos y 100
iteraciones. FIXED, FRESH, GREEN y projected_step se conservan textualmente.
Las semillas originales se mantienen. Softmax usa SEED_SOFTMAX=SEED_FRESH y
la misma fórmula SEED_FRESH+10000*k, aplicando CRN entre sus bancos nuevos.
Las decisiones distintas reconstruyen señales distintas sobre el mismo ruido.

SOFTMAX_BETA tiene unidades inversas de recompensa por trayectoria: con 0.5,
un gap de 2 unidades implica una razón de pesos exp(-1). La intensidad efectiva
del suavizado depende de la escala de recompensas y del horizonte; por eso los
diagnósticos de la prueba reducida no sustituyen a los del experimento de 45
trabajos. El gap log-sum-exp respecto de max está acotado por log(144)/beta;
con beta=0.5 la cota es aproximadamente 9.94. Ese surrogate nunca sustituye J
en las curvas comunes ni en el oracle.

Las funciones principales añadidas son:
- softmax_weights: pesos estables y comprobación de normalización.
- combine_softmax_gradients: combinación convexa, coste único, max w y K_eff.
- softmax_estimator: medias/G_i de un banco Fresh y su gradiente combinado.
- optimize_softmax: búsqueda con projected_step sin alterar su regla.
- save_four_results: resultados exactos JLS, CSV y metadatos TOML.

Las funciones de evaluación y gráficos se amplían a cuatro métodos. Las cuatro
trazas se evalúan sobre EL MISMO banco CRN independiente, utilizando siempre
max_i mean(Y_i)-alpha'z. El heatmap conserva el procedimiento original completo.
El objetivo Softmax/log-sum-exp solo se registra como diagnóstico; no se usa como
eje Y de la comparación principal. Los nuevos gráficos tienen márgenes mayores
para evitar los recortes de etiquetas observados en los gráficos originales.

Ejecución reproducible, desde la raíz de SimOpt, usando el entorno preparado:

    julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno \
        -e 'using Pkg; Pkg.instantiate()'

    cd resultados/003probing_softmax_20261021/completo
    env GKSwstype=100 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        julia --startup-file=no --project=../entorno \
        ../../../003probing_four_searches_softmax.jl

Para ejecutar en otro entorno Julia, basta con disponer de Plots; el resto son
bibliotecas estándar. No se instala ningún paquete adicional respecto al entorno
ya preparado. La ejecución comprobada utiliza Julia 1.11.7 y Plots 1.41.1.

Validaciones, desde la raíz de SimOpt:

    env GKSwstype=100 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno \
        tests/validate_four_searches_softmax.jl

Las pruebas cargan fixtures reducidas en módulos separados, sin editar fuentes.
Comprueban pesos finitos/no negativos/normalizados, invariancia a traslación,
valores extremos y empates, coste único, gradientes finitos, límite beta grande,
dominio, semillas comunes y evaluación hard-max con banco CRN común. También
reproducen exactamente los tres algoritmos originales en un problema reducido.
La validación por diferencias finitas usa un banco fijo con LR y el surrogate
log-sum-exp. No deriva las realizaciones CRN discontinuas por diferencias finitas.

La prueba integral reducida se guarda por separado en resultados/.../prueba/.
Cambia solo tamaños en una copia: 12 trabajos, 40 trayectorias, 4 iteraciones,
oracle de 100 trayectorias y malla 3×3/curvas con 40 trayectorias. No modifica
las constantes del archivo 003 ni del original 002. Es una prueba funcional,
no un resultado comparable al horizonte de 45 trabajos.

Salidas nuevas, escritas en el directorio de ejecución:
    probing_four_objective_vs_iteration.png          (gráfico principal)
    probing_four_trajectories_crn_oracle_surface.png
    probing_four_coordinate_paths.png
    probing_four_raw_search_objective_vs_iteration.png
    probing_four_oracle_endpoints.png
    probing_four_traces.csv
    probing_four_softmax_diagnostics.csv
    probing_four_oracle_endpoints.csv
    probing_four_oracle_surface.csv
    probing_four_metadata.toml
    probing_four_searches.jls
    probing_four_results.jls

Los CSV conservan precisión de Float64 (17 cifras significativas). El JLS guarda
los arrays completos sin redondeo. El archivo searches.jls se guarda antes de
las evaluaciones costosas; results.jls incluye también las trazas y la superficie.
El CSV de Softmax registra max w, K_eff=1/sum w², el gradiente sin/con recorte,
paso, semilla y surrogate. Los metadatos registran parámetros, semillas, versiones,
huellas SHA-256 y presupuesto. El nuevo archivo no ejecuta main al incluirlo;
ejecute como script o llame explícitamente a main en una sesión/módulo separado.

Interpretación estadística y presupuestos:
- FIXED genera 500 trayectorias de búsqueda; FRESH/GREEN/SOFTMAX generan 50000
  cada uno. Igual número de pasos no implica igual presupuesto total.
- Los cuatro puntos finales se evalúan por simulación directa con 30000
  trayectorias cada uno. Los tres originales mantienen semillas oracle +1,+2,+3;
  Softmax usa +4. No se alteran los resultados originales al añadir el cuarto.
- Los intervalos oracle se basan en la media de una política seleccionada sobre
  la misma muestra. No ajustan por selección de política ni por la búsqueda de z.
- Una sola ejecución por método no establece superioridad media de algoritmos.
- Fresh, Green y Softmax conservan el objetivo/política anteriores en un eventual
  punto terminal añadido sin simulación adicional. El CSV lo marca mediante
  internal_evaluated=false y el gráfico bruto omite ese valor provisional. La
  curva común y la evaluación oracle sí evalúan el punto final correcto.

El experimento original 002 permanece intacto y sigue ejecutándose como antes.
No se implementa la ecuación (3), bootstrap, aproximación normal multivariante,
nueva información eta, nuevas políticas ni cambios en el sistema de routing.

Precisión sobre independencia de las evaluaciones auxiliares:
Se conserva una coincidencia de semillas del experimento original:
TRACE_EVAL_SEED=20260726 coincide con la semilla oracle de Fresh (+2), y
ORACLE_SURFACE_SEED=20260725 con la semilla oracle de Fixed (+1). En cada caso,
las primeras 5000 primitivas son compartidas entre esas evaluaciones auxiliares.
El banco común es independiente de las búsquedas, pero no de TODOS los otros
bancos de evaluación. Los cuatro oracle finales sí usan semillas distintas
entre sí. Se documenta esta dependencia sin alterar las semillas originales.
