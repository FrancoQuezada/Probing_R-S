# Two-Server Routing with Probing (job-level model)

Jobs arrive and are routed to a FAST or STANDARD server. Jobs have a class and a latent complexity;
probing `z` improves a *binary* signal (hit/false-positive probabilities depend on `z`). A routing policy
`a_v·value − a_q·(q_F − q_S) + a_s·signal ≥ τ` is chosen from a finite family. The objective is expected
reward minus a linear probing cost. Theory: the `Probing_R_S*.pdf` files in `SimOpt/` and `lr_baseline_2d/`.

This is **not** the Production and Rework model (see `../production_rework/`).

## Contents

| Path | Role |
|---|---|
| `SimOpt/` | Main research workspace. Run everything from this directory. |
| `SimOpt/002probing_three_searches_n500_full45iters.jl` | Reference simulator and original searches (fixed-baseline LR / CRN oracle). |
| `SimOpt/003probing_four_searches_softmax.jl`, `004`–`008` | Softmax variants, sensitivity, multiseed, sample efficiency, probability-softmax. |
| `SimOpt/009*`, `010`, `011`, `probing_regime_*.jl` | Parameter regimes, instance landscape search, high-contrast interior instances (I0252). |
| `SimOpt/012`–`019` | Anytime budget, Green efficiency, Green + Softmax (β = 0.10/0.90/2.00), Green bank variants, five-method comparison, I0252 statistical study. |
| `SimOpt/resultados/020…`, `021_green_variance_softmax/`, `022_differential_evolution/` | Four-instance statistical study, Green Variance (+Softmax), differential-evolution benchmark. Their code is in each directory's `code/`. |
| `SimOpt/tests/` | Validation scripts (most use reduced fixtures). |
| `SimOpt/PROTOCOLO_*.txt`, `README.md`, `CONTEX.md`, `CODEX_CONTEXT_Probing_RS.md` | Protocols and notes. |
| `SimOpt/resultados/` | All results (~49 GB), including oracle/reference runs (e.g. `011…`, `017…`) and Julia environments (`003probing_softmax_20261021/entorno` is the shared one). |
| `lr_baseline_2d/` | Self-contained Julia project (Project/Manifest, tests) for the fixed-baseline LR estimator with projected gradient on `(z3, z4)`; heatmaps and policy regions. Previously in `Assemblylines/`. |
| `original_scripts/` | Earliest job-level scripts kept for traceability. |

## Running

```bash
cd two_servers/SimOpt
# shared environment used by the experiment drivers
julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno -e 'using Pkg; Pkg.status()'
# reduced-fixture validation (verified)
julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno tests/validate_four_searches_softmax.jl
# experiment drivers (see the PROTOCOLO_* and each results README for arguments; some are expensive)
python3 run_green_five_comparison_017.py
```

```bash
cd two_servers/lr_baseline_2d
julia --project=. test/runtests.jl
julia --project=. probing_heatmap_2d.jl      # writes PNGs into the current directory
```

## Why `SimOpt/` was moved as one unit

Scripts, Python drivers and analysis code reference `resultados/...` and `--project=resultados/…/entorno`
relative to `SimOpt/`, and results record fingerprints of the scripts. Splitting scripts from results into the
suggested `experiments/`/`results/` layout would have broken reproducibility of ~90,000 files, so the internal
layout was left untouched.
