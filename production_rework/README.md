# Production and Rework System

Several assembly lines have latent conditions Θ_c ~ N(μ_c, τ_c²) that drive the rework arrival rate
λ_c(Θ) = λ⁰ exp(γ(Θ−μ)) and the probability of complex rework (logistic). Before the shift the decision maker
chooses probing effort `z ∈ [0,1]^m`, observing η_c | Θ_c ~ N(Θ_c, ν_c/(T_max z_c)); then a routing policy
(`a_v·v_c − a_q(q_F − q_S) ≥ τ`) is chosen and fixed for the whole shift. The objective is
`J(z) = −αᵀz + E_η|z[ max_i E[Y(x_i, ξ) | η, z] ]`.

Theory: [docs/Probing_R_S-2.pdf](docs/Probing_R_S-2.pdf); spec and background: `docs/`.
This is **not** the job-level two-server model (see `../two_servers/`).

## Package `assembly_line_model/` (reference implementation)

Julia package `AssemblyLineModel` (Julia 1.11). Architecture: `assembly_line_model/ARCHITECTURE.md`;
scientific audit: `assembly_line_model/FINAL_VALIDATION_REPORT.md`.
Implemented: nested Monte Carlo (`evaluate_z`), conditional policy selection, finite grid search, optional CRN
(`evaluate_z_crn`). **Not implemented:** likelihood-ratio / Green / gradient methods.

All commands below are run from `production_rework/`:

```bash
julia --project=assembly_line_model -e 'using Pkg; Pkg.instantiate()'
julia --project=assembly_line_model assembly_line_model/test/runtests.jl

# small smoke experiment, independent streams
julia --project=assembly_line_model assembly_line_model/examples/run_grid_experiment.jl \
  --config=assembly_line_model/configs/reference_experiment.toml \
  --outer-reps=4 --inner-train=10 --inner-eval=15 --z-step=0.5 \
  --output-dir=assembly_line_model/experiment_outputs/my_smoke
# same with common random numbers: add  --randomization=crn
# dense grid (expensive):
julia --project=assembly_line_model assembly_line_model/examples/run_grid_experiment.jl \
  --config=assembly_line_model/configs/reference_experiment.toml \
  --seed=20260726 --outer-reps=100 --inner-train=300 --inner-eval=1000 --z-step=0.05 \
  --output-dir=assembly_line_model/experiment_outputs/dense_grid_current
```

`docs/Commands` has the heatmap command. Runner options: `assembly_line_model/EXPERIMENT_RUNNER.md`.

## Results

`assembly_line_model/experiment_outputs/` (notably `dense_grid_crn`, `dense_grid_current`,
`crn_validation`, `independent_validation`) and `assembly_line_model/validation_outputs/`. Their
`resolved_configuration.toml` / reports record `source_config` with the absolute path at the time of the
run (the old `2serversModel/` location for older runs); this is provenance only.

## `docs/`

`ASSEMBLY_LINES_IMPLEMENTATION_SPEC.md`, `CODEX_ASSEMBLY_LINES_IMPLEMENTATION_PROMPT.md` (original
implementation spec/prompt), `background_rework.md`, `prompt_codex_rework.md` (background and prompt for the
planned LR extension), `Probing_R_S-2.pdf` + `.txt` (theory, rework version), `Commands`.
