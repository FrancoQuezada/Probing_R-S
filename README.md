# Probing & RS

Research code on **simulation optimization with probing**: choosing how much to invest in
information (probing) before routing work to a FAST or STANDARD server. The repository holds
two *different* models that must not be confused.

| | `two_servers/` | `production_rework/` |
|---|---|---|
| Model | Job-level two-server routing | Production and Rework System (assembly lines) |
| What is probed | Each job: binary signal, accuracy depends on `z` | Each line: Gaussian diagnostic of a latent line condition Θ, precision depends on `z` |
| Decision structure | Policy chosen *before* signals are observed (max over policies inside the expectation) | Two-stage: probe `z`, observe η, **then** select a policy for the whole shift |
| Methods implemented | Fixed-baseline LR, Fresh, Green, Green Softmax, Green Variance, (+Softmax), differential evolution | Nested Monte Carlo, conditional policy selection, finite grid search, optional CRN |
| Language / env | Julia scripts + Python drivers | Julia package `AssemblyLineModel` |
| Entry point | `two_servers/SimOpt/` | `production_rework/assembly_line_model/` |

Fresh/Green likelihood-ratio methods exist **only** for the two-server model. They are not yet
implemented for Production and Rework.

## Layout

```text
Probing&RS/
├── two_servers/
│   ├── SimOpt/            main experiments 002–022, tests, protocols AND resultados/ (~49 GB)
│   ├── lr_baseline_2d/    stand-alone Julia project: job-level LR baseline, 2D probing (z3, z4)
│   └── original_scripts/  earliest scripts (CRN oracle surface, likelihood method)
├── production_rework/
│   ├── assembly_line_model/   validated Julia package (src, test, examples, configs, experiment_outputs)
│   └── docs/                  spec, prompts, background for LR extension, reference PDF
├── archive/               historical material, see archive/README.md
└── docs/                  MIGRATION_MAP.md, CLEANUP_CANDIDATES.md
```

## Environments and tests

Julia 1.11 is used throughout. Nothing here installs packages for you; use `Pkg.instantiate()` where noted.

```bash
# Production & Rework (run from production_rework/)
cd production_rework
julia --project=assembly_line_model -e 'using Pkg; Pkg.instantiate()'
julia --project=assembly_line_model assembly_line_model/test/runtests.jl      # 470 assertions

# Two-server job-level LR baseline
cd two_servers/lr_baseline_2d
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl                                             # 63 assertions

# Two-server SimOpt (example: reduced-fixture test; run from two_servers/SimOpt)
cd two_servers/SimOpt
julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno tests/validate_four_searches_softmax.jl
```

See each example's README for experiments and outputs.

## Where results live

- Two-server: `two_servers/SimOpt/resultados/<NNN>…/` (one directory per experiment). It stays beside
  the scripts because scripts and Python drivers address it as `resultados/...` relative to `SimOpt/`.
- Production & Rework: `production_rework/assembly_line_model/experiment_outputs/` (dense-grid
  experiments `dense_grid_crn`, `dense_grid_current`, …) and `validation_outputs/`.

## Legacy material

`archive/` (details and provenance in [archive/README.md](archive/README.md)): early two-server scripts
(`AllFiles`), an older assembly-line prototype, and a snapshot of the old `2serversModel` directory.
Nothing was deleted. History of the restructuring: [docs/MIGRATION_MAP.md](docs/MIGRATION_MAP.md);
possible further cleanup: [docs/CLEANUP_CANDIDATES.md](docs/CLEANUP_CANDIDATES.md).

Note: this directory is not under Git. Reports and logs inside the results contain absolute paths of
the machine and directory layout in which they were produced; they are provenance and were not rewritten.
