# Cleanup candidates (nothing below has been deleted)

Exact duplicates (md5-identical):
- `archive/old_snapshots/2serversModel_org/{probing_three_searches_crn_oracle_surface.jl,LikelihoodMethod002.jl}` = `two_servers/original_scripts/*`.
- `archive/old_snapshots/2serversModel_org/{ASSEMBLY_LINES_IMPLEMENTATION_SPEC.md,CODEX_…_PROMPT.md,Probing_R_S-2.pdf,audit_outputs/Probing_R_S-2.txt}` = `production_rework/docs/*`.
  The only unique file in that snapshot is `probing_surface_parameter_screen.jl`.
- `archive/two_servers_legacy/AllFiles/SimOpt009_2D_clean.jl` = `two_servers/SimOpt/SimOpt009_2D_clean.jl`.
- `two_servers/SimOpt/SimOpt009.jl` = `two_servers/SimOpt/SimOpt010.jl` (two names, one content).

Reproducible or large:
- `two_servers/SimOpt/resultados/*.zip` (`002…`, `003…`, `004…`, `010…`) are archives of directories that also exist unzipped.
- `__pycache__/` directories and the `quick_test*`/`prueba*` subfolders inside results; many empty folders in `resultados/`.
- `SimOpt/resultados` is ~49 GB: consider Git LFS or external storage if the repo is versioned.
- Production & Rework: `experiment_outputs/*smoke*` and `reference_smoke*` are tiny regression outputs; `dense_grid_crn/outer_replications.csv` is 6 MB.

Consistency/organisation:
- Several different `Probing_R_S*.pdf` versions exist; decide which one is canonical.
- `two_servers/SimOpt` mixes Julia, Python drivers and Spanish/English naming; a `src/`+`experiments/` split would need path changes in drivers and re-fingerprinting.
- `assembly_line_model/Project.toml` declares CSV, DataFrames and Plots; `src/` itself uses only `Random` (examples use the rest).
- Initialise Git (not done) so history exists from this point.
