# Migration map (31 Oct 2026)

Recovery point (outside the repository): `../Probing&RS_backup_pre_restructure_20261031/` containing
`code_and_small_files.tar.gz` (everything except `SimOpt/resultados`), `md5_non_resultados.txt` and
`resultados_manifest.txt` (89,837 files, 52.17 GB). `SimOpt/resultados` was only *renamed* (same filesystem),
never copied; after the move its file list and sizes match the manifest exactly. The repository was not under Git.

| Old location | New location |
|---|---|
| `SimOpt/` (entire directory, incl. `resultados/`) | `two_servers/SimOpt/` |
| `2serversModel/assembly_line_model/` | `production_rework/assembly_line_model/` |
| `2serversModel/ASSEMBLY_LINES_IMPLEMENTATION_SPEC.md`, `CODEX_ASSEMBLY_LINES_IMPLEMENTATION_PROMPT.md`, `Commands`, `Probing_R_S-2.pdf` | `production_rework/docs/` |
| `2serversModel/audit_outputs/Probing_R_S-2.txt` | `production_rework/docs/Probing_R_S-2.txt` |
| `Assemblylines/background_rework.md`, `prompt_codex_rework.md` | `production_rework/docs/` |
| `2serversModel/probing_three_searches_crn_oracle_surface.jl`, `LikelihoodMethod002.jl` | `two_servers/original_scripts/` |
| `Assemblylines/{Project,Manifest}.toml, README.md, Probing_R_S-1.pdf, single_job_two_stage*.jl, probing_heatmap_2d.jl, probing_policy_regions_2d.jl, test/test_single_job_two_stage.jl, results/single_job_*` | `two_servers/lr_baseline_2d/` |
| `Assemblylines/assembly_line_two_stage*.jl`, `test/test_assembly_line_two_stage.jl`, `results/assembly_line_*` | `archive/rework_prototypes/assemblylines_classical_two_stage/` |
| `AllFiles/` | `archive/two_servers_legacy/AllFiles/` |
| `2serversModel_org/` | `archive/old_snapshots/2serversModel_org/` |

Removed: the now-empty directories `Assemblylines/` and `2serversModel/`.

## Content edits (only these; no model or numerical code)

- `two_servers/lr_baseline_2d/test/runtests.jl`: includes only the single-job test (it also included the prototype test).
- `archive/…/assemblylines_classical_two_stage/test/runtests.jl` (new, includes only the prototype test) and
  `…/test/test_assembly_line_two_stage.jl` line ~120 (path to the two scripts that moved to `lr_baseline_2d/`).
- `archive/…/assemblylines_classical_two_stage/{Project,Manifest}.toml`: copies of the original Assemblylines environment.
- `production_rework/assembly_line_model/{ARCHITECTURE,EXPERIMENT_RUNNER,REFERENCE_EXPERIMENT}.md`: the phrase "repository root" now names `production_rework/`.
- New: root `.gitignore`, READMEs, this file.

## Deviations from the proposed tree

- No `two_servers/{src,experiments,configs,results,plots,test,docs}` split: SimOpt's scripts, drivers and results are path-coupled (`resultados/…`, `--project=resultados/…/entorno`, script fingerprints). Kept as `two_servers/SimOpt/`.
- No `docs/shared/`: there was no genuinely shared documentation. `Probing_R_S-1/-2.pdf` versions in `SimOpt/`, `lr_baseline_2d/` and `production_rework/docs/` differ from one another (different checksums) and were all kept next to the code they document.
- Provenance text in about 676 logs/reports/metadata files inside the results still names the old absolute paths (`…/SimOpt`, `…/2serversModel`). They were not rewritten.
