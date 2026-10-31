# Archive

Historical material, moved here (not deleted) during the restructuring. Original locations are relative to
the old repository root.

| Archived path | Original location | Purpose | Why archived | Superseded by |
|---|---|---|---|---|
| `rework_prototypes/assemblylines_classical_two_stage/` | `Assemblylines/assembly_line_two_stage.jl`, `…_2d.jl`, its test, `results/assembly_line_*` (22 Jul) | First assembly-line prototype: Θ/η two-stage with a *different* model (probit complexity, 45 fixed jobs, exponential interarrivals, binary signal, random job values) | Not the validated model; kept for traceability. 68 tests pass (`julia --project=. test/runtests.jl` there; `Project/Manifest` are copies of `two_servers/lr_baseline_2d`). | `production_rework/assembly_line_model/` |
| `two_servers_legacy/AllFiles/` | `AllFiles/` | Early two-server scripts `SimOpt001–009`, 2D hill-climb/SGD/policy comparison (they `include("SimOpt009.jl")`) | Early versions of the job-level code. | `two_servers/SimOpt/` (`SimOpt005/009` are still there) |
| `old_snapshots/2serversModel_org/` | `2serversModel_org/` | Snapshot of the old `2serversModel` before the assembly-line work | Byte-identical duplicates of `two_servers/original_scripts/*`, `production_rework/docs/*` (spec, prompt, PDF, audit txt). Only unique file: `probing_surface_parameter_screen.jl`. | the above |

Duplicates inside the archive were deliberately not removed; see `docs/CLEANUP_CANDIDATES.md`.
