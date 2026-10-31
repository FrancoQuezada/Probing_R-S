# Generic Grid Experiment Runner

The generic runner separates scientific inputs from Julia source code:

```text
TOML + CLI overrides -> validated configuration -> ModelParameters
-> build_policy_grid -> build_z_grid -> finite_grid_search -> outputs
```

Run it from the `production_rework/` directory (the parent of this package):

```bash
julia --project=assembly_line_model \
  assembly_line_model/examples/run_grid_experiment.jl \
  --config=assembly_line_model/configs/reference_experiment.toml \
  --z-step=0.5 \
  --output-dir=assembly_line_model/experiment_outputs/reference_smoke_generic
```

`--config` is required except with `--help`. CLI values override TOML; TOML
overrides only non-scientific runner defaults. Model fields have no hidden
defaults and must be supplied in TOML.

## TOML schema

The supplied [reference configuration](configs/reference_experiment.toml) has
four sections plus one `[[lines]]` table per assembly line:

- `[experiment]`: `name`, `seed`, `outer_reps`, `inner_train`, `inner_eval`,
  `output_dir`.
- `[model]`: `horizon`, economic parameters, and `probing_costs`.
- `[[lines]]`: the exact `LineParameters` fields, using
  `baseline_arrival_rate` for `base_arrival_rate` and separate simple/complex
  means for each server.
- `[policies]`: `av_values`, `aq_values`, `tau_values`, `max_policies`.
- `[probing_grid]`: either `mode="step"` with `step`, or `mode="explicit"`
  with `coordinate_values`, plus `max_candidates`.
- `[randomization]` (optional): `mode="independent"` (the default) or
  `mode="crn"`, with an optional exact `reference_z` for paired reporting.

Step mode requires a positive step dividing one within `1e-12`; it constructs
coordinates through integer intervals, so `0.05` gives exactly 21 values from
0 to 1. Explicit mode uses one comma-separated coordinate vector per line.

## CLI overrides

Supported options are `--seed`, `--outer-reps`, `--inner-train`,
`--inner-eval`, `--z-step`, `--z-values`, `--av-values`, `--aq-values`,
`--tau-values`, `--max-candidates`, `--max-policies`, `--output-dir`, and
`--experiment-name`. Both `--key=value` and `--key value` forms work. `--z-step`
and `--z-values` are mutually exclusive.

`--randomization=crn` aligns base draws by outer replication, phase, inner path,
line, job-generation index, and primitive. It does not share a mutable RNG:
each primitive has a deterministic domain-separated substream. This induces
correlation between candidate estimates to reduce uncertainty in paired
differences; individual-candidate outer SEs need not decline. In CRN mode the
runner additionally writes `outer_replications.csv` and `paired_vs_reference.csv`.
If no `--crn-reference-z`/`reference_z` is supplied, the first deterministic
grid candidate is used as the reference.

Examples:

```bash
# Dense regular grid; do not start this research-size run unintentionally.
julia --project=assembly_line_model assembly_line_model/examples/run_grid_experiment.jl \
  --config=assembly_line_model/configs/reference_experiment.toml \
  --seed=20260726 --outer-reps=100 --inner-train=300 --inner-eval=1000 \
  --z-step=0.05 --output-dir=assembly_line_model/experiment_outputs/dense_grid_current

# Explicit nonuniform two-line grid.
julia --project=assembly_line_model assembly_line_model/examples/run_grid_experiment.jl \
  --config=assembly_line_model/configs/reference_experiment.toml \
  --z-values="0,0.25,0.5,0.75,1;0,0.1,0.2,0.5,1" \
  --output-dir=assembly_line_model/experiment_outputs/nonuniform_grid
```

## Outputs and reproducibility

The selected output directory receives only `candidate_results.csv`,
`experiment_report.md`, and `resolved_configuration.toml`; unrelated files are
preserved. The resolved TOML records every effective input, including CLI
overrides and final coordinate vectors. Outputs omit timestamps.

Candidate streams remain independent across z. Rankings are finite-grid,
Monte-Carlo estimates, not significance tests or continuous optima. No CRN is
implemented by this runner.
