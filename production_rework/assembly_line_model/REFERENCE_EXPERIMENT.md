# Reference Assembly-Line Experiment

`examples/reference_experiment.jl` is a small synthetic end-to-end example of
the implemented assembly-line model. Its parameters are structural validation
inputs, not empirical calibration or a recommended operational policy.

The reference wrapper now delegates to the generic TOML runner. See
[EXPERIMENT_RUNNER.md](EXPERIMENT_RUNNER.md) to change grids, sample sizes,
policy coefficients, or model parameters without editing Julia source.

Run the default smoke experiment from the `production_rework/` directory (the parent of this package):

```bash
julia --project=assembly_line_model \
  assembly_line_model/examples/reference_experiment.jl \
  --mode=smoke \
  --seed=20260726 \
  --output-dir=assembly_line_model/experiment_outputs/reference_smoke
```

The supported options are `--mode=smoke`, `--mode=reference`, `--seed=<Int>`,
and `--output-dir=<path>`. Smoke mode uses `(outer, train, eval) = (5, 20,
30)`; reference mode uses `(30, 100, 200)`. The latter remains deliberately
moderate but is not run by the test suite.

The example constructs a 12-policy `Policy(av, aq, tau)` family with
`build_policy_grid`, a nine-point two-line z grid with `build_z_grid`, and
calls `finite_grid_search` once. It writes only these deterministic files to
the selected output directory (the generic runner also writes
`resolved_configuration.toml`):

- `candidate_results.csv`: one ranked row per unique z candidate, including
  gross value, probing cost, net objective, outer SE, and policy frequencies.
- `experiment_report.md`: configuration, model reminder, ranked
  candidates, best-candidate summary, and interpretation limits.

Candidate streams are the independent deterministic streams assigned by Stage
7. The reported point is therefore only the best estimated candidate on this
finite grid; the report makes no significance or continuous-optimum claim.

For fixed-z convergence, sensitivity, performance measurements, and the final
readiness assessment, see [FINAL_VALIDATION_REPORT.md](FINAL_VALIDATION_REPORT.md).
