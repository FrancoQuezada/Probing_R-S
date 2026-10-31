# Assembly-line model architecture

This directory is an independent Julia package.  It deliberately does not
import or modify the legacy two-server scripts (now under `two_servers/`).

## Implemented in Stages 0–10

- `src/types.jl`: immutable domain and result types with validation-only
  constructors.
- `src/parameters.jl`: parameter accessors and validation of a probing vector.
- `src/bayesian_update.jl`: prior sampling, diagnostics, and the exact
  coordinatewise normal-normal posterior update.
- `src/predictive_simulation.jl`: posterior-predictive latent conditions,
  per-line Poisson arrivals, complexity draws, potential service times, and
  globally ordered shift paths. Arrival ties are ordered by line index and
  then stable generation order. It has no routing or policy logic.
- `src/routing_simulation.jl`: deterministic fixed-policy routing over two
  FCFS servers, isolated reward calculation, and common-path policy reward
  evaluation. It performs no policy selection or stochastic sampling.
- `src/conditional_policy_selection.jl`: common-bank conditional policy-value
  estimation, deterministic first-maximum selection, and independent
  posterior-predictive evaluation of the selected policy. It does not perform
  outer diagnostic averaging or first-stage evaluation.
- `src/nested_monte_carlo.jl`: outer prior/diagnostic/posterior replications,
  Stage-5 conditional selection and independent evaluation, deterministic
  probing-cost subtraction, and outer Monte Carlo summaries for one fixed z.
- `src/finite_grid_search.jl`: deterministic evaluation of one finite,
  user-supplied candidate set (or a bounded Cartesian grid) by delegating each
  unique z to `evaluate_z`. Exact duplicate Float64 candidates are compressed
  with an original-position mapping. Candidate streams are independently
  derived from a master seed and z bits; ranking uses net objective followed by
  documented deterministic tie-breaks. This layer performs no refinement,
  independent candidate streams by default; optional CRN uses indexed base
  streams and supports paired comparisons without changing the estimator.
- `src/crn_sampling.jl`: domain-separated deterministic substreams for CRN.
  The key is master seed plus outer/phase/path/line/job indices, never z or
  candidate identity. Shared job prefixes may diverge in count or queue order.
- `src/policy_family.jl`: pure construction of finite `Policy(av, aq, tau)`
  families from caller-provided coefficient vectors. Exact duplicates are
  removed independently while preserving first occurrence; Cartesian order is
  `av` slowest, `aq` next, and `tau` fastest. The bounded builder performs no
  simulation, policy-quality assessment, or coefficient optimization.
- `examples/reference_experiment.jl`: an opt-in synthetic reference experiment
  which composes the public Stage-7 and Stage-8 builders/search APIs, then
  emits deterministic ranked CSV and Markdown summaries. It does not alter the
  estimator or implement an additional search method.
- `examples/experiment_runner.jl` and `examples/run_grid_experiment.jl`: a
  TOML-driven reusable runner and CLI which validate resolved inputs, delegate
  to Stage 7, and write deterministic CSV/Markdown/resolved-TOML outputs. The
  reference wrapper uses this same layer.
- `examples/convergence_validation.jl`, `examples/sensitivity_validation.jl`,
  and `examples/performance_validation.jl`: opt-in fixed-z convergence,
  structural sensitivity, and warmed serial-performance validation. They do not
  modify the estimator or add an optimization method.
- `FINAL_VALIDATION_REPORT.md`: final scientific audit, measured validation
  outcomes, limitations, and conservative run-size guidance.
- `test/runtests.jl`: focused deterministic tests for Stages 0–10.

The package uses only Julia's `Random` standard library.  Run its tests from
the `production_rework/` directory with:

```bash
julia --project=assembly_line_model assembly_line_model/test/runtests.jl
```

## Planned, intentionally unimplemented search features

Adaptive/local refinement, coordinate or pattern search, likelihood-ratio or
gradient methods, green simulation, Bayesian optimization, and parallel search
remain intentionally unimplemented. The finite-grid result is
only the best estimated point in the supplied finite set; its outer standard
errors are not pairwise-comparison standard errors and ranking does not make a
statistical-significance claim.
