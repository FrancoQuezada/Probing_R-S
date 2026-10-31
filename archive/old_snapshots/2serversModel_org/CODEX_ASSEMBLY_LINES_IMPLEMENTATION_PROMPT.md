# Prompt for Codex — Assembly-Line Example, Staged Implementation

You are implementing a new Julia research example for probing-enhanced stochastic optimization.

The repository currently contains a working but conceptually different two-server example with job-level probing. The new example concerns multiple assembly lines, pre-shift information acquisition, Bayesian updating, and selection of one routing policy for an entire rework shift.

## Authoritative specification

Read and follow:

- `ASSEMBLY_LINES_IMPLEMENTATION_SPEC.md`
- `Probing_R_S-2.pdf`, especially the assembly-line example, only as supporting context
- the existing two-server Julia code, only to identify reusable queueing and policy-evaluation logic

When the PDF, comments, and implementation specification differ, treat `ASSEMBLY_LINES_IMPLEMENTATION_SPEC.md` as the authoritative source for this task.

## Scope restriction

Do not implement likelihood ratios, gradients, green simulation, or continuous optimization.

Do not add job-level signals, `p_hit`, `p_false`, or an `a_s` policy coefficient.

Do not modify the existing two-server model unless a change is strictly necessary for repository setup and is explicitly documented. Prefer creating a separate module or directory.

The uncertainty vector for the new example is

\[
\xi=(\Theta,\mathcal A,\mathcal D,\mathcal B).
\]

Product values are deterministic parameters \(v_c\) attached to assembly lines. They are not random variables.

The decision sequence is

\[
\Theta \text{ realized but hidden}
\rightarrow z
\rightarrow \eta
\rightarrow i^\star(\eta,z)
\rightarrow (\mathcal A,\mathcal D,\mathcal B)
\rightarrow R.
\]

One routing policy is selected after observing \(\eta\), and that policy remains fixed throughout the shift.

## Development protocol

Work in phases. Do not implement the entire model in one pass.

For every phase:

1. inspect the relevant existing code;
2. state the files and interfaces you plan to add or change;
3. implement only the current phase;
4. add focused tests;
5. run the smallest relevant test set;
6. report results, assumptions, and unresolved issues;
7. stop if any test fails or if the specification is ambiguous.

Do not run large experiments.

Store temporary outputs outside the source tree or in a dedicated `audit_outputs/` or `tmp/` directory.

## This run: implement only Stages 0–2

Complete only the following stages in this run, then stop and present a reviewable report.

### Stage 0 — Repository protection and implementation plan

Tasks:

- inspect the repository structure;
- identify the existing two-server source files and leave them unchanged;
- locate `ASSEMBLY_LINES_IMPLEMENTATION_SPEC.md`;
- create a separate assembly-line implementation area;
- determine whether a `Project.toml` and test setup already exist;
- add the minimum reproducible Julia project/test structure if necessary;
- write a short architecture note describing planned files and dependencies.

Preferred structure:

```text
assembly_line_model/
├── AssemblyLineModel.jl
├── types.jl
├── parameters.jl
├── bayesian_update.jl
├── predictive_simulation.jl
├── routing_simulation.jl
├── nested_monte_carlo.jl
├── first_stage_search.jl
├── experiments/
│   └── run_small_example.jl
└── test/
    └── runtests.jl
```

You may use a simpler structure if justified, but preserve separation between:

- types and parameters;
- Bayesian update;
- predictive simulation;
- routing simulation;
- nested Monte Carlo;
- first-stage search.

Stage-0 acceptance criteria:

- existing Julia source files remain unchanged;
- the new module can be loaded;
- the test runner starts successfully;
- dependencies are documented.

### Stage 1 — Types and parameter validation

Implement immutable Julia types for at least:

- `LineParameters`;
- `ModelParameters`;
- `DiagnosticObservation`;
- `PosteriorState`;
- `ReworkJob`;
- `ShiftPath`;
- `Policy`;
- `PolicySelectionResult`;
- `ZEvaluationResult`.

Required semantics:

- line value \(v_c\) belongs to `LineParameters`;
- `ReworkJob` stores line origin but not a random value and not a signal;
- service-time parameters can depend on line, complexity, and server;
- dimensions across all line-specific vectors must agree;
- all variances, rates, horizons, and service means must be validated;
- \(z\) must eventually be compatible with \([0,1]^m\).

Add tests for:

- valid parameter construction;
- invalid dimension combinations;
- negative or zero variances where prohibited;
- invalid product values;
- invalid horizon;
- confirmation that `ReworkJob` has no signal field;
- confirmation that value is obtained from the line parameterization.

Stage-1 acceptance criteria:

- all tests pass;
- error messages are informative;
- no model logic is hidden in constructors beyond validation and simple normalization.

### Stage 2 — Bayesian information layer

Implement:

```julia
sample_prior_theta(rng, params)
sample_diagnostics(rng, theta, z, params)
posterior_parameters(diagnostics, z, params)
```

Model:

\[
\Theta_c\sim\mathcal N(\mu_c,\tau_c^2),
\]

and for \(z_c>0\),

\[
\eta_c\mid\Theta_c,z_c
\sim
\mathcal N\left(
\Theta_c,
\frac{\nu_c}{T_c^{\max}z_c}
\right).
\]

Posterior:

\[
V_c(z_c)
=
\left(
\frac{1}{\tau_c^2}
+
\frac{T_c^{\max}z_c}{\nu_c}
\right)^{-1},
\]

\[
m_c(\eta_c,z_c)
=
V_c(z_c)
\left(
\frac{\mu_c}{\tau_c^2}
+
\frac{T_c^{\max}z_c}{\nu_c}\eta_c
\right).
\]

For \(z_c=0\):

- no measurement is generated;
- posterior mean equals \(\mu_c\);
- posterior variance equals \(\tau_c^2\);
- no division by zero is allowed.

Represent missing measurements explicitly. Do not invent a fake numerical signal for \(z_c=0\).

Add tests for:

1. prior sampling dimensions;
2. reproducibility under fixed seeds;
3. posterior equals prior at \(z_c=0\);
4. analytical posterior mean and variance at a hand-checkable point;
5. posterior variance decreases as \(z_c\) increases;
6. diagnostic variance decreases as \(z_c\) increases;
7. mixed vectors where some coordinates have \(z_c=0\) and others have \(z_c>0\);
8. rejection of \(z_c<0\) or \(z_c>1\);
9. no leakage of the true latent \(\Theta\) into `PosteriorState` beyond what is mathematically required for test fixtures.

Stage-2 acceptance criteria:

- all tests pass;
- formulas match the specification exactly;
- the API is ready for posterior predictive simulation;
- the implementation does not contain arrival, complexity, routing, likelihood-ratio, or gradient logic yet.

## Required report after Stage 2

After completing Stages 0–2, stop and provide:

1. files created or modified;
2. confirmation that the existing two-server files were not changed;
3. implemented type definitions and public interfaces;
4. exact validation rules;
5. test results;
6. any ambiguity found in the specification;
7. a proposed plan for Stage 3 only, without implementing it.

Do not proceed to posterior predictive arrivals, routing, nested Monte Carlo, or first-stage search in this run.
