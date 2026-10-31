# AGENTS.md

## Repository Guidance For Coding Agents

This repository contains research-prototype Julia code for probing, simulation
optimization, and ranking-and-selection experiments. Treat the code as
experimental research software, not production infrastructure.

## Most Important Conceptual Rule

`SimOpt005.jl` is the reference baseline for the two-stage train/evaluate
experiment.

It enumerates binary probing vectors `z in {0,1}^4`, selects a policy using
training replications, and estimates the value of that selected policy using
independent evaluation replications.

`SimOpt009.jl` must not be interpreted as a direct continuous extension of the
same estimator. It changes the estimator structure:

1. Generate one full sample path under continuous `z`.
2. Evaluate every policy on that same path.
3. Take the best realized policy value on that path.
4. Estimate a score-function / likelihood-ratio gradient of this pathwise-max
   objective.

This distinction must be preserved in documentation, comments, refactors, and
future experiments.

## File Roles

- `SimOpt005.jl`: reference two-stage train/evaluate baseline. Auto-runs with
  `run()` at the end.
- `SimOpt009.jl`: continuous probing model with likelihood-ratio gradient for a
  pathwise-max objective. Include-safe because `run()` is commented out.
- `SimOpt009_2D_clean.jl`: two-dimensional wrapper around `SimOpt009.jl`.
  Fixes `z[2] = 1.0` and `z[3] = 1.0`, optimizes dimensions 1 and 4, and
  auto-runs with `main()`.
- `CODEX_CONTEXT_Probing_RS.md`: longer project context and analysis notes.
- `CONTEX.md`: concise project context for humans and agents.
- `README.md`: user-facing repository overview.

## Auto-Run Behavior

Be careful when including Julia files:

- `SimOpt005.jl` ends with `run()`, so `include("SimOpt005.jl")` executes the
  full binary two-stage experiment.
- `SimOpt009.jl` ends with `#run()`, so it is safe to include as a library of
  definitions.
- `SimOpt009_2D_clean.jl` ends with `main()`, so running or including it
  executes the 2D SGD experiment.

## Development Rules

Before changing code, identify which objective is being targeted:

- two-stage selected-policy value from `SimOpt005.jl`;
- pathwise-max LR objective from `SimOpt009.jl`;
- theoretical conditional-expectation R&S objective from the project notes.

Do not silently replace one objective with another.

Avoid broad refactors unless they are explicitly requested. If refactoring is
needed, preserve experiment settings, random-seed logic, auto-run behavior, and
objective definitions unless the researcher asks to change them.

## Numerical And Statistical Cautions

- Probability clamps in `SimOpt009.jl` are numerical safeguards for LR scores.
- The LR gradient estimator can have high variance.
- The pathwise maximum over policies is not the same as independent
  train/evaluate selection.
- The 2D wrapper's clamp `[0.5, 0.95]` is an experimental restriction, not a
  general model constraint.
- `SimOpt009_2D_clean.jl` defines `SEED`, but the current code does not appear
  to use it.

## Preferred Analysis Workflow

1. Read `README.md`.
2. Read `CONTEX.md`.
3. Inspect `SimOpt005.jl` for the baseline two-stage estimator.
4. Inspect `SimOpt009.jl` for the continuous LR pathwise-max estimator.
5. Inspect `SimOpt009_2D_clean.jl` only after understanding `SimOpt009.jl`.

