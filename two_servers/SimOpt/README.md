# Probing and Ranking-and-Selection Julia Experiments

This repository contains Julia research prototypes for simulation optimization
with probing, inspection effort, and ranking-and-selection style policy choice.

The model represents a two-stage service or production system with a front end,
a transfer stage, and two back-end routing options: fast and standard. Jobs have
stochastic arrivals, demands, classes, and complexity. Probing affects the
information available to routing policies and can reduce disruption risk.

## Files

- `SimOpt005.jl`: reference baseline for the two-stage train/evaluate
  experiment.
- `SimOpt009.jl`: continuous probing experiment with a likelihood-ratio
  gradient estimator for a pathwise-max objective.
- `SimOpt009_2D_clean.jl`: 2D wrapper around `SimOpt009.jl`.
- `CODEX_CONTEXT_Probing_RS.md`: detailed project notes for analysis.
- `CONTEX.md`: concise conceptual context.
- `AGENTS.md`: guidance for coding agents working in this repository.

## Main Conceptual Distinction

`SimOpt005.jl` is the reference baseline.

It enumerates binary probing vectors:

```text
z in {0,1}^4
```

For each `z`, it selects a policy using training replications and estimates the
value of that selected policy using independent evaluation replications.

`SimOpt009.jl` should not be interpreted as a direct continuous extension of
that same estimator. It estimates a different structure:

```text
For each sample path:
    evaluate every policy on the same path
    take the best realized policy value on that path
Estimate a likelihood-ratio gradient of that pathwise-max objective
```

The maximizing policy in `SimOpt009.jl` can vary across sample paths. This is
not the same as selecting one policy in a training stage and evaluating it
independently.

## Requirements

The Julia scripts use:

```julia
Random
Distributions
Statistics
```

`Random` and `Statistics` are standard libraries. `Distributions` must be
available in the active Julia environment.

## Running Experiments

Run the two-stage binary baseline:

```bash
julia SimOpt005.jl
```

Important: `SimOpt005.jl` ends with `run()`, so including it also executes the
experiment.

Run the 4D continuous LR experiment by enabling the commented call in
`SimOpt009.jl`, or by calling `run()` after including it:

```julia
include("SimOpt009.jl")
run()
```

`SimOpt009.jl` is include-safe because it ends with `#run()`.

Run the 2D wrapper:

```bash
julia SimOpt009_2D_clean.jl
```

Important: `SimOpt009_2D_clean.jl` ends with `main()`, so including it also
executes the 2D experiment.

## Policy Grid

Policies are linear threshold rules:

```text
score = a_demand * demand
      - a_queue * (fast_queue - standard_queue)
      + a_signal * signal
```

A job is routed to the fast back end if:

```text
score >= threshold
```

The current grid has:

```text
6 * 6 * 5 * 5 = 900 policies
```

## Probing Effects

For class `c`, probing intensity `z[c]` affects:

- inspection probability;
- signal hit and false-positive probabilities;
- disruption probability for complex jobs.

Signal quality:

```text
ph = p_hit[c] + (1 - p_hit[c]) * (1 - exp(-beta * z[c]))
pf = p_false[c] * exp(-beta * z[c])
```

Disruption probability for complex jobs:

```text
p_disrupt = 1 - 0.75 * z[c]
```

## Notes For Future Work

Before changing the code, decide which objective is being studied:

- selected-policy value from the two-stage `SimOpt005.jl` procedure;
- pathwise-max LR objective from `SimOpt009.jl`;
- theoretical conditional-expectation R&S objective from the research notes.

These objectives are related but not identical.

