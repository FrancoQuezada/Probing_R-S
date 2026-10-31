# CONTEX.md

## Project Context

This project studies a simulation-optimization problem involving probing,
inspection effort, and ranking-and-selection style policy choice.

The motivating system is a two-stage service or production model with:

- a front-end resource;
- a transfer stage;
- a fast back-end resource;
- a standard back-end resource;
- stochastic arrivals and demands;
- heterogeneous job classes;
- class-dependent complexity probabilities;
- probing decisions that affect information quality and disruption risk;
- a finite grid of routing policies.

The main research question is how to allocate probing or inspection effort
across job classes so that improved routing decisions justify probing cost.

## Policy Representation

Policies are linear threshold rules. A policy has:

- `a_demand`;
- `a_queue`;
- `a_signal`;
- `threshold`.

At routing time, the policy computes:

```text
score = a_demand * demand
      - a_queue * (fast_queue - standard_queue)
      + a_signal * signal
```

The job is sent to the fast back end when:

```text
score >= threshold
```

The policy grid contains:

```text
6 demand weights * 6 queue weights * 5 signal weights * 5 thresholds = 900 policies
```

## Reference Baseline: SimOpt005.jl

`SimOpt005.jl` is the reference baseline for the two-stage train/evaluate
experiment.

It evaluates binary probing vectors:

```text
z in {0,1}^4
```

For each probing vector:

1. Simulate all policies on training replications.
2. Select the policy with the best training mean.
3. Evaluate that selected policy on independent evaluation replications.
4. Report the independent evaluation value, standard error, and selected policy.

This structure is important because the final value estimates the performance
of one selected policy, not the pathwise maximum over policies.

`SimOpt005.jl` ends with:

```julia
run()
```

Therefore, including the file executes the experiment immediately.

## Continuous LR Experiment: SimOpt009.jl

`SimOpt009.jl` is not a direct continuous extension of the `SimOpt005.jl`
estimator.

It changes the estimator structure. For each replication:

1. Generate a full sample path under continuous `z`.
2. Evaluate every policy on the same path.
3. Take the best realized policy value on that path.
4. Use the sample-path likelihood-ratio score to estimate a gradient.

The estimated objective is closer to:

```text
Q(z) = E_path [ max_i Y_i(path; z) ]
```

with optional probing cost:

```text
Q(z) - alpha' z
```

This pathwise-max objective differs from the two-stage selected-policy
objective in `SimOpt005.jl`. The maximizing policy in `SimOpt009.jl` can vary
from path to path.

`SimOpt009.jl` ends with:

```julia
#run()
```

So it is safe to include.

## How Probing Affects The Model

For a job of class `c`, probing intensity `z[c]` affects:

1. The probability that the job is probed.
2. The signal quality if it is probed.
3. The disruption probability for complex jobs.

Signal probabilities use:

```text
ph = p_hit[c] + (1 - p_hit[c]) * (1 - exp(-beta * z[c]))
pf = p_false[c] * exp(-beta * z[c])
```

Higher `z[c]` increases the hit probability for complex jobs and decreases the
false-positive probability for simple jobs.

For complex jobs, disruption probability is:

```text
p_disrupt = 1 - 0.75 * z[c]
```

In `SimOpt009.jl`, probability values are clamped away from 0 and 1 to avoid
unstable likelihood-ratio scores.

## 2D Wrapper: SimOpt009_2D_clean.jl

`SimOpt009_2D_clean.jl` includes `SimOpt009.jl` and optimizes a reduced
two-dimensional problem:

```text
make_full_z(z1, z2) = [z1, 1.0, 1.0, z2]
```

Thus:

- class 2 probing is fixed at `1.0`;
- class 3 probing is fixed at `1.0`;
- only dimensions 1 and 4 are optimized;
- the full 4D gradient is computed, but only components `[g[1], g[4]]` are
  used.

The file defines `SEED = 1234`, but the current implementation does not appear
to use that constant.

`SimOpt009_2D_clean.jl` ends with:

```julia
main()
```

So it auto-runs.

## Caveats To Preserve

- Do not conflate the two-stage selected-policy estimator with the pathwise-max
  estimator.
- Do not describe `SimOpt009.jl` as merely the continuous version of
  `SimOpt005.jl`.
- The pathwise maximum can be optimistic relative to a train/evaluate selected
  policy procedure.
- LR gradients may have high variance.
- Boundary probabilities need safeguards because LR scores involve terms like
  `1/p` and `1/(1-p)`.
- The current code is research-prototype code and intentionally prioritizes
  experimental clarity over architecture.

