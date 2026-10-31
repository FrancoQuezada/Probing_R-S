# Codex Context: Probing and Ranking-and-Selection Simulation Codebase

This document gives Codex the project context needed to understand the Julia code files before making any implementation changes.

**Important instruction for Codex:** do not modify any source code yet. This stage is read-only analysis. Produce explanations, dependency maps, theoretical alignment notes, and implementation-risk notes only.

## 1. Project purpose

This project studies a simulation-optimization problem combining probing / inspection decisions with ranking-and-selection style policy choice.

The motivating system is a two-stage service / production system with:

- a front-end resource (FE),
- a transfer stage,
- two alternative back-end resources (BE): a fast BE and a standard BE,
- stochastic job arrivals,
- heterogeneous job classes,
- class-dependent complexity probabilities,
- probing / inspection decisions that affect information quality and disruption risk,
- a finite grid of routing policies.

The high-level question is: how should the system allocate probing or inspection effort across job classes so that downstream routing decisions improve enough to justify the probing cost?

## 2. Theoretical foundation

The broader formulation is a probing-enhanced ranking-and-selection problem. A first-stage decision chooses what information to acquire. After observing the probed information, the decision maker selects the best policy / alternative from a finite set.

A stylized binary probing formulation is:

```text
max_z { - alpha' z + E_eta [ max_i E[Y(x_i, xi) | observed probes under z] ] }
```

where:

- `z` is the probing decision,
- `alpha` is the probing cost vector,
- `x_i` is a candidate policy / alternative,
- `Y(x_i, xi)` is the simulated performance of policy `x_i` under random primitives `xi`,
- the inner maximization over `i` represents ranking and selection conditional on the information produced by probing.

The project also explores a continuous probing version:

```text
z in [0,1]^m
```

where `z_j` is the inspection intensity for class `j`. Higher `z_j` means more frequent inspection and/or more reliable signal generation. In the code, continuous probing affects:

1. the probability that a job of class `j` is probed,
2. the signal hit/false-positive probabilities,
3. the probability of disruption at the standard BE for complex jobs.

A later algorithmic direction uses likelihood-ratio / score-function gradient estimation. The key estimator has the form:

```text
grad Q(z) approx (1/R) sum_r (Ymax_r - mean(Ymax)) * S_r(z)
```

where:

- `Ymax_r` is the best policy value on replication / sample path `r`,
- `S_r(z)` is the score, i.e., the gradient of the log-likelihood of the generated sample path with respect to `z`,
- the baseline `mean(Ymax)` is used for variance reduction.

This LR-gradient approach is implemented in `SimOpt009.jl`.

## 3. Source files in this codebase

Current relevant Julia files:

```text
SimOpt005.jl
SimOpt009.jl
SimOpt009_2D_clean.jl
```

Supporting project documents:

```text
Probing_R_S-1.pdf
Probing_R_S-2.pdf
```

The PDFs contain the theoretical notes and experimental framing. Codex should treat the Julia code as prototype / research code connected to those notes, not as production software.

## 4. File: SimOpt005.jl

### 4.1 Role

`SimOpt005.jl` is a complete standalone simulation and evaluation script for the two-stage probing model.

It implements:

- policy definition,
- policy-grid generation,
- stochastic signal generation,
- discrete-event simulation for one policy and one probing vector,
- two-stage evaluation of probing decisions,
- enumeration of binary probing vectors,
- a `run()` function that executes the experiment immediately.

### 4.2 Main functions

```julia
choose_fast_be(p::Policy, demand, fast_q, std_q, signal)
generate_policies()
generate_signal(rng, true_class, complex, z; ...)
simulate_replication(policy::Policy, z, rng; ...)
evaluate_Q_two_stage(z, policies; ...)
generate_all_binary_z(m)
run()
```

### 4.3 Policy model

A policy is a linear scoring rule:

```text
score = a_demand * demand - a_queue * (fast_q - std_q) + a_signal * signal
send_fast = score >= threshold
```

The policy grid contains:

```text
6 demand weights * 6 queue weights * 5 signal weights * 5 thresholds = 900 policies
```

### 4.4 Probing and signal model

The probing vector is `z`, with one component per class.

For a job of class `c`:

- it is inspected with probability `z[c]`,
- if not inspected, no signal is generated,
- if inspected and complex, the probability of a positive signal is `ph`,
- if inspected and simple, the probability of a positive signal is `pf`,
- higher probing increases `ph` and decreases `pf`.

The code uses:

```text
ph = p_hit[c] + (1 - p_hit[c]) * (1 - exp(-beta*z[c]))
pf = p_false[c] * exp(-beta*z[c])
```

The routing signal used by the policy is:

```text
if no signal: class baseline complexity probability
if signal exists: 0.5 * baseline + 0.5 * signal
```

### 4.5 Disruption model

If a complex job is routed to standard BE, it may suffer a disruption. Probing reduces disruption probability:

```text
p_disrupt = 1 - 0.75 * z[class]
```

If disruption occurs, an exponential delay is added.

### 4.6 Evaluation method

The key function is:

```julia
evaluate_Q_two_stage(z, policies; train_reps, eval_reps, seed, alpha)
```

It performs:

1. training stage: simulate every policy and select the best policy by sample mean,
2. evaluation stage: evaluate the selected policy on an independent sample,
3. return objective value, standard error, selected policy id, and training value.

This is a classical train/evaluate structure intended to reduce selection bias.

### 4.7 Outer experiment

The `run()` function enumerates all binary probing vectors:

```text
z in {0,1}^4
```

Although the simulator accepts continuous `z`, the actual `run()` experiment in `SimOpt005.jl` is binary enumeration.

### 4.8 Auto-run behavior

`SimOpt005.jl` ends with:

```julia
run()
```

Therefore, including or loading this file executes the experiment immediately.

## 5. File: SimOpt009.jl

### 5.1 Role

`SimOpt009.jl` is the continuous probing and likelihood-ratio gradient version.

It is not simply `SimOpt005.jl` plus an optimizer. It reorganizes the simulator around full sample paths so that every policy can be evaluated on the same random realization. This enables common random numbers across policies and supports a score-function gradient estimator.

### 5.2 Main functions

```julia
choose_fast_be(p::Policy, demand, fast_q, std_q, signal::Float64)
generate_policies()
signal_probs(true_class, z; ...)
generate_sample_path(z, rng; ...)
evaluate_policy_on_path(policy, path; ...)
estimate_Q_and_gradient(z, policies, seed; ...)
project!(z, eps=0.05)
optimize_z(policies; ...)
evaluate_Q(z, policies; ...)
run()
```

### 5.3 Sample-path design

Instead of simulating one policy directly, `SimOpt009.jl` first generates a full path:

```julia
generate_sample_path(z, rng; N=200, ...)
```

The generated path stores arrays for:

- interarrival times,
- demands,
- classes,
- complexity flags,
- FE processing times,
- transfer times,
- fast BE times,
- standard BE times,
- disruption delays,
- probing events,
- signal bits,
- disruption bits,
- signal values used by policies,
- likelihood-ratio score vector.

Then each policy is evaluated on that same path using:

```julia
evaluate_policy_on_path(policy, path)
```

### 5.4 Objective estimator

The function:

```julia
estimate_Q_and_gradient(z, policies, seed; reps, alpha, N)
```

for each replication:

1. generates a path under `z`,
2. evaluates all policies on that path,
3. records the best value over policies,
4. records the path score,
5. estimates the value and gradient.

The estimator uses:

```text
baseline = mean(max_vals)
grad = average_r (max_vals[r] - baseline) * score[r]
grad = grad - alpha
objective = baseline - dot(alpha, z)
```

### 5.5 Score-function terms

The likelihood-ratio score includes contributions from the `z`-dependent random mechanisms:

1. probing decision:

```text
p_probe = z[c]
score += 1/p_probe if probed, else -1/(1-p_probe)
```

2. signal hit probability for complex jobs:

```text
dph/dz = (1 - p_hit[c]) * beta * exp(-beta*z[c])
```

3. false-positive probability for simple jobs:

```text
dpf/dz = -p_false[c] * beta * exp(-beta*z[c])
```

4. disruption probability for complex jobs:

```text
p_disrupt = 1 - 0.75*z[c]
dp_disrupt/dz = -0.75
```

The code clamps probabilities away from 0 and 1 to avoid numerical explosion.

### 5.6 Optimization method

`optimize_z` performs projected stochastic gradient ascent:

```text
z <- z + step_size * gradient
z <- projection to [eps, 1-eps]^m
```

Default projection uses `eps=0.05`, so the default box is:

```text
z_j in [0.05, 0.95]
```

### 5.7 Auto-run behavior

The file currently ends with:

```julia
#run()
```

This is intentional. The file defines the 4D model and optimizer but does not execute automatically. This is important because `SimOpt009_2D_clean.jl` includes it.

## 6. File: SimOpt009_2D_clean.jl

### 6.1 Role

`SimOpt009_2D_clean.jl` is a two-dimensional wrapper around `SimOpt009.jl`.

It does not define a new simulation model. It imports `SimOpt009.jl` and optimizes only two components of `z`.

### 6.2 Main functions

```julia
make_full_z(z1, z2)
gradient_2d(z1, z2, iter)
optimize_z_2d(; max_iter, step_size, z0)
main()
```

### 6.3 2D reduction

The wrapper defines:

```julia
make_full_z(z1, z2) = [z1, 1.0, 1.0, z2]
```

So the effective optimization problem is:

```text
maximize Q(z1, 1.0, 1.0, z4)
```

Only components 1 and 4 are optimized. Components 2 and 3 are fixed at full probing.

The wrapper obtains the full 4D gradient from `estimate_Q_and_gradient` and keeps only:

```text
[g[1], g[4]]
```

### 6.4 Optimization settings

Current settings:

```text
GRAD_REPS = 300
mini-batch size = 3
max_iter = 300
step_size = 3e-5
z0 = [0.8, 0.8]
projection / clamp = [0.5, 0.95]
```

The lower bound is 0.5 in the 2D wrapper, not 0.05 as in `SimOpt009.jl`.

### 6.5 Auto-run behavior

`SimOpt009_2D_clean.jl` ends with:

```julia
main()
```

Therefore, running this file executes the 2D experiment immediately.

## 7. Critical conceptual difference between SimOpt005 and SimOpt009

Codex must not treat `SimOpt009.jl` as a direct drop-in continuation of `SimOpt005.jl`.

### SimOpt005 objective behavior

`SimOpt005.jl` is closer to:

```text
For each z:
    choose one policy using training simulations
    evaluate that selected policy independently
```

The selected policy is fixed during evaluation.

### SimOpt009 objective behavior

`SimOpt009.jl` is closer to:

```text
For each generated path:
    evaluate every policy on the same path
    take the pathwise maximum value
Estimate E[max_i Y_i(path; z)] and its LR gradient
```

This means the maximizing policy can vary by sample path. This is a different estimator structure.

This distinction should be preserved in any documentation, refactoring plan, or future manuscript description.

## 8. Known implementation notes and caveats

1. `SimOpt005.jl` has a header comment saying `SimOpt004_twostage_fixed.jl`, but the file is currently named `SimOpt005.jl`. This may be historical naming drift.

2. `SimOpt005.jl` auto-runs with `run()`. This is convenient for standalone experiments but inconvenient for reuse by other scripts.

3. `SimOpt009.jl` correctly keeps `run()` commented out so it can be included by the 2D wrapper.

4. `SimOpt009_2D_clean.jl` defines `const SEED = 1234`, but that constant is currently unused.

5. `SimOpt009_2D_clean.jl` uses a lower bound of 0.5 for the optimized dimensions, whereas `SimOpt009.jl` uses `eps=0.05` by default. This should be understood as an experimental restriction, not a general model constraint.

6. `SimOpt009.jl` generates disruption events for all complex jobs at sample-path generation time, but a disruption affects performance only if a policy routes that job to the standard BE.

7. The computational burden is high because `estimate_Q_and_gradient` evaluates all policies for every sample path. With 900 policies, `N=200`, and many replications, experiments can be expensive.

8. The current code uses simple arrays and named tuples. No module structure is used yet.

9. The code is research-prototype style and prioritizes conceptual clarity over software architecture.

## 9. What Codex should do in the current stage

In the current stage, Codex should only:

- read and understand the three Julia files,
- produce a function dependency map,
- explain the simulation flow,
- explain the difference between two-stage evaluation and LR-gradient optimization,
- identify assumptions and caveats,
- identify possible future refactoring opportunities without applying them,
- identify potential numerical or statistical issues without changing code,
- preserve the distinction between the theoretical R&S objective and the pathwise max estimator in `SimOpt009.jl`.

## 10. What Codex should not do yet

Codex should not:

- edit any `.jl` file,
- rename files,
- change auto-run behavior,
- change seeds,
- change objective definitions,
- change gradient formulas,
- add modules,
- add package environments,
- optimize performance,
- rewrite the simulator,
- introduce new mathematical assumptions.

Any proposed change should be written as a recommendation only.

## 11. Suggested read order for Codex

1. Read `Probing_R_S-1.pdf` and `Probing_R_S-2.pdf` only for conceptual context.
2. Read `SimOpt005.jl` to understand the two-stage baseline.
3. Read `SimOpt009.jl` to understand the continuous LR-gradient version.
4. Read `SimOpt009_2D_clean.jl` to understand the reduced 2D experiment.
5. Produce a read-only technical report.

