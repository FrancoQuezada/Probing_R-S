# Probing and routing with likelihood-ratio simulation

> Location note: this project lives at `two_servers/lr_baseline_2d/` (formerly `Assemblylines/`). Run the commands below from this directory. `single_job_two_stage*.jl` is the job-level two-stage example; the assembly-line prototype that used to share this folder is archived in `archive/rework_prototypes/`.

This repository is a small Julia simulation study of whether it is worthwhile to
acquire better job information before routing work to one of two servers.  It
implements the two-server example described in
[`Probing_R_S-1.pdf`](Probing_R_S-1.pdf).

The study evaluates a two-dimensional probing decision with likelihood-ratio
(LR) / importance-sampling reweighting.  It then selects the best policy from a
finite family of threshold routing rules and uses projected-gradient ascent to
find a promising probing allocation.

## Model at a glance

Jobs arrive to a fast server and a standard server.  The fast server has lower
service times, especially for complex jobs, but charges an extra routing cost.
Each job has a value, one of four classes, an unobserved complexity state, and a
binary probing signal.  The signal informs the routing decision but is imperfect.

A routing policy has the form

```text
score = a_v * value - a_q * (fast_workload - standard_workload) + a_s * signal
route to fast server when score >= tau
```

For class `c`, probing effort `z_c` improves the signal as follows:

```text
p_hit,c(z_c)   = p_hit,c(0) + (1 - p_hit,c(0)) * (1 - exp(-beta * z_c))
p_false,c(z_c) = p_false,c(0) * exp(-beta * z_c)
```

The scripts optimize only `z = (z_3, z_4)`, while `z_1 = z_2 = 0`.  The net
objective is the best estimated gross routing reward minus linear probing cost:

```text
J(z) = max_i mean(Y_i * L(z)) - alpha' * z
```

where `i` indexes a finite grid of 144 threshold policies and `L(z)` is the
likelihood ratio between the candidate signal distribution and the baseline
(`z = 0`) signal distribution.

## Repository contents

| File | Purpose |
| --- | --- |
| `Probing_R_S-1.pdf` | Model notes, likelihood-ratio derivation, and research discussion. |
| `probing_heatmap_2d.jl` | Main experiment: diagnostics, gradient check, projected-gradient search, objective heatmap, and convergence charts. |
| `probing_policy_regions_2d.jl` | The main experiment plus a map of the policy that is optimal at each probing-grid point. |

The two Julia scripts duplicate the shared model and differ slightly in their
calibration: `ALPHA[1]` is `6.1` in `probing_heatmap_2d.jl` and `6.0` in
`probing_policy_regions_2d.jl`.  Use the latter when policy regions are needed;
do not compare outputs from the two scripts as though they used identical costs.

## Requirements and quick start

Julia 1.11 or newer is recommended.  The only non-standard package is `Plots`.

```bash
julia -e 'using Pkg; Pkg.add("Plots")'
julia probing_heatmap_2d.jl
```

To also produce the optimal-policy region map:

```bash
julia probing_policy_regions_2d.jl
```

Both scripts use `N_PATHS = 5000`, `JOBS_PER_PATH = 45`, a 41 x 41 probing
grid, and a fixed random seed.  A full run precomputes rewards for all 144
policies and can take noticeable time; lower `N_PATHS` and `GRID_N` are useful
for a quick smoke run.  The seed makes results reproducible for a fixed Julia
and package environment.

## Outputs

The scripts write PNG files in the directory from which Julia is launched:

| Output | Produced by |
| --- | --- |
| `probing_heatmap_alpha3_<...>_alpha4_<...>_beta_<...>.png` | Both scripts; contour plot of the LR objective and search trajectory. |
| `probing_convergence.png` | Both scripts; evolution of `z_3` and `z_4`. |
| `probing_objective_convergence.png` | Both scripts; objective by iteration. |
| `probing_policy_regions_alpha3_<...>_alpha4_<...>_beta_<...>.png` | `probing_policy_regions_2d.jl` only; discrete optimal-policy regions. |

The console also reports the baseline result, selected policy indices, an
analytic-versus-finite-difference gradient check, LR mean/standard deviation,
and the best plotted-grid point.

## How the estimator works

1. Generate a bank of complete job-day paths under baseline probing.  A path
   contains arrivals, class, complexity, value, service-time draws for both
   servers, and its baseline binary signal.
2. Simulate every candidate policy once on each fixed path and store its reward
   `Y(policy, path)`.
3. For any proposed `(z_3, z_4)`, reweight each path by the product of the
   Bernoulli signal probability ratios.  No job primitive other than the signal
   distribution changes with `z`.
4. Choose the policy with the largest LR-weighted sample mean.  The gradient
   for that active policy is `mean(Y * L * grad(log L)) - alpha`.
5. Apply projected gradient ascent over `[0, 1]^2`, then evaluate the same
   empirical objective over a grid for visualization.

This matches the PDF's production-routing interpretation: a policy is chosen
before the job signals are observed, so the maximization is over policies and is
outside the sample expectation.

## Parameters to tune

All experiment controls are constants near the top of each script:

- `P_COMPLEX`, `P_HIT0`, `P_FALSE0`, and `BETA` define classes and signal quality.
- `FAST_MEAN`, `STD_MEAN`, `MEAN_INTERARRIVAL`, and the economic constants define
  the queueing and reward model.
- `ALPHA` sets the marginal probing costs for classes 3 and 4.
- `N_PATHS`, `JOBS_PER_PATH`, `GRID_N`, and `SEED` control simulation accuracy,
  runtime, and reproducibility.
- `Z0`, `MAX_ITERS`, `BASE_STEP`, and `GRAD_CLIP` control the search.

## Audit notes and limitations

- The Julia sources parse successfully with Julia 1.11.7.  The repository now
  has a Julia project environment and package-manager-generated manifest.
- LR reweighting is exact in expectation for the stated model, because the
  probing decision changes only the conditional signal distribution.  It can
  nevertheless have high variance as efforts move away from zero or paths get
  longer.  Monitor the printed `E[L]` and `SD[L]` diagnostics.
- `log L` is clipped to `[-35, 35]` before exponentiation.  This prevents
  numerical overflow but introduces bias if clipping occurs; it should be
  reported or replaced with a numerically stable estimator for larger studies.
- The objective is non-smooth at policy-switch boundaries.  The finite-difference
  check is intentionally run away from such a boundary, but projected-gradient
  results remain local estimates on a finite Monte Carlo surrogate.
- The implementation treats class as generated but not as an input to the
  routing score.  This is consistent with the current script, while the PDF
  explicitly identifies observing class in the policy as a possible extension.
- The policy grid is a deliberately restricted family, not a globally optimized
  routing rule.  “Optimal” in output names means optimal within that grid and
  for the finite sample bank.

## Suggested next steps

For a more durable research codebase, factor shared functions into a Julia
module, add a `Project.toml`, write small deterministic tests for the LR and
gradient, and save experiment parameters/results alongside each figure.

## Simplest original two-stage formulation

[`single_job_two_stage.jl`](single_job_two_stage.jl) implements a separate,
direct simulation of the original single-job theoretical setting.  It is not a
variation of the threshold-policy grid in the scripts above.

The decision sequence is:

```text
choose probing effort z
-> observe context H = (class, value, fast workload, standard workload)
-> observe one binary signal S
-> choose FAST or STANDARD
-> observe hidden complexity and service-time uncertainty
-> receive reward
```

Here `eta` is the one binary signal `S`, and the second-stage variable `x` is
the direct action `a ∈ {FAST, STANDARD}`.  The implemented objective is:

```text
J(z)
  =
  - alpha' z
  + E_H [
      E_{S | H,z} [
          max_{a in {FAST, STANDARD}}
          E[Y(a, Omega) | H,S,z]
      ]
    ]
```

The maximum is deliberately taken after observing `S`.  For each context the
implementation enumerates `S=0` and `S=1` exactly, applies Bayes' rule to
integrate hidden complexity, and computes expected service time analytically
before comparing the two action values.  It therefore estimates the binary
signal expectation without sampling the signal in the primary estimator.

The complementary realized-reward simulation samples complexity, signal, and
the service time only for validation.  It chooses the action before service time
is generated and never provides the realized complexity to the action rule.  The
module also supports common random numbers for reproducible comparisons among
probing vectors; this first implementation uses direct simulation, not
likelihood-ratio reweighting.

The single-job context distribution is intentionally simpler than the full-day
queue model: classes are uniform, values use the existing mild class effect, and
the two observed workloads are independent exponential draws with configurable
means.  These are modelling assumptions specific to this experiment.

Run the tests with:

```bash
julia --project=. test/runtests.jl
```

Run the two-dimensional common-random-number grid experiment with:

```bash
julia --project=. single_job_two_stage_2d.jl
```

It fixes `z1=z2=0`, searches `z3,z4 ∈ [0,1]`, and accepts these environment
variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GRID_N` | `41` | Grid points per probing dimension. |
| `REPS` | `50000` | Context replications in the shared noise set. |
| `SEED` | `20260720` | Shared-noise random seed. |
| `ALPHA` | `0.0,0.0,0.0,0.0` | Four comma-separated probing-cost coefficients; a neutral, uncalibrated default. |
| `OUTPUT_DIR` | `results` | Directory for figures and the grid CSV. |

For example:

```bash
GRID_N=21 REPS=10000 ALPHA=0,0,0.3,0.4 OUTPUT_DIR=results/example \
  julia --project=. single_job_two_stage_2d.jl
```

The driver writes a grid CSV plus heatmaps for net objective, gross value,
unconditional FAST-routing probability, and mean value of information.  Each
filename contains alpha, replication count, and seed.

The future batch-policy-selection formulation remains separate and is not yet
implemented by this module:

```text
E_eta [ max_i E[Y(x_i, xi) | eta, z] ]
```

That expression selects among complete candidate policies.  In contrast, this
single-job experiment is a direct simulation implementation of the optimal
post-signal routing action between FAST and STANDARD; it does not select an
optimal threshold policy or use a pathwise policy maximum.

## Assembly-line diagnosis and classical two-stage policy selection

[`assembly_line_two_stage.jl`](assembly_line_two_stage.jl) models a centralized
rework center serving four assembly lines.  Units that fail quality control enter
the dynamically evolving rework queue, where one fixed routing policy dispatches
each job to FAST (expert, costly) or STANDARD (regular, slower) repair.

Before a shift, each line has a latent operating condition `Theta_c` representing
wear, misalignment, contamination, or similar process degradation.  The
decision `z_c ∈ [0,1]` is normalized diagnostic time: its actual shutdown time
is `T_max[c] * z_c`.  The cost `alpha' * z` represents lost production margin
plus diagnostic-resource cost during that line downtime.

The diagnostic result is accumulated sensor evidence:

```text
eta_c = t_c * Theta_c + sigma_sensor[c] * sqrt(t_c) * epsilon_c
```

with an exact Gaussian posterior.  At zero effort, `eta_c=0` and the posterior
is exactly the prior.  Condition affects only each line's rework-job complexity
probability through a probit model.  The intercept is calibrated so that the
original marginal complexity rates `[0.02, 0.12, 0.55, 0.95]` are preserved.
Version 1 does not let condition change rework volume, arrivals, values, service
distributions, job-level quick-signal quality, or disruption probability.

The timing and objective are classical two-stage policy selection:

```text
choose z -> observe eta -> select one policy -> run the complete operating shift

J(z) = -alpha' z + E_{eta|z}[ max_i E[Y(x_i, Xi) | eta, z] ]
```

The selected policy is one of the existing 144 rules
`(a_value, a_queue, a_signal, threshold)` and remains fixed throughout its
shift.  It generates many ordinary FAST/STANDARD routing actions, but those are
not further policy-selection stages.  Conditional policy ranking uses training
paths; the selected policy alone is evaluated on independent paths.  A separate
unconditional fixed-policy baseline estimates `max_i E[Y(x_i,Xi)]`, and the
reported information value is adaptive gross value minus that baseline.

This differs from the existing ex-ante/LR policy-grid objective, the pathwise
maximum `E_Xi[max_i Y(x_i,Xi;z)]`, the single-job action model
`E_H E_S[max_a E[Y(a,Omega)|H,S,z]]`, and a multistage dynamic program that
reoptimizes after every job.  The implementation uses common random numbers and
direct conditional simulation, not likelihood-ratio reweighting.

Run the small smoke experiment:

```bash
GRID_N=3 OUTER_REPS=3 INNER_TRAIN=4 INNER_EVAL=6 JOBS_PER_PATH=8 \
  OUTPUT_DIR=results/assembly-smoke julia --project=. assembly_line_two_stage_2d.jl
```

The default research-scale command is:

```bash
julia --project=. assembly_line_two_stage_2d.jl
```

The driver accepts `GRID_N`, `OUTER_REPS`, `INNER_TRAIN`, `INNER_EVAL`,
`JOBS_PER_PATH`, five independent seed variables, `ALPHA`, `T_MAX`,
`TAU_THETA`, `SIGMA_SENSOR`, `CONDITION_SENSITIVITY`, `ACTIVE_LINES`, `FIXED_Z`,
and `OUTPUT_DIR`.  Vector settings are comma-separated.  It writes a grid CSV,
a best-point context CSV, and heatmaps of net objective, gross adaptive value,
information value, posterior uncertainty, and training optimism.

Computational work grows approximately with
`GRID_N^2 * OUTER_REPS * INNER_TRAIN * 144 * JOBS_PER_PATH`, plus independent
evaluation work.  Start with the smoke settings before increasing the nested
sample sizes.
