# Assembly-Line Probing and Rework Routing Model

## Implementation specification for Codex

## 1. Purpose and scope

This document specifies the first implementation of the **assembly-line probing and centralized rework-routing example**.

The immediate objective is to implement and validate the model using **nested Monte Carlo simulation** and a simple first-stage search over the probing decision \(z\).

This version must **not** implement:

- likelihood-ratio estimators;
- score-function gradients;
- green simulation;
- continuous gradient-based optimization;
- job-level probing signals;
- the previous two-server example's `p_hit`, `p_false`, or `a_s` terms.

The implementation should reuse only the parts of the existing two-server code that remain conceptually valid, especially:

- two single-server queues, FAST and STANDARD;
- FCFS workload updates;
- evaluation of a finite family of routing policies;
- evaluation of all policies on the same simulated shift paths;
- reproducible random-number streams;
- optional common random numbers after the basic implementation is validated.

The central decision and information sequence is

\[
\Theta \text{ is realized but unobserved}
\;\longrightarrow\;
z
\;\longrightarrow\;
\eta
\;\longrightarrow\;
i^\star(\eta,z)
\;\longrightarrow\;
(\mathcal A,\mathcal D,\mathcal B)
\;\longrightarrow\;
R.
\]

The selected routing policy is chosen **after** the pre-shift information \(\eta\) is observed and remains fixed during the complete operating shift.

---

## 2. Mathematical formulation

Let

\[
\mathcal C=\{1,\ldots,m\}
\]

be the set of assembly lines.

Each line \(c\in\mathcal C\) produces one product family with deterministic value

\[
v_c\in[0,1].
\]

The value is not random. Before the next job arrives, its value is unknown only because its line of origin is unknown. Once a job arrives and its line is observed, its value is determined as \(v_c\).

The uncertainty vector is

\[
\xi=(\Theta,\mathcal A,\mathcal D,\mathcal B),
\]

where:

- \(\Theta=(\Theta_1,\ldots,\Theta_m)\) contains latent line conditions;
- \(\mathcal A\) contains arrival times and corresponding lines of origin;
- \(\mathcal D\) contains job-complexity realizations;
- \(\mathcal B\) contains potential service times at FAST and STANDARD.

Do not introduce a separate random value vector.

The first-stage objective is

\[
J(z)
=
-\alpha^\top z
+
\mathbb E_{\eta\mid z}
\left[
\max_{i\in\mathcal X}
\mathbb E_{\xi\mid\eta,z}
\left[R(x_i;\xi)\right]
\right].
\]

---

## 3. Latent assembly-line conditions

For each line \(c\),

\[
\Theta_c\sim\mathcal N(\mu_c,\tau_c^2).
\]

Larger values of \(\Theta_c\) represent worse operating conditions.

Assumptions for the first version:

- line conditions are mutually independent;
- \(\Theta_c\) remains constant during one complete shift;
- probing does not change \(\Theta_c\);
- probing only changes the precision of the information obtained about \(\Theta_c\).

---

## 4. First-stage probing decision

The probing decision is

\[
z=(z_1,\ldots,z_m)\in[0,1]^m.
\]

For line \(c\), the diagnostic duration is

\[
t_c(z_c)=T_c^{\max}z_c.
\]

The probing cost is

\[
C^{\mathrm{probe}}(z)=\alpha^\top z.
\]

The cost must be subtracted exactly once from the expected operational reward.

---

## 5. Diagnostic observation model

For \(z_c>0\), the pre-shift measurement satisfies

\[
\eta_c\mid\Theta_c,z_c
\sim
\mathcal N\left(
\Theta_c,
\frac{\nu_c}{T_c^{\max}z_c}
\right).
\]

Equivalently,

\[
\eta_c
=
\Theta_c
+
\sqrt{\frac{\nu_c}{T_c^{\max}z_c}}\,\varepsilon_c,
\qquad
\varepsilon_c\sim\mathcal N(0,1).
\]

For \(z_c=0\):

- no measurement is obtained;
- the code must not divide by zero;
- the posterior remains equal to the prior;
- the absence of a measurement should be represented explicitly, preferably through `missing`, a Boolean observation mask, or a dedicated posterior state.

---

## 6. Posterior distribution

For \(z_c>0\), conjugacy gives

\[
\Theta_c\mid\eta_c,z_c
\sim
\mathcal N\left(m_c(\eta_c,z_c),V_c(z_c)\right),
\]

with

\[
V_c(z_c)
=
\left(
\frac{1}{\tau_c^2}
+
\frac{T_c^{\max}z_c}{\nu_c}
\right)^{-1},
\]

and

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

For \(z_c=0\),

\[
m_c=\mu_c,
\qquad
V_c=\tau_c^2.
\]

The implementation should expose posterior means and variances through a dedicated immutable data structure.

---

## 7. Conditional arrival process

Conditional on \(\Theta_c\), the rework-arrival rate for line \(c\) is

\[
\lambda_c(\Theta_c)
=
\lambda_c^0
\exp\left(
\gamma_c^\lambda(\Theta_c-\mu_c)
\right),
\qquad
\gamma_c^\lambda\ge 0.
\]

For the first implementation, use a homogeneous Poisson process on \([0,H]\).

A simple exact simulation method is:

\[
N_c\mid\Theta_c
\sim
\operatorname{Poisson}\left(\lambda_c(\Theta_c)H\right),
\]

then, conditional on \(N_c\), generate

\[
A_{c,1},\ldots,A_{c,N_c}
\overset{iid}{\sim}
\operatorname{Uniform}(0,H)
\]

and sort the arrival times.

After generating arrivals for every line, merge all jobs and sort them globally by arrival time.

Each arrival record must contain its line of origin. No additional random job-class variable is needed.

---

## 8. Conditional job complexity

For the \(j\)-th rework job generated by line \(c\),

\[
D_{c,j}\mid\Theta_c
\sim
\operatorname{Bernoulli}(p_c(\Theta_c)),
\]

where

\[
p_c(\theta)
=
\frac{1}{
1+
\exp\left[-\left(
\gamma_{c,0}^{p}
+
\gamma_{c,1}^{p}(\theta-\mu_c)
\right)\right]
}.
\]

Use a numerically stable logistic implementation.

No additional Bernoulli variable is required to determine whether a produced unit becomes a rework job. Every simulated arrival is already a job requiring processing at the rework center.

---

## 9. Potential service times

For every rework job, generate both potential service times:

\[
B_{c,j}^{F},
\qquad
B_{c,j}^{S}.
\]

Their distributions may depend on:

- line of origin \(c\);
- complexity \(D_{c,j}\);
- server FAST or STANDARD.

A first implementation may use exponential distributions parameterized by mean:

\[
B_{c,j}^{F}\mid D_{c,j}=d
\sim
\operatorname{ExponentialMean}(\beta_{c,d}^{F}),
\]

\[
B_{c,j}^{S}\mid D_{c,j}=d
\sim
\operatorname{ExponentialMean}(\beta_{c,d}^{S}).
\]

The implementation should not hard-code a single service distribution if a parameter table can represent line, complexity, and server effects cleanly.

Generating both potential service times allows all policies to be evaluated on the same sample path.

---

## 10. Job and sample-path structures

A suitable initial Julia structure is

```julia
struct ReworkJob
    arrival::Float64
    line::Int
    complex::Bool
    service_fast::Float64
    service_standard::Float64
end
```

The value should be retrieved deterministically from the line parameters:

```julia
value = params.lines[job.line].value
```

A sample path may be represented as

```julia
struct ShiftPath
    jobs::Vector{ReworkJob}
end
```

Do not include a job-level probing signal in `ReworkJob`.

---

## 11. Routing policies

Let

\[
\mathcal X=\{x_1,\ldots,x_k\}
\]

be a finite family of candidate routing policies.

Each policy is

\[
x_i=(a_v^i,a_q^i,\tau^i).
\]

For an arriving job from line \(c\), with current residual workloads \(q_F\) and \(q_S\), define

\[
\operatorname{score}_i
=
 a_v^i v_c
-
 a_q^i(q_F-q_S).
\]

Route the job to FAST if

\[
\operatorname{score}_i\ge\tau^i,
\]

and to STANDARD otherwise.

A suitable Julia structure is

```julia
struct Policy
    av::Float64
    aq::Float64
    tau::Float64
end
```

There is no coefficient \(a_s\), because there is no job-level signal.

The pre-shift diagnostic vector \(\eta\) affects operations only by determining which policy is selected before the shift.

---

## 12. Queueing simulation

The queueing logic from the existing two-server example can be reused.

Maintain:

```julia
free_fast = 0.0
free_standard = 0.0
```

For a job arriving at time \(t\), define residual workloads

```julia
qf = max(free_fast - t, 0.0)
qs = max(free_standard - t, 0.0)
```

If routed to FAST:

```julia
start = max(t, free_fast)
finish = start + job.service_fast
free_fast = finish
```

If routed to STANDARD:

```julia
start = max(t, free_standard)
finish = start + job.service_standard
free_standard = finish
```

The policy is fixed for the complete shift.

Individual routing decisions are applications of the fixed policy and are not additional optimization stages.

---

## 13. Initial reward convention

The reward function should be isolated in one function so it can be changed without modifying the probabilistic or queueing architecture.

For the first implementation, use a configurable reward of the form

\[
R(x_i;\xi)
=
\sum_j
\left[
\rho v_{c(j)}
-
c_F\mathbf 1\{j\text{ routed to FAST}\}
-
c_T(C_j-A_j)
\right],
\]

where:

- \(A_j\) is the arrival time;
- \(C_j\) is the completion time;
- \(\rho\) is a revenue scale;
- \(c_F\) is the additional FAST routing cost;
- \(c_T\) is the delay penalty per unit time in system.

Initial horizon convention:

- generate arrivals only on \([0,H]\);
- all jobs arriving before \(H\) are processed;
- jobs may complete after \(H\);
- their full time in system is charged;
- no job is discarded solely because its completion occurs after \(H\).

These choices must be parameterized and documented as version-1 conventions rather than embedded assumptions.

---

## 14. Nested Monte Carlo estimator

For a fixed first-stage decision \(z\), estimate

\[
J(z)
=
-\alpha^\top z
+
\mathbb E_{\eta\mid z}
\left[
\max_i
\mathbb E_{\xi\mid\eta,z}
[R(x_i;\xi)]
\right].
\]

### 14.1 Outer replication

For each outer replication \(\ell=1,\ldots,L\):

1. Sample the true latent condition
   \[
   \Theta^{(\ell)}\sim P_\Theta.
   \]

2. Generate the pre-shift diagnostics
   \[
   \eta^{(\ell)}\sim P(\eta\mid\Theta^{(\ell)},z).
   \]

3. Compute the posterior parameters of
   \[
   P(\Theta\mid\eta^{(\ell)},z).
   \]

The true \(\Theta^{(\ell)}\) used to generate \(\eta^{(\ell)}\) must not be revealed to the policy-selection routine.

### 14.2 Inner training bank

For each inner training replication \(r=1,\ldots,N_{\mathrm{train}}\):

1. Sample
   \[
   \widetilde\Theta^{(\ell,r)}
   \sim
   P(\Theta\mid\eta^{(\ell)},z).
   \]

2. Conditional on \(\widetilde\Theta^{(\ell,r)}\), generate:
   - arrival processes;
   - line origins;
   - job complexities;
   - potential service times.

3. Evaluate every policy on the generated shift path.

For each policy \(i\), estimate

\[
\widehat Q_i(\eta^{(\ell)},z)
=
\frac{1}{N_{\mathrm{train}}}
\sum_{r=1}^{N_{\mathrm{train}}}
R(x_i;\xi^{(\ell,r)}).
\]

Select

\[
\widehat i_\ell^\star
\in
\arg\max_i
\widehat Q_i(\eta^{(\ell)},z).
\]

### 14.3 Independent inner evaluation bank

Generate an independent conditional bank of size \(N_{\mathrm{eval}}\) from the same posterior predictive distribution and compute

\[
\widehat V_\ell(z)
=
\frac{1}{N_{\mathrm{eval}}}
\sum_{r=1}^{N_{\mathrm{eval}}}
R(x_{\widehat i_\ell^\star};\xi_{\mathrm{eval}}^{(\ell,r)}).
\]

Using a separate evaluation bank is recommended from the beginning because it avoids conflating policy selection and policy evaluation.

### 14.4 Final estimator

Compute

\[
\widehat J(z)
=
-\alpha^\top z
+
\frac{1}{L}
\sum_{\ell=1}^{L}
\widehat V_\ell(z).
\]

Estimate the Monte Carlo standard error using outer-replication values:

\[
\widehat{\operatorname{SE}}(\widehat J(z))
=
\frac{
\operatorname{sd}(\widehat V_1(z),\ldots,\widehat V_L(z))
}{\sqrt L}.
\]

---

## 15. Core function interfaces

The implementation should aim for interfaces similar to the following.

```julia
sample_prior_theta(rng, params) -> Vector{Float64}

sample_diagnostics(rng, theta, z, params) -> DiagnosticObservation

posterior_parameters(diagnostics, z, params) -> PosteriorState

arrival_rate(c, theta_c, params) -> Float64

complexity_probability(c, theta_c, params) -> Float64

generate_predictive_path(rng, posterior, params) -> ShiftPath

generate_predictive_bank(rng, posterior, params, npaths) -> Vector{ShiftPath}

build_policy_grid(params) -> Vector{Policy}

evaluate_policy(policy, path, params) -> Float64

precompute_rewards(policies, bank, params) -> Matrix{Float64}

select_policy_given_information(rng, diagnostics, z, params, policies;
                                inner_train) -> PolicySelectionResult

evaluate_z(z, params, policies;
           outer_reps, inner_train, inner_eval, seed) -> ZEvaluationResult
```

Results should be returned through named tuples or explicit immutable result structs rather than long positional tuples.

---

## 16. Suggested module architecture

A clean first implementation may use the following files.

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

If the repository is intentionally kept small, these components may initially be grouped into fewer files, but the conceptual separation should remain.

Do not modify the existing two-server implementation except for shared utility extraction that is clearly safe and separately tested.

---

## 17. First-stage search without gradients

The first implementation should optimize \(z\) using derivative-free methods.

### 17.1 Two-dimensional validation case

Start with two active probing coordinates so the objective can be visualized.

Use a coarse grid such as

\[
z_c\in\{0,0.1,\ldots,1\}.
\]

Then refine around the best point using a finer local grid.

### 17.2 Higher-dimensional case

For larger \(m\), avoid a dense Cartesian grid.

Use one of:

- coordinate search;
- pattern search;
- coarse grid \(\{0,0.5,1\}^m\) followed by local refinement;
- a derivative-free package only after the objective evaluator has been validated.

The first priority is a correct and reproducible `evaluate_z`, not a sophisticated optimizer.

---

## 18. Common random numbers

Common random numbers may be introduced only after the independent Monte Carlo implementation passes all tests.

Potential reusable primitives include:

- standard normals for prior \(\Theta\);
- standard normals for diagnostic noise;
- standard normals for posterior draws;
- uniforms for Poisson-process construction;
- uniforms for Bernoulli complexity;
- unit exponentials for service times.

A CRN implementation must preserve the correct marginal distribution at every \(z\).

Do not add CRN in the first correctness pass unless it is needed for a specific validation test.

---

## 19. Phased implementation plan

Each phase must end with tests and a short implementation report. Do not proceed when tests fail.

### Phase 0 — Repository protection and execution plan

Tasks:

- inspect the current repository;
- preserve the existing two-server scripts unchanged;
- create a new assembly-line module or directory;
- add a minimal `Project.toml` if the repository does not already provide one;
- document how to run tests and the small example.

Acceptance criteria:

- no existing source file is modified unintentionally;
- the new module loads successfully;
- the test runner executes.

### Phase 1 — Types and parameters

Implement:

- `LineParameters`;
- `ModelParameters`;
- `PosteriorState`;
- `DiagnosticObservation`;
- `ReworkJob`;
- `ShiftPath`;
- `Policy`;
- result structs for policy selection and \(z\)-evaluation.

Acceptance criteria:

- dimensions are validated;
- all rates, variances, costs, and horizon values are checked;
- deterministic values \(v_c\) are stored only in line parameters;
- no job-level signal field exists.

### Phase 2 — Bayesian information layer

Implement:

- prior sampling;
- diagnostic sampling;
- exact posterior mean and variance;
- safe handling of \(z_c=0\).

Required tests:

- posterior equals prior at \(z_c=0\);
- posterior variance decreases as \(z_c\) increases;
- posterior mean matches the analytical formula;
- no division by zero;
- repeated runs with the same seed are identical.

### Phase 3 — Posterior predictive shift generation

Implement:

- arrival-rate function;
- stable complexity-probability function;
- per-line Poisson arrivals;
- global merge and sort;
- complexity sampling;
- potential FAST and STANDARD service times.

Required tests:

- arrival rate increases with adverse \(\Theta_c\) when \(\gamma_c^\lambda>0\);
- complexity probability increases with adverse \(\Theta_c\) when \(\gamma_{c,1}^p>0\);
- all arrivals lie in \([0,H]\);
- global path is sorted;
- every job has a valid line origin;
- value is obtained deterministically from line origin;
- zero-arrival paths are valid;
- generated service times are positive.

### Phase 4 — Routing simulator

Implement:

- policy score;
- FAST/STANDARD decision;
- FCFS workload updates;
- reward calculation;
- all-policy evaluation on a common path bank.

Required tests:

- hand-constructed 3–5 job path;
- exact start, finish, workload, and reward values;
- selected policy remains fixed for the complete shift;
- no signal term appears in the routing rule;
- simulation state resets between paths.

### Phase 5 — Conditional policy selection

Implement:

- inner training-bank generation from a fixed posterior;
- reward matrix;
- conditional sample-mean value of each policy;
- deterministic tie handling;
- independent evaluation bank for the selected policy.

Required tests:

- the true outer latent \(\Theta\) is never passed to policy selection;
- train and evaluation random streams are independent;
- with one policy, selection is trivial and evaluation is correct;
- with deterministic paths, the known best policy is selected.

### Phase 6 — Nested Monte Carlo evaluator

Implement `evaluate_z`.

Required outputs:

- gross operational value;
- probing cost;
- net objective;
- Monte Carlo standard error;
- outer values;
- selected-policy frequencies;
- configuration and seeds.

Required tests:

- probing cost is subtracted exactly once;
- \(z=0\) executes correctly;
- same seed gives identical output;
- increasing outer replications reduces empirical Monte Carlo noise in a controlled experiment;
- selected-policy frequencies sum to one.

### Phase 7 — First-stage search

Implement:

- small grid search;
- ranking of candidate \(z\) values;
- local grid refinement;
- export of results to CSV or a structured text format.

Required tests:

- all grid points are evaluated exactly once;
- best reported point matches the maximum stored objective;
- no search routine changes the semantics of `evaluate_z`.

### Phase 8 — Scientific validation

Run small controlled experiments:

- no-information case;
- near-perfect information case;
- zero probing cost;
- very high probing cost;
- line conditions with no effect on arrivals or complexity;
- identical product values;
- one-line case;
- one-policy case.

The observed behavior must be explained before running large experiments.

### Phase 9 — Performance improvements

Only after correctness is established, consider:

- common random numbers;
- threading;
- batching;
- caching policy-independent path information;
- optimized policy evaluation;
- derivative-free optimization packages.

No performance optimization should alter the validated statistical estimator.

---

## 20. Critical implementation invariants

The following invariants must be enforced throughout development.

1. \(v_c\) is deterministic and obtained from line origin.
2. There is no job-level probing signal.
3. \(\eta\) is observed before policy selection.
4. One policy is selected for the entire shift.
5. Policy selection uses the posterior, not the true latent \(\Theta\).
6. Conditional on sampled posterior \(\Theta\), arrivals and complexities are generated from their structural models.
7. Probing affects information precision, not the physical line condition.
8. Probing cost is subtracted exactly once.
9. Training and evaluation banks are independent unless a deliberate CRN design is explicitly documented.
10. The first implementation uses Monte Carlo, not likelihood ratios or gradients.

---

## 21. Minimal initial experiment

The first executable example should be deliberately small.

Suggested configuration:

- \(m=2\) lines;
- 2–3 candidate values for each policy coefficient;
- 5–20 outer replications;
- 20–100 inner training paths;
- 20–100 inner evaluation paths;
- a short horizon;
- fixed seed;
- a small \(z\)-grid.

The purpose is to validate model logic, not to produce final numerical conclusions.

The experiment should print:

- evaluated \(z\);
- net objective and standard error;
- probing cost;
- selected-policy frequencies;
- average number of jobs per shift and per line;
- average fraction of complex jobs;
- reproducibility seed.

---

## 22. Completion criterion for the first implementation

The first implementation is complete when:

- all phases through Phase 7 pass their tests;
- the small experiment runs reproducibly;
- the information sequence is verified;
- the nested Monte Carlo estimator matches the formulation in this document;
- the existing two-server code remains available and unchanged;
- no likelihood-ratio or gradient code has been introduced.
