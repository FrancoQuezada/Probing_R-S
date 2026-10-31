# Final Validation of the Assembly-Line Probing Model

## 1. Executive conclusion

The implementation is consistent with the intended assembly-line formulation in
the relevant rework-system sections of `Probing_R_S-2.pdf` and is suitable for
reproducible exploratory numerical experiments with synthetic parameters. It is
not yet suitable for publication-quality comparative claims: outer uncertainty
can be material, z candidates use independent rather than paired streams, and
no empirical calibration has been performed.

No Critical or High-severity scientific inconsistency was found. No model or
estimator correction was made during Stage 10. One low-severity validation-script
stream-separation defect was corrected before recording final convergence results.

## 2. Implemented mathematical model

For each line, `Theta_c ~ Normal(mu_c, tau_c^2)`. For positive effort,
`eta_c | Theta_c,z_c ~ Normal(Theta_c, nu_c/(Tmax_c z_c))`; at zero effort the
diagnostic is absent and the posterior is the prior. Given a posterior draw,
each line has a homogeneous Poisson rework process with rate
`lambda0_c exp(gamma_lambda_c(Theta_c-mu_c))`, logistic complexity probability,
and both potential exponential service times conditional on line and complexity.

The policy `x_i=(a_v,a_q,tau)` sends a job to FAST iff
`a_v v_c-a_q(q_FAST-q_STANDARD) >= tau`. Shift reward is line-value revenue
minus FAST cost and delay cost. The outer objective is
`J(z)=-alpha' z+E_eta|z[max_i E[R(x_i;xi)|eta,z]]`.

## 3. Information and decision timeline

`z -> true Theta -> eta -> posterior -> conditional policy selection -> fixed-policy shift simulation -> reward`.

`nested_monte_carlo.jl:5-13` uses true `Theta` only to generate diagnostics.
Selection receives `PosteriorState`, not true `Theta` or diagnostics. The chosen
policy remains fixed while `routing_simulation.jl:28-65` routes the full shift.

## 4. Nested Monte Carlo estimator

For each outer replication, `evaluate_z` forms a posterior, chooses a policy on
`Ntrain` common posterior-predictive paths, and evaluates that fixed choice on
an independent `Neval` bank. Gross value is the outer mean; `alpha' z` is
subtracted once afterwards; outer SE is the sample SE of outer evaluation means.
Finite `Ntrain` therefore evaluates a finite-sample policy-selection procedure,
not the exact analytical inner maximum.

## 5. Confirmed implementation consistencies

- `bayesian_update.jl:7-60`: stated observation model, zero-effort handling,
  and coordinatewise normal-normal posterior.
- `predictive_simulation.jl:105-146`: one posterior condition vector per
  shift, line-labelled arrivals, complexity-dependent potential services, and
  deterministic global ordering.
- `routing_simulation.jl:1-102`: no signal coefficient, FCFS residual-workload
  routing, deterministic line value, and one reward accounting per job.
- `conditional_policy_selection.jl:34-86`: common training bank, first-index
  argmax, and independent selected-policy evaluation bank.
- `nested_monte_carlo.jl:17-52`: independent outer replications, evaluation
  means rather than training means, and exactly one probing-cost subtraction.
- `finite_grid_search.jl:94-123`: Stage-6 delegation, duplicate compression,
  independent deterministic candidate streams, and net-objective ranking.
- `policy_family.jl:62-85`: stable finite policy order (`av`, then `aq`, then
  `tau`) retained through evaluation and search.

## 6. Findings and limitations

| ID | Severity | Classification | Location | Effect and action |
| --- | --- | --- | --- | --- |
| F-01 | Medium | Statistical limitation | `conditional_policy_selection.jl:79-86` | Finite `Ntrain` gives a noisy sample-average policy-selection rule, rather than an exact conditional maximum. Intended; report `Ntrain` in research work. |
| F-02 | Medium | Statistical limitation | `finite_grid_search.jl:107-116` | z candidates use independent streams. Ranking is reproducible but is not a paired statistical comparison; do not infer significance. |
| F-03 | Medium | Numerical/performance risk | `predictive_simulation.jl:13-23,112-126` | Normal conditions enter exponential rates, so rare adverse draws can create large job counts, runtime variation, and large outer SE. Overflow is rejected rather than truncated; monitor it. |
| F-04 | Low | Reporting limitation | `nested_monte_carlo.jl:48-52` | Direct `evaluate_z` has an RNG rather than a stored integer seed. Stage 7/reference results retain master and candidate seeds. |
| F-05 | Informational | Scope limitation | `reference_experiment.jl:54-70` | Parameters and finite policy/z grids are synthetic reference inputs, not calibration or global optimization. |
| F-06 | Low | Implementation defect (corrected) | `convergence_validation.jl:13-44` | Preliminary validation reused a master stream across distinct z points. The final script assigns fixed distinct stream keys per z, and the convergence CSV was rerun. No Stage-0–9 or reference-smoke result was affected. |

## 7. Convergence results

`examples/convergence_validation.jl` evaluated 15 fixed configurations with
seed `20261001`; distinct deterministic streams were assigned to distinct z
points and no adaptive search was performed. Results are in
`validation_outputs/convergence_results.csv`.

Holding `(Ntrain,Neval)=(50,100)`, the outer comparison was:

| z | Objective at L=10 (SE) | L=30 (SE) | L=100 (SE) |
| --- | ---: | ---: | ---: |
| (0,0) | 18.389 (0.445) | 17.069 (0.492) | 16.967 (0.301) |
| (0,1) | 16.631 (3.438) | 17.204 (1.206) | 16.377 (1.103) |
| (1,1) | 8.277 (9.573) | 8.654 (7.322) | 14.466 (2.294) |

SE is lower at L=100 than L=30 for all three points, but finite samples are not
monotone and full probing remains noisy. At `z=(0,1),L=30,Neval=100`, `Ntrain=50,100,300` gave
objectives 17.204, 17.481, and 16.845; this reflects a changed finite policy
selection rule. At `Ntrain=100`, `Neval=100,300,1000` gave objectives 17.481,
16.821, and 16.313 with SE 1.121, 1.183, and 1.258. This one-seed sensitivity is
not evidence of monotone behavior.

## 8. Sensitivity results

`examples/sensitivity_validation.jl` used seed `20261101`, fixed `z=(0,1)`,
and `(L,Ntrain,Neval)=(10,50,100)`. Results are in
`validation_outputs/sensitivity_results.csv`.

- Changing only probing cost retained gross value exactly at 11.8660 under the
  same stream, while net objective moved from 11.3660 (low cost) to 9.8660
  (high cost).
- Lower/higher diagnostic noise changed line-2 posterior variance from 0.3143
  to 0.1833/0.4889, respectively.
- Higher latent sensitivity produced objective -1.878 with SE 12.264 versus
  22.730 with SE 0.455 under lower sensitivity: consequences intensify, but
  this is not a calibrated value-of-information conclusion.
- Higher delay penalty lowered the observed objective. Policy identity and
  FAST-advantage checks showed no stable monotone pattern at these sample sizes.

## 9. Computational performance

Warm measurements from `examples/performance_validation.jl` were: one path
0.000043 s / 23 KB; a 20-path bank 0.000066 s / 78 KB; 12-by-20 rewards
0.000061 s / 10 KB; Stage-5 `(20,30)` 0.000182 s / 158 KB; and one smoke
`evaluate_z` 0.0745 s / 8.61 MB. The 20-path bank had 222 jobs and 240
policy-path evaluations. A warmed development `(30,100,200)` evaluation took
0.110 s for 9,000 paths and roughly 42,000 policy-path evaluations.

Repeated path generation and policy-path reward evaluation inside Stage 5 are
the dominant costs. Seed-dependent job counts make single timing observations
variable.

## 10. Reproducibility

Validation uses documented integer seeds and local `MersenneTwister` instances.
Stage 7 derives independent candidate streams from master seed and Float64 z
bits; serializers omit timestamps. Run:

```bash
julia --project=assembly_line_model assembly_line_model/test/runtests.jl
julia --project=assembly_line_model assembly_line_model/examples/convergence_validation.jl
julia --project=assembly_line_model assembly_line_model/examples/sensitivity_validation.jl
julia --project=assembly_line_model assembly_line_model/examples/performance_validation.jl
```

## 11. Recommended experiment settings

| Setting | outer_reps | inner_train | inner_eval | Runtime guidance | Outer-SE guidance |
| --- | ---: | ---: | ---: | --- | --- |
| Smoke | 5 | 20 | 30 | 9-point reference smoke: 3.62 s cold | Software verification only; smoke-best SE was 1.064. |
| Development | 30 | 100 | 200 | 0.110 s per measured warmed z; allow seconds for a grid | L=30 observed SE: 0.492–7.322. |
| Research | 100 | 300 | 1000 | About 11 times measured development work: 1–5 s/z and 9–45 s/grid, transparently extrapolated | At L=100 with smaller inner banks: 0.301–2.294; repeat master seeds. |

The research row is a workload extrapolation, not a precision guarantee.

## 12. Interpretation limits

Parameters are synthetic; policy and z families are finite; Monte Carlo error
remains; rankings do not establish statistical significance; no continuous
global optimum is claimed; and no empirical calibration has been performed.

## 13. Readiness for future methods

Candidate streams in `finite_grid_search.jl` are the clean point for CRN across
z. The posterior/diagnostic layer is the point for likelihood-ratio or gradient
estimators; predictive/routing separation supports green simulation; and the
finite-grid layer can host advanced z optimization. None is implemented here.

## 14. Common-random-number comparison mode

The finite grid now retains `randomization_mode=:independent` as its default
and additionally supports `:crn`. CRN uses fixed domain-separated indexed
substreams for prior conditions, diagnostics, posterior draws, arrival counts,
arrival-time prefixes, complexity uniforms, and FAST/STANDARD exponential
potential services. Keys never contain z or a candidate index. Training and
evaluation use separate domains; queue ordering may still differ after coupled
line/job-generation prefixes are globally sorted.

CRN supports `paired_comparison`, which reports the mean net-objective
difference, its paired SE, and gross-value correlation. It is a variance-
reduction device for differences, not a claim that marginal candidate SEs or
the heatmap are automatically smoother. Finite-sample policy selection can
weaken coupling where the selected policy changes.
