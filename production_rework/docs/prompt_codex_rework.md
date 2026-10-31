# Codex implementation prompt -- five LR/SF methods for the production/rework example

You are a senior Julia researcher/developer specializing in stochastic simulation optimization, likelihood-ratio/score-function estimators, and two-stage decision-making under uncertainty.

**Task:** Implement, test, and document FIVE algorithms for the CURRENT production/rework model:

1. `Fresh`
2. `Green`
3. `GreenSoftmax`
4. `GreenVariance`
5. `GreenVarianceSoftmax`

You have access to this repository. Before changing code, read the entire attached technical reference **`REWORK_LR_IMPLEMENTATION_BACKGROUND.md`**. If the source PDF is present, use **`Probing_R_S(1).pdf` (October 31, 2026), Sections 8.1-8.9, especially the box on p. 27 and notes on pp. 28-30**. The Markdown reference is self-contained where the PDF is unavailable. The older Julia examples `LikelihoodMethod002.jl` and `probing_three_searches_crn_oracle_surface.jl` illustrate some reusable programming ideas but solve a DIFFERENT model; do not copy their Bernoulli probing signals or global policy selection into rework.

## First: audit the repository

Locate the current rework model, package entry points, main experiment configuration, random-stream handling, simulation functions, posterior/policy-selection functions, and tests. Identify actual paths and dependencies; do not invent file names. Produce a short audit stating what already exists, what is missing, and which functions will be reused. Keep existing functional behavior intact. If rework code is absent, build a minimal but fully runnable, modular implementation of the model specified in the background; do not silently substitute the old two-server Bernoulli example.

Do not stop after writing a plan: **implement the algorithms, run the tests that the environment permits, execute a reproducible small comparison, and report the outputs and limitations.**

## Mathematical invariants (must follow)

The physical system has latent per-line states `Theta`, rework arrivals `A` (including lines of origin), complexities `D`, and potential service times `B`. The line value `v_c` is deterministic. One pre-shift measurement `eta_c` is obtained for each probed line, then ONE routing policy is selected and retained for the whole shift. This is a genuine two-stage selection problem:

```text
J(z) = -alpha' z + E_{eta|z}[ max_i E[Y_i(Theta,A,D,B) | eta,z] ].
```

NEVER replace `E[max conditional policy value]` with `max E[policy value]`. NEVER give a policy selector access to the outer shift's true `Theta`. No job-level probing signals occur in this version.

For each active line `c` with `z_c>0`, the measurement variance is `nu_c/(Tmax_c*z_c)`. Complete simulation output must include `(Theta,eta,A,D,B)` so historical LR reweighting does NOT require integrating out Theta. For **target z** and **source zsource** with the SAME active measurement mask:

```text
logL = sum_active_c [ 0.5*log(z_c/zsource_c)
                    - Tmax_c*(z_c-zsource_c)*(eta_c-Theta_c)^2/(2*nu_c) ]
score_c = 1/(2*z_c) - Tmax_c*(eta_c-Theta_c)^2/(2*nu_c)
L = exp(logL)
```

**IMPORTANT:** The PDF's p. 27 BOX has the prefactor reversed and incorrect signs in the exponential and score. Implement the corrected equations above, not the boxed expressions verbatim. Treat score as an m-dimensional vector; `-alpha` is added to the overall gradient exactly once. Validate against direct Normal log-pdf differences and finite differences before implementing the optimizer.

At `z_c=0`, there is NO measurement, and the positive-z Gaussian LR formula is undefined. Track explicit active observation masks. Reject cross-mask LR reuse. Search positive coordinates with a configurable `zmin>0`, leave inactive coordinates exactly 0, and handle candidate zero/positive masks explicitly. For a small number of lines, enumerate masks and compare their independently evaluated endpoints; for a larger instance use documented feasible masks. Never replace zero by epsilon without disclosing that it changes the model.

## Conditional policy selector and score contributions

Implement a reusable selector `conditional_policy_values(eta,z,...)` that estimates each candidate policy's expected reward given a particular pre-shift measurement using independent inner simulation from the exact Gaussian posterior `Theta|eta,z` (or a mathematically equivalent validated importance-sampling approach). Reuse identical inner physical paths across the candidate policies for paired comparisons, but keep inner policy-training paths independent of the outer gradient-evaluation paths. A source bank's outer `Theta` is used for LR/gradient calculations, not for selecting the policy.

For each realized outer shift `xi=(Theta,eta,A,D,B)` and target `z`, the hard contribution from a bank generated at `zsource` is:

```text
G_hard(xi; z,zsource) = L(xi;z,zsource) *
                        Y_{argmax_i mu_hat_i(eta,z)}(xi_physical) * score(xi;z)
```

The chosen policy is PER OBSERVED eta at the TARGET z, not one global best policy across the bank and not a policy frozen at its historical source z. The weighted gross objective contribution uses the same selected policy but no score. The result is an estimate of the ORIGINAL hard two-stage problem, with finite-inner-sample and adaptive-reuse limitations disclosed.

For softmax methods use a stable inverse-temperature `beta>0` with weights computed from conditional policy means:

```text
w_i(eta,z) = softmax_i(beta * mu_hat_i(eta,z))
G_soft(xi; z,zsource) = L(xi;z,zsource) *
                        (sum_i w_i(eta,z) * Y_i(xi_physical)) * score(xi;z)
```

Softmax is a gradient-smoothing heuristic; keep the original hard objective and its independent validation as the common comparison criterion. Include a `beta` sensitivity option and test that softmax approaches hard argmax for large beta and unique best policies.

## Exact definitions of the five searches

Use a common shared projected-gradient-ascent driver, same initialization(s), step schedule, stopping rules, policies, and simulation budget conventions for ALL five methods. At each iterate generate a fresh bank of `M_outer` complete shifts under current `z`. Save each bank's source z, mask, full primitives, and reward of every candidate policy on the same physical shift. Inner policy-training replications are separate.

- **Fresh:** use ONLY the current bank, with `L=1` pathwise. Discard historical banks for gradient purposes.
- **Green:** retain all banks from earlier iterates with compatible observation masks. Evaluate each at CURRENT target z using `f_z/f_zsource` and the target-specific hard conditional policy selector. Average all included per-path hard contributions, weighted correctly by sample counts.
- **GreenSoftmax:** same history as Green, but use target-specific per-observation softmax contributions.
- **GreenVariance:** extend hard Green by screening historical banks using an estimated variance of their reweighted per-path gradient vectors. For bank tau compute `V_tau = tr(sample_cov(G_hard_tau_l))` and accept if `V_tau <= threshold` (plus any explicitly configured ESS rule). Always accept the CURRENT bank. Compare variance **without** the `1/M_outer` factor when testing the threshold, as proposed on p. 30. Normalize accepted contributions with the actual number of accepted observations. This no-softmax variant is a documented design extension of the notes, not a verbatim formula from the PDF.
- **GreenVarianceSoftmax:** same screening logic but compute `V_tau = tr(sample_cov(G_soft_tau_l))` using the actual combined per-path soft contribution, then pool accepted soft contributions. The source's p. 30 `sum_i sigma_i^2*w_i` is only a heuristic and ignores covariance between policy contributions (and their varying weights in rework); do not mislabel that proxy as the exact variance of the combination. If implemented, expose it only as a separately named research diagnostic, NEVER as an undocumented substitute.

Optional but useful: variance screening from pilot paths disjoint from gradient-evaluation paths to study sample-dependent screening bias; count pilot costs in experimental budgets. In every method, calculate/log ESS, mean LR, extreme weights, source-to-target distance and rejected-bank reasons. Pure Green must not silently omit low-ESS banks unless explicitly configured as a separate study.

## Required software architecture

Implement as Julia modules/functions consistent with existing package layout, not five giant duplicated scripts. Factor out: model simulation, posterior conditional generation, all-policy path rewards, complete-data `logLR/score`, per-record hard/soft contributions, bank/history storage, variance filter, step rule, direct evaluation and reporting. Make methods selectable from a single experiment CLI/config with adjustable `method`, `seed`, `n_outer`, `n_inner_train`, `n_outer_validation`, `max_iters`, `z0`, `zmin`, `beta`, `variance_threshold`, `min_ess`, and `output_dir` (adapt names to actual conventions). Preserve existing commands and output formats when possible.

Use stable log-likelihood arithmetic and stable softmax (`subtract(max)`), but do not arbitrarily clip raw LR values and still call the resulting estimator unbiased. Detect and report numerical overflows and low ESS. Keep paths and seeds reproducible. Write machine-readable iteration-level results and a final cross-method summary. Optional 2D trajectory plots must use an independently computed direct-resimulation reference surface, not a historical LR surface mislabeled as oracle.

## Tests and acceptance gates -- run them

1. Gaussian LR and score vs `Distributions.Normal` logpdf and central finite differences, with heterogeneous line parameters. At `z=zsource`, `L=1`. Under source sampling `mean(L)~=1`; under target `mean(score)~=0`; under source `mean(L*score)~=0`.
2. Closed-form test with toy reward `Y=eta^2`: `E[Y]=mu^2+tau^2+nu/(Tmax*z)` and derivative `-nu/(Tmax*z^2)`. Both fresh and reweighted green gradients must agree with the analytic sign and value within sampling uncertainty.
3. **Analytic two-stage test:** `Theta~N(0,tau^2)`, two policies `Y_+=Theta`, `Y_-=-Theta`, and Gaussian observation; exact `q(z)=sqrt(2/pi)*tau^2/sqrt(tau^2+nu/(Tmax*z))`. Test its derivative specified in the Markdown, using the per-eta policy selector. This specifically checks the two-stage `E[max]` structure.
4. Prior/posterior tests, correct missing measurement at zero, cross-mask LR rejection, no information leakage, all-policy pairing of physical paths, deterministic product value by origin line.
5. Algorithmic invariants: Fresh equals single-bank Green when sharing data; no variance rejection implies GreenVariance = Green and GreenVarianceSoftmax = GreenSoftmax; large-beta softmax approaches hard argmax for distinct policy means; covariance-aware variance passes constructed correlated-policy tests.
6. Run a small deterministic smoke comparison for all five methods, then independently re-simulate their final z with NEW seeds and the correct nested selection logic. Report oracle mean/SE (or replications-based interval), physical simulation counts, inner simulation counts, time, and ESS/variance-screening diagnostics.

## Deliverables in the repository

- Implemented Julia code for all five methods and a minimal reproducible runner/config.
- Unit/integration tests that are actually executed, plus command lines and results.
- README/methodology documenting model, mathematical corrections to p. 27, zero-effort masks, how nested policy selection is implemented, variance-screening statistical caveats and softmax's heuristic nature.
- CSV/other machine-readable results with method, iteration, `z`, net/hard objective, gradient, selected-policy counts, active/retained/rejected banks and reasons, ESS, variance statistic, beta/threshold, seeds, runtime, and independent endpoint evaluation.
- A concise implementation audit and a final summary of changed files, passed/failed tests, smoke-run outcomes and unresolved issues. Distinguish completed work from work that the runtime prevented you from executing.

**Priorities:** mathematical correctness and correct information structure first; validated simulation second; clear modularity third; optimization/plots fourth. Do not change the underlying stochastic model merely to make LR or gradient search easy. Avoid asking clarifying questions when the supplied specification resolves the issue: implement a reasonable, explicitly documented choice and test it.
