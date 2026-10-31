# Rework example: likelihood-ratio and score-function implementation background

**Document type:** technical specification for Codex (Julia)
**Source:** `Probing_R_S(1).pdf`, dated October 31, 2026, especially Sections 8.1-8.9, pp. 15-27, and the softmax/variance notes, pp. 27-30.
**Related code supplied for orientation only:** `LikelihoodMethod002.jl` and `probing_three_searches_crn_oracle_surface.jl`. These examples are NOT implementations of the current rework model.
**Status vocabulary:** **SOURCE** = model or proposal stated in the PDF; **CORRECTION** = mathematical correction needed for consistency; **DESIGN** = proposed implementation choice, to be documented and tested rather than presented as an original formula from the PDF.

## 1. Read this first: nonnegotiable distinctions

1. **SOURCE:** Rework is a genuine two-stage *information acquisition* problem. Choose pre-shift diagnostic effort `z`; observe one line-level measurement `eta_c` for each probed line; choose ONE policy `x_i` for the whole subsequent shift; execute that policy as jobs arrive. The optimal policy generally depends on the observation `eta` and on `z`.
2. **SOURCE:** There are no job-level probing signals and no independent random job values in this rework example. Each line `c` has a deterministic value `v_c`, observed when the arrival's line of origin becomes known. Do not reuse the older Bernoulli hit/false-positive probing model.
3. **SOURCE:** The latent physical states `Theta_c` do not depend on probing. Their realization is saved by the simulator for likelihood calculations, **but is never revealed to the decision maker choosing the policy**.
4. **CORRECTION:** The boxed likelihood ratio and score on p. 27 contain sign/reversal mistakes. The corrected formulas in Section 4 below MUST be used, with independent analytic checks.
5. **CORRECTION:** A vector score is required: one derivative per active probing coordinate, not an undifferentiated scalar sum.
6. **DESIGN:** Strict zero probing is a no-observation regime. It cannot be treated as an ordinary Gaussian observation of variance `nu/(Tmax * 0)`. LR recycling is only permitted between compatible observation masks.
7. **SOURCE:** The last-page softmax + green-variance notes expressly discuss a **simplified model without eta**. They are a methodological idea, not a ready-made estimator of the rework nested objective. The adaptation below preserves the outer observation / conditional-policy-selection structure.
8. **CORRECTION:** The p. 30 `sum_i sigma_i^2 * w_i` expression is not, in general, the variance of a weighted sum of correlated policy gradients. The within-path covariance terms matter; see Section 8.

## 2. Production/rework stochastic model

Let `C = {1,...,m}` denote assembly lines, and let `H` be the shift horizon. The physical uncertainty is

\[
\xi_{\mathrm{phys}}=(\Theta,\mathcal A,\mathcal D,\mathcal B).
\]

- `Theta_c ~ Normal(mu_c, tau_c^2)` independently between lines; adverse conditions have larger `Theta_c`.
- Conditional on `Theta_c`, line `c` produces rework arrivals (already defective units) at rate

\[
\lambda_c(\Theta_c)=\lambda_c^0\exp\big(\gamma_c^\lambda(\Theta_c-\mu_c)\big).
\]

- Each rework job from `c` has unobserved complexity `D_{c,j} in {0,1}` with

\[
P(D_{c,j}=1\mid\Theta_c)=p_c(\Theta_c)=
\operatorname{logistic}\big(\gamma^p_{c,0}+\gamma^p_{c,1}(\Theta_c-\mu_c)\big).
\]

- `A` includes arrivals AND their line-of-origin labels. `D` includes individual rework complexities. `B` includes potential service times for both fast and standard rework resources. Preserve the actual service-time model already in the repository.
- The value `v_c` is a known deterministic line-level parameter, **not** another random component of `xi`.
- Every modeled arrival already needs rework: no extra quality-control pass/fail Bernoulli is required.
- Conditional on the physical states, future arrivals, complexities and service times do not directly depend on `z` or `eta`. Probing costs represent pre-shift downtime; do NOT silently modify the subsequent arrival process.

The candidate routing policies are a finite set `X={x_1,...,x_k}`. A typical source rule (Sec. 8.5) sends an arriving job from line `c` to the fast server when

\[
a_v^{i}v_c-a_q^{i}(q^F-q^S)\ge\tau^{i}.
\]

The selected `i` is held fixed for the whole shift. Within-shift routing applies this rule as a function of observed arrivals and queue states. Do not run a new optimization at each job arrival. Let `Y_i(omega)` denote gross reward of routing policy `i` for a complete physical shift `omega=(Theta,A,D,B)`; it can be evaluated for all candidate policies on the same recorded physical realization.

## 3. Pre-shift measurements, posterior and two-stage objective

The first-stage decision is `z in [0,1]^m`. For `z_c > 0`, diagnostic duration is `Tmax_c * z_c` and

\[
\eta_c\mid \Theta_c,z_c\sim
\mathcal N\left(\Theta_c,\;\frac{\nu_c}{T_c^{\max}z_c}\right),
\qquad \nu_c>0,\;T_c^{\max}>0.
\]

The PDF uses `nu_c` in Sec. 8.3 and also `sigma_c^2` in Sec. 8.9 for this noise-scale numerator. Treat these as the same conceptual parameter; map the actual Julia configuration explicitly. Do NOT square a stored variance again.

For probed coordinates, normal-normal conjugacy gives

\[
V_c(z_c)=\left(\frac{1}{\tau_c^2}+\frac{T_c^{\max}z_c}{\nu_c}\right)^{-1},
\qquad
m_c(\eta_c,z_c)=V_c(z_c)\left(
\frac{\mu_c}{\tau_c^2}+
\frac{T_c^{\max}z_c}{\nu_c}\eta_c\right).
\]

Thus `Theta_c | eta_c,z_c ~ Normal(m_c,V_c)`. For an unprobed coordinate `z_c=0`, there is NO `eta_c`, and the posterior equals the prior `Normal(mu_c,tau_c^2)`.

For positive effort, the marginal signal density is known exactly:

\[
\eta_c\mid z_c\sim\mathcal N\left(\mu_c,\tau_c^2+\frac{\nu_c}{T_c^{\max}z_c}\right).
\]

Define the conditional gross value of policy `i` as

\[
\mu_i(\eta,z)=\mathbb E[\,Y_i(\xi_{\mathrm{phys}})\mid\eta,z\,].
\]

Then the actual optimization objective is

\[
\boxed{J(z)=-\alpha^\top z+q(z),\qquad
q(z)=\mathbb E_{\eta\mid z}\!\left[\max_{i=1,\ldots,k}\mu_i(\eta,z)\right].}
\]

The decision after seeing `eta` is `i^*(eta,z)=argmax_i mu_i(eta,z)`. **Never replace** `E_eta[max_i mu_i(eta,z)]` by `max_i E_z[Y_i]`. They are generally different, and the latter discards the value of pre-shift information.

## 4. Complete-data density, corrected LR, and corrected score

For likelihood purposes, retain complete simulation output

\[
\xi=(\Theta,\eta,\mathcal A,\mathcal D,\mathcal B).
\]

On a fixed observation mask (all relevant `z_c > 0`), its density factors as

\[
p_z(\xi)=\pi(\Theta)\,g_z(\eta\mid\Theta)\,
 f(\mathcal A,\mathcal D,\mathcal B\mid\Theta),
\qquad
g_z(\eta\mid\Theta)=\prod_{c\,\mathrm{probed}}
\phi\left(\eta_c;\Theta_c,\frac{\nu_c}{T_c^{\max}z_c}\right).
\]

The service-time term need not be independent of arrival/complexity; it only needs to be `z`-independent conditional on the physical variables. The prior and physical-process factors cancel between two probing vectors with compatible masks. **Do not integrate out `Theta`**: the simulator has realized and stored it, even if the real controller does not observe it.

For a record generated at `zsource` and evaluated at `ztarget` (same positive-coordinate mask), define `delta_c=eta_c-Theta_c`. The **CORRECT** log likelihood ratio is

\[
\boxed{\log L(\xi;z,z^{\mathrm{src}})=
\sum_{c\,\mathrm{probed}}\left[
\frac12\log\frac{z_c}{z^{\mathrm{src}}_c}
-\frac{T_c^{\max}(z_c-z^{\mathrm{src}}_c)\,\delta_c^2}{2\nu_c}
\right].}
\]

Equivalently,

\[
L(\xi;z,z^{\mathrm{src}})=\prod_{c\,\mathrm{probed}}
\sqrt{\frac{z_c}{z^{\mathrm{src}}_c}}
\exp\!\left(-\frac{T_c^{\max}(z_c-z^{\mathrm{src}}_c)\delta_c^2}{2\nu_c}\right).
\]

Its derivative with respect to the **target** vector is

\[
\boxed{s_c(\xi;z)=\frac{\partial\log p_z(\xi)}{\partial z_c}=
\frac{1}{2z_c}-\frac{T_c^{\max}(\eta_c-\Theta_c)^2}{2\nu_c},
\quad c\ \mathrm{probed}.}
\]

For inactive coordinates in a fixed mask the directional score is not used (internally use zero as a placeholder only if documented). The gradient of the probing cost is `-alpha`, once and only once.

### What is wrong in the PDF's boxed expression (p. 27)?

- The PDF box writes `sqrt(zsource/z)`; correct is `sqrt(z/zsource)`.
- The PDF box has a **positive** exponent involving `(z-zsource)`; correct is **negative**.
- The PDF box has a **positive** quadratic term in the score; correct is **negative**.
- The score must retain distinct components `c` for an `m`-dimensional decision.
- Thus the box must NOT be copied into code verbatim. These statements correct the algebra; they do not change the Gaussian measurement model in Sec. 8.3.

Deterministic numerical reference for a single active line: `Tmax=5`, `nu=1.7`, `Theta=0.2`, `eta=0.8`, `zsource=0.3`, `ztarget=0.7` give `logLR=0.21188422431124887`, `LR=1.23600477749006`, and `score=0.18487394957983183`. Include these numbers in an isolated unit test.

Essential sanity identities (with samples drawn from `zsource`):

\[
L(\xi;z,z)=1,\quad
\mathbb E_{z^{\mathrm{src}}}[L(\xi;z,z^{\mathrm{src}})]=1,\quad
\mathbb E_z[s_c(\xi;z)]=0,\quad
\mathbb E_{z^{\mathrm{src}}}[L(\xi;z,z^{\mathrm{src}})s_c(\xi;z)]=0.
\]

## 5. Exact zero probing and compatibility of historical banks

**SOURCE:** `z_c=0` means no measurement. It is NOT a draw from a proper Gaussian with infinite variance. Consequently, the Gaussian density family in Section 4 applies only to `z_c>0`. Comparing a record with missing `eta_c` against a target that observes `eta_c`, using a density ratio of those observations, is not valid without a separately justified augmentation/change of sample space.

**DESIGN for a correct initial implementation:**

- Track an explicit `observed_mask::BitVector`, identifying active lines. Store missing `eta_c` as `missing` (or a typed sentinel with a validated mask), never as a fake sampled value.
- A historical bank can be LR-reweighted to target `z` **only if its observation mask is identical** to the target mask. Otherwise it is ineligible; report it separately as `mask_mismatch`.
- Within each active mask, use a configurable strictly positive lower bound `zmin` for active coordinates (e.g. `1e-3` as an **algorithmic bound**, not as a substitute for zero). Inactive coordinates remain exactly zero and never enter a log/division.
- For small `m`, optimize each candidate active mask separately and compare final solutions by **independent direct two-stage Monte Carlo evaluation**. For larger `m`, use a documented set of candidate masks or a distinct boundary-search strategy; do not silently enumerate exponentially many masks.
- At the all-zero mask, no signal is present, so the policy decision is made using prior information only. The standard positive-coordinate score is inapplicable to opening a new probed line.
- Explicitly state which masks were searched. For comparisons of methods, keep masks and budgets identical.

If the repository already implements a rigorous compatible zero-effort treatment, keep it after verifying its probabilistic validity. Otherwise use the above.

## 6. Selecting the routing policy conditional on a measurement

For each measurement `eta` at a target `z`, estimate

\[
\widehat\mu_i(\eta,z)\approx\mathbb E_{\Theta\mid\eta,z}
  \mathbb E_{A,D,B\mid\Theta}[Y_i(A,D,B)].
\]

**Recommended DESIGN:** Use **independent inner simulations** drawn directly from the exact Gaussian posterior from Section 3, followed by conditional arrivals/complexities/services. Evaluate **all policies on the same inner physical realizations** (CRN/pairing) and calculate `mu_hat_i` as their mean reward. Use a stable deterministic tie-breaking rule. The inner *policy-training* replications must be independent of the outer/historical *gradient-evaluation* shifts; otherwise selected-policy/maximization bias can be severe.

Alternative if compatible with existing code: a single, independent prior-generated physical bank can be reweighted at a given `eta,z` by `g_z(eta|Theta_l) / f_z(eta)`, where `f_z(eta)` is the exactly known marginal measurement density (product over active lines). This provides an unnormalized-importance-sampling estimator of conditional means. If self-normalized IS is used instead, label its finite-sample bias; do not confuse a normalized weight ratio with an ordinary LR estimator.

Separate at least three roles: (i) inner policy training, (ii) outer samples for gradient estimation, and (iii) independent final oracle/holdout evaluation. Do not let the controller see the true outer `Theta` when selecting a policy.

## 7. Two-stage score-function gradient: why eta and policy selection matter

Let `omega=(Theta,A,D,B)` and `h(omega|Theta)` denote the physical simulator. Write

\[
a_i(\eta,z)=\int Y_i(\omega)\,
\pi(\Theta)g_z(\eta\mid\Theta)h(A,D,B\mid\Theta)
\,d\Theta\,dA\,dD\,dB.
\]

The unconditional marginal of `eta` is `f_z(eta)`. Since `a_i(eta,z)=f_z(eta) mu_i(eta,z)`, the outer objective is

\[
q(z)=\int\max_i a_i(\eta,z)\,d\eta.
\]

Assuming differentiation/integration can be interchanged and a unique maximizer almost everywhere (or adopting a valid subgradient at ties), the envelope theorem yields

\[
\boxed{\nabla q(z)=\mathbb E_z\left[
Y_{i^*(\eta,z)}(\omega)\,s(\Theta,\eta;z)
\right].}
\]

Therefore

\[
\boxed{\nabla J(z)=-\alpha+\mathbb E_z\left[
Y_{i^*(\eta,z)}(\omega)\,s(\Theta,\eta;z)
\right].}
\]

This does NOT mean the objective is `max_i E[Y_i]`; the chosen policy inside the expectation **depends on each observed eta**. A fresh Monte Carlo estimate uses independent joint paths `xi_l~p_z`, chooses `i_hat(eta_l,z)` via inner training, and averages `Y_{i_hat}(omega_l)*s_l`.

Given samples generated under a different `zsource`, replace the contribution by

\[
G_{\ell}(z;z^{\mathrm{src}})=
L(\xi_\ell;z,z^{\mathrm{src}})
Y_{\widehat i(\eta_\ell,z)}(\omega_\ell)
\,s(\Theta_\ell,\eta_\ell;z).
\]

This is the building block for historical green reuse. The estimated policy must be **recomputed for the target** `z` and for the individual retained observation `eta_l`, not frozen at the source iteration. Policy-selection noise/finite inner sample size creates additional estimation error; cross-fitting/independent inner training reduces, but does not eliminate, non-smooth argmax-related finite-sample effects.

For an estimate of the gross objective, use corresponding weighted rewards `L*Y_selected`, and subtract `alpha' z` once. Keep separate a hard-objective estimate, the gradient estimator used to search, and the independently validated final value.

## 8. The five requested methods: precise operational definitions

Define a **bank** `B_tau` generated at iterate `z_tau` as `M_tau` independent complete shifts, each storing `Theta`, line-level `eta`/mask, physical event paths, and rewards for all candidate policies. Recompute target-policy choices or target-softmax weights from independent inner training when evaluating bank `B_tau` at the current `z`. Define `L_tau,l=L(xi_tau,l;z,z_tau)` and `s_tau,l=s(xi_tau,l;z)`. Use equal bank weights if all `M_tau` are the same, or weight by `M_tau / sum M_tau` if sizes differ.

### 8.1 Fresh (local SF, no historical reuse)

At every iterate `z_t`, simulate a NEW bank `B_t ~ p_{z_t}`. For each observation `eta_t,l`, estimate conditional values and choose `i_hat(eta_t,l,z_t)` using independent inner training. Then

\[
\widehat g_{\rm Fresh}(z_t)=-\alpha+
\frac1{M_t}\sum_{\ell=1}^{M_t}
Y_{\widehat i(\eta_{t,\ell},z_t)}(\omega_{t,\ell})s_{t,\ell}.
\]

The LR is identically 1, since target and source agree. Discard earlier outer banks for gradient estimation (can retain summaries for diagnostics).

### 8.2 Green (LR reuse, hard conditional argmax)

At `z_t`, simulate a new bank and retain past **compatible-mask** banks. Use

\[
\widehat g_{\rm Green}(z_t)=-\alpha+
\frac{1}{N_t}\sum_{\tau\in\mathcal H_t}
\sum_{\ell=1}^{M_\tau}
L_{\tau\ell}Y_{\widehat i(\eta_{\tau\ell},z_t)}(\omega_{\tau\ell})s_{\tau\ell},
\quad N_t=\sum_{\tau\in\mathcal H_t}M_\tau.
\]

This is ordinary multiple-source IS with a separate denominator for the source that actually generated each observation. It is NOT the old example's `max_i` of a single pooled mean across all eta's.

### 8.3 Green Softmax (LR reuse, smooth conditional-policy gradient)

For each outer observation `eta` at target `z`, let `mu_hat_i(eta,z)` be independently estimated conditional policy values, and set

\[
w_i(\eta,z;\beta)=
\frac{\exp(\beta[\widehat\mu_i(\eta,z)-\max_j\widehat\mu_j(\eta,z)])}
{\sum_j\exp(\beta[\widehat\mu_j(\eta,z)-\max_j\widehat\mu_j(\eta,z)])},\qquad \beta>0.
\]

The per-path soft contribution is

\[
G_{\rm soft}(\xi;z,z^{\rm src})=L(\xi;z,z^{\rm src})
\left(\sum_i w_i(\eta,z;\beta)Y_i(\omega)\right)s(\xi;z).
\]

Green Softmax averages these contributions across all compatible banks. As `beta -> infinity`, this recovers the hard-argmax contribution when the conditional best policy is unique. This is the paper's **gradient-smoothing heuristic** adapted to rework; without extra terms it is NOT automatically the exact gradient of a log-sum-exp-smoothed two-stage objective. Report the original hard objective with independent evaluation, not an unlabelled soft surrogate.

### 8.4 Green Variance (hard conditional argmax, variance-screened banks)

This method is a **DESIGN extension**: the PDF gives the filtering idea for softmax, not a standalone fully specified no-softmax rework algorithm. Use the hard per-path green contribution `G_hard` as in 8.2. For each bank, estimate the `m x m` sample covariance matrix of its per-path **vector** contributions, and define

\[
V_\tau^{\rm hard}(z)=\operatorname{tr}\left(\widehat{\operatorname{Cov}}_\ell
[G_{\rm hard}(\xi_{\tau\ell};z,z_\tau)]\right).
\]

Retain historical banks with `V_tau <= variance_threshold` (optional separate ESS threshold), and always keep the fresh/current bank. Average accepted bank contributions only, with the correct count denominator. Compare the raw per-replication covariance trace with the threshold: **do not divide it by M_tau**, consistent with the heuristic on p. 30. Record rejected banks and the reason. A threshold, ESS cutoff, and minimum retention policy must be configurable and fixed before independent final evaluation.

### 8.5 Green Variance + Softmax (soft contributions, variance-screened banks)

As in 8.3, compute softmax weights for each `eta` at the target `z`. For each bank define its per-replication vector `G_soft` as above; screen with

\[
V_\tau^{\rm soft}(z)=\operatorname{tr}\left(\widehat{\operatorname{Cov}}_\ell
[G_{\rm soft}(\xi_{\tau\ell};z,z_\tau)]\right).
\]

Accept banks below the threshold (always include current bank); average accepted soft contributions. **Why use covariance-aware variance?** Policies evaluated on the same path share stochastic primitives and therefore have correlated gradients. For constant weights, the actual covariance is

\[
\operatorname{Cov}\left(\sum_i w_i G_i\right)=
\sum_{i,j}w_iw_j\operatorname{Cov}(G_i,G_j),
\]

not simply `sum_i sigma_i^2 w_i`; even the independent-policy expression would use `w_i^2`. In the **rework model** the softmax weights additionally vary by `eta`, so computing the actual combined per-path contribution and then its sample covariance is the transparent default. The PDF's linear-weight expression may be included as a separately labelled **paper heuristic / diagnostic**, not reported as the exact statistical variance. Sample covariance treats estimated soft weights as fixed conditional on policy-training data; it does not capture all randomness of training the weights. Report that limitation.

For more defensible variance screening, optionally compute bank-acceptance statistics using an independent pilot subset/bank, and estimate the gradient using a disjoint evaluation subset. Charge all pilot simulations to the simulation budget. If the same samples are used for screening and gradient estimation, explicitly note the selection-induced bias. All forms of historical reuse during adaptive search can introduce additional dependence bias because past `z_tau` were chosen using previous observations; final independent oracle validation is mandatory.

## 9. Julia implementation architecture (adapt names to the actual repository)

Recommended abstractions, without prescribing repository paths:

- `ReworkConfig`: physical distribution parameters, `alpha`, `H`, policy family, `zmin`, MC budgets, random seeds, and numeric tolerances.
- `ReworkShift`: realized latent `Theta`, observed `eta` plus mask, physical arrivals/complexities/potential service times, and optionally a cached reward vector for the `k` candidate policies.
- `ReworkBank`: `z_source`, observed mask, vector of independent shifts, reward matrix `k x M`, creation iteration/seed, and per-bank diagnostics.
- `posterior_params(eta, z, config)`: exact normal-normal posterior; prior for masked-off coordinates.
- `conditional_policy_values(eta, z, config, inner_rng, n_inner)`: vector of policy means and uncertainty/pairing diagnostics; **must not consult the outer shift's true Theta**.
- `log_lr_and_score(theta, eta, z_target, z_source, config, mask)`: returns scalar `logLR` and vector `score`, checks strict positivity/mask compatibility.
- `softmax_policy_weights(mu_hat, beta)`: numerically stable, sums to 1.
- `score_contributions(bank, z_target, selector, smoothing_mode, ...)`: vector-valued per-shift contributions, gross rewards and diagnostics.
- `estimate_gradient(method, z, history, ...)`: returns gradient, hard-objective estimate, selected/rejected banks, ESS, etc.
- `search_probing(method, ...)`: shared projected-gradient ascent driver on each active mask; same initial z, step-size rule, gradient clipping (if any), and stopping criteria across methods.
- `evaluate_direct_two_stage(z, ...)`: genuinely independent direct Monte Carlo oracle using fresh outer signals and independent inner policy training/evaluation. NEVER use historical LR to validate an endpoint.

Use Julia `Random`, `Statistics`, `LinearAlgebra`, `Distributions` as appropriate; preserve and reuse existing project code. Use log-pdf differences or the closed-form logLR in Section 4. Do not silently clamp `logLR` to a finite interval: exponential clipping changes the estimator. Monitor max/min log weights, `mean(L)`, ESS, max normalized weight, finite values and overflow; surface diagnostics/errors and optionally support explicit bias-labelled truncation as a research experiment.

### Gradient update

For each mask's active coordinates, use a shared projected ascent rule, e.g.

\[
z_{t+1}=\Pi_{[z_{\min},1]^{|\mathcal C_{\rm active}|}}
\big[z_t+\gamma_t\,\widehat\nabla J(z_t)\big].
\]

Apply `-alpha` exactly once to the gradient, and `-alpha' z` exactly once to the reported net objective. The softmax temperature and variance threshold must not be tuned using the independent final test oracle. A projected step at a bound may be zero despite nonzero unprojected gradient; implement a projected-gradient stopping criterion appropriately.

### Data quality and RNG

Generate one `Theta_c` per line per SHIFT (not once per job). Generate one pre-shift measurement per PROBED line per shift. Generate arrivals and complexities conditional on the SAME latent line condition. When all policies are evaluated on a simulated shift, reuse exactly the same exogenous physical realization, including potential service times, but allow queues and decisions to diverge by policy. Distinct policy-training and gradient-evaluation RNG streams are mandatory. Distinct independent oracle seeds are mandatory. Save configurations and seeds with output artifacts.

## 10. Validation suite: required, not optional

### 10.1 Closed-form Gaussian tests

For positive target/source `z`, any fixed `theta` and `eta`, compare closed-form `logLR` with the difference of normal log-pdfs using variance `nu/(Tmax*z)` and standard deviation `sqrt(variance)`. Compare score to central finite differences of the target normal log-pdf. Test multiple lines with heterogeneous `Tmax` and `nu`; ensure vector dimension is correct.

- `logLR(z,z) == 0`, hence `L=1` pathwise.
- `score` is independent of the source `zsource`, for fixed `(theta,eta,z)`.
- Under `eta|theta,z`, empirical `E[score] ~= 0`.
- Under source simulations, empirical `E[L] ~= 1` and `E[L*score] ~= 0`.
- Under the source, `E[L * eta^2]` should approach the target `E_z[eta^2]`.
- Verify score/LR with a numerical example so tests fail for the **incorrect signs and inverted root in the original PDF box**.

### 10.2 Analytic single-line derivative (no policy selection)

For `Theta ~ N(mu,tau^2)` and `eta|Theta,z ~ N(Theta,nu/(Tmax*z))`, take the toy payoff `Y=eta^2`. Then

\[
E_z[Y]=\mu^2+\tau^2+\frac{\nu}{T^{\max}z},
\qquad
\frac{d}{dz}E_z[Y]= -\frac{\nu}{T^{\max}z^2}.
\]

The direct fresh score estimate `mean(Y*score)` and reweighted green score estimate `mean(L*Y*score)` must agree with this analytic negative derivative within MC uncertainty. This isolates the LR/SF layer from the routing model.

### 10.3 Analytic TWO-STAGE test (very important)

One latent scalar `Theta ~ N(0,tau^2)` and signal as above. Create two synthetic policies with physical reward `Y_+(Theta)=Theta` and `Y_-(Theta)=-Theta`. The optimal policy after observing `eta` is `sign(E[Theta|eta,z])`, which here equals `sign(eta)` for `z>0`. Its exact gross objective is

\[
q(z)=\sqrt{\frac2\pi}\frac{\tau^2}
{\sqrt{\tau^2+\nu/(T^{\max}z)}}.
\]

Its derivative is

\[
q'(z)=\sqrt{\frac2\pi}\frac{\tau^2(\nu/T^{\max})}
{2z^2[\tau^2+\nu/(T^{\max}z)]^{3/2}}.
\]

Check fresh and green estimates of `E[Y_selected * score]` against this analytic derivative; compare `q(z)` as well. This test catches accidentally swapping `E[max]` for `max[E]`, omitting `eta` from the target-specific selector, and using an incorrect LR/score.

### 10.4 Posterior and information-flow tests

- Posterior mean/variance match Gaussian conjugacy numerically; posterior variance decreases with effort; `z=0` returns prior and **no measurement**.
- A policy selector cannot access the outer latent `Theta` value; controlled tests should change that outer Theta while holding `(eta,z)` and the inner training RNG fixed, and obtain the same chosen policy.
- Holding a policy and physical realization fixed, its reward is independent of `eta` and `z` (the measurement selects a policy; it does not directly change the physical shift).
- Correctly resample/condition arrival and complexity via latent `Theta`; verify line-origin labels and deterministic job value `v_c`.
- A single probed line can be inactive while others are active. All-zero mask is valid as direct simulation; cross-mask LR is rejected.

### 10.5 All five method equivalences

- At iteration 1 (a single fresh bank) Fresh and Green hard-gradient contributions match exactly when they operate on the same bank and policy-training random numbers.
- If the variance filter accepts every compatible bank, Green Variance equals Green; Green Variance + Softmax equals Green Softmax, up to FP tolerance.
- As `beta` becomes very large for unique conditional best policies, Green Softmax approaches Green, and Green Variance + Softmax approaches Green Variance **if the variance filters accept identical banks**. At a finite variance threshold, acceptance sets may differ; do not claim unconditional equivalence.
- All variants use the same hard original objective for independent endpoint comparison.
- Variance statistics reflect within-path covariance across policies; a constructed case with `G_1=U`, `G_2=-U`, and `w_1=w_2=1/2` must have **zero** variance in the combined contribution, despite positive individual variances. A linear-weighted variance proxy fails this test.
- Require at least two independent outer shifts for a sample covariance; if `M_outer<2`, report variance as undefined and reject a variance-screening run with a clear error.
- Rejections by variance / ESS and incompatible masks must be counted, logged and reproducible.

### 10.6 Independent oracle and scientific reporting

Before scaling experiments, test against direct two-stage re-simulation at selected `z` values with independent nested Monte Carlo (plus CRN-based paired comparisons where appropriate). A direct oracle must independently select a policy given `eta` before running the holdout/evaluation shift, not select one global policy across shifts. Use fresh independent repetitions for standard errors and intervals. Compare endpoints using separate validation data, report matched total number of simulated **physical shifts**, inner policy-training shifts, wall-clock time, ESS and variance filters. Avoid interpreting a visually smooth reference heatmap as ground truth.

## 11. Expected deliverables from Codex

1. A short repository audit identifying the current rework simulator, policy selection logic, random-number plumbing, current tests and any mismatches with the PDF.
2. Correct complete-data LR and score routines with analytic unit tests.
3. A two-stage policy-selection interface with independent inner simulations and a direct oracle.
4. Runnable and configurable searches named exactly `Fresh`, `Green`, `GreenSoftmax`, `GreenVariance`, `GreenVarianceSoftmax` (CLI/selector spellings may be normalized but must be documented).
5. One fair comparison runner with same physical model, policy family, starting points, step rule and explicit budgets; optional graphical trajectories for `m=2`, not required for correctness.
6. Machine-readable per-iteration outputs: method, iteration, `z`, mask, original hard objective estimate, gradient components and norm, selected policy distributions, number of generated/retained/rejected banks, reason for rejection, ESS/mean-L/maximum-weight diagnostics, variance statistic, beta/threshold/step size, seeds, elapsed time, independent final oracle means and uncertainty intervals.
7. A technical README explaining source-vs-correction-vs-design choices and the zero boundary, plus a test report. Do not claim successful tests unless they actually ran.

## 12. Known statistical caveats to disclose

- The sample maximum/selection of the policy creates finite-sample bias; use independent inner policy training and outer evaluation, and study increasing inner sample sizes.
- Green reuse along an adaptive trajectory induces dependence beyond independent fixed-source importance sampling; do not claim the green estimator is exactly unbiased under adaptive source selection.
- Sample-dependent variance filtering creates further selection effects; pilot/evaluation splitting or independent holdouts provide useful sensitivity checks.
- Softmax gradients in the source notes are weighted-gradient heuristics; they are not by themselves proofs of unbiased gradients of a smoothed nested objective.
- Finite-sample importance sampling may have extremely low ESS when target and historical probing vectors are far apart. Do not silently renormalize/clamp ordinary LR terms and then call the result unbiased.
- The PDF's p. 30 variance expression is tentative and omits policy cross-covariance. Distinguish a paper-style proxy from the covariance-aware estimator used by the implementation.
- All mathematical identities requiring derivatives are stated for interior positive probing coordinates and suitable regularity / tie conditions.

## 13. Traceability summary

| Subject | Evidence in source | Implementation requirement |
|---|---|---|
| Assembly/rework model; known `v_c` | Secs. 8.1-8.2, pp. 15-17 | Keep physical arrivals/complexities/queueing and deterministic value by line |
| Gaussian pre-shift measurement and no observation at zero | Sec. 8.3, p. 18 | Posterior, explicit mask, correct variance parameter |
| Normal conjugate posterior and two-stage policy choice | Secs. 8.4-8.6, pp. 19-22 | Per-`eta` policy selector; no global `max E` shortcut |
| Complete-data factorization | Sec. 8.9, p. 25 | Store simulated `Theta`; physical factors cancel in LR |
| LR and score box (algebraically inconsistent) | Sec. 8.9, pp. 26-27 | Use corrected signs/root; validate against normal logpdf |
| Softmax of policy estimates | Notes on Softmax, pp. 27-29 | Stable conditional-policy weights; beta controls sharpness |
| Variance-informed recycling idea | Softmax + Green Variance, pp. 29-30 | Filter historical banks with disclosed variance statistic/threshold; no `1/n` in threshold comparison |
| Prior toy algorithms | `probing_three_searches_crn_oracle_surface.jl` | Reuse software ideas only: fresh banks, historical banks, ESS and independent oracle; not its old Bernoulli signal model |

