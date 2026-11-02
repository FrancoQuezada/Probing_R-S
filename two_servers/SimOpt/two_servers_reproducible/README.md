# Two-server simulation optimization with probing: 5 instances × 6 methods (reproducible)

This is a standalone Julia project. It reproduces the archived experiments 019 and 020 (Fresh, Green, Green Softmax, Green Variance), 021 (Green Variance + Softmax) and 022 (Differential Evolution) on the instances **I0252, L000, L025, L026 and L030**, together with their figures.

Replication 1 of all **30 method × instance cases** reproduces the archived results **bit for bit**, and so do the archived 21×21 objective surfaces. See *Verification* below.

The folder can be copied anywhere; nothing else from the original repository is needed.

## Model

Each sample path contains 45 jobs.
- **Arrivals:** exponential interarrival times with mean 0.8.
- **Class:** uniform on 1..4. The job is complex with probability `P_COMPLEX[c]`.
- **Value:** 0.15 + 0.16c + 0.35(U − ½), clipped to [0.05, 1].
- **Service times:** exponential, on a FAST and a STANDARD server, with means that depend on complexity.

**Probing.** The decision is z = (z3, z4) ∈ [0, 1]², the probing effort on classes 3 and 4. Probing produces a binary signal:
- P(1 | complex) = p_hit0 + (1 − p_hit0)(1 − e^(−BETA·z));
- P(1 | simple) = p_false0·e^(−BETA·z).

**Routing.** A routing policy (av, aq, as, τ) sends a job to FAST iff av·value − aq·(queue_FAST − queue_STD) + as·signal ≥ τ. There are 144 policies (3 × 3 × 4 × 4 grid).

**Reward.** The reward of a path is the revenue (`REVENUE_SCALE`·value), minus `FAST_ROUTING_COST` for each job routed to FAST, minus `DELAY_PENALTY`·(time in system).

**Objective (net):**

    J(z) = max_i E_z[ Y(x_i, ξ) ] − αᵀz

The maximum is outside the expectation: one routing policy is chosen ex ante.

## Instances (`instances/<ID>.toml`)

The `[model]` blocks are copied programmatically, at full precision, from the frozen configurations of experiments 019/020; they are identical in 021/022. The source file and its SHA-256 are noted in each instance file.

All five instances share NCLASS = 4, probing on classes 3 and 4, P_COMPLEX = (0.02, 0.12, 0.45, 0.70), interarrival mean 0.8, revenue scale 8 and FAST routing cost 0.7. They differ in:

| ID | α | BETA | P_HIT0 (classes 3, 4) | P_FALSE0 (classes 3, 4) | FAST / STD complex means | delay penalty |
|---|---|---|---|---|---|---|
| I0252 | (8.5293, 10.0839) | 4.0 | 0.55, 0.65 | 0.20, 0.25 | 1.8 / 12 | 0.84 |
| L000 | (10, 4.5) | 2.2 | 0.75, 0.92 | 0.15, 0.20 | 2.4 / 8 | 0.42 |
| L025 | (5, 2.25) | 4.0 | 0.75, 0.92 | 0.15, 0.20 | 1.8 / 12 | 0.84 |
| L026 | (7.5, 3.375) | 4.0 | 0.55, 0.65 | 0.20, 0.25 | 1.8 / 12 | 3.36 |
| L030 | (5, 2.25) | 2.2 | 0.75, 0.92 | 0.15, 0.20 | 2.4 / 8 | 3.36 |

**Shared protocol (`protocol.toml`):**
- start z0 = (0.12, 0.10), 100 iterations, banks of 500 paths × 45 jobs;
- step 0.035/√k, gradient clipping ±15, projection onto [0, 1]²;
- β_softmax = 0.5, Filter2 κ = 2;
- evaluation: trace 1000 paths, selection 10,000, holdout 50,000;
- Differential Evolution: 100 objective evaluations × 500 paths.

**Seeds.** Replication r of instance ID uses the original seeds, e.g. `Probing010-v1/<ID>/compare/<19000000+r>/search/<k>`. Every method sees the same paths within a replication, and the execution order does not matter.

## Methods (`methods.jl`)

These are identical to the validated `I0252_reproducible`.

| id | method | update direction |
|---|---|---|
| `fresh` | Fresh | LR/score gradient of the hard-max policy on the new 500-path bank only |
| `green` | Green | same, on all past banks reweighted by L = exp(clamp(log LR, ±35)) |
| `green_softmax` | Green Softmax | softmax(β = 0.5) mixture of the per-policy Green gradients |
| `green_variance` | Green Variance (Filter2) | Green on the banks whose LR-weighted variance (provisional best policy) is ≤ 2 × the current bank's |
| `green_variance_softmax` | Green Variance + Softmax | Filter2 banks, then softmax mixture; α subtracted once |
| `differential_evolution` | Differential Evolution | BlackBoxOptim 0.6.12 `adaptive_de_rand_1_bin_radiuslimited`, population 20, direct objective on 500 fixed CRN paths, 100 evaluations including z0 (best incumbent reported) |

## Install

You need Julia **1.11.7**, the version of the original runs. `Manifest.toml` pins every package to the versions of experiment 022, including BlackBoxOptim 0.6.12 and Plots 1.41.1.

```bash
cd two_servers_reproducible
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run experiments

```bash
julia --project=. run.jl --instance I0252 --method green --reps 1           # one method, one instance
julia --project=. run.jl --instance L026 --method all --reps 3              # all methods, replications 1-3
julia --project=. run.jl --instance L026 --method all --reps 4:10           # a range of replications
julia --project=. run.jl --instance L000 --method fresh --rep 7             # one replication id
julia --project=. run.jl --instance all --method all --reps 50              # complete design: 5 x 6 x 50 = 1500 runs
julia --project=. run.jl --instance all --method all --reps 1 --quick       # smoke test (~15 s)
```

**Identifiers:**
- instances: `I0252 L000 L025 L026 L030 all` (or a comma list);
- methods: `fresh green green_softmax green_variance green_variance_softmax differential_evolution all`.

**Run time.** One full replication of the six methods takes about 1.5 min per instance on one core. The complete design takes about 6–7 h on one core. Instances can run in parallel processes, as long as each process writes to a different instance directory.

**`--quick`.** It reduces only the budgets: 5 iterations × 50 paths, trace 100, selection 500, holdout 1000, DE 10 × 50, surface 5×5×100. Quick results go to `results/quick/` and are **not scientific results**; their figures carry the label "QUICK MODE: preliminary".

**Resuming.** A completed (instance, method, replication) is never recomputed or overwritten, so an interrupted campaign resumes with the same command. `run_info.toml` refuses to mix results produced with a different model or different settings.

## Outputs: `results/<full|quick>/<ID>/`

| file | content |
|---|---|
| `final_results.csv` | one row per method and replication (details below) |
| `trajectories.csv` | per iteration t (details below) |
| `trajectory_evaluations.csv` | independent direct-CRN evaluation of every visited z (details below) |
| `objective_surface.csv` | optional surface: z3, z4, objective, gross, cost, policy, SE, seed, paths |
| `run_info.toml` | instance parameters, settings, seeds protocol, versions, SHA-256 of the sources |

**`final_results.csv`:**
- the final z;
- gross reward, probing cost, **net objective** and its holdout MC standard error / 95 % interval;
- selected policy;
- optimization, trace and final times;
- effort (new, reused, simulated paths, policy-path evaluations);
- evaluation budgets and all seeds.

**`trajectories.csv`:**
- the decision zᵗ, the optimizer's internal objective and policy;
- for the gradient methods: g3, g4, ESS, LR clips, K_eff and banks used. Row t holds the estimate computed at zᵗ;
- for DE: the candidate and the incumbent after t evaluations;
- cumulative time and effort.

**`trajectory_evaluations.csv`.** Every zᵗ is evaluated on one fixed bank of 1000 paths per replication, shared by all methods (common random numbers). This is J = max_i mean Y_i − αᵀz, with no likelihood ratios.

**Effort accounting.** Effort is counted in three ways:
- **new paths:** newly generated primitive paths;
- **simulated paths:** paths on which all 144 policies were simulated. DE re-simulates its fixed bank at every evaluation;
- **reused paths:** historical paths that were only LR-reweighted by the Green methods, never re-simulated.

The three are not equal costs. Measured times come from this simplified code on the user's machine, and are not comparable with the archived timings.

## Figures

```bash
julia --project=. plots.jl --instance I0252                 # one instance (from results/full)
julia --project=. plots.jl --instance all                   # all instances + cross-instance figure
julia --project=. plots.jl --instance all --rep 3           # trajectories of replication 3
julia --project=. plots.jl --instance all --quick           # figures of the quick results
```

`plots.jl` reads only the saved CSV files. It never runs an optimizer or a simulation, and never modifies results. Figures go to `figures/<full|quick>/<ID>/` as PNG and PDF:

| figure | content |
|---|---|
| `trajectories` | the six paths in (z3, z4) for one replication (default 1); DE shows its best-incumbent path. Coordinates are never averaged across replications. |
| `convergence` | independently evaluated objective of the visited decisions: (a) gradient methods by iteration, (b) DE by objective evaluation, (c) all methods by cumulative simulated paths. Mean with pointwise 95 % Student-t band over replications (unit = replication). |
| `final_comparison` | final holdout objective: each replication with its holdout MC 95 % interval; ± 1 sd across replications; mean with 95 % t-CI. With one replication only the MC interval is shown, and the figure says so. |
| `efficiency_time` | objective vs measured cumulative optimization time |
| `efficiency_budget` | (a) objective vs cumulative simulated paths; (b) new / reused / simulated paths per run |
| `objective_heatmap` | surface J(z3, z4), with the trajectories of the chosen replication (only when the surface was computed) |
| `cross_instance_paired_vs_fresh` | per instance (own y-scale): final objective, and J(method) − J(Fresh) paired within each replication |

**About the cross-instance figure.** The paired difference to Fresh is meaningful because all methods share the selection and holdout seeds within a replication. No normalization is applied, and the original J values are shown.

### Objective surfaces (explicit and expensive)

```bash
julia --project=. run.jl --instance I0252 --surface         # 21 x 21 grid x 5000 paths, ~2-3 min per instance
julia --project=. run.jl --instance all --surface
julia --project=. plots.jl --instance all                   # heatmaps from the saved surfaces
```

**How the surface is built.**
- One bank of 5000 primitive paths, shared by every grid point.
- At each z the signals are re-materialized and all 144 policies are simulated; the best sample mean is taken and αᵀz subtracted. No LR reweighting is used.

**Seed and grid.** The seed and the grid reproduce the archived reference surfaces:
- I0252: `Probing011-v1/shared/high/0/surface/0`;
- L instances: `Probing010-v1/shared/high/0/surface/0`.

An existing surface is never overwritten. To change the grid or budget, edit `[surface]` in `protocol.toml`.

## Verification

```bash
julia --project=. test/runtests.jl                          # 252 checks, ~1 min
FULL_REGRESSION=1 julia --project=. test/runtests.jl        # + full replication 1, 30 cases vs archived, ~8 min
SURFACE_REGRESSION=1 julia --project=. test/runtests.jl     # + full 21x21x5000 surfaces vs archived, ~12 min
```

With both variables set, 382 checks pass in about 22 minutes. This was verified in a copy of this folder outside the repository.

**Reference data.** For each instance, `test/reference/<ID>/` holds reference data produced by the **original frozen code** (provenance in `PROVENANCE.toml`), plus the archived replication-1 results and the reference surface.

**Bitwise checks.** For every instance the tests check, bit for bit:
- model digest, policy grid and seeds;
- primitive paths, signals and 144×20 rewards;
- LR and scores, cost accounting;
- 4-iteration searches of the five gradient methods, the Filter2 retained banks, and 20 DE evaluations;
- direct trace, selection + holdout, and three surface points.

**Other checks:**
- the run order does not change results (Green alone = Green within `all`);
- completed runs are skipped;
- figures come from saved results and repeat identically, and plotting leaves the results untouched;
- quick figures are flagged as preliminary.

**Results of the long checks.**
- **`FULL_REGRESSION`:** compares 33,396 values (trajectories, internal estimates, gradients, LR diagnostics, trace evaluations, final holdouts) with the archived files of 019/020/021/022, with 0 differences.
- **`SURFACE_REGRESSION`:** compares the five complete surfaces (441 points each) with the archived reference surfaces, with 0 differences (19,845 values).

`test/make_reference.jl` is a maintainer tool that regenerates the reference data from the original repository; it is not needed here.

## Interpreting the numbers

**Net objective.** All objectives are net: gross reward minus αᵀz, subtracted once. Compare methods only through the independent evaluations (`trajectory_evaluations.csv` and the holdout in `final_results.csv`). Do not use the optimizers' internal estimates, which use different estimators.

**Three different uncertainties.** They should not be confused:
1. The **holdout MC standard error** is the precision of one final evaluation.
2. The **across-replication sd** is how much the final solution quality varies between independent runs.
3. The **95 % t-CI of the mean** uses the replications as the statistical unit, with n − 1 degrees of freedom.

**Effort comparisons.** One DE objective evaluation is not one gradient iteration. Efficiency is therefore compared through time and simulated paths.

## Differences from the original code (no effect on the numbers)

- Code is consolidated into `model.jl` and `methods.jl`. The Green runners' unused Fresh estimate, the history-provenance checks, checkpoint files and worker orchestration are omitted.
- The surface standard error keeps the original `collect` expression, `surface_point`. `direct` keeps the trace expression, a view. The two can differ in the last bit, so each is used where the original used it.
- The top-level `019probing_i0252_statistical.jl` in the repository is the later multi-instance version. The frozen per-experiment copies in `resultados/0*/<ID>/reproduction/` are the sources of truth.
