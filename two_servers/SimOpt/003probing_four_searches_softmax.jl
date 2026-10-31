# 003probing_four_searches_softmax.jl
# Four searches for J(z) = max_i E_z[Y_i] - ALPHA'z, z=(z3,z4) in [0,1]^2.
# Adds Fresh + Softmax from eq. (2), Notes on Softmax, Probing_R_S.pdf p. 25.
# No outer eta stage is introduced. Only Softmax's search gradient is smoothed;
# all four trajectories are evaluated using the ORIGINAL hard-max objective
# on ONE independent common CRN bank. The oracle heat map is direct CRN simulation.
#
# The model, policy grid, LR, projected step, and all THREE original searches
# below are copied unchanged from 002probing_three_searches_n500_full45iters.jl.
# The original file is neither included nor modified; its automatic main() is
# therefore never triggered by this file. This new file is safe to include.
#
# Default experiment: 500 paths/bank, 45 jobs/path, 100 iterations, beta_soft=0.5.
# Fresh and Softmax share iteration seeds but materialize signals at their own z.
# Fixed generates 500 search paths total; the other methods generate 50000 each.
# Outputs use probing_four_* names and never overwrite probing_three_* files.
# Dependency: Plots; other imports are Julia standard libraries.
# Run: env GKSwstype=100 OPENBLAS_NUM_THREADS=1 julia --project=<environment> 003probing_four_searches_softmax.jl

using Random
using Statistics
using Printf
using LinearAlgebra
using Plots
using Serialization
using TOML
using SHA
using Dates

# =========================
# 1. PARAMETERS
# =========================

const NCLASS = 4
const P_COMPLEX = [0.02, 0.12, 0.45, 0.70]
const P_HIT0   = [0.60, 0.65, 0.75, 0.92]
const P_FALSE0 = [0.05, 0.08, 0.15, 0.20]
const BETA = 2.20

# Eq. (2), Notes on Softmax, Probing_R_S.pdf, p. 25.
# Independent of BETA, which controls probing-signal accuracy.
const SOFTMAX_BETA = 0.5

const MEAN_INTERARRIVAL = 0.80
const FAST_MEAN = (0.75, 2.40)
const STD_MEAN  = (1.00, 8.00)

const REVENUE_SCALE = 8.0
const FAST_ROUTING_COST = 0.70
const DELAY_PENALTY = 0.42

const ACTIVE_CLASSES = (3, 4)
const ALPHA = [10.0, 4.5]
# Other possible choices (10,2.6),(10,3.5),(9,4.5)

# Same n for each newly generated bank in all three methods.
# Coauthor robustness experiment: use only 500 replications per bank.
# Thus:
#   FIXED uses 500 search paths total;
#   FRESH uses 500 new paths per iteration;
#   GREEN uses 500 new paths per iteration and accumulates them.
const JOBS_PER_PATH = 45
const N_PATHS = 500

# Separate deterministic seed streams.
const SEED_FIXED = 20260720
const SEED_FRESH = 20260721
const SEED_SOFTMAX = SEED_FRESH  # paired CRN stream with Fresh
const SEED_GREEN = 20260722
const SEED_SURFACE = 20260723

const GRID_N = 41

# Independent direct-resimulation ("oracle") evaluation.
# At each evaluated z, signals are generated directly under z, so no LR is used.
const ORACLE_PATHS_FINAL = 30000
const ORACLE_SEED = 20260724

# Oracle heat-map controls. This affects ONLY the background surface;
# the three optimization trajectories are unchanged.
const ORACLE_GRID_N = 21
const ORACLE_PATHS_PER_GRID_POINT = 5000
const ORACLE_SURFACE_SEED = 20260725

# Common independent evaluation bank used ONLY to plot objective value vs iteration.
# Every visited point from all three searches is evaluated on the SAME CRN primitives,
# so the curves are directly comparable and are not contaminated by the fact that
# FIXED/FRESH/GREEN use different search estimators.
const TRACE_EVAL_PATHS = 5000
const TRACE_EVAL_SEED = 20260726

const Z0 = [0.12, 0.10]
const MAX_ITERS = 100
const BASE_STEP = 0.035
const GRAD_CLIP = 15.0
const STEP_TOL = 2e-4  # retained for reference; early stopping is disabled in the three searches

# Unlike the original fixed-bank code, we do NOT use objective backtracking for
# the three-way comparison. Fresh/green estimators change as new banks arrive,
# so using a fixed-bank line search for one method but not the others would make
# the search rules incomparable. All three use exactly the same projected,
# diminishing gradient step.
const USE_COMMON_STEP_RULE = true

# =========================
# 2. DATA STRUCTURES
# =========================

struct JobPrimitive
    arrival::Float64
    class::Int
    complex::Bool
    value::Float64
    service_fast::Float64
    service_std::Float64
    signal::Int
end

struct SamplePath
    jobs::Vector{JobPrimitive}
end

struct Policy
    av::Float64
    aq::Float64
    as::Float64
    tau::Float64
end

struct HistoricalBank
    zgen::Vector{Float64}   # 2-D (z3,z4) at which signals were generated
    bank::Vector{SamplePath}
    Y::Matrix{Float64}
end

# =========================
# 3. BASIC HELPERS
# =========================

randexp_mean(rng, mean::Float64) = -mean * log(rand(rng))

function p_hit(c::Int, zc::Real)
    p0 = P_HIT0[c]
    return p0 + (1.0 - p0) * (1.0 - exp(-BETA * Float64(zc)))
end

function p_false(c::Int, zc::Real)
    return P_FALSE0[c] * exp(-BETA * Float64(zc))
end

signal_probability(c::Int, complex::Bool, zc::Real) =
    complex ? p_hit(c, zc) : p_false(c, zc)

function full_z(z2::AbstractVector{<:Real})
    zz = zeros(Float64, 4)
    zz[ACTIVE_CLASSES[1]] = z2[1]
    zz[ACTIVE_CLASSES[2]] = z2[2]
    return zz
end

# =========================
# 4. GENERATE A BANK UNDER AN ARBITRARY z
# =========================

function generate_path(rng::AbstractRNG, zgen::AbstractVector{<:Real};
                       njobs::Int=JOBS_PER_PATH)
    zz = full_z(zgen)
    jobs = Vector{JobPrimitive}(undef, njobs)
    t = 0.0

    for r in 1:njobs
        t += randexp_mean(rng, MEAN_INTERARRIVAL)

        c = rand(rng, 1:NCLASS)
        d = rand(rng) < P_COMPLEX[c]

        basev = 0.15 + 0.16 * c
        v = clamp(basev + 0.35 * (rand(rng) - 0.5), 0.05, 1.0)

        muf = d ? FAST_MEAN[2] : FAST_MEAN[1]
        mus = d ? STD_MEAN[2]  : STD_MEAN[1]

        sf = randexp_mean(rng, muf)
        ss = randexp_mean(rng, mus)

        # KEY DIFFERENCE FROM THE OLD CODE:
        # signal is generated under zgen, not hard-coded at z=0.
        ps = signal_probability(c, d, zz[c])
        s = rand(rng) < ps ? 1 : 0

        jobs[r] = JobPrimitive(t, c, d, v, sf, ss, s)
    end

    return SamplePath(jobs)
end

function generate_bank_at(zgen::AbstractVector{<:Real};
                          npaths::Int=N_PATHS,
                          seed::Int)
    rng = MersenneTwister(seed)
    return [generate_path(rng, zgen) for _ in 1:npaths]
end

# =========================
# 5. POLICY FAMILY
# =========================

function build_policy_grid()
    policies = Policy[]
    for av in (0.4, 0.8, 1.2),
        aq in (0.15, 0.35, 0.65),
        as in (0.0, 0.6, 1.2, 2.0),
        tau in (0.10, 0.45, 0.80, 1.15)
        push!(policies, Policy(av, aq, as, tau))
    end
    return policies
end

# =========================
# 6. POLICY REWARD ON A REALIZED PATH
# =========================

function evaluate_policy(policy::Policy, path::SamplePath)
    free_fast = 0.0
    free_std  = 0.0
    total = 0.0

    for job in path.jobs
        t = job.arrival
        qf = max(free_fast - t, 0.0)
        qs = max(free_std  - t, 0.0)

        score = policy.av * job.value -
                policy.aq * (qf - qs) +
                policy.as * job.signal

        route_fast = score >= policy.tau

        if route_fast
            start = max(t, free_fast)
            finish = start + job.service_fast
            free_fast = finish
            time_in_system = finish - t

            total += REVENUE_SCALE * job.value
            total -= FAST_ROUTING_COST
            total -= DELAY_PENALTY * time_in_system
        else
            start = max(t, free_std)
            finish = start + job.service_std
            free_std = finish
            time_in_system = finish - t

            total += REVENUE_SCALE * job.value
            total -= DELAY_PENALTY * time_in_system
        end
    end
    return total
end

function precompute_rewards(policies, bank; verbose::Bool=false)
    npol = length(policies)
    npaths = length(bank)
    Y = Matrix{Float64}(undef, npol, npaths)

    verbose && @printf("Precomputing Y: %d policies x %d paths...\n", npol, npaths)

    for i in 1:npol
        for ell in 1:npaths
            Y[i, ell] = evaluate_policy(policies[i], bank[ell])
        end
    end
    return Y
end

# =========================
# 7. GENERAL LR: target z / generating z
# =========================

"""
Return
    logL  = log f(xi; ztarget) / f(xi; zsource)
    score = grad_{ztarget} log f(xi; ztarget)

This generalizes the old code, whose source was always z=(0,0).
"""
function loglr_and_score(path::SamplePath,
                         ztarget::AbstractVector{<:Real},
                         zsource::AbstractVector{<:Real})
    zt = full_z(ztarget)
    zs = full_z(zsource)

    logL = 0.0
    score = zeros(2)

    for job in path.jobs
        c = job.class
        active_pos = c == ACTIVE_CLASSES[1] ? 1 :
                     c == ACTIVE_CLASSES[2] ? 2 : 0
        active_pos == 0 && continue

        pt = clamp(signal_probability(c, job.complex, zt[c]), 1e-12, 1.0 - 1e-12)
        ps = clamp(signal_probability(c, job.complex, zs[c]), 1e-12, 1.0 - 1e-12)

        if job.signal == 1
            logL += log(pt) - log(ps)
        else
            logL += log1p(-pt) - log1p(-ps)
        end

        dp =
            if job.complex
                (1.0 - P_HIT0[c]) * BETA * exp(-BETA * zt[c])
            else
                -P_FALSE0[c] * BETA * exp(-BETA * zt[c])
            end

        score[active_pos] += job.signal == 1 ? dp / pt : -dp / (1.0 - pt)
    end

    return logL, score
end

# =========================
# 8. ONE-BANK ESTIMATOR
# =========================

"""
Estimate objective/gradient at ztarget from a bank generated under zsource.

If ztarget == zsource, L == 1 pathwise. This is the FRESH-LOCAL estimator.
If zsource == (0,0), this is the original FIXED-BASELINE LR estimator.
"""
function one_bank_estimator(
    ztarget::AbstractVector{<:Real},
    zsource::AbstractVector{<:Real},
    bank,
    Y;
    need_gradient::Bool=true
)
    npaths = length(bank)
    npol = size(Y, 1)

    logL = Vector{Float64}(undef, npaths)
    scores = need_gradient ? Matrix{Float64}(undef, 2, npaths) : zeros(0, 0)

    for ell in 1:npaths
        ll, sc = loglr_and_score(bank[ell], ztarget, zsource)
        logL[ell] = ll
        if need_gradient
            scores[:, ell] .= sc
        end
    end

    L = exp.(clamp.(logL, -35.0, 35.0))

    policy_values = Vector{Float64}(undef, npol)
    for i in 1:npol
        policy_values[i] = mean(@view(Y[i, :]) .* L)
    end

    best_i = argmax(policy_values)
    gross = policy_values[best_i]
    J = gross - dot(ALPHA, ztarget)

    if !need_gradient
        return J, best_i, gross, mean(L), std(L)
    end

    g = zeros(2)
    ystar = @view Y[best_i, :]
    for j in 1:2
        g[j] = mean(ystar .* L .* @view(scores[j, :])) - ALPHA[j]
    end

    return J, g, best_i, gross, mean(L), std(L)
end

# =========================
# 9. GREEN ESTIMATOR
# =========================

"""
Green estimator at current z using ALL historical banks.

For each historical bank tau generated under z^tau:
  - reweight to current z by f(xi;z)/f(xi;z^tau)
  - compute policy values
Average policy values equally across banks, choose one common best policy i*,
then average the LR-score gradient contributions for that SAME i* across banks.

This implements:
  -alpha + (1/T) sum_tau (1/n) sum_ell
      Y(i*, xi_{tau,ell}) L_{tau,ell}(z) score_{tau,ell}(z).
"""
function green_estimator(z::AbstractVector{<:Real},
                         history::Vector{HistoricalBank})
    T = length(history)
    npol = size(history[1].Y, 1)

    policy_values = zeros(npol)

    # Store each bank's LR and score because after selecting the common i*
    # we need them again for the gradient.
    Ls = Vector{Vector{Float64}}(undef, T)
    Scores = Vector{Matrix{Float64}}(undef, T)

    meanLs = zeros(T)
    stdLs = zeros(T)

    for tau in 1:T
        hb = history[tau]
        npaths = length(hb.bank)
        logL = Vector{Float64}(undef, npaths)
        scores = Matrix{Float64}(undef, 2, npaths)

        for ell in 1:npaths
            ll, sc = loglr_and_score(hb.bank[ell], z, hb.zgen)
            logL[ell] = ll
            scores[:, ell] .= sc
        end

        L = exp.(clamp.(logL, -35.0, 35.0))
        Ls[tau] = L
        Scores[tau] = scores
        meanLs[tau] = mean(L)
        stdLs[tau] = std(L)

        for i in 1:npol
            policy_values[i] += mean(@view(hb.Y[i, :]) .* L) / T
        end
    end

    best_i = argmax(policy_values)
    gross = policy_values[best_i]
    J = gross - dot(ALPHA, z)

    g = -copy(ALPHA)
    for tau in 1:T
        ystar = @view history[tau].Y[best_i, :]
        L = Ls[tau]
        scores = Scores[tau]
        for j in 1:2
            g[j] += mean(ystar .* L .* @view(scores[j, :])) / T
        end
    end

    return J, g, best_i, gross, mean(meanLs), mean(stdLs)
end

# =========================
# 10. COMMON STEP RULE
# =========================

function projected_step(z, g, k)
    gc = clamp.(g, -GRAD_CLIP, GRAD_CLIP)
    eta = BASE_STEP / sqrt(k)
    znew = clamp.(z .+ eta .* gc, 0.0, 1.0)
    return znew, eta, gc
end

# =========================
# 11A. SEARCH 1: FIXED z=0 BANK
# =========================

function optimize_fixed(z0, policies)
    println("\n==============================================================")
    println(" SEARCH 1: FIXED BASELINE BANK AT z=(0,0)")
    println("==============================================================")

    zsource = [0.0, 0.0]
    bank = generate_bank_at(zsource; seed=SEED_FIXED)
    Y = precompute_rewards(policies, bank; verbose=true)

    z = clamp.(copy(z0), 0.0, 1.0)
    zhist = Vector{Vector{Float64}}()
    Jhist = Float64[]
    ihist = Int[]

    for k in 1:MAX_ITERS
        J, g, best_i, _, meanL, stdL =
            one_bank_estimator(z, zsource, bank, Y)

        push!(zhist, copy(z))
        push!(Jhist, J)
        push!(ihist, best_i)

        @printf("FIXED iter %2d | z=(%.4f,%.4f) | J=%10.4f | g=(% .4f,% .4f) | pol=%3d | E[L]=%.3f SD[L]=%.3f\n",
                k, z[1], z[2], J, g[1], g[2], best_i, meanL, stdL)

        znew, eta, gc = projected_step(z, g, k)

        # For the three-way comparison, always run the full MAX_ITERS.
        # In particular, a projected step can be zero at the boundary even when
        # later noisy/LR gradient estimates might point back into the feasible set.
        # Early stopping would also give the methods unequal iteration budgets.
        z = znew
    end

    J, _, best_i, _, _, _ = one_bank_estimator(z, zsource, bank, Y)
    push!(zhist, copy(z)); push!(Jhist, J); push!(ihist, best_i)

    return z, J, zhist, Jhist, ihist, bank, Y
end

# =========================
# 11B. SEARCH 2: FRESH LOCAL BANK
# =========================

function optimize_fresh(z0, policies)
    println("\n==============================================================")
    println(" SEARCH 2: FRESH LOCAL BANK AT EACH ITERATE")
    println("==============================================================")

    z = clamp.(copy(z0), 0.0, 1.0)
    zhist = Vector{Vector{Float64}}()
    Jhist = Float64[]
    ihist = Int[]

    lastJ = NaN

    for k in 1:MAX_ITERS
        # New independent bank generated under CURRENT z.
        seed = SEED_FRESH + 10000 * k
        bank = generate_bank_at(z; seed=seed)
        Y = precompute_rewards(policies, bank)

        # source == target, hence L == 1 exactly.
        J, g, best_i, _, meanL, stdL =
            one_bank_estimator(z, z, bank, Y)

        push!(zhist, copy(z))
        push!(Jhist, J)
        push!(ihist, best_i)
        lastJ = J

        @printf("FRESH iter %2d | z=(%.4f,%.4f) | J=%10.4f | g=(% .4f,% .4f) | pol=%3d | E[L]=%.3f SD[L]=%.3f\n",
                k, z[1], z[2], J, g[1], g[2], best_i, meanL, stdL)

        znew, eta, gc = projected_step(z, g, k)

        # For the three-way comparison, always run the full MAX_ITERS.
        # In particular, a projected step can be zero at the boundary even when
        # later noisy/LR gradient estimates might point back into the feasible set.
        # Early stopping would also give the methods unequal iteration budgets.
        z = znew
    end

    # Do not generate an extra bank just to append a final noisy point.
    # The reported endpoint is the last decision reached by the search.
    if norm(z - zhist[end]) > 1e-12
        push!(zhist, copy(z))
        push!(Jhist, lastJ)   # plotting placeholder; trajectory is the main comparison
        push!(ihist, ihist[end])
    end

    return z, lastJ, zhist, Jhist, ihist
end

# =========================
# 11C. SEARCH 3: GREEN SIMULATION
# =========================

function optimize_green(z0, policies)
    println("\n==============================================================")
    println(" SEARCH 3: GREEN SIMULATION (ACCUMULATE ALL BANKS)")
    println("==============================================================")

    z = clamp.(copy(z0), 0.0, 1.0)
    history = HistoricalBank[]

    zhist = Vector{Vector{Float64}}()
    Jhist = Float64[]
    ihist = Int[]
    lastJ = NaN

    for k in 1:MAX_ITERS
        # Generate a NEW bank under current z and retain it forever.
        seed = SEED_GREEN + 10000 * k
        bank = generate_bank_at(z; seed=seed)
        Y = precompute_rewards(policies, bank)
        push!(history, HistoricalBank(copy(z), bank, Y))

        # Estimate at current z using ALL banks collected so far.
        J, g, best_i, _, meanL, meanStdL = green_estimator(z, history)

        push!(zhist, copy(z))
        push!(Jhist, J)
        push!(ihist, best_i)
        lastJ = J

        @printf("GREEN iter %2d | z=(%.4f,%.4f) | J=%10.4f | g=(% .4f,% .4f) | pol=%3d | banks=%2d | avg E[L]=%.3f avg SD[L]=%.3f\n",
                k, z[1], z[2], J, g[1], g[2], best_i, length(history), meanL, meanStdL)

        znew, eta, gc = projected_step(z, g, k)

        # For the three-way comparison, always run the full MAX_ITERS.
        # In particular, a projected step can be zero at the boundary even when
        # later noisy/LR gradient estimates might point back into the feasible set.
        # Early stopping would also give the methods unequal iteration budgets.
        z = znew
    end

    if norm(z - zhist[end]) > 1e-12
        push!(zhist, copy(z))
        push!(Jhist, lastJ)
        push!(ihist, ihist[end])
    end

    return z, lastJ, zhist, Jhist, ihist, history
end

# =========================
# 12. COMMON REFERENCE SURFACE
# =========================

# =========================
# 12A. CRN PRIMITIVES FOR THE ORACLE SURFACE
# =========================
#
# We generate the exogenous primitives ONCE and reuse them at every grid point:
# arrivals, classes, degradation/complexity states, values, service times, and
# one U(0,1) signal uniform per job.
#
# At each z, the signal is reconstructed as
#     S_r(z) = 1{ U_r <= p_r(z) }.
#
# Therefore each grid point has exactly the correct direct-simulation marginal
# distribution under that z. NO likelihood ratio is used. The common uniforms
# merely induce positive correlation across grid points (CRN), reducing noise
# in differences between neighboring z values.

struct CRNJobPrimitive
    arrival::Float64
    class::Int
    complex::Bool
    value::Float64
    service_fast::Float64
    service_std::Float64
    u_signal::Float64
end

struct CRNPath
    jobs::Vector{CRNJobPrimitive}
end

function generate_crn_primitive_path(rng::AbstractRNG;
                                     njobs::Int=JOBS_PER_PATH)
    jobs = Vector{CRNJobPrimitive}(undef, njobs)
    t = 0.0

    for r in 1:njobs
        t += randexp_mean(rng, MEAN_INTERARRIVAL)

        c = rand(rng, 1:NCLASS)
        d = rand(rng) < P_COMPLEX[c]

        basev = 0.15 + 0.16 * c
        v = clamp(basev + 0.35 * (rand(rng) - 0.5), 0.05, 1.0)

        muf = d ? FAST_MEAN[2] : FAST_MEAN[1]
        mus = d ? STD_MEAN[2]  : STD_MEAN[1]

        sf = randexp_mean(rng, muf)
        ss = randexp_mean(rng, mus)

        # Store the uniform, NOT the realized signal.
        u = rand(rng)

        jobs[r] = CRNJobPrimitive(t, c, d, v, sf, ss, u)
    end

    return CRNPath(jobs)
end

function generate_crn_primitive_bank(; npaths::Int=ORACLE_PATHS_PER_GRID_POINT,
                                     seed::Int=ORACLE_SURFACE_SEED)
    rng = MersenneTwister(seed)
    return [generate_crn_primitive_path(rng) for _ in 1:npaths]
end

function materialize_crn_path(path::CRNPath,
                              z::AbstractVector{<:Real})
    zz = full_z(z)
    jobs = Vector{JobPrimitive}(undef, length(path.jobs))

    for r in eachindex(path.jobs)
        jp = path.jobs[r]
        ps = signal_probability(jp.class, jp.complex, zz[jp.class])
        s = jp.u_signal < ps ? 1 : 0

        jobs[r] = JobPrimitive(
            jp.arrival,
            jp.class,
            jp.complex,
            jp.value,
            jp.service_fast,
            jp.service_std,
            s
        )
    end

    return SamplePath(jobs)
end

function materialize_crn_bank(crn_bank,
                              z::AbstractVector{<:Real})
    return [materialize_crn_path(path, z) for path in crn_bank]
end

function build_oracle_surface(policies)
    println("\nBuilding DIRECT-RESIMULATION ORACLE heat map with CRN...")
    @printf("grid = %d x %d, CRN paths = %d\n",
            ORACLE_GRID_N, ORACLE_GRID_N, ORACLE_PATHS_PER_GRID_POINT)
    println("One common primitive bank is reused across ALL grid points.")
    println("Signals are regenerated correctly at each z using common uniforms.")
    println("No likelihood ratios are used for the oracle surface.")

    # ONE common primitive bank for the entire surface.
    crn_bank = generate_crn_primitive_bank(
        npaths=ORACLE_PATHS_PER_GRID_POINT,
        seed=ORACLE_SURFACE_SEED
    )

    grid = range(0.0, 1.0; length=ORACLE_GRID_N)
    Z = Matrix{Float64}(undef, ORACLE_GRID_N, ORACLE_GRID_N)

    for (ix, z3) in enumerate(grid)
        for (iy, z4) in enumerate(grid)
            z = [z3, z4]

            # Direct simulation under z using the SAME underlying primitive
            # randomness at every grid point. No LR.
            bank_z = materialize_crn_bank(crn_bank, z)
            Yz = precompute_rewards(policies, bank_z)

            vals = vec(mean(Yz; dims=2))
            gross = maximum(vals)
            Z[iy, ix] = gross - dot(ALPHA, z)
        end
        @printf("  CRN oracle heatmap progress: %d/%d rows\n", ix, ORACLE_GRID_N)
    end

    return grid, Z
end

# =========================
# 12B. COMMON CRN OBJECTIVE TRACE ALONG THE SEARCHES
# =========================
#
# IMPORTANT:
# The J values produced internally by the three search algorithms are based on
# different estimators:
#   FIXED: one fixed LR bank,
#   FRESH: a new local bank each iteration,
#   GREEN: all accumulated historical banks.
#
# Plotting those raw Jhist curves is useful diagnostically, but it is not a clean
# apples-to-apples performance comparison. Therefore we also evaluate EVERY visited
# z on one independent common CRN primitive bank. At each z we directly reconstruct
# signals under z, optimize over the same policy family, and subtract probing cost.
# No likelihood ratio is used in this evaluation.

function evaluate_trace_on_common_crn(zhist, policies, crn_bank)
    Jtrace = Vector{Float64}(undef, length(zhist))

    for k in eachindex(zhist)
        z = zhist[k]
        bank_z = materialize_crn_bank(crn_bank, z)
        Yz = precompute_rewards(policies, bank_z)
        vals = vec(mean(Yz; dims=2))
        Jtrace[k] = maximum(vals) - dot(ALPHA, z)
    end

    return Jtrace
end


# Direct oracle evaluation: unchanged from the original experiment.
function oracle_evaluate(z, policies;
                         npaths::Int=ORACLE_PATHS_FINAL,
                         seed::Int=ORACLE_SEED)
    bank = generate_bank_at(z; npaths=npaths, seed=seed)
    Y = precompute_rewards(policies, bank)

    vals = vec(mean(Y; dims=2))
    best_i = argmax(vals)
    gross = vals[best_i]
    J = gross - dot(ALPHA, z)
    se = std(@view(Y[best_i, :])) / sqrt(npaths)

    return J, gross, best_i, se
end

# =========================
# 15. FRESH + SOFTMAX: equation (2), without an outer eta expectation
# =========================

"""Stable gap-based Softmax weights. beta is independent of signal-model BETA."""
function softmax_weights(values::AbstractVector{<:Real}; beta::Real=SOFTMAX_BETA)
    isfinite(beta) && beta > 0 || throw(ArgumentError("Softmax beta must be finite and positive"))
    isempty(values) && throw(ArgumentError("Softmax requires at least one policy"))
    all(isfinite, values) || throw(ArgumentError("Policy values must be finite"))
    weights = exp.(beta .* (values .- maximum(values)))
    weights ./= sum(weights)
    all(w -> isfinite(w) && w >= 0, weights) || error("Invalid Softmax weights")
    isapprox(sum(weights), 1.0; atol=1e-12, rtol=1e-12) || error("Softmax weights do not sum to one")
    return weights
end

"""
Combine policy-specific gross gradients according to eq. (2).
The deterministic probing cost is subtracted ONCE, after the convex combination.
The log-sum-exp surrogate is diagnostic only; evaluation always uses hard max.
For fixed beta, these are log-sum-exp weights, not argmax-probability estimates.
"""
function combine_softmax_gradients(values, policy_gradients; beta::Real=SOFTMAX_BETA)
    size(policy_gradients) == (length(values), length(ALPHA)) ||
        throw(DimensionMismatch("Expected one gradient row per policy"))
    all(isfinite, policy_gradients) || error("Nonfinite policy gradient")
    weights = softmax_weights(values; beta=beta)
    gradient = transpose(policy_gradients) * weights - ALPHA
    all(isfinite, gradient) || error("Nonfinite Softmax gradient")
    vmax = maximum(values)
    surrogate_gross = vmax + log(sum(exp.(beta .* (values .- vmax)))) / beta
    return (gradient=gradient, weights=weights,
            max_weight=maximum(weights), effective_policies=1 / sum(abs2, weights),
            surrogate_gross=surrogate_gross)
end

"""
Fresh-local Softmax estimator. bank must have been generated UNDER z.
For each path, loglr_and_score(path,z,z) has log L=0 exactly. Reuse the existing
reward matrix and score function; no additional paths or rewards are simulated.
G[i,j] = mean_l Y[i,l] score[j,l]. There is no per-policy cost in G.
The reported objective is max_i mean(Y[i,:])-ALPHA'z, not log-sum-exp.
"""
function softmax_estimator(z, bank, Y; beta::Real=SOFTMAX_BETA)
    npaths = length(bank)
    npaths > 0 || throw(ArgumentError("A nonempty local bank is required"))
    size(Y, 2) == npaths || throw(DimensionMismatch("Reward columns must match paths"))
    scores = Matrix{Float64}(undef, 2, npaths)
    for ell in eachindex(bank)
        logL, score = loglr_and_score(bank[ell], z, z)
        logL == 0.0 || error("Fresh-local likelihood ratio must be exactly one")
        scores[:, ell] .= score
    end
    values = vec(mean(Y; dims=2))
    policy_gradients = (Y * transpose(scores)) / npaths
    combined = combine_softmax_gradients(values, policy_gradients; beta=beta)
    best_i = argmax(values)
    cost = dot(ALPHA, z)
    return (objective=values[best_i] - cost, gradient=combined.gradient,
            best_i=best_i, gross=values[best_i],
            surrogate=combined.surrogate_gross - cost,
            max_weight=combined.max_weight, effective_policies=combined.effective_policies,
            weights=combined.weights)
end

function optimize_softmax(z0, policies; beta::Real=SOFTMAX_BETA)
    println("\n==============================================================")
    println(" SEARCH 4: FRESH + SOFTMAX (EQ. 2)")
    @printf(" beta_soft = %.6g; iteration seeds paired with FRESH\n", beta)
    println("==============================================================")
    z = clamp.(copy(z0), 0.0, 1.0)
    zhist = Vector{Vector{Float64}}()
    Jhist = Float64[]
    ihist = Int[]
    diagnostics = NamedTuple[]
    lastJ = NaN

    for k in 1:MAX_ITERS
        seed = SEED_SOFTMAX + 10000 * k
        bank = generate_bank_at(z; seed=seed)
        Y = precompute_rewards(policies, bank)
        est = softmax_estimator(z, bank, Y; beta=beta)
        push!(zhist, copy(z))
        push!(Jhist, est.objective)
        push!(ihist, est.best_i)
        lastJ = est.objective
        znew, eta, gc = projected_step(z, est.gradient, k)
        push!(diagnostics, (iteration=k, seed=seed, z=copy(z),
                           gradient=copy(est.gradient), clipped_gradient=copy(gc),
                           step=eta, max_weight=est.max_weight,
                           effective_policies=est.effective_policies,
                           surrogate=est.surrogate))
        @printf("SOFTMAX iter %2d | z=(%.4f,%.4f) | J=%10.4f | g=(% .4f,% .4f) | pol=%3d | max w=%.5f | K_eff=%.3f\n",
                k, z[1], z[2], est.objective, est.gradient[1], est.gradient[2],
                est.best_i, est.max_weight, est.effective_policies)
        flush(stdout)
        z = znew
    end

    # Match Fresh's accounting: no extra simulation at the terminal decision.
    # Its raw J/policy are placeholders; independent CRN/oracle evaluate final z.
    if norm(z - zhist[end]) > 1e-12
        push!(zhist, copy(z))
        push!(Jhist, lastJ)
        push!(ihist, ihist[end])
    end
    return (z=z, J=lastJ, zhist=zhist, Jhist=Jhist, ihist=ihist,
            diagnostics=diagnostics, beta=Float64(beta))
end

# =========================
# 16. ONE COMMON CRN BANK AND THE UNMODIFIED HARD-MAX OBJECTIVE
# =========================

function build_common_objective_traces(histF, histR, histG, histS, policies)
    println("\nEvaluating FOUR trajectories on ONE independent common CRN bank...")
    @printf("paths = %d, seed = %d; objective remains max_i E[Y_i] - alpha'z\n",
            TRACE_EVAL_PATHS, TRACE_EVAL_SEED)
    flush(stdout)
    crn_bank = generate_crn_primitive_bank(npaths=TRACE_EVAL_PATHS, seed=TRACE_EVAL_SEED)
    histories = (histF, histR, histG, histS)
    names = ("FIXED", "FRESH", "GREEN", "SOFTMAX")
    traces = ntuple(4) do i
        # The SAME object crn_bank is passed to all four evaluations.
        trace = evaluate_trace_on_common_crn(histories[i], policies, crn_bank)
        @printf("  %s: %d visited points evaluated\n", names[i], length(trace))
        flush(stdout)
        trace
    end
    return traces
end

function oracle_compare_endpoints(zF, zR, zG, zS, policies)
    println("\n================================================================")
    println(" INDEPENDENT DIRECT ORACLE EVALUATION: FOUR FINAL POINTS")
    println("================================================================")
    @printf("%d paths per endpoint; no LR; seeds +1,+2,+3 unchanged, Softmax +4\n", ORACLE_PATHS_FINAL)
    flush(stdout)
    zs = (zF, zR, zG, zS)
    names = ("FIXED", "FRESH", "GREEN", "SOFTMAX")
    results = ntuple(4) do i
        J, gross, policy, se = oracle_evaluate(zs[i], policies; seed=ORACLE_SEED+i)
        @printf("%s | z=(%.6f, %.6f) | J_oracle=%10.4f | gross=%10.4f | policy=%3d | SE≈%.4f\n",
                names[i], zs[i][1], zs[i][2], J, gross, policy, se)
        flush(stdout)
        (method=names[i], z=copy(zs[i]), J=J, gross=gross, policy=policy, se=se,
         seed=ORACLE_SEED+i)
    end
    best = argmax([r.J for r in results])
    @printf("Largest endpoint estimate in this run: %s (J=%.4f)\n", names[best], results[best].J)
    println("SE bars are for selected-policy means, without adjustment for policy selection.")
    return results
end

# =========================
# 17. FOUR-METHOD FIGURES (CONSISTENT COLORS AND ADEQUATE MARGINS)
# =========================

const METHOD_LABELS = ("Fixed", "Fresh", "Green", "Softmax")
const METHOD_COLORS = (:royalblue, :darkorange, :forestgreen, :purple)
const METHOD_MARKERS = (:circle, :diamond, :utriangle, :rect)
const METHOD_STYLES = (:solid, :dash, :dot, :dashdot)

function trajectory_plot(grid, Z, histF, histR, histG, histS, zF, zR, zG, zS)
    p = contourf(grid, grid, Z;
        xlabel="z3 (probing effort, class 3)", ylabel="z4 (probing effort, class 4)",
        title="Four search trajectories on the CRN direct-simulation oracle surface\nalpha=(10,4.5), signal beta=$(BETA), Softmax beta=$(SOFTMAX_BETA)",
        levels=24, colorbar_title="Oracle J(z)", size=(1150,850),
        left_margin=12Plots.mm, bottom_margin=10Plots.mm, top_margin=7Plots.mm,
        right_margin=8Plots.mm, titlefontsize=12, legendfontsize=9)
    histories = (histF,histR,histG,histS)
    endpoints = (zF,zR,zG,zS)
    for i in 1:4
        hist = histories[i]
        plot!(p, [z[1] for z in hist], [z[2] for z in hist];
              color=METHOD_COLORS[i], lw=2.5, marker=METHOD_MARKERS[i], ms=3,
              linestyle=METHOD_STYLES[i], label=METHOD_LABELS[i])
        scatter!(p, [endpoints[i][1]], [endpoints[i][2]];
                 color=METHOD_COLORS[i], ms=8, marker=METHOD_MARKERS[i], label="")
    end
    scatter!(p, [Z0[1]], [Z0[2]]; color=:gold, marker=:star5, ms=10, label="Common z0")
    savefig(p,"probing_four_trajectories_crn_oracle_surface.png")
    return p
end

function coordinate_plot(histF, histR, histG, histS)
    histories = (histF,histR,histG,histS)
    panels = map(1:2) do j
        p = plot(; xlabel="Iteration", ylabel=j==1 ? "z3" : "z4",
                 title=j==1 ? "Class-3 probing" : "Class-4 probing",
                 left_margin=7Plots.mm, bottom_margin=9Plots.mm, legendfontsize=8)
        for i in 1:4
            hist = histories[i]
            plot!(p, 0:length(hist)-1, [z[j] for z in hist];
                  label=METHOD_LABELS[i], color=METHOD_COLORS[i],
                  marker=METHOD_MARKERS[i], linestyle=METHOD_STYLES[i], ms=2.5, lw=2)
        end
        p
    end
    p = plot(panels...; layout=(1,2), size=(1250,550))
    savefig(p,"probing_four_coordinate_paths.png")
    return p
end

function objective_iteration_plot(JF, JR, JG, JS)
    p = plot(; xlabel="Iteration", ylabel="Objective value J(z)",
             title="Convergence of four searches\nOriginal hard-max objective on the SAME independent CRN bank",
             size=(1100,700), left_margin=12Plots.mm, bottom_margin=10Plots.mm,
             top_margin=7Plots.mm, titlefontsize=12, legend=:bottomright)
    for (i,trace) in enumerate((JF,JR,JG,JS))
        plot!(p, 0:length(trace)-1, trace;
              label=METHOD_LABELS[i], color=METHOD_COLORS[i], marker=METHOD_MARKERS[i],
              linestyle=METHOD_STYLES[i], lw=2.5, ms=3)
    end
    savefig(p,"probing_four_objective_vs_iteration.png")
    return p
end

function raw_search_objective_plot(JF, JR, JG, JS)
    p = plot(; xlabel="Iteration", ylabel="Internal hard-max objective estimate",
             title="Raw search estimates (different banks/estimators)\nSoftmax changes the gradient; its reported objective still uses hard max",
             size=(1100,700), left_margin=12Plots.mm, bottom_margin=10Plots.mm,
             top_margin=7Plots.mm, titlefontsize=12, legend=:bottomright)
    for (i,trace) in enumerate((JF,JR,JG,JS))
        # Fresh/Green/Softmax append stale terminal placeholders; do not draw them
        # as new objective evaluations. Original search functions remain unchanged.
        n = i==1 ? length(trace) : min(MAX_ITERS,length(trace))
        plot!(p, 0:n-1, trace[1:n]; label=METHOD_LABELS[i], color=METHOD_COLORS[i],
              marker=METHOD_MARKERS[i], linestyle=METHOD_STYLES[i], lw=2.3, ms=3)
    end
    savefig(p,"probing_four_raw_search_objective_vs_iteration.png")
    return p
end

function oracle_endpoint_plot(results)
    p = bar(collect(METHOD_LABELS), [r.J for r in results];
            yerror=1.96 .* [r.se for r in results], color=collect(METHOD_COLORS),
            ylabel="Independent oracle objective J(z)", legend=false,
            title="Direct-simulation evaluation of four final points\n95% MC bars for selected-policy means (no selection adjustment)",
            size=(1000,700), left_margin=12Plots.mm, bottom_margin=10Plots.mm,
            top_margin=8Plots.mm, titlefontsize=12)
    savefig(p,"probing_four_oracle_endpoints.png")
    return p
end

# =========================
# 18. EXACT DATA, READABLE CSV AND REPRODUCIBILITY METADATA
# =========================

function save_four_results(searches, traces, oracle, grid, surface)
    serialize("probing_four_results.jls", (searches=searches,traces=traces,
              oracle=oracle,grid=collect(grid),surface=surface))
    open("probing_four_traces.csv","w") do io
        println(io,"method,iteration,z3,z4,J_common_crn,J_internal,policy_internal,internal_evaluated")
        for (i,s) in enumerate(searches), k in eachindex(s.zhist)
            evaluated = i==1 || k<=MAX_ITERS
            @printf(io,"%s,%d,%.17g,%.17g,%.17g,%.17g,%d,%s\n",
                    ("FIXED","FRESH","GREEN","SOFTMAX")[i],k-1,s.zhist[k]...,
                    traces[i][k],s.Jhist[k],s.ihist[k],string(evaluated))
        end
    end
    open("probing_four_softmax_diagnostics.csv","w") do io
        println(io,"iteration,seed,z3,z4,g3,g4,clipped_g3,clipped_g4,step,max_weight,K_eff,logsumexp_surrogate")
        for d in searches[4].diagnostics
            @printf(io,"%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    d.iteration,d.seed,d.z...,d.gradient...,d.clipped_gradient...,
                    d.step,d.max_weight,d.effective_policies,d.surrogate)
        end
    end
    open("probing_four_oracle_endpoints.csv","w") do io
        println(io,"method,z3,z4,J_oracle,gross,policy,SE,seed,npaths")
        for r in oracle
            @printf(io,"%s,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%d,%d\n",
                    r.method,r.z...,r.J,r.gross,r.policy,r.se,r.seed,ORACLE_PATHS_FINAL)
        end
    end
    open("probing_four_oracle_surface.csv","w") do io
        println(io,"z3,z4,J_oracle_crn")
        for ix in eachindex(grid), iy in eachindex(grid)
            @printf(io,"%.17g,%.17g,%.17g\n",grid[ix],grid[iy],surface[iy,ix])
        end
    end
    metadata = Dict(
        "created_utc"=>string(Dates.now(Dates.UTC)), "julia_version"=>string(VERSION),
        "plots_version"=>string(pkgversion(Plots)), "julia_threads"=>Threads.nthreads(),
        "blas_threads"=>BLAS.get_num_threads(),
        "source_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "objective"=>"max_i E_z[Y_i] - alpha'z; no outer eta stage",
        "softmax_equation"=>"Probing_R_S.pdf, Notes on Softmax, equation (2), page 25",
        "SOFTMAX_BETA"=>SOFTMAX_BETA, "BETA"=>BETA,
        "N_PATHS"=>N_PATHS,"JOBS_PER_PATH"=>JOBS_PER_PATH,"MAX_ITERS"=>MAX_ITERS,
        "Z0"=>Z0,"ALPHA"=>ALPHA,"BASE_STEP"=>BASE_STEP,"GRAD_CLIP"=>GRAD_CLIP,
        "P_COMPLEX"=>P_COMPLEX,"P_HIT0"=>P_HIT0,"P_FALSE0"=>P_FALSE0,
        "MEAN_INTERARRIVAL"=>MEAN_INTERARRIVAL,
        "FAST_MEAN"=>collect(FAST_MEAN),"STD_MEAN"=>collect(STD_MEAN),
        "REVENUE_SCALE"=>REVENUE_SCALE,"FAST_ROUTING_COST"=>FAST_ROUTING_COST,
        "DELAY_PENALTY"=>DELAY_PENALTY,"policy_count"=>144,
        "SEED_FIXED"=>SEED_FIXED,"SEED_FRESH"=>SEED_FRESH,"SEED_GREEN"=>SEED_GREEN,
        "SEED_SOFTMAX"=>SEED_SOFTMAX,"iteration_seed_stride"=>10000,
        "TRACE_EVAL_PATHS"=>TRACE_EVAL_PATHS,"TRACE_EVAL_SEED"=>TRACE_EVAL_SEED,
        "ORACLE_PATHS_FINAL"=>ORACLE_PATHS_FINAL,"ORACLE_SEED"=>ORACLE_SEED,
        "ORACLE_GRID_N"=>ORACLE_GRID_N,"ORACLE_PATHS_PER_GRID_POINT"=>ORACLE_PATHS_PER_GRID_POINT,
        "ORACLE_SURFACE_SEED"=>ORACLE_SURFACE_SEED,
        "search_paths_generated"=>Dict("FIXED"=>N_PATHS,"FRESH"=>N_PATHS*MAX_ITERS,
                                      "GREEN"=>N_PATHS*MAX_ITERS,"SOFTMAX"=>N_PATHS*MAX_ITERS),
        "evaluation_note"=>"All traces use one shared independent CRN bank and hard max. Oracle SE excludes policy-selection adjustment.",
        "internal_terminal_placeholders"=>"Fresh, Green, Softmax retain previous raw J/policy at an appended final z; flagged false in internal_evaluated and omitted from raw plot.")
    for (key,filename) in (("original_source_sha256","002probing_three_searches_n500_full45iters.jl"),
                           ("pdf_sha256","Probing_R_S.pdf"))
        path = joinpath(@__DIR__,filename)
        isfile(path) && (metadata[key]=bytes2hex(sha256(read(path))))
    end
    open("probing_four_metadata.toml","w") do io
        TOML.print(io,metadata)
    end
    return nothing
end

# =========================
# 19. FOUR-SEARCH MAIN (ORIGINAL EXPERIMENT PARAMETERS)
# =========================

function main()
    println("================================================================")
    println(" TWO-SERVER PROBING: FIXED / FRESH / GREEN / SOFTMAX")
    println("================================================================")
    @printf("n=%d paths/bank, jobs/path=%d, iterations=%d\n",N_PATHS,JOBS_PER_PATH,MAX_ITERS)
    @printf("signal BETA=%.3f; independent SOFTMAX_BETA=%.3f\n",BETA,SOFTMAX_BETA)
    @printf("initial z=(%.3f,%.3f); cost alpha=(%.3f,%.3f)\n",Z0...,ALPHA...)
    @printf("Search paths: FIXED=%d; FRESH=%d; GREEN=%d; SOFTMAX=%d\n",
            N_PATHS,N_PATHS*MAX_ITERS,N_PATHS*MAX_ITERS,N_PATHS*MAX_ITERS)
    println("Fresh and Softmax use the SAME iteration seeds; evaluation uses independent streams.")
    println("The three original searches and all problem parameters are unchanged.")
    flush(stdout)
    policies = build_policy_grid()
    zF,JF,histF,JhistF,ihistF,_,_ = optimize_fixed(Z0,policies)
    flush(stdout)
    zR,JR,histR,JhistR,ihistR = optimize_fresh(Z0,policies)
    flush(stdout)
    zG,JG,histG,JhistG,ihistG,_ = optimize_green(Z0,policies)
    flush(stdout)
    soft = optimize_softmax(Z0,policies)
    searches = ((z=zF,J=JF,zhist=histF,Jhist=JhistF,ihist=ihistF),
                (z=zR,J=JR,zhist=histR,Jhist=JhistR,ihist=ihistR),
                (z=zG,J=JG,zhist=histG,Jhist=JhistG,ihist=ihistG),soft)
    # Preserve full-precision search data before the more expensive evaluations.
    serialize("probing_four_searches.jls",searches)
    oracle = oracle_compare_endpoints(zF,zR,zG,soft.z,policies)
    traces = build_common_objective_traces(histF,histR,histG,soft.zhist,policies)
    # This is the UNCHANGED surface procedure copied from the three-search file.
    grid,surface = build_oracle_surface(policies)
    flush(stdout)
    save_four_results(searches,traces,oracle,grid,surface)
    plots = (trajectory_plot(grid,surface,histF,histR,histG,soft.zhist,zF,zR,zG,soft.z),
             coordinate_plot(histF,histR,histG,soft.zhist),
             objective_iteration_plot(traces...),
             raw_search_objective_plot(JhistF,JhistR,JhistG,soft.Jhist),
             oracle_endpoint_plot(oracle))
    println("\nFINAL POINTS (full precision is saved in CSV/JLS):")
    for r in oracle
        @printf("%s: z=(%.9f,%.9f), oracle J=%.6f, SE=%.6f\n",r.method,r.z...,r.J,r.se)
    end
    println("Saved five probing_four_*.png figures, CSV traces/diagnostics/oracle/surface,")
    println("probing_four_metadata.toml and full-precision .jls results.")
    flush(stdout)
    for p in plots
        display(p)
    end
    return (searches=searches,traces=traces,oracle=oracle,grid=grid,surface=surface,plots=plots)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
