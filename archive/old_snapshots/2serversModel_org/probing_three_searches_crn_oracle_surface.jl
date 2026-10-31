# probing_three_searches.jl
#
# Two-server routing with probing: comparison of THREE gradient-search schemes
#
#   1) FIXED BASELINE BANK:
#      Generate n paths once under z=(0,0), then reuse them forever with LR weights.
#
#   2) FRESH LOCAL BANK:
#      At each visited iterate z^t, generate a fresh bank under z^t, use only that
#      bank for the gradient at z^t (so the LR is identically 1), then discard it.
#
#   3) GREEN SIMULATION:
#      At each visited iterate z^t, generate a fresh bank under z^t and KEEP all
#      previous banks.  To estimate at the current z^t, reweight every historical
#      bank generated under z^tau by f(xi;z^t)/f(xi;z^tau), then average across banks.
#
# All three searches:
#   - use the same model and threshold-policy family;
#   - start from the same Z0;
#   - use the same projected gradient update / diminishing step;
#   - are plotted on ONE common reference heat map.
#
# IMPORTANT:
# The common heat map is only a visual reference surface. It is built from one
# independent, fixed bank generated under z=0. It does NOT drive the fresh or green
# searches.
#
# The green implementation follows the estimator described on pp. 9-10 of the
# write-up: historical observations generated at z^tau are reweighted to the
# current z^t.
#
# Packages:
#   import Pkg
#   Pkg.add("Plots")
#
# Run:
#   julia probing_three_searches.jl

using Random
using Statistics
using Printf
using LinearAlgebra
using Plots

# =========================
# 1. PARAMETERS
# =========================

const NCLASS = 4
const P_COMPLEX = [0.02, 0.12, 0.45, 0.70]
const P_HIT0   = [0.60, 0.65, 0.75, 0.92]
const P_FALSE0 = [0.05, 0.08, 0.15, 0.20]
const BETA = 2.20

const MEAN_INTERARRIVAL = 0.80
const FAST_MEAN = (0.75, 2.40)
const STD_MEAN  = (1.00, 8.00)

const REVENUE_SCALE = 8.0
const FAST_ROUTING_COST = 0.70
const DELAY_PENALTY = 0.42

const ACTIVE_CLASSES = (3, 4)
const ALPHA = [6.1, 2.6]

# Same n for each bank in all three methods.
# WARNING: fresh/green generate a new bank at every iteration, so N_PATHS=5000
# is substantially more expensive than the original fixed-bank experiment.
const JOBS_PER_PATH = 45
const N_PATHS = 5000

# Separate deterministic seed streams.
const SEED_FIXED = 20260720
const SEED_FRESH = 20260721
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

const Z0 = [0.12, 0.10]
const MAX_ITERS = 45
const BASE_STEP = 0.035
const GRAD_CLIP = 15.0
const STEP_TOL = 2e-4

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
        if norm(znew - z) < STEP_TOL
            z = znew
            break
        end
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
        if norm(znew - z) < STEP_TOL
            z = znew
            break
        end
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
        if norm(znew - z) < STEP_TOL
            z = znew
            break
        end
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
# 12B. INDEPENDENT ORACLE / DIRECT RESIMULATION
# =========================

"""
Directly evaluate a candidate z using fresh paths generated UNDER that z.

No likelihood ratio is used. For each policy i we estimate E_z[Y_i] directly,
select the policy with the largest sample mean, and subtract probing cost.

Returns:
    J, gross, best_i, SE_of_selected_policy_mean
"""
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

function oracle_compare_endpoints(zF, zR, zG, policies)
    println("\n================================================================")
    println(" INDEPENDENT ORACLE EVALUATION OF THE THREE FINAL POINTS")
    println("================================================================")
    println("Fresh direct simulation under each candidate z; NO LR.")
    @printf("oracle paths per endpoint = %d\n\n", ORACLE_PATHS_FINAL)

    # Different seeds make the three evaluations fully independent.
    JF, grossF, polF, seF =
        oracle_evaluate(zF, policies; seed=ORACLE_SEED + 1)
    JR, grossR, polR, seR =
        oracle_evaluate(zR, policies; seed=ORACLE_SEED + 2)
    JG, grossG, polG, seG =
        oracle_evaluate(zG, policies; seed=ORACLE_SEED + 3)

    @printf("FIXED | z=(%.6f, %.6f) | J_oracle=%10.4f | gross=%10.4f | policy=%3d | SE≈%.4f\n",
            zF[1], zF[2], JF, grossF, polF, seF)
    @printf("FRESH | z=(%.6f, %.6f) | J_oracle=%10.4f | gross=%10.4f | policy=%3d | SE≈%.4f\n",
            zR[1], zR[2], JR, grossR, polR, seR)
    @printf("GREEN | z=(%.6f, %.6f) | J_oracle=%10.4f | gross=%10.4f | policy=%3d | SE≈%.4f\n",
            zG[1], zG[2], JG, grossG, polG, seG)

    Js = [JF, JR, JG]
    names = ["FIXED", "FRESH", "GREEN"]
    ibest = argmax(Js)
    @printf("\nBest independently evaluated endpoint: %s  (J_oracle=%.4f)\n",
            names[ibest], Js[ibest])

    return (JF=JF, JR=JR, JG=JG,
            grossF=grossF, grossR=grossR, grossG=grossG,
            polF=polF, polR=polR, polG=polG,
            seF=seF, seR=seR, seG=seG)
end

# =========================
# 13. PLOTS: THREE TRAJECTORIES
# =========================

function trajectory_plot(grid, Z,
                         zhist_fixed, zhist_fresh, zhist_green,
                         zstar_fixed, zstar_fresh, zstar_green)

    p = contourf(
        grid, grid, Z;
        xlabel="z3  (probing effort, class 3)",
        ylabel="z4  (probing effort, class 4)",
        title="Three search trajectories on CRN direct-resimulation oracle surface\nalpha3=$(ALPHA[1]), alpha4=$(ALPHA[2]), beta=$(BETA), n=$(N_PATHS) per search bank",
        levels=24,
        colorbar_title="Oracle J(z)",
        size=(1050, 800)
    )

    xf = [z[1] for z in zhist_fixed]; yf = [z[2] for z in zhist_fixed]
    xr = [z[1] for z in zhist_fresh]; yr = [z[2] for z in zhist_fresh]
    xg = [z[1] for z in zhist_green]; yg = [z[2] for z in zhist_green]

    plot!(p, xf, yf; lw=3.0, marker=:circle, ms=3.5, label="Fixed bank at z=0")
    plot!(p, xr, yr; lw=3.0, marker=:diamond, ms=3.5, linestyle=:dash,
          label="Fresh bank at each iterate")
    plot!(p, xg, yg; lw=3.0, marker=:utriangle, ms=4.0, linestyle=:dot,
          label="Green accumulated banks")

    scatter!(p, [Z0[1]], [Z0[2]]; ms=10, marker=:star5, label="common z0")

    scatter!(p, [zstar_fixed[1]], [zstar_fixed[2]];
             ms=8, marker=:circle, label="fixed endpoint")
    scatter!(p, [zstar_fresh[1]], [zstar_fresh[2]];
             ms=8, marker=:diamond, label="fresh endpoint")
    scatter!(p, [zstar_green[1]], [zstar_green[2]];
             ms=9, marker=:utriangle, label="green endpoint")

    savefig(p, "probing_three_trajectories_crn_oracle_surface.png")
    return p
end

function coordinate_plot(zhist_fixed, zhist_fresh, zhist_green)
    # Plot z3 and z4 in separate panels, each with the three methods.
    itf = 0:(length(zhist_fixed)-1)
    itr = 0:(length(zhist_fresh)-1)
    itg = 0:(length(zhist_green)-1)

    p3 = plot(itf, [z[1] for z in zhist_fixed];
              lw=2.5, marker=:circle, label="Fixed z=0 bank",
              xlabel="iteration", ylabel="z3", title="Class-3 probing trajectory")
    plot!(p3, itr, [z[1] for z in zhist_fresh];
          lw=2.5, marker=:diamond, linestyle=:dash, label="Fresh local bank")
    plot!(p3, itg, [z[1] for z in zhist_green];
          lw=2.5, marker=:utriangle, linestyle=:dot, label="Green")

    p4 = plot(itf, [z[2] for z in zhist_fixed];
              lw=2.5, marker=:circle, label="Fixed z=0 bank",
              xlabel="iteration", ylabel="z4", title="Class-4 probing trajectory")
    plot!(p4, itr, [z[2] for z in zhist_fresh];
          lw=2.5, marker=:diamond, linestyle=:dash, label="Fresh local bank")
    plot!(p4, itg, [z[2] for z in zhist_green];
          lw=2.5, marker=:utriangle, linestyle=:dot, label="Green")

    p = plot(p3, p4; layout=(1,2), size=(1200,500))
    savefig(p, "probing_three_coordinate_paths.png")
    return p
end

function oracle_endpoint_plot(zF, zR, zG, oracle_results)
    labels = ["Fixed z=0 bank", "Fresh local", "Green"]
    vals = [oracle_results.JF, oracle_results.JR, oracle_results.JG]
    ses = [oracle_results.seF, oracle_results.seR, oracle_results.seG]

    p = bar(
        labels, vals;
        yerror=1.96 .* ses,
        ylabel="Independent oracle objective J(z)",
        title="Independent direct-resimulation evaluation of final endpoints\n95% MC error bars for selected-policy mean",
        legend=false,
        size=(900, 600),
        xrotation=10
    )
    savefig(p, "probing_three_oracle_endpoints.png")
    return p
end

# =========================
# 14. MAIN
# =========================

function main()
    println("================================================================")
    println(" TWO-SERVER PROBING: THREE GRADIENT-SEARCH SCHEMES")
    println("================================================================")
    @printf("active classes = (%d,%d)\n", ACTIVE_CLASSES...)
    @printf("alpha = (%.3f, %.3f), beta = %.3f\n", ALPHA[1], ALPHA[2], BETA)
    @printf("n per bank = %d, jobs/path = %d\n", N_PATHS, JOBS_PER_PATH)
    @printf("common start z0 = (%.3f, %.3f)\n", Z0[1], Z0[2])
    println("NOTE: fresh and green generate n new paths at every iteration.")
    println("================================================================")

    policies = build_policy_grid()
    @printf("policy family = %d threshold policies\n", length(policies))

    # Run the three searches.
    zF, JF, histF, JhistF, ihistF, fixed_bank, fixedY =
        optimize_fixed(Z0, policies)

    zR, JR, histR, JhistR, ihistR =
        optimize_fresh(Z0, policies)

    zG, JG, histG, JhistG, ihistG, green_history =
        optimize_green(Z0, policies)

    # Independent direct-resimulation evaluation of the three endpoints.
    oracle_results = oracle_compare_endpoints(zF, zR, zG, policies)

    # Independent DIRECT-RESIMULATION oracle surface for visualization only.
    # This does not alter or rerun the three search trajectories.
    grid, Z = build_oracle_surface(policies)

    p1 = trajectory_plot(grid, Z, histF, histR, histG, zF, zR, zG)
    p2 = coordinate_plot(histF, histR, histG)
    p3 = oracle_endpoint_plot(zF, zR, zG, oracle_results)

    println("\n================================================================")
    println(" FINAL ENDPOINTS")
    println("================================================================")
    @printf("Fixed z=0 bank : z=(%.6f, %.6f)\n", zF[1], zF[2])
    @printf("Fresh local    : z=(%.6f, %.6f)\n", zR[1], zR[2])
    @printf("Green          : z=(%.6f, %.6f)\n", zG[1], zG[2])
    println("================================================================")

    println("\nSaved:")
    println("  probing_three_trajectories_crn_oracle_surface.png")
    println("  probing_three_coordinate_paths.png")
    println("  probing_three_oracle_endpoints.png")

    display(p1)
    display(p2)
    display(p3)
end

main()
