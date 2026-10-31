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
const ALPHA_BASELINE = [6.1, 2.6]
const ALPHA = ALPHA_BASELINE  # retained for compatibility with shared helper functions

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
const MAX_ITERS = 45
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



# ============================================================
# PARAMETER-SCREENING CONTROLS
# ============================================================
#
# Goal: diagnose the geometry/flatness of the TRUE direct-simulation objective
# BEFORE running Fixed/Fresh/Green optimization searches.
#
# Important computational shortcut:
#   Gross(z) = max_policy E[Y(policy,z)]
# does NOT depend on alpha.
# Therefore we simulate Gross(z) ONCE on a common CRN grid, then for every
# candidate alpha compute
#
#   J_alpha(z) = Gross(z) - alpha3*z3 - alpha4*z4
#
# essentially for free.
#
# This means we can screen many alpha pairs without rerunning simulation.

const SCREEN_GRID_N = 31
const SCREEN_PATHS = 10000
const SCREEN_SEED = 20260730

# Candidate probing costs. Edit these grids freely.
# Baseline (6.1, 2.6) is included explicitly.
const ALPHA3_GRID = [5.0, 6.1, 7.0, 8.0, 9.0, 10.0, 12.0]
const ALPHA4_GRID = [2.0, 2.6, 3.5, 4.5, 5.5, 7.0]

# Absolute objective-loss tolerances used to measure near-optimal-region size.
const LOSS_TOLS = [0.10, 0.25, 0.50, 1.00]

# Optional criteria used only for a rough ranking flag.
# We do NOT automatically declare a "winner"; the script prints diagnostics
# so the parameter choice can be made substantively.
const REQUIRE_INTERIOR_MARGIN = 0.10
const MAX_FRAC_WITHIN_025_FOR_COMPACT = 0.20

# ============================================================
# BUILD GROSS-BENEFIT SURFACE ONCE
# ============================================================

function build_gross_surface_once(policies)
    println("\n==============================================================")
    println(" BUILDING COMMON DIRECT-SIMULATION GROSS-BENEFIT SURFACE")
    println("==============================================================")
    @printf("grid = %d x %d\n", SCREEN_GRID_N, SCREEN_GRID_N)
    @printf("CRN paths per grid point = %d\n", SCREEN_PATHS)
    @printf("seed = %d\n", SCREEN_SEED)
    println("This simulation is done ONCE. Alpha is applied afterward analytically.")

    crn_bank = generate_crn_primitive_bank(
        npaths=SCREEN_PATHS,
        seed=SCREEN_SEED
    )

    grid = collect(range(0.0, 1.0; length=SCREEN_GRID_N))
    G = Matrix{Float64}(undef, SCREEN_GRID_N, SCREEN_GRID_N)

    for (ix, z3) in enumerate(grid)
        for (iy, z4) in enumerate(grid)
            z = [z3, z4]
            bank_z = materialize_crn_bank(crn_bank, z)
            Yz = precompute_rewards(policies, bank_z)
            vals = vec(mean(Yz; dims=2))
            G[iy, ix] = maximum(vals)
        end
        @printf("  gross-surface progress: %d/%d z3 columns\n", ix, SCREEN_GRID_N)
    end

    return grid, G
end

function objective_surface_from_gross(grid, G, alpha3, alpha4)
    J = similar(G)
    for (ix, z3) in enumerate(grid)
        for (iy, z4) in enumerate(grid)
            J[iy, ix] = G[iy, ix] - alpha3*z3 - alpha4*z4
        end
    end
    return J
end

# ============================================================
# SURFACE DIAGNOSTICS
# ============================================================

function nearest_grid_index(grid, x)
    return argmin(abs.(grid .- x))
end

function surface_diagnostics(grid, J, alpha3, alpha4)
    CI = CartesianIndices(J)
    imax = argmax(J)
    I = CI[imax]
    iy, ix = Tuple(I)

    z3star = grid[ix]
    z4star = grid[iy]
    Jstar = J[iy, ix]

    i0 = nearest_grid_index(grid, 0.0)
    i1 = nearest_grid_index(grid, 1.0)

    J00 = J[i0, i0]
    J11 = J[i1, i1]

    losses = Jstar .- J
    fracs = [mean(losses .<= tol + 1e-12) for tol in LOSS_TOLS]

    # Simple boundary/interior diagnostic.
    interior =
        (z3star >= REQUIRE_INTERIOR_MARGIN) &&
        (z3star <= 1.0 - REQUIRE_INTERIOR_MARGIN) &&
        (z4star >= REQUIRE_INTERIOR_MARGIN) &&
        (z4star <= 1.0 - REQUIRE_INTERIOR_MARGIN)

    # Objective ranges along slices through the grid maximizer.
    row_at_star = @view J[iy, :]
    col_at_star = @view J[:, ix]
    z3_slice_drop = Jstar - minimum(row_at_star)
    z4_slice_drop = Jstar - minimum(col_at_star)

    # Rough compactness flag, deliberately transparent rather than "optimizing"
    # parameters behind the scenes.
    compact025 = fracs[2] <= MAX_FRAC_WITHIN_025_FOR_COMPACT

    return (
        alpha3=alpha3,
        alpha4=alpha4,
        z3star=z3star,
        z4star=z4star,
        Jstar=Jstar,
        J00=J00,
        J11=J11,
        loss00=Jstar-J00,
        loss11=Jstar-J11,
        frac010=fracs[1],
        frac025=fracs[2],
        frac050=fracs[3],
        frac100=fracs[4],
        z3_slice_drop=z3_slice_drop,
        z4_slice_drop=z4_slice_drop,
        interior=interior,
        compact025=compact025
    )
end

function print_diagnostic_header()
    println("\n==============================================================================================================================")
    println(" PARAMETER SCREEN")
    println("==============================================================================================================================")
    @printf("%6s %6s | %6s %6s | %9s | %8s %8s | %7s %7s %7s %7s | %8s %8s | %s\n",
            "a3","a4","z3*","z4*","J*","loss00","loss11",
            "%<=.10","%<=.25","%<=.50","%<=1.0",
            "drop-z3","drop-z4","flags")
    println(repeat("-", 126))
end

function print_diagnostic_row(d)
    flags = String[]
    d.interior && push!(flags, "interior")
    d.compact025 && push!(flags, "compact.25")
    isempty(flags) && push!(flags, "-")

    @printf("%6.2f %6.2f | %6.3f %6.3f | %9.3f | %8.3f %8.3f | %7.1f %7.1f %7.1f %7.1f | %8.3f %8.3f | %s\n",
            d.alpha3, d.alpha4,
            d.z3star, d.z4star, d.Jstar,
            d.loss00, d.loss11,
            100*d.frac010, 100*d.frac025, 100*d.frac050, 100*d.frac100,
            d.z3_slice_drop, d.z4_slice_drop,
            join(flags, ","))
end

# ============================================================
# PLOTS FOR SELECTED / PROMISING PARAMETERIZATIONS
# ============================================================

function plot_surface(grid, J, d; prefix="screen")
    p = contourf(
        grid, grid, J;
        xlabel="z3  (probing effort, class 3)",
        ylabel="z4  (probing effort, class 4)",
        title=@sprintf("Direct-simulation oracle J(z): alpha=(%.2f, %.2f)\nz*=(%.3f, %.3f), J*=%.3f, grid within 0.25=%.1f%%",
                       d.alpha3, d.alpha4, d.z3star, d.z4star, d.Jstar, 100*d.frac025),
        levels=28,
        colorbar_title="J(z)",
        size=(900,700)
    )
    scatter!(p, [d.z3star], [d.z4star];
             marker=:star5, ms=11, label="grid maximizer")
    scatter!(p, [0.0,1.0], [0.0,1.0];
             marker=:circle, ms=5, label="(0,0), (1,1)")

    fname = @sprintf("%s_alpha3_%.2f_alpha4_%.2f.png",
                     prefix, d.alpha3, d.alpha4)
    fname = replace(fname, "." => "p", count=2) # keep extension correction below
    fname = replace(fname, "ppng" => ".png")
    savefig(p, fname)
    return p, fname
end

function near_optimal_loss_plot(grid, J, d)
    losses = d.Jstar .- J
    p = contourf(
        grid, grid, losses;
        xlabel="z3",
        ylabel="z4",
        title=@sprintf("Loss from grid optimum: alpha=(%.2f, %.2f)", d.alpha3, d.alpha4),
        levels=[0.0,0.10,0.25,0.50,1.0,2.0,4.0,8.0,Inf],
        colorbar_title="J* - J(z)",
        size=(900,700)
    )
    scatter!(p, [d.z3star], [d.z4star];
             marker=:star5, ms=11, label="grid maximizer")
    fname = @sprintf("loss_alpha3_%.2f_alpha4_%.2f.png", d.alpha3, d.alpha4)
    savefig(p, fname)
    return p, fname
end

# ============================================================
# MAIN SCREEN
# ============================================================

function main()
    println("================================================================")
    println(" PROBING SURFACE PARAMETER SCREEN -- NO OPTIMIZATION SEARCHES")
    println("================================================================")
    @printf("beta = %.3f\n", BETA)
    @printf("active classes = (%d,%d)\n", ACTIVE_CLASSES...)
    println("Purpose: choose an informative J(z) landscape BEFORE comparing algorithms.")

    policies = build_policy_grid()
    @printf("policy family = %d threshold policies\n", length(policies))

    grid, G = build_gross_surface_once(policies)

    results = NamedTuple[]
    surfaces = Dict{Tuple{Float64,Float64}, Matrix{Float64}}()

    print_diagnostic_header()

    for a3 in ALPHA3_GRID
        for a4 in ALPHA4_GRID
            J = objective_surface_from_gross(grid, G, a3, a4)
            d = surface_diagnostics(grid, J, a3, a4)
            push!(results, d)
            surfaces[(a3,a4)] = J
            print_diagnostic_row(d)
        end
    end

    # Rank primarily by compactness of the 0.25-near-optimal set, but only
    # display this as a diagnostic ranking. We do not automatically choose alpha.
    ranked = sort(results; by=d -> (
        d.interior ? 0 : 1,
        d.frac025,
        -min(d.loss00, d.loss11)
    ))

    println("\n================================================================")
    println(" TOP 10 DIAGNOSTIC CANDIDATES")
    println(" Preference shown: interior grid maximizer, smaller <=0.25 region,")
    println(" then larger loss at the better of the two extreme corners.")
    println(" This is NOT an automatic scientific choice of parameters.")
    println("================================================================")
    for d in ranked[1:min(10,length(ranked))]
        print_diagnostic_row(d)
    end

    # Always plot baseline plus up to four top-ranked distinct candidates.
    selected = NamedTuple[]
    baseline = findfirst(d -> isapprox(d.alpha3,6.1) && isapprox(d.alpha4,2.6), results)
    if baseline !== nothing
        push!(selected, results[baseline])
    end
    for d in ranked
        if !any(x -> x.alpha3 == d.alpha3 && x.alpha4 == d.alpha4, selected)
            push!(selected, d)
        end
        length(selected) >= 5 && break
    end

    println("\nGenerating detailed plots for baseline + leading diagnostic candidates:")
    plots = Any[]
    for d in selected
        J = surfaces[(d.alpha3,d.alpha4)]
        p1, f1 = plot_surface(grid, J, d)
        p2, f2 = near_optimal_loss_plot(grid, J, d)
        push!(plots, p1); push!(plots, p2)
        @printf("  alpha=(%.2f,%.2f): %s, %s\n", d.alpha3,d.alpha4,f1,f2)
    end

    println("\nInterpretation guide:")
    println("  loss00/loss11 = objective loss versus the grid optimum at (0,0)/(1,1).")
    println("  %<=.25 = percentage of the entire z-grid within 0.25 objective units of optimum.")
    println("             Smaller means a more discriminating / less-flat test landscape.")
    println("  drop-z3/drop-z4 = total objective drop along horizontal/vertical slices through z*.")
    println("  'interior' means both coordinates of the grid maximizer lie in [0.10,0.90].")
    println("\nNo Fixed/Fresh/Green search was run.")

    # Display only the first few to avoid flooding the plot pane.
    for p in plots[1:min(4,length(plots))]
        display(p)
    end
end

main()
