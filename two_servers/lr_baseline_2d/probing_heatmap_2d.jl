
# probing_heatmap_2d.jl
#
# Two-server routing with probing, likelihood-ratio (LR) evaluation,
# a 2-D heat map, a starting point z0, and projected-gradient evolution
# toward an estimated interior optimum z*.
#
# The two optimized probing coordinates are z3 and z4 (classes 3 and 4).
# Classes 1 and 2 are fixed at zero probing.
#
# Packages:
#   import Pkg
#   Pkg.add("Plots")
#
# Run:
#   julia probing_heatmap_2d.jl
#
# Output:
#   probing_heatmap_trajectory.png
#   probing_convergence.png
#
# IMPORTANT MODELING POINT
# ------------------------
# The sample-path bank is generated under z=(0,0,0,0). Each path includes the
# binary signals generated under that baseline distribution. For any candidate z,
# we reuse those exact paths and change their probability through the LR
#
#     L(xi;z) = f(xi;z) / f(xi;0).
#
# The routing reward Y(policy, xi) is therefore computed ONCE for each
# policy/path pair. Candidate z changes the weighted expectation, not Y itself.
#
# The analytic gradient correctly differentiates E_0[Y L(z)]:
#
#   d/dz E_0[Y L(z)] = E_0[Y L(z) * d log f(xi;z)/dz].
#
# ------------------------------------------------------------------------------

using Random
using Statistics
using Printf
using LinearAlgebra
using Plots

# =========================
# 1. PARAMETERS
# =========================

const NCLASS = 4

# Intrinsic complexity probabilities
const P_COMPLEX = [0.02, 0.12, 0.55, 0.95]

# Signal quality at z=0
const P_HIT0   = [0.25, 0.40, 0.75, 0.92]
const P_FALSE0 = [0.05, 0.08, 0.15, 0.20]

# How quickly probing improves signal quality
const BETA = 2.20

# Arrival / service parameters.
# Chosen so congestion matters and the fast server is valuable but not dominant.
const MEAN_INTERARRIVAL = 0.80

# Mean service times by [simple, complex].
# Complex jobs benefit strongly from fast routing.
const FAST_MEAN = (0.75, 2.40)
const STD_MEAN  = (1.00, 5.80)

# Economic parameters.
# Revenue is obtained upon completion; total net benefit subtracts routing cost
# and a delay penalty proportional to time in system.
const REVENUE_SCALE = 8.0
const FAST_ROUTING_COST = 0.70
const DELAY_PENALTY = 0.42

# Two-dimensional experiment:
# optimize probing for classes 3 and 4 only.
const ACTIVE_CLASSES = (3, 4)

# Probing cost vector for z3,z4.
#
# These values were deliberately chosen to make "probe nothing" and "probe fully"
# unattractive in typical runs: class 3 has substantial ambiguity and class 4 is
# usually complex already, so the two dimensions have different marginal values.
#
# If your Monte Carlo seed/sample size changes substantially, use the calibration
# diagnostic printed by the script and adjust these slightly.
const ALPHA = [6.1, 2.6]

# Simulation size.
# LR products become more variable as jobs/path grows, so we use moderately sized
# independent "days" and many replications.
const JOBS_PER_PATH = 45
const N_PATHS = 5000
const SEED = 20260720

# Heatmap resolution
const GRID_N = 41

# Projected gradient ascent
const Z0 = [0.12, 0.10]
const MAX_ITERS = 45
const BASE_STEP = 0.035
const GRAD_CLIP = 15.0

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
    signal::Int      # baseline signal generated under z=0
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

# =========================
# 3. BASIC RANDOM HELPERS
# =========================

# Exponential random variable with specified mean, using only Base/Random.
randexp_mean(rng, mean::Float64) = -mean * log(rand(rng))

function p_hit(c::Int, zc::Float64)
    p0 = P_HIT0[c]
    return p0 + (1.0 - p0) * (1.0 - exp(-BETA * zc))
end

function p_false(c::Int, zc::Float64)
    return P_FALSE0[c] * exp(-BETA * zc)
end

function signal_probability(c::Int, complex::Bool, zc::Float64)
    return complex ? p_hit(c, zc) : p_false(c, zc)
end

# =========================
# 4. SAMPLE-PATH BANK AT z=0
# =========================

function generate_path(rng::AbstractRNG; njobs::Int=JOBS_PER_PATH)
    jobs = Vector{JobPrimitive}(undef, njobs)
    t = 0.0

    for r in 1:njobs
        t += randexp_mean(rng, MEAN_INTERARRIVAL)

        c = rand(rng, 1:NCLASS)
        d = rand(rng) < P_COMPLEX[c]

        # Job value: mild class effect, but still heterogeneous within classes.
        # Kept inside [0,1].
        basev = 0.15 + 0.16 * c
        v = clamp(basev + 0.35 * (rand(rng) - 0.5), 0.05, 1.0)

        μf = d ? FAST_MEAN[2] : FAST_MEAN[1]
        μs = d ? STD_MEAN[2]  : STD_MEAN[1]

        sf = randexp_mean(rng, μf)
        ss = randexp_mean(rng, μs)

        # Baseline signal distribution: z_c = 0
        ps = signal_probability(c, d, 0.0)
        s = rand(rng) < ps ? 1 : 0

        jobs[r] = JobPrimitive(t, c, d, v, sf, ss, s)
    end

    return SamplePath(jobs)
end

function generate_bank(; npaths::Int=N_PATHS, seed::Int=SEED)
    rng = MersenneTwister(seed)
    return [generate_path(rng) for _ in 1:npaths]
end

# =========================
# 5. THRESHOLD POLICY FAMILY
# =========================

function build_policy_grid()
    policies = Policy[]

    # A compact but heterogeneous family. The optimizer over policies is inside
    # the LR-weighted expectation, exactly as in the write-up.
    for av in (0.4, 0.8, 1.2),
        aq in (0.15, 0.35, 0.65),
        as in (0.0, 0.6, 1.2, 2.0),
        tau in (0.10, 0.45, 0.80, 1.15)

        push!(policies, Policy(av, aq, as, tau))
    end

    return policies
end

# =========================
# 6. SIMULATE ONE POLICY ON ONE FIXED PATH
# =========================

function evaluate_policy(policy::Policy, path::SamplePath)
    # next-free times
    free_fast = 0.0
    free_std  = 0.0
    total = 0.0

    for job in path.jobs
        t = job.arrival

        # Workloads visible at arrival.
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

function precompute_rewards(policies, bank)
    npol = length(policies)
    npaths = length(bank)
    Y = Matrix{Float64}(undef, npol, npaths)

    @printf("Precomputing Y(policy, path): %d policies x %d paths...\n", npol, npaths)

    for i in 1:npol
        for ℓ in 1:npaths
            Y[i, ℓ] = evaluate_policy(policies[i], bank[ℓ])
        end
    end

    return Y
end

# =========================
# 7. LR AND SCORE FUNCTION
# =========================

# Convert 2-D decision (z3,z4) to full 4-D probing vector.
function full_z(z::AbstractVector{<:Real})
    zz = zeros(Float64, 4)
    zz[ACTIVE_CLASSES[1]] = z[1]
    zz[ACTIVE_CLASSES[2]] = z[2]
    return zz
end

"""
Return:
  logL  = log f(xi;z)/f(xi;0)
  score = gradient wrt the 2 active z coordinates of log f(xi;z)

Only classes 3 and 4 contribute because z1=z2=0 and are not decision variables.
"""
function loglr_and_score(path::SamplePath, z2::AbstractVector{<:Real})
    zz = full_z(z2)
    logL = 0.0
    score = zeros(2)

    for job in path.jobs
        c = job.class

        # Skip inactive classes: numerator == denominator.
        active_pos = c == ACTIVE_CLASSES[1] ? 1 :
                     c == ACTIVE_CLASSES[2] ? 2 : 0
        active_pos == 0 && continue

        zc = zz[c]

        pz = signal_probability(c, job.complex, zc)
        p0 = signal_probability(c, job.complex, 0.0)

        # Numerical guard
        pz = clamp(pz, 1e-12, 1.0 - 1e-12)
        p0 = clamp(p0, 1e-12, 1.0 - 1e-12)

        if job.signal == 1
            logL += log(pz) - log(p0)
        else
            logL += log1p(-pz) - log1p(-p0)
        end

        # dp/dz
        dp =
            if job.complex
                (1.0 - P_HIT0[c]) * BETA * exp(-BETA * zc)
            else
                -P_FALSE0[c] * BETA * exp(-BETA * zc)
            end

        # d/dz log Bernoulli probability of realized signal
        score[active_pos] += job.signal == 1 ? dp / pz : -dp / (1.0 - pz)
    end

    return logL, score
end

# =========================
# 8. OBJECTIVE AND ANALYTIC GRADIENT
# =========================

"""
Compute LR-weighted policy values, select the best policy, then return
J(z) = max_i mean(Y_i * L(z)) - alpha'z.

For numerical stability, likelihood ratios are built from logL.
The baseline expectation E_0[L]=1 is NOT self-normalized here, because the
ordinary LR estimator is the estimator implied by the model derivation.
"""
function objective_and_gradient(
    z::AbstractVector{<:Real},
    bank,
    Y;
    alpha::AbstractVector{<:Real}=ALPHA,
    need_gradient::Bool=true
)
    npaths = length(bank)
    npol = size(Y, 1)

    logL = Vector{Float64}(undef, npaths)
    scores = need_gradient ? Matrix{Float64}(undef, 2, npaths) : zeros(0,0)

    for ℓ in 1:npaths
        ll, sc = loglr_and_score(bank[ℓ], z)
        logL[ℓ] = ll
        if need_gradient
            scores[:, ℓ] .= sc
        end
    end

    # Direct exponentiation is acceptable with our moderate path length, but clip
    # extreme Monte Carlo outliers to avoid a single numerical overflow.
    L = exp.(clamp.(logL, -35.0, 35.0))

    policy_values = Vector{Float64}(undef, npol)
    for i in 1:npol
        policy_values[i] = mean(@view(Y[i, :]) .* L)
    end

    best_i = argmax(policy_values)
    best_value = policy_values[best_i]
    J = best_value - dot(alpha, z)

    if !need_gradient
        return J, best_i, best_value, mean(L), std(L)
    end

    # Correct LR gradient:
    # E_0[Y_i* L(z) * score(z)] - alpha
    g = zeros(2)
    ystar = @view Y[best_i, :]

    for j in 1:2
        g[j] = mean(ystar .* L .* @view(scores[j, :])) - alpha[j]
    end

    return J, g, best_i, best_value, mean(L), std(L)
end

# =========================
# 9. FINITE-DIFFERENCE GRADIENT CHECK
# =========================

function gradient_check(z, bank, Y; h=1e-4)
    J, g, best_i, _, _, _ = objective_and_gradient(z, bank, Y)

    gfd = zeros(2)
    for j in 1:2
        zp = copy(z); zm = copy(z)
        zp[j] += h
        zm[j] -= h

        Jp, _, _, _, _ = objective_and_gradient(zp, bank, Y; need_gradient=false)
        Jm, _, _, _, _ = objective_and_gradient(zm, bank, Y; need_gradient=false)
        gfd[j] = (Jp - Jm) / (2h)
    end

    @printf("\nGradient check at z=(%.3f, %.3f), active policy=%d\n", z[1], z[2], best_i)
    @printf(" analytic = [% .6f, % .6f]\n", g[1], g[2])
    @printf(" finite-d = [% .6f, % .6f]\n", gfd[1], gfd[2])
    @printf(" abs diff = [% .3e, % .3e]\n\n", abs(g[1]-gfd[1]), abs(g[2]-gfd[2]))

    return J, g, gfd
end

# =========================
# 10. PROJECTED GRADIENT ASCENT
# =========================

function optimize_z(z0, bank, Y)
    z = clamp.(copy(z0), 0.0, 1.0)

    zhist = Vector{Vector{Float64}}()
    Jhist = Float64[]
    ihist = Int[]

    best_z = copy(z)
    best_J = -Inf

    for k in 1:MAX_ITERS
        J, g, best_i, _, meanL, stdL = objective_and_gradient(z, bank, Y)

        push!(zhist, copy(z))
        push!(Jhist, J)
        push!(ihist, best_i)

        if J > best_J
            best_J = J
            best_z .= z
        end

        @printf(
            "iter %2d | z=(%.4f, %.4f) | J=%10.4f | g=(% .4f,% .4f) | policy=%3d | E[L]=%.3f SD[L]=%.3f\n",
            k, z[1], z[2], J, g[1], g[2], best_i, meanL, stdL
        )

        # Clip only the step-driving gradient, not the reported gradient.
        gc = clamp.(g, -GRAD_CLIP, GRAD_CLIP)

        # Diminishing step + simple backtracking safeguard.
        η = BASE_STEP / sqrt(k)
        accepted = false

        for bt in 0:8
            znew = clamp.(z .+ η .* gc, 0.0, 1.0)
            Jnew, _, _, _, _ = objective_and_gradient(znew, bank, Y; need_gradient=false)

            if Jnew >= J - 1e-8
                z = znew
                accepted = true
                break
            end
            η *= 0.5
        end

        if !accepted
            @printf("No improving projected step found; stopping.\n")
            break
        end

        if norm(η .* gc) < 2e-4
            @printf("Step is very small; stopping.\n")
            break
        end
    end

    # Add final point
    J, _, best_i, _, _, _ = objective_and_gradient(z, bank, Y)
    push!(zhist, copy(z))
    push!(Jhist, J)
    push!(ihist, best_i)

    if J > best_J
        best_J = J
        best_z .= z
    end

    return best_z, best_J, zhist, Jhist, ihist
end

# =========================
# 11. HEATMAP
# =========================

function build_heatmap(bank, Y, zhist, zstar)
    grid = range(0.0, 1.0; length=GRID_N)
    Z = Matrix{Float64}(undef, GRID_N, GRID_N)

    println("\nBuilding heat map...")
    for (ix, z3) in enumerate(grid)
        for (iy, z4) in enumerate(grid)
            J, _, _, _, _ = objective_and_gradient([z3, z4], bank, Y; need_gradient=false)
            Z[iy, ix] = J  # rows=y(z4), cols=x(z3)
        end
        @printf("  heatmap row/column progress: %d/%d\n", ix, GRID_N)
    end

    p = contourf(
        grid, grid, Z;
        xlabel="z₃  (probing effort, class 3)",
        ylabel="z₄  (probing effort, class 4)",
        title="LR objective surface and optimization trajectory\nα₃=$(ALPHA[1]), α₄=$(ALPHA[2]), β=$(BETA)",        levels=24,
        colorbar_title="J(z)",
        size=(900, 720)
    )

    xs = [z[1] for z in zhist]
    ys = [z[2] for z in zhist]

    plot!(p, xs, ys; lw=2.5, marker=:circle, ms=3.5, label="optimization path")
    scatter!(p, [xs[1]], [ys[1]]; ms=9, marker=:star5, label="z₀")
    scatter!(p, [zstar[1]], [zstar[2]]; ms=10, marker=:diamond, label="estimated z*")

    filename = "probing_heatmap_alpha3_$(ALPHA[1])_alpha4_$(ALPHA[2])_beta_$(BETA).png"
    savefig(p, filename)    
    return p, grid, Z
end

function convergence_plot(zhist, Jhist)
    it = 0:(length(zhist)-1)
    z3 = [z[1] for z in zhist]
    z4 = [z[2] for z in zhist]

    p = plot(
        it, z3;
        xlabel="iteration",
        ylabel="value",
        label="z₃",
        lw=2,
        marker=:circle,
        title="Evolution of probing decisions"
    )
    plot!(p, it, z4; label="z₄", lw=2, marker=:circle)
    savefig(p, "probing_convergence.png")

    pJ = plot(
        it, Jhist;
        xlabel="iteration",
        ylabel="J(z)",
        label="objective",
        lw=2,
        marker=:circle,
        title="Objective evolution"
    )
    savefig(pJ, "probing_objective_convergence.png")

    return p, pJ
end

# =========================
# 12. INTERIOR-OPTIMUM DIAGNOSTIC
# =========================

function corner_and_midpoint_diagnostic(bank, Y)
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.50],
        [0.50, 0.25],
        [0.75, 0.50],
        [0.50, 0.75],
    ]

    println("\nObjective diagnostic:")
    for z in pts
        J, i, gross, meanL, stdL =
            objective_and_gradient(z, bank, Y; need_gradient=false)
        @printf(
            " z=(%.2f,%.2f): J=%10.4f  gross=%10.4f  policy=%3d  E[L]=%.3f SD[L]=%.3f\n",
            z[1], z[2], J, gross, i, meanL, stdL
        )
    end
end

# =========================
# 13. MAIN
# =========================

function main()
    println("==============================================================")
    println(" Two-server probing model: 2-D LR heatmap + z trajectory")
    println("==============================================================")
    @printf("Active probing dimensions: classes %d and %d\n", ACTIVE_CLASSES...)
    @printf("alpha = (%.3f, %.3f), beta = %.3f\n", ALPHA[1], ALPHA[2], BETA)
    @printf("paths = %d, jobs/path = %d, seed = %d\n\n", N_PATHS, JOBS_PER_PATH, SEED)

    bank = generate_bank()
    policies = build_policy_grid()

    @printf("Policy grid contains %d threshold policies.\n", length(policies))
    Y = precompute_rewards(policies, bank)

    # Sanity check: under z=0, L=1 exactly.
    J00, i00, gross00, meanL00, stdL00 =
        objective_and_gradient([0.0, 0.0], bank, Y; need_gradient=false)

    @printf("\nBaseline z=(0,0): J=%.4f, gross=%.4f, best policy=%d, E[L]=%.6f, SD[L]=%.6f\n",
            J00, gross00, i00, meanL00, stdL00)

    corner_and_midpoint_diagnostic(bank, Y)

    # Validate analytic gradient away from policy-switch boundaries.
    gradient_check([0.35, 0.35], bank, Y)

    println("Projected gradient ascent:")
    zstar, Jstar, zhist, Jhist, ihist = optimize_z(Z0, bank, Y)

    println("\n==============================================================")
    @printf("FINAL RESULT\n")
    @printf("z0                  = [%.6f, %.6f]\n", Z0[1], Z0[2])
    @printf("estimated z*        = [%.6f, %.6f]\n", zstar[1], zstar[2])
    @printf("estimated J(z*)     = %.6f\n", Jstar)
    @printf("probing costs alpha = [%.6f, %.6f]\n", ALPHA[1], ALPHA[2])
    println("==============================================================")

    pheat, grid, Z = build_heatmap(bank, Y, zhist, zstar)
    pconv, pobj = convergence_plot(zhist, Jhist)

    # Also report best point on the plotted grid.
    idx = argmax(Z)
    iy, ix = Tuple(idx)
    zg = [grid[ix], grid[iy]]
    @printf("\nBest heat-map grid point = [%.4f, %.4f], J=%.6f\n",
            zg[1], zg[2], Z[iy, ix])

    println("\nSaved:")
    println("  probing_heatmap_trajectory.png")
    println("  probing_convergence.png")
    println("  probing_objective_convergence.png")

    display(pheat)
end

main()
