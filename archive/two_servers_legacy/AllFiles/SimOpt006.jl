using Random
using Distributions
using Statistics
using LinearAlgebra
using Printf

# ============================================================
# POLICY
# ============================================================

struct Policy
    id::Int
    a_demand::Float64
    a_queue::Float64
    a_signal::Float64
    threshold::Float64
end

function choose_fast_be(p::Policy, demand, fast_q, std_q, signal)
    queue_diff = fast_q - std_q
    score = p.a_demand * demand - p.a_queue * queue_diff + p.a_signal * signal
    return score >= p.threshold
end

# ============================================================
# POLICY GRID
# ============================================================

function generate_policies()
    a_demand_grid = range(0.5, 2.5, length=6)
    a_queue_grid = range(0.2, 1.8, length=6)
    a_signal_grid = range(0.0, 2.0, length=5)
    threshold_grid = range(0.2, 1.4, length=5)

    policies = Policy[]
    id = 1
    for a_d in a_demand_grid,
        a_q in a_queue_grid,
        a_s in a_signal_grid,
        th in threshold_grid
        push!(policies, Policy(id, a_d, a_q, a_s, th))
        id += 1
    end
    return policies
end

# ============================================================
# Z-DEPENDENT EVENT GENERATION + SCORE TERMS
# ============================================================

# Returns:
#   signal::Union{Nothing, Float64}
#   score::Vector{Float64}   -> contribution to ∇_z log P(path)
#
# The score-function estimator uses:
#   ∇ E[Y(z)] = E[Y(z) * ∇ log P(path; z)]
#
# We therefore accumulate the score contributions from all random events whose
# distribution depends on z.

function signal_and_score!(rng, true_class::Int, complex::Bool, z::Vector{Float64}, score::Vector{Float64};
    p_hit   = [0.25, 0.40, 0.75, 0.92],
    p_false = [0.05, 0.08, 0.15, 0.20],
    beta = 3.0)

    j = true_class
    zj = z[j]

    # Clamp away from 0 and 1 so log-derivatives remain finite.
    zj = clamp(zj, 1e-6, 1 - 1e-6)

    # Probe occurrence: B ~ Bernoulli(zj)
    probed = rand(rng) < zj
    if probed
        score[j] += 1.0 / zj
    else
        score[j] += -1.0 / (1.0 - zj)
        return nothing
    end

    # Signal quality conditional on probing
    ph0 = p_hit[j]
    pf0 = p_false[j]

    e = exp(-beta * zj)
    ph = ph0 + (1.0 - ph0) * (1.0 - e)
    pf = pf0 * e

    dph = beta * (1.0 - ph0) * e
    dpf = -beta * pf0 * e

    if complex
        s = rand(rng) < ph ? 1.0 : 0.0
        if s == 1.0
            score[j] += dph / ph
        else
            score[j] += -dph / (1.0 - ph)
        end
        return s
    else
        s = rand(rng) < pf ? 1.0 : 0.0
        if s == 1.0
            score[j] += dpf / pf
        else
            score[j] += -dpf / (1.0 - pf)
        end
        return s
    end
end

function disruption_extra_and_score!(rng, true_class::Int, complex::Bool, z::Vector{Float64}, score::Vector{Float64})
    if !complex
        return 0.0
    end

    j = true_class
    zj = z[j]
    # disruption probability if complex and sent to standard
    p = 1.0 - 0.75 * zj
    p = clamp(p, 1e-6, 1 - 1e-6)
    dp = -0.75

    disrupted = rand(rng) < p
    if disrupted
        score[j] += dp / p
        return rand(rng, Exponential(20.0))
    else
        score[j] += -dp / (1.0 - p)
        return 0.0
    end
end

# ============================================================
# SIMULATION
# ============================================================

# Returns one replication and its LR gradient estimator for a fixed policy.
# Gradient is for E[profit(policy,z)] - alpha'z.

function simulate_replication_with_lr(policy::Policy, z::Vector{Float64}, rng;
    N=200,
    mean_interarrival=5.0,
    revenue_scale=40.0,
    due_offset=20.0,
    complex_prob_by_class=[0.02, 0.12, 0.55, 0.95],
    fast_be_premium=3.0,
    alpha=zeros(Float64, length(z)))

    m = length(z)

    fe_available = 0.0
    be_fast_available = 0.0
    be_std_available = 0.0

    current_time = 0.0
    total_profit = 0.0
    total_score = zeros(Float64, m)

    for n in 1:N
        interarrival = rand(rng, Exponential(mean_interarrival))
        arrival = current_time + interarrival
        current_time = arrival

        demand = rand(rng)
        true_class = rand(rng, 1:m)
        complex = rand(rng) < complex_prob_by_class[true_class]

        # score contributions for this job
        job_score = zeros(Float64, m)

        signal = signal_and_score!(rng, true_class, complex, z, job_score)

        baseline = complex_prob_by_class[true_class]
        signal_used = isnothing(signal) ? baseline : 0.5 * baseline + 0.5 * signal

        t_fe = rand(rng, Gamma(2.0, 0.9))
        fe_start = max(arrival, fe_available)
        fe_end = fe_start + t_fe
        fe_available = fe_end

        transfer = rand(rng, Exponential(1.0))
        transfer_end = fe_end + transfer

        fast_q = max(0.0, be_fast_available - transfer_end)
        std_q = max(0.0, be_std_available - transfer_end)

        send_fast = choose_fast_be(policy, demand, fast_q, std_q, signal_used)

        if send_fast
            t_be = rand(rng, Gamma(2.0, 1.15))
            start = max(transfer_end, be_fast_available)
            extra = 0.0
            finish = start + t_be
            be_fast_available = finish
            routing_premium = fast_be_premium
        else
            t_be = rand(rng, Gamma(2.0, 1.5))
            start = max(transfer_end, be_std_available)
            extra = disruption_extra_and_score!(rng, true_class, complex, z, job_score)
            finish = start + t_be + extra
            be_std_available = finish
            routing_premium = 0.0
        end

        tardiness = max(0.0, finish - (arrival + due_offset))
        revenue = revenue_scale * (0.5 + demand)

        cost =
            1.0 * t_fe +
            0.8 * transfer +
            1.2 * (t_be + extra) +
            2.5 * tardiness +
            routing_premium

        total_profit += revenue - cost
        total_score .+= job_score
    end

    value = total_profit - dot(alpha, z)
    grad = value .* total_score .- alpha
    return value, grad
end

# Plain simulation for training / policy selection
function simulate_replication(policy::Policy, z::Vector{Float64}, rng;
    N=200,
    mean_interarrival=5.0,
    revenue_scale=40.0,
    due_offset=20.0,
    complex_prob_by_class=[0.02, 0.12, 0.55, 0.95],
    fast_be_premium=3.0)

    m = length(z)

    fe_available = 0.0
    be_fast_available = 0.0
    be_std_available = 0.0

    current_time = 0.0
    total_profit = 0.0

    for n in 1:N
        interarrival = rand(rng, Exponential(mean_interarrival))
        arrival = current_time + interarrival
        current_time = arrival

        demand = rand(rng)
        true_class = rand(rng, 1:m)
        complex = rand(rng) < complex_prob_by_class[true_class]

        signal = begin
            zj = clamp(z[true_class], 1e-6, 1 - 1e-6)
            if rand(rng) > zj
                nothing
            else
                ph0 = [0.25, 0.40, 0.75, 0.92][true_class]
                pf0 = [0.05, 0.08, 0.15, 0.20][true_class]
                e = exp(-3.0 * zj)
                ph = ph0 + (1.0 - ph0) * (1.0 - e)
                pf = pf0 * e
                if complex
                    rand(rng) < ph ? 1.0 : 0.0
                else
                    rand(rng) < pf ? 1.0 : 0.0
                end
            end
        end

        baseline = complex_prob_by_class[true_class]
        signal_used = isnothing(signal) ? baseline : 0.5 * baseline + 0.5 * signal

        t_fe = rand(rng, Gamma(2.0, 0.9))
        fe_start = max(arrival, fe_available)
        fe_end = fe_start + t_fe
        fe_available = fe_end

        transfer = rand(rng, Exponential(1.0))
        transfer_end = fe_end + transfer

        fast_q = max(0.0, be_fast_available - transfer_end)
        std_q = max(0.0, be_std_available - transfer_end)

        send_fast = choose_fast_be(policy, demand, fast_q, std_q, signal_used)

        if send_fast
            t_be = rand(rng, Gamma(2.0, 1.15))
            start = max(transfer_end, be_fast_available)
            extra = 0.0
            finish = start + t_be
            be_fast_available = finish
            routing_premium = fast_be_premium
        else
            t_be = rand(rng, Gamma(2.0, 1.5))
            start = max(transfer_end, be_std_available)
            if complex && rand(rng) < 1.0 - 0.75 * z[true_class]
                extra = rand(rng, Exponential(20.0))
            else
                extra = 0.0
            end
            finish = start + t_be + extra
            be_std_available = finish
            routing_premium = 0.0
        end

        tardiness = max(0.0, finish - (arrival + due_offset))
        revenue = revenue_scale * (0.5 + demand)
        cost =
            1.0 * t_fe +
            0.8 * transfer +
            1.2 * (t_be + extra) +
            2.5 * tardiness +
            routing_premium

        total_profit += revenue - cost
    end

    return total_profit
end

# ============================================================
# POLICY SELECTION FOR A FIXED z
# ============================================================

function select_best_policy(z::Vector{Float64}, policies;
    train_reps=50,
    seed=1234)

    best_policy = nothing
    best_value = -Inf

    for p in policies
        vals = Float64[]
        for r in 1:train_reps
            rng = MersenneTwister(seed + 10_000 * p.id + r)
            push!(vals, simulate_replication(p, z, rng))
        end
        μ = mean(vals)
        if μ > best_value
            best_value = μ
            best_policy = p
        end
    end

    return best_policy, best_value
end

# ============================================================
# EVALUATION + LR GRADIENT FOR A FIXED POLICY
# ============================================================

function evaluate_policy_value_and_gradient(policy::Policy, z::Vector{Float64};
    reps=200,
    seed=777,
    alpha=zeros(Float64, length(z)))

    vals = Float64[]
    grads = Vector{Float64}[]

    for r in 1:reps
        rng = MersenneTwister(seed + 1000 * r)
        v, g = simulate_replication_with_lr(policy, z, rng; alpha=alpha)
        push!(vals, v)
        push!(grads, g)
    end

    gmat = reduce(hcat, grads)
    mean_grad = vec(mean(gmat, dims=2))

    return (
        value = mean(vals),
        se = std(vals) / sqrt(reps),
        grad = mean_grad,
        grad_se = vec(std(gmat, dims=2)) ./ sqrt(reps)
    )
end

# ============================================================
# CONTINUOUS OUTER OPTIMIZATION
# ============================================================

project01!(z) = (z .= clamp.(z, 1e-4, 1 - 1e-4))

function optimize_z_lr(policies;
    m=4,
    alpha=fill(40.0, m),
    iters=20,
    train_reps=40,
    grad_reps=200,
    step0=0.15,
    seed=2026,
    z0=fill(0.5, m))

    z = copy(z0)
    project01!(z)

    history = []

    @printf("%-5s %-28s %-12s %-8s %s\n", "iter", "z", "value", "policy", "grad")
    println(repeat("-", 85))

    for k in 1:iters
        pol, train_val = select_best_policy(z, policies; train_reps=train_reps, seed=seed + 50_000 * k)
        out = evaluate_policy_value_and_gradient(pol, z; reps=grad_reps, seed=seed + 100_000 * k, alpha=alpha)

        η = step0 / sqrt(k)
        z .+= η .* out.grad
        project01!(z)

        push!(history, (
            iter = k,
            z = copy(z),
            value = out.value,
            se = out.se,
            policy = pol.id,
            grad = copy(out.grad),
            grad_se = copy(out.grad_se),
            train_value = train_val
        ))

        @printf("%-5d %-28s %-12.2f %-8d [%7.3f, %7.3f, %7.3f, %7.3f]\n",
            k,
            string(round.(z, digits=3)),
            out.value,
            pol.id,
            out.grad[1], out.grad[2], out.grad[3], out.grad[4])
    end

    return z, history
end

# ============================================================
# FINAL INDEPENDENT EVALUATION
# ============================================================

function final_assessment(z::Vector{Float64}, policies;
    train_reps=100,
    eval_reps=1000,
    seed=9999,
    alpha=fill(40.0, length(z)))

    pol, train_val = select_best_policy(z, policies; train_reps=train_reps, seed=seed)
    vals = Float64[]
    for r in 1:eval_reps
        rng = MersenneTwister(seed + 1_000_000 + r)
        push!(vals, simulate_replication(pol, z, rng) - dot(alpha, z))
    end

    return (
        z = copy(z),
        policy = pol.id,
        train_value = train_val,
        value = mean(vals),
        se = std(vals) / sqrt(eval_reps)
    )
end

# ============================================================
# MAIN
# ============================================================

function run()
    m = 4
    alpha = [35.0, 35.0, 35.0, 35.0]

    policies = generate_policies()
    println("Policies: ", length(policies))
    println("Running continuous probing with LR gradients...\n")

    z_star, history = optimize_z_lr(policies;
        m=m,
        alpha=alpha,
        iters=20,
        train_reps=40,
        grad_reps=250,
        step0=0.12,
        seed=2026,
        z0=[0.5, 0.5, 0.5, 0.5])

    println("\nFinal z from optimizer: ", round.(z_star, digits=4))

    out = final_assessment(z_star, policies;
        train_reps=100,
        eval_reps=1000,
        seed=4242,
        alpha=alpha)

    println("\nIndependent final assessment")
    println("z        = ", round.(out.z, digits=4))
    println("policy   = ", out.policy)
    println("value    = ", round(out.value, digits=2))
    println("se       = ", round(out.se, digits=2))
    println("trainval = ", round(out.train_value, digits=2))
end

run()
