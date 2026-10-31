using Random
using Distributions
using Statistics

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

function choose_fast_be(p::Policy, demand, fast_q, std_q, signal::Float64)
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
# SIGNAL MODEL
# ============================================================

function signal_probs(true_class, z;
    p_hit=[0.25, 0.40, 0.75, 0.92],
    p_false=[0.05, 0.08, 0.15, 0.20],
    beta=3.0)

    ph = p_hit[true_class] + (1.0 - p_hit[true_class]) * (1.0 - exp(-beta * z[true_class]))
    pf = p_false[true_class] * exp(-beta * z[true_class])

    return ph, pf
end

# ============================================================
# SAMPLE PATH GENERATION
# ============================================================
# For a fixed z, we generate one full sample path eta_z.
# Then every policy is evaluated on that same path.
# The score is the likelihood-ratio score of the entire path.
# ============================================================

function generate_sample_path(z, rng;
    N=200,
    mean_interarrival=5.0,
    complex_prob_by_class=[0.02, 0.12, 0.55, 0.95],
    p_hit=[0.25, 0.40, 0.75, 0.92],
    p_false=[0.05, 0.08, 0.15, 0.20],
    beta=3.0)

    m = length(z)

    interarrivals = Vector{Float64}(undef, N)
    demands = Vector{Float64}(undef, N)
    classes = Vector{Int}(undef, N)
    complex_flags = Vector{Bool}(undef, N)

    t_fe = Vector{Float64}(undef, N)
    transfer = Vector{Float64}(undef, N)
    t_be_fast = Vector{Float64}(undef, N)
    t_be_std = Vector{Float64}(undef, N)
    disruption_delay = Vector{Float64}(undef, N)

    probed = Vector{Bool}(undef, N)
    signal_bits = falses(N)
    disruption_bits = falses(N)
    signal_used = Vector{Float64}(undef, N)

    score = zeros(m)

    for n in 1:N
        interarrivals[n] = rand(rng, Exponential(mean_interarrival))
        demands[n] = rand(rng)
        classes[n] = rand(rng, 1:m)
        c = classes[n]

        complex_flags[n] = rand(rng) < complex_prob_by_class[c]
        complex = complex_flags[n]

        t_fe[n] = rand(rng, Gamma(2.0, 0.9))
        transfer[n] = rand(rng, Exponential(1.0))
        t_be_fast[n] = rand(rng, Gamma(2.0, 1.15))
        t_be_std[n] = rand(rng, Gamma(2.0, 1.5))
        disruption_delay[n] = rand(rng, Exponential(20.0))

        p_probe = clamp(z[c], 1e-6, 1 - 1e-6)
        u_probe = rand(rng)
        probed[n] = u_probe < p_probe
        score[c] += probed[n] ? 1 / p_probe : -1 / (1 - p_probe)

        baseline = complex_prob_by_class[c]

        if probed[n]
            ph, pf = signal_probs(c, z; p_hit=p_hit, p_false=p_false, beta=beta)
            ph = clamp(ph, 1e-6, 1 - 1e-6)
            pf = clamp(pf, 1e-6, 1 - 1e-6)

            if complex
                signal_bits[n] = rand(rng) < ph
                dph_dz = (1 - p_hit[c]) * beta * exp(-beta * z[c])
                score[c] += signal_bits[n] ? (1 / ph) * dph_dz : (-1 / (1 - ph)) * dph_dz
            else
                signal_bits[n] = rand(rng) < pf
                dpf_dz = -p_false[c] * beta * exp(-beta * z[c])
                score[c] += signal_bits[n] ? (1 / pf) * dpf_dz : (-1 / (1 - pf)) * dpf_dz
            end

            signal_used[n] = 0.5 * baseline + 0.5 * (signal_bits[n] ? 1.0 : 0.0)
        else
            signal_bits[n] = false
            signal_used[n] = baseline
        end

        if complex
            p_disrupt = clamp(1.0 - 0.75 * z[c], 1e-6, 1 - 1e-6)
            disruption_bits[n] = rand(rng) < p_disrupt
            dp_dz = -0.75
            score[c] += disruption_bits[n] ? (1 / p_disrupt) * dp_dz : (-1 / (1 - p_disrupt)) * dp_dz
        else
            disruption_bits[n] = false
        end
    end

    return (
        N=N,
        interarrivals=interarrivals,
        demands=demands,
        classes=classes,
        complex_flags=complex_flags,
        t_fe=t_fe,
        transfer=transfer,
        t_be_fast=t_be_fast,
        t_be_std=t_be_std,
        disruption_delay=disruption_delay,
        probed=probed,
        signal_bits=signal_bits,
        disruption_bits=disruption_bits,
        signal_used=signal_used,
        score=score,
    )
end

# ============================================================
# EVALUATE ONE POLICY ON A FIXED SAMPLE PATH
# ============================================================

function evaluate_policy_on_path(policy, path;
    revenue_scale=40.0,
    due_offset=20.0,
    fast_be_premium=3.0)

    fe_available = 0.0
    be_fast_available = 0.0
    be_std_available = 0.0

    current_time = 0.0
    total_profit = 0.0

    for n in 1:path.N
        arrival = current_time + path.interarrivals[n]
        current_time = arrival

        demand = path.demands[n]
        complex = path.complex_flags[n]
        signal = path.signal_used[n]

        fe_start = max(arrival, fe_available)
        fe_end = fe_start + path.t_fe[n]
        fe_available = fe_end

        transfer_end = fe_end + path.transfer[n]

        fast_q = max(0.0, be_fast_available - transfer_end)
        std_q = max(0.0, be_std_available - transfer_end)

        send_fast = choose_fast_be(policy, demand, fast_q, std_q, signal)

        if send_fast
            t_be = path.t_be_fast[n]
            start = max(transfer_end, be_fast_available)
            extra = 0.0
            finish = start + t_be
            be_fast_available = finish
            routing_premium = fast_be_premium
        else
            t_be = path.t_be_std[n]
            start = max(transfer_end, be_std_available)
            extra = (complex && path.disruption_bits[n]) ? path.disruption_delay[n] : 0.0
            finish = start + t_be + extra
            be_std_available = finish
            routing_premium = 0.0
        end

        tardiness = max(0.0, finish - (arrival + due_offset))
        revenue = revenue_scale * (0.5 + demand)

        cost =
            1.0 * path.t_fe[n] +
            0.8 * path.transfer[n] +
            1.2 * (t_be + extra) +
            2.5 * tardiness +
            routing_premium

        total_profit += revenue - cost
    end

    return total_profit
end

# ============================================================
# MAX-INSIDE-EXPECTATION ESTIMATORS
# ============================================================

function estimate_Q_and_gradient(z, policies, seed;
    reps=1000,
    alpha=zeros(length(z)),
    N=200)

    m = length(z)
    max_vals = Vector{Float64}(undef, reps)
    scores = Matrix{Float64}(undef, reps, m)

    for r in 1:reps
        rng = MersenneTwister(seed + r)
        path = generate_sample_path(z, rng; N=N)

        best_val = -Inf
        for p in policies
            val = evaluate_policy_on_path(p, path)
            if val > best_val
                best_val = val
            end
        end

        max_vals[r] = best_val
        scores[r, :] .= path.score
    end

    baseline = mean(max_vals)
    grad = zeros(m)
    for r in 1:reps
        grad .+= (max_vals[r] - baseline) .* @view(scores[r, :])
    end
    grad ./= reps
    grad .-= alpha

    obj = baseline - dot(alpha, z)
    se = std(max_vals) / sqrt(reps)

    return (value=baseline, objective=obj, gradient=grad, se=se)
end

function project!(z, eps=0.05)
    for j in eachindex(z)
        z[j] = min(1.0 - eps, max(eps, z[j]))
    end
end

# ============================================================
# OPTIMIZATION
# ============================================================

function optimize_z(policies;
    max_iter=50,
    step_size=5e-5,
    reps=1000,
    alpha=zeros(4),
    eps=0.05,
    N=200,
    z0=fill(0.8, length(alpha)))

    z = copy(z0)

    println("iter  z                                  Qhat         objective    grad")
    println("-----------------------------------------------------------------------------------------------------------")

    for it in 1:max_iter
        est = estimate_Q_and_gradient(z, policies, 10_000 * it;
            reps=reps,
            alpha=alpha,
            N=N)

        grad = est.gradient
        z .+= step_size .* grad
        project!(z, eps)

        println(rpad(it, 5),
                rpad(string(round.(z, digits=4)), 30),
                rpad(string(round(est.value, digits=2)), 13),
                rpad(string(round(est.objective, digits=2)), 13),
                round.(grad, digits=2))
    end

    return z
end

# ============================================================
# EVALUATION
# ============================================================

function evaluate_Q(z, policies;
    reps=2000,
    seed=1234,
    alpha=zeros(length(z)),
    N=200)

    est = estimate_Q_and_gradient(z, policies, seed;
        reps=reps,
        alpha=alpha,
        N=N)

    return (
        z=copy(z),
        Q=est.value,
        objective=est.objective,
        se=est.se,
    )
end

# ============================================================
# MAIN
# ============================================================

function run()
    policies = generate_policies()

    #alpha = [30.0, 40.0, 50.0, 60.0] # probing costs
    alpha = [0.0, 0.0, 0.0, 0.0] # probing costs

    println("Policies: ", length(policies))
    println("Running correct max-inside-expectation LR optimization...\n")

    z = optimize_z(policies;
        max_iter=10, # Number of iterations of the algorithm.
        step_size=3e-5,
        reps=1000,
        alpha=alpha,
        N=200,
        z0=[0.8, 0.8, 0.8, 0.8], # Initial z
    )

    println("\nFinal z = ", z)

    println("\nSANITY CHECKS:\n")
    println("z = 0000 -> ", evaluate_Q([0.0, 0.0, 0.0, 0.0], policies; reps=1000, alpha=alpha))
    println("z = 1111 -> ", evaluate_Q([1.0, 1.0, 1.0, 1.0], policies; reps=1000, alpha=alpha))
    println("z = 0.5  -> ", evaluate_Q([0.5, 0.5, 0.5, 0.5], policies; reps=1000, alpha=alpha))
    println("z = z*   -> ", evaluate_Q(z, policies; reps=1000, alpha=alpha))

    return z
end

#run() # Turn on if you want the 4D experiment to be run
