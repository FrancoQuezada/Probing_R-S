# SimOpt007_continuous_lr_stabilized.jl
# Continuous probing with stabilized LR gradients


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
# SIGNAL MODEL
# ============================================================

function signal_probs(true_class, z;
    p_hit=[0.25,0.40,0.75,0.92],
    p_false=[0.05,0.08,0.15,0.20],
    beta=3.0)

    ph = p_hit[true_class] + (1.0 - p_hit[true_class]) * (1.0 - exp(-beta * z[true_class]))
    pf = p_false[true_class] * exp(-beta * z[true_class])

    return ph, pf
end

# ============================================================
# SIMULATION WITH SCORE FUNCTION
# ============================================================

function simulate_with_score(policy, z, rng;
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

    score = zeros(m)

    for n in 1:N

        # arrivals
        interarrival = rand(rng, Exponential(mean_interarrival))
        arrival = current_time + interarrival
        current_time = arrival

        demand = rand(rng)
        true_class = rand(rng, 1:m)
        complex = rand(rng) < complex_prob_by_class[true_class]

        # --- probing decision ---
        u = rand(rng)
        prob_probe = z[true_class]

        prob_probe = clamp(prob_probe, 1e-6, 1-1e-6)

        if u < prob_probe
            score[true_class] += 1 / prob_probe
            probed = true
        else
            score[true_class] += -1 / (1 - prob_probe)
            probed = false
        end

        signal = nothing

        if probed
            ph, pf = signal_probs(true_class, z)

            ph = clamp(ph, 1e-6, 1-1e-6)
            pf = clamp(pf, 1e-6, 1-1e-6)

            if complex
                s = rand(rng) < ph
                score[true_class] += (s ? 1/ph : -1/(1-ph))
            else
                s = rand(rng) < pf
                score[true_class] += (s ? 1/pf : -1/(1-pf))
            end

            signal = s ? 1.0 : 0.0
        end

        baseline = complex_prob_by_class[true_class]
        signal_used = isnothing(signal) ? baseline : 0.5*baseline + 0.5*signal

        # FE
        t_fe = rand(rng, Gamma(2.0, 0.9))
        fe_start = max(arrival, fe_available)
        fe_end = fe_start + t_fe
        fe_available = fe_end

        # transfer
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

            if complex
                p_disrupt = 1.0 - 0.75 * z[true_class]
                p_disrupt = clamp(p_disrupt, 1e-6, 1-1e-6)

                d = rand(rng) < p_disrupt
                score[true_class] += (d ? 1/p_disrupt : -1/(1-p_disrupt))

                extra = d ? rand(rng, Exponential(20.0)) : 0.0
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

    return total_profit, score
end

# ============================================================
# GRADIENT ESTIMATION (WITH BASELINE)
# ============================================================

function estimate_gradient(policy, z, seed; reps=50, alpha=zeros(length(z)))

    m = length(z)
    vals = Float64[]

    for r in 1:reps
        rng = MersenneTwister(seed + r)
        val, _ = simulate_with_score(policy, z, rng)
        push!(vals, val)
    end

    baseline = mean(vals)

    grad = zeros(m)

    for r in 1:reps
        rng = MersenneTwister(seed + r)
        val, score = simulate_with_score(policy, z, rng)
        grad .+= (val - baseline) .* score
    end

    grad ./= reps
    grad .-= alpha

    return grad
end

# ============================================================
# PROJECTION
# ============================================================

function project!(z, eps=0.05)
    for j in eachindex(z)
        z[j] = min(1.0 - eps, max(eps, z[j]))
    end
end

# ============================================================
# OPTIMIZATION
# ============================================================

function optimize_z(policies;
    max_iter=20,
    step_size=1e-5,
    train_reps=50,
    grad_reps=50,
    alpha=zeros(4),
    eps=0.05,
    z_tol=1e-3,
    patience=5)

    m = length(alpha)
    z = fill(0.5, m)

       z = fill(0.5, m)
    small_change_count = 0

    println("iter  z                              value        policy   grad")
    println("-------------------------------------------------------------------------------------------------------")

    for it in 1:max_iter

        best_policy = nothing
        best_val = -Inf

        for p in policies
            vals = Float64[]
            for r in 1:train_reps
                rng = MersenneTwister(1000*it + r + p.id)
                push!(vals, simulate_with_score(p, z, rng)[1])
            end

            μ = mean(vals)

            if μ > best_val
                best_val = μ
                best_policy = p
            end
        end

        grad = estimate_gradient(best_policy, z, 10_000*it; reps=grad_reps, alpha=alpha)

        z_old = copy(z)
        z .+= step_size .* grad
        project!(z, eps)

        println(rpad(it,5),
                rpad(string(round.(z, digits=4)),30),
                round(best_val, digits=2),"      ",
                best_policy.id,"      ",
                round.(grad, digits=2))

        if norm(z - z_old) < z_tol
            small_change_count += 1
        else
            small_change_count = 0
        end

        if small_change_count >= patience
            println("\nStopping: z changed by less than ", z_tol,
                    " for ", patience, " consecutive iterations.")
            break
        end
    end

    return z
end

# ============================================================
# MAIN
# ============================================================

function run()
    policies = generate_policies()
    alpha = [0.0, 0.0, 0.0, 0.0] # Probing costs

    println("Policies: ", length(policies))
    println("Running stabilized LR optimization...\n")

    z = optimize_z(policies;
        max_iter=20,
        step_size=1e-5,
        train_reps=50,
        grad_reps=100,
        alpha=alpha,
        eps=0.05,
        z_tol=5e-4,
        patience=8
    )

    println("\nFinal z = ", z)
end

run()