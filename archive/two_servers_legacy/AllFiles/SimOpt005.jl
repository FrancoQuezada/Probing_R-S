# SimOpt005_twostage_fixed.jl
# This version includes a premium cost for using the fast BE, which prevents degenerate policies that always choose the fast BE even when it's only slightly faster.
# Probing model + two-stage evaluation
# The two-stage evaluation procedure is as follows: in the first stage, we evaluate all policies on a training set of replications and select the best policy based on average profit. 
# In the second stage, we take the selected best policy and evaluate it on an independent test set of replications to get an unbiased estimate of its performance. 
# This allows us to see how well the best policy from the training stage generalizes to new data, and to compare different probing decisions (z-vectors) in a more robust way.
# Fixed version:
# - removes degenerate always-fast dominance
# - adds a premium cost for using fast BE
# - keeps probing as direct signal generation (no posterior, no latent inference layer)

using Random
using Distributions
using Statistics
using LinearAlgebra

# ============================================================
# POLICY
# ============================================================

struct Policy # Data container for policies. The 4 parameters are weights for a simple linear score that determines whether to send to fast BE or not.
    id::Int
    a_demand::Float64 # weight on demand, which is a proxy for how valuable the job is. Higher demand means more revenue, so the policy might want to prioritize those jobs for the fast BE.
    a_queue::Float64 # weight on queue difference, which is the difference in queue lengths (or waiting times) between the fast BE and the standard BE. A positive weight means the policy prefers to send to the fast BE when its queue is shorter than the standard BE, and a negative weight means the opposite.
    a_signal::Float64 # weight on signal, which is the output of the probing. The signal is a noisy indicator of whether the job is complex or not. A higher weight means the policy relies more on the probing signal to make the routing decision.
    threshold::Float64 # threshold for sending to fast BE. The policy calculates a score based on the demand, queue difference, and signal, and compares it to this threshold to decide whether to send to the fast BE or not.
end

function choose_fast_be(p::Policy, demand, fast_q, std_q, signal) # simple linear policy based on demand, queue difference, and signal
    queue_diff = fast_q - std_q
    score = p.a_demand * demand - p.a_queue * queue_diff + p.a_signal * signal
    return score >= p.threshold
end

# ============================================================
# POLICY GRID
# ============================================================

function generate_policies() 
    # narrower, less degenerate grid
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
# SIGNAL GENERATION
# ============================================================

function generate_signal(rng, true_class, complex, z; # parameters for signal quality as a function of probing effort
    p_hit   = [0.25, 0.40, 0.75, 0.92], # probability of correctly identifying a complex job, by class
    p_false = [0.05, 0.08, 0.15, 0.20], # probability of incorrectly classifying a simple job as complex, by class
    beta = 3.0) # This parenthesis closes the list of keyword arguments for signal generation

    # no inspection
    if rand(rng) > z[true_class]
        return nothing
    end

    # signal quality improves with more probing effort, but with diminishing returns
    ph = p_hit[true_class] + (1.0 - p_hit[true_class]) * (1.0 - exp(-beta * z[true_class]))
    pf = p_false[true_class] * exp(-beta * z[true_class])

    if complex
        return rand(rng) < ph ? 1.0 : 0.0
    else
        return rand(rng) < pf ? 1.0 : 0.0
    end
end

# ============================================================
# SIMULATION
# ============================================================

function simulate_replication(policy::Policy, z, rng; # function that simulates one replication of the system under a given policy and probing vector, returning total profit for that replication
    N=200,
    mean_interarrival=5.0,
    revenue_scale=40.0,
    due_offset=20.0,
    complex_prob_by_class=[0.02, 0.12, 0.55, 0.95],
    fast_be_premium = 3.0) # Sending to the faster BE (BE stands for back-end) is more expensive, so we add a fixed premium cost for using it. 
    #This is important to prevent degenerate policies that always choose the fast BE even when it's only slightly faster, which can happen if the cost of using the fast BE is not properly accounted for. 
    # The premium cost encourages the policy to be more selective about when to use the fast BE, and to rely on the probing signal and other factors to make that decision.
    
    m = length(z)

    fe_available = 0.0
    be_fast_available = 0.0
    be_std_available = 0.0

    current_time = 0.0
    total_profit = 0.0

    for n in 1:N

        # arrivals and lot attributes
        interarrival = rand(rng, Exponential(mean_interarrival))
        arrival = current_time + interarrival
        current_time = arrival

        demand = rand(rng)
        true_class = rand(rng, 1:m)
        complex = rand(rng) < complex_prob_by_class[true_class]

        # probing signal
        signal = generate_signal(rng, true_class, complex, z)

        # use probing to refine the baseline rather than overwrite it
        baseline = complex_prob_by_class[true_class]
        signal_used = isnothing(signal) ? baseline : 0.5 * baseline + 0.5 * signal

        # FE
        t_fe = rand(rng, Gamma(2.0, 0.9))
        fe_start = max(arrival, fe_available)
        fe_end = fe_start + t_fe
        fe_available = fe_end

        # transfer
        transfer = rand(rng, Exponential(1.0))
        transfer_end = fe_end + transfer

        # queue states at routing
        fast_q = max(0.0, be_fast_available - transfer_end)
        std_q = max(0.0, be_std_available - transfer_end)

        # routing
        send_fast = choose_fast_be(policy, demand, fast_q, std_q, signal_used)

        if send_fast
            # fast BE is faster, but not absurdly better
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
                # probing reduces the probability of a disruption if sent to standard
                disruption_prob = 1.0 - 0.75 * z[true_class]

                if rand(rng) < disruption_prob
                    extra = rand(rng, Exponential(20.0))
                else
                    extra = 0.0
                end
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
# TWO-STAGE EVALUATION
# ============================================================

function evaluate_Q_two_stage(z, policies; # function that evaluates a set of policies using a two-stage procedure: first it selects the best policy based on a training set of replications, then it evaluates that policy on an independent test set of replications, returning the test set mean and standard error
    train_reps=100,
    eval_reps=1000,
    seed=1234,
    alpha=zeros(Float64, length(z)))

    best_policy = nothing
    best_train_value = -Inf

    # Stage 1: policy selection
    for p in policies
        vals = Float64[]

        for r in 1:train_reps
            rng = MersenneTwister(seed + 10_000 * p.id + r)
            push!(vals, simulate_replication(p, z, rng))
        end

        μ = mean(vals)

        if μ > best_train_value
            best_train_value = μ
            best_policy = p
        end
    end

    # Stage 2: independent evaluation of selected policy
    eval_vals = Float64[]

    for r in 1:eval_reps
        rng = MersenneTwister(seed + 1_000_000 + 1000 * r)
        push!(eval_vals, simulate_replication(best_policy, z, rng))
    end

    μ_eval = mean(eval_vals)
    se_eval = std(eval_vals) / sqrt(eval_reps)

    return (
        value = μ_eval - dot(alpha, z),
        se = se_eval,
        policy_id = best_policy.id,
        train_value = best_train_value
    )
end

# ============================================================
# PROBING VECTORS
# ============================================================

function generate_all_binary_z(m)
    all_z = Vector{Vector{Int}}()
    for mask in 0:(2^m - 1)
        z = [ (mask >> (j - 1)) & 1 for j in 1:m ]
        push!(all_z, z)
    end
    return all_z
end

# ============================================================
# MAIN
# ============================================================

function run()
    m = 4
    alpha = zeros(Float64, m)

    policies = generate_policies()
    all_z = generate_all_binary_z(m)

    println("Policies: ", length(policies))
    println("Evaluating probing decisions (two-stage)...\n")

    results = []

    for z in all_z
        out = evaluate_Q_two_stage(z, policies; train_reps=100, eval_reps=1000, alpha=alpha)
        push!(results, (
            z = z,
            value = out.value,
            se = out.se,
            policy = out.policy_id
        ))
    end

    sort!(results, by = x -> x.value, rev = true)

    println("z-vector   value      se      policy")
    println("-----------------------------------------")

    for r in results
        zstr = join(r.z, "")
        println(
            rpad(zstr, 10),
            round(r.value, digits=2), "   ",
            round(r.se, digits=2), "   ",
            r.policy
        )
    end
end

run()