# SimOpt004_clean.jl
# Probing model WITHOUT latent inference or posterior.
# Probing affects signal generation directly.

using Random
using Distributions
using Statistics
using LinearAlgebra

# ============================================================
# POLICY
# ============================================================

struct Policy
    id::Int
    a_demand::Float64 # weight on demand
    a_queue::Float64 # weight on queue difference
    a_signal::Float64 # weight on signal (0 if no signal, 1 if signal = 1, and baseline prob if signal is missing)
    threshold::Float64 # threshold for sending to fast BE
end

function choose_fast_be(p::Policy, demand, fast_q, std_q, signal)
    queue_diff = fast_q - std_q
    score = p.a_demand * demand - p.a_queue * queue_diff + p.a_signal * signal # signal is 0 if no signal, 1 if signal = 1, and baseline prob if signal is missing
    return score >= p.threshold
end

# ============================================================
# POLICY GRID
# ============================================================

function generate_policies()
    a_demand_grid = range(0.5, 3.0, length=6)
    a_queue_grid = range(0.0, 1.2, length=6)
    a_signal_grid = range(0.0, 2.0, length=5)
    threshold_grid = [-1.0, 0.0, 1.0]

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
# SIGNAL GENERATION (KEY PART)
# ============================================================

function generate_signal(rng, true_class, complex, z;
    p_hit   = [0.3, 0.4, 0.7, 0.9],
    p_false = [0.05, 0.08, 0.15, 0.2],
    beta = 3.0)

    if rand(rng) > z[true_class]
        return nothing
    end

    # Improve accuracy with z
    ph = p_hit[true_class] + (1 - p_hit[true_class])*(1 - exp(-beta*z[true_class]))
    pf = p_false[true_class] * exp(-beta*z[true_class])

    if complex
        return rand(rng) < ph ? 1.0 : 0.0
    else
        return rand(rng) < pf ? 1.0 : 0.0
    end
end

# ============================================================
# SIMULATION
# ============================================================

function simulate_replication(policy::Policy, z, rng;
    N=200,
    mean_interarrival=5.0,
    revenue_scale=40.0,
    due_offset=20.0,
    complex_prob_by_class=[0.01, 0.10, 0.60, 0.95])

    m = length(z)

    fe_available = 0.0
    be_fast_available = 0.0
    be_std_available = 0.0

    current_time = 0.0
    total_profit = 0.0
    total_tardiness = 0.0

    n_fast = 0
    n_std = 0
    complex_on_std = 0

    for n in 1:N

        # Arrival
        interarrival = rand(rng, Exponential(mean_interarrival))
        arrival = current_time + interarrival
        current_time = arrival

        demand = rand(rng)

        # Observable class
        true_class = rand(rng, 1:m)

        # Complexity affects BE processing (never inferred)
        complex = rand(rng) < complex_prob_by_class[true_class]

        # Signal (depends on z)
        signal = generate_signal(rng, true_class, complex, z)

        baseline = complex_prob_by_class[true_class]
        signal_used = isnothing(signal) ? baseline : signal

        # FE
        t_fe = rand(rng, Gamma(2.0, 0.9))
        fe_start = max(arrival, fe_available)
        fe_end = fe_start + t_fe
        fe_available = fe_end

        # Transfer
        transfer = rand(rng, Exponential(1.0))
        transfer_end = fe_end + transfer

        # Queues
        fast_q = max(0.0, be_fast_available - transfer_end)
        std_q = max(0.0, be_std_available - transfer_end)

        # Decision
        send_fast = choose_fast_be(policy, demand, fast_q, std_q, signal_used)

        if send_fast
            n_fast += 1
            t_be = rand(rng, Gamma(2.0, 1.0))
            start = max(transfer_end, be_fast_available)
            extra = 0.0
            finish = start + t_be
            be_fast_available = finish
        else
            n_std += 1
            t_be = rand(rng, Gamma(2.0, 1.5))
            start = max(transfer_end, be_std_available)

            if complex
                complex_on_std += 1
                scale = 8.0 * (1 - 0.8 * z[true_class])   # probing reduces disruption
                extra = rand(rng, Exponential(scale))   
            else
                extra = 0.0
            end

            finish = start + t_be + extra
            be_std_available = finish
        end

        tardiness = max(0.0, finish - (arrival + due_offset))

        revenue = revenue_scale * (0.5 + demand)
        cost = 1.0 * t_fe + 0.8 * transfer + 1.2 * (t_be + extra) + 2.5 * tardiness

        total_profit += revenue - cost
        total_tardiness += tardiness
    end

    return (
        profit = total_profit,
        tardiness = total_tardiness,
        n_fast = n_fast,
        n_std = n_std,
        complex_on_std = complex_on_std
    )
end

# ============================================================
# EVALUATE Q(z)
# ============================================================

function evaluate_Q(z, policies;
    reps=100,
    seed=1234,
    alpha=zeros(Float64, length(z)))

    policy_vals = Float64[]

    for p in policies
        vals = Float64[]

        for r in 1:reps
            rng = MersenneTwister(seed + 1000*r)
            out = simulate_replication(p, z, rng)
            push!(vals, out.profit)
        end

        push!(policy_vals, mean(vals))
    end

    return maximum(policy_vals) - dot(alpha, z)
end

# ============================================================
# MAIN
# ============================================================

function run()

    m = 4
    policies = generate_policies()

    # Try all probing vectors (binary case)
    all_z = []
    for mask in 0:(2^m - 1)
        z = [ (mask >> (j-1)) & 1 for j in 1:m ]
        push!(all_z, z)
    end

    println("Policies: ", length(policies))
    println("Evaluating probing decisions...\n")

    results = []

    for z in all_z
        val = evaluate_Q(z, policies)
        push!(results, (z=z, value=val))
    end

    sort!(results, by = x -> x.value, rev = true)

    println("z-vector   value")
    println("-----------------------")

    for r in results
        zstr = join(r.z, "")
        println(rpad(zstr, 10), round(r.value, digits=2))
    end
end

run()