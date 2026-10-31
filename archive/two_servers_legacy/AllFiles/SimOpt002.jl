# SimOpt002.jl
# Introduces classes, but the policy does not use classes.
# No signal in decision. The decision is based on demand and queue difference only.

using Random
using Distributions
using Statistics

struct Policy # A data container for policies. The 3 parameters are weights for a simple linear score that determines whether to send to fast BE or not.
    id::Int
    a_demand::Float64
    a_queue::Float64
    threshold::Float64
end

function choose_fast_be(p::Policy, demand, fast_q, std_q) # simple linear policy based on demand and queue difference
    queue_diff = fast_q - std_q
    score = p.a_demand * demand - p.a_queue * queue_diff
    return score >= p.threshold
end

function generate_policies() # Generates a grid of policies by varying the parameters a_demand, a_queue, and threshold
    a_demand_grid = range(0.5, 3.0, length=10)
    a_queue_grid = range(0.0, 2.0, length=10)
    threshold_grid = range(-2.0, 2.0, length=5)

    policies = Policy[]
    id = 1

    for a in a_demand_grid, b in a_queue_grid, t in threshold_grid
        push!(policies, Policy(id, a, b, t))
        id += 1
    end

    return policies
end

function simulate_replication(policy::Policy, rng; # Simulates one replication of the system under a given policy and random seed. Returns total profit or detailed stats if collect_stats=true.
    N=200,
    mean_interarrival=5.0,
    revenue_scale=40.0,
    due_offset=20.0,
    collect_stats=false)

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
        interarrival = rand(rng, Exponential(mean_interarrival))
        arrival = current_time + interarrival
        current_time = arrival

        demand = rand(rng)
        lot_class = rand(rng, 1:4)
        complex_prob_by_class = [0.05, 0.15, 0.50, 0.85] # different classes have different probabilities of being complex, which adds extra time if sent to standard BE
        complex = rand(rng) < complex_prob_by_class[lot_class] # determine if the job is complex based on its class

        t_fe = rand(rng, Gamma(2.0, 1.5))
        fe_start = max(arrival, fe_available)
        fe_end = fe_start + t_fe
        fe_available = fe_end

        transfer = rand(rng, Exponential(1.0))
        transfer_end = fe_end + transfer

        fast_q = max(0.0, be_fast_available - transfer_end) # calculate the queue length for fast BE and standard BE at the time the job would arrive at BE, which is after transfer. The queue length is how much time until the BE is available, so if it's negative it means it's available and there is no queue.
        std_q = max(0.0, be_std_available - transfer_end) # the policy will use these queue lengths and the demand to decide whether to send to fast BE or standard BE. The decision is made by the choose_fast_be function, which calculates a score based on the demand and the difference in queue lengths, and compares it to a threshold.

        send_fast = choose_fast_be(policy, demand, fast_q, std_q) # decide whether to send to fast BE or standard BE based on the policy and the current demand and queue lengths

        if send_fast
            n_fast += 1
            t_be = rand(rng, Gamma(2.0, 1.4))
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
                extra = rand(rng, Exponential(4.0))
            else
                extra = 0.0
            end

            finish = start + t_be + extra
            be_std_available = finish
        end

        tardiness = max(0.0, finish - (arrival + due_offset)) # calculate tardiness based on the completion time (finish) and the due date (arrival + due_offset). If finish is less than due date, there is no tardiness, otherwise it's the difference.

        revenue = revenue_scale * (0.5 + demand) # calculate revenue based on the demand. The revenue is higher for higher demand, and the revenue_scale parameter controls how much revenue we get per unit of demand.
        cost = 1.0 * t_fe + 0.8 * transfer + 1.2 * (t_be + extra) + 2.5 * tardiness

        total_profit += revenue - cost
        total_tardiness += tardiness
    end

    if collect_stats
        return (
            profit = total_profit,
            tardiness = total_tardiness,
            n_fast = n_fast,
            n_std = n_std,
            complex_on_std = complex_on_std
        )
    else
        return total_profit
    end
end

function evaluate_policies(policies; reps=300) # Evaluates a list of policies by simulating multiple replications for each and collecting average profit and other stats.
    results = []

    for p in policies
        profits = Float64[]

        for r in 1:reps
            rng = MersenneTwister(1234 + r)
            push!(profits, simulate_replication(p, rng))
        end

        push!(results, (
            policy = p.id,
            mean_profit = mean(profits),
            std_profit = std(profits),
            a_demand = p.a_demand,
            a_queue = p.a_queue,
            threshold = p.threshold
        ))
    end

    sort!(results, by = x -> x.mean_profit, rev = true)
    return results
end

function inspect_policy(policy::Policy; reps=300) # Runs multiple replications for a single policy and prints detailed stats about profit, tardiness, and routing decisions.
    profits = Float64[]
    tardiness = Float64[]
    fast_counts = Float64[]
    std_counts = Float64[]
    complex_std = Float64[]

    for r in 1:reps
        rng = MersenneTwister(1234 + r)
        out = simulate_replication(policy, rng; collect_stats=true)

        push!(profits, out.profit)
        push!(tardiness, out.tardiness)
        push!(fast_counts, out.n_fast)
        push!(std_counts, out.n_std)
        push!(complex_std, out.complex_on_std)
    end

    println("\nPolicy ", policy.id)
    println("mean profit = ", mean(profits))
    println("mean tardiness = ", mean(tardiness))
    println("avg routed fast = ", mean(fast_counts))
    println("avg routed std = ", mean(std_counts))
    println("complex sent to std = ", mean(complex_std))
end

policies = generate_policies()

println("Policies: ", length(policies))

results = evaluate_policies(policies, reps=200)

println("\nTop 10 policies:\n")
for r in results[1:10]
    println(r)
end

best_policy = policies[results[1].policy]

inspect_policy(best_policy)
inspect_policy(policies[1])
inspect_policy(policies[250])
inspect_policy(policies[500])