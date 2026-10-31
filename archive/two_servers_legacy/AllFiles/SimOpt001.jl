# SimOpt001.jl - A simple simulation optimization example for a two-stage service system with a fast and a standard back-end. 
# The policy is a simple linear score based on demand and queue difference, compared to a threshold. 
# We evaluate 500 policies generated from a grid of parameters, and inspect the best one in more detail.
# There are no classes, no signal and decision isies demand + queue difference only.

using Random
using Distributions
using Statistics

# A data container for policies. The 3 parameters are weights for a simple linear score that determines whether to send to fast BE or not. 
#The score is a weighted combination of demand and queue difference, compared to a threshold.
struct Policy
    id::Int
    a_demand::Float64 # sensitivity to demand. It is essentially a priority signal for lots that bring more revenue.
    a_queue::Float64 # sensitivity to queue difference
    threshold::Float64 # threshold for choosing fast BE
end

function choose_fast_be(p::Policy, demand, fast_q, std_q) # simple linear policy based on demand and queue difference
    queue_diff = fast_q - std_q
    score = p.a_demand * demand - p.a_queue * queue_diff
    return score >= p.threshold # if score is above threshold, send to fast BE/ otherwise send to standard BE. The function returns true if sent to fast BE, false if sent to standard BE
end

function generate_policies()
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

function simulate_replication(policy::Policy, rng;
    N=200, # number of arrivals in a replication
    mean_interarrival=5.0, # mean interarrival time
    complex_prob=0.25, # probability that a job is complex, which adds extra time if sent to standard BE
    revenue_scale=40.0, # scale for revenue calculation
    due_offset=20.0) # offset for due date calculation

    fe_available = 0.0 # time when FE is available
    be_fast_available = 0.0 # time when fast BE is available
    be_std_available = 0.0 # time when standard BE is available

    current_time = 0.0 # current simulation time
    total_profit = 0.0 # total profit accumulated in the replication

    for n in 1:N

        interarrival = rand(rng, Exponential(mean_interarrival))
        arrival = current_time + interarrival
        current_time = arrival

        demand = rand(rng) # random demand between 0 and 1, determines how valuable the job is
        complex = rand(rng) < complex_prob
 
        t_fe = rand(rng, Gamma(2.0,1.5)) # time to process at FE
        fe_start = max(arrival, fe_available) # FE can only start after the job arrives and after it is available
        fe_end = fe_start + t_fe
        fe_available = fe_end

        transfer = rand(rng, Exponential(1.0)) # time to transfer from FE to BE, can start after FE is done
        transfer_end = fe_end + transfer

        fast_q = max(0.0, be_fast_available - transfer_end) 
        std_q = max(0.0, be_std_available - transfer_end)

        send_fast = choose_fast_be(policy, demand, fast_q, std_q)

        if send_fast
            t_be = rand(rng, Gamma(2.0,1.4)) # time to process at fast BE
            start = max(transfer_end, be_fast_available)
            extra = 0.0
            finish = start + t_be + extra
            be_fast_available = finish
        else
            t_be = rand(rng, Gamma(2.0,1.5))
            start = max(transfer_end, be_std_available)
            extra = complex ? rand(rng, Exponential(4.0)) : 0.0
            finish = start + t_be + extra
            be_std_available = finish
        end

        completion = finish
        due = arrival + due_offset
        tardiness = max(0.0, completion - due)

        revenue = revenue_scale * (0.5 + demand)

        cost = 1.0*t_fe + 0.8*transfer + 1.2*(t_be+extra) + 2.5*tardiness

        total_profit += revenue - cost
    end

    return total_profit
end

function evaluate_policies(policies; reps=50)

    results = []

    for p in policies

        profits = Float64[]

        for r in 1:reps
            rng = MersenneTwister(1234+r)
            push!(profits, simulate_replication(p,rng))
        end

        push!(results, (
            policy=p.id,
            mean_profit=mean(profits),
            std_profit=std(profits),
            a_demand=p.a_demand,
            a_queue=p.a_queue,
            threshold=p.threshold
        ))
    end

    sort!(results, by=x->x.mean_profit, rev=true)

    return results
end

policies = generate_policies()

println("Policies: ", length(policies))

results = evaluate_policies(policies, reps=100)

println("\nTop 10 policies:\n")

for r in results[1:10]
    println(r)
end

function simulate_replication(policy::Policy, rng;
    N=200,
    mean_interarrival=5.0,
    complex_prob=0.25,
    revenue_scale=40.0,
    due_offset=25.0)

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
        complex = rand(rng) < complex_prob

        t_fe = rand(rng, Gamma(2.0,1.5))
        fe_start = max(arrival, fe_available)
        fe_end = fe_start + t_fe
        fe_available = fe_end

        transfer = rand(rng, Exponential(1.0))
        transfer_end = fe_end + transfer

        fast_q = max(0.0, be_fast_available - transfer_end)
        std_q = max(0.0, be_std_available - transfer_end)

        send_fast = choose_fast_be(policy, demand, fast_q, std_q)

        if send_fast
            n_fast += 1
            t_be = rand(rng, Gamma(2.0,1.4))
            start = max(transfer_end, be_fast_available)
            extra = 0.0
            finish = start + t_be + extra
            be_fast_available = finish
        else
            n_std += 1
            t_be = rand(rng, Gamma(2.0,1.5))
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

        completion = finish
        due = arrival + due_offset
        tardiness = max(0.0, completion - due)

        revenue = revenue_scale * (0.5 + demand)

        cost = 1.0*t_fe + 0.8*transfer + 1.2*(t_be+extra) + 2.5*tardiness

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

function inspect_policy(policy::Policy; reps=200)

    profits = Float64[]
    tardiness = Float64[]
    fast_counts = Float64[]
    std_counts = Float64[]
    complex_std = Float64[]

    for r in 1:reps
        rng = MersenneTwister(999 + r)

        out = simulate_replication(policy, rng)

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

best_policy = policies[results[1].policy]

inspect_policy(best_policy)
inspect_policy(policies[1])
inspect_policy(policies[250])
inspect_policy(policies[500])