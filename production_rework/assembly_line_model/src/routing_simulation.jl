"""Compute the fixed-policy routing score for one already-realized rework job."""
function routing_score(policy::Policy, job::ReworkJob, q_fast::Real,
                       q_standard::Real, params::ModelParameters)::Float64
    qf = Float64(q_fast)
    qs = Float64(q_standard)
    isfinite(qf) && qf >= 0.0 ||
        throw(ArgumentError("FAST workload must be finite and nonnegative"))
    isfinite(qs) && qs >= 0.0 ||
        throw(ArgumentError("STANDARD workload must be finite and nonnegative"))

    score = policy.av * line_value(params, job.line) - policy.aq * (qf - qs)
    isfinite(score) ||
        throw(ArgumentError("routing score is not finite for the supplied workloads"))
    return score
end

function _job_reward(value::Float64, routed_fast::Bool, time_in_system::Float64,
                     params::ModelParameters)::Float64
    isfinite(time_in_system) && time_in_system >= 0.0 ||
        throw(ArgumentError("time in system must be finite and nonnegative"))
    reward = params.revenue_scale * value -
             (routed_fast ? params.fast_routing_cost : 0.0) -
             params.delay_penalty * time_in_system
    isfinite(reward) || throw(ArgumentError("job reward is not finite"))
    return reward
end

function _simulate_policy(policy::Policy, path::ShiftPath, params::ModelParameters;
                          collect_trace::Bool=false)
    free_fast = 0.0
    free_standard = 0.0
    total_reward = 0.0
    trace = NamedTuple[]

    for job in path.jobs
        arrival = job.arrival
        q_fast = max(free_fast - arrival, 0.0)
        q_standard = max(free_standard - arrival, 0.0)
        score = routing_score(policy, job, q_fast, q_standard, params)
        routed_fast = score >= policy.tau

        if routed_fast
            start = max(arrival, free_fast)
            finish = start + job.service_fast
            free_fast = finish
        else
            start = max(arrival, free_standard)
            finish = start + job.service_standard
            free_standard = finish
        end

        time_in_system = finish - arrival
        reward = _job_reward(line_value(params, job.line), routed_fast,
                             time_in_system, params)
        total_reward += reward
        isfinite(total_reward) || throw(ArgumentError("total shift reward is not finite"))

        if collect_trace
            push!(trace, (; arrival, line=job.line, value=line_value(params, job.line),
                          q_fast, q_standard, score, routed_fast, start, finish,
                          time_in_system, reward, cumulative_reward=total_reward))
        end
    end

    return collect_trace ? (total_reward, trace) : total_reward
end

"""Evaluate the total operational reward of one fixed policy on one shift path."""
function evaluate_policy(policy::Policy, path::ShiftPath,
                         params::ModelParameters)::Float64
    return _simulate_policy(policy, path, params)
end

"""Internal deterministic trace used only for focused routing tests."""
function _simulate_policy_trace(policy::Policy, path::ShiftPath,
                                params::ModelParameters)
    _, trace = _simulate_policy(policy, path, params; collect_trace=true)
    return trace
end

"""Evaluate each policy in order on the same realized shift path."""
function evaluate_policies_on_path(policies::AbstractVector{<:Policy}, path::ShiftPath,
                                   params::ModelParameters)::Vector{Float64}
    rewards = Vector{Float64}(undef, length(policies))
    for i in eachindex(policies)
        rewards[i] = evaluate_policy(policies[i], path, params)
    end
    return rewards
end

"""Return Y[i, ell], the reward of policy i on supplied path ell."""
function precompute_rewards(policies::AbstractVector{<:Policy},
                            paths::AbstractVector{<:ShiftPath},
                            params::ModelParameters)::Matrix{Float64}
    rewards = Matrix{Float64}(undef, length(policies), length(paths))
    for i in eachindex(policies)
        for ell in eachindex(paths)
            rewards[i, ell] = evaluate_policy(policies[i], paths[ell], params)
        end
    end
    return rewards
end
