function _positive_sample_size(n::Integer, name::AbstractString)::Int
    n > 0 || throw(ArgumentError("$name must be positive"))
    return Int(n)
end

function _first_argmax(values::AbstractVector{<:Real})::Int
    isempty(values) && throw(ArgumentError("cannot select a policy from an empty value vector"))
    best_index = firstindex(values)
    best_value = values[best_index]
    for index in (best_index + 1):lastindex(values)
        if values[index] > best_value
            best_index = index
            best_value = values[index]
        end
    end
    return best_index
end

function _mean_and_se(rewards::AbstractVector{<:Real})
    count = length(rewards)
    count > 0 || throw(ArgumentError("at least one evaluation reward is required"))
    mean_reward = sum(rewards) / count
    isfinite(mean_reward) || throw(ArgumentError("evaluation mean must be finite"))
    if count == 1
        return Float64(mean_reward), 0.0
    end
    sum_squared_deviations = sum((reward - mean_reward)^2 for reward in rewards)
    standard_error = sqrt(sum_squared_deviations / (count - 1)) / sqrt(count)
    isfinite(standard_error) && standard_error >= 0.0 ||
        throw(ArgumentError("evaluation standard error must be finite and nonnegative"))
    return Float64(mean_reward), Float64(standard_error)
end

"""Estimate all candidate policies on one common posterior-predictive training bank."""
function estimate_policy_values(rng::AbstractRNG, posterior::PosteriorState,
                                params::ModelParameters,
                                policies::AbstractVector{<:Policy},
                                ntrain::Integer)::Vector{Float64}
    count = _positive_sample_size(ntrain, "ntrain")
    isempty(policies) && return Float64[]
    training_paths = generate_predictive_bank(rng, posterior, params, count)
    rewards = precompute_rewards(policies, training_paths, params)
    return [sum(@view rewards[i, :]) / count for i in axes(rewards, 1)]
end

"""Select the first maximum-reward policy from a common posterior-predictive bank."""
function select_policy_given_posterior(rng::AbstractRNG, posterior::PosteriorState,
                                       params::ModelParameters,
                                       policies::AbstractVector{<:Policy},
                                       ntrain::Integer)::PolicySelectionResult
    isempty(policies) && throw(ArgumentError("at least one candidate policy is required"))
    count = _positive_sample_size(ntrain, "ntrain")
    training_values = estimate_policy_values(rng, posterior, params, policies, count)
    selected_index = _first_argmax(training_values)
    return PolicySelectionResult(selected_index, policies[selected_index], Tuple(training_values),
                                 training_values[selected_index], nothing, nothing,
                                 count, nothing)
end

"""Independently evaluate one fixed policy under a supplied posterior distribution."""
function evaluate_selected_policy(rng::AbstractRNG, posterior::PosteriorState,
                                  params::ModelParameters, policy::Policy,
                                  neval::Integer)
    count = _positive_sample_size(neval, "neval")
    evaluation_paths = generate_predictive_bank(rng, posterior, params, count)
    rewards = [evaluate_policy(policy, path, params) for path in evaluation_paths]
    evaluation_mean, evaluation_standard_error = _mean_and_se(rewards)
    return (; evaluation_mean, evaluation_standard_error, neval=count)
end

"""Select on a common training bank, then independently evaluate that fixed policy."""
function select_and_evaluate_policy(rng::AbstractRNG, posterior::PosteriorState,
                                    params::ModelParameters,
                                    policies::AbstractVector{<:Policy},
                                    ntrain::Integer, neval::Integer)::PolicySelectionResult
    isempty(policies) && throw(ArgumentError("at least one candidate policy is required"))
    train_count = _positive_sample_size(ntrain, "ntrain")
    evaluation_count = _positive_sample_size(neval, "neval")
    training_values = estimate_policy_values(rng, posterior, params, policies, train_count)
    selected_index = _first_argmax(training_values)
    evaluation = evaluate_selected_policy(rng, posterior, params,
                                          policies[selected_index], evaluation_count)
    return PolicySelectionResult(selected_index, policies[selected_index], training_values,
                                 training_values[selected_index], evaluation.evaluation_mean,
                                 evaluation.evaluation_standard_error, train_count,
                                 evaluation.neval)
end
