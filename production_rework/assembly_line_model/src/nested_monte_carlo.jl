function _probing_cost(z::AbstractVector{<:Real}, params::ModelParameters)::Float64
    return sum(params.probe_cost[c] * z[c] for c in eachindex(params.probe_cost))
end

function _evaluate_outer_replication(rng::AbstractRNG, z::AbstractVector{<:Real},
                                     params::ModelParameters,
                                     policies::AbstractVector{<:Policy},
                                     inner_train::Int, inner_eval::Int)
    theta_true = sample_prior_theta(rng, params)
    diagnostics = sample_diagnostics(rng, theta_true, z, params)
    posterior = posterior_parameters(diagnostics, z, params)
    return select_and_evaluate_policy(rng, posterior, params, policies,
                                      inner_train, inner_eval)
end

"""Estimate the nested objective for one fixed probing decision `z`."""
function evaluate_z(rng::AbstractRNG, z::AbstractVector{<:Real},
                    params::ModelParameters, policies::AbstractVector{<:Policy};
                    outer_reps::Integer, inner_train::Integer,
                    inner_eval::Integer)::ZEvaluationResult
    z_values = validate_z(z, params)
    outer_count = _positive_sample_size(outer_reps, "outer_reps")
    training_count = _positive_sample_size(inner_train, "inner_train")
    evaluation_count = _positive_sample_size(inner_eval, "inner_eval")
    isempty(policies) && throw(ArgumentError("at least one candidate policy is required"))

    outer_values = Vector{Float64}(undef, outer_count)
    selected_indices = Vector{Int}(undef, outer_count)
    selected_training_values = Vector{Float64}(undef, outer_count)
    inner_standard_errors = Vector{Float64}(undef, outer_count)
    selection_counts = zeros(Int, length(policies))

    for replication in eachindex(outer_values)
        selection = _evaluate_outer_replication(rng, z_values, params, policies,
                                                training_count, evaluation_count)
        outer_values[replication] = something(selection.evaluation_mean)
        selected_indices[replication] = selection.selected_index
        selected_training_values[replication] = selection.selected_training_value
        inner_standard_errors[replication] = something(selection.evaluation_standard_error)
        selection_counts[selection.selected_index] += 1
    end

    gross_value, outer_standard_error = _mean_and_se(outer_values)
    cost = _probing_cost(z_values, params)
    objective = gross_value - cost
    frequencies = selection_counts ./ outer_count

    return ZEvaluationResult(Tuple(z_values), gross_value, cost, objective,
                             outer_standard_error, Tuple(outer_values), Tuple(frequencies),
                             nothing, Tuple(selected_indices), Tuple(selection_counts),
                             outer_count, training_count, evaluation_count,
                             Tuple(selected_training_values), Tuple(inner_standard_errors))
end
