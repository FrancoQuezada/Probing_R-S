"""Indexed CRN sampling.  Seeds use fixed SplitMix64-style folding, never `hash`.

Domains separate prior, diagnostics, posterior draws, counts, arrivals,
complexity, and the two potential service streams.  Keys contain only the
master seed and logical replication/path/line/job indices; they never contain z
or a candidate index.
"""
const CRN_PRIOR_THETA = UInt64(0x10a1)
const CRN_DIAGNOSTIC_NOISE = UInt64(0x10a2)
const CRN_TRAIN_POSTERIOR_THETA = UInt64(0x10b1)
const CRN_EVAL_POSTERIOR_THETA = UInt64(0x10b2)
const CRN_TRAIN_ARRIVAL_COUNT = UInt64(0x10c1)
const CRN_EVAL_ARRIVAL_COUNT = UInt64(0x10c2)
const CRN_TRAIN_ARRIVAL_TIME = UInt64(0x10d1)
const CRN_EVAL_ARRIVAL_TIME = UInt64(0x10d2)
const CRN_TRAIN_COMPLEXITY = UInt64(0x10e1)
const CRN_EVAL_COMPLEXITY = UInt64(0x10e2)
const CRN_TRAIN_FAST_SERVICE = UInt64(0x10f1)
const CRN_EVAL_FAST_SERVICE = UInt64(0x10f2)
const CRN_TRAIN_STANDARD_SERVICE = UInt64(0x1101)
const CRN_EVAL_STANDARD_SERVICE = UInt64(0x1102)

function _crn_seed(master_seed::Integer, domain::UInt64, indices::Vararg{Integer})::UInt64
    state = reinterpret(UInt64, Int64(_normalize_master_seed(master_seed))) ⊻ domain ⊻ 0x9e3779b97f4a7c15
    for index in indices
        state += reinterpret(UInt64, Int64(Int(index))) + 0x9e3779b97f4a7c15
        state = _mix_seed_word(state)
    end
    return state
end
# Xoshiro is a local, seeded RNG with the same required marginal primitives as
# the independent implementation; creating it per logical primitive avoids any
# mutable stream shared across candidates and is substantially lighter than a
# full MersenneTwister state.
_crn_rng(master_seed, domain, indices...) = Xoshiro(_crn_seed(master_seed, domain, indices...))
_crn_domains(phase::Symbol) = phase === :train ? (CRN_TRAIN_POSTERIOR_THETA, CRN_TRAIN_ARRIVAL_COUNT, CRN_TRAIN_ARRIVAL_TIME, CRN_TRAIN_COMPLEXITY, CRN_TRAIN_FAST_SERVICE, CRN_TRAIN_STANDARD_SERVICE) :
                              phase === :eval ? (CRN_EVAL_POSTERIOR_THETA, CRN_EVAL_ARRIVAL_COUNT, CRN_EVAL_ARRIVAL_TIME, CRN_EVAL_COMPLEXITY, CRN_EVAL_FAST_SERVICE, CRN_EVAL_STANDARD_SERVICE) : throw(ArgumentError("CRN phase must be :train or :eval"))

function _crn_prior_theta(master_seed, outer_rep::Int, params::ModelParameters)
    return [line.prior_mean + sqrt(line.prior_variance) * randn(_crn_rng(master_seed, CRN_PRIOR_THETA, outer_rep, c))
            for (c, line) in pairs(params.lines)]
end
function _crn_diagnostics(master_seed, outer_rep::Int, theta, z, params)
    values = Vector{Union{Missing,Float64}}(undef, nlines(params))
    for c in eachindex(params.lines)
        values[c] = z[c] == 0.0 ? missing : theta[c] + sqrt(diagnostic_variance(c, z[c], params)) * randn(_crn_rng(master_seed, CRN_DIAGNOSTIC_NOISE, outer_rep, c))
    end
    return DiagnosticObservation(values)
end

function generate_predictive_path_crn(master_seed::Integer, outer_rep::Integer, phase::Symbol,
                                      path_index::Integer, posterior::PosteriorState,
                                      params::ModelParameters)::ShiftPath
    _validate_posterior_dimensions(posterior, params)
    posterior_domain, count_domain, arrival_domain, complexity_domain, fast_domain, standard_domain = _crn_domains(phase)
    jobs = ReworkJob[]
    for c in eachindex(params.lines)
        line = params.lines[c]
        theta = posterior.means[c] + sqrt(posterior.variances[c]) * randn(_crn_rng(master_seed, posterior_domain, outer_rep, path_index, c))
        n_arrivals = _rand_poisson(_crn_rng(master_seed, count_domain, outer_rep, path_index, c), arrival_rate(theta, line) * params.horizon)
        for job_index in 1:n_arrivals
            arrival = rand(_crn_rng(master_seed, arrival_domain, outer_rep, path_index, c, job_index)) * params.horizon
            complex = rand(_crn_rng(master_seed, complexity_domain, outer_rep, path_index, c, job_index)) < complexity_probability(theta, line)
            service_index = complex ? 2 : 1
            fast = line.fast_service_means[service_index] * randexp(_crn_rng(master_seed, fast_domain, outer_rep, path_index, c, job_index))
            standard = line.standard_service_means[service_index] * randexp(_crn_rng(master_seed, standard_domain, outer_rep, path_index, c, job_index))
            push!(jobs, ReworkJob(arrival, c, complex, fast, standard))
        end
    end
    sort!(jobs; by=job -> (job.arrival, job.line), alg=MergeSort)
    return ShiftPath(jobs)
end

function _crn_bank(master_seed, outer_rep, phase, posterior, params, count)
    return [generate_predictive_path_crn(master_seed, outer_rep, phase, path, posterior, params) for path in 1:count]
end
function select_and_evaluate_policy_crn(master_seed, outer_rep, posterior, params, policies, ntrain, neval)
    train_paths = _crn_bank(master_seed, outer_rep, :train, posterior, params, ntrain)
    rewards = precompute_rewards(policies, train_paths, params)
    values = [sum(@view rewards[i, :]) / ntrain for i in axes(rewards, 1)]
    selected = _first_argmax(values)
    eval_paths = _crn_bank(master_seed, outer_rep, :eval, posterior, params, neval)
    evaluation_mean, evaluation_standard_error = _mean_and_se([evaluate_policy(policies[selected], path, params) for path in eval_paths])
    return PolicySelectionResult(selected, policies[selected], values, values[selected], evaluation_mean, evaluation_standard_error, ntrain, neval)
end

"""Evaluate a fixed z with CRN-indexed base streams aligned across candidates."""
function evaluate_z_crn(master_seed::Integer, z::AbstractVector{<:Real}, params::ModelParameters,
                        policies::AbstractVector{<:Policy}; outer_reps::Integer, inner_train::Integer, inner_eval::Integer)::ZEvaluationResult
    z_values = validate_z(z, params); outer = _positive_sample_size(outer_reps, "outer_reps"); train = _positive_sample_size(inner_train, "inner_train"); evaluation = _positive_sample_size(inner_eval, "inner_eval")
    isempty(policies) && throw(ArgumentError("at least one candidate policy is required"))
    values = Float64[]; indices = Int[]; training = Float64[]; errors = Float64[]; counts = zeros(Int, length(policies))
    for replication in 1:outer
        theta = _crn_prior_theta(master_seed, replication, params)
        posterior = posterior_parameters(_crn_diagnostics(master_seed, replication, theta, z_values, params), z_values, params)
        selection = select_and_evaluate_policy_crn(master_seed, replication, posterior, params, policies, train, evaluation)
        push!(values, something(selection.evaluation_mean)); push!(indices, selection.selected_index); push!(training, selection.selected_training_value); push!(errors, something(selection.evaluation_standard_error)); counts[selection.selected_index] += 1
    end
    gross, se = _mean_and_se(values); cost = _probing_cost(z_values, params)
    return ZEvaluationResult(Tuple(z_values), gross, cost, gross - cost, se, Tuple(values), Tuple(counts ./ outer), nothing, Tuple(indices), Tuple(counts), outer, train, evaluation, Tuple(training), Tuple(errors))
end
