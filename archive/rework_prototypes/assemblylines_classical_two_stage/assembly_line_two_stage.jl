module AssemblyLineTwoStage

using Random
using Statistics
using LinearAlgebra
using Distributions

export AssemblyLineParams, ExperimentParams, Policy, BaseNoisePath, NoiseBanks,
       ConditionalSelectionResult, TwoStageEstimate, FixedPolicyEstimate,
       build_policy_grid, validate_params, diagnostic_time, measurement_marginal_variance,
       sample_measurement, posterior_line_condition, sample_theta_posterior,
       conditional_complexity_probability, posterior_expected_complexity,
       generate_noise_banks, outer_measurement, conditional_operating_path,
       unconditional_operating_path, evaluate_policy, conditional_selection,
       estimate_two_stage, estimate_fixed_policy_baseline

"""Calibration shared with the original two-server routing experiments."""
Base.@kwdef struct AssemblyLineParams
    nclass::Int = 4
    p_bar::Vector{Float64} = [0.02, 0.12, 0.55, 0.95]
    p_hit::Vector{Float64} = [0.25, 0.40, 0.75, 0.92]
    p_false::Vector{Float64} = [0.05, 0.08, 0.15, 0.20]
    tau_theta::Vector{Float64} = ones(4)
    sigma_sensor::Vector{Float64} = fill(2.0, 4)
    condition_sensitivity::Vector{Float64} = [0.35, 0.55, 0.80, 0.60]
    t_max::Vector{Float64} = fill(20.0, 4)
    mean_interarrival::Float64 = 0.80
    fast_mean_simple::Float64 = 0.75
    fast_mean_complex::Float64 = 2.40
    std_mean_simple::Float64 = 1.00
    std_mean_complex::Float64 = 5.80
    revenue_scale::Float64 = 8.0
    fast_routing_cost::Float64 = 0.70
    delay_penalty::Float64 = 0.42
    disruption_probability::Float64 = 0.0
    disruption_delay_mean::Float64 = 0.0
end

"""Sampling sizes, costs, and independent random seeds for one nested experiment."""
Base.@kwdef struct ExperimentParams
    outer_reps::Int = 100
    inner_train::Int = 300
    inner_eval::Int = 500
    jobs_per_path::Int = 45
    alpha::Vector{Float64} = zeros(4)
    outer_seed::Int = 20260724
    train_seed::Int = 20261724
    eval_seed::Int = 20262724
    baseline_train_seed::Int = 20263724
    baseline_eval_seed::Int = 20264724
end

"""One complete routing policy, fixed for the entirety of an operating shift."""
struct Policy
    a_value::Float64
    a_queue::Float64
    a_signal::Float64
    threshold::Float64
end

"""Uniforms and normals sufficient to reparameterize one full operating path."""
struct BaseNoisePath
    theta_normals::Vector{Float64}
    arrivals::Vector{Float64}
    classes::Vector{Float64}
    values::Vector{Float64}
    complexities::Vector{Float64}
    signals::Vector{Float64}
    fast_services::Vector{Float64}
    std_services::Vector{Float64}
    disruptions::Vector{Float64}
    disruption_delays::Vector{Float64}
end

struct NoiseBanks
    outer_normals::Matrix{Float64}
    train::Vector{Vector{BaseNoisePath}}
    eval::Vector{Vector{BaseNoisePath}}
    fixed_train::Vector{BaseNoisePath}
    fixed_eval::Vector{BaseNoisePath}
end

struct AssemblyJob
    arrival::Float64
    class::Int
    value::Float64
    complex::Bool
    signal::Int
    fast_service::Float64
    std_service::Float64
end

struct OperatingPath
    theta::Vector{Float64}
    jobs::Vector{AssemblyJob}
end

struct ConditionalSelectionResult
    outer_id::Int
    eta::Vector{Float64}
    posterior_mean::Vector{Float64}
    posterior_variance::Vector{Float64}
    posterior_complexity::Vector{Float64}
    selected_policy_id::Int
    best_training_mean::Float64
    second_training_mean::Float64
    training_gap::Float64
    evaluation_mean::Float64
    evaluation_se::Float64
    fast_probability::Float64
    evaluation_policy_ids::Vector{Int}
end

struct FixedPolicyEstimate
    selected_policy_id::Int
    training_mean::Float64
    evaluation_mean::Float64
    evaluation_se::Float64
end

struct TwoStageEstimate
    z::Vector{Float64}
    diagnostic_times::Vector{Float64}
    gross_value::Float64
    probing_cost::Float64
    objective::Float64
    standard_error::Float64
    optimistic_training_value::Float64
    fixed_policy::FixedPolicyEstimate
    information_value::Float64
    mean_posterior_variance::Vector{Float64}
    mean_posterior_complexity::Vector{Float64}
    distinct_selected_policies::Int
    most_frequent_policy_id::Int
    most_frequent_policy_frequency::Float64
    average_training_gap::Float64
    average_fast_probability::Float64
    contexts::Vector{ConditionalSelectionResult}
end

_unit(x::Real) = isfinite(x) && 0.0 <= x <= 1.0
_uniform(x::Real) = isfinite(x) && 0.0 <= x < 1.0
_normal() = Normal()

function validate_params(params::AssemblyLineParams, experiment::Union{Nothing,ExperimentParams}=nothing)
    params.nclass > 0 || throw(ArgumentError("nclass must be positive"))
    for (name, values) in (("p_bar", params.p_bar), ("p_hit", params.p_hit), ("p_false", params.p_false),
                           ("tau_theta", params.tau_theta), ("sigma_sensor", params.sigma_sensor),
                           ("condition_sensitivity", params.condition_sensitivity), ("t_max", params.t_max))
        length(values) == params.nclass || throw(ArgumentError("$name must have length $(params.nclass)"))
        all(isfinite, values) || throw(ArgumentError("$name entries must be finite"))
    end
    all(_unit, params.p_bar) && all(_unit, params.p_hit) && all(_unit, params.p_false) ||
        throw(ArgumentError("probability vectors must lie in [0,1]"))
    all(>(0.0), params.tau_theta) && all(>(0.0), params.sigma_sensor) && all(>(0.0), params.t_max) ||
        throw(ArgumentError("prior scales, sensor scales, and diagnostic maxima must be positive"))
    all(>=(0.0), params.condition_sensitivity) || throw(ArgumentError("condition sensitivities must be nonnegative"))
    for (name, value) in (("mean_interarrival", params.mean_interarrival),
                          ("fast_mean_simple", params.fast_mean_simple), ("fast_mean_complex", params.fast_mean_complex),
                          ("std_mean_simple", params.std_mean_simple), ("std_mean_complex", params.std_mean_complex))
        isfinite(value) && value > 0 || throw(ArgumentError("$name must be positive"))
    end
    _unit(params.disruption_probability) || throw(ArgumentError("disruption probability must be in [0,1]"))
    isfinite(params.disruption_delay_mean) && params.disruption_delay_mean >= 0 ||
        throw(ArgumentError("disruption delay mean must be nonnegative"))
    if experiment !== nothing
        for (name, value) in (("outer_reps", experiment.outer_reps), ("inner_train", experiment.inner_train),
                              ("inner_eval", experiment.inner_eval), ("jobs_per_path", experiment.jobs_per_path))
            value > 0 || throw(ArgumentError("$name must be positive"))
        end
        length(experiment.alpha) == params.nclass || throw(ArgumentError("alpha must have length $(params.nclass)"))
        all(isfinite, experiment.alpha) || throw(ArgumentError("alpha entries must be finite"))
    end
    return nothing
end

function _validate_z(z::AbstractVector{<:Real}, params::AssemblyLineParams)
    length(z) == params.nclass || throw(ArgumentError("z must have length $(params.nclass)"))
    all(_unit, z) || throw(ArgumentError("all diagnostic efforts must be in [0,1]"))
end

function _check_class(c::Int, params::AssemblyLineParams)
    1 <= c <= params.nclass || throw(ArgumentError("class must be in 1:$(params.nclass)"))
end

diagnostic_time(zc::Real, c::Int, params::AssemblyLineParams) = begin
    validate_params(params); _check_class(c, params); _unit(zc) || throw(ArgumentError("zc must be in [0,1]"))
    params.t_max[c] * zc
end

function measurement_marginal_variance(zc::Real, c::Int, params::AssemblyLineParams)
    t = diagnostic_time(zc, c, params)
    t == 0.0 && return 0.0
    return t^2 * params.tau_theta[c]^2 + params.sigma_sensor[c]^2 * t
end

function sample_measurement(theta_c::Real, zc::Real, base_normal::Real, c::Int, params::AssemblyLineParams)
    t = diagnostic_time(zc, c, params)
    return t == 0.0 ? 0.0 : t * theta_c + params.sigma_sensor[c] * sqrt(t) * base_normal
end

function posterior_line_condition(eta_c::Real, zc::Real, c::Int, params::AssemblyLineParams)
    t = diagnostic_time(zc, c, params)
    t == 0.0 && return (mean=0.0, variance=params.tau_theta[c]^2)
    variance = 1.0 / (1.0 / params.tau_theta[c]^2 + t / params.sigma_sensor[c]^2)
    return (mean=variance * eta_c / params.sigma_sensor[c]^2, variance=variance)
end

function sample_theta_posterior(eta::AbstractVector{<:Real}, z::AbstractVector{<:Real},
                                base_normals::AbstractVector{<:Real}, params::AssemblyLineParams)
    _validate_z(z, params); length(eta) == params.nclass == length(base_normals) ||
        throw(ArgumentError("eta and base normals must have nclass entries"))
    return [begin post = posterior_line_condition(eta[c], z[c], c, params); post.mean + sqrt(post.variance) * base_normals[c] end
            for c in 1:params.nclass]
end

function _intercept(c::Int, params::AssemblyLineParams)
    _check_class(c, params)
    return quantile(_normal(), params.p_bar[c]) *
           sqrt(1.0 + params.condition_sensitivity[c]^2 * params.tau_theta[c]^2)
end

function conditional_complexity_probability(c::Int, theta_c::Real, params::AssemblyLineParams)
    validate_params(params); _check_class(c, params)
    p = cdf(_normal(), _intercept(c, params) + params.condition_sensitivity[c] * theta_c)
    _unit(p) || throw(ArgumentError("computed complexity probability is invalid"))
    return p
end

function posterior_expected_complexity(eta_c::Real, zc::Real, c::Int, params::AssemblyLineParams)
    post = posterior_line_condition(eta_c, zc, c, params)
    denominator = sqrt(1.0 + params.condition_sensitivity[c]^2 * post.variance)
    return cdf(_normal(), (_intercept(c, params) + params.condition_sensitivity[c] * post.mean) / denominator)
end

function build_policy_grid()
    [Policy(av, aq, as, tau) for av in (0.4, 0.8, 1.2), aq in (0.15, 0.35, 0.65),
     as in (0.0, 0.6, 1.2, 2.0), tau in (0.10, 0.45, 0.80, 1.15)] |> vec
end

_exp_from_uniform(u::Real, mean::Real) = begin
    _uniform(u) || throw(ArgumentError("uniform primitive must be in [0,1)"))
    -mean * log1p(-u)
end

function _bernoulli(u::Real, p::Real)
    _uniform(u) || throw(ArgumentError("Bernoulli primitive must be in [0,1)"))
    isfinite(p) || throw(ArgumentError("Bernoulli probability must be finite"))
    return u < clamp(p, 0.0, 1.0) # protects only floating-point endpoint error
end

function _noise_path(rng::AbstractRNG, jobs::Int, nclass::Int)
    uniforms() = rand(rng, jobs)
    return BaseNoisePath(randn(rng, nclass), uniforms(), uniforms(), uniforms(), uniforms(), uniforms(),
                         uniforms(), uniforms(), uniforms(), uniforms())
end

function generate_noise_banks(params::AssemblyLineParams, experiment::ExperimentParams)
    validate_params(params, experiment)
    m, nt, ne, jobs = experiment.outer_reps, experiment.inner_train, experiment.inner_eval, experiment.jobs_per_path
    outer = randn(MersenneTwister(experiment.outer_seed), params.nclass, m)
    train_rng, eval_rng = MersenneTwister(experiment.train_seed), MersenneTwister(experiment.eval_seed)
    fixed_train_rng, fixed_eval_rng = MersenneTwister(experiment.baseline_train_seed), MersenneTwister(experiment.baseline_eval_seed)
    train = [[_noise_path(train_rng, jobs, params.nclass) for _ in 1:nt] for _ in 1:m]
    eval = [[_noise_path(eval_rng, jobs, params.nclass) for _ in 1:ne] for _ in 1:m]
    fixed_train = [_noise_path(fixed_train_rng, jobs, params.nclass) for _ in 1:nt]
    fixed_eval = [_noise_path(fixed_eval_rng, jobs, params.nclass) for _ in 1:ne]
    return NoiseBanks(outer, train, eval, fixed_train, fixed_eval)
end

function outer_measurement(z::AbstractVector{<:Real}, outer_normal::AbstractVector{<:Real}, params::AssemblyLineParams)
    _validate_z(z, params); length(outer_normal) == params.nclass || throw(ArgumentError("outer normal has wrong length"))
    return [sqrt(measurement_marginal_variance(z[c], c, params)) * outer_normal[c] for c in 1:params.nclass]
end

function _path_from_theta(theta::AbstractVector{<:Real}, noise::BaseNoisePath, params::AssemblyLineParams)
    length(theta) == params.nclass || throw(ArgumentError("theta has wrong length"))
    jobs = AssemblyJob[]; arrival = 0.0
    for n in eachindex(noise.arrivals)
        arrival += _exp_from_uniform(noise.arrivals[n], params.mean_interarrival)
        c = min(floor(Int, noise.classes[n] * params.nclass) + 1, params.nclass)
        value = clamp(0.15 + 0.16 * c + 0.35 * (noise.values[n] - 0.5), 0.05, 1.0)
        complex = _bernoulli(noise.complexities[n], conditional_complexity_probability(c, theta[c], params))
        p_signal = complex ? params.p_hit[c] : params.p_false[c]
        signal = _bernoulli(noise.signals[n], p_signal) ? 1 : 0
        fast_mean = complex ? params.fast_mean_complex : params.fast_mean_simple
        std_mean = complex ? params.std_mean_complex : params.std_mean_simple
        disruption = _bernoulli(noise.disruptions[n], params.disruption_probability)
        delay = disruption ? _exp_from_uniform(noise.disruption_delays[n], params.disruption_delay_mean) : 0.0
        push!(jobs, AssemblyJob(arrival, c, value, complex, signal,
                                _exp_from_uniform(noise.fast_services[n], fast_mean) + delay,
                                _exp_from_uniform(noise.std_services[n], std_mean) + delay))
    end
    return OperatingPath(collect(Float64, theta), jobs)
end

function conditional_operating_path(eta::AbstractVector{<:Real}, z::AbstractVector{<:Real},
                                    noise::BaseNoisePath, params::AssemblyLineParams)
    theta = sample_theta_posterior(eta, z, noise.theta_normals, params)
    return _path_from_theta(theta, noise, params)
end

function unconditional_operating_path(noise::BaseNoisePath, params::AssemblyLineParams)
    theta = params.tau_theta .* noise.theta_normals
    return _path_from_theta(theta, noise, params)
end

function evaluate_policy(policy::Policy, path::OperatingPath, params::AssemblyLineParams)
    free_fast = 0.0; free_std = 0.0; reward = 0.0; fast_count = 0
    for job in path.jobs
        qf, qs = max(free_fast - job.arrival, 0.0), max(free_std - job.arrival, 0.0)
        route_fast = policy.a_value * job.value - policy.a_queue * (qf - qs) + policy.a_signal * job.signal >= policy.threshold
        if route_fast
            finish = max(job.arrival, free_fast) + job.fast_service; free_fast = finish; fast_count += 1
            reward += params.revenue_scale * job.value - params.fast_routing_cost - params.delay_penalty * (finish - job.arrival)
        else
            finish = max(job.arrival, free_std) + job.std_service; free_std = finish
            reward += params.revenue_scale * job.value - params.delay_penalty * (finish - job.arrival)
        end
    end
    return (reward=reward, fast_count=fast_count, jobs=length(path.jobs))
end

function _select_policy(paths::Vector{OperatingPath}, policies::Vector{Policy}, params::AssemblyLineParams)
    isempty(policies) && throw(ArgumentError("policy set must be nonempty"))
    means = [mean(evaluate_policy(policy, path, params).reward for path in paths) for policy in policies]
    best = 1
    for i in 2:length(means)
        means[i] > means[best] && (best = i) # strict comparison gives lowest-ID tie breaking
    end
    second = length(means) == 1 ? means[best] : maximum(means[i] for i in eachindex(means) if i != best)
    return (policy_id=best, best_mean=means[best], second_mean=second, gap=means[best] - second, means=means)
end

function conditional_selection(outer_id::Int, eta::Vector{Float64}, z::AbstractVector{<:Real},
                               train_noise::Vector{BaseNoisePath}, eval_noise::Vector{BaseNoisePath},
                               policies::Vector{Policy}, params::AssemblyLineParams)
    train_paths = [conditional_operating_path(eta, z, noise, params) for noise in train_noise]
    selection = _select_policy(train_paths, policies, params)
    selected = policies[selection.policy_id]
    eval_results = [evaluate_policy(selected, conditional_operating_path(eta, z, noise, params), params) for noise in eval_noise]
    rewards = [result.reward for result in eval_results]
    post = [posterior_line_condition(eta[c], z[c], c, params) for c in 1:params.nclass]
    return ConditionalSelectionResult(outer_id, eta, [x.mean for x in post], [x.variance for x in post],
        [posterior_expected_complexity(eta[c], z[c], c, params) for c in 1:params.nclass],
        selection.policy_id, selection.best_mean, selection.second_mean, selection.gap, mean(rewards),
        length(rewards) > 1 ? std(rewards) / sqrt(length(rewards)) : 0.0,
        sum(result.fast_count for result in eval_results) / sum(result.jobs for result in eval_results),
        fill(selection.policy_id, length(eval_results)))
end

function estimate_fixed_policy_baseline(params::AssemblyLineParams, experiment::ExperimentParams,
                                        policies::Vector{Policy}, noise_banks::NoiseBanks)
    train_paths = [unconditional_operating_path(noise, params) for noise in noise_banks.fixed_train]
    selection = _select_policy(train_paths, policies, params)
    eval_rewards = [evaluate_policy(policies[selection.policy_id], unconditional_operating_path(noise, params), params).reward
                    for noise in noise_banks.fixed_eval]
    return FixedPolicyEstimate(selection.policy_id, selection.best_mean, mean(eval_rewards),
                               length(eval_rewards) > 1 ? std(eval_rewards) / sqrt(length(eval_rewards)) : 0.0)
end

function estimate_two_stage(z::AbstractVector{<:Real}, params::AssemblyLineParams, experiment::ExperimentParams,
                            policies::Vector{Policy}, noise_banks::NoiseBanks;
                            fixed_policy::Union{Nothing,FixedPolicyEstimate}=nothing)
    validate_params(params, experiment); _validate_z(z, params); isempty(policies) && throw(ArgumentError("policy set must be nonempty"))
    fixed = isnothing(fixed_policy) ? estimate_fixed_policy_baseline(params, experiment, policies, noise_banks) : fixed_policy
    contexts = ConditionalSelectionResult[]
    for r in 1:experiment.outer_reps
        eta = outer_measurement(z, view(noise_banks.outer_normals, :, r), params)
        push!(contexts, conditional_selection(r, eta, z, noise_banks.train[r], noise_banks.eval[r], policies, params))
    end
    values = [context.evaluation_mean for context in contexts]
    ids = [context.selected_policy_id for context in contexts]
    counts = [count(==(i), ids) for i in eachindex(policies)]
    most = findfirst(==(maximum(counts)), counts)
    gross = mean(values); cost = dot(experiment.alpha, z)
    return TwoStageEstimate(collect(Float64, z), [diagnostic_time(z[c], c, params) for c in 1:params.nclass],
        gross, cost, gross - cost, length(values) > 1 ? std(values) / sqrt(length(values)) : 0.0,
        mean(context.best_training_mean for context in contexts), fixed, gross - fixed.evaluation_mean,
        [mean(context.posterior_variance[c] for context in contexts) for c in 1:params.nclass],
        [mean(context.posterior_complexity[c] for context in contexts) for c in 1:params.nclass],
        count(>(0), counts), most, maximum(counts) / length(contexts),
        mean(context.training_gap for context in contexts), mean(context.fast_probability for context in contexts), contexts)
end

end # module AssemblyLineTwoStage
