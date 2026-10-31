module SingleJobTwoStage

using Random
using Statistics
using LinearAlgebra

export RoutingAction, FAST, STANDARD, ModelParams, JobContext, SingleJobNoise,
       validate_params, p_hit, p_false, signal_one_probability,
       marginal_signal_probability, posterior_complex_probability,
       expected_service_time, conditional_action_value, evaluate_actions,
       select_action, context_value, context_information_value, generate_context,
       generate_noise_set, context_from_noise, complexity_from_noise,
       signal_from_noise, service_time_from_noise, simulate_realized_replication,
       estimate_objective, estimate_realized_objective,
       estimate_objective_given_noise, estimate_realized_objective_given_noise

"""The only second-stage routing choices in the single-job model."""
@enum RoutingAction FAST STANDARD

"""Parameters for the direct single-job two-stage experiment."""
Base.@kwdef struct ModelParams
    nclass::Int = 4
    p_complex::Vector{Float64} = [0.02, 0.12, 0.55, 0.95]
    p_hit0::Vector{Float64} = [0.25, 0.40, 0.75, 0.92]
    p_false0::Vector{Float64} = [0.05, 0.08, 0.15, 0.20]
    beta::Float64 = 2.20
    fast_mean_simple::Float64 = 0.75
    fast_mean_complex::Float64 = 2.40
    std_mean_simple::Float64 = 1.00
    std_mean_complex::Float64 = 5.80
    revenue_scale::Float64 = 8.0
    fast_routing_cost::Float64 = 0.70
    delay_penalty::Float64 = 0.42
    mean_fast_workload::Float64 = 1.0
    mean_std_workload::Float64 = 1.0
end

"""Freely observed job context at the routing epoch."""
struct JobContext
    class::Int
    value::Float64
    fast_workload::Float64
    std_workload::Float64
end

"""Uniform variates used to make common-random-number comparisons reproducible."""
struct SingleJobNoise
    u_class::Float64
    u_value::Float64
    u_fast_workload::Float64
    u_std_workload::Float64
    u_complex::Float64
    u_signal::Float64
    u_fast_service::Float64
    u_std_service::Float64
end

_valid_unit_interval(x::Real) = isfinite(x) && 0.0 <= x <= 1.0
_valid_uniform(x::Real) = isfinite(x) && 0.0 <= x < 1.0

function _check_class(c::Int, params::ModelParams)
    1 <= c <= params.nclass || throw(ArgumentError("class must be in 1:$(params.nclass), got $c"))
end

function _check_signal(signal::Integer)
    signal in (0, 1) || throw(ArgumentError("signal must be 0 or 1, got $signal"))
end

"""Validate dimensions, probabilities, and positive time-scale inputs."""
function validate_params(params::ModelParams)
    params.nclass > 0 || throw(ArgumentError("nclass must be positive"))
    for (name, values) in (("p_complex", params.p_complex), ("p_hit0", params.p_hit0),
                           ("p_false0", params.p_false0))
        length(values) == params.nclass ||
            throw(ArgumentError("$name must have length $(params.nclass)"))
        all(_valid_unit_interval, values) ||
            throw(ArgumentError("$name entries must be finite probabilities in [0, 1]"))
    end
    isfinite(params.beta) && params.beta >= 0 || throw(ArgumentError("beta must be finite and nonnegative"))
    for (name, value) in (("fast_mean_simple", params.fast_mean_simple),
                          ("fast_mean_complex", params.fast_mean_complex),
                          ("std_mean_simple", params.std_mean_simple),
                          ("std_mean_complex", params.std_mean_complex),
                          ("mean_fast_workload", params.mean_fast_workload),
                          ("mean_std_workload", params.mean_std_workload))
        isfinite(value) && value > 0 || throw(ArgumentError("$name must be finite and positive"))
    end
    for (name, value) in (("revenue_scale", params.revenue_scale),
                          ("fast_routing_cost", params.fast_routing_cost),
                          ("delay_penalty", params.delay_penalty))
        isfinite(value) || throw(ArgumentError("$name must be finite"))
    end
    return params
end

function _validate_context(context::JobContext, params::ModelParams)
    validate_params(params)
    _check_class(context.class, params)
    isfinite(context.value) || throw(ArgumentError("context value must be finite"))
    isfinite(context.fast_workload) && context.fast_workload >= 0 ||
        throw(ArgumentError("fast workload must be finite and nonnegative"))
    isfinite(context.std_workload) && context.std_workload >= 0 ||
        throw(ArgumentError("standard workload must be finite and nonnegative"))
    return context
end

function _validate_effort(z::AbstractVector{<:Real}, params::ModelParams)
    validate_params(params)
    length(z) == params.nclass ||
        throw(ArgumentError("probing-effort vector must have length $(params.nclass)"))
    all(x -> isfinite(x) && 0.0 <= x <= 1.0, z) ||
        throw(ArgumentError("all probing efforts must be finite values in [0, 1]"))
    return z
end

function _validate_alpha(alpha::AbstractVector{<:Real}, params::ModelParams)
    length(alpha) == params.nclass ||
        throw(ArgumentError("alpha must have length $(params.nclass)"))
    all(isfinite, alpha) || throw(ArgumentError("alpha entries must be finite"))
    return alpha
end

"""True-positive probability for class `c` at effort `zc`."""
function p_hit(c::Int, zc::Real, params::ModelParams)
    validate_params(params); _check_class(c, params)
    isfinite(zc) && 0.0 <= zc <= 1.0 || throw(ArgumentError("zc must be in [0, 1]"))
    p = params.p_hit0[c] + (1.0 - params.p_hit0[c]) * (1.0 - exp(-params.beta * zc))
    _valid_unit_interval(p) || throw(ArgumentError("computed hit probability is invalid"))
    return p
end

"""False-positive probability for class `c` at effort `zc`."""
function p_false(c::Int, zc::Real, params::ModelParams)
    validate_params(params); _check_class(c, params)
    isfinite(zc) && 0.0 <= zc <= 1.0 || throw(ArgumentError("zc must be in [0, 1]"))
    p = params.p_false0[c] * exp(-params.beta * zc)
    _valid_unit_interval(p) || throw(ArgumentError("computed false-positive probability is invalid"))
    return p
end

"""Probability of `S=1` conditional on class and the hidden complexity state."""
function signal_one_probability(c::Int, complex::Bool, zc::Real, params::ModelParams)
    return complex ? p_hit(c, zc, params) : p_false(c, zc, params)
end

"""Marginal probability of the requested binary signal after integrating complexity."""
function marginal_signal_probability(c::Int, signal::Integer, zc::Real, params::ModelParams)
    validate_params(params); _check_class(c, params); _check_signal(signal)
    rho = params.p_complex[c]
    p1 = rho * p_hit(c, zc, params) + (1.0 - rho) * p_false(c, zc, params)
    _valid_unit_interval(p1) || throw(ArgumentError("computed marginal signal probability is invalid"))
    return signal == 1 ? p1 : 1.0 - p1
end

function _checked_ratio(numerator::Float64, denominator::Float64, description::AbstractString)
    denominator > 0.0 || throw(DomainError(denominator, "$description conditions on a zero-probability signal"))
    p = numerator / denominator
    _valid_unit_interval(p) || throw(ArgumentError("$description produced an invalid probability"))
    return p
end

"""Bayesian probability that a job is complex given its observed signal."""
function posterior_complex_probability(c::Int, signal::Integer, zc::Real, params::ModelParams)
    validate_params(params); _check_class(c, params); _check_signal(signal)
    rho = params.p_complex[c]
    ph = p_hit(c, zc, params)
    pf = p_false(c, zc, params)
    if signal == 1
        return _checked_ratio(rho * ph, rho * ph + (1.0 - rho) * pf, "positive-signal posterior")
    end
    return _checked_ratio(rho * (1.0 - ph), rho * (1.0 - ph) + (1.0 - rho) * (1.0 - pf),
                          "negative-signal posterior")
end

"""Expected service time under a routing action and posterior complexity belief."""
function expected_service_time(action::RoutingAction, posterior_complex::Real, params::ModelParams)
    validate_params(params)
    _valid_unit_interval(posterior_complex) ||
        throw(ArgumentError("posterior complexity probability must be in [0, 1]"))
    if action == FAST
        return posterior_complex * params.fast_mean_complex +
               (1.0 - posterior_complex) * params.fast_mean_simple
    end
    return posterior_complex * params.std_mean_complex +
           (1.0 - posterior_complex) * params.std_mean_simple
end

"""Conditional expected reward of one action after observing the binary signal."""
function conditional_action_value(action::RoutingAction, context::JobContext, signal::Integer,
                                  z::AbstractVector{<:Real}, params::ModelParams)
    _validate_context(context, params); _validate_effort(z, params); _check_signal(signal)
    posterior = posterior_complex_probability(context.class, signal, z[context.class], params)
    revenue = params.revenue_scale * context.value
    if action == FAST
        expected_time = context.fast_workload + expected_service_time(FAST, posterior, params)
        return revenue - params.fast_routing_cost - params.delay_penalty * expected_time
    end
    expected_time = context.std_workload + expected_service_time(STANDARD, posterior, params)
    return revenue - params.delay_penalty * expected_time
end

"""Evaluate both routing actions; ties are deterministically assigned to FAST."""
function evaluate_actions(context::JobContext, signal::Integer, z::AbstractVector{<:Real}, params::ModelParams)
    _validate_context(context, params); _validate_effort(z, params); _check_signal(signal)
    posterior = posterior_complex_probability(context.class, signal, z[context.class], params)
    value_fast = conditional_action_value(FAST, context, signal, z, params)
    value_standard = conditional_action_value(STANDARD, context, signal, z, params)
    action = value_fast >= value_standard ? FAST : STANDARD
    selected_value = action == FAST ? value_fast : value_standard
    return (posterior_complex=posterior, value_fast=value_fast, value_standard=value_standard,
            selected_action=action, selected_value=selected_value)
end

"""Select FAST exactly when its conditional value is at least STANDARD's value."""
select_action(context::JobContext, signal::Integer, z::AbstractVector{<:Real}, params::ModelParams) =
    evaluate_actions(context, signal, z, params).selected_action

"""Exact post-signal value for one context, obtained by enumerating `S=0,1`."""
function context_value(context::JobContext, z::AbstractVector{<:Real}, params::ModelParams)
    _validate_context(context, params); _validate_effort(z, params)
    c = context.class
    p0 = marginal_signal_probability(c, 0, z[c], params)
    p1 = marginal_signal_probability(c, 1, z[c], params)
    s0 = evaluate_actions(context, 0, z, params)
    s1 = evaluate_actions(context, 1, z, params)
    value = p0 * s0.selected_value + p1 * s1.selected_value
    return (value=value, p_signal0=p0, p_signal1=p1,
            selected_action_s0=s0.selected_action, selected_action_s1=s1.selected_action,
            posterior_s0=s0.posterior_complex, posterior_s1=s1.posterior_complex,
            value_fast_s0=s0.value_fast, value_standard_s0=s0.value_standard,
            value_fast_s1=s1.value_fast, value_standard_s1=s1.value_standard,
            selected_value_s0=s0.selected_value, selected_value_s1=s1.selected_value)
end

"""Compare action selection before the signal with the exact post-signal value."""
function context_information_value(context::JobContext, z::AbstractVector{<:Real}, params::ModelParams)
    cv = context_value(context, z, params)
    expected_fast = cv.p_signal0 * cv.value_fast_s0 + cv.p_signal1 * cv.value_fast_s1
    expected_standard = cv.p_signal0 * cv.value_standard_s0 + cv.p_signal1 * cv.value_standard_s1
    value_before = max(expected_fast, expected_standard)
    info_value = cv.value - value_before
    info_value >= -1e-12 || error("post-signal value should not be below pre-signal value")
    return (value_before_signal=value_before, value_after_signal=cv.value,
            information_value=max(info_value, 0.0), expected_fast_before_signal=expected_fast,
            expected_standard_before_signal=expected_standard)
end

_exp_from_uniform(u::Real, mean::Real) = begin
    _valid_uniform(u) || throw(ArgumentError("uniform variates must be in [0, 1)"))
    -mean * log1p(-u)
end

"""Generate one context under the single-job simplifying context distribution."""
function generate_context(rng::AbstractRNG, params::ModelParams)
    validate_params(params)
    c = rand(rng, 1:params.nclass)
    base_value = 0.15 + 0.16 * c
    value = clamp(base_value + 0.35 * (rand(rng) - 0.5), 0.05, 1.0)
    return JobContext(c, value, _exp_from_uniform(rand(rng), params.mean_fast_workload),
                      _exp_from_uniform(rand(rng), params.mean_std_workload))
end

"""Generate a deterministic set of primitive uniforms for common-random-number comparisons."""
function generate_noise_set(seed::Integer, reps::Integer)
    reps > 0 || throw(ArgumentError("reps must be positive"))
    rng = MersenneTwister(seed)
    return [SingleJobNoise(rand(rng), rand(rng), rand(rng), rand(rng), rand(rng), rand(rng),
                           rand(rng), rand(rng)) for _ in 1:reps]
end

"""Transform base noise into a context using the configured simplifying distribution."""
function context_from_noise(noise::SingleJobNoise, params::ModelParams)
    validate_params(params)
    _valid_uniform(noise.u_class) || throw(ArgumentError("u_class must be in [0, 1)"))
    c = min(floor(Int, noise.u_class * params.nclass) + 1, params.nclass)
    base_value = 0.15 + 0.16 * c
    value = clamp(base_value + 0.35 * (noise.u_value - 0.5), 0.05, 1.0)
    return JobContext(c, value, _exp_from_uniform(noise.u_fast_workload, params.mean_fast_workload),
                      _exp_from_uniform(noise.u_std_workload, params.mean_std_workload))
end

"""Transform base noise into the hidden complexity state for a context."""
function complexity_from_noise(noise::SingleJobNoise, context::JobContext, params::ModelParams)
    _validate_context(context, params); _valid_uniform(noise.u_complex) ||
        throw(ArgumentError("u_complex must be in [0, 1)"))
    return noise.u_complex < params.p_complex[context.class]
end

"""Transform base noise into a signal conditional on a hidden complexity state."""
function signal_from_noise(noise::SingleJobNoise, context::JobContext, complex::Bool,
                           z::AbstractVector{<:Real}, params::ModelParams)
    _validate_context(context, params); _validate_effort(z, params); _valid_uniform(noise.u_signal) ||
        throw(ArgumentError("u_signal must be in [0, 1)"))
    return noise.u_signal < signal_one_probability(context.class, complex, z[context.class], params) ? 1 : 0
end

"""Transform base noise into a realized service time for the selected action."""
function service_time_from_noise(noise::SingleJobNoise, action::RoutingAction, complex::Bool,
                                 params::ModelParams)
    validate_params(params)
    if action == FAST
        mean = complex ? params.fast_mean_complex : params.fast_mean_simple
        return _exp_from_uniform(noise.u_fast_service, mean)
    end
    mean = complex ? params.std_mean_complex : params.std_mean_simple
    return _exp_from_uniform(noise.u_std_service, mean)
end

function _realized_reward(context::JobContext, action::RoutingAction, service::Real, params::ModelParams)
    workload = action == FAST ? context.fast_workload : context.std_workload
    return params.revenue_scale * context.value -
           (action == FAST ? params.fast_routing_cost : 0.0) -
           params.delay_penalty * (workload + service)
end

"""Simulate one realized validation replication without exposing complexity to action selection."""
function simulate_realized_replication(z::AbstractVector{<:Real}, rng::AbstractRNG, params::ModelParams)
    _validate_effort(z, params)
    context = generate_context(rng, params)
    complex = rand(rng) < params.p_complex[context.class]
    signal = rand(rng) < signal_one_probability(context.class, complex, z[context.class], params) ? 1 : 0
    actions = evaluate_actions(context, signal, z, params)
    action = actions.selected_action
    mean = action == FAST ? (complex ? params.fast_mean_complex : params.fast_mean_simple) :
                            (complex ? params.std_mean_complex : params.std_mean_simple)
    service = _exp_from_uniform(rand(rng), mean)
    return (context=context, complex=complex, signal=signal, selected_action=action,
            conditional_fast_value=actions.value_fast, conditional_standard_value=actions.value_standard,
            realized_service_time=service, realized_reward=_realized_reward(context, action, service, params))
end

function _standard_error(values::AbstractVector{<:Real})
    length(values) > 1 ? std(values) / sqrt(length(values)) : 0.0
end

function _summarize_exact(contexts, z::AbstractVector{<:Real}, alpha::AbstractVector{<:Real},
                          params::ModelParams; seed=nothing)
    _validate_effort(z, params); _validate_alpha(alpha, params)
    n = length(contexts); n > 0 || throw(ArgumentError("at least one context is required"))
    values = Vector{Float64}(undef, n)
    information_values = Vector{Float64}(undef, n)
    class_count = zeros(Int, params.nclass)
    class_positive = zeros(params.nclass)
    class_fast = zeros(params.nclass)
    fast_unconditional = 0.0
    fast_s0_numerator = 0.0; fast_s0_denominator = 0.0
    fast_s1_numerator = 0.0; fast_s1_denominator = 0.0
    positive_signal = 0.0
    for (i, context) in enumerate(contexts)
        cv = context_value(context, z, params)
        iv = context_information_value(context, z, params)
        c = context.class
        values[i] = cv.value; information_values[i] = iv.information_value
        class_count[c] += 1
        class_positive[c] += cv.p_signal1
        class_fast[c] += cv.p_signal0 * (cv.selected_action_s0 == FAST) +
                         cv.p_signal1 * (cv.selected_action_s1 == FAST)
        positive_signal += cv.p_signal1
        fast_unconditional += cv.p_signal0 * (cv.selected_action_s0 == FAST) +
                              cv.p_signal1 * (cv.selected_action_s1 == FAST)
        fast_s0_denominator += cv.p_signal0
        fast_s1_denominator += cv.p_signal1
        fast_s0_numerator += cv.p_signal0 * (cv.selected_action_s0 == FAST)
        fast_s1_numerator += cv.p_signal1 * (cv.selected_action_s1 == FAST)
    end
    class_positive_probability = [class_count[c] == 0 ? NaN : class_positive[c] / class_count[c]
                                  for c in 1:params.nclass]
    class_fast_probability = [class_count[c] == 0 ? NaN : class_fast[c] / class_count[c]
                              for c in 1:params.nclass]
    gross = mean(values); cost = dot(alpha, z)
    return (z=collect(Float64, z), gross_value=gross, probing_cost=cost, objective=gross - cost,
            standard_error=_standard_error(values), reps=n, seed=seed, params=params,
            expected_positive_signal_probability=positive_signal / n,
            probability_fast_after_s0=fast_s0_denominator == 0 ? NaN : fast_s0_numerator / fast_s0_denominator,
            probability_fast_after_s1=fast_s1_denominator == 0 ? NaN : fast_s1_numerator / fast_s1_denominator,
            unconditional_probability_fast=fast_unconditional / n,
            mean_information_value=mean(information_values),
            class_specific_positive_signal_probabilities=class_positive_probability,
            class_specific_fast_selection_probabilities=class_fast_probability,
            class_counts=class_count)
end

"""Primary estimator: Monte Carlo over contexts and exact enumeration over both signal values."""
function estimate_objective(z::AbstractVector{<:Real}; reps::Integer=10_000, seed::Integer=20260720,
                            params::ModelParams=ModelParams(), alpha::AbstractVector{<:Real}=zeros(params.nclass))
    _validate_effort(z, params); _validate_alpha(alpha, params); reps > 0 || throw(ArgumentError("reps must be positive"))
    rng = MersenneTwister(seed)
    contexts = [generate_context(rng, params) for _ in 1:reps]
    return _summarize_exact(contexts, z, alpha, params; seed=seed)
end

"""Primary estimator evaluated on an existing common-random-number noise set."""
function estimate_objective_given_noise(z::AbstractVector{<:Real}, noise_set::AbstractVector{SingleJobNoise},
                                        alpha::AbstractVector{<:Real}, params::ModelParams)
    contexts = [context_from_noise(noise, params) for noise in noise_set]
    return _summarize_exact(contexts, z, alpha, params)
end

function _summarize_realized(records, z::AbstractVector{<:Real}, alpha::AbstractVector{<:Real},
                             params::ModelParams; seed=nothing)
    _validate_effort(z, params); _validate_alpha(alpha, params)
    n = length(records); n > 0 || throw(ArgumentError("at least one replication is required"))
    rewards = [record.realized_reward for record in records]
    gross = mean(rewards); cost = dot(alpha, z)
    return (z=collect(Float64, z), gross_value=gross, probing_cost=cost, objective=gross - cost,
            standard_error=_standard_error(rewards), reps=n, seed=seed, params=params)
end

"""Validation estimator based on realized complexity and service-time draws."""
function estimate_realized_objective(z::AbstractVector{<:Real}; reps::Integer=10_000, seed::Integer=20260720,
                                     params::ModelParams=ModelParams(), alpha::AbstractVector{<:Real}=zeros(params.nclass))
    _validate_effort(z, params); _validate_alpha(alpha, params); reps > 0 || throw(ArgumentError("reps must be positive"))
    rng = MersenneTwister(seed)
    records = [simulate_realized_replication(z, rng, params) for _ in 1:reps]
    return _summarize_realized(records, z, alpha, params; seed=seed)
end

"""Realized validation estimator evaluated on an existing common-random-number noise set."""
function estimate_realized_objective_given_noise(z::AbstractVector{<:Real}, noise_set::AbstractVector{SingleJobNoise},
                                                 alpha::AbstractVector{<:Real}, params::ModelParams)
    _validate_effort(z, params); _validate_alpha(alpha, params)
    records = map(noise_set) do noise
        context = context_from_noise(noise, params)
        complex = complexity_from_noise(noise, context, params)
        signal = signal_from_noise(noise, context, complex, z, params)
        actions = evaluate_actions(context, signal, z, params)
        action = actions.selected_action
        service = service_time_from_noise(noise, action, complex, params)
        (context=context, complex=complex, signal=signal, selected_action=action,
         conditional_fast_value=actions.value_fast, conditional_standard_value=actions.value_standard,
         realized_service_time=service, realized_reward=_realized_reward(context, action, service, params))
    end
    return _summarize_realized(records, z, alpha, params)
end

end # module SingleJobTwoStage
