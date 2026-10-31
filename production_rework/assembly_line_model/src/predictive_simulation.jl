"""Draw one latent condition per line from an already formed posterior state."""
function sample_posterior_theta(rng::AbstractRNG, posterior::PosteriorState)
    return [posterior.means[c] + sqrt(posterior.variances[c]) * randn(rng)
            for c in eachindex(posterior.means)]
end

"""Conditional rework-arrival rate for a line at latent condition `theta`."""
function arrival_rate(theta::Real, line::LineParameters)::Float64
    theta_value = Float64(theta)
    isfinite(theta_value) || throw(ArgumentError("arrival-rate theta must be finite; got $theta"))
    line.arrival_sensitivity == 0.0 && return line.base_arrival_rate

    displacement = theta_value - line.prior_mean
    exponent = line.arrival_sensitivity * displacement
    log_rate = log(line.base_arrival_rate) + exponent
    max_log = log(floatmax(Float64))
    min_log = log(nextfloat(0.0))
    isfinite(log_rate) && min_log <= log_rate <= max_log ||
        throw(ArgumentError("arrival-rate overflow or underflow for theta=$theta_value (log rate=$log_rate)"))
    rate = exp(log_rate)
    isfinite(rate) && rate > 0.0 ||
        throw(ArgumentError("arrival rate must be finite and positive for theta=$theta_value"))
    return rate
end

"""Stable conditional probability that a rework job is complex."""
function complexity_probability(theta::Real, line::LineParameters)::Float64
    theta_value = Float64(theta)
    isfinite(theta_value) ||
        throw(ArgumentError("complexity-probability theta must be finite; got $theta"))
    linear_predictor = line.complexity_sensitivity == 0.0 ?
        line.complexity_intercept :
        line.complexity_intercept + line.complexity_sensitivity * (theta_value - line.prior_mean)
    if linear_predictor >= 0.0
        return 1.0 / (1.0 + exp(-linear_predictor))
    end
    exponential = exp(linear_predictor)
    return exponential / (1.0 + exponential)
end

"""Exact Poisson sampler: product-of-uniforms for small means and PTRS otherwise."""
function _rand_poisson(rng::AbstractRNG, mean_count::Float64)::Int
    isfinite(mean_count) && mean_count >= 0.0 ||
        throw(ArgumentError("Poisson mean must be finite and nonnegative"))
    mean_count == 0.0 && return 0

    if mean_count < 30.0
        threshold = exp(-mean_count)
        product = 1.0
        count = 0
        while true
            count += 1
            product *= rand(rng)
            product <= threshold && return count - 1
        end
    end

    # Hörmann's transformed-rejection sampler.  `lgamma(k + 1)` is evaluated
    # through Julia's bundled libm, avoiding an external distribution package.
    sqrt_mean = sqrt(mean_count)
    b = 0.931 + 2.53 * sqrt_mean
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2.0)
    log_mean = log(mean_count)

    while true
        u = rand(rng) - 0.5
        v = rand(rng)
        u_abs = 0.5 - abs(u)
        u_abs == 0.0 && continue
        candidate = floor((2.0 * a / u_abs + b) * u + mean_count + 0.43)
        candidate < 0.0 && continue
        candidate > typemax(Int) &&
            throw(ArgumentError("Poisson draw exceeds the supported Int range"))
        k = Int(candidate)

        u_abs >= 0.07 && v <= v_r && return k
        (u_abs < 0.013 && v > u_abs) && continue

        log_acceptance = log(v * inv_alpha / (a / u_abs^2 + b))
        log_pmf = -mean_count + k * log_mean - _logfactorial(k)
        log_acceptance <= log_pmf && return k
    end
end

_logfactorial(k::Int) =
    ccall((:lgamma, Base.Math.libm), Cdouble, (Cdouble,), Float64(k) + 1.0)

function _validate_posterior_dimensions(posterior::PosteriorState,
                                        params::ModelParameters)
    length(posterior.means) == nlines(params) ||
        throw(ArgumentError("posterior dimension $(length(posterior.means)) must equal the number of lines $(nlines(params))"))
    return nothing
end

function _sample_service_time(rng::AbstractRNG, mean_service::Float64,
                              server_name::AbstractString)::Float64
    service = mean_service * randexp(rng)
    isfinite(service) && service > 0.0 ||
        throw(ArgumentError("generated $server_name service time is not finite and positive"))
    return service
end

"""Generate one posterior-predictive shift path, without routing or policy logic."""
function generate_predictive_path(rng::AbstractRNG, posterior::PosteriorState,
                                  params::ModelParameters)::ShiftPath
    _validate_posterior_dimensions(posterior, params)
    theta = sample_posterior_theta(rng, posterior)
    jobs = ReworkJob[]

    for c in eachindex(params.lines)
        line = params.lines[c]
        rate = arrival_rate(theta[c], line)
        mean_count = rate * params.horizon
        isfinite(mean_count) ||
            throw(ArgumentError("Poisson mean over the horizon overflows for line $c"))
        n_arrivals = _rand_poisson(rng, mean_count)

        for _ in 1:n_arrivals
            arrival = rand(rng) * params.horizon
            complex = rand(rng) < complexity_probability(theta[c], line)
            service_index = complex ? 2 : 1
            service_fast = _sample_service_time(rng, line.fast_service_means[service_index], "FAST")
            service_standard = _sample_service_time(rng, line.standard_service_means[service_index], "STANDARD")
            push!(jobs, ReworkJob(arrival, c, complex, service_fast, service_standard))
        end
    end

    # Ties are ordered by line index; `MergeSort` preserves generation order
    # for any remaining same-line ties.
    sort!(jobs; by=job -> (job.arrival, job.line), alg=MergeSort)
    return ShiftPath(jobs)
end

"""Generate independent posterior-predictive shifts, each with a fresh Θ draw."""
function generate_predictive_bank(rng::AbstractRNG, posterior::PosteriorState,
                                  params::ModelParameters,
                                  npaths::Integer)::Vector{ShiftPath}
    npaths >= 0 || throw(ArgumentError("npaths must be nonnegative"))
    count = Int(npaths)
    paths = Vector{ShiftPath}(undef, count)
    for path_index in eachindex(paths)
        paths[path_index] = generate_predictive_path(rng, posterior, params)
    end
    return paths
end
