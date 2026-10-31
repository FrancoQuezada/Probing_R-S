"""Draw independent latent line conditions from their prior distributions."""
function sample_prior_theta(rng::AbstractRNG, params::ModelParameters)
    return [line.prior_mean + sqrt(line.prior_variance) * randn(rng)
            for line in params.lines]
end

"""Sample pre-shift diagnostics, retaining `missing` where no effort is allocated."""
function sample_diagnostics(rng::AbstractRNG, theta, z, params::ModelParameters)
    theta_values = Float64.(theta)
    length(theta_values) == nlines(params) ||
        throw(ArgumentError("theta length $(length(theta_values)) must equal the number of lines $(nlines(params))"))
    all(isfinite, theta_values) || throw(ArgumentError("theta values must be finite"))
    z_values = validate_z(z, params)

    values = Vector{Union{Missing,Float64}}(undef, nlines(params))
    for c in eachindex(params.lines)
        if z_values[c] == 0.0
            values[c] = missing
        else
            variance = diagnostic_variance(c, z_values[c], params)
            values[c] = theta_values[c] + sqrt(variance) * randn(rng)
        end
    end
    return DiagnosticObservation(values)
end

"""Compute the exact independent normal-normal posterior after diagnostics."""
function posterior_parameters(diagnostics::DiagnosticObservation, z,
                              params::ModelParameters)
    length(diagnostics.values) == nlines(params) ||
        throw(ArgumentError("diagnostic length $(length(diagnostics.values)) must equal the number of lines $(nlines(params))"))
    z_values = validate_z(z, params)

    means = Vector{Float64}(undef, nlines(params))
    variances = Vector{Float64}(undef, nlines(params))
    observed = Vector{Bool}(undef, nlines(params))

    for c in eachindex(params.lines)
        line = params.lines[c]
        eta = diagnostics.values[c]
        if z_values[c] == 0.0
            ismissing(eta) ||
                throw(ArgumentError("diagnostic for line $c must be missing when z[$c] == 0"))
            means[c] = line.prior_mean
            variances[c] = line.prior_variance
            observed[c] = false
        else
            ismissing(eta) &&
                throw(ArgumentError("diagnostic for line $c is required when z[$c] > 0"))
            precision = inv(line.prior_variance) +
                        line.max_diagnostic_time * z_values[c] / line.diagnostic_noise
            variance = inv(precision)
            means[c] = variance * (line.prior_mean / line.prior_variance +
                                   line.max_diagnostic_time * z_values[c] /
                                   line.diagnostic_noise * eta)
            variances[c] = variance
            observed[c] = true
        end
    end
    return PosteriorState(means, variances, observed)
end
