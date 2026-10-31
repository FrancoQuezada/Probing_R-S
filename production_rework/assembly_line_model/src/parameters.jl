"""Number of assembly lines in a validated model parameterization."""
nlines(params::ModelParameters) = length(params.lines)

"""Deterministic product value attached to an assembly line."""
function line_value(params::ModelParameters, line::Integer)
    1 <= line <= nlines(params) ||
        throw(ArgumentError("line index $line is outside 1:$(nlines(params))"))
    return params.lines[line].value
end

"""Validate and normalize a probing vector for this model."""
function validate_z(z, params::ModelParameters)
    length(z) == nlines(params) ||
        throw(ArgumentError("z length $(length(z)) must equal the number of lines $(nlines(params))"))
    normalized = Float64.(z)
    all(x -> isfinite(x) && 0.0 <= x <= 1.0, normalized) ||
        throw(ArgumentError("every z coordinate must be finite and lie in [0, 1]"))
    return normalized
end

"""Conditional diagnostic variance ν[c] / (Tmax[c] * z[c]) for positive effort."""
function diagnostic_variance(c::Integer, zc::Real, params::ModelParameters)
    1 <= c <= nlines(params) ||
        throw(ArgumentError("line index $c is outside 1:$(nlines(params))"))
    isfinite(zc) && 0.0 < zc <= 1.0 ||
        throw(ArgumentError("diagnostic variance is defined only for z in (0, 1]"))
    line = params.lines[c]
    return line.diagnostic_noise / (line.max_diagnostic_time * Float64(zc))
end
