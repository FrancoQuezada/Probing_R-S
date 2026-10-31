"""Normalize one policy-coefficient family and retain exact first occurrences."""
function _canonical_policy_values(values::AbstractVector{<:Real},
                                  family::AbstractString;
                                  nonnegative::Bool)::Vector{Float64}
    isempty(values) && throw(ArgumentError("$family values must be nonempty"))
    normalized = Float64[]
    for (position, value) in pairs(values)
        coefficient = Float64(value)
        valid = isfinite(coefficient) && (!nonnegative || coefficient >= 0.0)
        valid || throw(ArgumentError("$family[$position] = $value must be finite" *
                                     (nonnegative ? " and nonnegative" : "")))
        # Julia's exact `==` is deliberately used here; no tolerance is applied.
        findfirst(==(coefficient), normalized) === nothing && push!(normalized, coefficient)
    end
    return normalized
end

function _policy_grid_count(av_values::AbstractVector{<:Real},
                            aq_values::AbstractVector{<:Real},
                            tau_values::AbstractVector{<:Real})::Int
    count = 1
    for values in (av_values, aq_values, tau_values)
        length(values) > div(typemax(Int), count) &&
            throw(ArgumentError("requested policy-grid size overflows Int"))
        count *= length(values)
    end
    return count
end

function _policy_grid_limit(max_policies::Integer)::Int
    try
        limit = Int(max_policies)
        limit > 0 || throw(ArgumentError("max_policies must be positive"))
        return limit
    catch error
        error isa InexactError || rethrow()
        throw(ArgumentError("max_policies must be representable as an Int"))
    end
end

"""Return the number of distinct policies implied by three coefficient families.

Each input is validated and converted to Float64; exact duplicate values within
one family are counted once in their first-occurrence order.
"""
function policy_grid_size(av_values::AbstractVector{<:Real},
                          aq_values::AbstractVector{<:Real},
                          tau_values::AbstractVector{<:Real})::Int
    av = _canonical_policy_values(av_values, "av"; nonnegative=true)
    aq = _canonical_policy_values(aq_values, "aq"; nonnegative=true)
    tau = _canonical_policy_values(tau_values, "tau"; nonnegative=false)
    return _policy_grid_count(av, aq, tau)
end

"""Build a deterministic Cartesian family of routing policies.

Exact duplicates are removed independently within `av_values`, `aq_values`,
and `tau_values`, retaining their first occurrence.  The returned order is
`av` slowest, `aq` next, and `tau` fastest, so its indices are stable for the
selection and evaluation stages.
"""
function build_policy_grid(av_values::AbstractVector{<:Real},
                           aq_values::AbstractVector{<:Real},
                           tau_values::AbstractVector{<:Real};
                           max_policies::Integer=10_000)::Vector{Policy}
    # Normalize every family before allocating the output policy vector.
    av = _canonical_policy_values(av_values, "av"; nonnegative=true)
    aq = _canonical_policy_values(aq_values, "aq"; nonnegative=true)
    tau = _canonical_policy_values(tau_values, "tau"; nonnegative=false)
    limit = _policy_grid_limit(max_policies)
    count = _policy_grid_count(av, aq, tau)
    count <= limit ||
        throw(ArgumentError("requested policy grid contains $count policies, exceeding max_policies=$limit"))

    policies = Vector{Policy}()
    sizehint!(policies, count)
    for av_coefficient in av
        for aq_coefficient in aq
            for tau_coefficient in tau
                push!(policies, Policy(av_coefficient, aq_coefficient, tau_coefficient))
            end
        end
    end
    return policies
end
