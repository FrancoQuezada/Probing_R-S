"""Normalize an integer master seed without accepting values outside `Int`."""
function _normalize_master_seed(master_seed::Integer)::Int
    try
        return Int(master_seed)
    catch error
        error isa InexactError || rethrow()
        throw(ArgumentError("master_seed must be representable as an Int"))
    end
end

"""Canonical Float64 representation used for candidate identity and seeding."""
_canonical_coordinate(value::Float64) = iszero(value) ? 0.0 : value

"""One fixed SplitMix64-style mixing step, with deliberately fixed constants."""
function _mix_seed_word(word::UInt64)::UInt64
    word ⊻= word >> 30
    word *= 0xbf58476d1ce4e5b9
    word ⊻= word >> 27
    word *= 0x94d049bb133111eb
    return word ⊻ (word >> 31)
end

"""Derive a deterministic local RNG seed from an `Int` seed and Float64 bits.

The procedure is intentionally small and stable: it folds the two's-complement
bits of `master_seed` and the IEEE-754 bits of each canonical coordinate using
fixed SplitMix64 constants.  It is not intended to be collision resistant.
"""
function _candidate_seed(master_seed::Integer, z)::Int
    normalized_seed = _normalize_master_seed(master_seed)
    state = reinterpret(UInt64, Int64(normalized_seed)) ⊻ 0x9e3779b97f4a7c15
    for coordinate in z
        value = _canonical_coordinate(Float64(coordinate))
        state += reinterpret(UInt64, value) + 0x9e3779b97f4a7c15
        state = _mix_seed_word(state)
    end
    return Int(mod(state, UInt64(typemax(Int))))
end

function _positive_search_count(value::Integer, name::AbstractString)::Int
    return _positive_sample_size(value, name)
end

"""Validate all candidates and compress exact Float64 duplicates by first occurrence."""
function _canonicalize_candidates(candidates::AbstractVector{<:AbstractVector{<:Real}},
                                  params::ModelParameters)
    isempty(candidates) && throw(ArgumentError("at least one probing candidate is required"))
    unique_candidates = Vector{Tuple{Vararg{Float64}}}()
    original_to_unique = Int[]
    for candidate in candidates
        normalized = validate_z(candidate, params)
        canonical = Tuple(_canonical_coordinate(value) for value in normalized)
        unique_index = findfirst(==(canonical), unique_candidates)
        if isnothing(unique_index)
            push!(unique_candidates, canonical)
            unique_index = length(unique_candidates)
        end
        push!(original_to_unique, unique_index)
    end
    return unique_candidates, original_to_unique
end

"""Return whether `left` ranks before `right` under the Stage-7 rule."""
function _candidate_precedes(left_index::Int, right_index::Int,
                             candidates::AbstractVector{<:Tuple},
                             evaluations::AbstractVector{<:ZEvaluationResult})::Bool
    left = evaluations[left_index]
    right = evaluations[right_index]
    left.objective != right.objective && return left.objective > right.objective
    left.gross_operational_value != right.gross_operational_value &&
        return left.gross_operational_value > right.gross_operational_value
    left.standard_error != right.standard_error && return left.standard_error < right.standard_error
    left.probing_cost != right.probing_cost && return left.probing_cost < right.probing_cost
    candidates[left_index] != candidates[right_index] &&
        return isless(candidates[left_index], candidates[right_index])
    return left_index < right_index
end

"""Rank unique candidates by stored Stage-6 summaries without altering them."""
function _rank_candidates(candidates::AbstractVector{<:Tuple},
                          evaluations::AbstractVector{<:ZEvaluationResult})
    length(candidates) == length(evaluations) ||
        throw(ArgumentError("candidates and evaluations must have equal lengths"))
    isempty(candidates) && throw(ArgumentError("at least one candidate is required for ranking"))
    return sort(collect(eachindex(candidates));
                lt=(left, right) -> _candidate_precedes(left, right, candidates, evaluations))
end

"""Evaluate every unique candidate once through Stage 6 using independent streams.

Exact Float64 duplicates are compressed in first-occurrence order and recorded
in `original_to_unique` in the returned `FiniteGridSearchResult`.
"""
function evaluate_z_candidates(master_seed::Integer,
                               candidates::AbstractVector{<:AbstractVector{<:Real}},
                               params::ModelParameters,
                               policies::AbstractVector{<:Policy};
                               outer_reps::Integer, inner_train::Integer,
                               inner_eval::Integer,
                               randomization_mode::Symbol=:independent)::FiniteGridSearchResult
    normalized_seed = _normalize_master_seed(master_seed)
    outer_count = _positive_search_count(outer_reps, "outer_reps")
    training_count = _positive_search_count(inner_train, "inner_train")
    evaluation_count = _positive_search_count(inner_eval, "inner_eval")
    isempty(policies) && throw(ArgumentError("at least one candidate policy is required"))
    unique_candidates, original_to_unique = _canonicalize_candidates(candidates, params)

    randomization_mode in (:independent, :crn) ||
        throw(ArgumentError("randomization_mode must be :independent or :crn"))
    if randomization_mode === :crn
        evaluations = [evaluate_z_crn(normalized_seed, collect(candidate), params, policies;
                                      outer_reps=outer_count, inner_train=training_count,
                                      inner_eval=evaluation_count)
                       for candidate in unique_candidates]
        ranking = _rank_candidates(unique_candidates, evaluations)
        best_index = ranking[1]
        return FiniteGridSearchResult(Tuple(unique_candidates), Tuple(fill(0, length(unique_candidates))),
                                      Tuple(evaluations), Tuple(ranking), best_index,
                                      unique_candidates[best_index], evaluations[best_index],
                                      Tuple(original_to_unique), normalized_seed, outer_count,
                                      training_count, evaluation_count, :crn, nothing)
    end

    candidate_seeds = Int[]
    evaluations = ZEvaluationResult[]
    for candidate in unique_candidates
        seed = _candidate_seed(normalized_seed, candidate)
        push!(candidate_seeds, seed)
        push!(evaluations, evaluate_z(MersenneTwister(seed), collect(candidate), params, policies;
                                      outer_reps=outer_count, inner_train=training_count,
                                      inner_eval=evaluation_count))
    end
    ranking = _rank_candidates(unique_candidates, evaluations)
    best_index = ranking[1]
    return FiniteGridSearchResult(Tuple(unique_candidates), Tuple(candidate_seeds),
                                  Tuple(evaluations), Tuple(ranking), best_index,
                                  unique_candidates[best_index], evaluations[best_index],
                                  Tuple(original_to_unique), normalized_seed, outer_count,
                                  training_count, evaluation_count)
end

function _positive_grid_limit(max_candidates::Integer)::Int
    try
        limit = Int(max_candidates)
        limit > 0 || throw(ArgumentError("max_candidates must be positive"))
        return limit
    catch error
        error isa InexactError || rethrow()
        throw(ArgumentError("max_candidates must be representable as an Int"))
    end
end

"""Construct a bounded Cartesian product, with line 1 varying slowest."""
function build_z_grid(coordinate_values::AbstractVector{<:AbstractVector{<:Real}},
                      params::ModelParameters;
                      max_candidates::Integer=10_000)::Vector{Vector{Float64}}
    length(coordinate_values) == nlines(params) ||
        throw(ArgumentError("coordinate_values length must equal the number of lines"))
    limit = _positive_grid_limit(max_candidates)
    coordinates = Vector{Vector{Float64}}(undef, nlines(params))
    for line in eachindex(coordinate_values)
        isempty(coordinate_values[line]) &&
            throw(ArgumentError("coordinate_values[$line] must be nonempty"))
        values = Vector{Float64}()
        for value in coordinate_values[line]
            normalized = _canonical_coordinate(Float64(value))
            isfinite(normalized) && 0.0 <= normalized <= 1.0 ||
                throw(ArgumentError("grid coordinates must be finite and lie in [0, 1]"))
            findfirst(==(normalized), values) === nothing && push!(values, normalized)
        end
        coordinates[line] = values
    end

    candidate_count = 1
    for values in coordinates
        length(values) > div(typemax(Int), candidate_count) &&
            throw(ArgumentError("requested Cartesian grid size overflows Int"))
        candidate_count *= length(values)
    end
    candidate_count <= limit ||
        throw(ArgumentError("requested Cartesian grid contains $candidate_count candidates, exceeding max_candidates=$limit"))

    grid = [Float64[]]
    for values in coordinates
        next_grid = Vector{Vector{Float64}}(undef, 0)
        sizehint!(next_grid, length(grid) * length(values))
        for prefix in grid, value in values
            push!(next_grid, [prefix; value])
        end
        grid = next_grid
    end
    return grid
end

"""Build one finite Cartesian probing grid and evaluate it once, without refinement."""
function finite_grid_search(master_seed::Integer, coordinate_values,
                            params::ModelParameters,
                            policies::AbstractVector{<:Policy};
                            outer_reps::Integer, inner_train::Integer,
                            inner_eval::Integer, max_candidates::Integer=10_000,
                            randomization_mode::Symbol=:independent)
    candidates = build_z_grid(coordinate_values, params; max_candidates=max_candidates)
    return evaluate_z_candidates(master_seed, candidates, params, policies;
                                 outer_reps=outer_reps, inner_train=inner_train,
                                 inner_eval=inner_eval, randomization_mode=randomization_mode)
end

"""Return the CRN paired difference (including deterministic probing costs)."""
function paired_comparison(search_result::FiniteGridSearchResult,
                           first_index::Integer, second_index::Integer)::PairedComparisonResult
    search_result.randomization_mode === :crn ||
        throw(ArgumentError("paired_comparison is available only for randomization_mode=:crn"))
    first, second = Int(first_index), Int(second_index)
    count = length(search_result.evaluations)
    1 <= first <= count && 1 <= second <= count || throw(ArgumentError("candidate index is out of bounds"))
    left, right = search_result.evaluations[first], search_result.evaluations[second]
    length(left.outer_values) == length(right.outer_values) || throw(ArgumentError("outer replication counts must match"))
    differences = [left.outer_values[i] - right.outer_values[i] - (left.probing_cost - right.probing_cost)
                   for i in eachindex(left.outer_values)]
    mean_difference, paired_se = _mean_and_se(differences)
    if first == second
        mean_difference, paired_se = 0.0, 0.0
    end
    x, y = left.outer_values, right.outer_values
    mean_x, mean_y = sum(x) / length(x), sum(y) / length(y)
    denominator = sqrt(sum((v - mean_x)^2 for v in x) * sum((v - mean_y)^2 for v in y))
    correlation = denominator == 0.0 ? NaN : sum((x[i] - mean_x) * (y[i] - mean_y) for i in eachindex(x)) / denominator
    return PairedComparisonResult(first, second, search_result.candidates[first], search_result.candidates[second],
                                  mean_difference, paired_se, correlation, length(x))
end
