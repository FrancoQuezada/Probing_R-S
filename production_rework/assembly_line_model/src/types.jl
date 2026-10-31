"""One assembly line's fixed physical, diagnostic, and service parameters."""
struct LineParameters
    value::Float64
    prior_mean::Float64
    prior_variance::Float64
    diagnostic_noise::Float64
    max_diagnostic_time::Float64
    base_arrival_rate::Float64
    arrival_sensitivity::Float64
    complexity_intercept::Float64
    complexity_sensitivity::Float64
    fast_service_means::NTuple{2,Float64}      # (simple, complex)
    standard_service_means::NTuple{2,Float64}  # (simple, complex)

    function LineParameters(
        value::Float64,
        prior_mean::Float64,
        prior_variance::Float64,
        diagnostic_noise::Float64,
        max_diagnostic_time::Float64,
        base_arrival_rate::Float64,
        arrival_sensitivity::Float64,
        complexity_intercept::Float64,
        complexity_sensitivity::Float64,
        fast_service_means::NTuple{2,Float64},
        standard_service_means::NTuple{2,Float64},
    )
        isfinite(value) && 0.0 <= value <= 1.0 ||
            throw(ArgumentError("line value must be finite and lie in [0, 1]"))
        isfinite(prior_mean) ||
            throw(ArgumentError("prior mean must be finite"))
        isfinite(prior_variance) && prior_variance > 0.0 ||
            throw(ArgumentError("prior variance must be finite and positive"))
        isfinite(diagnostic_noise) && diagnostic_noise > 0.0 ||
            throw(ArgumentError("diagnostic noise ν must be finite and positive"))
        isfinite(max_diagnostic_time) && max_diagnostic_time > 0.0 ||
            throw(ArgumentError("maximum diagnostic time must be finite and positive"))
        isfinite(base_arrival_rate) && base_arrival_rate > 0.0 ||
            throw(ArgumentError("base arrival rate must be finite and positive"))
        isfinite(arrival_sensitivity) && arrival_sensitivity >= 0.0 ||
            throw(ArgumentError("arrival sensitivity must be finite and nonnegative"))
        isfinite(complexity_intercept) ||
            throw(ArgumentError("complexity intercept must be finite"))
        isfinite(complexity_sensitivity) && complexity_sensitivity >= 0.0 ||
            throw(ArgumentError("complexity sensitivity must be finite and nonnegative"))
        all(x -> isfinite(x) && x > 0.0, fast_service_means) ||
            throw(ArgumentError("FAST service means must be finite and positive"))
        all(x -> isfinite(x) && x > 0.0, standard_service_means) ||
            throw(ArgumentError("STANDARD service means must be finite and positive"))
        new(value, prior_mean, prior_variance, diagnostic_noise,
            max_diagnostic_time, base_arrival_rate, arrival_sensitivity,
            complexity_intercept, complexity_sensitivity,
            fast_service_means, standard_service_means)
    end
end

function LineParameters(value::Real, prior_mean::Real, prior_variance::Real,
                        diagnostic_noise::Real, max_diagnostic_time::Real,
                        base_arrival_rate::Real, arrival_sensitivity::Real,
                        complexity_intercept::Real, complexity_sensitivity::Real,
                        fast_service_means, standard_service_means)
    fast = _service_means(fast_service_means, "FAST")
    standard = _service_means(standard_service_means, "STANDARD")
    return LineParameters(Float64(value), Float64(prior_mean), Float64(prior_variance),
                          Float64(diagnostic_noise), Float64(max_diagnostic_time),
                          Float64(base_arrival_rate), Float64(arrival_sensitivity),
                          Float64(complexity_intercept), Float64(complexity_sensitivity),
                          fast, standard)
end

function _service_means(means, server_name::AbstractString)
    length(means) == 2 ||
        throw(ArgumentError("$server_name service means must contain (simple, complex)"))
    return (Float64(means[1]), Float64(means[2]))
end

"""Model-wide parameters. Product values remain exclusively in `lines`."""
struct ModelParameters
    lines::Tuple{Vararg{LineParameters}}
    probe_cost::Tuple{Vararg{Float64}}
    horizon::Float64
    revenue_scale::Float64
    fast_routing_cost::Float64
    delay_penalty::Float64

    function ModelParameters(lines::Tuple{Vararg{LineParameters}},
                             probe_cost::Tuple{Vararg{Float64}}, horizon::Float64,
                             revenue_scale::Float64, fast_routing_cost::Float64,
                             delay_penalty::Float64)
        isempty(lines) && throw(ArgumentError("at least one line is required"))
        length(probe_cost) == length(lines) ||
            throw(ArgumentError("probe_cost length must equal the number of lines"))
        all(x -> isfinite(x) && x >= 0.0, probe_cost) ||
            throw(ArgumentError("probe costs must be finite and nonnegative"))
        isfinite(horizon) && horizon > 0.0 ||
            throw(ArgumentError("horizon must be finite and positive"))
        isfinite(revenue_scale) && revenue_scale >= 0.0 ||
            throw(ArgumentError("revenue scale must be finite and nonnegative"))
        isfinite(fast_routing_cost) && fast_routing_cost >= 0.0 ||
            throw(ArgumentError("FAST routing cost must be finite and nonnegative"))
        isfinite(delay_penalty) && delay_penalty >= 0.0 ||
            throw(ArgumentError("delay penalty must be finite and nonnegative"))
        new(lines, probe_cost, horizon, revenue_scale, fast_routing_cost, delay_penalty)
    end
end

function ModelParameters(lines::AbstractVector{<:LineParameters}, probe_cost,
                         horizon::Real;
                         revenue_scale::Real=8.0,
                         fast_routing_cost::Real=0.7,
                         delay_penalty::Real=0.42)
    return ModelParameters(Tuple(lines), Tuple(Float64.(probe_cost)), Float64(horizon),
                           Float64(revenue_scale), Float64(fast_routing_cost),
                           Float64(delay_penalty))
end

"""Pre-shift diagnostic values; `missing` denotes a coordinate with z[c] == 0."""
struct DiagnosticObservation
    values::Tuple{Vararg{Union{Missing,Float64}}}

    function DiagnosticObservation(values::Tuple{Vararg{Union{Missing,Float64}}})
        all(x -> ismissing(x) || isfinite(x), values) ||
            throw(ArgumentError("diagnostic observations must be finite or missing"))
        new(values)
    end
end

function DiagnosticObservation(values)
    normalized = map(values) do value
        ismissing(value) ? missing : Float64(value)
    end
    return DiagnosticObservation(Tuple(normalized))
end

"""Coordinatewise posterior moments; it deliberately stores no latent truth."""
struct PosteriorState
    means::Tuple{Vararg{Float64}}
    variances::Tuple{Vararg{Float64}}
    observed::Tuple{Vararg{Bool}}

    function PosteriorState(means::Tuple{Vararg{Float64}},
                            variances::Tuple{Vararg{Float64}},
                            observed::Tuple{Vararg{Bool}})
        length(means) == length(variances) == length(observed) ||
            throw(ArgumentError("posterior means, variances, and mask must have equal lengths"))
        all(isfinite, means) || throw(ArgumentError("posterior means must be finite"))
        all(x -> isfinite(x) && x > 0.0, variances) ||
            throw(ArgumentError("posterior variances must be finite and positive"))
        new(means, variances, observed)
    end
end

PosteriorState(means, variances, observed) =
    PosteriorState(Tuple(Float64.(means)), Tuple(Float64.(variances)), Tuple(Bool.(observed)))

"""One rework job. Its value is `params.lines[job.line].value`, never a random field."""
struct ReworkJob
    arrival::Float64
    line::Int
    complex::Bool
    service_fast::Float64
    service_standard::Float64

    function ReworkJob(arrival::Float64, line::Int, complex::Bool,
                       service_fast::Float64, service_standard::Float64)
        isfinite(arrival) && arrival >= 0.0 ||
            throw(ArgumentError("job arrival must be finite and nonnegative"))
        line >= 1 || throw(ArgumentError("job line index must be positive"))
        isfinite(service_fast) && service_fast > 0.0 ||
            throw(ArgumentError("FAST service time must be finite and positive"))
        isfinite(service_standard) && service_standard > 0.0 ||
            throw(ArgumentError("STANDARD service time must be finite and positive"))
        new(arrival, line, complex, service_fast, service_standard)
    end
end

ReworkJob(arrival::Real, line::Integer, complex::Bool,
          service_fast::Real, service_standard::Real) =
    ReworkJob(Float64(arrival), Int(line), complex, Float64(service_fast), Float64(service_standard))

struct ShiftPath
    jobs::Tuple{Vararg{ReworkJob}}
    ShiftPath(jobs::Tuple{Vararg{ReworkJob}}) = new(jobs)
end

ShiftPath(jobs::AbstractVector{<:ReworkJob}) = ShiftPath(Tuple(jobs))

struct Policy
    av::Float64
    aq::Float64
    tau::Float64

    function Policy(av::Float64, aq::Float64, tau::Float64)
        isfinite(av) && av >= 0.0 || throw(ArgumentError("policy av must be finite and nonnegative"))
        isfinite(aq) && aq >= 0.0 || throw(ArgumentError("policy aq must be finite and nonnegative"))
        isfinite(tau) || throw(ArgumentError("policy tau must be finite"))
        new(av, aq, tau)
    end
end

Policy(av::Real, aq::Real, tau::Real) = Policy(Float64(av), Float64(aq), Float64(tau))

struct PolicySelectionResult
    selected_index::Int
    selected_policy::Policy
    training_values::Tuple{Vararg{Float64}}
    selected_training_value::Float64
    evaluation_mean::Union{Nothing,Float64}
    evaluation_standard_error::Union{Nothing,Float64}
    ntrain::Int
    neval::Union{Nothing,Int}

    function PolicySelectionResult(selected_index::Int, selected_policy::Policy,
                                   training_values::Tuple{Vararg{Float64}},
                                   selected_training_value::Float64,
                                   evaluation_mean::Union{Nothing,Float64},
                                   evaluation_standard_error::Union{Nothing,Float64},
                                   ntrain::Int, neval::Union{Nothing,Int})
        selected_index >= 1 || throw(ArgumentError("selected policy index must be positive"))
        selected_index <= length(training_values) ||
            throw(ArgumentError("selected policy index must index training_values"))
        all(isfinite, training_values) ||
            throw(ArgumentError("training values must be finite"))
        isfinite(selected_training_value) ||
            throw(ArgumentError("selected training value must be finite"))
        selected_training_value == training_values[selected_index] ||
            throw(ArgumentError("selected training value must match its indexed training value"))
        ntrain > 0 || throw(ArgumentError("ntrain must be positive"))
        if isnothing(evaluation_mean) != isnothing(evaluation_standard_error)
            throw(ArgumentError("evaluation mean and standard error must either both be present or both be absent"))
        end
        if !isnothing(evaluation_mean)
            isfinite(evaluation_mean) || throw(ArgumentError("evaluation mean must be finite"))
            isfinite(evaluation_standard_error) && evaluation_standard_error >= 0.0 ||
                throw(ArgumentError("evaluation standard error must be finite and nonnegative"))
            !isnothing(neval) && neval > 0 ||
                throw(ArgumentError("neval must be positive when evaluation results are present"))
        elseif !isnothing(neval)
            throw(ArgumentError("neval is meaningful only when evaluation results are present"))
        end
        new(selected_index, selected_policy, training_values, selected_training_value,
            evaluation_mean, evaluation_standard_error, ntrain, neval)
    end
end

function PolicySelectionResult(selected_index::Integer, selected_policy::Policy,
                               training_values)
    values = Tuple(Float64.(training_values))
    index = Int(selected_index)
    return PolicySelectionResult(index, selected_policy, values, values[index],
                                 nothing, nothing, length(values), nothing)
end

function PolicySelectionResult(selected_index::Integer, selected_policy::Policy,
                               training_values, selected_training_value::Real,
                               evaluation_mean::Real,
                               evaluation_standard_error::Real,
                               ntrain::Integer, neval::Integer)
    return PolicySelectionResult(Int(selected_index), selected_policy,
                                 Tuple(Float64.(training_values)),
                                 Float64(selected_training_value),
                                 Float64(evaluation_mean),
                                 Float64(evaluation_standard_error),
                                 Int(ntrain), Int(neval))
end

struct ZEvaluationResult
    z::Tuple{Vararg{Float64}}
    gross_operational_value::Float64
    probing_cost::Float64
    objective::Float64
    standard_error::Float64
    outer_values::Tuple{Vararg{Float64}}
    selected_policy_frequencies::Tuple{Vararg{Float64}}
    seed::Union{Nothing,Int}
    selected_policy_indices::Tuple{Vararg{Int}}
    selected_policy_counts::Tuple{Vararg{Int}}
    outer_reps::Int
    inner_train::Union{Nothing,Int}
    inner_eval::Union{Nothing,Int}
    selected_training_values::Tuple{Vararg{Float64}}
    inner_evaluation_standard_errors::Tuple{Vararg{Float64}}

    function ZEvaluationResult(z::Tuple{Vararg{Float64}}, gross_operational_value::Float64,
                               probing_cost::Float64, objective::Float64,
                               standard_error::Float64, outer_values::Tuple{Vararg{Float64}},
                               selected_policy_frequencies::Tuple{Vararg{Float64}},
                               seed::Union{Nothing,Int},
                               selected_policy_indices::Tuple{Vararg{Int}},
                               selected_policy_counts::Tuple{Vararg{Int}},
                               outer_reps::Int, inner_train::Union{Nothing,Int},
                               inner_eval::Union{Nothing,Int},
                               selected_training_values::Tuple{Vararg{Float64}},
                               inner_evaluation_standard_errors::Tuple{Vararg{Float64}})
        all(x -> isfinite(x) && 0.0 <= x <= 1.0, z) ||
            throw(ArgumentError("z coordinates must be finite and lie in [0, 1]"))
        all(isfinite, (gross_operational_value, probing_cost, objective, standard_error)) ||
            throw(ArgumentError("evaluation summaries must be finite"))
        probing_cost >= 0.0 || throw(ArgumentError("probing cost must be nonnegative"))
        standard_error >= 0.0 || throw(ArgumentError("standard error must be nonnegative"))
        all(isfinite, outer_values) || throw(ArgumentError("outer values must be finite"))
        outer_reps > 0 || throw(ArgumentError("outer_reps must be positive"))
        length(outer_values) == outer_reps ||
            throw(ArgumentError("outer_values length must equal outer_reps"))
        all(x -> isfinite(x) && 0.0 <= x <= 1.0, selected_policy_frequencies) ||
            throw(ArgumentError("policy frequencies must lie in [0, 1]"))
        all(isfinite, selected_training_values) ||
            throw(ArgumentError("selected training values must be finite"))
        all(x -> isfinite(x) && x >= 0.0, inner_evaluation_standard_errors) ||
            throw(ArgumentError("inner evaluation standard errors must be finite and nonnegative"))
        if !isempty(selected_policy_indices) || !isempty(selected_policy_counts)
            length(selected_policy_indices) == outer_reps ||
                throw(ArgumentError("selected policy indices length must equal outer_reps"))
            length(selected_policy_counts) == length(selected_policy_frequencies) ||
                throw(ArgumentError("policy counts and frequencies must have equal lengths"))
            all(index -> 1 <= index <= length(selected_policy_counts), selected_policy_indices) ||
                throw(ArgumentError("selected policy indices must be valid candidate indices"))
            all(count -> count >= 0, selected_policy_counts) ||
                throw(ArgumentError("selected policy counts must be nonnegative"))
            sum(selected_policy_counts) == outer_reps ||
                throw(ArgumentError("selected policy counts must sum to outer_reps"))
            all(selected_policy_counts[i] == count(==(i), selected_policy_indices)
                for i in eachindex(selected_policy_counts)) ||
                throw(ArgumentError("selected policy counts must match selected indices"))
            all(isapprox(selected_policy_frequencies[i],
                         selected_policy_counts[i] / outer_reps)
                for i in eachindex(selected_policy_counts)) ||
                throw(ArgumentError("selected policy frequencies must match selected counts"))
            length(selected_training_values) == outer_reps ||
                throw(ArgumentError("selected training values length must equal outer_reps"))
            length(inner_evaluation_standard_errors) == outer_reps ||
                throw(ArgumentError("inner evaluation standard errors length must equal outer_reps"))
            !isnothing(inner_train) && inner_train > 0 ||
                throw(ArgumentError("inner_train must be positive for a nested evaluation"))
            !isnothing(inner_eval) && inner_eval > 0 ||
                throw(ArgumentError("inner_eval must be positive for a nested evaluation"))
        end
        new(z, gross_operational_value, probing_cost, objective, standard_error,
            outer_values, selected_policy_frequencies, seed, selected_policy_indices,
            selected_policy_counts, outer_reps, inner_train, inner_eval,
            selected_training_values, inner_evaluation_standard_errors)
    end
end

function ZEvaluationResult(z, gross_operational_value::Real, probing_cost::Real,
                           objective::Real, standard_error::Real, outer_values,
                           selected_policy_frequencies, seed::Integer)
    values = Tuple(Float64.(outer_values))
    return ZEvaluationResult(Tuple(Float64.(z)), Float64(gross_operational_value),
                             Float64(probing_cost), Float64(objective),
                             Float64(standard_error), values,
                             Tuple(Float64.(selected_policy_frequencies)), Int(seed),
                             (), (), length(values), nothing, nothing, (), ())
end

"""Immutable summary of one deterministic finite candidate-set evaluation.

`candidates`, `evaluations`, and `candidate_seeds` contain unique decisions in
first-occurrence order.  `original_to_unique` maps every supplied candidate
position to that unique-decision index; exact duplicates are therefore
evaluated once and are not repeated in `ranking`.
"""
struct FiniteGridSearchResult
    candidates::Tuple{Vararg{Tuple{Vararg{Float64}}}}
    candidate_seeds::Tuple{Vararg{Int}}
    evaluations::Tuple{Vararg{ZEvaluationResult}}
    ranking::Tuple{Vararg{Int}}
    best_index::Int
    best_z::Tuple{Vararg{Float64}}
    best_evaluation::ZEvaluationResult
    original_to_unique::Tuple{Vararg{Int}}
    master_seed::Int
    outer_reps::Int
    inner_train::Int
    inner_eval::Int
    randomization_mode::Symbol
    reference_candidate_index::Union{Nothing,Int}

    function FiniteGridSearchResult(
        candidates::Tuple{Vararg{Tuple{Vararg{Float64}}}},
        candidate_seeds::Tuple{Vararg{Int}},
        evaluations::Tuple{Vararg{ZEvaluationResult}},
        ranking::Tuple{Vararg{Int}}, best_index::Int,
        best_z::Tuple{Vararg{Float64}}, best_evaluation::ZEvaluationResult,
        original_to_unique::Tuple{Vararg{Int}}, master_seed::Int,
        outer_reps::Int, inner_train::Int, inner_eval::Int,
        randomization_mode::Symbol=:independent,
        reference_candidate_index::Union{Nothing,Int}=nothing,
    )
        count = length(candidates)
        count > 0 || throw(ArgumentError("at least one unique candidate is required"))
        length(candidate_seeds) == count == length(evaluations) ||
            throw(ArgumentError("candidates, seeds, and evaluations must have equal lengths"))
        all(z -> all(x -> isfinite(x) && 0.0 <= x <= 1.0, z), candidates) ||
            throw(ArgumentError("candidate coordinates must be finite and lie in [0, 1]"))
        all(evaluations[i].z == candidates[i] for i in eachindex(candidates)) ||
            throw(ArgumentError("each evaluation must correspond to its stored candidate"))
        length(ranking) == count && sort(collect(ranking)) == collect(1:count) ||
            throw(ArgumentError("ranking must be a permutation of unique candidate indices"))
        best_index == ranking[1] ||
            throw(ArgumentError("best_index must be the first ranked candidate"))
        best_z == candidates[best_index] ||
            throw(ArgumentError("best_z must match the best stored candidate"))
        best_evaluation == evaluations[best_index] ||
            throw(ArgumentError("best_evaluation must match the best stored evaluation"))
        !isempty(original_to_unique) ||
            throw(ArgumentError("original_to_unique must retain at least one input position"))
        all(index -> 1 <= index <= count, original_to_unique) ||
            throw(ArgumentError("original_to_unique contains an invalid candidate index"))
        outer_reps > 0 && inner_train > 0 && inner_eval > 0 ||
            throw(ArgumentError("search sample sizes must be positive"))
        all(evaluation -> evaluation.outer_reps == outer_reps &&
                          evaluation.inner_train == inner_train &&
                          evaluation.inner_eval == inner_eval, evaluations) ||
            throw(ArgumentError("evaluation sample sizes must match search metadata"))
        randomization_mode in (:independent, :crn) ||
            throw(ArgumentError("randomization_mode must be :independent or :crn"))
        isnothing(reference_candidate_index) || 1 <= reference_candidate_index <= count ||
            throw(ArgumentError("reference_candidate_index must identify a stored candidate"))
        new(candidates, candidate_seeds, evaluations, ranking, best_index, best_z,
            best_evaluation, original_to_unique, master_seed, outer_reps,
            inner_train, inner_eval, randomization_mode, reference_candidate_index)
    end
end

"""Paired CRN comparison of two finite-grid candidates; correlation is NaN if degenerate."""
struct PairedComparisonResult
    first_index::Int
    second_index::Int
    first_z::Tuple{Vararg{Float64}}
    second_z::Tuple{Vararg{Float64}}
    objective_difference::Float64
    paired_standard_error::Float64
    sample_correlation::Float64
    outer_reps::Int
end
