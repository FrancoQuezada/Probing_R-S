module ExperimentRunner

using AssemblyLineModel
using TOML
using Printf
using Statistics

export GridExperimentConfig,
       parse_cli,
       load_experiment_config,
       run_grid_experiment,
       write_experiment_outputs,
       execute_grid_experiment,
       print_summary,
       help_text

const DEFAULT_OUTPUT_DIR = "assembly_line_model/experiment_outputs/grid_experiment"
const REQUIRED_LINE_FIELDS = Set([
    "value", "prior_mean", "prior_variance", "diagnostic_noise", "max_diagnostic_time",
    "baseline_arrival_rate", "arrival_sensitivity", "complexity_intercept",
    "complexity_sensitivity", "fast_simple_mean", "fast_complex_mean",
    "standard_simple_mean", "standard_complex_mean",
])

"""Fully resolved, immutable experiment inputs; mutable caller vectors are copied."""
struct GridExperimentConfig
    name::String
    mode::Symbol
    seed::Int
    outer_reps::Int
    inner_train::Int
    inner_eval::Int
    output_dir::String
    params::ModelParameters
    av_values::Tuple{Vararg{Float64}}
    aq_values::Tuple{Vararg{Float64}}
    tau_values::Tuple{Vararg{Float64}}
    policy_max::Int
    policies::Tuple{Vararg{Policy}}
    z_coordinate_values::Tuple{Vararg{Tuple{Vararg{Float64}}}}
    z_grid::Tuple{Vararg{Tuple{Vararg{Float64}}}}
    candidate_max::Int
    grid_mode::String
    grid_step::Union{Nothing,Float64}
    randomization_mode::Symbol
    crn_reference_z::Union{Nothing,Tuple{Vararg{Float64}}}
    source_config::String
end

_format_number(value::Real) = @sprintf("%.17g", Float64(value))
_format_z(z) = "(" * join((_format_number(value) for value in z), ", ") * ")"
_format_values(values) = "[" * join((_format_number(value) for value in values), ", ") * "]"

function _as_float(value, context::AbstractString)::Float64
    value isa Real && !(value isa Bool) || throw(ArgumentError("$context must be numeric"))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError("$context must be finite"))
    return converted
end

function _as_int(value, context::AbstractString)::Int
    value isa Integer && !(value isa Bool) || throw(ArgumentError("$context must be an integer"))
    try
        return Int(value)
    catch error
        error isa InexactError || rethrow()
        throw(ArgumentError("$context must be representable as Int"))
    end
end

function _positive_int(value, context::AbstractString)::Int
    converted = _as_int(value, context)
    converted > 0 || throw(ArgumentError("$context must be positive"))
    return converted
end

function _table(root::AbstractDict, name::String)::AbstractDict
    haskey(root, name) || throw(ArgumentError("missing TOML [$name] section"))
    root[name] isa AbstractDict || throw(ArgumentError("TOML [$name] section must be a table"))
    return root[name]
end

function _required(table::AbstractDict, key::String, context::AbstractString)
    haskey(table, key) || throw(ArgumentError("missing required $context.$key"))
    return table[key]
end

function _float_vector(value, context::AbstractString)::Vector{Float64}
    value isa AbstractVector || throw(ArgumentError("$context must be an array"))
    isempty(value) && throw(ArgumentError("$context must be nonempty"))
    return [_as_float(entry, "$context[$index]") for (index, entry) in pairs(value)]
end

function _parse_csv_floats(text::AbstractString, context::AbstractString)::Vector{Float64}
    isempty(text) && throw(ArgumentError("$context must be nonempty"))
    pieces = split(text, ','; keepempty=true)
    any(isempty, pieces) && throw(ArgumentError("$context contains an empty value"))
    values = Float64[]
    for (index, piece) in pairs(pieces)
        try
            push!(values, _as_float(parse(Float64, piece), "$context[$index]"))
        catch error
            error isa ArgumentError || rethrow()
            throw(ArgumentError("$context[$index] is not a finite floating-point value: $piece"))
        end
    end
    return values
end

function _parse_z_values(text::AbstractString)::Vector{Vector{Float64}}
    segments = split(text, ';'; keepempty=true)
    isempty(segments) || any(isempty, segments) &&
        throw(ArgumentError("--z-values contains an empty coordinate segment"))
    return [_parse_csv_floats(segment, "--z-values line $line")
            for (line, segment) in pairs(segments)]
end

function _deduplicate_coordinates(values::Vector{Float64})
    output = Float64[]
    for value in values
        isfinite(value) && 0.0 <= value <= 1.0 ||
            throw(ArgumentError("probing coordinate $value must be finite and lie in [0, 1]"))
        findfirst(==(value), output) === nothing && push!(output, value)
    end
    return output
end

function _step_coordinates(step_value)::Vector{Float64}
    step = _as_float(step_value, "probing_grid.step")
    0.0 < step <= 1.0 || throw(ArgumentError("probing_grid.step must lie in (0, 1]"))
    intervals = round(Int, inv(step))
    intervals > 0 && isapprox(intervals * step, 1.0; rtol=0.0, atol=1e-12) ||
        throw(ArgumentError("probing_grid.step must divide 1 exactly within 1e-12"))
    return Float64[index / intervals for index in 0:intervals]
end

function _explicit_coordinates(value, line_count::Int)::Vector{Vector{Float64}}
    value isa AbstractVector || throw(ArgumentError("probing_grid.coordinate_values must be an array of arrays"))
    length(value) == line_count ||
        throw(ArgumentError("probing_grid.coordinate_values must contain one vector per line"))
    coordinates = Vector{Vector{Float64}}(undef, line_count)
    for line in 1:line_count
        coordinates[line] = _deduplicate_coordinates(_float_vector(value[line], "probing_grid.coordinate_values[$line]"))
    end
    return coordinates
end

function _line_parameters(value::AbstractDict, index::Int)::LineParameters
    unknown = setdiff(Set(String.(keys(value))), REQUIRED_LINE_FIELDS)
    isempty(unknown) || throw(ArgumentError("lines[$index] contains unknown fields: $(join(sort!(collect(unknown)), ", "))"))
    missing = setdiff(REQUIRED_LINE_FIELDS, Set(String.(keys(value))))
    isempty(missing) || throw(ArgumentError("lines[$index] is missing fields: $(join(sort!(collect(missing)), ", "))"))
    field(name) = _as_float(_required(value, name, "lines[$index]"), "lines[$index].$name")
    return LineParameters(field("value"), field("prior_mean"), field("prior_variance"),
                          field("diagnostic_noise"), field("max_diagnostic_time"),
                          field("baseline_arrival_rate"), field("arrival_sensitivity"),
                          field("complexity_intercept"), field("complexity_sensitivity"),
                          (field("fast_simple_mean"), field("fast_complex_mean")),
                          (field("standard_simple_mean"), field("standard_complex_mean")))
end

function _model_parameters(root::AbstractDict)::ModelParameters
    model = _table(root, "model")
    lines_raw = _required(root, "lines", "TOML root")
    lines_raw isa AbstractVector && !isempty(lines_raw) ||
        throw(ArgumentError("TOML [[lines]] must be a nonempty array of tables"))
    lines = LineParameters[]
    for (index, line) in pairs(lines_raw)
        line isa AbstractDict || throw(ArgumentError("lines[$index] must be a table"))
        push!(lines, _line_parameters(line, index))
    end
    costs = _float_vector(_required(model, "probing_costs", "model"), "model.probing_costs")
    length(costs) == length(lines) ||
        throw(ArgumentError("model.probing_costs length must equal the number of lines"))
    return ModelParameters(lines, costs, _as_float(_required(model, "horizon", "model"), "model.horizon");
                           revenue_scale=_as_float(_required(model, "revenue_scale", "model"), "model.revenue_scale"),
                           fast_routing_cost=_as_float(_required(model, "fast_routing_cost", "model"), "model.fast_routing_cost"),
                           delay_penalty=_as_float(_required(model, "delay_penalty", "model"), "model.delay_penalty"))
end

function _parse_cli_value(args::AbstractVector{<:AbstractString}, index::Int)
    argument = args[index]
    argument == "--help" && return (:help, "", index)
    startswith(argument, "--") || throw(ArgumentError("unknown argument: $argument"))
    if occursin('=', argument)
        key, value = split(argument[3:end], '='; limit=2)
        isempty(value) && throw(ArgumentError("--$key requires a value"))
        return (Symbol(replace(key, '-' => '_')), value, index)
    end
    index < length(args) || throw(ArgumentError("$argument requires a value"))
    value = args[index + 1]
    startswith(value, "--") && throw(ArgumentError("$argument requires a value"))
    return (Symbol(replace(argument[3:end], '-' => '_')), value, index + 1)
end

"""Parse generic CLI arguments without loading a TOML configuration."""
function parse_cli(args::AbstractVector{<:AbstractString}=String[])
    allowed = Set([:config, :seed, :outer_reps, :inner_train, :inner_eval, :z_step,
                   :z_values, :av_values, :aq_values, :tau_values, :max_candidates,
                   :max_policies, :output_dir, :experiment_name, :randomization,
                   :crn_reference_z, :help])
    values = Dict{Symbol,String}()
    index = 1
    while index <= length(args)
        key, value, next_index = _parse_cli_value(args, index)
        key in allowed || throw(ArgumentError("unknown argument: --$(replace(String(key), '_' => '-'))"))
        key in keys(values) && throw(ArgumentError("duplicate argument: --$(replace(String(key), '_' => '-'))"))
        key === :help || (values[key] = value)
        key === :help && return (; help=true, values=values)
        index = next_index + 1
    end
    haskey(values, :z_step) && haskey(values, :z_values) &&
        throw(ArgumentError("--z-step and --z-values are mutually exclusive"))
    return (; help=false, values=values)
end

function _cli_int(overrides::Dict{Symbol,String}, key::Symbol, fallback, context::String)
    value = get(overrides, key, fallback)
    try
        return _positive_int(value isa AbstractString ? parse(Int, value) : value, context)
    catch error
        error isa ArgumentError || rethrow()
        throw(ArgumentError("$context must be a positive integer"))
    end
end

function _cli_seed(overrides::Dict{Symbol,String}, fallback)
    value = get(overrides, :seed, fallback)
    try
        return _as_int(value isa AbstractString ? parse(Int, value) : value, "seed")
    catch error
        error isa ArgumentError || rethrow()
        throw(ArgumentError("seed must be a valid Int"))
    end
end

function _resolved_coordinates(grid::AbstractDict, overrides::Dict{Symbol,String}, line_count::Int)
    if haskey(overrides, :z_step)
        step = try parse(Float64, overrides[:z_step]) catch; throw(ArgumentError("--z-step must be numeric")) end
        values = _step_coordinates(step)
        return [copy(values) for _ in 1:line_count], "step", step
    elseif haskey(overrides, :z_values)
        values = _parse_z_values(overrides[:z_values])
        length(values) == line_count || throw(ArgumentError("--z-values must contain one coordinate vector per line"))
        return [_deduplicate_coordinates(value) for value in values], "explicit", nothing
    end
    mode = _required(grid, "mode", "probing_grid")
    mode isa AbstractString || throw(ArgumentError("probing_grid.mode must be a string"))
    if mode == "step"
        step = _as_float(_required(grid, "step", "probing_grid"), "probing_grid.step")
        values = _step_coordinates(step)
        return [copy(values) for _ in 1:line_count], "step", step
    elseif mode == "explicit"
        return _explicit_coordinates(_required(grid, "coordinate_values", "probing_grid"), line_count), "explicit", nothing
    end
    throw(ArgumentError("unknown probing_grid.mode: $mode"))
end

"""Load TOML, apply CLI-overrides, and validate all experiment inputs before simulation."""
function load_experiment_config(config_path::AbstractString;
                                overrides::Dict{Symbol,String}=Dict{Symbol,String}(),
                                mode::Symbol=:generic)
    isfile(config_path) || throw(ArgumentError("configuration file does not exist: $config_path"))
    root = try TOML.parsefile(config_path) catch error; throw(ArgumentError("unable to parse TOML $config_path: $(sprint(showerror, error))")) end
    experiment = _table(root, "experiment")
    params = _model_parameters(root)
    policies_table = _table(root, "policies")
    grid_table = _table(root, "probing_grid")
    randomization_table = get(root, "randomization", Dict{String,Any}())
    randomization_table isa AbstractDict || throw(ArgumentError("TOML [randomization] section must be a table"))
    name = get(overrides, :experiment_name, _required(experiment, "name", "experiment"))
    name isa AbstractString && !isempty(name) || throw(ArgumentError("experiment.name must be a nonempty string"))
    seed = _cli_seed(overrides, _required(experiment, "seed", "experiment"))
    outer = _cli_int(overrides, :outer_reps, _required(experiment, "outer_reps", "experiment"), "outer_reps")
    train = _cli_int(overrides, :inner_train, _required(experiment, "inner_train", "experiment"), "inner_train")
    evaluation = _cli_int(overrides, :inner_eval, _required(experiment, "inner_eval", "experiment"), "inner_eval")
    output_dir = String(get(overrides, :output_dir, get(experiment, "output_dir", DEFAULT_OUTPUT_DIR)))
    isempty(output_dir) && throw(ArgumentError("output_dir must be nonempty"))
    av = haskey(overrides, :av_values) ? _parse_csv_floats(overrides[:av_values], "--av-values") : _float_vector(_required(policies_table, "av_values", "policies"), "policies.av_values")
    aq = haskey(overrides, :aq_values) ? _parse_csv_floats(overrides[:aq_values], "--aq-values") : _float_vector(_required(policies_table, "aq_values", "policies"), "policies.aq_values")
    tau = haskey(overrides, :tau_values) ? _parse_csv_floats(overrides[:tau_values], "--tau-values") : _float_vector(_required(policies_table, "tau_values", "policies"), "policies.tau_values")
    policy_max = _cli_int(overrides, :max_policies, _required(policies_table, "max_policies", "policies"), "max_policies")
    candidate_max = _cli_int(overrides, :max_candidates, _required(grid_table, "max_candidates", "probing_grid"), "max_candidates")
    coordinates, grid_mode, grid_step = _resolved_coordinates(grid_table, overrides, nlines(params))
    policies = build_policy_grid(av, aq, tau; max_policies=policy_max)
    grid = build_z_grid(coordinates, params; max_candidates=candidate_max)
    randomization_text = get(overrides, :randomization, get(randomization_table, "mode", "independent"))
    randomization_text isa AbstractString || throw(ArgumentError("randomization.mode must be a string"))
    randomization_mode = Symbol(randomization_text)
    randomization_mode in (:independent, :crn) || throw(ArgumentError("randomization must be independent or crn"))
    reference_text = get(overrides, :crn_reference_z, get(randomization_table, "reference_z", nothing))
    reference_z = if isnothing(reference_text)
        nothing
    elseif reference_text isa AbstractString
        Tuple(_parse_csv_floats(reference_text, "--crn-reference-z"))
    else
        Tuple(_float_vector(reference_text, "randomization.reference_z"))
    end
    isnothing(reference_z) || length(reference_z) == nlines(params) ||
        throw(ArgumentError("CRN reference z must contain one coordinate per line"))
    isnothing(reference_z) || all(value -> 0.0 <= value <= 1.0, reference_z) ||
        throw(ArgumentError("CRN reference z coordinates must lie in [0, 1]"))
    isnothing(reference_z) || reference_z in Tuple(Tuple(z) for z in grid) ||
        throw(ArgumentError("CRN reference z must exist exactly in the generated candidate set"))
    return GridExperimentConfig(String(name), mode, seed, outer, train, evaluation, output_dir,
                                params, Tuple(av), Tuple(aq), Tuple(tau), policy_max,
                                Tuple(policies), Tuple(Tuple(values) for values in coordinates),
                                Tuple(Tuple(z) for z in grid), candidate_max, grid_mode,
                                grid_step, randomization_mode, reference_z, abspath(config_path))
end

"""Delegate the resolved experiment to Stage 7 without reimplementing its estimator."""
function run_grid_experiment(config::GridExperimentConfig)::FiniteGridSearchResult
    result = finite_grid_search(config.seed, [collect(values) for values in config.z_coordinate_values],
                                config.params, collect(config.policies);
                                outer_reps=config.outer_reps, inner_train=config.inner_train,
                                inner_eval=config.inner_eval, max_candidates=config.candidate_max,
                                randomization_mode=config.randomization_mode)
    if config.randomization_mode === :crn
        reference = _reference_index(config, result)
        return FiniteGridSearchResult(result.candidates, result.candidate_seeds, result.evaluations,
                                      result.ranking, result.best_index, result.best_z,
                                      result.best_evaluation, result.original_to_unique, result.master_seed,
                                      result.outer_reps, result.inner_train, result.inner_eval, :crn, reference)
    end
    return result
end

function _path_within(path::AbstractString, directory::AbstractString)
    relative = relpath(path, directory)
    return relative == "." || (relative != ".." && !startswith(relative, "..$(Base.Filesystem.path_separator)"))
end

function _output_paths(output_dir::AbstractString)
    directory = abspath(output_dir)
    isfile(directory) && throw(ArgumentError("output_dir resolves to a regular file: $directory"))
    protected = (abspath(@__DIR__), abspath(joinpath(@__DIR__, "..", "src")),
                 abspath(joinpath(@__DIR__, "..", "test")))
    any(_path_within(directory, root) for root in protected) &&
        throw(ArgumentError("output_dir must not be a package source directory"))
    paths = (csv_path=normpath(joinpath(directory, "candidate_results.csv")),
             markdown_path=normpath(joinpath(directory, "experiment_report.md")),
             resolved_path=normpath(joinpath(directory, "resolved_configuration.toml")),
             outer_path=normpath(joinpath(directory, "outer_replications.csv")),
             paired_path=normpath(joinpath(directory, "paired_vs_reference.csv")))
    all(_path_within(path, directory) for path in values(paths)) ||
        throw(ArgumentError("output paths must remain inside output_dir"))
    return directory, paths
end

function _dominant(evaluation::ZEvaluationResult)
    frequency, index = findmax(evaluation.selected_policy_frequencies)
    return index, frequency
end

function _write_csv(path::AbstractString, result::FiniteGridSearchResult)
    headers = ["rank", "candidate_index", "candidate_seed"]
    append!(headers, ["z_$index" for index in eachindex(result.best_z)])
    append!(headers, ["gross_value", "probing_cost", "objective", "outer_standard_error",
                      "outer_reps", "inner_train", "inner_eval", "dominant_policy_index",
                      "dominant_policy_frequency"])
    append!(headers, ["policy_$(index)_frequency" for index in eachindex(result.best_evaluation.selected_policy_frequencies)])
    open(path, "w") do io
        println(io, join(headers, ","))
        for (rank, index) in enumerate(result.ranking)
            evaluation = result.evaluations[index]
            dominant_index, dominant_frequency = _dominant(evaluation)
            seed_text = result.randomization_mode === :crn ? "common_crn" : string(result.candidate_seeds[index])
            fields = String[string(rank), string(index), seed_text]
            append!(fields, [_format_number(value) for value in result.candidates[index]])
            append!(fields, [_format_number(evaluation.gross_operational_value), _format_number(evaluation.probing_cost),
                             _format_number(evaluation.objective), _format_number(evaluation.standard_error),
                             string(result.outer_reps), string(result.inner_train), string(result.inner_eval),
                             string(dominant_index), _format_number(dominant_frequency)])
            append!(fields, [_format_number(value) for value in evaluation.selected_policy_frequencies])
            println(io, join(fields, ","))
        end
    end
end

function _resolved_dict(config::GridExperimentConfig)
    lines = [Dict("value" => line.value, "prior_mean" => line.prior_mean,
                  "prior_variance" => line.prior_variance, "diagnostic_noise" => line.diagnostic_noise,
                  "max_diagnostic_time" => line.max_diagnostic_time,
                  "baseline_arrival_rate" => line.base_arrival_rate,
                  "arrival_sensitivity" => line.arrival_sensitivity,
                  "complexity_intercept" => line.complexity_intercept,
                  "complexity_sensitivity" => line.complexity_sensitivity,
                  "fast_simple_mean" => line.fast_service_means[1], "fast_complex_mean" => line.fast_service_means[2],
                  "standard_simple_mean" => line.standard_service_means[1], "standard_complex_mean" => line.standard_service_means[2])
             for line in config.params.lines]
    return Dict("experiment" => Dict("name" => config.name, "seed" => config.seed,
                                      "outer_reps" => config.outer_reps, "inner_train" => config.inner_train,
                                      "inner_eval" => config.inner_eval, "output_dir" => config.output_dir,
                                      "source_config" => config.source_config),
                "model" => Dict("horizon" => config.params.horizon, "revenue_scale" => config.params.revenue_scale,
                                "fast_routing_cost" => config.params.fast_routing_cost,
                                "delay_penalty" => config.params.delay_penalty,
                                "probing_costs" => collect(config.params.probe_cost)),
                "lines" => lines,
                "policies" => Dict("av_values" => collect(config.av_values), "aq_values" => collect(config.aq_values),
                                   "tau_values" => collect(config.tau_values), "max_policies" => config.policy_max),
                "probing_grid" => Dict("grid_mode" => config.grid_mode, "step" => config.grid_step,
                                        "coordinate_values" => [collect(values) for values in config.z_coordinate_values],
                                        "max_candidates" => config.candidate_max),
                "randomization" => Dict("mode" => String(config.randomization_mode),
                                          "master_seed" => config.seed,
                                          "reference_z" => isnothing(config.crn_reference_z) ? nothing : collect(config.crn_reference_z)))
end

function _write_resolved(path::AbstractString, config::GridExperimentConfig)
    data = _resolved_dict(config)
    if isnothing(data["probing_grid"]["step"])
        delete!(data["probing_grid"], "step")
    end
    if isnothing(data["randomization"]["reference_z"])
        delete!(data["randomization"], "reference_z")
    end
    open(path, "w") do io
        TOML.print(io, data; sorted=true)
    end
end

function _write_report(path::AbstractString, config::GridExperimentConfig, result::FiniteGridSearchResult)
    objectives = [evaluation.objective for evaluation in result.evaluations]
    ses = [evaluation.standard_error for evaluation in result.evaluations]
    best = result.best_evaluation
    open(path, "w") do io
        println(io, "# Grid Experiment Report\n")
        println(io, "## Resolved configuration\n")
        println(io, "- Name: `$(config.name)`")
        println(io, "- Source configuration: `$(config.source_config)`")
        println(io, "- Seed: `$(config.seed)`; lines: $(nlines(config.params)); horizon: $(_format_number(config.params.horizon))")
        println(io, "- Economics: revenue=$(_format_number(config.params.revenue_scale)), FAST cost=$(_format_number(config.params.fast_routing_cost)), delay=$(_format_number(config.params.delay_penalty))")
        println(io, "- Probing costs: $(_format_values(config.params.probe_cost))")
        println(io, "- Policy grids: av=$(_format_values(config.av_values)), aq=$(_format_values(config.aq_values)), tau=$(_format_values(config.tau_values)); policies=$(length(config.policies))")
        println(io, "- Probing coordinates: $([collect(values) for values in config.z_coordinate_values]); candidates=$(length(result.candidates))")
        println(io, "- Samples: outer=$(result.outer_reps), train=$(result.inner_train), eval=$(result.inner_eval)\n")
        println(io, "## Estimator\n")
        println(io, "`Jhat(z) = (1/L) sum_l Vhat_l(z) - alpha' z`. Finite inner training selects a policy on common posterior-predictive paths; a separate inner bank evaluates it.")
        if result.randomization_mode === :crn
            println(io, "Common random numbers are used to create positive correlation between candidate estimators. Their primary purpose is to reduce the variance of pairwise objective differences. The marginal outer standard error reported for an individual candidate need not decrease.\n")
        else
            println(io, "Candidate streams are independent and no CRN is used.\n")
        end
        println(io, "## Ranked candidates\n")
        println(io, "| Rank | z | Gross | Cost | Objective | Outer SE | Dominant policy | Frequency |")
        println(io, "| ---: | :-- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for (rank, index) in enumerate(result.ranking[1:min(20, length(result.ranking))])
            evaluation = result.evaluations[index]
            dominant_index, dominant_frequency = _dominant(evaluation)
            println(io, "| $rank | $(_format_z(result.candidates[index])) | $(_format_number(evaluation.gross_operational_value)) | $(_format_number(evaluation.probing_cost)) | $(_format_number(evaluation.objective)) | $(_format_number(evaluation.standard_error)) | $dominant_index | $(_format_number(dominant_frequency)) |")
        end
        println(io, "\n## Best candidate\n")
        seed_label = result.randomization_mode === :crn ? "common CRN master seed $(result.master_seed)" : "candidate seed $(result.candidate_seeds[result.best_index])"
        println(io, "- z: `$(_format_z(result.best_z))`; $seed_label")
        println(io, "- Gross: $(_format_number(best.gross_operational_value)); cost: $(_format_number(best.probing_cost)); objective: $(_format_number(best.objective)); outer SE: $(_format_number(best.standard_error))")
        println(io, "- Policy counts: $(collect(best.selected_policy_counts)); frequencies: $(_format_values(best.selected_policy_frequencies))\n")
        println(io, "## Diagnostics\n")
        println(io, "- Complete Cartesian grid: $(length(result.candidates)) candidates")
        println(io, "- Objective min/max: $(_format_number(minimum(objectives))) / $(_format_number(maximum(objectives)))")
        println(io, "- Outer SE min/mean/median/max: $(_format_number(minimum(ses))) / $(_format_number(sum(ses) / length(ses))) / $(_format_number(median(ses))) / $(_format_number(maximum(ses)))\n")
        println(io, "## Interpretation limits\n")
        stream_note = result.randomization_mode === :crn ?
            "CRN aligns candidate base streams for paired comparisons, but policy-selection discontinuities can weaken that coupling." :
            "independent streams are used across z."
        println(io, "Parameters come from the supplied configuration. The policy family and probing grid are finite; estimates contain Monte Carlo noise; $stream_note Ranking is not a significance test; and no continuous optimum is claimed.")
    end
end

function _rank_positions(result::FiniteGridSearchResult)
    positions = zeros(Int, length(result.evaluations))
    for (rank, index) in enumerate(result.ranking); positions[index] = rank; end
    return positions
end
function _reference_index(config::GridExperimentConfig, result::FiniteGridSearchResult)
    requested = isnothing(config.crn_reference_z) ? result.candidates[1] : config.crn_reference_z
    index = findfirst(==(requested), result.candidates)
    isnothing(index) && throw(ArgumentError("CRN reference z must exist exactly in the candidate set"))
    return index
end
function _write_crn_outputs(paths, config, result)
    reference = _reference_index(config, result); ranks = _rank_positions(result)
    open(paths.outer_path, "w") do io
        headers = ["candidate_index", "rank"]; append!(headers, ["z_$i" for i in eachindex(result.best_z)])
        append!(headers, ["outer_replication", "outer_evaluation_value", "selected_policy_index", "selected_training_value", "inner_evaluation_standard_error", "probing_cost", "net_outer_value"]); println(io, join(headers, ","))
        for index in eachindex(result.evaluations), replication in 1:result.outer_reps
            e = result.evaluations[index]
            fields = [string(index), string(ranks[index])]; append!(fields, _format_number.(result.candidates[index]))
            append!(fields, [string(replication), _format_number(e.outer_values[replication]), string(e.selected_policy_indices[replication]), _format_number(e.selected_training_values[replication]), _format_number(e.inner_evaluation_standard_errors[replication]), _format_number(e.probing_cost), _format_number(e.outer_values[replication] - e.probing_cost)])
            println(io, join(fields, ","))
        end
    end
    open(paths.paired_path, "w") do io
        headers = ["candidate_index"]; append!(headers, ["z_$i" for i in eachindex(result.best_z)]); push!(headers, "reference_candidate_index"); append!(headers, ["reference_z_$i" for i in eachindex(result.best_z)]); append!(headers, ["objective", "reference_objective", "objective_difference", "paired_standard_error", "unpaired_standard_error_approximation", "sample_correlation", "outer_reps"]); println(io, join(headers, ","))
        reference_evaluation = result.evaluations[reference]
        for index in eachindex(result.evaluations)
            comparison = paired_comparison(result, index, reference); evaluation = result.evaluations[index]
            fields = [string(index)]; append!(fields, _format_number.(result.candidates[index])); push!(fields, string(reference)); append!(fields, _format_number.(result.candidates[reference])); append!(fields, [_format_number(evaluation.objective), _format_number(reference_evaluation.objective), _format_number(comparison.objective_difference), _format_number(comparison.paired_standard_error), _format_number(sqrt(evaluation.standard_error^2 + reference_evaluation.standard_error^2)), _format_number(comparison.sample_correlation), string(result.outer_reps)]); println(io, join(fields, ","))
        end
    end
end

"""Write deterministic CSV, Markdown, and fully resolved TOML outputs."""
function write_experiment_outputs(config::GridExperimentConfig, result::FiniteGridSearchResult)
    directory, paths = _output_paths(config.output_dir)
    mkpath(directory)
    _write_csv(paths.csv_path, result)
    _write_report(paths.markdown_path, config, result)
    _write_resolved(paths.resolved_path, config)
    result.randomization_mode === :crn && _write_crn_outputs(paths, config, result)
    return (; output_dir=directory, paths...)
end

function execute_grid_experiment(config::GridExperimentConfig)
    result = run_grid_experiment(config)
    paths = write_experiment_outputs(config, result)
    return (; configuration=config, result, paths)
end

function print_summary(execution)
    config, result, paths = execution.configuration, execution.result, execution.paths
    best = result.best_evaluation
    println("Grid experiment: $(config.name)")
    println("seed=$(config.seed), lines=$(nlines(config.params)), policies=$(length(config.policies)), candidates=$(length(result.candidates))")
    println("samples=(outer=$(result.outer_reps), train=$(result.inner_train), eval=$(result.inner_eval))")
    println("randomization=$(result.randomization_mode)")
    println("best_z=$(_format_z(result.best_z)), gross=$(_format_number(best.gross_operational_value)), cost=$(_format_number(best.probing_cost)), objective=$(_format_number(best.objective)), outer_se=$(_format_number(best.standard_error))")
    println("candidate_results_csv=$(paths.csv_path)")
    println("experiment_report=$(paths.markdown_path)")
    println("resolved_configuration=$(paths.resolved_path)")
    result.randomization_mode === :crn && println("outer_replications=$(paths.outer_path)\npaired_vs_reference=$(paths.paired_path)")
end

function help_text()
    return """
Generic assembly-line grid experiment runner

Usage: julia --project=assembly_line_model assembly_line_model/examples/run_grid_experiment.jl --config=PATH [overrides]

Required: --config=PATH (unless --help)
Overrides: --seed, --outer-reps, --inner-train, --inner-eval, --z-step,
  --z-values, --av-values, --aq-values, --tau-values, --max-candidates,
  --max-policies, --output-dir, --experiment-name, --randomization, --crn-reference-z.
CLI overrides TOML. --z-step=0.05 creates {0,0.05,...,1}; --z-values uses
semicolon-separated line vectors, e.g. "0,0.5,1;0,0.25,1".
Outputs: candidate_results.csv, experiment_report.md, resolved_configuration.toml.
Smoke: --config=assembly_line_model/configs/reference_experiment.toml --z-step=0.5
Dense: --config=assembly_line_model/configs/reference_experiment.toml --z-step=0.05 --outer-reps=100 --inner-train=300 --inner-eval=1000
"""
end

end # module ExperimentRunner
