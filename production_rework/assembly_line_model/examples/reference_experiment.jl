module ReferenceExperiment

include(joinpath(@__DIR__, "experiment_runner.jl"))
using .ExperimentRunner

export reference_configuration,
       parse_cli,
       run_reference_experiment,
       write_experiment_outputs,
       execute_reference_experiment,
       main

const DEFAULT_SEED = 20_260_726
const REFERENCE_CONFIG = normpath(joinpath(@__DIR__, "..", "configs", "reference_experiment.toml"))

function _mode_sizes(mode::Symbol)
    mode === :smoke && return nothing
    mode === :reference && return (outer_reps=30, inner_train=100, inner_eval=200)
    throw(ArgumentError("unknown experiment mode: $mode (expected :smoke or :reference)"))
end

"""Load the historical reference setup through the shared TOML runner layer."""
function reference_configuration(mode::Symbol=:smoke;
                                 master_seed::Integer=DEFAULT_SEED,
                                 output_dir::AbstractString="assembly_line_model/experiment_outputs/reference_smoke",
                                 sample_sizes=nothing)
    defaults = _mode_sizes(mode)
    sizes = isnothing(sample_sizes) ? defaults : sample_sizes
    overrides = Dict{Symbol,String}(:seed => string(master_seed), :output_dir => String(output_dir))
    if !isnothing(sizes)
        for name in (:outer_reps, :inner_train, :inner_eval)
            hasproperty(sizes, name) || throw(ArgumentError("sample_sizes must include $name"))
            value = getproperty(sizes, name)
            value isa Integer && value > 0 || throw(ArgumentError("$name must be a positive integer"))
            overrides[name] = string(value)
        end
    end
    return load_experiment_config(REFERENCE_CONFIG; overrides=overrides, mode=mode)
end

"""Parse the legacy reference CLI; the generic runner owns all execution logic."""
function parse_cli(args::AbstractVector{<:AbstractString}=String[])
    mode, seed, output_dir = :smoke, DEFAULT_SEED,
                             "assembly_line_model/experiment_outputs/reference_smoke"
    seen = Set{Symbol}()
    for argument in args
        key, value = if startswith(argument, "--mode=")
            (:mode, argument[8:end])
        elseif startswith(argument, "--seed=")
            (:seed, argument[8:end])
        elseif startswith(argument, "--output-dir=")
            (:output_dir, argument[14:end])
        else
            throw(ArgumentError("unknown argument: $argument"))
        end
        key in seen && throw(ArgumentError("duplicate argument: $argument"))
        push!(seen, key)
        if key === :mode
            value == "smoke" || value == "reference" || throw(ArgumentError("invalid mode: $value"))
            mode = Symbol(value)
        elseif key === :seed
            try
                seed = parse(Int, value)
            catch error
                error isa ArgumentError || rethrow()
                throw(ArgumentError("seed must be a valid Int: $value"))
            end
        else
            isempty(value) && throw(ArgumentError("output directory must be nonempty"))
            output_dir = value
        end
    end
    return (; mode, master_seed=seed, output_dir)
end

run_reference_experiment(configuration) = run_grid_experiment(configuration)
write_experiment_outputs(configuration, result) = ExperimentRunner.write_experiment_outputs(configuration, result)

function execute_reference_experiment(configuration)
    result = run_reference_experiment(configuration)
    paths = write_experiment_outputs(configuration, result)
    return (; configuration, result, paths)
end

function main(args::AbstractVector{<:AbstractString}=ARGS)
    options = parse_cli(args)
    configuration = reference_configuration(options.mode; master_seed=options.master_seed,
                                            output_dir=options.output_dir)
    execution = execute_reference_experiment(configuration)
    print_summary(execution)
    return execution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module ReferenceExperiment
