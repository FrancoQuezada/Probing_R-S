include(joinpath(@__DIR__, "experiment_runner.jl"))
using .ExperimentRunner

function main(args::AbstractVector{<:AbstractString}=ARGS)
    parsed = parse_cli(args)
    if parsed.help
        print(help_text())
        return nothing
    end
    haskey(parsed.values, :config) || throw(ArgumentError("--config is required; run with --help for usage"))
    config_path = parsed.values[:config]
    overrides = copy(parsed.values)
    delete!(overrides, :config)
    config = load_experiment_config(config_path; overrides=overrides)
    println("Resolved configuration: name=$(config.name), lines=$(length(config.params.lines)), policies=$(length(config.policies)), candidates=$(length(config.z_grid)), samples=($(config.outer_reps),$(config.inner_train),$(config.inner_eval))")
    execution = execute_grid_experiment(config)
    print_summary(execution)
    return execution
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
