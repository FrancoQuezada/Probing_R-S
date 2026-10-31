using Printf
using Plots

if !isdefined(@__MODULE__, :SingleJobTwoStage)
    include(joinpath(@__DIR__, "single_job_two_stage.jl"))
end
using .SingleJobTwoStage

function _parse_positive_int_env(name::AbstractString, default::Int)
    raw = get(ENV, name, string(default))
    value = tryparse(Int, raw)
    value === nothing && throw(ArgumentError("$name must be a positive integer; got '$raw'"))
    value > 0 || throw(ArgumentError("$name must be a positive integer; got '$raw'"))
    return value
end

function _parse_alpha_env(params::ModelParams)
    raw = get(ENV, "ALPHA", join(zeros(params.nclass), ","))
    pieces = split(raw, ',')
    length(pieces) == params.nclass ||
        throw(ArgumentError("ALPHA must contain $(params.nclass) comma-separated values; got '$raw'"))
    alpha = Float64[]
    for piece in pieces
        value = tryparse(Float64, strip(piece))
        value === nothing && throw(ArgumentError("ALPHA contains an invalid number: '$piece'"))
        isfinite(value) || throw(ArgumentError("ALPHA contains a non-finite number: '$piece'"))
        push!(alpha, value)
    end
    return alpha
end

_file_number(x::Real) = replace(replace(string(x), "-" => "m"), "." => "p")

function _file_stem(alpha, reps::Integer, seed::Integer)
    "alpha_$(join(_file_number.(alpha), "_"))_reps_$(reps)_seed_$(seed)"
end

function _write_grid_csv(filename::AbstractString, rows)
    open(filename, "w") do io
        println(io, "z3,z4,gross_value,probing_cost,objective,standard_error,positive_signal_probability,unconditional_fast_probability,fast_probability_after_s0,fast_probability_after_s1,mean_information_value")
        for row in rows
            @printf(io, "%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                    row.z3, row.z4, row.gross_value, row.probing_cost, row.objective,
                    row.standard_error, row.positive_signal_probability,
                    row.unconditional_fast_probability, row.fast_probability_after_s0,
                    row.fast_probability_after_s1, row.mean_information_value)
        end
    end
end

function _save_heatmap(grid, values, title_text::AbstractString, colorbar_text::AbstractString,
                       filename::AbstractString)
    plot_object = heatmap(grid, grid, values;
                          xlabel="z₃ (probing effort, class 3)",
                          ylabel="z₄ (probing effort, class 4)",
                          title=title_text, colorbar_title=colorbar_text,
                          size=(900, 720))
    savefig(plot_object, filename)
    return filename
end

"""Run the direct CRN grid experiment for efforts `(z3, z4)` with `z1=z2=0`."""
function run_grid_experiment(; grid_n::Integer=41, reps::Integer=50_000, seed::Integer=20260720,
                             params::ModelParams=ModelParams(), alpha::AbstractVector{<:Real}=zeros(params.nclass),
                             output_dir::AbstractString="results")
    grid_n >= 2 || throw(ArgumentError("grid_n must be at least 2"))
    reps > 0 || throw(ArgumentError("reps must be positive"))
    validate_params(params)
    length(alpha) == params.nclass || throw(ArgumentError("alpha must have length $(params.nclass)"))
    all(isfinite, alpha) || throw(ArgumentError("alpha entries must be finite"))
    mkpath(output_dir)

    noise_set = generate_noise_set(seed, reps)
    grid = collect(range(0.0, 1.0; length=grid_n))
    objective = Matrix{Float64}(undef, grid_n, grid_n)
    gross = Matrix{Float64}(undef, grid_n, grid_n)
    fast_probability = Matrix{Float64}(undef, grid_n, grid_n)
    information_value = Matrix{Float64}(undef, grid_n, grid_n)
    rows = NamedTuple[]

    println("Direct single-job two-stage grid search")
    @printf("grid=%d x %d, reps=%d, seed=%d, alpha=[%s]\n", grid_n, grid_n, reps, seed,
            join(string.(alpha), ", "))
    for (ix, z3) in enumerate(grid)
        for (iy, z4) in enumerate(grid)
            z = [0.0, 0.0, z3, z4]
            result = estimate_objective_given_noise(z, noise_set, alpha, params)
            objective[iy, ix] = result.objective
            gross[iy, ix] = result.gross_value
            fast_probability[iy, ix] = result.unconditional_probability_fast
            information_value[iy, ix] = result.mean_information_value
            push!(rows, (z3=z3, z4=z4, gross_value=result.gross_value,
                         probing_cost=result.probing_cost, objective=result.objective,
                         standard_error=result.standard_error,
                         positive_signal_probability=result.expected_positive_signal_probability,
                         unconditional_fast_probability=result.unconditional_probability_fast,
                         fast_probability_after_s0=result.probability_fast_after_s0,
                         fast_probability_after_s1=result.probability_fast_after_s1,
                         mean_information_value=result.mean_information_value))
        end
        @printf("  completed z₃ grid column %d/%d\n", ix, grid_n)
    end

    stem = _file_stem(alpha, reps, seed)
    csv_file = joinpath(output_dir, "single_job_two_stage_grid_$(stem).csv")
    objective_file = joinpath(output_dir, "single_job_two_stage_objective_$(stem).png")
    gross_file = joinpath(output_dir, "single_job_two_stage_gross_value_$(stem).png")
    fast_file = joinpath(output_dir, "single_job_two_stage_fast_probability_$(stem).png")
    information_file = joinpath(output_dir, "single_job_two_stage_information_value_$(stem).png")
    _write_grid_csv(csv_file, rows)
    _save_heatmap(grid, objective, "Net objective: direct two-stage action", "J(z)", objective_file)
    _save_heatmap(grid, gross, "Gross expected value", "gross value", gross_file)
    _save_heatmap(grid, fast_probability, "Unconditional FAST-routing probability", "P(FAST)", fast_file)
    _save_heatmap(grid, information_value, "Mean value of the binary signal", "information value", information_file)

    index = argmax(objective)
    iy, ix = Tuple(index)
    best = rows[(ix - 1) * grid_n + iy]
    println("\nFinal summary")
    @printf("best grid point: z=(%.4f, %.4f)\n", best.z3, best.z4)
    @printf("gross value=%.6f, probing cost=%.6f, net objective=%.6f, SE=%.6f\n",
            best.gross_value, best.probing_cost, best.objective, best.standard_error)
    @printf("P(FAST)=%.6f, P(FAST | S=0)=%.6f, P(FAST | S=1)=%.6f\n",
            best.unconditional_fast_probability, best.fast_probability_after_s0,
            best.fast_probability_after_s1)
    @printf("P(S=1)=%.6f, mean information value=%.6f\n",
            best.positive_signal_probability, best.mean_information_value)
    println("outputs:")
    foreach(file -> println("  $file"), (csv_file, objective_file, gross_file, fast_file, information_file))

    return (best=best, rows=rows, grid=grid, objective=objective, gross=gross,
            fast_probability=fast_probability, information_value=information_value,
            output_files=(csv_file, objective_file, gross_file, fast_file, information_file))
end

function main()
    params = ModelParams()
    grid_n = _parse_positive_int_env("GRID_N", 41)
    reps = _parse_positive_int_env("REPS", 50_000)
    seed = _parse_positive_int_env("SEED", 20260720)
    alpha = _parse_alpha_env(params)
    output_dir = get(ENV, "OUTPUT_DIR", "results")
    isempty(output_dir) && throw(ArgumentError("OUTPUT_DIR must not be empty"))
    run_grid_experiment(grid_n=grid_n, reps=reps, seed=seed, alpha=alpha,
                        output_dir=output_dir, params=params)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
