using Printf
using Plots
using Statistics

if !isdefined(@__MODULE__, :AssemblyLineTwoStage)
    include(joinpath(@__DIR__, "assembly_line_two_stage.jl"))
end
using .AssemblyLineTwoStage

function assembly_parse_int(name::AbstractString, default::Int; minimum::Int=1)
    raw = get(ENV, name, string(default)); value = tryparse(Int, raw)
    value === nothing && throw(ArgumentError("$name must be an integer; got '$raw'"))
    value >= minimum || throw(ArgumentError("$name must be at least $minimum; got '$raw'"))
    return value
end

function assembly_parse_vector(name::AbstractString, default::AbstractVector{<:Real}, n::Int)
    raw = get(ENV, name, join(default, ",")); fields = split(raw, ',')
    length(fields) == n || throw(ArgumentError("$name must contain $n comma-separated values; got '$raw'"))
    values = Float64[]
    for field in fields
        value = tryparse(Float64, strip(field))
        value === nothing && throw(ArgumentError("$name contains an invalid value: '$field'"))
        isfinite(value) || throw(ArgumentError("$name values must be finite"))
        push!(values, value)
    end
    return values
end

function assembly_parse_active_lines(raw::AbstractString, nclass::Int)
    fields = split(raw, ','); length(fields) == 2 || throw(ArgumentError("ACTIVE_LINES must contain exactly two line IDs"))
    lines = [tryparse(Int, strip(field)) for field in fields]
    any(isnothing, lines) && throw(ArgumentError("ACTIVE_LINES must contain integers"))
    active = Int[lines[1], lines[2]]
    all(1 .<= active .<= nclass) || throw(ArgumentError("ACTIVE_LINES must lie in 1:$nclass"))
    active[1] != active[2] || throw(ArgumentError("ACTIVE_LINES must be distinct"))
    return active
end

assembly_file_number(x::Real) = replace(replace(string(x), "-" => "m"), "." => "p")
assembly_stem(experiment::ExperimentParams) = "outer_$(experiment.outer_reps)_train_$(experiment.inner_train)_eval_$(experiment.inner_eval)_jobs_$(experiment.jobs_per_path)_seed_$(experiment.outer_seed)_alpha_$(join(assembly_file_number.(experiment.alpha), "_"))"

function assembly_write_grid_csv(filename::AbstractString, rows, nclass::Int)
    open(filename, "w") do io
        header = ["z_active_1", "z_active_2", "z_full", "diagnostic_times", "gross_adaptive_value", "probing_cost", "net_objective", "objective_se", "optimistic_training_value", "fixed_policy_evaluation_value", "information_value", "distinct_selected_policies", "most_frequent_policy_id", "most_frequent_policy_frequency", "average_training_gap", "average_fast_probability", "runtime_seconds"]
        append!(header, ["mean_posterior_variance_$c" for c in 1:nclass])
        append!(header, ["mean_posterior_complexity_$c" for c in 1:nclass])
        println(io, join(header, ','))
        for row in rows
            base = [row.z_active_1, row.z_active_2, join(row.z, ';'), join(row.diagnostic_times, ';'), row.gross_value,
                    row.probing_cost, row.objective, row.standard_error, row.optimistic_training_value,
                    row.fixed_policy_value, row.information_value, row.distinct_selected_policies,
                    row.most_frequent_policy_id, row.most_frequent_policy_frequency, row.average_training_gap,
                    row.average_fast_probability, row.runtime_seconds]
            println(io, join(string.(vcat(base, row.mean_posterior_variance, row.mean_posterior_complexity)), ','))
        end
    end
end

function assembly_write_context_csv(filename::AbstractString, estimate::TwoStageEstimate, policies::Vector{Policy})
    open(filename, "w") do io
        n = length(estimate.z)
        header = ["outer_context_id"]
        append!(header, ["eta_$c" for c in 1:n]); append!(header, ["posterior_mean_$c" for c in 1:n])
        append!(header, ["posterior_sd_$c" for c in 1:n]); append!(header, ["posterior_complexity_$c" for c in 1:n])
        append!(header, ["selected_policy_id", "a_value", "a_queue", "a_signal", "threshold", "best_training_mean", "second_training_mean", "training_gap", "evaluation_mean", "evaluation_se", "fast_probability"])
        println(io, join(header, ','))
        for context in estimate.contexts
            policy = policies[context.selected_policy_id]
            values = Any[context.outer_id]
            append!(values, context.eta); append!(values, context.posterior_mean)
            append!(values, sqrt.(context.posterior_variance)); append!(values, context.posterior_complexity)
            append!(values, (context.selected_policy_id, policy.a_value, policy.a_queue, policy.a_signal, policy.threshold,
                             context.best_training_mean, context.second_training_mean, context.training_gap,
                             context.evaluation_mean, context.evaluation_se, context.fast_probability))
            println(io, join(string.(values), ','))
        end
    end
end

function assembly_heatmap(grid, values, title_text, colorbar_text, filename)
    figure = heatmap(grid, grid, values; xlabel="diagnostic effort, active line 1", ylabel="diagnostic effort, active line 2",
                     title=title_text, colorbar_title=colorbar_text, size=(900, 720))
    savefig(figure, filename)
end

"""Exhaustive CRN grid search for the classical multi-job two-stage experiment."""
function run_assembly_line_grid(; grid_n::Int=5, params::AssemblyLineParams=AssemblyLineParams(),
                                experiment::ExperimentParams=ExperimentParams(), active_lines::Vector{Int}=[3, 4],
                                fixed_z::Vector{Float64}=zeros(4), output_dir::AbstractString="results")
    grid_n >= 2 || throw(ArgumentError("grid_n must be at least 2"))
    validate_params(params, experiment); length(fixed_z) == params.nclass || throw(ArgumentError("FIXED_Z has wrong length"))
    all(x -> 0.0 <= x <= 1.0, fixed_z) || throw(ArgumentError("FIXED_Z must lie in [0,1]"))
    length(active_lines) == 2 && active_lines[1] != active_lines[2] && all(1 .<= active_lines .<= params.nclass) ||
        throw(ArgumentError("active_lines must be two distinct valid line IDs"))
    mkpath(output_dir)
    policies = build_policy_grid(); noise = generate_noise_banks(params, experiment)
    fixed = estimate_fixed_policy_baseline(params, experiment, policies, noise)
    grid = collect(range(0.0, 1.0; length=grid_n)); rows = NamedTuple[]
    objective = Matrix{Float64}(undef, grid_n, grid_n); gross = similar(objective); information = similar(objective)
    uncertainty = similar(objective); optimism_gap = similar(objective); estimates = Matrix{TwoStageEstimate}(undef, grid_n, grid_n)
    println("Assembly-line classical two-stage grid search with $(length(policies)) policies")
    for (ix, first_effort) in enumerate(grid)
        for (iy, second_effort) in enumerate(grid)
            z = copy(fixed_z); z[active_lines[1]] = first_effort; z[active_lines[2]] = second_effort
            started = time_ns(); estimate = estimate_two_stage(z, params, experiment, policies, noise; fixed_policy=fixed)
            elapsed = (time_ns() - started) / 1e9
            estimates[iy, ix] = estimate; objective[iy, ix] = estimate.objective; gross[iy, ix] = estimate.gross_value
            information[iy, ix] = estimate.information_value; uncertainty[iy, ix] = mean(estimate.mean_posterior_variance[active_lines])
            optimism_gap[iy, ix] = estimate.optimistic_training_value - estimate.gross_value
            push!(rows, (z_active_1=first_effort, z_active_2=second_effort, z=estimate.z, diagnostic_times=estimate.diagnostic_times,
                         gross_value=estimate.gross_value, probing_cost=estimate.probing_cost, objective=estimate.objective,
                         standard_error=estimate.standard_error, optimistic_training_value=estimate.optimistic_training_value,
                         fixed_policy_value=fixed.evaluation_mean, information_value=estimate.information_value,
                         mean_posterior_variance=estimate.mean_posterior_variance, mean_posterior_complexity=estimate.mean_posterior_complexity,
                         distinct_selected_policies=estimate.distinct_selected_policies, most_frequent_policy_id=estimate.most_frequent_policy_id,
                         most_frequent_policy_frequency=estimate.most_frequent_policy_frequency, average_training_gap=estimate.average_training_gap,
                         average_fast_probability=estimate.average_fast_probability, runtime_seconds=elapsed))
        end
        @printf("  completed active-line grid column %d/%d\n", ix, grid_n)
    end
    stem = assembly_stem(experiment); csv = joinpath(output_dir, "assembly_line_grid_$(stem).csv")
    objective_png = joinpath(output_dir, "assembly_line_objective_$(stem).png")
    gross_png = joinpath(output_dir, "assembly_line_gross_$(stem).png")
    info_png = joinpath(output_dir, "assembly_line_information_$(stem).png")
    uncertainty_png = joinpath(output_dir, "assembly_line_uncertainty_$(stem).png")
    optimism_png = joinpath(output_dir, "assembly_line_optimism_gap_$(stem).png")
    assembly_write_grid_csv(csv, rows, params.nclass)
    assembly_heatmap(grid, objective, "Net classical two-stage objective", "J(z)", objective_png)
    assembly_heatmap(grid, gross, "Gross adaptive value", "gross value", gross_png)
    assembly_heatmap(grid, information, "Estimated pre-shift information value", "adaptive - fixed", info_png)
    assembly_heatmap(grid, uncertainty, "Mean posterior variance: active lines", "variance", uncertainty_png)
    assembly_heatmap(grid, optimism_gap, "Optimistic training minus evaluation", "gap", optimism_png)
    iy, ix = Tuple(argmax(objective)); best = estimates[iy, ix]
    context_csv = joinpath(output_dir, "assembly_line_best_contexts_$(stem).csv")
    assembly_write_context_csv(context_csv, best, policies)
    @printf("\nBest z=%s, diagnostic times=%s\n", string(best.z), string(best.diagnostic_times))
    @printf("gross=%.6f cost=%.6f objective=%.6f SE=%.6f fixed=%.6f information=%.6f\n", best.gross_value, best.probing_cost, best.objective, best.standard_error, best.fixed_policy.evaluation_mean, best.information_value)
    @printf("most frequent policy=%d (%.1f%%), distinct policies=%d\n", best.most_frequent_policy_id, 100best.most_frequent_policy_frequency, best.distinct_selected_policies)
    return (best=best, rows=rows, fixed_policy=fixed, output_files=(csv, context_csv, objective_png, gross_png, info_png, uncertainty_png, optimism_png))
end

function assembly_main()
    defaults = AssemblyLineParams(); n = defaults.nclass
    params = AssemblyLineParams(tau_theta=assembly_parse_vector("TAU_THETA", defaults.tau_theta, n),
        sigma_sensor=assembly_parse_vector("SIGMA_SENSOR", defaults.sigma_sensor, n),
        condition_sensitivity=assembly_parse_vector("CONDITION_SENSITIVITY", defaults.condition_sensitivity, n),
        t_max=assembly_parse_vector("T_MAX", defaults.t_max, n))
    experiment = ExperimentParams(outer_reps=assembly_parse_int("OUTER_REPS", 12), inner_train=assembly_parse_int("INNER_TRAIN", 25),
        inner_eval=assembly_parse_int("INNER_EVAL", 50), jobs_per_path=assembly_parse_int("JOBS_PER_PATH", 45),
        alpha=assembly_parse_vector("ALPHA", zeros(n), n), outer_seed=assembly_parse_int("OUTER_SEED", 20260724; minimum=0),
        train_seed=assembly_parse_int("TRAIN_SEED", 20261724; minimum=0), eval_seed=assembly_parse_int("EVAL_SEED", 20262724; minimum=0),
        baseline_train_seed=assembly_parse_int("BASELINE_TRAIN_SEED", 20263724; minimum=0), baseline_eval_seed=assembly_parse_int("BASELINE_EVAL_SEED", 20264724; minimum=0))
    active = assembly_parse_active_lines(get(ENV, "ACTIVE_LINES", "3,4"), n)
    fixed_z = assembly_parse_vector("FIXED_Z", zeros(n), n)
    run_assembly_line_grid(grid_n=assembly_parse_int("GRID_N", 5; minimum=2), params=params, experiment=experiment,
                           active_lines=active, fixed_z=fixed_z, output_dir=get(ENV, "OUTPUT_DIR", "results"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    assembly_main()
end
