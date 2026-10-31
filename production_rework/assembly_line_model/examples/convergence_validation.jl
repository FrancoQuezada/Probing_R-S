module ConvergenceValidation

using AssemblyLineModel
using Printf
using Random

include(joinpath(@__DIR__, "reference_experiment.jl"))
using .ReferenceExperiment

export convergence_plan, run_convergence_validation

const DEFAULT_OUTPUT = normpath(joinpath(@__DIR__, "..", "validation_outputs", "convergence_results.csv"))

"""A structured, non-Cartesian convergence plan for three fixed probing points."""
function convergence_plan()
    zero, smoke_best, full = [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]
    plan = NamedTuple[]
    for (stream_key, z) in enumerate((zero, smoke_best, full)), outer_reps in (10, 30, 100)
        push!(plan, (; study="outer", z, stream_key, outer_reps, inner_train=50, inner_eval=100))
    end
    for inner_train in (50, 100, 300)
        push!(plan, (; study="inner_train", z=smoke_best, stream_key=2, outer_reps=30, inner_train,
                     inner_eval=100))
    end
    for inner_eval in (100, 300, 1_000)
        push!(plan, (; study="inner_eval", z=smoke_best, stream_key=2, outer_reps=30, inner_train=100,
                     inner_eval))
    end
    return plan
end

_number(value::Real) = @sprintf("%.17g", Float64(value))
_frequencies(values) = join((_number(value) for value in values), ";")

function _dominant(result::ZEvaluationResult)
    frequency, index = findmax(result.selected_policy_frequencies)
    return index, frequency
end

function _z_stream_seed(master_seed::Integer, stream_key::Integer)::Int
    base = reinterpret(UInt64, Int64(Int(master_seed)))
    mixed = base + UInt64(stream_key) * 0x9e3779b97f4a7c15
    return Int(mod(mixed, UInt64(typemax(Int))))
end

function _write_results(path::AbstractString, rows)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "study,z_1,z_2,outer_reps,inner_train,inner_eval,gross_value,probing_cost,objective,outer_standard_error,dominant_policy_index,dominant_policy_frequency,policy_frequencies,runtime_seconds")
        for row in rows
            println(io, join((row.study, _number(row.z[1]), _number(row.z[2]),
                              string(row.outer_reps), string(row.inner_train), string(row.inner_eval),
                              _number(row.gross_value), _number(row.probing_cost), _number(row.objective),
                              _number(row.outer_standard_error), string(row.dominant_policy_index),
                              _number(row.dominant_policy_frequency), row.policy_frequencies,
                              _number(row.runtime_seconds)), ","))
        end
    end
end

"""Evaluate the fixed convergence plan; no z optimization or refinement is performed."""
function run_convergence_validation(; master_seed::Integer=20_261_001,
                                    output_path::AbstractString=DEFAULT_OUTPUT,
                                    plan=convergence_plan())
    configuration = reference_configuration(:smoke;
        sample_sizes=(outer_reps=1, inner_train=1, inner_eval=1))
    rows = NamedTuple[]
    for specification in plan
        stream_key = hasproperty(specification, :stream_key) ? specification.stream_key : 1
        elapsed = @elapsed result = evaluate_z(MersenneTwister(_z_stream_seed(master_seed, stream_key)), specification.z,
                                               configuration.params, collect(configuration.policies);
                                               outer_reps=specification.outer_reps,
                                               inner_train=specification.inner_train,
                                               inner_eval=specification.inner_eval)
        dominant_index, dominant_frequency = _dominant(result)
        push!(rows, (; specification.study, z=Tuple(result.z), specification.outer_reps,
                     specification.inner_train, specification.inner_eval,
                     gross_value=result.gross_operational_value, probing_cost=result.probing_cost,
                     objective=result.objective, outer_standard_error=result.standard_error,
                     dominant_policy_index=dominant_index,
                     dominant_policy_frequency=dominant_frequency,
                     policy_frequencies=_frequencies(result.selected_policy_frequencies),
                     runtime_seconds=elapsed))
    end
    _write_results(output_path, rows)
    return (; rows, output_path=abspath(output_path), policy_count=length(configuration.policies))
end

if abspath(PROGRAM_FILE) == @__FILE__
    validation = run_convergence_validation()
    println("wrote $(length(validation.rows)) convergence rows to $(validation.output_path)")
end

end # module ConvergenceValidation
