module SensitivityValidation

using AssemblyLineModel
using Printf
using Random

include(joinpath(@__DIR__, "reference_experiment.jl"))
using .ReferenceExperiment

export run_sensitivity_validation

const DEFAULT_OUTPUT = normpath(joinpath(@__DIR__, "..", "validation_outputs", "sensitivity_results.csv"))

_number(value::Real) = @sprintf("%.17g", Float64(value))

function _line_copy(line::LineParameters; diagnostic_noise=line.diagnostic_noise,
                    arrival_sensitivity=line.arrival_sensitivity,
                    complexity_sensitivity=line.complexity_sensitivity,
                    standard_service_means=line.standard_service_means)
    return LineParameters(line.value, line.prior_mean, line.prior_variance,
                          diagnostic_noise, line.max_diagnostic_time,
                          line.base_arrival_rate, arrival_sensitivity,
                          line.complexity_intercept, complexity_sensitivity,
                          line.fast_service_means, standard_service_means)
end

function _parameter_copy(params::ModelParameters; lines=collect(params.lines),
                         probe_cost=collect(params.probe_cost),
                         delay_penalty=params.delay_penalty)
    return ModelParameters(lines, probe_cost, params.horizon;
                           revenue_scale=params.revenue_scale,
                           fast_routing_cost=params.fast_routing_cost,
                           delay_penalty=delay_penalty)
end

function _dominant(result::ZEvaluationResult)
    frequency, index = findmax(result.selected_policy_frequencies)
    return index, frequency
end

function _write_results(path::AbstractString, rows)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "scenario,changed_parameter,reference_value,alternative_value,z_1,z_2,gross_value,probing_cost,objective,outer_standard_error,dominant_policy_index,dominant_policy_frequency,posterior_variance_1,posterior_variance_2,runtime_seconds")
        for row in rows
            println(io, join((row.scenario, row.changed_parameter, row.reference_value,
                              row.alternative_value, _number(row.z[1]), _number(row.z[2]),
                              _number(row.gross_value), _number(row.probing_cost),
                              _number(row.objective), _number(row.outer_standard_error),
                              string(row.dominant_policy_index), _number(row.dominant_policy_frequency),
                              _number(row.posterior_variances[1]), _number(row.posterior_variances[2]),
                              _number(row.runtime_seconds)), ","))
        end
    end
end

"""Run one-factor synthetic sanity checks at the fixed point z = (0, 1)."""
function run_sensitivity_validation(; master_seed::Integer=20_261_101,
                                    output_path::AbstractString=DEFAULT_OUTPUT,
                                    outer_reps::Integer=10, inner_train::Integer=50,
                                    inner_eval::Integer=100)
    configuration = reference_configuration(:smoke;
        sample_sizes=(outer_reps=1, inner_train=1, inner_eval=1))
    reference = configuration.params
    z = [0.0, 1.0]
    low_noise = _parameter_copy(reference;
        lines=[_line_copy(line; diagnostic_noise=line.diagnostic_noise * 0.5) for line in reference.lines])
    high_noise = _parameter_copy(reference;
        lines=[_line_copy(line; diagnostic_noise=line.diagnostic_noise * 2.0) for line in reference.lines])
    low_sensitivity = _parameter_copy(reference;
        lines=[_line_copy(line; arrival_sensitivity=line.arrival_sensitivity * 0.5,
                          complexity_sensitivity=line.complexity_sensitivity * 0.5)
               for line in reference.lines])
    high_sensitivity = _parameter_copy(reference;
        lines=[_line_copy(line; arrival_sensitivity=line.arrival_sensitivity * 1.5,
                          complexity_sensitivity=line.complexity_sensitivity * 1.5)
               for line in reference.lines])
    weaker_fast_advantage = _parameter_copy(reference;
        lines=[_line_copy(line; standard_service_means=(line.standard_service_means[1],
                                                        line.standard_service_means[2] * 0.7))
               for line in reference.lines])
    stronger_fast_advantage = _parameter_copy(reference;
        lines=[_line_copy(line; standard_service_means=(line.standard_service_means[1],
                                                        line.standard_service_means[2] * 1.4))
               for line in reference.lines])
    scenarios = [
        ("reference", "none", "reference", "reference", reference),
        ("low_probing_cost", "probe_cost", "[0.6;1]", "[0.3;0.5]",
         _parameter_copy(reference; probe_cost=[0.3, 0.5])),
        ("high_probing_cost", "probe_cost", "[0.6;1]", "[1.2;2]",
         _parameter_copy(reference; probe_cost=[1.2, 2.0])),
        ("low_diagnostic_noise", "diagnostic_noise", "reference", "0.5x", low_noise),
        ("high_diagnostic_noise", "diagnostic_noise", "reference", "2x", high_noise),
        ("low_latent_sensitivity", "arrival_and_complexity_sensitivity", "reference", "0.5x", low_sensitivity),
        ("high_latent_sensitivity", "arrival_and_complexity_sensitivity", "reference", "1.5x", high_sensitivity),
        ("weaker_fast_advantage", "complex_STANDARD_mean", "reference", "0.7x", weaker_fast_advantage),
        ("stronger_fast_advantage", "complex_STANDARD_mean", "reference", "1.4x", stronger_fast_advantage),
        ("low_delay_penalty", "delay_penalty", "0.5", "0.25", _parameter_copy(reference; delay_penalty=0.25)),
        ("high_delay_penalty", "delay_penalty", "0.5", "1.0", _parameter_copy(reference; delay_penalty=1.0)),
    ]
    rows = NamedTuple[]
    for (scenario, parameter, reference_value, alternative_value, params) in scenarios
        elapsed = @elapsed result = evaluate_z(MersenneTwister(Int(master_seed)), z, params,
                                               collect(configuration.policies); outer_reps, inner_train, inner_eval)
        posterior = posterior_parameters(DiagnosticObservation([missing, 0.0]), z, params)
        dominant_index, dominant_frequency = _dominant(result)
        push!(rows, (; scenario, changed_parameter=parameter, reference_value, alternative_value,
                     z=Tuple(result.z), gross_value=result.gross_operational_value,
                     probing_cost=result.probing_cost, objective=result.objective,
                     outer_standard_error=result.standard_error,
                     dominant_policy_index=dominant_index,
                     dominant_policy_frequency=dominant_frequency,
                     posterior_variances=posterior.variances, runtime_seconds=elapsed))
    end
    _write_results(output_path, rows)
    return (; rows, output_path=abspath(output_path), policy_count=length(configuration.policies))
end

if abspath(PROGRAM_FILE) == @__FILE__
    validation = run_sensitivity_validation()
    println("wrote $(length(validation.rows)) sensitivity rows to $(validation.output_path)")
end

end # module SensitivityValidation
