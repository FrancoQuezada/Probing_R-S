module PerformanceValidation

using AssemblyLineModel
using Random

include(joinpath(@__DIR__, "reference_experiment.jl"))
using .ReferenceExperiment

export performance_snapshot

function _measurement(work, label::AbstractString)
    timed = @timed work()
    return (; label, seconds=timed.time, bytes=timed.bytes, value=timed.value)
end

"""Warm and measure representative serial operations without changing the estimator."""
function performance_snapshot()
    configuration = reference_configuration(:smoke)
    params, policies = configuration.params, collect(configuration.policies)
    posterior = PosteriorState([line.prior_mean for line in params.lines],
                               [line.prior_variance for line in params.lines], [false, false])
    # Warm-up first so reported calls largely exclude compilation.
    path = generate_predictive_path(MersenneTwister(20_262_001), posterior, params)
    bank = generate_predictive_bank(MersenneTwister(20_262_002), posterior, params, 5)
    precompute_rewards(policies, bank, params)
    select_and_evaluate_policy(MersenneTwister(20_262_003), posterior, params, policies, 2, 2)
    evaluate_z(MersenneTwister(20_262_004), [0.0, 1.0], params, policies;
               outer_reps=1, inner_train=2, inner_eval=2)
    finite_grid_search(20_262_005, configuration.z_coordinate_values, params, policies;
                       outer_reps=1, inner_train=2, inner_eval=2)

    path_measurement = _measurement("generate_predictive_path") do
        generate_predictive_path(MersenneTwister(20_262_011), posterior, params)
    end
    bank_measurement = _measurement("generate_predictive_bank_20") do
        generate_predictive_bank(MersenneTwister(20_262_012), posterior, params, 20)
    end
    measured_bank = bank_measurement.value
    reward_measurement = _measurement("precompute_rewards_12x20") do
        precompute_rewards(policies, measured_bank, params)
    end
    selection_measurement = _measurement("stage5_select_and_evaluate_20_30") do
        select_and_evaluate_policy(MersenneTwister(20_262_013), posterior, params, policies, 20, 30)
    end
    z_measurement = _measurement("evaluate_z_smoke") do
        evaluate_z(MersenneTwister(20_262_014), [0.0, 1.0], params, policies;
                   outer_reps=5, inner_train=20, inner_eval=30)
    end
    grid_measurement = _measurement("finite_grid_search_9_smoke") do
        finite_grid_search(20_262_015, configuration.z_coordinate_values, params, policies;
                           outer_reps=5, inner_train=20, inner_eval=30)
    end
    jobs_in_bank = sum(length(path.jobs) for path in measured_bank)
    return (; measurements=[path_measurement, bank_measurement, reward_measurement,
                            selection_measurement, z_measurement, grid_measurement],
            policy_count=length(policies), paths_in_bank=length(measured_bank), jobs_in_bank,
            policy_path_evaluations=length(policies) * length(measured_bank))
end

if abspath(PROGRAM_FILE) == @__FILE__
    snapshot = performance_snapshot()
    for measurement in snapshot.measurements
        println("$(measurement.label): seconds=$(measurement.seconds), bytes=$(measurement.bytes)")
    end
    println("policies=$(snapshot.policy_count), bank_paths=$(snapshot.paths_in_bank), jobs=$(snapshot.jobs_in_bank), policy_path_evaluations=$(snapshot.policy_path_evaluations)")
end

end # module PerformanceValidation
