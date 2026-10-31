using Test
using Random
using AssemblyLineModel

include(joinpath(@__DIR__, "..", "examples", "experiment_runner.jl"))
include(joinpath(@__DIR__, "..", "examples", "reference_experiment.jl"))
include(joinpath(@__DIR__, "..", "examples", "convergence_validation.jl"))

function example_line(; value=0.55, prior_mean=0.2, prior_variance=1.6,
                      diagnostic_noise=2.0, max_diagnostic_time=5.0,
                      base_arrival_rate=1.1, arrival_sensitivity=0.4,
                      complexity_intercept=-0.3, complexity_sensitivity=0.7,
                      fast_service_means=(0.7, 2.1),
                      standard_service_means=(1.0, 6.0))
    return LineParameters(value, prior_mean, prior_variance, diagnostic_noise,
                          max_diagnostic_time, base_arrival_rate,
                          arrival_sensitivity, complexity_intercept,
                          complexity_sensitivity, fast_service_means,
                          standard_service_means)
end

@testset "CRN indexed finite-grid comparisons" begin
    @test AssemblyLineModel._crn_seed(101, AssemblyLineModel.CRN_PRIOR_THETA, 1, 2) ==
          AssemblyLineModel._crn_seed(101, AssemblyLineModel.CRN_PRIOR_THETA, 1, 2)
    @test AssemblyLineModel._crn_seed(101, AssemblyLineModel.CRN_PRIOR_THETA, 1, 2) !=
          AssemblyLineModel._crn_seed(101, AssemblyLineModel.CRN_PRIOR_THETA, 2, 2)
    @test AssemblyLineModel._crn_seed(101, AssemblyLineModel.CRN_PRIOR_THETA, 1, 2) !=
          AssemblyLineModel._crn_seed(101, AssemblyLineModel.CRN_DIAGNOSTIC_NOISE, 1, 2)

    params = ModelParameters([example_line(value=0.35), example_line(value=0.80)],
                             [1.2, 2.4], 8.0; revenue_scale=8.0,
                             fast_routing_cost=0.7, delay_penalty=0.42)
    policies = [Policy(0.0, 0.0, 0.0), Policy(1.0, 0.0, 0.0)]
    left, right = [0.0, 0.5], [0.5, 1.0]
    first = evaluate_z_crn(12_345, left, params, policies; outer_reps=3, inner_train=3, inner_eval=4)
    @test first == evaluate_z_crn(12_345, left, params, policies; outer_reps=3, inner_train=3, inner_eval=4)
    @test all(isfinite, first.outer_values)
    theta = AssemblyLineModel._crn_prior_theta(12_345, 1, params)
    @test theta == AssemblyLineModel._crn_prior_theta(12_345, 1, params)
    @test theta != AssemblyLineModel._crn_prior_theta(12_345, 2, params)
    zero_diagnostic = AssemblyLineModel._crn_diagnostics(12_345, 1, theta, [0.0, 0.0], params)
    @test all(ismissing, zero_diagnostic.values)
    @test posterior_parameters(zero_diagnostic, [0.0, 0.0], params).means == Tuple(line.prior_mean for line in params.lines)

    candidates = [left, right, [1.0, 0.0]]
    search = evaluate_z_candidates(12_345, candidates, params, policies;
                                   outer_reps=3, inner_train=3, inner_eval=4,
                                   randomization_mode=:crn)
    repeat_search = evaluate_z_candidates(12_345, candidates, params, policies;
                                          outer_reps=3, inner_train=3, inner_eval=4,
                                          randomization_mode=:crn)
    @test search == repeat_search
    @test search.randomization_mode == :crn
    @test all(seed == 0 for seed in search.candidate_seeds)
    reordered = evaluate_z_candidates(12_345, reverse(candidates), params, policies;
                                      outer_reps=3, inner_train=3, inner_eval=4,
                                      randomization_mode=:crn)
    by_z(result) = Dict(evaluation.z => evaluation for evaluation in result.evaluations)
    @test by_z(search) == by_z(reordered)
    paired = paired_comparison(search, 1, 2)
    left_evaluation, right_evaluation = search.evaluations[1], search.evaluations[2]
    differences = [left_evaluation.outer_values[i] - right_evaluation.outer_values[i] -
                   (left_evaluation.probing_cost - right_evaluation.probing_cost)
                   for i in 1:3]
    expected_mean, expected_se = AssemblyLineModel._mean_and_se(differences)
    @test paired.objective_difference == expected_mean && paired.paired_standard_error == expected_se
    identical = paired_comparison(search, 1, 1)
    @test identical.objective_difference == 0.0 && identical.paired_standard_error == 0.0
    @test_throws ArgumentError paired_comparison(evaluate_z_candidates(12_345, candidates, params, policies;
        outer_reps=1, inner_train=1, inner_eval=1), 1, 2)
    @test_throws ArgumentError finite_grid_search(1, [[0.0, 1.0], [0.0, 1.0]], params, policies;
        outer_reps=1, inner_train=1, inner_eval=1, randomization_mode=:invalid)

    config_path = joinpath(@__DIR__, "..", "configs", "reference_experiment.toml")
    mktempdir() do directory
        config = ExperimentRunner.load_experiment_config(config_path; overrides=Dict(
            :randomization => "crn", :crn_reference_z => "0,0", :output_dir => joinpath(directory, "crn"),
            :outer_reps => "2", :inner_train => "2", :inner_eval => "2", :z_values => "0,1;0,1"))
        execution = ExperimentRunner.execute_grid_experiment(config)
        @test execution.result.reference_candidate_index == 1
        @test length(readlines(execution.paths.outer_path)) == 1 + 4 * 2
        @test length(readlines(execution.paths.paired_path)) == 1 + 4
        reference_row = split(readlines(execution.paths.paired_path)[2], ',')
        @test reference_row[9] == "0" && reference_row[10] == "0"
    end
end

function example_params()
    return ModelParameters([example_line(value=0.35), example_line(value=0.80)],
                           [1.2, 2.4], 8.0;
                           revenue_scale=8.0,
                           fast_routing_cost=0.7,
                           delay_penalty=0.42)
end

function routing_params(; revenue_scale=8.0, fast_routing_cost=0.7,
                        delay_penalty=0.42, horizon=1.0)
    return ModelParameters([example_line(value=0.5), example_line(value=0.9)],
                           [1.2, 2.4], horizon;
                           revenue_scale=revenue_scale,
                           fast_routing_cost=fast_routing_cost,
                           delay_penalty=delay_penalty)
end

sample_mean(values) = sum(values) / length(values)
sample_variance(values) = sum((value - sample_mean(values))^2 for value in values) /
                          (length(values) - 1)

@testset "Stage 0: package setup" begin
    @test nlines(example_params()) == 2
end

@testset "Stage 1: types and parameter validation" begin
    params = example_params()
    @test params.lines[1].value == 0.35
    @test line_value(params, 2) == 0.80
    @test_throws ArgumentError ModelParameters([example_line()], [1.0, 2.0], 8.0)
    @test_throws ArgumentError ModelParameters([example_line()], [1.0], 0.0)
    @test_throws ArgumentError LineParameters(1.01, 0.0, 1.0, 1.0, 1.0, 1.0,
                                               0.0, 0.0, 0.0, (1.0, 2.0), (1.0, 2.0))
    @test_throws ArgumentError LineParameters(0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
                                               0.0, 0.0, 0.0, (1.0, 2.0), (1.0, 2.0))
    @test_throws ArgumentError LineParameters(0.5, 0.0, 1.0, 0.0, 1.0, 1.0,
                                               0.0, 0.0, 0.0, (1.0, 2.0), (1.0, 2.0))
    @test_throws ArgumentError LineParameters(0.5, 0.0, 1.0, 1.0, 1.0, -0.1,
                                               0.0, 0.0, 0.0, (1.0, 2.0), (1.0, 2.0))
    @test_throws ArgumentError LineParameters(0.5, 0.0, 1.0, 1.0, 1.0, 0.0,
                                               0.0, 0.0, 0.0, (1.0, 2.0), (1.0, 2.0))
    @test_throws ArgumentError LineParameters(0.5, 0.0, 1.0, 1.0, 1.0, 1.0,
                                               0.0, 0.0, 0.0, (0.0, 2.0), (1.0, 2.0))
    @test !hasfield(ReworkJob, :signal)
    @test !hasfield(ReworkJob, :value)
    @test fieldnames(ReworkJob) == (:arrival, :line, :complex, :service_fast, :service_standard)
    job = ReworkJob(0.5, 2, true, 1.5, 4.0)
    @test line_value(params, job.line) == 0.80
    @test_throws ArgumentError validate_z([0.0], params)
end

@testset "Stage 2: Bayesian information layer" begin
    params = example_params()
    @test length(sample_prior_theta(MersenneTwister(7), params)) == 2
    @test sample_prior_theta(MersenneTwister(7), params) ==
          sample_prior_theta(MersenneTwister(7), params)

    theta = [1.0, -0.5]
    diagnostics = sample_diagnostics(MersenneTwister(99), theta, [0.0, 0.5], params)
    @test ismissing(diagnostics.values[1])
    @test !ismissing(diagnostics.values[2])
    @test diagnostics == sample_diagnostics(MersenneTwister(99), theta, [0.0, 0.5], params)

    prior_posterior = posterior_parameters(DiagnosticObservation([missing, missing]),
                                         [0.0, 0.0], params)
    @test prior_posterior.means == Tuple(line.prior_mean for line in params.lines)
    @test prior_posterior.variances == Tuple(line.prior_variance for line in params.lines)
    @test prior_posterior.observed == (false, false)

    hand_line = example_line(prior_mean=1.0, prior_variance=4.0,
                             diagnostic_noise=2.0, max_diagnostic_time=5.0)
    hand_params = ModelParameters([hand_line], [1.0], 8.0)
    hand_posterior = posterior_parameters(DiagnosticObservation([3.0]), [0.5], hand_params)
    @test hand_posterior.variances[1] ≈ 2 / 3
    @test hand_posterior.means[1] ≈ 8 / 3
    @test hand_posterior.observed == (true,)

    low_effort = posterior_parameters(DiagnosticObservation([0.0, 0.0]), [0.2, 0.2], params)
    high_effort = posterior_parameters(DiagnosticObservation([0.0, 0.0]), [0.8, 0.8], params)
    @test all(high_effort.variances .< low_effort.variances)
    @test diagnostic_variance(1, 0.8, params) < diagnostic_variance(1, 0.2, params)

    mixed = posterior_parameters(DiagnosticObservation([missing, 1.4]), [0.0, 1.0], params)
    @test mixed.means[1] == params.lines[1].prior_mean
    @test mixed.variances[1] == params.lines[1].prior_variance
    @test mixed.observed == (false, true)

    @test_throws ArgumentError sample_diagnostics(MersenneTwister(1), theta, [-0.1, 0.1], params)
    @test_throws ArgumentError sample_diagnostics(MersenneTwister(1), theta, [0.1, 1.1], params)
    @test_throws ArgumentError posterior_parameters(DiagnosticObservation([0.0, missing]), [0.0, 0.0], params)
    @test_throws ArgumentError posterior_parameters(DiagnosticObservation([missing, missing]), [0.5, 0.5], params)
    @test !(:theta in fieldnames(PosteriorState))
end

@testset "Stage 3: posterior predictive simulation" begin
    params = example_params()
    posterior = PosteriorState([0.3, -0.4], [1.4, 0.25], [true, false])

    draw_a = sample_posterior_theta(MersenneTwister(42), posterior)
    draw_b = sample_posterior_theta(MersenneTwister(42), posterior)
    @test draw_a == draw_b
    @test length(draw_a) == 2
    @test all(isfinite, draw_a)
    sequential_rng = MersenneTwister(42)
    @test sample_posterior_theta(sequential_rng, posterior) !=
          sample_posterior_theta(sequential_rng, posterior)

    posterior_rng = MersenneTwister(101)
    posterior_draws = [sample_posterior_theta(posterior_rng, posterior) for _ in 1:2_000]
    first_coordinates = [draw[1] for draw in posterior_draws]
    second_coordinates = [draw[2] for draw in posterior_draws]
    @test sample_mean(first_coordinates) ≈ posterior.means[1] atol=0.10
    @test sample_variance(first_coordinates) ≈ posterior.variances[1] rtol=0.12
    @test sample_mean(second_coordinates) ≈ posterior.means[2] atol=0.05
    @test sample_variance(second_coordinates) ≈ posterior.variances[2] rtol=0.12

    line = params.lines[1]
    @test arrival_rate(line.prior_mean, line) == line.base_arrival_rate
    @test arrival_rate(line.prior_mean + 1.0, line) > arrival_rate(line.prior_mean, line)
    constant_arrival_line = example_line(arrival_sensitivity=0.0)
    @test arrival_rate(-100.0, constant_arrival_line) ==
          arrival_rate(100.0, constant_arrival_line) == constant_arrival_line.base_arrival_rate
    @test isfinite(arrival_rate(0.5, line)) && arrival_rate(0.5, line) > 0.0
    overflow_line = example_line(prior_mean=-1e308, arrival_sensitivity=1.0)
    @test_throws ArgumentError arrival_rate(1e308, overflow_line)

    logistic_at_mean = 1.0 / (1.0 + exp(-line.complexity_intercept))
    @test complexity_probability(line.prior_mean, line) ≈ logistic_at_mean
    @test complexity_probability(line.prior_mean + 1.0, line) >
          complexity_probability(line.prior_mean, line)
    constant_complexity_line = example_line(complexity_sensitivity=0.0)
    @test complexity_probability(-100.0, constant_complexity_line) ==
          complexity_probability(100.0, constant_complexity_line)
    @test 0.0 <= complexity_probability(1e308, line) <= 1.0
    @test 0.0 <= complexity_probability(-1e308, line) <= 1.0

    path = generate_predictive_path(MersenneTwister(23), posterior, params)
    @test issorted([job.arrival for job in path.jobs])
    @test all(0.0 <= job.arrival <= params.horizon for job in path.jobs)
    @test all(1 <= job.line <= nlines(params) for job in path.jobs)
    @test all(isfinite(job.service_fast) && job.service_fast > 0.0 &&
              isfinite(job.service_standard) && job.service_standard > 0.0 for job in path.jobs)
    @test all(job.complex isa Bool for job in path.jobs)
    @test !hasfield(ReworkJob, :value)
    @test !hasfield(ReworkJob, :signal)
    @test all(line_value(params, job.line) == params.lines[job.line].value for job in path.jobs)
    @test isempty(ShiftPath(ReworkJob[]).jobs)

    near_zero_line = example_line(base_arrival_rate=1e-20)
    near_zero_params = ModelParameters([near_zero_line], [1.0], 1.0)
    near_zero_posterior = PosteriorState([0.0], [1e-12], [false])
    @test isempty(generate_predictive_path(MersenneTwister(4), near_zero_posterior,
                                           near_zero_params).jobs)
    empty_bank = generate_predictive_bank(MersenneTwister(4), posterior, params, 0)
    @test empty_bank isa Vector{ShiftPath}
    @test isempty(empty_bank)
    @test_throws ArgumentError generate_predictive_bank(MersenneTwister(4), posterior, params, -1)
    @test_throws ArgumentError generate_predictive_path(MersenneTwister(4),
                                                         PosteriorState([0.0], [1.0], [false]), params)

    reproducible_a = generate_predictive_path(MersenneTwister(73), posterior, params)
    reproducible_b = generate_predictive_path(MersenneTwister(73), posterior, params)
    distinct_seed = generate_predictive_path(MersenneTwister(74), posterior, params)
    @test reproducible_a == reproducible_b
    @test reproducible_a != distinct_seed
    bank_a = generate_predictive_bank(MersenneTwister(81), posterior, params, 4)
    bank_b = generate_predictive_bank(MersenneTwister(81), posterior, params, 4)
    @test bank_a == bank_b

    behavior_line = example_line(prior_mean=0.0, prior_variance=1.0,
                                 base_arrival_rate=2.0, arrival_sensitivity=0.7,
                                 complexity_intercept=-1.0, complexity_sensitivity=1.2,
                                 fast_service_means=(1.5, 4.0),
                                 standard_service_means=(2.0, 8.0))
    behavior_params = ModelParameters([behavior_line], [1.0], 3.0)
    good_posterior = PosteriorState([-1.0], [1e-12], [true])
    bad_posterior = PosteriorState([1.0], [1e-12], [true])
    good_bank = generate_predictive_bank(MersenneTwister(301), good_posterior, behavior_params, 300)
    bad_bank = generate_predictive_bank(MersenneTwister(302), bad_posterior, behavior_params, 300)
    good_counts = [length(path.jobs) for path in good_bank]
    bad_counts = [length(path.jobs) for path in bad_bank]
    @test sample_mean(bad_counts) > sample_mean(good_counts)
    expected_bad_count = arrival_rate(1.0, behavior_line) * behavior_params.horizon
    @test sample_mean(bad_counts) ≈ expected_bad_count atol=0.65
    good_jobs = reduce(vcat, [collect(path.jobs) for path in good_bank]; init=ReworkJob[])
    bad_jobs = reduce(vcat, [collect(path.jobs) for path in bad_bank]; init=ReworkJob[])
    @test sample_mean([job.complex for job in bad_jobs]) >
          sample_mean([job.complex for job in good_jobs])

    simple_line = example_line(base_arrival_rate=40.0, complexity_intercept=-40.0,
                               complexity_sensitivity=0.0,
                               fast_service_means=(1.5, 4.0),
                               standard_service_means=(2.0, 8.0))
    simple_params = ModelParameters([simple_line], [1.0], 2.0)
    simple_posterior = PosteriorState([0.0], [1e-12], [false])
    simple_bank = generate_predictive_bank(MersenneTwister(901), simple_posterior,
                                            simple_params, 100)
    simple_jobs = reduce(vcat, [collect(path.jobs) for path in simple_bank]; init=ReworkJob[])
    @test !isempty(simple_jobs)
    @test sample_mean([job.service_fast for job in simple_jobs]) ≈ 1.5 rtol=0.08
    @test sample_mean([job.service_standard for job in simple_jobs]) ≈ 2.0 rtol=0.08

    complex_line = example_line(base_arrival_rate=40.0, complexity_intercept=40.0,
                                complexity_sensitivity=0.0,
                                fast_service_means=(1.5, 4.0),
                                standard_service_means=(2.0, 8.0))
    complex_params = ModelParameters([complex_line], [1.0], 2.0)
    complex_bank = generate_predictive_bank(MersenneTwister(902), simple_posterior,
                                             complex_params, 100)
    complex_jobs = reduce(vcat, [collect(path.jobs) for path in complex_bank]; init=ReworkJob[])
    @test all(job.complex for job in complex_jobs)
    @test sample_mean([job.service_fast for job in complex_jobs]) ≈ 4.0 rtol=0.08
    @test sample_mean([job.service_standard for job in complex_jobs]) ≈ 8.0 rtol=0.08
end

@testset "Stage 4: routing simulation and common-path evaluation" begin
    params = routing_params()
    line_one_job = ReworkJob(0.0, 1, false, 2.0, 5.0)
    line_two_job = ReworkJob(0.0, 2, true, 7.0, 9.0)
    score_policy = Policy(1.0, 0.5, 0.4)

    @test !hasfield(Policy, :as)
    @test routing_score(score_policy, line_one_job, 0.0, 0.0, params) == 0.5
    @test routing_score(score_policy, line_two_job, 0.0, 0.0, params) == 0.9
    @test routing_score(score_policy, line_one_job, 2.0, 0.0, params) == -0.5
    @test routing_score(score_policy, line_one_job, 0.0, 2.0, params) == 1.5
    @test routing_score(score_policy, ReworkJob(0.0, 1, true, 99.0, 101.0), 1.0, 0.0, params) ==
          routing_score(score_policy, ReworkJob(0.0, 1, false, 0.1, 0.2), 1.0, 0.0, params)
    @test_throws ArgumentError routing_score(score_policy, ReworkJob(0.0, 3, false, 1.0, 1.0),
                                              0.0, 0.0, params)
    @test_throws ArgumentError routing_score(score_policy, line_one_job, -0.1, 0.0, params)
    @test_throws ArgumentError routing_score(score_policy, line_one_job, Inf, 0.0, params)
    @test_throws ArgumentError routing_score(score_policy, line_one_job, 0.0, NaN, params)
    @test_throws ArgumentError Policy(-0.1, 0.0, 0.0)
    @test_throws ArgumentError Policy(0.0, -0.1, 0.0)
    @test_throws ArgumentError Policy(0.0, 0.0, Inf)

    fast_policy = Policy(1.0, 0.0, 0.0)
    standard_policy = Policy(0.0, 0.0, 1.0)
    tie_policy = Policy(0.0, 0.0, 0.0)
    fast_trace = AssemblyLineModel._simulate_policy_trace(fast_policy,
                                                           ShiftPath([line_one_job]), params)
    standard_trace = AssemblyLineModel._simulate_policy_trace(standard_policy,
                                                               ShiftPath([line_one_job]), params)
    tie_trace = AssemblyLineModel._simulate_policy_trace(tie_policy,
                                                          ShiftPath([line_one_job]), params)
    @test fast_trace[1].routed_fast
    @test !standard_trace[1].routed_fast
    @test tie_trace[1].routed_fast # score == tau routes to FAST
    @test fast_trace[1].start == 0.0 && fast_trace[1].finish == 2.0
    @test standard_trace[1].start == 0.0 && standard_trace[1].finish == 5.0
    @test fast_trace[1].time_in_system == 2.0
    @test evaluate_policy(fast_policy, ShiftPath([line_one_job]), params) ≈ 2.46
    @test evaluate_policy(standard_policy, ShiftPath([line_one_job]), params) ≈ 1.90

    # Hand-checked four-job shift under (av, aq, tau) = (1, 0.5, 0.4).
    # Routes: FAST, FAST, STANDARD, STANDARD; total reward = 18.522.
    hand_path = ShiftPath([
        ReworkJob(0.0, 1, false, 1.0, 2.0),
        ReworkJob(0.2, 2, true, 2.0, 3.0),
        ReworkJob(0.5, 1, true, 0.5, 1.0),
        ReworkJob(1.0, 2, false, 1.0, 0.6),
    ])
    hand_trace = AssemblyLineModel._simulate_policy_trace(score_policy, hand_path, params)
    @test [record.q_fast for record in hand_trace] ≈ [0.0, 0.8, 2.5, 2.0]
    @test [record.q_standard for record in hand_trace] ≈ [0.0, 0.0, 0.0, 0.5]
    @test [record.routed_fast for record in hand_trace] == [true, true, false, false]
    @test [record.start for record in hand_trace] ≈ [0.0, 1.0, 0.5, 1.5]
    @test [record.finish for record in hand_trace] ≈ [1.0, 3.0, 1.5, 2.1]
    @test [record.time_in_system for record in hand_trace] ≈ [1.0, 2.8, 1.0, 1.1]
    @test [record.reward for record in hand_trace] ≈ [2.88, 5.324, 3.58, 6.738]
    @test evaluate_policy(score_policy, hand_path, params) ≈ 18.522
    @test hand_trace[end].finish > params.horizon

    two_fast = ShiftPath([ReworkJob(0.0, 1, false, 2.0, 9.0),
                          ReworkJob(0.5, 1, false, 1.0, 9.0)])
    two_fast_trace = AssemblyLineModel._simulate_policy_trace(fast_policy, two_fast, params)
    @test [record.start for record in two_fast_trace] == [0.0, 2.0]
    @test [record.finish for record in two_fast_trace] == [2.0, 3.0]
    @test [record.q_fast for record in two_fast_trace] == [0.0, 1.5]

    two_standard = ShiftPath([ReworkJob(0.0, 1, false, 9.0, 2.0),
                              ReworkJob(0.5, 1, false, 9.0, 1.0)])
    two_standard_trace = AssemblyLineModel._simulate_policy_trace(standard_policy, two_standard, params)
    @test [record.start for record in two_standard_trace] == [0.0, 2.0]
    @test [record.finish for record in two_standard_trace] == [2.0, 3.0]
    @test [record.q_standard for record in two_standard_trace] == [0.0, 1.5]

    idle_fast = ShiftPath([ReworkJob(0.0, 1, false, 1.0, 9.0),
                           ReworkJob(2.0, 1, false, 1.0, 9.0)])
    idle_fast_trace = AssemblyLineModel._simulate_policy_trace(fast_policy, idle_fast, params)
    @test [record.q_fast for record in idle_fast_trace] == [0.0, 0.0]
    @test [record.start for record in idle_fast_trace] == [0.0, 2.0]

    simultaneous = ShiftPath([ReworkJob(0.0, 1, false, 1.0, 9.0),
                              ReworkJob(0.0, 1, false, 1.0, 9.0)])
    simultaneous_trace = AssemblyLineModel._simulate_policy_trace(fast_policy, simultaneous, params)
    @test [record.q_fast for record in simultaneous_trace] == [0.0, 1.0]
    @test [record.start for record in simultaneous_trace] == [0.0, 1.0]

    empty_path = ShiftPath(ReworkJob[])
    @test evaluate_policy(fast_policy, empty_path, params) == 0.0
    @test evaluate_policies_on_path([fast_policy, standard_policy], empty_path, params) == [0.0, 0.0]

    common_rewards = evaluate_policies_on_path([score_policy, fast_policy, standard_policy],
                                                hand_path, params)
    @test common_rewards == [evaluate_policy(policy, hand_path, params)
                             for policy in [score_policy, fast_policy, standard_policy]]
    path_before = hand_path
    policy_before = score_policy
    params_before = params
    @test evaluate_policy(score_policy, hand_path, params) == evaluate_policy(score_policy, hand_path, params)
    @test hand_path == path_before && score_policy == policy_before && params == params_before
    @test evaluate_policies_on_path([fast_policy, standard_policy], hand_path, params) ==
          reverse(evaluate_policies_on_path([standard_policy, fast_policy], hand_path, params))
    Random.seed!(4_444)
    expected_random = rand()
    Random.seed!(4_444)
    evaluate_policy(score_policy, hand_path, params)
    @test rand() == expected_random

    reward_matrix = precompute_rewards([score_policy, standard_policy], [hand_path, empty_path], params)
    @test size(reward_matrix) == (2, 2)
    @test eltype(reward_matrix) == Float64
    @test reward_matrix[1, 1] == evaluate_policy(score_policy, hand_path, params)
    @test reward_matrix[2, 1] == evaluate_policy(standard_policy, hand_path, params)
    @test reward_matrix[:, 2] == [0.0, 0.0]
    @test size(precompute_rewards(Policy[], [hand_path], params)) == (0, 1)
    @test size(precompute_rewards([fast_policy], ShiftPath[], params)) == (1, 0)
    @test size(precompute_rewards(Policy[], ShiftPath[], params)) == (0, 0)

    zero_revenue = routing_params(revenue_scale=0.0)
    zero_fast_cost = routing_params(fast_routing_cost=0.0)
    zero_delay = routing_params(delay_penalty=0.0)
    one_job_path = ShiftPath([line_one_job])
    @test evaluate_policy(fast_policy, one_job_path, zero_revenue) ≈ -1.54
    @test evaluate_policy(fast_policy, one_job_path, zero_fast_cost) ≈ 3.16
    @test evaluate_policy(fast_policy, one_job_path, zero_delay) ≈ 3.30
    @test evaluate_policy(standard_policy, one_job_path, params) ≈ 1.90
    @test_throws ArgumentError routing_params(revenue_scale=-0.1)
    @test_throws ArgumentError routing_params(fast_routing_cost=-0.1)
    @test_throws ArgumentError routing_params(delay_penalty=-0.1)
    @test_throws ArgumentError routing_params(revenue_scale=Inf)
    @test_throws ArgumentError routing_params(fast_routing_cost=NaN)
    @test_throws ArgumentError routing_params(delay_penalty=Inf)
end

@testset "Stage 5: conditional policy selection and independent evaluation" begin
    params = routing_params()
    posterior = PosteriorState([0.1, -0.2], [0.4, 0.3], [true, true])
    policies = [Policy(1.0, 0.0, 0.0), Policy(0.0, 0.0, 1.0),
                Policy(1.0, 0.5, 0.4)]

    estimate_rng = MersenneTwister(5_001)
    estimated_values = estimate_policy_values(estimate_rng, posterior, params, policies, 5)
    manual_rng = MersenneTwister(5_001)
    manual_training_paths = generate_predictive_bank(manual_rng, posterior, params, 5)
    manual_rewards = precompute_rewards(policies, manual_training_paths, params)
    manual_values = [sum(@view manual_rewards[i, :]) / 5 for i in axes(manual_rewards, 1)]
    @test length(estimated_values) == length(policies)
    @test estimated_values == manual_values
    @test estimate_policy_values(MersenneTwister(5_001), posterior, params, policies, 5) ==
          estimated_values
    @test estimate_policy_values(MersenneTwister(5_001), posterior, params, Policy[], 5) == Float64[]
    @test_throws ArgumentError estimate_policy_values(MersenneTwister(1), posterior, params, policies, 0)
    @test_throws ArgumentError estimate_policy_values(MersenneTwister(1), posterior, params, policies, -1)

    selection_rng = MersenneTwister(5_002)
    training_selection = select_policy_given_posterior(selection_rng, posterior, params, policies, 5)
    reproduced_training_values = estimate_policy_values(MersenneTwister(5_002), posterior, params, policies, 5)
    @test training_selection.selected_index == argmax(reproduced_training_values)
    @test training_selection.training_values == Tuple(reproduced_training_values)
    @test training_selection.selected_policy == policies[training_selection.selected_index]
    @test training_selection.selected_training_value ==
          training_selection.training_values[training_selection.selected_index]
    @test isnothing(training_selection.evaluation_mean)
    @test isnothing(training_selection.evaluation_standard_error)
    @test isnothing(training_selection.neval)
    @test training_selection.ntrain == 5
    @test AssemblyLineModel._first_argmax([2.0, 2.0, 1.0]) == 1
    @test AssemblyLineModel._first_argmax([1.0, 3.0, 3.0]) == 2
    @test_throws ArgumentError AssemblyLineModel._first_argmax(Float64[])
    @test_throws ArgumentError select_policy_given_posterior(MersenneTwister(1), posterior, params,
                                                              Policy[], 5)
    @test_throws ArgumentError select_policy_given_posterior(MersenneTwister(1), posterior, params,
                                                              policies, 0)

    evaluation_rng = MersenneTwister(5_003)
    evaluation = evaluate_selected_policy(evaluation_rng, posterior, params, policies[1], 4)
    manual_evaluation_rng = MersenneTwister(5_003)
    manual_evaluation_paths = generate_predictive_bank(manual_evaluation_rng, posterior, params, 4)
    manual_evaluation_rewards = [evaluate_policy(policies[1], path, params)
                                 for path in manual_evaluation_paths]
    manual_evaluation_mean = sample_mean(manual_evaluation_rewards)
    manual_evaluation_se = sqrt(sample_variance(manual_evaluation_rewards) / 4)
    @test evaluation.evaluation_mean == manual_evaluation_mean
    @test evaluation.evaluation_standard_error == manual_evaluation_se
    @test evaluation.neval == 4
    @test evaluation.evaluation_standard_error >= 0.0
    one_evaluation = evaluate_selected_policy(MersenneTwister(5_004), posterior, params, policies[1], 1)
    @test one_evaluation.evaluation_standard_error == 0.0
    @test_throws ArgumentError evaluate_selected_policy(MersenneTwister(1), posterior, params, policies[1], 0)
    @test_throws ArgumentError evaluate_selected_policy(MersenneTwister(1), posterior, params, policies[1], -1)

    combined_rng = MersenneTwister(5_005)
    result = select_and_evaluate_policy(combined_rng, posterior, params, policies, 5, 4)
    manual_combined_rng = MersenneTwister(5_005)
    combined_train_paths = generate_predictive_bank(manual_combined_rng, posterior, params, 5)
    combined_train_values = [sum(@view values[i, :]) / 5
                             for values in (precompute_rewards(policies, combined_train_paths, params),)
                             for i in 1:length(policies)]
    combined_index = argmax(combined_train_values)
    combined_eval_paths = generate_predictive_bank(manual_combined_rng, posterior, params, 4)
    combined_eval_rewards = [evaluate_policy(policies[combined_index], path, params)
                             for path in combined_eval_paths]
    @test result.selected_index == combined_index
    @test result.selected_policy == policies[combined_index]
    @test result.training_values == Tuple(combined_train_values)
    @test result.selected_training_value == combined_train_values[combined_index]
    @test result.evaluation_mean == sample_mean(combined_eval_rewards)
    @test result.evaluation_standard_error == sqrt(sample_variance(combined_eval_rewards) / 4)
    @test result.ntrain == 5 && result.neval == 4
    @test result == select_and_evaluate_policy(MersenneTwister(5_005), posterior, params, policies, 5, 4)
    @test result != select_and_evaluate_policy(MersenneTwister(5_006), posterior, params, policies, 5, 4)
    @test result.evaluation_mean != result.selected_training_value

    one_policy_result = select_and_evaluate_policy(MersenneTwister(5_007), posterior, params,
                                                    [policies[2]], 3, 2)
    @test one_policy_result.selected_index == 1
    @test one_policy_result.selected_policy == policies[2]
    @test one_policy_result.ntrain == 3 && one_policy_result.neval == 2
    @test isfinite(one_policy_result.evaluation_mean)
    @test one_policy_result.evaluation_standard_error >= 0.0
    @test_throws ArgumentError select_and_evaluate_policy(MersenneTwister(1), posterior, params,
                                                            Policy[], 3, 2)
    @test_throws ArgumentError select_and_evaluate_policy(MersenneTwister(1), posterior, params,
                                                            policies, 0, 2)
    @test_throws ArgumentError select_and_evaluate_policy(MersenneTwister(1), posterior, params,
                                                            policies, 2, 0)

    near_zero_line = example_line(base_arrival_rate=1e-20)
    zero_params = ModelParameters([near_zero_line], [999.0], 1.0)
    zero_posterior = PosteriorState([0.0], [1e-12], [false])
    zero_policies = [Policy(1.0, 0.0, 0.0), Policy(0.0, 0.0, 1.0)]
    zero_result = select_and_evaluate_policy(MersenneTwister(5_008), zero_posterior,
                                              zero_params, zero_policies, 3, 2)
    @test zero_result.training_values == (0.0, 0.0)
    @test zero_result.selected_index == 1
    @test zero_result.evaluation_mean == 0.0
    @test zero_result.evaluation_standard_error == 0.0

    posterior_before = posterior
    policies_before = copy(policies)
    params_before = params
    Random.seed!(5_009)
    expected_random = rand()
    Random.seed!(5_009)
    select_and_evaluate_policy(MersenneTwister(5_010), posterior, params, policies, 3, 2)
    @test rand() == expected_random
    @test posterior == posterior_before
    @test policies == policies_before
    @test params == params_before
    legacy_result = PolicySelectionResult(1, policies[1], [1.0, 0.5])
    @test legacy_result.selected_training_value == 1.0
    @test isnothing(legacy_result.evaluation_mean) && isnothing(legacy_result.neval)
end

@testset "Stage 6: outer nested Monte Carlo evaluation" begin
    params = routing_params()
    policies = [Policy(1.0, 0.0, 0.0), Policy(0.0, 0.0, 1.0),
                Policy(1.0, 0.5, 0.4)]
    z = [0.2, 0.4]

    invalid_rng = MersenneTwister(6_001)
    @test_throws ArgumentError evaluate_z(invalid_rng, [0.2], params, policies;
                                          outer_reps=2, inner_train=2, inner_eval=2)
    @test rand(invalid_rng) == rand(MersenneTwister(6_001))
    @test_throws ArgumentError evaluate_z(MersenneTwister(1), [-0.1, 0.2], params, policies;
                                          outer_reps=2, inner_train=2, inner_eval=2)
    @test_throws ArgumentError evaluate_z(MersenneTwister(1), [0.2, 1.1], params, policies;
                                          outer_reps=2, inner_train=2, inner_eval=2)
    @test_throws ArgumentError evaluate_z(MersenneTwister(1), [NaN, 0.2], params, policies;
                                          outer_reps=2, inner_train=2, inner_eval=2)
    @test_throws ArgumentError evaluate_z(MersenneTwister(1), z, params, policies;
                                          outer_reps=0, inner_train=2, inner_eval=2)
    @test_throws ArgumentError evaluate_z(MersenneTwister(1), z, params, policies;
                                          outer_reps=2, inner_train=0, inner_eval=2)
    @test_throws ArgumentError evaluate_z(MersenneTwister(1), z, params, policies;
                                          outer_reps=2, inner_train=2, inner_eval=0)
    @test_throws ArgumentError evaluate_z(MersenneTwister(1), z, params, Policy[];
                                          outer_reps=2, inner_train=2, inner_eval=2)

    outer_reps = 3
    inner_train = 4
    inner_eval = 3
    result_rng = MersenneTwister(6_002)
    result = evaluate_z(result_rng, z, params, policies;
                        outer_reps=outer_reps, inner_train=inner_train,
                        inner_eval=inner_eval)

    manual_rng = MersenneTwister(6_002)
    manual_outer_values = Float64[]
    manual_indices = Int[]
    manual_training_values = Float64[]
    manual_inner_ses = Float64[]
    for _ in 1:outer_reps
        theta_true = sample_prior_theta(manual_rng, params)
        diagnostics = sample_diagnostics(manual_rng, theta_true, z, params)
        posterior = posterior_parameters(diagnostics, z, params)
        selection = select_and_evaluate_policy(manual_rng, posterior, params, policies,
                                               inner_train, inner_eval)
        push!(manual_outer_values, selection.evaluation_mean)
        push!(manual_indices, selection.selected_index)
        push!(manual_training_values, selection.selected_training_value)
        push!(manual_inner_ses, selection.evaluation_standard_error)
    end
    manual_counts = [count(==(i), manual_indices) for i in eachindex(policies)]
    manual_frequencies = manual_counts ./ outer_reps
    manual_gross = sample_mean(manual_outer_values)
    manual_outer_se = sqrt(sample_variance(manual_outer_values) / outer_reps)
    manual_cost = sum(params.probe_cost .* z)
    @test result.outer_values == Tuple(manual_outer_values)
    @test result.selected_policy_indices == Tuple(manual_indices)
    @test result.selected_training_values == Tuple(manual_training_values)
    @test result.inner_evaluation_standard_errors == Tuple(manual_inner_ses)
    @test result.gross_operational_value == manual_gross
    @test result.probing_cost == manual_cost
    @test result.objective == manual_gross - manual_cost
    @test result.standard_error == manual_outer_se
    @test result.selected_policy_counts == Tuple(manual_counts)
    @test result.selected_policy_frequencies == Tuple(manual_frequencies)
    @test result.outer_reps == outer_reps
    @test result.inner_train == inner_train && result.inner_eval == inner_eval
    @test isnothing(result.seed)
    @test sum(result.selected_policy_counts) == outer_reps
    @test sum(result.selected_policy_frequencies) ≈ 1.0
    @test all(1 <= index <= length(policies) for index in result.selected_policy_indices)
    @test !(:theta in fieldnames(ZEvaluationResult))
    @test !(:diagnostics in fieldnames(ZEvaluationResult))
    @test !(:posteriors in fieldnames(ZEvaluationResult))

    single_outer = evaluate_z(MersenneTwister(6_003), z, params, policies;
                              outer_reps=1, inner_train=3, inner_eval=2)
    @test single_outer.standard_error == 0.0
    @test isfinite(single_outer.gross_operational_value)
    @test isfinite(single_outer.objective)

    zero_z = [0.0, 0.0]
    zero_result = evaluate_z(MersenneTwister(6_004), zero_z, params, policies;
                             outer_reps=2, inner_train=2, inner_eval=2)
    zero_manual_rng = MersenneTwister(6_004)
    zero_theta = sample_prior_theta(zero_manual_rng, params)
    zero_diagnostics = sample_diagnostics(zero_manual_rng, zero_theta, zero_z, params)
    zero_posterior = posterior_parameters(zero_diagnostics, zero_z, params)
    @test all(ismissing, zero_diagnostics.values)
    @test zero_posterior.means == Tuple(line.prior_mean for line in params.lines)
    @test zero_posterior.variances == Tuple(line.prior_variance for line in params.lines)
    @test zero_result.probing_cost == 0.0
    @test isfinite(zero_result.objective)
    @test zero_result.standard_error >= 0.0

    zero_cost_params = ModelParameters([params.lines...], [0.0, 0.0], params.horizon;
                                        revenue_scale=params.revenue_scale,
                                        fast_routing_cost=params.fast_routing_cost,
                                        delay_penalty=params.delay_penalty)
    high_cost_params = ModelParameters([params.lines...], [10.0, 20.0], params.horizon;
                                        revenue_scale=params.revenue_scale,
                                        fast_routing_cost=params.fast_routing_cost,
                                        delay_penalty=params.delay_penalty)
    zero_cost_result = evaluate_z(MersenneTwister(6_005), z, zero_cost_params, policies;
                                  outer_reps=2, inner_train=3, inner_eval=2)
    high_cost_result = evaluate_z(MersenneTwister(6_005), z, high_cost_params, policies;
                                  outer_reps=2, inner_train=3, inner_eval=2)
    @test zero_cost_result.gross_operational_value == high_cost_result.gross_operational_value
    @test zero_cost_result.outer_values == high_cost_result.outer_values
    @test zero_cost_result.selected_policy_indices == high_cost_result.selected_policy_indices
    @test zero_cost_result.standard_error == high_cost_result.standard_error
    @test zero_cost_result.probing_cost == 0.0
    @test high_cost_result.probing_cost == 10.0 * z[1] + 20.0 * z[2]
    @test high_cost_result.objective == high_cost_result.gross_operational_value -
                                        high_cost_result.probing_cost

    one_policy = [policies[1]]
    one_policy_result = evaluate_z(MersenneTwister(6_006), z, params, one_policy;
                                   outer_reps=3, inner_train=2, inner_eval=2)
    @test one_policy_result.selected_policy_indices == (1, 1, 1)
    @test one_policy_result.selected_policy_counts == (3,)
    @test one_policy_result.selected_policy_frequencies == (1.0,)
    @test one_policy_result.gross_operational_value == sample_mean(one_policy_result.outer_values)

    near_zero_line = example_line(base_arrival_rate=1e-20)
    empty_params = ModelParameters([near_zero_line], [3.0], 1.0)
    empty_result = evaluate_z(MersenneTwister(6_007), [0.5], empty_params,
                              [Policy(1.0, 0.0, 0.0), Policy(0.0, 0.0, 1.0)];
                              outer_reps=2, inner_train=2, inner_eval=2)
    @test empty_result.outer_values == (0.0, 0.0)
    @test empty_result.selected_policy_indices == (1, 1)
    @test empty_result.gross_operational_value == 0.0
    @test empty_result.objective == -1.5
    @test empty_result.standard_error == 0.0

    z_before = copy(z)
    policies_before = copy(policies)
    params_before = params
    Random.seed!(6_008)
    expected_random = rand()
    Random.seed!(6_008)
    reproducible_a = evaluate_z(MersenneTwister(6_009), z, params, policies;
                                outer_reps=2, inner_train=2, inner_eval=2)
    reproducible_b = evaluate_z(MersenneTwister(6_009), z, params, policies;
                                outer_reps=2, inner_train=2, inner_eval=2)
    distinct_seed = evaluate_z(MersenneTwister(6_010), z, params, policies;
                               outer_reps=2, inner_train=2, inner_eval=2)
    @test rand() == expected_random
    @test reproducible_a == reproducible_b
    @test reproducible_a != distinct_seed
    @test z == z_before
    @test policies == policies_before
    @test params == params_before

    legacy_z_result = ZEvaluationResult([0.0], 1.0, 0.0, 1.0, 0.0, [1.0], [1.0], 123)
    @test legacy_z_result.seed == 123
    @test isempty(legacy_z_result.selected_policy_indices)
    @test legacy_z_result.outer_reps == 1
end

@testset "Stage 7: finite candidate evaluation and deterministic ranking" begin
    params = routing_params(horizon=1.0)
    policies = [Policy(1.0, 0.0, 0.0), Policy(0.0, 0.0, 1.0)]
    candidates = [[0.0, 0.0], [0.25, 0.5]]

    @test_throws ArgumentError evaluate_z_candidates(7, Vector{Vector{Float64}}(), params, policies;
                                                       outer_reps=1, inner_train=1, inner_eval=1)
    @test_throws ArgumentError evaluate_z_candidates(7, [[0.0]], params, policies;
                                                       outer_reps=1, inner_train=1, inner_eval=1)
    @test_throws ArgumentError evaluate_z_candidates(7, [[-0.1, 0.0]], params, policies;
                                                       outer_reps=1, inner_train=1, inner_eval=1)
    @test_throws ArgumentError evaluate_z_candidates(7, [[0.0, 1.1]], params, policies;
                                                       outer_reps=1, inner_train=1, inner_eval=1)
    @test_throws ArgumentError evaluate_z_candidates(7, [[NaN, 0.0]], params, policies;
                                                       outer_reps=1, inner_train=1, inner_eval=1)
    @test_throws ArgumentError evaluate_z_candidates(7, candidates, params, Policy[];
                                                       outer_reps=1, inner_train=1, inner_eval=1)
    @test_throws ArgumentError evaluate_z_candidates(7, candidates, params, policies;
                                                       outer_reps=0, inner_train=1, inner_eval=1)
    @test_throws ArgumentError evaluate_z_candidates(7, candidates, params, policies;
                                                       outer_reps=1, inner_train=0, inner_eval=1)
    @test_throws ArgumentError evaluate_z_candidates(7, candidates, params, policies;
                                                       outer_reps=1, inner_train=1, inner_eval=0)

    seed_a = AssemblyLineModel._candidate_seed(771, [0.25, 0.5])
    @test seed_a == AssemblyLineModel._candidate_seed(771, [0.25, 0.5])
    @test seed_a != AssemblyLineModel._candidate_seed(772, [0.25, 0.5])
    @test seed_a != AssemblyLineModel._candidate_seed(771, [0.5, 0.25])

    direct = evaluate_z(MersenneTwister(seed_a), [0.25, 0.5], params, policies;
                        outer_reps=2, inner_train=2, inner_eval=2)
    delegated = evaluate_z_candidates(771, [[0.25, 0.5]], params, policies;
                                      outer_reps=2, inner_train=2, inner_eval=2)
    @test delegated.evaluations == (direct,)
    @test delegated.candidate_seeds == (seed_a,)
    @test delegated.ranking == (1,)
    @test delegated.best_index == 1 && delegated.best_z == (0.25, 0.5)
    @test delegated.best_evaluation == direct
    @test delegated.outer_reps == 2 && delegated.inner_train == 2 && delegated.inner_eval == 2
    @test delegated.master_seed == 771

    ordered = evaluate_z_candidates(772, candidates, params, policies;
                                    outer_reps=1, inner_train=1, inner_eval=1)
    reordered = evaluate_z_candidates(772, reverse(candidates), params, policies;
                                      outer_reps=1, inner_train=1, inner_eval=1)
    for candidate in ((0.0, 0.0), (0.25, 0.5))
        ordered_index = findfirst(==(candidate), ordered.candidates)
        reordered_index = findfirst(==(candidate), reordered.candidates)
        @test ordered.candidate_seeds[ordered_index] == reordered.candidate_seeds[reordered_index]
        @test ordered.evaluations[ordered_index] == reordered.evaluations[reordered_index]
    end
    @test ordered.best_z == reordered.best_z

    duplicated = [[0.0, 0.0], [0.5, 0.5], [0.0, 0.0]]
    duplicate_result = evaluate_z_candidates(773, duplicated, params, policies;
                                             outer_reps=1, inner_train=1, inner_eval=1)
    @test duplicate_result.candidates == ((0.0, 0.0), (0.5, 0.5))
    @test duplicate_result.original_to_unique == (1, 2, 1)
    @test length(duplicate_result.evaluations) == 2
    @test sort(collect(duplicate_result.ranking)) == [1, 2]

    fake(z; gross=10.0, cost=1.0, objective=gross - cost, se=0.5) =
        ZEvaluationResult(z, gross, cost, objective, se, [gross], [1.0], 1)
    rank_for(candidates, evaluations) = AssemblyLineModel._rank_candidates(candidates, evaluations)
    @test rank_for([(0.0,), (0.5,)], [fake([0.0]; objective=1.0), fake([0.5]; objective=2.0)]) == [2, 1]
    @test rank_for([(0.0,), (0.5,)], [fake([0.0]; gross=2.0, objective=1.0),
                                       fake([0.5]; gross=3.0, objective=1.0)]) == [2, 1]
    @test rank_for([(0.0,), (0.5,)], [fake([0.0]; se=0.4), fake([0.5]; se=0.3)]) == [2, 1]
    @test rank_for([(0.0,), (0.5,)], [fake([0.0]; cost=0.4), fake([0.5]; cost=0.3)]) == [2, 1]
    @test rank_for([(0.5, 0.0), (0.0, 0.5)], [fake([0.5, 0.0]), fake([0.0, 0.5])]) == [2, 1]
    @test rank_for([(0.0,), (0.0,)], [fake([0.0]), fake([0.0])]) == [1, 2]

    coordinates = [[0.0, 0.5, 0.0], [0.0, 1.0]]
    coordinates_before = deepcopy(coordinates)
    @test build_z_grid([[0.0, 0.5]], ModelParameters([params.lines[1]], [1.0], 1.0)) ==
          [[0.0], [0.5]]
    @test build_z_grid(coordinates, params) == [[0.0, 0.0], [0.0, 1.0],
                                                 [0.5, 0.0], [0.5, 1.0]]
    @test length(build_z_grid([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]],
                              ModelParameters([params.lines..., params.lines[1]], [1.0, 1.0, 1.0], 1.0))) == 8
    @test coordinates == coordinates_before
    @test_throws ArgumentError build_z_grid([[0.0]], params)
    @test_throws ArgumentError build_z_grid([Float64[], [0.0]], params)
    @test_throws ArgumentError build_z_grid([[0.0], [Inf]], params)
    @test length(build_z_grid([[0.0, 0.5], [0.0, 1.0]], params; max_candidates=4)) == 4
    @test_throws ArgumentError build_z_grid([[0.0, 0.5], [0.0, 1.0]], params; max_candidates=3)
    @test_throws ArgumentError build_z_grid([[0.0], [0.0]], params; max_candidates=0)

    candidate_copy = deepcopy(candidates)
    policies_before = copy(policies)
    params_before = params
    reproducible_a = finite_grid_search(774, [[0.0, 0.25], [0.0, 0.5]], params, policies;
                                        outer_reps=1, inner_train=1, inner_eval=1)
    reproducible_b = finite_grid_search(774, [[0.0, 0.25], [0.0, 0.5]], params, policies;
                                        outer_reps=1, inner_train=1, inner_eval=1)
    @test reproducible_a == reproducible_b
    @test candidates == candidate_copy && policies == policies_before && params == params_before
    mutable_result = evaluate_z_candidates(775, candidates, params, policies;
                                           outer_reps=1, inner_train=1, inner_eval=1)
    candidates[1][1] = 1.0
    @test mutable_result.candidates[1] == (0.0, 0.0)
    candidates[1][1] = candidate_copy[1][1]
    Random.seed!(776)
    expected_random = rand()
    Random.seed!(776)
    evaluate_z_candidates(776, candidate_copy, params, policies;
                          outer_reps=1, inner_train=1, inner_eval=1)
    @test rand() == expected_random

    empty_line = example_line(base_arrival_rate=1e-20)
    cost_params = ModelParameters([empty_line], [3.0], 1.0)
    cost_result = evaluate_z_candidates(777, [[0.0], [0.5]], cost_params, policies;
                                        outer_reps=1, inner_train=1, inner_eval=1)
    @test [evaluation.gross_operational_value for evaluation in cost_result.evaluations] == [0.0, 0.0]
    @test cost_result.best_z == (0.0,)
    @test cost_result.best_evaluation.objective == 0.0
end

@testset "Stage 8: deterministic finite routing-policy families" begin
    single = build_policy_grid([0.4], [0.1], [0.0])
    @test single == [Policy(0.4, 0.1, 0.0)]
    @test eltype(single) == Policy
    @test policy_grid_size([0.4], [0.1], [0.0]) == 1

    av = [0.4, 0.8]
    aq = [0.1, 0.3]
    tau = [0.0, 0.5]
    policies = build_policy_grid(av, aq, tau)
    expected = [
        Policy(0.4, 0.1, 0.0), Policy(0.4, 0.1, 0.5),
        Policy(0.4, 0.3, 0.0), Policy(0.4, 0.3, 0.5),
        Policy(0.8, 0.1, 0.0), Policy(0.8, 0.1, 0.5),
        Policy(0.8, 0.3, 0.0), Policy(0.8, 0.3, 0.5),
    ]
    @test policies == expected
    @test length(policies) == 8 == policy_grid_size(av, aq, tau)

    duplicated = build_policy_grid([0.8, 0.4, 0.8], [0.3, 0.1, 0.3],
                                   [0.5, 0.0, 0.5])
    @test duplicated == [
        Policy(0.8, 0.3, 0.5), Policy(0.8, 0.3, 0.0),
        Policy(0.8, 0.1, 0.5), Policy(0.8, 0.1, 0.0),
        Policy(0.4, 0.3, 0.5), Policy(0.4, 0.3, 0.0),
        Policy(0.4, 0.1, 0.5), Policy(0.4, 0.1, 0.0),
    ]
    @test length(unique((policy.av, policy.aq, policy.tau) for policy in duplicated)) == length(duplicated)
    @test policy_grid_size([0.8, 0.4, 0.8], [0.3, 0.1, 0.3], [0.5, 0.0, 0.5]) == 8

    @test_throws ArgumentError build_policy_grid(Float64[], [0.0], [0.0])
    @test_throws ArgumentError build_policy_grid([0.0], Float64[], [0.0])
    @test_throws ArgumentError build_policy_grid([0.0], [0.0], Float64[])
    @test_throws ArgumentError build_policy_grid([-0.1], [0.0], [0.0])
    @test_throws ArgumentError build_policy_grid([0.0], [-0.1], [0.0])
    @test_throws ArgumentError build_policy_grid([NaN], [0.0], [0.0])
    @test_throws ArgumentError build_policy_grid([0.0], [Inf], [0.0])
    @test_throws ArgumentError build_policy_grid([0.0], [0.0], [-Inf])
    @test_throws ArgumentError build_policy_grid([0.0], [0.0], [0.0]; max_policies=0)
    invalid_error = try
        build_policy_grid([0.0, -0.1], [0.0], [0.0])
        ""
    catch error
        sprint(showerror, error)
    end
    @test occursin("av[2] = -0.1", invalid_error)

    @test length(build_policy_grid([0.0, 1.0], [0.0], [0.0, 1.0]; max_policies=4)) == 4
    @test_throws ArgumentError build_policy_grid([0.0, 1.0], [0.0], [0.0, 1.0]; max_policies=3)
    @test length(build_policy_grid([0.0, 0.0], [0.0], [0.0, 1.0]; max_policies=2)) == 2
    @test_throws ArgumentError AssemblyLineModel._policy_grid_count(Base.OneTo(typemax(Int)), [0], [0, 1])

    av_before, aq_before, tau_before = copy(av), copy(aq), copy(tau)
    copied = build_policy_grid(av, aq, tau)
    av[1] = 9.0
    aq[1] = 9.0
    tau[1] = 9.0
    @test copied == expected
    @test build_policy_grid(av_before, aq_before, tau_before) == expected
    Random.seed!(8_001)
    expected_random = rand()
    Random.seed!(8_001)
    build_policy_grid(av_before, aq_before, tau_before)
    @test rand() == expected_random

    params = routing_params(horizon=1.0)
    path = ShiftPath([ReworkJob(0.0, 1, false, 1.0, 2.0),
                      ReworkJob(0.3, 2, true, 1.5, 3.0)])
    routing_policies = build_policy_grid([0.0, 1.0], [0.0], [0.0, 0.5])
    @test evaluate_policies_on_path(routing_policies, path, params) ==
          [evaluate_policy(policy, path, params) for policy in routing_policies]

    posterior = PosteriorState([0.0, 0.0], [0.5, 0.5], [true, true])
    values = estimate_policy_values(MersenneTwister(8_002), posterior, params, routing_policies, 2)
    selection = select_policy_given_posterior(MersenneTwister(8_002), posterior, params,
                                               routing_policies, 2)
    @test selection.training_values == Tuple(values)
    @test selection.selected_index == argmax(values)
    @test selection.selected_policy == routing_policies[selection.selected_index]

    empty_line = example_line(base_arrival_rate=1e-20)
    empty_params = ModelParameters([empty_line], [1.0], 1.0)
    empty_posterior = PosteriorState([0.0], [1e-12], [false])
    tied_policies = build_policy_grid([0.0, 1.0], [0.0], [0.0])
    tied_selection = select_policy_given_posterior(MersenneTwister(8_003), empty_posterior,
                                                    empty_params, tied_policies, 2)
    @test tied_selection.training_values == (0.0, 0.0)
    @test tied_selection.selected_index == 1

    evaluation = evaluate_z(MersenneTwister(8_004), [0.2, 0.4], params, routing_policies;
                            outer_reps=2, inner_train=1, inner_eval=1)
    @test all(1 <= index <= length(routing_policies) for index in evaluation.selected_policy_indices)
    @test length(evaluation.selected_policy_frequencies) == length(routing_policies)
    @test sum(evaluation.selected_policy_counts) == 2

    search = evaluate_z_candidates(8_005, [[0.0, 0.0], [0.2, 0.4]], params,
                                   routing_policies; outer_reps=1, inner_train=1, inner_eval=1)
    @test length(search.evaluations) == 2
    @test all(length(result.selected_policy_frequencies) == length(routing_policies)
              for result in search.evaluations)
end

@testset "Stage 9: reproducible end-to-end reference experiment" begin
    smoke = ReferenceExperiment.reference_configuration(:smoke)
    reference = ReferenceExperiment.reference_configuration(:reference)
    @test nlines(smoke.params) == 2
    @test length(smoke.policies) == 12
    @test length(smoke.z_grid) == 9
    @test (0.0, 0.0) in smoke.z_grid
    @test smoke.outer_reps == 5 && smoke.inner_train == 20 && smoke.inner_eval == 30
    @test reference.outer_reps == 30 && reference.inner_train == 100 && reference.inner_eval == 200
    @test_throws ArgumentError ReferenceExperiment.reference_configuration(:unknown)
    @test_throws ArgumentError ReferenceExperiment.reference_configuration(:unknown;
                                                                            sample_sizes=(outer_reps=1, inner_train=1, inner_eval=1))

    defaults = ReferenceExperiment.parse_cli()
    @test defaults.mode == :smoke && defaults.master_seed == 20_260_726
    @test ReferenceExperiment.parse_cli(["--mode=reference"]).mode == :reference
    parsed = ReferenceExperiment.parse_cli(["--mode=smoke", "--seed=42", "--output-dir=temporary/results"])
    @test parsed == (mode=:smoke, master_seed=42, output_dir="temporary/results")
    @test_throws ArgumentError ReferenceExperiment.parse_cli(["--unknown=value"])
    @test_throws ArgumentError ReferenceExperiment.parse_cli(["--seed=not-an-int"])
    @test_throws ArgumentError ReferenceExperiment.parse_cli(["--mode=invalid"])
    @test_throws ArgumentError ReferenceExperiment.parse_cli(["--mode=smoke", "--mode=reference"])

    mktempdir() do temporary_root
        output_one = joinpath(temporary_root, "nested", "one")
        config_one = ReferenceExperiment.reference_configuration(:smoke;
            master_seed=9_001, output_dir=output_one,
            sample_sizes=(outer_reps=1, inner_train=1, inner_eval=1))
        direct = finite_grid_search(config_one.seed, [collect(values) for values in config_one.z_coordinate_values],
                                    config_one.params, collect(config_one.policies);
                                    outer_reps=config_one.outer_reps,
                                    inner_train=config_one.inner_train,
                                    inner_eval=config_one.inner_eval)
        execution_one = ReferenceExperiment.execute_reference_experiment(config_one)
        @test execution_one.result isa FiniteGridSearchResult
        @test execution_one.result == direct
        @test length(execution_one.result.candidates) == 9
        @test length(execution_one.result.ranking) == 9
        @test all(isfinite(evaluation.objective) for evaluation in execution_one.result.evaluations)
        @test execution_one.result.best_index == execution_one.result.ranking[1]
        @test isfile(execution_one.paths.csv_path) && isfile(execution_one.paths.markdown_path)

        csv_one = read(execution_one.paths.csv_path, String)
        csv_lines = split(chomp(csv_one), '\n')
        header = split(first(csv_lines), ',')
        @test length(csv_lines) == 10
        @test header[1:3] == ["rank", "candidate_index", "candidate_seed"]
        @test all(column in header for column in ["z_1", "z_2", "gross_value", "probing_cost",
                                                   "objective", "outer_standard_error", "outer_reps",
                                                   "inner_train", "inner_eval", "dominant_policy_index",
                                                   "dominant_policy_frequency"])
        first_row = split(csv_lines[2], ',')
        candidate_column = findfirst(==("candidate_index"), header)
        objective_column = findfirst(==("objective"), header)
        @test parse(Int, first_row[candidate_column]) == execution_one.result.best_index
        @test parse(Float64, first_row[objective_column]) == execution_one.result.best_evaluation.objective

        markdown_one = read(execution_one.paths.markdown_path, String)
        @test occursin("# Grid Experiment Report", markdown_one)
        @test occursin("## Resolved configuration", markdown_one)
        @test occursin("## Ranked candidates", markdown_one)
        @test occursin("## Best candidate", markdown_one)
        @test occursin("## Interpretation limits", markdown_one)
        @test occursin("Objective", markdown_one)
        @test occursin("outer=1, train=1, eval=1", markdown_one)
        @test occursin("Monte Carlo noise", markdown_one)

        keep_path = joinpath(output_one, "keep.txt")
        write(keep_path, "preserve")
        ReferenceExperiment.write_experiment_outputs(config_one, execution_one.result)
        @test read(keep_path, String) == "preserve"
        @test read(execution_one.paths.csv_path, String) == csv_one
        @test read(execution_one.paths.markdown_path, String) == markdown_one

        output_two = joinpath(temporary_root, "nested", "two")
        config_two = ReferenceExperiment.reference_configuration(:smoke;
            master_seed=9_001, output_dir=output_two,
            sample_sizes=(outer_reps=1, inner_train=1, inner_eval=1))
        execution_two = ReferenceExperiment.execute_reference_experiment(config_two)
        @test execution_two.result == execution_one.result
        @test read(execution_two.paths.csv_path, String) == csv_one
        @test read(execution_two.paths.markdown_path, String) == markdown_one

        different_seed = ReferenceExperiment.reference_configuration(:smoke;
            master_seed=9_002, output_dir=joinpath(temporary_root, "different"),
            sample_sizes=(outer_reps=1, inner_train=1, inner_eval=1))
        @test ReferenceExperiment.run_reference_experiment(different_seed) != execution_one.result
        @test_throws ArgumentError ReferenceExperiment.write_experiment_outputs(
            ReferenceExperiment.reference_configuration(:smoke; output_dir=joinpath(@__DIR__, "..", "src")),
            execution_one.result)
    end
end

@testset "Stage 10: validation-script regression checks" begin
    plan = ConvergenceValidation.convergence_plan()
    @test length(plan) == 15
    @test Set(Tuple(specification.z) for specification in plan if specification.study == "outer") ==
          Set([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    outer_stream_keys = Set(specification.stream_key for specification in plan if specification.study == "outer")
    @test outer_stream_keys == Set([1, 2, 3])
    @test length(Set(ConvergenceValidation._z_stream_seed(10_000, key) for key in outer_stream_keys)) == 3
    @test all(specification.outer_reps > 0 && specification.inner_train > 0 &&
              specification.inner_eval > 0 for specification in plan)

    mktempdir() do directory
        tiny_plan = [(study="regression", z=[0.0, 1.0], outer_reps=1,
                      inner_train=1, inner_eval=1)]
        validation = ConvergenceValidation.run_convergence_validation(
            master_seed=10_001, output_path=joinpath(directory, "convergence.csv"), plan=tiny_plan)
        configuration = ReferenceExperiment.reference_configuration(:smoke;
            sample_sizes=(outer_reps=1, inner_train=1, inner_eval=1))
        direct_seed = ConvergenceValidation._z_stream_seed(10_001, 1)
        direct = evaluate_z(MersenneTwister(direct_seed), [0.0, 1.0], configuration.params,
                            collect(configuration.policies); outer_reps=1, inner_train=1, inner_eval=1)
        @test length(validation.rows) == 1
        @test validation.rows[1].objective == direct.objective
        @test isfile(validation.output_path)
        @test length(readlines(validation.output_path)) == 2
    end
end

@testset "Generic TOML grid-experiment runner" begin
    config_path = joinpath(@__DIR__, "..", "configs", "reference_experiment.toml")
    config = ExperimentRunner.load_experiment_config(config_path)
    @test config isa ExperimentRunner.GridExperimentConfig
    @test config.name == "reference_experiment" && config.seed == 20_260_726
    @test nlines(config.params) == 2 && length(config.policies) == 12 && length(config.z_grid) == 9
    @test config.z_coordinate_values == ((0.0, 0.5, 1.0), (0.0, 0.5, 1.0))
    @test config.outer_reps == 5 && config.inner_train == 20 && config.inner_eval == 30

    parsed = ExperimentRunner.parse_cli(["--config=$config_path", "--seed", "42", "--z-step=0.25"])
    @test !parsed.help && parsed.values[:seed] == "42" && parsed.values[:z_step] == "0.25"
    @test ExperimentRunner.parse_cli(["--help"]).help
    @test_throws ArgumentError ExperimentRunner.parse_cli(["--unknown=1"])
    @test_throws ArgumentError ExperimentRunner.parse_cli(["--z-step=0.5", "--z-values=0,1;0,1"])
    @test_throws ArgumentError ExperimentRunner.parse_cli(["--seed"])

    for (step, count) in (("1.0", 2), ("0.5", 3), ("0.25", 5), ("0.05", 21))
        stepped = ExperimentRunner.load_experiment_config(config_path;
            overrides=Dict(:z_step => step))
        @test length(stepped.z_coordinate_values[1]) == count
        @test first(stepped.z_coordinate_values[1]) == 0.0
        @test last(stepped.z_coordinate_values[1]) == 1.0
        @test length(stepped.z_grid) == count^2
    end
    @test_throws ArgumentError ExperimentRunner.load_experiment_config(config_path;
        overrides=Dict(:z_step => "0.3"))
    @test_throws ArgumentError ExperimentRunner.load_experiment_config(config_path;
        overrides=Dict(:z_values => "0,0.5,1"))

    explicit = ExperimentRunner.load_experiment_config(config_path;
        overrides=Dict(:z_values => "0,0.5,1;0,0.25,1", :av_values => "1,0.5",
                       :max_candidates => "100", :max_policies => "12", :seed => "99"))
    @test explicit.seed == 99
    @test explicit.z_coordinate_values == ((0.0, 0.5, 1.0), (0.0, 0.25, 1.0))
    @test length(explicit.z_grid) == 9
    @test [policy.av for policy in explicit.policies[1:3]] == [1.0, 1.0, 1.0]
    @test_throws ArgumentError ExperimentRunner.load_experiment_config(config_path;
        overrides=Dict(:aq_values => "-0.1"))

    mktempdir() do directory
        source = read(config_path, String)
        missing_model = joinpath(directory, "missing_model.toml")
        write(missing_model, replace(source, "[model]" => "[model_missing]"; count=1))
        @test_throws ArgumentError ExperimentRunner.load_experiment_config(missing_model)
        wrong_costs = joinpath(directory, "wrong_costs.toml")
        write(wrong_costs, replace(source, "probing_costs = [0.6, 1.0]" => "probing_costs = [0.6]"; count=1))
        @test_throws ArgumentError ExperimentRunner.load_experiment_config(wrong_costs)
        unknown_line = joinpath(directory, "unknown_line.toml")
        write(unknown_line, replace(source, "standard_complex_mean = 4.80" =>
                                    "standard_complex_mean = 4.80\nunexpected_line_field = 1"; count=1))
        @test_throws ArgumentError ExperimentRunner.load_experiment_config(unknown_line)
        missing_line = joinpath(directory, "missing_line.toml")
        write(missing_line, replace(source, "fast_simple_mean = 0.70\n" => ""; count=1))
        @test_throws ArgumentError ExperimentRunner.load_experiment_config(missing_line)
        three_line = joinpath(directory, "three_line.toml")
        three_source = replace(source, "probing_costs = [0.6, 1.0]" => "probing_costs = [0.6, 1.0, 1.2]"; count=1) * """

[[lines]]
value = 0.55
prior_mean = 0.0
prior_variance = 1.0
diagnostic_noise = 1.5
max_diagnostic_time = 4.0
baseline_arrival_rate = 0.70
arrival_sensitivity = 0.20
complexity_intercept = -0.50
complexity_sensitivity = 0.60
fast_simple_mean = 0.70
fast_complex_mean = 1.90
standard_simple_mean = 0.90
standard_complex_mean = 4.50
"""
        write(three_line, three_source)
        three_config = ExperimentRunner.load_experiment_config(three_line;
            overrides=Dict(:z_values => "0,1;0,0.5,1;0,1"))
        @test nlines(three_config.params) == 3
        @test three_config.z_coordinate_values == ((0.0, 1.0), (0.0, 0.5, 1.0), (0.0, 1.0))
        @test length(three_config.z_grid) == 12

        output_dir = joinpath(directory, "nested", "run")
        tiny = ExperimentRunner.load_experiment_config(config_path;
            overrides=Dict(:output_dir => output_dir, :seed => "123", :outer_reps => "1",
                           :inner_train => "1", :inner_eval => "1", :z_values => "0,1;0,1",
                           :av_values => "0,1", :aq_values => "0", :tau_values => "0"))
        direct = finite_grid_search(tiny.seed, [collect(values) for values in tiny.z_coordinate_values],
                                    tiny.params, collect(tiny.policies); outer_reps=1,
                                    inner_train=1, inner_eval=1, max_candidates=tiny.candidate_max)
        execution = ExperimentRunner.execute_grid_experiment(tiny)
        @test execution.result == direct
        @test length(execution.result.candidates) == 4 && length(tiny.policies) == 2
        @test all(isfile, [execution.paths.csv_path, execution.paths.markdown_path, execution.paths.resolved_path])
        @test length(readlines(execution.paths.csv_path)) == 5
        resolved = read(execution.paths.resolved_path, String)
        @test occursin("seed = 123", resolved) && occursin("coordinate_values", resolved)
        keep_path = joinpath(output_dir, "keep.txt")
        write(keep_path, "keep")
        ExperimentRunner.write_experiment_outputs(tiny, execution.result)
        @test read(keep_path, String) == "keep"
        blocked_path = joinpath(directory, "regular_file")
        write(blocked_path, "not a directory")
        @test_throws ArgumentError ExperimentRunner.write_experiment_outputs(
            ExperimentRunner.load_experiment_config(config_path; overrides=Dict(:output_dir => blocked_path)), execution.result)
        @test_throws ArgumentError ExperimentRunner.write_experiment_outputs(
            ExperimentRunner.load_experiment_config(config_path; overrides=Dict(:output_dir => joinpath(@__DIR__, "..", "src"))), execution.result)
    end
end
