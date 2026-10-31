using Random
using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "single_job_two_stage.jl"))
using .SingleJobTwoStage

const PARAMS = ModelParams()
const ZERO_Z = zeros(4)

@testset "single-job two-stage formulation" begin
    @testset "signal probabilities" begin
        for c in 1:PARAMS.nclass
            hits = [p_hit(c, zc, PARAMS) for zc in (0.0, 0.25, 0.5, 0.75, 1.0)]
            false_positives = [p_false(c, zc, PARAMS) for zc in (0.0, 0.25, 0.5, 0.75, 1.0)]
            @test all(0.0 .<= hits .<= 1.0)
            @test all(0.0 .<= false_positives .<= 1.0)
            @test all(diff(hits) .>= -eps())
            @test all(diff(false_positives) .<= eps())
        end

        c, zc = 3, 0.4
        rho = PARAMS.p_complex[c]
        manual = rho * p_hit(c, zc, PARAMS) + (1.0 - rho) * p_false(c, zc, PARAMS)
        @test marginal_signal_probability(c, 1, zc, PARAMS) ≈ manual
        @test marginal_signal_probability(c, 0, zc, PARAMS) ≈ 1.0 - manual
    end

    @testset "Bayes posterior" begin
        c, zc = 3, 0.35
        rho, ph, pf = PARAMS.p_complex[c], p_hit(c, zc, PARAMS), p_false(c, zc, PARAMS)
        direct_positive = rho * ph / (rho * ph + (1.0 - rho) * pf)
        direct_negative = rho * (1.0 - ph) / (rho * (1.0 - ph) + (1.0 - rho) * (1.0 - pf))
        @test posterior_complex_probability(c, 1, zc, PARAMS) ≈ direct_positive
        @test posterior_complex_probability(c, 0, zc, PARAMS) ≈ direct_negative
        for c in 1:PARAMS.nclass, zc in (0.0, 0.5, 1.0)
            @test posterior_complex_probability(c, 1, zc, PARAMS) >=
                  posterior_complex_probability(c, 0, zc, PARAMS)
        end
    end

    @testset "direct action comparison and no hidden-state leakage" begin
        context = JobContext(3, 0.65, 0.3, 1.1)
        actions = evaluate_actions(context, 1, ZERO_Z, PARAMS)
        @test actions.selected_action == (actions.value_fast >= actions.value_standard ? FAST : STANDARD)
        @test select_action(context, 1, ZERO_Z, PARAMS) == actions.selected_action

        # Complexity is intentionally absent from the action-selection API.
        action_if_simple = select_action(context, 1, ZERO_Z, PARAMS)
        action_if_complex = select_action(context, 1, ZERO_Z, PARAMS)
        @test action_if_simple == action_if_complex
        @test evaluate_actions(context, 1, ZERO_Z, PARAMS) == evaluate_actions(context, 1, ZERO_Z, PARAMS)
    end

    @testset "exact binary-signal expectation and information value" begin
        contexts = (JobContext(1, 0.2, 0.1, 0.7), JobContext(3, 0.6, 0.4, 0.9),
                    JobContext(4, 0.9, 1.0, 0.2))
        z = [0.0, 0.0, 0.5, 0.7]
        for context in contexts
            cv = context_value(context, z, PARAMS)
            manual = cv.p_signal0 * max(cv.value_fast_s0, cv.value_standard_s0) +
                     cv.p_signal1 * max(cv.value_fast_s1, cv.value_standard_s1)
            @test cv.value ≈ manual
            iv = context_information_value(context, z, PARAMS)
            @test iv.value_after_signal + 1e-12 >= iv.value_before_signal
            @test iv.information_value >= 0.0
        end
    end

    @testset "objective and CRN reproducibility" begin
        z = [0.0, 0.0, 0.25, 0.75]
        alpha = [0.1, 0.2, 0.3, 0.4]
        result = estimate_objective(z; reps=300, seed=91, alpha=alpha, params=PARAMS)
        @test result.objective ≈ result.gross_value - dot(alpha, z)
        repeated = estimate_objective(z; reps=300, seed=91, alpha=alpha, params=PARAMS)
        @test result.gross_value == repeated.gross_value
        @test result.objective == repeated.objective
        @test result.mean_information_value == repeated.mean_information_value

        noise = generate_noise_set(123, 300)
        crn_one = estimate_objective_given_noise(z, noise, alpha, PARAMS)
        crn_two = estimate_objective_given_noise(z, noise, alpha, PARAMS)
        @test crn_one == crn_two
    end

    @testset "tower-property validation" begin
        z = [0.0, 0.0, 0.4, 0.8]
        alpha = zeros(4)
        noise = generate_noise_set(2026, 40_000)
        exact = estimate_objective_given_noise(z, noise, alpha, PARAMS)
        realized = estimate_realized_objective_given_noise(z, noise, alpha, PARAMS)
        tolerance = 4.0 * sqrt(exact.standard_error^2 + realized.standard_error^2) + 0.02
        @test abs(exact.gross_value - realized.gross_value) <= tolerance
    end

    @testset "increasing effort changes beliefs and can change an action" begin
        context = JobContext(3, 0.6, 0.0, 0.9)
        low_effort = [0.0, 0.0, 0.0, 0.0]
        high_effort = [0.0, 0.0, 1.0, 0.0]
        @test posterior_complex_probability(3, 0, high_effort[3], PARAMS) <
              posterior_complex_probability(3, 0, low_effort[3], PARAMS)
        @test select_action(context, 0, low_effort, PARAMS) !=
              select_action(context, 0, high_effort, PARAMS)
    end

    @testset "source isolation" begin
        source = read(joinpath(@__DIR__, "..", "single_job_two_stage.jl"), String)
        for forbidden in (r"\bPolicy\b", r"\bbuild_policy_grid\b", r"\bgenerate_policies\b",
                          r"\bchoose_fast_be\b", r"\ba_demand\b", r"\ba_queue\b",
                          r"\ba_signal\b", r"\bthreshold\b")
            @test !occursin(forbidden, source)
        end
    end

    @testset "small grid writes all outputs" begin
        include(joinpath(@__DIR__, "..", "single_job_two_stage_2d.jl"))
        output_dir = mktempdir()
        grid_result = run_grid_experiment(grid_n=3, reps=200, seed=77, alpha=zeros(4), output_dir=output_dir)
        @test length(grid_result.rows) == 9
        @test all(isfile, grid_result.output_files)
    end
end
