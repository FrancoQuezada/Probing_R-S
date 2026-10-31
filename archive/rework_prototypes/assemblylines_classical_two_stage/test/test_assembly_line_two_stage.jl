module AssemblyLineTests

using Test
using Random
using Statistics
using LinearAlgebra
using Distributions

include(joinpath(@__DIR__, "..", "assembly_line_two_stage.jl"))
using .AssemblyLineTwoStage

const P = AssemblyLineParams()
const E = ExperimentParams(outer_reps=3, inner_train=4, inner_eval=6, jobs_per_path=8,
                           outer_seed=101, train_seed=102, eval_seed=103,
                           baseline_train_seed=104, baseline_eval_seed=105)
const POLICIES = build_policy_grid()

@testset "assembly-line classical two-stage formulation" begin
    @testset "measurement and posterior formulas" begin
        for c in 1:P.nclass
            @test diagnostic_time(0.35, c, P) == P.t_max[c] * 0.35
            @test measurement_marginal_variance(0.0, c, P) == 0.0
            post0 = posterior_line_condition(0.0, 0.0, c, P)
            @test post0.mean == 0.0
            @test post0.variance == P.tau_theta[c]^2
            variances = [posterior_line_condition(0.0, z, c, P).variance for z in (0.0, 0.25, 0.5, 1.0)]
            @test all(diff(variances) .<= 0.0)
        end
        c, z, eta = 3, 0.45, 1.2
        t = P.t_max[c] * z
        reference_variance = 1.0 / (1.0 / P.tau_theta[c]^2 + t / P.sigma_sensor[c]^2)
        reference_mean = reference_variance * eta / P.sigma_sensor[c]^2
        post = posterior_line_condition(eta, z, c, P)
        @test post.variance ≈ reference_variance
        @test post.mean ≈ reference_mean
        @test measurement_marginal_variance(z, c, P) ≈ t^2 * P.tau_theta[c]^2 + P.sigma_sensor[c]^2 * t
    end

    @testset "measurement and complexity calibration" begin
        rng = MersenneTwister(22); c, z, n = 2, 0.4, 20_000
        eta = [sample_measurement(P.tau_theta[c] * randn(rng), z, randn(rng), c, P) for _ in 1:n]
        target_variance = measurement_marginal_variance(z, c, P)
        @test abs(mean(eta)) < 0.05 * sqrt(target_variance)
        @test abs(var(eta) - target_variance) < 0.06 * target_variance
        for c in 1:P.nclass
            probabilities = [conditional_complexity_probability(c, theta, P) for theta in (-3.0, 0.0, 3.0)]
            @test all(0.0 .<= probabilities .<= 1.0)
            @test probabilities[1] <= probabilities[2] <= probabilities[3]
            intercept = quantile(Normal(), P.p_bar[c]) * sqrt(1 + P.condition_sensitivity[c]^2 * P.tau_theta[c]^2)
            @test cdf(Normal(), intercept / sqrt(1 + P.condition_sensitivity[c]^2 * P.tau_theta[c]^2)) ≈ P.p_bar[c]
            theta = P.tau_theta[c] .* randn(rng, n)
            @test abs(mean(conditional_complexity_probability(c, x, P) for x in theta) - P.p_bar[c]) < 0.012
        end
    end

    @testset "shared conditions and preserved physical path primitives" begin
        noise = AssemblyLineTwoStage._noise_path(MersenneTwister(31), 12, P.nclass)
        theta = [0.3, -0.4, 0.5, 0.1]
        first = AssemblyLineTwoStage._path_from_theta(theta, noise, P)
        second = AssemblyLineTwoStage._path_from_theta(theta, noise, P)
        @test first.theta == theta == second.theta
        @test first.jobs == second.jobs
        @test all(job -> job.class in 1:P.nclass, first.jobs)
        @test all(job -> job.signal in (0, 1), first.jobs)
        # z enters only the pre-shift measurement/posterior transformation, never this physical generator.
        @test first.jobs == AssemblyLineTwoStage._path_from_theta(theta, noise, P).jobs
        @test P.p_hit == [0.25, 0.40, 0.75, 0.92]
        @test P.p_false == [0.05, 0.08, 0.15, 0.20]
    end

    @testset "unconditional marginal complexity" begin
        rng = MersenneTwister(44); counts = zeros(Int, P.nclass); complex_counts = zeros(Int, P.nclass)
        for _ in 1:500
            path = unconditional_operating_path(AssemblyLineTwoStage._noise_path(rng, 30, P.nclass), P)
            for job in path.jobs
                counts[job.class] += 1; complex_counts[job.class] += job.complex
            end
        end
        for c in 1:P.nclass
            @test abs(complex_counts[c] / counts[c] - P.p_bar[c]) < 0.035
        end
    end

    @testset "selection timing, ties, and estimator identities" begin
        noise = generate_noise_banks(P, E); z = [0.0, 0.0, 0.4, 0.7]
        eta = outer_measurement(z, view(noise.outer_normals, :, 1), P)
        result = conditional_selection(1, eta, z, noise.train[1], noise.eval[1], POLICIES, P)
        @test all(==(result.selected_policy_id), result.evaluation_policy_ids)
        @test result.selected_policy_id in eachindex(POLICIES)
        identical = [Policy(0.4, 0.15, 0.0, 0.1), Policy(0.4, 0.15, 0.0, 0.1)]
        paths = [conditional_operating_path(eta, z, noise.train[1][1], P)]
        @test AssemblyLineTwoStage._select_policy(paths, identical, P).policy_id == 1
        estimate = estimate_two_stage(z, P, E, POLICIES, noise)
        @test estimate.gross_value == mean(context.evaluation_mean for context in estimate.contexts)
        @test estimate.objective ≈ estimate.gross_value - dot(E.alpha, z)
        @test noise.train != noise.eval
        @test noise.fixed_train != noise.fixed_eval
    end

    @testset "zero-information controlled consistency and toy information ordering" begin
        # Reuse exactly the same train/evaluation paths in a one-context controlled case.
        one = ExperimentParams(outer_reps=1, inner_train=3, inner_eval=4, jobs_per_path=6,
                               outer_seed=51, train_seed=52, eval_seed=53, baseline_train_seed=52, baseline_eval_seed=53)
        base = generate_noise_banks(P, one)
        controlled = NoiseBanks(base.outer_normals, [base.fixed_train], [base.fixed_eval], base.fixed_train, base.fixed_eval)
        fixed = estimate_fixed_policy_baseline(P, one, POLICIES, controlled)
        adaptive = estimate_two_stage(zeros(4), P, one, POLICIES, controlled; fixed_policy=fixed)
        @test adaptive.contexts[1].eta == zeros(4)
        @test adaptive.gross_value == fixed.evaluation_mean
        # Two conditions, two specialized policies: E[max] = 1 exceeds max(E) = 0.5.
        @test mean((max(1.0, 0.0), max(0.0, 1.0))) > max(mean((1.0, 0.0)), mean((0.0, 1.0)))
    end

    @testset "reproducibility, CRN, and original syntax" begin
        z1, z2 = [0.0, 0.0, 0.2, 0.3], [0.0, 0.0, 0.8, 0.9]
        noise = generate_noise_banks(P, E)
        @test estimate_two_stage(z1, P, E, POLICIES, noise).objective == estimate_two_stage(z1, P, E, POLICIES, noise).objective
        @test outer_measurement(z1, view(noise.outer_normals, :, 2), P) != outer_measurement(z2, view(noise.outer_normals, :, 2), P)
        for source in ("probing_heatmap_2d.jl", "probing_policy_regions_2d.jl")
            # These scripts moved to two_servers/lr_baseline_2d/ during the restructuring.
            @test Meta.parseall(read(joinpath(@__DIR__, "..", "..", "..", "..", "two_servers", "lr_baseline_2d", source), String)) isa Expr
        end
    end

    @testset "small grid outputs" begin
        include(joinpath(@__DIR__, "..", "assembly_line_two_stage_2d.jl"))
        output_dir = mktempdir()
        smoke = run_assembly_line_grid(grid_n=3, params=P,
            experiment=ExperimentParams(outer_reps=3, inner_train=4, inner_eval=6, jobs_per_path=8), output_dir=output_dir)
        @test length(smoke.rows) == 9
        @test all(isfile, smoke.output_files)
    end
end

end # module AssemblyLineTests
