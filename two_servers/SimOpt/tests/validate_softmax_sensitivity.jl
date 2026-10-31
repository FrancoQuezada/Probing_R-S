using Test, LinearAlgebra, Statistics, SHA, Random
include(joinpath(@__DIR__,"../004probing_softmax_sensitivity.jl"))

@testset "Parameters, stable weights and monotonic effective policies" begin
    cfg=configuration()
    @test (cfg.npaths,cfg.njobs,cfg.iterations,cfg.trace_paths,cfg.final_paths)==(500,45,100,5000,30000)
    @test cfg.betas==[.025,.05,.1,.2,.5,1.0]
    @test parse_arguments(["--betas","0.2,0.025"]).betas==[.025,.2]
    @test cfg.search_seed==20260721
    @test C.BETA==2.20 && C.Z0==[.12,.10]
    values=[-1000.,0.,1.,2.,3.,100.]
    keffs=Float64[]
    for b in cfg.betas
        w=C.softmax_weights(values;beta=b)
        @test sum(w)≈1 atol=1e-14
        @test all(isfinite,w) && all(>=(0),w)
        push!(keffs,inv(sum(abs2,w)))
    end
    @test all(diff(keffs).<=0)
    G=[12. 5.;9. 4.;8. 3.]
    est=C.combine_softmax_gradients([2.,1.,0.],G;beta=1000.)
    @test est.gradient≈G[1,:]-C.ALPHA atol=1e-14
    @test bytes2hex(sha256(read(joinpath(@__DIR__,"../003probing_four_searches_softmax.jl"))))=="cb084d435519328e77972b17812c9117d42b761d085ccaa4bd254af12e6e0a98"
end

@testset "Same-bank gradients, single cost, primitive CRN and update diagnostics" begin
    cfg=configuration(smoke=true);policies=C.build_policy_grid();z=[.31,.27]
    primitive=primitive_bank(cfg,50,123)
    bank=C.materialize_crn_bank(primitive,z)
    data=local_policy_data(z,bank,policies)
    Y=C.precompute_rewards(policies,bank)
    hard=C.one_bank_estimator(z,z,bank,Y)
    @test data.hard==hard[2]
    @test data.objective≈hard[1] atol=1e-12
    for beta in cfg.betas
        ours=smoothing_diagnostics(data,z,1,beta,cfg.epsilon)
        reference=C.softmax_estimator(z,bank,Y;beta)
        @test ours.gradient==reference.gradient
        @test ours.diagnostics.relative_difference≈norm(reference.gradient-hard[2])/max(norm(hard[2]),cfg.epsilon)
        @test ours.znew==C.projected_step(z,reference.gradient,1)[1]
        @test ours.diagnostics.weight_sum≈1 atol=1e-14
        @test all(x->0<=x<=1,ours.znew)
    end
    full=configuration();direct=C.generate_bank_at(z;npaths=3,seed=789)
    materialized=C.materialize_crn_bank(primitive_bank(full,3,789),z)
    @test all(direct[i].jobs==materialized[i].jobs for i in 1:3)
    cosine,status=safe_cosine(zeros(2),zeros(2),cfg.epsilon)
    @test ismissing(cosine) && status=="both_near_zero"
    cosine,status=safe_cosine(zeros(2),ones(2),cfg.epsilon)
    @test ismissing(cosine) && status=="soft_near_zero"
    zero_data=(values=[1.,0.],G=[10. 4.5;12. 4.5],hard=zeros(2),best=1,objective=0.)
    zero_est=smoothing_diagnostics(zero_data,z,1,.1,cfg.epsilon)
    @test isfinite(zero_est.diagnostics.relative_difference)
    @test ismissing(zero_est.diagnostics.cosine)
end

@testset "Execution order, exact Fresh reproduction and common CRN traces" begin
    cfg=configuration(smoke=true);policies=C.build_policy_grid()
    a,anchors,_=redirect_stdout(devnull) do
        run_searches(cfg,policies)
    end
    b,_,_=redirect_stdout(devnull) do
        run_searches(cfg,policies;order=collect(7:-1:1))
    end
    @test isequal(a,b)
    @test length(anchors)==cfg.iterations*6
    z=copy(C.Z0);hist=[copy(z)]
    for k in 1:cfg.iterations
        rng=MersenneTwister(cfg.search_seed+10000*k)
        bank=[C.generate_path(rng,z;njobs=cfg.njobs) for _ in 1:cfg.npaths]
        Y=C.precompute_rewards(policies,bank)
        _,g,_,_,_,_=C.one_bank_estimator(z,z,bank,Y)
        z=C.projected_step(z,g,k)[1];push!(hist,copy(z))
    end
    @test a[1].zhist==hist
    @test all(all(all(x->0<=x<=1,z) for z in r.zhist) for r in a)
    @test all(all(isfinite(d.g3)&&isfinite(d.g4) for d in r.diagnostics) for r in a)
    traces,_=redirect_stdout(devnull) do
        evaluate_traces(cfg,a,policies,nothing)
    end
    bank=primitive_bank(cfg,cfg.trace_paths,cfg.trace_seed)
    @test length(unique(t[1] for t in traces))==1
    for i in eachindex(a)
        @test traces[i]==C.evaluate_trace_on_common_crn(a[i].zhist,policies,bank)
    end
    oracle,pairs,rewards=redirect_stdout(devnull) do
        evaluate_endpoints(cfg,a,policies)
    end
    @test size(rewards)==(cfg.final_paths,7)
    @test length(pairs)==21
    firstpair=pairs[1]
    @test firstpair.paired_SE≈std(rewards[:,1]-rewards[:,2])/sqrt(cfg.final_paths)
    @test firstpair.difference≈oracle[1].objective-oracle[2].objective
end
