using Test, Random, LinearAlgebra
include(joinpath(@__DIR__,"../probing_regime_experiment.jl"))
const X=ProbingRegimeExperiment
const R=X.R
BLAS.set_num_threads(1)
@testset "Additional configurable economic and signal parameters" begin
    cfg=R.ModelConfig();z=[.3,.4];p=X.C.build_policy_grid()[62]
    path=R.materialize(R.primitive_bank(cfg,1,321)[1],z,cfg)
    detail=R.routing_details(p,path,cfg)
    highrev=R.ModelConfig(revenue_scale=16.)
    @test R.evaluate_policy(p,path,highrev)≈detail.reward+8sum(j.value for j in path.jobs) atol=1e-10
    highdelay=R.ModelConfig(delay_penalty=.84)
    @test R.evaluate_policy(p,path,highdelay)≈detail.reward-detail.delay_penalty atol=1e-10
    beta=R.ModelConfig(beta=.7)
    @test R.p_hit(3,.4,beta)≈.75+.25*(1-exp(-.7*.4))
    @test R.p_false(4,.4,beta)≈.20*exp(-.7*.4)
    @test R.loglr_and_score(path,z,[.5,.5],beta)!=R.loglr_and_score(path,z,[.5,.5],cfg)
end
@testset "Reference equivalence and projected-step uncertainty use the actual point" begin
    Y=[1. 2. 3.;1. 2. 3.;4. 1. 7.]
    @test X.reward_equivalence(Y,Y,1e-8)==[1,1,3]
    cfg=R.ModelConfig();z=[.9,.95];yi=ones(20);yj=2ones(20);S=repeat([10.,5.],1,20)
    ci=X.bootstrap_gradient_metrics(yi,yj,S,100,MersenneTwister(1),cfg,z)
    for k in (1,100)
        expected=norm(R.projected_step(z,[10.,5.]-collect(cfg.alpha),k)[1]-R.projected_step(z,[20.,10.]-collect(cfg.alpha),k)[1])
        @test getproperty(ci,Symbol("step$(k)_low"))≈expected atol=1e-14
        @test getproperty(ci,Symbol("step$(k)_high"))≈expected atol=1e-14
    end
end
