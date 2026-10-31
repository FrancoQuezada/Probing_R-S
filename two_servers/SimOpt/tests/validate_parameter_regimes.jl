using Test, Random, Statistics, LinearAlgebra, Serialization
include(joinpath(@__DIR__,"../probing_regime_model.jl"))
const R=ProbingRegimeModel
const O=R.C
const E=R.E
BLAS.set_num_threads(1)

@testset "Immutable scenarios and exact base primitives/rewards/LR" begin
    scenarios=R.scenarios();cfg=R.ModelConfig();policies=O.build_policy_grid()
    @test length(scenarios)==8 && length(policies)==144
    @test length(unique(R.model_digest(s.model) for s in scenarios))==8
    @test R.model_dict(scenarios[1].model)==R.model_dict(cfg)
    for s in scenarios,c in 1:4,z in range(0,1;length=101)
        @test 0<=R.p_hit(c,z,s.model)<=1 && 0<=R.p_false(c,z,s.model)<=1
    end
    primitives=R.primitive_bank(cfg,35,789)
    original=O.generate_crn_primitive_bank(npaths=35,seed=789)
    @test all(p.jobs==q.jobs for (p,q) in zip(primitives,original))
    for z in ([.12,.1],[0.,0.],[1.,1.],[.31,.57])
        b=R.materialize_bank(primitives,z,cfg);ob=O.materialize_crn_bank(original,z)
        @test all(p.jobs==q.jobs for (p,q) in zip(b,ob))
        Y=R.rewards(policies,b,cfg)
        @test Y==O.precompute_rewards(policies,ob)
        d=R.local_data(z,b,Y,cfg)
        @test d.hard==E.local_data(z,ob,Y).hard
        @test d.G==E.local_data(z,ob,Y).G
        for path in b,source in ([.4,.7],[0.,0.],z)
            @test R.loglr_and_score(path,z,source,cfg)==O.loglr_and_score(path,z,source)
        end
    end
end

@testset "Parameters reach primitives, routing, LR, scores, alpha and CRN marginals" begin
    base=R.ModelConfig();p=O.build_policy_grid()[62];z=[.3,.4]
    b=R.primitive_bank(base,200,777)
    busy=R.ModelConfig(mean_interarrival=.6)
    bb=R.primitive_bank(busy,200,777)
    for (a,bp) in zip(b,bb),j in eachindex(a.jobs)
        @test bp.jobs[j].arrival≈a.jobs[j].arrival*.75 atol=1e-12
        @test bp.jobs[j].service_fast==a.jobs[j].service_fast
    end
    allcomplex=R.ModelConfig(p_complex=(1.,1.,1.,1.))
    @test all(j.complex for p in R.primitive_bank(allcomplex,10,123) for j in p.jobs)
    changedservice=R.ModelConfig(fast_mean=(1.5,4.8),std_mean=(2.,16.))
    cb=R.primitive_bank(changedservice,200,777)
    @test all(y.service_fast==2*x.service_fast && y.service_std==2*x.service_std for (a,bp) in zip(b,cb) for (x,y) in zip(a.jobs,bp.jobs))
    path=R.materialize(b[1],z,base);stats=R.routing_details(p,path,base)
    @test stats.reward==R.evaluate_policy(p,path,base)
    expensive=R.ModelConfig(fast_routing_cost=1.4)
    @test R.evaluate_policy(p,path,expensive)≈stats.reward-.7*count(stats.decisions) atol=1e-11
    alpha=R.ModelConfig(alpha=(3.,7.))
    @test R.combine(zeros(3,2),[.2,.3,.5],alpha)==[-3.,-7.]
    @test R.cost(alpha,z)==dot([3.,7.],z)
    for s in R.scenarios()
        cfg=s.model;bank=R.materialize_bank(R.primitive_bank(cfg,12,123),z,cfg)
        for path in bank
            ll,score=R.loglr_and_score(path,z,[.5,.6],cfg)
            for j in 1:2
                zp=copy(z);zm=copy(z);zp[j]+=1e-6;zm[j]-=1e-6
                fd=(R.loglr_and_score(path,zp,[.5,.6],cfg)[1]-R.loglr_and_score(path,zm,[.5,.6],cfg)[1])/2e-6
                @test score[j]≈fd atol=2e-7 rtol=1e-6
            end
        end
    end
    # Conditional Bernoulli frequencies retain the configured marginals.
    primitives=R.primitive_bank(R.scenarios()[5].model,3000,991;njobs=45)
    cfg=R.scenarios()[5].model;bank=R.materialize_bank(primitives,z,cfg)
    for c in (3,4),complex in (false,true)
        signals=[j.signal for path in bank for j in path.jobs if j.class==c && j.complex==complex]
        prob=R.signal_probability(c,complex,z[c-2],cfg)
        @test abs(mean(signals)-prob)<5sqrt(prob*(1-prob)/length(signals))
    end
end

@testset "Scenario guard and unchanged Softmax/PSoft definitions" begin
    s=R.scenarios()[1];p=O.build_policy_grid();z=[.31,.27]
    bank=R.materialize_bank(R.primitive_bank(s.model,40,101),z,s.model);Y=R.rewards(p,bank,s.model)
    h=[R.historical_bank(s.id,s.model,z,bank,Y)]
    d=R.green_data(z,h,s.model,s.id)
    old=E.green_data(z,[O.HistoricalBank(z,bank,Y)])
    @test d.G==old.G && d.hard==old.hard && d.values==old.values
    @test_throws ErrorException R.green_data(z,h,R.scenarios()[2].model,"S001")
    @test_throws ErrorException R.green_data(z,h,s.model,"S111")
    @test R.softmax_weights(d.values)==O.softmax_weights(d.values;beta=.5)
    @test R.bootstrap_weights(Y,300,MersenneTwister(12))==E.bootstrap_weights(Y,300,MersenneTwister(12))
end

function replay_saved(directory,stage,index)
    saved=deserialize(joinpath(@__DIR__,"../resultados",directory,"checkpoints/rep_01_n_0500_search.jls")).runs[index]
    cfg=R.ModelConfig();policies=O.build_policy_grid();z=copy(O.Z0);history=R.HistoricalBank[]
    actual=[copy(z)]
    for k in 1:100
        primitives=R.primitive_bank(cfg,500,E.stream_seed(stage,1,"search",k))
        bank=R.materialize_bank(primitives,z,cfg);Y=R.rewards(policies,bank,cfg)
        m=saved.method
        if m.kind in (:green,:green_softmax)
            push!(history,R.historical_bank("S000",cfg,z,bank,Y))
            data=R.green_data(z,history,cfg,"S000")
        else;data=R.local_data(z,bank,Y,cfg);end
        g=if m.kind in (:fresh,:green);data.hard
        elseif m.kind==:psoft
            R.combine(data.G,R.bootstrap_weights(Y,300,MersenneTwister(E.stream_seed(8,1,"bootstrap",k))),cfg)
        else;R.combine(data.G,R.softmax_weights(data.values;beta=m.beta),cfg);end
        z=R.projected_step(z,g,k)[1];push!(actual,copy(z))
        @test all(x->0<=x<=1,z) && all(isfinite,g)
    end
    @test actual==saved.zhist
    println("Exact 100-iteration stored-trajectory regression: ",saved.method.name)
end
@testset "Exact regression against completed 005/007/008 searches, 500 paths x 100 iterations" begin
    replay_saved("005probing_softmax_multiseed",5,1)
    replay_saved("005probing_softmax_multiseed",5,5)
    replay_saved("007probing_green_softmax",7,1)
    replay_saved("007probing_green_softmax",7,3)
    replay_saved("008probing_probability_softmax",8,4)
end
