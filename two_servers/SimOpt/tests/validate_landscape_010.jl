using Test, Random, Statistics, LinearAlgebra, Serialization, TOML
include(joinpath(@__DIR__,"../010probing_instance_landscape_search.jl"))
const X=ProbingLandscapeSearch
const R=X.R
const C=X.C
BLAS.set_num_threads(1)
@testset "010 unchanged objective, design, probabilities and Fixed" begin
    ss=X.design(56);p=C.build_policy_grid();cfg=R.ModelConfig()
    @test length(ss)==56 && length(p)==144
    @test length(unique(R.model_digest(s.model) for s in ss))==56
    @test R.model_dict(ss[1].model)==R.model_dict(cfg)
    @test all(s.model.revenue_scale==8. && s.model.fast_routing_cost==.7 for s in ss)
    for s in ss,c in 1:4,z in 0:.05:1
        @test 0<=R.p_hit(c,z,s.model)<=1
        @test 0<=R.p_false(c,z,s.model)<=1
    end
    prim=R.primitive_bank(cfg,500,C.SEED_FIXED)
    bank=R.materialize_bank(prim,[0.,0.],cfg);old=C.generate_bank_at([0.,0.];npaths=500,seed=C.SEED_FIXED)
    @test all(a.jobs==b.jobs for (a,b) in zip(bank,old))
    Y=R.rewards(p,bank,cfg);@test Y==C.precompute_rewards(p,old)
    z=copy(C.Z0);hist=[copy(z)]
    for k in 1:100
        a=X.fixed_data(z,bank,Y,cfg);b=C.one_bank_estimator(z,[0.,0.],old,Y)
        @test a.hard==b[2] && a.best==b[3]
        @test maximum(a.values)-R.cost(cfg,z)==b[1]
        z=R.projected_step(z,a.hard,k)[1];push!(hist,copy(z))
    end
    saved=deserialize(joinpath(@__DIR__,"../resultados/003probing_softmax_20261021/completo/probing_four_results.jls"))
    @test hist==saved.searches[1].zhist
    # Reproduce saved direct-simulation surface at preselected grid points, exact original bank.
    prim=R.primitive_bank(cfg,5000,C.ORACLE_SURFACE_SEED)
    for (i,j) in ((1,1),(11,11),(21,21),(7,9))
        z=[saved.grid[i],saved.grid[j]]
        values=vec(mean(R.rewards(p,R.materialize_bank(prim,z,cfg),cfg);dims=2))
        @test maximum(values)-R.cost(cfg,z)==saved.surface[j,i]
    end
    @test X.components(Bool[1 0 1;0 0 1;1 1 0])==3
    seeds=[X.seed(s.id,stage,r,stream,k) for s in ss[1:4] for stage in ("screen","compare") for r in 1:5 for stream in ("search","trace","selection","holdout","bootstrap") for k in 0:100]
    @test length(unique(seeds))==length(seeds)
end
if !isempty(ARGS)
    out=abspath(ARGS[1]);d=X.arguments(["--quick-test","--output",out,"--resume"])
    @testset "010 seven-method quick executor independent replay" begin
        ss=deserialize(joinpath(out,"configs/design.jls"));p=C.build_policy_grid()
        selected=deserialize(joinpath(out,"configs/selected_instances.jls"))
        for id in [s.scenario for s in selected]
            s=only(filter(s->s.id==id,ss));saved=deserialize(joinpath(out,"checkpoints/search_$(id)_1.jls"))
            for run in saved.runs
                z=copy(C.Z0);hist=[copy(z)];history=R.HistoricalBank[];m=run.method
                prim=R.primitive_bank(s.model,20,X.seed(id,"compare",1,"fixed"));bank0=R.materialize_bank(prim,[0.,0.],s.model);Y0=R.rewards(p,bank0,s.model)
                for k in 1:4
                    prim=R.primitive_bank(s.model,20,X.seed(id,"compare",1,"search",k))
                    bank=R.materialize_bank(prim,z,s.model);Y=R.rewards(p,bank,s.model)
                    if m.kind==:fixed;data=X.fixed_data(z,bank0,Y0,s.model)
                    elseif m.kind in (:green,:green_softmax)
                        push!(history,R.historical_bank(id,s.model,z,bank,Y));data=R.green_data(z,history,s.model,id)
                    else;data=R.local_data(z,bank,Y,s.model);end
                    g=if m.kind in (:fixed,:fresh,:green);data.hard
                    elseif m.kind==:psoft;R.combine(data.G,R.bootstrap_weights(Y,300,MersenneTwister(X.seed(id,"compare",1,"bootstrap",k))),s.model)
                    else;R.combine(data.G,R.softmax_weights(data.values;beta=m.beta),s.model);end
                    z=R.projected_step(z,g,k)[1];push!(hist,copy(z))
                end
                @test hist==run.zhist
            end
        end
    end
end
