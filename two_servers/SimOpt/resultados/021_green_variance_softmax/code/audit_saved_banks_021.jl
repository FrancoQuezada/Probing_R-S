# Read-only audit of production banks; never generates a path or updates a decision.
output=abspath(ARGS[1])
include(joinpath(output,"reproduction/021_green_variance_softmax.jl"))
using .GreenVarianceSoftmax021, Serialization, Statistics, LinearAlgebra, Test, Printf
const G=GreenVarianceSoftmax021
const S=G.S;const R=G.R;const V=G.V
BLAS.set_num_threads(1)
c,cfg=S.model(output)
G.OUTPUT[]=output
nreps=length(ARGS)>1 ? parse(Int,ARGS[2]) : 50
result=@testset "021 persisted bank identities and original-estimator replay" begin
    for rep in 1:nreps
        dir=joinpath(output,"raw",G.METHOD,@sprintf("rep_%04d",rep))
        raw=deserialize(joinpath(dir,"search.jls")).raw
        history=R.HistoricalBank[];seeds=S.seeds(rep,100)
        for k in 1:100
            cp=deserialize(joinpath(dir,"banks",@sprintf("bank_%03d.jls",k)))
            h=cp.historical
            @test cp.iteration==k
            @test cp.seed==seeds["search"][k]
            @test h.scenario==V.ID && h.model_sha256==c["science"]["model_sha256"]
            @test h.zgen==raw.zhist[k]
            @test length(h.bank)==500 && size(h.Y)==(144,500)
            @test all(p->length(p.jobs)==45,h.bank)
            @test all(isfinite,h.Y)
            # Replay original estimation from SAVED observations at the first,
            # middle and final search targets for rep01. No simulation is run.
            if rep==1
                push!(history,h)
                if k in (1,50,100)
                    old=V.estimate(h.zgen,history,cfg,G.FILTER)
                    new=G.estimate(h.zgen,history,cfg)
                    @test isequal(old,new.est)
                    @test join(old.indices,";")==raw.rows[k].retained_bank_indices
                    @test old.current_var==raw.rows[k].filter_current_variance
                    @test old.threshold==raw.rows[k].filter_threshold
                    @test isapprox(new.gradient,[raw.core[k].g3,raw.core[k].g4];rtol=1e-12,atol=1e-10)
                    @test maximum(old.data.values)-R.cost(cfg,h.zgen)==raw.core[k].objective_internal
                end
            end
        end
    end
end
name=nreps==50 ? "persisted_banks_audit.toml" : "persisted_banks_audit_first$(nreps).toml"
S.atomic_toml(joinpath(output,"validation",name),Dict("passed"=>true,
    "instance"=>V.ID,"assertions"=>Test.get_test_counts(result).passes,
    "saved_banks_checked"=>100nreps,"paths_checked"=>50000nreps,"new_simulations"=>0,
    "exact_original_filter_replays"=>3,"replay_replication"=>1,"replay_iterations"=>[1,50,100]))
