# Additional zero-threshold and independent weighted-sum checks; no search edits.
include(joinpath(@__DIR__,"016probing_green_bank_variants.jl"))
using .ProbingGreenBankVariants
using Statistics, LinearAlgebra, TOML, Test
const V=ProbingGreenBankVariants
BLAS.set_num_threads(1)
cfg=V.H.from_dict(TOML.parsefile(joinpath(V.SOURCE,"configs/main_instance_parameters.toml"))["model"])
history=V.R.HistoricalBank[];policies=V.C.build_policy_grid();target=[.65,.60]
for (k,z) in enumerate(([.1,.1],[.3,.4],target))
    p=V.R.primitive_bank(cfg,25,V.X.seed(V.ID,"edge_audit",1,"search",k);njobs=45)
    bank=V.R.materialize_bank(p,z,cfg);Y=V.R.rewards(policies,bank,cfg)
    push!(history,V.R.historical_bank(V.ID,cfg,z,bank,Y))
end
results=@testset "Additional Green edge and weighted-sum audit" begin
    e=V.estimate(target,history,cfg,(family=:exponential,param=.7))
    values=zeros(144);gradient=zeros(144,2)
    for (h,a) in zip(e.indices,e.weights)
        hb=history[h];L,S=e.cache[h].L,e.cache[h].S
        for i in 1:144
            values[i]+=a*sum(hb.Y[i,j]*L[j] for j in 1:25)/25
            for d in 1:2
                gradient[i,d]+=a*sum(hb.Y[i,j]*L[j]*S[d,j] for j in 1:25)/25
            end
        end
    end
    best=argmax(values)
    @test e.data.best==best
    @test e.data.values≈values
    @test e.data.hard≈gradient[best,:]-collect(cfg.alpha)
    @test e.weights≈[.7^2,.7,1]/sum([.7^2,.7,1])
    # Synthetic contributions ONLY for a deterministic edge test, never searches.
    synthetic=[V.R.historical_bank(V.ID,cfg,h.zgen,h.bank,
        j==3 ? zeros(size(h.Y)) : [i*k for i in 1:144,k in 1:25].*1.) for (j,h) in enumerate(history)]
    filtered=V.estimate(target,synthetic,cfg,(family=:filtered,param=1.))
    @test filtered.current_var==0.
    @test filtered.filter_variances[1]>0&&filtered.filter_variances[2]>0
    @test filtered.indices==[3]
    @test filtered.data.best==1
    @test filtered.data.hard == -collect(cfg.alpha)
    @test all(iszero,filtered.data.values)
    # kappa=0 tests the always-current rule with positive current variance.
    forced=V.estimate(target,history,cfg,(family=:filtered,param=0.))
    @test forced.current_var>0
    @test length(history) in forced.indices
    @test all(i==length(history)||forced.filter_variances[i]==0. for i in forced.indices)
end
open(joinpath(V.OUTPUT,"validation/additional_audit.toml"),"w") do io
    TOML.print(io,Dict("passed"=>true,"assertions"=>Test.get_test_counts(results).passes,
        "zero_current_positive_history"=>true,"weighted_sum_independent_check"=>true))
end
