using Test, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__,"../010probing_instance_landscape_search.jl"))
const X=ProbingLandscapeSearch
const R=X.R
const C=X.C
BLAS.set_num_threads(1)
@testset "010 economic changes are routing changes, not reward rescaling" begin
    base=R.ModelConfig();high=R.ModelConfig(delay_penalty=3.36)
    prim=R.primitive_bank(base,20,42);p=C.build_policy_grid()
    for z in ([0.,0.],[.3,.6],[1.,1.]),path in R.materialize_bank(prim,z,base),policy in p[1:15:end]
        a=R.routing_details(policy,path,base);b=R.routing_details(policy,path,high)
        @test a.decisions==b.decisions
        @test b.reward≈a.reward-(high.delay_penalty/base.delay_penalty-1)*a.delay_penalty atol=2e-10
    end
    changed=R.ModelConfig(std_mean=(1.,12.),fast_mean=(.75,1.8))
    nb=R.primitive_bank(changed,20,42)
    for (a,b) in zip(prim,nb),(x,y) in zip(a.jobs,b.jobs)
        @test x.arrival==y.arrival && x.value==y.value && x.complex==y.complex
        @test y.service_std≈x.service_std*(x.complex ? 1.5 : 1.)
        @test y.service_fast≈x.service_fast*(x.complex ? .75 : 1.)
    end
    for s in X.design(56)
        @test s.model.p_hit0[1:2]==base.p_hit0[1:2]
        @test s.model.p_false0[1:2]==base.p_false0[1:2]
        @test s.model.fast_mean[1]==.75 && s.model.std_mean[1]==1.
        @test s.model.fast_mean[2]<s.model.std_mean[2]
        for c in 1:2,z in (0.,.5,1.)
            @test R.signal_probability(c,true,C.full_z([z,z])[c],s.model)==base.p_hit0[c]
            @test R.signal_probability(c,false,C.full_z([z,z])[c],s.model)==base.p_false0[c]
        end
    end
end
