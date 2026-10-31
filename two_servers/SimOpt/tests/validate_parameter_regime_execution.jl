# Independent uncached replay of the quick executor in all eight configurations.
using Test, Serialization, Random, LinearAlgebra
include(joinpath(@__DIR__,"../probing_regime_experiment.jl"))
const X=ProbingRegimeExperiment
const R=X.R
BLAS.set_num_threads(1)
d=X.parse_arguments(["--quick-test"])
policies=X.C.build_policy_grid()
@testset "Five methods, eight scenarios: executor equals uncached estimator replay" begin
    for s in R.scenarios()
        path=joinpath(d["output"],"checkpoints/pilot_$(s.id)_rep_01_search.jls")
        saved=deserialize(path)
        for (i,m) in enumerate(X.METHODS)
            z=copy(X.C.Z0);hist=[copy(z)];history=R.HistoricalBank[]
            for k in 1:d["max_iters"]
                primitives=R.primitive_bank(s.model,d["n_paths"],X.seed(s.id,"pilot",1,"search",k))
                bank=R.materialize_bank(primitives,z,s.model);Y=R.rewards(policies,bank,s.model)
                if m.kind in (:green,:green_softmax)
                    push!(history,R.historical_bank(s.id,s.model,z,bank,Y))
                    data=R.green_data(z,history,s.model,s.id)
                else;data=R.local_data(z,bank,Y,s.model);end
                g=if m.kind in (:fresh,:green);data.hard
                elseif m.kind==:psoft
                    R.combine(data.G,R.bootstrap_weights(Y,300,MersenneTwister(X.seed(s.id,"pilot",1,"bootstrap",k))),s.model)
                else;R.combine(data.G,R.softmax_weights(data.values;beta=.5),s.model);end
                z=R.projected_step(z,g,k)[1];push!(hist,copy(z))
            end
            @test hist==saved.runs[i].zhist
        end
    end
end
@testset "Independent seed namespaces, history guards and cached oracle identity" begin
    seeds=[X.seed(s.id,phase,r,stream,k) for s in R.scenarios() for phase in ("pilot","confirmation")
        for r in 1:20 for stream in ("search","bootstrap","final_selection","final_holdout") for k in 0:100]
    @test length(unique(seeds))==length(seeds)
    for s in R.scenarios()
        oracle=deserialize(joinpath(d["output"],"checkpoints/oracle_$(s.id).jls"))
        @test oracle.scenario==s.id && oracle.fingerprint==d["fingerprint"]
    end
end
