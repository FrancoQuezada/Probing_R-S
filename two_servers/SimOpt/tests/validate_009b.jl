using Test,Random,Statistics,LinearAlgebra,Serialization,TOML
include(joinpath(@__DIR__,"../probing_validation_experiment.jl"))
const V=ProbingRegimeValidation
const X=V.X
const R=V.R
BLAS.set_num_threads(1)

@testset "Saved configurations, paired gradient covariance, common-path bootstrap" begin
    scenarios=V.configured_scenarios();@test [s.id for s in scenarios]==V.IDS
    @test R.model_dict(scenarios[1].model)==R.model_dict(R.ModelConfig())
    for s in scenarios
        old=only(filter(x->x.id==s.id,R.scenarios()));@test s.model==old.model
    end
    rng=MersenneTwister(91);n=200;x=100randn(rng,n);d=randn(rng,n)
    yi=x+d;yj=x;S=randn(rng,2,n)
    h=V.paired_moments(yi,yj,S);F=hcat((yi-yj).*S[1,:],(yi-yj).*S[2,:])
    @test h.delta==vec(mean(F;dims=1))
    @test h.covariance==cov(F;dims=1)/n
    @test h.delta≈[mean(yi.*S[j,:])-mean(yj.*S[j,:]) for j in 1:2] atol=1e-12
    @test maximum(diag(h.covariance))<.1
    @test maximum([var(yi.*S[j,:])/n+var(yj.*S[j,:])/n for j in 1:2])>10
    zero=V.paired_moments(yi,yi,S);@test zero.delta==[0.,0.]&&zero.covariance==zeros(2,2)
    Y=randn(rng,144,12);scores=vcat(ones(1,12),fill(2.,1,12));candidates=[3,7,60]
    boot=V.bootstrap_joint(Y,scores,candidates,300,MersenneTwister(100))
    _,counts,old=R.E.bootstrap_weights(Y,300,MersenneTwister(100);return_counts=true)
    @test boot.values≈old atol=1e-14
    @test boot.G3≈old[candidates,:] atol=1e-14
    @test boot.G4≈2old[candidates,:] atol=1e-14
    @test all(sum(counts;dims=1).==12)
    z=[.12,.1]
    for t in (1,25,50,100),g in ([100.,-100.],[-100.,100.],[.1,.2])
        @test R.projected_step(z,g,t)==V.C.projected_step(z,g,t)
        zz,eta,clipped=R.projected_step(z,g,t)
        @test zz==clamp.(z+(.035/sqrt(t))*clamp.(g,-15.,15.),0.,1.)
        @test eta==.035/sqrt(t) && clipped==clamp.(g,-15.,15.)
    end
end

@testset "New namespaces independent of 009 and across replications/evaluation streams" begin
    seeds=[X.seed(s,V.arguments(String[])["phase"],rep,stream,k) for s in V.IDS for rep in 1:20
        for stream in ("search","bootstrap","trace","final_selection","final_holdout") for k in 0:100]
    @test length(seeds)==length(unique(seeds))
    oldfile=joinpath(V.SOURCE,"tables/seed_manifest.csv")
    lines=readlines(oldfile);header=split(lines[1],',');column=findfirst(==("seed"),header)
    old=Set(parse(Int,split(line,',')[column]) for line in lines[2:end])
    @test isempty(intersect(Set(seeds),old))
    @test length(unique(X.seed("S001","009b-confirmation",rep,"trace") for rep in 1:20))==20
end

if !isempty(ARGS)
    @testset "Five configurations/methods: cached executor equals independent uncached replay" begin
        out=abspath(ARGS[1]);d=V.arguments(["--quick-test"]);policies=V.C.build_policy_grid()
        for s in V.configured_scenarios()
            saved=deserialize(joinpath(out,"checkpoints/confirm_$(s.id)_rep_01_search.jls"))
            for (i,m) in enumerate(V.METHODS)
                z=copy(V.C.Z0);history=R.HistoricalBank[];actual=[copy(z)]
                for k in 1:d["max_iters"]
                    primitive=R.primitive_bank(s.model,d["n_paths"],X.seed(s.id,d["phase"],1,"search",k))
                    bank=R.materialize_bank(primitive,z,s.model);Y=R.rewards(policies,bank,s.model)
                    if m.kind in (:green,:green_softmax)
                        push!(history,R.historical_bank(s.id,s.model,z,bank,Y));data=R.green_data(z,history,s.model,s.id)
                    else;data=R.local_data(z,bank,Y,s.model);end
                    gradient=if m.kind in (:fresh,:green);data.hard
                    elseif m.kind==:psoft;R.combine(data.G,R.bootstrap_weights(Y,300,MersenneTwister(X.seed(s.id,d["phase"],1,"bootstrap",k))),s.model)
                    else;R.combine(data.G,R.softmax_weights(data.values;beta=.5),s.model);end
                    z=R.projected_step(z,gradient,k)[1];push!(actual,copy(z))
                end
                @test actual==saved.runs[i].zhist
            end
        end
    end
    @testset "Actual convergence banks are shared within replications and independent between them" begin
        out=abspath(ARGS[1]);d=V.arguments(["--quick-test"]);policies=V.C.build_policy_grid()
        for s in V.configured_scenarios(),rep in 1:2
            primitive=R.primitive_bank(s.model,d["trace_paths"],X.seed(s.id,d["phase"],rep,"trace"))
            Y=R.rewards(policies,R.materialize_bank(primitive,V.C.Z0,s.model),s.model)
            expected=maximum(vec(mean(Y;dims=2)))-R.cost(s.model,V.C.Z0)
            for method in 1:5
                evaluation=deserialize(joinpath(out,"checkpoints/confirm_$(s.id)_rep_$(lpad(rep,2,'0'))_M$(method)_evaluation.jls"))
                @test evaluation.traces[1].objective==expected
            end
        end
    end
    @testset "Reference prefixes reproduce exactly the newly seeded direct sample and scores" begin
        out=abspath(ARGS[1]);d=V.arguments(["--quick-test"]);d["output"]=out;policies=V.C.build_policy_grid()
        for p in V.selected_points(d)
            s=only(filter(s->s.id==p.scenario,V.configured_scenarios()))
            reference=deserialize(joinpath(out,"checkpoints/A_$(s.id)_$(p.point)_reference.jls"))
            n=first(d["reference_sizes"]);seed=V.diagseed(d,s.id,p.point,"reference_B")
            bank=R.materialize_bank(R.primitive_bank(s.model,n,seed),p.z,s.model)
            @test R.rewards(policies,bank,s.model)==reference.Y[:,1:n]
            @test R.bank_scores(bank,p.z,p.z,s.model)[2]==reference.S[:,1:n]
        end
    end
end
