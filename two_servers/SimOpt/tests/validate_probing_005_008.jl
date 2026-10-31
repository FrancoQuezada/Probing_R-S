using Test, Random, LinearAlgebra, Statistics, Serialization, SHA
include(joinpath(@__DIR__,"../probing_experiments_common.jl"))
const E=ProbingExperiments
const M=E.C
BLAS.set_num_threads(1)

@testset "Preserved model, seeds and experimental designs" begin
    @test length(M.build_policy_grid())==144
    @test M.ALPHA==[10.,4.5] && M.Z0==[.12,.10]
    @test M.P_COMPLEX==[.02,.12,.45,.70]
    @test M.BASE_STEP==.035 && M.GRAD_CLIP==15.
    @test bytes2hex(sha256(read(joinpath(E.ROOT,"003probing_four_searches_softmax.jl"))))=="cb084d435519328e77972b17812c9117d42b761d085ccaa4bd254af12e6e0a98"
    @test E.stream_seed(8,1,"search",1)==E.stream_seed(8,1,"search",1)
    seeds=[E.stream_seed(s,r,t,k) for s in 5:8 for r in 1:20 for t in ("search","bootstrap","trace","final_selection","final_evaluation") for k in 0:100]
    @test length(unique(seeds))==length(seeds)
    d=E.parse_arguments(6,["--quick-test"])
    @test E.search_iterations(d,10)==10
    @test E.search_iterations(d,20)==5
    @test 4 in E.eval_iterations(d,10) && 10 in E.eval_iterations(d,10)
    @test_throws ErrorException E.parse_arguments(8,["--quick-test","--bootstrap-reps","0"])
    @test_throws ErrorException E.parse_arguments(5,["--quick-test","--betas","-1"])
end

@testset "PSoft: joint complete-path bootstrap, ties and hard limit" begin
    # The returned frequency matrix proves that reward columns, not jobs or
    # independent policy rows, are resampled jointly.
    Y=[10. 12. 14. 16.;1. 2. 3. 4.;3. 2. 1. 0.]
    p,C,V=E.bootstrap_weights(Y,300,MersenneTwister(18);return_counts=true)
    @test all(0 .<= p .<= 1) && sum(p)≈1
    @test p==[1.,0.,0.]
    @test all(vec(sum(C;dims=1)).==4)
    @test V==Y*C/4
    @test E.bootstrap_weights(Y,300,MersenneTwister(18))==p
    tied=repeat(reshape([1.,4.,-2.,8.],1,:),3,1)
    pt=E.bootstrap_weights(tied,1000,MersenneTwister(1))
    @test all(isapprox.(pt,1/3;atol=1e-15))
    near=[1. 1.;1+1e-11 1+1e-11]
    @test E.bootstrap_weights(near,100,MersenneTwister(2))≈[.5,.5]
    G=[30. 7.;-2. 8.;12. 15.]
    @test E.combine_gradients(G,p)==G[1,:]-M.ALPHA
    @test E.combine_gradients(zeros(3,2),p)==-M.ALPHA
    # More B reduces average Monte Carlo error; it need not improve every draw.
    uncertain=[2. -2. 2. -2.;-2. 2. -2. 2.]
    err=Dict(B=>mean(sum(abs2,E.bootstrap_weights(uncertain,B,MersenneTwister(100000*B+q)).-[.5,.5]) for q in 1:150) for B in (100,300,1000))
    @test err[1000]<err[300]<err[100]
end

@testset "Original-data individual gradients; exact Fresh/Softmax preservation" begin
    cfg=E.parse_arguments(8,["--quick-test"]);policies=M.build_policy_grid();z=[.31,.27]
    bank=M.materialize_crn_bank(E.primitive_bank(cfg,50,123),z)
    Y=M.precompute_rewards(policies,bank);original=copy(Y)
    data=E.local_data(z,bank,Y)
    ref=M.one_bank_estimator(z,z,bank,Y)
    @test data.hard==ref[2]
    @test M.softmax_estimator(z,bank,Y;beta=.1).gradient==E.combine_gradients(data.G,M.softmax_weights(data.values;beta=.1))
    p=E.bootstrap_weights(Y,300,MersenneTwister(99))
    @test Y==original
    for i in 1:length(policies),j in 1:2
        expected=mean(Y[i,l]*M.loglr_and_score(bank[l],z,z)[2][j] for l in eachindex(bank))
        @test data.G[i,j]≈expected atol=1e-11
    end
    @test E.combine_gradients(data.G,p)≈sum(p[i]*data.G[i,:] for i in eachindex(p))-M.ALPHA
    @test all(isfinite,E.combine_gradients(data.G,p))
    @test all(0 .<= M.projected_step(z,E.combine_gradients(data.G,p),1)[1] .<= 1)
    @test M.one_bank_estimator(z,z,bank,Y)[2]==ref[2]
    @test M.softmax_estimator(z,bank,Y;beta=.1).gradient==E.combine_gradients(data.G,M.softmax_weights(data.values;beta=.1))
end

@testset "Green aggregates all banks and all policies before weights" begin
    policies=M.build_policy_grid();history=M.HistoricalBank[];z=[.31,.27]
    for (k,zg) in enumerate(([.12,.1],[.55,.4],[.2,.8]))
        bank=M.generate_bank_at(zg;npaths=25,seed=100+k)
        push!(history,M.HistoricalBank(zg,bank,M.precompute_rewards(policies,bank)))
    end
    data=E.green_data(z,history);ref=M.green_estimator(z,history)
    @test data.hard==ref[2]
    @test data.best==ref[3]
    @test maximum(data.values)-dot(M.ALPHA,z)==ref[1]
    manual=zeros(length(policies),2)
    for hb in history, l in eachindex(hb.bank)
        ll,sc=M.loglr_and_score(hb.bank[l],z,hb.zgen)
        for i in eachindex(policies),j in 1:2
            manual[i,j]+=hb.Y[i,l]*exp(clamp(ll,-35.,35.))*sc[j]/(length(history)*length(hb.bank))
        end
    end
    @test data.G≈manual atol=1e-10
    w=M.softmax_weights(data.values;beta=.1)
    @test E.combine_gradients(data.G,w)≈transpose(manual)*w-M.ALPHA
    gap=sort(data.values;rev=true)[1]-sort(data.values;rev=true)[2]
    @test gap>0
    @test E.combine_gradients(data.G,M.softmax_weights(data.values;beta=1000/gap))≈data.hard atol=1e-10
    # Artificial differing banks expose the incorrect per-bank-weight estimator.
    V1=[10.,0.];V2=[0.,8.];G1=[1. 0.;0. 1.];G2=[2. 0.;0. 3.]
    correct=E.combine_gradients((G1+G2)/2,M.softmax_weights((V1+V2)/2;beta=1.))
    wrong=(E.combine_gradients(G1,M.softmax_weights(V1;beta=1.))+E.combine_gradients(G2,M.softmax_weights(V2;beta=1.)))/2
    @test norm(correct-wrong)>.1
    @test 0<data.ess<=75.00001 && data.banks==3
end

@testset "Reproducible searches and PSoft does not alter Fresh or Softmax" begin
    d8=E.parse_arguments(8,["--quick-test"]);policies=M.build_policy_grid()
    a=redirect_stdout(devnull) do;E.search_rep(d8,1,20,policies);end
    b=redirect_stdout(devnull) do;E.search_rep(d8,1,20,policies);end
    for i in eachindex(a.runs)
        @test a.runs[i].zhist==b.runs[i].zhist
        @test [(x.g3,x.g4) for x in a.runs[i].diagnostics]==[(x.g3,x.g4) for x in b.runs[i].diagnostics]
    end
    # Same stage/seed stream, remove PSoft only by using Fresh/Softmax directly.
    for (idx,beta) in enumerate((0.,.1))
        z=copy(M.Z0);hist=[copy(z)]
        for k in 1:d8["max_iters"]
            bank=M.materialize_crn_bank(E.primitive_bank(d8,20,E.stream_seed(8,1,"search",k)),z)
            Y=M.precompute_rewards(policies,bank)
            g=beta==0 ? M.one_bank_estimator(z,z,bank,Y)[2] : M.softmax_estimator(z,bank,Y;beta=beta).gradient
            z=M.projected_step(z,g,k)[1];push!(hist,copy(z))
        end
        @test a.runs[idx].zhist==hist
    end
    mktempdir() do dir
        path=joinpath(dir,"checkpoint.jls");E.atomic_serialize(path,(a=1,b=[2.,3.]))
        @test deserialize(path)==(a=1,b=[2.,3.])
        @test !isfile(path*".tmp")
    end
end
