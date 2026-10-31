using Test, LinearAlgebra, Statistics, SHA

const ROOT = normpath(joinpath(@__DIR__, ".."))
const ORIGINAL_FILE = joinpath(ROOT,"002probing_three_searches_n500_full45iters.jl")
const FOUR_FILE = joinpath(ROOT,"003probing_four_searches_softmax.jl")
const ORIGINAL_SHA = "50000bdc269da66ea4324373869bc3f38f8d51f86db0d5c017b086a24efb6021"

# Reduced fixtures only: change sizes in memory, without editing either source.
function load_reduced(path, name; original=false)
    source_text = read(path,String)
    if original
        source_text = replace(source_text,r"(?m)^main\(\)\s*$"=>"")
    end
    for (key,value) in (("N_PATHS",40),("JOBS_PER_PATH",12),("MAX_ITERS",4),
                         ("TRACE_EVAL_PATHS",40),("ORACLE_PATHS_FINAL",100),
                         ("ORACLE_GRID_N",3),("ORACLE_PATHS_PER_GRID_POINT",40))
        source_text = replace(source_text,Regex("(?m)^const "*key*" = \\d+")=>"const $key = $value")
    end
    mod = Module(name)
    Base.include_string(mod,source_text,path)
    return mod
end

const Original = load_reduced(ORIGINAL_FILE,:Original;original=true)
const Four = load_reduced(FOUR_FILE,:Four)

@testset "Original experiment is preserved" begin
    @test bytes2hex(sha256(read(ORIGINAL_FILE))) == ORIGINAL_SHA
    for key in (:BETA,:ALPHA,:Z0,:SEED_FIXED,:SEED_FRESH,:SEED_GREEN,
                :ORACLE_SEED,:TRACE_EVAL_SEED,:ORACLE_SURFACE_SEED)
        @test getproperty(Original,key) == getproperty(Four,key)
    end
    oldpol = Original.build_policy_grid()
    newpol = Four.build_policy_grid()
    @test length(oldpol) == length(newpol) == 144
    for name in (:optimize_fixed,:optimize_fresh,:optimize_green)
        old = redirect_stdout(devnull) do
            getproperty(Original,name)(Original.Z0,oldpol)
        end
        new = redirect_stdout(devnull) do
            getproperty(Four,name)(Four.Z0,newpol)
        end
        # Full decisions, objectives, policy histories and endpoints match exactly.
        for k in 1:5
            @test old[k] == new[k]
        end
    end
    @test Four.SEED_SOFTMAX == Four.SEED_FRESH
end

@testset "Stable Softmax weights" begin
    w = Four.softmax_weights([1000.0,999.0,-1000.0])
    @test all(isfinite,w)
    @test all(>=(0),w)
    @test sum(w) ≈ 1.0 atol=1e-14
    @test w ≈ Four.softmax_weights([1001000.0,1000999.0,999000.0])
    @test Four.softmax_weights([-1e300,1e300];beta=1e300) == [0.0,1.0]
    @test Four.softmax_weights([2.0,2.0];beta=1e6) == [0.5,0.5]
    @test_throws ArgumentError Four.softmax_weights([1.0];beta=0.0)
    @test_throws ArgumentError Four.softmax_weights([1.0];beta=Inf)
    @test_throws ArgumentError Four.softmax_weights([NaN,1.0])
    @test_throws ArgumentError Four.softmax_weights(Float64[])
end

@testset "Local score gradient, single cost, and large-beta limit" begin
    z=[0.31,0.27]
    bank=Four.generate_bank_at(z;npaths=80,seed=12345)
    Y=Four.precompute_rewards(Four.build_policy_grid(),bank)
    scores=hcat([Four.loglr_and_score(path,z,z)[2] for path in bank]...)
    @test all(Four.loglr_and_score(path,z,z)[1]==0.0 for path in bank)
    est=Four.softmax_estimator(z,bank,Y)
    values=vec(mean(Y;dims=2))
    per_policy=[mean(@view(Y[i,:]) .* @view(scores[j,:])) for i in axes(Y,1), j in 1:2]
    direct=vec(sum(per_policy .* est.weights;dims=1))-Four.ALPHA
    @test est.gradient ≈ direct atol=1e-11 rtol=1e-12
    @test est.gradient+Four.ALPHA ≈ transpose(per_policy)*est.weights atol=1e-11
    @test all(isfinite,est.gradient)
    @test est.objective ≈ maximum(values)-dot(Four.ALPHA,z)
    @test 1<=est.effective_policies<=length(values)
    @test est.max_weight ≈ maximum(est.weights)
    @test est.effective_policies ≈ inv(sum(abs2,est.weights))

    # Pick two policies with distinct sample means, ensuring a UNIQUE maximum.
    # Ties converge to a mixture of tied gradients, not the first argmax's gradient.
    best=argmax(values); worst=argmin(values)
    @test values[best]>values[worst]
    Yunique=Y[[best,worst],:]
    beta_large=40/(values[best]-values[worst])
    hard=Four.one_bank_estimator(z,z,bank,Yunique)
    smooth=Four.softmax_estimator(z,bank,Yunique;beta=beta_large)
    @test smooth.weights[1] ≈ 1 atol=1e-14
    @test smooth.gradient ≈ hard[2] atol=1e-10 rtol=1e-11
    println("Large-beta unique-argmax gradient error: ",maximum(abs.(smooth.gradient-hard[2])))
end

@testset "Fixed-LR log-sum-exp surrogate finite differences" begin
    sourcez=[0.0,0.0]; z=[0.31,0.27]; beta=Four.SOFTMAX_BETA
    bank=Four.generate_bank_at(sourcez;npaths=100,seed=986)
    Y=Four.precompute_rewards(Four.build_policy_grid(),bank)
    function surrogate_data(target)
        pairs=[Four.loglr_and_score(path,target,sourcez) for path in bank]
        logs=first.(pairs)
        @test maximum(abs.(logs))<35  # no LR clamp active in this derivative check
        L=exp.(logs)
        scores=hcat(last.(pairs)...)
        values=vec(Y*L/length(bank))
        G=(Y*transpose(scores .* transpose(L)))/length(bank)
        combined=Four.combine_softmax_gradients(values,G;beta=beta)
        return combined.surrogate_gross-dot(Four.ALPHA,target),combined.gradient
    end
    _,gradient=surrogate_data(z)
    h=1e-6
    fd=zeros(2)
    for j in 1:2
        zp=copy(z); zm=copy(z); zp[j]+=h; zm[j]-=h
        fd[j]=(surrogate_data(zp)[1]-surrogate_data(zm)[1])/(2h)
    end
    println("Softmax fixed-LR gradient: ",gradient,"; finite differences: ",fd,
            "; maximum error: ",maximum(abs.(gradient-fd)))
    @test gradient ≈ fd atol=2e-6 rtol=1e-6
end

@testset "Softmax search domain, shared seed streams and finite diagnostics" begin
    policies=Four.build_policy_grid()
    soft=redirect_stdout(devnull) do
        Four.optimize_softmax(Four.Z0,policies)
    end
    @test all(all(x->0<=x<=1,z) for z in soft.zhist)
    @test length(soft.diagnostics)==Four.MAX_ITERS
    @test all(all(isfinite,d.gradient) for d in soft.diagnostics)
    @test all(0<d.max_weight<=1 && 1<=d.effective_policies<=144 for d in soft.diagnostics)
    @test [d.seed for d in soft.diagnostics] ==
          [Four.SEED_FRESH+10000*k for k in 1:Four.MAX_ITERS]

    # Same seed retains all exogenous primitives and signal uniforms, even if z diverges.
    seed=Four.SEED_FRESH+10000
    a=Four.generate_bank_at([0.1,0.2];seed=seed)
    b=Four.generate_bank_at([0.7,0.8];seed=seed)
    fields=(:arrival,:class,:complex,:value,:service_fast,:service_std)
    @test all(all(getproperty(ja,f)==getproperty(jb,f) for f in fields)
              for (pa,pb) in zip(a,b) for (ja,jb) in zip(pa.jobs,pb.jobs))
end

@testset "Four traces share the original independent CRN hard-max evaluation" begin
    policies=Four.build_policy_grid()
    histories=([copy(Four.Z0),[0.2,0.3]], [copy(Four.Z0),[0.4,0.5]],
               [copy(Four.Z0),[0.6,0.7]], [copy(Four.Z0),[0.1,0.9]])
    traces=redirect_stdout(devnull) do
        Four.build_common_objective_traces(histories...,policies)
    end
    bank=Four.generate_crn_primitive_bank(npaths=Four.TRACE_EVAL_PATHS,seed=Four.TRACE_EVAL_SEED)
    @test traces[1][1] == traces[2][1] == traces[3][1] == traces[4][1]
    for i in 1:4
        expected=Four.evaluate_trace_on_common_crn(histories[i],policies,bank)
        @test traces[i]==expected
        materialized=Four.materialize_crn_bank(bank,histories[i][end])
        Y=Four.precompute_rewards(policies,materialized)
        @test traces[i][end] ≈ maximum(vec(mean(Y;dims=2)))-dot(Four.ALPHA,histories[i][end])
    end
end
