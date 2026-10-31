# Only NEW production searches: Exp50, I0252 reps 1:3. Never call old main/tests.
include(joinpath(@__DIR__,"016probing_green_bank_variants.jl"))
module ProbingGreenFiveComparison
using ..ProbingGreenBankVariants
using Statistics, LinearAlgebra, Serialization, SHA, TOML, Dates, Printf, Test
const V=ProbingGreenBankVariants;const A=V.A;const H=V.H;const X=V.X;const R=V.R;const E=V.E;const C=V.C
const ROOT=@__DIR__;const ID="I0252";const REPS=1:3;const RHO=.5
const SOURCE=V.SOURCE;const BANKS=V.OUTPUT;const TIMING=V.DONOR
const OUTPUT=joinpath(ROOT,"resultados/017probing_green_five_method_comparison")
const NEW=(id="Exp50",family=:exponential,param=RHO)
const METHODS=["Green","GreenSoftmax050","Exp50","Filter2","Rolling5"]
const Z95=1.959963984540054
sha(p)=bytes2hex(sha256(read(p)))
oldarchive(rep)=deserialize(joinpath(SOURCE,"checkpoints/search_$(ID)_$rep.jls"))
function archived(rep,id)
    if id=="GreenSoftmax050"
        return only(filter(r->r.method.id==id,oldarchive(rep).runs))
    end
    deserialize(joinpath(BANKS,"checkpoints/search_$(rep)_$id.jls"))
end

"""No simulation or optimization. Fail before a new search if compatibility
cannot be established. The old 003 standalone experiment is explicitly excluded.
"""
function preflight()
    s011=TOML.parsefile(joinpath(SOURCE,"configs/experiment.toml"))["science"]
    s016=TOML.parsefile(joinpath(BANKS,"configs/experiment.toml"))["science"]
    s012=TOML.parsefile(joinpath(TIMING,"configs/experiment.toml"))["science"]
    for (file,h) in merge(s011["sources"],s016["sources"])
        @assert sha(joinpath(ROOT,file))==h "Changed scientific source: $file"
    end
    md=TOML.parsefile(joinpath(SOURCE,"configs/main_instance_parameters.toml"))["model"]
    @assert md==TOML.parsefile(joinpath(BANKS,"configs/I0252.toml"))["model"]
    cfg=H.from_dict(md)
    @assert R.model_digest(cfg)==s016["model_sha256"]==s012["reference_model_sha256"]
    @assert s011["max_iters"]==s016["max_iters"]==100&&s011["n_paths"]==s016["n_paths"]==500
    @assert s011["jobs"]==s016["jobs"]==45&&length(C.build_policy_grid())==144
    @assert s011["Z0"]==s016["Z0"]==C.Z0==[.12,.10]
    @assert s011["BASE_STEP"]==s016["BASE_STEP"]==C.BASE_STEP==.035
    @assert s011["GRAD_CLIP"]==s016["GRAD_CLIP"]==C.GRAD_CLIP==15.
    @assert sha(joinpath(SOURCE,"configs/policies.csv"))==sha(joinpath(BANKS,"configs/policies.csv"))
    @assert sha(joinpath(SOURCE,"high/I0252_surface.csv"))==sha(joinpath(BANKS,"configs/I0252_surface.csv"))
    checks=NamedTuple[];seedrows=NamedTuple[]
    for rep in REPS
        a=oldarchive(rep);green=only(filter(r->r.method.id=="Green",a.runs))
        g=archived(rep,"Green");@assert isequal(green.zhist,g.zhist)&&isequal(green.diagnostics,g.core)
        oldfinal=deserialize(joinpath(SOURCE,"checkpoints/evaluation_$(ID)_$rep.jls"))
        for id in filter(!=("Exp50"),METHODS)
            v=archived(rep,id);@assert length(v.zhist)==101&&v.zhist[1]==C.Z0
            if id=="GreenSoftmax050"
                @assert v.method.kind==:green_softmax&&v.method.beta==.5
                timing=deserialize(joinpath(TIMING,"checkpoints/timing_$(rep)_$id.jls"))
                @assert isequal(v.zhist,timing.run.zhist)&&isequal(v.diagnostics,timing.run.diagnostics)
                f=only(filter(r->r.method==id,oldfinal.finals))
            else
                @assert v.iterations==100&&v.new_paths==50000
                id=="Filter2"&&(@assert v.method.family==:filtered&&v.method.param==2.)
                id=="Rolling5"&&(@assert v.method.family==:rolling&&v.method.param==5.)
                f=deserialize(joinpath(BANKS,"checkpoints/final_$(rep)_$id.jls")).row
            end
            @assert [f.z3,f.z4]==v.zhist[end]
            @assert f.selection_paths==10000&&f.holdout_paths==50000
            @assert f.selection_seed==X.seed(ID,"compare",rep,"selection")&&f.holdout_seed==X.seed(ID,"compare",rep,"holdout")
            push!(checks,(rep=rep,method=id,model_sha256=R.model_digest(cfg),policies=144,
                iterations=100,new_paths_per_iteration=500,z0_3=C.Z0[1],z0_4=C.Z0[2],jobs=45,
                base_step=C.BASE_STEP,component_clip=C.GRAD_CLIP,lr_log_clip=35.,probability_clamp=1e-12,
                same_search_seeds=true,compatible=true,source=id=="GreenSoftmax050" ? "011 search/evaluation; 012 exact archived timing" : "016 search/evaluation/timing",
                final_protocol_equivalent=true))
        end
        for k in 1:100
            ss=X.seed(ID,"compare",rep,"search",k)
            @assert only(filter(r->r.stream=="search"&&r.iteration==k,a.seeds)).seed==ss
            for id in METHODS;push!(seedrows,(rep=rep,method=id,iteration=k,seed=ss));end
        end
    end
    # beta is inverse temperature: multiply value gaps, do NOT divide by beta.
    @assert R.softmax_weights([0.,2.];beta=.5)≈[1/(1+exp(1)),exp(1)/(1+exp(1))]
    standalone=TOML.parsefile(joinpath(ROOT,"resultados/003probing_softmax_20261021/completo/probing_four_metadata.toml"))
    @assert standalone["BETA"]!=cfg.beta&&standalone["ALPHA"]!=collect(cfg.alpha)
    E.write_csv(joinpath(OUTPUT,"validation/compatibility.csv"),checks)
    E.write_csv(joinpath(OUTPUT,"configs/search_seeds.csv"),seedrows)
    open(joinpath(OUTPUT,"validation/compatibility.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true,"reps"=>collect(REPS),"reused_searches"=>12,"new_searches_authorized"=>3,
            "standalone003_excluded"=>true,"reason003"=>"Different model/alpha, signal beta, service means and RNG protocol; standalone Softmax is fresh-local, not Green+Softmax",
            "softmax_beta_definition"=>"q_i=exp(0.5*(v_i-max(v)))/sum_j exp(0.5*(v_j-max(v))); inverse temperature",
            "softmax_gradient"=>"G' q-alpha; LSE surrogate with beta=.5; uniform weighting across historical banks",
            "softmax_internal_field"=>"Archived objective_internal is HARD max(v)-cost, not LSE; surrogate was not archived",
            "final_protocol_identical"=>true,"old_endpoint_reevaluation_needed"=>false,
            "timing_limit"=>"Archived 012/016 and new 017 measurements; sequential BLAS=1, similar scope, different sessions and diagnostic instrumentation; descriptive only"))
    end
    println("Compatibility PASSED: reuse 12 searches, only 3 Exp50 searches authorized; no rep4/rep5.");flush(stdout)
    cfg
end

function initialize()
    for f in ("configs","tables","checkpoints","validation","figures","reproduction","logs");mkpath(joinpath(OUTPUT,f));end
    cfg=preflight()
    d=Dict{String,Any}("output"=>OUTPUT,"max_iters"=>100,"n_paths"=>500,"jobs"=>45)
    files=["017probing_green_five_method_comparison.jl","PROTOCOLO_green_five_comparison_017.txt"]
    science=Dict("scenario"=>ID,"rep_ids"=>collect(REPS),"methods"=>METHODS,"only_new_method"=>"Exp50","rho"=>RHO,
        "iterations"=>100,"new_paths_per_iteration"=>500,"jobs"=>45,"policy_count"=>144,
        "Z0"=>C.Z0,"BASE_STEP"=>C.BASE_STEP,"GRAD_CLIP"=>C.GRAD_CLIP,"model_sha256"=>R.model_digest(cfg),
        "trace_paths"=>1000,"selection_paths"=>10000,"holdout_paths"=>50000,"julia_version"=>string(VERSION),
        "sources"=>Dict(f=>sha(joinpath(ROOT,f)) for f in [files;"016probing_green_bank_variants.jl";"probing_regime_model.jl";"010probing_instance_landscape_search.jl";"003probing_four_searches_softmax.jl"]),
        "objective"=>"max_i E_z[Y_i]-alpha'z; hard-max original objective for independent evaluation")
    path=joinpath(OUTPUT,"configs/experiment.toml");fp=R.digest(science)
    isfile(path)&&(@assert TOML.parsefile(path)["fingerprint"]==fp "Source/config changed; do not mix experiments")
    open(path,"w") do io;TOML.print(io,Dict("fingerprint"=>fp,"science"=>science);sorted=true);end
    for file in ("I0252.toml","policies.csv","I0252_surface.csv")
        cp(joinpath(BANKS,"configs",file),joinpath(OUTPUT,"configs",file);force=true)
    end
    for file in files;cp(joinpath(ROOT,file),joinpath(OUTPUT,"reproduction",file);force=true);end
    d,cfg
end

"""Tests only rho=.5. No original/Softmax/filter/window optimizations run."""
function tests(d,cfg)
    history=R.HistoricalBank[];policies=C.build_policy_grid();target=[.62,.61]
    results=@testset "Exp50 basic estimator and reproducibility tests" begin
        for (k,z) in enumerate(([.12,.10],[.40,.35],target))
            p=R.primitive_bank(cfg,20,X.seed(ID,"compare",1,"search",k);njobs=45)
            bank=R.materialize_bank(p,z,cfg);Y=R.rewards(policies,bank,cfg)
            push!(history,R.historical_bank(ID,cfg,z,bank,Y))
        end
        e=V.estimate(target,history,cfg,NEW)
        @test e.indices==[1,2,3]
        @test e.weights≈[.5^2,.5,1.]/sum([.5^2,.5,1.])
        @test all(e.weights.>=0)&&isapprox(sum(e.weights),1.;atol=1e-14)
        @test all(==(1.),e.cache[3].L)
        val=zeros(144);G=zeros(144,2)
        for h in 1:3,i in 1:144
            hb=history[h];L=e.cache[h].L;S=e.cache[h].S;a=e.weights[h]
            val[i]+=a*sum(hb.Y[i,j]*L[j] for j in 1:20)/20
            for dim in 1:2;G[i,dim]+=a*sum(hb.Y[i,j]*L[j]*S[dim,j] for j in 1:20)/20;end
        end
        @test e.data.values≈val
        @test e.data.best==argmax(val)
        @test e.data.hard≈G[argmax(val),:]-collect(cfg.alpha)
        dg=V.diagnostics(target,history,cfg,e)
        @test isfinite(dg.objective_variance)&&all(isfinite,dg.gradient_variance)
        # Small prefixes with the SAME rho=.5 and N=500, not other methods.
        u=V.search(d,cfg,1,NEW;K=3,n=500,extra=true)
        v=V.search(d,cfg,1,NEW;K=3,n=500,extra=true)
        @test isequal(u.zhist,v.zhist)&&isequal(u.core,v.core)
        E.atomic_serialize(joinpath(OUTPUT,"validation/exp50_test_prefix.jls"),u)
    end
    open(joinpath(OUTPUT,"validation/tests.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true,"assertions"=>Test.get_test_counts(results).passes,
            "only_rho"=>RHO,"other_optimizations_executed"=>0,"small_prefix_paths"=>3000,"fixture_paths"=>60))
    end
end

function searches(d,cfg)
    runs=Dict{Tuple{Int,String},Any}();coords=NamedTuple[];diags=NamedTuple[];provenance=NamedTuple[];newbanks=NamedTuple[]
    for rep in REPS
        p=joinpath(OUTPUT,"checkpoints/search_$(rep)_Exp50.jls")
        if isfile(p);v=deserialize(p)
        else
            GC.gc();v=V.search(d,cfg,rep,NEW);E.atomic_serialize(p,v)
        end
        @assert v.method==NEW&&v.iterations==100&&v.new_paths==50000
        if rep==1
            q=deserialize(joinpath(OUTPUT,"validation/exp50_test_prefix.jls"))
            @assert isequal(v.zhist[1:4],q.zhist)&&isequal(v.core[1:3],q.core)
        end
        runs[(rep,"Exp50")]=v;append!(newbanks,v.bankrows)
        @printf("NEW search complete: Exp50 rep%d z=(%.8f,%.8f), %.3fs\n",rep,v.zhist[end]...,v.search_seconds);flush(stdout)
        for id in METHODS
            id=="Exp50"||(runs[(rep,id)]=archived(rep,id))
            run=runs[(rep,id)];origin=id=="Exp50" ? "017" : id=="GreenSoftmax050" ? "011" : "016"
            soft=id=="GreenSoftmax050";core=soft ? run.diagnostics : run.core
            timing=soft ? deserialize(joinpath(TIMING,"checkpoints/timing_$(rep)_$id.jls")) : nothing
            for (j,z) in enumerate(run.zhist)
                push!(coords,(scenario=ID,rep=rep,method=id,iteration=j-1,z3=z[1],z4=z[2],search_source=origin))
            end
            for k in 1:100
                r=core[k];z=run.zhist[k];zn=run.zhist[k+1]
                row=soft ? nothing : run.rows[k]
                retained=soft ? k : row.retained_banks
                push!(diags,(rep=rep,method=id,iteration=k,target_z3=z[1],target_z4=z[2],next_z3=zn[1],next_z4=zn[2],
                    g3=r.g3,g4=r.g4,gradient_norm=r.gradient_norm,step_norm=r.step_norm,
                    clipping=r.clipping,projection=r.projection,ess=r.ess,ess_fraction=r.ess/(500retained),lr_clips=r.lr_clips,
                    selected_hard_policy=r.hard_policy,effective_policies=r.K_eff,
                    internal_hard_max=r.objective_internal,internal_lse_surrogate=missing,
                    gradient_target=soft ? "LSE beta=.5" : "common selected hard-max policy",
                    internal_value_definition="max estimated policy means minus probing cost; not LSE",
                    generated_banks=k,retained_banks=retained,retained_observations=500retained,reused_observations=500(retained-1),
                    logical_new_paths=500,cumulative_logical_paths=500k,new_paths_generated_this_study=id=="Exp50" ? 500 : 0,
                    search_seconds=soft ? timing.rows[k+1].search_seconds : row.search_seconds,
                    cumulative_search_seconds=soft ? timing.rows[k+1].cumulative_search_seconds : row.cumulative_search_seconds,
                    diagnostic_seconds=soft ? missing : row.diagnostic_seconds,
                    search_source=origin,timing_source=soft ? "012 archived sequential exact replay" : "$origin sequential measurement"))
            end
            seconds=soft ? timing.rows[end].cumulative_search_seconds : run.search_seconds
            push!(provenance,(rep=rep,method=id,search_source=origin,optimization_executed_this_study=id=="Exp50",
                optimization_seconds=seconds,diagnostic_seconds=soft ? missing : run.diagnostic_seconds,
                timing_source=soft ? "012 archived exact replay" : "$origin sequential; extra diagnostics excluded",
                logical_search_paths=50000,new_paths_this_study=id=="Exp50" ? 50000 : 0))
        end
    end
    E.write_csv(joinpath(OUTPUT,"tables/coordinates.csv"),coords)
    E.write_csv(joinpath(OUTPUT,"tables/iteration_diagnostics.csv"),diags)
    E.write_csv(joinpath(OUTPUT,"tables/provenance_and_runtime.csv"),provenance)
    E.write_csv(joinpath(OUTPUT,"tables/exp50_banks.csv"),newbanks)
    runs
end

function direct_point(cfg,rep,z,bank,seed)
    key=bytes2hex(sha256(repr(z)));path=joinpath(OUTPUT,"checkpoints/trace_$(rep)_$key.jls")
    isfile(path)&&return deserialize(path)
    candidates=[joinpath(BANKS,"checkpoints/trace_$(rep)_$key.jls"),joinpath(TIMING,"checkpoints/point_main_$(rep)_$key.jls")]
    donor=findfirst(isfile,candidates)
    if donor!==nothing
        p=deserialize(candidates[donor]);@assert isequal(p.z,z)
        n=hasproperty(p,:n) ? p.n : p.n_paths;@assert n==1000
        hasproperty(p,:seed)&&(@assert p.seed==seed)
        v=(z=z,objective=p.objective,se=p.se,policy=p.policy,n=1000,seed=seed,source=donor==1 ? "016 exact direct point reuse" : "012 exact direct point reuse")
    else
        Y=R.rewards(C.build_policy_grid(),R.materialize_bank(bank,z,cfg),cfg);means=vec(mean(Y;dims=2));i=argmax(means)
        v=(z=z,objective=means[i]-R.cost(cfg,z),se=std(view(Y,i,:))/sqrt(1000),policy=i,n=1000,seed=seed,source="017 new direct original-objective trace")
    end
    E.atomic_serialize(path,v);v
end

function evaluate_rep(cfg,rep,runs)
    path=joinpath(OUTPUT,"checkpoints/evaluation_$rep.jls");isfile(path)&&return deserialize(path)
    ts=X.seed(ID,"compare",rep,"trace");ss=X.seed(ID,"compare",rep,"selection");hs=X.seed(ID,"compare",rep,"holdout")
    @assert isempty(intersect([X.seed(ID,"compare",rep,"search",k) for k in 1:100],[ts,ss,hs]))
    trace=R.primitive_bank(cfg,1000,ts;njobs=45);traces=NamedTuple[];finals=NamedTuple[];pairs=NamedTuple[]
    oldeval=deserialize(joinpath(SOURCE,"checkpoints/evaluation_$(ID)_$rep.jls"))
    for id in METHODS
        run=runs[(rep,id)]
        for (j,z) in enumerate(run.zhist)
            p=direct_point(cfg,rep,z,trace,ts)
            if id=="GreenSoftmax050"
                row=only(filter(r->r.method==id&&r.iteration==j-1,oldeval.traces))
                @assert isequal(z,[row.z3,row.z4])&&isapprox(row.objective,p.objective;atol=1e-12)
                @assert row.trace_paths==1000&&row.seed==ts
            end
            push!(traces,(rep=rep,method=id,iteration=j-1,z3=z[1],z4=z[2],objective=p.objective,conditional_se=p.se,
                selected_policy=p.policy,trace_paths=1000,evaluation_seed=ts,source=p.source,
                estimator="direct max of 144 policy means minus cost; common independent CRN"))
        end
    end
    # Existing selected-policy final evaluations are equivalent: reuse them.
    for id in filter(!=("Exp50"),METHODS)
        z=runs[(rep,id)].zhist[end]
        if id=="GreenSoftmax050"
            f=only(filter(r->r.method==id,oldeval.finals));obj=f.objective_holdout;se=f.evaluation_se
            p=only(filter(r->r.method_a==id&&r.method_b=="Green",oldeval.paired))
            push!(pairs,(rep=rep,method=id,reference="Green",difference=p.difference,paired_mc_se=p.paired_se,
                ci95_low=p.difference-Z95*p.paired_se,ci95_high=p.difference+Z95*p.paired_se,holdout_paths=50000,
                source="011 archived paired holdout; Green control exactly identical to 016"))
            origin="011 original 10000/50000 selected-policy holdout; no reevaluation"
        else
            v=deserialize(joinpath(BANKS,"checkpoints/final_$(rep)_$id.jls"));f=v.row;obj=f.objective;se=f.evaluation_se
            if id!="Green"
                p=only(filter(r->r.method==id,deserialize(joinpath(BANKS,"checkpoints/evaluation_$rep.jls")).paired))
                push!(pairs,merge(p,(source="016 archived paired holdout",)))
            end
            origin="016 original 10000/50000 selected-policy holdout; no reevaluation"
        end
        @assert [f.z3,f.z4]==z&&f.selection_seed==ss&&f.holdout_seed==hs
        push!(finals,(rep=rep,method=id,z3=z[1],z4=z[2],selected_policy=f.selected_policy,objective=obj,evaluation_se=se,
            ci95_low=obj-Z95*se,ci95_high=obj+Z95*se,probing_cost=R.cost(cfg,z),selection_paths=10000,holdout_paths=50000,
            selection_seed=ss,holdout_seed=hs,source=origin))
    end
    ep=joinpath(OUTPUT,"checkpoints/final_$(rep)_Exp50.jls")
    if isfile(ep);v=deserialize(ep)
    else
        z=runs[(rep,"Exp50")].zhist[end];policies=C.build_policy_grid();cost=R.cost(cfg,z)
        selector=R.primitive_bank(cfg,10000,ss;njobs=45);hold=R.primitive_bank(cfg,50000,hs;njobs=45)
        Ys=R.rewards(policies,R.materialize_bank(selector,z,cfg),cfg);chosen=argmax(vec(mean(Ys;dims=2)))
        raw=[R.evaluate_policy(policies[chosen],p,cfg) for p in R.materialize_bank(hold,z,cfg)]
        obj=mean(raw)-cost;se=std(raw)/sqrt(50000)
        row=(rep=rep,method="Exp50",z3=z[1],z4=z[2],selected_policy=chosen,objective=obj,evaluation_se=se,
            ci95_low=obj-Z95*se,ci95_high=obj+Z95*se,probing_cost=cost,selection_paths=10000,holdout_paths=50000,
            selection_seed=ss,holdout_seed=hs,source="017 new independent direct selected-policy holdout")
        v=(row=row,rewards=raw.-cost);E.atomic_serialize(ep,v)
    end
    push!(finals,v.row)
    original=deserialize(joinpath(BANKS,"checkpoints/final_$(rep)_Green.jls"))
    delta=v.rewards-original.rewards;mu=mean(delta);se=std(delta)/sqrt(50000)
    push!(pairs,(rep=rep,method="Exp50",reference="Green",difference=mu,paired_mc_se=se,
        ci95_low=mu-Z95*se,ci95_high=mu+Z95*se,holdout_paths=50000,source="017 Exp50 minus unchanged 016 Green holdout vectors"))
    v=(traces=traces,finals=finals,paired=pairs);E.atomic_serialize(path,v)
    println("Evaluation complete: rep$rep; only Exp50 endpoints newly simulated; four other evaluations reused.");flush(stdout);v
end

function main(args=ARGS)
    all(a in ("--resume","--no-plots","--plots-only","--preflight-only","--tests-only") for a in args)||error("Unknown argument")
    BLAS.set_num_threads(1);d,cfg=initialize()
    "--preflight-only" in args&&return
    if !("--plots-only" in args)
        # Never run V.tests or V.main: those would rerun unrequested methods.
        if !isfile(joinpath(OUTPUT,"validation/tests.toml"))
            tests(d,cfg)
        elseif any(!isfile(joinpath(OUTPUT,"checkpoints/search_$(rep)_Exp50.jls")) for rep in REPS)
            # Same rho=.5 only, pure estimator warmup; no optimization replay.
            p=R.primitive_bank(cfg,20,X.seed(ID,"compare",1,"search",1);njobs=45)
            bank=R.materialize_bank(p,C.Z0,cfg);Y=R.rewards(C.build_policy_grid(),bank,cfg)
            V.estimate(C.Z0,[R.historical_bank(ID,cfg,C.Z0,bank,Y)],cfg,NEW)
        end
        "--tests-only" in args&&return
        runs=searches(d,cfg)
        results=Vector{Any}(undef,3)
        Threads.@threads :dynamic for j in 1:3;results[j]=evaluate_rep(cfg,j,runs);end
        for (name,field) in (("objective_traces",:traces),("final_results",:finals),("paired_vs_green",:paired))
            E.write_csv(joinpath(OUTPUT,"tables/$name.csv"),vcat([getproperty(v,field) for v in results]...))
        end
        open(joinpath(OUTPUT,"validation/science_complete.toml"),"w") do io
            TOML.print(io,Dict("complete"=>true,"new_optimizations"=>3,"reused_optimizations"=>12,"rep_ids"=>[1,2,3],
                "new_production_search_paths"=>150000,"reused_logical_search_paths"=>600000,"new_final_evaluations"=>3,
                "reused_final_evaluations"=>12,"extra_parameter_combinations"=>0,"rep4_or_rep5_executed"=>false))
        end
    end
    if !("--no-plots" in args)
        run(`python3 $(joinpath(ROOT,"analyze_green_five_comparison_017.py"))`)
    end
end
end
if abspath(PROGRAM_FILE)==@__FILE__;ProbingGreenFiveComparison.main();end
