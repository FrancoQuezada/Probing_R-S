# Batch-size configuration study only. All optimizers/estimators are reused.
include(joinpath(@__DIR__,"017probing_green_five_method_comparison.jl"))
module ProbingGreenSampleSize
using ..ProbingGreenFiveComparison
using Statistics, LinearAlgebra, Serialization, SHA, TOML, Printf, Test
const P=ProbingGreenFiveComparison;const V=P.V;const A=P.A;const H=P.H;const X=P.X;const R=P.R;const E=P.E;const C=P.C
const ROOT=@__DIR__;const ID="I0252";const REPS=1:3;const METHODS=P.METHODS;const Z95=P.Z95
const SOURCE=P.OUTPUT;const OUTPUT=joinpath(ROOT,"resultados/018probing_green_n250_vs_n500")
const EFF=joinpath(ROOT,"resultados/013probing_green_simulation_efficiency")
const HARD=Dict("Green"=>(id="Green",family=:original,param=0.),"Exp50"=>P.NEW,
    "Filter2"=>(id="Filter2",family=:filtered,param=2.),"Rolling5"=>(id="Rolling5",family=:rolling,param=5.))
const SOFT=(id="GreenSoftmax050",kind=:green_softmax,beta=.5)
sha(p)=bytes2hex(sha256(read(p)))
zs(v,id)=id==SOFT.id ? v.run.zhist : v.zhist
core(v,id)=id==SOFT.id ? v.run.diagnostics : v.core
seconds(v,id)=id==SOFT.id ? v.rows[end].cumulative_search_seconds : v.search_seconds
function oldraw(rep,id)
    id==SOFT.id&&return deserialize(joinpath(P.TIMING,"checkpoints/timing_$(rep)_$id.jls"))
    id=="Exp50" ? deserialize(joinpath(SOURCE,"checkpoints/search_$(rep)_$id.jls")) : P.archived(rep,id)
end
function run250(d,cfg,rep,id;K=100)
    id==SOFT.id ? A.timed_method(d,cfg,rep,SOFT;K=K) : V.search(d,cfg,rep,HARD[id];K=K,n=250)
end

function initialize()
    for dir in ("configs","tables","checkpoints","validation","figures","logs","reproduction");mkpath(joinpath(OUTPUT,dir));end
    prior=TOML.parsefile(joinpath(SOURCE,"configs/experiment.toml"))["science"]
    for (f,h) in prior["sources"];@assert sha(joinpath(ROOT,f))==h "Changed original source $f";end
    oldscience=TOML.parsefile(joinpath(P.SOURCE,"configs/experiment.toml"))["science"]
    for (f,h) in oldscience["sources"];@assert sha(joinpath(ROOT,f))==h;end
    cfg=H.from_dict(TOML.parsefile(joinpath(SOURCE,"configs/I0252.toml"))["model"])
    @assert R.model_digest(cfg)==prior["model_sha256"]&&prior["rep_ids"]==[1,2,3]
    @assert prior["new_paths_per_iteration"]==500&&prior["iterations"]==100&&prior["jobs"]==45
    @assert prior["Z0"]==C.Z0==[.12,.10]&&prior["BASE_STEP"]==C.BASE_STEP==.035&&prior["GRAD_CLIP"]==C.GRAD_CLIP==15.
    @assert length(C.build_policy_grid())==144&&prior["selection_paths"]==10000&&prior["holdout_paths"]==50000
    @assert sha(joinpath(SOURCE,"configs/I0252_surface.csv"))==sha(joinpath(P.SOURCE,"high/I0252_surface.csv"))
    compatibility=NamedTuple[];seeds=NamedTuple[]
    for rep in REPS
        archive=P.oldarchive(rep);ev=deserialize(joinpath(SOURCE,"checkpoints/evaluation_$rep.jls"))
        for id in METHODS
            v=oldraw(rep,id);@assert length(zs(v,id))==101&&length(core(v,id))==100&&zs(v,id)[1]==C.Z0
            f=only(filter(r->r.method==id,ev.finals));@assert [f.z3,f.z4]==zs(v,id)[end]
            @assert f.selection_paths==10000&&f.holdout_paths==50000
            @assert f.selection_seed==X.seed(ID,"compare",rep,"selection")&&f.holdout_seed==X.seed(ID,"compare",rep,"holdout")
            if id==SOFT.id
                original=only(filter(r->r.method.id==id,archive.runs))
                @assert original.method.kind==:green_softmax&&original.method.beta==.5
                @assert isequal(original.zhist,v.run.zhist)&&isequal(original.diagnostics,v.run.diagnostics)
            end
            push!(compatibility,(rep=rep,method=id,old_n=500,new_n=250,iterations=100,policies=144,
                model_sha256=R.model_digest(cfg),only_optimization_setting_changed="new paths per iteration",
                old_optimization_reused=true,old_final_protocol_identical=true))
        end
        for k in 1:100
            ss=X.seed(ID,"compare",rep,"search",k)
            @assert only(filter(r->r.stream=="search"&&r.iteration==k,archive.seeds)).seed==ss
            for n in (500,250),id in METHODS;push!(seeds,(n_paths=n,rep=rep,method=id,iteration=k,seed=ss));end
        end
    end
    @assert R.softmax_weights([0.,2.];beta=.5)≈[1/(1+exp(1)),exp(1)/(1+exp(1))]
    E.write_csv(joinpath(OUTPUT,"validation/compatibility.csv"),compatibility)
    E.write_csv(joinpath(OUTPUT,"configs/search_seeds.csv"),seeds)
    d=Dict{String,Any}("n_paths"=>250,"jobs"=>45,"max_iters"=>100,"bootstrap_reps"=>300)
    sourcefiles=["018probing_green_n250_vs_n500.jl","PROTOCOLO_green_n250_vs_n500_018.txt",
        "016probing_green_bank_variants.jl","017probing_green_five_method_comparison.jl","probing_anytime_012_runtime.jl",
        "probing_regime_model.jl","003probing_four_searches_softmax.jl"]
    science=Dict("scenario"=>ID,"rep_ids"=>collect(REPS),"methods"=>METHODS,"new_n"=>250,"old_n"=>500,
        "iterations"=>100,"jobs"=>45,"policies"=>144,"Z0"=>C.Z0,"BASE_STEP"=>C.BASE_STEP,"GRAD_CLIP"=>C.GRAD_CLIP,
        "model_sha256"=>R.model_digest(cfg),"trace_paths"=>1000,"selection_paths"=>10000,"holdout_paths"=>50000,
        "new_optimizations"=>15,"old_optimizations_reexecuted"=>0,"julia_version"=>string(VERSION),
        "sources"=>Dict(f=>sha(joinpath(ROOT,f)) for f in sourcefiles))
    fp=R.digest(science);path=joinpath(OUTPUT,"configs/experiment.toml")
    isfile(path)&&(@assert TOML.parsefile(path)["fingerprint"]==fp "Different source/config; do not mix results")
    open(path,"w") do io;TOML.print(io,Dict("fingerprint"=>fp,"science"=>science);sorted=true);end
    for f in ("I0252.toml","policies.csv","I0252_surface.csv");cp(joinpath(SOURCE,"configs",f),joinpath(OUTPUT,"configs",f);force=true);end
    for f in sourcefiles[1:2];cp(joinpath(ROOT,f),joinpath(OUTPUT,"reproduction",f);force=true);end
    println("Compatibility PASSED: only n changes 500 -> 250; 15 new searches; no n500 replay.");flush(stdout)
    d,cfg
end

function tests(d,cfg)
    policies=C.build_policy_grid();history=R.HistoricalBank[]
    result=@testset "n250 fixed configuration and estimator tests" begin
        for (k,z) in enumerate(([.12,.10],[.4,.35],[.65,.6]))
            p=R.primitive_bank(cfg,20,X.seed(ID,"compare",1,"search",k);njobs=45)
            b=R.materialize_bank(p,z,cfg);push!(history,R.historical_bank(ID,cfg,z,b,R.rewards(policies,b,cfg)))
        end
        z=history[end].zgen
        for id in keys(HARD)
            e=V.estimate(z,history,cfg,HARD[id]);mom=V.diagnostics(z,history,cfg,e)
            @test all(e.weights.>=0)&&isapprox(sum(e.weights),1.;atol=1e-14)
            @test 3 in e.indices
            @test all(==(1.),V.bank_kernel(z,history[end],cfg).L)
            @test all(isfinite,e.data.hard)&&isfinite(mom.objective_variance)
        end
        data=R.green_data(z,history,cfg,ID);q=R.softmax_weights(data.values;beta=.5)
        expected=exp.(.5 .* (data.values .- maximum(data.values)))
        @test q≈expected./sum(expected)
        @test R.combine(data.G,q,cfg)≈transpose(data.G)*q-collect(cfg.alpha)
        for id in METHODS
            a=run250(d,cfg,1,id;K=4);b=run250(d,cfg,1,id;K=4)
            @test isequal(zs(a,id),zs(b,id))&&isequal(core(a,id),core(b,id))
            @test length(zs(a,id))==5
            E.atomic_serialize(joinpath(OUTPUT,"validation/test_prefix_$id.jls"),a)
        end
    end
    open(joinpath(OUTPUT,"validation/tests.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true,"assertions"=>Test.get_test_counts(result).passes,
            "new_n"=>250,"n500_optimizations_executed"=>0,"test_prefix_paths"=>10000,"fixture_paths"=>60))
    end
end

function searches(d,cfg)
    runs=Dict{Tuple{Int,Int,String},Any}();coordinates=NamedTuple[];runtime=NamedTuple[]
    for rep in REPS,id in METHODS
        path=joinpath(OUTPUT,"checkpoints/search_250_$(rep)_$id.jls")
        if isfile(path);raw=deserialize(path).raw
        else
            GC.gc();raw=run250(d,cfg,rep,id);E.atomic_serialize(path,(n_paths=250,rep=rep,method=id,raw=raw))
        end
        @assert length(zs(raw,id))==101
        if rep==1
            q=deserialize(joinpath(OUTPUT,"validation/test_prefix_$id.jls"))
            @assert isequal(zs(raw,id)[1:5],zs(q,id))&&isequal(core(raw,id)[1:4],core(q,id))
        end
        # Independently archived n250 Green also checks exact algorithm reuse.
        if id=="Green"
            donor=joinpath(EFF,"checkpoints/search_Green-250_$rep.jls")
            if isfile(donor)
                old=deserialize(donor).run
                @assert isequal(raw.zhist,old.zhist)&&isequal(raw.core,old.diagnostics)
            end
        end
        runs[(250,rep,id)]=raw;runs[(500,rep,id)]=oldraw(rep,id)
        for n in (500,250)
            v=runs[(n,rep,id)]
            for (j,z) in enumerate(zs(v,id));push!(coordinates,(n_paths=n,rep=rep,method=id,iteration=j-1,z3=z[1],z4=z[2]));end
            push!(runtime,(n_paths=n,rep=rep,method=id,optimization_seconds=seconds(v,id),
                additional_moment_seconds=id==SOFT.id ? missing : v.diagnostic_seconds,
                iterations=100,logical_search_paths=100n,new_search_paths_this_study=n==250 ? 25000 : 0,
                source=n==250 ? (id==SOFT.id ? "018 using unchanged A.timed_method" : "018 using unchanged V.search") : "017 archived controls; original timings 012/016/017"))
        end
        @printf("n250 search ready: rep%d %-16s z=(%.8f,%.8f), %.3fs\n",rep,id,zs(raw,id)[end]...,seconds(raw,id));flush(stdout)
    end
    E.write_csv(joinpath(OUTPUT,"tables/coordinates.csv"),coordinates)
    E.write_csv(joinpath(OUTPUT,"tables/runtime.csv"),runtime)
    runs
end

"""Diagnostic reconstruction at FIXED archived decisions. No projected steps
or optimizer calls. Softmax gradient moments use its ACTUAL policy mixture,
while objective moments use the hard-max diagnostic policy for all methods.
"""
function soft_moments(cfg,n,rep,raw)
    path=joinpath(OUTPUT,"checkpoints/soft_moments_$(n)_$rep.jls");isfile(path)&&return deserialize(path)
    rows=NamedTuple[];banks=NamedTuple[];history=R.HistoricalBank[];policies=C.build_policy_grid();a=time_ns()
    for k in 1:100
        z=zs(raw,SOFT.id)[k];p=R.primitive_bank(cfg,n,X.seed(ID,"compare",rep,"search",k);njobs=45)
        bank=R.materialize_bank(p,z,cfg);Y=R.rewards(policies,bank,cfg)
        push!(history,R.historical_bank(ID,cfg,z,bank,Y))
        data=R.green_data(z,history,cfg,ID);q=R.softmax_weights(data.values;beta=.5);grad=R.combine(data.G,q,cfg)
        c=core(raw,SOFT.id)[k]
        @assert isequal(grad,[c.g3,c.g4])&&data.best==c.hard_policy&&isequal(data.ess,c.ess)&&data.lr_clips==c.lr_clips
        vo=0.;vg=zeros(2);vop=0.;vgp=zeros(2);vsoft=0.;gm=-collect(cfg.alpha)
        for (h,hb) in enumerate(history)
            @assert length(hb.bank)==n
            ker=V.bank_kernel(z,hb,cfg);hard=view(hb.Y,data.best,:).*ker.L
            mixture=(transpose(hb.Y)*q).*ker.L
            gcontrib=[mixture.*view(ker.S,j,:) for j in 1:2]
            ov=var(hard);gv=[var(v) for v in gcontrib];sv=var(mixture)
            vo+=ov/k;vg.+=gv/k;vop+=ov/(k^2*n);vgp.+=gv/(k^2*n);vsoft+=sv/k
            for j in 1:2;gm[j]+=mean(gcontrib[j])/k;end
            h==k&&(@assert all(==(1.),ker.L))
            push!(banks,(n_paths=n,rep=rep,method=SOFT.id,iteration=k,bank_iteration=h,bank_paths=n,bank_weight=1/k,
                hard_policy=data.best,objective_variance=ov,soft_mixture_objective_variance=sv,
                gradient3_variance=gv[1],gradient4_variance=gv[2]))
        end
        @assert all(isfinite,[vo;vg;vop;vgp;vsoft])&&all(isapprox.(gm,grad;atol=1e-8,rtol=1e-12))
        push!(rows,(iteration=k,objective_contribution_variance=vo,gradient3_contribution_variance=vg[1],
            gradient4_contribution_variance=vg[2],objective_mean_variance_proxy=vop,
            gradient3_mean_variance_proxy=vgp[1],gradient4_mean_variance_proxy=vgp[2],
            soft_mixture_objective_variance=vsoft,policy_weights=join(q,";")))
    end
    v=(rows=rows,banks=banks,seconds=(time_ns()-a)/1e9,reconstructed_primitive_paths=100n,optimizer_updates=0)
    E.atomic_serialize(path,v);println("Softmax contribution moments verified: n=$n rep$rep; zero optimizer updates.");flush(stdout);v
end

function diagnostics_rep(cfg,rep,runs)
    path=joinpath(OUTPUT,"checkpoints/diagnostics_$rep.jls");isfile(path)&&return deserialize(path)
    rows=NamedTuple[];bankrows=NamedTuple[];ledger=NamedTuple[]
    for n in (500,250),id in METHODS
        raw=runs[(n,rep,id)];soft=id==SOFT.id
        sm=soft ? soft_moments(cfg,n,rep,raw) : nothing
        if soft
            append!(bankrows,sm.banks)
            push!(ledger,(n_paths=n,rep=rep,phase="Softmax moment reconstruction at fixed decisions",primitive_paths=sm.reconstructed_primitive_paths,
                optimizer_updates=0,seconds=sm.seconds,paths_charged_to_optimization=0))
        elseif n==250
            append!(bankrows,[(n_paths=n,rep=rep,method=id,iteration=r.iteration,bank_iteration=r.bank_iteration,bank_paths=r.paths,
                bank_weight=r.bank_weight,hard_policy=r.selected_policy,objective_variance=r.objective_contribution_sample_variance,
                soft_mixture_objective_variance=missing,gradient3_variance=r.gradient3_contribution_sample_variance,
                gradient4_variance=r.gradient4_contribution_sample_variance) for r in raw.bankrows])
        end
        for k in 1:100
            c=core(raw,id)[k];m=soft ? sm.rows[k] : raw.rows[k]
            z=zs(raw,id)[k];zn=zs(raw,id)[k+1];retained=soft ? k : raw.rows[k].retained_banks
            newpaths=soft ? raw.rows[k+1].new_trajectories : raw.rows[k].new_paths
            @assert newpaths==n
            push!(rows,(n_paths=n,rep=rep,method=id,iteration=k,target_z3=z[1],target_z4=z[2],next_z3=zn[1],next_z4=zn[2],
                g3=c.g3,g4=c.g4,gradient_norm=c.gradient_norm,step_norm=c.step_norm,clipping=c.clipping,projection=c.projection,
                ess=c.ess,ess_fraction=c.ess/(n*retained),lr_clips=c.lr_clips,hard_policy=c.hard_policy,effective_policies=c.K_eff,
                internal_hard_objective=c.objective_internal,generated_banks=k,retained_banks=retained,
                retained_observations=n*retained,reused_observations=n*(retained-1),logical_new_paths=n,cumulative_logical_paths=n*k,
                objective_contribution_variance=m.objective_contribution_variance,
                gradient3_contribution_variance=m.gradient3_contribution_variance,gradient4_contribution_variance=m.gradient4_contribution_variance,
                objective_mean_variance_proxy=m.objective_mean_variance_proxy,
                gradient3_mean_variance_proxy=m.gradient3_mean_variance_proxy,gradient4_mean_variance_proxy=m.gradient4_mean_variance_proxy,
                soft_mixture_objective_variance=soft ? m.soft_mixture_objective_variance : missing,
                search_seconds=soft ? raw.rows[k+1].search_seconds : m.search_seconds,
                cumulative_search_seconds=soft ? raw.rows[k+1].cumulative_search_seconds : m.cumulative_search_seconds,
                gradient_contribution_definition=soft ? "LR times Softmax policy-mixture reward times target score" : "LR times common selected-policy reward times target score"))
        end
    end
    v=(rows=rows,banks=bankrows,ledger=ledger);E.atomic_serialize(path,v);v
end

function trace_point(cfg,rep,z,bank,seed)
    key=bytes2hex(sha256(repr(z)));path=joinpath(OUTPUT,"checkpoints/trace_$(rep)_$key.jls")
    isfile(path)&&return deserialize(path)
    donors=[joinpath(SOURCE,"checkpoints/trace_$(rep)_$key.jls"),joinpath(P.BANKS,"checkpoints/trace_$(rep)_$key.jls"),
        joinpath(EFF,"checkpoints/point_main_$(rep)_$key.jls")]
    ix=findfirst(isfile,donors)
    if ix!==nothing
        p=deserialize(donors[ix]);@assert isequal(p.z,z)
        @assert (hasproperty(p,:n) ? p.n : p.n_paths)==1000
        hasproperty(p,:seed)&&(@assert p.seed==seed)
        v=(z=z,objective=p.objective,se=p.se,policy=p.policy,source="existing compatible direct CRN point")
    else
        Y=R.rewards(C.build_policy_grid(),R.materialize_bank(bank,z,cfg),cfg);means=vec(mean(Y;dims=2));i=argmax(means)
        v=(z=z,objective=means[i]-R.cost(cfg,z),se=std(view(Y,i,:))/sqrt(1000),policy=i,source="new direct original-objective point")
    end
    E.atomic_serialize(path,v);v
end
function endpoint_row(n,rep,id,z,chosen,obj,se,cfg,source)
    (n_paths=n,rep=rep,method=id,z3=z[1],z4=z[2],selected_policy=chosen,objective=obj,evaluation_se=se,
        ci95_low=obj-Z95*se,ci95_high=obj+Z95*se,probing_cost=R.cost(cfg,z),selection_paths=10000,holdout_paths=50000,
        selection_seed=X.seed(ID,"compare",rep,"selection"),holdout_seed=X.seed(ID,"compare",rep,"holdout"),source=source)
end
function evaluate_rep(cfg,rep,runs)
    path=joinpath(OUTPUT,"checkpoints/evaluation_$rep.jls");isfile(path)&&return deserialize(path)
    archived=deserialize(joinpath(SOURCE,"checkpoints/evaluation_$rep.jls"))
    ts=X.seed(ID,"compare",rep,"trace");ss=X.seed(ID,"compare",rep,"selection");hs=X.seed(ID,"compare",rep,"holdout")
    @assert isempty(intersect([X.seed(ID,"compare",rep,"search",k) for k in 1:100],[ts,ss,hs]))
    trace=R.primitive_bank(cfg,1000,ts;njobs=45);traces=NamedTuple[];finals=NamedTuple[];pairs=NamedTuple[];cross=NamedTuple[]
    for r in archived.traces
        @assert r.trace_paths==1000&&r.evaluation_seed==ts
        push!(traces,(n_paths=500,rep=rep,method=r.method,iteration=r.iteration,z3=r.z3,z4=r.z4,objective=r.objective,
            conditional_se=r.conditional_se,selected_policy=r.selected_policy,trace_paths=1000,seed=ts,source="017 exact archived trace reuse"))
    end
    for id in METHODS
        for (j,z) in enumerate(zs(runs[(250,rep,id)],id))
            v=trace_point(cfg,rep,z,trace,ts)
            push!(traces,(n_paths=250,rep=rep,method=id,iteration=j-1,z3=z[1],z4=z[2],objective=v.objective,
                conditional_se=v.se,selected_policy=v.policy,trace_paths=1000,seed=ts,source=v.source))
        end
        println("n250 direct trace ready: rep$rep $id");flush(stdout)
    end
    selector=R.primitive_bank(cfg,10000,ss;njobs=45);hold=R.primitive_bank(cfg,50000,hs;njobs=45);policies=C.build_policy_grid()
    vectors=Dict{Tuple{Int,String},Vector{Float64}}()
    for n in (500,250),id in METHODS
        p=joinpath(OUTPUT,"checkpoints/final_$(n)_$(rep)_$id.jls")
        if isfile(p);v=deserialize(p)
        elseif n==500
            f=only(filter(r->r.method==id,archived.finals));z=zs(runs[(n,rep,id)],id)[end]
            @assert [f.z3,f.z4]==z&&f.selection_seed==ss&&f.holdout_seed==hs
            if id==SOFT.id
                # Needed only to calculate paired n250-minus-n500 MC SE.
                net=[R.evaluate_policy(policies[f.selected_policy],p,cfg)-R.cost(cfg,z) for p in R.materialize_bank(hold,z,cfg)]
            else
                donor=id=="Exp50" ? joinpath(SOURCE,"checkpoints/final_$(rep)_$id.jls") : joinpath(P.BANKS,"checkpoints/final_$(rep)_$id.jls")
                net=deserialize(donor).rewards
            end
            @assert isapprox(mean(net),f.objective;atol=1e-10)&&isapprox(std(net)/sqrt(50000),f.evaluation_se;atol=1e-12)
            row=endpoint_row(n,rep,id,z,f.selected_policy,f.objective,f.evaluation_se,cfg,"017 archived scalar evaluation; original holdout payoffs reused/reconstructed for pairing")
            v=(row=row,rewards=net);E.atomic_serialize(p,v)
        else
            z=zs(runs[(n,rep,id)],id)[end];cost=R.cost(cfg,z)
            Y=R.rewards(policies,R.materialize_bank(selector,z,cfg),cfg);chosen=argmax(vec(mean(Y;dims=2)))
            raw=[R.evaluate_policy(policies[chosen],p,cfg) for p in R.materialize_bank(hold,z,cfg)]
            row=endpoint_row(n,rep,id,z,chosen,mean(raw)-cost,std(raw)/sqrt(50000),cfg,"018 new independent direct selector/holdout")
            v=(row=row,rewards=raw.-cost);E.atomic_serialize(p,v)
        end
        push!(finals,v.row);vectors[(n,id)]=v.rewards
    end
    for n in (500,250),id in METHODS[2:end]
        dv=vectors[(n,id)]-vectors[(n,"Green")];mu=mean(dv);se=std(dv)/sqrt(50000)
        if n==500
            old=only(filter(r->r.method==id,archived.paired))
            @assert isapprox(mu,old.difference;atol=1e-10)&&isapprox(se,old.paired_mc_se;atol=1e-12)
        end
        push!(pairs,(n_paths=n,rep=rep,method=id,reference="Green",difference=mu,paired_mc_se=se,ci95_low=mu-Z95*se,ci95_high=mu+Z95*se))
    end
    for id in METHODS
        dv=vectors[(250,id)]-vectors[(500,id)];mu=mean(dv);se=std(dv)/sqrt(50000)
        push!(cross,(rep=rep,method=id,difference_n250_minus_n500=mu,paired_mc_se=se,ci95_low=mu-Z95*se,ci95_high=mu+Z95*se,
            n250_search_paths=25000,n500_search_paths=50000,holdout_paths=50000))
    end
    v=(traces=traces,finals=finals,paired=pairs,cross=cross);E.atomic_serialize(path,v)
    println("Independent evaluations and paired sample-size comparisons complete: rep$rep");flush(stdout);v
end

function parallel_reps(f)
    results=Vector{Any}(undef,3)
    Threads.@threads :dynamic for rep in 1:3;results[rep]=f(rep);end
    results
end
function main(args=ARGS)
    all(a in ("--resume","--no-plots","--plots-only","--tests-only","--preflight-only") for a in args)||error("Unknown argument")
    BLAS.set_num_threads(1);d,cfg=initialize()
    "--preflight-only" in args&&return
    if !("--plots-only" in args)
        if !isfile(joinpath(OUTPUT,"validation/tests.toml"));tests(d,cfg)
        elseif any(!isfile(joinpath(OUTPUT,"checkpoints/search_250_$(rep)_$id.jls")) for rep in REPS for id in METHODS)
            # Only n250 warmup prefixes, no n500 optimizer execution.
            for id in METHODS;run250(d,cfg,1,id;K=2);end
        end
        "--tests-only" in args&&return
        runs=searches(d,cfg)
        # All search clocks have finished before external diagnostics/evaluations.
        diags=parallel_reps(rep->diagnostics_rep(cfg,rep,runs))
        for (name,field) in (("iteration_diagnostics",:rows),("bank_diagnostics",:banks),("diagnostic_workload",:ledger))
            E.write_csv(joinpath(OUTPUT,"tables/$name.csv"),vcat([getproperty(v,field) for v in diags]...))
        end
        evals=parallel_reps(rep->evaluate_rep(cfg,rep,runs))
        for (name,field) in (("objective_traces",:traces),("final_results",:finals),("paired_vs_green",:paired),("paired_n250_vs_n500",:cross))
            E.write_csv(joinpath(OUTPUT,"tables/$name.csv"),vcat([getproperty(v,field) for v in evals]...))
        end
        open(joinpath(OUTPUT,"validation/science_complete.toml"),"w") do io
            TOML.print(io,Dict("complete"=>true,"new_optimizations_n250"=>15,"n500_optimizations_reexecuted"=>0,
                "rep_ids"=>[1,2,3],"iterations"=>100,"new_production_search_paths"=>375000,
                "archived_logical_search_paths"=>750000,"surface_recomputed"=>false,
                "softmax_diagnostic_reconstruction_paths_n500"=>150000,"softmax_diagnostic_reconstruction_paths_n250"=>75000))
        end
    end
    if !("--no-plots" in args);run(`python3 $(joinpath(ROOT,"analyze_green_sample_size_018.py"))`);end
end
end
if abspath(PROGRAM_FILE)==@__FILE__;ProbingGreenSampleSize.main();end
