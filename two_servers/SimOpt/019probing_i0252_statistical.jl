# Campaign orchestration only: all model, estimators and updates are reused.
include(joinpath(@__DIR__, "016probing_green_bank_variants.jl"))
module I0252Statistical
using ..ProbingGreenBankVariants
using Statistics, LinearAlgebra, Serialization, SHA, TOML, Dates, Printf, Test
const V = ProbingGreenBankVariants
const A = V.A; const H = V.H; const X = V.X; const R = V.R; const E = V.E; const C = V.C
const IDS = Dict("fresh"=>"Fresh", "green"=>"Green", "green_softmax"=>"GreenSoftmax050", "green_variance"=>"Filter2")
const REP_OFFSET = 19_000_000
const INSTANCE = V.ID
sha(p) = bytes2hex(sha256(read(p)))
utcnow() = string(Dates.now(Dates.UTC))*"Z"

function atomic_toml(path, data)
    tmp=path*".tmp"
    open(tmp,"w") do io
        TOML.print(io,data;sorted=true);flush(io)
        ccall(:fsync,Cint,(Cint,),Base.fd(io))==0||error("fsync failed")
    end
    mv(tmp,path;force=true)
end

function model(output)
    d=TOML.parsefile(joinpath(output,"config/experiment.toml"))
    instance=d["science"]["scenario"]
    @assert instance==INSTANCE==A.ID
    md=TOML.parsefile(joinpath(output,"config","$instance.toml"))["model"]
    cfg=H.from_dict(md)
    @assert R.model_dict(cfg)==md && R.model_digest(cfg)==d["science"]["model_sha256"]
    @assert C.Z0==d["science"]["Z0"]==[.12,.10]
    @assert C.BASE_STEP==d["science"]["BASE_STEP"]==.035 && C.GRAD_CLIP==15.
    @assert length(C.build_policy_grid())==144
    for (name,hash) in d["science"]["numerical_sources"]
        @assert sha(joinpath(output,"reproduction",name))==hash "Changed frozen source: $name"
    end
    d,cfg
end

function search(d,cfg,rep,method)
    # Existing timing wrappers retain the exact estimator and summation order.
    if method=="green_variance"
        m=only(filter(x->x.id=="Filter2",V.METHODS))
        raw=V.search(d,cfg,rep,m;extra=true,check_original=false)
        return (zhist=raw.zhist,core=raw.core,rows=raw.rows,
            cumulative=[0.;[r.cumulative_search_seconds for r in raw.rows]],
            seconds=raw.search_seconds,diagnostic_seconds=raw.diagnostic_seconds)
    end
    m=only(filter(x->x.id==IDS[method],X.METHODS))
    raw=A.timed_method(d,cfg,rep,m)
    (zhist=raw.run.zhist,core=raw.run.diagnostics,rows=raw.rows,
        cumulative=[r.cumulative_search_seconds for r in raw.rows],
        seconds=raw.rows[end].cumulative_search_seconds,diagnostic_seconds=0.)
end

function seeds(rep,K)
    n=REP_OFFSET+rep
    Dict("search"=>[X.seed(INSTANCE,"compare",n,"search",k) for k in 1:K],
        "trace"=>X.seed(INSTANCE,"compare",n,"trace"),
        "selection"=>X.seed(INSTANCE,"compare",n,"selection"),
        "holdout"=>X.seed(INSTANCE,"compare",n,"holdout"))
end

function direct_trace(cfg,zs,n,seed,jobs)
    # No LR, scores, optimizer state, policy selection feedback or step adaptation.
    bank=R.primitive_bank(cfg,n,seed;njobs=jobs);policies=C.build_policy_grid()
    cache=Dict{Tuple{Float64,Float64},Any}();rows=NamedTuple[]
    for z in zs
        p=get!(cache,Tuple(z)) do
            Y=R.rewards(policies,R.materialize_bank(bank,z,cfg),cfg)
            values=vec(mean(Y;dims=2));i=argmax(values)
            (objective=values[i]-R.cost(cfg,z),policy=i,se=std(view(Y,i,:))/sqrt(n))
        end
        push!(rows,p)
    end
    rows
end

function endpoint(cfg,z,d,ss)
    policies=C.build_policy_grid()
    selector=R.primitive_bank(cfg,d["selection_paths"],ss["selection"];njobs=d["jobs"])
    Y=R.rewards(policies,R.materialize_bank(selector,z,cfg),cfg)
    chosen=argmax(vec(mean(Y;dims=2)))
    holdout=R.primitive_bank(cfg,d["holdout_paths"],ss["holdout"];njobs=d["jobs"])
    raw=[R.evaluate_policy(policies[chosen],p,cfg) for p in R.materialize_bank(holdout,z,cfg)]
    p=policies[chosen]
    (objective=mean(raw)-R.cost(cfg,z),gross=mean(raw),se=std(raw)/sqrt(length(raw)),policy=chosen,
        av=p.av,aq=p.aq,as=p.as,tau=p.tau,selection_gross=maximum(vec(mean(Y;dims=2))))
end

function run_rep(output,c,cfg,method,rep)
    d=c["science"];K=d["max_iters"];fp=c["fingerprint"];ss=seeds(rep,K)
    @assert length(unique(vcat(ss["search"],[ss[k] for k in ("trace","selection","holdout")])))==K+3
    parent=joinpath(output,"raw",method);mkpath(parent)
    name=@sprintf("rep_%04d",rep);final=joinpath(parent,name);stage=joinpath(parent,".$name.partial")
    locks=joinpath(output,"raw/.locks");mkpath(locks)
    open(joinpath(locks,"$(method)_$name.lock"),"a") do lockio
        ccall(:flock,Cint,(Cint,Cint),Base.fd(lockio),6)==0||error("Replication already running")
        isdir(final)&&error("Published replication exists; runner must validate and skip it")
        mkpath(stage)
        state=joinpath(stage,"state.toml")
        started=utcnow()
        if isfile(state)
            old=TOML.parsefile(state)
            @assert old["fingerprint"]==fp && old["method"]==method && old["replication"]==rep
            started=old["started_utc"]
        end
        atomic_toml(state,Dict("fingerprint"=>fp,"method"=>method,"replication"=>rep,"started_utc"=>started,"status"=>"running"))
        wall=time_ns();searchfile=joinpath(stage,"search.jls")
        if isfile(searchfile)
            checkpoint=deserialize(searchfile);@assert checkpoint.fingerprint==fp
            raw=checkpoint.raw
        else
            raw=search(d,cfg,REP_OFFSET+rep,method)
            E.atomic_serialize(searchfile,(fingerprint=fp,raw=raw))
        end
        @assert length(raw.zhist)==K+1 && length(raw.core)==K
        @assert all(z->length(z)==2&&all(isfinite,z)&&all(0 .<= z .<= 1),raw.zhist)
        println("SEARCH\t$method\t$rep\t$(raw.seconds)");flush(stdout)
        E.write_csv(joinpath(stage,"iteration_diagnostics.csv"),raw.rows)
        E.write_csv(joinpath(stage,"search_seeds.csv"),[(iteration=k,seed=s) for (k,s) in enumerate(ss["search"])])
        tracefile=joinpath(stage,"convergence.jls")
        if isfile(tracefile)
            cp=deserialize(tracefile);@assert cp.fingerprint==fp;trace=cp.rows;trace_seconds=cp.seconds
        else
            t=time_ns();trace=direct_trace(cfg,raw.zhist,d["trace_paths"],ss["trace"],d["jobs"])
            trace_seconds=(time_ns()-t)/1e9
            E.atomic_serialize(tracefile,(fingerprint=fp,rows=trace,seconds=trace_seconds))
        end
        rows=NamedTuple[]
        for t in 0:K
            z=raw.zhist[t+1];p=trace[t+1];g=t<K ? raw.core[t+1] : nothing
            push!(rows,(replication=rep,method=method,iteration=t,z3=z[1],z4=z[2],objective_crn=p.objective,
                objective_internal=g===nothing ? missing : g.objective_internal,
                g3=g===nothing ? missing : g.g3,g4=g===nothing ? missing : g.g4,
                gradient_policy=g===nothing ? missing : g.hard_policy,
                ess=g===nothing ? missing : g.ess,lr_clips=g===nothing ? missing : g.lr_clips,
                K_eff=g===nothing ? missing : g.K_eff,cumulative_optimization_seconds=raw.cumulative[t+1],
                crn_selected_policy=p.policy,crn_selected_policy_se=p.se,evaluation_paths=d["trace_paths"],evaluation_seed=ss["trace"],
                cumulative_new_search_paths=t*d["n_paths"]))
        end
        E.write_csv(joinpath(stage,"trajectory.csv"),rows)
        finalfile=joinpath(stage,"final.jls")
        if isfile(finalfile)
            cp=deserialize(finalfile);@assert cp.fingerprint==fp;p=cp.result;final_seconds=cp.seconds
        else
            t=time_ns();p=endpoint(cfg,raw.zhist[end],d,ss);final_seconds=(time_ns()-t)/1e9
            E.atomic_serialize(finalfile,(fingerprint=fp,result=p,seconds=final_seconds))
        end
        z=raw.zhist[end]
        E.write_csv(joinpath(stage,"final.csv"),[(replication=rep,method=method,z3=z[1],z4=z[2],selected_policy=p.policy,
            policy_av=p.av,policy_aq=p.aq,policy_as=p.as,policy_tau=p.tau,objective=p.objective,gross=p.gross,
            probing_cost=R.cost(cfg,z),evaluation_se=p.se,mc_ci95_low=p.objective-1.959963984540054*p.se,
            mc_ci95_high=p.objective+1.959963984540054*p.se,selection_gross=p.selection_gross,
            optimization_seconds=raw.seconds,diagnostic_seconds=raw.diagnostic_seconds,
            convergence_evaluation_seconds=trace_seconds,final_evaluation_seconds=final_seconds,
            iterations=K,new_search_paths=K*d["n_paths"],selection_paths=d["selection_paths"],holdout_paths=d["holdout_paths"],
            selection_seed=ss["selection"],holdout_seed=ss["holdout"])])
        meta=Dict("status"=>"complete","fingerprint"=>fp,"replication"=>rep,"replication_id"=>name,"method"=>method,"scenario"=>INSTANCE,
            "original_method_id"=>IDS[method],"original_seed_replication"=>REP_OFFSET+rep,"seeds"=>ss,
            "started_utc"=>started,"completed_utc"=>utcnow(),"optimization_seconds"=>raw.seconds,
            "diagnostic_seconds"=>raw.diagnostic_seconds,"convergence_evaluation_seconds"=>trace_seconds,
            "final_evaluation_seconds"=>final_seconds,"last_attempt_seconds"=>(time_ns()-wall)/1e9,
            "julia_version"=>string(VERSION),"blas_threads"=>BLAS.get_num_threads(),"cpu"=>Sys.CPU_NAME,
            "science"=>d,"model"=>R.model_dict(cfg),"indexing"=>"t=0..T; internal/gradient at t update z[t] to z[t+1]; absent at T")
        atomic_toml(joinpath(stage,"metadata.toml"),meta)
        files=["trajectory.csv","final.csv","iteration_diagnostics.csv","search_seeds.csv","metadata.toml","search.jls","convergence.jls","final.jls"]
        for f in files
            open(joinpath(stage,f),"r") do io;ccall(:fsync,Cint,(Cint,),Base.fd(io))==0||error("fsync failed");end
        end
        atomic_toml(joinpath(stage,"complete.toml"),Dict("status"=>"complete","fingerprint"=>fp,"files"=>Dict(f=>sha(joinpath(stage,f)) for f in files)))
        mv(stage,final;force=false)
        for dir in (final,parent)
            fd=ccall(:open,Cint,(Cstring,Cint),dir,0)
            fd>=0||error("Cannot open directory");ccall(:fsync,Cint,(Cint,),fd);ccall(:close,Cint,(Cint,),fd)
        end
    end
end

function instance_audit(output,root)
    c,cfg=model(output);d=copy(c["science"]);d["max_iters"]=3;d["n_paths"]=500
    result=@testset "Instance binding, historical regression and unchanged Filter2" begin
        archive=deserialize(joinpath(root,"resultados/010probing_instance_landscape_search/checkpoints/search_$(INSTANCE)_1.jls"))
        for method in ("fresh","green","green_softmax")
            old=only(filter(r->r.method.id==IDS[method],archive.runs));new=search(d,cfg,1,method)
            @test isequal(new.zhist,old.zhist[1:4])
            @test isequal(new.core,old.diagnostics[1:3])
        end
        # Independently reconstruct the original 016 rejection rule on this model.
        history=R.HistoricalBank[];policies=C.build_policy_grid();z=[.35,.45]
        for k in 1:4
            source=[.12+.04k,.10+.06k]
            prim=R.primitive_bank(cfg,20,X.seed(INSTANCE,"compare",1,"search",k);njobs=45)
            bank=R.materialize_bank(prim,source,cfg);Y=R.rewards(policies,bank,cfg)
            push!(history,R.historical_bank(INSTANCE,cfg,source,bank,Y))
        end
        full=R.green_data(z,history,cfg,INSTANCE);variances=Float64[]
        for hb in history
            ll,_=R.bank_scores(hb.bank,z,hb.zgen,cfg)
            push!(variances,var(collect(view(hb.Y,full.best,:)).*exp.(clamp.(ll,-35.,35.));corrected=true))
        end
        indices=sort!(unique(vcat(findall(v->v<=2*variances[end],variances),[length(history)])))
        filtered=V.estimate(z,history,cfg,(id="Filter2",family=:filtered,param=2.))
        @test filtered.indices==indices
        @test filtered.weights==fill(1/length(indices),length(indices))
        @test filtered.provisional==full.best
        @test filtered.threshold==2*variances[end]
        @test isequal(filtered.data,R.green_data(z,history[indices],cfg,INSTANCE))
        @test length(history) in filtered.indices
        small=copy(d);small["n_paths"]=20;small["max_iters"]=4
        for method in keys(IDS)
            a=search(small,cfg,REP_OFFSET+1,method);original=deepcopy(a.zhist)
            direct_trace(cfg,a.zhist,20,123456,45)
            b=search(small,cfg,REP_OFFSET+1,method)
            @test isequal(a.zhist,b.zhist)&&isequal(a.core,b.core)&&isequal(a.zhist,original)
        end
        ss=seeds(1,100)
        @test length(unique(vcat(ss["search"],[ss[k] for k in ("trace","selection","holdout")])))==103
        @test ss["search"][1]==X.seed(INSTANCE,"compare",REP_OFFSET+1,"search",1)
        @test C.Z0==[.12,.10]
    end
    atomic_toml(joinpath(output,"validation/algorithm_audit.toml"),Dict("passed"=>true,"instance"=>INSTANCE,
        "assertions"=>Test.get_test_counts(result).passes,"archived_replication"=>1,"full_size_prefix_updates"=>3,
        "original_sources_modified"=>false,"filter2_rule_independently_checked"=>true,
        "direct_trace_no_lr"=>true,"evaluation_does_not_affect_search"=>true))
end

function audit(output,root)
    INSTANCE!="I0252"&&return instance_audit(output,root)
    c,cfg=model(output);d=copy(c["science"]);d["max_iters"]=3;d["n_paths"]=500
    result=@testset "Original algorithm regression and evaluation separation" begin
        archive=deserialize(joinpath(root,"resultados/011probing_high_contrast_interior/checkpoints/search_I0252_1.jls"))
        for method in ("fresh","green","green_softmax")
            old=only(filter(r->r.method.id==IDS[method],archive.runs));new=search(d,cfg,1,method)
            @test isequal(new.zhist,old.zhist[1:4])
            @test isequal(new.core,old.diagnostics[1:3])
        end
        old=deserialize(joinpath(root,"resultados/016probing_green_bank_variants/checkpoints/search_1_Filter2.jls"))
        new=search(d,cfg,1,"green_variance")
        @test isequal(new.zhist,old.zhist[1:4])
        @test isequal(new.core,old.core[1:3])
        # Reuse exact 016 diagnostics to check the actual retained-bank plan.
        @test [r.retained_bank_indices for r in new.rows]==[r.retained_bank_indices for r in old.rows[1:3]]
        @test [r.filter_threshold for r in new.rows]==[r.filter_threshold for r in old.rows[1:3]]
        @test R.softmax_weights([0.,2.];beta=.5)≈[1.,exp(1)]./(1+exp(1))
        small=copy(d);small["n_paths"]=20;small["max_iters"]=4
        for method in keys(IDS)
            a=search(small,cfg,REP_OFFSET+1,method)
            before=deepcopy(a.zhist)
            direct_trace(cfg,a.zhist,20,123456,45)
            direct_trace(cfg,a.zhist,20,987654,45)
            b=search(small,cfg,REP_OFFSET+1,method)
            @test isequal(a.zhist,b.zhist)&&isequal(a.core,b.core)&&isequal(a.zhist,before)
        end
        streams=[seeds(rep,100) for rep in 1:50]
        allseeds=vcat([vcat(s["search"],[s[k] for k in ("trace","selection","holdout")]) for s in streams]...)
        @test length(unique(allseeds))==50*103
        p=R.primitive_bank(cfg,20,12345;njobs=45);z=[.4,.5];policies=C.build_policy_grid()
        Y=R.rewards(policies,R.materialize_bank(p,z,cfg),cfg);i=argmax(vec(mean(Y;dims=2)))
        v=only(direct_trace(cfg,[z],20,12345,45))
        @test v.objective==mean(view(Y,i,:))-R.cost(cfg,z)
        @test v.policy==i
    end
    mkpath(joinpath(output,"validation"))
    atomic_toml(joinpath(output,"validation/algorithm_audit.toml"),Dict("passed"=>true,
        "assertions"=>Test.get_test_counts(result).passes,"archived_replication"=>1,"full_size_prefix_updates"=>3,
        "original_sources_modified"=>false,"direct_trace_no_lr"=>true,"evaluation_does_not_affect_search"=>true))
end

function main()
    BLAS.set_num_threads(1)
    output=abspath(ARGS[1]);c,cfg=model(output)
    if length(ARGS)>1&&ARGS[2]=="audit";audit(output,ARGS[3]);return;end
    # Compile before measured work. These small runs are never campaign records.
    warm=copy(c["science"]);warm["max_iters"]=2;warm["n_paths"]=20
    for m in ("fresh","green","green_softmax","green_variance");search(warm,cfg,REP_OFFSET,m);end
    direct_trace(cfg,[C.Z0],20,12345,45)
    endpoint(cfg,C.Z0,merge(warm,Dict("selection_paths"=>20,"holdout_paths"=>20)),seeds(0,2))
    println("READY");flush(stdout)
    for line in eachline(stdin)
        isempty(line)&&continue
        method,id=split(line,'\t');rep=parse(Int,id)
        try
            run_rep(output,c,cfg,String(method),rep)
            println("COMPLETE\t$method\t$rep");flush(stdout)
        catch err
            showerror(stdout,err,catch_backtrace());println();println("FAILED\t$method\t$rep");flush(stdout)
        end
        GC.gc()
    end
end
end
if abspath(PROGRAM_FILE)==@__FILE__;I0252Statistical.main();end
