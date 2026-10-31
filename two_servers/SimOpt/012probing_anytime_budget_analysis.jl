# New experiment; all archived model, gradient and search sources stay untouched.
if !isdefined(@__MODULE__,:ProbingHighContrastInterior)
    include(joinpath(@__DIR__,"011probing_high_contrast_interior.jl"))
end
module ProbingAnytimeBudget
using ..ProbingHighContrastInterior
using Random, Statistics, LinearAlgebra, Serialization, SHA, TOML, Dates, Printf
const H=ProbingHighContrastInterior
const X=H.X
const R=H.R
const E=H.E
const C=H.C
const ROOT=@__DIR__
const ID="I0252"
const EPS=[.5,1.,2.]
const ZREF=[.70,.70]
cohort(rep)=rep<=5 ? "original_011" : "new_012"
sha(p)=bytes2hex(sha256(read(p)))
seed(rep,stream)=Int(parse(UInt64,bytes2hex(sha256("Probing012-v1/$ID/$rep/$stream"))[1:15];base=16))+1
function arguments(args)
    d=Dict{String,Any}("output"=>joinpath(ROOT,"resultados/012probing_anytime_budget"),
        "source"=>joinpath(ROOT,"resultados/011probing_high_contrast_interior"),"quick"=>false,"resume"=>false,
        "reuse"=>true,"recompute"=>false,"plots_only"=>false,"extra_reps"=>0,
        "validation_paths"=>5000,"trace_paths"=>1000,"n_paths"=>500,"jobs"=>45,"max_iters"=>100,"bootstrap_reps"=>300)
    i=1
    while i<=length(args)
        a=args[i]
        if a=="--quick-test";d["quick"]=true
        elseif a=="--resume";d["resume"]=true
        elseif a=="--reuse-011";d["reuse"]=true
        elseif a=="--recompute-evaluations";d["recompute"]=true
        elseif a=="--plots-only";d["plots_only"]=true
        elseif a=="--help"
            println("012 [--quick-test] [--reuse-011] [--recompute-evaluations] [--extra-reps N] [--plots-only] [--resume] [--output DIR] [--source DIR] [--validation-paths 5000] [--trace-paths 1000]");return nothing
        elseif a in ("--extra-reps","--output","--source","--validation-paths","--trace-paths")
            i+=1;i<=length(args)||error("Missing argument for $a");k=replace(a[3:end],'-'=>'_')
            d[k]=k in ("output","source") ? abspath(args[i]) : parse(Int,args[i])
        else;error("Unknown option $a");end
        i+=1
    end
    if d["quick"]
        d["output"]==joinpath(ROOT,"resultados/012probing_anytime_budget")&&(d["output"]=joinpath(d["output"],"quick_test"))
        d["max_iters"]=6;d["validation_paths"]=min(d["validation_paths"],100)
    end
    d["extra_reps"]>=0&&d["validation_paths"]>=2&&d["trace_paths"]>=2||error("Invalid sample sizes")
    d["recompute"]||d["trace_paths"]==1000||error("Use --recompute-evaluations for a different trace bank size")
    d["rep_ids"]=[collect(1:(d["quick"] ? 2 : 5));collect(6:5+d["extra_reps"])]
    d
end
function initialize(d)
    out=d["output"];src=d["source"]
    for p in ("configs","source_011","tables","validation","logs","reproduction","checkpoints","figures");mkpath(joinpath(out,p));end
    original=TOML.parsefile(joinpath(src,"configs/experiment.toml"))["science"]
    @assert original["n_paths"]==500&&original["jobs"]==45&&original["max_iters"]==100&&original["reps"]==5
    @assert original["BASE_STEP"]==C.BASE_STEP==.035&&original["GRAD_CLIP"]==C.GRAD_CLIP==15.
    @assert original["Z0"]==C.Z0==[.12,.10]
    cfgdict=TOML.parsefile(joinpath(src,"configs/models.toml"))[ID]
    @assert cfgdict==TOML.parsefile(joinpath(src,"configs/main_instance_parameters.toml"))["model"]
    cfg=H.from_dict(cfgdict);@assert R.model_dict(cfg)==cfgdict
    @assert length(C.build_policy_grid())==144
    for (f,h) in original["sources"];@assert sha(joinpath(ROOT,f))==h "Original scientific source changed: $f";end
    science=Dict("quick"=>d["quick"],"n_paths"=>500,"jobs"=>45,"max_iters"=>d["max_iters"],
        "trace_paths"=>d["trace_paths"],"validation_paths"=>d["validation_paths"],"bootstrap_reps"=>300,
        "recompute_evaluations"=>d["recompute"],"epsilons"=>EPS,"sustained_window"=>5,"z_ref"=>ZREF,
        "reference_model_sha256"=>R.model_digest(cfg),"source_experiment_sha256"=>sha(joinpath(src,"configs/experiment.toml")),
        "sources"=>Dict(f=>sha(joinpath(ROOT,f)) for f in ["012probing_anytime_budget_analysis.jl","probing_anytime_012_runtime.jl","PROTOCOLO_012.txt"]))
    fingerprint=R.digest(science);d["fingerprint"]=fingerprint
    path=joinpath(out,"configs/experiment.toml")
    if isfile(path)
        old=TOML.parsefile(path);@assert old["fingerprint"]==fingerprint "Different protocol/source/budget: use a separate --output"
        d["resume"]||d["plots_only"]||error("Existing results: use --resume")
        d["extra_reps"]=max(d["extra_reps"],old["extra_reps"])
        d["rep_ids"]=[collect(1:(d["quick"] ? 2 : 5));collect(6:5+d["extra_reps"])]
    end
    open(path,"w") do io;TOML.print(io,Dict("science"=>science,"fingerprint"=>fingerprint,"extra_reps"=>d["extra_reps"],"rep_ids"=>d["rep_ids"]);sorted=true);end
    cp(joinpath(src,"configs/main_instance_parameters.toml"),joinpath(out,"configs/I0252.toml");force=true)
    for f in ["012probing_anytime_budget_analysis.jl","probing_anytime_012_runtime.jl","PROTOCOLO_012.txt"]
        cp(joinpath(ROOT,f),joinpath(out,"reproduction",f);force=true)
    end
    for f in ("Project.toml","Manifest.toml")
        cp(joinpath(src,"reproduction",f),joinpath(out,"reproduction",f);force=true)
    end
    cp(joinpath(src,"high/I0252_surface.csv"),joinpath(out,"source_011/I0252_surface.csv");force=true)
    inputhash=Dict{String,String}();archives=Dict{Int,Any}()
    for rep in 1:(d["quick"] ? 2 : 5)
        path=joinpath(src,"checkpoints/search_$(ID)_$rep.jls");archives[rep]=deserialize(path);inputhash[relpath(path,src)]=sha(path)
        evaluation=joinpath(src,"checkpoints/evaluation_$(ID)_$rep.jls");inputhash[relpath(evaluation,src)]=sha(evaluation)
        for suffix in ("coordinates","traces","diagnostics","finals","paired","grid_gaps","seeds")
            name=@sprintf("rep_%02d_%s.csv",rep,suffix);p=joinpath(src,"comparisons",ID,name)
            inputhash[relpath(p,src)]=sha(p);cp(p,joinpath(out,"source_011",name);force=true)
        end
    end
    open(joinpath(out,"configs/input_sha256.toml"),"w") do io;TOML.print(io,inputhash;sorted=true);end
    audit=deserialize(joinpath(src,"checkpoints/geometry_audit_$(ID).jls"))
    a=only(filter(r->r.role=="anchor",audit.rows));@assert [a.z3,a.z4]==ZREF
    reference=Dict("z_ref"=>ZREF,"selected_before_metrics"=>true,"role"=>"anchor",
        "source"=>"011 independent geometry audit, not a certified optimum",
        "source_sha256"=>sha(joinpath(src,"audits/$(ID)_geometry.csv")),
        "high_precision"=>Dict(string(k)=>getproperty(a,k) for k in keys(a)))
    open(joinpath(out,"configs/reference.toml"),"w") do io;TOML.print(io,reference;sorted=true);end
    d["science"]=science
    cfg,archives
end
include(joinpath(@__DIR__,"probing_anytime_012_runtime.jl"))
function direct_point(d,cfg,rep,phase,z,bank)
    key=bytes2hex(sha256(repr(z)));path=joinpath(d["output"],"checkpoints/point_$(phase)_$(rep)_$key.jls")
    isfile(path)&&return deserialize(path)
    a=time_ns();Y=R.rewards(C.build_policy_grid(),R.materialize_bank(bank,z,cfg),cfg)
    means=vec(mean(Y;dims=2));idx=argmax(means);v=collect(view(Y,idx,:));cost=R.cost(cfg,z)
    p=(z=copy(z),objective=means[idx]-cost,cost=cost,policy=idx,rewards=v,se=std(v)/sqrt(length(v)),
        n_paths=length(bank),seconds=(time_ns()-a)/1e9)
    E.atomic_serialize(path,p);p
end
function evaluation_phase(d,cfg,runs)
    mainrows=NamedTuple[];refrows=NamedTuple[];ledger=NamedTuple[]
    for rep in d["rep_ids"]
        path=joinpath(d["output"],"checkpoints/main_evaluation_$rep.jls")
        if isfile(path);v=deserialize(path)
        else
            reuse=rep<=5&&!d["recompute"]
            ss=reuse ? X.seed(ID,"compare",rep,"trace") : seed(rep,"main_evaluation")
            bank=R.primitive_bank(cfg,d["trace_paths"],ss;njobs=d["jobs"])
            phase=d["recompute"] ? "main_new" : "main";reference=direct_point(d,cfg,rep,phase,ZREF,bank)
            rows=NamedTuple[]
            if reuse
                archived=deserialize(joinpath(d["source"],"checkpoints/evaluation_$(ID)_$rep.jls"))
                # Directly verify seed/budget and exact hard-max reconstruction.
                for (method,k) in (("Fresh",0),("Fresh",1),("Green",min(50,d["max_iters"])))
                    old=only(filter(r->r.method==method&&r.iteration==k,archived.traces))
                    @assert old.seed==ss&&old.trace_paths==d["trace_paths"]
                    p=direct_point(d,cfg,rep,phase,[old.z3,old.z4],bank)
                    @assert isequal(p.objective,old.objective) "Trace bank reconstruction failed"
                end
                for r in archived.traces
                    r.iteration>d["max_iters"]&&continue
                    push!(rows,(scenario=ID,rep=rep,cohort=cohort(rep),method=r.method,iteration=r.iteration,
                        z3=r.z3,z4=r.z4,objective=r.objective,reference_objective=reference.objective,
                        deficit=reference.objective-r.objective,n_paths=d["trace_paths"],evaluation_seed=ss,source="011 exact trace reuse"))
                end
            else
                for run in runs[rep].runs,(j,z) in enumerate(run.zhist)
                    p=direct_point(d,cfg,rep,phase,z,bank)
                    push!(rows,(scenario=ID,rep=rep,cohort=cohort(rep),method=run.method.id,iteration=j-1,z3=z[1],z4=z[2],
                        objective=p.objective,reference_objective=reference.objective,deficit=reference.objective-p.objective,
                        n_paths=d["trace_paths"],evaluation_seed=ss,source="012 new common direct evaluation"))
                end
            end
            ref=(rep=rep,cohort=cohort(rep),z3=ZREF[1],z4=ZREF[2],objective=reference.objective,
                conditional_mc_se=reference.se,policy=reference.policy,n_paths=length(bank),seed=ss,reused_011_bank=reuse)
            countpoints=length(filter(f->startswith(basename(f),"point_$(phase)_$(rep)_"),readdir(joinpath(d["output"],"checkpoints"))))
            led=(rep=rep,cohort=cohort(rep),phase="main_reference_and_reconstruction",primitive_paths_rebuilt=length(bank),
                new_unique_exogenous_paths=reuse ? 0 : length(bank),materialized_paths=countpoints*length(bank),
                policy_path_evaluations=144*countpoints*length(bank),reused_trace_values=reuse ? length(rows) : 0)
            v=(rows=rows,reference=ref,ledger=led);E.atomic_serialize(path,v)
        end
        append!(mainrows,v.rows);push!(refrows,v.reference);push!(ledger,v.ledger)
        println("Comparable direct reference/traces complete: rep=$rep");flush(stdout)
    end
    E.write_csv(joinpath(d["output"],"tables/objectives_by_iteration.csv"),mainrows)
    E.write_csv(joinpath(d["output"],"tables/reference_by_rep.csv"),refrows)
    mainrows,ledger
end
function validation_plan(d,rows)
    plan=NamedTuple[]
    for rep in d["rep_ids"],m in X.METHODS
        q=sort(filter(r->r.rep==rep&&r.method==m.id,rows);by=r->r.iteration)
        @assert length(q)==d["max_iters"]+1
        initial=first(q);relative=.05*(initial.reference_objective-initial.objective)
        tolerances=[(name=string(e),value=e) for e in EPS]
        relative>0&&push!(tolerances,(name="relative_5pct",value=relative))
        for e in tolerances,kind in ("first","sustained5")
            good=[r.deficit<=e.value for r in q]
            pos=kind=="first" ? findfirst(good) : findfirst(i->all(good[i:i+4]),1:(length(good)-4))
            start=pos===nothing ? missing : pos-1
            confirm=ismissing(start) ? missing : start+(kind=="first" ? 0 : 4)
            times=ismissing(start) ? Int[] : kind=="first" ? unique([max(start-1,0),start]) : collect(start:confirm)
            push!(plan,(rep=rep,cohort=cohort(rep),method=m.id,epsilon_name=e.name,epsilon=e.value,kind=kind,
                achieved=pos!==nothing,start_iteration=start,confirmation_iteration=confirm,validation_iterations=join(times,";")))
        end
    end
    path=joinpath(d["output"],"configs/validation_plan.jls")
    if isfile(path);@assert isequal(deserialize(path),plan)
    else;E.atomic_serialize(path,plan);end
    E.write_csv(joinpath(d["output"],"configs/validation_plan.csv"),plan);plan
end
function validate_crossings(d,cfg,searches,plan)
    results=Vector{Any}(undef,length(d["rep_ids"]))
    # AFTER all sequential timing measurements: these workers never contaminate them.
    Threads.@threads :dynamic for j in eachindex(d["rep_ids"])
        local rep=d["rep_ids"][j];path=joinpath(d["output"],"checkpoints/crossing_validation_$rep.jls")
        if isfile(path);results[j]=deserialize(path);continue;end
        ss=seed(rep,"threshold_validation");bank=R.primitive_bank(cfg,d["validation_paths"],ss;njobs=d["jobs"])
        reference=direct_point(d,cfg,rep,"validation",ZREF,bank);local rows=NamedTuple[]
        for p in filter(p->p.rep==rep&&p.achieved,plan)
            run=only(filter(r->r.method.id==p.method,searches[rep].runs))
            for t in parse.(Int,split(p.validation_iterations,';'))
                point=direct_point(d,cfg,rep,"validation",run.zhist[t+1],bank)
                delta=reference.rewards-point.rewards .-(reference.cost-point.cost)
                push!(rows,(rep=rep,cohort=cohort(rep),method=p.method,kind=p.kind,epsilon_name=p.epsilon_name,epsilon=p.epsilon,
                    start_iteration=p.start_iteration,confirmation_iteration=p.confirmation_iteration,iteration=t,
                    role=t<p.start_iteration ? "preceding" : "candidate",objective=point.objective,
                    reference_objective=reference.objective,deficit=mean(delta),paired_mc_se=std(delta)/sqrt(length(delta)),
                    meets_threshold=mean(delta)<=p.epsilon,reference_policy=reference.policy,policy=point.policy,
                    n_paths=length(bank),evaluation_seed=ss))
            end
        end
        countpoints=length(filter(f->startswith(basename(f),"point_validation_$(rep)_"),readdir(joinpath(d["output"],"checkpoints"))))
        led=(rep=rep,cohort=cohort(rep),phase="independent_crossing_validation",primitive_paths_rebuilt=length(bank),
            new_unique_exogenous_paths=length(bank),materialized_paths=countpoints*length(bank),
            policy_path_evaluations=144*countpoints*length(bank),reused_trace_values=0)
        results[j]=(rows=rows,ledger=led);E.atomic_serialize(path,results[j])
        println("Independent crossing/window validation complete: rep=$rep unique_points=$countpoints");flush(stdout)
    end
    for (j,v) in enumerate(results)
        rep=d["rep_ids"][j]
        @assert all(r.rep==rep for r in v.rows) "Mixed validation repetitions"
        expected=sum(length(split(p.validation_iterations,';')) for p in plan if p.rep==rep&&p.achieved)
        @assert length(v.rows)==expected "Missing/duplicated validation rows"
    end
    combined=vcat([v.rows for v in results]...)
    keys=[(r.rep,r.method,r.kind,r.epsilon_name,r.iteration) for r in combined]
    @assert length(unique(keys))==length(keys) "Duplicate validation point records"
    E.write_csv(joinpath(d["output"],"tables/crossing_validation.csv"),combined)
    [v.ledger for v in results]
end
function driver_audit(d,archives)
    rows=NamedTuple[];n=d["n_paths"]
    for rep in sort(collect(keys(archives))),k in 1:d["max_iters"]
        seen=Set{Tuple{Float64,Float64}}()
        for run in archives[rep].runs
            fixed=run.method.kind==:fixed;z=Tuple(run.zhist[k]);miss=!fixed&&!(z in seen)
            !fixed&&push!(seen,z)
            push!(rows,(rep=rep,method=run.method.id,iteration=k,
                common_new_primitive_paths=run.method.id=="Fresh" ? n : 0,
                fixed_setup_paths=fixed&&k==1 ? n : 0,
                new_materialized_paths=fixed ? (k==1 ? n : 0) : miss ? n : 0,
                new_policy_path_evaluations=(fixed ? (k==1 ? n : 0) : miss ? n : 0)*144,
                inter_method_payload_cache_hit=!fixed&&!miss))
        end
    end
    E.write_csv(joinpath(d["output"],"tables/joint_011_driver_accounting.csv"),rows)
end
function main(args=ARGS)
    BLAS.set_num_threads(1);d=arguments(args);d===nothing&&return
    cfg,archives=initialize(d)
    if !d["plots_only"]
        searches,budget=timing_phase(d,cfg,archives);driver_audit(d,archives)
        rows,ledger=evaluation_phase(d,cfg,searches);plan=validation_plan(d,rows)
        append!(ledger,validate_crossings(d,cfg,searches,plan))
        E.write_csv(joinpath(d["output"],"tables/external_evaluation_budget.csv"),ledger)
        open(joinpath(d["output"],"COMPUTATION_COMPLETE.toml"),"w") do io
            TOML.print(io,Dict("complete"=>true,"quick"=>d["quick"],"fingerprint"=>d["fingerprint"],"rep_ids"=>d["rep_ids"],
                "archived_searches_reused"=>7*(d["quick"] ? 2 : 5),"timing_replays"=>7*length(d["rep_ids"])))
        end
    end
    isfile(joinpath(d["output"],"COMPUTATION_COMPLETE.toml"))||error("Complete computations first")
    run(Cmd(["python3",joinpath(ROOT,"analyze_anytime_012.py"),d["output"]]))
    for (f,h) in d["science"]["sources"];@assert sha(joinpath(ROOT,f))==h "Scientific source changed during execution";end
    println("012 completed output=",d["output"])
end
end
if abspath(PROGRAM_FILE)==@__FILE__;ProbingAnytimeBudget.main();end
