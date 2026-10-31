module ProbingRegimeValidation
include(joinpath(@__DIR__,"probing_regime_experiment.jl"))
using .ProbingRegimeExperiment
using Random,Statistics,LinearAlgebra,Serialization,TOML,SHA,Printf,Dates
const X=ProbingRegimeExperiment
const R=X.R
const E=X.E
const C=X.C
const ROOT=@__DIR__
const SOURCE=joinpath(ROOT,"resultados/009probing_parameter_regimes")
const IDS=["S000","S001","S010","S101","S111"]
const METHODS=X.METHODS

function configured_scenarios(source=TOML.parsefile(joinpath(SOURCE,"configs/configuration.toml")))
    R.digest(source["science"])==source["fingerprint"]||error("009 configuration hash mismatch")
    for (file,sha) in source["science"]["source_sha256"]
        bytes2hex(sha256(read(joinpath(ROOT,file))))==sha||error("Changed original source $file")
    end
    [(id=id,A=parse(Int,id[2:2]),B=parse(Int,id[3:3]),C=parse(Int,id[4:4]),model=begin
        m=source["science"]["models"][id]
        cfg=R.validate(R.ModelConfig(p_complex=Tuple(m["P_COMPLEX"]),p_hit0=Tuple(m["P_HIT0"]),
            p_false0=Tuple(m["P_FALSE0"]),beta=m["BETA"],mean_interarrival=m["MEAN_INTERARRIVAL"],
            fast_mean=Tuple(m["FAST_MEAN"]),std_mean=Tuple(m["STD_MEAN"]),revenue_scale=m["REVENUE_SCALE"],
            fast_routing_cost=m["FAST_ROUTING_COST"],delay_penalty=m["DELAY_PENALTY"],alpha=Tuple(m["ALPHA"])))
        R.model_dict(cfg)==m||error("Model mapping differs from saved configuration")
        cfg
    end) for id in IDS]
end

function arguments(args)
    quick="--quick-test" in args
    d=Dict{String,Any}("quick"=>quick,"resume"=>false,"actions"=>String[],"scenarios"=>String[],
        "output"=>joinpath(ROOT,"resultados/009b_probing_regime_validation",quick ? "prueba_validada" : ""),
        "n_reps"=>quick ? 2 : 20,"n_paths"=>quick ? 20 : 500,"max_iters"=>quick ? 4 : 100,
        "jobs_per_path"=>45,"trace_paths"=>quick ? 30 : 1000,"selection_paths"=>quick ? 40 : 5000,
        "final_paths"=>quick ? 80 : 30000,"candidate_paths"=>quick ? 120 : 3000,
        "reference_sizes"=>quick ? [30,60,120] : [3000,10000,30000],
        "selection_reps"=>quick ? 5 : 100,"diagnostic_bootstrap"=>300,
        "bootstrap_reps"=>300,"beta_soft"=>.5,"phase"=>quick ? "009b-confirmation-quick" : "009b-confirmation",
        "diagnostic_phase"=>quick ? "009b-diagnostics-quick" : "009b-diagnostics")
    i=1
    while i<=length(args)
        a=args[i]
        if a=="--quick-test"
        elseif a=="--resume";d["resume"]=true
        elseif a in ("--diagnostics","--confirm","--plots","--report","--select-points")
            push!(d["actions"],a[3:end])
        elseif a in ("--output","--scenario","--n-reps","--n-paths","--max-iters","--trace-paths","--selection-paths","--final-paths","--candidate-paths","--reference-sizes","--selection-reps")
            i+=1;i<=length(args)||error("Missing value for $a");key=replace(a[3:end],'-'=>'_')
            if key=="output";d[key]=abspath(args[i])
            elseif key=="scenario";d["scenarios"]=split(args[i],',')
            elseif key=="reference_sizes";d[key]=parse.(Int,split(args[i],','))
            else;d[key]=parse(Int,args[i]);end
        elseif a=="--help"
            println("009b [--quick-test] [--resume] [--select-points] [--diagnostics] [--confirm] [--plots] [--report] [--scenario S000,S001] [--n-reps 20] [--reference-sizes 3000,10000,30000] [--output DIR]");return nothing
        else;error("Unknown option $a");end
        i+=1
    end
    isempty(d["actions"])&&(d["actions"]=["select-points","diagnostics","confirm","plots","report"])
    all(s->s in IDS,d["scenarios"])||error("Scenario outside prespecified five")
    for key in ("n_paths","trace_paths","selection_paths","final_paths","candidate_paths")
        d[key]>=2||error("$key must be >=2")
    end
    d["n_reps"]>=2&&d["max_iters"]>0&&d["selection_reps"]>0||error("Invalid budget")
    issorted(d["reference_sizes"])&&all(>=(3),d["reference_sizes"])||error("Reference sizes must be ascending >=3")
    return d
end

function initialize(d)
    out=d["output"]
    for f in ("configs","diagnostics","confirmation","figures","tables","logs","checkpoints","validation");mkpath(joinpath(out,f));end
    run(`python3 $(joinpath(ROOT,"prepare_009b.py")) $out`)
    old=TOML.parsefile(joinpath(SOURCE,"configs/configuration.toml"));d["models"]=configured_scenarios(old)
    files=[collect(keys(old["science"]["source_sha256"]));"009b_probing_regime_validation.jl";
        "probing_validation_experiment.jl";"probing_validation_diagnostics.jl";"prepare_009b.py";"PROTOCOLO_009b.txt"]
    science=Dict(k=>v for (k,v) in d if !(k in ("actions","resume","scenarios","output","n_reps","models")))
    science["source_009_fingerprint"]=old["fingerprint"]
    science["models"]=Dict(s.id=>R.model_dict(s.model) for s in d["models"])
    science["source_sha256"]=Dict(f=>bytes2hex(sha256(read(joinpath(ROOT,f)))) for f in files)
    science["points_sha256"]=bytes2hex(sha256(read(joinpath(out,"selected_points.csv"))))
    science["point_selection_sha256"]=bytes2hex(sha256(read(joinpath(out,"configs/point_selection.json"))))
    science["julia_version"]=string(VERSION);science["blas_threads"]=1
    d["science"]=science;d["fingerprint"]=R.digest(science)
    path=joinpath(out,"configs/configuration.toml")
    if isfile(path)
        TOML.parsefile(path)["fingerprint"]==d["fingerprint"]||error("Configuration/source mismatch: choose another --output")
        any(a->a in ("diagnostics","confirm"),d["actions"])&&!d["resume"]&&error("Use --resume")
    else
        open(path,"w") do io;TOML.print(io,Dict("fingerprint"=>d["fingerprint"],"science"=>science,"threads"=>Threads.nthreads());sorted=true);end
    end
    open(joinpath(out,"logs/invocations.log"),"a") do io
        println(io,Dates.now()," ARGS=",repr(ARGS)," threads=",Threads.nthreads()," fingerprint=",d["fingerprint"])
    end
    return d
end
verify(d)=all(bytes2hex(sha256(read(joinpath(ROOT,f))))==v for (f,v) in d["science"]["source_sha256"])||error("Scientific source changed")
targets(d)=isempty(d["scenarios"]) ? d["models"] : filter(s->s.id in d["scenarios"],d["models"])
diagseed(d,s,p,stream,k=0)=X.seed(s,d["diagnostic_phase"],0,"$p/$stream",k)

include(joinpath(@__DIR__,"probing_validation_diagnostics.jl"))

function evaluate_rep(d,s,rep,search,stem,policies)
    phase=d["phase"];cfg=s.model;cachepath=stem*"_trace_cache.jls"
    cache=if isfile(cachepath)
        x=deserialize(cachepath);x.fingerprint==d["fingerprint"]||error("Invalid trace cache");x.values
    else;Dict{Tuple{Float64,Float64},Any}();end
    tracebank=nothing;selector=nothing;holdout=nothing
    finals=NamedTuple[];traces=NamedTuple[];stability=NamedTuple[];evaluations=Any[]
    for (m,run) in enumerate(search.runs)
        path=stem*"_M$(m)_evaluation.jls"
        ev=if isfile(path)
            x=deserialize(path);x.fingerprint==d["fingerprint"]||error("Invalid method evaluation");x
        else
            tracebank===nothing&&(tracebank=R.primitive_bank(cfg,d["trace_paths"],X.seed(s.id,phase,rep,"trace");njobs=d["jobs_per_path"]))
            rows=NamedTuple[];started=time_ns()
            for k in 0:d["max_iters"]
                z=run.zhist[k+1];key=Tuple(z)
                v=get!(cache,key) do
                    Y=R.rewards(policies,R.materialize_bank(tracebank,z,cfg),cfg);mu=vec(mean(Y;dims=2));best=argmax(mu)
                    (objective=mu[best]-R.cost(cfg,z),mc_se=std(view(Y,best,:))/sqrt(d["trace_paths"]),policy=best)
                end
                push!(rows,(scenario=s.id,phase=phase,rep=rep,method=run.method.name,iteration=k,z3=z[1],z4=z[2],
                    objective=v.objective,mc_se=v.mc_se,policy=v.policy,search_paths=k*d["n_paths"]))
                k%10==0&&E.atomic_serialize(cachepath,(fingerprint=d["fingerprint"],values=cache))
            end
            E.atomic_serialize(cachepath,(fingerprint=d["fingerprint"],values=cache))
            trace_seconds=(time_ns()-started)/1e9;started=time_ns()
            selector===nothing&&(selector=R.primitive_bank(cfg,d["selection_paths"],X.seed(s.id,phase,rep,"final_selection");njobs=d["jobs_per_path"]))
            holdout===nothing&&(holdout=R.primitive_bank(cfg,d["final_paths"],X.seed(s.id,phase,rep,"final_holdout");njobs=d["jobs_per_path"]))
            z=run.zhist[end];Ys=R.rewards(policies,R.materialize_bank(selector,z,cfg),cfg)
            selected=argmax(vec(mean(Ys;dims=2)))
            Ye=R.rewards(policies,R.materialize_bank(holdout,z,cfg),cfg);mu=vec(mean(Ye;dims=2));best=argmax(mu)
            cost=R.cost(cfg,z);v=collect(view(Ye,selected,:)).-cost;dv=collect(view(Ye,best,:)).-cost
            final_seconds=(time_ns()-started)/1e9
            final=(scenario=s.id,A=s.A,B=s.B,C=s.C,phase=phase,rep=rep,method=run.method.name,z3=z[1],z4=z[2],
                selection_policy=selected,direct_policy=best,objective_holdout=mean(v),objective_direct=mu[best]-cost,
                gross_benefit=mu[selected],probing_cost=cost,evaluation_se=std(v)/sqrt(length(v)),
                direct_selected_se=std(dv)/sqrt(length(dv)),search_seconds=run.search_seconds,bootstrap_seconds=run.bootstrap_seconds,
                trace_evaluation_seconds=trace_seconds,final_evaluation_seconds=final_seconds,search_paths=d["n_paths"]*d["max_iters"])
            tail=rows[max(2,length(rows)-19):end];contact(z)=any(x->x<=1e-12||x>=1-1e-12,z)
            stab=(scenario=s.id,rep=rep,method=run.method.name,best_visited_objective=maximum(r.objective for r in rows),
                tail_mean_objective=mean(r.objective for r in tail),tail_temporal_sd=std([r.objective for r in tail]),
                tail_rms_mc_se=sqrt(mean(r.mc_se^2 for r in tail)),tail_iterations=length(tail),
                total_variation=sum(norm(run.zhist[k+1]-run.zhist[k]) for k in 1:d["max_iters"]),
                boundary_frequency=mean(contact(z) for z in run.zhist[2:end]),final_objective_trace=rows[end].objective,
                final_objective_holdout=final.objective_holdout,z3=z[1],z4=z[2],search_seconds=run.search_seconds,
                search_paths=d["n_paths"]*d["max_iters"])
            x=(fingerprint=d["fingerprint"],traces=rows,final=final,stability=stab,holdout=v,direct=dv)
            E.atomic_serialize(path,x);x
        end
        append!(traces,ev.traces);push!(finals,ev.final);push!(stability,ev.stability);push!(evaluations,ev)
    end
    paired=NamedTuple[]
    for (i,j) in ((2,1),(3,1),(5,4),(4,1)),estimand in ("holdout","direct")
        delta=getproperty(evaluations[i],Symbol(estimand))-getproperty(evaluations[j],Symbol(estimand))
        push!(paired,(scenario=s.id,phase=phase,rep=rep,estimand=estimand,method_a=METHODS[i].name,method_b=METHODS[j].name,
            difference=mean(delta),evaluation_paired_se=std(delta)/sqrt(length(delta))))
    end
    return (traces=traces,finals=finals,stability=stability,paired=paired)
end

function confirmation(d)
    out=d["output"];policies=C.build_policy_grid();chosen=targets(d)
    tasks=[(s,r) for s in chosen for r in 1:d["n_reps"]]
    Threads.@threads :dynamic for (s,rep) in tasks
        dir=joinpath(out,"confirmation",s.id);mkpath(dir)
        stem=joinpath(out,"checkpoints",@sprintf("confirm_%s_rep_%02d",s.id,rep));complete=stem*"_complete.jls"
        if isfile(complete)
            deserialize(complete).fingerprint==d["fingerprint"]||error("Invalid completed repetition")
            println("REUSE 009b $(s.id) rep=$rep");continue
        end
        sp=stem*"_search.jls"
        search=if isfile(sp)
            x=deserialize(sp);x.fingerprint==d["fingerprint"]||error("Invalid search");x
        else
            x=X.search(d,s,d["phase"],rep,policies)
            y=merge(x,(fingerprint=d["fingerprint"],));E.atomic_serialize(sp,y);y
        end
        for (m,run) in enumerate(search.runs)
            path=stem*"_M$(m)_search.jls"
            isfile(path)||E.atomic_serialize(path,(fingerprint=d["fingerprint"],run=run))
        end
        ev=evaluate_rep(d,s,rep,search,stem,policies);prefix=joinpath(dir,@sprintf("rep_%02d",rep))
        for field in (:traces,:finals,:stability,:paired);E.write_csv(prefix*"_$(field).csv",getproperty(ev,field));end
        diagnostics=NamedTuple[];coordinates=NamedTuple[]
        for run in search.runs
            append!(diagnostics,[merge((scenario=s.id,phase=d["phase"],rep=rep,method=run.method.name),x) for x in run.diagnostics])
            append!(coordinates,[(scenario=s.id,phase=d["phase"],rep=rep,method=run.method.name,iteration=k-1,z3=z[1],z4=z[2]) for (k,z) in enumerate(run.zhist)])
        end
        E.write_csv(prefix*"_diagnostics.csv",diagnostics);E.write_csv(prefix*"_coordinates.csv",coordinates)
        seeds=vcat(search.seeds,[(scenario=s.id,phase=d["phase"],rep=rep,stream=stream,iteration=0,seed=X.seed(s.id,d["phase"],rep,stream)) for stream in ("trace","final_selection","final_holdout")])
        E.write_csv(prefix*"_seeds.csv",seeds)
        E.atomic_serialize(complete,(fingerprint=d["fingerprint"],scenario=s.id,rep=rep))
        println("COMPLETE 009b $(s.id) rep=$rep");flush(stdout)
    end
    open(joinpath(out,"configs/confirmation_plan.toml"),"w") do io
        TOML.print(io,Dict("scenarios"=>[s.id for s in chosen],"n_reps"=>d["n_reps"],"fingerprint"=>d["fingerprint"],"complete"=>true);sorted=true)
    end
end

function main(args=ARGS)
    BLAS.set_num_threads(1);d=arguments(args);d===nothing&&return;initialize(d)
    "diagnostics" in d["actions"]&&diagnostics(d)
    "confirm" in d["actions"]&&confirmation(d)
    verify(d)
    if "plots" in d["actions"]||"report" in d["actions"]
        run(`python3 $(joinpath(ROOT,"analyze_009b.py")) $(d["output"])`)
    end
    println("009b actions complete: ",d["actions"]);return d
end
end
