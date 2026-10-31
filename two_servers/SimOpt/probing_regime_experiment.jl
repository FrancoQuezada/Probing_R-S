module ProbingRegimeExperiment
include(joinpath(@__DIR__,"probing_regime_model.jl"))
using .ProbingRegimeModel
using Random, Statistics, LinearAlgebra, Serialization, SHA, TOML, Printf, Dates
const R=ProbingRegimeModel
const E=R.E
const C=R.C
const ROOT=@__DIR__
const METHODS=[(name="Fresh",kind=:fresh),(name="Fresh + Softmax beta=0.5",kind=:softmax),
    (name="Fresh + PSoft B=300",kind=:psoft),(name="Green",kind=:green),
    (name="Green + Softmax beta=0.5",kind=:green_softmax)]
const OLD_FILES=["002probing_three_searches_n500_full45iters.jl","003probing_four_searches_softmax.jl",
    "004probing_softmax_sensitivity.jl","005probing_softmax_multiseed.jl","006probing_softmax_sample_efficiency.jl",
    "007probing_green_softmax.jl","008probing_probability_softmax.jl","probing_experiments_common.jl",
    "SimOpt005.jl","SimOpt009.jl","SimOpt009_2D_clean.jl"]

function seed(scenario,phase,rep,stream,k=0)
    h=bytes2hex(sha256("Probing009-v1/$scenario/$phase/$rep/$stream/$k"))
    Int(parse(UInt64,h[1:15];base=16))+1
end
function parse_arguments(args)
    d=Dict{String,Any}("quick"=>false,"resume"=>false,"n_paths"=>500,"max_iters"=>100,
        "jobs_per_path"=>45,"reference_paths"=>3000,"selection_reps"=>100,
        "pilot_reps"=>5,"confirm_reps"=>20,"trace_paths"=>1000,"selection_paths"=>5000,
        "final_paths"=>30000,"gradient_bootstrap_reps"=>300,"oracle_grid_n"=>11,"oracle_paths"=>3000,
        "bootstrap_reps"=>300,"beta_soft"=>.5,"epsilon"=>.5,"equivalence_tolerance"=>1e-8,
        "gradient_zero_tolerance"=>1e-8,"stable_modal_min"=>.90,"stable_wilson_min"=>.80,
        "uncertain_modal_max"=>.80,"uncertain_wilson_max"=>.90,
        "relative_gradient_boundary"=>.20,"similar_cosine_min"=>.95,"different_cosine_max"=>.90,
        "output"=>joinpath(ROOT,"resultados/009probing_parameter_regimes"),
        "scenarios"=>String[],"all_scenarios"=>false,"actions"=>String[])
    supplied=Set{String}();nrep=nothing;i=1
    while i<=length(args)
        a=args[i]
        if a=="--quick-test";d["quick"]=true
        elseif a=="--resume";d["resume"]=true
        elseif a=="--all-scenarios";d["all_scenarios"]=true
        elseif a in ("--characterize","--pilot","--confirm","--plots","--report")
            push!(d["actions"],a[3:end])
        elseif a in ("--scenario","--n-reps","--n-paths","--max-iters","--reference-paths",
            "--selection-reps","--pilot-reps","--confirm-reps","--output","--trace-paths",
            "--selection-paths","--final-paths","--oracle-grid-n","--oracle-paths","--epsilon")
            i+=1;i<=length(args)||error("Missing value for $a")
            key=replace(a[3:end],'-'=>'_');push!(supplied,key)
            if key=="scenario";d["scenarios"]=split(args[i],',')
            elseif key=="output";d[key]=abspath(args[i])
            elseif key=="epsilon";d[key]=parse(Float64,args[i])
            elseif key=="n_reps";nrep=parse(Int,args[i])
            else;d[key]=parse(Int,args[i]);end
        elseif a=="--help"
            println("Usage: julia 009probing_parameter_regimes.jl [--quick-test] [--resume] [--characterize] [--pilot] [--confirm] [--plots] [--report] [--scenario S000,S111] [--all-scenarios] [--n-reps N] [--n-paths 500] [--max-iters 100] [--reference-paths 3000] [--selection-reps 100] [--oracle-grid-n 11] [--oracle-paths 3000] [--output DIR]")
            return nothing
        else;error("Unknown option $a");end
        i+=1
    end
    isempty(d["actions"])&&(d["actions"]=["characterize","pilot","confirm","plots","report"])
    if d["quick"]
        for (k,v) in ("n_paths"=>20,"max_iters"=>4,"reference_paths"=>40,"selection_reps"=>4,
            "pilot_reps"=>2,"confirm_reps"=>2,"trace_paths"=>30,"selection_paths"=>40,
            "final_paths"=>80,"gradient_bootstrap_reps"=>30,"oracle_grid_n"=>3,"oracle_paths"=>40)
            k in supplied||(d[k]=v)
        end
        "output" in supplied||(d["output"]=joinpath(d["output"],"prueba_validada"))
    end
    nrep!==nothing&&("pilot" in d["actions"])&&(d["pilot_reps"]=nrep)
    nrep!==nothing&&("confirm" in d["actions"])&&(d["confirm_reps"]=nrep)
    for k in ("n_paths","reference_paths","trace_paths","selection_paths","final_paths")
        d[k]>=2||error("$k must be at least two")
    end
    for k in ("max_iters","pilot_reps","confirm_reps","selection_reps","gradient_bootstrap_reps","oracle_grid_n","oracle_paths")
        d[k]>0||error("$k must be positive")
    end
    all(s->s in [sc.id for sc in R.scenarios()],d["scenarios"])||error("Invalid scenario")
    critical_code="from scipy.stats import t;import sys;n=int(sys.argv[1]);print(t.ppf(1-.05/(2*(144*143//2)),n-1));print(t.ppf(1-.05/(2*6),n-1))"
    critical=parse.(Float64,split(read(`python3 -c $critical_code $(string(d["reference_paths"]))`,String)))
    d["reference_pair_critical"]=critical[1];d["gradient_critical"]=critical[2]
    sourcefiles=[OLD_FILES;"Probing_R_S.pdf";"probing_regime_model.jl";"probing_regime_experiment.jl";
        "probing_regime_characterization.jl";"009probing_parameter_regimes.jl";
        "probing_regime_structure.py";"PROTOCOLO_009.txt"]
    d["source_sha256"]=Dict(f=>bytes2hex(sha256(read(joinpath(ROOT,f)))) for f in sourcefiles)
    operational=Set(["actions","scenarios","all_scenarios","resume","output","pilot_reps","confirm_reps"])
    science=Dict(k=>v for (k,v) in d if !(k in operational))
    science["models"]=Dict(s.id=>R.model_dict(s.model) for s in R.scenarios())
    science["julia_version"]=string(VERSION);science["blas_threads"]=1
    d["science"]=science;d["fingerprint"]=R.digest(science)
    return d
end

function initialize(d)
    out=d["output"]
    for folder in ("configs","characterization","pilot","confirmation","figures","tables","logs","checkpoints","validation")
        mkpath(joinpath(out,folder))
    end
    path=joinpath(out,"configs/configuration.toml")
    if isfile(path)
        previous=TOML.parsefile(path)
        previous["fingerprint"]==d["fingerprint"]||error("Configuration/source mismatch. Choose a new --output for changed scientific settings.")
        any(a->a in ("characterize","pilot","confirm"),d["actions"])&&!d["resume"]&&error("Existing experiment: use --resume")
    else
        open(path,"w") do io;TOML.print(io,Dict("science"=>d["science"],"fingerprint"=>d["fingerprint"],"threads"=>Threads.nthreads());sorted=true);end
        rows=NamedTuple[]
        for s in R.scenarios()
            c=s.model
            push!(rows,(scenario=s.id,A=s.A,B=s.B,C=s.C,P_COMPLEX=join(c.p_complex,';'),P_HIT0=join(c.p_hit0,';'),
                P_FALSE0=join(c.p_false0,';'),BETA=c.beta,MEAN_INTERARRIVAL=c.mean_interarrival,
                FAST_MEAN=join(c.fast_mean,';'),STD_MEAN=join(c.std_mean,';'),REVENUE_SCALE=c.revenue_scale,
                FAST_ROUTING_COST=c.fast_routing_cost,DELAY_PENALTY=c.delay_penalty,ALPHA=join(c.alpha,';'),
                JOBS_PER_PATH=d["jobs_per_path"],NCLASS=4,ACTIVE_CLASSES="3;4",policies=144,model_sha256=R.model_digest(c)))
        end
        E.write_csv(joinpath(out,"scenarios.csv"),rows)
        E.write_csv(joinpath(out,"configs/policies.csv"),[(policy=i,av=p.av,aq=p.aq,as=p.as,tau=p.tau) for (i,p) in enumerate(C.build_policy_grid())])
    end
    invocation=joinpath(out,"logs/invocations.log")
    open(invocation,"a") do io
        println(io,Dates.now()," threads=",Threads.nthreads()," actions=",d["actions"]," scenarios=",d["scenarios"],
            " all_scenarios=",d["all_scenarios"]," pilot_reps=",d["pilot_reps"]," confirm_reps=",d["confirm_reps"]," ARGS=",repr(ARGS))
    end
end
verify_sources(d)=all(bytes2hex(sha256(read(joinpath(ROOT,f))))==v for (f,v) in d["source_sha256"])||error("Source changed during experiment")
scenario_targets(d)=isempty(d["scenarios"]) ? R.scenarios() : filter(s->s.id in d["scenarios"],R.scenarios())

include(joinpath(@__DIR__,"probing_regime_characterization.jl"))

function search(d,s,phase,rep,policies)
    cfg=s.model;n=d["n_paths"];K=d["max_iters"]
    states=[(method=m,zhist=[copy(C.Z0)],diagnostics=NamedTuple[],history=R.HistoricalBank[],
        search_seconds=Ref(0.),bootstrap_seconds=Ref(0.)) for m in METHODS]
    seedrows=NamedTuple[]
    for k in 1:K
        search_seed=seed(s.id,phase,rep,"search",k);bootstrap_seed=seed(s.id,phase,rep,"bootstrap",k)
        push!(seedrows,(scenario=s.id,phase=phase,rep=rep,stream="search",iteration=k,seed=search_seed))
        push!(seedrows,(scenario=s.id,phase=phase,rep=rep,stream="bootstrap",iteration=k,seed=bootstrap_seed))
        ts=time_ns();primitives=R.primitive_bank(cfg,n,search_seed;njobs=d["jobs_per_path"])
        primitive_seconds=(time_ns()-ts)/1e9
        payloads=Dict{Tuple{Float64,Float64},Any}()
        for i in circshift(collect(eachindex(states)),mod(rep+k,5))
            st=states[i];m=st.method;z=st.zhist[end]
            payload=get!(payloads,Tuple(z)) do
                ts=time_ns();bank=R.materialize_bank(primitives,z,cfg);Y=R.rewards(policies,bank,cfg)
                data=R.local_data(z,bank,Y,cfg)
                (bank=bank,Y=Y,data=data,seconds=(time_ns()-ts)/1e9)
            end
            ts=time_ns()
            if m.kind in (:green,:green_softmax)
                push!(st.history,R.historical_bank(s.id,cfg,z,payload.bank,payload.Y))
                data=R.green_data(z,st.history,cfg,s.id)
            else;data=payload.data;end
            bt=0.
            if m.kind in (:fresh,:green)
                g=data.hard;w=zeros(144);w[data.best]=1.
            elseif m.kind==:psoft
                tb=time_ns();w=R.bootstrap_weights(payload.Y,d["bootstrap_reps"],MersenneTwister(bootstrap_seed))
                bt=(time_ns()-tb)/1e9;g=R.combine(data.G,w,cfg)
            else
                w=R.softmax_weights(data.values;beta=d["beta_soft"]);g=R.combine(data.G,w,cfg)
            end
            all(isfinite,g)||error("Nonfinite gradient in $(s.id)")
            znew,eta,gc=R.projected_step(z,g,k)
            all(x->0<=x<=1,znew)||error("Projection failure")
            elapsed=primitive_seconds+payload.seconds+(time_ns()-ts)/1e9
            st.search_seconds[]+=elapsed;st.bootstrap_seconds[]+=bt
            push!(st.diagnostics,(iteration=k,z3=z[1],z4=z[2],g3=g[1],g4=g[2],hard_policy=data.best,
                objective_search=data.values[data.best]-R.cost(cfg,z),gradient_norm=norm(g),
                K_eff=1/sum(abs2,w),entropy=E.weight_entropy(w),max_weight=maximum(w),
                hard_gradient_gap=norm(g-data.hard),step_norm=norm(znew-z),
                clipped_components=count(x->abs(x)>C.GRAD_CLIP,g),ess=data.ess,banks=data.banks,
                lr_clips=data.lr_clips,iteration_seconds=elapsed,cumulative_seconds=st.search_seconds[],
                bootstrap_seconds=bt,search_paths=k*n))
            push!(st.zhist,znew)
        end
        (k==1||k%25==0||k==K)&&println("$phase $(s.id) rep=$rep iteration=$k/$K")
        flush(stdout)
    end
    return (runs=[(method=st.method,zhist=st.zhist,diagnostics=st.diagnostics,search_seconds=st.search_seconds[],bootstrap_seconds=st.bootstrap_seconds[]) for st in states],seeds=seedrows)
end

const CACHE_LOCK=ReentrantLock()
const TRACE_BANKS=Dict{String,Any}()
const TRACE_TASKS=Dict{String,Task}()
function trace_value(d,s,phase,z,policies)
    key=d["fingerprint"]*"/$(s.id)/$phase"
    bank=lock(CACHE_LOCK) do
        get!(TRACE_BANKS,key) do
            R.primitive_bank(s.model,d["trace_paths"],seed(s.id,phase,0,"trace");njobs=d["jobs_per_path"])
        end
    end
    pointkey=bytes2hex(sha256(key*"/"*repr(z)))
    cachepath=joinpath(d["output"],"checkpoints/trace_"*pointkey*".jls")
    task=lock(CACHE_LOCK) do
        get!(TRACE_TASKS,pointkey) do
            Threads.@spawn begin
                if isfile(cachepath);deserialize(cachepath)
                else
                    Y=R.rewards(policies,R.materialize_bank(bank,z,s.model),s.model)
                    v=maximum(vec(mean(Y;dims=2)))-R.cost(s.model,z)
                    E.atomic_serialize(cachepath,v);v
                end
            end
        end
    end
    return fetch(task)
end

function evaluate_search(d,s,phase,rep,policies,search,stem)
    cfg=s.model;path=stem*"_evaluation.jls"
    done=isfile(path) ? deserialize(path) : (completed=String[],traces=NamedTuple[],finals=NamedTuple[],values=Dict{String,Vector{Float64}}(),seconds=0.)
    completed=done.completed;traces=done.traces;finals=done.finals;values=done.values;seconds=done.seconds
    if length(completed)<5
        ts=time_ns()
        select=R.primitive_bank(cfg,d["selection_paths"],seed(s.id,phase,rep,"final_selection");njobs=d["jobs_per_path"])
        hold=R.primitive_bank(cfg,d["final_paths"],seed(s.id,phase,rep,"final_holdout");njobs=d["jobs_per_path"])
        seconds+=(time_ns()-ts)/1e9
        ks=sort(unique([0;1;5;collect(10:10:d["max_iters"]);d["max_iters"]]))
        ks=filter(k->k<=d["max_iters"],ks)
        for run in search.runs
            name=run.method.name;name in completed&&continue;ts=time_ns()
            for k in ks
                z=run.zhist[k+1];J=trace_value(d,s,phase,z,policies)
                push!(traces,(scenario=s.id,phase=phase,rep=rep,method=name,iteration=k,z3=z[1],z4=z[2],objective=J,search_paths=k*d["n_paths"]))
            end
            z=run.zhist[end]
            Ys=R.rewards(policies,R.materialize_bank(select,z,cfg),cfg);selected=argmax(vec(mean(Ys;dims=2)))
            Ye=R.rewards(policies,R.materialize_bank(hold,z,cfg),cfg);means=vec(mean(Ye;dims=2));best=argmax(means)
            v=collect(view(Ye,selected,:)).-R.cost(cfg,z);dv=collect(view(Ye,best,:)).-R.cost(cfg,z)
            values[name]=v;values["direct/"*name]=dv
            push!(finals,(scenario=s.id,A=s.A,B=s.B,C=s.C,phase=phase,rep=rep,method=name,z3=z[1],z4=z[2],
                objective_holdout=mean(v),objective_direct=means[best]-R.cost(cfg,z),selection_policy=selected,direct_policy=best,
                evaluation_se=std(v)/sqrt(length(v)),direct_selected_se=std(dv)/sqrt(length(dv)),
                search_seconds=run.search_seconds,bootstrap_seconds=run.bootstrap_seconds,
                total_variation=sum(x.step_norm for x in run.diagnostics),search_paths=d["max_iters"]*d["n_paths"]))
            seconds+=(time_ns()-ts)/1e9;push!(completed,name)
            E.atomic_serialize(path,(completed=completed,traces=traces,finals=finals,values=values,seconds=seconds))
        end
    end
    pairs=[(METHODS[2].name,METHODS[1].name),(METHODS[3].name,METHODS[1].name),
           (METHODS[5].name,METHODS[4].name),(METHODS[4].name,METHODS[1].name)]
    paired=NamedTuple[]
    for (a,b) in pairs,estimand in ("holdout","direct")
        prefix=estimand=="direct" ? "direct/" : "";delta=values[prefix*a]-values[prefix*b]
        push!(paired,(scenario=s.id,A=s.A,B=s.B,C=s.C,phase=phase,rep=rep,estimand=estimand,method_a=a,method_b=b,
            difference=mean(delta),evaluation_paired_se=std(delta)/sqrt(length(delta))))
    end
    return (traces=traces,finals=finals,paired=paired,evaluation_seconds=seconds)
end

function optimization_phase(d,phase,targets,reps)
    policies=C.build_policy_grid();out=d["output"]
    tasks=[(s,r) for s in targets for r in 1:reps]
    Threads.@threads :dynamic for (s,rep) in tasks
        dir=joinpath(out,phase,s.id);mkpath(dir)
        stem=joinpath(out,"checkpoints",@sprintf("%s_%s_rep_%02d",phase,s.id,rep))
        complete=stem*"_complete.jls"
        if isfile(complete)
            deserialize(complete).fingerprint==d["fingerprint"]||error("Invalid checkpoint")
            println("REUSE $phase $(s.id) rep=$rep");continue
        end
        sp=stem*"_search.jls"
        sr=if isfile(sp);deserialize(sp)
        else;x=search(d,s,phase,rep,policies);E.atomic_serialize(sp,x);x;end
        ev=evaluate_search(d,s,phase,rep,policies,sr,stem)
        prefix=joinpath(dir,@sprintf("rep_%02d",rep))
        E.write_csv(prefix*"_traces.csv",ev.traces);E.write_csv(prefix*"_finals.csv",ev.finals);E.write_csv(prefix*"_paired.csv",ev.paired)
        diag=NamedTuple[];coords=NamedTuple[]
        for run in sr.runs
            append!(diag,[merge((scenario=s.id,phase=phase,rep=rep,method=run.method.name,),x) for x in run.diagnostics])
            append!(coords,[(scenario=s.id,phase=phase,rep=rep,method=run.method.name,iteration=k-1,z3=z[1],z4=z[2]) for (k,z) in enumerate(run.zhist)])
        end
        E.write_csv(prefix*"_diagnostics.csv",diag);E.write_csv(prefix*"_coordinates.csv",coords)
        seeds=vcat(sr.seeds,NamedTuple[(scenario=s.id,phase=phase,rep=0,stream="trace",iteration=0,seed=seed(s.id,phase,0,"trace")),
            (scenario=s.id,phase=phase,rep=rep,stream="final_selection",iteration=0,seed=seed(s.id,phase,rep,"final_selection")),
            (scenario=s.id,phase=phase,rep=rep,stream="final_holdout",iteration=0,seed=seed(s.id,phase,rep,"final_holdout"))])
        E.write_csv(prefix*"_seeds.csv",seeds)
        E.write_csv(prefix*"_costs.csv",[(scenario=s.id,phase=phase,rep=rep,evaluation_seconds=ev.evaluation_seconds,
            selection_paths=d["selection_paths"],final_paths=d["final_paths"],trace_paths=d["trace_paths"])])
        E.atomic_serialize(complete,(fingerprint=d["fingerprint"],scenario=s.id,phase=phase,rep=rep))
        println("COMPLETE $phase $(s.id) rep=$rep");flush(stdout)
    end
    open(joinpath(out,"configs/$(phase)_plan.toml"),"w") do io
        TOML.print(io,Dict("scenarios"=>[s.id for s in targets],"n_reps"=>reps,"complete"=>true,"fingerprint"=>d["fingerprint"]);sorted=true)
    end
end

function oracle_surfaces(d,targets)
    policies=C.build_policy_grid()
    Threads.@threads :dynamic for s in targets
        path=joinpath(d["output"],"checkpoints/oracle_$(s.id).jls")
        if isfile(path);println("REUSE oracle $(s.id)");continue;end
        bank=R.primitive_bank(s.model,d["oracle_paths"],seed(s.id,"oracle",0,"surface");njobs=d["jobs_per_path"])
        grid=range(0,1;length=d["oracle_grid_n"]);rows=NamedTuple[];ts=time_ns()
        for z3 in grid,z4 in grid
            z=[z3,z4];Y=R.rewards(policies,R.materialize_bank(bank,z,s.model),s.model)
            push!(rows,(scenario=s.id,z3=z3,z4=z4,objective=maximum(vec(mean(Y;dims=2)))-R.cost(s.model,z),n_paths=d["oracle_paths"]))
        end
        E.write_csv(joinpath(d["output"],"characterization/oracle_$(s.id).csv"),rows)
        E.atomic_serialize(path,(fingerprint=d["fingerprint"],scenario=s.id,seconds=(time_ns()-ts)/1e9))
        println("COMPLETE oracle $(s.id)")
    end
end

function main(args=ARGS)
    BLAS.set_num_threads(1);d=parse_arguments(args);d===nothing&&return
    initialize(d);out=d["output"];targets=scenario_targets(d)
    if "characterize" in d["actions"]
        characterize(d,targets)
        run(`python3 $(joinpath(ROOT,"analyze_parameter_regimes.py")) $out --classify`)
    end
    "pilot" in d["actions"]&&optimization_phase(d,"pilot",targets,d["pilot_reps"])
    confirmed=NamedTuple[]
    if "confirm" in d["actions"]
        confirmed=if !isempty(d["scenarios"])||d["all_scenarios"];targets
        else
            p=joinpath(out,"configs/selected_scenarios.toml");isfile(p)||error("Run --characterize before automatic confirmation")
            choice=TOML.parsefile(p);choice["fingerprint"]==d["fingerprint"]||error("Stale structural selection")
            filter(s->s.id in choice["scenarios"],R.scenarios())
        end
        optimization_phase(d,"confirmation",confirmed,d["confirm_reps"])
        oracle_surfaces(d,confirmed)
    end
    verify_sources(d)
    if "plots" in d["actions"]||"report" in d["actions"]
        run(`python3 $(joinpath(ROOT,"analyze_parameter_regimes.py")) $out --analyze`)
    end
    println("Requested actions complete: ",d["actions"],"; ",out)
    return d
end
end
