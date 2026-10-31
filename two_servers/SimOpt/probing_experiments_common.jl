# Additive experiments 005--008. The ex-ante objective is always
# J(z)=max_i E_z[Y_i]-ALPHA'z. No outer conditional-expectation stage is added.
module ProbingExperiments
module Model
include(joinpath(@__DIR__, "003probing_four_searches_softmax.jl"))
end
using .Model
using Random, Statistics, LinearAlgebra, Serialization, TOML, SHA, Printf, Dates
const C = Model
const ROOT = @__DIR__
const NAMES = Dict(5=>"005probing_softmax_multiseed", 6=>"006probing_softmax_sample_efficiency",
                   7=>"007probing_green_softmax", 8=>"008probing_probability_softmax")
const OLD_FILES = ["002probing_three_searches_n500_full45iters.jl",
    "003probing_four_searches_softmax.jl", "004probing_softmax_sensitivity.jl",
    "SimOpt005.jl", "SimOpt009.jl", "SimOpt009_2D_clean.jl"]
const TIE_ATOL = 1e-10
const TIE_RTOL = 1e-12

"""Independent deterministic streams; method is intentionally absent for CRN."""
function stream_seed(stage, rep, stream, iteration=0)
    h = bytes2hex(sha256("Probing005-008-v1/$stage/$rep/$stream/$iteration"))
    return Int(parse(UInt64, h[1:15]; base=16)) + 1
end

function atomic_serialize(path, value)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, value); flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 || error("fsync failed")
    end
    mv(tmp, path; force=true)
end

csv_value(x::Missing) = ""
csv_value(x::AbstractFloat) = @sprintf("%.17g", x)
csv_value(x) = occursin(r"[,\"\n]",string(x)) ? "\""*replace(string(x),"\""=>"\"\"")*"\"" : string(x)
function write_csv(path, rows)
    isempty(rows) && return
    ks = propertynames(first(rows))
    tmp = path * ".tmp"
    open(tmp,"w") do io
        println(io,join(string.(ks),','))
        for r in rows
            println(io,join((csv_value(getproperty(r,k)) for k in ks),','))
        end
    end
    mv(tmp,path;force=true)
end

function parse_arguments(stage, args)
    d = Dict{String,Any}("stage"=>stage,"quick"=>false,"resume"=>false,
        "n_reps"=>20,"n_paths"=>stage==6 ? [50,100,200,500,1000] : [500],
        "max_iters"=>100,"jobs_per_path"=>45,"budget"=>50000,
        "betas"=>[.05,.10,.20,.50],"bootstrap_reps"=>300,
        "trace_paths"=>1000,"selection_paths"=>5000,"final_paths"=>30000,
        "output"=>joinpath(ROOT,"resultados",NAMES[stage]),
        "selection"=>joinpath(ROOT,"resultados",NAMES[5],"selected_beta.toml"))
    supplied=Set{String}(); i=1
    while i<=length(args)
        a=args[i]
        if a=="--quick-test"; d["quick"]=true
        elseif a=="--resume"; d["resume"]=true
        elseif a=="--help"
            println("Usage: julia $(NAMES[stage]).jl [--quick-test] [--resume] [--n-reps 20] [--n-paths 500 (or 50,100,200,500,1000 for 006)] [--max-iters 100] [--betas .05,.1,.2,.5] [--bootstrap-reps 300] [--output DIR] [--selection FILE] [--trace-paths 1000] [--selection-paths 5000] [--final-paths 30000]")
            return nothing
        elseif a in ("--n-reps","--n-paths","--max-iters","--betas","--bootstrap-reps",
                     "--output","--selection","--trace-paths","--selection-paths","--final-paths","--budget")
            i+=1; i<=length(args) || error("Missing value for $a")
            key=replace(a[3:end],'-'=>'_'); push!(supplied,key)
            d[key] = key in ("output","selection") ? abspath(args[i]) :
                     key=="betas" ? sort(unique(parse.(Float64,split(args[i],',')))) :
                     key=="n_paths" ? sort(unique(parse.(Int,split(args[i],',')))) : parse(Int,args[i])
        else; error("Unknown argument: $a")
        end
        i+=1
    end
    if d["quick"]
        for (k,v) in ("n_reps"=>2,"n_paths"=>stage==6 ? [10,20] : [20],
            "max_iters"=>4,"budget"=>100,"trace_paths"=>30,"selection_paths"=>40,"final_paths"=>80)
            k in supplied || (d[k]=v)
        end
        "output" in supplied || (d["output"]=joinpath(d["output"],"prueba"))
    end
    all(k->d[k]>0,("n_reps","max_iters","budget","trace_paths","selection_paths","final_paths","bootstrap_reps")) || error("Counts must be positive")
    all(>(1),d["n_paths"]) || error("At least two paths required")
    all(b->isfinite(b)&&b>0,d["betas"]) && !isempty(d["betas"]) || error("Invalid betas")
    stage==6 || length(d["n_paths"])==1 || error("This stage accepts one N_PATHS")
    if stage!=5
        if d["quick"]
            # Quick tests validate code only and never select a scientific beta.
            d["betas"]=[.1];d["selection_sha256"]="quick-test-no-selection"
        else
            isfile(d["selection"]) || error("Complete stage 005 first: missing $(d["selection"])")
            s=TOML.parsefile(d["selection"])
            get(s,"complete",false) && !get(s,"quick",true) || error("005 selection is incomplete or a quick test")
            d["betas"]=sort(unique([.1,Float64(s["selected_beta"])]))
            d["selection_sha256"]=bytes2hex(sha256(read(d["selection"])))
        end
    end
    d["source_sha256"]=Dict(f=>bytes2hex(sha256(read(joinpath(ROOT,f)))) for f in [OLD_FILES;"probing_experiments_common.jl";NAMES[stage]*".jl"])
    d["julia_version"]=string(VERSION)
    d["blas_threads"]=1
    d["threads"]=Threads.nthreads()
    d["tie_atol"]=TIE_ATOL; d["tie_rtol"]=TIE_RTOL
    d["model"]=Dict("ALPHA"=>C.ALPHA,"Z0"=>C.Z0,"BETA"=>C.BETA,
        "BASE_STEP"=>C.BASE_STEP,"GRAD_CLIP"=>C.GRAD_CLIP,"policies"=>144,
        "P_COMPLEX"=>C.P_COMPLEX,"P_HIT0"=>C.P_HIT0,"P_FALSE0"=>C.P_FALSE0)
    return d
end

"""Resample complete reward columns jointly. No policy simulation occurs here.
Ties receive 1/number_of_ties credit when max-value <= atol+rtol*abs(max).
The frequency matrix uses the SAME sampled path counts for every policy.
"""
function bootstrap_weights(Y, B, rng; atol=TIE_ATOL, rtol=TIE_RTOL, return_counts=false)
    m,n=size(Y); m>0 && n>0 && B>0 || error("Empty bootstrap input")
    all(isfinite,Y) || error("Nonfinite bootstrap rewards")
    counts=zeros(Float64,n,B)
    for b in 1:B, l in 1:n
        counts[rand(rng,1:n),b]+=1.0
    end
    boot=(Y*counts)/n
    p=zeros(m)
    for b in 1:B
        vals=view(boot,:,b); vmax=maximum(vals)
        ties=findall(x->vmax-x<=atol+rtol*abs(vmax),vals)
        p[ties].+=inv(B*length(ties))
    end
    p./=sum(p)
    all(x->isfinite(x)&&0<=x<=1,p) && isapprox(sum(p),1;atol=1e-12) || error("Invalid bootstrap probabilities")
    return return_counts ? (p,counts,boot) : p
end

combine_gradients(G,w)=transpose(G)*w-C.ALPHA
weight_entropy(w)=-sum(x>0 ? x*log(x) : 0. for x in w)
cosine(a,b)=norm(a)>1e-12 && norm(b)>1e-12 ? clamp(dot(a,b)/(norm(a)*norm(b)),-1.,1.) : missing

function primitive_bank(d,n,seed)
    rng=MersenneTwister(seed)
    return [C.generate_crn_primitive_path(rng;njobs=d["jobs_per_path"]) for _ in 1:n]
end

function local_data(z,bank,Y)
    n=length(bank); scores=zeros(2,n)
    for l in 1:n
        ll,sc=C.loglr_and_score(bank[l],z,z)
        ll==0 || error("Fresh LR is not one")
        scores[:,l].=sc
    end
    values=vec(mean(Y;dims=2)); best=argmax(values)
    G=(Y*transpose(scores))/n
    # Preserve Fresh's exact reduction order from 003.
    hard=[mean(view(Y,best,:).*view(scores,j,:))-C.ALPHA[j] for j in 1:2]
    return (values=values,G=G,hard=hard,best=best,ess=Float64(n),ess_min=Float64(n),banks=1,lr_clips=0)
end

"""All-policy extension of Green. Equal average over banks, THEN Softmax.
The hard gradient retains the original Green summation order and estimator.
LR clipping is inherited unchanged. No bank-normalization of LR is introduced.
"""
function green_data(z,history)
    T=length(history); m=size(history[1].Y,1)
    values=zeros(m); G=zeros(m,2); Ls=Vector{Vector{Float64}}(); Ss=Matrix{Float64}[]
    ess=Float64[]; sumL=0.;sumL2=0.; clips=0
    for hb in history
        n=length(hb.bank); L=zeros(n); S=zeros(2,n)
        for l in 1:n
            ll,sc=C.loglr_and_score(hb.bank[l],z,hb.zgen)
            clips+=abs(ll)>35
            L[l]=exp(clamp(ll,-35.,35.)); S[:,l].=sc
        end
        push!(Ls,L);push!(Ss,S)
        for i in 1:m
            values[i]+=mean(view(hb.Y,i,:).*L)/T
        end
        G .+= (hb.Y*transpose(S.*transpose(L)))/(n*T)
        push!(ess,sum(L)^2/sum(abs2,L));sumL+=sum(L);sumL2+=sum(abs2,L)
    end
    best=argmax(values);hard=-copy(C.ALPHA)
    for (t,hb) in enumerate(history), j in 1:2
        hard[j]+=mean(view(hb.Y,best,:).*Ls[t].*view(Ss[t],j,:))/T
    end
    return (values=values,G=G,hard=hard,best=best,ess=sumL^2/sumL2,ess_min=minimum(ess),banks=T,lr_clips=clips)
end

function methods(d)
    green=d["stage"]==7
    out=[(name=green ? "Green" : "Fresh",kind=green ? :green : :fresh,beta=0.)]
    append!(out,[(name=(green ? "Green + Softmax" : "Softmax")*" beta=$(b)",kind=green ? :green_softmax : :softmax,beta=b) for b in d["betas"]])
    d["stage"]==8 && push!(out,(name="PSoft B=$(d["bootstrap_reps"])",kind=:psoft,beta=0.))
    return out
end

function search_iterations(d,n)
    return d["stage"]==6 ? max(d["max_iters"],max(1,fld(d["budget"],n))) : d["max_iters"]
end
function eval_iterations(d,n)
    k=search_iterations(d,n)
    if d["quick"]; return collect(0:k); end
    if d["stage"]==6
        return sort(unique([0;round.(Int,10 .^ range(0,log10(k);length=14));d["max_iters"];max(1,fld(d["budget"],n))]))
    end
    return sort(unique([0;1;5;collect(10:10:k);k]))
end

function search_rep(d,rep,n,policies)
    ms=methods(d); iters=search_iterations(d,n)
    states=[(method=m,zhist=[copy(C.Z0)],diagnostics=NamedTuple[],weights=NamedTuple[],
        history=C.HistoricalBank[],search_seconds=Ref(0.),bootstrap_seconds=Ref(0.)) for m in ms]
    sensitivity=NamedTuple[];diagnostic_seconds=0.
    for k in 1:iters
        ts=time_ns(); primitives=primitive_bank(d,n,stream_seed(d["stage"],rep,"search",k))
        primitive_seconds=(time_ns()-ts)/1e9
        # Rotate processing order; all methods use the same exogenous RNG stream.
        order=circshift(collect(eachindex(states)),mod(rep+k,length(states)))
        for idx in order
            st=states[idx];m=st.method;z=st.zhist[end];ts=time_ns()
            bank=C.materialize_crn_bank(primitives,z);Y=C.precompute_rewards(policies,bank)
            if m.kind in (:green,:green_softmax)
                push!(st.history,C.HistoricalBank(copy(z),bank,Y));data=green_data(z,st.history)
            else
                data=local_data(z,bank,Y)
            end
            softw=C.softmax_weights(data.values;beta=m.beta>0 ? m.beta : .1)
            softg=combine_gradients(data.G,softw)
            p=Float64[]; bg_seconds=0.
            if m.kind==:psoft
                tb=time_ns()
                p=bootstrap_weights(Y,d["bootstrap_reps"],MersenneTwister(stream_seed(8,rep,"bootstrap",k)))
                bg_seconds=(time_ns()-tb)/1e9; st.bootstrap_seconds[]+=bg_seconds
            end
            w=m.kind==:psoft ? p : m.kind in (:softmax,:green_softmax) ? softw : Float64.(eachindex(data.values).==data.best)
            g=m.kind in (:fresh,:green) ? data.hard : combine_gradients(data.G,w)
            all(isfinite,g) || error("Nonfinite gradient")
            znew,eta,gc=C.projected_step(z,g,k)
            all(x->0<=x<=1,znew) || error("Iterate outside unit square")
            push!(st.zhist,znew)
            elapsed=(time_ns()-ts)/1e9+primitive_seconds
            st.search_seconds[]+=elapsed
            # Every comparison below is at this method's SAME point and bank/history.
            push!(st.diagnostics,(iteration=k,z3=z[1],z4=z[2],g3=g[1],g4=g[2],
                hard_g3=data.hard[1],hard_g4=data.hard[2],soft_g3=softg[1],soft_g4=softg[2],
                gradient_norm=norm(g),hard_gap=norm(g-data.hard),soft_gap=norm(g-softg),
                hard_relative_gap=norm(g-data.hard)/max(norm(data.hard),1e-12),
                hard_cosine=cosine(g,data.hard),soft_cosine=cosine(g,softg),
                hard_policy=data.best,top_policy=argmax(w),max_weight=maximum(w),
                K_eff=inv(sum(abs2,w)),entropy=weight_entropy(w),
                soft_K_eff=inv(sum(abs2,softw)),soft_entropy=weight_entropy(softw),
                step_norm=norm(znew-z),clipped_components=count(x->abs(x)>C.GRAD_CLIP,g),
                ess=data.ess,ess_min=data.ess_min,banks=data.banks,lr_clips=data.lr_clips,
                iteration_seconds=elapsed,bootstrap_seconds=bg_seconds,cumulative_seconds=st.search_seconds[],
                search_paths=k*n))
            if d["stage"] in (7,8)
                push!(st.weights,(iteration=k,values=data.values,G=data.G,weights=w,soft_weights=softw))
            end
            # Sensitivity uses stored Y only, with independent bootstrap streams.
            # Two prespecified banks (iterations 1 and 50) per repetition.
            if m.kind==:psoft && k in (1,min(50,iters))
                sensitivity_start=time_ns()
                reference=bootstrap_weights(Y,10000,MersenneTwister(stream_seed(8,rep,"bootstrap_reference",k)))
                for B in (100,300,1000), q in 1:10
                    tb=time_ns()
                    pp=bootstrap_weights(Y,B,MersenneTwister(stream_seed(8,rep,"sensitivity_$B/$q",k)))
                    push!(sensitivity,(rep=rep,iteration=k,B=B,draw=q,l1_error=sum(abs,pp-reference),
                        squared_error=sum(abs2,pp-reference),gradient_error=norm(combine_gradients(data.G,pp)-combine_gradients(data.G,reference)),
                        seconds=(time_ns()-tb)/1e9))
                end
                diagnostic_seconds+=(time_ns()-sensitivity_start)/1e9
            end
        end
        (k==1 || k%25==0 || k==iters) && println("stage=$(d["stage"]) rep=$rep n=$n iteration=$k/$iters")
        flush(stdout)
    end
    runs=[(method=st.method,zhist=st.zhist,diagnostics=st.diagnostics,weights=st.weights,
        search_seconds=st.search_seconds[],bootstrap_seconds=st.bootstrap_seconds[]) for st in states]
    return (runs=runs,sensitivity=sensitivity,diagnostic_seconds=diagnostic_seconds)
end

function evaluate_rep(d,rep,n,policies,search,stem)
    ep=stem*"_evaluation.jls"
    done=isfile(ep) ? deserialize(ep) : (traces=NamedTuple[],finals=NamedTuple[],rewards=Dict{String,Vector{Float64}}(),completed=String[],seconds=0.)
    traces=done.traces; finals=done.finals; rewards=done.rewards; completed=done.completed; secs=done.seconds
    ts=time_ns()
    tracebank=primitive_bank(d,d["trace_paths"],stream_seed(d["stage"],rep,"trace"))
    selectbank=primitive_bank(d,d["selection_paths"],stream_seed(d["stage"],rep,"final_selection"))
    finalbank=primitive_bank(d,d["final_paths"],stream_seed(d["stage"],rep,"final_evaluation"))
    secs+=(time_ns()-ts)/1e9
    checkpoints=eval_iterations(d,n)
    horizons=d["stage"]==6 ? [("fixed_iterations",d["max_iters"]),("fixed_budget",max(1,fld(d["budget"],n)))] : [("final",d["max_iters"])]
    for run in search.runs
        name=run.method.name; name in completed && continue
        ts=time_ns()
        # Deduplicate repeated decisions, but evaluate every prespecified checkpoint.
        cache=Dict{Tuple{Float64,Float64},Float64}()
        for k in checkpoints
            z=run.zhist[k+1]
            J=get!(cache,Tuple(z)) do
                Y=C.precompute_rewards(policies,C.materialize_crn_bank(tracebank,z))
                maximum(vec(mean(Y;dims=2)))-dot(C.ALPHA,z)
            end
            search_time=k==0 ? 0. : run.diagnostics[k].cumulative_seconds
            push!(traces,(rep=rep,n_paths=n,method=name,iteration=k,search_paths=k*n,
                search_seconds=search_time,z3=z[1],z4=z[2],objective=J))
        end
        for (analysis,k) in horizons
            z=run.zhist[k+1]
            Ys=C.precompute_rewards(policies,C.materialize_crn_bank(selectbank,z))
            selected=argmax(vec(mean(Ys;dims=2)))
            Ye=C.precompute_rewards(policies,C.materialize_crn_bank(finalbank,z))
            means=vec(mean(Ye;dims=2));best=argmax(means);cost=dot(C.ALPHA,z)
            vals=collect(view(Ye,selected,:)).-cost
            maxvals=collect(view(Ye,best,:)).-cost
            key="$analysis/$name";rewards[key]=vals;rewards["direct/$key"]=maxvals
            push!(finals,(rep=rep,n_paths=n,analysis=analysis,method=name,beta=run.method.beta,
                iteration=k,search_paths=k*n,z3=z[1],z4=z[2],objective_direct=means[best]-cost,
                objective_holdout=mean(vals),selection_policy=selected,direct_policy=best,
                evaluation_se=std(vals)/sqrt(length(vals)),direct_selected_se=std(maxvals)/sqrt(length(maxvals)),
                search_seconds=run.diagnostics[k].cumulative_seconds,
                bootstrap_seconds=sum(x.bootstrap_seconds for x in run.diagnostics[1:k]),
                total_variation=sum(x.step_norm for x in run.diagnostics[1:k]),
                terminal_step_sd=std([x.step_norm for x in run.diagnostics[max(1,k-19):k]])))
        end
        secs+=(time_ns()-ts)/1e9;push!(completed,name)
        atomic_serialize(ep,(traces=traces,finals=finals,rewards=rewards,completed=completed,seconds=secs))
    end
    paired=NamedTuple[]
    for (analysis,k) in horizons
        baseline=search.runs[1].method.name
        pairs=[(r.method.name,baseline) for r in search.runs[2:end]]
        if d["stage"]==8
            append!(pairs,[(search.runs[end].method.name,r.method.name) for r in search.runs[2:end-1]])
        end
        for (a,b) in pairs
            ar=only(filter(r->r.analysis==analysis&&r.method==a,finals));br=only(filter(r->r.analysis==analysis&&r.method==b,finals))
            for estimand in ("holdout","direct")
                prefix=estimand=="direct" ? "direct/" : ""
                delta=rewards["$prefix$analysis/$a"]-rewards["$prefix$analysis/$b"]
                estimate=estimand=="direct" ? ar.objective_direct-br.objective_direct : ar.objective_holdout-br.objective_holdout
                push!(paired,(rep=rep,n_paths=n,analysis=analysis,estimand=estimand,method_a=a,method_b=b,
                    difference=estimate,evaluation_paired_se=std(delta)/sqrt(length(delta))))
            end
        end
    end
    return (traces=traces,finals=finals,paired=paired,evaluation_seconds=secs)
end

function export_rep(d,rep,n,search,evaluation,stem)
    write_csv(stem*"_traces.csv",evaluation.traces)
    write_csv(stem*"_finals.csv",evaluation.finals)
    write_csv(stem*"_paired.csv",evaluation.paired)
    diag=NamedTuple[];weights=NamedTuple[];coords=NamedTuple[]
    for r in search.runs
        append!(diag,[merge((rep=rep,n_paths=n,method=r.method.name,),x) for x in r.diagnostics])
        append!(coords,[(rep=rep,n_paths=n,method=r.method.name,iteration=k-1,z3=z[1],z4=z[2]) for (k,z) in enumerate(r.zhist)])
        for x in r.weights, i in eachindex(x.values)
            push!(weights,(rep=rep,n_paths=n,method=r.method.name,iteration=x.iteration,policy=i,
                value=x.values[i],G3=x.G[i,1],G4=x.G[i,2],weight=x.weights[i],soft_weight=x.soft_weights[i]))
        end
    end
    write_csv(stem*"_diagnostics.csv",diag);write_csv(stem*"_coordinates.csv",coords)
    write_csv(stem*"_weights.csv",weights);write_csv(stem*"_bootstrap_sensitivity.csv",search.sensitivity)
    write_csv(stem*"_costs.csv",[(rep=rep,n_paths=n,evaluation_seconds=evaluation.evaluation_seconds,
        sensitivity_seconds=search.diagnostic_seconds,
        trace_paths=d["trace_paths"],selection_paths=d["selection_paths"],final_paths=d["final_paths"])])
end

function initialize(d)
    out=d["output"];mkpath(out);mkpath(joinpath(out,"checkpoints"));mkpath(joinpath(out,"replications"))
    fp=joinpath(out,"configuration.toml"); saved=filter(p->p.first!="resume",d)
    if isfile(fp)
        d["resume"] || error("Output already exists; use --resume or --output NEW_DIRECTORY: $out")
        old=TOML.parsefile(fp)
        old==saved || error("Resume configuration/source mismatch; keep the original code/options or choose a new output")
    else
        open(fp,"w") do io;TOML.print(io,saved;sorted=true);end
        seeds=NamedTuple[]
        for rep in 1:d["n_reps"]
            for stream in ("trace","final_selection","final_evaluation")
                push!(seeds,(rep=rep,stream=stream,iteration=0,seed=stream_seed(d["stage"],rep,stream)))
            end
            for k in 1:maximum(search_iterations(d,n) for n in d["n_paths"])
                push!(seeds,(rep=rep,stream="search",iteration=k,seed=stream_seed(d["stage"],rep,"search",k)))
                d["stage"]==8 && push!(seeds,(rep=rep,stream="bootstrap",iteration=k,seed=stream_seed(8,rep,"bootstrap",k)))
            end
        end
        length(unique(s.seed for s in seeds))==length(seeds) || error("RNG seed collision")
        write_csv(joinpath(out,"seeds.csv"),seeds)
        write_csv(joinpath(out,"policies.csv"),[(policy=i,av=p.av,aq=p.aq,as=p.as,tau=p.tau) for (i,p) in enumerate(C.build_policy_grid())])
        open(joinpath(out,"reproduce.txt"),"w") do io
            println(io,"cd ",repr(ROOT))
            opts=["--n-reps",string(d["n_reps"]),"--n-paths",join(d["n_paths"],','),"--max-iters",string(d["max_iters"]),
                "--trace-paths",string(d["trace_paths"]),"--selection-paths",string(d["selection_paths"]),"--final-paths",string(d["final_paths"]),
                "--bootstrap-reps",string(d["bootstrap_reps"]),"--budget",string(d["budget"]),"--betas",join(d["betas"],','),
                "--output",d["output"],"--selection",d["selection"]]
            d["quick"] && push!(opts,"--quick-test")
            println(io,"env GKSwstype=100 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=$(d["threads"]) julia --startup-file=no --project=resultados/003probing_softmax_20261021/entorno ",NAMES[d["stage"]],".jl ",join(repr.(opts),' ')," --resume")
        end
    end
end

function main(stage,args=ARGS)
    BLAS.set_num_threads(1)
    d=parse_arguments(stage,args);d===nothing && return
    initialize(d);out=d["output"]
    logpath=joinpath(out,"execution.log")
    open(logpath,"a") do io;println(io,"$(Dates.now()) START stage=$stage quick=$(d["quick"]) resume=$(d["resume"]) threads=$(Threads.nthreads())");end
    policies=C.build_policy_grid();tasks=[(rep,n) for rep in 1:d["n_reps"] for n in d["n_paths"]]
    Threads.@threads :dynamic for (rep,n) in tasks
        base=@sprintf("rep_%02d_n_%04d",rep,n)
        stem=joinpath(out,"checkpoints",base)
        complete=stem*"_complete.jls"
        if isfile(complete)
            println("REUSE complete $base");continue
        end
        sp=stem*"_search.jls"
        search=if isfile(sp)
            println("REUSE search $base");deserialize(sp)
        else
            s=search_rep(d,rep,n,policies);atomic_serialize(sp,s);s
        end
        evaluation=evaluate_rep(d,rep,n,policies,search,stem)
        export_rep(d,rep,n,search,evaluation,joinpath(out,"replications",base))
        atomic_serialize(complete,(rep=rep,n_paths=n,complete=true))
        println("COMPLETE $base");flush(stdout)
    end
    all(isfile(joinpath(out,"checkpoints",@sprintf("rep_%02d_n_%04d_complete.jls",rep,n))) for (rep,n) in tasks) || error("Incomplete stage")
    for f in OLD_FILES
        bytes2hex(sha256(read(joinpath(ROOT,f))))==d["source_sha256"][f] || error("Original source changed: $f")
    end
    # The analysis program writes COMPLETE only after tables, plots and report succeed.
    run(`python3 $(joinpath(ROOT,"analyze_probing_experiments.py")) $out`)
    open(logpath,"a") do io;println(io,"$(Dates.now()) COMPLETE stage=$stage replications=$(length(tasks))");end
    println("Stage $stage complete: $out")
    return d
end
end # module
