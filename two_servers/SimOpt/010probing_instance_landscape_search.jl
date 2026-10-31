# Additive landscape experiment. Original model and algorithm files are untouched.
module ProbingLandscapeSearch
include(joinpath(@__DIR__, "probing_regime_model.jl"))
using .ProbingRegimeModel
using Random, Statistics, LinearAlgebra, Serialization, SHA, TOML, Dates, Printf
const R=ProbingRegimeModel
const E=R.E
const C=R.C
const ROOT=@__DIR__
const METHODS=[(id="Fixed",kind=:fixed,beta=0.),(id="Fresh",kind=:fresh,beta=0.),
    (id="Green",kind=:green,beta=0.),(id="Softmax010",kind=:softmax,beta=.10),
    (id="Softmax050",kind=:softmax,beta=.50),(id="PSoft300",kind=:psoft,beta=0.),
    (id="GreenSoftmax050",kind=:green_softmax,beta=.50)]
const FILES=["002probing_three_searches_n500_full45iters.jl","003probing_four_searches_softmax.jl",
    "004probing_softmax_sensitivity.jl","008probing_probability_softmax.jl",
    "009probing_parameter_regimes.jl","009b_probing_regime_validation.jl",
    "probing_regime_model.jl","probing_experiments_common.jl",
    "010probing_instance_landscape_search.jl","analyze_landscape_010.py","PROTOCOLO_010.txt"]
seed(s,stage,rep,stream,k=0)=Int(parse(UInt64,bytes2hex(sha256("Probing010-v1/$s/$stage/$rep/$stream/$k"))[1:15];base=16))+1
sha(path)=bytes2hex(sha256(read(path)))
# A zero difference with zero SE has undefined SNR, not infinite evidence.
safe_snr(delta,se)=se>0 ? abs(delta)/se : iszero(delta) ? NaN : Inf
function arguments(args)
    d=Dict{String,Any}("output"=>joinpath(ROOT,"resultados/010probing_instance_landscape_search"),
        "quick"=>false,"resume"=>false,"actions"=>String[],"configs"=>56,"refine_count"=>8,"selected_count"=>4,
        "screen_grid"=>5,"screen_paths"=>400,"refine_grid"=>11,"refine_paths"=>3000,
        "high_grid"=>21,"high_paths"=>5000,"n_paths"=>500,"jobs"=>45,"max_iters"=>100,
        "reps"=>5,"trace_paths"=>1000,"selection_paths"=>10000,"final_paths"=>50000,"bootstrap_reps"=>300)
    supplied=Set{String}();i=1
    while i<=length(args)
        a=args[i]
        if a=="--quick-test";d["quick"]=true
        elseif a=="--resume";d["resume"]=true
        elseif a in ("--screen","--refine","--compare","--plots");push!(d["actions"],a[3:end])
        elseif a=="--help"
            println("010 landscape search: [--quick-test] [--screen] [--refine] [--compare] [--plots] [--resume] [--output DIR] [--configs 56] [--screen-paths 400] [--refine-paths 3000] [--high-paths 5000] [--trace-paths 1000] [--selection-paths 10000] [--final-paths 50000] [--reps 5] [--n-paths 500] [--max-iters 100] [--screen-grid 5] [--refine-grid 11] [--high-grid 21] [--refine-count 8] [--selected-count 4]")
            return nothing
        elseif startswith(a,"--")
            key=replace(a[3:end],'-'=>'_');haskey(d,key)||error("Unknown option $a")
            i+=1;i<=length(args)||error("Missing value for $a")
            d[key]=key=="output" ? abspath(args[i]) : parse(Int,args[i]);push!(supplied,key)
        else;error("Unexpected $a");end
        i+=1
    end
    isempty(d["actions"])&&(d["actions"]=["screen","refine","compare","plots"])
    if d["quick"]
        for (k,v) in ("configs"=>5,"refine_count"=>3,"selected_count"=>3,"screen_grid"=>3,"screen_paths"=>20,
            "refine_grid"=>3,"refine_paths"=>30,"high_grid"=>3,"high_paths"=>40,
            "n_paths"=>20,"max_iters"=>4,"reps"=>2,"trace_paths"=>30,"selection_paths"=>40,"final_paths"=>80)
            k in supplied||(d[k]=v)
        end
        "output" in supplied||(d["output"]=joinpath(d["output"],"quick_test"))
    end
    for k in keys(d)
        d[k] isa Int && !(d[k] isa Bool) && d[k]<2 && error("$k must be >=2")
    end
    3<=d["selected_count"]<=d["refine_count"]<=d["configs"]||error("Invalid selection sizes")
    sources=Dict(f=>sha(joinpath(ROOT,f)) for f in FILES)
    science=Dict{String,Any}(k=>v for (k,v) in d if !(k in ("output","actions","resume")))
    science["sources"]=sources;science["julia_version"]=string(VERSION)
    science["Z0"]=C.Z0;science["BASE_STEP"]=C.BASE_STEP;science["GRAD_CLIP"]=C.GRAD_CLIP
    d["science"]=science;d["fingerprint"]=R.digest(science);d
end

function design(n)
    levels=([.42,.84,1.68,3.36],[(8.,2.4),(12.,2.4),(12.,1.8),(16.,1.8)],
        [0,1],[1.,2.2,4.],[(5.,2.25),(7.5,3.375),(10.,4.5),(15.,6.75)],[.60,.80,1.])
    original=(1,1,1,2,3,2);codes=[original];origins=["original"]
    for j in 1:6,lev in eachindex(levels[j])
        c=ntuple(k->k==j ? lev : original[k],6)
        c in codes&&continue;push!(codes,c);push!(origins,"one_factor_$j")
    end
    rng=MersenneTwister(seed("design","frozen",0,"categorical"))
    # Reproducible discrete design, no full factorial evaluation.
    while length(codes)<n
        c=ntuple(j->rand(rng,eachindex(levels[j])),6)
        c in codes&&continue;push!(codes,c);push!(origins,"random_combination")
    end
    out=NamedTuple[]
    for (i,c) in enumerate(codes[1:n])
        delay,service,signal,beta,alpha,arrival=(levels[j][c[j]] for j in 1:6)
        cfg=R.validate(R.ModelConfig(delay_penalty=delay,std_mean=(1.,service[1]),fast_mean=(.75,service[2]),
            p_hit0=signal==0 ? (.60,.65,.75,.92) : (.60,.65,.55,.65),
            p_false0=signal==0 ? (.05,.08,.15,.20) : (.05,.08,.20,.25),beta=beta,alpha=alpha,mean_interarrival=arrival))
        id=@sprintf("L%03d",i-1)
        push!(out,(id=id,model=cfg,origin=origins[i],code=c))
    end
    out
end
function initialize(d)
    out=d["output"]
    for folder in ("configs","checkpoints","screen","refine","high","comparisons","figures","tables","logs","validation","reproduction")
        mkpath(joinpath(out,folder))
    end
    path=joinpath(out,"configs/experiment.toml")
    if isfile(path)
        TOML.parsefile(path)["fingerprint"]==d["fingerprint"]||error("Source/budget mismatch; use a new output directory")
        d["resume"]||d["actions"]==["plots"]||error("Existing results: use --resume")
    else
        open(path,"w") do io;TOML.print(io,Dict("fingerprint"=>d["fingerprint"],"science"=>d["science"],"threads"=>Threads.nthreads());sorted=true);end
        ss=design(d["configs"])
        E.atomic_serialize(joinpath(out,"configs/design.jls"),ss)
        models=Dict(s.id=>merge(R.model_dict(s.model),Dict("origin"=>s.origin,"code"=>collect(s.code),"model_sha256"=>R.model_digest(s.model))) for s in ss)
        open(joinpath(out,"configs/models.toml"),"w") do io;TOML.print(io,models;sorted=true);end
        E.write_csv(joinpath(out,"configs/configurations.csv"),[(scenario=s.id,origin=s.origin,delay=s.model.delay_penalty,
            std_complex=s.model.std_mean[2],fast_complex=s.model.fast_mean[2],signal=s.code[3]-1,beta_signal=s.model.beta,
            alpha3=s.model.alpha[1],alpha4=s.model.alpha[2],mean_interarrival=s.model.mean_interarrival,
            revenue_scale=s.model.revenue_scale,model_sha256=R.model_digest(s.model)) for s in ss])
        E.write_csv(joinpath(out,"configs/policies.csv"),[(policy=i,av=p.av,aq=p.aq,as=p.as,tau=p.tau) for (i,p) in enumerate(C.build_policy_grid())])
        for f in FILES;cp(joinpath(ROOT,f),joinpath(out,"reproduction",f));end
        env=joinpath(ROOT,"resultados/003probing_softmax_20261021/entorno")
        for f in ("Project.toml","Manifest.toml");cp(joinpath(env,f),joinpath(out,"reproduction",f));end
        for f in ("probing_regime_model.jl","probing_experiments_common.jl","003probing_four_searches_softmax.jl")
            sha(joinpath(ROOT,f))==d["science"]["sources"][f]||error("Source mutation")
        end
    end
    open(joinpath(out,"logs/invocations.log"),"a") do io;println(io,Dates.now()," ",repr(ARGS)," threads=",Threads.nthreads());end
    deserialize(joinpath(out,"configs/design.jls"))
end
physical_digest(cfg)=R.digest(Dict(k=>v for (k,v) in R.model_dict(cfg) if k!="ALPHA"))
function point(d,s,stage,z,primitives,policies)
    # Gross values do not depend on ALPHA. Reuse exact simulations across cost-only cases.
    key=bytes2hex(sha256("$(physical_digest(s.model))/$stage/$(repr(z))/$(length(primitives))/$(d["jobs"])"))
    path=joinpath(d["output"],"checkpoints/gross_$key.jls")
    if isfile(path);return deserialize(path);end
    Y=R.rewards(policies,R.materialize_bank(primitives,z,s.model),s.model)
    values=vec(mean(Y;dims=2));best=argmax(values);v=collect(view(Y,best,:))
    ans=(z=copy(z),gross=values[best],policy=best,rewards=v,se=std(v)/sqrt(length(v)),means=values)
    E.atomic_serialize(path,ans);ans
end
function components(mask)
    seen=falses(size(mask));n=0
    for I in CartesianIndices(mask)
        (!mask[I]||seen[I])&&continue;n+=1;queue=[I];seen[I]=true
        while !isempty(queue)
            q=pop!(queue)
            for step in (CartesianIndex(1,0),CartesianIndex(-1,0),CartesianIndex(0,1),CartesianIndex(0,-1))
                j=q+step
                if checkbounds(Bool,mask,j)&&mask[j]&&!seen[j];seen[j]=true;push!(queue,j);end
            end
        end
    end
    n
end
function metrics(s,grid,pts,initial)
    n=length(grid);J=reshape([p.gross-R.cost(s.model,p.z) for p in pts],n,n)
    best=argmax(J);search=vec([grid[I[1]]<=.75+1e-12&&grid[I[2]]<=.75+1e-12 for I in CartesianIndices(J)])
    inds=findall(search);ranked=sort(inds;by=i->J[i]);worst=first(ranked);med=ranked[cld(length(ranked),2)]
    rsearch=maximum(J[inds])-minimum(J[inds]);threshold=maximum(J)-.15*rsearch
    top=J.>=threshold;localmax=0
    for I in CartesianIndices(J)
        ns=[J[K] for K in CartesianIndices(J) if max(abs(K[1]-I[1]),abs(K[2]-I[2]))==1]
        all(J[I]>=v-1e-12 for v in ns)&&any(J[I]>v+1e-12 for v in ns)&&(localmax+=1)
    end
    vbest=pts[LinearIndices(J)[best]].rewards.-R.cost(s.model,pts[LinearIndices(J)[best]].z)
    function contrast(p)
        delta=vbest.-(p.rewards.-R.cost(s.model,p.z));se=std(delta)/sqrt(length(delta))
        (mean(delta),se,safe_snr(mean(delta),se))
    end
    ini=contrast(initial);low=contrast(pts[worst]);mid=contrast(pts[med]);zbest=pts[LinearIndices(J)[best]].z
    sensitivity=mean(abs.([vec(diff(J;dims=1));vec(diff(J;dims=2))]))/(grid[2]-grid[1])
    (scenario=s.id,R_global=maximum(J)-minimum(J),R_search=rsearch,R_initial=ini[1],
        best_minus_search_median=maximum(J)-median(J[inds]),best_z3=zbest[1],best_z4=zbest[2],best_value=maximum(J),
        initial_value=initial.gross-R.cost(s.model,C.Z0),best_policy=pts[LinearIndices(J)[best]].policy,
        boundary_distance=minimum(vcat(zbest,1 .- zbest)),corner_best=all(x->x==0||x==1,zbest),
        competitive_fraction=mean(top),competitive_components=components(top),epsilon05_fraction=mean(J.>=maximum(J)-.5),
        local_maxima=localmax,mean_local_sensitivity=sensitivity,
        initial_delta_se=ini[2],initial_snr=ini[3],search_delta_se=low[2],search_snr=low[3],
        median_delta_se=mid[2],median_snr=mid[3],gross_range=maximum(p.gross for p in pts)-minimum(p.gross for p in pts),
        cost_range=sum(s.model.alpha))
end
function surface(d,s,stage)
    out=d["output"];path=joinpath(out,"checkpoints/surface_$(stage)_$(s.id).jls")
    if isfile(path);return deserialize(path);end
    n=d[stage*"_grid"];np=d[stage*"_paths"];policies=C.build_policy_grid()
    rngseed=seed("shared",stage,0,"surface");prim=R.primitive_bank(s.model,np,rngseed;njobs=d["jobs"])
    grid=collect(range(0.,1.;length=n));pts=NamedTuple[]
    for x in grid,y in grid;push!(pts,point(d,s,stage,[x,y],prim,policies));end
    initial=point(d,s,stage,C.Z0,prim,policies)
    met=metrics(s,grid,pts,initial)
    ans=(scenario=s.id,model=s.model,stage=stage,grid=grid,points=pts,initial=initial,metrics=met,seed=rngseed,n_paths=np,fingerprint=d["fingerprint"])
    E.write_csv(joinpath(out,stage,"$(s.id)_surface.csv"),[(scenario=s.id,z3=p.z[1],z4=p.z[2],objective=p.gross-R.cost(s.model,p.z),
        gross=p.gross,cost=R.cost(s.model,p.z),policy=p.policy,conditional_se=p.se,n_paths=np,seed=rngseed) for p in pts])
    E.atomic_serialize(path,ans);println("$stage $(s.id) complete R_search=",round(met.R_search;digits=3)," best=",(met.best_z3,met.best_z4));flush(stdout);ans
end
function surfaces(d,ss,stage)
    # Parallelize physical models; process identical physical configurations sequentially for cache safety.
    groups=Dict{String,Vector{Any}}()
    for s in ss;push!(get!(groups,physical_digest(s.model),Any[]),s);end
    Threads.@threads :dynamic for group in collect(values(groups))
        for s in group;surface(d,s,stage);end
    end
    results=[surface(d,s,stage) for s in ss]
    E.write_csv(joinpath(d["output"],stage,"metrics.csv"),[a.metrics for a in results]);results
end
function choose(results,nselect)
    # Select using landscapes only, before any algorithm results exist.
    ms=[a.metrics for a in results];N=length(ms)
    ordinal(v)=[count(x->x<=y,v)/length(v) for y in v]
    score=.35 .*ordinal([m.R_search for m in ms])+.25 .*ordinal([m.R_initial for m in ms])+
          .20 .*ordinal([m.best_minus_search_median for m in ms])+.20 .*ordinal([m.mean_local_sensitivity for m in ms])
    for i in 1:N;ms[i].corner_best&&(score[i]*=.65);end
    vectors=Vector{Float64}[]
    for a in results
        vals=[p.gross-R.cost(a.model,p.z) for p in a.points]
        shape=(vals.-minimum(vals))./max(maximum(vals)-minimum(vals),1e-12)
        push!(vectors,vcat(shape/sqrt(length(shape)),[a.metrics.best_z3,a.metrics.best_z4,a.metrics.competitive_fraction]))
    end
    original=findfirst(a->a.scenario=="L000",results);chosen=[original];reasons=["original reference"]
    # First seek a contrasted noncorner landscape; then localized and diverse ones.
    while length(chosen)<nselect
        available=setdiff(1:N,chosen)
        if length(chosen)==1
            eligible=filter(i->!ms[i].corner_best&&ms[i].R_initial>0,available)
            isempty(eligible)&&(eligible=available)
            i=eligible[argmax([score[j] for j in eligible])];reason="largest relevant-contrast composite among noncorner candidates"
        elseif length(chosen)==2
            eligible=filter(i->!ms[i].corner_best&&ms[i].R_search>=median([m.R_search for m in ms]),available)
            isempty(eligible)&&(eligible=available)
            values=[.6score[j]+.4*(1-ms[j].competitive_fraction) for j in eligible]
            i=eligible[argmax(values)];reason="localized competitive region with appreciable search contrast"
        else
            distance=[minimum(norm(vectors[j]-vectors[k]) for k in chosen) for j in available]
            distmax=max(maximum(distance),1e-12)
            i=available[argmax([.45score[j]+.55distance[l]/distmax for (l,j) in enumerate(available)])]
            reason="shape diversity plus relevant contrast"
        end
        push!(chosen,i);push!(reasons,reason)
    end
    [(scenario=results[i].scenario,order=k,reason=reasons[k],score=score[i],source_stage=results[i].stage) for (k,i) in enumerate(chosen)]
end
function freeze_selection(d,name,rows)
    path=joinpath(d["output"],"configs/$(name).jls")
    if isfile(path);return deserialize(path);end
    E.atomic_serialize(path,rows);E.write_csv(joinpath(d["output"],"configs/$(name).csv"),rows)
    open(joinpath(d["output"],"configs/$(name)_provenance.toml"),"w") do io
        TOML.print(io,Dict("frozen_at"=>string(Dates.now()),"fingerprint"=>d["fingerprint"],
            "basis"=>"direct-simulation landscape metrics and shape diversity; no algorithm outcomes",
            "source_metrics_sha256"=>sha(joinpath(d["output"],rows[1].source_stage,"metrics.csv"))))
    end
    rows
end
function fixed_data(z,bank,Y,cfg)
    ll,S=R.bank_scores(bank,z,[0.,0.],cfg);L=exp.(clamp.(ll,-35.,35.));m=size(Y,1)
    values=[mean(view(Y,i,:).*L) for i in 1:m];best=argmax(values)
    hard=[mean(view(Y,best,:).*L.*view(S,j,:))-cfg.alpha[j] for j in 1:2]
    (values=values,best=best,hard=hard,ess=sum(L)^2/sum(abs2,L),banks=1,lr_clips=count(x->abs(x)>35,ll))
end
function search(d,s,rep)
    path=joinpath(d["output"],"checkpoints/search_$(s.id)_$(rep).jls")
    isfile(path)&&return deserialize(path)
    cfg=s.model;policies=C.build_policy_grid();n=d["n_paths"];K=d["max_iters"]
    states=[(method=m,zhist=[copy(C.Z0)],diagnostics=NamedTuple[],history=R.HistoricalBank[]) for m in METHODS]
    primfix=R.primitive_bank(cfg,n,seed(s.id,"compare",rep,"fixed");njobs=d["jobs"])
    fixedbank=R.materialize_bank(primfix,[0.,0.],cfg);fixedY=R.rewards(policies,fixedbank,cfg)
    seeds=NamedTuple[]
    push!(seeds,(scenario=s.id,rep=rep,stream="fixed",iteration=0,seed=seed(s.id,"compare",rep,"fixed")))
    for k in 1:K
        ks=seed(s.id,"compare",rep,"search",k);bs=seed(s.id,"compare",rep,"bootstrap",k)
        push!(seeds,(scenario=s.id,rep=rep,stream="search",iteration=k,seed=ks))
        push!(seeds,(scenario=s.id,rep=rep,stream="bootstrap",iteration=k,seed=bs))
        primitives=R.primitive_bank(cfg,n,ks;njobs=d["jobs"]);payloads=Dict{Tuple{Float64,Float64},Any}()
        for st in states
            m=st.method;z=st.zhist[end]
            if m.kind==:fixed
                data=fixed_data(z,fixedbank,fixedY,cfg)
            else
                payload=get!(payloads,Tuple(z)) do
                    bank=R.materialize_bank(primitives,z,cfg);Y=R.rewards(policies,bank,cfg)
                    (bank=bank,Y=Y,data=R.local_data(z,bank,Y,cfg))
                end
                if m.kind in (:green,:green_softmax)
                    push!(st.history,R.historical_bank(s.id,cfg,z,payload.bank,payload.Y))
                    data=R.green_data(z,st.history,cfg,s.id)
                else;data=payload.data;end
            end
            if m.kind in (:fixed,:fresh,:green)
                g=data.hard;w=zeros(144);w[data.best]=1.
            elseif m.kind==:psoft
                w=R.bootstrap_weights(payload.Y,d["bootstrap_reps"],MersenneTwister(bs));g=R.combine(data.G,w,cfg)
            else
                w=R.softmax_weights(data.values;beta=m.beta);g=R.combine(data.G,w,cfg)
            end
            all(isfinite,g)||error("Nonfinite gradient in $(s.id)/$(m.id)")
            znew,eta,gc=R.projected_step(z,g,k)
            push!(st.diagnostics,(iteration=k,g3=g[1],g4=g[2],step_norm=norm(znew-z),gradient_norm=norm(g),
                clipping=count(x->abs(x)>C.GRAD_CLIP,g),projection=norm(znew-(z+eta*gc)),
                ess=data.ess,lr_clips=data.lr_clips,hard_policy=data.best,K_eff=1/sum(abs2,w),
                objective_internal=maximum(data.values)-R.cost(cfg,z)))
            push!(st.zhist,znew)
        end
        (k==1||k%25==0||k==K)&&println("search $(s.id) rep=$rep iteration=$k/$K");flush(stdout)
    end
    result=(scenario=s.id,rep=rep,runs=[(method=st.method,zhist=st.zhist,diagnostics=st.diagnostics) for st in states],seeds=seeds)
    E.atomic_serialize(path,result);result
end
function evaluate_repetition(d,s,rep,search_result)
    path=joinpath(d["output"],"checkpoints/evaluation_$(s.id)_$(rep).jls")
    isfile(path)&&return deserialize(path)
    cfg=s.model;policies=C.build_policy_grid();out=d["output"]
    dir=joinpath(out,"comparisons",s.id);mkpath(dir);stem=joinpath(dir,@sprintf("rep_%02d",rep))
    # Separate banks per repetition; methods share CRN within a repetition.
    trace_seed=seed(s.id,"compare",rep,"trace");selection_seed=seed(s.id,"compare",rep,"selection")
    hold_seed=seed(s.id,"compare",rep,"holdout")
    trace=R.primitive_bank(cfg,d["trace_paths"],trace_seed;njobs=d["jobs"])
    select=R.primitive_bank(cfg,d["selection_paths"],selection_seed;njobs=d["jobs"])
    hold=R.primitive_bank(cfg,d["final_paths"],hold_seed;njobs=d["jobs"])
    cache=Dict{Tuple{Float64,Float64},Any}();traces=NamedTuple[];coords=NamedTuple[];diagnostics=NamedTuple[]
    for run in search_result.runs
        partial=joinpath(out,"checkpoints/trace_$(s.id)_$(rep)_$(run.method.id).jls")
        done=isfile(partial) ? deserialize(partial) : NamedTuple[]
        for k in length(done):d["max_iters"]
            z=run.zhist[k+1]
            v=get!(cache,Tuple(z)) do
                Y=R.rewards(policies,R.materialize_bank(trace,z,cfg),cfg)
                maximum(vec(mean(Y;dims=2)))-R.cost(cfg,z)
            end
            push!(done,(scenario=s.id,rep=rep,method=run.method.id,iteration=k,z3=z[1],z4=z[2],objective=v,
                trace_paths=d["trace_paths"],seed=trace_seed))
            (k%10==0||k==d["max_iters"])&&E.atomic_serialize(partial,done)
        end
        for row in done;cache[(row.z3,row.z4)]=row.objective;end
        append!(traces,done)
        for (k,z) in enumerate(run.zhist);push!(coords,(scenario=s.id,rep=rep,method=run.method.id,iteration=k-1,z3=z[1],z4=z[2]));end
        for row in run.diagnostics;push!(diagnostics,merge((scenario=s.id,rep=rep,method=run.method.id),row));end
    end
    surf=deserialize(joinpath(out,"checkpoints/surface_high_$(s.id).jls"));m=surf.metrics
    targets=[(name=run.method.id,z=run.zhist[end]) for run in search_result.runs]
    append!(targets,[(name="GridBest",z=[m.best_z3,m.best_z4]),(name="Initial",z=copy(C.Z0))])
    # Audit nearby regions with independent high precision; no further optimization.
    for (j,delta) in enumerate(([.1,0.],[-.1,0.],[0.,.1],[0.,-.1]))
        push!(targets,(name="GridNeighbor$j",z=clamp.([m.best_z3,m.best_z4]+delta,0.,1.)))
    end
    finals=NamedTuple[];vectors=Dict{String,Vector{Float64}}();evalcache=Dict{Tuple{Float64,Float64},Any}()
    for t in targets
        v=get!(evalcache,Tuple(t.z)) do
            Ys=R.rewards(policies,R.materialize_bank(select,t.z,cfg),cfg);chosen=argmax(vec(mean(Ys;dims=2)))
            Ye=R.rewards(policies,R.materialize_bank(hold,t.z,cfg),cfg);means=vec(mean(Ye;dims=2));direct=argmax(means)
            (chosen=chosen,direct=direct,values=collect(view(Ye,chosen,:)),direct_gross=means[direct])
        end
        cost=R.cost(cfg,t.z);vectors[t.name]=v.values.-cost
        push!(finals,(scenario=s.id,rep=rep,method=t.name,z3=t.z[1],z4=t.z[2],selected_policy=v.chosen,
            gross=mean(v.values),probing_cost=cost,objective_holdout=mean(v.values)-cost,
            evaluation_se=std(v.values)/sqrt(length(v.values)),objective_direct=v.direct_gross-cost,direct_policy=v.direct,
            selection_paths=d["selection_paths"],holdout_paths=d["final_paths"],selection_seed=selection_seed,holdout_seed=hold_seed))
    end
    pairs=NamedTuple[]
    for a in [m.id for m in METHODS],b in [m.id for m in METHODS if m.id!=a]
        delta=vectors[a]-vectors[b];se=std(delta)/sqrt(length(delta))
        push!(pairs,(scenario=s.id,rep=rep,method_a=a,method_b=b,difference=mean(delta),paired_se=se,snr=safe_snr(mean(delta),se)))
    end
    gaps=NamedTuple[]
    for t in targets
        delta=vectors["GridBest"]-vectors[t.name];se=std(delta)/sqrt(length(delta))
        push!(gaps,(scenario=s.id,rep=rep,method=t.name,grid_reference_gap=mean(delta),paired_se=se,
            snr=safe_snr(mean(delta),se)))
    end
    E.write_csv(stem*"_traces.csv",traces);E.write_csv(stem*"_coordinates.csv",coords)
    E.write_csv(stem*"_diagnostics.csv",diagnostics);E.write_csv(stem*"_finals.csv",finals)
    E.write_csv(stem*"_paired.csv",pairs);E.write_csv(stem*"_grid_gaps.csv",gaps)
    E.write_csv(stem*"_seeds.csv",[search_result.seeds;
        [(scenario=s.id,rep=rep,stream=stream,iteration=0,seed=seed(s.id,"compare",rep,stream)) for stream in ("trace","selection","holdout")]])
    result=(traces=traces,coordinates=coords,diagnostics=diagnostics,finals=finals,paired=pairs,gaps=gaps)
    E.atomic_serialize(path,result);println("evaluation $(s.id) rep=$rep complete");flush(stdout);result
end
function compare(d,ss)
    targets=deserialize(joinpath(d["output"],"configs/selected_instances.jls"));ids=[t.scenario for t in targets]
    selected=filter(s->s.id in ids,ss);surfaces(d,selected,"high")
    tasks=[(s,r) for s in selected for r in 1:d["reps"]]
    Threads.@threads :dynamic for (s,r) in tasks
        search_result=search(d,s,r);evaluate_repetition(d,s,r,search_result)
    end
    open(joinpath(d["output"],"COMPUTATION_COMPLETE.toml"),"w") do io
        TOML.print(io,Dict("complete"=>true,"quick"=>d["quick"],"scenarios"=>ids,"reps"=>d["reps"],"fingerprint"=>d["fingerprint"]))
    end
end
function main(args=ARGS)
    BLAS.set_num_threads(1);d=arguments(args);d===nothing&&return
    ss=initialize(d)
    if "screen" in d["actions"];surfaces(d,ss,"screen");end
    if "refine" in d["actions"]
        all(isfile(joinpath(d["output"],"checkpoints/surface_screen_$(s.id).jls")) for s in ss)||error("Run --screen first")
        screen_results=[deserialize(joinpath(d["output"],"checkpoints/surface_screen_$(s.id).jls")) for s in ss]
        selection=freeze_selection(d,"refinement_selection",choose(screen_results,d["refine_count"]))
        refined=surfaces(d,filter(s->s.id in [a.scenario for a in selection],ss),"refine")
        freeze_selection(d,"selected_instances",choose(refined,d["selected_count"]))
    end
    "compare" in d["actions"]&&compare(d,ss)
    if "plots" in d["actions"]
        run(`python3 $(joinpath(ROOT,"analyze_landscape_010.py")) $(d["output"])`)
    end
    all(sha(joinpath(ROOT,f))==v for (f,v) in d["science"]["sources"])||error("Source changed during execution")
    println("010 completed actions=",d["actions"]," output=",d["output"])
end
end
if abspath(PROGRAM_FILE)==@__FILE__;ProbingLandscapeSearch.main();end
