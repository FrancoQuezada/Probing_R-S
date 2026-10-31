# New experiment only. Reuse the exact 010 model/estimators/searches unchanged.
# Load at Main so archived ModelConfig objects from 010 retain their module path.
if !isdefined(@__MODULE__, :ProbingLandscapeSearch)
    include(joinpath(@__DIR__,"010probing_instance_landscape_search.jl"))
end
module ProbingHighContrastInterior
using ..ProbingLandscapeSearch
using Random, Statistics, LinearAlgebra, Serialization, SHA, TOML, Printf, Dates
const X=ProbingLandscapeSearch
const R=X.R
const E=X.E
const C=X.C
const ROOT=@__DIR__
const CONTROL_IDS=["L000","L025","L026"]
const ANCHORS=[[.30,.30],[.40,.50],[.50,.50],[.60,.65],[.70,.70],[.75,.80]]
const SCALES=[(1.,1.),(.8,.8),(1.2,1.2),(.8,1.2),(1.2,.8)]
const FILES=["011probing_high_contrast_interior.jl","probing_interior_011_diagnostics.jl",
    "PROTOCOLO_011.txt","010probing_instance_landscape_search.jl",
    "probing_regime_model.jl","probing_experiments_common.jl","003probing_four_searches_softmax.jl",
    "002probing_three_searches_n500_full45iters.jl",
    "009probing_parameter_regimes.jl","009b_probing_regime_validation.jl"]
sha(p)=bytes2hex(sha256(read(p)))
seed(s,phase,rep,stream,k=0)=Int(parse(UInt64,bytes2hex(sha256("Probing011-v1/$s/$phase/$rep/$stream/$k"))[1:15];base=16))+1
function arguments(args)
    d=Dict{String,Any}("output"=>joinpath(ROOT,"resultados/011probing_high_contrast_interior"),
        "reference"=>joinpath(ROOT,"resultados/010probing_instance_landscape_search"),
        "quick"=>false,"resume"=>false,"actions"=>String[],"ops"=>48,"calibration_paths"=>5000,
        "screen_grid"=>5,"screen_paths"=>500,"refine_count"=>10,"refine_grid"=>11,"refine_paths"=>3000,
        "selected_count"=>5,"high_grid"=>21,"high_paths"=>5000,"n_paths"=>500,"jobs"=>45,"max_iters"=>100,
        "reps"=>5,"trace_paths"=>1000,"selection_paths"=>10000,"final_paths"=>50000,"bootstrap_reps"=>300,
        "diagnostic_reps"=>30,"audit_paths"=>50000,"slope_snr_min"=>2.5,"interior_min"=>.10)
    supplied=Set{String}();i=1
    while i<=length(args)
        a=args[i]
        if a=="--quick-test";d["quick"]=true
        elseif a=="--resume";d["resume"]=true
        elseif a in ("--calibrate","--screen","--refine","--compare","--plots");push!(d["actions"],a[3:end])
        elseif a=="--help"
            println("011 [--quick-test] [--calibrate] [--screen] [--refine] [--compare] [--plots] [--resume] [--output DIR] [--reference DIR] [--ops 48] [--calibration-paths 5000] [--screen-paths 500] [--refine-paths 3000] [--high-paths 5000] [--diagnostic-reps 30] [--audit-paths 50000] [--n-paths 500] [--max-iters 100] [--reps 5] [--trace-paths 1000] [--selection-paths 10000] [--final-paths 50000]");return nothing
        else
            key=replace(a[3:end],'-'=>'_');haskey(d,key)||error("Unknown option $a")
            i+=1;i<=length(args)||error("Missing value for $a");push!(supplied,key)
            d[key]=key in ("output","reference") ? abspath(args[i]) : key in ("slope_snr_min","interior_min") ? parse(Float64,args[i]) : parse(Int,args[i])
        end
        i+=1
    end
    isempty(d["actions"])&&(d["actions"]=["calibrate","screen","refine","compare","plots"])
    if d["quick"]
        for (k,v) in ("ops"=>4,"calibration_paths"=>40,"screen_grid"=>3,"screen_paths"=>20,"refine_count"=>5,
            "refine_grid"=>3,"refine_paths"=>30,"high_grid"=>3,"high_paths"=>40,"n_paths"=>20,"max_iters"=>4,
            "reps"=>2,"trace_paths"=>30,"selection_paths"=>40,"final_paths"=>80,"diagnostic_reps"=>3,"audit_paths"=>80)
            k in supplied||(d[k]=v)
        end
        "output" in supplied||(d["output"]=joinpath(d["output"],"quick_test"))
    end
    for (k,v) in d
        v isa Int && !(v isa Bool) && v<2 && error("$k must be >=2")
    end
    3<=d["selected_count"]<=5&&d["selected_count"]<=d["refine_count"]||error("Select 3 to 5 instances")
    4<=d["ops"]<=216||error("Invalid operational design size")
    sources=Dict(f=>sha(joinpath(ROOT,f)) for f in FILES)
    science=Dict{String,Any}(k=>v for (k,v) in d if !(k in ("output","reference","actions","resume")))
    science["sources"]=sources;science["reference_models_sha256"]=sha(joinpath(d["reference"],"configs/models.toml"))
    science["Z0"]=C.Z0;science["BASE_STEP"]=C.BASE_STEP;science["GRAD_CLIP"]=C.GRAD_CLIP
    science["julia_version"]=string(VERSION);science["anchors"]=ANCHORS;science["alpha_scales"]=[collect(v) for v in SCALES]
    d["science"]=science;d["fingerprint"]=R.digest(science);d
end
function from_dict(m)
    R.validate(R.ModelConfig(p_complex=Tuple(m["P_COMPLEX"]),p_hit0=Tuple(m["P_HIT0"]),p_false0=Tuple(m["P_FALSE0"]),
        beta=m["BETA"],mean_interarrival=m["MEAN_INTERARRIVAL"],fast_mean=Tuple(m["FAST_MEAN"]),std_mean=Tuple(m["STD_MEAN"]),
        revenue_scale=m["REVENUE_SCALE"],fast_routing_cost=m["FAST_ROUTING_COST"],delay_penalty=m["DELAY_PENALTY"],alpha=Tuple(m["ALPHA"])))
end
function changed(cfg;kwargs...)
    fields=NamedTuple{fieldnames(R.ModelConfig)}(Tuple(getfield(cfg,k) for k in fieldnames(R.ModelConfig)))
    R.validate(R.ModelConfig(;merge(fields,(;kwargs...))...))
end
function controls(d)
    models=TOML.parsefile(joinpath(d["reference"],"configs/models.toml"))
    [(id=id,op=id,model=from_dict(models[id]),origin="reference010",calibrated=false,anchor=[.5,.5],scale=[1.,1.]) for id in CONTROL_IDS]
end
function operational_design(d)
    base=controls(d)[2].model;levels=([.84,1.,1.2,1.5,1.8,2.2],[10.,12.,14.],[1.8,2.4],[2.2,3.,4.],[0,1])
    original=(1,2,1,3,1);codes=[original];origins=["L025 physics"]
    for j in 1:5,lev in eachindex(levels[j])
        c=ntuple(k->k==j ? lev : original[k],5)
        c in codes&&continue;push!(codes,c);push!(origins,"one_factor_$j")
    end
    # Balanced discrete fractional blocks within the stated local neighborhood.
    rng=MersenneTwister(seed("design","operational",0,"balanced"))
    while length(codes)<d["ops"]
        n=48;columns=[shuffle(rng,[mod(i-1,length(lev))+1 for i in 1:n]) for lev in levels]
        for i in 1:n
            c=ntuple(j->columns[j][i],5);c in codes&&continue
            push!(codes,c);push!(origins,"balanced_fractional");length(codes)>=d["ops"]&&break
        end
    end
    result=NamedTuple[]
    for (i,c) in enumerate(codes[1:d["ops"]])
        delay,std,fast,beta,sig=(levels[j][c[j]] for j in 1:5)
        cfg=changed(base;delay_penalty=delay,std_mean=(1.,std),fast_mean=(.75,fast),beta=beta,
            p_hit0=sig==0 ? base.p_hit0 : (.60,.65,.55,.65),p_false0=sig==0 ? base.p_false0 : (.05,.08,.20,.25))
        push!(result,(id=@sprintf("O%03d",i),model=cfg,origin=origins[i],code=c))
    end
    result
end
function initialize(d)
    out=d["output"]
    for f in ("configs","calibration","screen","refine","high","diagnostics","audits","checkpoints","comparisons","tables","figures","logs","validation","reproduction")
        mkpath(joinpath(out,f))
    end
    path=joinpath(out,"configs/experiment.toml")
    if isfile(path)
        TOML.parsefile(path)["fingerprint"]==d["fingerprint"]||error("Scientific source/budget mismatch: choose a new --output")
        d["resume"]||d["actions"]==["plots"]||error("Existing experiment: use --resume")
    else
        open(path,"w") do io;TOML.print(io,Dict("fingerprint"=>d["fingerprint"],"science"=>d["science"],"threads"=>Threads.nthreads());sorted=true);end
        ops=operational_design(d);E.atomic_serialize(joinpath(out,"configs/operational_design.jls"),ops)
        E.write_csv(joinpath(out,"configs/operational_design.csv"),[(op=s.id,origin=s.origin,delay=s.model.delay_penalty,std_complex=s.model.std_mean[2],
            fast_complex=s.model.fast_mean[2],beta_signal=s.model.beta,signal=s.code[5]-1,mean_interarrival=s.model.mean_interarrival,
            revenue_scale=s.model.revenue_scale,physical_sha256=X.physical_digest(s.model)) for s in ops])
        open(joinpath(out,"configs/operational_models.toml"),"w") do io;TOML.print(io,Dict(s.id=>R.model_dict(s.model) for s in ops);sorted=true);end
        E.write_csv(joinpath(out,"configs/policies.csv"),[(policy=i,av=p.av,aq=p.aq,as=p.as,tau=p.tau) for (i,p) in enumerate(C.build_policy_grid())])
        for f in [FILES;"analyze_interior_011.py";"analyze_landscape_010.py"];cp(joinpath(ROOT,f),joinpath(out,"reproduction",f));end
        env=joinpath(ROOT,"resultados/003probing_softmax_20261021/entorno")
        isfile(joinpath(env,"Project.toml"))||(env=ROOT)
        for f in ("Project.toml","Manifest.toml");cp(joinpath(env,f),joinpath(out,"reproduction",f));end
        cp(joinpath(d["reference"],"configs/models.toml"),joinpath(out,"configs/reference010_models.toml"))
    end
    open(joinpath(out,"logs/invocations.log"),"a") do io;println(io,Dates.now()," ",repr(ARGS)," threads=",Threads.nthreads());end
    deserialize(joinpath(out,"configs/operational_design.jls"))
end
function gross_point(d,op,phase,z,prim,policies)
    key=bytes2hex(sha256("$(X.physical_digest(op.model))/$phase/$(repr(z))/$(length(prim))/$(d["jobs"])"))
    path=joinpath(d["output"],"checkpoints/gross_$key.jls")
    isfile(path)&&return deserialize(path)
    Y=R.rewards(policies,R.materialize_bank(prim,z,op.model),op.model)
    means=vec(mean(Y;dims=2));policy=argmax(means);v=collect(view(Y,policy,:))
    result=(z=copy(z),gross=means[policy],policy=policy,rewards=v,se=std(v)/sqrt(length(v)),means=means)
    E.atomic_serialize(path,result);result
end
function calibrate_op(d,op)
    path=joinpath(d["output"],"checkpoints/calibration_$(op.id).jls")
    isfile(path)&&return deserialize(path)
    prim=R.primitive_bank(op.model,d["calibration_paths"],seed("shared","calibration",0,"primitives");njobs=d["jobs"])
    policies=C.build_policy_grid();rows=NamedTuple[]
    for (i,z) in enumerate(ANCHORS)
        center=gross_point(d,op,"calibration",z,prim,policies);fds=Dict{Tuple{Int,Float64},Vector{Float64}}();points=Dict{Tuple{Int,Float64,Int},Any}()
        for j in 1:2,h in (.05,.10)
            zp=copy(z);zm=copy(z);zp[j]+=h;zm[j]-=h
            plus=gross_point(d,op,"calibration",zp,prim,policies);minus=gross_point(d,op,"calibration",zm,prim,policies)
            points[(j,h,1)]=plus;points[(j,h,-1)]=minus;fds[(j,h)]=(plus.rewards-minus.rewards)/(2h)
        end
        g=[mean(fds[(j,.05)]) for j in 1:2];se=[std(fds[(j,.05)])/sqrt(length(prim)) for j in 1:2]
        g10=[mean(fds[(j,.10)]) for j in 1:2];se10=[std(fds[(j,.10)])/sqrt(length(prim)) for j in 1:2]
        sensitivity=[mean(fds[(j,.05)]-fds[(j,.10)]) for j in 1:2]
        sensitivity_se=[std(fds[(j,.05)]-fds[(j,.10)])/sqrt(length(prim)) for j in 1:2]
        precise=all(g[j]>d["slope_snr_min"]*se[j]&&g10[j]>1.96se10[j]&&
            abs(sensitivity[j])<=2sensitivity_se[j]+.25max(abs(g[j]),1.) for j in 1:2)
        curv=[(points[(j,.05,1)].rewards-2center.rewards+points[(j,.05,-1)].rewards)/.05^2 for j in 1:2]
        switches=any(points[(j,h,k)].policy!=center.policy for j in 1:2 for h in (.05,.10) for k in (-1,1))
        push!(rows,(op=op.id,anchor=i,z3=z[1],z4=z[2],slope3=g[1],slope4=g[2],se3=se[1],se4=se[2],
            slope3_h10=g10[1],slope4_h10=g10[2],se3_h10=se10[1],se4_h10=se10[2],
            step_difference3=sensitivity[1],step_difference4=sensitivity[2],step_difference_se3=sensitivity_se[1],step_difference_se4=sensitivity_se[2],
            curvature3=mean(curv[1]),curvature4=mean(curv[2]),curvature_se3=std(curv[1])/sqrt(length(prim)),curvature_se4=std(curv[2])/sqrt(length(prim)),
            policy_switch=switches,center_policy=center.policy,usable=precise,n_paths=length(prim),seed=seed("shared","calibration",0,"primitives")))
    end
    E.write_csv(joinpath(d["output"],"calibration/$(op.id)_slopes.csv"),rows);E.atomic_serialize(path,rows)
    println("calibration $(op.id): ",count(r->r.usable,rows),"/6 precise positive anchors");flush(stdout);rows
end
function calibrate(d,ops)
    Threads.@threads :dynamic for op in ops;calibrate_op(d,op);end
    allrows=vcat([calibrate_op(d,op) for op in ops]...);E.write_csv(joinpath(d["output"],"calibration/slopes.csv"),allrows)
    path=joinpath(d["output"],"configs/scenarios.jls")
    isfile(path)&&return deserialize(path)
    scenarios=NamedTuple[controls(d)...];digests=Set(R.model_digest(s.model) for s in scenarios)
    function add(op,cfg,origin,calibrated,anchor,scale)
        digest=R.model_digest(cfg);digest in digests&&return
        id=@sprintf("I%04d",length(scenarios)-2)
        push!(scenarios,(id=id,op=op.id,model=cfg,origin=origin,calibrated=calibrated,anchor=copy(anchor),scale=collect(scale)));push!(digests,digest)
    end
    for op in ops
        for alpha in ((5.,2.25),(10.,4.5));add(op,changed(op.model;alpha=alpha),"fixed_alpha_control",false,[.5,.5],(1.,1.));end
        for r in filter(r->r.op==op.id&&r.usable,allrows),scale in SCALES
            alpha=(r.slope3*scale[1],r.slope4*scale[2]);all(x->isfinite(x)&&x>0,alpha)||error("Invalid calibrated cost")
            add(op,changed(op.model;alpha=alpha),"calibrated_anchor_$(r.anchor)",true,[r.z3,r.z4],scale)
        end
    end
    E.atomic_serialize(path,scenarios)
    E.write_csv(joinpath(d["output"],"configs/alpha_candidates.csv"),[(scenario=s.id,op=s.op,origin=s.origin,calibrated=s.calibrated,
        alpha3=s.model.alpha[1],alpha4=s.model.alpha[2],anchor3=s.anchor[1],anchor4=s.anchor[2],scale3=s.scale[1],scale4=s.scale[2],
        delay=s.model.delay_penalty,std_complex=s.model.std_mean[2],fast_complex=s.model.fast_mean[2],beta_signal=s.model.beta,
        signal=s.model.p_hit0[3]==.75 ? 0 : 1,model_sha256=R.model_digest(s.model)) for s in scenarios])
    open(joinpath(d["output"],"configs/models.toml"),"w") do io;TOML.print(io,Dict(s.id=>R.model_dict(s.model) for s in scenarios);sorted=true);end
    println("Frozen ",length(scenarios)," net configurations, including three references; no search outcomes used.");scenarios
end
function raw_surface(d,s,phase)
    physical=X.physical_digest(s.model);path=joinpath(d["output"],"checkpoints/physical_$(phase)_$physical.jls")
    isfile(path)&&return deserialize(path)
    n=d[phase*"_grid"];np=d[phase*"_paths"];reused=nothing
    if !d["quick"]&&phase in ("refine","high")&&d["jobs"]==45
        for control in controls(d)
            X.physical_digest(control.model)==physical||continue
            source=joinpath(d["reference"],"checkpoints/surface_$(phase)_$(control.id).jls")
            isfile(source)||continue;old=deserialize(source)
            if length(old.grid)==n&&old.n_paths==np
                @assert R.model_digest(old.model)==R.model_digest(control.model)
                reused=(grid=old.grid,points=old.points,initial=old.initial,seed=old.seed,n_paths=np,source="010/$(control.id)/$phase");break
            end
        end
    end
    if reused===nothing
        rngseed=seed("shared",phase,0,"surface");prim=R.primitive_bank(s.model,np,rngseed;njobs=d["jobs"])
        grid=collect(range(0.,1.;length=n));policies=C.build_policy_grid();points=NamedTuple[]
        for x in grid,y in grid;push!(points,gross_point(d,s,phase,[x,y],prim,policies));end
        initial=gross_point(d,s,phase,C.Z0,prim,policies)
        reused=(grid=grid,points=points,initial=initial,seed=rngseed,n_paths=np,source="011 direct CRN")
    end
    E.atomic_serialize(path,reused);reused
end
mean_or_nan(v)=isempty(v) ? NaN : mean(v)
function landscape_metrics(s,raw)
    pts=raw.points;grid=raw.grid;n=length(grid);values=[p.gross-R.cost(s.model,p.z) for p in pts];J=reshape(values,n,n)
    bi=argmax(values);best=pts[bi];z=best.z;Jmax=values[bi]
    search=findall(p->all(x->x<=.85+1e-12,p.z),pts);rsearch=maximum(values[search])-minimum(values[search])
    function pair(i)
        delta=best.rewards-pts[i].rewards .-(R.cost(s.model,z)-R.cost(s.model,pts[i].z))
        (difference=mean(delta),se=std(delta)/sqrt(length(delta)))
    end
    neighbors=[i for i in eachindex(pts) if isapprox(norm(pts[i].z-z),grid[2]-grid[1];atol=1e-9)]
    mcse=median([pair(i).se for i in neighbors]);epsmc=max(.05,2mcse)
    mask=J.>=Jmax-epsmc;indices=findall(values.>=Jmax-epsmc)
    positions=reduce(hcat,[pts[i].z for i in indices]);centroid=vec(mean(positions;dims=2))
    spatial=sqrt(mean(sum(abs2,pts[i].z-centroid) for i in indices))
    diameter=maximum(norm(pts[i].z-pts[j].z) for i in indices for j in indices)
    function axis_indices(distance)
        [i for i in eachindex(pts) if count(x->abs(x)>1e-9,pts[i].z-z)==1&&isapprox(norm(pts[i].z-z),distance;atol=1e-9)]
    end
    n10=axis_indices(.10);n20=axis_indices(.20)
    initialdelta=best.rewards-raw.initial.rewards .-(R.cost(s.model,z)-R.cost(s.model,C.Z0))
    initialse=std(initialdelta)/sqrt(length(initialdelta));worst=search[argmin(values[search])];worstpair=pair(worst)
    # Second differences are descriptive; report policy switching separately.
    curv=fill(NaN,2);curvse=fill(NaN,2);policy_switch=false;h=grid[2]-grid[1]
    for j in 1:2
        ip=findfirst(p->isapprox(p.z,z+[j==1 ? h : 0.,j==2 ? h : 0.];atol=1e-9),pts)
        im=findfirst(p->isapprox(p.z,z-[j==1 ? h : 0.,j==2 ? h : 0.];atol=1e-9),pts)
        if ip!==nothing&&im!==nothing
            v=(pts[ip].rewards-2best.rewards+pts[im].rewards)/h^2
            curv[j]=mean(v);curvse[j]=std(v)/sqrt(length(v))
            policy_switch|=pts[ip].policy!=best.policy||pts[im].policy!=best.policy
        end
    end
    localmaxima=0
    for I in CartesianIndices(J)
        adjacent=[J[K] for K in CartesianIndices(J) if maximum(abs.(Tuple(K-I)))==1]
        all(J[I]>=v-1e-12 for v in adjacent)&&any(J[I]>v+1e-12 for v in adjacent)&&(localmaxima+=1)
    end
    (scenario=s.id,op=s.op,calibrated=s.calibrated,R_global=maximum(values)-minimum(values),R_search=rsearch,
        R_initial=mean(initialdelta),best_minus_search_median=Jmax-median(values[search]),best_z3=z[1],best_z4=z[2],
        best_value=Jmax,initial_value=raw.initial.gross-R.cost(s.model,C.Z0),best_policy=best.policy,
        boundary_distance=minimum(vcat(z,1 .- z)),corner_best=all(x->x==0||x==1,z),
        epsilon05_fraction=mean(values.>=Jmax-.5),epsilon1_fraction=mean(values.>=Jmax-1.),epsilon2_fraction=mean(values.>=Jmax-2.),
        epsilon_mc=epsmc,epsilon_mc_fraction=mean(mask),competitive_components=X.components(mask),
        competitive_spatial_rms=spatial,competitive_diameter=diameter,relative_fraction=mean(values.>=Jmax-.1rsearch),
        local_loss_mean=mean([pair(i).difference for i in neighbors]),loss10_mean=mean_or_nan([pair(i).difference for i in n10]),
        loss20_mean=mean_or_nan([pair(i).difference for i in n20]),local_pair_se=mcse,
        initial_se=initialse,initial_snr=X.safe_snr(mean(initialdelta),initialse),search_se=worstpair.se,
        search_snr=X.safe_snr(worstpair.difference,worstpair.se),curvature3=curv[1],curvature4=curv[2],
        curvature_se3=curvse[1],curvature_se4=curvse[2],curvature_step=h,curvature_policy_switch=policy_switch,local_maxima=localmaxima,
        gross_range=maximum(p.gross for p in pts)-minimum(p.gross for p in pts),cost_range=sum(s.model.alpha),source=raw.source,n_paths=raw.n_paths)
end
function surface(d,s,phase;raw=nothing)
    path=joinpath(d["output"],"checkpoints/surface_$(phase)_$(s.id).jls")
    isfile(path)&&return deserialize(path)
    raw===nothing&&(raw=raw_surface(d,s,phase));metrics=landscape_metrics(s,raw)
    result=merge(raw,(scenario=s.id,model=s.model,stage=phase,metrics=metrics,fingerprint=d["fingerprint"]))
    E.write_csv(joinpath(d["output"],phase,"$(s.id)_surface.csv"),[(scenario=s.id,z3=p.z[1],z4=p.z[2],objective=p.gross-R.cost(s.model,p.z),
        gross=p.gross,cost=R.cost(s.model,p.z),policy=p.policy,conditional_se=p.se,n_paths=raw.n_paths,seed=raw.seed) for p in raw.points])
    E.atomic_serialize(path,result);result
end
function surfaces(d,scenarios,phase)
    groups=Dict{String,Vector{Any}}()
    for s in scenarios;push!(get!(groups,X.physical_digest(s.model),Any[]),s);end
    Threads.@threads :dynamic for group in collect(values(groups))
        raw=raw_surface(d,first(group),phase)
        for s in group;surface(d,s,phase;raw=raw);end
        println("$phase physical model: $(first(group).op), ",length(group)," net surfaces, source=",raw.source);flush(stdout)
    end
    results=[surface(d,s,phase) for s in scenarios]
    E.write_csv(joinpath(d["output"],phase,"metrics.csv"),[a.metrics for a in results]);results
end
function ranking(d,results;diagnostics=NamedTuple[])
    m=[a.metrics for a in results];ordinal(v)=[count(x->x<=y,v)/length(v) for y in v]
    scores=.30 .*ordinal([a.R_search for a in m])+.20 .*ordinal([a.R_initial for a in m])+
        .20 .*ordinal([1-a.epsilon_mc_fraction for a in m])+.20 .*ordinal([a.local_loss_mean for a in m])+.10 .*ordinal([a.boundary_distance for a in m])
    for (i,a) in enumerate(m)
        a.boundary_distance<d["interior_min"]-1e-9&&(scores[i]*=.4)
        if !isempty(diagnostics)
            q=filter(r->r.scenario==a.scenario&&r.method=="Fresh"&&r.point=="grid_best"&&r.iteration==20,diagnostics)
            if !isempty(q);scores[i]*=1-.35mean(r.clipped_fraction for r in q)-.25mean(r.projected for r in q);end
        end
    end
    scores
end
function pick(d,results,n;diagnostics=NamedTuple[])
    score=ranking(d,results;diagnostics=diagnostics);chosen=Int[findfirst(a->a.scenario==id,results)::Int for id in CONTROL_IDS]
    reasons=["mandatory original control","mandatory interior reference L025","mandatory stress/boundary control L026"]
    usedops=Set(a.metrics.op for a in results[chosen]);vectors=Vector{Float64}[]
    for a in results
        v=[p.gross-R.cost(a.model,p.z) for p in a.points];push!(vectors,(v.-minimum(v))./max(maximum(v)-minimum(v),1e-12))
    end
    while length(chosen)<n
        available=setdiff(collect(eachindex(results)),chosen)
        preferred=filter(i->results[i].metrics.boundary_distance>=d["interior_min"]-1e-9&&!(results[i].metrics.op in usedops),available)
        isempty(preferred)&&(preferred=filter(i->!(results[i].metrics.op in usedops),available))
        isempty(preferred)&&(preferred=available)
        distance=[minimum(norm(vectors[i]-vectors[j])/sqrt(length(vectors[i])) for j in chosen) for i in preferred]
        maxdist=max(maximum(distance),1e-12)
        utility=[.75score[i]+.25distance[k]/maxdist+.03results[i].metrics.calibrated for (k,i) in enumerate(preferred)]
        i=preferred[argmax(utility)];push!(chosen,i);push!(usedops,results[i].metrics.op)
        push!(reasons,"interiority, relevant contrast, localization, shape diversity"*(isempty(diagnostics) ? "" : ", preliminary saturation"))
    end
    [(scenario=results[i].scenario,order=k,reason=reasons[k],score=score[i],source_stage=results[i].stage) for (k,i) in enumerate(chosen)]
end
include(joinpath(@__DIR__,"probing_interior_011_diagnostics.jl"))
function refine(d,scenarios)
    all(isfile(joinpath(d["output"],"checkpoints/surface_screen_$(s.id).jls")) for s in scenarios)||error("Run --screen first")
    screen=[surface(d,s,"screen") for s in scenarios]
    selection=X.freeze_selection(d,"refinement_selection",pick(d,screen,d["refine_count"]))
    targets=filter(s->s.id in [a.scenario for a in selection],scenarios)
    refined=surfaces(d,targets,"refine");diagnostics=diagnostic_phase(d,targets)
    provisional=X.freeze_selection(d,"high_precision_candidates",pick(d,refined,d["selected_count"];diagnostics=diagnostics))
    high_targets=filter(s->s.id in [a.scenario for a in provisional],scenarios)
    high=surfaces(d,high_targets,"high")
    Threads.@threads :dynamic for s in high_targets;geometry_audit(d,s,surface(d,s,"high"));end
    # Up to two geometry-only fallbacks if a proposed new case loses interiority.
    for attempt in 1:2
        good=count(a->!(a.scenario in CONTROL_IDS)&&a.metrics.boundary_distance>=d["interior_min"]-1e-9,high)
        good>=d["selected_count"]-3&&break
        available=filter(a->!(a.scenario in [h.scenario for h in high]),refined);isempty(available)&&break
        scores=ranking(d,available;diagnostics=diagnostics);candidate=available[argmax(scores)]
        s=only(filter(s->s.id==candidate.scenario,scenarios));a=surface(d,s,"high");push!(high,a);geometry_audit(d,s,a)
    end
    E.write_csv(joinpath(d["output"],"high/all_candidate_metrics.csv"),[a.metrics for a in high])
    final=X.freeze_selection(d,"selected_instances",pick(d,high,d["selected_count"];diagnostics=diagnostics))
    E.write_csv(joinpath(d["output"],"high/metrics.csv"),[surface(d,only(filter(s->s.id==a.scenario,scenarios)),"high").metrics for a in final])
    mainpath=joinpath(d["output"],"configs/main_instance.toml")
    if !isfile(mainpath)
        proposed=filter(r->!(r.scenario in CONTROL_IDS),final)
        interior=filter(r->surface(d,only(filter(s->s.id==r.scenario,scenarios)),"high").metrics.boundary_distance>=d["interior_min"]-1e-9,proposed)
        primary=isempty(interior) ? (isempty(proposed) ? "L025" : first(proposed).scenario) : first(interior).scenario
        open(mainpath,"w") do io
            TOML.print(io,Dict("scenario"=>primary,"selected_before_algorithms"=>true,"interior_grid_candidate"=>!isempty(interior),
                "criterion"=>"high-grid interiority, relevant contrast/localization and presearch saturation; no optimization outcomes"))
        end
    end
    final
end
function reuse_reference_comparisons(d,scenarios)
    d["quick"]&&return
    oldcfg=TOML.parsefile(joinpath(d["reference"],"configs/experiment.toml"))["science"]
    keys=("n_paths","jobs","max_iters","trace_paths","selection_paths","final_paths","bootstrap_reps","BASE_STEP","GRAD_CLIP","Z0")
    all(oldcfg[k]==d["science"][k] for k in keys)||return
    rows=NamedTuple[]
    for s in filter(s->s.id in CONTROL_IDS,scenarios),rep in 1:d["reps"]
        source=joinpath(d["reference"],"checkpoints/evaluation_$(s.id)_$rep.jls")
        isfile(source)||continue
        @assert R.model_digest(s.model)==R.model_digest(only(filter(c->c.id==s.id,controls(d))).model)
        for kind in ("search","evaluation")
            src=joinpath(d["reference"],"checkpoints/$(kind)_$(s.id)_$rep.jls");dest=joinpath(d["output"],"checkpoints/$(kind)_$(s.id)_$rep.jls")
            isfile(dest)||cp(src,dest)
        end
        folder=joinpath(d["output"],"comparisons",s.id);mkpath(folder);prefix=@sprintf("rep_%02d_",rep)
        for file in readdir(joinpath(d["reference"],"comparisons",s.id))
            startswith(file,prefix)||continue;dest=joinpath(folder,file)
            isfile(dest)||cp(joinpath(d["reference"],"comparisons",s.id,file),dest)
        end
        push!(rows,(scenario=s.id,rep=rep,source="010 exact model/seed/budget reuse",methods=7))
    end
    isempty(rows)||E.write_csv(joinpath(d["output"],"configs/reused_comparisons.csv"),rows)
end
function compare(d,scenarios)
    isfile(joinpath(d["output"],"configs/selected_instances.jls"))||error("Run --refine first")
    selection=deserialize(joinpath(d["output"],"configs/selected_instances.jls"));selected=filter(s->s.id in [a.scenario for a in selection],scenarios)
    reuse_reference_comparisons(d,selected)
    # Exact unchanged 010 search, estimators, projected step and independent evaluator.
    X.compare(d,selected)
    postsearch_audits(d,selected)
end
function main(args=ARGS)
    BLAS.set_num_threads(1);d=arguments(args);d===nothing&&return
    ops=initialize(d)
    "calibrate" in d["actions"]&&calibrate(d,ops)
    scenarios=isfile(joinpath(d["output"],"configs/scenarios.jls")) ? deserialize(joinpath(d["output"],"configs/scenarios.jls")) : nothing
    for phase in ("screen","refine","compare")
        phase in d["actions"]||continue;scenarios===nothing&&error("Run --calibrate first")
        phase=="screen" ? surfaces(d,scenarios,"screen") : phase=="refine" ? refine(d,scenarios) : compare(d,scenarios)
    end
    if "plots" in d["actions"];run(`python3 $(joinpath(ROOT,"analyze_interior_011.py")) $(d["output"])`);end
    all(sha(joinpath(ROOT,f))==h for (f,h) in d["science"]["sources"])||error("Source changed during experiment")
    println("011 completed actions=",d["actions"]," output=",d["output"])
end
end
if abspath(PROGRAM_FILE)==@__FILE__;ProbingHighContrastInterior.main();end
