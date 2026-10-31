# Additional method only. The 010/011/012 scientific implementations are unchanged.
include(joinpath(@__DIR__,"012probing_anytime_budget_analysis.jl"))
module ProbingGreenSoftmax090
using ..ProbingAnytimeBudget
using Statistics, LinearAlgebra, Serialization, SHA, TOML, Printf, Dates
const A=ProbingAnytimeBudget
const H=A.H;const X=A.X;const R=A.R;const E=A.E;const C=A.C
const ID="I0252"
const NEW=(id="GreenSoftmax090",kind=:green_softmax,beta=.90)
const PREVIOUS=(id="GreenSoftmax010",kind=:green_softmax,beta=.10)
const METHODS=[X.METHODS;[PREVIOUS,NEW]]
const ROOT=@__DIR__
const SOURCE=joinpath(ROOT,"resultados/011probing_high_contrast_interior")
const DONOR=joinpath(ROOT,"resultados/012probing_anytime_budget")
const PRIOR=joinpath(ROOT,"resultados/013probing_green_softmax010")
sha(p)=bytes2hex(sha256(read(p)))

function initialize(args)
    quick="--quick-test" in args;plots="--plots-only" in args
    all(a in ("--quick-test","--plots-only","--resume") for a in args)||error("Unknown option")
    out=joinpath(ROOT,"resultados/014probing_green_softmax090",quick ? "quick_test" : "")
    d=Dict{String,Any}("output"=>out,"n_paths"=>500,"jobs"=>45,"max_iters"=>quick ? 6 : 100,
        "trace_paths"=>1000,"validation_paths"=>quick ? 100 : 5000,"bootstrap_reps"=>300,
        "quick"=>quick,"rep_ids"=>collect(1:(quick ? 2 : 5)))
    for p in ("configs","tables","checkpoints","validation","figures","logs","reproduction","source_011","comparisons/I0252");mkpath(joinpath(out,p));end
    source=TOML.parsefile(joinpath(SOURCE,"configs/experiment.toml"))["science"]
    @assert source["n_paths"]==500&&source["max_iters"]==100&&source["jobs"]==45&&source["reps"]==5
    @assert source["BASE_STEP"]==C.BASE_STEP==.035 && source["GRAD_CLIP"]==C.GRAD_CLIP==15.0 && source["Z0"]==C.Z0==[.12,.10]
    for (f,h) in source["sources"];@assert sha(joinpath(ROOT,f))==h "Original scientific source changed: $f";end
    cfg=H.from_dict(TOML.parsefile(joinpath(SOURCE,"configs/models.toml"))[ID])
    @assert R.model_digest(cfg)==TOML.parsefile(joinpath(DONOR,"configs/experiment.toml"))["science"]["reference_model_sha256"]
    @assert R.model_digest(cfg)==TOML.parsefile(joinpath(PRIOR,"configs/experiment.toml"))["science"]["model_sha256"]
    @assert length(C.build_policy_grid())==144
    science=Dict("scenario"=>ID,"quick"=>quick,"n_paths"=>500,"jobs"=>45,"max_iters"=>d["max_iters"],
        "trace_paths"=>1000,"validation_paths"=>d["validation_paths"],"bootstrap_reps"=>300,
        "new_method"=>NEW.id,"softmax_beta"=>.90,"methods"=>[m.id for m in METHODS],
        "model_sha256"=>R.model_digest(cfg),"source_011_sha256"=>sha(joinpath(SOURCE,"configs/experiment.toml")),
        "source_012_sha256"=>sha(joinpath(DONOR,"configs/experiment.toml")),
        "source_013_sha256"=>sha(joinpath(PRIOR,"configs/experiment.toml")),
        "driver_sha256"=>sha(@__FILE__),"runtime_sha256"=>sha(joinpath(ROOT,"probing_anytime_012_runtime.jl")),
        "rep_ids"=>d["rep_ids"],"selection_paths"=>10000,"holdout_paths"=>50000,
        "BASE_STEP"=>C.BASE_STEP,"GRAD_CLIP"=>C.GRAD_CLIP,"Z0"=>C.Z0,
        "objective"=>"max_i E_z[Y_i] - alpha'z; independent CRN hard-max trace evaluation")
    path=joinpath(out,"configs/experiment.toml");fingerprint=R.digest(science)
    if isfile(path);@assert TOML.parsefile(path)["fingerprint"]==fingerprint "Protocol changed: use a separate output";end
    open(path,"w") do io;TOML.print(io,Dict("science"=>science,"fingerprint"=>fingerprint);sorted=true);end
    for f in ("I0252.toml","reference.toml");cp(joinpath(DONOR,"configs",f),joinpath(out,"configs",f);force=true);end
    cp(joinpath(SOURCE,"high/I0252_surface.csv"),joinpath(out,"source_011/I0252_surface.csv");force=true)
    for rep in d["rep_ids"],suffix in ("coordinates","traces","diagnostics","finals","paired","grid_gaps","seeds")
        name=@sprintf("rep_%02d_%s.csv",rep,suffix)
        cp(joinpath(SOURCE,"comparisons",ID,name),joinpath(out,"source_011",name);force=true)
    end
    d,cfg,plots
end

# Point cache may be copied only with the same model, seed, bank size and phase.
function point(d,cfg,rep,phase,z,bank)
    key=bytes2hex(sha256(repr(z)));name="point_$(phase)_$(rep)_$key.jls"
    dst=joinpath(d["output"],"checkpoints",name)
    for folder in (PRIOR,DONOR)
        src=joinpath(folder,"checkpoints",name)
        if !isfile(dst)&&isfile(src)
            p=deserialize(src);@assert p.n_paths==length(bank)&&p.z==z
            cp(src,dst)
        end
    end
    A.direct_point(d,cfg,rep,phase,z,bank)
end

function previous_search(rep)
    old=deserialize(joinpath(SOURCE,"checkpoints/search_$(ID)_$rep.jls"))
    previous=deserialize(joinpath(PRIOR,"checkpoints/search_new_$rep.jls"))
    @assert previous.run.method==PREVIOUS&&previous.iterations==100
    merge(old,(runs=[old.runs;[previous.run]],))
end

function searches(d,cfg)
    archives=Dict(rep=>previous_search(rep) for rep in d["rep_ids"])
    proof=joinpath(d["output"],"validation/kernel_regression.toml")
    if !isfile(proof)
        old=only(filter(r->r.method.id=="GreenSoftmax050",archives[1].runs))
        A.timed_method(d,cfg,1,old.method;K=6,expected=old)
        v=A.timed_method(d,cfg,1,NEW;K=6)
        # Same data/weights formula at the first update, coupled to archived inputs.
        b=R.primitive_bank(cfg,500,X.seed(ID,"compare",1,"search",1);njobs=45)
        bank=R.materialize_bank(b,C.Z0,cfg);Y=R.rewards(C.build_policy_grid(),bank,cfg)
        data=R.green_data(C.Z0,[R.historical_bank(ID,cfg,C.Z0,bank,Y)],cfg,ID)
        w=R.softmax_weights(data.values;beta=.90);g=R.combine(data.G,w,cfg)
        z,eta,gc=R.projected_step(C.Z0,g,1)
        @assert isequal(z,v.run.zhist[2])&&isequal(g,[v.run.diagnostics[1].g3,v.run.diagnostics[1].g4])
        @assert all(all(isfinite,z)&&all(0 .<=z.<=1) for z in v.run.zhist)
        open(proof,"w") do io;TOML.print(io,Dict("archived_050_first_six_decisions_and_diagnostics_exact"=>true,
            "new_090_first_gradient_and_update_exact"=>true,"finite_projected_prefix"=>true));end
        println("Kernel checks passed; warmup excluded from search timing");flush(stdout)
    elseif any(!isfile(joinpath(d["output"],"checkpoints/search_new_$rep.jls")) for rep in d["rep_ids"])
        A.timed_method(d,cfg,1,NEW;K=2)
    end
    allruns=Dict{Int,Any}();budgets=NamedTuple[];checks=NamedTuple[];prefix_reps=Int[]
    for rep in d["rep_ids"]
        old=archives[rep];newpath=joinpath(d["output"],"checkpoints/search_new_$rep.jls")
        if isfile(newpath);v=deserialize(newpath)
        else
            GC.gc();v=A.timed_method(d,cfg,rep,NEW)
            v=merge(v,(rows=[merge(r,(timing_source="014 new method, separate sequential timing session",)) for r in v.rows],))
            E.atomic_serialize(newpath,v)
        end
        @assert v.iterations==d["max_iters"]
        @assert all(all(isfinite,z)&&all(0 .<=z.<=1) for z in v.run.zhist)
        @assert all(isfinite(s.g3)&&isfinite(s.g4)&&isfinite(s.ess)&&1-1e-8<=s.K_eff<=144+1e-8 for s in v.run.diagnostics)
        @assert all(r.cumulative_trajectories==500*r.iteration&&r.history_banks==r.iteration for r in v.rows[2:end])
        quickpath=joinpath(d["output"],"quick_test/checkpoints/search_new_$rep.jls")
        if !d["quick"]&&isfile(quickpath)
            q=deserialize(quickpath);nk=q.iterations
            @assert isequal(v.run.zhist[1:nk+1],q.run.zhist)&&isequal(v.run.diagnostics[1:nk],q.run.diagnostics)
            push!(prefix_reps,rep)
        end
        append!(budgets,v.rows)
        for m in METHODS[1:end-1]
            p=m.id==PREVIOUS.id ? joinpath(PRIOR,"checkpoints/search_new_$rep.jls") : joinpath(DONOR,"checkpoints/timing_$(rep)_$(m.id).jls");t=deserialize(p)
            r=only(filter(r->r.method.id==m.id,old.runs))
            @assert isequal(t.run.zhist,r.zhist)&&isequal(t.run.diagnostics,r.diagnostics)
            append!(budgets,filter(x->x.iteration<=d["max_iters"],t.rows))
            push!(checks,(rep=rep,method=m.id,source=m.id==PREVIOUS.id ? "013 unchanged beta .10 search and timing" : "011 unchanged search + 012 exact timing archive",exact_original=true,
                search_sha256=sha(m.id==PREVIOUS.id ? joinpath(PRIOR,"checkpoints/search_new_$rep.jls") : joinpath(SOURCE,"checkpoints/search_$(ID)_$rep.jls")),timing_sha256=sha(p)))
        end
        allruns[rep]=(runs=[[(method=r.method,zhist=r.zhist[1:d["max_iters"]+1],diagnostics=r.diagnostics[1:d["max_iters"]]) for r in old.runs];[v.run]],)
        println("New Green+Softmax 0.90 search complete: rep=$rep z=",v.run.zhist[end]);flush(stdout)
    end
    E.write_csv(joinpath(d["output"],"tables/search_budget_runtime.csv"),budgets)
    E.write_csv(joinpath(d["output"],"validation/archived_search_reuse.csv"),checks)
    open(joinpath(d["output"],"validation/full_run_checks.toml"),"w") do io
        TOML.print(io,Dict("all_runs_finite_projected"=>true,"original_search_budgets_verified"=>true,
            "quick_full_prefix_repetitions_exact"=>prefix_reps,"search_paths_per_rep"=>500*d["max_iters"]))
    end
    allruns
end

function evaluate_rep(d,cfg,rep,runs)
    path=joinpath(d["output"],"checkpoints/evaluation_new_$rep.jls")
    if isfile(path);return deserialize(path);end
    ss=X.seed(ID,"compare",rep,"trace");bank=R.primitive_bank(cfg,1000,ss;njobs=45)
    reference=point(d,cfg,rep,"main",A.ZREF,bank)
    old=deserialize(joinpath(PRIOR,"checkpoints/evaluation_new_$rep.jls"))
    @assert isequal(reference.objective,old.reference.objective)
    rows=NamedTuple[r for r in old.rows if r.iteration<=d["max_iters"]]
    new=only(filter(r->r.method.id==NEW.id,runs.runs))
    for (j,z) in enumerate(new.zhist)
        p=point(d,cfg,rep,"main",z,bank)
        push!(rows,(scenario=ID,rep=rep,cohort=A.cohort(rep),method=NEW.id,iteration=j-1,z3=z[1],z4=z[2],
            objective=p.objective,reference_objective=reference.objective,deficit=reference.objective-p.objective,
            n_paths=1000,evaluation_seed=ss,source="014 new method; same 011 independent CRN trace bank"))
    end
    v=(rows=rows,reference=old.reference);E.atomic_serialize(path,v)
    println("1000-path comparable traces complete: rep=$rep");flush(stdout);v
end

function plan(d,rows)
    plans=NamedTuple[]
    for rep in d["rep_ids"],m in METHODS
        q=sort(filter(r->r.rep==rep&&r.method==m.id,rows);by=r->r.iteration)
        @assert length(q)==d["max_iters"]+1
        rel=.05*(first(q).reference_objective-first(q).objective)
        for e in [(name=string(v),value=v) for v in A.EPS] ∪ [(name="relative_5pct",value=rel)],kind in ("first","sustained5")
            good=[r.deficit<=e.value for r in q]
            pos=kind=="first" ? findfirst(good) : findfirst(i->all(good[i:i+4]),1:(length(good)-4))
            start=pos===nothing ? missing : pos-1;confirm=ismissing(start) ? missing : start+(kind=="first" ? 0 : 4)
            times=ismissing(start) ? Int[] : kind=="first" ? unique([max(start-1,0),start]) : collect(start:confirm)
            push!(plans,(rep=rep,cohort=A.cohort(rep),method=m.id,epsilon_name=e.name,epsilon=e.value,kind=kind,
                achieved=pos!==nothing,start_iteration=start,confirmation_iteration=confirm,validation_iterations=join(times,";")))
        end
    end
    p=joinpath(d["output"],"configs/validation_plan.jls")
    isfile(p) ? (@assert isequal(deserialize(p),plans)) : E.atomic_serialize(p,plans)
    E.write_csv(joinpath(d["output"],"configs/validation_plan.csv"),plans);plans
end

function validate_rep(d,cfg,rep,runs,plans)
    path=joinpath(d["output"],"checkpoints/crossings_$rep.jls")
    if isfile(path);return deserialize(path);end
    ss=A.seed(rep,"threshold_validation");bank=R.primitive_bank(cfg,d["validation_paths"],ss;njobs=45)
    phase=d["quick"] ? "validation_quick" : "validation"
    reference=point(d,cfg,rep,phase,A.ZREF,bank);rows=NamedTuple[]
    old=d["quick"] ? nothing : (rows=deserialize(joinpath(PRIOR,"checkpoints/crossings_$rep.jls")),)
    for p in filter(p->p.rep==rep&&p.achieved,plans)
        times=parse.(Int,split(p.validation_iterations,';'))
        if old!==nothing&&p.method!=NEW.id
            q=filter(r->r.method==p.method&&r.kind==p.kind&&r.epsilon_name==p.epsilon_name,old.rows)
            @assert sort([r.iteration for r in q])==sort(times)
            @assert all(r.n_paths==5000&&r.evaluation_seed==ss&&r.start_iteration==p.start_iteration&&r.confirmation_iteration==p.confirmation_iteration for r in q)
            append!(rows,q);continue
        end
        run=only(filter(r->r.method.id==p.method,runs.runs))
        for t in times
            v=point(d,cfg,rep,phase,run.zhist[t+1],bank)
            delta=reference.rewards-v.rewards .-(reference.cost-v.cost)
            push!(rows,(rep=rep,cohort=A.cohort(rep),method=p.method,kind=p.kind,epsilon_name=p.epsilon_name,epsilon=p.epsilon,
                start_iteration=p.start_iteration,confirmation_iteration=p.confirmation_iteration,iteration=t,
                role=t<p.start_iteration ? "preceding" : "candidate",objective=v.objective,reference_objective=reference.objective,
                deficit=mean(delta),paired_mc_se=std(delta)/sqrt(length(delta)),meets_threshold=mean(delta)<=p.epsilon,
                reference_policy=reference.policy,policy=v.policy,n_paths=length(bank),evaluation_seed=ss))
        end
    end
    @assert length(rows)==sum(length(split(p.validation_iterations,';')) for p in plans if p.rep==rep&&p.achieved)
    E.atomic_serialize(path,rows);println("Independent crossing rechecks complete: rep=$rep");flush(stdout);rows
end

function final_rep(d,cfg,rep,runs)
    path=joinpath(d["output"],"checkpoints/final_$rep.jls")
    if isfile(path);return deserialize(path);end
    old=deserialize(joinpath(PRIOR,"checkpoints/final_$rep.jls"))
    finals=copy(old.finals);policies=C.build_policy_grid();new=only(filter(r->r.method.id==NEW.id,runs.runs));z=new.zhist[end]
    selss=X.seed(ID,"compare",rep,"selection");holdss=X.seed(ID,"compare",rep,"holdout")
    select=R.primitive_bank(cfg,10000,selss;njobs=45);hold=R.primitive_bank(cfg,50000,holdss;njobs=45)
    Ysel=R.rewards(policies,R.materialize_bank(select,z,cfg),cfg);idx=argmax(vec(mean(Ysel;dims=2)))
    Yhold=R.rewards(policies,R.materialize_bank(hold,z,cfg),cfg);means=vec(mean(Yhold;dims=2));direct=argmax(means)
    values=Dict{String,Vector{Float64}}(NEW.id=>collect(view(Yhold,idx,:)).-R.cost(cfg,z))
    vals=values[NEW.id];cost=R.cost(cfg,z)
    push!(finals,(scenario=ID,rep=rep,method=NEW.id,z3=z[1],z4=z[2],selected_policy=idx,
        gross=mean(view(Yhold,idx,:)),probing_cost=cost,objective_holdout=mean(view(Yhold,idx,:))-cost,evaluation_se=std(view(Yhold,idx,:))/sqrt(length(vals)),
        objective_direct=means[direct]-cost,direct_policy=direct,selection_paths=10000,holdout_paths=50000,selection_seed=selss,holdout_seed=holdss))
    pairs=copy(old.paired)
    for m in METHODS[1:end-1]
        r=only(filter(r->r.method==m.id,old.finals));zz=[r.z3,r.z4];bank=R.materialize_bank(hold,zz,cfg)
        raw=[R.evaluate_policy(policies[r.selected_policy],p,cfg) for p in bank]
        v=raw.-R.cost(cfg,zz)
        @assert isapprox(mean(v),r.objective_holdout;atol=1e-10,rtol=1e-12)
        @assert isapprox(std(v)/sqrt(length(v)),r.evaluation_se;atol=1e-12)
        delta=vals-v;se=std(delta)/sqrt(length(delta));mu=mean(delta)
        for (a,b,sgn) in ((NEW.id,m.id,1.),(m.id,NEW.id,-1.))
            push!(pairs,(scenario=ID,rep=rep,method_a=a,method_b=b,difference=sgn*mu,paired_se=se,snr=X.safe_snr(mu,se)))
        end
    end
    v=(finals=finals,paired=pairs);E.atomic_serialize(path,v)
    println("Independent 10000/50000-path final evaluation complete: rep=$rep J=",mean(vals));flush(stdout);v
end

function main(args=ARGS)
    BLAS.set_num_threads(1);d,cfg,plots=initialize(args)
    if !plots
        runs=searches(d,cfg)
        evals=[evaluate_rep(d,cfg,rep,runs[rep]) for rep in d["rep_ids"]]
        rows=vcat([v.rows for v in evals]...)
        E.write_csv(joinpath(d["output"],"tables/objectives_by_iteration.csv"),rows)
        E.write_csv(joinpath(d["output"],"tables/reference_by_rep.csv"),[v.reference for v in evals])
        plans=plan(d,rows)
        crossings=vcat([validate_rep(d,cfg,rep,runs[rep],plans) for rep in d["rep_ids"]]...)
        E.write_csv(joinpath(d["output"],"tables/crossing_validation.csv"),crossings)
        for rep in d["rep_ids"]
            prefix=joinpath(d["output"],"comparisons",ID,@sprintf("rep_%02d",rep));q=filter(r->r.rep==rep,rows)
            E.write_csv(prefix*"_traces.csv",q)
            E.write_csv(prefix*"_coordinates.csv",[(scenario=ID,rep=rep,method=r.method,iteration=r.iteration,z3=r.z3,z4=r.z4) for r in q])
            E.write_csv(prefix*"_diagnostics.csv",[merge((scenario=ID,rep=rep,method=r.method.id),s) for r in runs[rep].runs for s in r.diagnostics])
            if !d["quick"]
                v=final_rep(d,cfg,rep,runs[rep]);E.write_csv(prefix*"_finals.csv",v.finals);E.write_csv(prefix*"_paired.csv",v.paired)
            end
        end
        E.write_csv(joinpath(d["output"],"tables/new_method_search_seeds.csv"),[(rep=rep,method=NEW.id,iteration=k,
            seed=X.seed(ID,"compare",rep,"search",k),n_paths=500,jobs=45,stage="search") for rep in d["rep_ids"] for k in 1:d["max_iters"]])
    end
    run(`python3 $(joinpath(ROOT,"analyze_green_softmax090_014.py")) $(d["output"])`)
end
end
if abspath(PROGRAM_FILE)==@__FILE__;ProbingGreenSoftmax090.main();end
