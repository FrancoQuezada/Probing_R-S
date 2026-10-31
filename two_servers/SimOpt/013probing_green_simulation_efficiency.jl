# New experiment only: sample size changes; all scientific kernels stay frozen.
include(joinpath(@__DIR__,"012probing_anytime_budget_analysis.jl"))
module ProbingGreenSimulationEfficiency
using ..ProbingAnytimeBudget
using Statistics, LinearAlgebra, Serialization, SHA, TOML, Printf, Dates
const A=ProbingAnytimeBudget;const H=A.H;const X=A.X;const R=A.R;const E=A.E;const C=A.C
const ROOT=@__DIR__;const ID="I0252"
const SOURCE=joinpath(ROOT,"resultados/011probing_high_contrast_interior")
const DONOR=joinpath(ROOT,"resultados/012probing_anytime_budget")
const OUTPUT=joinpath(ROOT,"resultados/013probing_green_simulation_efficiency")
const METHODS=[(id="$b-$n",kind=b=="Fresh" ? :fresh : :green,beta=0.,base=b,n=n) for b in ("Fresh","Green") for n in (100,250,500)]
cohort(rep)=A.cohort(rep)
sha(p)=bytes2hex(sha256(read(p)))
include(joinpath(@__DIR__,"probing_green_efficiency_013_runtime.jl"))

function initialize(args)
    all(a in ("--quick-test","--resume","--plots-only","--no-plots") for a in args)||error("Unknown argument")
    quick="--quick-test" in args;out=quick ? joinpath(OUTPUT,"quick_test") : OUTPUT
    d=Dict{String,Any}("output"=>out,"quick"=>quick,"max_iters"=>quick ? 6 : 100,"jobs"=>45,
        "n_paths"=>500,"bootstrap_reps"=>300,"trace_paths"=>1000,"validation_paths"=>quick ? 100 : 5000,
        "rep_ids"=>collect(1:(quick ? 2 : 5)),"budgets"=>quick ? [500,1000] : [10000,25000,50000],
        "horizons"=>quick ? [2,4,6] : [20,50,100])
    for p in ("configs","tables","trajectories","checkpoints","validation","figures","logs","reproduction");mkpath(joinpath(out,p));end
    original=TOML.parsefile(joinpath(SOURCE,"configs/experiment.toml"))["science"]
    @assert original["n_paths"]==500&&original["max_iters"]==100&&original["reps"]==5&&original["jobs"]==45
    @assert original["BASE_STEP"]==C.BASE_STEP==.035&&original["GRAD_CLIP"]==C.GRAD_CLIP==15.0&&original["Z0"]==C.Z0==[.12,.10]
    for (f,h) in original["sources"];@assert sha(joinpath(ROOT,f))==h "Scientific source changed: $f";end
    cfgdict=TOML.parsefile(joinpath(SOURCE,"configs/main_instance_parameters.toml"))["model"]
    cfg=H.from_dict(cfgdict);@assert R.model_dict(cfg)==cfgdict
    @assert R.model_digest(cfg)==TOML.parsefile(joinpath(DONOR,"configs/experiment.toml"))["science"]["reference_model_sha256"]
    @assert length(C.build_policy_grid())==144&&cfg.beta==4.
    files=["013probing_green_simulation_efficiency.jl","probing_green_efficiency_013_runtime.jl","PROTOCOLO_green_simulation_efficiency_013.txt"]
    science=Dict("scenario"=>ID,"quick"=>quick,"max_iters"=>d["max_iters"],"jobs"=>45,"rep_ids"=>d["rep_ids"],
        "methods"=>[m.id for m in METHODS],"sample_sizes"=>[100,250,500],"trace_paths"=>1000,"validation_paths"=>d["validation_paths"],
        "selection_paths"=>10000,"holdout_paths"=>50000,"BASE_STEP"=>C.BASE_STEP,"GRAD_CLIP"=>C.GRAD_CLIP,"Z0"=>C.Z0,
        "z_ref"=>A.ZREF,"epsilons"=>A.EPS,"sustained_window"=>5,"comparability_margin"=>.5,
        "budgets"=>d["budgets"],"horizons"=>d["horizons"],"model_sha256"=>R.model_digest(cfg),
        "source_011_sha256"=>sha(joinpath(SOURCE,"configs/experiment.toml")),"source_012_sha256"=>sha(joinpath(DONOR,"configs/experiment.toml")),
        "sources"=>Dict(f=>sha(joinpath(ROOT,f)) for f in files),"objective"=>"max_i E_z[Y_i]-alpha'z; ex-ante policy")
    fp=R.digest(science);path=joinpath(out,"configs/experiment.toml")
    isfile(path)&&(@assert TOML.parsefile(path)["fingerprint"]==fp "Different source/protocol: use a separate output")
    open(path,"w") do io;TOML.print(io,Dict("science"=>science,"fingerprint"=>fp);sorted=true);end
    cp(joinpath(SOURCE,"configs/main_instance_parameters.toml"),joinpath(out,"configs/I0252.toml");force=true)
    cp(joinpath(DONOR,"configs/reference.toml"),joinpath(out,"configs/reference.toml");force=true)
    cp(joinpath(SOURCE,"high/I0252_surface.csv"),joinpath(out,"configs/I0252_surface.csv");force=true)
    for f in files;cp(joinpath(ROOT,f),joinpath(out,"reproduction",f);force=true);end
    d,cfg
end

function oldrun(rep,base)
    q=deserialize(joinpath(SOURCE,"checkpoints/search_$(ID)_$rep.jls"))
    only(filter(r->r.method.id==base,q.runs))
end
function sameprimitives(a,b)
    length(a)==length(b)&&all(length(pa.jobs)==45&&length(pb.jobs)==45&&
        all(isequal(getfield(x,f),getfield(y,f)) for (x,y) in zip(pa.jobs,pb.jobs) for f in fieldnames(typeof(x))) for (pa,pb) in zip(a,b))
end
function audit(d,cfg)
    path=joinpath(d["output"],"checkpoints/crn_audit.jls");isfile(path)&&return deserialize(path)
    rows=NamedTuple[];seeds=Int[];checks=NamedTuple[]
    for rep in d["rep_ids"]
        archive=deserialize(joinpath(SOURCE,"checkpoints/search_$(ID)_$rep.jls"))
        for k in 1:d["max_iters"]
            ss=X.seed(ID,"compare",rep,"search",k);push!(seeds,ss)
            @assert only(filter(r->r.stream=="search"&&r.iteration==k,archive.seeds)).seed==ss
            b500=R.primitive_bank(cfg,500,ss;njobs=45)
            b250=R.primitive_bank(cfg,250,ss;njobs=45);b100=R.primitive_bank(cfg,100,ss;njobs=45)
            @assert sameprimitives(b100,b500[1:100])&&sameprimitives(b250,b500[1:250])
            firsthash=bytes2hex(sha256(repr([Tuple(getfield(j,f) for f in fieldnames(typeof(j))) for j in b500[1].jobs])))
            push!(rows,(rep=rep,iteration=k,seed=ss,prefix100_exact=true,prefix250_exact=true,first_path_sha256=firsthash,
                audit_primitive_paths_generated=850,search_paths_charged=0))
        end
        for base in ("Fresh","Green")
            a=oldrun(rep,base);b=deserialize(joinpath(DONOR,"checkpoints/timing_$(rep)_$base.jls"))
            @assert isequal(a.zhist,b.run.zhist)&&isequal(a.diagnostics,b.run.diagnostics)
            @assert length(a.zhist)==101
            push!(checks,(rep=rep,method="$base-500",exact_decisions=true,exact_gradients_and_diagnostics=true,
                original_search_sha256=sha(joinpath(SOURCE,"checkpoints/search_$(ID)_$rep.jls")),
                original_timing_sha256=sha(joinpath(DONOR,"checkpoints/timing_$(rep)_$base.jls"))))
        end
        println("CRN prefixes and original controls audited: rep=$rep");flush(stdout)
    end
    @assert length(unique(seeds))==length(seeds)
    @assert length(unique(r.first_path_sha256 for r in rows))==length(rows)
    E.write_csv(joinpath(d["output"],"validation/crn_prefix_checks.csv"),rows)
    E.write_csv(joinpath(d["output"],"validation/control_reuse.csv"),checks)
    E.atomic_serialize(path,(rows=rows,controls=checks));(rows=rows,controls=checks)
end

function search_phase(d,cfg)
    proof=joinpath(d["output"],"validation/kernel_equivalence.toml")
    if !isfile(proof)
        for m in METHODS
            dd=merge(d,Dict("n_paths"=>m.n))
            if m.n==500
                timed_search(dd,cfg,1,m;K=6,expected=oldrun(1,m.base))
            else
                expected=A.timed_method(dd,cfg,1,m;K=6).run
                timed_search(dd,cfg,1,m;K=6,expected=expected)
            end
        end
        open(proof,"w") do io;TOML.print(io,Dict("all_six_configurations_first_six_updates_exact"=>true,"only_bank_size_changes"=>true));end
    elseif any(!isfile(joinpath(d["output"],"checkpoints/search_$(m.id)_$rep.jls")) for m in METHODS if m.n!=500 for rep in d["rep_ids"])
        for m in filter(m->m.n!=500,METHODS);timed_search(merge(d,Dict("n_paths"=>m.n)),cfg,1,m;K=2);end
    end
    runs=Dict{Tuple{Int,String},Any}();budgets=NamedTuple[];diagnostics=NamedTuple[]
    for rep in d["rep_ids"],m in METHODS
        if m.n==500
            old=oldrun(rep,m.base);v=deserialize(joinpath(DONOR,"checkpoints/timing_$(rep)_$(m.base).jls"))
            run=(method=m,zhist=old.zhist[1:d["max_iters"]+1],diagnostics=old.diagnostics[1:d["max_iters"]])
            br=[merge(r,(method=m.id,reuse_seconds=missing,cumulative_reuse_seconds=missing,)) for r in v.rows if r.iteration<=d["max_iters"]]
        else
            path=joinpath(d["output"],"checkpoints/search_$(m.id)_$rep.jls")
            if isfile(path);v=deserialize(path)
            else;GC.gc();v=timed_search(merge(d,Dict("n_paths"=>m.n)),cfg,rep,m);E.atomic_serialize(path,v);end
            run=v.run;br=v.rows
            qp=joinpath(d["output"],"quick_test/checkpoints/search_$(m.id)_$rep.jls")
            if !d["quick"]&&isfile(qp)
                q=deserialize(qp);@assert isequal(run.zhist[1:q.iterations+1],q.run.zhist)&&isequal(run.diagnostics[1:q.iterations],q.run.diagnostics)
            end
        end
        @assert length(run.zhist)==d["max_iters"]+1
        for (k,s) in enumerate(run.diagnostics)
            z,eta,gc=R.projected_step(run.zhist[k],[s.g3,s.g4],k)
            @assert isequal(z,run.zhist[k+1])&&all(isfinite,z)&&all(0 .<=z.<=1)
            @assert isfinite(s.g3)&&isfinite(s.g4)
            @assert br[k+1].cumulative_trajectories==m.n*k&&br[k+1].new_trajectories==m.n
            @assert br[k+1].history_banks==(m.kind==:green ? k : 1)
            push!(diagnostics,merge((rep=rep,method=m.id,algorithm=m.base,N_PATHS=m.n,source_z3=run.zhist[k][1],source_z4=run.zhist[k][2]),s))
        end
        runs[(rep,m.id)]=run
        append!(budgets,[merge(r,(algorithm=m.base,N_PATHS=m.n,available_history_paths=m.kind==:green ? m.n*r.iteration : (r.iteration==0 ? 0 : m.n),)) for r in br])
        println("Search ready: $(m.id) rep=$rep ",m.n==500 ? "archived" : "new", " z=",run.zhist[end]);flush(stdout)
    end
    # Needed only for the previously unrecorded control history-clock split.
    profiles=NamedTuple[]
    for rep in d["rep_ids"]
        m=only(filter(m->m.id=="Green-500",METHODS));path=joinpath(d["output"],"checkpoints/profile_Green500_$rep.jls")
        if isfile(path);v=deserialize(path)
        else;GC.gc();v=timed_search(d,cfg,rep,m;expected=oldrun(rep,"Green"));E.atomic_serialize(path,v);end
        append!(profiles,[merge(r,(timing_source="separate exact Green-500 profiling replay; primary clock remains 012",)) for r in v.rows])
        println("Green-500 history-clock replay exact: rep=$rep");flush(stdout)
    end
    E.write_csv(joinpath(d["output"],"tables/search_budget_runtime.csv"),budgets)
    E.write_csv(joinpath(d["output"],"tables/gradients_and_diagnostics.csv"),diagnostics)
    E.write_csv(joinpath(d["output"],"tables/control_history_profiling.csv"),profiles)
    E.write_csv(joinpath(d["output"],"configs/search_seeds.csv"),[(rep=rep,method=m.id,N_PATHS=m.n,iteration=k,
        seed=X.seed(ID,"compare",rep,"search",k)) for rep in d["rep_ids"] for m in METHODS for k in 1:d["max_iters"]])
    runs
end

function point(d,cfg,rep,phase,z,bank,ss)
    name="point_$(phase)_$(rep)_$(bytes2hex(sha256(repr(z)))).jls";path=joinpath(d["output"],"checkpoints",name)
    isfile(path)&&return deserialize(path)
    donor=joinpath(DONOR,"checkpoints",name)
    if isfile(donor)
        p=deserialize(donor);@assert p.z==z&&p.n_paths==length(bank)
        v=merge(p,(phase=phase,rep=rep,seed=ss,origin="012 exact evaluation point reuse",new_policy_evaluations=0,))
    else
        Y=R.rewards(C.build_policy_grid(),R.materialize_bank(bank,z,cfg),cfg);means=vec(mean(Y;dims=2));i=argmax(means)
        rewards=collect(view(Y,i,:));cost=R.cost(cfg,z)
        v=(z=copy(z),objective=means[i]-cost,cost=cost,policy=i,rewards=rewards,se=std(rewards)/sqrt(length(bank)),n_paths=length(bank),
            phase=phase,rep=rep,seed=ss,origin="new independent direct evaluation",new_policy_evaluations=length(Y))
    end
    E.atomic_serialize(path,v);v
end

function evaluate_rep(d,cfg,rep,runs)
    path=joinpath(d["output"],"checkpoints/evaluation_$rep.jls");isfile(path)&&return deserialize(path)
    ss=X.seed(ID,"compare",rep,"trace");bank=R.primitive_bank(cfg,1000,ss;njobs=45)
    ref=point(d,cfg,rep,"main",A.ZREF,bank,ss)
    old=deserialize(joinpath(DONOR,"checkpoints/main_evaluation_$rep.jls"));@assert isequal(ref.objective,old.reference.objective)
    rows=NamedTuple[]
    for m in METHODS
        run=runs[(rep,m.id)]
        for (j,z) in enumerate(run.zhist)
            k=j-1
            if m.n==500
                r=only(filter(r->r.method==m.base&&r.iteration==k,old.rows));@assert isequal(z,[r.z3,r.z4])&&r.evaluation_seed==ss
                obj=r.objective;origin="011/012 exact control trace reuse"
            else
                p=point(d,cfg,rep,"main",z,bank,ss);obj=p.objective;origin="new configuration on original 012 CRN bank"
            end
            push!(rows,(scenario=ID,rep=rep,cohort=cohort(rep),method=m.id,algorithm=m.base,N_PATHS=m.n,iteration=k,
                z3=z[1],z4=z[2],objective=obj,reference_objective=ref.objective,deficit=ref.objective-obj,
                evaluation_paths=1000,evaluation_seed=ss,source=origin))
        end
    end
    v=(rows=rows,reference=old.reference);E.atomic_serialize(path,v)
    println("Independent common traces complete: rep=$rep");flush(stdout);v
end

function validation_plan(d,rows)
    plans=NamedTuple[]
    for rep in d["rep_ids"],m in METHODS
        q=sort(filter(r->r.rep==rep&&r.method==m.id,rows);by=r->r.iteration)
        for eps in A.EPS,kind in ("first","sustained5")
            good=[r.deficit<=eps for r in q];w=kind=="first" ? 1 : 5
            pos=findfirst(i->all(good[i:i+w-1]),1:(length(good)-w+1))
            start=pos===nothing ? missing : pos-1;confirm=ismissing(start) ? missing : start+w-1
            ts=ismissing(start) ? Int[] : w==1 ? unique([max(0,start-1),start]) : collect(start:confirm)
            push!(plans,(rep=rep,method=m.id,epsilon=eps,epsilon_name=string(eps),kind=kind,achieved=pos!==nothing,
                start_iteration=start,confirmation_iteration=confirm,validation_iterations=join(ts,";")))
        end
    end
    path=joinpath(d["output"],"configs/validation_plan.jls")
    if isfile(path);@assert isequal(deserialize(path),plans);else;E.atomic_serialize(path,plans);end
    E.write_csv(joinpath(d["output"],"configs/validation_plan.csv"),plans);plans
end

function validate_rep(d,cfg,rep,runs,plans)
    path=joinpath(d["output"],"checkpoints/validation_$rep.jls");isfile(path)&&return deserialize(path)
    ss=A.seed(rep,"threshold_validation");np=d["validation_paths"];bank=R.primitive_bank(cfg,np,ss;njobs=45)
    phase=d["quick"] ? "validation_quick" : "validation";ref=point(d,cfg,rep,phase,A.ZREF,bank,ss)
    rows=NamedTuple[];budgets=NamedTuple[];pairs=NamedTuple[]
    old=d["quick"] ? nothing : deserialize(joinpath(DONOR,"checkpoints/crossing_validation_$rep.jls"))
    for p in filter(p->p.rep==rep&&p.achieved,plans)
        m=only(filter(m->m.id==p.method,METHODS))
        for t in parse.(Int,split(p.validation_iterations,';'))
            if old!==nothing&&m.n==500
                r=only(filter(r->r.method==m.base&&r.kind==p.kind&&r.epsilon_name==p.epsilon_name&&r.iteration==t,old.rows))
                @assert r.start_iteration==p.start_iteration&&r.confirmation_iteration==p.confirmation_iteration
                push!(rows,merge(r,(method=m.id,source="012 unchanged candidate recheck",)));continue
            end
            v=point(d,cfg,rep,phase,runs[(rep,m.id)].zhist[t+1],bank,ss)
            delta=ref.rewards-v.rewards .-(ref.cost-v.cost)
            push!(rows,(rep=rep,cohort=cohort(rep),method=m.id,kind=p.kind,epsilon_name=p.epsilon_name,epsilon=p.epsilon,
                start_iteration=p.start_iteration,confirmation_iteration=p.confirmation_iteration,iteration=t,
                role=t<p.start_iteration ? "preceding" : "candidate",objective=v.objective,reference_objective=ref.objective,
                deficit=mean(delta),paired_mc_se=std(delta)/sqrt(np),meets_threshold=mean(delta)<=p.epsilon,
                reference_policy=ref.policy,policy=v.policy,n_paths=np,evaluation_seed=ss,source="independent selected-candidate recheck"))
        end
    end
    for B in d["budgets"]
        pts=Dict{String,Any}()
        for m in METHODS
            B%m.n==0&&B÷m.n<=d["max_iters"]||continue
            t=B÷m.n;v=point(d,cfg,rep,phase,runs[(rep,m.id)].zhist[t+1],bank,ss);pts[m.id]=v
            dv=ref.rewards-v.rewards .-(ref.cost-v.cost)
            push!(budgets,(rep=rep,method=m.id,algorithm=m.base,N_PATHS=m.n,budget=B,iteration=t,objective=v.objective,
                reference_objective=ref.objective,deficit=mean(dv),paired_mc_se=std(dv)/sqrt(np),evaluation_paths=np,evaluation_seed=ss))
        end
        for ma in METHODS,mb in METHODS
            ma.id==mb.id||!haskey(pts,ma.id)||!haskey(pts,mb.id)&&continue
            haskey(pts,ma.id)&&haskey(pts,mb.id)&&ma.id!=mb.id||continue
            a=pts[ma.id];b=pts[mb.id];dv=a.rewards-b.rewards .-(a.cost-b.cost)
            push!(pairs,(rep=rep,budget=B,method_a=ma.id,method_b=mb.id,difference=mean(dv),paired_mc_se=std(dv)/sqrt(np),evaluation_paths=np))
        end
    end
    expected=sum(length(split(p.validation_iterations,';')) for p in plans if p.rep==rep&&p.achieved)
    @assert length(rows)==expected
    v=(rows=rows,budgets=budgets,pairs=pairs);E.atomic_serialize(path,v)
    println("Independent candidate and equal-budget validation complete: rep=$rep");flush(stdout);v
end

function green_diagnostics_rep(d,cfg,rep,runs)
    path=joinpath(d["output"],"checkpoints/green_diagnostics_$rep.jls");isfile(path)&&return deserialize(path)
    rows=NamedTuple[];bankrows=NamedTuple[];histrows=NamedTuple[]
    for m in filter(m->m.kind==:green,METHODS)
        run=runs[(rep,m.id)];banks=Any[]
        for k in 1:d["max_iters"]
            source=run.zhist[k];primitive=R.primitive_bank(cfg,m.n,X.seed(ID,"compare",rep,"search",k);njobs=45)
            push!(banks,(zgen=source,bank=R.materialize_bank(primitive,source,cfg)))
            @assert length(banks)==k&&sum(length(b.bank) for b in banks)==m.n*k
            ls=Float64[];lls=Float64[];mass=Float64[];sumL=0.;sumL2=0.;clips=0
            for b in banks
                ll,_=R.bank_scores(b.bank,source,b.zgen,cfg);L=exp.(clamp.(ll,-35.,35.))
                append!(ls,L);append!(lls,ll);push!(mass,sum(L));sumL+=sum(L);sumL2+=sum(abs2,L);clips+=count(x->abs(x)>35,ll)
            end
            ess=sumL^2/sumL2;s=run.diagnostics[k]
            @assert isequal(ess,s.ess)&&clips==s.lr_clips "LR diagnostic does not match actual Green estimator"
            probs=[0.,.01,.1,.5,.9,.99,1.];ql=quantile(lls,probs);qw=quantile(ls,probs)
            push!(rows,(rep=rep,method=m.id,N_PATHS=m.n,iteration=k,history_banks=k,available_paths=m.n*k,
                ess=ess,ess_fraction=ess/(m.n*k),lr_sum=sumL,lr_sum_squares=sumL2,lr_mean=mean(ls),lr_sd=std(ls),
                lr_min=qw[1],lr_p01=qw[2],lr_p10=qw[3],lr_median=qw[4],lr_p90=qw[5],lr_p99=qw[6],lr_max=qw[7],
                loglr_min=ql[1],loglr_p01=ql[2],loglr_p10=ql[3],loglr_median=ql[4],loglr_p90=ql[5],loglr_p99=ql[6],loglr_max=ql[7],
                truncations=clips,truncation_fraction=clips/(m.n*k),newest_bank_lr_mass=mass[end]/sumL,
                historical_lr_mass=k==1 ? 0. : sum(mass[1:end-1])/sumL,gradient_norm=s.gradient_norm,clipping=s.clipping,projection=s.projection))
            if k in d["horizons"]
                for j in 1:k;push!(bankrows,(rep=rep,method=m.id,iteration=k,bank=j,bank_age=k-j,paths=m.n,
                    estimator_bank_coefficient=1/k,mean_lr=mass[j]/m.n,relative_lr_mass=mass[j]/sumL,is_newest=j==k));end
                edges=[-Inf,-35.,-20.,-10.,-5.,-2.,-1.,0.,1.,2.,5.,10.,20.,35.,Inf]
                for j in 1:length(edges)-1
                    c=count(x->edges[j]<=x<edges[j+1],lls)
                    push!(histrows,(rep=rep,method=m.id,iteration=k,bin=j,left=edges[j],right=edges[j+1],count=c,fraction=c/length(lls)))
                end
            end
        end
    end
    v=(rows=rows,banks=bankrows,histograms=histrows);E.atomic_serialize(path,v)
    println("All-iteration exact Green LR/ESS diagnostics complete: rep=$rep");flush(stdout);v
end

function final_rep(d,cfg,rep,runs)
    path=joinpath(d["output"],"checkpoints/final_holdout_$rep.jls");isfile(path)&&return deserialize(path)
    old=deserialize(joinpath(SOURCE,"checkpoints/evaluation_$(ID)_$rep.jls"))
    ss=X.seed(ID,"compare",rep,"selection");hs=X.seed(ID,"compare",rep,"holdout")
    selector=R.primitive_bank(cfg,10000,ss;njobs=45);hold=R.primitive_bank(cfg,50000,hs;njobs=45);policies=C.build_policy_grid()
    rows=NamedTuple[];vectors=Dict{String,Any}();pairs=NamedTuple[]
    for m in METHODS
        z=runs[(rep,m.id)].zhist[end];cost=R.cost(cfg,z)
        if m.n==500
            r=only(filter(r->r.method==m.base,old.finals));@assert [r.z3,r.z4]==z
            bank=R.materialize_bank(hold,z,cfg);raw=[R.evaluate_policy(policies[r.selected_policy],p,cfg) for p in bank]
            @assert isapprox(mean(raw)-cost,r.objective_holdout;atol=1e-10)
            vectors[m.id]=raw.-cost
            push!(rows,merge(r,(method=m.id,algorithm=m.base,N_PATHS=m.n,source="011 unchanged final selected-policy holdout",)))
        else
            Ys=R.rewards(policies,R.materialize_bank(selector,z,cfg),cfg);chosen=argmax(vec(mean(Ys;dims=2)))
            Ye=R.rewards(policies,R.materialize_bank(hold,z,cfg),cfg);vals=collect(view(Ye,chosen,:));av=vec(mean(Ye;dims=2));best=argmax(av)
            vectors[m.id]=vals.-cost
            push!(rows,(scenario=ID,rep=rep,method=m.id,z3=z[1],z4=z[2],selected_policy=chosen,gross=mean(vals),probing_cost=cost,
                objective_holdout=mean(vals)-cost,evaluation_se=std(vals)/sqrt(50000),objective_direct=av[best]-cost,direct_policy=best,
                selection_paths=10000,holdout_paths=50000,selection_seed=ss,holdout_seed=hs,algorithm=m.base,N_PATHS=m.n,
                source="new independent selected-policy holdout"))
        end
    end
    for a in METHODS,b in METHODS
        a.id==b.id&&continue;dv=vectors[a.id]-vectors[b.id]
        push!(pairs,(rep=rep,method_a=a.id,method_b=b.id,difference=mean(dv),paired_mc_se=std(dv)/sqrt(length(dv))))
    end
    v=(rows=rows,pairs=pairs);E.atomic_serialize(path,v);println("Independent 10000/50000-path final holdout complete: rep=$rep");flush(stdout);v
end

function parallel_reps(f,d)
    results=Vector{Any}(undef,length(d["rep_ids"]))
    Threads.@threads :dynamic for j in eachindex(d["rep_ids"]);results[j]=f(d["rep_ids"][j]);end
    results
end
function write_phase(d,name,results,field)
    rows=vcat([getproperty(v,field) for v in results]...)
    isempty(rows)||E.write_csv(joinpath(d["output"],"tables",name*".csv"),rows)
end
function main(args=ARGS)
    BLAS.set_num_threads(1);d,cfg=initialize(args)
    if !("--plots-only" in args)
        audit(d,cfg);runs=search_phase(d,cfg)
        ev=parallel_reps(rep->evaluate_rep(d,cfg,rep,runs),d);rows=vcat([v.rows for v in ev]...)
        E.write_csv(joinpath(d["output"],"tables/objectives_by_iteration.csv"),rows)
        E.write_csv(joinpath(d["output"],"tables/reference_by_rep.csv"),[v.reference for v in ev])
        plans=validation_plan(d,rows)
        val=parallel_reps(rep->validate_rep(d,cfg,rep,runs,plans),d)
        for (name,field) in (("crossing_validation",:rows),("budget_validation",:budgets),("budget_validation_pairs",:pairs));write_phase(d,name,val,field);end
        gd=parallel_reps(rep->green_diagnostics_rep(d,cfg,rep,runs),d)
        for (name,field) in (("green_lr_diagnostics",:rows),("green_bank_weights",:banks),("green_loglr_histograms",:histograms));write_phase(d,name,gd,field);end
        if !d["quick"]
            final=parallel_reps(rep->final_rep(d,cfg,rep,runs),d)
            write_phase(d,"final_selected_holdout",final,:rows);write_phase(d,"final_selected_holdout_pairs",final,:pairs)
        end
        ledger=NamedTuple[]
        for f in readdir(joinpath(d["output"],"checkpoints");join=true)
            startswith(basename(f),"point_")||continue;p=deserialize(f)
            push!(ledger,(rep=p.rep,phase=p.phase,z3=p.z[1],z4=p.z[2],evaluation_paths=p.n_paths,evaluation_seed=p.seed,
                origin=p.origin,materialized_paths=p.new_policy_evaluations÷144,policy_path_evaluations=p.new_policy_evaluations,search_paths_charged=0))
        end
        E.write_csv(joinpath(d["output"],"tables/external_point_workload.csv"),ledger)
        for rep in d["rep_ids"]
            E.write_csv(joinpath(d["output"],"trajectories",@sprintf("rep%02d.csv",rep)),filter(r->r.rep==rep,rows))
        end
    end
    "--no-plots" in args||run(`python3 $(joinpath(ROOT,"analyze_green_simulation_efficiency_013.py")) $(d["output"])`)
    println("Simulation-efficiency experiment complete: ",d["output"])
end
end
if abspath(PROGRAM_FILE)==@__FILE__;ProbingGreenSimulationEfficiency.main();end
