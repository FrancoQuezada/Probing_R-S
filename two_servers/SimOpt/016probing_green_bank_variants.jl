# Additive I0252 study: only bank selection/weighting changes. No old source edits.
include(joinpath(@__DIR__,"012probing_anytime_budget_analysis.jl"))
module ProbingGreenBankVariants
using ..ProbingAnytimeBudget
using Statistics, LinearAlgebra, Serialization, SHA, TOML, Dates, Printf, Test
const A=ProbingAnytimeBudget;const H=A.H;const X=A.X;const R=A.R;const E=A.E;const C=A.C
const ROOT=@__DIR__;const ID="I0252"
const SOURCE=joinpath(ROOT,"resultados/011probing_high_contrast_interior")
const DONOR=joinpath(ROOT,"resultados/012probing_anytime_budget")
const OUTPUT=joinpath(ROOT,"resultados/016probing_green_bank_variants")
const METHODS=vcat([(id="Green",family=:original,param=0.)],
    [(id="Rolling$(w)",family=:rolling,param=Float64(w)) for w in (5,10,20)],
    [(id="Exp$(round(Int,100r))",family=:exponential,param=r) for r in (.7,.9,.95)],
    [(id="Filter$(k)",family=:filtered,param=Float64(k)) for k in (1,2,5)])
sha(p)=bytes2hex(sha256(read(p)))
function original(rep)
    only(filter(r->r.method.id=="Green",deserialize(joinpath(SOURCE,"checkpoints/search_$(ID)_$rep.jls")).runs))
end

function initialize(args)
    all(a in ("--quick-test","--resume","--no-plots","--plots-only","--tests-only") for a in args)||error("Unknown argument")
    quick="--quick-test" in args;out=quick ? joinpath(OUTPUT,"quick_test") : OUTPUT
    d=Dict{String,Any}("output"=>out,"quick"=>quick,"max_iters"=>quick ? 6 : 100,
        "n_paths"=>500,"jobs"=>45,"rep_ids"=>quick ? [1] : [1,2,3],
        "trace_paths"=>quick ? 100 : 1000,"selection_paths"=>quick ? 100 : 10000,
        "holdout_paths"=>quick ? 500 : 50000)
    for folder in ("configs","tables","checkpoints","validation","figures","reproduction","logs")
        mkpath(joinpath(out,folder))
    end
    old=TOML.parsefile(joinpath(SOURCE,"configs/experiment.toml"))["science"]
    @assert old["max_iters"]==100&&old["n_paths"]==500&&old["jobs"]==45
    @assert old["Z0"]==C.Z0==[.12,.10]&&old["BASE_STEP"]==C.BASE_STEP==.035&&old["GRAD_CLIP"]==C.GRAD_CLIP==15.
    for (file,hash) in old["sources"];@assert sha(joinpath(ROOT,file))==hash "Changed original science: $file";end
    model=TOML.parsefile(joinpath(SOURCE,"configs/main_instance_parameters.toml"))["model"]
    cfg=H.from_dict(model);@assert R.model_dict(cfg)==model&&length(C.build_policy_grid())==144
    files=["016probing_green_bank_variants.jl","PROTOCOLO_green_bank_variants_016.txt"]
    science=Dict(k=>v for (k,v) in d if k!="output")
    merge!(science,Dict("methods"=>[m.id for m in METHODS],"Z0"=>C.Z0,"BASE_STEP"=>C.BASE_STEP,
        "GRAD_CLIP"=>C.GRAD_CLIP,"model_sha256"=>R.model_digest(cfg),"julia_version"=>string(VERSION),
        "objective"=>"max_i E_z[Y_i]-alpha'z; common ex-ante policy",
        "sources"=>Dict(f=>sha(joinpath(ROOT,f)) for f in files)))
    fp=R.digest(science);p=joinpath(out,"configs/experiment.toml")
    isfile(p)&&(@assert TOML.parsefile(p)["fingerprint"]==fp "Changed implementation/configuration: use a new output")
    open(p,"w") do io;TOML.print(io,Dict("fingerprint"=>fp,"science"=>science);sorted=true);end
    for (a,b) in (("configs/main_instance_parameters.toml","configs/I0252.toml"),
        ("configs/policies.csv","configs/policies.csv"),("high/I0252_surface.csv","configs/I0252_surface.csv"))
        cp(joinpath(SOURCE,a),joinpath(out,b);force=true)
    end
    for f in files;cp(joinpath(ROOT,f),joinpath(out,"reproduction",f);force=true);end
    d,cfg
end

"""The archived, target-z score and clipped LR; no normalization by observed LR."""
function bank_kernel(z,hb,cfg)
    hb.scenario==ID&&hb.model_sha256==R.model_digest(cfg)||error("Invalid history provenance")
    ll,S=R.bank_scores(hb.bank,z,hb.zgen,cfg)
    all(isfinite,ll)&&all(isfinite,S)&&all(isfinite,hb.Y)||error("Nonfinite LR/score/reward")
    L=exp.(clamp.(ll,-35.,35.))
    (L=L,S=S,clips=count(x->abs(x)>35.,ll))
end
function contribution_variance(hb,kernel,policy)
    n=length(hb.bank);n>=2||error("Sample variance requires >=2 paths")
    y=collect(view(hb.Y,policy,:)).*kernel.L
    all(isfinite,y)||error("Nonfinite LR-reweighted objective contribution")
    v=var(y;corrected=true)
    isfinite(v)&&v>=0||error("Invalid empirical objective variance: $v")
    v
end

"""Returns the estimator plus an explicit bank selection plan. Equal-bank
cases delegate to the original reduction, including its exact summation order.
The variance filter is adaptive; no claim of unbiasedness is made for it.
"""
function estimate(z,history,cfg,m;force_keep_all=false)
    T=length(history);T>0||error("Empty history")
    indices=collect(1:T);cache=Dict{Int,Any}();vf=fill(NaN,T)
    provisional=0;threshold=NaN;current_var=NaN
    if m.family==:rolling
        W=Int(m.param);W>=1||error("Invalid window")
        indices=collect(max(1,T-W+1):T)
    elseif m.family==:filtered
        isfinite(m.param)&&m.param>=0||error("Invalid kappa")
        # Step 2 explicitly uses ORIGINAL full Green, before bank rejection.
        full=R.green_data(z,history,cfg,ID);provisional=full.best
        for h in 1:T
            cache[h]=bank_kernel(z,history[h],cfg)
            # Invalid variances stop explicitly: they are never imputed.
            vf[h]=contribution_variance(history[h],cache[h],provisional)
        end
        current_var=vf[T];threshold=m.param*current_var
        isfinite(threshold)&&threshold>=0||error("Invalid filtering threshold")
        # <= explicitly retains zero-variance ties when current variance is zero.
        indices=force_keep_all ? collect(1:T) : findall(v->v<=threshold,vf)
        T in indices||push!(indices,T);sort!(indices)
    elseif !(m.family in (:original,:exponential))
        error("Unknown family")
    end
    weights=fill(1/length(indices),length(indices))
    if m.family==:exponential
        rho=m.param;isfinite(rho)&&0<rho<=1||error("rho must be in (0,1]")
        weights=rho.^(T .- indices);weights./=sum(weights)
    end
    all(isfinite,weights)&&all(weights.>=0)&&isapprox(sum(weights),1.;atol=1e-14)||error("Invalid bank weights")
    @assert T in indices
    if m.family!=:exponential||m.param==1.
        data=R.green_data(z,history[indices],cfg,ID)
    else
        np=size(history[1].Y,1);values=zeros(np);G=zeros(np,2)
        weightedL=0.;weightedL2=0.;clips=0
        for (h,a) in zip(indices,weights)
            hb=history[h];kernel=bank_kernel(z,hb,cfg);cache[h]=kernel
            L,S=kernel.L,kernel.S;n=length(L);clips+=kernel.clips
            for i in 1:np;values[i]+=a*mean(view(hb.Y,i,:).*L);end
            G .+=a*(hb.Y*transpose(S.*transpose(L)))/n
            weightedL+=a*sum(L)/n;weightedL2+=a^2*sum(abs2,L)/n^2
        end
        best=argmax(values);g=-collect(cfg.alpha)
        for (h,a) in zip(indices,weights),j in 1:2
            g[j]+=a*mean(view(history[h].Y,best,:).*cache[h].L.*view(cache[h].S,j,:))
        end
        data=(values=values,G=G,hard=g,best=best,ess=weightedL^2/weightedL2,banks=T,lr_clips=clips)
    end
    all(isfinite,data.values)&&all(isfinite,data.hard)||error("Nonfinite estimate")
    (data=data,indices=indices,weights=weights,cache=cache,filter_variances=vf,
        provisional=provisional,threshold=threshold,current_var=current_var)
end

"""Optional contribution moments, outside optimization timing. Per-bank sample
variances and sum a_h^2*s_h^2/n_h are descriptive under adaptive reuse, not
valid unconditional MC variance estimates for the adaptive optimizer."""
function diagnostics(z,history,cfg,est)
    T=length(history);selected=Set(est.indices);policy=est.data.best
    rows=NamedTuple[];vobj=0.;vg=zeros(2);vm=0.;vgm=zeros(2);meanobj=0.;grad=-collect(cfg.alpha)
    sumL=0.;sumL2=0.;weightedL=0.;weightedL2=0.;clips=0
    for h in 1:T
        keep=h in selected;a=keep ? est.weights[findfirst(==(h),est.indices)] : 0.
        # Window-excluded banks were not queried by the estimator.
        assessed=keep||est.provisional>0
        if assessed
            kernel=get!(est.cache,h) do;bank_kernel(z,history[h],cfg);end
            n=length(kernel.L);y=collect(view(history[h].Y,policy,:)).*kernel.L
            gy=[y.*view(kernel.S,j,:) for j in 1:2]
            vo=var(y);vv=[var(gy[j]) for j in 1:2]
            all(isfinite,[vo;vv])||error("Nonfinite diagnostic variance")
            if keep
                vobj+=a*vo;vg.+=a.*vv;vm+=a^2*vo/n;vgm.+=a^2 .*vv./n
                meanobj+=a*mean(y)
                for j in 1:2;grad[j]+=a*mean(gy[j]);end
                sumL+=sum(kernel.L);sumL2+=sum(abs2,kernel.L);clips+=kernel.clips
                weightedL+=a*sum(kernel.L)/n;weightedL2+=a^2*sum(abs2,kernel.L)/n^2
            end
            if h==T;@assert all(==(1.),kernel.L);end
            om=mean(y);g3m=mean(gy[1]);g4m=mean(gy[2]);meanlr=mean(kernel.L)
            localess=sum(kernel.L)^2/sum(abs2,kernel.L)
        else
            vo=missing;vv=[missing,missing];om=missing;g3m=missing;g4m=missing;meanlr=missing;localess=missing
        end
        fv=est.provisional>0 ? est.filter_variances[h] : missing
        reason=keep ? (h==T&&est.provisional>0&&fv>est.threshold ? "current_always_retained" : "retained") :
            (est.provisional>0 ? "variance_above_threshold" : "outside_window")
        push!(rows,(bank_iteration=h,bank_age=T-h,bank_z3=history[h].zgen[1],bank_z4=history[h].zgen[2],
            paths=length(history[h].bank),retained=keep,bank_weight=a,is_current=h==T,reason=reason,
            selected_policy=policy,objective_contribution_mean=om,objective_contribution_sample_variance=vo,
            gradient3_contribution_mean=g3m,gradient4_contribution_mean=g4m,
            gradient3_contribution_sample_variance=vv[1],gradient4_contribution_sample_variance=vv[2],
            mean_lr=meanlr,bank_lr_ess=localess,provisional_policy=est.provisional,
            filter_sample_variance=fv,filter_threshold=est.provisional>0 ? est.threshold : missing))
    end
    @assert isapprox(meanobj,est.data.values[policy];atol=1e-9,rtol=1e-12)
    @assert all(isapprox.(grad,est.data.hard;atol=1e-8,rtol=1e-12)) "Policy/weight gradient mismatch"
    @assert clips==est.data.lr_clips
    ess=weightedL^2/weightedL2
    @assert isapprox(ess,est.data.ess;rtol=1e-12)
    (rows=rows,objective_variance=vobj,gradient_variance=vg,objective_mean_variance_proxy=vm,
        gradient_mean_variance_proxy=vgm,ess=ess,raw_lr_ess=sumL^2/sumL2)
end

function search(d,cfg,rep,m;K=d["max_iters"],n=d["n_paths"],extra=true,check_original=false)
    policies=C.build_policy_grid();history=R.HistoricalBank[];zhist=[copy(C.Z0)]
    core=NamedTuple[];rows=NamedTuple[];bankrows=NamedTuple[];elapsed=0.;diagnostic_elapsed=0.
    for k in 1:K
        ts=time_ns();z=zhist[end]
        p=R.primitive_bank(cfg,n,X.seed(ID,"compare",rep,"search",k);njobs=d["jobs"])
        bank=R.materialize_bank(p,z,cfg);Y=R.rewards(policies,bank,cfg)
        # Preserve the local-data work performed by the original execution path.
        R.local_data(z,bank,Y,cfg)
        push!(history,R.historical_bank(ID,cfg,z,bank,Y))
        est=estimate(z,history,cfg,m);data=est.data;g=data.hard
        znew,eta,gc=R.projected_step(z,g,k)
        record=(iteration=k,g3=g[1],g4=g[2],step_norm=norm(znew-z),gradient_norm=norm(g),
            clipping=count(x->abs(x)>C.GRAD_CLIP,g),projection=norm(znew-(z+eta*gc)),
            ess=data.ess,lr_clips=data.lr_clips,hard_policy=data.best,K_eff=1.,
            objective_internal=maximum(data.values)-R.cost(cfg,z))
        push!(core,record);push!(zhist,znew)
        seconds=(time_ns()-ts)/1e9;elapsed+=seconds
        if extra
            a=time_ns();dg=diagnostics(z,history,cfg,est);ds=(time_ns()-a)/1e9;diagnostic_elapsed+=ds
            append!(bankrows,[merge((rep=rep,method=m.id,iteration=k),r) for r in dg.rows])
            retained=length(est.indices);nret=sum(length(history[h].bank) for h in est.indices)
            push!(rows,merge((scenario=ID,rep=rep,method=m.id,family=string(m.family),parameter=m.param,
                target_z3=z[1],target_z4=z[2],next_z3=znew[1],next_z4=znew[2]),record,
                (objective_policy=data.best,gradient_policy=data.best,probing_cost=R.cost(cfg,z),
                generated_banks=k,retained_banks=retained,new_paths=n,cumulative_new_paths=n*k,
                retained_observations=nret,reused_observations=nret-n,bank_weights=join(est.weights,";"),
                retained_bank_indices=join(est.indices,";"),ess_fraction=data.ess/nret,raw_lr_ess=dg.raw_lr_ess,
                objective_contribution_variance=dg.objective_variance,
                gradient3_contribution_variance=dg.gradient_variance[1],gradient4_contribution_variance=dg.gradient_variance[2],
                objective_mean_variance_proxy=dg.objective_mean_variance_proxy,
                gradient3_mean_variance_proxy=dg.gradient_mean_variance_proxy[1],gradient4_mean_variance_proxy=dg.gradient_mean_variance_proxy[2],
                provisional_policy=est.provisional,filter_current_variance=est.provisional>0 ? est.current_var : missing,
                filter_threshold=est.provisional>0 ? est.threshold : missing,
                search_seconds=seconds,cumulative_search_seconds=elapsed,diagnostic_seconds=ds,
                cumulative_diagnostic_seconds=diagnostic_elapsed)))
        end
    end
    if check_original
        old=original(rep)
        @assert isequal(zhist,old.zhist[1:K+1])&&isequal(core,old.diagnostics[1:K]) "Original control changed"
    end
    (method=m,rep=rep,zhist=zhist,core=core,rows=rows,bankrows=bankrows,
        search_seconds=elapsed,diagnostic_seconds=diagnostic_elapsed,iterations=K,new_paths=n*K)
end

function tests(d,cfg)
    history=R.HistoricalBank[];policies=C.build_policy_grid()
    summary=Dict{String,Any}();res=@testset "Green bank variants: deterministic invariants" begin
        for k in 1:4
            z=[.12+.1k,.10+.08k];p=R.primitive_bank(cfg,20,X.seed(ID,"compare",1,"search",k);njobs=45)
            bank=R.materialize_bank(p,z,cfg);Y=R.rewards(policies,bank,cfg)
            push!(history,R.historical_bank(ID,cfg,z,bank,Y));old=R.green_data(z,history,cfg,ID)
            @test all(==(1.),bank_kernel(z,history[end],cfg).L)
            for m in ((id="wide",family=:rolling,param=100.),(id="rho1",family=:exponential,param=1.),(id="all",family=:filtered,param=1.))
                e=estimate(z,history,cfg,m;force_keep_all=true)
                @test isequal(e.data,old)
            end
            for m in METHODS
                e=estimate(z,history,cfg,m);dg=diagnostics(z,history,cfg,e)
                @test k in e.indices
                @test all(e.weights.>=0)&&isapprox(sum(e.weights),1.;atol=1e-14)
                @test e.data.best==argmax(e.data.values)
                @test isfinite(dg.objective_variance)&&all(isfinite,dg.gradient_variance)
            end
        end
        # Zero rewards give exactly zero variances. <= retains ties explicitly.
        zeros_history=[R.historical_bank(ID,cfg,h.zgen,h.bank,zeros(size(h.Y))) for h in history]
        e=estimate(history[end].zgen,zeros_history,cfg,(family=:filtered,param=1.))
        @test e.current_var==0. && e.threshold==0. && e.indices==collect(1:4)
        @test isequal(e.data,R.green_data(history[end].zgen,zeros_history,cfg,ID))
        # An invalid reward must raise, not turn into a fabricated finite variance.
        bad=deepcopy(zeros_history);bad[1].Y[1,1]=NaN
        @test_throws ErrorException estimate(history[end].zgen,bad,cfg,(family=:filtered,param=1.))
        @test_throws ErrorException estimate(history[end].zgen,history,cfg,(family=:exponential,param=0.))
        for m in METHODS
            u=search(d,cfg,1,m;K=4,n=20,extra=false)
            v=search(d,cfg,1,m;K=4,n=20,extra=false)
            @test isequal(u.zhist,v.zhist)&&isequal(u.core,v.core)
        end
        # Full-size small replay also compiles all estimators before measured runs.
        for m in METHODS
            search(d,cfg,1,m;K=3,n=500,extra=true,check_original=m.family==:original)
        end
    end
    summary["passed"]=true;summary["tests_passed"]=Test.get_test_counts(res).passes
    summary["zero_variance_rule"]="retain ties <= threshold, always current"
    summary["nonfinite_rule"]="explicit error; never impute or observation-filter"
    summary["full_size_control_prefix_exact"]=true
    open(joinpath(d["output"],"validation/deterministic_tests.toml"),"w") do io;TOML.print(io,summary);end
    summary
end

function searches(d,cfg)
    runs=Dict{Tuple{Int,String},Any}();allrows=NamedTuple[];allbanks=NamedTuple[];coords=NamedTuple[]
    seedrows=NamedTuple[]
    for rep in d["rep_ids"],m in METHODS
        path=joinpath(d["output"],"checkpoints/search_$(rep)_$(m.id).jls")
        if isfile(path);v=deserialize(path)
        else
            GC.gc();v=search(d,cfg,rep,m;check_original=m.family==:original)
            E.atomic_serialize(path,v)
        end
        @assert v.method==m&&v.iterations==d["max_iters"]&&v.new_paths==500*d["max_iters"]
        if m.family==:original
            old=original(rep);@assert isequal(v.zhist,old.zhist[1:d["max_iters"]+1])&&isequal(v.core,old.diagnostics[1:d["max_iters"]])
        end
        qp=joinpath(OUTPUT,"quick_test/checkpoints/search_$(rep)_$(m.id).jls")
        if !d["quick"]&&isfile(qp)
            q=deserialize(qp);@assert isequal(v.zhist[1:q.iterations+1],q.zhist)&&isequal(v.core[1:q.iterations],q.core)
        end
        for (j,z) in enumerate(v.zhist);push!(coords,(rep=rep,method=m.id,family=string(m.family),iteration=j-1,z3=z[1],z4=z[2]));end
        for k in 1:d["max_iters"]
            ss=X.seed(ID,"compare",rep,"search",k)
            archive=deserialize(joinpath(SOURCE,"checkpoints/search_$(ID)_$rep.jls"))
            @assert only(filter(r->r.stream=="search"&&r.iteration==k,archive.seeds)).seed==ss
            push!(seedrows,(rep=rep,method=m.id,iteration=k,seed=ss))
        end
        append!(allrows,v.rows);append!(allbanks,v.bankrows);runs[(rep,m.id)]=v
        @printf("Search completed: rep=%d %-12s z=(%.6f,%.6f) time=%.3fs\n",rep,m.id,v.zhist[end]...,v.search_seconds);flush(stdout)
    end
    E.write_csv(joinpath(d["output"],"tables/iterations.csv"),allrows)
    E.write_csv(joinpath(d["output"],"tables/banks.csv"),allbanks)
    E.write_csv(joinpath(d["output"],"tables/coordinates.csv"),coords)
    E.write_csv(joinpath(d["output"],"configs/search_seeds.csv"),seedrows)
    runs
end

function trace_point(d,cfg,rep,z,bank,ss)
    key=bytes2hex(sha256(repr(z)));path=joinpath(d["output"],"checkpoints/trace_$(rep)_$(key).jls")
    isfile(path)&&return deserialize(path)
    donor=joinpath(DONOR,"checkpoints/point_main_$(rep)_$(key).jls")
    if !d["quick"]&&isfile(donor)
        p=deserialize(donor);@assert p.z==z&&p.n_paths==length(bank)
        v=(z=z,objective=p.objective,se=p.se,policy=p.policy,n=length(bank),seed=ss,source="exact independent 012 point reuse")
    else
        Y=R.rewards(C.build_policy_grid(),R.materialize_bank(bank,z,cfg),cfg)
        vals=vec(mean(Y;dims=2));i=argmax(vals)
        v=(z=z,objective=vals[i]-R.cost(cfg,z),se=std(view(Y,i,:))/sqrt(length(bank)),policy=i,n=length(bank),seed=ss,source="direct independent trace")
    end
    E.atomic_serialize(path,v);v
end
function evaluate_rep(d,cfg,rep,runs)
    path=joinpath(d["output"],"checkpoints/evaluation_$rep.jls");isfile(path)&&return deserialize(path)
    policies=C.build_policy_grid();ts=X.seed(ID,"compare",rep,"trace")
    bank=R.primitive_bank(cfg,d["trace_paths"],ts;njobs=45);traces=NamedTuple[]
    for m in METHODS
        for (j,z) in enumerate(runs[(rep,m.id)].zhist)
            p=trace_point(d,cfg,rep,z,bank,ts)
            push!(traces,(rep=rep,method=m.id,iteration=j-1,z3=z[1],z4=z[2],objective=p.objective,
                conditional_se=p.se,selected_policy=p.policy,paths=p.n,seed=p.seed,source=p.source))
        end
        println("Direct trace ready: rep=$rep $(m.id)");flush(stdout)
    end
    # Independent selector followed by untouched holdout avoids reporting a
    # max of holdout means as an unbiased selected-policy evaluation.
    ss=X.seed(ID,"compare",rep,"selection");hs=X.seed(ID,"compare",rep,"holdout")
    @assert length(unique([ts,ss,hs]))==3
    training=[X.seed(ID,"compare",rep,"search",k) for k in 1:d["max_iters"]]
    @assert isempty(intersect(training,[ts,ss,hs]))
    selector=R.primitive_bank(cfg,d["selection_paths"],ss;njobs=45)
    hold=R.primitive_bank(cfg,d["holdout_paths"],hs;njobs=45)
    finals=NamedTuple[];pairs=NamedTuple[];vectors=Dict{String,Vector{Float64}}()
    for m in METHODS
        cp=joinpath(d["output"],"checkpoints/final_$(rep)_$(m.id).jls")
        if isfile(cp);v=deserialize(cp)
        else
            run=runs[(rep,m.id)];z=run.zhist[end];cost=R.cost(cfg,z)
            Ys=R.rewards(policies,R.materialize_bank(selector,z,cfg),cfg);chosen=argmax(vec(mean(Ys;dims=2)))
            # Only the independently selected policy is evaluated on holdout.
            raw=[R.evaluate_policy(policies[chosen],p,cfg) for p in R.materialize_bank(hold,z,cfg)]
            obj=mean(raw)-cost;se=std(raw)/sqrt(length(raw))
            r=(scenario=ID,rep=rep,method=m.id,family=string(m.family),parameter=m.param,z3=z[1],z4=z[2],
                selected_policy=chosen,objective=obj,evaluation_se=se,ci95_low=obj-1.959963984540054se,
                ci95_high=obj+1.959963984540054se,probing_cost=cost,selection_paths=length(selector),holdout_paths=length(hold),
                selection_seed=ss,holdout_seed=hs,optimization_seconds=run.search_seconds,
                diagnostic_seconds=run.diagnostic_seconds,iterations=run.iterations,new_search_paths=run.new_paths)
            if m.family==:original&&!d["quick"]
                old=deserialize(joinpath(SOURCE,"checkpoints/evaluation_$(ID)_$rep.jls"))
                o=only(filter(x->x.method=="Green",old.finals))
                @assert chosen==o.selected_policy&&[o.z3,o.z4]==z
                @assert isapprox(obj,o.objective_holdout;atol=1e-10)&&isapprox(se,o.evaluation_se;atol=1e-12)
            end
            v=(row=r,rewards=raw.-cost);E.atomic_serialize(cp,v)
        end
        push!(finals,v.row);vectors[m.id]=v.rewards
    end
    for m in METHODS
        m.id=="Green"&&continue;delta=vectors[m.id]-vectors["Green"];se=std(delta)/sqrt(length(delta));mu=mean(delta)
        push!(pairs,(rep=rep,method=m.id,reference="Green",difference=mu,paired_mc_se=se,
            ci95_low=mu-1.959963984540054se,ci95_high=mu+1.959963984540054se,holdout_paths=length(delta)))
    end
    v=(traces=traces,finals=finals,paired=pairs);E.atomic_serialize(path,v)
    println("Independent final selection/holdout complete: rep=$rep");flush(stdout);v
end
function main(args=ARGS)
    BLAS.set_num_threads(1)
    d,cfg=initialize(args)
    if !("--plots-only" in args)
        tests(d,cfg)
        "--tests-only" in args&&return
        runs=searches(d,cfg)
        results=Vector{Any}(undef,length(d["rep_ids"]))
        # Timing is complete before these independent, parallel evaluations.
        Threads.@threads :dynamic for j in eachindex(d["rep_ids"])
            results[j]=evaluate_rep(d,cfg,d["rep_ids"][j],runs)
        end
        for (name,field) in (("objective_traces",:traces),("final_results",:finals),("paired_final_vs_green",:paired))
            E.write_csv(joinpath(d["output"],"tables/$name.csv"),vcat([getproperty(v,field) for v in results]...))
        end
        open(joinpath(d["output"],"validation/science_complete.toml"),"w") do io
            TOML.print(io,Dict("complete"=>true,"searches"=>length(runs),"new_search_paths"=>sum(v.new_paths for v in values(runs)),
                "original_controls_exact"=>true,"searches_sequential"=>true,"blas_threads"=>BLAS.get_num_threads(),
                "lr_score_source"=>"unchanged probing_regime_model.jl","surface_recomputed"=>false))
        end
    end
    if !("--no-plots" in args)
        run(`python3 $(joinpath(ROOT,"analyze_green_bank_variants_016.py")) --output $(d["output"])`)
    end
end
end
if abspath(PROGRAM_FILE)==@__FILE__;ProbingGreenBankVariants.main();end
