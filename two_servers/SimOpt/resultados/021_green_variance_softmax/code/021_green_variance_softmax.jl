# Additive composition: all historical functions execute from frozen 019/020 copies.
include(joinpath(@__DIR__, "019probing_i0252_statistical.jl"))
module GreenVarianceSoftmax021
using ..I0252Statistical
using Statistics, LinearAlgebra, Serialization, TOML, Test, Printf
const S=I0252Statistical; const V=S.V; const R=S.R; const C=S.C; const X=S.X; const E=S.E
const METHOD="green_variance_softmax"
const FILTER=(id="Filter2",family=:filtered,param=2.)
const OUTPUT=Ref("")

function estimate(z,history,cfg)
    # This is the ORIGINAL 016 filter, including provisional full-Green policy.
    est=V.estimate(z,history,cfg,FILTER)
    # Only the accepted-bank policy estimates reach the historical stable softmax.
    w=R.softmax_weights(est.data.values;beta=.5)
    g=R.combine(est.data.G,w,cfg) # G is GROSS reward; subtract alpha exactly once.
    @assert all(isfinite,est.data.G) && all(isfinite,w) && all(w.>=0)
    @assert isapprox(sum(w),1.;atol=1e-14) && all(isfinite,g)
    (est=est,weights=w,gradient=g)
end

function search(d,cfg,seedrep;persist=true)
    K=d["max_iters"];n=d["n_paths"];rep=seedrep-S.REP_OFFSET
    stage=joinpath(OUTPUT[],"raw",METHOD,@sprintf(".rep_%04d.partial",rep))
    statefile=joinpath(stage,"iteration_state.jls")
    fp=persist ? TOML.parsefile(joinpath(OUTPUT[],"config/experiment.toml"))["fingerprint"] : "test"
    history=R.HistoricalBank[];zhist=[copy(C.Z0)];core=NamedTuple[];rows=NamedTuple[]
    bankrows=NamedTuple[];policyrows=NamedTuple[];elapsed=0.;diagnostic_elapsed=0.;first=1
    if persist
        mkpath(joinpath(stage,"banks"))
        if isfile(statefile)
            cp=deserialize(statefile);@assert cp.fingerprint==fp
            zhist=cp.zhist;core=cp.core;rows=cp.rows;bankrows=cp.bankrows;policyrows=cp.policyrows
            elapsed=cp.elapsed;diagnostic_elapsed=cp.diagnostic_elapsed;first=length(core)+1
            for k in 1:first-1
                h=deserialize(joinpath(stage,"banks",@sprintf("bank_%03d.jls",k)))
                @assert h.iteration==k && h.seed==X.seed(V.ID,"compare",seedrep,"search",k)
                @assert h.historical.zgen==zhist[k] && length(h.historical.bank)==n
                push!(history,h.historical)
            end
        end
    end
    policies=C.build_policy_grid()
    for k in first:K
        ts=time_ns();z=zhist[end];seed=X.seed(V.ID,"compare",seedrep,"search",k)
        p=R.primitive_bank(cfg,n,seed;njobs=d["jobs"])
        bank=R.materialize_bank(p,z,cfg);Y=R.rewards(policies,bank,cfg)
        R.local_data(z,bank,Y,cfg) # Same local-data work as the original runner.
        hb=R.historical_bank(V.ID,cfg,z,bank,Y);push!(history,hb)
        result=estimate(z,history,cfg);est=result.est;data=est.data;w=result.weights;g=result.gradient
        znew,eta,gc=R.projected_step(z,g,k)
        record=(iteration=k,g3=g[1],g4=g[2],step_norm=norm(znew-z),gradient_norm=norm(g),
            clipping=count(x->abs(x)>C.GRAD_CLIP,g),projection=norm(znew-(z+eta*gc)),
            ess=data.ess,lr_clips=data.lr_clips,hard_policy=data.best,K_eff=1/sum(abs2,w),
            objective_internal=maximum(data.values)-R.cost(cfg,z))
        push!(core,record);push!(zhist,znew)
        seconds=(time_ns()-ts)/1e9;elapsed+=seconds
        a=time_ns();dg=V.diagnostics(z,history,cfg,est)
        # V.diagnostics describes the accepted estimator's HARD policy, not the mixture.
        append!(bankrows,[merge((replication=rep,method=METHOD,iteration=k),r) for r in dg.rows])
        for i in eachindex(w)
            push!(policyrows,(replication=rep,iteration=k,policy=i,gross_mean=data.values[i],
                gross_g3=data.G[i,1],gross_g4=data.G[i,2],softmax_weight=w[i]))
        end
        ds=(time_ns()-a)/1e9;diagnostic_elapsed+=ds;retained=length(est.indices)
        # Stable LSE is explicitly separate from the archived hard-max internal field.
        vmax=maximum(data.values);lse=vmax+log(sum(exp.(.5 .* (data.values .- vmax))))/.5-R.cost(cfg,z)
        push!(rows,merge((scenario=V.ID,replication=rep,method=METHOD,
            target_z3=z[1],target_z4=z[2],next_z3=znew[1],next_z4=znew[2]),record,
            (probing_cost=R.cost(cfg,z),internal_lse_surrogate=lse,generated_banks=k,
            retained_banks=retained,rejected_banks=k-retained,new_paths=n,cumulative_new_paths=n*k,
            retained_observations=n*retained,retained_bank_indices=join(est.indices,";"),
            bank_weights=join(est.weights,";"),ess_fraction=data.ess/(n*retained),raw_lr_ess=dg.raw_lr_ess,
            provisional_policy=est.provisional,filter_current_variance=est.current_var,filter_threshold=est.threshold,
            objective_contribution_variance=dg.objective_variance,
            diagnostic_variance_policy="accepted hard-max policy, not softmax mixture",
            objective_mean_variance_proxy=dg.objective_mean_variance_proxy,
            search_seconds=seconds,cumulative_search_seconds=elapsed,diagnostic_seconds=ds,
            cumulative_diagnostic_seconds=diagnostic_elapsed)))
        if persist
            E.atomic_serialize(joinpath(stage,"banks",@sprintf("bank_%03d.jls",k)),
                (iteration=k,seed=seed,historical=hb))
            E.atomic_serialize(statefile,(fingerprint=fp,zhist=zhist,core=core,rows=rows,bankrows=bankrows,
                policyrows=policyrows,elapsed=elapsed,diagnostic_elapsed=diagnostic_elapsed))
        end
    end
    if persist
        E.write_csv(joinpath(stage,"bank_diagnostics.csv"),bankrows)
        E.write_csv(joinpath(stage,"policy_estimates.csv"),policyrows)
    end
    (zhist=zhist,core=core,rows=rows,cumulative=[0.;[r.cumulative_search_seconds for r in rows]],
        seconds=elapsed,diagnostic_seconds=diagnostic_elapsed)
end

# More-specific dispatch affects only the new method in this process, never old files.
function S.search(d::Dict,cfg::R.ModelConfig,rep::Int,method::String)
    @assert method==METHOD
    search(d,cfg,rep)
end
S.IDS[METHOD]="GreenVarianceSoftmax021"

function tests(c,cfg)
    results=@testset "021 filter -> accepted estimates -> stable softmax -> update" begin
        history=R.HistoricalBank[];policies=C.build_policy_grid();target=[.73,.81]
        for (k,z) in enumerate(([.02,.03],[.3,.4],target))
            p=R.primitive_bank(cfg,30,12340+k;njobs=45);bank=R.materialize_bank(p,z,cfg)
            push!(history,R.historical_bank(V.ID,cfg,z,bank,R.rewards(policies,bank,cfg)))
        end
        result=estimate(target,history,cfg);e=result.est;old=V.estimate(target,history,cfg,FILTER)
        @test isequal(e,old)
        full=R.green_data(target,history,cfg,V.ID)
        variances=[var(h.Y[full.best,:].*exp.(clamp.(R.bank_scores(h.bank,target,h.zgen,cfg)[1],-35.,35.));corrected=true) for h in history]
        expected=sort!(unique(vcat(findall(v->v<=2variances[end],variances),[3])))
        @test e.indices==expected
        @test 3 in e.indices
        @test isequal(e.data,R.green_data(target,history[e.indices],cfg,V.ID))
        @test all(result.weights.>=0) && sum(result.weights)≈1.
        @test result.weights==R.softmax_weights(e.data.values;beta=.5)
        @test R.softmax_weights([1e6,1e6+2];beta=.5)≈[1.,exp(1)]./(1+exp(1))
        @test all(isfinite,R.softmax_weights([-1e300,1e300];beta=.5))
        manualvalues=zeros(144);manualG=zeros(144,2)
        for h in e.indices
            hb=history[h];ll,score=R.bank_scores(hb.bank,target,hb.zgen,cfg);L=exp.(clamp.(ll,-35.,35.))
            for i in 1:144
                manualvalues[i]+=mean(hb.Y[i,:].*L)/length(e.indices)
                for j in 1:2
                    manualG[i,j]+=mean(hb.Y[i,:].*L.*score[j,:])/length(e.indices)
                end
            end
            # Independent Bernoulli path probability and finite-difference scores.
            path=hb.bank[1];manual_ll=0.
            for job in path.jobs
                job.class in (3,4)||continue
                pt=clamp(R.signal_probability(job.class,job.complex,target[job.class-2],cfg),1e-12,1-1e-12)
                ps=clamp(R.signal_probability(job.class,job.complex,hb.zgen[job.class-2],cfg),1e-12,1-1e-12)
                manual_ll+=job.signal==1 ? log(pt)-log(ps) : log1p(-pt)-log1p(-ps)
            end
            @test ll[1]≈manual_ll
            for j in 1:2
                plus=copy(target);minus=copy(target);plus[j]+=1e-6;minus[j]-=1e-6
                fd=(R.loglr_and_score(path,plus,hb.zgen,cfg)[1]-R.loglr_and_score(path,minus,hb.zgen,cfg)[1])/2e-6
                @test isapprox(fd,score[j,1];atol=1e-6,rtol=1e-6)
            end
        end
        @test e.data.values≈manualvalues
        @test e.data.G≈manualG
        @test result.gradient≈transpose(manualG)*result.weights-collect(cfg.alpha)
        @test all(isfinite,result.gradient)
        zeros_history=[R.historical_bank(V.ID,cfg,h.zgen,h.bank,zeros(size(h.Y))) for h in history]
        zero=estimate(target,zeros_history,cfg)
        @test zero.est.indices==[1,2,3] && zero.est.current_var==0.
        @test zero.gradient≈-collect(cfg.alpha)
        small=merge(c["science"],Dict("max_iters"=>3,"n_paths"=>20))
        u=search(small,cfg,S.REP_OFFSET+1;persist=false)
        v=search(small,cfg,S.REP_OFFSET+1;persist=false)
        @test isequal(u.zhist,v.zhist) && isequal(u.core,v.core)
        @test S.direct_trace(cfg,u.zhist,20,123456,45)==S.direct_trace(cfg,u.zhist,20,123456,45)
        @test isequal(u.zhist,search(small,cfg,S.REP_OFFSET+1;persist=false).zhist)
        # Real checkpoint continuation: first two updates are neither regenerated
        # nor counted twice. This separate validation directory is never analyzed.
        production=OUTPUT[]
        sandbox=mktempdir(joinpath(production,"validation"))
        mkpath(joinpath(sandbox,"config"))
        cp(joinpath(production,"config/experiment.toml"),joinpath(sandbox,"config/experiment.toml"))
        try
            OUTPUT[]=sandbox
            prefix=search(merge(small,Dict("max_iters"=>2)),cfg,S.REP_OFFSET+1)
            stage=joinpath(sandbox,"raw",METHOD,".rep_0001.partial")
            before=[S.sha(joinpath(stage,"banks",@sprintf("bank_%03d.jls",k))) for k in 1:2]
            resumed=search(small,cfg,S.REP_OFFSET+1)
            @test isequal(resumed.zhist,u.zhist) && isequal(resumed.core,u.core)
            @test before==[S.sha(joinpath(stage,"banks",@sprintf("bank_%03d.jls",k))) for k in 1:2]
            @test length(readdir(joinpath(stage,"banks")))==3
        finally
            OUTPUT[]=production
        end
    end
    S.atomic_toml(joinpath(OUTPUT[],"validation/algorithm_tests.toml"),Dict("passed"=>true,
        "assertions"=>Test.get_test_counts(results).passes,"instance"=>V.ID,
        "filter_exact_016"=>true,"finite_difference_scores"=>true,"cost_subtracted_once"=>true))
end

function finish_rep(c,cfg,rep)
    dir=joinpath(OUTPUT[],"raw",METHOD,@sprintf("rep_%04d",rep))
    # Preserve original run_rep atomic publication, then supplement with bank hashes.
    names=["bank_diagnostics.csv","policy_estimates.csv","iteration_state.jls"]
    append!(names,[joinpath("banks",@sprintf("bank_%03d.jls",k)) for k in 1:c["science"]["max_iters"]])
    if V.ID=="I0252" && rep<=3
        name="trace_crn017.csv";path=joinpath(dir,name)
        if !isfile(path)
            raw=deserialize(joinpath(dir,"search.jls")).raw;seed=X.seed(V.ID,"compare",rep,"trace")
            trace=S.direct_trace(cfg,raw.zhist,1000,seed,45)
            E.write_csv(path,[(replication=rep,iteration=k-1,z3=z[1],z4=z[2],objective=p.objective,
                conditional_se=p.se,selected_policy=p.policy,evaluation_seed=seed,trace_paths=1000)
                for (k,(z,p)) in enumerate(zip(raw.zhist,trace))])
        end
        push!(names,name)
    end
    S.atomic_toml(joinpath(dir,"supplement_complete.toml"),Dict("fingerprint"=>c["fingerprint"],
        "status"=>"complete","files"=>Dict(name=>S.sha(joinpath(dir,name)) for name in names)))
end

function main()
    BLAS.set_num_threads(1);OUTPUT[]=abspath(ARGS[1]);c,cfg=S.model(OUTPUT[])
    if length(ARGS)>1 && ARGS[2]=="test";tests(c,cfg);return;end
    warm=merge(c["science"],Dict("max_iters"=>2,"n_paths"=>20))
    search(warm,cfg,S.REP_OFFSET;persist=false)
    S.direct_trace(cfg,[C.Z0],20,12345,45)
    S.endpoint(cfg,C.Z0,merge(warm,Dict("selection_paths"=>20,"holdout_paths"=>20)),S.seeds(0,2))
    println("READY");flush(stdout)
    for line in eachline(stdin)
        isempty(line)&&continue;rep=parse(Int,line)
        try
            dir=joinpath(OUTPUT[],"raw",METHOD,@sprintf("rep_%04d",rep))
            if !isdir(dir);S.run_rep(OUTPUT[],c,cfg,METHOD,rep);end
            finish_rep(c,cfg,rep)
            println("COMPLETE\t$rep");flush(stdout)
        catch err
            showerror(stdout,err,catch_backtrace());println();println("FAILED\t$rep");flush(stdout)
        end
        GC.gc()
    end
end
end
if abspath(PROGRAM_FILE)==@__FILE__;GreenVarianceSoftmax021.main();end
