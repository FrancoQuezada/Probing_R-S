# Included in ProbingAnytimeBudget. Only timing/accounting orchestration is new.
# Every estimator, matrix reduction and projected update calls unchanged 010/R.
function timed_method(d,cfg,rep,m;K=d["max_iters"],expected=nothing)
    policies=C.build_policy_grid();n=d["n_paths"];history=R.HistoricalBank[]
    zhist=[copy(C.Z0)];diagnostics=NamedTuple[];rows=NamedTuple[]
    cumulative=zeros(8);total_new=0;total_policy=0
    # Phase order: primitive generation, materialization, policy simulation,
    # gradient, bootstrap, projected update, bookkeeping/other, total wall time.
    initial=zeros(8);fixedbank=nothing;fixedY=nothing
    if m.kind==:fixed
        a=time_ns();p=R.primitive_bank(cfg,n,X.seed(ID,"compare",rep,"fixed");njobs=d["jobs"]);b=time_ns()
        fixedbank=R.materialize_bank(p,[0.,0.],cfg);c=time_ns()
        fixedY=R.rewards(policies,fixedbank,cfg);e=time_ns()
        initial[1:3].=[b-a,c-b,e-c]./1e9;initial[8]=(e-a)/1e9
    end
    function row(k,new,policy,banks,lr,bootdraws,phase)
        (scenario=ID,rep=rep,cohort=cohort(rep),method=m.id,iteration=k,
         new_trajectories=new,cumulative_trajectories=total_new,history_banks=banks,
         new_policy_path_evaluations=policy,cumulative_policy_path_evaluations=total_policy,
         lr_path_evaluations=lr,bootstrap_path_draws=bootdraws,
         generation_seconds=phase[1],materialization_seconds=phase[2],policy_seconds=phase[3],
         gradient_seconds=phase[4],bootstrap_seconds=phase[5],update_seconds=phase[6],
         other_seconds=phase[7],search_seconds=phase[8],
         cumulative_search_seconds=cumulative[8],cumulative_gradient_seconds=cumulative[4],
         cumulative_bootstrap_seconds=cumulative[5],timing_source="012 sequential exact replay")
    end
    push!(rows,row(0,0,0,0,0,0,zeros(8)))
    for k in 1:K
        phase=zeros(8);a=time_ns();z=zhist[end];payload=nothing
        if m.kind==:fixed
            ts=time_ns();data=X.fixed_data(z,fixedbank,fixedY,cfg);phase[4]+=(time_ns()-ts)/1e9
        else
            ts=time_ns();p=R.primitive_bank(cfg,n,X.seed(ID,"compare",rep,"search",k);njobs=d["jobs"]);phase[1]+=(time_ns()-ts)/1e9
            ts=time_ns();bank=R.materialize_bank(p,z,cfg);phase[2]+=(time_ns()-ts)/1e9
            ts=time_ns();Y=R.rewards(policies,bank,cfg);phase[3]+=(time_ns()-ts)/1e9
            ts=time_ns();localdata=R.local_data(z,bank,Y,cfg);phase[4]+=(time_ns()-ts)/1e9
            payload=(bank=bank,Y=Y,data=localdata)
            if m.kind in (:green,:green_softmax)
                ts=time_ns();push!(history,R.historical_bank(ID,cfg,z,payload.bank,payload.Y))
                data=R.green_data(z,history,cfg,ID);phase[4]+=(time_ns()-ts)/1e9
            else;data=payload.data;end
        end
        ts=time_ns()
        if m.kind in (:fixed,:fresh,:green)
            g=data.hard;w=zeros(144);w[data.best]=1.
            phase[4]+=(time_ns()-ts)/1e9
        elseif m.kind==:psoft
            w=R.bootstrap_weights(payload.Y,d["bootstrap_reps"],MersenneTwister(X.seed(ID,"compare",rep,"bootstrap",k)))
            phase[5]+=(time_ns()-ts)/1e9
            ts=time_ns();g=R.combine(data.G,w,cfg);phase[4]+=(time_ns()-ts)/1e9
        else
            w=R.softmax_weights(data.values;beta=m.beta);g=R.combine(data.G,w,cfg)
            phase[4]+=(time_ns()-ts)/1e9
        end
        all(isfinite,g)||error("Nonfinite replay gradient")
        ts=time_ns();znew,eta,gc=R.projected_step(z,g,k);phase[6]+=(time_ns()-ts)/1e9
        push!(diagnostics,(iteration=k,g3=g[1],g4=g[2],step_norm=norm(znew-z),gradient_norm=norm(g),
            clipping=count(x->abs(x)>C.GRAD_CLIP,g),projection=norm(znew-(z+eta*gc)),
            ess=data.ess,lr_clips=data.lr_clips,hard_policy=data.best,K_eff=1/sum(abs2,w),
            objective_internal=maximum(data.values)-R.cost(cfg,z)))
        push!(zhist,znew)
        phase[8]=(time_ns()-a)/1e9
        if k==1;phase .+=initial;end
        phase[7]=max(0.,phase[8]-sum(phase[1:6]))
        cumulative .+=phase
        new=m.kind==:fixed ? (k==1 ? length(fixedbank) : 0) : length(payload.bank)
        total_new+=new;policy=m.kind==:fixed ? (k==1 ? length(fixedY) : 0) : length(payload.Y);total_policy+=policy
        green=m.kind in (:green,:green_softmax)
        banks=green ? length(history) : 1
        lr=m.kind==:fixed ? length(fixedbank) : length(payload.bank)+(green ? sum(length(hb.bank) for hb in history) : 0)
        push!(rows,row(k,new,policy,banks,lr,m.kind==:psoft ? n*d["bootstrap_reps"] : 0,phase))
    end
    if expected!==nothing
        @assert isequal(zhist,expected.zhist[1:K+1]) "Replay decisions differ from original"
        for k in 1:K
            @assert isequal(diagnostics[k],expected.diagnostics[k]) "Replay diagnostics differ at $k"
        end
    end
    (run=(method=m,zhist=zhist,diagnostics=diagnostics),rows=rows,
     exact_reproduction=expected!==nothing,iterations=K)
end
function timing_phase(d,cfg,archives)
    warmfile=joinpath(d["output"],"logs/warmup.toml")
    # Re-evaluating an existing experiment needs no new timing replay.
    reused=NamedTuple[]
    donors=[joinpath(ROOT,"resultados/012probing_anytime_budget"),joinpath(ROOT,"resultados/012probing_anytime_budget/quick_test")]
    compatible_keys=["n_paths","jobs","max_iters","bootstrap_reps","reference_model_sha256","source_experiment_sha256"]
    for rep in d["rep_ids"],m in X.METHODS
        path=joinpath(d["output"],"checkpoints/timing_$(rep)_$(m.id).jls")
        isfile(path)&&continue
        for donor in donors
            from=joinpath(donor,"checkpoints/timing_$(rep)_$(m.id).jls");config=joinpath(donor,"configs/experiment.toml")
            isfile(from)&&isfile(config)||continue
            prior=TOML.parsefile(config)["science"]
            all(get(prior,k,nothing)==d["science"][k] for k in compatible_keys)||continue
            v=deserialize(from);v.iterations==d["max_iters"]||continue
            if rep<=5
                expected=only(filter(r->r.method.id==m.id,archives[rep].runs))
                @assert isequal(v.run.zhist,expected.zhist[1:d["max_iters"]+1])
                @assert isequal(v.run.diagnostics,expected.diagnostics[1:d["max_iters"]])
            end
            cp(from,path);push!(reused,(rep=rep,method=m.id,source=from,sha256=sha(from)))
            println("Compatible timing archive reused: rep=$rep method=$(m.id)");break
        end
    end
    isempty(reused)||E.write_csv(joinpath(d["output"],"configs/reused_timing_archives.csv"),reused)
    pending=any(!isfile(joinpath(d["output"],"checkpoints/timing_$(rep)_$(m.id).jls")) for rep in d["rep_ids"] for m in X.METHODS)
    # Warm only when an actual new timing measurement remains; costs excluded.
    if pending
    for m in X.METHODS
        expected=only(filter(r->r.method.id==m.id,archives[1].runs))
        timed_method(d,cfg,1,m;K=2,expected=expected)
    end
    open(warmfile,"w") do io
        TOML.print(io,Dict("methods"=>7,"updates_per_method"=>2,"n_paths"=>d["n_paths"],
            "excluded_from_budgets_and_timings"=>true,"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
            "cpu"=>Sys.CPU_NAME,"julia"=>string(VERSION),"measurement"=>"sequential methods, single process, warmed; OS/GC not isolated"))
    end
    else
        println("All timing archives available: no warmup or search replay required.")
    end
    allrows=NamedTuple[];results=Dict{Int,Any}();checks=NamedTuple[]
    for rep in d["rep_ids"]
        runs=Any[]
        for idx in circshift(collect(eachindex(X.METHODS)),mod(rep-1,length(X.METHODS)))
            m=X.METHODS[idx];path=joinpath(d["output"],"checkpoints/timing_$(rep)_$(m.id).jls")
            if isfile(path);v=deserialize(path)
            else
                GC.gc();expected=rep<=5 ? only(filter(r->r.method.id==m.id,archives[rep].runs)) : nothing
                v=timed_method(d,cfg,rep,m;expected=expected);E.atomic_serialize(path,v)
                println("timing rep=$rep method=$(m.id) seconds=$(round(last(v.rows).cumulative_search_seconds;digits=3)) exact=$(v.exact_reproduction)");flush(stdout)
            end
            push!(runs,v.run);append!(allrows,v.rows)
            push!(checks,(rep=rep,cohort=cohort(rep),method=m.id,exact_original=v.exact_reproduction,
                decisions=length(v.run.zhist),gradients=length(v.run.diagnostics)))
        end
        results[rep]=(runs=sort(runs;by=r->findfirst(m->m.id==r.method.id,X.METHODS)),)
    end
    E.write_csv(joinpath(d["output"],"tables/search_budget_runtime.csv"),allrows)
    E.write_csv(joinpath(d["output"],"validation/exact_replays.csv"),checks)
    results,allrows
end
