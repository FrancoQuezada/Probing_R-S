# Included in ProbingAnytimeBudget. Only timing/accounting orchestration is new.
# Every estimator, matrix reduction and projected update calls unchanged 010/R.
function timed_search(d,cfg,rep,m;K=d["max_iters"],expected=nothing)
    policies=C.build_policy_grid();n=d["n_paths"];history=R.HistoricalBank[]
    zhist=[copy(C.Z0)];diagnostics=NamedTuple[];rows=NamedTuple[]
    cumulative=zeros(8);total_new=0;total_policy=0
    # Phase order: primitive generation, materialization, policy simulation,
    # gradient, bootstrap, projected update, bookkeeping/other, total wall time.
    initial=zeros(8);fixedbank=nothing;fixedY=nothing;cumulative_reuse=0.
    if m.kind==:fixed
        a=time_ns();p=R.primitive_bank(cfg,n,X.seed(ID,"compare",rep,"fixed");njobs=d["jobs"]);b=time_ns()
        fixedbank=R.materialize_bank(p,[0.,0.],cfg);c=time_ns()
        fixedY=R.rewards(policies,fixedbank,cfg);e=time_ns()
        initial[1:3].=[b-a,c-b,e-c]./1e9;initial[8]=(e-a)/1e9
    end
    function row(k,new,policy,banks,lr,bootdraws,phase,reuse_s)
        (scenario=ID,rep=rep,cohort=cohort(rep),method=m.id,iteration=k,
         new_trajectories=new,cumulative_trajectories=total_new,history_banks=banks,
         new_policy_path_evaluations=policy,cumulative_policy_path_evaluations=total_policy,
         lr_path_evaluations=lr,bootstrap_path_draws=bootdraws,
         generation_seconds=phase[1],materialization_seconds=phase[2],policy_seconds=phase[3],
         gradient_seconds=phase[4],reuse_seconds=reuse_s,cumulative_reuse_seconds=cumulative_reuse,bootstrap_seconds=phase[5],update_seconds=phase[6],
         other_seconds=phase[7],search_seconds=phase[8],
         cumulative_search_seconds=cumulative[8],cumulative_gradient_seconds=cumulative[4],
         cumulative_bootstrap_seconds=cumulative[5],timing_source="013 simulation efficiency; sequential original-kernel measurement")
    end
    push!(rows,row(0,0,0,0,0,0,zeros(8),0.))
    for k in 1:K
        phase=zeros(8);reuse_s=0.;a=time_ns();z=zhist[end];payload=nothing
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
                data=R.green_data(z,history,cfg,ID);reuse_s=(time_ns()-ts)/1e9;phase[4]+=reuse_s
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
        cumulative .+=phase;cumulative_reuse+=reuse_s
        new=m.kind==:fixed ? (k==1 ? length(fixedbank) : 0) : length(payload.bank)
        total_new+=new;policy=m.kind==:fixed ? (k==1 ? length(fixedY) : 0) : length(payload.Y);total_policy+=policy
        green=m.kind in (:green,:green_softmax)
        banks=green ? length(history) : 1
        lr=m.kind==:fixed ? length(fixedbank) : length(payload.bank)+(green ? sum(length(hb.bank) for hb in history) : 0)
        push!(rows,row(k,new,policy,banks,lr,m.kind==:psoft ? n*d["bootstrap_reps"] : 0,phase,reuse_s))
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
