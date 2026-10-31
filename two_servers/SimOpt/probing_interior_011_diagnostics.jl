# Included inside ProbingHighContrastInterior; original estimators are called unchanged.
function diagnostic_case(d,s)
    path=joinpath(d["output"],"checkpoints/diagnostic_$(s.id).jls")
    isfile(path)&&return deserialize(path)
    surf=surface(d,s,"refine");m=surf.metrics
    points=[("initial",copy(C.Z0)),("grid_best",[m.best_z3,m.best_z4]),("anchor",s.anchor)]
    rows=NamedTuple[];policies=C.build_policy_grid();cfg=s.model
    for rep in 1:d["diagnostic_reps"]
        ss=seed(s.id,"presearch",rep,"fresh");fs=seed(s.id,"presearch",rep,"fixed")
        primitive=R.primitive_bank(cfg,d["n_paths"],ss;njobs=d["jobs"])
        fixed=R.materialize_bank(R.primitive_bank(cfg,d["n_paths"],fs;njobs=d["jobs"]),[0.,0.],cfg)
        fixedY=R.rewards(policies,fixed,cfg)
        for (label,z) in points
            bank=R.materialize_bank(primitive,z,cfg);Y=R.rewards(policies,bank,cfg);localdata=R.local_data(z,bank,Y,cfg)
            greendata=R.green_data(z,[R.historical_bank(s.id,cfg,z,bank,Y)],cfg,s.id)
            p=R.bootstrap_weights(Y,d["bootstrap_reps"],MersenneTwister(seed(s.id,"presearch",rep,"bootstrap/"*label)))
            for method in X.METHODS
                g=if method.kind==:fixed;X.fixed_data(z,fixed,fixedY,cfg).hard
                elseif method.kind==:fresh;localdata.hard
                elseif method.kind==:green;greendata.hard
                elseif method.kind==:psoft;R.combine(localdata.G,p,cfg)
                elseif method.kind==:green_softmax;R.combine(greendata.G,R.softmax_weights(greendata.values;beta=method.beta),cfg)
                else;R.combine(localdata.G,R.softmax_weights(localdata.values;beta=method.beta),cfg);end
                for k in (1,20,50,100)
                    znew,eta,gc=C.projected_step(z,g,k);projected=norm(znew-(z+eta*gc))>1e-12
                    push!(rows,(scenario=s.id,point=label,z3=z[1],z4=z[2],rep=rep,method=method.id,iteration=k,
                        gradient3=g[1],gradient4=g[2],gradient_norm=norm(g),clipped_fraction=count(x->abs(x)>C.GRAD_CLIP,g)/2,
                        any_clipping=any(x->abs(x)>C.GRAD_CLIP,g),projected=projected,
                        boundary_contact=any(x->x==0||x==1,znew),effective_step=norm(znew-z),history_banks=1,
                        fresh_seed=ss,fixed_seed=fs,n_paths=d["n_paths"]))
                end
            end
        end
    end
    E.write_csv(joinpath(d["output"],"diagnostics/$(s.id)_gradients.csv"),rows);E.atomic_serialize(path,rows)
    println("Presearch saturation diagnostic complete: ",s.id);flush(stdout);rows
end
function diagnostic_phase(d,scenarios)
    Threads.@threads :dynamic for s in scenarios;diagnostic_case(d,s);end
    rows=vcat([diagnostic_case(d,s) for s in scenarios]...)
    E.write_csv(joinpath(d["output"],"diagnostics/presearch_gradients.csv"),rows);rows
end
function audit_point(d,s,phase,z,selector,hold,policies)
    key=bytes2hex(sha256(repr(z)));path=joinpath(d["output"],"checkpoints/audit_$(phase)_$(s.id)_$key.jls")
    isfile(path)&&return deserialize(path)
    Ys=R.rewards(policies,R.materialize_bank(selector,z,s.model),s.model);selected=argmax(vec(mean(Ys;dims=2)))
    Ye=R.rewards(policies,R.materialize_bank(hold,z,s.model),s.model);means=vec(mean(Ye;dims=2));direct=argmax(means)
    v=collect(view(Ye,selected,:));result=(z=copy(z),selected_policy=selected,direct_policy=direct,rewards=v,
        gross=mean(v),cost=R.cost(s.model,z),objective=mean(v)-R.cost(s.model,z),direct_objective=means[direct]-R.cost(s.model,z),se=std(v)/sqrt(length(v)))
    E.atomic_serialize(path,result);result
end
function geometry_audit(d,s,surf)
    path=joinpath(d["output"],"checkpoints/geometry_audit_$(s.id).jls");isfile(path)&&return deserialize(path)
    z=[surf.metrics.best_z3,surf.metrics.best_z4];targets=[(role="grid_best",z=z),(role="initial",z=copy(C.Z0)),(role="anchor",z=s.anchor)]
    for h in (.10,.20),j in 1:2,sign in (-1,1)
        point=copy(z);point[j]+=sign*h
        all(x->0<=x<=1,point)&&push!(targets,(role="axis$(j)_h$(h)_s$(sign)",z=point))
    end
    for a in (-1,1),b in (-1,1)
        point=z+[a*.1,b*.1];all(x->0<=x<=1,point)&&push!(targets,(role="diagonal_$(a)_$(b)",z=point))
    end
    for j in 1:2,value in (0.,1.)
        point=copy(z);point[j]=value;push!(targets,(role="boundary$(j)_$(Int(value))",z=point))
    end
    select_seed=seed(s.id,"geometry_audit",0,"selection");hold_seed=seed(s.id,"geometry_audit",0,"holdout")
    selector=R.primitive_bank(s.model,d["selection_paths"],select_seed;njobs=d["jobs"])
    hold=R.primitive_bank(s.model,d["audit_paths"],hold_seed;njobs=d["jobs"]);policies=C.build_policy_grid()
    points=[audit_point(d,s,"geometry",t.z,selector,hold,policies) for t in targets];best=points[1]
    rows=NamedTuple[]
    for (t,p) in zip(targets,points)
        delta=best.rewards-p.rewards .-(best.cost-p.cost);se=std(delta)/sqrt(length(delta))
        push!(rows,(scenario=s.id,role=t.role,z3=p.z[1],z4=p.z[2],selected_policy=p.selected_policy,direct_policy=p.direct_policy,
            objective_holdout=p.objective,objective_direct=p.direct_objective,gross=p.gross,cost=p.cost,se=p.se,
            grid_reference_loss=mean(delta),paired_se=se,snr=X.safe_snr(mean(delta),se),selection_paths=d["selection_paths"],holdout_paths=d["audit_paths"],
            selection_seed=select_seed,holdout_seed=hold_seed))
    end
    localrows=NamedTuple[]
    for h in (.10,.20),j in 1:2
        ip=findfirst(t->t.role=="axis$(j)_h$(h)_s1",targets);im=findfirst(t->t.role=="axis$(j)_h$(h)_s-1",targets)
        (ip===nothing||im===nothing)&&continue
        plus=points[ip];minus=points[im]
        gradient=(plus.rewards-minus.rewards .-(plus.cost-minus.cost))/(2h)
        second=(plus.rewards-2best.rewards+minus.rewards)/h^2
        push!(localrows,(scenario=s.id,dimension=j+2,h=h,net_slope=mean(gradient),slope_se=std(gradient)/sqrt(length(gradient)),
            second_difference=mean(second),second_difference_se=std(second)/sqrt(length(second)),
            selected_policy_switch=length(unique([plus.selected_policy,best.selected_policy,minus.selected_policy]))>1))
    end
    E.write_csv(joinpath(d["output"],"audits/$(s.id)_geometry.csv"),rows)
    E.write_csv(joinpath(d["output"],"audits/$(s.id)_local_differences.csv"),localrows)
    result=(rows=rows,local_differences=localrows);E.atomic_serialize(path,result)
    println("Independent geometry audit complete: ",s.id);flush(stdout);result
end
function trajectory_audit(d,s)
    path=joinpath(d["output"],"checkpoints/trajectory_audit_$(s.id).jls");isfile(path)&&return deserialize(path)
    evaluation=deserialize(joinpath(d["output"],"checkpoints/evaluation_$(s.id)_1.jls"));traces=evaluation.traces
    allks=sort(unique(r.iteration for r in traces));nonfixed=filter(r->r.method!="Fixed",traces)
    spread(k)=maximum(r.objective for r in nonfixed if r.iteration==k)-minimum(r.objective for r in nonfixed if r.iteration==k)
    tail=filter(k->k>=min(20,d["max_iters"]),allks)
    peak=allks[argmax(spread.(allks))];tailpeak=tail[argmax(spread.(tail))]
    times=sort(unique([filter(k->k<=d["max_iters"],[20,50,80,100]);d["max_iters"];peak;tailpeak]))
    search=deserialize(joinpath(d["output"],"checkpoints/search_$(s.id)_1.jls"));plans=NamedTuple[]
    for k in times
        at=filter(r->r.iteration==k,nonfixed);hi=at[argmax([r.objective for r in at])];lo=at[argmin([r.objective for r in at])]
        push!(plans,(scenario=s.id,rep=1,iteration=k,method_high=hi.method,method_low=lo.method,trace_difference=hi.objective-lo.objective,
            global_peak=k==peak,tail_peak=k==tailpeak))
    end
    planpath=joinpath(d["output"],"configs/trajectory_audit_$(s.id).jls")
    if isfile(planpath);@assert isequal(deserialize(planpath),plans)
    else;E.atomic_serialize(planpath,plans);E.write_csv(joinpath(d["output"],"configs/trajectory_audit_$(s.id).csv"),plans);end
    ss=seed(s.id,"trajectory_validation",1,"selection");hs=seed(s.id,"trajectory_validation",1,"holdout")
    selector=nothing;hold=nothing;policies=C.build_policy_grid();results=NamedTuple[]
    for plan in plans
        zhi=only(filter(r->r.method.id==plan.method_high,search.runs)).zhist[plan.iteration+1]
        zlo=only(filter(r->r.method.id==plan.method_low,search.runs)).zhist[plan.iteration+1]
        # Reuse independent final comparisons when this is the endpoint.
        if zhi==zlo
            delta=0.;se=0.;source="identical decisions; exact zero";usedss=missing;usedhs=missing;usedns=0;usednh=0
        elseif plan.iteration==d["max_iters"]
            old=only(filter(r->r.method_a==plan.method_high&&r.method_b==plan.method_low,evaluation.paired))
            delta=old.difference;se=old.paired_se;source="same independently evaluated final pair"
            f=only(filter(r->r.method==plan.method_high,evaluation.finals));usedss=f.selection_seed;usedhs=f.holdout_seed;usedns=f.selection_paths;usednh=f.holdout_paths
        else
            # Exact 010 peak audit reuse for unchanged reference traces and budgets.
            oldpath=joinpath(d["reference"],"checkpoints/trajectory_audit_$(s.id).jls")
            old=(!d["quick"]&&isfile(oldpath)) ? deserialize(oldpath).summary : nothing
            if old!==nothing&&old.iteration==plan.iteration&&old.method_high==plan.method_high&&old.method_low==plan.method_low&&old.holdout_paths==d["audit_paths"]&&old.selection_paths==d["selection_paths"]
                delta=old.independent_difference;se=old.paired_se;source="010 exact independent peak audit reuse"
                usedss=old.selection_seed;usedhs=old.holdout_seed;usedns=old.selection_paths;usednh=old.holdout_paths
            else
                selector===nothing&&(selector=R.primitive_bank(s.model,d["selection_paths"],ss;njobs=d["jobs"]))
                hold===nothing&&(hold=R.primitive_bank(s.model,d["audit_paths"],hs;njobs=d["jobs"]))
                hi=audit_point(d,s,"trajectory",zhi,selector,hold,policies);lo=audit_point(d,s,"trajectory",zlo,selector,hold,policies)
                values=hi.rewards-lo.rewards .-(hi.cost-lo.cost);delta=mean(values);se=std(values)/sqrt(length(values));source="011 new independent selector/holdout"
                usedss=ss;usedhs=hs;usedns=d["selection_paths"];usednh=d["audit_paths"]
            end
        end
        push!(results,merge(plan,(independent_difference=delta,paired_se=se,snr=X.safe_snr(delta,se),source=source,
            selection_paths=usedns,holdout_paths=usednh,selection_seed=usedss,holdout_seed=usedhs)))
    end
    E.write_csv(joinpath(d["output"],"audits/$(s.id)_trajectory_contrast.csv"),results);E.atomic_serialize(path,results)
    println("Independent temporal contrast audit complete: ",s.id);flush(stdout);results
end
function postsearch_audits(d,scenarios)
    Threads.@threads :dynamic for s in scenarios;trajectory_audit(d,s);end
    E.write_csv(joinpath(d["output"],"tables/trajectory_contrast_validation.csv"),vcat([trajectory_audit(d,s) for s in scenarios]...))
end
