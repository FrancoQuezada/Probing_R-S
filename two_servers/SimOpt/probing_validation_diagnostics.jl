# Included in ProbingRegimeValidation. Scores and policy simulation are reused.
function selected_points(d)
    [begin
        (scenario=row["scenario"],point=row["point"],role=row["role"],z=[row["z3"],row["z4"]],
            source_point=row["source_point"],old_regime=row["old_regime"])
    end for row in TOML.parsefile(joinpath(d["output"],"configs/selected_points.toml"))["points"]]
end

function candidate_reference(d,s,p,policies)
    stem=joinpath(d["output"],"checkpoints","A_$(s.id)_$(p.point)")
    path=stem*"_candidates.jls"
    if isfile(path)
        x=deserialize(path);x.fingerprint==d["fingerprint"]||error("Invalid candidate reference");return x
    end
    cfg=s.model;n=d["candidate_paths"];started=time_ns()
    primitive=R.primitive_bank(cfg,n,diagseed(d,s.id,p.point,"candidate_A");njobs=d["jobs_per_path"])
    YA=R.rewards(policies,R.materialize_bank(primitive,p.z,cfg),cfg);mu=vec(mean(YA;dims=2))
    groups=X.reward_equivalence(YA,YA,1e-8);order=sortperm(mu;rev=true)
    distinct=unique(groups[order]);representatives=[first(filter(i->groups[i]==g,order)) for g in distinct[1:min(2,length(distinct))]]
    draws=NamedTuple[];means=Matrix{Float64}(undef,144,0);seeds=NamedTuple[]
    push!(seeds,(scenario=s.id,point=p.point,phase=d["diagnostic_phase"],rep=0,stream="candidate_A",iteration=0,seed=diagseed(d,s.id,p.point,"candidate_A")))
    for size in (50,100,500),rep in 1:d["selection_reps"]
        stream="ranking_n=$size";seed=diagseed(d,s.id,p.point,stream,rep)
        bank=R.materialize_bank(R.primitive_bank(cfg,size,seed;njobs=d["jobs_per_path"]),p.z,cfg)
        values=vec(mean(R.rewards(policies,bank,cfg);dims=2));selected=argmax(values)
        means=hcat(means,values)
        push!(draws,(scenario=s.id,point=p.point,n_paths=size,rep=rep,policy=selected,
            ties=count(x->maximum(values)-x<=E.TIE_ATOL+E.TIE_RTOL*abs(maximum(values)),values)))
        push!(seeds,(scenario=s.id,point=p.point,phase=d["diagnostic_phase"],rep=rep,stream=stream,iteration=0,seed=seed))
    end
    candidates=sort(unique([order[1:5];representatives;[r.policy for r in draws]]))
    result=(fingerprint=d["fingerprint"],YA=YA,muA=mu,groupsA=groups,candidates=candidates,
        draws=draws,bank_means=means,seeds=seeds,seconds=(time_ns()-started)/1e9)
    E.atomic_serialize(path,result)
    println("FROZEN candidates $(s.id) $(p.point): ",candidates);flush(stdout)
    return result
end

"""Paired observations, not subtraction of independent policy gradient errors."""
function paired_features(yi,yj,S)
    delta=yi-yj
    hcat(delta.*view(S,1,:),delta.*view(S,2,:))
end
function paired_moments(yi,yj,S)
    F=paired_features(yi,yj,S);n=size(F,1)
    (delta=vec(mean(F;dims=1)),covariance=cov(F;dims=1)/n)
end

function bootstrap_joint(Y,S,candidates,B,rng)
    n=size(Y,2);counts=zeros(n,B)
    for b in 1:B,l in 1:n;counts[rand(rng,1:n),b]+=1.;end
    # One shared resampling matrix for values, all policy gradients and steps.
    F=vcat(Y,Y[candidates,:].*transpose(view(S,1,:)),Y[candidates,:].*transpose(view(S,2,:)))
    boot=F*counts/n;m=length(candidates)
    (values=boot[1:144,:],G3=boot[145:144+m,:],G4=boot[145+m:144+2m,:])
end

function diagnostic_criticals(d,totalpairs,totalpolicies)
    code="from scipy.stats import t\nimport sys\nns=list(map(int,sys.argv[1].split(',')));m=int(sys.argv[2]);p=int(sys.argv[3])\nfor n in ns: print(n,t.ppf(.975,n-1),t.ppf(1-.05/(2*6),n-1),t.ppf(1-.05/(2*10296),n-1),t.ppf(1-.05/(4*m),n-1),t.ppf(1-.05/(4*p),n-1),t.ppf(1-.05/(2*m),n-1))"
    lines=split(strip(read(`python3 -c $code $(join(d["reference_sizes"],',')) $totalpairs $totalpolicies`,String)),'\n')
    Dict(parse(Int,split(line)[1])=>parse.(Float64,split(line)[2:end]) for line in lines)
end

function norm_bounds(g,se,critical)
    lo=g-critical*se;hi=g+critical*se
    lower=norm([lo[j]>0 ? lo[j] : hi[j]<0 ? hi[j] : 0. for j in eachindex(g)])
    upper=norm(max.(abs.(lo),abs.(hi)))
    (lower,upper)
end
cosvalue(a,b)=norm(a)>1e-8&&norm(b)>1e-8 ? clamp(dot(a,b)/(norm(a)*norm(b)),-1.,1.) : NaN
interval(x)=isempty(x) ? (NaN,NaN) : (quantile(x,.025),quantile(x,.975))

function reference_data(d,s,p,policies)
    path=joinpath(d["output"],"checkpoints","A_$(s.id)_$(p.point)_reference.jls")
    if isfile(path)
        x=deserialize(path);x.fingerprint==d["fingerprint"]||error("Invalid final reference");return x
    end
    started=time_ns();n=maximum(d["reference_sizes"]);cfg=s.model
    primitive=R.primitive_bank(cfg,n,diagseed(d,s.id,p.point,"reference_B");njobs=d["jobs_per_path"])
    bank=R.materialize_bank(primitive,p.z,cfg);Y=R.rewards(policies,bank,cfg)
    _,S=R.bank_scores(bank,p.z,p.z,cfg)
    x=(fingerprint=d["fingerprint"],Y=Y,S=S,seconds=(time_ns()-started)/1e9)
    E.atomic_serialize(path,x);x
end

function diagnose_point(d,s,p,a,reference,criticals)
    cfg=s.model;candidate_rows=NamedTuple[];value_rows=NamedTuple[];gradient_rows=NamedTuple[]
    component_rows=NamedTuple[];policy_rows=NamedTuple[];step_rows=NamedTuple[];class_rows=NamedTuple[]
    bootstrap_rows=NamedTuple[];routing_rows=NamedTuple[];seedrows=copy(a.seeds);started=time_ns()
    candidates=a.candidates;positions=Dict(i=>k for (k,i) in enumerate(candidates));alpha=collect(cfg.alpha)
    for i in candidates
        push!(candidate_rows,(scenario=s.id,point=p.point,role=p.role,z3=p.z[1],z4=p.z[2],policy=i,
            candidate_mean_A=a.muA[i],candidate_se_A=std(view(a.YA,i,:))/sqrt(size(a.YA,2)),
            selected_independent_banks=count(r->r.policy==i,a.draws),equivalence_group_A=a.groupsA[i]))
    end
    push!(seedrows,(scenario=s.id,point=p.point,phase=d["diagnostic_phase"],rep=0,stream="reference_B",iteration=0,seed=diagseed(d,s.id,p.point,"reference_B")))
    for n in d["reference_sizes"]
        Y=view(reference.Y,:,1:n);S=view(reference.S,:,1:n);mu=vec(mean(Y;dims=2));G=Y*transpose(S)/n
        Gse=hcat([std(view(Y,i,:).*view(S,j,:))/sqrt(n) for i in 1:144,j in 1:2])
        marginal,local6,reference_critical,global_component,global_policy,global_value=criticals[n]
        seed=diagseed(d,s.id,p.point,"diagnostic_bootstrap",n)
        boot=bootstrap_joint(Y,S,candidates,d["diagnostic_bootstrap"],MersenneTwister(seed))
        push!(seedrows,(scenario=s.id,point=p.point,phase=d["diagnostic_phase"],rep=0,stream="diagnostic_bootstrap",iteration=n,seed=seed))
        groups=X.reward_equivalence(a.YA,Y,1e-8);bestA=argmax(a.muA);bestB=argmax(mu)
        covariance=cov(Y;dims=2);lower=zeros(144);upper=zeros(144);statuses=String[]
        for i in 1:144
            for j in 1:144
                se=sqrt(max(0.,covariance[i,i]+covariance[j,j]-2covariance[i,j])/n)
                lower[i]=max(lower[i],mu[j]-mu[i]-reference_critical*se)
                upper[i]=max(upper[i],mu[j]-mu[i]+reference_critical*se)
            end
            push!(statuses,upper[i]<=.5 ? "certified_inside" : lower[i]>.5 ? "certified_outside" : "reference_ambiguous")
        end
        counts=zeros(Int,144)
        for row in a.draws;row.n_paths==500&&(counts[row.policy]+=1);end
        groupcounts=Dict(g=>sum(counts[findall(==(g),groups)]) for g in unique(groups))
        grouporder=sort(collect(keys(groupcounts));by=g->(-groupcounts[g],g))
        modal_i=argmax([groups[i]==grouporder[1] ? counts[i] : -1 for i in 1:144])
        modal_j=length(grouporder)>1 ? argmax([groups[i]==grouporder[2] ? counts[i] : -1 for i in 1:144]) : modal_i
        alternatives=filter(i->groups[i]!=groups[bestA],sortperm(a.muA;rev=true));second=isempty(alternatives) ? bestA : first(alternatives)
        gap=a.muA[bestA]-a.muA[second];gapse=std(view(a.YA,bestA,:)-view(a.YA,second,:))/sqrt(size(a.YA,2))
        prefix=(scenario=s.id,point=p.point,role=p.role,z3=p.z[1],z4=p.z[2],n_reference=n,primary=n==maximum(d["reference_sizes"]))
        for i in 1:144
            push!(policy_rows,merge(prefix,(policy=i,mean_A=a.muA[i],mean_B=mu[i],se_B=std(view(Y,i,:))/sqrt(n),
                equivalence_group=groups[i],reference_status=statuses[i],regret_lower=lower[i],regret_upper=upper[i],
                G3=G[i,1],G4=G[i,2],G3_se=Gse[i,1],G4_se=Gse[i,2])))
        end
        weights=zeros(144)
        for b in axes(boot.values,2)
            v=view(boot.values,:,b);top=maximum(v);ties=findall(x->top-x<=E.TIE_ATOL+E.TIE_RTOL*abs(top),v)
            weights[ties].+=1/length(ties)/size(boot.values,2)
        end
        for i in 1:144;push!(bootstrap_rows,merge(prefix,(policy=i,bootstrap_selection_probability=weights[i],equivalence_group=groups[i])));end
        for i in candidates,j in 1:2
            push!(component_rows,merge(prefix,(policy=i,coordinate=j+2,estimate=G[i,j],se=Gse[i,j],
                ci95_low=G[i,j]-marginal*Gse[i,j],ci95_high=G[i,j]+marginal*Gse[i,j],
                simultaneous_low=G[i,j]-global_policy*Gse[i,j],simultaneous_high=G[i,j]+global_policy*Gse[i,j])))
        end
        classification_pair=nothing
        for ai in 1:length(candidates)-1,aj in ai+1:length(candidates)
            i=candidates[ai];j=candidates[aj];yi=view(Y,i,:);yj=view(Y,j,:);delta=yi-yj
            deltaZ=mean(delta);seZ=std(delta)/sqrt(n);paired=paired_moments(yi,yj,S)
            dg=paired.delta;covg=paired.covariance;se=sqrt.(max.(0.,diag(covg)));gi=G[i,:];gj=G[j,:]
            nl,nh=norm_bounds(dg,se,global_component);localnl,localnh=norm_bounds(dg,se,local6)
            gli,ghi=norm_bounds(gi,Gse[i,:],local6);glj,ghj=norm_bounds(gj,Gse[j,:],local6)
            relative=norm(dg)/max(norm(gi),norm(gj),1.);relative_low=localnl/max(ghi,ghj,1.);relative_high=localnh/max(gli,glj,1.)
            c=cosvalue(gi,gj);cosines=Float64[]
            for b in axes(boot.values,2)
                bg1=[boot.G3[ai,b],boot.G4[ai,b]];bg2=[boot.G3[aj,b],boot.G4[aj,b]]
                cc=cosvalue(bg1,bg2);isfinite(cc)&&push!(cosines,cc)
            end
            clo,chi=interval(cosines)
            inversion=Dict{Int,Any}()
            reference_tie=abs(a.muA[i]-a.muA[j])<=E.TIE_ATOL
            refsign=sign(a.muA[i]-a.muA[j])
            for size in (50,100,500)
                indices=findall(r->r.n_paths==size,a.draws);diff=a.bank_means[i,indices]-a.bank_means[j,indices]
                inversion[size]=(reference_tie ? missing : mean(x->-refsign*x>E.TIE_ATOL,diff),mean(abs.(diff).<=E.TIE_ATOL))
            end
            push!(value_rows,merge(prefix,(policy_i=i,policy_j=j,equivalent_rewards=groups[i]==groups[j],
                delta_value=deltaZ,paired_se=seZ,ranking_reference_A_tie=reference_tie,ci95_low=deltaZ-marginal*seZ,ci95_high=deltaZ+marginal*seZ,
                simultaneous_low=deltaZ-global_value*seZ,simultaneous_high=deltaZ+global_value*seZ,
                inversion_n50=inversion[50][1],inversion_n100=inversion[100][1],inversion_n500=inversion[500][1],
                tie_n50=inversion[50][2],tie_n100=inversion[100][2],tie_n500=inversion[500][2])))
            push!(gradient_rows,merge(prefix,(policy_i=i,policy_j=j,equivalent_rewards=groups[i]==groups[j],
                delta_G3=dg[1],delta_G4=dg[2],G3_se=se[1],G4_se=se[2],
                covariance_33=covg[1,1],covariance_34=covg[1,2],covariance_44=covg[2,2],
                G3_ci95_low=dg[1]-marginal*se[1],G3_ci95_high=dg[1]+marginal*se[1],
                G4_ci95_low=dg[2]-marginal*se[2],G4_ci95_high=dg[2]+marginal*se[2],
                G3_simultaneous_low=dg[1]-global_component*se[1],G3_simultaneous_high=dg[1]+global_component*se[1],
                G4_simultaneous_low=dg[2]-global_component*se[2],G4_simultaneous_high=dg[2]+global_component*se[2],
                distance=norm(dg),distance_low=nl,distance_high=nh,relative_gradient_gap=relative,
                local6_distance_low=localnl,local6_distance_high=localnh,relative_gap_low=relative_low,relative_gap_high=relative_high,
                cosine=c,cosine_low=clo,cosine_high=chi,global_component_critical=global_component)))
            if i==min(modal_i,modal_j)&&j==max(modal_i,modal_j)
                classification_pair=(relative_low=relative_low,relative_high=relative_high,cosine_low=clo,cosine_high=chi,distance_low=localnl)
            end
            for t in (1,25,50,100)
                rawi=gi-alpha;rawj=gj-alpha;zi,_,clipi=R.projected_step(p.z,rawi,t);zj,_,clipj=R.projected_step(p.z,rawj,t)
                distances=Float64[];clippeddistances=Float64[];identical=Bool[];boundary_i=Bool[];boundary_j=Bool[]
                for b in axes(boot.values,2)
                    bi=[boot.G3[ai,b],boot.G4[ai,b]]-alpha;bj=[boot.G3[aj,b],boot.G4[aj,b]]-alpha
                    aplus,_,ac=R.projected_step(p.z,bi,t);bplus,_,bc=R.projected_step(p.z,bj,t)
                    dd=norm(aplus-bplus);push!(distances,dd);push!(clippeddistances,norm(ac-bc));push!(identical,dd<=1e-12)
                    push!(boundary_i,any(x->x<=1e-12||x>=1-1e-12,aplus));push!(boundary_j,any(x->x<=1e-12||x>=1-1e-12,bplus))
                end
                dlo,dhi=interval(distances);clo2,chi2=interval(clippeddistances)
                push!(step_rows,merge(prefix,(policy_i=i,policy_j=j,iteration=t,raw_gradient_distance=norm(dg),
                    clipped_gradient_distance=norm(clipi-clipj),clipped_distance_low=clo2,clipped_distance_high=chi2,
                    projected_step_distance=norm(zi-zj),step_ci95_low=dlo,step_ci95_high=dhi,
                    identical_update_bootstrap_frequency=mean(identical),step_over_001_bootstrap_frequency=mean(distances.>.01),
                    boundary_i_bootstrap_frequency=mean(boundary_i),boundary_j_bootstrap_frequency=mean(boundary_j),
                    z3_i=zi[1],z4_i=zi[2],z3_j=zj[1],z4_j=zj[2])))
            end
        end
        modalcount=groupcounts[grouporder[1]];secondcount=length(grouporder)>1 ? groupcounts[grouporder[2]] : 0
        push!(class_rows,merge(prefix,(best_A=bestA,best_B=bestB,modal_i=modal_i,modal_j=modal_j,
            modal_group=groups[modal_i],best_A_group=groups[bestA],best_B_group=groups[bestB],
            n_equivalence_groups=length(unique(groups)),gap_A=gap,gap_A_paired_se=gapse,
            standardized_gap_A=gapse>0 ? gap/gapse : Inf,modal_count=modalcount,second_count=secondcount,
            selection_reps=d["selection_reps"],modal_reference_status=statuses[modal_i],
            nominal_selection_error=mean(maximum(a.muA)-a.muA[r.policy]>.5 for r in a.draws if r.n_paths==500),
            reference_ambiguity=mean(statuses[r.policy]=="reference_ambiguous" for r in a.draws if r.n_paths==500),
            relative_gap_low=classification_pair===nothing ? NaN : classification_pair.relative_low,
            relative_gap_high=classification_pair===nothing ? NaN : classification_pair.relative_high,
            cosine_low=classification_pair===nothing ? NaN : classification_pair.cosine_low,
            cosine_high=classification_pair===nothing ? NaN : classification_pair.cosine_high,
            local6_distance_low=classification_pair===nothing ? NaN : classification_pair.distance_low,
            old_regime=p.old_regime)))
    end
    # Detailed own-queue routing only on the final reference, using the same B paths.
    n=maximum(d["reference_sizes"]);primitive=R.primitive_bank(cfg,n,diagseed(d,s.id,p.point,"reference_B");njobs=d["jobs_per_path"])
    bank=R.materialize_bank(primitive,p.z,cfg);policies=C.build_policy_grid();details=Dict(i=>[R.routing_details(policies[i],path,cfg) for path in bank] for i in candidates)
    for ai in 1:length(candidates)-1,aj in ai+1:length(candidates)
        i=candidates[ai];j=candidates[aj]
        disagreement=[count(xor.(details[i][l].decisions,details[j][l].decisions))/d["jobs_per_path"] for l in 1:n]
        push!(routing_rows,(scenario=s.id,point=p.point,policy_i=i,policy_j=j,n_reference=n,
            disagreement=mean(disagreement),disagreement_se=std(disagreement)/sqrt(n),
            waiting_difference=mean(details[i][l].total_wait-details[j][l].total_wait for l in 1:n),
            utilization_difference=mean(details[i][l].fast_utilization-details[j][l].fast_utilization for l in 1:n)))
    end
    return (fingerprint=d["fingerprint"],candidates=candidate_rows,values=value_rows,gradients=gradient_rows,
        components=component_rows,steps=step_rows,classification=class_rows,bootstrap=bootstrap_rows,routing=routing_rows,
        policies=policy_rows,seeds=seedrows,seconds=(time_ns()-started)/1e9+reference.seconds+a.seconds)
end

function diagnostics(d)
    points=selected_points(d);policies=C.build_policy_grid();models=Dict(s.id=>s for s in d["models"])
    candidates=Dict{String,Any}();lk=ReentrantLock()
    Threads.@threads :dynamic for p in points
        s=models[p.scenario];a=candidate_reference(d,s,p,policies)
        lock(lk) do;candidates["$(p.scenario)/$(p.point)"]=a;end
    end
    totalpairs=sum(length(a.candidates)*(length(a.candidates)-1)÷2 for a in values(candidates))
    totalpolicies=sum(length(a.candidates) for a in values(candidates));criticals=diagnostic_criticals(d,totalpairs,totalpolicies)
    open(joinpath(d["output"],"configs/candidate_family.toml"),"w") do io
        TOML.print(io,Dict("fingerprint"=>d["fingerprint"],"pairs"=>totalpairs,"policy_components"=>2totalpolicies,
            "gradient_difference_components"=>2totalpairs,"frozen_before_reference_B"=>true);sorted=true)
    end
    Threads.@threads :dynamic for p in points
        s=models[p.scenario];path=joinpath(d["output"],"checkpoints","A_$(s.id)_$(p.point)_complete.jls")
        x=if isfile(path)
            r=deserialize(path);r.fingerprint==d["fingerprint"]||error("Invalid diagnostic checkpoint");r
        else
            reference=reference_data(d,s,p,policies);r=diagnose_point(d,s,p,candidates["$(s.id)/$(p.point)"],reference,criticals)
            E.atomic_serialize(path,r);r
        end
        dir=joinpath(d["output"],"diagnostics",s.id);mkpath(dir)
        for field in (:candidates,:values,:gradients,:components,:steps,:classification,:bootstrap,:routing,:policies,:seeds)
            E.write_csv(joinpath(dir,"$(p.point)_$(field).csv"),getproperty(x,field))
        end
        E.write_csv(joinpath(dir,"$(p.point)_costs.csv"),[(scenario=s.id,point=p.point,diagnostics_seconds=x.seconds)])
        println("COMPLETE 009b diagnostics $(s.id) $(p.point)");flush(stdout)
    end
end
