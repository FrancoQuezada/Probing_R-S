# Included inside ProbingRegimeExperiment. Independent of optimization outcomes.
const CHARACTERIZATION_LOCK=ReentrantLock()
const CHARACTERIZATION_BANKS=Dict{String,Any}()
function reference_primitives(d,s,label)
    key=d["fingerprint"]*"/$(s.id)/$label"
    lock(CHARACTERIZATION_LOCK) do
        get!(CHARACTERIZATION_BANKS,key) do
            R.primitive_bank(s.model,d["reference_paths"],seed(s.id,"characterization",0,label);njobs=d["jobs_per_path"])
        end
    end
end
function points()
    grid=vec([(@sprintf("g%02d",k-1),[a,b]) for (k,(a,b)) in enumerate(Iterators.product((.1,.5,.9),(.1,.5,.9)))])
    [grid;[("Z0",copy(C.Z0))]]
end

function reward_equivalence(YA,YB,tolerance)
    m=size(YA,1);groups=zeros(Int,m);means=vec(mean(YA;dims=2))
    for i in 1:m
        groups[i]!=0&&continue;groups[i]=i
        for j in i+1:m
            groups[j]!=0&&continue
            if abs(means[i]-means[j])<=tolerance &&
               maximum(abs,view(YA,i,:)-view(YA,j,:))<=tolerance &&
               maximum(abs,view(YB,i,:)-view(YB,j,:))<=tolerance
                groups[j]=i
            end
        end
    end
    return groups
end

function bootstrap_gradient_metrics(Yi,Yj,S,B,rng,cfg,z)
    n=length(Yi);F=vcat(transpose(Yi.*view(S,1,:)),transpose(Yi.*view(S,2,:)),
                       transpose(Yj.*view(S,1,:)),transpose(Yj.*view(S,2,:)))
    counts=zeros(n,B)
    for b in 1:B,l in 1:n;counts[rand(rng,1:n),b]+=1.;end
    means=F*counts/n
    cosines=Float64[];steps1=Float64[];steps100=Float64[]
    for b in 1:B
        gi=means[1:2,b];gj=means[3:4,b]
        c=E.cosine(gi,gj);!ismissing(c)&&push!(cosines,c)
        for (k,dest) in ((1,steps1),(100,steps100))
            a=R.projected_step(z,gi-collect(cfg.alpha),k)[1]
            bstep=R.projected_step(z,gj-collect(cfg.alpha),k)[1]
            push!(dest,norm(a-bstep))
        end
    end
    return (cosine_low=isempty(cosines) ? missing : quantile(cosines,.025),
        cosine_high=isempty(cosines) ? missing : quantile(cosines,.975),
        step1_low=quantile(steps1,.025),step1_high=quantile(steps1,.975),
        step100_low=quantile(steps100,.025),step100_high=quantile(steps100,.975))
end

function characterize_point(d,s,point,z,policies)
    cfg=s.model;n=d["reference_paths"];ts=time_ns()
    bankA=R.materialize_bank(reference_primitives(d,s,"reference_A"),z,cfg)
    bankB=R.materialize_bank(reference_primitives(d,s,"reference_B"),z,cfg)
    YA=R.rewards(policies,bankA,cfg);YB=R.rewards(policies,bankB,cfg)
    data=R.local_data(z,bankB,YB,cfg);S=data.scores
    muA=vec(mean(YA;dims=2));muB=data.values;m=length(policies)
    seA=vec(std(YA;dims=2))/sqrt(n);seB=vec(std(YB;dims=2))/sqrt(n)
    covariance=cov(YB;dims=2)
    groups=reward_equivalence(YA,YB,d["equivalence_tolerance"])
    bestA=argmax(muA);bestB=argmax(muB);nominal=(maximum(muA) .- muA) .<= d["epsilon"]
    lower=zeros(m);upper=zeros(m);status=String[]
    for i in 1:m
        for j in 1:m
            se=sqrt(max(0.,covariance[i,i]+covariance[j,j]-2covariance[i,j])/n)
            lower[i]=max(lower[i],muB[j]-muB[i]-d["reference_pair_critical"]*se)
            upper[i]=max(upper[i],muB[j]-muB[i]+d["reference_pair_critical"]*se)
        end
        push!(status,upper[i]<=d["epsilon"] ? "certified_inside" : lower[i]>d["epsilon"] ? "certified_outside" : "reference_ambiguous")
    end
    gradients=data.G
    gradSE=zeros(m,2)
    for i in 1:m,j in 1:2
        gradSE[i,j]=std(view(YB,i,:).*view(S,j,:))/sqrt(n)
    end
    prefix=(scenario=s.id,A=s.A,B=s.B,C=s.C,point=point,z3=z[1],z4=z[2])
    policyrows=[merge(prefix,(policy=i,equivalence_group=groups[i],mean_A=muA[i],mean_B=muB[i],
        se_A=seA[i],se_B=seB[i],nominal_epsilon=nominal[i],regret_B_lower=lower[i],regret_B_upper=upper[i],
        reference_status=status[i],G3=gradients[i,1],G4=gradients[i,2],G3_se=gradSE[i,1],G4_se=gradSE[i,2])) for i in 1:m]
    seeds=NamedTuple[(scenario=s.id,phase="characterization",point="ALL",rep=0,stream=label,seed=seed(s.id,"characterization",0,label)) for label in ("reference_A","reference_B")]
    selections=NamedTuple[];counts500=zeros(Int,m)
    for size in (50,100,500),r in 1:d["selection_reps"]
        stream="selection/$point/n=$size";rngseed=seed(s.id,"characterization",r,stream)
        primitive=R.primitive_bank(cfg,size,rngseed;njobs=d["jobs_per_path"])
        Y=R.rewards(policies,R.materialize_bank(primitive,z,cfg),cfg)
        values=vec(mean(Y;dims=2));selected=argmax(values);size==500&&(counts500[selected]+=1)
        ties=count(v->maximum(values)-v<=E.TIE_ATOL+E.TIE_RTOL*abs(maximum(values)),values)
        push!(selections,merge(prefix,(n_paths=size,selection_rep=r,policy=selected,equivalence_group=groups[selected],
            outside_nominal_epsilon=!nominal[selected],reference_status=status[selected],ties=ties)))
        push!(seeds,(scenario=s.id,phase="characterization",point=point,rep=r,stream=stream,seed=rngseed))
    end
    group_counts=Dict(g=>sum(counts500[findall(==(g),groups)]) for g in unique(groups))
    grouporder=sort(collect(keys(group_counts));by=g->(-group_counts[g],g))
    modal_i=argmax([groups[i]==grouporder[1] ? counts500[i] : -1 for i in 1:m])
    modal_j=length(grouporder)>1 ? argmax([groups[i]==grouporder[2] ? counts500[i] : -1 for i in 1:m]) : modal_i
    top=sortperm(muA;rev=true)[1:5];principals=sort(unique([top;modal_i;modal_j;bestB]))
    routing=Dict{Int,Any}();routingrows=NamedTuple[]
    for i in principals
        details=[R.routing_details(policies[i],path,cfg) for path in bankB]
        routing[i]=details
        all(details[l].reward==YB[i,l] for l in 1:n)||error("Instrumented routing reward differs")
        push!(routingrows,merge(prefix,(policy=i,fast_fraction=mean(x.fast_fraction for x in details),
            fast_utilization=mean(x.fast_utilization for x in details),total_wait=mean(x.total_wait for x in details),
            total_wait_se=std([x.total_wait for x in details])/sqrt(n),delay_penalty=mean(x.delay_penalty for x in details),
            routing_cost=mean(x.routing_cost for x in details),reward=muB[i],reward_se=seB[i])))
    end
    pairs=sort(unique([[(principals[a],principals[b]) for a in 1:length(principals)-1 for b in a+1:length(principals)];[(min(modal_i,modal_j),max(modal_i,modal_j))]]))
    pairrows=NamedTuple[];routingpairs=NamedTuple[]
    for (i,j) in pairs
        yi=collect(view(YB,i,:));yj=collect(view(YB,j,:));gi=collect(view(gradients,i,:));gj=collect(view(gradients,j,:))
        delta=gi-gj;se=[std((yi-yj).*view(S,k,:))/sqrt(n) for k in 1:2]
        crit=d["gradient_critical"]
        normlo=sqrt(sum(max(abs(delta[k])-crit*se[k],0.)^2 for k in 1:2))
        normhi=sqrt(sum((abs(delta[k])+crit*se[k])^2 for k in 1:2))
        gilow=norm(max.(abs.(gi)-crit*gradSE[i,:],0.));gjlow=norm(max.(abs.(gj)-crit*gradSE[j,:],0.))
        gihi=norm(abs.(gi)+crit*gradSE[i,:]);gjhi=norm(abs.(gj)+crit*gradSE[j,:])
        relative=norm(delta)/max(norm(gi),norm(gj),1.)
        rel_low=normlo/max(gihi,gjhi,1.);rel_high=normhi/max(gilow,gjlow,1.)
        cos=norm(gi)>d["gradient_zero_tolerance"]&&norm(gj)>d["gradient_zero_tolerance"] ? E.cosine(gi,gj) : missing
        stream="gradient/$point/$i/$j";rngseed=seed(s.id,"characterization",0,stream)
        ci=bootstrap_gradient_metrics(yi,yj,S,d["gradient_bootstrap_reps"],MersenneTwister(rngseed),cfg,z)
        push!(seeds,(scenario=s.id,phase="characterization",point=point,rep=0,stream=stream,seed=rngseed))
        step1=norm(R.projected_step(z,gi-collect(cfg.alpha),1)[1]-R.projected_step(z,gj-collect(cfg.alpha),1)[1])
        step100=norm(R.projected_step(z,gi-collect(cfg.alpha),100)[1]-R.projected_step(z,gj-collect(cfg.alpha),100)[1])
        gap=muB[i]-muB[j];gapse=std(yi-yj)/sqrt(n)
        push!(pairrows,merge(prefix,(policy_i=i,policy_j=j,equivalent=groups[i]==groups[j],
            gap_B=gap,gap_paired_se=gapse,delta_G3=delta[1],delta_G4=delta[2],delta_G3_se=se[1],delta_G4_se=se[2],
            gradient_distance=norm(delta),distance_low=normlo,distance_high=normhi,
            relative_gradient_gap=relative,relative_gap_low=rel_low,relative_gap_high=rel_high,
            cosine=cos,directional_gap=ismissing(cos) ? missing : 1-cos,
            projected_step1_difference=step1,projected_step100_difference=step100),ci))
        differences=[count(xor.(routing[i][l].decisions,routing[j][l].decisions))/d["jobs_per_path"] for l in 1:n]
        push!(routingpairs,merge(prefix,(policy_i=i,policy_j=j,equivalent_rewards=groups[i]==groups[j],
            disagreement=mean(differences),disagreement_se=std(differences)/sqrt(n),
            fast_fraction_difference=mean(routing[i][l].fast_fraction-routing[j][l].fast_fraction for l in 1:n),
            fast_utilization_difference=mean(routing[i][l].fast_utilization-routing[j][l].fast_utilization for l in 1:n),
            waiting_difference=mean(routing[i][l].total_wait-routing[j][l].total_wait for l in 1:n),
            delay_penalty_difference=mean(routing[i][l].delay_penalty-routing[j][l].delay_penalty for l in 1:n),
            routing_cost_difference=mean(routing[i][l].routing_cost-routing[j][l].routing_cost for l in 1:n),
            reward_difference=gap,reward_difference_se=gapse)))
    end
    alternatives=filter(i->groups[i]!=groups[bestA],sortperm(muA;rev=true))
    secondA=isempty(alternatives) ? bestA : first(alternatives)
    gap=muA[bestA]-muA[secondA];gapSE=std(view(YA,bestA,:)-view(YA,secondA,:))/sqrt(n)
    summary=merge(prefix,(objective_A=maximum(muA)-R.cost(cfg,z),objective_B=maximum(muB)-R.cost(cfg,z),
        best_A=bestA,best_B=bestB,second_nonequivalent_A=secondA,gap_A=gap,gap_A_paired_se=gapSE,
        standardized_gap_A=gapSE>0 ? gap/gapSE : missing,modal_i=modal_i,modal_j=modal_j,
        best_A_group=groups[bestA],best_B_group=groups[bestB],modal_group=groups[modal_i],
        n_equivalence_groups=length(unique(groups)),nominal_epsilon_count=count(nominal),
        certified_inside_count=count(==("certified_inside"),status),certified_outside_count=count(==("certified_outside"),status),
        reference_ambiguous_count=count(==("reference_ambiguous"),status),reference_paths=n,
        reference_A_seed=seed(s.id,"characterization",0,"reference_A"),reference_B_seed=seed(s.id,"characterization",0,"reference_B"),
        seconds=(time_ns()-ts)/1e9))
    return (policies=policyrows,selections=selections,gradients=pairrows,routing=routingrows,routing_pairs=routingpairs,
        summary=summary,seeds=seeds,fingerprint=d["fingerprint"],scenario=s.id,point=point)
end

function characterize(d,targets)
    policies=C.build_policy_grid();tasks=[(s,p,z) for s in targets for (p,z) in points()]
    Threads.@threads :dynamic for (s,p,z) in tasks
        dir=joinpath(d["output"],"characterization",s.id);mkpath(dir)
        checkpoint=joinpath(d["output"],"checkpoints/characterization_$(s.id)_$p.jls")
        result=if isfile(checkpoint)
            x=deserialize(checkpoint);x.fingerprint==d["fingerprint"]||error("Invalid characterization cache")
            println("REUSE characterization $(s.id) $p");x
        else
            x=characterize_point(d,s,p,z,policies);E.atomic_serialize(checkpoint,x);x
        end
        for (field,suffix) in ((:policies,"policy_values"),(:selections,"selection_draws"),(:gradients,"gradient_pairs"),
            (:routing,"routing_policies"),(:routing_pairs,"routing_pairs"),(:seeds,"seeds"))
            E.write_csv(joinpath(dir,"$(p)_$suffix.csv"),getproperty(result,field))
        end
        E.write_csv(joinpath(dir,"$(p)_summary.csv"),[result.summary])
        println("COMPLETE characterization $(s.id) $p");flush(stdout)
    end
    empty!(CHARACTERIZATION_BANKS)
end
