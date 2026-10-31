# Refresh the existing 011 temporal-contrast audit with the nine-method traces.
# Only presentation/selection bookkeeping is new; audit_point is unchanged.
include(joinpath(@__DIR__,"014probing_green_softmax090.jl"))
using .ProbingGreenSoftmax090
using Statistics, LinearAlgebra, Serialization, SHA, TOML
const B=ProbingGreenSoftmax090

function main()
    BLAS.set_num_threads(1)
    out=joinpath(@__DIR__,"resultados/014probing_green_softmax090")
    d=Dict{String,Any}("output"=>out);cfg=B.H.from_dict(TOML.parsefile(joinpath(B.SOURCE,"configs/models.toml"))[B.ID])
    trace=deserialize(joinpath(out,"checkpoints/evaluation_new_1.jls")).rows
    new=deserialize(joinpath(out,"checkpoints/search_new_1.jls")).run
    old=deserialize(joinpath(B.SOURCE,"checkpoints/search_I0252_1.jls"));previous=deserialize(joinpath(B.PRIOR,"checkpoints/search_new_1.jls")).run;runs=[old.runs;[previous,new]]
    q=filter(r->r.method!="Fixed",trace);ks=sort(unique(r.iteration for r in q))
    spread(k)=maximum(r.objective for r in q if r.iteration==k)-minimum(r.objective for r in q if r.iteration==k)
    peak=ks[argmax(spread.(ks))];tail=filter(k->k>=20,ks);tailpeak=tail[argmax(spread.(tail))]
    times=sort(unique([20,50,80,100,peak,tailpeak]));plans=NamedTuple[]
    for k in times
        at=filter(r->r.iteration==k,q);hi=at[argmax([r.objective for r in at])];lo=at[argmin([r.objective for r in at])]
        push!(plans,(scenario=B.ID,rep=1,iteration=k,method_high=hi.method,method_low=lo.method,
            trace_difference=hi.objective-lo.objective,global_peak=k==peak,tail_peak=k==tailpeak))
    end
    planpath=joinpath(out,"configs/temporal_audit_plan.jls")
    if isfile(planpath);@assert isequal(deserialize(planpath),plans);else;B.E.atomic_serialize(planpath,plans);end
    B.E.write_csv(joinpath(out,"configs/temporal_audit_plan.csv"),plans)
    source=deserialize(joinpath(B.PRIOR,"checkpoints/temporal_contrast_validation.jls"))
    ss=B.H.seed(B.ID,"trajectory_validation",1,"selection");hs=B.H.seed(B.ID,"trajectory_validation",1,"holdout")
    selector=nothing;hold=nothing;policies=B.C.build_policy_grid();rows=NamedTuple[]
    for p in plans
        zhi=only(filter(r->r.method.id==p.method_high,runs)).zhist[p.iteration+1]
        zlo=only(filter(r->r.method.id==p.method_low,runs)).zhist[p.iteration+1]
        match=findfirst(r->r.iteration==p.iteration&&r.method_high==p.method_high&&r.method_low==p.method_low,source)
        if match!==nothing
            r=source[match];delta=r.independent_difference;se=r.paired_se;origin="013 unchanged independent pair reused"
            ns=r.selection_paths;nh=r.holdout_paths;usedss=r.selection_seed;usedhs=r.holdout_seed
        elseif zhi==zlo
            delta=0.;se=0.;origin="identical decisions";ns=0;nh=0;usedss=missing;usedhs=missing
        elseif p.iteration==100
            f=deserialize(joinpath(out,"checkpoints/final_1.jls"))
            r=only(filter(r->r.method_a==p.method_high&&r.method_b==p.method_low,f.paired))
            delta=r.difference;se=r.paired_se;origin="014 independent final pair reused"
            ns=10000;nh=50000;usedss=B.X.seed(B.ID,"compare",1,"selection");usedhs=B.X.seed(B.ID,"compare",1,"holdout")
        else
            selector===nothing&&(selector=B.R.primitive_bank(cfg,10000,ss;njobs=45))
            hold===nothing&&(hold=B.R.primitive_bank(cfg,50000,hs;njobs=45))
            v=Any[]
            for z in (zhi,zlo)
                name="audit_trajectory_I0252_$(bytes2hex(sha256(repr(z)))).jls"
                dst=joinpath(out,"checkpoints",name)
                for folder in (B.PRIOR,B.SOURCE)
                    src=joinpath(folder,"checkpoints",name)
                    if !isfile(dst)&&isfile(src);cp(src,dst);end
                end
                a=B.H.audit_point(d,(id=B.ID,model=cfg),"trajectory",z,selector,hold,policies)
                @assert a.z==z&&length(a.rewards)==50000;push!(v,a)
            end
            dv=v[1].rewards-v[2].rewards .-(v[1].cost-v[2].cost)
            delta=mean(dv);se=std(dv)/sqrt(length(dv));origin="014 independent selected pair; unchanged selector/holdout estimator"
            ns=10000;nh=50000;usedss=ss;usedhs=hs
        end
        push!(rows,merge(p,(independent_difference=delta,paired_se=se,source=origin,selection_paths=ns,
            holdout_paths=nh,selection_seed=usedss,holdout_seed=usedhs)))
    end
    B.E.write_csv(joinpath(out,"tables/temporal_contrast_validation.csv"),rows)
    B.E.atomic_serialize(joinpath(out,"checkpoints/temporal_contrast_validation.jls"),rows)
    println("Nine-method temporal audit complete: ",length(rows)," pairs; ",count(r->startswith(r.source,"014 independent selected"),rows)," new independent pairs")
end
if abspath(PROGRAM_FILE)==@__FILE__;main();end
