# Independent precision audit of maximum non-Fixed trajectory spread in the
# PREDEFINED illustrative replication 1. No seed or instance winner selection.
include(joinpath(@__DIR__,"010probing_instance_landscape_search.jl"))
using .ProbingLandscapeSearch
using Serialization, Statistics, LinearAlgebra, TOML
const X=ProbingLandscapeSearch
const R=X.R
const E=X.E
const C=X.C
BLAS.set_num_threads(1)
const OUT=joinpath(@__DIR__,"resultados/010probing_instance_landscape_search")
ss=deserialize(joinpath(OUT,"configs/design.jls"));chosen=deserialize(joinpath(OUT,"configs/selected_instances.jls"))
const IDS=[a.scenario for a in chosen]
function rows(path)
    raw=readlines(path);ks=Symbol.(split(raw[1],','))
    [Dict(zip(ks,split(line,','))) for line in raw[2:end] if !isempty(line)]
end
selections=NamedTuple[]
for id in IDS
    trace=rows(joinpath(OUT,"comparisons",id,"rep_01_traces.csv"))
    iters=sort(unique(parse(Int,r[:iteration]) for r in trace));spreads=Float64[]
    for k in iters
        q=[parse(Float64,r[:objective]) for r in trace if parse(Int,r[:iteration])==k&&r[:method]!="Fixed"]
        push!(spreads,maximum(q)-minimum(q))
    end
    k=iters[argmax(spreads)];at=[r for r in trace if parse(Int,r[:iteration])==k&&r[:method]!="Fixed"]
    hi=at[argmax([parse(Float64,r[:objective]) for r in at])][:method]
    lo=at[argmin([parse(Float64,r[:objective]) for r in at])][:method]
    push!(selections,(scenario=id,rep=1,iteration=k,trace_spread=maximum(spreads),trace_high=hi,trace_low=lo,
        selection_seed=X.seed(id,"trajectory_audit",1,"selection"),holdout_seed=X.seed(id,"trajectory_audit",1,"holdout")))
end
frozen=joinpath(OUT,"configs/trajectory_audit.jls")
if isfile(frozen);@assert deserialize(frozen)==selections
else;E.atomic_serialize(frozen,selections);E.write_csv(joinpath(OUT,"configs/trajectory_audit.csv"),selections);end
Threads.@threads :dynamic for a in selections
    path=joinpath(OUT,"checkpoints/trajectory_audit_$(a.scenario).jls");isfile(path)&&continue
    s=only(filter(s->s.id==a.scenario,ss));cfg=s.model;p=C.build_policy_grid()
    result=deserialize(joinpath(OUT,"checkpoints/search_$(s.id)_1.jls"))
    selector=R.primitive_bank(cfg,10000,a.selection_seed);hold=R.primitive_bank(cfg,50000,a.holdout_seed)
    values=Dict{String,Vector{Float64}}();points=NamedTuple[]
    cache=Dict{Tuple{Float64,Float64},Any}()
    for run in result.runs
        z=run.zhist[a.iteration+1]
        v=get!(cache,Tuple(z)) do
            Y=R.rewards(p,R.materialize_bank(selector,z,cfg),cfg);policy=argmax(vec(mean(Y;dims=2)))
            Ye=R.rewards(p,R.materialize_bank(hold,z,cfg),cfg)
            (policy=policy,v=collect(view(Ye,policy,:)),direct=maximum(vec(mean(Ye;dims=2))))
        end
        cost=R.cost(cfg,z);values[run.method.id]=v.v.-cost
        push!(points,(scenario=s.id,rep=1,iteration=a.iteration,method=run.method.id,z3=z[1],z4=z[2],selected_policy=v.policy,
            objective_holdout=mean(v.v)-cost,se=std(v.v)/sqrt(50000),objective_direct=v.direct-cost,gross=mean(v.v),cost=cost))
    end
    delta=values[a.trace_high]-values[a.trace_low];se=std(delta)/sqrt(length(delta))
    summary=(scenario=s.id,rep=1,iteration=a.iteration,method_high=a.trace_high,method_low=a.trace_low,
        exploratory_trace_spread=a.trace_spread,independent_difference=mean(delta),paired_se=se,
        snr=se>0 ? abs(mean(delta))/se : 0.,selection_paths=10000,holdout_paths=50000,
        selection_seed=a.selection_seed,holdout_seed=a.holdout_seed)
    E.atomic_serialize(path,(points=points,summary=summary));println("Independent trajectory contrast audit complete: ",s.id);flush(stdout)
end
results=[deserialize(joinpath(OUT,"checkpoints/trajectory_audit_$id.jls")) for id in IDS]
E.write_csv(joinpath(OUT,"tables/trajectory_contrast_validation.csv"),[r.summary for r in results])
E.write_csv(joinpath(OUT,"tables/trajectory_contrast_points.csv"),vcat([r.points for r in results]...))
