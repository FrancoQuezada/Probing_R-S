# Post-selection economic attribution at two fixed, prealgorithm points.
# Does not change the frozen design, selected landscapes, or algorithms.
include(joinpath(@__DIR__,"010probing_instance_landscape_search.jl"))
using .ProbingLandscapeSearch
using Statistics, LinearAlgebra, Serialization, TOML
const X=ProbingLandscapeSearch
const R=X.R
const C=X.C
const E=X.E
BLAS.set_num_threads(1)
const OUT=joinpath(@__DIR__,"resultados/010probing_instance_landscape_search")
ss=deserialize(joinpath(OUT,"configs/design.jls"));s=only(filter(s->s.id=="L026",ss))
ref=deserialize(joinpath(OUT,"checkpoints/surface_refine_L026.jls"))
z=[ref.metrics.best_z3,ref.metrics.best_z4];base=s.model
fields=NamedTuple{fieldnames(R.ModelConfig)}(Tuple(getfield(base,k) for k in fieldnames(R.ModelConfig)))
change(;kw...)=R.validate(R.ModelConfig(;merge(fields,(;kw...))...))
variants=[(name="selected_L026",cfg=base),
    (name="restore_delay_0p42",cfg=change(delay_penalty=.42)),
    (name="restore_service_8_2p4",cfg=change(std_mean=(1.,8.),fast_mean=(.75,2.4))),
    (name="restore_original_signals",cfg=change(p_hit0=(.6,.65,.75,.92),p_false0=(.05,.08,.15,.20))),
    (name="restore_beta_signal_2p2",cfg=change(beta=2.2)),
    (name="restore_alpha_10_4p5",cfg=change(alpha=(10.,4.5)))]
selection_seed=X.seed("attribution","economic",0,"selection")
hold_seed=X.seed("attribution","economic",0,"holdout")
meta=Dict("purpose"=>"local attribution at fixed original Z0 and L026 refined-grid best, before comparing algorithm outcomes",
    "target_z"=>z,"initial_z"=>C.Z0,"selection_paths"=>10000,"holdout_paths"=>50000,
    "selection_seed"=>selection_seed,"holdout_seed"=>hold_seed,
    "models"=>Dict(v.name=>R.model_dict(v.cfg) for v in variants))
path=joinpath(OUT,"configs/economic_attribution.toml")
if isfile(path);@assert TOML.parsefile(path)==meta
else;open(path,"w") do io;TOML.print(io,meta;sorted=true);end;end
Threads.@threads :dynamic for variant in variants
    name=variant.name;cfg=variant.cfg;checkpoint=joinpath(OUT,"checkpoints/attribution_$name.jls")
    isfile(checkpoint)&&continue
    p=C.build_policy_grid();selector=R.primitive_bank(cfg,10000,selection_seed)
    hold=R.primitive_bank(cfg,50000,hold_seed);vals=Vector{Float64}[];rows=NamedTuple[]
    for (role,point) in (("initial",C.Z0),("target",z))
        Ys=R.rewards(p,R.materialize_bank(selector,point,cfg),cfg);selected=argmax(vec(mean(Ys;dims=2)))
        Y=R.rewards(p,R.materialize_bank(hold,point,cfg),cfg);v=collect(view(Y,selected,:));push!(vals,v)
        push!(rows,(variant=name,role=role,z3=point[1],z4=point[2],selected_policy=selected,gross=mean(v),cost=R.cost(cfg,point),
            objective_holdout=mean(v)-R.cost(cfg,point),objective_direct=maximum(vec(mean(Y;dims=2)))-R.cost(cfg,point),se=std(v)/sqrt(length(v))))
    end
    dv=vals[2]-vals[1];cost_delta=R.cost(cfg,z)-R.cost(cfg,C.Z0)
    summary=(variant=name,gross_gain=mean(dv),cost_increase=cost_delta,net_gain=mean(dv)-cost_delta,
        paired_se=std(dv)/sqrt(length(dv)),z3=z[1],z4=z[2],n_selection=10000,n_holdout=50000)
    E.atomic_serialize(checkpoint,(points=rows,summary=summary));println("Economic attribution complete: ",name);flush(stdout)
end
results=[deserialize(joinpath(OUT,"checkpoints/attribution_$(v.name).jls")) for v in variants]
E.write_csv(joinpath(OUT,"tables/economic_attribution_points.csv"),vcat([r.points for r in results]...))
E.write_csv(joinpath(OUT,"tables/economic_attribution_gains.csv"),[r.summary for r in results])
