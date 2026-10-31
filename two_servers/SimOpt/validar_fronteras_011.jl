# Independent audit of each edge's best fine-grid node, before optimization.
include(joinpath(@__DIR__,"011probing_high_contrast_interior.jl"))
using .ProbingHighContrastInterior
using Serialization, Statistics, LinearAlgebra, TOML, SHA
const H=ProbingHighContrastInterior
const X=H.X
const E=H.E
const R=H.R
const C=H.C
const OUT=joinpath(@__DIR__,"resultados/011probing_high_contrast_interior")
BLAS.set_num_threads(1)
function validate_edges(d,s)
    path=joinpath(OUT,"checkpoints/boundary_validation_$(s.id).jls")
    isfile(path)&&return deserialize(path)
    surf=H.surface(d,s,"high");z=[surf.metrics.best_z3,surf.metrics.best_z4]
    selector_seed=H.seed(s.id,"geometry_audit",0,"selection");holdout_seed=H.seed(s.id,"geometry_audit",0,"holdout")
    selector=R.primitive_bank(s.model,d["selection_paths"],selector_seed;njobs=d["jobs"])
    hold=R.primitive_bank(s.model,d["audit_paths"],holdout_seed;njobs=d["jobs"]);policies=C.build_policy_grid()
    center=H.audit_point(d,s,"geometry",z,selector,hold,policies);rows=NamedTuple[]
    for j in 1:2,edge in (0.,1.)
        eligible=filter(p->p.z[j]==edge,surf.points)
        p=eligible[argmax([p.gross-R.cost(s.model,p.z) for p in eligible])]
        evaluated=H.audit_point(d,s,"geometry",p.z,selector,hold,policies)
        delta=center.rewards-evaluated.rewards .-(center.cost-evaluated.cost);se=std(delta)/sqrt(length(delta))
        push!(rows,(scenario=s.id,edge="z$(j+2)=$(Int(edge))",z3=p.z[1],z4=p.z[2],
            center_z3=z[1],center_z4=z[2],center_policy=center.selected_policy,edge_policy=evaluated.selected_policy,
            center_holdout=center.objective,edge_holdout=evaluated.objective,center_minus_edge=mean(delta),paired_se=se,
            descriptive_lower95=mean(delta)-1.96se,snr=X.safe_snr(mean(delta),se),
            selection_paths=d["selection_paths"],holdout_paths=d["audit_paths"],selection_seed=selector_seed,holdout_seed=holdout_seed))
    end
    E.write_csv(joinpath(OUT,"audits/$(s.id)_best_boundaries.csv"),rows);E.atomic_serialize(path,rows)
    println("Independent best-edge audit: ",s.id);flush(stdout);rows
end
function main()
    d=H.arguments(["--resume"]);scenarios=deserialize(joinpath(OUT,"configs/scenarios.jls"))
    selection=deserialize(joinpath(OUT,"configs/selected_instances.jls"));ids=[r.scenario for r in selection]
    selected=filter(s->s.id in ids,scenarios)
    Threads.@threads :dynamic for s in selected;validate_edges(d,s);end
    rows=vcat([validate_edges(d,s) for s in selected]...);E.write_csv(joinpath(OUT,"tables/independent_best_boundaries.csv"),rows)
    # This is a recommendation based ONLY on geometry, never algorithm outcomes.
    path=joinpath(OUT,"configs/main_instance.toml");meta=TOML.parsefile(path)
    if !haskey(meta,"boundary_audit_reviewed")
        any(startswith(basename(f),"search_I") for f in readdir(joinpath(OUT,"checkpoints")))&&error("Review geometry before new searches")
        new=filter(s->!(s.id in H.CONTROL_IDS),selected)
        passed=filter(s->H.surface(d,s,"high").metrics.boundary_distance>=.1-1e-9&&all(r.descriptive_lower95>0 for r in rows if r.scenario==s.id),new)
        prior=meta["scenario"]
        function rates(id)
            rs=deserialize(joinpath(OUT,"checkpoints/diagnostic_$id.jls"))
            q=filter(r->r.method=="Fresh"&&r.point=="grid_best"&&r.iteration==20,rs)
            (component=mean(r.clipped_fraction for r in q),step=mean(r.any_clipping for r in q),projection=mean(r.projected for r in q))
        end
        usable=filter(s->begin r=rates(s.id);r.component<.3&&r.step<.5&&r.projection<.2 end,passed)
        if !(prior in [s.id for s in passed])&&!isempty(usable)
            cp(path,joinpath(OUT,"configs/main_instance_pre_boundary.toml");force=false)
            meta["scenario"]=first(usable).id;meta["criterion"]*="; independent validation against each edge's best fine-grid node"
        end
        r=rates(meta["scenario"])
        meta["preliminary_clip_components"]=r.component;meta["preliminary_clip_steps"]=r.step;meta["preliminary_projection"]=r.projection
        meta["meets_preliminary_usable_thresholds"]=r.component<.3&&r.step<.5&&r.projection<.2
        meta["boundary_audit_reviewed"]=true;meta["beats_tested_best_edge_nodes_at_descriptive95"]=meta["scenario"] in [s.id for s in passed]
        meta["boundary_audit_note"]="Exploratory paired intervals for four selected edge nodes; no global optimum certification"
        meta["boundary_audit_source_sha256"]=bytes2hex(sha256(read(@__FILE__)))
        open(path,"w") do io;TOML.print(io,meta;sorted=true);end
    end
end
main()
