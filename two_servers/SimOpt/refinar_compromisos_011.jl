# Adaptive, GEOMETRY/GRADIENT-only extension, before any full new searches.
# No original source, algorithm or hyperparameter is modified.
include(joinpath(@__DIR__,"011probing_high_contrast_interior.jl"))
using .ProbingHighContrastInterior
using Serialization, Statistics, TOML, SHA, LinearAlgebra
const H=ProbingHighContrastInterior
const X=H.X
const E=H.E
const R=H.R
const OUT=joinpath(@__DIR__,"resultados/011probing_high_contrast_interior")
BLAS.set_num_threads(1)
# Each grid point uses the same exact frozen primitive bank. Only independent
# point tasks run concurrently; policy/path reductions and algorithms are unchanged.
function parallel_high(d,s)
    surfacepath=joinpath(OUT,"checkpoints/surface_high_$(s.id).jls")
    isfile(surfacepath)&&return deserialize(surfacepath)
    physical=X.physical_digest(s.model);rawpath=joinpath(OUT,"checkpoints/physical_high_$physical.jls")
    if !isfile(rawpath)
        rngseed=H.seed("shared","high",0,"surface")
        primitive=R.primitive_bank(s.model,d["high_paths"],rngseed;njobs=d["jobs"])
        grid=collect(range(0.,1.;length=d["high_grid"]));zs=[[x,y] for x in grid for y in grid]
        policies=H.C.build_policy_grid();points=Vector{Any}(undef,length(zs))
        Threads.@threads :dynamic for i in eachindex(zs)
            points[i]=H.gross_point(d,s,"high",zs[i],primitive,policies)
        end
        initial=H.gross_point(d,s,"high",H.C.Z0,primitive,policies)
        raw=(grid=grid,points=points,initial=initial,seed=rngseed,n_paths=d["high_paths"],source="011 direct CRN; independent nodes in parallel")
        E.atomic_serialize(rawpath,raw)
    end
    H.surface(d,s,"high")
end
function main()
    d=H.arguments(["--resume"]);scenarios=deserialize(joinpath(OUT,"configs/scenarios.jls"))
    completed=joinpath(OUT,"configs/extension_complete.toml")
    isfile(completed)&&(println("Adaptive refinement already complete; frozen selection reused.");return)
    any(startswith(basename(f),"search_I") for f in readdir(joinpath(OUT,"checkpoints")))&&error("Freeze extension BEFORE new algorithm searches")
    initial=deserialize(joinpath(OUT,"configs/refinement_selection.jls"));initialids=[r.scenario for r in initial]
    rows=vcat([H.diagnostic_case(d,only(filter(s->s.id==id,scenarios))) for id in initialids]...)
    newids=filter(id->!(id in H.CONTROL_IDS),initialids)
    saturation(id)=mean(r.clipped_fraction for r in rows if r.scenario==id&&r.point=="grid_best"&&r.method=="Fresh"&&r.iteration==20)
    @assert all(saturation(id)>=.5-1e-12 for id in newids) "Extension trigger: all original new cases clip at least half their gradient components"
    path=joinpath(OUT,"configs/refinement_extension.jls")
    if isfile(path);additional=deserialize(path)
    else
        allscreen=[H.surface(d,s,"screen") for s in scenarios];scores=H.ranking(d,allscreen);additional=NamedTuple[]
        for delay in (.84,1.,1.2)
            available=findall(i->scenarios[i].calibrated&&scenarios[i].model.delay_penalty==delay&&
                allscreen[i].metrics.boundary_distance>=.1-1e-9&&!(scenarios[i].id in initialids),eachindex(scenarios))
            isempty(available)&&continue;i=available[argmax(scores[available])]
            push!(additional,(scenario=scenarios[i].id,delay=delay,score=scores[i],reason="geometry-selected lower-delay stratum after all original refined new cases showed >=50% component clipping"))
        end
        E.atomic_serialize(path,additional);E.write_csv(joinpath(OUT,"configs/refinement_extension.csv"),additional)
        open(joinpath(OUT,"configs/refinement_extension_protocol.toml"),"w") do io
            TOML.print(io,Dict("trigger"=>"all seven original new refined cases: Fresh clipped component fraction >=.5 at best node, t20, 30 independent banks",
                "strata"=>[.84,1.,1.2],"criterion"=>"one previously calibrated, interior screen candidate per delay; highest original landscape score, no search outcomes",
                "main_usable_thresholds"=>"Fresh any clipping <.5, component clipping <.3, projection <.2, measured at refined best node with original step",
                "frozen_before_new_searches"=>true,"source_sha256"=>bytes2hex(sha256(read(@__FILE__))),"initial_new_ids"=>newids,
                "initial_component_clipping"=>[saturation(id) for id in newids]))
        end
    end
    extra=filter(s->s.id in [r.scenario for r in additional],scenarios)
    Threads.@threads :dynamic for s in extra;H.surface(d,s,"refine");H.diagnostic_case(d,s);end
    "--prepare" in ARGS&&(println("Additional lower-delay refinements and diagnostics prepared.");return)
    isfile(joinpath(OUT,"configs/selected_instances.jls"))||error("Complete core --refine before finalizing")
    allids=unique([initialids;[s.id for s in extra]]);allscenarios=filter(s->s.id in allids,scenarios)
    refined=[H.surface(d,s,"refine") for s in allscenarios];allrows=vcat([H.diagnostic_case(d,s) for s in allscenarios]...)
    E.write_csv(joinpath(OUT,"refine/all_metrics.csv"),[a.metrics for a in refined]);E.write_csv(joinpath(OUT,"diagnostics/all_presearch_gradients.csv"),allrows)
    scores=H.ranking(d,refined;diagnostics=allrows)
    function rates(id)
        q=filter(r->r.scenario==id&&r.point=="grid_best"&&r.method=="Fresh"&&r.iteration==20,allrows)
        (component=mean(r.clipped_fraction for r in q),step=mean(r.any_clipping for r in q),projection=mean(r.projected for r in q))
    end
    eligible=filter(i->!(refined[i].scenario in H.CONTROL_IDS)&&refined[i].metrics.boundary_distance>=.1-1e-9,eachindex(refined))
    usable=filter(i->begin r=rates(refined[i].scenario);r.component<.3&&r.step<.5&&r.projection<.2 end,eligible)
    priority=isempty(usable) ? eligible : usable
    firsti=priority[argmax(scores[priority])];chosen=[firsti]
    rest=filter(i->i!=firsti&&refined[i].metrics.op!=refined[firsti].metrics.op,eligible)
    preferred=filter(i->i in usable,rest);isempty(preferred)&&(preferred=rest)
    push!(chosen,preferred[argmax(scores[preferred])])
    proposed=refined[chosen];high=Any[]
    for candidate in proposed
        s=only(filter(s->s.id==candidate.scenario,scenarios));push!(high,parallel_high(d,s))
    end
    Threads.@threads :dynamic for i in eachindex(high)
        s=only(filter(s->s.id==high[i].scenario,scenarios));H.geometry_audit(d,s,high[i])
    end
    # Final quality check is still geometry/gradient-only, before searches.
    acceptable=filter(i->high[i].metrics.boundary_distance>=.1-1e-9,eachindex(high))
    isempty(acceptable)||(high=[high[acceptable];high[setdiff(eachindex(high),acceptable)]])
    controls=[H.surface(d,s,"high") for s in H.controls(d)];finalsurfaces=[controls;high]
    selected=[(scenario=a.scenario,order=i,reason=i<=3 ? "mandatory 010 reference" : "adaptive lower-delay geometry plus original-estimator saturation; frozen before algorithms",score=0.,source_stage="high") for (i,a) in enumerate(finalsurfaces)]
    for file in ("selected_instances.csv","selected_instances.jls","selected_instances_provenance.toml","main_instance.toml")
        src=joinpath(OUT,"configs",file);dest=joinpath(OUT,"configs/initial_"*file);isfile(dest)||cp(src,dest)
    end
    E.atomic_serialize(joinpath(OUT,"configs/selected_instances.jls"),selected);E.write_csv(joinpath(OUT,"configs/selected_instances.csv"),selected)
    primary=high[1].scenario;rate=rates(primary)
    open(joinpath(OUT,"configs/main_instance.toml"),"w") do io
        TOML.print(io,Dict("scenario"=>primary,"selected_before_algorithms"=>true,"interior_grid_candidate"=>high[1].metrics.boundary_distance>=.1-1e-9,
            "criterion"=>"calibrated geometry and low-saturation stratum; no algorithm outcomes","preliminary_clip_components"=>rate.component,
            "preliminary_clip_steps"=>rate.step,"preliminary_projection"=>rate.projection,"meets_preliminary_usable_thresholds"=>rate.component<.3&&rate.step<.5&&rate.projection<.2))
    end
    open(joinpath(OUT,"configs/selected_instances_provenance.toml"),"w") do io
        TOML.print(io,Dict("fingerprint"=>d["fingerprint"],"adaptive_protocol"=>"refinement_extension_protocol.toml","no_search_outcomes_used"=>true,
            "initial_selection_preserved"=>true,"source_sha256"=>bytes2hex(sha256(read(@__FILE__)))))
    end
    E.write_csv(joinpath(OUT,"high/metrics.csv"),[a.metrics for a in finalsurfaces])
    allhigh=unique(vcat([r.scenario for r in deserialize(joinpath(OUT,"configs/high_precision_candidates.jls"))],[a.scenario for a in high]))
    E.write_csv(joinpath(OUT,"high/extended_candidate_metrics.csv"),[H.surface(d,only(filter(s->s.id==id,scenarios)),"high").metrics for id in allhigh])
    open(completed,"w") do io;TOML.print(io,Dict("complete"=>true,"additional_refinements"=>length(extra),"selected"=>[a.scenario for a in finalsurfaces],"main_instance"=>primary,"source_sha256"=>bytes2hex(sha256(read(@__FILE__)))));end
    println("Adaptive selection frozen BEFORE algorithms: ",[a.scenario for a in finalsurfaces],"; main=",primary," rates=",rate)
end
main()
