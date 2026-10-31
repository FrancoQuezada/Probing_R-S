# Archive per-job routing decisions from the already fixed reference-B samples.
# This adds an audit artifact; it does not alter characterization or selection.
using Serialization, TOML, SHA, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "probing_regime_experiment.jl"))
const X=ProbingRegimeExperiment
const R=X.R
BLAS.set_num_threads(1)

function archive_routing(args)
    d=X.parse_arguments(vcat(["--characterize", "--resume"],args))
    X.verify_sources(d)
    out=d["output"];stored=TOML.parsefile(joinpath(out,"configs/configuration.toml"))
    stored["fingerprint"]==d["fingerprint"]||error("Reference configuration mismatch")
    source_sha=bytes2hex(sha256(read(@__FILE__)))
    policies=X.C.build_policy_grid();totals=zeros(Int,Threads.nthreads())
    Threads.@threads :static for s in X.scenario_targets(d)
        primitive=R.primitive_bank(s.model,d["reference_paths"],X.seed(s.id,"characterization",0,"reference_B");njobs=d["jobs_per_path"])
        for (point,z) in X.points()
            result=deserialize(joinpath(out,"checkpoints/characterization_$(s.id)_$point.jls"))
            result.fingerprint==d["fingerprint"]||error("Invalid characterization")
            path=joinpath(out,"characterization",s.id,"$(point)_routing_decisions.csv.gz")
            meta=joinpath(out,"checkpoints/routing_decisions_$(s.id)_$point.toml")
            if isfile(path)&&isfile(meta)
                m=TOML.parsefile(meta)
                if m["fingerprint"]==d["fingerprint"]&&m["source_sha256"]==source_sha&&m["sha256"]==bytes2hex(sha256(read(path)))
                    totals[Threads.threadid()]+=m["rows"]
                    println("REUSE routing decisions $(s.id) $point");continue
                end
            end
            bank=R.materialize_bank(primitive,z,s.model)
            masks=Dict{Int,Vector{UInt64}}();rows=0;temporary=path*".tmp"
            open(pipeline(`gzip -c`,stdout=temporary),"w") do io
                println(io,"scenario,point,policy,path_index,fast_route_mask,fast_fraction,fast_utilization,total_wait,delay_penalty,routing_cost,reward")
                for row in result.routing
                    details=[R.routing_details(policies[row.policy],p,s.model) for p in bank]
                    isapprox(mean(x.reward for x in details),row.reward;atol=1e-10,rtol=1e-12)||error("Routing replay reward mismatch")
                    policy_masks=UInt64[]
                    for (l,x) in enumerate(details)
                        mask=foldl((a,j)->a | (UInt64(x.decisions[j])<<(j-1)),eachindex(x.decisions);init=UInt64(0))
                        push!(policy_masks,mask)
                        println(io,join((s.id,point,row.policy,l,mask,x.fast_fraction,x.fast_utilization,x.total_wait,x.delay_penalty,x.routing_cost,x.reward),','))
                        rows+=1
                    end
                    masks[row.policy]=policy_masks
                end
            end
            for row in result.routing_pairs
                delta=[count_ones(xor(a,b))/d["jobs_per_path"] for (a,b) in zip(masks[row.policy_i],masks[row.policy_j])]
                isapprox(mean(delta),row.disagreement;atol=1e-14,rtol=1e-12)||error("Routing decision disagreement mismatch")
            end
            mv(temporary,path;force=true)
            open(meta,"w") do io
                TOML.print(io,Dict("fingerprint"=>d["fingerprint"],"source_sha256"=>source_sha,
                    "sha256"=>bytes2hex(sha256(read(path))),"rows"=>rows);sorted=true)
            end
            totals[Threads.threadid()]+=rows
            println("COMPLETE routing decisions $(s.id) $point rows=$rows");flush(stdout)
        end
    end
    open(joinpath(out,"validation/routing_decisions.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true,"rows"=>sum(totals),"fingerprint"=>d["fingerprint"],
            "source_sha256"=>source_sha,"encoding"=>"UInt64: bit (job_index-1) equals 1 for FAST, 0 for STD; jobs and paths indexed from 1.",
            "reference"=>"Exact existing reference_B seeds, same principal policies, no new sample or changed classification.");sorted=true)
    end
end
archive_routing(ARGS)
