# Compatible correction of a DERIVED diagnostic, after the complete computation.
# Search decisions, objective estimates, SEs, seeds and model parameters are unchanged.
include(joinpath(@__DIR__,"010probing_instance_landscape_search.jl"))
using .ProbingLandscapeSearch
using Serialization, TOML, SHA
const X=ProbingLandscapeSearch
const E=X.E
const R=X.R
const OUT=joinpath(@__DIR__,"resultados/010probing_instance_landscape_search")
configpath=joinpath(OUT,"configs/experiment.toml")
old=TOML.parsefile(configpath);d=X.arguments(["--resume"])
a=deepcopy(old["science"]);b=deepcopy(d["science"])
oldsource=a["sources"]["010probing_instance_landscape_search.jl"]
newsource=b["sources"]["010probing_instance_landscape_search.jl"]
a["sources"]["010probing_instance_landscape_search.jl"]=newsource
@assert isequal(a,b) "Only the documented SNR source correction is compatible"
@assert isfile(joinpath(OUT,"COMPUTATION_COMPLETE.toml"))
changed=Ref(0)
for file in filter(f->startswith(basename(f),"evaluation_L")&&endswith(f,".jls"),readdir(joinpath(OUT,"checkpoints");join=true))
    result=deserialize(file)
    pairs=[merge(r,(snr=X.safe_snr(r.difference,r.paired_se),)) for r in result.paired]
    gaps=[merge(r,(snr=X.safe_snr(r.grid_reference_gap,r.paired_se),)) for r in result.gaps]
    changed[]+=count(!isequal(x.snr,y.snr) for (x,y) in zip(result.paired,pairs))
    changed[]+=count(!isequal(x.snr,y.snr) for (x,y) in zip(result.gaps,gaps))
    # Preserve ALL other fields exactly, including every estimated difference and SE.
    for (before,after) in zip(result.paired,pairs),k in propertynames(before)
        k==:snr||(@assert isequal(getproperty(before,k),getproperty(after,k)))
    end
    for (before,after) in zip(result.gaps,gaps),k in propertynames(before)
        k==:snr||(@assert isequal(getproperty(before,k),getproperty(after,k)))
    end
    E.atomic_serialize(file,merge(result,(paired=pairs,gaps=gaps)))
    sample=result.finals[1];stem=joinpath(OUT,"comparisons",sample.scenario,"rep_"*lpad(string(sample.rep),2,'0'))
    E.write_csv(stem*"_paired.csv",pairs);E.write_csv(stem*"_grid_gaps.csv",gaps)
end
old["fingerprint"]=d["fingerprint"];old["science"]=d["science"]
old["computation_fingerprint"]=get(old,"computation_fingerprint",TOML.parsefile(joinpath(OUT,"COMPUTATION_COMPLETE.toml"))["fingerprint"])
old["diagnostic_revision"]="0/0 SNR is undefined (NaN); no simulations or algorithm decisions changed"
open(configpath,"w") do io;TOML.print(io,old;sorted=true);end
open(joinpath(OUT,"validation/snr_revision.toml"),"w") do io
    TOML.print(io,Dict("old_source_sha256"=>oldsource,"new_source_sha256"=>newsource,
        "computation_fingerprint"=>old["computation_fingerprint"],"current_fingerprint"=>d["fingerprint"],
        "changed_diagnostic_entries"=>changed[],"unchanged_model_search_objectives_and_se"=>true,
        "note"=>"Original generating surface fingerprints remain archived with the exact source used. Compatible revision changes only SNR diagnostic for zero SE; every difference and SE is preserved."))
end
cp(joinpath(@__DIR__,"010probing_instance_landscape_search.jl"),joinpath(OUT,"reproduction/010probing_instance_landscape_search.jl");force=true)
println("Corrected derived SNR entries: ",changed[],"; no scientific simulation changed.")
