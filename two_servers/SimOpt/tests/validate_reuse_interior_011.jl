using Test, Serialization, SHA
include(joinpath(@__DIR__,"../011probing_high_contrast_interior.jl"))
const H=ProbingHighContrastInterior
const X=H.X
const R=H.R
const E=H.E
@testset "011 exact reuse of all three 010 controls" begin
    d=H.arguments(String[]);d["output"]=joinpath(@__DIR__,"../resultados/011probing_high_contrast_interior/validation/reference_reuse_test")
    mkpath(joinpath(d["output"],"checkpoints"));mkpath(joinpath(d["output"],"configs"));ss=H.controls(d)
    for s in ss,phase in ("refine","high")
        a=H.raw_surface(d,s,phase);old=deserialize(joinpath(d["reference"],"checkpoints/surface_$(phase)_$(s.id).jls"))
        @test a.points==old.points&&a.initial==old.initial&&a.seed==old.seed
        @test occursin("010/",a.source)
        @test H.landscape_metrics(s,a).best_value==old.metrics.best_value
    end
    H.reuse_reference_comparisons(d,ss)
    for s in ss,rep in 1:5,kind in ("search","evaluation")
        file="checkpoints/$(kind)_$(s.id)_$rep.jls"
        @test sha256(read(joinpath(d["output"],file)))==sha256(read(joinpath(d["reference"],file)))
    end
end
