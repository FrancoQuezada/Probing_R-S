using Test, Statistics, Serialization, Random, LinearAlgebra
include(joinpath(@__DIR__,"../011probing_high_contrast_interior.jl"))
const H=ProbingHighContrastInterior
const R=H.R
const C=H.C
BLAS.set_num_threads(1)
@testset "011 reference regressions, unchanged physical model and finite-difference geometry" begin
    d=H.arguments(String[]);ops=H.operational_design(d);controls=H.controls(d);p=C.build_policy_grid()
    @test length(ops)==48&&length(unique(H.X.physical_digest(s.model) for s in ops))==48
    @test length(p)==144
    @test all(H.R.projected_step([.12,.1],[g,-g],k)==C.projected_step([.12,.1],[g,-g],k) for g in [-30.,-2.,5.,40.] for k in [1,20,50,100])
    @test H.R.model_dict(ops[1].model)==H.R.model_dict(controls[2].model)
    for op in ops
        @test op.model.delay_penalty in [.84,1.,1.2,1.5,1.8,2.2]
        @test op.model.std_mean[1]==1.0 &&op.model.fast_mean[1]==.75
        @test op.model.revenue_scale==8.0 &&op.model.mean_interarrival==.8 &&op.model.fast_routing_cost==.7
        for c in 1:4,z in 0:.1:1
            @test 0<=R.p_hit(c,z,op.model)<=1&&0<=R.p_false(c,z,op.model)<=1
        end
    end
    for s in controls
        old=deserialize(joinpath(d["reference"],"checkpoints/surface_high_$(s.id).jls"))
        @test R.model_dict(old.model)==R.model_dict(s.model)
        @test H.landscape_metrics(s,merge(old,(source="010 exact reference",))).best_value==old.metrics.best_value
        primitive=R.primitive_bank(s.model,5000,old.seed)
        for (i,j) in ((6,11),(16,16))
            z=[old.grid[i],old.grid[j]];Y=R.rewards(p,R.materialize_bank(primitive,z,s.model),s.model)
            @test maximum(vec(mean(Y;dims=2)))-R.cost(s.model,z)==old.points[(i-1)*21+j].gross-R.cost(s.model,z)
        end
    end
    # A known concave NET surface; the gross benefit includes its linear cost.
    cfg=controls[2].model;s=(id="synthetic",op="test",model=cfg,calibrated=true)
    grid=collect(0:.25:1);noise=randn(MersenneTwister(44),30);noise.-=mean(noise)
    net(z)=100-20(z[1]-.5)^2-30(z[2]-.5)^2
    point(z)=(z=z,gross=net(z)+R.cost(cfg,z),policy=1,rewards=net(z)+R.cost(cfg,z).+noise,se=std(noise)/sqrt(30),means=[net(z)+R.cost(cfg,z)])
    raw=(grid=grid,points=[point([x,y]) for x in grid for y in grid],initial=point(C.Z0),source="synthetic",n_paths=30)
    metrics=H.landscape_metrics(s,raw)
    @test metrics.best_z3==metrics.best_z4==.5
    @test metrics.boundary_distance==.5
    @test metrics.R_search≈12.5
    @test metrics.curvature3≈-40&&metrics.curvature4≈-60
    @test metrics.epsilon05_fraction==1/25
    @test metrics.competitive_components==1
    @test isnan(H.X.safe_snr(0.,0.))
end
