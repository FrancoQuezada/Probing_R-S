# Validate published data and replay stored simulator inputs; no optimizer reruns.
output=abspath(ARGS[1])
include(joinpath(output,"reproduction/022_differential_evolution.jl"))
using .DifferentialEvolution022, Serialization, Test, LinearAlgebra, Printf, TOML
const D=DifferentialEvolution022;const S=D.S;const R=D.R;const C=D.C
BLAS.set_num_threads(1)
c,cfg=S.model(output);d=c["science"];cohort=d["cohort"]
count=cohort=="017" ? 3 : 50
result=@testset "022 saved fixed CRN, objective replays and reproducible prefix" begin
    policies=C.build_policy_grid()
    for rep in 1:count
        path=joinpath(output,"raw",D.METHOD,@sprintf("rep_%04d",rep))
        cp=deserialize(joinpath(path,"search.jls"));@test cp.fingerprint==c["fingerprint"]
        raw=cp.raw;ss=D.seeds(cohort,rep);n=d["de"]["paths_per_evaluation"]
        @test length(raw.primitives)==n
        @test all(p->length(p.jobs)==45,raw.primitives)
        regenerated=R.primitive_bank(cfg,n,ss["training"];njobs=45)
        @test repr(raw.primitives)==repr(regenerated)
        @test raw.rows[end].cumulative_path_evaluations==n*raw.objective_evaluations<=d["de"]["path_budget"]
        @test raw.rows[end].cumulative_policy_path_evaluations==n*144*raw.objective_evaluations
        @test raw.library_objective_evaluations+1==raw.objective_evaluations
        @test length(raw.final_f_parameters)==length(raw.final_cr_parameters)==d["de"]["population_size"]
        # Directly replay three evaluated candidates for every published run.
        for k in unique([1,cld(length(raw.rows),2),length(raw.rows)])
            row=raw.rows[k];z=[row.candidate_z3,row.candidate_z4]
            p=D.direct(z,raw.primitives,policies,cfg)
            @test p.objective==row.candidate_objective && p.policy==row.candidate_policy
        end
        @test raw.best_z==raw.zhist[end]
        @test all(diff([r.objective_internal for r in raw.rows]).>=0)
        if rep==1
            smoke=deserialize(joinpath(output,"validation/smoke_search.jls"))
            @test D.signature(smoke)==D.signature(raw)[1:length(smoke.rows)]
        end
    end
end
S.atomic_toml(joinpath(output,"validation/saved_de_audit.toml"),Dict("passed"=>true,
    "assertions"=>Test.get_test_counts(result).passes,"instance"=>D.ID,"cohort"=>cohort,
    "published_replications_checked"=>count,"direct_candidate_replays"=>3count,
    "optimization_runs_started"=>0,"primitive_banks_regenerated_for_validation"=>count,
    "primitive_regeneration_paths"=>count*d["de"]["paths_per_evaluation"],
    "audit_policy_path_evaluations"=>3count*d["de"]["paths_per_evaluation"]*144,
    "audit_source_sha256"=>S.sha(@__FILE__),
    "published_manifest_sha256"=>Dict(string(rep)=>S.sha(joinpath(output,"raw",D.METHOD,@sprintf("rep_%04d",rep),"complete.toml")) for rep in 1:count),
    "audit_work_excluded_from_optimization_budget"=>true))
