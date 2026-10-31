# Only the external optimizer is new. All model and validation functions are original.
include(joinpath(@__DIR__,"019probing_i0252_statistical.jl"))
module DifferentialEvolution022
using ..I0252Statistical
using BlackBoxOptim, Random, Statistics, LinearAlgebra, Serialization, TOML, SHA, Test, Printf
const S=I0252Statistical;const V=S.V;const R=S.R;const C=S.C;const E=S.E;const X=S.X
const METHOD="differential_evolution"
const VARIANT=:adaptive_de_rand_1_bin_radiuslimited
const ID=V.ID
de_seed(cohort,rep)=parse(Int,bytes2hex(sha256("DE022-v1/$ID/$cohort/$rep/optimizer"))[1:15];base=16)+1

function seeds(cohort,rep)
    original=cohort=="statistical" ? S.REP_OFFSET+rep : rep
    @assert cohort in ("statistical","017")
    Dict("optimizer"=>de_seed(cohort,rep),"training"=>X.seed(ID,"compare",original,"search",1),
        "trace"=>X.seed(ID,"compare",original,"trace"),
        "selection"=>X.seed(ID,"compare",original,"selection"),"holdout"=>X.seed(ID,"compare",original,"holdout"))
end

"""Direct original simulator: max of policy sample MEANS, with cost once.
No LR, score, Green estimator or gradient is invoked here."""
function direct(z,primitives,policies,cfg)
    @assert length(z)==2 && all(isfinite,z) && all(0 .<= z .<= 1)
    bank=R.materialize_bank(primitives,z,cfg)
    Y=R.rewards(policies,bank,cfg)
    values=vec(mean(Y;dims=2));i=argmax(values)
    objective=values[i]-R.cost(cfg,z)
    @assert all(isfinite,values) && isfinite(objective)
    (objective=objective,gross=values[i],policy=i,se=std(view(Y,i,:))/sqrt(length(primitives)),values=values)
end

function search(d,cfg,ss;cap=d["de"]["max_objective_evaluations"],n=d["de"]["paths_per_evaluation"])
    # BlackBoxOptim uses task-local default RNG in this version. A fresh seeded
    # task isolates it from explicit MersenneTwister simulator streams.
    task=Task() do
        Random.seed!(ss["optimizer"])
        started=time_ns();policies=C.build_policy_grid()
        primitives=R.primitive_bank(cfg,n,ss["training"];njobs=d["jobs"])
        primitive_seconds=(time_ns()-started)/1e9
        rows=NamedTuple[];best=-Inf;best_z=copy(C.Z0);best_policy=0
        phase="initial_point"
        function objective(z)
            count=length(rows)+1
            @assert count<=cap "Hard simulation budget guard; no extra simulator call made"
            ts=time_ns();p=direct(z,primitives,policies,cfg);seconds=(time_ns()-ts)/1e9
            if p.objective>best
                best=p.objective;best_z=copy(z);best_policy=p.policy
            end
            push!(rows,(objective_evaluation=count,iteration=count,phase=phase,
                candidate_z3=z[1],candidate_z4=z[2],candidate_objective=p.objective,candidate_policy=p.policy,
                z3=best_z[1],z4=best_z[2],objective_internal=best,selected_policy=best_policy,
                new_primitive_paths=count==1 ? n : 0,cumulative_new_primitive_paths=n,
                paths_evaluated=n,cumulative_path_evaluations=n*count,
                policy_path_evaluations=n*length(policies),cumulative_policy_path_evaluations=n*count*length(policies),
                simulator_seconds=seconds,cumulative_optimization_seconds=(time_ns()-started)/1e9))
            -p.objective
        end
        initial_fitness=objective(C.Z0) # Count the common starting point explicitly.
        phase="differential_evolution"
        # Standard OptimizationProblem avoids bbsetup(function)'s EXTRA random
        # type-check evaluation, which is not included in library f_calls.
        problem=BlackBoxOptim.FunctionBasedProblem(objective,"022 original routing simulator",
            BlackBoxOptim.MinimizingFitnessScheme,BlackBoxOptim.RectSearchSpace(2,(0.,1.)),nothing)
        callback=ctrl->(length(rows)>=cap-1 && BlackBoxOptim.shutdown!(ctrl);nothing)
        ctrl=bbsetup(problem;Method=VARIANT,PopulationSize=d["de"]["population_size"],
            MaxFuncEvals=cap,TraceMode=:silent,CallbackFunction=callback,CallbackInterval=0.)
        @assert length(rows)==1 "Unexpected unaccounted library setup evaluation"
        result=bboptimize(ctrl,copy(C.Z0))
        elapsed=(time_ns()-started)/1e9
        @assert cap-1<=length(rows)<=cap
        @assert BlackBoxOptim.f_calls(result)+1==length(rows)
        @assert isapprox(best,-min(initial_fitness,best_fitness(result));atol=1e-10,rtol=1e-12)
        @assert length(unique([ss[k] for k in keys(ss)]))==5
        # Include all optimizer setup/termination overhead in the endpoint clock.
        rows[end]=merge(rows[end],(cumulative_optimization_seconds=elapsed,))
        (rows=rows,zhist=vcat([copy(C.Z0)],[[r.z3,r.z4] for r in rows]),
            cumulative=vcat([0.],[r.cumulative_optimization_seconds for r in rows]),
            seconds=elapsed,primitive_seconds=primitive_seconds,primitives=primitives,
            objective_evaluations=length(rows),library_objective_evaluations=BlackBoxOptim.f_calls(result),
            best_z=best_z,best_training_objective=best,training_policy=best_policy,
            stop_reason=BlackBoxOptim.stop_reason(result),library_best_z=best_candidate(result),
            final_f_parameters=copy(BlackBoxOptim.optimizer(ctrl).modify.params.fs),
            final_cr_parameters=copy(BlackBoxOptim.optimizer(ctrl).modify.params.crs))
    end
    schedule(task);fetch(task)
end

signature(raw)=[(r.candidate_z3,r.candidate_z4,r.candidate_objective,r.z3,r.z4,r.objective_internal) for r in raw.rows]
function tests(output,c,cfg)
    d=c["science"];cohort=d["cohort"];ss=seeds(cohort,1);policies=C.build_policy_grid()
    result=@testset "022 direct simulator, CRN, bounds, seeds, DE and budgets" begin
        @test pkgversion(BlackBoxOptim)==v"0.6.12"
        @test R.model_digest(cfg)==d["model_sha256"]
        @test length(policies)==144 && C.Z0==d["Z0"]
        primitive=R.primitive_bank(cfg,30,ss["training"];njobs=45)
        original=deepcopy(primitive)
        for z in ([0.,0.],[1.,1.],C.Z0,[.3,.7],[.9,.2])
            got=direct(z,primitive,policies,cfg)
            expected=only(S.direct_trace(cfg,[z],30,ss["training"],45))
            @test got.objective==expected.objective && got.policy==expected.policy && got.se==expected.se
            Y=R.rewards(policies,R.materialize_bank(primitive,z,cfg),cfg)
            @test got.objective==maximum(vec(mean(Y;dims=2)))-R.cost(cfg,z)
            materialized=R.materialize_bank(primitive,z,cfg)
            for (path,base) in zip(materialized,primitive),(job,p) in zip(path.jobs,base.jobs)
                probability=R.signal_probability(p.class,p.complex,C.full_z(z)[p.class],cfg)
                @test job.signal==(p.u_signal<probability ? 1 : 0)
                @test (job.arrival,job.class,job.complex,job.value,job.service_fast,job.service_std)==
                    (p.arrival,p.class,p.complex,p.value,p.service_fast,p.service_std)
            end
        end
        @test repr(primitive)==repr(original)
        # Empirical Bernoulli checks on a common uniform sample, separately from
        # exact threshold identities. Includes endpoints and both active classes.
        uniforms=rand(MersenneTwister(220022),20000)
        for cl in (3,4),complex in (false,true),z in (0.,.3,.7,1.)
            p=R.signal_probability(cl,complex,z,cfg)
            @test 0<=p<=1
            @test abs(mean(uniforms.<p)-p)<=6sqrt(p*(1-p)/length(uniforms))+1/length(uniforms)
        end
        @test_throws AssertionError direct([-.01,.5],primitive,policies,cfg)
        @test_throws AssertionError direct([.5,1.01],primitive,policies,cfg)
        @test_throws AssertionError direct([NaN,.5],primitive,policies,cfg)
        a=search(d,cfg,ss;cap=20,n=20);b=search(d,cfg,ss;cap=20,n=20)
        @test isequal(signature(a),signature(b))
        @test a.objective_evaluations==b.objective_evaluations
        @test a.library_objective_evaluations+1==a.objective_evaluations
        @test all(r->0<=r.z3<=1 && 0<=r.z4<=1,a.rows)
        @test all(diff([r.objective_internal for r in a.rows]).>=0)
        @test sum(r.new_primitive_paths for r in a.rows)==20
        @test sum(r.paths_evaluated for r in a.rows)==20a.objective_evaluations
        @test sum(r.policy_path_evaluations for r in a.rows)==20*144*a.objective_evaluations
        S.direct_trace(cfg,a.zhist,20,ss["trace"],45)
        @test isequal(signature(a),signature(search(d,cfg,ss;cap=20,n=20)))
        streams=[seeds(cohort,r) for r in 1:50]
        flat=[s[k] for s in streams for k in keys(s)]
        @test length(unique(flat))==250
        # An inexpensive fixed-observation regression, no old optimizer rerun.
        old=d["historical_output"]
        archived=deserialize(joinpath(old,"raw/fresh/rep_0001/convergence.jls")).rows[1]
        historical_seed=S.seeds(1,100)["trace"]
        p=only(S.direct_trace(cfg,[C.Z0],1000,historical_seed,45))
        @test isequal(p,archived)
        smoke=search(d,cfg,ss;cap=8,n=500)
        @test 7<=smoke.objective_evaluations<=8
        @test smoke.rows[end].cumulative_path_evaluations<=4000
        E.atomic_serialize(joinpath(output,"validation/smoke_search.jls"),smoke)
        endp=S.endpoint(cfg,smoke.best_z,merge(d,Dict("selection_paths"=>30,"holdout_paths"=>80)),ss)
        @test isfinite(endp.objective) && isfinite(endp.se)
    end
    S.atomic_toml(joinpath(output,"validation/preflight_tests.toml"),Dict("passed"=>true,
        "assertions"=>Test.get_test_counts(result).passes,"instance"=>ID,"cohort"=>cohort,
        "original_optimizers_executed"=>0,"blackboxoptim_version"=>string(pkgversion(BlackBoxOptim)),
        "trained_validation_independent"=>true,"fixed_observation_original_regression"=>true))
end

function run_rep(output,c,cfg,rep)
    d=c["science"];ss=seeds(d["cohort"],rep);fp=c["fingerprint"]
    parent=joinpath(output,"raw",METHOD);mkpath(parent)
    name=@sprintf("rep_%04d",rep);final=joinpath(parent,name);stage=joinpath(parent,".$name.partial")
    mkpath(joinpath(output,"raw/.locks"))
    open(joinpath(output,"raw/.locks","$name.lock"),"a") do lock
        ccall(:flock,Cint,(Cint,Cint),Base.fd(lock),6)==0||error("Replication already running")
        isdir(final)&&error("Published result exists; validate and skip it")
        mkpath(stage);wall=time_ns()
        state=joinpath(stage,"state.toml");started=S.utcnow()
        if isfile(state)
            old=TOML.parsefile(state);@assert old["fingerprint"]==fp;started=old["started_utc"]
        end
        S.atomic_toml(state,Dict("fingerprint"=>fp,"started_utc"=>started,"replication"=>rep,"status"=>"running"))
        path=joinpath(stage,"search.jls")
        if isfile(path)
            cp=deserialize(path);@assert cp.fingerprint==fp;raw=cp.raw
        else
            raw=search(d,cfg,ss);E.atomic_serialize(path,(fingerprint=fp,raw=raw))
        end
        println("SEARCH\t$rep\t$(raw.seconds)\t$(raw.objective_evaluations)");flush(stdout)
        E.write_csv(joinpath(stage,"objective_history.csv"),raw.rows)
        E.write_csv(joinpath(stage,"iteration_diagnostics.csv"),raw.rows)
        E.write_csv(joinpath(stage,"search_seeds.csv"),[(stream=k,seed=ss[k]) for k in sort!(collect(keys(ss)))])
        path=joinpath(stage,"convergence.jls")
        if isfile(path)
            cp=deserialize(path);@assert cp.fingerprint==fp;trace=cp.rows;trace_seconds=cp.seconds
        else
            ts=time_ns();trace=S.direct_trace(cfg,raw.zhist,d["trace_paths"],ss["trace"],d["jobs"])
            trace_seconds=(time_ns()-ts)/1e9;E.atomic_serialize(path,(fingerprint=fp,rows=trace,seconds=trace_seconds))
        end
        rows=NamedTuple[];n=d["de"]["paths_per_evaluation"]
        for t in 0:raw.objective_evaluations
            z=raw.zhist[t+1];p=trace[t+1]
            push!(rows,(replication=rep,method=METHOD,iteration=t,objective_evaluation=t,z3=z[1],z4=z[2],
                objective_crn=p.objective,objective_internal=t==0 ? missing : raw.rows[t].objective_internal,
                cumulative_optimization_seconds=raw.cumulative[t+1],crn_selected_policy=p.policy,
                crn_selected_policy_se=p.se,evaluation_paths=d["trace_paths"],evaluation_seed=ss["trace"],
                cumulative_new_search_paths=t==0 ? 0 : n,cumulative_path_evaluations=t*n,
                cumulative_policy_path_evaluations=t*n*length(C.build_policy_grid()),
                ess=missing,lr_clips=missing,K_eff=missing,g3=missing,g4=missing))
        end
        E.write_csv(joinpath(stage,"trajectory.csv"),rows)
        path=joinpath(stage,"final.jls")
        if isfile(path)
            cp=deserialize(path);@assert cp.fingerprint==fp;p=cp.result;final_seconds=cp.seconds
        else
            ts=time_ns();p=S.endpoint(cfg,raw.best_z,d,ss);final_seconds=(time_ns()-ts)/1e9
            E.atomic_serialize(path,(fingerprint=fp,result=p,seconds=final_seconds))
        end
        z=raw.best_z;paths=n*raw.objective_evaluations;policies=length(C.build_policy_grid())
        E.write_csv(joinpath(stage,"final.csv"),[(replication=rep,method=METHOD,z3=z[1],z4=z[2],selected_policy=p.policy,
            objective=p.objective,gross=p.gross,probing_cost=R.cost(cfg,z),evaluation_se=p.se,
            mc_ci95_low=p.objective-1.959963984540054*p.se,mc_ci95_high=p.objective+1.959963984540054*p.se,
            objective_internal=raw.best_training_objective,training_policy=raw.training_policy,
            optimization_seconds=raw.seconds,diagnostic_seconds=0.,convergence_evaluation_seconds=trace_seconds,
            final_evaluation_seconds=final_seconds,iterations=raw.objective_evaluations,
            objective_evaluations=raw.objective_evaluations,library_objective_evaluations=raw.library_objective_evaluations,
            new_search_paths=n,path_evaluations=paths,policy_path_evaluations=paths*policies,
            selection_paths=d["selection_paths"],holdout_paths=d["holdout_paths"],
            selection_seed=ss["selection"],holdout_seed=ss["holdout"],stop_reason=raw.stop_reason)])
        meta=Dict("status"=>"complete","fingerprint"=>fp,"replication"=>rep,"method"=>METHOD,"scenario"=>ID,
            "cohort"=>d["cohort"],"seeds"=>ss,"science"=>d,"model"=>R.model_dict(cfg),
            "started_utc"=>started,"completed_utc"=>S.utcnow(),"optimization_seconds"=>raw.seconds,
            "convergence_evaluation_seconds"=>trace_seconds,"final_evaluation_seconds"=>final_seconds,
            "last_attempt_seconds"=>(time_ns()-wall)/1e9,"julia_version"=>string(VERSION),
            "blackboxoptim_version"=>string(pkgversion(BlackBoxOptim)),"blas_threads"=>BLAS.get_num_threads(),
            "cpu"=>Sys.CPU_NAME,"stop_reason"=>raw.stop_reason,"new_primitive_paths"=>n,
            "path_evaluations"=>paths,"policy_path_evaluations"=>paths*policies,
            "objective_evaluations"=>raw.objective_evaluations,"max_path_evaluations"=>d["de"]["path_budget"],
            "optimization_uses_lr"=>false,"optimization_uses_gradients"=>false)
        S.atomic_toml(joinpath(stage,"metadata.toml"),meta)
        files=["trajectory.csv","final.csv","iteration_diagnostics.csv","objective_history.csv","search_seeds.csv",
            "metadata.toml","search.jls","convergence.jls","final.jls"]
        S.atomic_toml(joinpath(stage,"complete.toml"),Dict("status"=>"complete","fingerprint"=>fp,
            "files"=>Dict(f=>S.sha(joinpath(stage,f)) for f in files)))
        mv(stage,final;force=false)
    end
end

function main()
    BLAS.set_num_threads(1);output=abspath(ARGS[1]);c,cfg=S.model(output)
    @assert pkgversion(BlackBoxOptim)==v"0.6.12"
    if length(ARGS)>1 && ARGS[2]=="test";tests(output,c,cfg);return;end
    ss=seeds(c["science"]["cohort"],0)
    search(c["science"],cfg,ss;cap=8,n=20)
    S.direct_trace(cfg,[C.Z0],20,ss["trace"],45)
    S.endpoint(cfg,C.Z0,merge(c["science"],Dict("selection_paths"=>20,"holdout_paths"=>20)),ss)
    println("READY");flush(stdout)
    for line in eachline(stdin)
        isempty(line)&&continue;rep=parse(Int,line)
        try
            run_rep(output,c,cfg,rep);println("COMPLETE\t$rep");flush(stdout)
        catch err
            showerror(stdout,err,catch_backtrace());println();println("FAILED\t$rep");flush(stdout)
        end
        GC.gc()
    end
end
end
if abspath(PROGRAM_FILE)==@__FILE__;DifferentialEvolution022.main();end
