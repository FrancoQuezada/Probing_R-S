# Run the two-server experiments: five instances x six methods x replications, plus the optional objective surface.
#
#   julia --project=. run.jl --instance I0252 --method green --reps 1
#   julia --project=. run.jl --instance L026 --method all --reps 3
#   julia --project=. run.jl --instance all --method all --reps 50            (complete design: 5 x 6 x 50)
#   julia --project=. run.jl --instance all --method all --reps 1 --quick     (smoke test, reduced budgets)
#   julia --project=. run.jl --instance I0252 --surface                       (direct-CRN objective surface, explicit)
#
# Options: --instance <id|all|id,id>  --method <id|all|id,id>  --reps N (replications 1..N) | --reps a:b | --rep r
#          --quick  --surface  --output <dir> (default results/full or results/quick)
# Per (instance, method, replication): search -> direct-CRN evaluation of every visited z -> final selection + holdout.
# Completed (instance, method, replication) rows are never recomputed or overwritten; results are appended.
using LinearAlgebra
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "methods.jl"))

function parse_list(x, all)
    x == "all" && return all
    v = String.(split(x, ','))
    bad = filter(!in(all), v)
    isempty(bad) || error("Unknown value(s) $(join(bad, ", ")); use one of $(join(all, ", ")) or all")
    return v
end

function arguments(args)
    a = Dict{String,Any}("instance" => "all", "method" => "all", "reps" => "1", "quick" => false, "surface" => false)
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("Unexpected argument $(args[i])")
        k = args[i][3:end]
        if k in ("quick", "surface")
            a[k] = true; i += 1
        elseif k in ("instance", "method", "reps", "rep", "output")
            i + 1 <= length(args) || error("--$k needs a value"); a[k] = args[i+1]; i += 2
        else
            error("Unknown option --$k")
        end
    end
    reps = haskey(a, "rep") ? (parse(Int, a["rep"]):parse(Int, a["rep"])) :
           occursin(':', a["reps"]) ? (parse(Int, split(a["reps"], ':')[1]):parse(Int, split(a["reps"], ':')[2])) :
           (1:parse(Int, a["reps"]))
    first(reps) >= 1 || error("Replication ids start at 1")
    output = get(a, "output", joinpath(@__DIR__, "results", a["quick"] ? "quick" : "full"))
    return (instances=parse_list(a["instance"], INSTANCES), methods=parse_list(a["method"], ALL_METHODS), reps=reps,
            quick=a["quick"], surface=a["surface"], output=output)
end

source_sha(f) = bytes2hex(sha256(read(joinpath(@__DIR__, f))))
"""Write run_info.toml once per output directory; refuse to mix results produced under different settings."""
function check_run_info(dir, id, cfg, s, raw)
    info = Dict("instance" => id, "mode" => s["quick"] ? "quick" : "full", "model_sha256" => model_digest(cfg),
                "settings" => Dict(k => v for (k, v) in s if k != "quick"), "instance_file" => raw["instance"],
                "julia_version" => string(VERSION), "blackboxoptim_version" => string(pkgversion(BlackBoxOptim)),
                "source_sha256" => Dict(f => source_sha(f) for f in ("model.jl", "methods.jl", "run.jl", "protocol.toml",
                                                                        "instances/$id.toml", "Manifest.toml")))
    path = joinpath(dir, "run_info.toml")
    if isfile(path)
        old = TOML.parsefile(path)
        old["model_sha256"] == info["model_sha256"] && old["settings"] == TOML.parse(sprint(io -> TOML.print(io, info["settings"]))) ||
            error("$path was produced with different model or settings; use another --output")
        old["source_sha256"] == info["source_sha256"] || @warn "Source files changed since $path was written"
    else
        open(io -> TOML.print(io, info; sorted=true), path, "w")
    end
end

function completed(dir)
    f = joinpath(dir, "final_results.csv"); isfile(f) || return Set{Tuple{String,Int}}()
    lines = readlines(f); h = split(lines[1], ','); im = findfirst(==("method"), h); ir = findfirst(==("replication"), h)
    return Set((String(split(l, ',')[im]), parse(Int, split(l, ',')[ir])) for l in lines[2:end])
end

function run_one(id, method, rep, cfg, s, dir)
    seeds = replication_seeds(s, rep); mode = s["quick"] ? "quick" : "full"
    result = search(method, cfg, s, rep)                                   # optimization (timed)
    t = time_ns(); trace = direct_trace(cfg, result.zhist, s["trace_paths"], seeds.trace, s["jobs"])
    trace_seconds = (time_ns() - t) / 1e9
    t = time_ns(); fin = endpoint(cfg, result.final_z, s, seeds); final_seconds = (time_ns() - t) / 1e9
    de = method == "differential_evolution"
    n = de ? s["de"]["paths_per_evaluation"] : s["n_paths"]
    # Original indexing: gradient methods, row t carries the estimate computed AT z^t (moving to z^(t+1); empty at T);
    # DE, row t is the incumbent after t objective evaluations (row 0 = z0).
    est(t) = de ? (t == 0 ? nothing : result.rows[t]) : (t < length(result.rows) ? result.rows[t+1] : nothing)
    field(t, f) = (r = est(t); r === nothing ? missing : getproperty(r, f))
    lr(t, f) = de ? missing : field(t, f)
    dec(t, f) = de && t > 0 ? getproperty(result.rows[t], f) : missing
    # Effort after t updates / evaluations. New = newly generated primitive paths; simulated = paths on which all 144
    # policies were simulated (DE re-simulates its fixed bank at every evaluation); reused = historical paths only
    # LR-reweighted (Green family), never re-simulated.
    reused(t) = de ? 0 : sum((n * (result.rows[k].banks_used - 1) for k in 1:t); init=0)
    rows = [(instance=id, mode=mode, method=method, replication=rep, iteration=t, z3=z[1], z4=z[2],
             objective_internal=field(t, :objective_internal), internal_policy=field(t, :hard_policy),
             g3=lr(t, :g3), g4=lr(t, :g4), ess=lr(t, :ess), lr_clips=lr(t, :lr_clips), K_eff=lr(t, :K_eff),
             banks_used=lr(t, :banks_used), candidate_z3=dec(t, :candidate_z3), candidate_z4=dec(t, :candidate_z4),
             candidate_objective=dec(t, :candidate_objective),
             cumulative_optimization_seconds=t == 0 ? 0.0 : result.rows[t].cumulative_optimization_seconds,
             cumulative_new_paths=de ? (t == 0 ? 0 : n) : n * t, cumulative_reused_paths=reused(t),
             cumulative_simulated_paths=n * t, cumulative_policy_path_evaluations=144 * n * t)
            for (t, z) in zip(0:length(result.zhist)-1, result.zhist)]
    evals = [(instance=id, mode=mode, method=method, replication=rep, iteration=t, z3=z[1], z4=z[2], objective=p.objective,
              gross=p.gross, probing_cost=cost(cfg, z), policy=p.policy, standard_error=p.se, trace_paths=s["trace_paths"],
              trace_seed=seeds.trace) for (t, z, p) in zip(0:length(trace)-1, result.zhist, trace)]
    z = result.final_z
    final = (instance=id, mode=mode, method=method, replication=rep, z3=z[1], z4=z[2], gross=fin.gross, probing_cost=cost(cfg, z),
             objective=fin.objective, standard_error=fin.se, mc_ci95_low=fin.objective - 1.959963984540054 * fin.se,
             mc_ci95_high=fin.objective + 1.959963984540054 * fin.se, selected_policy=fin.policy, policy_av=fin.av,
             policy_aq=fin.aq, policy_as=fin.as, policy_tau=fin.tau, selection_gross=fin.selection_gross,
             optimization_seconds=result.seconds, trace_seconds=trace_seconds, final_evaluation_seconds=final_seconds,
             iterations=de ? result.objective_evaluations : s["max_iters"], new_paths=rows[end].cumulative_new_paths,
             reused_paths=rows[end].cumulative_reused_paths, simulated_paths=rows[end].cumulative_simulated_paths,
             policy_path_evaluations=rows[end].cumulative_policy_path_evaluations, trace_paths=s["trace_paths"],
             selection_paths=s["selection_paths"], holdout_paths=s["holdout_paths"],
             search_seed_first=de ? seeds.de_training : seeds.search[1], de_optimizer_seed=de ? seeds.de_optimizer : missing,
             trace_seed=seeds.trace, selection_seed=seeds.selection, holdout_seed=seeds.holdout)
    write_csv(joinpath(dir, "trajectories.csv"), rows; append=true)
    write_csv(joinpath(dir, "trajectory_evaluations.csv"), evals; append=true)
    write_csv(joinpath(dir, "final_results.csv"), [final]; append=true)     # written last: marks completion
    @printf("%-6s %-24s rep %3d  z=(%.4f, %.4f)  J=%.4f ± %.4f  policy %3d  search %.1fs\n",
            id, method, rep, z[1], z[2], fin.objective, fin.se, fin.policy, result.seconds)
    flush(stdout)
end

function main(args=ARGS)
    BLAS.set_num_threads(1)                       # as in the original runs
    a = arguments(args)
    a.quick && println("QUICK MODE: reduced budgets, for testing only (not scientific results)")
    for id in a.instances
        cfg, s, raw = load_instance(id; quick=a.quick)
        dir = joinpath(a.output, id); mkpath(dir)
        check_run_info(dir, id, cfg, s, raw)
        if a.surface
            path = joinpath(dir, "objective_surface.csv")
            isfile(path) && error("$path exists; delete it explicitly to recompute")
            println("$id: objective surface, $(s["surface_grid"])^2 points x $(s["surface_paths"]) paths ...")
            t = time_ns(); write_csv(path, objective_surface(cfg, s))
            @printf("%s: surface written in %.1fs -> %s\n", id, (time_ns() - t) / 1e9, path)
            continue
        end
        done = completed(dir)
        println("$id (model $(model_digest(cfg)[1:12])): methods $(join(a.methods, ", ")); replications $(first(a.reps))-$(last(a.reps))")
        for rep in a.reps, method in a.methods
            (method, rep) in done && (println("$id $method rep $rep already complete, skipped"); continue)
            run_one(id, method, rep, cfg, s, dir)
        end
    end
    println("Results: $(a.output)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
