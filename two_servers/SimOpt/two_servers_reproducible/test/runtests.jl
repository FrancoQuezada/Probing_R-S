# Regression and correctness tests for the five instances. Standalone: they read only test/reference/<ID>/, produced
# from the ORIGINAL frozen implementation by test/make_reference.jl (see test/reference/<ID>/PROVENANCE.toml).
#
#   julia --project=. test/runtests.jl                              (~1 min)
#   FULL_REGRESSION=1 julia --project=. test/runtests.jl            (+ full-size replication 1, 5 instances x 6 methods, ~8 min)
#   SURFACE_REGRESSION=1 julia --project=. test/runtests.jl         (+ full 21 x 21 x 5000 surfaces vs archived, ~15 min)
using Test, LinearAlgebra
BLAS.set_num_threads(1)
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "run.jl"))            # loads model.jl and methods.jl
include(joinpath(ROOT, "plots.jl"))          # plotting helpers (no simulation)

function readcsv(path)
    lines = readlines(path); header = split(lines[1], ',')
    [Dict(zip(header, split(l, ','; keepempty=true))) for l in lines[2:end]]
end
num(x) = parse(Float64, x)
eqv(a, b) = a == b || (!isempty(a) && !isempty(b) && num(a) == num(b))
const DIGEST = Dict("I0252" => "1ecdc3dc6161f7783a79a4e9d575f1f753a6fbb1ff4fb0d9337af56d5da54fd7",
                    "L000" => "4eea8a117dad2dd949964b82aaf5879b1d8fb913af2887ed2408ba7e2886dcf1",
                    "L025" => "9bf6c5ac745c089aaf620c2383f2a664fd21ed08b669ab6b0653b66da80ad0fa",
                    "L026" => "11cf7b3d2f08fd90eab5d0c8284451d016189fb6ed74ec10bc5e6f66d8057f7d",
                    "L030" => "8398a57ab7bc035528d3bcd551bb57eba125bc0f40715cedb60e9fea2468f2a0")
policies = policy_grid()

@testset "two_servers_reproducible" begin
for id in INSTANCES
    REF = joinpath(@__DIR__, "reference", id); O = joinpath(REF, "original_small")
    cfg, s, raw = load_instance(id)
    small = merge(s, Dict("max_iters" => 4, "n_paths" => 20))
    small["de"] = merge(s["de"], Dict("max_objective_evaluations" => 20, "paths_per_evaluation" => 20))
    @testset "$id" begin
        @testset "configuration identical to the archived experiments" begin
            prov = TOML.parsefile(joinpath(REF, "PROVENANCE.toml"))
            @test model_digest(cfg) == prov["model_sha256"] == DIGEST[id]
            io = IOBuffer(); println(io, "policy,av,aq,as,tau")
            for (i, p) in enumerate(policies); println(io, join([string(i); csv_value.([p.av, p.aq, p.as, p.tau])], ',')); end
            @test length(policies) == 144 && bytes2hex(sha256(take!(io))) == prov["policy_grid_sha256"]
            @test s["Z0"] == [0.12, 0.1] && (s["max_iters"], s["n_paths"], s["jobs"]) == (100, 500, 45)
            @test (s["trace_paths"], s["selection_paths"], s["holdout_paths"]) == (1000, 10000, 50000)
            @test (s["BASE_STEP"], s["GRAD_CLIP"], s["beta_softmax"], s["filter_kappa"]) == (0.035, 15.0, 0.5, 2.0)
            @test (s["de"]["max_objective_evaluations"], s["de"]["paths_per_evaluation"], s["de"]["population_size"]) == (100, 500, 20)
            @test pkgversion(BlackBoxOptim) == v"0.6.12"
        end
        @testset "seeds" begin
            for r in readcsv(joinpath(O, "seeds.csv"))
                ss = replication_seeds(small, parse(Int, r["rep"]))
                @test ss.search == [parse(Int, r["search$k"]) for k in 1:4]
                @test (ss.trace, ss.selection, ss.holdout, ss.de_optimizer, ss.de_training) ==
                      Tuple(parse(Int, r[k]) for k in ("trace", "selection", "holdout", "de_optimizer", "de_training"))
            end
        end
        prim = primitive_bank(cfg, 20, 12345; njobs=45); bank = materialize_bank(prim, [0.4, 0.5], cfg); Y = rewards(policies, bank, cfg)
        @testset "paths, signals, rewards, LR and scores (bitwise)" begin
            P = readcsv(joinpath(O, "primitives.csv"))
            @test length(P) == 900 && all(r -> (j = prim[parse(Int, r["path"])].jobs[parse(Int, r["job"])];
                (j.arrival, j.class, j.complex, j.value, j.service_fast, j.service_std, j.u_signal) ==
                (num(r["arrival"]), parse(Int, r["class"]), r["complex"] == "1", num(r["value"]), num(r["service_fast"]),
                 num(r["service_std"]), num(r["u_signal"]))), P)
            @test all(r -> bank[parse(Int, r["path"])].jobs[parse(Int, r["job"])].signal == parse(Int, r["signal"]), readcsv(joinpath(O, "signals.csv")))
            @test all(r -> Y[parse(Int, r["policy"]), parse(Int, r["path"])] == num(r["reward"]), readcsv(joinpath(O, "rewards.csv")))
            ll, S = bank_scores(bank, [0.35, 0.45], [0.4, 0.5], cfg)
            @test all(r -> (l = parse(Int, r["path"]); (ll[l], S[1, l], S[2, l]) == (num(r["loglr"]), num(r["score3"]), num(r["score4"]))),
                      readcsv(joinpath(O, "lr_scores.csv")))
            for path in bank[1:3], j in 1:2                       # score = d/dz log LR (central finite differences)
                t0 = [0.73, 0.81]; src = [0.3, 0.4]; plus = copy(t0); minus = copy(t0); plus[j] += 1e-6; minus[j] -= 1e-6
                fd = (loglr_and_score(path, plus, src, cfg)[1] - loglr_and_score(path, minus, src, cfg)[1]) / 2e-6
                @test isapprox(fd, loglr_and_score(path, t0, src, cfg)[2][j]; atol=1e-6, rtol=1e-6)
            end
        end
        @testset "probing cost subtracted exactly once" begin
            p = direct([0.4, 0.5], prim, policies, cfg)
            @test p.objective == maximum(vec(mean(Y; dims=2))) - cost(cfg, [0.4, 0.5]) && p.objective == p.gross - cost(cfg, [0.4, 0.5])
            d = green_data([0.4, 0.5], [HistoricalBank([0.3, 0.4], bank, zeros(size(Y)))], cfg)
            @test d.hard == -collect(cfg.alpha) && combine(d.G, softmax_weights(d.values), cfg) ≈ -collect(cfg.alpha)
        end
        @testset "five gradient methods = original drivers (4 iterations x 20 paths)" begin
            ref = readcsv(joinpath(O, "searches.csv"))
            for method in GRADIENT_METHODS
                r = gradient_search(method, cfg, small, 1); rows = filter(x -> x["method"] == method, ref)
                @test [[num(x["z3"]), num(x["z4"])] for x in rows] == r.zhist
                @test all(((row, x),) -> (row.g3, row.g4, row.objective_internal, row.hard_policy, row.ess, row.lr_clips, row.K_eff) ==
                          (num(x["g3"]), num(x["g4"]), num(x["objective_internal"]), parse(Int, x["hard_policy"]), num(x["ess"]),
                           parse(Int, x["lr_clips"]), num(x["K_eff"])), zip(r.rows, rows))
            end
            hist = HistoricalBank[]; got = String[]; z = Float64.(small["Z0"])
            for k in 1:4
                b = materialize_bank(primitive_bank(cfg, 20, stream_seed(small, 1, "search", k); njobs=45), z, cfg)
                push!(hist, HistoricalBank(copy(z), b, rewards(policies, b, cfg)))
                e = filter_data(z, hist, cfg); push!(got, join(e.indices, ";")); z = projected_step(z, e.data.hard, k)[1]
            end
            @test got == [x["retained_bank_indices"] for x in readcsv(joinpath(O, "filter2_retained.csv"))]
        end
        @testset "differential evolution (20 evaluations x 20 paths)" begin
            ref = readcsv(joinpath(O, "de_small.csv")); r = de_search(cfg, small, 1)
            @test r.objective_evaluations == length(ref) && all(((row, x),) ->
                (row.candidate_z3, row.candidate_z4, row.candidate_objective, row.incumbent_z3, row.incumbent_z4, row.objective_internal) ==
                Tuple(num(x[k]) for k in ("candidate_z3", "candidate_z4", "candidate_objective", "z3", "z4", "objective_internal")), zip(r.rows, ref))
        end
        @testset "independent evaluation and objective surface" begin
            zs = [[0.0, 0.0], [0.12, 0.1], [0.4, 0.5], [1.0, 1.0], [0.4, 0.5]]
            @test all(((p, x),) -> (p.objective, p.policy, p.se) == (num(x["objective"]), parse(Int, x["policy"]), num(x["se"])),
                      zip(direct_trace(cfg, zs, 20, 123456, 45), readcsv(joinpath(O, "direct_trace.csv"))))
            ep = endpoint(cfg, [0.7, 0.65], merge(small, Dict("selection_paths" => 30, "holdout_paths" => 80)), replication_seeds(small, 1))
            x = only(readcsv(joinpath(O, "endpoint.csv")))
            @test (ep.objective, ep.gross, ep.se, ep.policy, ep.selection_gross) ==
                  (num(x["objective"]), num(x["gross"]), num(x["se"]), parse(Int, x["policy"]), num(x["selection_gross"]))
            # three archived surface points (full 5000-path bank), plus the seed and grid conventions
            surf = readcsv(joinpath(REF, "reference_surface.csv"))
            seed = Int(parse(UInt64, bytes2hex(sha256(s["surface_seed_string"]))[1:15]; base=16)) + 1
            @test all(x -> parse(Int, x["seed"]) == seed && x["n_paths"] == "5000", surf) && length(surf) == 441
            grid = collect(range(0.0, 1.0; length=21))
            @test sort(unique(num.(getindex.(surf, "z3")))) == grid
            sprim = primitive_bank(cfg, 5000, seed; njobs=45)
            for x in surf[[1, 221, 441]]
                p = surface_point([num(x["z3"]), num(x["z4"])], sprim, policies, cfg)
                @test (p.objective, p.gross, p.policy, p.se) == (num(x["objective"]), num(x["gross"]), parse(Int, x["policy"]), num(x["conditional_se"]))
            end
        end
    end
end

@testset "execution order and reproducibility" begin
    cfg, s, _ = load_instance("L026")
    small = merge(s, Dict("max_iters" => 3, "n_paths" => 20)); small["de"] = merge(s["de"], Dict("max_objective_evaluations" => 8, "paths_per_evaluation" => 20))
    alone = search("green", cfg, small, 2)
    for m in ALL_METHODS; search(m, cfg, small, 2); end           # running other methods first changes nothing
    @test isequal(search("green", cfg, small, 2).zhist, alone.zhist)
    dir = mktempdir()
    main(["--instance", "L026", "--method", "green", "--rep", "2", "--quick", "--output", dir])
    main(["--instance", "L026", "--method", "all", "--reps", "2:2", "--quick", "--output", joinpath(dir, "all")])
    a = filter(x -> x["method"] == "green", readcsv(joinpath(dir, "L026", "final_results.csv")))
    b = filter(x -> x["method"] == "green", readcsv(joinpath(dir, "all", "L026", "final_results.csv")))
    @test [a[1][k] for k in ("z3", "z4", "objective", "selected_policy")] == [b[1][k] for k in ("z3", "z4", "objective", "selected_policy")]
    before = read(joinpath(dir, "L026", "final_results.csv"))
    main(["--instance", "L026", "--method", "green", "--rep", "2", "--quick", "--output", dir])   # already complete: skipped
    @test read(joinpath(dir, "L026", "final_results.csv")) == before
end

@testset "figures from saved results (no simulation)" begin
    dir = mktempdir()
    main(["--instance", "all", "--method", "all", "--reps", "1", "--quick", "--output", dir])
    snapshot = Dict(f => read(f) for f in readdir.(joinpath.(dir, INSTANCES); join=true) |> Iterators.flatten)
    figdir = mktempdir()
    data1 = make_figures(INSTANCES; results=dir, figures=figdir, rep=1)
    data2 = make_figures(INSTANCES; results=dir, figures=mktempdir(), rep=1)
    @test isequal(data1, data2)                                    # identical plotted numbers on repetition
    @test all(f -> read(f) == snapshot[f], keys(snapshot))         # plotting never changes results
    for id in INSTANCES, f in ("trajectories", "convergence", "final_comparison", "efficiency_time", "efficiency_budget")
        @test isfile(joinpath(figdir, id, "$f.png")) && isfile(joinpath(figdir, id, "$f.pdf"))
    end
    @test isfile(joinpath(figdir, "cross_instance_paired_vs_fresh.png"))
    @test all(d -> d.preliminary, values(data1))                   # quick results are flagged preliminary
end

if get(ENV, "FULL_REGRESSION", "0") == "1"
    @testset "full-size replication 1 = archived 019/020/021/022 results (5 instances x 6 methods, bitwise)" begin
        out = mktempdir()
        main(["--instance", "all", "--method", "all", "--reps", "1", "--output", out])
        pairs = [("z3", "z3"), ("z4", "z4"), ("objective_internal", "objective_internal"), ("g3", "g3"), ("g4", "g4"),
                 ("internal_policy", "gradient_policy"), ("ess", "ess"), ("lr_clips", "lr_clips"), ("K_eff", "K_eff")]
        for id in INSTANCES
            tr = readcsv(joinpath(out, id, "trajectories.csv")); ev = readcsv(joinpath(out, id, "trajectory_evaluations.csv"))
            fi = readcsv(joinpath(out, id, "final_results.csv"))
            for method in ALL_METHODS
                old = readcsv(joinpath(@__DIR__, "reference", id, "archived_rep1", "$(method)_trajectory.csv"))
                t = filter(x -> x["method"] == method, tr); e = filter(x -> x["method"] == method, ev)
                @test length(old) == length(t) == length(e)
                @test all(eqv(n[a], o[b]) for (n, o) in zip(t, old) for (a, b) in pairs if haskey(o, b) && !isempty(o[b]))
                @test all(eqv(n["objective"], o["objective_crn"]) && eqv(n["policy"], o["crn_selected_policy"]) &&
                          eqv(n["standard_error"], o["crn_selected_policy_se"]) for (n, o) in zip(e, old))
                of = only(readcsv(joinpath(@__DIR__, "reference", id, "archived_rep1", "$(method)_final.csv")))
                nf = only(filter(x -> x["method"] == method, fi))
                @test all(eqv(nf[a], of[b]) for (a, b) in [("z3", "z3"), ("z4", "z4"), ("objective", "objective"), ("gross", "gross"),
                          ("probing_cost", "probing_cost"), ("standard_error", "evaluation_se"), ("selected_policy", "selected_policy"),
                          ("selection_seed", "selection_seed"), ("holdout_seed", "holdout_seed")])
            end
        end
    end
end

if get(ENV, "SURFACE_REGRESSION", "0") == "1"
    @testset "full objective surfaces = archived reference surfaces (bitwise)" begin
        for id in INSTANCES
            cfg, s, _ = load_instance(id)
            new = objective_surface(cfg, s); old = readcsv(joinpath(@__DIR__, "reference", id, "reference_surface.csv"))
            @test length(new) == length(old) == 441
            @test all((r.z3, r.z4, r.objective, r.gross, r.cost, r.policy, r.conditional_se) ==
                      Tuple(num(o[k]) for k in ("z3", "z4", "objective", "gross", "cost", "policy", "conditional_se")) for (r, o) in zip(new, old))
        end
    end
end
end
