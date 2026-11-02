# Maintainer tool (NOT needed to run or test the project): regenerates test/reference/<ID>/ from the ORIGINAL frozen
# implementation inside the Probing&RS repository. Run from two_servers_reproducible/ with the frozen 022 environment,
# one process per instance (the frozen sources bind the instance id as module constants):
#
#   for i in I0252 L000 L025 L026 L030; do
#     julia --project=../resultados/022_differential_evolution/environment test/make_reference.jl $i; done
#
# It executes the original 019/021/022 code unchanged (021 is evaluated without its own include line so that both
# compositions share one copy of the frozen 019 module), at small sizes, and also copies the archived full-size
# replication-1 results of all six methods. Nothing outside test/reference/ is written.
using Serialization, TOML, SHA, Statistics, Printf
const SIMOPT = normpath(joinpath(@__DIR__, "..", ".."))
const ID = ARGS[1]
const REPRO = joinpath(SIMOPT, "resultados", "022_differential_evolution", ID, "reproduction")
const OUT = joinpath(@__DIR__, "reference", ID)
include(joinpath(REPRO, "022_differential_evolution.jl"))                       # 022 -> frozen 019 -> 016 -> 012 -> ... -> 003
const S = Main.I0252Statistical; const V = S.V; const R = S.R; const C = S.C; const X = S.X
const DE = Main.DifferentialEvolution022
# 021 adds a more specific S.search method that intercepts every method name, as in its own single-method process;
# it is therefore loaded only after the 019 searches below have run.
load021() = include_string(Main, replace(read(joinpath(SIMOPT, "resultados", "021_green_variance_softmax", "code",
    "021_green_variance_softmax.jl"), String), r"^include\(.*\)\n"m => ""; count=1), "021_green_variance_softmax.jl")
csv(x) = x isa AbstractFloat ? @sprintf("%.17g", x) : string(x)
function table(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for r in rows; println(io, join(csv.(r), ',')); end
    end
end

mkpath(joinpath(OUT, "original_small")); mkpath(joinpath(OUT, "archived_rep1"))
c, cfg = S.model(joinpath(SIMOPT, "resultados", "022_differential_evolution", ID))
@assert c["science"]["scenario"] == ID
d = copy(c["science"])
small = merge(d, Dict("max_iters" => 4, "n_paths" => 20))
policies = C.build_policy_grid()
o = joinpath(OUT, "original_small")

# seeds for replications 1..3 (019/021 streams and the 022 DE streams)
rows = []
for rep in 1:3
    ss = S.seeds(rep, 4); de = DE.seeds("statistical", rep)
    push!(rows, (rep, ss["search"]..., ss["trace"], ss["selection"], ss["holdout"], de["optimizer"], de["training"]))
end
table(joinpath(o, "seeds.csv"), ["rep", "search1", "search2", "search3", "search4", "trace", "selection", "holdout", "de_optimizer", "de_training"], rows)

# primitives, materialized signals, rewards, LR and scores on a 20-path bank
prim = R.primitive_bank(cfg, 20, 12345; njobs=45)
table(joinpath(o, "primitives.csv"), ["path", "job", "arrival", "class", "complex", "value", "service_fast", "service_std", "u_signal"],
      [(l, r, j.arrival, j.class, Int(j.complex), j.value, j.service_fast, j.service_std, j.u_signal) for (l, p) in enumerate(prim) for (r, j) in enumerate(p.jobs)])
bank = R.materialize_bank(prim, [0.4, 0.5], cfg)
table(joinpath(o, "signals.csv"), ["path", "job", "signal"], [(l, r, j.signal) for (l, p) in enumerate(bank) for (r, j) in enumerate(p.jobs)])
Y = R.rewards(policies, bank, cfg)
table(joinpath(o, "rewards.csv"), ["policy", "path", "reward"], [(i, l, Y[i, l]) for i in axes(Y, 1) for l in axes(Y, 2)])
ll, Sc = R.bank_scores(bank, [0.35, 0.45], [0.4, 0.5], cfg)
table(joinpath(o, "lr_scores.csv"), ["path", "loglr", "score3", "score4"], [(l, ll[l], Sc[1, l], Sc[2, l]) for l in eachindex(ll)])

# short searches (4 iterations x 20 paths, replication 1) with the original drivers
rows = []
for method in ("fresh", "green", "green_softmax", "green_variance")
    raw = S.search(small, cfg, S.REP_OFFSET + 1, method)
    for (k, z) in enumerate(raw.zhist)
        g = k <= length(raw.core) ? raw.core[k] : nothing
        push!(rows, (method, k - 1, z[1], z[2], g === nothing ? "" : g.g3, g === nothing ? "" : g.g4,
                     g === nothing ? "" : g.objective_internal, g === nothing ? "" : g.hard_policy,
                     g === nothing ? "" : g.ess, g === nothing ? "" : g.lr_clips, g === nothing ? "" : g.K_eff))
    end
end
filter = V.search(small, cfg, S.REP_OFFSET + 1, only(Base.filter(x -> x.id == "Filter2", V.METHODS)); extra=true)
de = DE.search(d, cfg, DE.seeds("statistical", 1); cap=20, n=20)
load021()
raw = Base.invokelatest(Main.GreenVarianceSoftmax021.search, small, cfg, S.REP_OFFSET + 1; persist=false)
for (k, z) in enumerate(raw.zhist)
    g = k <= length(raw.core) ? raw.core[k] : nothing
    push!(rows, ("green_variance_softmax", k - 1, z[1], z[2], g === nothing ? "" : g.g3, g === nothing ? "" : g.g4,
                 g === nothing ? "" : g.objective_internal, g === nothing ? "" : g.hard_policy,
                 g === nothing ? "" : g.ess, g === nothing ? "" : g.lr_clips, g === nothing ? "" : g.K_eff))
end
table(joinpath(o, "searches.csv"), ["method", "iteration", "z3", "z4", "g3", "g4", "objective_internal", "hard_policy", "ess", "lr_clips", "K_eff"], rows)
table(joinpath(o, "filter2_retained.csv"), ["iteration", "retained_bank_indices", "filter_threshold"],
      [(r.iteration, r.retained_bank_indices, r.filter_threshold) for r in filter.rows])

# DE: 20 objective evaluations on 20 paths (replication 1), run above before 021 was loaded
table(joinpath(o, "de_small.csv"), ["evaluation", "candidate_z3", "candidate_z4", "candidate_objective", "z3", "z4", "objective_internal"],
      [(r.objective_evaluation, r.candidate_z3, r.candidate_z4, r.candidate_objective, r.z3, r.z4, r.objective_internal) for r in de.rows])

# evaluation: direct trace and final selection + holdout
zs = [[0.0, 0.0], [0.12, 0.1], [0.4, 0.5], [1.0, 1.0], [0.4, 0.5]]
tr = S.direct_trace(cfg, zs, 20, 123456, 45)
table(joinpath(o, "direct_trace.csv"), ["z3", "z4", "objective", "policy", "se"], [(z[1], z[2], p.objective, p.policy, p.se) for (z, p) in zip(zs, tr)])
ep = S.endpoint(cfg, [0.7, 0.65], merge(d, Dict("selection_paths" => 30, "holdout_paths" => 80)), S.seeds(1, 4))
table(joinpath(o, "endpoint.csv"), ["objective", "gross", "se", "policy", "selection_gross"], [(ep.objective, ep.gross, ep.se, ep.policy, ep.selection_gross)])

# archived full-size replication 1 of all six methods (original files, copied unchanged)
base = ID == "I0252" ? "019probing_i0252_four_methods_statistical" : "020probing_four_instances_statistical/$ID"
src = Dict("fresh" => "$base/raw/fresh/rep_0001", "green" => "$base/raw/green/rep_0001",
           "green_softmax" => "$base/raw/green_softmax/rep_0001", "green_variance" => "$base/raw/green_variance/rep_0001",
           "green_variance_softmax" => "021_green_variance_softmax/$ID/raw/green_variance_softmax/rep_0001",
           "differential_evolution" => "022_differential_evolution/$ID/raw/differential_evolution/rep_0001")
prov = Dict{String,Any}()
for (m, rel) in src, f in ("trajectory.csv", "final.csv")
    from = joinpath(SIMOPT, "resultados", rel, f); to = joinpath(OUT, "archived_rep1", "$(m)_$f")
    cp(from, to; force=true)
    prov["$(m)_$f"] = Dict("source" => "two_servers/SimOpt/resultados/$rel/$f", "sha256" => bytes2hex(sha256(read(to))))
end
surface = "022_differential_evolution/$ID/config/$(ID)_reference_surface.csv"
cp(joinpath(SIMOPT, "resultados", surface), joinpath(OUT, "reference_surface.csv"); force=true)
prov["reference_surface.csv"] = Dict("source" => "two_servers/SimOpt/resultados/$surface", "sha256" => bytes2hex(sha256(read(joinpath(OUT, "reference_surface.csv")))))
open(joinpath(OUT, "PROVENANCE.toml"), "w") do io
    TOML.print(io, Dict("instance" => ID, "generated_by" => "test/make_reference.jl", "julia" => string(VERSION),
                        "original_sources" => d["numerical_sources"], "model_sha256" => d["model_sha256"],
                        "policy_grid_sha256" => d["policy_grid_sha256"], "archived_rep1" => prov); sorted=true)
end
println("reference written to $OUT")
