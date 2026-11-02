# Two-server routing model with probing, shared by the five instances (I0252, L000, L025, L026, L030) and six methods.
#
#
# Objective: J(z) = max_i E_z[Y(x_i, xi)] - alpha'z  (maximum OUTSIDE the expectation).
using Random, Statistics, LinearAlgebra, SHA, TOML, Printf

# ----------------------------------------------------------------------------- configuration
Base.@kwdef struct ModelConfig
    p_complex::NTuple{4,Float64}
    p_hit0::NTuple{4,Float64}
    p_false0::NTuple{4,Float64}
    beta::Float64
    mean_interarrival::Float64
    fast_mean::NTuple{2,Float64}
    std_mean::NTuple{2,Float64}
    revenue_scale::Float64
    fast_routing_cost::Float64
    delay_penalty::Float64
    alpha::NTuple{2,Float64}
end

function model_config(m::AbstractDict)
    m["NCLASS"] == 4 && m["ACTIVE_CLASSES"] == [3, 4] || error("Only 4 classes with probing on classes 3 and 4 are implemented")
    cfg = ModelConfig(p_complex=Tuple(m["P_COMPLEX"]), p_hit0=Tuple(m["P_HIT0"]), p_false0=Tuple(m["P_FALSE0"]),
                      beta=m["BETA"], mean_interarrival=m["MEAN_INTERARRIVAL"], fast_mean=Tuple(m["FAST_MEAN"]),
                      std_mean=Tuple(m["STD_MEAN"]), revenue_scale=m["REVENUE_SCALE"],
                      fast_routing_cost=m["FAST_ROUTING_COST"], delay_penalty=m["DELAY_PENALTY"], alpha=Tuple(m["ALPHA"]))
    all(x -> isfinite(x) && 0 <= x <= 1, [cfg.p_complex...; cfg.p_hit0...; cfg.p_false0...]) || error("Invalid probabilities")
    all(x -> isfinite(x) && x > 0, [cfg.mean_interarrival; cfg.fast_mean...; cfg.std_mean...]) || error("Invalid time means")
    all(x -> isfinite(x) && x >= 0, [cfg.beta; cfg.revenue_scale; cfg.fast_routing_cost; cfg.delay_penalty; cfg.alpha...]) ||
        error("Invalid economic parameters")
    return cfg
end

"""SHA-256 of the model exactly as the original `model_digest` (sorted TOML); identifies the I0252 instance."""
function model_digest(cfg::ModelConfig)
    d = Dict("P_COMPLEX" => collect(cfg.p_complex), "P_HIT0" => collect(cfg.p_hit0), "P_FALSE0" => collect(cfg.p_false0),
             "BETA" => cfg.beta, "MEAN_INTERARRIVAL" => cfg.mean_interarrival, "FAST_MEAN" => collect(cfg.fast_mean),
             "STD_MEAN" => collect(cfg.std_mean), "REVENUE_SCALE" => cfg.revenue_scale, "FAST_ROUTING_COST" => cfg.fast_routing_cost,
             "DELAY_PENALTY" => cfg.delay_penalty, "ALPHA" => collect(cfg.alpha), "NCLASS" => 4, "ACTIVE_CLASSES" => [3, 4])
    io = IOBuffer()
    TOML.print(io, d; sorted=true)
    return bytes2hex(sha256(take!(io)))
end

const INSTANCES = ["I0252", "L000", "L025", "L026", "L030"]
const PROJECT = @__DIR__

"""Load instances/<id>.toml and the shared protocol.toml. `quick=true` overrides only sample sizes and budgets,
never the model or the methods. Returns (cfg, settings, raw instance + protocol tables)."""
function load_instance(id::AbstractString; quick::Bool=false, dir=PROJECT)
    inst = TOML.parsefile(joinpath(dir, "instances", "$id.toml"))
    inst["id"] == id || error("instances/$id.toml declares id $(inst["id"])")
    proto = TOML.parsefile(joinpath(dir, "protocol.toml"))
    cfg = model_config(inst["model"])
    s = merge(proto["search"], proto["evaluation"], proto["seeds"], inst["seeds"],
              Dict("de" => copy(proto["differential_evolution"]), "instance" => id,
                   "surface_grid" => proto["surface"]["grid"], "surface_paths" => proto["surface"]["paths"],
                   "surface_seed_string" => inst["surface"]["seed_string"]))
    if quick
        q = proto["quick"]
        for k in ("max_iters", "n_paths", "trace_paths", "selection_paths", "holdout_paths")
            s[k] = q[k]
        end
        s["de"]["max_objective_evaluations"] = q["de_max_objective_evaluations"]
        s["de"]["paths_per_evaluation"] = q["de_paths_per_evaluation"]
        s["surface_grid"] = q["surface_grid"]; s["surface_paths"] = q["surface_paths"]
    end
    s["quick"] = quick
    return cfg, s, Dict("instance" => inst, "protocol" => proto)
end

# ----------------------------------------------------------------------------- seeds
"""Original Probing010-v1 stream seed: first 15 hex digits of SHA-256 + 1 (prefix carries the instance id)."""
stream_seed(s, rep::Integer, stream::AbstractString, k::Integer=0) =
    Int(parse(UInt64, bytes2hex(sha256("$(s["seed_prefix"])/$(s["rep_offset"] + rep)/$stream/$k"))[1:15]; base=16)) + 1
"""Original DE022-v1 optimizer seed (statistical cohort)."""
de_seed(s, rep::Integer) = parse(Int, bytes2hex(sha256("$(s["de_seed_prefix"])/$rep/optimizer"))[1:15]; base=16) + 1

replication_seeds(s, rep) = (search=[stream_seed(s, rep, "search", k) for k in 1:s["max_iters"]],
                             trace=stream_seed(s, rep, "trace"), selection=stream_seed(s, rep, "selection"),
                             holdout=stream_seed(s, rep, "holdout"), de_optimizer=de_seed(s, rep),
                             de_training=stream_seed(s, rep, "search", 1))

# ----------------------------------------------------------------------------- sample paths
struct PrimitiveJob            # common random numbers of one job; the signal is materialized later from u_signal
    arrival::Float64
    class::Int
    complex::Bool
    value::Float64
    service_fast::Float64
    service_std::Float64
    u_signal::Float64
end
struct PrimitivePath
    jobs::Vector{PrimitiveJob}
end
struct Job
    arrival::Float64
    class::Int
    complex::Bool
    value::Float64
    service_fast::Float64
    service_std::Float64
    signal::Int
end
struct SamplePath
    jobs::Vector{Job}
end

randexp_mean(rng, mean::Float64) = -mean * log(rand(rng))

"""Probing levels of the four classes: z = (z3, z4) for the active classes 3 and 4, zero elsewhere."""
function full_z(z2::AbstractVector{<:Real})
    zz = zeros(Float64, 4)
    zz[3] = z2[1]
    zz[4] = z2[2]
    return zz
end
p_hit(c, z, cfg::ModelConfig) = cfg.p_hit0[c] + (1.0 - cfg.p_hit0[c]) * (1.0 - exp(-cfg.beta * Float64(z)))
p_false(c, z, cfg::ModelConfig) = cfg.p_false0[c] * exp(-cfg.beta * Float64(z))
signal_probability(c, complex, z, cfg) = complex ? p_hit(c, z, cfg) : p_false(c, z, cfg)
"""Probing cost alpha'z, subtracted exactly once from the gross reward."""
cost(cfg, z) = dot(collect(cfg.alpha), z)

"""One path of `njobs` jobs; the RNG draw order is the original one."""
function primitive_path(rng, cfg::ModelConfig; njobs=45)
    jobs = Vector{PrimitiveJob}(undef, njobs); t = 0.0
    for r in 1:njobs
        t += randexp_mean(rng, cfg.mean_interarrival)
        c = rand(rng, 1:4); d = rand(rng) < cfg.p_complex[c]
        basev = 0.15 + 0.16 * c; v = clamp(basev + 0.35 * (rand(rng) - 0.5), 0.05, 1.0)
        muf = d ? cfg.fast_mean[2] : cfg.fast_mean[1]
        mus = d ? cfg.std_mean[2] : cfg.std_mean[1]
        sf = randexp_mean(rng, muf); ss = randexp_mean(rng, mus); u = rand(rng)
        jobs[r] = PrimitiveJob(t, c, d, v, sf, ss, u)
    end
    return PrimitivePath(jobs)
end
primitive_bank(cfg, n, seed; njobs=45) = (rng = MersenneTwister(seed); [primitive_path(rng, cfg; njobs) for _ in 1:n])

"""Signals under probing z from the stored uniforms (CRN across z)."""
function materialize(path, z, cfg)
    zz = full_z(z); jobs = Vector{Job}(undef, length(path.jobs))
    for (r, j) in enumerate(path.jobs)
        s = j.u_signal < signal_probability(j.class, j.complex, zz[j.class], cfg) ? 1 : 0
        jobs[r] = Job(j.arrival, j.class, j.complex, j.value, j.service_fast, j.service_std, s)
    end
    return SamplePath(jobs)
end
materialize_bank(bank, z, cfg) = [materialize(p, z, cfg) for p in bank]

# ----------------------------------------------------------------------------- routing policies and rewards
struct Policy
    av::Float64
    aq::Float64
    as::Float64
    tau::Float64
end
"""The original 144 policies: route FAST iff av*value - aq*(queue_fast - queue_std) + as*signal >= tau."""
function policy_grid()
    policies = Policy[]
    for av in (0.4, 0.8, 1.2), aq in (0.15, 0.35, 0.65), as in (0.0, 0.6, 1.2, 2.0), tau in (0.10, 0.45, 0.80, 1.15)
        push!(policies, Policy(av, aq, as, tau))
    end
    return policies
end

"""Gross reward of one policy on one path: revenue - FAST routing cost - delay penalty x time in system."""
function evaluate_policy(policy, path, cfg::ModelConfig)
    free_fast = 0.0; free_std = 0.0; total = 0.0
    for job in path.jobs
        t = job.arrival; qf = max(free_fast - t, 0.0); qs = max(free_std - t, 0.0)
        score = policy.av * job.value - policy.aq * (qf - qs) + policy.as * job.signal
        if score >= policy.tau
            start = max(t, free_fast); finish = start + job.service_fast; free_fast = finish
            time_in_system = finish - t
            total += cfg.revenue_scale * job.value
            total -= cfg.fast_routing_cost
            total -= cfg.delay_penalty * time_in_system
        else
            start = max(t, free_std); finish = start + job.service_std; free_std = finish
            time_in_system = finish - t
            total += cfg.revenue_scale * job.value
            total -= cfg.delay_penalty * time_in_system
        end
    end
    return total
end
function rewards(policies, bank, cfg)
    Y = Matrix{Float64}(undef, length(policies), length(bank))
    for i in eachindex(policies), l in eachindex(bank)
        Y[i, l] = evaluate_policy(policies[i], bank[l], cfg)
    end
    return Y
end

# ----------------------------------------------------------------------------- likelihood ratio and score function
"""Log signal probabilities and their z-derivatives (scores) for classes 3, 4 x complex/simple x signal 1/0."""
function distribution_terms(z, cfg)
    zz = full_z(z); logs = zeros(8); scores = zeros(8)
    for pos in 1:2, complex in (true, false)
        c = pos + 2
        p = clamp(signal_probability(c, complex, zz[c], cfg), 1e-12, 1.0 - 1e-12)
        dp = complex ? (1.0 - cfg.p_hit0[c]) * cfg.beta * exp(-cfg.beta * zz[c]) :
                       -cfg.p_false0[c] * cfg.beta * exp(-cfg.beta * zz[c])
        idx = (pos - 1) * 4 + (complex ? 0 : 2)
        logs[idx+1] = log(p); logs[idx+2] = log1p(-p)
        scores[idx+1] = dp / p; scores[idx+2] = -dp / (1.0 - p)
    end
    return logs, scores
end
function lr_kernel(z, zsource, cfg)
    lt, s = distribution_terms(z, cfg); ls, _ = distribution_terms(zsource, cfg)
    return lt - ls, s
end
"""log L(z | zsource) and the score d/dz log p_z of one path (only classes 3 and 4 depend on z)."""
function lr_with_kernel(path, kernel)
    logs, sc = kernel; ll = 0.0; score = zeros(2)
    for job in path.jobs
        pos = job.class == 3 ? 1 : job.class == 4 ? 2 : 0
        pos == 0 && continue
        idx = (pos - 1) * 4 + (job.complex ? 0 : 2) + (job.signal == 1 ? 1 : 2)
        ll += logs[idx]; score[pos] += sc[idx]
    end
    return ll, score
end
loglr_and_score(path, z, zsource, cfg) = lr_with_kernel(path, lr_kernel(z, zsource, cfg))
function bank_scores(bank, z, zsource, cfg)
    kernel = lr_kernel(z, zsource, cfg); n = length(bank); ll = zeros(n); S = zeros(2, n)
    for l in 1:n
        ll[l], s = lr_with_kernel(bank[l], kernel); S[:, l] .= s
    end
    return ll, S
end

# ----------------------------------------------------------------------------- estimators
"""Fresh (local) estimator on a bank generated at z: policy means, per-policy gross gradients G, hard-max gradient."""
function local_data(z, bank, Y, cfg)
    n = length(bank); ll, S = bank_scores(bank, z, z, cfg)
    all(==(0.0), ll) || error("Fresh LR differs from one")
    values = vec(mean(Y; dims=2)); best = argmax(values); G = (Y * transpose(S)) / n
    hard = [mean(view(Y, best, :) .* view(S, j, :)) - cfg.alpha[j] for j in 1:2]
    return (values=values, G=G, hard=hard, best=best, ess=Float64(n), banks=1, lr_clips=0)
end

struct HistoricalBank
    zgen::Vector{Float64}          # z at which the signals of this bank were generated
    bank::Vector{SamplePath}
    Y::Matrix{Float64}             # 144 x n policy rewards
end

"""Green: equal-weight average over all historical banks, each reweighted by L = exp(clamp(log LR, -35, 35))."""
function green_data(z, history, cfg)
    T = length(history); m = size(history[1].Y, 1); values = zeros(m); G = zeros(m, 2)
    Ls = Vector{Vector{Float64}}(); Ss = Matrix{Float64}[]; sumL = 0.0; sumL2 = 0.0; clips = 0
    for hb in history
        n = length(hb.bank); ll, S = bank_scores(hb.bank, z, hb.zgen, cfg)
        clips += count(x -> abs(x) > 35, ll); L = exp.(clamp.(ll, -35.0, 35.0))
        push!(Ls, L); push!(Ss, S)
        for i in 1:m; values[i] += mean(view(hb.Y, i, :) .* L) / T; end
        G .+= (hb.Y * transpose(S .* transpose(L))) / (n * T)
        sumL += sum(L); sumL2 += sum(abs2, L)
    end
    best = argmax(values); hard = -collect(cfg.alpha)
    for (t, hb) in enumerate(history), j in 1:2
        hard[j] += mean(view(hb.Y, best, :) .* Ls[t] .* view(Ss[t], j, :)) / T
    end
    return (values=values, G=G, hard=hard, best=best, ess=sumL^2 / sumL2, banks=T, lr_clips=clips)
end

"""Filter2 (experiment 016): provisional policy = full-Green argmax; keep banks whose LR-weighted contribution
variance for that policy is <= kappa x the current bank's variance (current bank always kept); equal weights."""
function filter_data(z, history, cfg; kappa=2.0)
    T = length(history)
    full = green_data(z, history, cfg); provisional = full.best; vf = fill(NaN, T)
    for h in 1:T
        hb = history[h]
        ll, _ = bank_scores(hb.bank, z, hb.zgen, cfg)
        L = exp.(clamp.(ll, -35.0, 35.0))
        length(hb.bank) >= 2 || error("Sample variance requires >=2 paths")
        vf[h] = var(collect(view(hb.Y, provisional, :)) .* L; corrected=true)
        isfinite(vf[h]) && vf[h] >= 0 || error("Invalid empirical objective variance")
    end
    threshold = kappa * vf[T]
    indices = findall(v -> v <= threshold, vf)
    T in indices || push!(indices, T); sort!(indices)
    return (data=green_data(z, history[indices], cfg), indices=indices, provisional=provisional, threshold=threshold)
end

"""Stable gap-based softmax weights (experiment 003); the optimization surrogate only, never the reported objective."""
function softmax_weights(values::AbstractVector{<:Real}; beta::Real=0.5)
    isfinite(beta) && beta > 0 || throw(ArgumentError("Softmax beta must be finite and positive"))
    all(isfinite, values) || throw(ArgumentError("Policy values must be finite"))
    weights = exp.(beta .* (values .- maximum(values)))
    weights ./= sum(weights)
    isapprox(sum(weights), 1.0; atol=1e-12, rtol=1e-12) || error("Softmax weights do not sum to one")
    return weights
end
"""Convex combination of per-policy GROSS gradients; probing cost subtracted once."""
combine(G, w, cfg) = transpose(G) * w - collect(cfg.alpha)

"""Projected SGA step: clip each gradient coordinate to [-clip, clip], step base/sqrt(k), project onto [0, 1]^2."""
function projected_step(z, g, k; base_step=0.035, clip=15.0)
    gc = clamp.(g, -clip, clip)
    eta = base_step / sqrt(k)
    znew = clamp.(z .+ eta .* gc, 0.0, 1.0)
    return znew, eta, gc
end

# ----------------------------------------------------------------------------- independent evaluation (no LR)
"""Direct hard-max objective of z on a fixed primitive bank (CRN across z): max_i mean_l Y_il - alpha'z."""
function direct(z, primitives, policies, cfg)
    Y = rewards(policies, materialize_bank(primitives, z, cfg), cfg)
    values = vec(mean(Y; dims=2)); i = argmax(values)
    return (objective=values[i] - cost(cfg, z), gross=values[i], policy=i, se=std(view(Y, i, :)) / sqrt(length(primitives)))
end

"""Convergence trace: every z of a trajectory on ONE primitive bank of `n` paths (repeated z evaluated once)."""
function direct_trace(cfg, zs, n, seed, jobs)
    bank = primitive_bank(cfg, n, seed; njobs=jobs); policies = policy_grid()
    cache = Dict{Tuple{Float64,Float64},Any}()
    return [get!(() -> direct(z, bank, policies, cfg), cache, Tuple(z)) for z in zs]
end

"""Final evaluation: select the policy on an independent selection bank, then score it alone on a holdout bank."""
function endpoint(cfg, z, s, seeds)
    policies = policy_grid()
    selector = primitive_bank(cfg, s["selection_paths"], seeds.selection; njobs=s["jobs"])
    Y = rewards(policies, materialize_bank(selector, z, cfg), cfg)
    chosen = argmax(vec(mean(Y; dims=2)))
    holdout = primitive_bank(cfg, s["holdout_paths"], seeds.holdout; njobs=s["jobs"])
    raw = [evaluate_policy(policies[chosen], p, cfg) for p in materialize_bank(holdout, z, cfg)]
    p = policies[chosen]
    return (objective=mean(raw) - cost(cfg, z), gross=mean(raw), se=std(raw) / sqrt(length(raw)), policy=chosen,
            av=p.av, aq=p.aq, as=p.as, tau=p.tau, selection_gross=maximum(vec(mean(Y; dims=2))))
end

"""Direct-CRN objective surface (archived 010/011 protocol): one primitive bank of `paths` paths shared by every grid
point (seed = SHA-256 of `surface_seed_string`, first 15 hex + 1); at each z the signals are re-materialized, all 144
policies simulated, the best sample mean taken and alpha'z subtracted. No likelihood ratios."""
function objective_surface(cfg, s)
    seed = Int(parse(UInt64, bytes2hex(sha256(s["surface_seed_string"]))[1:15]; base=16)) + 1
    prim = primitive_bank(cfg, s["surface_paths"], seed; njobs=s["jobs"]); policies = policy_grid()
    grid = collect(range(0.0, 1.0; length=s["surface_grid"]))
    rows = NamedTuple[]
    for x in grid, y in grid
        p = surface_point([x, y], prim, policies, cfg)
        push!(rows, (instance=s["instance"], z3=x, z4=y, objective=p.objective, gross=p.gross, cost=cost(cfg, [x, y]),
                     policy=p.policy, conditional_se=p.se, n_paths=s["surface_paths"], seed=seed))
    end
    return rows
end
"""One surface point with the expressions of the original 010/011 `point` (the SE uses a collected copy of the reward
row; `direct` uses a view, as the original trace did; the two can differ in the last bit)."""
function surface_point(z, prim, policies, cfg)
    Y = rewards(policies, materialize_bank(prim, z, cfg), cfg)
    values = vec(mean(Y; dims=2)); best = argmax(values); v = collect(view(Y, best, :))
    return (objective=values[best] - cost(cfg, z), gross=values[best], policy=best, se=std(v) / sqrt(length(v)))
end

# ----------------------------------------------------------------------------- CSV output (original %.17g convention)
csv_value(x::Missing) = ""
csv_value(x::AbstractFloat) = @sprintf("%.17g", x)
csv_value(x) = occursin(r"[,\"\n]", string(x)) ? "\"" * replace(string(x), "\"" => "\"\"") * "\"" : string(x)
function write_csv(path, rows; append::Bool=false)
    isempty(rows) && return
    ks = propertynames(first(rows))
    header = !(append && isfile(path))
    open(path, append ? "a" : "w") do io
        header && println(io, join(string.(ks), ','))
        for r in rows
            println(io, join((csv_value(getproperty(r, k)) for k in ks), ','))
        end
    end
end
