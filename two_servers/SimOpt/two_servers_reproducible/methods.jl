# The six optimization methods (unchanged from I0252_reproducible). All share model.jl and, within an instance, the same seeds per replication.
#
#   fresh                   hard-max LR gradient on the new 500-path bank only            (010/012 "Fresh")
#   green                   hard-max LR gradient on ALL historical banks (LR-reweighted)   (010/012 "Green")
#   green_softmax           softmax(beta=0.5) mixture of per-policy Green gradients         (010/012 "GreenSoftmax050")
#   green_variance          Green on the banks retained by Filter2 (kappa = 2)              (016 "Filter2")
#   green_variance_softmax  Filter2 banks, then softmax(beta=0.5) mixture                   (021)
#   differential_evolution  BlackBoxOptim adaptive_de_rand_1_bin_radiuslimited, direct
#                           objective on 500 fixed CRN paths, 100 objective evaluations     (022)
#
# Every gradient method draws, at iteration k, the primitive bank with seed stream_seed(rep, "search", k),
# materializes it at the current z and updates z <- proj(z + 0.035/sqrt(k) * clip(g, 15)).
using BlackBoxOptim

const GRADIENT_METHODS = ["fresh", "green", "green_softmax", "green_variance", "green_variance_softmax"]
const ALL_METHODS = [GRADIENT_METHODS; "differential_evolution"]

"""Gradient (and the estimator data used for diagnostics) of one method at iteration k."""
function method_gradient(method, z, bank, Y, history, cfg, s)
    beta = s["beta_softmax"]
    if method == "fresh"
        data = local_data(z, bank, Y, cfg); return data.hard, data, onehot(data.best), 1
    end
    push!(history, HistoricalBank(copy(z), bank, Y))
    if method == "green"
        data = green_data(z, history, cfg); return data.hard, data, onehot(data.best), length(history)
    elseif method == "green_softmax"
        data = green_data(z, history, cfg); w = softmax_weights(data.values; beta)
        return combine(data.G, w, cfg), data, w, length(history)
    elseif method == "green_variance"
        est = filter_data(z, history, cfg; kappa=s["filter_kappa"])
        return est.data.hard, est.data, nothing, length(est.indices)
    elseif method == "green_variance_softmax"
        est = filter_data(z, history, cfg; kappa=s["filter_kappa"]); w = softmax_weights(est.data.values; beta)
        return combine(est.data.G, w, cfg), est.data, w, length(est.indices)
    end
    error("Unknown gradient method $method")
end
onehot(i) = (w = zeros(144); w[i] = 1.0; w)

"""Projected stochastic gradient ascent from z0 for `max_iters` iterations (no early stopping)."""
function gradient_search(method, cfg, s, rep)
    policies = policy_grid(); n = s["n_paths"]; history = HistoricalBank[]
    zhist = [Float64.(s["Z0"])]; rows = NamedTuple[]; started = time_ns()
    for k in 1:s["max_iters"]
        z = zhist[end]
        primitives = primitive_bank(cfg, n, stream_seed(s, rep, "search", k); njobs=s["jobs"])
        bank = materialize_bank(primitives, z, cfg); Y = rewards(policies, bank, cfg)
        g, data, w, banks = method_gradient(method, z, bank, Y, history, cfg, s)
        all(isfinite, g) || error("Nonfinite gradient")
        znew, eta, gc = projected_step(z, g, k; base_step=s["BASE_STEP"], clip=s["GRAD_CLIP"])
        push!(rows, (iteration=k, g3=g[1], g4=g[2], step_norm=norm(znew - z), gradient_norm=norm(g),
                     clipping=count(x -> abs(x) > s["GRAD_CLIP"], g), projection=norm(znew - (z + eta * gc)),
                     ess=data.ess, lr_clips=data.lr_clips, hard_policy=data.best,
                     K_eff=w === nothing ? 1.0 : 1 / sum(abs2, w),
                     objective_internal=maximum(data.values) - cost(cfg, z), banks_used=banks,
                     cumulative_optimization_seconds=(time_ns() - started) / 1e9))
        push!(zhist, znew)
    end
    return (zhist=zhist, rows=rows, final_z=zhist[end], seconds=(time_ns() - started) / 1e9,
            new_paths=n * s["max_iters"], policy_path_evaluations=144 * n * s["max_iters"])
end

"""Differential Evolution (experiment 022): BlackBoxOptim 0.6.12, adaptive_de_rand_1_bin_radiuslimited,
population 20, bounds [0,1]^2. The objective is the direct hard-max estimate on ONE fixed bank of
`paths_per_evaluation` primitive paths (training seed = the gradient methods' first search seed); no LR.
The start z0 is evaluated first and counts towards the budget; the run stops after max_objective_evaluations - 1
or max_objective_evaluations calls. The returned z is the best evaluated point (incumbent)."""
function de_search(cfg, s, rep)
    d = s["de"]; cap = d["max_objective_evaluations"]; n = d["paths_per_evaluation"]
    z0 = Float64.(s["Z0"]); seeds = replication_seeds(s, rep)
    # BlackBoxOptim draws from the task-local default RNG: a fresh seeded task isolates it.
    task = Task() do
        Random.seed!(seeds.de_optimizer)
        started = time_ns(); policies = policy_grid()
        primitives = primitive_bank(cfg, n, seeds.de_training; njobs=s["jobs"])
        rows = NamedTuple[]; best = -Inf; best_z = copy(z0); best_policy = 0
        function objective(z)
            count = length(rows) + 1
            @assert count <= cap "Hard simulation budget guard"
            p = direct(z, primitives, policies, cfg)
            if p.objective > best
                best = p.objective; best_z = copy(z); best_policy = p.policy
            end
            push!(rows, (iteration=count, candidate_z3=z[1], candidate_z4=z[2], candidate_objective=p.objective,
                         candidate_policy=p.policy, objective_internal=best, hard_policy=best_policy,
                         incumbent_z3=best_z[1], incumbent_z4=best_z[2],
                         cumulative_optimization_seconds=(time_ns() - started) / 1e9))
            return -p.objective
        end
        initial_fitness = objective(z0)
        problem = BlackBoxOptim.FunctionBasedProblem(objective, "022 original routing simulator",
            BlackBoxOptim.MinimizingFitnessScheme, BlackBoxOptim.RectSearchSpace(2, (0.0, 1.0)), nothing)
        callback = ctrl -> (length(rows) >= cap - 1 && BlackBoxOptim.shutdown!(ctrl); nothing)
        ctrl = bbsetup(problem; Method=Symbol(d["method"]), PopulationSize=d["population_size"],
                       MaxFuncEvals=cap, TraceMode=:silent, CallbackFunction=callback, CallbackInterval=0.0)
        @assert length(rows) == 1 "Unexpected unaccounted library setup evaluation"
        result = bboptimize(ctrl, copy(z0))
        elapsed = (time_ns() - started) / 1e9
        @assert cap - 1 <= length(rows) <= cap
        @assert BlackBoxOptim.f_calls(result) + 1 == length(rows)
        @assert isapprox(best, -min(initial_fitness, best_fitness(result)); atol=1e-10, rtol=1e-12)
        (zhist=vcat([copy(z0)], [[r.incumbent_z3, r.incumbent_z4] for r in rows]), rows=rows, final_z=best_z,
         seconds=elapsed, objective_evaluations=length(rows), new_paths=n, path_evaluations=n * length(rows),
         policy_path_evaluations=144 * n * length(rows), stop_reason=BlackBoxOptim.stop_reason(result))
    end
    schedule(task)
    return fetch(task)
end

search(method, cfg, s, rep) = method == "differential_evolution" ? de_search(cfg, s, rep) : gradient_search(method, cfg, s, rep)
