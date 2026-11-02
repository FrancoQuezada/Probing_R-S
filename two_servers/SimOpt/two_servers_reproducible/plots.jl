# Figures from SAVED results only: never runs an optimizer or a simulation.
#
#   julia --project=. plots.jl --instance I0252
#   julia --project=. plots.jl --instance all [--rep 1] [--quick] [--results DIR] [--figures DIR]
#
# Reads results/<mode>/<ID>/{final_results,trajectories,trajectory_evaluations}.csv (+ objective_surface.csv if present)
# and writes figures/<mode>/<ID>/*.png|pdf and figures/<mode>/cross_instance_paired_vs_fresh.png|pdf.
# Objectives are always the NET objective J = max_i mean Y_i - alpha'z from the independent direct-CRN evaluations
# (trace: every visited z on one fixed bank per replication; final: independent selection + holdout).
using Plots, Statistics, Printf, TOML, Distributions
gr()
default(fontfamily="sans-serif", titlefontsize=10, guidefontsize=9, tickfontsize=8, legendfontsize=7, framestyle=:box,
        grid=true, gridalpha=0.25, dpi=200)

const PLOT_INSTANCES = ["I0252", "L000", "L025", "L026", "L030"]
const PLOT_METHODS = ["fresh", "green", "green_softmax", "green_variance", "green_variance_softmax", "differential_evolution"]
const LABEL = Dict("fresh" => "Fresh", "green" => "Green", "green_softmax" => "Green Softmax", "green_variance" => "Green Variance",
                   "green_variance_softmax" => "Green Variance + Softmax", "differential_evolution" => "Differential Evolution")
const COLOR = Dict("fresh" => colorant"#0072B2", "green" => colorant"#009E73", "green_softmax" => colorant"#D55E00",
                   "green_variance" => colorant"#CC79A7", "green_variance_softmax" => colorant"#E69F00",
                   "differential_evolution" => colorant"#000000")
const SHORT = Dict("fresh" => "Fresh", "green" => "Green", "green_softmax" => "G-Soft", "green_variance" => "G-Var",
                   "green_variance_softmax" => "G-Var+Soft", "differential_evolution" => "DE")
const STYLE = Dict(m => (m == "differential_evolution" ? :dash : :solid) for m in PLOT_METHODS)
const MARK = Dict("fresh" => :circle, "green" => :rect, "green_softmax" => :utriangle, "green_variance" => :diamond,
                  "green_variance_softmax" => :dtriangle, "differential_evolution" => :star5)

function load_csv(path)
    isfile(path) || return Dict{String,String}[]
    lines = readlines(path); header = split(lines[1], ',')
    [Dict(String(h) => String(v) for (h, v) in zip(header, split(l, ','; keepempty=true))) for l in lines[2:end]]
end
fnum(x) = isempty(x) ? NaN : parse(Float64, x)
col(rows, k) = [fnum(r[k]) for r in rows]

"""Mean, sd and Student-t 95% CI half-width across independent replications (statistical unit = replication)."""
function rep_stats(x)
    x = filter(isfinite, x); n = length(x)
    n == 0 && return (n=0, mean=NaN, sd=NaN, half=NaN)
    n == 1 && return (n=1, mean=x[1], sd=NaN, half=NaN)
    sd = std(x); return (n=n, mean=mean(x), sd=sd, half=quantile(TDist(n - 1), 0.975) * sd / sqrt(n))
end

"""Saved results of one instance; rows of incomplete (method, replication) pairs are ignored, duplicates keep the last."""
function load_results(dir)
    final = load_csv(joinpath(dir, "final_results.csv"))
    done = Set((r["method"], r["replication"]) for r in final)
    dedup(rows) = collect(values(Dict((r["method"], r["replication"], r["iteration"]) => r for r in rows if (r["method"], r["replication"]) in done)))
    sortit(rows) = sort(rows; by=r -> (r["method"], parse(Int, r["replication"]), parse(Int, r["iteration"])))
    info = isfile(joinpath(dir, "run_info.toml")) ? TOML.parsefile(joinpath(dir, "run_info.toml")) : Dict("mode" => "unknown")
    return (final=final, traj=sortit(dedup(load_csv(joinpath(dir, "trajectories.csv")))),
            evals=sortit(dedup(load_csv(joinpath(dir, "trajectory_evaluations.csv")))),
            surface=load_csv(joinpath(dir, "objective_surface.csv")), preliminary=info["mode"] != "full", info=info)
end
select(rows; kw...) = filter(r -> all(r[String(k)] == string(v) for (k, v) in kw), rows)
methods_in(R) = filter(m -> any(r -> r["method"] == m, R.final), PLOT_METHODS)
reps_of(R, m) = sort(unique(parse.(Int, getindex.(select(R.final; method=m), "replication"))))
tag(R) = R.preliminary ? "  [QUICK MODE: preliminary]" : ""
function save(p, dir, name)
    mkpath(dir); savefig(p, joinpath(dir, name * ".png")); savefig(p, joinpath(dir, name * ".pdf"))
end

"""Per iteration: mean and t-CI over replications of the trace objective, mean cumulative effort columns."""
function curves(R, m)
    ev = select(R.evals; method=m); tr = select(R.traj; method=m)
    its = sort(unique(parse.(Int, getindex.(ev, "iteration"))))
    out = NamedTuple[]
    for t in its
        e = select(ev; iteration=t); x = select(tr; iteration=t); st = rep_stats(col(e, "objective"))
        push!(out, (iteration=t, mean=st.mean, half=st.half, n=st.n, seconds=mean(col(x, "cumulative_optimization_seconds")),
                    simulated=mean(col(x, "cumulative_simulated_paths"))))
    end
    return out
end

function trajectory_panel!(p, R, rep)
    plotted = Dict{String,Any}()
    for m in methods_in(R)
        tr = select(R.traj; method=m, replication=rep); isempty(tr) && continue
        z3 = col(tr, "z3"); z4 = col(tr, "z4"); plotted[m] = (z3, z4)
        lab = m == "differential_evolution" ? "Differential Evolution (best incumbent)" : LABEL[m]
        plot!(p, z3, z4; color=COLOR[m], ls=STYLE[m], lw=1.6, alpha=0.9, label=lab)
        scatter!(p, [z3[end]], [z4[end]]; color=COLOR[m], marker=MARK[m], ms=7, msc=:white, label="")
        if !haskey(plotted, "start")
            scatter!(p, [z3[1]], [z4[1]]; color=:gold, marker=:star8, ms=10, msc=:black, label="common start z0"); plotted["start"] = (z3[1], z4[1])
        end
    end
    return plotted
end

function instance_figures(id, R, figdir; rep=1)
    data = Dict{String,Any}("preliminary" => R.preliminary)
    ms = methods_in(R); isempty(ms) && return data
    nrep = maximum(length(reps_of(R, m)) for m in ms)
    # 1. trajectories
    p = plot(; xlims=(-0.02, 1.02), ylims=(-0.02, 1.02), aspect_ratio=:equal, size=(620, 560), legend=:outerbottom, legendcolumns=2,
             xlabel="probing effort z3 (class 3)", ylabel="probing effort z4 (class 4)",
             title="$id: optimization trajectories, replication $rep$(tag(R))")
    data["trajectories"] = trajectory_panel!(p, R, rep)
    save(p, figdir, "trajectories")
    # 2. convergence: gradient methods by iteration, DE by objective evaluation, all by simulated paths
    band(c) = (c.n >= 2 ? c.half : 0.0)
    pa = plot(; xlabel="iteration (gradient update)", ylabel="net objective J (direct CRN evaluation)", title="(a) gradient methods")
    pb = plot(; xlabel="objective evaluation (DE)", title="(b) Differential Evolution (best incumbent)")
    pc = plot(; xlabel="cumulative simulated paths (each: 144 policies)", xscale=:log10, title="(c) all methods vs simulation effort")
    data["convergence"] = Dict{String,Any}()
    for m in ms
        c = curves(R, m); data["convergence"][m] = c
        x = [r.iteration for r in c]; y = [r.mean for r in c]; h = [band(r) for r in c]
        target = m == "differential_evolution" ? pb : pa
        plot!(target, x, y; ribbon=h, fillalpha=0.15, color=COLOR[m], ls=STYLE[m], lw=1.5, label=LABEL[m])
        keep = [r.simulated > 0 for r in c]
        plot!(pc, [r.simulated for r in c][keep], y[keep]; color=COLOR[m], ls=STYLE[m], lw=1.5, label=LABEL[m])
    end
    sub = nrep >= 2 ? "mean over replications with pointwise 95% t-CI (n = $nrep)" : "single replication (no across-replication band)"
    p = plot(pa, pb, pc; layout=(1, 3), size=(1500, 460), legend=:bottomright, left_margin=6Plots.mm, bottom_margin=6Plots.mm,
             plot_title="$id: independently evaluated objective of the visited decisions; $sub$(tag(R))", plot_titlefontsize=10)
    save(p, figdir, "convergence")
    # 3. final comparison (independent selection + holdout)
    note = nrep >= 2 ? "replications (bars: holdout MC 95%), gray: ±1 sd across replications, black: mean and 95% t-CI" :
                       "single replication: bars are the holdout MC 95% interval; across-replication variability not estimated"
    p = plot(; size=(860, 560), ylabel="final net objective J (holdout)", xticks=(1:length(ms), [LABEL[m] for m in ms]),
             xrotation=15, legend=:outerbottom, legendcolumns=nrep >= 2 ? 3 : 1, bottom_margin=4Plots.mm, left_margin=6Plots.mm,
             title="$id: final solutions on the independent holdout$(tag(R))\n$note", titlefontsize=9)
    data["final"] = Dict{String,Any}()
    for (i, m) in enumerate(ms)
        f = select(R.final; method=m); y = col(f, "objective"); se = col(f, "standard_error"); st = rep_stats(y)
        data["final"][m] = (values=y, stats=st)
        xs = i .+ (length(y) > 1 ? range(-0.22, 0.22; length=length(y)) : [0.0])
        scatter!(p, xs, y; yerror=1.96 .* se, color=COLOR[m], marker=MARK[m], ms=3.5, alpha=0.6, msc=COLOR[m],
                 label=i == 1 ? "replication (bar: holdout MC 95% interval)" : "")
        if st.n >= 2
            scatter!(p, [i + 0.32], [st.mean]; yerror=st.sd, color=:gray60, msc=:gray60, marker=:rect, ms=3, lw=3, label=i == 1 ? "mean ± 1 sd across replications" : "")
            scatter!(p, [i + 0.38], [st.mean]; yerror=st.half, color=:black, marker=:circle, ms=4, lw=2.5, label=i == 1 ? "mean, 95% t-CI (n = $(st.n))" : "")
        end
    end
    save(p, figdir, "final_comparison")
    # 4a. objective vs measured optimization time; 4b. vs simulation budget + accounting of paths
    p = plot(; size=(820, 480), xlabel="cumulative optimization time [s] (this machine; not comparable to archived timings)",
             ylabel="net objective J (direct CRN evaluation)", legend=:bottomright,
             title="$id: objective vs optimization time ($(nrep >= 2 ? "means over $nrep replications" : "replication $(first(reps_of(R, ms[1])))"))$(tag(R))")
    for m in ms
        c = data["convergence"][m]
        plot!(p, [r.seconds for r in c], [r.mean for r in c]; color=COLOR[m], ls=STYLE[m], lw=1.5, label=LABEL[m])
    end
    save(p, figdir, "efficiency_time")
    pa = plot(; xlabel="cumulative simulated paths (each: 144 policies)", ylabel="net objective J (direct CRN evaluation)",
              xscale=:log10, legend=:bottomright, title="(a) objective vs simulation effort")
    for m in ms
        c = filter(r -> r.simulated > 0, data["convergence"][m])
        plot!(pa, [r.simulated for r in c], [r.mean for r in c]; color=COLOR[m], ls=STYLE[m], lw=1.5, label=LABEL[m])
    end
    acc = [(m, [mean(col(select(R.final; method=m), k)) for k in ("new_paths", "reused_paths", "simulated_paths")]...) for m in ms]
    data["effort"] = acc
    top = maximum(maximum(a[2:4]) for a in acc)
    pb = plot(; yscale=:log10, xticks=(1:length(ms), [LABEL[m] for m in ms]), xrotation=20, legend=:outerbottom,
              ylims=(1, 10 * top),
              ylabel="paths per run (log scale; zero not drawn)", title="(b) path accounting per run")
    for (j, (lab, c)) in enumerate(zip(["newly generated paths", "reused historical paths (LR-reweighted, not re-simulated)",
                                        "simulated path evaluations (144 policies each)"], [:steelblue, :gray70, :black]))
        v = [a[j+1] > 0 ? a[j+1] : NaN for a in acc]
        bar!(pb, (1:length(ms)) .+ (j - 2) * 0.27, v; bar_width=0.27, fillto=1, color=c, lw=0, label=lab)
    end
    p = plot(pa, pb; layout=(1, 2), size=(1400, 500), left_margin=6Plots.mm, bottom_margin=10Plots.mm,
             plot_title="$id: simulation effort$(tag(R))", plot_titlefontsize=10)
    save(p, figdir, "efficiency_budget")
    # 5. heatmap (only if the surface was computed explicitly)
    if !isempty(R.surface)
        xs = sort(unique(col(R.surface, "z3"))); ys = sort(unique(col(R.surface, "z4")))
        Z = fill(NaN, length(ys), length(xs))
        for r in R.surface; Z[findfirst(==(fnum(r["z4"])), ys), findfirst(==(fnum(r["z3"])), xs)] = fnum(r["objective"]); end
        best = argmax(col(R.surface, "objective")); b = R.surface[best]
        p = heatmap(xs, ys, Z; c=:inferno, aspect_ratio=:equal, xlims=(-0.02, 1.02), ylims=(-0.02, 1.02), size=(700, 600),
                    
                    xlabel="probing effort z3 (class 3)", ylabel="probing effort z4 (class 4)", legend=:outerbottom, legendcolumns=2,
                    title="$id: objective surface ($(length(xs))x$(length(ys)) grid, $(R.surface[1]["n_paths"]) CRN paths per point; colour = net objective J)\nand trajectories of replication $rep$(tag(R))", titlefontsize=9)
        scatter!(p, [fnum(b["z3"])], [fnum(b["z4"])]; marker=:xcross, ms=9, color=:cyan, msw=3, label="best grid point (estimate)")
        trajectory_panel!(p, R, rep)
        data["surface"] = Z
        save(p, figdir, "objective_heatmap")
    end
    return data
end

"""Cross-instance figure: (top) final net objective per method, own y-scale per instance; (bottom) paired difference
J(method) - J(Fresh) of the final holdout values within each replication (same selection/holdout seeds for all methods)."""
function cross_figure(allR, figdir)
    ids = [id for id in PLOT_INSTANCES if haskey(allR, id) && !isempty(allR[id].final)]
    isempty(ids) && return nothing
    top = Any[]; bottom = Any[]; data = Dict{Tuple{String,String},Any}()
    for id in ids
        R = allR[id]; ms = methods_in(R)
        pt = plot(; title=id, xticks=(1:length(ms), [SHORT[m] for m in ms]), xrotation=45, legend=false, ylabel=id == ids[1] ? "final net objective J" : "")
        pb = plot(; xticks=(1:length(ms), [SHORT[m] for m in ms]), xrotation=45, legend=false, ylabel=id == ids[1] ? "J(method) - J(Fresh), paired" : "")
        lo = [Inf, Inf]; hi = [-Inf, -Inf]
        hline!(pb, [0.0]; color=:gray50, ls=:dot)
        fresh = Dict(r["replication"] => fnum(r["objective"]) for r in select(R.final; method="fresh"))
        for (i, m) in enumerate(ms)
            f = select(R.final; method=m); st = rep_stats(col(f, "objective"))
            d = [fnum(r["objective"]) - fresh[r["replication"]] for r in f if haskey(fresh, r["replication"])]; sd = rep_stats(d)
            data[(id, m)] = (final=st, paired=sd)
            scatter!(pt, [i], [st.mean]; yerror=st.n >= 2 ? st.half : 0, color=COLOR[m], marker=MARK[m], ms=5, lw=2)
            scatter!(pb, i .+ zeros(length(d)), d; color=COLOR[m], alpha=0.35, ms=2.5, msw=0)
            scatter!(pb, [i], [sd.mean]; yerror=sd.n >= 2 ? sd.half : 0, color=COLOR[m], marker=MARK[m], ms=5, lw=2)
            for (k, v) in ((1, [st.mean - (st.n >= 2 ? st.half : 0), st.mean + (st.n >= 2 ? st.half : 0)]), (2, [d; sd.mean - (sd.n >= 2 ? sd.half : 0); sd.mean + (sd.n >= 2 ? sd.half : 0)]))
                v = filter(isfinite, v); isempty(v) && continue; lo[k] = min(lo[k], minimum(v)); hi[k] = max(hi[k], maximum(v))
            end
        end
        for (k, q) in ((1, pt), (2, pb))
            pad = 0.08 * max(hi[k] - lo[k], 1e-9); ylims!(q, lo[k] - pad, hi[k] + pad); xlims!(q, 0.5, length(ms) + 0.5)
        end
        push!(top, pt); push!(bottom, pb)
    end
    pre = any(allR[id].preliminary for id in ids) ? "  [QUICK MODE: preliminary]" : ""
    p = plot(top..., bottom...; layout=(2, length(ids)), size=(320 * length(ids), 820), bottom_margin=8Plots.mm, left_margin=12Plots.mm,
             plot_title="Six methods across instances: final holdout objective (top) and paired difference to Fresh (bottom); mean and 95% t-CI over replications, dots = replications$pre\nG-Soft = Green Softmax, G-Var = Green Variance, G-Var+Soft = Green Variance + Softmax, DE = Differential Evolution; own y-scale per instance (units of J)",
             plot_titlefontsize=9)
    save(p, figdir, "cross_instance_paired_vs_fresh")
    return data
end

"""Make every figure for `instances` from saved results; returns the plotted numbers (for regression tests)."""
function make_figures(instances; results, figures, rep=1)
    allR = Dict{String,Any}(); out = Dict{String,Any}()
    for id in instances
        dir = joinpath(results, id)
        isdir(dir) || (println("$id: no results in $dir, skipped"); continue)
        allR[id] = load_results(dir)
        out[id] = (; preliminary=allR[id].preliminary, data=instance_figures(id, allR[id], joinpath(figures, id); rep))
        println("$id: figures written to $(joinpath(figures, id))")
    end
    if length(allR) >= 2
        out["cross"] = (; preliminary=any(R.preliminary for R in values(allR)), data=cross_figure(allR, figures))
    end
    return out
end

function plots_main(args=ARGS)
    a = Dict("instance" => "all", "rep" => "1", "quick" => "false"); i = 1
    while i <= length(args)
        k = args[i][3:end]
        if k == "quick"; a["quick"] = "true"; i += 1; else; a[k] = args[i+1]; i += 2; end
    end
    mode = a["quick"] == "true" ? "quick" : "full"
    instances = a["instance"] == "all" ? PLOT_INSTANCES : split(a["instance"], ',')
    make_figures(String.(instances); results=get(a, "results", joinpath(@__DIR__, "results", mode)),
                 figures=get(a, "figures", joinpath(@__DIR__, "figures", mode)), rep=parse(Int, a["rep"]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    plots_main()
end
