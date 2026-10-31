
using Random
using Plots

# ============================================================
# PARAMETERS
# ============================================================

function parse_alpha_env()
    raw = get(ENV, "ALPHA", "10.0,5.0,0.0,25.0")
    parts = strip.(split(raw, ","))

    if length(parts) != 4
        error("ALPHA must contain exactly 4 comma-separated numeric values; got $(length(parts)): \"$raw\"")
    end

    try
        return parse.(Float64, parts)
    catch
        error("ALPHA must contain exactly 4 numeric values; got: \"$raw\"")
    end
end

function alpha_filename_tag(alpha)
    parts = [replace(replace(string(x), "." => "p"), "-" => "m") for x in alpha]
    return "alpha_" * join(parts, "_")
end

const GRID_N = parse(Int, get(ENV, "GRID_N", "40"))
const GRID_REPS = parse(Int, get(ENV, "GRID_REPS", "50"))
const SEED = parse(Int, get(ENV, "SEED", "1234"))
const N = parse(Int, get(ENV, "N", "200"))
const CRN_REPS = parse(Int, get(ENV, "CRN_REPS", string(GRID_REPS)))
const CRN_SEED = parse(Int, get(ENV, "CRN_SEED", string(SEED + 100_000)))
const ALPHA = parse_alpha_env()

include("SimOpt009.jl")
const POLICIES = generate_policies()

make_full_z(z1, z4) = [z1, 1.0, 1.0, z4]

function evaluate_Q_2d(z1, z4; policies=POLICIES, reps=GRID_REPS, seed=SEED,
    alpha=ALPHA, N=N)

    z = make_full_z(z1, z4)
    return evaluate_Q(z, policies; reps=reps, seed=seed, alpha=alpha, N=N).objective
end

# ============================================================
# GRID
# ============================================================

function build_grid_values()
    z1_vals = collect(range(0.5, 1.0, length=GRID_N))
    z4_vals = collect(range(0.5, 1.0, length=GRID_N))

    return z1_vals, z4_vals
end

function compute_grid_no_crn(z1_vals, z4_vals, policies;
    reps=GRID_REPS,
    seed=SEED,
    alpha=ALPHA,
    N=N)

    Q = zeros(length(z4_vals), length(z1_vals))

    println("Building no-CRN grid...")

    for i in eachindex(z1_vals)
        for j in eachindex(z4_vals)
            point_seed = seed + 10_000 * i + j
            Q[j, i] = evaluate_Q_2d(z1_vals[i], z4_vals[j];
                policies=policies,
                reps=reps,
                seed=point_seed,
                alpha=alpha,
                N=N)
        end
        println("  no-CRN row ", i, " / ", length(z1_vals))
    end

    return Q
end

function compute_grid_crn(z1_vals, z4_vals, policies, noise_set;
    alpha=ALPHA)

    Q = zeros(length(z4_vals), length(z1_vals))

    println("Building CRN grid...")

    for i in eachindex(z1_vals)
        for j in eachindex(z4_vals)
            z = make_full_z(z1_vals[i], z4_vals[j])
            out = estimate_Q_given_noise(z, policies, noise_set; alpha=alpha)
            Q[j, i] = out.objective
        end
        println("  CRN row ", i, " / ", length(z1_vals))
    end

    return Q
end

# ============================================================
# HILL CLIMB PATH (STABLE!)
# ============================================================

function hill_climb(z1_vals, z4_vals, Q; label="")

    # start near your usual initial point
    i = findmin(abs.(z1_vals .- 0.8))[2]
    j = findmin(abs.(z4_vals .- 0.8))[2]

    path = [(z1_vals[i], z4_vals[j])]

    println("iter  z1_z4", isempty(label) ? "" : "  " * label)

    for it in 1:100

        best_i, best_j = i, j
        best_val = Q[j, i]

        for di in -1:1
            for dj in -1:1

                ni = clamp(i + di, 1, length(z1_vals))
                nj = clamp(j + dj, 1, length(z4_vals))

                if Q[nj, ni] > best_val
                    best_val = Q[nj, ni]
                    best_i, best_j = ni, nj
                end
            end
        end

        if best_i == i && best_j == j
            println("Stop at iter ", it)
            break
        end

        i, j = best_i, best_j

        println(rpad(it, 5), (z1_vals[i], z4_vals[j]))

        push!(path, (z1_vals[i], z4_vals[j]))
    end

    return path
end

function best_grid_point(z1_vals, z4_vals, Q)
    value, idx = findmax(Q)
    return (z1=z1_vals[idx[2]], z4=z4_vals[idx[1]], value=value)
end

function path_final_point(path, z1_vals, z4_vals, Q)
    z1, z4 = path[end]
    i = findfirst(==(z1), z1_vals)
    j = findfirst(==(z4), z4_vals)
    return (z1=z1, z4=z4, value=Q[j, i])
end

# ============================================================
# PLOT
# ============================================================

function plot_all(z1_vals, z4_vals, Q, path, filename; title="")

    p = contour(z1_vals, z4_vals, Q;
        fill=true,
        xlabel="z1",
        ylabel="z4 (plotted as second 2D coordinate)",
        title=title)

    xs = [z[1] for z in path]
    ys = [z[2] for z in path]

    plot!(p, xs, ys, marker=:circle, linewidth=2, label="Hill-climb path")

    scatter!(p, [1.0], [1.0], marker=:star, label="(z1=1, z4=1)")

    savefig(p, filename)

    println("Saved: ", filename)
end

function plot_difference(z1_vals, z4_vals, Qdiff, filename; title="CRN minus no-CRN objective")
    p = contour(z1_vals, z4_vals, Qdiff;
        fill=true,
        xlabel="z1",
        ylabel="z4 (plotted as second 2D coordinate)",
        title=title)

    savefig(p, filename)

    println("Saved: ", filename)
end

# ============================================================
# MAIN
# ============================================================

function main()

    alpha = ALPHA
    alpha_tag = alpha_filename_tag(alpha)
    alpha_label = string(alpha)
    saved_files = String[]

    println("Alpha used: ", alpha_label)

    z1_vals, z4_vals = build_grid_values()

    Q_no_crn = compute_grid_no_crn(z1_vals, z4_vals, POLICIES;
        reps=GRID_REPS,
        seed=SEED,
        alpha=alpha,
        N=N)

    path_no_crn = hill_climb(z1_vals, z4_vals, Q_no_crn; label="no-CRN")

    no_crn_file = "hill_climb_path_no_crn_$(alpha_tag).png"
    plot_all(z1_vals, z4_vals, Q_no_crn, path_no_crn, no_crn_file;
        title="No-CRN objective map, alpha=$(alpha_label)")
    push!(saved_files, no_crn_file)

    println("Building CRN noise set...")
    noise_set = [generate_base_noise(CRN_SEED + r; N=N, m=length(alpha)) for r in 1:CRN_REPS]

    Q_crn = compute_grid_crn(z1_vals, z4_vals, POLICIES, noise_set; alpha=alpha)

    path_crn = hill_climb(z1_vals, z4_vals, Q_crn; label="CRN")

    crn_file = "hill_climb_path_crn_$(alpha_tag).png"
    plot_all(z1_vals, z4_vals, Q_crn, path_crn, crn_file;
        title="CRN objective map, alpha=$(alpha_label)")
    push!(saved_files, crn_file)

    diff_file = "hill_climb_crn_minus_no_crn_$(alpha_tag).png"
    plot_difference(z1_vals, z4_vals, Q_crn .- Q_no_crn, diff_file;
        title="CRN minus no-CRN objective, alpha=$(alpha_label)")
    push!(saved_files, diff_file)

    best_no_crn = best_grid_point(z1_vals, z4_vals, Q_no_crn)
    best_crn = best_grid_point(z1_vals, z4_vals, Q_crn)
    final_no_crn = path_final_point(path_no_crn, z1_vals, z4_vals, Q_no_crn)
    final_crn = path_final_point(path_crn, z1_vals, z4_vals, Q_crn)

    println("\nSUMMARY")
    println("Alpha used: ", alpha_label)
    println("No-CRN best grid point: z1=", best_no_crn.z1,
        ", z4=", best_no_crn.z4,
        ", objective=", round(best_no_crn.value, digits=4))
    println("CRN best grid point: z1=", best_crn.z1,
        ", z4=", best_crn.z4,
        ", objective=", round(best_crn.value, digits=4))
    println("No-CRN hill-climb final point: z1=", final_no_crn.z1,
        ", z4=", final_no_crn.z4,
        ", objective=", round(final_no_crn.value, digits=4))
    println("CRN hill-climb final point: z1=", final_crn.z1,
        ", z4=", final_crn.z4,
        ", objective=", round(final_crn.value, digits=4))
    println("Saved files: ", join(saved_files, ", "))

    return (
        best_no_crn=best_no_crn,
        best_crn=best_crn,
        final_no_crn=final_no_crn,
        final_crn=final_crn,
        saved_files=saved_files,
    )
end

main()
