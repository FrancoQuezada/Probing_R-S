
using Random
using Plots

# ============================================================
# PARAMETERS
# ============================================================

const GRID_N = 40
const GRID_REPS = 50
const SEED = 1234

include("SimOpt009.jl")
const POLICIES = generate_policies()

make_full_z(z1,z2) = [z1,1.0,1.0,z2]

function evaluate_Q_2d(z1,z2)
    z = make_full_z(z1,z2)
    return evaluate_Q(z, POLICIES; reps=GRID_REPS, seed=SEED).objective
end

# ============================================================
# GRID
# ============================================================

function build_grid()

    z1_vals = collect(range(0.5,1.0,length=GRID_N))
    z2_vals = collect(range(0.5,1.0,length=GRID_N))

    Q = zeros(GRID_N,GRID_N)

    println("Building grid...")

    for i in 1:GRID_N
        for j in 1:GRID_N
            Q[j,i] = evaluate_Q_2d(z1_vals[i], z2_vals[j])
        end
    end

    return z1_vals, z2_vals, Q
end

# ============================================================
# HILL CLIMB PATH (STABLE!)
# ============================================================

function hill_climb(z1_vals, z2_vals, Q)

    # start near your usual initial point
    i = findmin(abs.(z1_vals .- 0.8))[2]
    j = findmin(abs.(z2_vals .- 0.8))[2]

    path = [(z1_vals[i], z2_vals[j])]

    println("iter  z")

    for it in 1:100

        best_i, best_j = i, j
        best_val = Q[j,i]

        for di in -1:1
            for dj in -1:1

                ni = clamp(i+di, 1, length(z1_vals))
                nj = clamp(j+dj, 1, length(z2_vals))

                if Q[nj,ni] > best_val
                    best_val = Q[nj,ni]
                    best_i, best_j = ni, nj
                end
            end
        end

        if best_i == i && best_j == j
            println("Stop at iter ", it)
            break
        end

        i, j = best_i, best_j

        println(rpad(it,5), (z1_vals[i], z2_vals[j]))

        push!(path, (z1_vals[i], z2_vals[j]))
    end

    return path
end

# ============================================================
# PLOT
# ============================================================

function plot_all(z1_vals, z2_vals, Q, path)

    p = contour(z1_vals, z2_vals, Q, fill=true)

    xs = [z[1] for z in path]
    ys = [z[2] for z in path]

    plot!(p, xs, ys, marker=:circle, linewidth=2, label="Hill-climb path")

    scatter!(p, [1.0], [1.0], marker=:star, label="(1,1)")

    savefig(p, "hill_climb_path.png")

    println("Saved: hill_climb_path.png")
end

# ============================================================
# MAIN
# ============================================================

function main()

    z1_vals, z2_vals, Q = build_grid()

    path = hill_climb(z1_vals, z2_vals, Q)

    plot_all(z1_vals, z2_vals, Q, path)

end

main()
