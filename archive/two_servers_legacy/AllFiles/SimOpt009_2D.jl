using Random
using Plots
using LinearAlgebra 

# ============================================================
# USER PARAMETERS (ADJUST IF NEEDED)
# ============================================================

const GRID_REPS = 2
const GRAD_REPS = 100
const SEED = 1234

# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

include("SimOpt009.jl")

# Assumes these exist after include:
# - evaluate_Q(z, policies; ...)
# - select_best_policy(z, policies; ...)
# - estimate_gradient(z, policy; ...)

# And a list of policies:
const POLICIES = generate_policies()

# ============================================================
# 2D WRAPPERS
# ============================================================

function make_full_z(z1, z2)
    return [z1, 1.0, 1.0, z2]
end

function evaluate_Q_2d(z1, z2)
    z = make_full_z(z1, z2)
    return evaluate_Q(z, POLICIES; reps=GRID_REPS, seed=SEED).objective
end

function gradient_2d(z1, z2, iter)

    z = make_full_z(z1, z2)

    est = estimate_Q_and_gradient(z, POLICIES, 10_000 * iter;
        reps=GRAD_REPS
    )

    g = est.gradient

    return [g[1], g[4]]
end


# ============================================================
# GRID
# ============================================================

function build_grid(n::Int=40; zmin=0.5, zmax=1.0)

    z1_vals = collect(range(zmin, zmax, length=n))
    z2_vals = collect(range(zmin, zmax, length=n))

    Q = zeros(n, n)

    best_val = -Inf
    best_point = (0.0, 0.0)

    for i in 1:n
        for j in 1:n

            z1 = z1_vals[i]
            z2 = z2_vals[j]

            val = evaluate_Q_2d(z1, z2)
            Q[j, i] = val

            if val > best_val
                best_val = val
                best_point = (z1, z2)
            end
        end
    end

    return z1_vals, z2_vals, Q, best_point
end


# ============================================================
# SGD
# ============================================================

function run_sgd_2d(; z0=[0.8,0.8], steps=100, lr=0.002)

    z = copy(z0)

    path = Vector{Vector{Float64}}()
    grads = Vector{Vector{Float64}}()

    push!(path, copy(z))

println("iter   z              grad")

    for k in 1:steps

        g = gradient_2d(z[1], z[2], k)   # ← note: pass k

        println(rpad(k,5),
            rpad(string(round.(z, digits=4)), 18),
            round.(g, digits=2))

        push!(grads, copy(g))

        z .= z .+ lr .* g
        z .= clamp.(z, 0.5, 0.95)

    push!(path, copy(z))
end

    return path, grads
end


# ============================================================
# PLOT
# ============================================================

function plot_results(z1_vals, z2_vals, Q, path, grads, best_point)

    contour(z1_vals, z2_vals, Q,
        xlabel="z₁",
        ylabel="z₂",
        title="Contour + SGD path",
        linewidth=1
    )

    xs = [z[1] for z in path]
    ys = [z[2] for z in path]

    plot!(xs, ys, marker=:circle, label="SGD path")

    for k in 1:5:length(grads)
        z = path[k]
        g = grads[k]

        quiver!([z[1]], [z[2]],
            quiver=([0.0005*g[1]], [0.0005*g[2]]),
            label=false)
    end

    scatter!([best_point[1]], [best_point[2]],
        marker=:star, markersize=10,
        label="Best grid point")
end


# ============================================================
# MAIN
# ============================================================

function main()

    #println("Building grid...")
    #z1_vals, z2_vals, Q, best_point = build_grid(40)

    #println("Best grid point: ", best_point)
    #println("Value at (1,1): ", evaluate_Q_2d(1.0, 1.0))

    println("Running SGD...")
    path, grads = run_sgd_2d()

    #println("Plotting...")
    #plot_results(z1_vals, z2_vals, Q, path, grads, best_point)

end

main()