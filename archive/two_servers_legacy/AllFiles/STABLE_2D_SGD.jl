
using Random
using LinearAlgebra

# ============================================================
# PARAMETERS
# ============================================================

const GRAD_REPS = 200
const SEED = 1234

# ============================================================
# LOAD MODEL
# ============================================================

include("SimOpt009.jl")
const POLICIES = generate_policies()

# ============================================================
# 2D WRAPPER
# ============================================================

function make_full_z(z1, z2)
    return [z1, 1.0, 1.0, z2]
end

function gradient_2d(z1, z2, iter)
    z = make_full_z(z1, z2)

    est = estimate_Q_and_gradient(
        z,
        POLICIES,
        10_000 * iter;
        reps = GRAD_REPS
    )

    g = est.gradient
    return [g[1], g[4]]
end

# ============================================================
# STABLE SGD (NO CRAZY JUMPS)
# ============================================================

function optimize_z_2d(; max_iter=100, z0=[0.8,0.8])

    z = copy(z0)

    println("iter  z              grad")
    println("--------------------------------------------------")

    for it in 1:max_iter

        g = zeros(2)

        # mini-batch averaging
        for s in 1:5
            g .+= gradient_2d(z[1], z[2], 10_000*it + s)
        end
        g ./= 5

        # normalize gradient (key fix)
        g_norm = norm(g)
        if g_norm > 0
            g ./= g_norm
        end

        # diminishing step size
        step = 0.05 / sqrt(it)

        # update
        z .+= step .* g

        # projection
        z .= clamp.(z, 0.5, 0.95)

        println(rpad(it,5),
                rpad(string(round.(z, digits=4)), 15),
                round.(g, digits=2))
    end

    return z
end

# ============================================================
# MAIN
# ============================================================

function main()
    println("Running STABLE 2D SGD...\n")

    z = optimize_z_2d(
        max_iter = 100,
        z0 = [0.8, 0.8]
    )

    println("\nFinal z = ", z)
end

main()
