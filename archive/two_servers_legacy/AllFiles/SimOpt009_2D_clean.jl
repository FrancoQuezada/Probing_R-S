
using Random

# ============================================================
# PARAMETERS
# ============================================================

const GRAD_REPS = 300
const SEED = 1234

# ============================================================
# LOAD MODEL (NO AUTO RUN)
# IMPORTANT: Make sure SimOpt009.jl does NOT call run() at the end
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

    est = estimate_Q_and_gradient(z, POLICIES, 10_000 * iter;
        reps=GRAD_REPS
    )

    g = est.gradient
    return [g[1], g[4]]
end

# ============================================================
# SGD (PRINTS LIKE 4D VERSION)
# ============================================================

function optimize_z_2d(; max_iter=100, step_size=5e-5, z0=[0.8,0.8])

    z = copy(z0)

    println("iter  z              grad")
    println("--------------------------------------------------")

    for it in 1:max_iter
        g = zeros(2)

        for s in 1:3   # ← mini-batch size = 3
            g .+= gradient_2d(z[1], z[2], 10_000*it + s)
        end

g ./= 3

        eta = step_size / sqrt(it)
        z .+= eta .* g
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
    println("Running 2D SGD (like 4D)...\n")

    z = optimize_z_2d(
        max_iter=300,
        step_size=3e-5,
        z0=[0.8, 0.8]
    )

    println("\nFinal z = ", z)
end

main()
