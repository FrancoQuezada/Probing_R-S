using Random
using Statistics

# ============================================================
# PARAMETERS
# ============================================================

const REPS = 2000   # increase if needed
const SEED = 1234

include("SimOpt009.jl")
const POLICIES = generate_policies()

# ============================================================
# 2D WRAPPER
# ============================================================

function make_full_z(z1, z2)
    return [z1, 1.0, 1.0, z2]
end

# ============================================================
# EVALUATE ONE z WITH CRN
# ============================================================

function evaluate_with_crn(z; reps=REPS)

    values = zeros(reps)

    for r in 1:reps
        rng = MersenneTwister(SEED + r)

        # IMPORTANT: same path used for all policies
        path = generate_sample_path(z, rng)

        best_val = -Inf

        for p in POLICIES
            val = evaluate_policy_on_path(p, path)
            if val > best_val
                best_val = val
            end
        end

        values[r] = best_val
    end

    mean_val = mean(values)
    se_val = std(values) / sqrt(reps)

    return mean_val, se_val
end

# ============================================================
# MAIN COMPARISON
# ============================================================

function main()

    zA = make_full_z(1.0, 1.0)
    zB = make_full_z(0.83, 0.85)

    println("Evaluating (1,1)...")
    meanA, seA = evaluate_with_crn(zA)

    println("Evaluating (0.83,0.85)...")
    meanB, seB = evaluate_with_crn(zB)

    println("\n------------------------------")
    println("RESULTS")
    println("------------------------------")

    println("(1,1):        ", round(meanA, digits=2),
            "   SE = ", round(seA, digits=2))

    println("(0.83,0.85):  ", round(meanB, digits=2),
            "   SE = ", round(seB, digits=2))

    diff = meanB - meanA
    se_diff = sqrt(seA^2 + seB^2)

    println("\nDifference (B - A): ",
            round(diff, digits=2),
            "   SE = ", round(se_diff, digits=2))

    println("------------------------------")
end

main()