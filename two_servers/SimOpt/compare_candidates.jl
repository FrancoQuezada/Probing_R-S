# compare_candidates.jl
#
# Standalone experiment script for SimOpt009.jl.
# It compares four probing candidates under a common independent CRN evaluation set:
#   1. z_base = [0.5, 0.5, 0.5, 0.5]
#   2. z_full = [0.95, 0.95, 0.95, 0.95]
#   3. z_crn  = result of optimize_z_crn(...)
#   4. z_sgd  = result of the original optimize_z(...)
#
# This script does not modify source files.
# It assumes SimOpt009.jl is in the same directory from which this script is run.

include("SimOpt009.jl")

# ============================================================
# SETTINGS
# ============================================================

policies = generate_policies()

# Probing costs.
# Use zeros for the no-cost case.
alpha = [0.0, 0.0, 0.0, 0.0]

# Uncomment this line if you want to test positive probing costs.
# alpha = [30.0, 40.0, 50.0, 60.0]

# Number of jobs per simulated path.
N = 200

# Optimization settings.
# For a quick smoke test, use smaller values such as:
#   opt_reps = 20
#   opt_iter = 2
#   eval_reps = 20
#   N = 30
opt_reps = 200
opt_iter = 20

# Independent CRN validation settings.
eval_reps = 500

# Seeds.
crn_opt_seed = 1234
crn_eval_seed = 900000

# Initial point for both optimizers.
z0 = [0.8, 0.8, 0.8, 0.8]

# Step size used by both optimizers.
step_size = 3e-5

# ============================================================
# BASELINE CANDIDATES
# ============================================================

z_base = [0.5, 0.5, 0.5, 0.5]
z_full = [0.95, 0.95, 0.95, 0.95]

println("\n============================================================")
println("Candidate comparison experiment")
println("============================================================")
println("Policies: ", length(policies))
println("N: ", N)
println("alpha: ", alpha)
println("opt_iter: ", opt_iter)
println("opt_reps: ", opt_reps)
println("eval_reps: ", eval_reps)
println("z0: ", z0)
println("step_size: ", step_size)

# ============================================================
# OPTIMIZE CANDIDATES
# ============================================================

println("\n============================================================")
println("Optimizing z_crn with optimize_z_crn")
println("============================================================\n")

z_crn = optimize_z_crn(
    policies;
    max_iter=opt_iter,
    reps=opt_reps,
    N=N,
    step_size=step_size,
    alpha=alpha,
    z0=z0,
    seed=crn_opt_seed,
)

println("\nz_crn = ", z_crn)

println("\n============================================================")
println("Optimizing z_sgd with original optimize_z")
println("============================================================\n")

z_sgd = optimize_z(
    policies;
    max_iter=opt_iter,
    reps=opt_reps,
    N=N,
    step_size=step_size,
    alpha=alpha,
    z0=z0,
)

println("\nz_sgd = ", z_sgd)

# ============================================================
# INDEPENDENT CRN VALIDATION SET
# ============================================================

println("\n============================================================")
println("Building independent CRN evaluation noise set")
println("============================================================\n")

noise_eval = [
    generate_base_noise(crn_eval_seed + r; N=N, m=length(alpha))
    for r in 1:eval_reps
]

# ============================================================
# EVALUATE CANDIDATES ON THE SAME EVALUATION SET
# ============================================================

candidates = [
    ("z_base", z_base),
    ("z_full", z_full),
    ("z_crn",  z_crn),
    ("z_sgd",  z_sgd),
]

println("\n============================================================")
println("Candidate evaluations on common CRN evaluation noise")
println("============================================================\n")

for (name, z) in candidates
    out = estimate_Q_given_noise(z, policies, noise_eval; alpha=alpha)

    println(name)
    println("  z         = ", z)
    println("  Q         = ", out.Q)
    println("  objective = ", out.objective)
    println("  se        = ", out.se)
    println()
end

# ============================================================
# PAIRED COMPARISONS AGAINST z_base
# ============================================================

println("\n============================================================")
println("Paired CRN comparisons against z_base")
println("============================================================\n")

for (name, z) in candidates
    if name == "z_base"
        continue
    end

    out = compare_z_pairwise(z_base, z, policies, noise_eval; alpha=alpha)

    println("z_base -> ", name)
    println("  mean_diff      = ", out.mean_diff)
    println("  se_diff        = ", out.se_diff)
    println("  signal/noise   = ", out.mean_diff / out.se_diff)
    println("  objective_diff = ", out.objective_diff)
    println("  QA             = ", out.QA)
    println("  QB             = ", out.QB)
    println("  objectiveA     = ", out.objectiveA)
    println("  objectiveB     = ", out.objectiveB)
    println()
end

println("\nDone.")
