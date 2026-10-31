include(joinpath(@__DIR__, "probing_experiments_common.jl"))
if abspath(PROGRAM_FILE) == @__FILE__
    ProbingExperiments.main(8)
end
