include(joinpath(@__DIR__,"probing_validation_experiment.jl"))
if abspath(PROGRAM_FILE)==@__FILE__
    ProbingRegimeValidation.main()
end
