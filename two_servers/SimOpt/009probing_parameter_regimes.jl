include(joinpath(@__DIR__,"probing_regime_experiment.jl"))
if abspath(PROGRAM_FILE)==@__FILE__
    ProbingRegimeExperiment.main()
end
