# In this file, we demonstrate the use of the likelihood ratio method to estimate the mean of a target distribution using samples 
# from a proposal distribution. We also compute the effective sample size (ESS) and visualize the top contributors to the estimate.

using Random, Distributions, Statistics, Printf, Plots

Random.seed!(1)

σ = 1.0
r = 50_000
N = 9
x1 = 0.0
Δ = 0.5
xs = [x1 + (n-1)*Δ for n in 1:N]

y1 = rand(Normal(x1, σ), r)

function lr_logstable_mean_and_ess(y_samples, x_target, x_proposal, σ)
    target = Normal(x_target, σ)
    prop   = Normal(x_proposal, σ)

    logw = logpdf.(target, y_samples) .- logpdf.(prop, y_samples)
    m = maximum(logw)
    w̃ = exp.(logw .- m)               # stabilized weights (proportional to true weights)

    # Unbiased LR estimator uses mean(w * y). With stabilized weights:
    # true w = exp(m) * w̃, so mean(true w * y) = exp(m) * mean(w̃ * y)
    lr = exp(m) * mean(w̃ .* y_samples)

    # ESS is scale-invariant, so we can use w̃
    ess = (sum(w̃)^2) / sum(w̃.^2)

    return lr, ess
end

function lr_extra_diagnostics(y_samples, x_target, x_proposal, σ; k = nothing)
    target = Normal(x_target, σ)
    prop   = Normal(x_proposal, σ)

    logw = logpdf.(target, y_samples) .- logpdf.(prop, y_samples)
    m    = maximum(logw)
    w̃    = exp.(logw .- m)                 # stabilized weights
    w    = exp(m) .* w̃                      # true weights

    lr = mean(w .* y_samples)

    # SumW = (1/r) sum(w) should be ~1 in expectation
    sumw = mean(w)

    # choose k
    if k === nothing
        # use ESS-based k (computed from normalized weights)
        ŵ = w̃ ./ sum(w̃)
        ess = 1 / sum(ŵ.^2)
        k = max(1, round(Int, ess))
    end

    ord = sortperm(w; rev=true)
    top = ord[1:k]

    total = sum(w .* y_samples)
    topsum = sum((w .* y_samples)[top])
    topshare = topsum / total

    return lr, sumw, topshare, k, w̃
end


@printf("%5s %10s %12s %14s %10s\n", "n", "x_n", "Exact", "LR(log)", "ESS")

plots = []   # will store 9 histogram objects

for (n, x) in enumerate(xs)

    lr, ess = lr_logstable_mean_and_ess(y1, x, x1, σ)
    @printf("%5d %10.3f %12.6f %14.6f %10.1f\n", n, x, x, lr, ess)

    # Recompute stabilized weights (only for selecting top contributors)
    target = Normal(x, σ)
    prop   = Normal(x1, σ)

    logw = logpdf.(target, y1) .- logpdf.(prop, y1)
    m    = maximum(logw)
    w̃    = exp.(logw .- m)

    ŵ = w̃ ./ sum(w̃)
    ord = sortperm(ŵ; rev=true)

    k = max(1, round(Int, ess))          # avoid k=0 edge case
    k = min(k, length(y1))              # safety

    top_y = y1[ord[1:k]]

    p = histogram(top_y,
                  bins=20,
                  title="x=$(round(x, digits=2)), k=$k",
                  xlabel="y",
                  ylabel="count",
                  legend=false)

    push!(plots, p)
end

# Combine into a 3x3 grid and show once
pgrid = plot(plots..., layout=(3,3), size=(1000,800))
display(pgrid)
