using SpecialFunctions
using NNlib
using Optimization
using OptimizationOptimJL
using Random
using Distributions
using StatsBase

include("estimate_params.jl")
include("smoothed_estimators.jl")


function freq_post(l_max::Integer, c::Integer, param::DPParams, 
                   J::Integer, logscale::Bool)
    theta = param.theta
    l_range = collect(0:min(l_max, c))
    out = zeros(length(l_range))

    out .+= log(theta / J) 
    out += loggamma(c + 1.0) .- loggamma.(c .- l_range .+ 1.0)
    out += loggamma.(c .- l_range .+ theta / J) .- loggamma(c + theta / J + 1.0)

    if l_max > c
        out = append!(out, zeros(l_max - c) .- Inf)
    end

    if !logscale
        out = exp.(out)
    end

    return out
end


Base.@kwdef mutable struct beta_integral_ngg
    params::NGGParams = NGGParams(1.0, 0.0, 0.0)
    n_mc::Integer = 1000
    vs::AbstractVector = [] 
    cache::Dict{Tuple{Float64, Float64}, Float64} = Dict()
    J::Integer = -1
end

function sample_v(param, J, n_mc)
    es = rand(Exponential(1.0), n_mc)
    bs = rand(Beta(1 - param.alpha, param.alpha), n_mc)
    betaj = param.theta / (param.alpha * J) * param.tau^param.alpha
    vs = bs .* (1.0 .- (betaj ./ (betaj .+ es)).^(1.0 / param.alpha) )
    return vs
end

function eval!(integral::beta_integral_ngg, a::Float64, b::Float64)

    if length(integral.vs) == 0
        integral.vs = sample_v(integral.params, integral.J, integral.n_mc)
    end 

    out = get(integral.cache, (a, b), nothing)
    if out === nothing
        integrand(x) = x^a * (1 - x)^b
        out = mean(integrand.(integral.vs))
        integral.cache[(a, b)] = out
    end

    return out

end


function freq_post!(l_max::Integer, c::Integer, param::NGGParams, 
                   J::Integer, logscale::Bool, 
                   integral_cache::Union{beta_integral_ngg, Nothing}=nothing)
    if integral_cache === nothing
        integral_cache = beta_integral_ngg
    end

    function mc_integral(l)
        log_binom = - log1p(c) - logbeta(l + 1, c - l + 1)
        return exp(
            log_binom + log(
                Sketch.eval!(integral_cache, Float64(l), Float64(c -l))))
    end

    l_range = collect(0:min(l_max, c))
    out = zeros(length(l_range))
    for l in 0:l_max
        out[l+1] = mc_integral(l)
    end

    if l_max > c
        out = append!(out, zeros(l_max - c))
    end

    if logscale
        out = log.(out)
    end

    return out
end


function dp_post(l_max::Integer, c::Integer, 
                 g::Float64, width::Integer, logscale::Bool)

    l_range = collect(0:min(l_max, c))
    out = zeros(length(l_range))
    out .+= log(g / width) 
    out += loggamma(c + 1.0) .- loggamma.(c .- l_range .+ 1.0)
    out += loggamma.(c .- l_range .+ g / width) .- loggamma(c + g / width + 1.0)

    if l_max > c
        out = append!(out, zeros(l_max - c) .- Inf)
    end

    if !logscale
        out = exp.(out)
    end

    return out
end


function fit_multiview(sketches, train_data, model)
    m = size(sketches)[1]
    if model == "NGG"
        p = fit_ngg(train_data)
        return [p for _ in 1:m]
    end

    out = []
    for i in 1:m
        if model == "DP"
            p = fit_dp(sketches[i, :])            
        elseif model == "NIG"
            p = fit_nig(sketches[i, :])
        end
        out = push!(out, p)
    end
    return out
end


function PoE_mean(logprobas)
    post_probas = mapreduce(permutedims, vcat, logprobas)
    post_prob = softmax(sum(post_probas, dims=1)[1, :])
    c_up = length(post_prob) - 1
    return sum((0:c_up) .* post_prob)
end


function MIN_mean(logprobas)
    post_probas = exp.(mapreduce(permutedims, vcat, logprobas))
    post_cdfs = cumsum(post_probas, dims=2)
    min_cdf = (1.0 .- prod(1.0 .- post_cdfs, dims=1))[1, :]
    post_prob = prepend!(diff(min_cdf), [min_cdf[1]])
    c_up = length(post_prob) - 1
    return sum((0:c_up) .* post_prob)
end


function product_expert_post(sketches, params, hs)
    m = size(sketches)[1]
    c_js = Int.(sketches[CartesianIndex.(collect(1:m), hs)])
    min_c = Int(minimum(c_js))
    if min_c == 0
        return [0]
    end

    post_probas = freq_post.(min_c, c_js, params, size(sketches)[2], true)
    post_probas = mapreduce(permutedims, vcat, post_probas)
    out = sum(post_probas, dims=1)[1, :]
    return softmax(out)
end


function min_expert_post(sketches, params, hs)
    m = size(sketches)[1]
    c_js = Int.(sketches[CartesianIndex.(collect(1:m), hs)])
    min_c = Int(minimum(c_js))
    if min_c == 0
        return [0]
    end

    post_probas = freq_post.(min_c, c_js, params, size(sketches)[2], false)
    post_probas = mapreduce(permutedims, vcat, post_probas)
    post_cdfs = cumsum(post_probas, dims=2)
    min_cdf = (1.0 .- prod(1.0 .- post_cdfs, dims=1))[1, :]
    return prepend!(diff(min_cdf), [min_cdf[1]])
end


function card_est_multiview(sketches, params, mean_fn=geomean)
    m = size(sketches)[1]
    local_cards = zeros(m)
    for j in 1:m
        local_cards[j] = card_est(params[j], sketches[j, :])
    end
    return mean_fn(local_cards)
end