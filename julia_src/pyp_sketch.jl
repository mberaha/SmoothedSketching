using SpecialFunctions
using NNlib
using Optimization
using OptimizationOptimJL
using Random
using Distributions
using StatsBase

include("params.jl")

function (fac_cache::LogGenFactorialCoeffs)(a, b)
    if (b > a)
        return - Inf
    end

    if (a == 0)
        return 0
    end

    if (b == 0)
        return - Inf
    end


    out = nothing 

    if a <= size(fac_cache.cache)[1]
        out = fac_cache.cache[a, b]
    end

    if out === nothing
        sigma = fac_cache.sigma
        new_cache = zeros(a, a)
        k = size(fac_cache.cache)[1]
        new_cache[1:k, 1:k] .= fac_cache.cache 

        for i in (k+1):a
            new_cache[i, 1] = log(-(sigma-(i-1))) + new_cache[i-1, 1]
            new_cache[i, i] = log(sigma) + new_cache[i-1, i-1]

            for j in 2:(i-1)
                tmp1 = log(sigma) + new_cache[i-1, j-1]
                tmp2 = log((i-1) - (sigma * j)) + new_cache[i-1, j]
                new_cache[i,j] = max(tmp1, tmp2) + log(1.0 + exp(min(tmp1,tmp2) - max(tmp1,tmp2)))
            end
        end
        fac_cache.cache = new_cache
        out = fac_cache.cache[a, b]
    end
    return out
end


function freq_post!(
        l_max::Integer, c::Integer, ndata::Integer, params::PYParams, 
        J::Integer, logscale::Bool)

    theta = params.theta
    alpha = params.alpha 
    log_gen_fac_coeffs = params.fac_coeffs

    l_range = collect(0:l_max)
    out = zeros(length(l_range))


    out .+= log(theta / J) 
    out .+= lpoch.(1 - alpha, l_range)
    out .+= lbinom.(c, l_range)
    
    ta = (theta + alpha) / alpha
    for l in l_range
        num_table = zeros(c-l + 1, ndata - c + 1)
        for i in 0:(c-l)
            for j in 0:(ndata - c)
                num_table[i+1, j+1] = (lpoch(ta, i + j) + i * log(1.0 / J) + 
                    j * log(1.0 - 1.0 / J) + log_gen_fac_coeffs(c - l, i) + log_gen_fac_coeffs(ndata - c, j))
            end
        end
        out[l+1] += logsumexp(vec(num_table))
    end

    out .-= logsumexp(out)

    if !logscale
        out = exp.(out)
    end

    return out
end


function freq_est!(params::PYParams, Cj::Integer, J::Integer, ndata::Integer)
    pmf = freq_post!(Cj, Cj, ndata, params, J, false)
    return sum(collect(0:Cj) .* pmf)
end

