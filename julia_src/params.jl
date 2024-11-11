struct DPParams 
    theta::Float64
end

struct NGGParams
    theta::Float64
    alpha::Float64
    tau::Float64
end

mutable struct LogGenFactorialCoeffs
    cache::Matrix{Float16}
    sigma::Float64
end


function LogGenFactorialCoeffs(sigma::Float64)
    cache = zeros(Float16, (1, 1))
    cache[1, 1] = log(sigma)
    return LogGenFactorialCoeffs(cache, sigma)
end


mutable struct PYParams
    theta::Float64
    alpha::Float64
    fac_coeffs::LogGenFactorialCoeffs
end


function PYParams(theta::Float64, sigma::Float64)
    return PYParams(theta, sigma, LogGenFactorialCoeffs(sigma))
end