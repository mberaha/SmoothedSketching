include("../julia_src/jsketch.jl")

using IterTools
using DataFrames
using CSV
using Serialization
using StatsBase
using Random

import Logging
Logging.disable_logging(Logging.Warn) # or e.g. Logging.Info

J = 128
NREP = 10
TRAIN_PERC = 0.05
NDATAMAX = 100000
NDATA = [100, 1000, 10000, 100000]

PY_ALPHAS = [0.0, 0.25, 0.5, 0.75]
PY_THETAS = [1.0, 10.0, 100.0, 1000.0]
ZIPF_C = [1.3, 1.6, 1.9, 2.2]
MODELS = ["DP", "NIG", "NGG"]

# NREP = 2
# PY_ALPHAS = [0.0]
# PY_THETAS = [1.0, 10.0]
# ZIPF_C = [1.9, 2.2]

Random.seed!(20230719)

function estimate_params(model::String, sketch::AbstractVector, train_data::AbstractVector)
    if model == "DP"
        return Sketch.fit_dp(sketch)
    elseif model == "NIG"
        return Sketch.fit_nig(sketch)
    elseif model == "NGG"
        return Sketch.fit_ngg(train_data)
    end
end

function estimate_params_mock(model::String, sketch::AbstractVector, 
                              train_data::AbstractVector)
    if model == "DP"
        return Sketch.DPParams(10.0)
    elseif model == "NIG"
        return Sketch.NGGParams(10.0, 0.5, 0.5)
    elseif model == "NGG"
        return Sketch.NGGParams(10.0, 0.5, 0.1)
    end
end


function run_simulation(true_dataset, data_gen, data_gen_params, nrep)
    println("Running Simulation")
    hf = Sketch.generate_hash(J, rand(10000:100000000))

    errors = zeros((length(MODELS), length(NDATA)))
    k_estim = zeros((length(MODELS), length(NDATA)))


    for (i, n) in enumerate(NDATA)
        data = true_dataset[1:n]
        train_data = data[1:Int(n * TRAIN_PERC)]
        sketch = Sketch.hash_dataset(data, [hf], J)[1, :]

        k_true = length(unique(data))
        params = []
        for (l, m) in enumerate(MODELS)
            p = estimate_params(m, sketch, train_data)
            params = push!(params, p)
            k_estim[l, i] = Sketch.card_est(p, sketch)
        end
        errors[:, i] = abs.(k_estim[:, i] .- k_true)
    end    

    if data_gen == "py"
        filename = "results/card_py_$(data_gen_params[1])_$(data_gen_params[2])_rep_$(nrep).jdat"
    else 
        filename = "results/card_zipf_$(data_gen_params)_rep_$(nrep).jdat"
    end

    out = Dict(
        "true_data" => true_dataset,
        "errors" => errors,
        "k_estim" => k_estim,
        "data_gen_params" => data_gen_params,
        "data_gen" => data_gen,
        "rep" => nrep
    )

    Serialization.serialize(filename, out)
    return errors
end

function run_simulation_mock(true_dataset, data_gen, data_gen_params, nrep)
    println(data_gen, ", params: ", data_gen_params, ", k: ", length(unique(true_dataset)))
    return zeros((length(MODELS), length(NDATA)))
end


function main()

    py_params = [
        repeat(PY_THETAS,1,length(PY_ALPHAS))'[:] repeat(PY_ALPHAS,length(PY_THETAS),1)[:]]
    error_dfs = []

    @Threads.threads for repnum in 1:NREP
        datasets = []
        data_gen = []
        for i in 1:size(py_params)[1]
            data = Sketch.sample_py(py_params[i, 1], py_params[i, 2], NDATAMAX)
            datasets = push!(datasets, data)
            data_gen = push!(data_gen, "py")
        end

        params = convert(Vector{Any}, [py_params[i, :] for i in 1:size(py_params,1)])
        
        for c in ZIPF_C
            datasets = push!(datasets, Sketch.rand_zipf(Sketch.Zipf(c), NDATAMAX))
            data_gen = push!(data_gen, "zipf")
            params = push!(params, c)
        end 

        errors = Array{Matrix{Float64}}(undef, length(params))
        @Threads.threads for i in 1:length(datasets)
            errors[i] = run_simulation(datasets[i], data_gen[i], params[i], repnum)
        end

        Serialization.serialize("results/card_sim_errors_rep_$(repnum).jdat", errors)

        println("ASSEMBLING DATAFRAME")


        colnames = [ "$(x)" for x in NDATA ]
        df = nothing

        for i in 1:length(errors)
            currdf = DataFrame(errors[i], colnames)
            currdf[!, "DataGen"] = repeat([data_gen[i]], size(errors[i], 1))
            currdf[!, "Params"] = repeat([params[i]], size(errors[i], 1))
            currdf[!, "Model"] = MODELS
            if df === nothing
                df = currdf
            else 
                df = [df; currdf]
            end
        end

        df[!, "repnum"] = repeat([repnum], size(df, 1))
        error_dfs = push!(error_dfs, df)
    end

    final_df = error_dfs[1]
    for j in 2:NREP
        final_df = [final_df; error_dfs[j]]
    end

    CSV.write("results/card_simulation_results.csv", final_df)
end 


main()

