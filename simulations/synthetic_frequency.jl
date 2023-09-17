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
NTRAIN = 5000
NDATA = 100000

PY_ALPHAS = [0.0, 0.25, 0.5, 0.75]
PY_THETAS = [1.0, 10.0, 100.0, 1000.0]
ZIPF_C = [1.3, 1.6, 1.9, 2.2]
# ZIPF_C = [1.3, 1.6, 1.9]

Random.seed!(20230719)

function get_freq_bin(f)
    bin = Int(ceil(log(f) / log(4) + 1))
    if bin > 6
        bin = 6
    end
    return bin
end


function estimate_params(model::String, sketch::AbstractVector, train_data::AbstractVector)
    if model == "DP"
        return Sketch.fit_dp(sketch)
    elseif model == "NIG"
        return Sketch.fit_nig(sketch)
    elseif model == "NGG"
        return Sketch.fit_ngg(train_data)
    end
end


function run_simulation(true_dataset, data_gen, data_gen_params, nrep)
    println("Running Simulation")

    hf = Sketch.generate_hash(J, rand(10000:100000000))
    sketch = Sketch.hash_dataset(true_dataset, [hf], J)[1, :]
    train_data = true_dataset[1:NTRAIN]
    uni2cnt = countmap(true_dataset)

    models = ["DP", "NIG", "NGG"]
    params = []

    for m in models
        params = push!(params, estimate_params(m, sketch, train_data))
    end

    nbins = get_freq_bin(1e12)
    freq_est_errors = zeros((length(models), nbins))
    n_by_bin = zeros(nbins)

    for (k, true_f) in uni2cnt
        bin = get_freq_bin(true_f)
        n_by_bin[bin] += 1.0
        cj = sketch[hf(k)]
        est_f = Sketch.freq_est.(params, cj, J)
        err = abs.(true_f .- est_f)
        freq_est_errors[:, bin] .+= err
    end

    for b in 1:nbins
        freq_est_errors[:, b] ./= n_by_bin[b]
    end


    if data_gen == "py"
        filename = "results/py_$(data_gen_params[1])_$(data_gen_params[2])_rep_$(nrep).jdat"
    else 
        filename = "results/zipf_$(data_gen_params)_rep_$(nrep).jdat"
    end

    out = Dict(
        "true_data" => true_dataset,
        "hf" => hf,
        "est_params" => params,
        "errors" => freq_est_errors,
        "data_gen_params" => data_gen_params,
        "data_gen" => data_gen,
        "rep" => nrep
    )

    Serialization.serialize(filename, out)

    return freq_est_errors
end

function run_simulation_mock(true_dataset, data_gen, data_gen_params, nrep)
    nbins = get_freq_bin(1e12)
    models = ["DP", "NIG", "NGG"]
    freq_est_errors = zeros((length(models), nbins))
    return freq_est_errors
end

function main()

    py_params = [
        repeat(PY_THETAS,1,length(PY_ALPHAS))'[:] repeat(PY_ALPHAS,length(PY_THETAS),1)[:]]
    error_dfs = []

    @Threads.threads for repnum in 1:NREP
        datasets = []
        data_gen = []
        for i in 1:size(py_params)[1]
            datasets = push!(datasets, Sketch.sample_py(
                py_params[i, 1], py_params[i, 2], NDATA))
            data_gen = push!(data_gen, "py")
        end

        params = convert(Vector{Any}, [py_params[i, :] for i in 1:size(py_params,1)])
        
        for c in ZIPF_C
            datasets = push!(datasets, Sketch.rand_zipf(Sketch.Zipf(c), NDATA))
            data_gen = push!(data_gen, "zipf")
            params = push!(params, c)
        end 

        errors = Array{Matrix{Float64}}(undef, length(params))
        @Threads.threads for i in 1:length(datasets)
            errors[i] = run_simulation(datasets[i], data_gen[i], params[i], repnum)
        end

        Serialization.serialize("results/frequency_sim_errors_rep_$(repnum).jdat", errors)

        println("ASSEMBLING DATAFRAME")

        bins = [0, 1, 4, 16, 64, 256, "Inf"]
        bins = collect(zip(bins[1:end-1], bins[2:end]))
        colnames = ["($(x), $(y)]" for (x, y) in bins ]
        df = nothing

        for i in 1:length(errors)
            currdf = DataFrame(errors[i], colnames)
            currdf[!, "DataGen"] = repeat([data_gen[i]], size(errors[i], 1))
            currdf[!, "Params"] = repeat([params[i]], size(errors[i], 1))
            currdf[!, "Model"] = ["DP", "NIG", "NGG"]
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

    CSV.write("results/frequency_simulation_results.csv", final_df)
end 


main()

