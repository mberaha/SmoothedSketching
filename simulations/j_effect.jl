include("../julia_src/jsketch.jl")

using Distributed
using IterTools
using DataFrames
using CSV
using Serialization
using StatsBase
using Random
using SharedArrays

import Logging
Logging.disable_logging(Logging.Warn) # or e.g. Logging.Info

NREP = 10
MODELS = ["DP", "NGG"]
Js = [10, 100, 1000, 10000]

PY_ALPHAS = [0.25,0.75]
PY_THETAS = [100.0]
NDATA = 100000
 
Random.seed!(20230719)


function get_freq_bin(f)
    bin = Int(ceil(log(f) / log(4) + 1))
    if bin > 6
        bin = 6
    end
    return bin
end


function one_simulation(data, data_gen, data_gen_params, repnum)
    println("RUNNING SIMULATION")

    n_mae_buckets = get_freq_bin(1e12)

    uniq2cnt = countmap(data)    
    k_true = length(unique(data))

    maes = zeros(length(MODELS), length(Js), n_mae_buckets)
    card_errors = zeros(length(MODELS), length(Js))
    k_estim = zeros(length(MODELS), length(Js))

    n_by_bin = zeros(n_mae_buckets)
    for (k, cnt) in uniq2cnt
        mae_bucket = get_freq_bin(cnt)
        n_by_bin[mae_bucket] += 1
    end
    
    ngg_params = Sketch.fit_ngg(data[1:5000])

    for (i, J) in enumerate(Js)
        hf = Sketch.generate_hash(J, rand(10000:100000000))
        sketch = Sketch.hash_dataset(data, [hf], J)[1, :]

        params = [Sketch.fit_dp(sketch), ngg_params]
        for p in 1:length(params)
            k_estim[p, i] =  Sketch.card_est(params[p], sketch)
        end
        card_errors[:, i] = abs.(k_estim[:, i] .- k_true)

        # for (k, true_f) in uniq2cnt
        #     bin = get_freq_bin(true_f)
        #     cj = sketch[hf(k)]
        #     est_f = Sketch.freq_est.(params, Int(cj), J)
        #     err = abs.(true_f .- est_f) ./ n_by_bin[bin]
        #     maes[:, i, bin] .+= err
        # end
    end

    out = Dict(
        "true_data" => data,
        "maes" => maes,
        "card_errors" => card_errors,
        "params" => data_gen_params,
        "rep" => repnum
    )

    filename = "results/j_effect_py_$(data_gen_params[1])_$(data_gen_params[2])_rep_$(repnum).jdat"
    Serialization.serialize(filename, out)

    return maes, card_errors
end


function main()

    py_params = [
        repeat(PY_THETAS,1,length(PY_ALPHAS))'[:] repeat(PY_ALPHAS,length(PY_THETAS),1)[:]]
    params = convert(Vector{Any}, [py_params[i, :] for i in 1:size(py_params,1)])
        
    maes_dfs = []
    card_dfs = []

    @Threads.threads for repnum in 1:NREP       
        maes = Array{Array{Float64, 3}}(undef, length(params))
        card_errors = Array{Array{Float64, 2}}(undef, length(params))

        @Threads.threads for i in 1:length(params)
            dataset = Sketch.sample_py(
                params[i][1], params[i][2], NDATA)
            tmp = one_simulation(dataset, "py", params[i], repnum)
            maes[i] = tmp[1]
            card_errors[i] = tmp[2]
        end

        # Serialization.serialize("results/multiview_errors_rep_$(repnum).jdat", errors)

        println("ASSEMBLING DATAFRAME")

        bins = [0, 1, 4, 16, 64, 256, "Inf"]
        bins = collect(zip(bins[1:end-1], bins[2:end]))
        mae_colnames = ["($(x), $(y)]" for (x, y) in bins ]
        card_colnames = ["$(x)" for x in Js]
        
        maes_df = nothing
        card_df = nothing
        for i in 1:length(maes)
            for l in 1:length(MODELS)
                nrows = size(maes[i][l, :, :], 1)
                currdf = DataFrame(maes[i][l, :, :], mae_colnames)
                currdf[!, "Params"] = repeat([params[i]], nrows)
                currdf[!, "Model"] = repeat([MODELS[l]], nrows)
                currdf[!, "J"] = Js

                if maes_df === nothing
                    maes_df = currdf
                else 
                    maes_df = [maes_df; currdf]
                end               
            end

            currdf = DataFrame(card_errors[i], card_colnames)
            currdf[!, "Model"] = MODELS
            currdf[!, "Params"] = repeat([params[i]], size(currdf, 1))

            if card_df === nothing
                card_df = currdf
            else 
                card_df = [card_df; currdf]
            end 
        end

        maes_dfs = push!(maes_dfs, maes_df)
        card_dfs = push!(card_dfs, card_df)

    end

    final_df_mae = maes_dfs[1]
    for j in 2:NREP
        final_df_mae = [final_df_mae; maes_dfs[j]]
    end
    # println("FINAL_DF_MAE: ", size(final_df_mae))
    # CSV.write("results/jeffect_freq_simulation_results.csv", final_df_mae)

    final_df_card = card_dfs[1]
    for j in 2:NREP
        final_df_card = [final_df_card; card_dfs[j]]
    end
    println("FINAL_DF_CARD: ", size(final_df_card))
    CSV.write("results/jeffect_card_simulation_results.csv", final_df_card)    
end 


main()