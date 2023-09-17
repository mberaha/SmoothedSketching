include("../julia_src/jsketch.jl")

using IterTools
using DataFrames
using CSV
using Serialization
using StatsBase
using Random

import Logging
Logging.disable_logging(Logging.Warn) # or e.g. Logging.Info


filenames = [
    "results/card_py_10.0_0.25_rep_1.jdat",
    "results/card_py_100.0_0.25_rep_1.jdat",
    "results/card_py_100.0_0.75_rep_1.jdat",
    "results/card_py_1000.0_0.25_rep_1.jdat",
    "results/card_zipf_1.6_rep_1.jdat", 
]
NDATA = [100, 1000, 10000, 100000]

function get_stats(res)
    true_data = res["true_data"]
    true_k = []
    for (i, n) in enumerate(NDATA)
        data = true_data[1:n]
        true_k = push!(true_k, length(unique(data)))
    end

    return true_k, res["k_estim"]'
end

function main()
    colnames = ["DP", "NIG", "NGG"]
    df = nothing

    for fname in filenames
        tmp = deserialize(fname)
        true_k, estim_k = get_stats(tmp)
        currdf = DataFrame(estim_k, colnames)
        currdf[!, "TrueK"] = true_k
        currdf[! "DataGen"] = repeat([tmp["data_gen"]], size(df, 1))
        currdf[!, "Params"] = repeat([tmp["data_gen_params"]], size(df, 1))

        if df === nothing
            df = currdf
        else 
            df = [df; currdf]
        end

    end

    CSV.write("results/card_plot_df.csv", final_df)
end

main()