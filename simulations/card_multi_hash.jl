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


Js = [50, 100, 500, 1000]
max_mem = 1000

PY_ALPHAS = [0.25,0.75]
PY_THETAS = [10.0, 100.0, 1000.0]
NDATA = [5000, 10000, 100000, 250000]

 
Random.seed!(20230719)




function one_simulation(full_data, py_theta, py_alpha, repnum)
    println("RUNNING SIMULATION WITH PARAMS: ($(py_theta), $(py_alpha)), REP: $(repnum)")
    
    col_name_and_type = Vector{Pair{Any, Any}}()
    col_name_and_type = push!(col_name_and_type, Pair("ndata", Int64[]))
    col_name_and_type = push!(col_name_and_type, Pair("model", String[]))
    col_name_and_type = push!(col_name_and_type, Pair("mean_fn", String[]))
    for j in Js
        col_name_and_type = push!(col_name_and_type, Pair("error_J_$(j)", Float64[]))
    end

    # ngg_error_geom = zeros((length(NDATA), length(Js)))
    # dp_error_geom = zeros((length(NDATA), length(Js)))
    # ngg_error_avg = zeros((length(NDATA), length(Js)))
    # dp_error_avg = zeros((length(NDATA), length(Js)))
    # ngg_k_estim = zeros((length(NDATA), length(Js)))
    # dp_k_estim = zeros((length(NDATA), length(Js)))
    true_k = zeros(length(NDATA))
    df = DataFrame(col_name_and_type)

    for (i, n) in enumerate(NDATA)
        data = full_data[1:n]
        true_k[i] = length(unique(data))
        ngg_p = Sketch.fit_ngg(data[1:Int(n * 0.1)])
        ngg_err_avg = zeros(length(Js))
        ngg_err_geom = zeros(length(Js))
        dp_err_avg = zeros(length(Js))
        dp_err_geom = zeros(length(Js))
        for (j, J) in enumerate(Js)
            m = Int(max_mem / J)
            hash_functions = [Sketch.generate_hash(J, rand(10000:100000000)) for _ in 1:m]
            hash_data = Sketch.hash_dataset(data, hash_functions, J)
            
            ngg_est_geom = Sketch.card_est_multiview(hash_data, [ngg_p for _ in 1:m], geomean)
            ngg_err_geom[j] = abs(true_k[i] - ngg_est_geom)
            ngg_est_avg = Sketch.card_est_multiview(hash_data, [ngg_p for _ in 1:m], mean)
            ngg_err_avg[j] = abs(true_k[i] - ngg_est_avg)

            dp_p = Sketch.fit_multiview(hash_data, [], "DP")
            dp_est_geom = Sketch.card_est_multiview(hash_data, dp_p, geomean)
            dp_err_geom[j] = abs(true_k[i] - dp_est_geom)
            dp_est_avg = Sketch.card_est_multiview(hash_data, dp_p, mean)
            dp_err_avg[j] = abs(true_k[i] - dp_est_avg)
        end
        curr_row = [[n, "NGG", "geom"]; ngg_err_geom]
        df = push!(df, curr_row)
        df = push!(df, [[n, "NGG", "avg"]; ngg_err_avg])
        df = push!(df, [[n, "DP", "geom"]; dp_err_geom])
        df = push!(df, [[n, "DP", "avg"]; dp_err_avg])
    end
    
    df[!, "PY_THETA"] = repeat([py_theta], size(df)[1])
    df[!, "PY_ALPHA"] = repeat([py_alpha], size(df)[1])
    df[!, "repnum"] = repeat([repnum], size(df)[1])

    CSV.write("results/multiview_card_simulation_results_rep$(repnum)_theta$(py_theta)_alpha$(py_alpha).csv", df)
    println("FINISHED SIMULATION WITH PARAMS: ", py_theta, ", ", py_alpha, ", REP: ", repnum)

    return df
end


function main()
    py_params = [
            repeat(PY_THETAS,1,length(PY_ALPHAS))'[:] repeat(PY_ALPHAS,length(PY_THETAS),1)[:]]
    params = convert(Vector{Any}, [py_params[i, :] for i in 1:size(py_params,1)])

    dfs = []


    # @Threads.threads 
    for repnum in 1:NREP
        # @Threads.threads 
        for i in 1:length(params)
            dataset = Sketch.sample_py(
                params[i][1], params[i][2], NDATA[end])
            res = one_simulation(dataset, params[i][1], params[i][2], repnum)
            dfs = push!(dfs, res)
        end
    end

    final_df = dfs[1]
    for i in 2:length(dfs)
        final_df = [final_df; dfs[i]]
    end

    CSV.write("results/multiview_card_simulation_results.csv", final_df)

    
end 


main()