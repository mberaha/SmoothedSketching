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

NREP = 50


Js = [50, 100, 500, 1000]
max_mem = 1000

PY_ALPHAS = [0.25,0.75]
PY_THETAS = [10.0, 100.0, 1000.0]
NDATA = 250000

 
Random.seed!(20230719)

function get_freq_bin(f)
    bin = Int(ceil(log(f) / log(4) + 1))
    if bin > 6
        bin = 6
    end
    return bin
end


function one_simulation(data, data_gen, data_gen_params, repnum)
    println("RUNNING SIMULATION WITH PARAMS: ", data_gen_params)

    n_mae_buckets = get_freq_bin(1e12)

    uniq2cnt = countmap(data)    
    maes_dp_prod = zeros(length(Js), n_mae_buckets)
    maes_dp_min = zeros(length(Js), n_mae_buckets)
    maes_ngg_prod = zeros(length(Js), n_mae_buckets)
    maes_ngg_min = zeros(length(Js), n_mae_buckets)
    maes_cms = zeros(length(Js), n_mae_buckets)
    maes_debiased = zeros(length(Js), n_mae_buckets)



    ndata_by_bucket = zeros(n_mae_buckets)
    for (k, cnt) in uniq2cnt
        mae_bucket = get_freq_bin(cnt)
        ndata_by_bucket[mae_bucket] += 1
    end
    
    ngg_p = Sketch.fit_ngg(data[1:10000])

    for (i, J) in enumerate(Js)
        println("J: ", J)
        m = Int(max_mem / J)
        hash_functions = [Sketch.generate_hash(J, rand(10000:100000000)) for _ in 1:m]
        hash_data = Sketch.hash_dataset(data, hash_functions, J)
        dp_p = Sketch.fit_multiview(hash_data, data[1:10000], "DP")
        ngg_intcache = Sketch.beta_integral_ngg(params=ngg_p, J=J)

        tmp = vec(hash_data)
        cms_bias = quantile(tmp, size(hash_data)[2] / length(tmp))


        for (k, cnt) in uniq2cnt
            hs = Int.([h(k) for h in hash_functions])
            mae_bucket = get_freq_bin(cnt)
            
            c_js = Int.(hash_data[CartesianIndex.(collect(1:m), hs)])
            min_c = Int(minimum(c_js))
            cms_est = min_c
            maes_cms[i, mae_bucket] += abs(cnt - cms_est) / ndata_by_bucket[mae_bucket]
            
            debiased_cms_est = min_c - cms_bias
            maes_debiased[i, mae_bucket] += abs(cnt - debiased_cms_est) / ndata_by_bucket[mae_bucket]

            dp_logprobas = Sketch.freq_post.(min_c, c_js, dp_p, J, true)
            ngg_logprobas = [
                Sketch.freq_post!(min_c, c, ngg_p, J, true, ngg_intcache) for c in c_js]

            maes_dp_prod[i, mae_bucket] += abs(
                cnt - Sketch.PoE_mean(dp_logprobas)) / ndata_by_bucket[mae_bucket]
            maes_dp_min[i, mae_bucket] += abs(
                cnt - Sketch.MIN_mean(dp_logprobas)) / ndata_by_bucket[mae_bucket]

            maes_ngg_prod[i, mae_bucket] += abs(
                cnt - Sketch.PoE_mean(ngg_logprobas)) / ndata_by_bucket[mae_bucket]
            maes_ngg_min[i, mae_bucket] += abs(
                cnt - Sketch.MIN_mean(ngg_logprobas)) / ndata_by_bucket[mae_bucket]
        end
    end

    out = Dict(
        "true_data" => data,
        "errors_dp_min" => maes_dp_min,
        "errors_dp_prod" => maes_dp_prod,
        "errors_ngg_min" => maes_ngg_min,
        "errors_ngg_prod" => maes_ngg_prod,
        "errors_cms" => maes_cms,
        "errors_debiased" => maes_debiased,
        "params" => data_gen_params,
        "rep" => repnum
    )

    # filename = "results/multiview_py_$(data_gen_params[1])_$(data_gen_params[2])_rep_$(repnum).jdat"
    # Serialization.serialize(filename, out)

    return maes_dp_min, maes_dp_prod, maes_ngg_min, maes_ngg_prod, maes_cms, maes_debiased
end


function main()

        
    error_dfs_prod = []
    error_dfs_min = []
    error_dfs_cms = []
    error_dfs_debiased = []


    @Threads.threads for repnum in 1:NREP
        py_params = [
            repeat(PY_THETAS,1,length(PY_ALPHAS))'[:] repeat(PY_ALPHAS,length(PY_THETAS),1)[:]]
        params = convert(Vector{Any}, [py_params[i, :] for i in 1:size(py_params,1)])

        min_dp_errors = Array{Array{Float64, 2}}(undef, length(params))
        prod_dp_errors = Array{Array{Float64, 2}}(undef, length(params))
        min_ngg_errors = Array{Array{Float64, 2}}(undef, length(params))
        prod_ngg_errors = Array{Array{Float64, 2}}(undef, length(params))
        cms_errors = Array{Array{Float64, 2}}(undef, length(params))
        debiased_errors = Array{Array{Float64, 2}}(undef, length(params))


        # @Threads.threads 
        for i in 1:length(params)
            dataset = Sketch.sample_py(
                params[i][1], params[i][2], NDATA)
            tmp = one_simulation(dataset, "py", params[i], repnum)
            min_dp_errors[i] = tmp[1]
            prod_dp_errors[i] = tmp[2]
            min_ngg_errors[i] = tmp[3]
            prod_ngg_errors[i] = tmp[4]
            cms_errors[i] = tmp[5]
            debiased_errors[i] = tmp[6]
        end


        println("ASSEMBLING DATAFRAME")

        bins = [0, 1, 4, 16, 64, 256, "Inf"]
        bins = collect(zip(bins[1:end-1], bins[2:end]))
        colnames = ["($(x), $(y)]" for (x, y) in bins ]
        
        prod_df = nothing
        for i in 1:length(prod_dp_errors)
            currdf_dp = DataFrame(prod_dp_errors[i], colnames)
            currdf_dp[!, "Model"] = repeat(["DP"], size(currdf_dp, 1))
            currdf_dp[!, "J"] = Js
            currdf_ngg = DataFrame(prod_ngg_errors[i], colnames)
            currdf_ngg[!, "Model"] = repeat(["NGG"], size(currdf_ngg, 1))
            currdf_ngg[!, "J"] = Js

            currdf = [currdf_dp; currdf_ngg]
            currdf[!, "DataGen"] = repeat(["py"], size(currdf, 1))
            currdf[!, "Params"] = repeat([params[i]], size(currdf, 1))
            
            if prod_df === nothing
                prod_df = currdf
            else 
                prod_df = [prod_df; currdf]
            end
        end
        prod_df[!, "repnum"] = repeat([repnum], size(prod_df, 1))
        error_dfs_prod = push!(error_dfs_prod, prod_df)

        # # CSV.write("results/multiview_prod_simulation_rep_$(repnum).csv", prod_df)

        min_df = nothing
        for i in 1:length(min_dp_errors)
            currdf_dp = DataFrame(min_dp_errors[i], colnames)
            currdf_dp[!, "Model"] = repeat(["DP"], size(currdf_dp, 1))
            currdf_dp[!, "J"] = Js
            currdf_ngg = DataFrame(min_ngg_errors[i], colnames)
            currdf_ngg[!, "Model"] = repeat(["NGG"], size(currdf_ngg, 1))
            currdf_ngg[!, "J"] = Js

            currdf = [currdf_dp; currdf_ngg]
            currdf[!, "DataGen"] = repeat(["py"], size(currdf, 1))
            currdf[!, "Params"] = repeat([params[i]], size(currdf, 1))

            if min_df === nothing
                min_df = currdf
            else 
                min_df = [min_df; currdf]
            end
        end
        min_df[!, "repnum"] = repeat([repnum], size(min_df, 1))
        error_dfs_min = push!(error_dfs_min, min_df)

        # CSV.write("results/multiview_min_simulation_rep_$(repnum).csv", min_df)

        cms_df = nothing
        for i in 1:length(cms_errors)
            currdf = DataFrame(cms_errors[i], colnames)
            currdf[!, "DataGen"] = repeat(["py"], size(cms_errors[i], 1))
            currdf[!, "Params"] = repeat([params[i]], size(cms_errors[i], 1))
            currdf[!, "Model"] = repeat(["CMS"], size(cms_errors[i], 1))
            currdf[!, "J"] = Js
            if cms_df === nothing
                cms_df = currdf
            else 
                cms_df = [cms_df; currdf]
            end
        end
        cms_df[!, "repnum"] = repeat([repnum], size(cms_df, 1))
        error_dfs_cms = push!(error_dfs_cms, cms_df)

        # CSV.write("results/multiview_cms_simulation_rep_$(repnum).csv", cms_df)

        debiased_df = nothing
        for i in 1:length(debiased_errors)
            currdf = DataFrame(debiased_errors[i], colnames)
            currdf[!, "DataGen"] = repeat(["py"], size(debiased_errors[i], 1))
            currdf[!, "Params"] = repeat([params[i]], size(debiased_errors[i], 1))
            currdf[!, "Model"] = repeat(["D-CMS"], size(debiased_errors[i], 1))
            currdf[!, "J"] = Js
            if debiased_df === nothing
                debiased_df = currdf
            else 
                debiased_df = [debiased_df; currdf]
            end
        end
        debiased_df[!, "repnum"] = repeat([repnum], size(debiased_df, 1))
        error_dfs_debiased = push!(error_dfs_debiased, debiased_df)

        # CSV.write("results/multiview_debiased_simulation_rep_$(repnum).csv", debiased_df)
    end

    final_df_prod = error_dfs_prod[1]
    for j in 2:NREP
        final_df_prod = [final_df_prod; error_dfs_prod[j]]
    end
    println("FINAL_DF_PROD: ", size(final_df_prod))

    CSV.write("results/multiview_prod_simulation_results.csv", final_df_prod)

    final_df_min = error_dfs_min[1]
    for j in 2:NREP
        final_df_min = [final_df_min; error_dfs_min[j]]
    end
    println("FINAL_DF_MIN: ", size(final_df_min))

    CSV.write("results/multiview_min_simulation_results.csv", final_df_min)

    final_df_cms = error_dfs_cms[1]
    for j in 2:NREP
        final_df_cms = [final_df_cms; error_dfs_cms[j]]
    end
    println("FINAL_DF_CMS: ", size(final_df_cms))

    CSV.write("results/multiview_cms_simulation_results.csv", final_df_cms)

    final_df_debiased = error_dfs_debiased[1]
    for j in 2:NREP
        final_df_debiased = [final_df_debiased; error_dfs_debiased[j]]
    end
    println("FINAL_DF_CMS: ", size(final_df_debiased))

    CSV.write("results/multiview_debiased_simulation_results.csv", final_df_debiased)
end 


main()