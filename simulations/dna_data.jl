include("../julia_src/jsketch.jl")

using Distributed
using IterTools
using DataFrames
using CSV
using Serialization
using StatsBase
using Random


Js = [1000, 2500, 5000, 10000]
MAXMEM = 10000
NDATA = [25000, 100000, 250000, 500000, 1000000]
MODELS = ["DP", "NGG"]

Random.seed!(20230719)


function get_freq_bin(f)
    if f == 0
        return 1
    end

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


function frequency_simulation(data, test_data)
    ngg_p = Sketch.fit_ngg(data[1:Int(length(data) * 0.05)])
    
    uniq2cnt_train = countmap(data)
    uniq2cnt_test = countmap(test_data)

    n_mae_buckets = get_freq_bin(1e12)
    maes_dp_prod = zeros(length(Js), n_mae_buckets)
    maes_dp_min = zeros(length(Js), n_mae_buckets)
    maes_ngg_prod = zeros(length(Js), n_mae_buckets)
    maes_ngg_min = zeros(length(Js), n_mae_buckets)
    maes_cms = zeros(length(Js), n_mae_buckets)

    ndata_by_bucket = zeros(n_mae_buckets)
    for (k, cnt) in uniq2cnt_test
        mae_bucket = get_freq_bin(cnt)
        ndata_by_bucket[mae_bucket] += 1
    end


    for (i, J) in enumerate(Js)
        println("Frequency, J: ", J)
        m = Int(MAXMEM / J)
        hash_functions = [Sketch.generate_hash(J, rand(10000:100000000)) for _ in 1:m]
        hash_data = Sketch.hash_dataset(data, hash_functions, J)
        dp_p = params = Sketch.fit_multiview(hash_data, data[1:10000], "DP")
        ngg_intcache = Sketch.beta_integral_ngg(params=ngg_p, J=J)
        for (k, test_cnt) in uniq2cnt_test
            hs = Int.([h(k) for h in hash_functions])

            cnt = get(uniq2cnt_train, k, 0)
            # println("CNT: ", cnt)
            mae_bucket = get_freq_bin(cnt)
            
            c_js = Int.(hash_data[CartesianIndex.(collect(1:m), hs)])
            min_c = Int(minimum(c_js))
            cms_est = min_c
            maes_cms[i, mae_bucket] += test_cnt * abs(cnt - cms_est) / ndata_by_bucket[mae_bucket]
            
            dp_logprobas = Sketch.freq_post.(min_c, c_js, dp_p, J, true)
            ngg_logprobas = [
                Sketch.freq_post!(min_c, c, ngg_p, J, true, ngg_intcache) for c in c_js]

            maes_dp_prod[i, mae_bucket] += test_cnt * abs(
                cnt - Sketch.PoE_mean(dp_logprobas)) / ndata_by_bucket[mae_bucket]
            maes_dp_min[i, mae_bucket] += test_cnt * abs(
                cnt - Sketch.MIN_mean(dp_logprobas)) / ndata_by_bucket[mae_bucket]

            maes_ngg_prod[i, mae_bucket] += test_cnt * abs(
                cnt - Sketch.PoE_mean(ngg_logprobas)) / ndata_by_bucket[mae_bucket]
            maes_ngg_min[i, mae_bucket] += test_cnt * abs(
                cnt - Sketch.MIN_mean(ngg_logprobas)) / ndata_by_bucket[mae_bucket]
        end
    end

    bins = [0, 1, 4, 16, 64, 256, "Inf"]
    bins = collect(zip(bins[1:end-1], bins[2:end]))
    colnames = ["($(x), $(y)]" for (x, y) in bins ]
    
    prod_df_dp = DataFrame(maes_dp_prod, colnames)
    prod_df_dp[!, "Model"] = repeat(["DP"], size(prod_df_dp, 1))
    prod_df_dp[!, "J"] = Js

    prod_df_ngg = DataFrame(maes_ngg_prod, colnames)
    prod_df_ngg[!, "Model"] = repeat(["NGG"], size(prod_df_ngg, 1))
    prod_df_ngg[!, "J"] = Js

    prod_df = [prod_df_dp; prod_df_ngg]
    prod_df[!, "Rule"] = repeat(["PoE"], size(prod_df, 1))

    min_df_dp = DataFrame(maes_dp_min, colnames)
    min_df_dp[!, "Model"] = repeat(["DP"], size(min_df_dp, 1))
    min_df_dp[!, "J"] = Js

    min_df_ngg = DataFrame(maes_ngg_min, colnames)
    min_df_ngg[!, "Model"] = repeat(["NGG"], size(min_df_ngg, 1))
    min_df_ngg[!, "J"] = Js

    min_df = [min_df_dp; min_df_ngg]
    min_df[!, "Rule"] = repeat(["MIN"], size(min_df, 1))

    cms_df = DataFrame(maes_cms, colnames)
    cms_df[!, "Model"] = repeat(["CMS"], size(cms_df, 1))
    cms_df[!, "J"] = Js
    cms_df[!, "Rule"] = repeat(["CMS"], size(cms_df, 1))

    maes_df = [prod_df; min_df; cms_df]
    CSV.write("results/dna_maes.csv", maes_df)
end

function cardinality_simulation(true_dataset)
    k_estim = zeros((2, length(NDATA)))
    k_true = zeros(length(NDATA))
    J = 10000
    for (i, n) in enumerate(NDATA)
        println("Cardinality, N: ", n)
        data = true_dataset[1:n]
        train_data = data[1:Int(n * 0.5)]
        hf = Sketch.generate_hash(J, rand(10000:100000000))
        sketch = Sketch.hash_dataset(data, [hf], J)[1, :]

        k_true[i] = length(unique(data))
        params = []
        for (l, m) in enumerate(MODELS)
            p = estimate_params(m, sketch, train_data)
            println("PARAMS: ", p)
            params = push!(params, p)
            k_estim[l, i] = Sketch.card_est(p, sketch)
        end
    end

    df = DataFrame(k_estim', MODELS)
    df[!, "ndata"] = NDATA
    df[!, "true_k"] = k_true
    CSV.write("results/dna_cardinality.csv", df)
end


function main()
    dna_seqs = CSV.read("data/dna_idx.csv", DataFrame)[:, 1]
    train = dna_seqs[1:1000000]
    test = dna_seqs[1000000:2000000]

    frequency_simulation(train, test)
    # cardinality_simulation(train)
end

main()