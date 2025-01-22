import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from itertools import product
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices

SEED = 20230810


Js = [50, 100, 500, 1000]
max_mem = 1000

PY_ALPHAS = [0.25]
PY_THETAS = [10.0, 100.0]
NDATA = 250000
NTRAIN = 25000
NTEST = 2500
NREP = 50

NJOBS = 32


# NDATA = 2500
# NTRAIN = 250
# NTEST = 500
# NJOBS = 4

np.random.seed(SEED)
seeds = np.random.randint(10000, 1000000, size=NREP)


def run_one(py_theta, py_alpha, method, model, J, rule, repnum):
    import os, sys
    sys.path.append("..")

    from cms.data import PYP
    from cms.cms import CMS, BayesianCMS
    from cms.conformal import ConformalCMS

    from .experiment_utils import process_results


    M = int(max_mem / J)
    rep_seed = seeds[repnum]
    stream = PYP(py_theta, py_alpha, rep_seed)
    cms = CMS(M, J, seed=rep_seed, conservative=False)
    method_unique = 0
    n_bins = 1
    n_track = NTRAIN
    sketch_name = "cms"
    if method == "conformal":
        worker = ConformalCMS(stream, cms,
                            n_track = NTRAIN,
                            unique = 0,
                            n_bins = 5,
                            scorer_type = "Bayesian-" + model, agg_rule=rule)
        method_name = method + "_" + rule

    else:
        worker = BayesianCMS(stream, cms, model=model, agg_rule=rule)
        method_name = method + "_" + rule

    
    results = worker.run(NDATA, NTEST, seed=rep_seed)
    outfile_prefix = \
        sketch_name + "_" + "PYP_" + str(py_theta) + "_" + str(py_alpha) + "_d" + str(M) + "_w" + str(J) + "_n" + str(NDATA) + "_repnum" + str(repnum)
    process_results(results, outfile_prefix, method_name, model, sketch_name, "PYP", M, J, 
                    method, False, "mcmc", n_bins, n_track, NDATA, rep_seed, 0.9, False)


def run_chunk(params):
    for p in params:
        theta, alpha, method, model, rule, j = p
        print("Running PYP({0}, {1}), J: {2}, Method: {3}, Model: {4}, Rule: {5}, REP: {6}".format(
                                    theta, alpha, args.J, method, model, rule, j))
        run_one(theta, alpha, method, model, args.J, rule, j)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None, choices=["conformal", "bayes", None])
    parser.add_argument("--model", type=str, default=None, choices=["NGG", "DP", None])
    parser.add_argument("--rule", type=str, default=None, choices=["PoE", "min", None])
    parser.add_argument("--J", type=int, default=100)
    
    args = parser.parse_args()

    methods = [args.method] if args.method else ["conformal", "bayes"]
    models = [args.model] if args.model else ["NGG", "DP"]
    rules = [args.rule] if args.rule else ["PoE", "min"]

    all_params = np.array(list(product(
        PY_THETAS, PY_ALPHAS, methods, models, rules, list(range(NREP)))), dtype=object)

    Parallel(n_jobs=NJOBS)(
        delayed(run_chunk)(all_params[s])
        for s in gen_even_slices(len(all_params), effective_n_jobs(NJOBS)))
