import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

from itertools import product
from joblib import Parallel, delayed


SEED = 20230810


Js = [50, 100, 500, 1000]
max_mem = 1000

PY_ALPHAS = [0.25]
PY_THETAS = [10.0, 100.0]
NDATA = 250000
NTRAIN = 25000
NTEST = 2500
NREP = 10

NJOBS = 16


# NDATA = 2500
# NTRAIN = 250
# NTEST = 500
# NJOBS = 4

np.random.seed(SEED)

seeds = np.random.randint(100000, 1000000, size=NREP)


def run_one(py_theta, py_alpha, model, J, repnum):
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

    # conformal PoE
    print("Running PYP({0}, {1}), J: {2}, Method: {3}, Model: {4}, Rule: {5}, REP: {6}".format(
            py_theta, py_alpha, J, "conformal", model, "PoE", repnum))
    method = "conformal"
    rule = "PoE"
    conf_worker = ConformalCMS(
        stream, cms, n_track = NTRAIN,
        unique = 0, n_bins = 5, scorer_type = "Bayesian-" + model, 
        agg_rule="PoE")
    method_name = method + "_" + rule
    results = conf_worker.run(NDATA, NTEST, seed=rep_seed)
    outfile_prefix = sketch_name + "_" + "PYP_" + str(py_theta) + "_" + str(py_alpha) + "_d" + str(M) + "_w" + str(J) + "_n" + str(NDATA) + "_repnum" + str(repnum)
    process_results(results, outfile_prefix, method_name, model, sketch_name, "PYP", M, J, 
                    method, False, "mcmc", n_bins, n_track, NDATA, rep_seed, 0.9, False)

    # Bayes PoE
    print("Running PYP({0}, {1}), J: {2}, Method: {3}, Model: {4}, Rule: {5}, REP: {6}".format(
            py_theta, py_alpha, J, "bayes", model, "PoE", repnum))
    method = "bayes"
    rule = "PoE"
    nggpoefit = conf_worker.model
    bayes_worker = BayesianCMS(stream, cms, model=model, agg_rule="poe")
    method_name = method + "_" + rule
    bayes_worker.run(NDATA, NTEST, seed=rep_seed, fitted_model=nggpoefit)
    outfile_prefix = sketch_name + "_" + "PYP_" + str(py_theta) + "_" + str(py_alpha) + "_d" + str(M) + "_w" + str(J) + "_n" + str(NDATA) + "_repnum" + str(repnum)
    process_results(results, outfile_prefix, method_name, model, sketch_name, "PYP", M, J, 
                    method, False, "mcmc", n_bins, n_track, NDATA, rep_seed, 0.9, False)

    # conformal min
    print("Running PYP({0}, {1}), J: {2}, Method: {3}, Model: {4}, Rule: {5}, REP: {6}".format(
            py_theta, py_alpha, J, "conformal", model, "min", repnum))
    conf_worker.change_rule("min")
    method = "conformal"
    rule = "min"
    method_name = method + "_" + rule
    results = conf_worker.run(NDATA, NTEST, seed=rep_seed, 
                         reuse_stream=True, reuse_model=True)
    outfile_prefix = sketch_name + "_" + "PYP_" + str(py_theta) + "_" + str(py_alpha) + "_d" + str(M) + "_w" + str(J) + "_n" + str(NDATA) + "_repnum" + str(repnum)
    process_results(results, outfile_prefix, method_name, model, sketch_name, "PYP", M, J, 
                    method, False, "mcmc", n_bins, n_track, NDATA, rep_seed, 0.9, False)
    
    # Bayes min
    print("Running PYP({0}, {1}), J: {2}, Method: {3}, Model: {4}, Rule: {5}, REP: {6}".format(
            py_theta, py_alpha, J, "bayes", model, "min", repnum))
    method = "bayes"
    rule = "min"
    nggminfit = conf_worker.model
    bayes_worker = BayesianCMS(stream, cms, model=model, agg_rule="min")
    method_name = method + "_" + rule
    bayes_worker.run(NDATA, NTEST, seed=rep_seed, fitted_model=nggminfit)
    outfile_prefix = sketch_name + "_" + "PYP_" + str(py_theta) + "_" + str(py_alpha) + "_d" + str(M) + "_w" + str(J) + "_n" + str(NDATA) + "_repnum" + str(repnum)
    process_results(results, outfile_prefix, method_name, model, sketch_name, "PYP", M, J, 
                    method, False, "mcmc", n_bins, n_track, NDATA, SEED, 0.9, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None, choices=["conformal", "bayes", None])
    parser.add_argument("--model", type=str, default=None, choices=["NGG", "DP", None])
    parser.add_argument("--rule", type=str, default=None, choices=["PoE", "min", None])
    parser.add_argument("--J", type=int, default=100)
    
    args = parser.parse_args()

    models = [args.model] if args.model else ["DP", "NGG"]

    for theta in PY_THETAS:
        for alpha in PY_ALPHAS:
            for model in models:
                for j in range(NREP):
                    run_one(theta, alpha, model, args.J, j)
