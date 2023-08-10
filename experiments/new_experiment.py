import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

from itertools import product
from joblib import Parallel, delayed


SEED = 20230810


Js = [50, 100, 500, 1000]
max_mem = 1000

PY_ALPHAS = [0.25,0.75]
PY_THETAS = [10.0, 100.0, 1000.0]
NDATA = 250000
NTRAIN = 25000
NTEST = 50000
NJOBS = 16


# NDATA = 2500
# NTRAIN = 250
# NTEST = 500
# NJOBS = 4


def run_one(py_theta, py_alpha, method, model, J):
    import os, sys
    sys.path.append("..")

    from cms.data import PYP
    from cms.cms import CMS, BayesianCMS
    from cms.conformal import ConformalCMS

    from .experiment_utils import process_results


    M = int(max_mem / J)
    stream = PYP(py_theta, py_alpha, SEED)
    cms = CMS(M, J, seed=SEED, conservative=False)
    method_unique = 0
    n_bins = 1
    n_track = NTRAIN
    sketch_name = "cms"
    if method == "conformal":
        worker = ConformalCMS(stream, cms,
                            n_track = NTRAIN,
                            unique = 0,
                            n_bins = 1,
                            scorer_type = "Bayesian-" + model)
        method_name = method + "_unique" + str(int(method_unique)) + "_bins" + str(n_bins) + "_track" + str(n_track)

    else:
        worker = BayesianCMS(stream, cms, model=model)
        method_name = method + "_unique" + str(int(method_unique)) + "_bins" + str(n_bins) + "_track" + str(n_track)

    
    results = worker.run(NDATA, NTEST, seed=SEED)
    outfile_prefix = "mario_" + sketch_name + "_" + "PYP" + "_d" + str(M) + "_w" + str(J) + "_n" + str(NDATA) + "_s" + str(SEED)
    process_results(results, outfile_prefix, method_name, sketch_name, "PYP", M, J, 
                    method, False, "mcmc", n_bins, n_track, NDATA, SEED, 0.9, False)
    

if __name__ == "__main__":
    params = list(product(
        PY_THETAS, PY_ALPHAS, ["conformal", "bayes"], ["DP", "NGG"], Js))
    
    
    Parallel(n_jobs=NJOBS)(delayed(run_one)(*p) for p in params)

    

    

