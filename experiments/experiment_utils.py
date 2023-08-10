
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import os, sys
sys.path.append("..")

from cms.diagnostics import evaluate_marginal, evaluate_conditional


def process_results(
        results, outfile_prefix, method_name, sketch_name, stream_name, d, w, method, 
        method_unique, posterior, n_bins, n_track, n, seed, confidence, two_sided):
    
    def add_header(df):
        df["sketch"] = sketch_name
        df["data"] = stream_name
        df["d"] = d
        df["w"] = w
        df["method"] = method
        df["method-unique"] = method_unique
        df["posterior"] = posterior
        df["n_bins"] = n_bins
        df["n_track"] = n_track
        df["n"] = n
        df["seed"] = seed
        df["confidence"] = confidence
        df["two_sided"] = two_sided
        return df
    
    confidence_str = str(confidence)

    ################
    # Save results #
    ################
    outfile = "results/" + stream_name + "/detailed/" + outfile_prefix + "_" + method_name + "_" + confidence_str + "_ts" + str(int(two_sided)) + ".txt"
    dir = os.path.dirname(outfile)
    os.makedirs(dir, exist_ok=True)
    add_header(results).to_csv(outfile, index=False)
    print("\nDetailed results written to {:s}\n".format(outfile))
    sys.stdout.flush()

    ###############
    # Diagnostics #
    ###############

    s1 = evaluate_marginal(results, include_seen=True)
    s2 = evaluate_marginal(results, include_seen=False)
    s1u = evaluate_marginal(results, include_seen=True, unique=True)
    s2u = evaluate_marginal(results, include_seen=False, unique=True)
    summary_marginal = pd.concat([s1, s2, s1u, s2u])
    print("Marginal summary:")
    print(summary_marginal)
    print()
    sys.stdout.flush()

    outfile = "results/" + stream_name + "/marginal/" + outfile_prefix + "_" + method_name + "_" + confidence_str + "_ts" + str(int(two_sided)) + ".txt"
    dir = os.path.dirname(outfile)
    os.makedirs(dir, exist_ok=True)
    add_header(summary_marginal).to_csv(outfile, index=False)
    print("\nMarginal summary written to {:s}\n".format(outfile))
    sys.stdout.flush()

    s1 = evaluate_conditional(results, nbins=5, include_seen=True)
    s2 = evaluate_conditional(results, nbins=5, include_seen=False)
    s1u = evaluate_conditional(results, nbins=5, include_seen=True, unique=True)
    s2u = evaluate_conditional(results, nbins=5, include_seen=False, unique=True)
    summary_conditional = pd.concat([s1, s2, s1u, s2u])
    print("Conditional summary:")
    print(summary_conditional)
    print()
    sys.stdout.flush()

    outfile = "results/" + stream_name + "/conditional/" + outfile_prefix + "_" + method_name + "_" + confidence_str + "_ts" + str(int(two_sided)) + ".txt"
    dir = os.path.dirname(outfile)
    os.makedirs(dir, exist_ok=True)
    add_header(summary_conditional).to_csv(outfile, index=False)
    print("\nConditional summary written to {:s}\n".format(outfile))
    sys.stdout.flush()
    return 