from hydra_utils import *
import argparse
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="physionet",
    choices=["physionet", "mimiciii", "eICU"]
)

parser.add_argument("--best_k", type=int, required=True)
parser.add_argument("--stat", type=str, default="weighted")

args = parser.parse_args()

dataset = args.dataset
best_k = args.best_k
stat = args.stat

# --------------------------------------------------
# Config
# --------------------------------------------------
TASK = "HOSPITAL_EXPIRE_FLAG"
exp_name = f"ALL_{TASK}"

distance = "euclidean"
mode = "split_only"

scaled = True
suffix = "_scaled_order" if scaled else ""

config = {
    "dataset": dataset,
    "exp_name": exp_name,
    "distance": distance,
    "mode": mode,
    "stat": stat,
    "suffix": suffix,
    "best_k": best_k
}

# --------------------------------------------------
# Data
# --------------------------------------------------
adms, features_df = load_features_target(config)
splits = load_splits(config)

full = features_df.join(splits)

sel_sub = splits[splits["balance_weight"] > 2]

# --------------------------------------------------
# Load importance table
# --------------------------------------------------
imp_table, means = combine_importances(config)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------
top = 10
methods = ["knn", "xgb", "permutation", "shap", "pairs"]

results = []

for method in methods:

    print(method)

    ranked_features = (
        imp_table
        .sort_values(by=method, ascending=False)
        .head(top)["feature"]
        .tolist()
    )

    res = run_subset_local_validation(
        sel_sub,
        full,
        ranked_features,
        best_k,
        min_feature_fraction=0.2
    )

    res["method"] = method
    res["top"] = top
    
    results.append(res)

results_df = pd.concat(results)

print(results_df.head())
print(compute_metrics_from_predictions(results_df))

# --------------------------------------------------
# Save
# --------------------------------------------------
outdir = Path(
    f"results/local_validation/{dataset}/{exp_name}/{stat}/{best_k}"
)

outdir.mkdir(parents=True, exist_ok=True)

outfile = outdir / "boundary_subset_performance.csv"

results_df.to_csv(outfile, index=False)

print(f"Saved results to {outfile}")