import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from hydra_utils import *
from hydra_method import *

# ---------------------------------------------------------------------
# Performance-based hardness: out-of-fold (OOF) prediction confidence
# from an unconstrained XGBoost model. A sample is "hard" if it is
# mispredicted, or predicted correctly with low confidence, when held
# out of training.
#
# See compute_hardness_disjunct.py for the disjunct-based alternative;
# prune.py consumes whichever one is selected via --criterion.
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])

parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")

args = parser.parse_args()

SEED = 42
dataset = args.dataset
TASK = args.task

DEPT = "ALL"
exp_name = f"{DEPT}_{TASK}"

print("Experiment:", exp_name)

adms = pd.read_csv(f"{dataset}/sub_adm_target.csv", index_col=0)
task_dict = adms[["HADM_ID", TASK]].set_index("HADM_ID")[TASK].to_dict()

features_df = pd.read_csv(f"{dataset}/ts_features_out.csv", index_col=0, header=[0, 1])
features_df = features_df.loc[adms["HADM_ID"].values]

y_all = np.array([task_dict[i] for i in features_df.index])
ids = np.array(features_df.index)
X_all = features_df.values

outdir = Path(f"results/hardness/{dataset}/{exp_name}")
outdir.mkdir(parents=True, exist_ok=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("--- Generating OOF confidence scores ---")
oof_probs = np.zeros(len(y_all))

for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y_all)):
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te = X_all[test_idx]

    scale_pos_weight = compute_scale_pos_weight(y_tr)
    model = get_xgboost_model(X_train=X_tr, scale_pos_weight=scale_pos_weight, constrained=False, seed=SEED)
    # n_jobs=1: hydra_method's get_xgboost_model doesn't expose n_jobs, and letting
    # XGBoost spawn its default multi-threaded pool across many repeated small fits
    # has been observed to deadlock in CPU-constrained/sandboxed environments.
    model.set_params(n_jobs=1)
    model.fit(X_tr, y_tr)

    # Save test predictions into global OOF array
    oof_probs[test_idx] = model.predict_proba(X_te)[:, 1]

# Compute master unbiased ranking
oof_preds = (oof_probs > 0.5).astype(int)
is_correct = (oof_preds == y_all)
confidence = 1.0 - np.abs(y_all - oof_probs)

# Score formula: correct samples get positive confidence; incorrect are penalized heavily.
# Oriented so that HIGH score = EASY, matching prune.py's convention.
difficulty_score = np.where(is_correct, confidence, -1.0 - (1.0 - confidence))

# --------------------------------------------------
# Save out-of-fold hardness measures
# --------------------------------------------------

hardness_df = pd.DataFrame({
    "HADM_ID": ids,
    "label": y_all,
    "y_prob": oof_probs,
    "y_pred": oof_preds,
    "is_correct": is_correct,
    "confidence": confidence,
    "difficulty_score": difficulty_score,
}).set_index("HADM_ID")

assert hardness_df.index.is_unique, \
    "Each HADM_ID should appear exactly once across concatenated test folds"
assert len(hardness_df) == len(ids), \
    "Concatenated out-of-fold results should cover the full cohort"

out_path = outdir / "performance_hardness_oof.csv"
hardness_df.to_csv(out_path)

n_total = len(y_all)
n_wrong = np.sum(~is_correct)
pct_wrong = 100 * n_wrong / n_total

print(f"\nTotal samples: {n_total}")
print(f"Wrong predictions: {n_wrong} ({pct_wrong:.2f}%)")
print(f"\nSaved out-of-fold hardness measures to {out_path}")
print(hardness_df[["difficulty_score"]].describe())
