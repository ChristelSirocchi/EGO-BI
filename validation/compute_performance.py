import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold

from hydra_utils import *
from hydra_method import *

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])

parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")

parser.add_argument("--prune_steps", type=int, default=5)
args = parser.parse_args()

SEED = 42
dataset = args.dataset
TASK = args.task
prune_steps = args.prune_steps

DEPT = "ALL"
exp_name = f"{DEPT}_{TASK}"


adms = pd.read_csv(f"{dataset}/sub_adm_target.csv", index_col=0)
task_dict = adms[["HADM_ID", TASK]].set_index("HADM_ID")[TASK].to_dict()

features_df = pd.read_csv(f"{dataset}/ts_features_out.csv", index_col=0, header=[0, 1])
features_df = features_df.loc[adms["HADM_ID"].values]

feature_names = [f"{c[0]}__{c[1]}" if isinstance(c, tuple) else str(c) for c in features_df.columns]
y_all = np.array([task_dict[i] for i in features_df.index])
ids = np.array(features_df.index)
X_all = features_df.values

outdir = Path(f"results/progressive_pruning_oof/{dataset}/{exp_name}/{prune_steps}")
outdir.mkdir(parents=True, exist_ok=True)

# Shared K-Fold setup to keep splits aligned
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


print("--- PHASE 1: Generating OOF Confidence Scores ---")
oof_probs = np.zeros(len(y_all))

for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y_all)):
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te = X_all[test_idx]
    
    scale_pos_weight = compute_scale_pos_weight(y_tr)
    model = get_xgboost_model(X_train=X_tr, scale_pos_weight=scale_pos_weight, constrained=False, seed=SEED)
    model.fit(X_tr, y_tr)
    
    # Save test predictions into global OOF array
    oof_probs[test_idx] = model.predict_proba(X_te)[:, 1]

# Compute master unbiased ranking
oof_preds = (oof_probs > 0.5).astype(int)
is_correct = (oof_preds == y_all)
confidence = 1.0 - np.abs(y_all - oof_probs)

# Score formula: Correct samples get positive confidence; incorrect get penalized heavily
master_scores = np.where(is_correct, confidence, -1.0 - (1.0 - confidence))

# --------------------------------------------------
# Save sample difficulty ranking
# --------------------------------------------------

difficulty_df = pd.DataFrame({
    "HADM_ID": ids,
    "y_true": y_all,
    "y_prob": oof_probs,
    "y_pred": oof_preds,
    "is_correct": is_correct,
    "confidence": confidence,
    "difficulty_score": master_scores
})

# easiest -> hardest
difficulty_df = difficulty_df.sort_values(
    "difficulty_score",
    ascending=False
).reset_index(drop=True)

difficulty_df.to_csv(
    outdir / "sample_difficulty.csv",
    index=False
)
# --------------------------------------------------
# Performance summary
# --------------------------------------------------

n_total = len(y_all)
n_wrong = np.sum(~is_correct)
pct_wrong = 100 * n_wrong / n_total

print(f"\nTotal samples: {n_total}")
print(f"Wrong predictions: {n_wrong}")
print(f"Percentage wrong: {pct_wrong:.2f}%")

print(f"\nCompleted! Saved matrix to: {outdir}/sample_difficulty.csv")
print(difficulty_df.head())