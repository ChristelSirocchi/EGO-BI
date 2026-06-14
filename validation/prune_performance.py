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

parser.add_argument("--prune_steps", type=int, default=10)
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

# Create a look-up dictionary mapping Sample ID -> Unbiased Difficulty Score
id_to_score = dict(zip(ids, master_scores))


# =========================================================
# PHASE 2: PROGRESSIVE PRUNING PER FOLD
# =========================================================
print("\n--- PHASE 2: Progressive Pruning Loop ---")
progressive_rows = []

for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y_all)):
    print(f"\nFold {fold + 1}")
    
    train_ids = ids[train_idx]
    X_train_fold = features_df.iloc[train_idx]
    y_train_fold = y_all[train_idx]
    
    # Gather the difficulty scores *specifically* for this fold's training samples
    fold_train_scores = np.array([id_to_score[sid] for sid in train_ids])
    
    # Sort training samples from easiest (highest score) to hardest (lowest score)
    sorted_indices = np.argsort(fold_train_scores)[::-1]
    total_samples = len(sorted_indices)
    
    for step in range(prune_steps):
        keep_fraction = 1.0 - (step / prune_steps)
        cutoff_idx = int(total_samples * keep_fraction)
        
        # Drop the easiest samples first (slicing out the front of the sorted array)
        drop_count = total_samples - cutoff_idx
        retained_indices = sorted_indices[drop_count:]
        
        if len(retained_indices) < 10 or len(np.unique(y_train_fold[retained_indices])) < 2:
            break
            
        X_train_subset = X_train_fold.iloc[retained_indices]
        y_train_subset = y_train_fold[retained_indices]
        
        print(f"  Step {step + 1} ({keep_fraction*100:.0f}% Kept): {len(y_train_subset)} samples remaining.")
        
        sub_scale_weight = compute_scale_pos_weight(y_train_subset)
        sub_model = get_xgboost_model(X_train=X_train_subset, scale_pos_weight=sub_scale_weight, constrained=False, seed=SEED)
        sub_model.fit(X_train_subset, y_train_subset)
        
        importances = sub_model.feature_importances_
        for feat, val in zip(feature_names, importances):
            progressive_rows.append({
                "fold": fold + 1,
                "step": step + 1,
                "percent_data_kept": int(keep_fraction * 100),
                "feature": feat,
                "importance": float(val)
            })

# =========================================================
# SAVE RESULT TABLES
# =========================================================
df_results = pd.DataFrame(progressive_rows)
df_pivot = df_results.pivot_table(index="feature", columns="percent_data_kept", values="importance", aggfunc="mean")
df_pivot = df_pivot[sorted(df_pivot.columns, reverse=True)]
df_pivot.to_csv(outdir / "summary_importance_shifts.csv")

print(f"\nCompleted! Saved matrix to: {outdir}/summary_importance_shifts.csv")