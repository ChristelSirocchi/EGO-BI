import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from hydra_utils import *
from hydra_method import *

# ---------------------------------------------------------------------
# Progressive pruning: given a per-patient hardness score (computed
# ahead of time by compute_hardness_performance.py or
# compute_hardness_disjunct.py), progressively drop the easiest
# training samples within each fold, retrain, and track how feature
# importance shifts as only harder cases remain.
#
# The pruning loop itself does not care which criterion produced the
# score -- only that higher = easier. --criterion selects which
# precomputed hardness file to load and, for disjunct, which column
# and orientation to use.
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])

parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")

parser.add_argument("--criterion", type=str, default="performance",
                    choices=["performance", "disjunct"],
                    help="Which precomputed hardness score to prune by. 'performance' reads "
                         "results/hardness/{dataset}/{exp_name}/performance_hardness_oof.csv "
                         "(from compute_hardness_performance.py); 'disjunct' reads "
                         "disjunct_hardness_oof.csv (from compute_hardness_disjunct.py).")

parser.add_argument("--disjunct-measure", type=str, default="DS",
                    choices=["DS", "TD"],
                    help="Only used when --criterion disjunct. DS = disjunct size (high = easy), "
                         "TD = disjunct/leaf depth (high = hard, so it's inverted internally).")

parser.add_argument("--prune_steps", type=int, default=10)

args = parser.parse_args()

SEED = 42
dataset = args.dataset
TASK = args.task
criterion = args.criterion
disjunct_measure = args.disjunct_measure
prune_steps = args.prune_steps

DEPT = "ALL"
exp_name = f"{DEPT}_{TASK}"

print("Experiment:", exp_name)
print("Criterion:", criterion if criterion == "performance" else f"disjunct ({disjunct_measure})")

# =========================================================
# LOAD PRECOMPUTED HARDNESS SCORES
# =========================================================

hardness_dir = Path(f"results/hardness/{dataset}/{exp_name}")

if criterion == "performance":
    hardness_path = hardness_dir / "performance_hardness_oof.csv"
    score_col = "difficulty_score"
    sign = 1.0
    criterion_label = "performance"
else:
    hardness_path = hardness_dir / "disjunct_hardness_oof.csv"
    score_col = disjunct_measure
    # DS: high = large, homogeneous disjunct = easy -> keep as is.
    # TD: high = deep, idiosyncratic leaf = hard -> invert so high = easy.
    sign = 1.0 if disjunct_measure == "DS" else -1.0
    criterion_label = f"disjunct_{disjunct_measure}"

if not hardness_path.exists():
    raise FileNotFoundError(
        f"{hardness_path} not found. Run "
        f"{'compute_hardness_performance.py' if criterion == 'performance' else 'compute_hardness_disjunct.py'} "
        f"for --dataset {dataset} --task {TASK} first."
    )

print(f"--- Loading precomputed hardness scores from {hardness_path} ---")
hardness_df = pd.read_csv(hardness_path, index_col="HADM_ID")

if score_col not in hardness_df.columns:
    raise ValueError(f"'{score_col}' not in {hardness_path} columns: {list(hardness_df.columns)}")

# =========================================================
# DATA
# =========================================================

adms = pd.read_csv(f"{dataset}/sub_adm_target.csv", index_col=0)
task_dict = adms[["HADM_ID", TASK]].set_index("HADM_ID")[TASK].to_dict()

features_df = pd.read_csv(f"{dataset}/ts_features_out.csv", index_col=0, header=[0, 1])
features_df = features_df.loc[adms["HADM_ID"].values]

feature_names = [f"{c[0]}__{c[1]}" if isinstance(c, tuple) else str(c) for c in features_df.columns]
y_all = np.array([task_dict[i] for i in features_df.index])
ids = np.array(features_df.index)

missing = set(ids) - set(hardness_df.index)
if missing:
    raise ValueError(
        f"{len(missing)} sample IDs from features_df not found in {hardness_path}, "
        f"e.g. {list(missing)[:5]}"
    )

id_to_score = (sign * hardness_df[score_col]).to_dict()

outdir = Path(f"results/progressive_pruning/{dataset}/{exp_name}/{criterion_label}/{prune_steps}")
outdir.mkdir(parents=True, exist_ok=True)

# Shared K-Fold setup; independent of whichever split produced the OOF
# hardness scores above, since those cover every sample regardless of fold.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# =========================================================
# PROGRESSIVE PRUNING PER FOLD
# =========================================================
print("\n--- Progressive Pruning Loop ---")
progressive_rows = []

for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y_all)):
    print(f"\nFold {fold + 1}")

    train_ids = ids[train_idx]
    X_train_fold = features_df.iloc[train_idx]
    y_train_fold = y_all[train_idx]

    # Gather the hardness scores *specifically* for this fold's training samples
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
        # n_jobs=1: see the matching comment in compute_hardness_performance.py --
        # repeated small XGBoost fits with default multi-threading have been
        # observed to deadlock in CPU-constrained/sandboxed environments.
        sub_model.set_params(n_jobs=1)
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
