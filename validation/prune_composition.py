import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

from hydra_utils import *
from hydra_method import *

# =========================================================
# CONFIG
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])

parser.add_argument("--k", type=int, default=5)
parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")
args = parser.parse_args()

SEED = 42

dataset = args.dataset
k = args.k

fractions = np.linspace(0.1, 1, k)

#DEPT = "EMERGENCY" if "mimiciii" in dataset else "ALL"
DEPT = "ALL"
TASK = args.task
exp_name = f"{DEPT}_{TASK}"

print("Experiment:", exp_name)

# =========================================================
# DATA
# =========================================================

adms = pd.read_csv(
    f"{dataset}/sub_adm_target.csv",
    index_col=0
)

task_dict = (
    adms[["HADM_ID", TASK]]
    .set_index("HADM_ID")[TASK]
    .to_dict()
)

features_df = pd.read_csv(
    f"{dataset}/{exp_name}/ts_features_out.csv",
    index_col=0,
    header=[0, 1]
)

features_df = features_df.loc[adms["HADM_ID"].values]

feature_names = features_df.columns.tolist()

y_all = np.array([task_dict[i] for i in features_df.index])
ids = np.array(features_df.index)

# fast lookup map (IMPORTANT FIX)
id_to_idx = {id_: i for i, id_ in enumerate(ids)}

# =========================================================
# PATHS
# =========================================================

fold_dir = Path(f"results/splits/{dataset}/{exp_name}/{k}")

out_dir = Path(f"results/importances/{dataset}/{exp_name}")
out_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# STORE
# =========================================================

all_importances = []

for fold in range(5):
    print(fold)
    train_stats = pd.read_csv(fold_dir / f"fold_{fold}_train.csv")

    train_ids = train_stats["id"].values
    train_idx = np.array([id_to_idx[i] for i in train_ids])

    X_train_full = features_df.iloc[train_idx].values
    y_train = y_all[train_idx]

    is_mixed = train_stats["is_mixed"].values

    homo_idx = np.where(is_mixed == 0)[0]
    hetero_idx = np.where(is_mixed == 1)[0]

    feature_names = features_df.columns

    for seed in range(3):
        print(seed)
        rng = np.random.RandomState(seed)

        for f in fractions:
            f = round(f, 2)
            print(f)
            n_remove = int(f * len(homo_idx))

            keep_homo = rng.choice(
                homo_idx,
                size=len(homo_idx) - n_remove,
                replace=False
            )

            train_keep_idx = np.concatenate([keep_homo, hetero_idx])

            X_train_sel = X_train_full[train_keep_idx]
            y_train_sel = y_train[train_keep_idx]

            scale_pos_weight = compute_scale_pos_weight(y_train_sel)

            model = get_xgboost_model(
                X_train=X_train_sel, 
                scale_pos_weight=scale_pos_weight, 
                constrained=False,
                seed=SEED
            )

            model.fit(X_train_sel, y_train_sel)

            # ✔ STORE FULL VECTOR WITH FEATURE NAMES
            all_importances.append(
                pd.DataFrame({
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                    "fold": fold,
                    "seed": seed,
                    "fraction": f
                })
            )

# =========================================================
# FINAL TABLE
# =========================================================

df = pd.concat(all_importances)

summary = df.groupby(["feature", "fraction"])["importance"].mean().reset_index()

table = summary.pivot(index="feature", columns="fraction", values="importance")

out_path = Path(f"results/importances/{dataset}/{exp_name}")
out_path.mkdir(parents=True, exist_ok=True)

table.to_csv(out_path / f"knn_{k}_fraction_seed_averaged_table_3.csv")