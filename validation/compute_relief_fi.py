import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from skrebate import ReliefF, MultiSURF
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
from model import build_importance_table

# =========================================================
# CONFIG
# =========================================================

SEED = 42

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])

parser.add_argument("--base-path", type=str, default=".",
                    help="Directory containing the {dataset}/ input folder. "
                         "Defaults to the current working directory.")

parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")

parser.add_argument("--algorithm", type=str, default="multisurf",
                    choices=["multisurf", "relieff"],
                    help="multisurf sets its neighbourhood radius automatically per instance "
                         "(no neighbour-count hyperparameter); relieff requires --n-neighbors.")

parser.add_argument("--n-neighbors", type=int, default=100,
                    help="Only used when --algorithm relieff.")

parser.add_argument("--n-splits", type=int, default=5)

parser.add_argument("--n-jobs", type=int, default=-1)

parser.add_argument("--n-samples", type=int, default=None,
                    help="If set, subsample this many patients (stratified by label) before "
                         "running CV. Relief-family algorithms are O(n^2) in patients, so this "
                         "is meant for quick test runs, not final results.")

args = parser.parse_args()

dataset = args.dataset
base_path = Path(args.base_path)

TASK = args.task

DEPT = "ALL"

exp_name = DEPT + "_" + TASK

print("Experiment:", exp_name)
print("Algorithm:", args.algorithm)

# =========================================================
# LOAD DATA
# =========================================================

dataset_dir = base_path / dataset

adms = pd.read_csv(
    dataset_dir / "sub_adm_target.csv",
    index_col=0
)

task_dict = (
    adms[["HADM_ID", TASK]]
    .set_index("HADM_ID")[TASK]
    .to_dict()
)

features_df = pd.read_csv(
    dataset_dir / "ts_features_out.csv",
    index_col=0,
    header=[0, 1]
)

features_df = features_df.loc[adms["HADM_ID"].values]

feature_names = [
    f"{c[0]}__{c[1]}" if isinstance(c, tuple) else str(c)
    for c in features_df.columns
]

y_all = np.array([
    task_dict[i]
    for i in features_df.index
])

ids = np.array(features_df.index)

if args.n_samples is not None and args.n_samples < len(ids):
    sample_pos, _ = train_test_split(
        np.arange(len(ids)),
        train_size=args.n_samples,
        stratify=y_all,
        random_state=SEED
    )
    features_df = features_df.iloc[sample_pos]
    y_all = y_all[sample_pos]
    ids = ids[sample_pos]
    print(f"Subsampled to {len(ids)} patients (stratified) for a quick test run.")

outdir = Path(
    f"results/feature_importances/{dataset}/{exp_name}"
)

outdir.mkdir(parents=True, exist_ok=True)

relief_rows = []

skf = StratifiedKFold(
    n_splits=args.n_splits,
    shuffle=True,
    random_state=SEED
)

for fold, (train_idx, test_idx) in enumerate(
    skf.split(ids, y_all)
):

    print(f"Fold {fold + 1}")

    X_train = features_df.iloc[train_idx].values.astype(float)
    y_train = y_all[train_idx]

    # skrebate has no API to accept a precomputed distance matrix, and D_full
    # (results/.../mean_distances_50_out.npz) is computed over a different,
    # fully-imputed time-series representation (discrete_ts_out.csv) than
    # ts_features_out.csv here -- reusing it would score importance on one
    # feature space using neighbours found in another. Missing values in
    # X_train are handled natively (np.isnan is checked internally), so no
    # imputation is needed either way.
    if args.algorithm == "relieff":
        relief = ReliefF(n_neighbors=args.n_neighbors, n_jobs=args.n_jobs)
    else:
        relief = MultiSURF(n_jobs=args.n_jobs)

    relief.fit(X_train, y_train)

    relief_imp = relief.feature_importances_

    for feat, val in zip(feature_names, relief_imp):

        relief_rows.append({
            "feature": feat,
            f"fold_{fold+1}": float(val)
        })

relief_df = build_importance_table(relief_rows)

relief_df.to_csv(
    outdir / f"relief_importances_{args.algorithm}.csv",
    index=False
)

print("\nSaved file to:")
print(outdir / f"relief_importances_{args.algorithm}.csv")
