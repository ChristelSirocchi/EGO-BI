import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from skrebate import (
    ReliefF, MuRelief,
    SURF, SURFstar,
    MultiSURF, MultiSURFstar,
    SWRF, SWRFstar,
    MultiSWRF, MultiSWRFstar,
    MultiSWRFDB, MultiSWRFDBstar,
)
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
from model import build_importance_table

# =========================================================
# ALGORITHM REGISTRY
# =========================================================
# All twelve are standalone, scikit-learn-style estimators (fit/
# feature_importances_) confirmed to handle NaN natively via the same
# pairwise-deletion distance calculation ReliefF uses (SURF/SWRF-family
# either subclass it directly or reimplement an equivalent NaN-masked
# scoring function -- verified by fitting all twelve on NaN-containing
# data before adding them here).
#
# NOT included: TURF, VLS, Iter. These are wrappers that take an already-
# constructed core algorithm as an argument (and, for TURF, a different
# fit() signature requiring `headers`) rather than being standalone
# choices that fit this script's uniform "pick one, .fit(), read
# .feature_importances_" pattern.
#
# Only relieff/murelief take a neighbour COUNT (--n-neighbors) -- every
# other variant below determines its own neighbourhood/weighting from the
# data and has no such hyperparameter to tune.
NEEDS_K = {"relieff", "murelief"}

ALGO_CLASSES = {
    "relieff":         ReliefF,        # k nearest by raw distance rank
    "murelief":        MuRelief,       # k neighbours by largest deviation from mean distance
    "surf":            SURF,           # single global radius (dataset-wide mean distance)
    "surfstar":        SURFstar,       # SURF + scores against far instances too
    "multisurf":       MultiSURF,      # per-instance radius (mean - std/2)
    "multisurfstar":   MultiSURFstar,  # MultiSURF + scores against far instances too
    "swrf":            SWRF,           # continuous distance-weighting, no hard radius (global)
    "swrfstar":        SWRFstar,       # swrf variant, near+far weighting
    "multiswrf":       MultiSWRF,      # continuous distance-weighting, per-instance
    "multiswrfstar":   MultiSWRFstar,  # multiswrf variant, near+far weighting
    "multiswrfdb":     MultiSWRFDB,    # multiswrf, alternate weighting curve
    "multiswrfdbstar": MultiSWRFDBstar,
}

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
                    choices=sorted(ALGO_CLASSES.keys()),
                    help="Which Relief-family variant to run. Only relieff/murelief take a "
                         "neighbour COUNT (--n-neighbors) -- all other variants (surf*/"
                         "multisurf*/swrf*/multiswrf*) determine their own neighbourhood or "
                         "distance-weighting from the data and have no count to tune. See the "
                         "ALGO_CLASSES comment block above for a one-line description of each.")

parser.add_argument("--n-neighbors", type=str, default="100",
                    help="Only used when --algorithm is relieff or murelief. Accepts either an "
                         "integer (a raw neighbour COUNT per side, e.g. 20) or a float containing "
                         "a decimal point (a FRACTION of the fold, split between hits/misses -- "
                         "e.g. 0.1 means 10%% of the fold total, per skrebate's own semantics: "
                         "int(fraction * n_fold * 0.5) per side). Use a fraction when comparing "
                         "across differently-sized datasets/folds; use an integer when you already "
                         "know the exact count you want (e.g. reusing a value tuned elsewhere).")

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

# "0.1" -> fraction of the fold (per skrebate's own float-vs-int dispatch);
# "20"  -> raw neighbour count. Parsed once here so the rest of the script
# (and every fold's fit) sees the right Python type either way.
n_neighbors_arg = float(args.n_neighbors) if "." in args.n_neighbors else int(args.n_neighbors)

print("Experiment:", exp_name)
print("Algorithm:", args.algorithm)
if args.algorithm in NEEDS_K:
    kind = "fraction of fold" if isinstance(n_neighbors_arg, float) else "raw count"
    print(f"  n_neighbors (k): {n_neighbors_arg} ({kind})")
else:
    print("  (no neighbour-count hyperparameter for this variant)")

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
    # X_train are handled natively by every algorithm in ALGO_CLASSES (all
    # confirmed NaN-aware -- see the registry comment above), so no
    # imputation is needed for any of them either way.
    cls = ALGO_CLASSES[args.algorithm]
    if args.algorithm in NEEDS_K:
        relief = cls(n_neighbors=n_neighbors_arg, n_jobs=args.n_jobs)
    else:
        relief = cls(n_jobs=args.n_jobs)

    relief.fit(X_train, y_train)

    relief_imp = relief.feature_importances_

    for feat, val in zip(feature_names, relief_imp):

        relief_rows.append({
            "feature": feat,
            f"fold_{fold+1}": float(val)
        })

relief_df = build_importance_table(relief_rows)

relief_df.to_csv(
    outdir / f"relief-{args.algorithm}-full_importances.csv",
    index=False
)

print("\nSaved file to:")
print(outdir / f"relief-{args.algorithm}-full_importances.csv")