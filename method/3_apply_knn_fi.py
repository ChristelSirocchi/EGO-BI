import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from scipy.spatial.distance import squareform

from sklearn.model_selection import StratifiedKFold

from utils import *
from model import *
from builder import *

# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="physionet",
    choices=["physionet", "mimiciii", "eICU"]
)

parser.add_argument(
    "--task",
    type=str,
    default="HOSPITAL_EXPIRE_FLAG"
)

parser.add_argument(
    "--k",
    type=int,
    default=5
)

parser.add_argument(
    "--mode",
    type=str,
    default="split",
    choices=["same", "subset", "split", "split_only"]
)

parser.add_argument(
    "--scaled",
    action="store_true",
    help="Whether to use sample weights for training"
)

parser.add_argument(
    "--filter",
    action="store_true",
    help="Whether to filter cases where >50% values are nan"
)

parser.add_argument(
    "--stat",
    type=str,
    default="mean",
    choices=["weighted", "mean", "max"],
    help="Aggregation statistical function used inside the neighborhood builder."
)

args = parser.parse_args()


# =========================================================
# CONFIG
# =========================================================

SEED = 42

dataset = args.dataset
TASK = args.task
k = args.k
mode = args.mode
stat = args.stat
scaled = args.scaled
filter = args.filter

DEPT = "ALL"

print(f"Dataset: {dataset}")
print(f"Task: {DEPT}_{TASK}")
print(f"Aggregation Function: {stat}")

exp_name = DEPT + "_" + TASK

distance = "euclidean"

# =========================================================
# DEFINE BUILDER
# =========================================================

# Dictionary routing modes to their respective Class references
BUILDER_MAPPING = {
    "same": SameBuilder,
    "subset": SubsetBuilder,
    "split": SplitBuilder,
    "split_only": SplitOnlyBuilder
}

if mode in BUILDER_MAPPING:
    # Instantiate dynamically using the mapping and passing the agg_type
    builder = BUILDER_MAPPING[mode](agg_type=stat,filter=filter)
    print(f"{builder.__class__.__name__} instantiated with '{stat}' aggregation.")
else:
    raise ValueError(f"Unknown mode: '{mode}'. Expected one of {list(BUILDER_MAPPING.keys())}.")


filter_suffix = "_filtered" if filter else ""

outdir = Path(
    f"results/feature_importances/{dataset}/{exp_name}/{k}/{mode}/{stat}{filter_suffix}"
)

outdir.mkdir(parents=True, exist_ok=True)

# Updated to nest by {stat} inside the knn_data directory
knn_data_dir = Path(
    f"results/knn_data/{dataset}/{exp_name}/{k}/{mode}/{stat}{filter_suffix}"
)

knn_data_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
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
    f"{dataset}/ts_features_out.csv",
    index_col=0,
    header=[0, 1]
)

features_df = features_df.loc[adms["HADM_ID"].values]


# ----------------------- DISTANCES -----------------------

dist_path = f"{dataset}/{distance}/mean_distances_50_out.npz"

dists = np.load(dist_path)["mean"].astype(np.float32)
dists_safe = np.where(np.isnan(dists), 999999.0, dists)

D_full = squareform(dists_safe)


# =========================================================
# FEATURES / LABELS
# =========================================================

feature_names = [
    f"{c[0]}__{c[1]}" if isinstance(c, tuple) else str(c)
    for c in features_df.columns
]

y_all = np.array([
    task_dict[i]
    for i in features_df.index
])

ids = np.array(features_df.index)

X_all = features_df.values


# =========================================================
# OUTPUT PATHS
# =========================================================


# =========================================================
# STORAGE
# =========================================================

xgb_rows_knn = []

metrics_knn = []


# =========================================================
# CROSS VALIDATION
# =========================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)


for fold, (train_idx, test_idx) in enumerate(
    skf.split(ids, y_all)
):

    print(f"\nFold {fold + 1}")

    # =====================================================
    # SPLIT
    # =====================================================

    train_ids = ids[train_idx]
    test_ids = ids[test_idx]

    X_train_nodes = X_all[train_idx]
    X_test_nodes = X_all[test_idx]

    y_train_nodes = y_all[train_idx]
    y_test_nodes = y_all[test_idx]


    # =====================================================
    # DISTANCES
    # =====================================================

    D_train = D_full[np.ix_(train_idx, train_idx)]
    D_test_train = D_full[np.ix_(test_idx, train_idx)]


    # =====================================================
    # KNN DATASET
    # =====================================================

    X_train_knn, y_train_knn, z_train_knn, id_train_knn, sample_weight = builder.fit_train(
        X=X_train_nodes,
        y=y_train_nodes,
        D=D_train,
        k=k,
        train_ids=train_ids
    )

    knn_columns = feature_names

    train_knn_df = pd.DataFrame(
        X_train_knn,
        index=id_train_knn,
        columns=knn_columns
    )

    train_knn_df["label"] = y_train_knn
    if scaled:
        train_knn_df["weight"] = sample_weight

    print(train_knn_df.head())
    
    train_knn_df.to_csv(
        knn_data_dir / f"fold_{fold}_train.csv",
        index=True
    )

    print(Counter(list(z_train_knn)))

    scale_pos_weight = compute_scale_pos_weight(y_train_knn) if not scaled else 1.0

    # =====================================================
    # TEST KNN
    # =====================================================

    X_test_knn, y_test_knn, z_test_knn, id_test_knn, tsw = builder.transform_test(
        X_test=X_test_nodes,
        y_test=y_test_nodes,
        X_train=X_train_nodes,
        y_train=y_train_nodes,
        D=D_test_train,
        k=k,
        test_ids=test_ids
    )

    test_knn_df = pd.DataFrame(
        X_test_knn,
        index=id_test_knn,
        columns=knn_columns
    )
    test_knn_df["label"] = y_test_knn
    if scaled:
        test_knn_df["weight"] = tsw

    test_knn_df.to_csv(
        knn_data_dir / f"fold_{fold}_test.csv",
        index=True
    )


    # =====================================================
    # KNN MODEL
    # =====================================================
    print("training")

    model_knn = get_xgboost_model(
        X_train=X_train_knn,
        scale_pos_weight=scale_pos_weight,
        constrained=True,
        seed=SEED,
        ntrees=1000
    )

    if scaled:
        model_knn.fit(X_train_knn, y_train_knn, sample_weight=sample_weight)
    else:
        model_knn.fit(X_train_knn, y_train_knn)

    prob_knn = model_knn.predict_proba(X_test_knn)[:, 1]
    pred_knn = (prob_knn > 0.5).astype(int)

    print(compute_metrics(
            y_test_knn,
            pred_knn,
            prob_knn
        ))
    
    # =====================================================
    # KNN IMPORTANCE
    # =====================================================

    knn_imp = model_knn.feature_importances_

    for feat, val in zip(feature_names, knn_imp):
        xgb_rows_knn.append({
            "feature": feat,
            f"fold_{fold+1}": float(val)
        })

    # =====================================================
    # METRICS
    # =====================================================

    metrics_knn.append(
        compute_metrics(
            y_test_knn,
            pred_knn,
            prob_knn
        )
    )


# =========================================================
# SUMMARY
# =========================================================

metrics_knn = pd.DataFrame(metrics_knn).mean()

print("\nKNN model:\n", metrics_knn)


# =========================================================
# FINAL TABLES
# =========================================================


# =========================================================
# SAVE RESULTS
# =========================================================

xgb_knn_df = build_importance_table(xgb_rows_knn)


suffix = "_scaled_order" if scaled else ""

xgb_knn_df.to_csv(outdir / f"knn_importances{suffix}.csv", index=False)
pd.DataFrame(metrics_knn).to_csv(outdir / f"knn_metrics{suffix}.csv")

print("\nSaved files to:")
print(outdir)
print(knn_data_dir)