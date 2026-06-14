import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold
from scipy.stats import entropy
import argparse

from hydra_utils import *
from hydra_method import *

SEED = 42

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])

parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")

parser.add_argument("--k", type=int, default=5)

args = parser.parse_args()

dataset = args.dataset

TASK = args.task

k = args.k

DEPT = "ALL"

exp_name = DEPT + "_" + TASK

print("Experiment:", exp_name)

distance="euclidean"

outdir = Path(f"results/splits/{dataset}/{exp_name}/{k}")
outdir.mkdir(parents=True, exist_ok=True)

dist_path = f"{dataset}/{distance}/mean_distances_50_out.npz"
dists = np.load(dist_path)["mean"].astype(np.float32)
D_full = squareform(dists)


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

y_all = np.array([task_dict[i] for i in features_df.index])
ids = np.array(features_df.index)


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)


for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y_all)):

    print(f"\nFold {fold}")

    D_train = D_full[np.ix_(train_idx, train_idx)]
    D_test_train = D_full[np.ix_(test_idx, train_idx)]

    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    train_ids = ids[train_idx]
    test_ids = ids[test_idx]

    knn_train = np.argsort(D_train, axis=1)[:, 1:k+1]

    train_rows = []

    for i in range(len(train_idx)):

        node_id = train_ids[i]
        node_label = y_train[i]

        neigh_labels = y_train[knn_train[i]]

        metrics = compute_neighbourhood_metrics(
            neigh_labels,
            node_label
        )

        train_rows.append({
            "id": node_id,
            "label": node_label,
            "neigh_id": train_ids[knn_train[i]],
            **metrics
        })

    knn_test = np.argsort(D_test_train, axis=1)[:, :k]

    test_rows = []

    for i in range(len(test_idx)):

        node_id = test_ids[i]
        node_label = y_test[i]

        neigh_labels = y_train[knn_test[i]]

        metrics = compute_neighbourhood_metrics(
            neigh_labels,
            node_label
        )

        test_rows.append({
            "id": node_id,
            "label": node_label,
            "neigh_id": train_ids[knn_test[i]],
            **metrics
        })

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)

    df_train.to_csv(outdir / f"fold_{fold}_train.csv", index=False)
    df_test.to_csv(outdir / f"fold_{fold}_test.csv", index=False)

    print(f"Saved fold {fold}")