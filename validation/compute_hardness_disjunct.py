import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import argparse

SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])
parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")
args = parser.parse_args()

dataset = args.dataset
TASK = args.task
DEPT = "ALL"
exp_name = DEPT + "_" + TASK

print("Experiment:", exp_name)

outdir = Path(f"results/hardness/{dataset}/{exp_name}")
outdir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Disjunct-based hardness measures (Smith et al. 2014), independent of
# any KNN/trajectory-distance construction — used to avoid circularity
# with the composition-based (B_q) stratification.
# ---------------------------------------------------------------------

def _augment(X, imputer):
    """Median-impute + append missingness indicators (approximation to
    C4.5's fractional handling of unknown attribute values)."""
    X_imp = imputer.transform(X)
    miss_mask = np.isnan(X).astype(float)
    return np.hstack([X_imp, miss_mask])


def fit_disjunct_tree(X_train, y_train, random_state=SEED):
    """Fully-grown, unpruned tree: every instance carried as far down
    as possible, so the only remaining impure leaves are instances
    that are feature-identical but differ in class."""
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_aug = _augment(X_train, imputer)

    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        random_state=random_state,
    )
    tree.fit(X_aug, y_train)
    return tree, imputer


def disjunct_metrics(tree, imputer, X_train, X_query):
    """Return DS and TD for each row of X_query, using disjuncts
    (leaves) defined by a tree fit strictly on X_train."""
    X_train_aug = _augment(X_train, imputer)
    X_query_aug = _augment(X_query, imputer)

    train_leaf_ids = tree.apply(X_train_aug)
    leaf_sizes = np.bincount(train_leaf_ids)
    largest_disjunct = leaf_sizes.max()

    query_leaf_ids = tree.apply(X_query_aug)
    sizes = np.array([
        leaf_sizes[lid] if lid < len(leaf_sizes) and leaf_sizes[lid] > 0 else 1
        for lid in query_leaf_ids
    ])
    ds = sizes / largest_disjunct

    node_depths = _compute_node_depths(tree)
    td = node_depths[query_leaf_ids]

    return ds, td


def _compute_node_depths(tree):
    """Depth of every node in the fitted tree (root = 0)."""
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    depths = np.zeros(n_nodes, dtype=int)
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
    return depths


# ---------------------------------------------------------------------
# Data loading (mirrors main pipeline; no distance matrix needed here)
# ---------------------------------------------------------------------

adms = pd.read_csv(f"{dataset}/sub_adm_target.csv", index_col=0)

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

X_all = features_df.values.astype(np.float64)
y_all = np.array([task_dict[i] for i in features_df.index])
ids = np.array(features_df.index)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

all_rows = []

for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y_all)):

    print(f"\nFold {fold}")

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    test_ids = ids[test_idx]

    tree, imputer = fit_disjunct_tree(X_train, y_train)
    ds_test, td_test = disjunct_metrics(tree, imputer, X_train, X_test)

    fold_df = pd.DataFrame({
        "HADM_ID": test_ids,
        "label": y_test,
        "DS": ds_test,
        "TD": td_test,
        "fold": fold,
    })

    all_rows.append(fold_df)
    print(f"  scored {len(test_ids)} held-out instances")

# ---------------------------------------------------------------------
# Concatenate out-of-fold results: exactly one row per patient
# ---------------------------------------------------------------------

hardness_df = pd.concat(all_rows, axis=0).set_index("HADM_ID")

assert hardness_df.index.is_unique, \
    "Each HADM_ID should appear exactly once across concatenated test folds"
assert len(hardness_df) == len(ids), \
    "Concatenated out-of-fold results should cover the full cohort"

out_path = outdir / "disjunct_hardness_oof.csv"
hardness_df.to_csv(out_path)
print(f"\nSaved out-of-fold hardness measures to {out_path}")
print(hardness_df[["DS", "TD"]].describe())