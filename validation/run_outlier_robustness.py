import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import nan_euclidean_distances

from sklearn.model_selection import StratifiedKFold

from hydra_utils import *
from hydra_method import *

import numpy as np

class BaseNeighborhoodBuilder:
    def __init__(self, agg_type='mean', filter=False, signed=False,
                 weight_type='count', exponent=1.0, winsorize=False,
                 winsor_limits=(0.01, 0.99)):
        self.agg_type = agg_type
        self.filter = filter
        self.signed = signed
        self.weight_type = weight_type
        self.exponent = exponent
        self.winsorize = winsorize
        self.winsor_limits = winsor_limits

        if agg_type not in ['mean', 'max', 'weighted']:
            raise ValueError("agg_type must be either 'mean', 'max', or 'weighted'")
        if weight_type not in ['count', 'entropy']:
            raise ValueError("weight_type must be either 'count' or 'entropy'")

    def fit_train(self, X, y, D, k, train_ids):
        knn_idx = np.argsort(D, axis=1)[:, 1:k+1]
        knn_dists = np.take_along_axis(D, knn_idx, axis=1)
        return self._build(X, y, knn_idx, knn_dists, X, y, train_ids)

    def transform_test(self, X_test, y_test, X_train, y_train, D, k, test_ids):
        knn_idx = np.argsort(D, axis=1)[:, :k]
        knn_dists = np.take_along_axis(D, knn_idx, axis=1)
        return self._build(X_test, y_test, knn_idx, knn_dists, X_train, y_train, test_ids)

    def _compute_entropy(self, p):
        if p == 0 or p == 1:
            return 0.0
        return (-p * np.log2(p) - (1 - p) * np.log2(1 - p))

    def _compute_diff(self, neigh_x, center_x):
        diff = neigh_x - center_x
        return diff if self.signed else np.abs(diff)

    def _compute_weight(self, neigh_labels, center_label):
        n_total = len(neigh_labels)
        n_opposite = np.sum(neigh_labels != center_label)
        n_same = n_total - n_opposite

        if self.weight_type == 'count':
            return min(n_opposite, n_same) + 1
        elif self.weight_type == 'entropy':
            p = n_opposite / n_total if n_total > 0 else 0.0
            ent = self._compute_entropy(p)
            return 1.0 + ent * (n_total / 2.0)

    def _build(self, X_query, y_query, knn_idx, knn_dists, X_ref, y_ref, query_ids):
        X_out, y_out, z_out, id_out, w_out = [], [], [], [], []

        for i in range(X_query.shape[0]):
            neigh_labels = y_ref[knn_idx[i]]

            feats = self._compute_features(
                center_x=X_query[i],
                center_label=y_query[i],
                neigh_x=X_ref[knn_idx[i]],
                neigh_labels=neigh_labels,
                neigh_dists=knn_dists[i]
            )

            weight = self._compute_weight(neigh_labels, y_query[i])

            for feat, mode, label in feats:
                X_out.append(feat)
                y_out.append(mode)
                z_out.append(label)
                id_out.append(query_ids[i])
                w_out.append(weight)

        return np.array(X_out), np.array(y_out), np.array(z_out), np.array(id_out), np.array(w_out)

    def _winsorize(self, diffs):
        lo, hi = self.winsor_limits
        lo_vals = np.nanpercentile(diffs, lo * 100, axis=0)
        hi_vals = np.nanpercentile(diffs, hi * 100, axis=0)
        return np.clip(diffs, lo_vals, hi_vals)

    def _aggregate(self, diffs, dists, max_missing_ratio=0.5):
        if self.winsorize:
            diffs = self._winsorize(diffs)

        n_neighbors = diffs.shape[0]
        nan_mask = np.isnan(diffs)
        missing_counts = np.sum(nan_mask, axis=0)
        missing_ratios = missing_counts / n_neighbors
        too_many_nans_mask = missing_ratios >= max_missing_ratio

        if self.agg_type == 'mean':
            aggregated = np.nanmean(diffs, axis=0)
        elif self.agg_type == 'max':
            aggregated = np.nanmax(diffs, axis=0)
        elif self.agg_type == 'weighted':
            weights = 1.0 / (dists + 1e-5) ** self.exponent

            nan_mask = np.isnan(diffs)
            if np.any(nan_mask):
                weights_expanded = np.repeat(weights[:, np.newaxis], diffs.shape[1], axis=1)
                weights_expanded[nan_mask] = 0.0

                numerator = np.nansum(diffs * weights_expanded, axis=0)
                denominator = np.sum(weights_expanded, axis=0)
                denominator = np.where(denominator == 0, 1, denominator)
                aggregated = numerator / denominator
            else:
                aggregated = np.average(diffs, axis=0, weights=weights)

        if self.filter:
            aggregated[too_many_nans_mask] = np.nan
        return aggregated

    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        raise NotImplementedError


class SplitOnlyBuilder(BaseNeighborhoodBuilder):
    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        same_mask = (neigh_labels == center_label)
        opp_mask = ~same_mask
        outputs = []

        if np.any(same_mask) and np.any(opp_mask):
            diffs_same = self._compute_diff(neigh_x[same_mask], center_x)
            feat_same = self._aggregate(diffs_same, neigh_dists[same_mask])
            outputs.append((feat_same, 0, center_label))

            diffs_opp = self._compute_diff(neigh_x[opp_mask], center_x)
            feat_opp = self._aggregate(diffs_opp, neigh_dists[opp_mask])
            outputs.append((feat_opp, 1, center_label))

        return outputs


class SameBuilder(BaseNeighborhoodBuilder):
    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        opp_mask = (neigh_labels != center_label)
        is_mixed = int(np.any(opp_mask))
        diffs = self._compute_diff(neigh_x, center_x)
        feat = self._aggregate(diffs, neigh_dists)
        return [(feat, is_mixed, -1 if is_mixed else center_label)]


class SubsetBuilder(BaseNeighborhoodBuilder):
    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        opp_mask = (neigh_labels != center_label)
        same_mask = ~opp_mask
        is_mixed = int(np.any(opp_mask))
        mask = opp_mask if np.any(opp_mask) else same_mask
        if np.any(mask):
            diffs = self._compute_diff(neigh_x[mask], center_x)
            feat = self._aggregate(diffs, neigh_dists[mask])
        else:
            feat = np.full(center_x.shape, np.nan)
        return [(feat, is_mixed, -1 if is_mixed else center_label)]


class SplitBuilder(BaseNeighborhoodBuilder):
    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        same_mask = (neigh_labels == center_label)
        opp_mask = ~same_mask
        outputs = []
        if np.any(same_mask):
            diffs_same = self._compute_diff(neigh_x[same_mask], center_x)
            feat_same = self._aggregate(diffs_same, neigh_dists[same_mask])
            outputs.append((feat_same, 0, center_label))
        if np.any(opp_mask):
            diffs_opp = self._compute_diff(neigh_x[opp_mask], center_x)
            feat_opp = self._aggregate(diffs_opp, neigh_dists[opp_mask])
            outputs.append((feat_opp, 1, -1))
        return outputs


def compute_outlier_scores(D_train, min_valid_comparisons=5, invalid_sentinel=999999.0):
    """For each training patient (row of D_train), summarise how different
    they are from the rest of the training pool: mean distance over the
    OTHER patients they have a genuinely valid (non-sentinel) comparison
    with. A patient with too few valid comparisons to compute this
    reliably is scored as +inf -- treated as automatically an outlier,
    since having almost no mutually-observed data with anyone is itself
    a legitimate reason not to trust them as a training neighbour.
    """
    n = D_train.shape[0]
    valid_mask = (D_train != invalid_sentinel)
    np.fill_diagonal(valid_mask, False)  # never compare a patient to themself

    valid_counts = valid_mask.sum(axis=1)
    D_masked = np.where(valid_mask, D_train, np.nan)
    scores = np.nanmean(D_masked, axis=1)

    scores = np.where(valid_counts < min_valid_comparisons, np.inf, scores)
    return scores


def remove_outlier_indices(D_train, remove_frac, min_valid_comparisons=5):
    """Returns (keep_positions, removed_positions) -- LOCAL positions within
    this fold's training pool (0..len(train_idx)-1), not global admission ids.
    """
    n = D_train.shape[0]
    n_remove = max(1, int(round(remove_frac * n))) if remove_frac > 0 else 0
    if n_remove == 0:
        return np.arange(n), np.array([], dtype=int)

    scores = compute_outlier_scores(D_train, min_valid_comparisons=min_valid_comparisons)
    order = np.argsort(-scores)  # most different from others first
    removed_positions = order[:n_remove]
    keep_positions = np.setdiff1d(np.arange(n), removed_positions, assume_unique=False)
    return keep_positions, removed_positions


# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])

parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")

parser.add_argument("--mode", type=str, default="split_only",
                    choices=["same", "subset", "split", "split_only"])

parser.add_argument("--scaled", action="store_true", default=True,
                    help="Matches full_method's setting; kept as an option for consistency.")

parser.add_argument("--filter", action="store_true")

parser.add_argument("--stat", type=str, default="weighted",
                    choices=["weighted", "mean", "max"])

parser.add_argument("--constrained", action="store_true", default=True,
                    help="Matches full_method's setting.")

parser.add_argument("--signed", action="store_true")

parser.add_argument("--weight_type", type=str, default="count", choices=["count", "entropy"])

parser.add_argument("--exponent", type=float, default=1.0)

parser.add_argument("--winsorize", action="store_true")

parser.add_argument("--distance_space", type=str, default="trajectory",
                    choices=["trajectory", "summary"],
                    help="Outlier scoring uses the SAME distance space as neighbourhood "
                         "construction, so an 'outlier' is defined consistently with "
                         "whatever space the method actually uses to find neighbours.")

# --- new for this robustness check ---
parser.add_argument("--remove_frac", type=float, default=0.01,
                    help="Fraction of TRAINING patients to remove each fold, chosen as those "
                         "with the largest mean distance to the rest of the (valid) training "
                         "pool -- i.e. the most different from everyone else. Test set is "
                         "never touched.")

parser.add_argument("--min_valid_comparisons", type=int, default=5,
                    help="A training patient with fewer than this many non-sentinel (valid "
                         "trajectory-overlap) comparisons to other training patients is scored "
                         "as an automatic outlier, rather than relying on an unreliable "
                         "few-comparison average.")

parser.add_argument("--run_name", type=str, default="remove_outliers_1pct",
                    help="Tag used in the output directory for this run.")

args = parser.parse_args()

# =========================================================
# CONFIG
# =========================================================

SEED = 42

BEST_K = {
     "physionet": [16, 14, 16, 16, 14],
     "mimiciii":  [22, 24, 24, 18, 16],
     "eICU":      [18, 16, 10, 16, 16],
}

dataset = args.dataset
TASK = args.task
mode = args.mode
stat = args.stat
scaled = args.scaled
filter = args.filter

constrained = args.constrained
signed = args.signed
weight_type = args.weight_type
exponent = args.exponent
winsorize = args.winsorize
distance_space = args.distance_space
remove_frac = args.remove_frac
min_valid_comparisons = args.min_valid_comparisons
run_name = args.run_name

DEPT = "ALL"
exp_name = DEPT + "_" + TASK
k_per_fold = BEST_K[dataset]

print(f"Dataset: {dataset}")
print(f"Task: {DEPT}_{TASK}")
print(f"Run: {run_name} (robustness check -- NOT an ablation)")
print(f"Per-fold k: {k_per_fold}")
print(f"remove_frac={remove_frac} (min_valid_comparisons={min_valid_comparisons})")
print(f"mode={mode} stat={stat} scaled={scaled} constrained={constrained} "
      f"signed={signed} weight_type={weight_type} exponent={exponent} "
      f"winsorize={winsorize} distance_space={distance_space}")

BUILDER_MAPPING = {
    "same": SameBuilder,
    "subset": SubsetBuilder,
    "split": SplitBuilder,
    "split_only": SplitOnlyBuilder
}

if mode not in BUILDER_MAPPING:
    raise ValueError(f"Unknown mode: '{mode}'. Expected one of {list(BUILDER_MAPPING.keys())}.")

builder = BUILDER_MAPPING[mode](
    agg_type=stat, filter=filter, signed=signed,
    weight_type=weight_type, exponent=exponent, winsorize=winsorize
)
print(f"{builder.__class__.__name__} instantiated.")

# NOTE: separate tree from results/ablations/ -- this is a robustness check,
# not a design-choice comparison, and shouldn't be mixed into the ablation table.
outdir = Path(f"results/robustness_outliers/{dataset}/{exp_name}/{run_name}")
outdir.mkdir(parents=True, exist_ok=True)

removed_ids_dir = Path(f"results/robustness_outliers_removed_ids/{dataset}/{exp_name}/{run_name}")
removed_ids_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

adms = pd.read_csv(f"{dataset}/sub_adm_target.csv", index_col=0)
task_dict = adms[["HADM_ID", TASK]].set_index("HADM_ID")[TASK].to_dict()

features_df = pd.read_csv(f"{dataset}/ts_features_out.csv", index_col=0, header=[0, 1])
features_df = features_df.loc[adms["HADM_ID"].values]

feature_names = [
    f"{c[0]}__{c[1]}" if isinstance(c, tuple) else str(c)
    for c in features_df.columns
]

y_all = np.array([task_dict[i] for i in features_df.index])
ids = np.array(features_df.index)
X_all = features_df.values

# ----------------------- DISTANCES -----------------------

if distance_space == "trajectory":
    dist_path = f"{dataset}/euclidean/mean_distances_50_out.npz"
else:
    dist_path = f"{dataset}/euclidean/mean_distances_over_stats_50_out.npz"

dists = np.load(dist_path)["mean"].astype(np.float32)
dists_safe = np.where(np.isnan(dists), 999999.0, dists)
D_full = squareform(dists_safe)

# =========================================================
# STORAGE
# =========================================================

xgb_rows_knn = []
metrics_knn = []
removed_records = []

# =========================================================
# CROSS VALIDATION
# =========================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y_all)):

    k = k_per_fold[fold]
    print(f"\nFold {fold + 1} (k={k})")

    train_ids = ids[train_idx]
    test_ids = ids[test_idx]

    X_train_nodes = X_all[train_idx]
    X_test_nodes = X_all[test_idx]

    y_train_nodes = y_all[train_idx]
    y_test_nodes = y_all[test_idx]

    D_train = D_full[np.ix_(train_idx, train_idx)]
    D_test_train = D_full[np.ix_(test_idx, train_idx)]

    # ----------------------- REMOVE OUTLIERS (training set only) -----------------------

    keep_pos, removed_pos = remove_outlier_indices(
        D_train, remove_frac=remove_frac, min_valid_comparisons=min_valid_comparisons
    )
    print(f"  Removing {len(removed_pos)}/{len(train_idx)} training patients "
          f"(most different from the rest of the training pool)")

    removed_records.append(pd.DataFrame({
        "fold": fold,
        "HADM_ID": train_ids[removed_pos],
    }))

    train_ids = train_ids[keep_pos]
    X_train_nodes = X_train_nodes[keep_pos]
    y_train_nodes = y_train_nodes[keep_pos]
    D_train = D_train[np.ix_(keep_pos, keep_pos)]
    D_test_train = D_test_train[:, keep_pos]  # test rows untouched, only train columns shrink

    # ----------------------- KNN DATASET -----------------------

    X_train_knn, y_train_knn, z_train_knn, id_train_knn, sample_weight = builder.fit_train(
        X=X_train_nodes, y=y_train_nodes, D=D_train, k=k, train_ids=train_ids
    )

    knn_columns = feature_names
    print(Counter(list(z_train_knn)))

    scale_pos_weight = compute_scale_pos_weight(y_train_knn) if not scaled else 1.0

    # ----------------------- TEST KNN -----------------------

    X_test_knn, y_test_knn, z_test_knn, id_test_knn, tsw = builder.transform_test(
        X_test=X_test_nodes, y_test=y_test_nodes,
        X_train=X_train_nodes, y_train=y_train_nodes,
        D=D_test_train, k=k, test_ids=test_ids
    )

    # ----------------------- MODEL -----------------------
    print("training")

    model_knn = get_xgboost_model(
        X_train=X_train_knn,
        scale_pos_weight=scale_pos_weight,
        constrained=constrained,
        seed=SEED,
        ntrees=1000
    )

    if scaled:
        model_knn.fit(X_train_knn, y_train_knn, sample_weight=sample_weight)
    else:
        model_knn.fit(X_train_knn, y_train_knn)

    prob_knn = model_knn.predict_proba(X_test_knn)[:, 1]
    pred_knn = (prob_knn > 0.5).astype(int)

    fold_metrics = compute_metrics(y_test_knn, pred_knn, prob_knn)
    print(fold_metrics)

    # ----------------------- IMPORTANCE -----------------------

    knn_imp = model_knn.feature_importances_
    for feat, val in zip(feature_names, knn_imp):
        xgb_rows_knn.append({"feature": feat, f"fold_{fold+1}": float(val)})

    metrics_knn.append(fold_metrics)


# =========================================================
# SUMMARY
# =========================================================

metrics_knn_df = pd.DataFrame(metrics_knn)
metrics_knn_mean = metrics_knn_df.mean()

print(f"\n[{run_name}] mean metrics:\n", metrics_knn_mean)

xgb_knn_df = build_importance_table(xgb_rows_knn)

xgb_knn_df.to_csv(outdir / "importances.csv", index=False)
metrics_knn_df.to_csv(outdir / "metrics_per_fold.csv", index=False)
metrics_knn_mean.to_csv(outdir / "metrics_mean.csv")

pd.concat(removed_records, ignore_index=True).to_csv(
    removed_ids_dir / "removed_patients_per_fold.csv", index=False
)

print("\nSaved files to:", outdir)
print(removed_ids_dir)

# python compute_outlier_robustness.py --run_name remove_outliers_1pct --remove_frac 0.01