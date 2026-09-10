import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from itertools import product
from scipy.spatial.distance import squareform
from scipy import stats as spstats

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from utils import *
from model import *
from builder import *
import gc

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
    "--base-path",
    type=str,
    default=".",
    help="Directory containing the {dataset}/ input folder (sub_adm_target.csv, ts_features_out.csv, "
         "{distance}/mean_distances_50_out.npz). Defaults to the current working directory."
)

parser.add_argument(
    "--task",
    type=str,
    default="HOSPITAL_EXPIRE_FLAG"
)

parser.add_argument(
    "--k-min",
    type=int,
    default=8,
    help="Lower bound (inclusive) of the neighbourhood size K search grid."
)

parser.add_argument(
    "--k-max",
    type=int,
    default=28,
    help="Upper bound (inclusive) of the neighbourhood size K search grid."
)

parser.add_argument(
    "--k-step",
    type=int,
    default=1,
    help="Step size of the neighbourhood size K search grid (e.g. 2 scans every other K)."
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

parser.add_argument(
    "--outer-splits",
    type=int,
    default=5,
    help="Number of outer folds used for the reported (final) performance estimate."
)

parser.add_argument(
    "--outer-repeats",
    type=int,
    default=1,
    help="Number of repeats of the outer CV (>1 gives repeated-CV confidence intervals)."
)

parser.add_argument(
    "--inner-splits",
    type=int,
    default=3,
    help="Number of inner folds used to select K (and XGBoost hyperparameters) within each outer training fold."
)

parser.add_argument(
    "--xgb-max-depths",
    type=str,
    default="6",
    help="Comma-separated list of XGBoost max_depth values to search in the inner loop. "
         "Default is a single value (the original fixed config) i.e. no depth tuning."
)

parser.add_argument(
    "--xgb-learning-rates",
    type=str,
    default="0.05",
    help="Comma-separated list of XGBoost learning_rate values to search in the inner loop. "
         "Default is a single value (the original fixed config) i.e. no learning-rate tuning."
)

parser.add_argument(
    "--n-estimators",
    type=int,
    default=1000,
    help="Number of trees used both for inner-loop candidate evaluation and the final refit."
)

parser.add_argument(
    "--xgb-n-jobs",
    type=int,
    default=1,
    help="XGBoost's own n_jobs. Defaults to 1: the inner loop fits many models back to back, and "
         "letting each spawn its own multi-threaded thread pool has been observed to deadlock in "
         "CPU-constrained/sandboxed environments. Raise this only if you've confirmed your machine "
         "doesn't hit that."
)

parser.add_argument(
    "--selection-metric",
    type=str,
    default="f1",
    choices=["f1", "recall", "precision", "roc_auc", "pr_auc", "average_precision", "balanced_accuracy"],
    help="Metric maximised by the inner CV to select K (and XGBoost hyperparameters, if tuned)."
)

parser.add_argument(
    "--k-selection-rule",
    type=str,
    default="one_se",
    choices=["best", "one_se"],
    help="How K is chosen from the inner-CV results for a given (max_depth, learning_rate). "
         "'best': the single (K, max_depth, learning_rate) combo with the highest mean inner "
         "score (the original behaviour). 'one_se': first pick (max_depth, learning_rate) via "
         "the same joint best-mean rule, then -- restricted to that combo -- pick the SMALLEST K "
         "whose mean inner score across inner folds is within one standard error of the best mean "
         "inner score across K (a simplicity-favouring '1-SE rule')."
)

parser.add_argument(
    "--baseline-metrics",
    type=str,
    default=None,
    help="Optional path to a per-fold metrics CSV (e.g. from a comparator/baseline run using the "
         "same outer-CV protocol and seed) against which a paired Wilcoxon signed-rank test on F1 "
         "is computed. Folds are matched by position, so the baseline must use the same "
         "--outer-splits/--outer-repeats/seed."
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()


# =========================================================
# CONFIG
# =========================================================

SEED = args.seed

dataset = args.dataset
base_path = Path(args.base_path)
TASK = args.task
k_min = args.k_min
k_max = args.k_max
k_step = args.k_step
mode = args.mode
stat = args.stat
scaled = args.scaled
filter = args.filter
outer_splits = args.outer_splits
outer_repeats = args.outer_repeats
inner_splits = args.inner_splits
n_estimators = args.n_estimators
xgb_n_jobs = args.xgb_n_jobs
selection_metric = args.selection_metric
k_selection_rule = args.k_selection_rule
depth_grid = sorted({int(x) for x in args.xgb_max_depths.split(",")})
lr_grid = sorted({float(x) for x in args.xgb_learning_rates.split(",")})
k_grid = list(range(k_min, k_max + 1, k_step))

DEPT = "ALL"

print(f"Dataset: {dataset}")
print(f"Base path: {base_path.resolve()}")
print(f"Task: {DEPT}_{TASK}")
print(f"Aggregation Function: {stat}")
print(f"K search grid: {k_grid} ({len(k_grid)} values, step {k_step})")
print(f"XGBoost max_depth grid: {depth_grid}")
print(f"XGBoost learning_rate grid: {lr_grid}")
if len(depth_grid) == 1 and len(lr_grid) == 1:
    print(
        "NOTE: XGBoost hyperparameters are NOT being tuned (single-value grid). "
        "The fixed configuration is being carried through the nested protocol unchanged; "
        "pass --xgb-max-depths/--xgb-learning-rates with multiple values to tune them jointly with K."
    )
print(f"Outer CV: {outer_splits}-fold, {outer_repeats} repeat(s)")
print(f"Inner CV: {inner_splits}-fold (selection confined to each outer training fold)")
print(f"Selection metric (maximised by the inner CV): {selection_metric}")
print(f"K selection rule: {k_selection_rule}")

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
    builder = BUILDER_MAPPING[mode](agg_type=stat, filter=filter)
    print(f"{builder.__class__.__name__} instantiated with '{stat}' aggregation.")
else:
    raise ValueError(f"Unknown mode: '{mode}'. Expected one of {list(BUILDER_MAPPING.keys())}.")


filter_suffix = "_filtered" if filter else ""
step_suffix = f"_step{k_step}" if k_step != 1 else ""

outdir = Path(
    f"results/feature_importances/{dataset}/{exp_name}/nested_k{k_min}-{k_max}{step_suffix}/{mode}/{stat}{filter_suffix}"
)

outdir.mkdir(parents=True, exist_ok=True)

knn_data_dir = Path(
    f"results/knn_data/{dataset}/{exp_name}/nested_k{k_min}-{k_max}{step_suffix}/{mode}/{stat}{filter_suffix}"
)

knn_data_dir.mkdir(parents=True, exist_ok=True)


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


# ----------------------- DISTANCES -----------------------

dist_path = dataset_dir / distance / "mean_distances_50_out.npz"

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

X_all = features_df.values.astype(np.float32)


# =========================================================
# HELPERS
# =========================================================

def build_knn_datasets(X_tr, y_tr, D_tr, ids_tr, X_te, y_te, D_te_tr, ids_te, k):
    """Constructs the boundary-derived train/test datasets for a given K.
    Mirrors builder.fit_train / builder.transform_test from 3_apply_knn_fi.py.
    """

    X_train_knn, y_train_knn, z_train_knn, id_train_knn, sample_weight = builder.fit_train(
        X=X_tr, y=y_tr, D=D_tr, k=k, train_ids=ids_tr
    )

    X_test_knn, y_test_knn, z_test_knn, id_test_knn, tsw = builder.transform_test(
        X_test=X_te, y_test=y_te, X_train=X_tr, y_train=y_tr, D=D_te_tr, k=k, test_ids=ids_te
    )

    return (X_train_knn, y_train_knn, sample_weight), (X_test_knn, y_test_knn, tsw)


def fit_and_score(X_tr, y_tr, sw, X_te, y_te, max_depth, learning_rate, ntrees):
    """Fits one XGBoost model and returns (model, metrics_dict) on (X_te, y_te)."""

    scale_pos_weight = compute_scale_pos_weight(y_tr) if not scaled else 1.0

    model = get_xgboost_model(
        X_train=X_tr,
        scale_pos_weight=scale_pos_weight,
        constrained=True,
        seed=SEED,
        ntrees=ntrees,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_jobs=xgb_n_jobs
    )

    if scaled:
        model.fit(X_tr, y_tr, sample_weight=sw)
    else:
        model.fit(X_tr, y_tr)

    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob > 0.5).astype(int)

    metrics = compute_metrics(y_te, pred, prob)

    return model, metrics


def compute_boundary_stats(D, y_query, y_ref, k, exclude_self):
    """Counts/proportion of patients whose K-neighbourhood contains at least one
    opposite-label case ("heterogeneous boundary cases"), for a given K.
    Uses the exact same neighbour-selection convention as builder.py:
    exclude_self=True mirrors fit_train (D is train x train, self is column 0),
    exclude_self=False mirrors transform_test (D is query x train, no self column).
    """

    if exclude_self:
        knn_idx = np.argsort(D, axis=1)[:, 1:k + 1]
    else:
        knn_idx = np.argsort(D, axis=1)[:, :k]

    neigh_labels = y_ref[knn_idx]
    is_mixed = np.any(neigh_labels != y_query[:, None], axis=1)

    return int(is_mixed.sum()), float(is_mixed.mean())


def mean_std_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    mean = float(values.mean()) if n > 0 else float("nan")

    if n > 1:
        std = float(values.std(ddof=1))
        se = std / np.sqrt(n)
        t_crit = spstats.t.ppf((1 + confidence) / 2.0, n - 1)
        half_width = float(t_crit * se)
    else:
        std = 0.0
        half_width = 0.0

    return mean, std, half_width, n


def select_k_one_se_rule(records_df, metric):

    stats = (
        records_df
        .groupby("k")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    stats["se"] = stats["std"] / np.sqrt(stats["count"])

    best = stats.loc[stats["mean"].idxmax()]
    threshold = best["mean"] - best["se"]

    eligible_k = stats.loc[stats["mean"] >= threshold, "k"]

    return int(eligible_k.min())


# =========================================================
# STORAGE
# =========================================================

xgb_rows_knn = []
fold_records = []
selection_records = []
boundary_records = []
inner_fold_records = []


# =========================================================
# OUTER CROSS-VALIDATION (reported performance)
# =========================================================

if outer_repeats > 1:
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=SEED
    )
else:
    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=SEED
    )

param_grid = list(product(k_grid, depth_grid, lr_grid))

for outer_iter, (train_idx, test_idx) in enumerate(outer_cv.split(ids, y_all)):

    repeat = outer_iter // outer_splits
    fold = outer_iter % outer_splits

    print(f"\n=== Outer fold {fold + 1}/{outer_splits} (repeat {repeat + 1}/{outer_repeats}) ===")

    # =====================================================
    # OUTER SPLIT
    # =====================================================

    train_ids = ids[train_idx]
    test_ids_fold = ids[test_idx]

    X_train_nodes = X_all[train_idx]
    X_test_nodes = X_all[test_idx]

    y_train_nodes = y_all[train_idx]
    y_test_nodes = y_all[test_idx]

    D_train = D_full[np.ix_(train_idx, train_idx)]
    D_test_train = D_full[np.ix_(test_idx, train_idx)]

    # =====================================================
    # BOUNDARY-SET COVERAGE ACROSS THE FULL K GRID
    # (computed on the outer TRAINING fold only, i.e. no test-fold information)
    # =====================================================
    compute_stats = False
    if compute_stats:
        for kk in k_grid:
            n_mixed, prop_mixed = compute_boundary_stats(
                D_train, y_train_nodes, y_train_nodes, kk, exclude_self=True
            )
            boundary_records.append({
                "repeat": repeat,
                "outer_fold": fold,
                "k": kk,
                "split": "outer_train",
                "n_patients": len(train_idx),
                "n_heterogeneous": n_mixed,
                "proportion_heterogeneous": prop_mixed
            })

    # =====================================================
    # INNER CROSS-VALIDATION: select K (and XGBoost params)
    # using ONLY the outer training fold
    # =====================================================

    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=SEED
    )

    inner_positions = list(inner_cv.split(train_ids, y_train_nodes))

    # The boundary-derived KNN dataset depends only on (k, inner fold), not on
    # (max_depth, learning_rate) -- build it once per (k, inner fold) and reuse
    # it across every XGBoost candidate in the grid, instead of rebuilding it
    # once per (k, depth, lr, inner fold).
    inner_scores_raw = {}
    n_skipped = 0

    # Records for THIS outer fold only (subset of inner_fold_records), used
    # below to apply the 1-SE rule for K without mixing in other outer folds.
    outer_inner_records = []

    for kk in k_grid:

        for inner_fold_idx, (in_tr_pos, in_val_pos) in enumerate(inner_positions):

            in_tr_idx = train_idx[in_tr_pos]
            in_val_idx = train_idx[in_val_pos]

            D_in_tr = D_full[np.ix_(in_tr_idx, in_tr_idx)]
            D_in_val_tr = D_full[np.ix_(in_val_idx, in_tr_idx)]

            try:
                (Xtr_knn, ytr_knn, sw), (Xval_knn, yval_knn, _) = build_knn_datasets(
                    X_all[in_tr_idx], y_all[in_tr_idx], D_in_tr, ids[in_tr_idx],
                    X_all[in_val_idx], y_all[in_val_idx], D_in_val_tr, ids[in_val_idx],
                    kk
                )
            except ValueError:
                n_skipped += len(depth_grid) * len(lr_grid)
                continue

            if len(np.unique(ytr_knn)) < 2 or len(Xval_knn) == 0 or len(np.unique(yval_knn)) < 2:
                n_skipped += len(depth_grid) * len(lr_grid)
                continue

            for depth, lr in product(depth_grid, lr_grid):

                try:
                    _, val_metrics = fit_and_score(
                        Xtr_knn, ytr_knn, sw, Xval_knn, yval_knn,
                        max_depth=depth, learning_rate=lr, ntrees=n_estimators
                    )
                except ValueError:
                    n_skipped += 1
                    continue

                inner_scores_raw.setdefault((kk, depth, lr), []).append(val_metrics[selection_metric])

                record = {
                    **val_metrics,
                    "repeat": repeat,
                    "outer_fold": fold,
                    "inner_fold": inner_fold_idx,
                    "k": kk,
                    "max_depth": depth,
                    "learning_rate": lr
                }
                inner_fold_records.append(record)
                outer_inner_records.append(record)

    inner_scores = {
        key: float(np.mean(vals))
        for key, vals in inner_scores_raw.items()
        if vals
    }

    if not inner_scores:
        raise RuntimeError(
            f"Inner CV produced no valid (K, max_depth, learning_rate) candidate for outer fold {fold}. "
            "Widen --k-min/--k-max or check class balance within folds."
        )

    if n_skipped:
        print(f"  (skipped {n_skipped} degenerate inner fold/candidate combinations)")

    # =====================================================
    # SELECT (max_depth, learning_rate, K)
    # =====================================================


    joint_best_k, best_depth, best_lr = max(inner_scores, key=inner_scores.get)

    if k_selection_rule == "one_se":
        outer_inner_df = pd.DataFrame(outer_inner_records)
        candidates_df = outer_inner_df[
            (outer_inner_df["max_depth"] == best_depth) &
            (outer_inner_df["learning_rate"] == best_lr)
        ]
        best_k = select_k_one_se_rule(candidates_df, selection_metric)
    else:
        best_k = joint_best_k

    best_inner_score = inner_scores.get((best_k, best_depth, best_lr), float("nan"))

    print(
        f"  Selected via inner {inner_splits}-fold CV ({k_selection_rule} rule): "
        f"K={best_k}, max_depth={best_depth}, learning_rate={best_lr} "
        f"(mean inner {selection_metric}={best_inner_score:.4f}, "
        f"joint best-mean K would have been {joint_best_k}, "
        f"{len(inner_scores)}/{len(param_grid)} candidates evaluated)"
    )

    selection_records.append({
        "repeat": repeat,
        "outer_fold": fold,
        "selected_k": best_k,
        "selected_max_depth": best_depth,
        "selected_learning_rate": best_lr,
        "selection_metric": selection_metric,
        "k_selection_rule": k_selection_rule,
        "joint_best_mean_k": joint_best_k,
        "inner_mean_selection_metric": best_inner_score,
        "n_candidates_evaluated": len(inner_scores),
        "n_candidates_total": len(param_grid)
    })

    # =====================================================
    # REFIT ON THE FULL OUTER TRAINING FOLD, EVALUATE ONCE
    # ON THE UNTOUCHED OUTER TEST FOLD
    # =====================================================

    (X_train_knn, y_train_knn, sample_weight), (X_test_knn, y_test_knn, tsw) = build_knn_datasets(
        X_train_nodes, y_train_nodes, D_train, train_ids,
        X_test_nodes, y_test_nodes, D_test_train, test_ids_fold,
        best_k
    )

    knn_columns = feature_names

    print(f"  Boundary-derived rows -> train: {len(X_train_knn)}, test: {len(X_test_knn)}")
    print(f"  Train label counts: {Counter(list(y_train_knn))}")

    model_knn, test_metrics = fit_and_score(
        X_train_knn, y_train_knn, sample_weight, X_test_knn, y_test_knn,
        max_depth=best_depth, learning_rate=best_lr, ntrees=n_estimators
    )

    print(f"  Outer test metrics: {test_metrics}")

    test_metrics.update({
        "repeat": repeat,
        "outer_fold": fold,
        "k": best_k,
        "max_depth": best_depth,
        "learning_rate": best_lr
    })
    fold_records.append(test_metrics)

    # Boundary-set coverage on the outer TEST fold, at the K actually used
    n_mixed_test, prop_mixed_test = compute_boundary_stats(
        D_test_train, y_test_nodes, y_train_nodes, best_k, exclude_self=False
    )
    boundary_records.append({
        "repeat": repeat,
        "outer_fold": fold,
        "k": best_k,
        "split": "outer_test_selected_k",
        "n_patients": len(test_idx),
        "n_heterogeneous": n_mixed_test,
        "proportion_heterogeneous": prop_mixed_test
    })

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    knn_imp = model_knn.feature_importances_

    for feat, val in zip(feature_names, knn_imp):
        xgb_rows_knn.append({
            "feature": feat,
            f"fold_{outer_iter + 1}": float(val)
        })

    # =====================================================
    # SAVE PER-FOLD BOUNDARY-DERIVED DATASETS
    # =====================================================

    pd.DataFrame(X_train_knn, columns=knn_columns).assign(label=y_train_knn).to_csv(
        knn_data_dir / f"outer_r{repeat}_fold_{fold}_k{best_k}_train.csv", index=False
    )
    pd.DataFrame(X_test_knn, columns=knn_columns).assign(label=y_test_knn).to_csv(
        knn_data_dir / f"outer_r{repeat}_fold_{fold}_k{best_k}_test.csv", index=False
    )
    del X_train_knn, y_train_knn, sample_weight
    del X_test_knn, y_test_knn, tsw
    del model_knn
    gc.collect()

# =========================================================
# SUMMARY: PER-FOLD METRICS + CONFIDENCE INTERVALS
# =========================================================

metrics_df = pd.DataFrame(fold_records)

summary_rows = []
metric_cols = [
    c for c in metrics_df.columns
    if c not in ("repeat", "outer_fold", "k", "max_depth", "learning_rate")
]

for col in metric_cols:
    mean, std, ci95, n = mean_std_ci(metrics_df[col].values)
    summary_rows.append({
        "metric": col,
        "mean": mean,
        "std": std,
        "ci95_halfwidth": ci95,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
        "n_folds": n
    })

summary_df = pd.DataFrame(summary_rows)

print("\n=== Nested-CV performance (mean +/- 95% CI across outer folds) ===")
for _, row in summary_df.iterrows():
    print(f"  {row['metric']:>18s}: {row['mean']:.4f} +/- {row['ci95_halfwidth']:.4f} (n={int(row['n_folds'])})")


# =========================================================
# UNIFIED K: PERFORMANCE POOLED ACROSS ALL INNER FOLDS
# =========================================================

inner_df = pd.DataFrame(inner_fold_records)

group_cols = ["k", "max_depth", "learning_rate"]
inner_metric_cols = [
    c for c in inner_df.columns
    if c not in group_cols + ["repeat", "outer_fold", "inner_fold"]
]

inner_k_summary_rows = []

for (kk, depth, lr), group in inner_df.groupby(group_cols):
    row = {"k": kk, "max_depth": depth, "learning_rate": lr, "n_inner_evals": len(group)}

    for col in inner_metric_cols:
        mean, std, ci95, n = mean_std_ci(group[col].values)
        row[f"{col}_mean"] = mean
        row[f"{col}_std"] = std
        row[f"{col}_ci95_halfwidth"] = ci95

    inner_k_summary_rows.append(row)

inner_k_summary_df = (
    pd.DataFrame(inner_k_summary_rows)
    .sort_values(group_cols)
    .reset_index(drop=True)
)

best_unified_row = inner_k_summary_df.loc[inner_k_summary_df[f"{selection_metric}_mean"].idxmax()]

print(
    "\n=== Unified K (pooled across all inner folds, all outer folds/repeats) ===\n"
    f"  Best by mean {selection_metric}: K={int(best_unified_row['k'])}, "
    f"max_depth={best_unified_row['max_depth']}, learning_rate={best_unified_row['learning_rate']} "
    f"({selection_metric}={best_unified_row[f'{selection_metric}_mean']:.4f} "
    f"+/- {best_unified_row[f'{selection_metric}_ci95_halfwidth']:.4f}, "
    f"n_inner_evals={int(best_unified_row['n_inner_evals'])})"
)


# =========================================================
# OPTIONAL PAIRED STATISTICAL COMPARISON VS. A BASELINE
# =========================================================

if args.baseline_metrics:
    baseline_df = pd.read_csv(args.baseline_metrics)

    if len(baseline_df) != len(metrics_df):
        print(
            f"\nWARNING: baseline metrics file has {len(baseline_df)} rows but this run has "
            f"{len(metrics_df)} outer folds -- skipping paired test. The baseline must use the "
            "same --outer-splits/--outer-repeats/seed for a valid paired comparison."
        )
    else:
        stat_w, p_w = spstats.wilcoxon(metrics_df[selection_metric].values, baseline_df[selection_metric].values)
        print(
            f"\nPaired Wilcoxon signed-rank test on per-fold {selection_metric} vs. baseline "
            f"({args.baseline_metrics}): statistic={stat_w:.4f}, p={p_w:.4g}"
        )
        pd.DataFrame([{
            "test": "wilcoxon_signed_rank",
            "metric": selection_metric,
            "statistic": stat_w,
            "p_value": p_w,
            "baseline_file": args.baseline_metrics
        }]).to_csv(outdir / "paired_test_vs_baseline.csv", index=False)


# =========================================================
# SAVE RESULTS
# =========================================================

suffix = "_scaled_order" if scaled else ""

xgb_knn_df = build_importance_table(xgb_rows_knn)
xgb_knn_df.to_csv(outdir / f"knn_importances{suffix}.csv", index=False)

metrics_df.to_csv(outdir / f"knn_metrics_per_fold{suffix}.csv", index=False)
summary_df.to_csv(outdir / f"knn_metrics_summary{suffix}.csv", index=False)

pd.DataFrame(selection_records).to_csv(outdir / f"knn_selected_params{suffix}.csv", index=False)
pd.DataFrame(boundary_records).to_csv(outdir / f"knn_boundary_stats{suffix}.csv", index=False)

inner_df.to_csv(outdir / f"knn_inner_fold_metrics{suffix}.csv", index=False)
inner_k_summary_df.to_csv(outdir / f"knn_inner_k_summary{suffix}.csv", index=False)

print("\nSaved files to:")
print(outdir)
print(knn_data_dir)
