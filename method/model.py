import numpy as np
import pandas as pd

from collections import Counter
from scipy.stats import spearmanr, pearsonr, rankdata

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    average_precision_score
)

from xgboost import XGBClassifier, XGBRegressor


def compute_scale_pos_weight(y_train):
    """
    Calculates the negative-to-positive class ratio factor for XGBoost's
    scale_pos_weight parameter to balance loss calculations.
    """
    counts = Counter(y_train)

    neg_count = counts.get(0, 1)
    pos_count = counts.get(1, 1)

    return float(neg_count / pos_count)


def compute_regression_metrics(y_true, pred):
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)

    mse = mean_squared_error(y_true, pred)
    rmse = np.sqrt(mse)

    rho, _ = spearmanr(y_true, pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "explained_variance": float(
            explained_variance_score(y_true, pred)
        ),
        "spearman": float(rho)
    }


def compute_metrics(y_true, pred, prob):

    precision, recall, _ = precision_recall_curve(
        y_true,
        prob
    )

    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": auc(recall[::-1], precision[::-1]),
        "average_precision": average_precision_score(y_true, prob),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "support": float(np.mean(np.asarray(y_true) == 1))
    }


def get_xgboost_model(
    X_train,
    scale_pos_weight=None,
    constrained=True,
    seed=42,
    ntrees=1000
):

    n_features = X_train.shape[1]

    if constrained:
        monotone_constraints = "(" + ",".join(["1"] * n_features) + ")"
    else:
        monotone_constraints = None

    return XGBClassifier(
        n_estimators=ntrees,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        monotone_constraints=monotone_constraints,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss"
    )


def get_xgboost_regressor(
    X_train,
    scale_pos_weight=None,
    constrained=True,
    seed=42,
    ntrees=1000
):

    n_features = X_train.shape[1]

    if constrained:
        monotone_constraints = "(" + ",".join(["1"] * n_features) + ")"
    else:
        monotone_constraints = None

    return XGBRegressor(
        n_estimators=ntrees,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints,
        random_state=seed,
        eval_metric="rmse"
    )


def build_importance_table(rows):

    df = pd.DataFrame(rows)

    df = (
        df.groupby("feature")
        .first()
        .reset_index()
    )

    fold_cols = [c for c in df.columns if "fold_" in c]

    df["mean"] = df[fold_cols].mean(axis=1)

    df = df.sort_values("mean", ascending=False)

    return df

