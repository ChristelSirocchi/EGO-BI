import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
import shap
import argparse

from hydra_utils import *
from hydra_method import *

from collections import Counter

# =========================================================
# CONFIG
# =========================================================

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

feature_names = [
    f"{c[0]}__{c[1]}" if isinstance(c, tuple) else str(c)
    for c in features_df.columns
]

y_all = np.array([
    task_dict[i]
    for i in features_df.index
])

ids = np.array(features_df.index)

outdir = Path(
    f"results/feature_importances/{dataset}/{exp_name}"
)

outdir.mkdir(parents=True, exist_ok=True)

xgb_rows = []
shap_rows = []
perm_rows = []


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

metrics = []

for fold, (train_idx, test_idx) in enumerate(
    skf.split(ids, y_all)
):

    print(f"Fold {fold + 1}")

    X_train = features_df.iloc[train_idx]
    X_test = features_df.iloc[test_idx]

    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    scale_pos_weight = compute_scale_pos_weight(y_train)

    model = get_xgboost_model(
        X_train=X_train, 
        scale_pos_weight=scale_pos_weight, 
        constrained=False,
        seed=SEED
    )

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob > 0.5).astype(int)

    metrics.append(
    compute_metrics(
        y_test,
        pred,
        prob
    )
    )

    # =====================================================
    # XGB IMPORTANCE
    # =====================================================
    xgb_imp = model.feature_importances_

    for feat, val in zip(feature_names, xgb_imp):

        xgb_rows.append({
            "feature": feat,
            f"fold_{fold+1}": float(val)
        })

    # =====================================================
    # SHAP IMPORTANCE
    # =====================================================
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_imp = np.abs(shap_values).mean(axis=0)

    for feat, val in zip(feature_names, shap_imp):

        shap_rows.append({
            "feature": feat,
            f"fold_{fold+1}": float(val)
        })

    # =====================================================
    # PERMUTATION IMPORTANCE
    # =====================================================
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=SEED,
        scoring="roc_auc",
        n_jobs=-1
    )

    perm_imp = perm.importances_mean

    for feat, val in zip(feature_names, perm_imp):

        perm_rows.append({
            "feature": feat,
            f"fold_{fold+1}": float(val)
        })
        
metrics = pd.DataFrame(metrics).mean()

pd.DataFrame(metrics).to_csv(
    outdir / "original_metrics.csv"
)


xgb_df = build_importance_table(xgb_rows)
shap_df = build_importance_table(shap_rows)
perm_df = build_importance_table(perm_rows)

xgb_df.to_csv(
    outdir / "xgb_importances.csv",
    index=False
)

shap_df.to_csv(
    outdir / "shap_importances.csv",
    index=False
)

perm_df.to_csv(
    outdir / "permutation_importances.csv",
    index=False
)

print("\nSaved files to:")
print(outdir)