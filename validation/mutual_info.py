import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

# ---------------------------------------------------------------------
# Mutual-information rank-dynamics analysis: for each feature-importance
# method's top-K features, bin patients by hardness (easiest -> hardest)
# and see whether that feature's mutual information with the outcome
# concentrates on easy or hard cases.
#
# Ported from the exploratory notebook workflow (hydra_utils.py's
# combine_importances / compute_mutual_info_bin / quantify_rank_dynamics),
# with two changes:
#   - the "composition" hardness criterion is dropped. compute_composition.py
#     / prune_composition.py no longer exist in this repo -- see prune.py's
#     docstring for why (circularity with the KNN-based boundary method).
#   - importances and hardness scores are loaded from the files this repo's
#     own scripts produce (compute_global_baselines.py, compute_relief_
#     baselines.py, compute_hardness_performance.py, compute_hardness_
#     disjunct.py), not the notebook's own results tree.
#
# One correctness fix vs. the notebook version: it bins TD (tree depth)
# with the same "highest raw value = bin 1" rule as DS, but high TD means
# HARD while high DS means EASY -- so TD bins came out backwards whenever
# it was actually used with measure="TD" (in practice it never was; every
# call in the notebook left disjunct_measure at its "DS" default, so this
# was latent). Here TD is inverted before binning, so bin 1 = easiest and
# bin q = hardest for both measures, matching prune.py's convention.
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])

parser.add_argument("--task", type=str, default="HOSPITAL_EXPIRE_FLAG")

parser.add_argument("--criteria", type=str, default="performance,disjunct",
                    help="Comma-separated subset of {performance, disjunct} to analyse.")

parser.add_argument("--disjunct-measure", type=str, default="DS",
                    choices=["DS", "TD"],
                    help="Only used when 'disjunct' is in --criteria.")

parser.add_argument("--methods", type=str, default=None,
                    help="Comma-separated importance-source names to include, matching "
                         "<name>_importances.csv under results/feature_importances/{dataset}/"
                         "{exp_name}/ (e.g. xgb,shap,permutation,relief-multisurf-full). "
                         "Defaults to every *_importances.csv found there.")

parser.add_argument("--knn-importances", type=str, default=None,
                    help="Optional path to the main boundary method's importance CSV (e.g. "
                         "knn_importances*.csv from method/3_apply_knn_fi_nested.py, which "
                         "lives in a nested mode/stat/K-range directory this script doesn't "
                         "try to reconstruct). If given, included as an extra method.")

parser.add_argument("--knn-importances-name", type=str, default="knn",
                    help="Method name to use for --knn-importances.")

parser.add_argument("--no-clip", dest="clip", action="store_false",
                    help="By default, negative importances are clipped to 0 before "
                         "L1-normalising each method (Relief-family convention: negative "
                         "score = 'no evidence', not 'negatively important'). Pass this to "
                         "keep raw signed values instead.")

parser.add_argument("--q", type=int, default=5,
                    help="Number of hardness quantile bins (1 = easiest, q = hardest).")

parser.add_argument("--top", type=int, default=10,
                    help="Top-K features per method (by positive importance) to analyse.")

parser.add_argument("--bin-mode", type=str, default="equal",
                    choices=["equal", "more", "less"],
                    help="'equal': patients in exactly bin b. 'more': bin >= b (b..hardest). "
                         "'less': bin <= b (easiest..b).")

parser.add_argument("--mid", type=int, default=None,
                    help="Bin index splitting 'easy'/'baseline' from 'hard'/'boundary' bins in "
                         "the activation-ratio summary. Defaults to q // 2.")

parser.set_defaults(clip=True)

args = parser.parse_args()

SEED = 42
dataset = args.dataset
TASK = args.task
criteria = [c.strip() for c in args.criteria.split(",") if c.strip()]
disjunct_measure = args.disjunct_measure
clip = args.clip
q = args.q
top = args.top
bin_mode = args.bin_mode
mid = args.mid

DEPT = "ALL"
exp_name = f"{DEPT}_{TASK}"

if not set(criteria) <= {"performance", "disjunct"}:
    raise ValueError(f"--criteria must be a subset of {{performance, disjunct}}, got {criteria}")

print("Experiment:", exp_name)
print("Criteria:", criteria)


# =========================================================
# DATA / IMPORTANCE LOADING
# =========================================================

def load_features_target(dataset):
    adms = pd.read_csv(f"{dataset}/sub_adm_target.csv", index_col=0)
    features_df = pd.read_csv(f"{dataset}/ts_features_out.csv", index_col=0, header=[0, 1])
    features_df = features_df.loc[adms["HADM_ID"].values]
    features_df.columns = [f"{a}__{b}" for a, b in features_df.columns]
    return adms, features_df


def discover_methods(fi_dir):
    return sorted(p.name[: -len("_importances.csv")] for p in fi_dir.glob("*_importances.csv"))


def combine_importances(fi_dir, methods, knn_path=None, knn_name="knn", clip=True):
    dfs = []
    all_methods = list(methods)

    def _load_one(name, path):
        df_imp = pd.read_csv(path)[["feature", "mean"]].rename(columns={"mean": name})
        if clip:
            df_imp[name] = df_imp[name].clip(lower=0)
        # abs().sum() rather than sum(): safe regardless of sign distribution, and
        # identical to plain sum() once clipped to non-negative.
        df_imp[name] = df_imp[name] / df_imp[name].abs().sum()
        return df_imp.set_index("feature")

    for name in methods:
        dfs.append(_load_one(name, fi_dir / f"{name}_importances.csv"))

    if knn_path is not None:
        dfs.append(_load_one(knn_name, knn_path))
        all_methods.append(knn_name)

    imp_table = pd.concat(dfs, axis=1).fillna(0).reset_index()

    sort_col = knn_name if knn_name in imp_table.columns else all_methods[0]
    imp_table = imp_table.sort_values(by=sort_col, ascending=False)

    return imp_table, all_methods


def load_hardness_meta(dataset, exp_name, criterion, disjunct_measure, q):
    hardness_dir = Path(f"results/hardness/{dataset}/{exp_name}")

    if criterion == "performance":
        path = hardness_dir / "performance_hardness_oof.csv"
        score_col = "difficulty_score"
        sign = 1.0
    else:
        path = hardness_dir / "disjunct_hardness_oof.csv"
        score_col = disjunct_measure
        # DS: high = large, homogeneous disjunct = easy -> keep as is.
        # TD: high = deep, idiosyncratic leaf = hard -> invert so high = easy.
        sign = 1.0 if disjunct_measure == "DS" else -1.0

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run "
            f"{'compute_hardness_performance.py' if criterion == 'performance' else 'compute_hardness_disjunct.py'} "
            f"for --dataset {dataset} --task {TASK} first."
        )

    df = pd.read_csv(path, index_col="HADM_ID")
    meta = df[["label", score_col]].copy()
    meta["bin"] = pd.qcut(
        (sign * meta[score_col]).rank(method="first"), q=q, labels=range(q, 0, -1)
    ).astype(int)
    return meta


# =========================================================
# MUTUAL INFORMATION x HARDNESS BIN
# =========================================================

def compute_mutual_info_bin(features_df, meta, imp_table, methods, mode="equal", top=10):
    print(meta["bin"].value_counts().sort_index())

    full = features_df.join(meta)
    results = []

    bins = sorted(meta["bin"].unique())

    for type_rank in methods:

        ranked_features = (
            imp_table.loc[imp_table[type_rank] > 0]
            .sort_values(by=type_rank, ascending=False)
            .head(top)["feature"]
        )
        if len(ranked_features) < top:
            print(f"[warning] only {len(ranked_features)} feature(s) have a positive "
                  f"'{type_rank}' score; using {len(ranked_features)} instead of the "
                  f"requested top={top} rather than padding with zero/negative-score "
                  f"features that show no evidence of importance.")

        for feature in ranked_features:

            for b in bins:

                if mode == "equal":
                    sub = full.loc[full["bin"] == b, ["label", feature]].dropna()
                elif mode == "more":
                    sub = full.loc[full["bin"] >= b, ["label", feature]].dropna()
                else:
                    sub = full.loc[full["bin"] <= b, ["label", feature]].dropna()

                # skip empty / degenerate cases
                if (
                    len(sub) < 5
                    or sub[feature].nunique() < 2
                    or sub["label"].nunique() < 2
                ):
                    mi = np.nan
                else:
                    mi = mutual_info_classif(sub[[feature]], sub["label"], random_state=SEED)[0]

                results.append({
                    "rank_type": type_rank,
                    "feature": feature,
                    "bin": b,
                    "mi": mi,
                    "n_samples": len(sub)
                })

    mi_table = pd.DataFrame(results)

    mi_df = mi_table.pivot(index=["rank_type", "feature"], columns="bin", values="mi").reset_index()
    df_bin_mean = mi_df.groupby("rank_type")[mi_df.columns[2:]].mean()
    return mi_table, df_bin_mean


def quantify_rank_dynamics(df_bin_mean, mid=None):
    results = []

    for rank_type, row in df_bin_mean.iterrows():

        trajectory = row.values.astype(float)
        x = np.arange(len(trajectory)) + 1

        # Linear trend
        slope, intercept = np.polyfit(x, trajectory, 1)

        # Early (easy) vs late (hard) information concentration
        current_mid = len(trajectory) // 2 if mid is None else mid
        baseline_avg = np.mean(trajectory[:current_mid])
        boundary_avg = np.mean(trajectory[current_mid:])
        activation_ratio = (boundary_avg + 1e-6) / (baseline_avg + 1e-6)

        # Center of mass
        positions = np.arange(1, len(trajectory) + 1)
        total_mass = trajectory.sum()
        center_mass = np.sum(positions * trajectory) / total_mass if total_mass > 0 else np.nan

        rho, _ = spearmanr(x, trajectory)

        results.append({
            "rank_type": rank_type,
            "slope": slope,
            "bin_correlation": rho,
            "activation_ratio": activation_ratio,
            "center_mass": center_mass
        })

    return pd.DataFrame(results)


# =========================================================
# RUN
# =========================================================

adms, features_df = load_features_target(dataset)

fi_dir = Path(f"results/feature_importances/{dataset}/{exp_name}")
methods = args.methods.split(",") if args.methods else discover_methods(fi_dir)

if not methods and not args.knn_importances:
    raise FileNotFoundError(
        f"No *_importances.csv found under {fi_dir} and no --knn-importances given. "
        "Run compute_global_baselines.py / compute_relief_baselines.py first, or pass "
        "--methods / --knn-importances explicitly."
    )

print("Methods:", methods + ([args.knn_importances_name] if args.knn_importances else []))

imp_table, all_methods = combine_importances(
    fi_dir, methods,
    knn_path=Path(args.knn_importances) if args.knn_importances else None,
    knn_name=args.knn_importances_name,
    clip=clip
)

outdir_base = Path(f"results/mutual_info/{dataset}/{exp_name}")
summaries = []

for criterion in criteria:
    criterion_label = criterion if criterion == "performance" else f"disjunct_{disjunct_measure}"
    print(f"\n=== Criterion: {criterion_label} ===")

    meta = load_hardness_meta(dataset, exp_name, criterion, disjunct_measure, q)

    mi_table, df_bin_mean = compute_mutual_info_bin(
        features_df, meta, imp_table, all_methods, mode=bin_mode, top=top
    )

    summary = quantify_rank_dynamics(df_bin_mean.dropna(axis=1), mid=mid)
    summary["dataset"] = dataset
    summary["criterion"] = criterion_label
    summaries.append(summary)

    outdir = outdir_base / criterion_label
    outdir.mkdir(parents=True, exist_ok=True)
    mi_table.to_csv(outdir / "mi_table.csv", index=False)
    df_bin_mean.to_csv(outdir / "mi_bin_means.csv")
    print(f"Saved {outdir}/mi_table.csv and mi_bin_means.csv")

summary_df = pd.concat(summaries, ignore_index=True)
outdir_base.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(outdir_base / "rank_dynamics_summary.csv", index=False)

print(f"\nCompleted! Saved summary to: {outdir_base}/rank_dynamics_summary.csv")
print(summary_df)
