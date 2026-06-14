import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy.stats import (
    rankdata,
    spearmanr,
    pearsonr,
)
import ast

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

def kendalls_w(matrix):
    n, m = matrix.shape
    ranked = np.apply_along_axis(rankdata, 0, matrix)
    R = np.sum(ranked, axis=1)
    R_bar = np.mean(R)
    S = np.sum((R - R_bar) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    return W


def gini(x):
    x = np.abs(x) + 1e-12
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - (n + 1) / n


def topk_jaccard(matrix, k=20):
    n_folds = matrix.shape[1]
    top_sets = []

    for i in range(n_folds):
        idx = np.argsort(-matrix[:, i])[:k]
        top_sets.append(set(idx))

    scores = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            scores.append(inter / union)

    return np.mean(scores)


def mean_spearman(matrix):
    m = matrix.shape[1]
    vals = []

    for i in range(m):
        for j in range(i + 1, m):
            r, _ = spearmanr(matrix[:, i], matrix[:, j])
            vals.append(r)

    return np.mean(vals)


def mean_pearson(matrix):
    corrs = []

    n_folds = matrix.shape[1]

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            r, _ = pearsonr(matrix[:, i], matrix[:, j])
            corrs.append(r)

    return np.mean(corrs)


def _compute_entropy(p):
    if p == 0 or p == 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _compute_balance_weight(neigh_labels, node_label):
    k = len(neigh_labels)
    n_opposite = np.sum(neigh_labels != node_label)
    return min(n_opposite, k - n_opposite) + 1


def compute_neighbourhood_metrics(neigh_labels, node_label):
    neigh_labels = np.asarray(neigh_labels)
    n_total = len(neigh_labels)

    same_ratio = np.mean(neigh_labels == node_label) if n_total > 0 else 0.0
    is_mixed = int(np.any(neigh_labels != node_label)) if n_total > 0 else 0
    is_1nn_different = int(neigh_labels[0] != node_label) if n_total > 0 else 0

    counts = Counter(neigh_labels)
    n_class_0 = counts.get(0, 0)
    n_class_1 = counts.get(1, 0)

    p_opposite = np.mean(neigh_labels != node_label) if n_total > 0 else 0.0
    label_entropy = _compute_entropy(p_opposite)

    balance_weight = _compute_balance_weight(neigh_labels, node_label)

    group = -1 if is_mixed else node_label

    return {
        "same_ratio": same_ratio,
        "is_mixed": is_mixed,
        "is_1nn_different": is_1nn_different,
        "entropy": label_entropy,
        "balance_weight": balance_weight,
        "group": group,
        "n_class_0": n_class_0,
        "n_class_1": n_class_1,
        "k": n_total
    }


def to_lab(df):
    df = df.copy()
    df["lab"] = df["feature"].str.split("__").str[-1]
    return df


def aggregate_lab(df):
    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    return (
        df.groupby("lab")[fold_cols + ["mean"]]
        .mean()
        .reset_index()
        .rename(columns={"lab": "feature"})
    )

####################################################################################

# EVALUATION - GLOBAL TO LOCAL
def add_ranks(imp_table):
    # Mean importance across global methods
    imp_table["global_mean"] = (
        imp_table[["permutation", "shap", "xgb"]]
        .mean(axis=1)
    )

    # Compute ranks (1 = most important)
    imp_table["rank_knn"] = imp_table["knn"].rank(
        ascending=False,
        method="average"
    )

    imp_table["rank_global"] = imp_table["global_mean"].rank(
        ascending=False,
        method="average"
    )

    # Positive => more important in KNN than global methods
    imp_table["rank_diff"] = (
        imp_table["rank_global"]
        - imp_table["rank_knn"]
    )
    return imp_table

def load_pruning(config,type="performance",feat="knn",kk=5):

    dataset = config["dataset"]
    exp_name = config["exp_name"]
    best_k = config["best_k"]
    mode = config["mode"]
    stat = config["stat"]
    suffix = config["suffix"]

    if type == "performance":
        path1 = f"results/progressive_pruning_oof/{dataset}/{exp_name}/{kk}/summary_importance_shifts.csv"
        ccols = list(pd.read_csv(path1).columns[1:]) + ['imp']

    elif type == "composition":
        path1 = f"results/importances/{dataset}/{exp_name}/knn_{kk}_fraction_seed_averaged_table_3.csv"
        ccols = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', 'imp']
    else:
        ValueError("possible options: performance, composition")
        
    df = pd.read_csv(path1)

    if "(" in df.loc[0,"feature"]:
        df["feature"] = df["feature"].apply(ast.literal_eval)
        df["feature"] = [f"{c[0]}__{c[1]}" for c in df["feature"]]

    df = df.set_index("feature")

    #path2 = f"results/feature_importances/{dataset}/{exp_name}/{best_k}/{mode}/{stat}/knn_importances{suffix}.csv"

    path3 = f"results/feature_importances/{dataset}/{exp_name}/xgb_importances.csv"

    #knn = pd.read_csv(path2).set_index("feature")[["mean"]].rename(columns={"mean":"knn"})

    og = pd.read_csv(path3).set_index("feature")[["mean"]].rename(columns={"mean":"0.0"})

    knn = load_importance(config, feat).set_index("feature")[["mean"]].rename(columns={"mean":"imp"})

    df = df.join(og).join(knn)
    df = df[ccols]
    df_norm = df.copy()

    df_norm[ccols] = df_norm[ccols].div(df_norm[ccols].sum(axis=0), axis=1)

    df_norm_lab = to_lab(df_norm.reset_index()).groupby("lab")[ccols].mean()

    return df_norm, df_norm_lab

def compute_distances(df):
    imp_vec = df["imp"].values

    results = {
        "euclidean": [],
        "l1": [],
        "cosine": [],
        "rank_l1": [],
        "pearson": [],
        "spearman": []
    }

    cols = df.columns[:-1]
    for c in cols:
        vec = df[c].values

        results["euclidean"].append(np.linalg.norm(vec - imp_vec))
        results["l1"].append(np.sum(np.abs(vec - imp_vec)))

        results["cosine"].append(
            cosine_distances(vec.reshape(1, -1), imp_vec.reshape(1, -1))[0, 0]
        )

        results["rank_l1"].append(
            np.sum(np.abs(np.argsort(vec) - np.argsort(imp_vec)))
        )

        r, _ = pearsonr(vec, imp_vec)
        results["pearson"].append(r)

        r, _ = spearmanr(vec, imp_vec)
        results["spearman"].append(r)

    return pd.DataFrame(results, index=cols)


def plot_distances(df_feat, df_lab):

    cols = df_feat.index
    metrics = df_feat.columns

    fig, axes = plt.subplots(
        2, len(metrics),
        figsize=(22, 7),
        sharex=True
    )

    for i, m in enumerate(metrics):

        axes[0, i].plot(cols, df_feat[m], marker="o")
        axes[0, i].set_title(f"Feature: {m}")
        axes[0, i].grid(True)

        axes[1, i].plot(cols, df_lab[m], marker="o")
        axes[1, i].set_title(f"Lab: {m}")
        axes[1, i].grid(True)

        if "pearson" in m or "spearman" in m:
            axes[0, i].set_ylabel("higher = better")
        else:
            axes[0, i].set_ylabel("lower = better")

    plt.tight_layout()
    plt.show()


def compute_slopes(df):
    x = df.index.astype(float).values

    slopes = {}

    for col in df.columns:
        y = df[col].astype(float).values

        # linear fit: y = ax + b
        slope, intercept = np.polyfit(x, y, 1)

        slopes[col] = slope

    return pd.Series(slopes).sort_values(ascending=False)

######################### BEST K

def plot_find_best_k(config):

    dataset = config["dataset"]
    exp_name = config["exp_name"]
    mode = config["mode"]
    stat = config["stat"]
    suffix = config["suffix"]

    base_path = Path(f"results/feature_importances/{dataset}/{exp_name}/")

    ks = list(range(2,30))

    all_results = []

    for k in ks:
        try:
            file_path = base_path / f"{k}" / f"{mode}" / f"{stat}" / f"knn_metrics{suffix}.csv"
            df = pd.read_csv(file_path, header=None)
            df.columns = ["metric", "value"]
            df["k"] = k
            all_results.append(df)
        except:
            pass

    df_all = pd.concat(all_results, ignore_index=True).dropna()

    df_pivot = df_all.pivot(index="k", columns="metric", values="value")
    df_pivot["pr_auc_gain"] = df_pivot["pr_auc"] - df_pivot["support"]
    metrics = df_pivot.columns.tolist()

    n_metrics = len(metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        axes[i].plot(df_pivot.index, df_pivot[metric], marker="o")
        axes[i].set_title(metric)
        axes[i].set_xlabel("k")
        axes[i].set_ylabel(metric)
        axes[i].grid(True)

    # remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

#### PLOT IMPORTANCE

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def get_purple_colormap():
    # Base colormap (excluding very light start)
    base_cmap = plt.get_cmap("PuRd")

    # Number of colors you want in the new palette
    n_colors = 256

    # Interpolate between white and the base colormap using RGB space
    # Create a range from 0 (white) to 1 (deepest PuRd)
    values = np.linspace(0, 1, n_colors)

    # Extract the original PuRd colormap values
    purd_colors = base_cmap(values)

    # Blend from white by linearly interpolating with white
    white = np.array([1, 1, 1, 1])  # RGBA for white
    blended_colors = np.zeros_like(purd_colors)

    for i, color in enumerate(purd_colors):
        alpha = values[i]
        blended_colors[i] = (1 - alpha) * white + alpha * color

    # Create the new colormap
    white_to_purd = mcolors.ListedColormap(blended_colors)

    return white_to_purd

def get_ordered_cols():
    return [
    'first     value',
    'last      value',
    'all       max',
    'all       mean',
    'all       min',
    'all       range',
    'all       slope',
    'all       std',
    'first 24h max',
    'first 24h mean',
    'first 24h min',
    'first 24h range',
    'first 24h slope',
    'first 24h std',
    'last 24h  max',
    'last 24h  mean',
    'last 24h  min',
    'last 24h  range',
    'last 24h  slope',
    'last 24h  std',
    'diff      max',
    'diff      mean',
    'diff      min',
    'diff      range',
    'diff      slope',
    'diff      std',
    ]

def load_importance(config, type="knn"):

    dataset = config["dataset"]
    exp_name = config["exp_name"]
    mode = config["mode"]
    stat = config["stat"]
    suffix = config["suffix"]
    best_k = config["best_k"]

    base_dir = Path(f"results/feature_importances/{dataset}/{exp_name}")
    if type == "knn":
        return pd.read_csv(base_dir / f"{best_k}/{mode}/{stat}/knn_importances{suffix}.csv")
    elif type in ["xgb","shap","permutation","pairs"]:
        return pd.read_csv(base_dir / f"{type}_importances.csv")
    else:
        ValueError("importance type must be knn, xbg, shap, permutation")

def plot_hm_importance(config, type="knn"):
    imp = load_importance(config, type)[["feature","mean"]]
    imp[["time","lab"]] = imp["feature"].str.split("__", expand = True)
    imp[["window","stat"]] = imp["time"].str.split("_", expand = True)
    imp["window"] = imp["window"].replace({"delta":"diff", "first24": "first 24h", "last24": "last 24h" })
    imp["stat"] = imp["stat"].replace({"val":"value"})

    hm = imp.copy()
    # Create composite index for rows and sort by stat
    hm["row"] = list(zip(hm["window"],hm["stat"]))
    hm["window"] = pd.Categorical(hm["window"], categories=["first", "last", "all", "first 24h", "last 24h", "diff"], ordered=True)
    hm = hm.sort_values("window")

    # Pivot to wide format
    heatmap_data = hm.pivot(index="lab", columns="row", values="mean")
    heatmap_data.columns = [f"{a:<10}{b}" for a, b in heatmap_data.columns]
    label_order = imp.groupby("lab")["mean"].mean().sort_values(ascending=False).index
    heatmap_data = heatmap_data.loc[label_order]
    ordered_cols = get_ordered_cols()
    heatmap_data = heatmap_data[ordered_cols]

    plt.figure(figsize=(5.5, 10))
    ax = sns.heatmap(
        heatmap_data,
        vmax=hm["mean"].quantile(0.995),
        cmap=get_purple_colormap(),
        annot=False,
        square=True,
        linewidths=0.5,
        linecolor='grey',
        cbar_kws={'orientation': 'horizontal'}
    )
    ax.xaxis.tick_top()               # Move x-axis ticks to the top
    plt.xticks(rotation=90)  
    # Move colorbar to top
    cbar = ax.collections[0].colorbar

    #vmin = heatmap_data.min().min()
    #vmax = hm["mean"].quantile(0.975)
    #tick_vals = np.linspace(vmin, vmax, 4)

    #cbar.set_ticks(tick_vals)


    # Optional: Format tick labels (e.g., fewer decimals)
    #cbar.set_ticklabels([f"{val:.3f}" for val in tick_vals])

    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.tick_top()


#################### COMPUTE FEATURE IMPORTANCE STABILITY

def compute_fi_stability(config):

    methods = ["permutation", "shap", "xgb", "knn","pairs"]

    results_stat = []
    results_lab = []

    for method in methods:

        df = load_importance(config, method)

        fold_cols = [c for c in df.columns if c.startswith("fold_")]

        col_sums = df[fold_cols].sum(axis=0).replace(0, np.nan)
        df[fold_cols] = df[fold_cols].div(col_sums, axis=1)

        matrix = np.nan_to_num(df[fold_cols].to_numpy(), nan=0.0)

        res = {
            "method": method,
            "kendalls_w": kendalls_w(matrix),
            "spearman_mean": mean_spearman(matrix),
            "pearson_mean": mean_pearson(matrix),
            "fold_var": matrix.var(axis=1).mean()
        }

        for topk in range(10, 50, 10):
            res[f"top{topk}_jaccard"] = topk_jaccard(matrix, k=topk)

        results_stat.append(res)

        df = aggregate_lab(to_lab(df))

        col_sums = df[fold_cols].sum(axis=0).replace(0, np.nan)
        df[fold_cols] = df[fold_cols].div(col_sums, axis=1)

        matrix = np.nan_to_num(df[fold_cols].to_numpy(), nan=0.0)

        res = {
            "method": method,
            "kendalls_w": kendalls_w(matrix),
            "spearman_mean": mean_spearman(matrix),
            "pearson_mean": mean_pearson(matrix),
            "fold_var": matrix.var(axis=1).mean()
        }

        for topk in range(4, 15, 3):
            res[f"top{topk}_jaccard"] = topk_jaccard(matrix, k=topk)

        results_lab.append(res)

    results_df_stat = pd.DataFrame(results_stat).sort_values("kendalls_w", ascending=False)

    results_df_lab = pd.DataFrame(results_lab).sort_values("kendalls_w", ascending=False)

    return results_df_stat, results_df_lab



############# COMPUTE STABILITY ACROSS K ####################

def compute_stability_across_k(config):

    dataset = config["dataset"]
    exp_name = config["exp_name"]
    mode = config["mode"]
    stat = config["stat"]
    suffix = config["suffix"]

    base_dir = Path(f"results/feature_importances/{dataset}/{exp_name}")

    dfs = []
    dfs_lab = []

    for k in range(10, 17, 2):

        file_path = base_dir / f"{k}/{mode}/{stat}/knn_importances{suffix}.csv"

        df = (
            pd.read_csv(file_path)
            .set_index("feature")[["mean"]]
            .rename(columns={"mean": f"mean_{k}"})
        )

        df_lab = (
            to_lab(pd.read_csv(file_path))
            .groupby("lab")[["mean"]]
            .mean()
            .rename(columns={"mean": f"mean_{k}"})
        )

        dfs.append(df)
        dfs_lab.append(df_lab)

    all_df = pd.concat(dfs, axis=1)
    all_df_lab = pd.concat(dfs_lab, axis=1)

    def compute_metrics(matrix):
        matrix = np.nan_to_num(matrix, nan=0.0)
        return {
            "kendalls_w": kendalls_w(matrix),
            "spearman_mean": mean_spearman(matrix),
            "pearson_mean": mean_pearson(matrix),
        }

    res = {
        "type": "stat",
        **compute_metrics(all_df.to_numpy())
    }

    res_lab = {
        "type": "lab",
        **compute_metrics(all_df_lab.to_numpy())
    }

    return pd.DataFrame([res, res_lab])

##############################################################################

def combine_importances(config):
    dfs = []
    methods = ["permutation", "shap", "xgb", "knn","pairs"]

    for name in methods:
        df_imp = load_importance(config, name)
        df_imp = df_imp[["feature", "mean"]].rename(columns={"mean": name})
        df_imp[name] = df_imp[name] / df_imp[name].sum()
        df_imp = df_imp.set_index("feature")
        dfs.append(df_imp)

    imp_table = pd.concat(dfs, axis=1).fillna(0)
    imp_table = imp_table.reset_index()
    imp_table.sort_values(by="knn", ascending=False)

    means = imp_table.copy()
    split_cols = means["feature"].str.split("__", n=1, expand=True)
    means["time_window"] = split_cols[0]
    means["feature_name"] = split_cols[1]

    means = means.groupby(["feature_name"])[methods].sum().sort_values(by="knn", ascending=False)
    return imp_table, means

def plot_imp_means(means):
    
    df_melted = means.reset_index().melt(
        id_vars=["feature_name"],
        var_name="method",
        value_name="importance"
    )

    sorted_labels = (
        means
        .sort_values("knn", ascending=False)
        .index
    )

    df_melted["feature_name"] = pd.Categorical(
        df_melted["feature_name"],
        categories=sorted_labels,
        ordered=True
    )

    n_features = df_melted["feature_name"].nunique()
    height_per_feature = 0.2
    plot_height = max(4.5, n_features * height_per_feature)

    plt.figure(figsize=(6, plot_height))

    ax = sns.scatterplot(
        data=df_melted,
        x="importance",
        y="feature_name",
        hue="method",
        palette="deep",
        s=40
    )

    ax.set_xlabel("Average Feature Importance")
    ax.set_ylabel("")
    ax.set_title("Mean Feature Importance Comparison")

    plt.tight_layout()
    plt.show()

############################## LOAD DATA

def load_features_target(config):
    dataset = config["dataset"]
    adms = pd.read_csv(f"{dataset}/sub_adm_target.csv", index_col=0)
    features_df = pd.read_csv(f"{dataset}/ts_features_out.csv", index_col=0, header=[0, 1])
    features_df = features_df.loc[adms["HADM_ID"].values]
    features_df.columns = [f"{a}__{b}" if isinstance((a, b), tuple) else str((a, b)) for a, b in features_df.columns]
    return adms, features_df

def load_splits(config):
    dataset = config["dataset"]
    exp_name = config["exp_name"]
    best_k = config["best_k"]
    dfs=[]
    for fold in range(5):
        dfs.append(pd.read_csv(f"results/splits/{dataset}/{exp_name}/{best_k}/fold_{fold}_test.csv"))
    return pd.concat(dfs).set_index("id")

def load_difficulty(config):
    dataset = config["dataset"]
    exp_name = config["exp_name"]
    best_k = config["best_k"]
    path=f"results/progressive_pruning_oof/{dataset}/{exp_name}/sample_difficulty.csv"
    return pd.read_csv(path).set_index("HADM_ID").rename(columns={"y_true":"label"})

######### MUTUAL INFORMATION ANALYSIS

def compute_mutual_info_bin(config, imp_table, type="composition", mode="equal", top=10):

    adms, features_df = load_features_target(config)

    if type == "composition":
        meta = load_splits(config)
        meta["bin"] = meta["balance_weight"]

    elif type == "performance":
        meta = load_difficulty(config)
        #meta["bin"] = pd.cut(meta["difficulty_score"], bins=10, labels=range(10, 0, -1))
        meta["bin"] = pd.qcut(meta["difficulty_score"], q=10, labels=range(10, 0, -1))
    else:
        ValueError("must be composition or performance")

    full = features_df.join(meta)
    results = []

    methods = ["knn","permutation","shap","xgb","pairs"]

    bins = sorted(meta["bin"].unique())

    for type_rank in methods:

        ranked_features = imp_table.sort_values(by=f"{type_rank}", ascending=False).head(top)["feature"]

        for feature in ranked_features:

            for b in bins:

                if mode=="equal":
                    sub = full.loc[full["bin"] == b, ["label", feature]].dropna()
                elif mode=="more":
                    sub = full.loc[full["bin"] >= b, ["label", feature]].dropna()
                elif mode=="less":
                    sub = full.loc[full["bin"] <= b, ["label", feature]].dropna()

                # skip empty / degenerate cases
                if (
                    len(sub) < 5
                    or sub[feature].nunique() < 2
                    or sub["label"].nunique() < 2
                ):
                    mi = np.nan
                else:
                    mi = mutual_info_classif(sub[[feature]], sub["label"], random_state=42)[0]

                results.append({
                    "rank_type": type_rank,
                    "feature": feature,
                    "balance_weight": b,
                    "mi": mi,
                    "n_samples": len(sub)
                })

    mi_table = pd.DataFrame(results)

    mi_df = mi_table.pivot(index=["rank_type","feature"], columns="balance_weight", values="mi").reset_index().dropna(axis=1)
    df_bin_mean = mi_df.groupby("rank_type")[mi_df.columns[2:]].mean()
    return mi_table, df_bin_mean


def quantify_rank_dynamics(df_bin_mean, mid=None):

    results = []

    for rank_type, row in df_bin_mean.iterrows():

        trajectory = row.values.astype(float)

        x = np.arange(len(trajectory)) + 1

        # Linear trend
        slope, intercept = np.polyfit(x, trajectory, 1)

        # Early vs late information concentration
        current_mid = len(trajectory) // 2 if mid is None else mid

        baseline_avg = np.mean(trajectory[:current_mid])
        boundary_avg = np.mean(trajectory[current_mid:])

        activation_ratio = (
            (boundary_avg + 1e-6) /
            (baseline_avg + 1e-6)
        )

        # Center of mass
        positions = np.arange(1, len(trajectory) + 1)

        total_mass = trajectory.sum()

        if total_mass > 0:
            center_mass = np.sum(positions * trajectory) / total_mass
        else:
            center_mass = np.nan

        rho, _ = spearmanr(x, trajectory)

        results.append({
            "rank_type": rank_type,
            "Slope": slope,
            "Bin_correlation":rho,
            "Activation_Ratio": activation_ratio,
            "Center_Mass": center_mass
        })

    return pd.DataFrame(results)

def plot_cum_r(df_bin_mean):
    # row-wise normalization
    df_norm = df_bin_mean.div(df_bin_mean.sum(axis=1), axis=0)

    # flip columns: high → low (important!)
    df_flipped = df_norm.iloc[:, ::-1]

    # cumulative sum in flipped direction
    df_cum = df_flipped.cumsum(axis=1)

    plt.figure(figsize=(6, 4))

    for method in df_cum.index:
        plt.plot(
            df_cum.columns,
            df_cum.loc[method],
            marker="o",
            label=method
        )

    plt.xlabel("balance_weight (high → low)")
    plt.ylabel("Cumulative normalized MI")
    plt.title("Cumulative MI (High → Low bins)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_lorenz(df_bin_mean):

    df_norm = df_bin_mean.div(df_bin_mean.sum(axis=1), axis=0)

    # ensure increasing x-axis order
    df_sorted = df_norm.copy()

    # cumulative from LOW → HIGH
    cum_standard = df_sorted.cumsum(axis=1)

    # reverse Lorenz: HIGH → LOW mass accumulation
    df_rev = df_sorted.iloc[:, ::-1]
    cum_reverse = df_rev.cumsum(axis=1)

    plt.figure(figsize=(6, 4))

    for method in cum_reverse.index:
        y = cum_reverse.loc[method].values
        x = np.linspace(0, 1, len(y))

        plt.plot(x, y, marker="o", label=method)

    # reference diagonal (uniform distribution)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)

    plt.xlabel("Cumulative fraction of bins (high → low)")
    plt.ylabel("Cumulative MI mass")
    plt.title("Reverse Lorenz Curve (mass concentration in high bins)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    auc
)

import numpy as np
import pandas as pd


def compute_metrics_from_predictions(
    df,
    min_features=0,
    in_all=False
):

    methods = df["method"].unique()

    if in_all:
        # samples present for every method
        valid_ids = (
            df.groupby("HADM_ID")["method"]
            .nunique()
            .loc[lambda x: x == len(methods)]
            .index
        )
        df = df[df["HADM_ID"].isin(valid_ids)].copy()

    # optional feature threshold
    if min_features > 0:

        valid_ids = (
            df.groupby("HADM_ID")["n_features"]
            .min()
            .loc[lambda x: x >= min_features]
            .index
        )

        df = df[df["HADM_ID"].isin(valid_ids)]

    results = []

    for method, sub in df.groupby("method"):

        y_true = sub["y_true"].values
        y_prob = sub["y_prob"].values
        y_pred = sub["y_pred"].values

        acc_global = accuracy_score(y_true, y_pred)

        if len(np.unique(y_true)) < 2:
            auc_roc = np.nan
            avg_prec = np.nan
            pr_auc = np.nan
        else:
            auc_roc = roc_auc_score(y_true, y_prob)

            avg_prec = average_precision_score(
                y_true,
                y_prob
            )

            precision, recall, _ = precision_recall_curve(
                y_true,
                y_prob
            )
            precision_cls1 = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            recall_cls1 = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            f1_cls1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            pr_auc = auc(recall, precision)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        acc_class_0 = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        acc_class_1 = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        brier = brier_score_loss(
            y_true,
            y_prob
        )

        results.append({
            "method": method,
            "N": len(sub),
            "Accuracy_Global": acc_global,
            "Accuracy_Class_0": acc_class_0,
            "Accuracy_Class_1": acc_class_1,
            "Precision": precision_cls1,
            "Recall": recall_cls1,
            "F1": f1_cls1,
            "AUC": auc_roc,
            "Avg_Precision": avg_prec,
            "PR_AUC": pr_auc,
            "Brier": brier
        })

    return pd.DataFrame(results)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

def run_subset_local_validation(
    sel_sub,
    features_df,
    feature_subset,
    k=10,
    missing_strategy="drop_features",
    min_feature_fraction=0.5
):

    sample_results = []
    y_true_all = []
    y_prob_all = []

    n_required_features = max(1, int(len(feature_subset) * min_feature_fraction))

    for idx, row in sel_sub.iterrows():
        weight = row.balance_weight
        center_id = row.name
        center_label = row["label"]

        n_ids = row["neigh_id"]
        if isinstance(n_ids, str):
            n_ids = n_ids.replace("[", "").replace("]", "").split()

        neighbor_ids = [int(nn) for nn in n_ids][:k]

        try:
            X_train_df = features_df.loc[neighbor_ids, feature_subset].copy()
            y_train = features_df.loc[neighbor_ids, "label"].values
            X_test_df = features_df.loc[[center_id], feature_subset].copy()
        except KeyError:
            continue

        # ----------------------------

        if missing_strategy == "drop_sample":
            if X_train_df.isna().any().any() or X_test_df.isna().any().any():
                continue

        elif missing_strategy == "drop_features":

            nan_cols = (
                X_train_df.columns[X_train_df.isna().any()].tolist()
                + X_test_df.columns[X_test_df.isna().any()].tolist()
            )
            nan_cols = list(set(nan_cols))

            X_train_df = X_train_df.drop(columns=nan_cols)
            X_test_df = X_test_df.drop(columns=nan_cols)

            # enforce minimum feature retention
            if X_train_df.shape[1] < n_required_features:
                continue

        else:
            raise ValueError("missing_strategy must be 'drop_sample' or 'drop_features'")

        # ----------------------------

        X_train = X_train_df.values
        X_test = X_test_df.values

        model = LogisticRegression(
            C=1.0,
            random_state=42,
            max_iter=1000
        )

        model.fit(X_train, y_train)
        prob_center = model.predict_proba(X_test)[0, 1]

        y_true_all.append(center_label)
        y_prob_all.append(prob_center)

        pred_center = int(prob_center >= 0.5)

        sample_results.append({
            "HADM_ID": center_id,
            "y_true": center_label,
            "y_prob": prob_center,
            "y_pred": pred_center,
            "n_features": X_train_df.shape[1],
            "weight": weight
        }) 

    results_df = pd.DataFrame(sample_results)

    if len(results_df) == 0:
        return results_df

    return results_df