import pandas as pd
import argparse
from utilities import *
import warnings
warnings.filterwarnings("ignore")

# ----------------- Parse command-line arguments -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])
parser.add_argument("--interval", type=int, default=1)
args = parser.parse_args()

dataset = args.dataset
interval = args.interval

interval_suffix = f"_{interval}" if interval != 1 else ""

print(f"[Checkpoint] Dataset: {dataset}")

# ----------------- Read input data -----------------
print("[Checkpoint] Reading input CSVs...")
all_labs = pd.read_csv(f"{dataset}/all_labs_out.csv", index_col=0)
adms = pd.read_csv(f"{dataset}/adm_target.csv", index_col=0)

# ----------------- Preprocess admissions and labs -----------------
all_labs["HADM_ID"] = all_labs["HADM_ID"].astype(int)
adms["HADM_ID"] = adms["HADM_ID"].astype(int)

df_adm1 = adms.copy()

df_lab1 = all_labs.loc[all_labs["HADM_ID"].isin(df_adm1["HADM_ID"])]

nlabs = all_labs["label"].nunique()
gg = df_lab1.groupby(["HADM_ID", "label"]).size().reset_index().groupby("HADM_ID").size().sort_values(ascending=False)
freq_adms = gg[gg >= nlabs * 0.50].index.astype(int)

df_adm1 = df_adm1.loc[df_adm1["HADM_ID"].isin(freq_adms)]
df_lab1 = df_lab1.loc[df_lab1["HADM_ID"].isin(df_adm1["HADM_ID"])]

df_adm1 = df_adm1.sort_values("HADM_ID").reset_index(drop=True)

df_lab1 = df_lab1.sort_values(by=['HADM_ID', 'label', 'minute'])
def transform_urine(group):
    group = group.copy()
    group['VALUENUM'] = group['VALUENUM'].cumsum()
    group['VALUENUM'] = group['VALUENUM'] - group['VALUENUM'].iloc[0]
    return group

if "Urine" in df_lab1['label'].unique():
    print("[Checkpoint] Making urine feature cumulative...")
    df_lab1 = df_lab1.groupby(['HADM_ID', 'label'], group_keys=False).apply(
        lambda g: transform_urine(g) if g.name[1] == 'Urine' else g
    )

df_adm1.to_csv(f"{dataset}/sub_adm_target.csv")
df_lab1.to_csv(f"{dataset}/sub_adm_lab_out.csv")

print("[Checkpoint] Filtered admissions and labs saved.")

# ----------------- Time Series Discretisation -----------------
df_lab1["HADM_ID"] = df_lab1["HADM_ID"].astype(int)
df_lab1["hour"] = df_lab1["hour"].astype(int)
df_lab1.dropna(subset=["HADM_ID", "label", "hour", "VALUENUM"], inplace=True)

# Compute stats per lab label for normalisation
label_stats = df_lab1.groupby("label")["VALUENUM"].agg(["mean", "std"]).rename(columns={"mean": "label_mean", "std": "label_std"})
df_lab1 = df_lab1.merge(label_stats, on="label")

# Remove constant-valued lab labels
df_lab1 = df_lab1[df_lab1["label_std"] > 0]

# Discretise by interval
df_lab1["interval"] = (df_lab1["hour"] // interval) * interval

print("[Checkpoint] Discretising time series...")

# Pivot time series
df1 = (
    df_lab1
    .groupby(["HADM_ID", "label", "label_mean", "label_std", "interval"])["VALUENUM"]
    .mean()
    .unstack(level=-1)
    .interpolate(method='linear', axis=1)
    .ffill(axis=1)
    .bfill(axis=1)
    .reset_index()
)

# Normalise values
time_bins = list(range(0, 48, interval))
df1[time_bins] = df1[time_bins].subtract(df1["label_mean"], axis=0).divide(df1["label_std"], axis=0)

# Store and index
df1 = df1.drop(columns=["label_mean", "label_std"]).set_index(["HADM_ID", "label"])
df1.to_csv(f"{dataset}/discrete_ts{interval_suffix}_out.csv")

print("[Checkpoint] Discrete normalised time series saved.")

features_df = compute_ts_metrics_window(df_lab1)
features_df.set_index(["HADM_ID", "label"], inplace=True)
features_df = features_df.unstack(level=-1)

features_df.to_csv(f"{dataset}/ts_features_out.csv")

print("[Checkpoint] Temporal features computed.")



