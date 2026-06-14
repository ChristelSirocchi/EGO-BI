from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from pathlib import Path


# first preprocess with mimic3benchmarks

ROOT = Path("/home/christel-sirocchi/GitHub/EHR-ANALYSIS/mimic3-benchmarks/data")

TASK = "in-hospital-mortality"

dataset = "mimiciii"


all_dfs = []

for split in ["test","train"]:

    target = pd.read_csv(ROOT / TASK / split / "listfile.csv")
    first = target[target["stay"].str.contains("_episode1_")]

    for _, row in first.iterrows():

        label = row.y_true
        file = row.stay

        admid = file.replace("_episode1_timeseries.csv", "")

        data_id = (
            pd.read_csv(ROOT / TASK / split / file)
            .set_index("Hours")
        )

        melted_df = (
            data_id
            .reset_index()
            .melt(
                id_vars=["Hours"],
                var_name="label",
                value_name="VALUENUM"
            )
            .dropna()
            .sort_values("Hours")
        )

        melted_df = melted_df.rename(columns={"Hours": "time"})

        melted_df["HADM_ID"] = admid
        melted_df["minute"] = (melted_df["time"] * 60).astype(int)
        melted_df["hour"] = np.floor(melted_df["time"]).astype(int)

        all_dfs.append(melted_df)

labs_df = pd.concat(all_dfs, ignore_index=True)

gcs_verbal_map = {
    "5 Oriented": 5,
    "Oriented": 5,

    "4 Confused": 4,
    "Confused": 4,

    "3 Inapprop words": 3,
    "Inappropriate Words": 3,

    "2 Incomp sounds": 2,
    "Incomprehensible sounds": 2,

    "1 No Response": 1,
    "No Response": 1,
    "No Response-ETT": 1,

    "1.0 ET/Trach": np.nan  # mechanically ventilated / not testable
}

mask = labs_df["label"] == "Glascow coma scale verbal response"

labs_df.loc[mask, "VALUENUM"] = (
    labs_df.loc[mask, "VALUENUM"].map(gcs_verbal_map)
)

gcs_motor_map = {
    "6 Obeys Commands": 6,
    "Obeys Commands": 6,

    "5 Localizes Pain": 5,
    "Localizes Pain": 5,

    "4 Flex-withdraws": 4,
    "Flex-withdraws": 4,

    "3 Abnorm flexion": 3,
    "Abnormal Flexion": 3,

    "2 Abnorm extensn": 2,
    "Abnormal extension": 2,

    "1 No Response": 1,
    "No Response": 1,
    "No response": 1
}

mask = labs_df["label"] == "Glascow coma scale motor response"

labs_df.loc[mask, "VALUENUM"] = (
    labs_df.loc[mask, "VALUENUM"].map(gcs_motor_map)
)


gcs_eye_map = {
    "4 Spontaneously": 4,
    "Spontaneously": 4,

    "3 To speech": 3,
    "To Speech": 3,

    "2 To pain": 2,
    "To Pain": 2,

    "1 No Response": 1,
    "No Response": 1
}

mask = labs_df["label"] == "Glascow coma scale eye opening"

labs_df.loc[mask, "VALUENUM"] = (
    labs_df.loc[mask, "VALUENUM"].map(gcs_eye_map)
)

labs_df = labs_df.dropna()
labs_df.to_csv(f"{dataset}/all_labs.csv", index=False)



TASK = "in-hospital-mortality"

all_ids = []

for split in ["test", "train"]:
    
    df = pd.read_csv(ROOT / TASK / split / "listfile.csv").rename(columns={"y_true": TASK})

    mask = df["stay"].astype(str).str.contains("_episode1_", na=False)

    all_ids.append(df[mask])

targets_df = pd.concat(all_ids, ignore_index=True)

targets_df["HADM_ID"] = targets_df["stay"].str.removesuffix("_episode1_timeseries.csv")
targets_df = targets_df.set_index("HADM_ID")

# TASK = "length-of-stay"

# all_ids = []

# for split in ["test", "train"]:
    
#     df = pd.read_csv(ROOT / TASK / split / "listfile.csv").rename(columns={"y_true": TASK})

#     mask = df["stay"].isin(targets_df.stay) & (df["period_length"] == 48)

#     all_ids.append(df[mask])

# los = pd.concat(all_ids, ignore_index=True).drop(columns=["period_length"])
# los["HADM_ID"] = los["stay"].str.removesuffix("_episode1_timeseries.csv")
# los = los.set_index("HADM_ID").drop(columns="stay")

TASK = "phenotyping"

all_ids = []

for split in ["test", "train"]:
    
    df = pd.read_csv(ROOT / TASK / split / "listfile.csv").rename(columns={"y_true": TASK})

    mask = df["stay"].isin(targets_df.stay)

    all_ids.append(df[mask])

pheno = pd.concat(all_ids, ignore_index=True)
pheno["HADM_ID"] = pheno["stay"].str.removesuffix("_episode1_timeseries.csv")
pheno = pheno.set_index("HADM_ID").drop(columns="stay")

targets_df = targets_df.join(pheno).drop(columns = "stay")

targets_df = targets_df.rename(columns={'in-hospital-mortality':"HOSPITAL_EXPIRE_FLAG"})

targets_df["LOS_DAYS"] = np.floor(targets_df["period_length"] / 24).astype(int)

targets_df["MID_STAY"] = (targets_df["LOS_DAYS"] >=7).astype(int)

targets_df["LONG_STAY"] = (targets_df["LOS_DAYS"] >=14).astype(int)


pheno_cols = [
    'Acute and unspecified renal failure',
    'Acute cerebrovascular disease',
    'Acute myocardial infarction',
    'Cardiac dysrhythmias',
    'Chronic kidney disease',
    'Chronic obstructive pulmonary disease and bronchiectasis',
    'Complications of surgical procedures or medical care',
    'Conduction disorders',
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Diabetes mellitus with complications',
    'Diabetes mellitus without complication',
    'Disorders of lipid metabolism',
    'Essential hypertension',
    'Fluid and electrolyte disorders',
    'Gastrointestinal hemorrhage',
    'Hypertension with complications and secondary hypertension',
    'Other liver diseases',
    'Other lower respiratory disease',
    'Other upper respiratory disease',
    'Pleurisy; pneumothorax; pulmonary collapse',
    'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
    'Respiratory failure; insufficiency; arrest (adult)',
    'Septicemia (except in labor)',
    'Shock'
]

rename_map = {col: f"PHENO{i+1}" for i, col in enumerate(pheno_cols)}

targets_df = targets_df.rename(columns=rename_map)

targets_df.reset_index().to_csv(f"{dataset}/adm_target.csv")

labs_df['z'] = labs_df.groupby(['label'])['VALUENUM'].transform(lambda x: np.abs((x - x.mean()) / x.std(ddof=0)))
labs_df = labs_df[labs_df['z'] < 5]
labs_df.reset_index(drop = True).to_csv(f"{dataset}/all_labs_out.csv")