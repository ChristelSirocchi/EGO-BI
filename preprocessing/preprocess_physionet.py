import os
import pandas as pd
import numpy as np

# Define dataset name (used in file naming)
dataset = "physionet"

os.makedirs(f"{dataset}", exist_ok=True)

# Define the path to raw MIMIC-III CSV files
RAW_DATA_PATH = "/home/christel-sirocchi/GitHub/DATASETS/Physionet_2012/data" #'../physionet2012/data'  # <-- modify as needed
# dataset source https://www.physionet.org/content/challenge-2012/1.0.0/


# ---------------------- Load lab time series from set-a and set-b ------------------------

all_labs = []

for subset in ["set-a", "set-b", "set-c"]:
    folder = os.path.join(RAW_DATA_PATH, subset)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder {folder} not found. Please check the path.")

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            lab = pd.read_csv(filepath, header=None, skiprows=1)
            lab["HADM_ID"] = filename.replace(".txt", "")  # Use filename as HADM_ID
            all_labs.append(lab)

print(f"[Checkpoint] Data imported.")

# concatenate records
labs_df = pd.concat(all_labs, ignore_index=True)
labs_df.columns = ["TIME", "label", "VALUENUM", "HADM_ID"]

# extract ICU type and admission
icus = labs_df[labs_df["label"] == "ICUType"][["HADM_ID", "VALUENUM"]].rename(columns={"VALUENUM": "ADMISSION_TYPE"})
icus["HADM_ID"] = icus["HADM_ID"].astype(int)
icus["ADMISSION_TYPE"] = icus["ADMISSION_TYPE"].astype(int)

# convert time to numeri minutes
labs_df = labs_df[labs_df["TIME"].str.contains(":")]

labs_df[['hour', 'minute']] = labs_df['TIME'].str.split(":", expand=True).astype(int)

mask = labs_df["TIME"] == "48:00"
labs_df.loc[mask, "hour"] = 47
labs_df.loc[mask, "minute"] = 59

labs_df['minute'] = labs_df['hour'] * 60 + labs_df['minute']

print(labs_df["hour"].unique())

# retain lab variables only
labs_df = labs_df[["HADM_ID", "label", "minute", "hour", "VALUENUM"]]

lab_list = [
    'GCS', 'HR', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Temp', 'Urine', 'BUN', 'Creatinine', 'Glucose',
    'HCO3', 'HCT', 'Mg', 'Platelets', 'K', 'Na', 'WBC', 'FiO2', 'TroponinT', 'pH', 'PaCO2', 'PaO2',
    'MechVent', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'DiasABP', 'MAP', 'SysABP', 'SaO2',
    'Lactate', 'RespRate', 'Cholesterol', 'TroponinI'
]
labs_df = labs_df[labs_df["label"].isin(lab_list)]
labs_df = labs_df[labs_df["minute"] >= 0]
# neg value indicates missingness.
labs_df = labs_df[labs_df["VALUENUM"]>=0]
labs_df["HADM_ID"] = labs_df["HADM_ID"].astype("int")
labs_df = labs_df.sort_values(["HADM_ID", "label", "minute"])

# Save processed lab records
labs_df.to_csv(f"{dataset}/all_labs.csv", index=False)

print(f"[Checkpoint] Lab data processed.")

#------------------------ Process admission data -------------------------

outcomes_a = pd.read_csv(os.path.join(RAW_DATA_PATH, "Outcomes-a.txt"))
outcomes_b = pd.read_csv(os.path.join(RAW_DATA_PATH, "Outcomes-b.txt"))
outcomes_c = pd.read_csv(os.path.join(RAW_DATA_PATH, "Outcomes-c.txt"))
outcomes = pd.concat([outcomes_a, outcomes_b, outcomes_c], ignore_index=True)

outcomes.columns = ['HADM_ID', 'SAPS-I', 'SOFA', 'LOS_DAYS', 'SURVIVAL', 'HOSPITAL_EXPIRE_FLAG']
outcomes['HADM_ID'] = outcomes['HADM_ID'].astype(int)
outcomes['LONG_STAY'] = (outcomes["LOS_DAYS"] >= 14).astype(int)

adm_data = outcomes.merge(icus, on="HADM_ID", how="inner")
adm_data = adm_data[adm_data["HADM_ID"].isin(labs_df["HADM_ID"].unique())]
print(len(adm_data))
adm_data.reset_index(drop=True).to_csv(f"{dataset}/adm_target.csv")

print(f"[Checkpoint] Admission data processed.")

# ------------------ Extra - remove outliers ------------------
#task_dict = adm_data[["HADM_ID", TASK]].set_index("HADM_ID")[TASK].to_dict()
#labs_df["target"] = labs_df["HADM_ID"].map(task_dict)
#labs_df['z'] = labs_df.groupby(['label','target'])['VALUENUM'].transform(lambda x: np.abs((x - x.mean()) / x.std(ddof=0)))

labs_df['z'] = labs_df.groupby(['label'])['VALUENUM'].transform(lambda x: np.abs((x - x.mean()) / x.std(ddof=0)))
labs_df = labs_df[labs_df['z'] < 5]
labs_df.reset_index(drop = True).to_csv(f"{dataset}/all_labs_out.csv")





