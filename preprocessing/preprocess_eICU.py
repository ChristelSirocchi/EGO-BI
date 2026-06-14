from pathlib import Path
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# first preprocessed with eICU benchmark
dataset = "eICU"

os.makedirs(f"{dataset}", exist_ok=True)

base_path = Path("/home/christel-sirocchi/GitHub/EHR-ANALYSIS/eICU_Benchmark/extracted_data")

all_dfs = []

for folder in base_path.iterdir():
    if folder.is_dir():
        file_path = folder / "pats.csv"
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)

final_df['unitdischargeoffset'] = final_df['unitdischargeoffset']/(1440)

final_df.to_csv(f"{dataset}/all_eICU_patient_summary.csv")

# filter suitable subjects

mortality_sub = final_df[(final_df.gender != 0) &
         (final_df.hospitaldischargestatus!=2) &
         (final_df.unitdischargeoffset>=2)
        ]

sub_ids = mortality_sub["patientunitstayid"].tolist()

all_labs = []

for pid in tqdm(sub_ids):

    folder = base_path / str(pid)

    if folder.is_dir():

        for file in ["lab.csv", "nc.csv"]:
            file_path = folder / file

            if file_path.exists():
                df = pd.read_csv(file_path)
                all_labs.append(df.dropna())

final_labs = pd.concat(all_labs, ignore_index=True)
final_labs.columns = ["HADM_ID", "minute","label", "VALUENUM"]
final_labs["hour"] = final_labs["minute"] // 60
final_labs = final_labs[(final_labs["minute"]>=0) & (final_labs["hour"]<48)]
final_labs.to_csv(f"{dataset}/all_labs.csv", index=False)

target_df = mortality_sub[['patientunitstayid', 'gender', 'age', 'hospitaldischargestatus', 'unitdischargeoffset']]
target_df.columns = ["HADM_ID", "gender","age","HOSPITAL_EXPIRE_FLAG", "LOS"]
target_df["LOS_DAYS"] = np.floor(target_df["LOS"]).astype(int)
target_df["MID_STAY"] = (target_df["LOS_DAYS"] >=7).astype(int)
target_df["LONG_STAY"] = (target_df["LOS_DAYS"] >=14).astype(int)

target_df.to_csv(f"{dataset}/adm_target.csv")

final_labs['z'] = final_labs.groupby(['label'])['VALUENUM'].transform(lambda x: np.abs((x - x.mean()) / x.std(ddof=0)))
labs_df = final_labs[final_labs['z'] < 5]
labs_df.to_csv(f"{dataset}/all_labs_out.csv")