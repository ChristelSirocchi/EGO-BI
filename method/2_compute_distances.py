import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances
from scipy.spatial.distance import squareform

# =====================================================================
# SETUP & CONFIGURATION
# =====================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "mimiciii", "eICU"])
parser.add_argument("--distance", type=str, default="euclidean")
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--min_overlap_pct", type=float, default=50) # 25% minimum lab overlap
args = parser.parse_args()

dataset = args.dataset
distance = args.distance
interval = args.interval
min_overlap_pct = args.min_overlap_pct/100

# Construct experiment name and suffix
interval_suffix = f"_{interval}" if interval != 1 else ""

output_dir = os.path.join(dataset, distance)
os.makedirs(output_dir, exist_ok=True)

print(f"Dataset: {dataset}, Distance: {distance}")

# Load discrete time series
df = pd.read_csv(f"{dataset}/discrete_ts{interval_suffix}_out.csv", index_col=[0, 1])

# Prepare for pairwise distance computation
adm_ids = df.index.get_level_values(0).unique().to_numpy()
labs = df.index.get_level_values(1).unique().to_numpy()
total_possible_labs = len(labs)

print(f"[Checkpoint] Data imported. Admissions: {len(adm_ids)}, Unique Labs: {total_possible_labs}")

df_unstacked = df.unstack(level=1) 
df_unstacked.columns = df_unstacked.columns.swaplevel(0, 1)
df_unstacked = df_unstacked.sort_index(axis=1)

X_matrix = df_unstacked.values.astype(np.float32)

lab_presence = ~np.isnan(X_matrix).reshape(len(adm_ids), total_possible_labs, -1).all(axis=2)

shared_labs_count = np.dot(lab_presence.astype(int), lab_presence.astype(int).T)

shared_labs_flattened = shared_labs_count[np.triu_indices(len(adm_ids), k=1)]

print("\n--- Distribution of Shared Labs Per Pair ---")
for p in [1, 5, 10, 25, 50, 75, 90]:
    print(f"{p}th Percentile: {np.percentile(shared_labs_flattened, p):.0f} shared labs")

print("Computing NaN-Euclidean pairwise distance matrix...")
D_matrix = nan_euclidean_distances(X_matrix, squared=False)

min_required_labs = int(total_possible_labs * min_overlap_pct)
print(f"Applying proximity constraint (Minimum required shared labs: {min_required_labs})")

# Identify pairs that drop below your baseline overlap requirement
invalid_pairs_mask = shared_labs_count < min_required_labs

# Calculate tracking metrics before setting elements to NaN
total_possible_pairs = (len(adm_ids) * (len(adm_ids) - 1)) // 2

# Exclude the diagonal (self-distances) from the invalid pairs count
np.fill_diagonal(invalid_pairs_mask, False)
removed_pairs_count = np.sum(invalid_pairs_mask) // 2 

print(f"--> Total unique patient pairs evaluated: {total_possible_pairs}")
print(f"--> Pairs removed due to low lab overlap (< {min_overlap_pct*100}%): {removed_pairs_count} ({ (removed_pairs_count/total_possible_pairs)*100 :.2f}%)")

# Apply the filter mask
D_matrix[invalid_pairs_mask] = np.nan

dists_condensed = squareform(D_matrix, force="tovector", checks=False)
# =====================================================================
# SAVE STORAGE ARTIFACTS
# =====================================================================
output_file = f"{output_dir}/mean_distances{interval_suffix}_{args.min_overlap_pct}_out.npz"
np.savez_compressed(output_file, mean=dists_condensed)

print(f"[Success] Scaled, constrained distance matrix saved cleanly to: {output_file}")