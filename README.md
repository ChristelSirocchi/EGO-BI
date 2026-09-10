# EGO-BI: Ego-network Boundary Importance for Clinical Trajectories

This repository implements a boundary-aware feature analysis framework for identifying variables that drive outcome divergence in clinically similar patient profiles. The method combines patient similarity modelling, K-nearest neighbour (KNN) neighbourhood extraction, and constrained tree-based learning on boundary-derived feature differences.

The pipeline is designed for longitudinal ICU datasets and is evaluated on PhysioNet 2012, MIMIC-III, and eICU.

---

## Repository Structure

---

## 1. Preprocessing

Scripts in `preprocessing/` prepare raw ICU datasets into structured longitudinal representations suitable for similarity computation and modelling.

### `preprocess_physionet.py`
- Loads and preprocesses the PhysioNet 2012 dataset.
- Handles missingness, temporal alignment, and feature standardisation.
- Outputs patient-level longitudinal trajectories.

### `preprocess_mimic_iii.py`
- Processes MIMIC-III ICU data into structured time series format.
- Extracts clinical variables and aligns them across time windows.
- Produces cleaned feature matrices for downstream analysis.

### `preprocess_eICU.py`
- Prepares the eICU dataset for modelling.
- Performs cohort filtering and feature harmonisation.
- Generates comparable longitudinal representations across patients.

---

## 2. Method

Scripts in `method/` implement the core boundary-aware learning framework.

### `1_prepare_data.py`
- Constructs patient-level representations from preprocessed data.
- Defines feature spaces for similarity computation.
- Prepares inputs for distance and neighbourhood modelling.

### `2_compute_distances.py`
- Computes pairwise patient distances in the embedding space.
- Supports multiple distance metrics for sensitivity analysis.
- Outputs similarity matrices used for KNN graph construction.

### `3_apply_knn_fi.py`
- Builds K-nearest neighbour (KNN) ego-networks for each patient.
- Identifies boundary cases with divergent outcomes.
- Computes local feature importance scores based on neighbourhood contrasts.

### `builder.py`
- Constructs boundary datasets from KNN subgraphs.
- Transforms raw trajectories into weighted feature-difference representations.
- Prepares training data for constrained tree ensembles.

### `model.py`
- Implements the constrained tree ensemble model.
- Learns feature interactions from boundary-derived datasets.
- Produces interpretable rules associated with outcome divergence.

### `utils.py`
- Contains helper functions for data processing, evaluation, and feature manipulation.
- Shared utilities used across the method pipeline.

---

## 3. Validation

Scripts in `validation/` evaluate the robustness, stability, and interpretability of the method.

### `compute_hardness_performance.py`
- Computes per-patient hardness from out-of-fold prediction confidence of an unconstrained model.
- Saves to `results/hardness/{dataset}/{exp_name}/performance_hardness_oof.csv`.

### `compute_hardness_disjunct.py`
- Computes per-patient hardness from decision-tree disjuncts (disjunct size / tree depth), independent of the KNN/trajectory-distance construction used by the main method.
- Saves to `results/hardness/{dataset}/{exp_name}/disjunct_hardness_oof.csv`.

### `prune.py`
- Progressively drops the easiest training samples per fold and retrains, tracking how feature importance shifts as only harder cases remain.
- `--criterion {performance, disjunct}` selects which precomputed hardness file to prune by (run the matching `compute_hardness_*.py` first); `--disjunct-measure {DS, TD}` selects the disjunct sub-measure.

### `train_linear.py`
- Trains linear baseline models on the same boundary datasets.
- Provides interpretable benchmark for comparison with tree ensemble method.

---

## Method Overview

The framework operates in three main steps:

1. **Patient embedding and similarity computation**
   - Longitudinal trajectories are embedded into a similarity space.

2. **Local neighbourhood construction**
   - KNN ego-networks identify clinically similar patients with divergent outcomes.

3. **Boundary learning**
   - Feature differences within neighbourhoods are used to train constrained tree models that identify divergence-driving variables.

---

## Output

The framework produces:
- Boundary-aware feature importance scores
- Interpretable decision rules
- Stability and robustness metrics
- Performance comparisons against baselines

---

## Datasets

- PhysioNet 2012
- MIMIC-III
- eICU

