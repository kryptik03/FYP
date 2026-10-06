# MLOps-Driven Partial Discharge (PD) Detection Pipeline

> **KIE4002 Final Year Project** — An end-to-end, production-grade Machine Learning pipeline designed to synthesize, ingest, isolate, classify, and localize Partial Discharge (PD) signals in complex industrial environments using a 4-sensor UHF array.

---

## Project Overview

This Final Year Project bridges the gap between academic theory and industrial application by solving the **Overlap Problem** — the challenge of detecting multiple, simultaneous PD events masked by heavy structured noise.

The system operates on two parallel data tracks:

| Track | Origin Tag | Description |
|-------|------------|-------------|
| **Synthetic (HFSS)** | `sy` | Mathematically rigorous data driven by HFSS co-simulation via MATLAB |
| **Synthetic (Equation)** | `eq` | Mathematically synthesized PD pulses using standard equations (SEDO, DED, DEDO, SMG) |
| **Measured (UHF)** | `ms` | Augmented real-world measurements captured from physical UHF sensors |
| **Measured (Bearing)** | `cw` | CWRU bearing dataset used for robust domain adversarial training |

To manage the immense complexity of multi-stage processing, the pipeline employs a strict **Data Lineage Tracking System** (SQLite DAG) and a highly modular, **Task-Driven PyTorch Architecture**.

---

## Key Features

- **Physics-Based Synthesis** — A MATLAB engine utilises HFSS `.s2p` Touchstone files to mathematically distort simulated Gaussian pulses based on calculated TDOA and 3D spatial geometry. A global absolute noise floor is applied probabilistically at runtime to prevent SNR scaling bugs when multiple PD events overlap.

- **Autonomous Ground Truth Labeling** — Dynamically generates 7-column physics-tracking labels that link every sensor spike back to its originating physical event:
  ```
  [Scene_ID, Channel_ID, Class_ID, Pulse_Instance_ID, TOA_Index, Start_Idx, End_Idx]
  ```
  All indices are **0-indexed**.

- **Directed Acyclic Graph (DAG) Tracking** — A centralised SQLite database (`src/utils/lineage.db`) tracks the exact ancestry of every dataset and model weight, enabling safe experiment branching and pruning. The DB is **version-controlled in git** so that Colab training runs can push lineage updates directly.

- **Task-Driven PyTorch Factory** — Decouples `Dataset`, `Backbone`, `Head`, and `Task Objective` into reusable Lego-style blocks. No monolithic training loops — ever.

- **Cloud-Ready HDF5 Sharding** — Handles massive continuous time-series tensors safely by sharding data into manageable files and applying in-RAM 500x Max-Pool decimation to prevent GPU OOM errors.

- **Schema Consistency** — Both synthetic and measured data are forced into an identical HDF5 scene-based schema, allowing models trained on synthetic data to validate against real-world measurements without architectural changes.

- **Built-in Performance Evaluation** — `predict.py` automatically evaluates every inference run against ground-truth labels (Precision, Recall, F1, mean IoU, per-class detection recall, timing) and saves `metrics.json` alongside the predictions.

- **Training Time Telemetry** — `train.py` records start/end time, per-epoch wall-clock times, total duration, and the exact GPU model assigned by Colab (T4, V100, A100), saved to `data/performance_evaluation/training/<NodeID>_timing.json`.

---

## Implemented Models

### `exp09_vit_dann` — ViT + SupCon + DANN + Spherical DEC
- **Task:** Solves Latent Space Fragmentation using Physics Invariance, Distance Invariance, and Domain Agnosticism.
  - *Phase 1:* Instance-Linked Supervised Contrastive Learning (SupCon) + Domain Adversarial Pre-Training (GRL).
  - *Phase 2:* Spherical DEC Cluster Refinement + Semi-Supervised Pairwise constraints + DANN.
- **Input:** 2-Channel Complex Bispectrum (Magnitude + Phase), `2×128×128` image grids. Captures the physics-based phase-coupling fingerprint that is stable across distances and source environments.
- **Architecture:** Custom Mini-ViT (d_model=384, heads=6, depth=6, patch_size=16) + 4-Domain Gradient Reversal Layer (GRL).
- **Configuration:** `src/models/configs/exp09_vit_dann.yaml`
- **Output Routing:** Results are saved to `data/classification_output/exp09_dec/<YYYYMMDD_HHMMSS>_inf-<NodeID>/`.
- **Generated Artifacts:**
  - `predictions.h5` — Cluster assignments and soft probabilities
  - `metrics.json` — Evaluation metrics
  - `fig4_soft_q_heatmap.png` & `fig_time_vs_distance.png` — Visualization plots
  - `analysis_history.txt` — Audit log

### How to Use Exp09

**1. Data Preparation**
Run feature extraction first to generate 2-channel bispectrum shards:
```bash
python src/features/extract_bispectra_v2.py --input_dir data/raw/<source_dir> --parent_node_id <ID>
```
*Update the paths in `src/models/configs/exp09_vit_dann.yaml` to point to these new shards.*

**Note:** steps 2 and 3 are done in colab and may take a few hours
Reference colab notebook for tuning and training: https://colab.research.google.com/drive/1fqfXguS4gZBtrj4mDTzsi8Eqh8JCOLDH?usp=sharing

Reference Kaggle Datasets for tuning and training: 

https://www.kaggle.com/datasets/chiangchangzee/ccz-fyp
https://www.kaggle.com/datasets/chiangchangzee/ccz-fyp-features-only

**2. Hyperparameter Tuning**
Run the Optuna tuning script to find optimal learning rates:
```bash
python src/models/hyperparam_tuning/tune_exp09.py
```

**3. Training**
Train the multi-phase model via the universal orchestrator:
```bash
python src/models/train.py --config src/models/configs/exp09_vit_dann.yaml
```

**4. Inference**
Run inference using a trained checkpoint (e.g., node `k7vt`) with UMAP reduction and HDBSCAN clustering:
```bash
python src/models/predictions/predict_exp09.py \
    --checkpoint_id <node_id> \
    --grouping_mode channel_capped \
    --reduce_method umap \
    --reduce_dims 2 \
    --time_threshold 1e-05 \
    --dist_threshold 0.5 \
    --source data/features/bispectra_v2/<folder_name>:<domain_type>:<shards>
```
*(Or use the batch helper script `run_batch_predictions_exp09.py` for automated bulk runs).*

### Command Reference

For a complete list of commands—including DAG management, data ingestion, feature extraction, and batch inference—please refer to the [`PowerShell Commands.txt`](file:///d:/Zee_Documents/Studies/Uni/Sem_8/KIE4002_FYP/Git_Cloned_Code/FYP/PowerShell%20Commands.txt) file.

---

## Directory Architecture

> **Note:** `data/raw/` (the large MATLAB-generated HDF5 shards, ~6 GB) is `.gitignore`'d. All other `data/` subfolders (performance evaluation records, prediction outputs, etc.) **are** version-controlled. Model weights are stored in `models/weights/` and pushed to GitHub after cloud training.

```
FYP/
├── data/
│   ├── touchstone_files/             # HFSS .s2p Touchstone channel files
│   ├── unprocessed_bearing/          # Raw CWRU bearing dataset files
│   ├── unprocessed_measured/         # Raw oscilloscope .wfm files (pre-ingestion)
│   ├── raw/                          # [git-ignored] Large HDF5 shards
│   │   ├── cwru/                     # Birthplace of 'cw' HDF5 shards
│   │   ├── eqn_generated/            # Birthplace of 'sy' (equation-based) HDF5 shards
│   │   ├── measured/                 # Birthplace of 'ms' HDF5 shards
│   │   └── synthesised/              # Birthplace of 'sy' (HFSS-based) HDF5 shards
│   ├── isolation_output/
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   ├── features/
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   ├── classification_output/        # Predictions, metrics, and figures
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   │           ├── predictions.h5
│   │           ├── metrics.json
│   │           └── analysis_history.txt
│   ├── tdoa/
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   ├── localisation_output/
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   └── performance_evaluation/
│       ├── training/
│       │   └── <NodeID>_timing.json  # GPU name, epoch times, total duration
│       ├── classification/
│       ├── tdoa/
│       └── localisation/
│
├── models/                           # Saved Neural Network Artifacts
│   ├── weights/                      # .pt checkpoint files (Named by NodeID)
│   └── configuration_snapshots/      # config.yaml snapshots (Named by NodeID)
│
└── src/                              # All source code (version-controlled)
    ├── bearing_dataset_generation/   # Scripts to ingest CWRU data
    ├── equation_signals_dataset_generation/ # Mathematical signal generation
    ├── hfss_signal_dataset_generation/      # MATLAB HFSS synthesis scripts
    ├── ingestion/                    # .wfm / .csv to HDF5 conversion & augmentation
    ├── isolation/                    # Non-DL signal isolation algorithms
    ├── features/                     # Feature extraction algorithms (e.g., Bispectrum)
    ├── classification/               # Non-DL classification algorithms
    ├── obtain_tdoa/                  # TDOA math algorithms (e.g., Cross-Correlation)
    ├── localisation/                 # Spatial algorithms (e.g., PSO)
    ├── utils/
    │   ├── lineage_tracker.py        # SQLite DAG manager (CLI + Python API)
    │   └── lineage.db                # Master tracking database (git-tracked)
    └── models/                       # Deep Learning Hub
        ├── configs/                  # Experiment YAML configs (source of truth)
        ├── data/                     # PyTorch Datasets & Transforms
        ├── hyperparam_tuning/        # Optuna hyperparameter optimization scripts
        ├── models/
        │   ├── backbones/            # CNN, ViT backbones
        │   └── heads/                # YOLO, Projection, DANN heads
        ├── predictions/              # Inference scripts per experiment
        ├── tasks/                    # PyTorch Lightning-style training logic
        ├── train/                    # Training orchestrators per experiment
        ├── utils/                    # PyTorch utilities
        └── train.py                  # Universal DAG training orchestrator
```

---

## Pipeline Stages

```
[A] Data Generation / Ingestion
        |
        v
[B] Signal Isolation  ─── src/isolation/<method>/  or  src/models/
        |                  → data/isolation_output/<method>/
        v
[C] Feature Extraction ── src/features/<method>/
        |                  → data/features/<method>/
        v
[D] Signal Classification  src/classification/<method>/  or  src/models/
        |                  → data/classification_output/<method>/
        v
[E] TDOA Calculation ──── src/obtain_tdoa/<method>/
        |                  → data/tdoa/<method>/
        v
[F] Localisation ──────── src/localisation/<method>/
                           → data/localisation_output/<method>/
```

> **Downstream Routing Rule:** A model that jointly performs Isolation + Classification (e.g., `cnn_yolo1d`) saves ALL artifacts to `classification_output/`. Outputs are never scattered across multiple stage folders.

### Stage Summary

| Stage | Input | Output | Status |
|---|---|---|---|
| **A1. Synthesis (HFSS)** | `data/touchstone_files/` | `data/raw/synthesised/` | Done — 20 shards (ShmH) |
| **A2. Synthesis (Equation)** | `None` | `data/raw/eqn_generated/` | Implemented |
| **A3. Ingestion (UHF)** | `data/unprocessed_measured/` | `data/raw/measured/` | Implemented |
| **A4. Ingestion (CWRU)** | `data/unprocessed_bearing/` | `data/raw/cwru/` | Implemented |
| **B+D. Detection (DL)** | `data/raw/synthesised/` | `data/classification_output/cnn_yolo1d/` | **Active** — model QIRE |
| **E. TDOA** | `data/classification_output/` | `data/tdoa/<method>/` | Pending |
| **F. Localisation** | `data/tdoa/` | `data/localisation_output/<method>/` | Pending |

---

## HDF5 Data Schema

All data (synthetic and measured) conforms to a shared **Scene-Based** schema:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `/scenes` | `(num_scenes, 4, N_points)` | Multi-channel raw waveform windows |
| `/labels` | `(7, num_pulses)` | Physics-tracking label matrix (transposed: h5py reads MATLAB column-major as `(7, N)`) |

**Label rows (0-indexed):**

| Row | Field | Description |
|-----|-------|-------------|
| 0 | `Scene_ID` | Scene index within the shard |
| 1 | `Channel_ID` | Sensor channel (0–3) |
| 2 | `Class_ID` | 0 = PD1, 1 = PD2 |
| 3 | `Pulse_Instance_ID` | Global pulse counter across the shard |
| 4 | `TOA_Index` | Time-of-Arrival sample index (0-indexed) |
| 5 | `Start_Idx` | Pulse start sample index (0-indexed) |
| 6 | `End_Idx` | Pulse end sample index (0-indexed) |

---

## Data Lineage & DAG Tracking

### Naming Convention

```
YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]
```

| Part | Description |
|------|-------------|
| `YYYYMMDD_HHMMSS` | Birth datetime of the original raw dataset — **inherited by all descendants** |
| `Origin` | `sy` (Synthesised) or `ms` (Measured) |
| `RootID` | 4-char ID of the raw dataset root (e.g., `ShmH`) |
| `NodeID` | 4-char ID unique to this specific process run (e.g., `QIRE`) |

**Example:** `20260427_170034_sy-ShmH-QIRE` — synthesised dataset, root `ShmH`, training run `QIRE`.

### SQLite Master Ledger (`src/utils/lineage.db`)

The DB is git-tracked. After each Colab training run, push `src/utils/lineage.db` alongside the weights.

```bash
# Visualise the experiment tree
python src/utils/lineage_tracker.py --action visualize --root_id ShmH

# Safely prune a dead-end node (leaf nodes only)
python src/utils/lineage_tracker.py --action prune --node_id <NodeID>
```

**Current DAG:**
```
[ROOT] ShmH [generation : generation]
  +-- [NODE] QIRE [classification : cnn_yolo1d]   <- trained model (25 epochs)
      +-- [NODE] gDZx [classification : cnn_yolo1d] <- inference run
```

### Registering a Node from Python

```python
from src.utils.lineage_tracker import register_process

register_process(
    parent_id="ShmH",
    stage="classification",
    method="cnn_yolo1d",
    folder_path="models/weights",
    appended_history="...",
    force_node_id="QIRE",   # omit to auto-generate
)
```

---

## Deep Learning Architecture

All experiments live in `src/models/`. Strictly decomposed into 5 components:

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **Config** | `src/models/configs/*.yaml` | Single source of truth for all hyperparameters |
| **Dataset** | `src/models/data/` | HDF5 I/O, YOLO target grid construction, Max-Pool decimation |
| **Backbone** | `src/models/models/backbones/` | Task-agnostic feature extraction |
| **Head** | `src/models/models/heads/` | Task-specific output projection |
| **Task** | `src/models/tasks/` | Loss function, optimizer, training/validation steps, checkpoint saving |

`train.py` and `predict.py` are the only user-facing entry points. They never change regardless of which backbone or head is swapped in.

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

| Dependency | Min Version |
|------------|-------------|
| `torch` | 2.1.0 |
| `numpy` | 1.26.0 |
| `h5py` | 3.10.0 |
| `PyYAML` | 6.0.0 |

> MATLAB is additionally required for synthetic data generation.

---

### Step 1 — Initialise the Lineage Database

```bash
python src/utils/lineage_tracker.py --action init
```

Run once. Creates `src/utils/lineage.db`.

---

### Step 2 — Generate Synthetic Data (MATLAB)

Open `src/generation/signal_generation.m` in MATLAB and run it. Reads HFSS `.s2p` Touchstone files, synthesises physics-distorted PD pulses, saves HDF5 shards to `data/raw/synthesised/`, and registers the Root Node in the lineage DB.

---

### Step 3 — Train on Google Colab

1. Clone the repo and mount Kaggle Dataset (`data/raw/synthesised/`)
2. Run:
   ```bash
   python src/models/train.py --config src/models/configs/exp01_yolo1d.yaml
   ```
3. After training, push back to git:
   ```bash
   git add models/weights/ models/configuration_snapshots/
   git add src/utils/lineage.db
   git add data/performance_evaluation/training/
   git commit -m "Training run NodeID <XXXX>"
   git push
   ```
4. Pull locally: `git pull` — weights and lineage DB arrive together.

For a quick CPU-only sanity check (2 shards, 2 epochs):
```bash
python src/models/train.py --config src/models/configs/exp01_yolo1d.yaml --smoke_test
```

---

### Step 4 — Inference & Evaluation

```bash
python src/models/predict.py \
    --checkpoint QIRE \
    --shards 17 18 19 20 \
    --threshold 0.5 \
    --iou_threshold 0.5
```

Automatically runs inference **and** evaluates against ground-truth labels. Outputs saved to `data/classification_output/cnn_yolo1d/<timestamp>/`:
- `predicted_boxes.h5` — bounding box predictions
- `predicted_classes.h5` — class predictions
- `metrics.json` — Precision, Recall, F1, mean IoU, per-class breakdown, inference time
- `analysis_history.txt` — one-line audit summary

**Sample results (QIRE, 4 val shards, IoU threshold 0.5):**

| Metric | Value |
|--------|-------|
| Precision | 0.740 |
| Recall | 0.777 |
| F1 | 0.758 |
| Mean IoU (TPs) | 0.803 |
| Cls Accuracy (TPs) | 1.000 |
| PD1 Detection Recall | 0.882 |
| PD2 Detection Recall | 0.675 |

---

## Architectural Rules

| Rule | Description |
|------|-------------|
| **Downstream Routing Rule** | Joint models (isolation + classification) save ALL artifacts to the most downstream stage folder. |
| **No Monolithic Models** | `Dataset + Backbone + Head + Task` must always be separate files. `train.py` never changes. |
| **DAG-Only Deletions** | Artifacts must be pruned via `lineage_tracker.py`, never deleted manually. |
| **No Hardcoded Hyperparameters** | All parameters live in YAML configs under `src/models/configs/`. |
| **Raw Data Git-Ignored** | `data/raw/` is git-ignored (~6 GB). All other `data/` subfolders are tracked. |
| **Lineage DB in Git** | `src/utils/lineage.db` is version-controlled so Colab can push lineage updates via `git push`. |

---

## Cloud Workflow

| Concern | Tool |
|---------|------|
| **Code + lineage versioning** | GitHub (`src/`, `models/`, `data/` except raw shards) |
| **Heavy data storage** | Kaggle Datasets (`data/raw/synthesised/` HDF5 shards) |
| **Model training** | Google Colab GPU instances |
| **Artifact return** | `git push` from Colab — weights, lineage DB, timing JSON, prediction outputs |

---

## License

This project is submitted in partial fulfilment of the requirements for the Bachelor of Engineering degree at the Faculty of Engineering, Universiti Malaya. All rights reserved.
