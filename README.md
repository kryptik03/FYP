# MLOps-Driven Partial Discharge (PD) Detection Pipeline

> **KIE4002 Final Year Project** — An end-to-end, production-grade Machine Learning pipeline designed to synthesize, ingest, isolate, classify, and localize Partial Discharge (PD) signals in complex industrial environments using a 4-sensor UHF array.

---

## 📖 Project Overview

This Final Year Project bridges the gap between academic theory and industrial application by solving the **Overlap Problem** — the challenge of detecting multiple, simultaneous PD events masked by heavy structured noise.

The system operates on two parallel data tracks:

| Track | Origin Tag | Description |
|-------|------------|-------------|
| 🔬 **Synthetic** | `sy` | Mathematically rigorous data driven by HFSS co-simulation via MATLAB |
| 📡 **Measured** | `ms` | Augmented real-world measurements captured from physical UHF sensors |

To manage the immense complexity of multi-stage processing, the pipeline employs a strict **Data Lineage Tracking System** (SQLite DAG) and a highly modular, **Task-Driven PyTorch Architecture**.

---

## ✨ Key Features

- **Physics-Based Synthesis** — A MATLAB engine utilises HFSS `.s2p` Touchstone files to mathematically distort simulated Gaussian pulses based on calculated TDOA and 3D spatial geometry. A global absolute noise floor is applied probabilistically at runtime to prevent SNR scaling bugs when multiple PD events overlap.

- **Autonomous Ground Truth Labeling** — Dynamically generates 7-column physics-tracking labels that link every sensor spike back to its originating physical event:
  ```
  [Scene_ID, Channel_ID, Class_ID, Pulse_Instance_ID, TOA_Index, Start_Idx, End_Idx]
  ```

- **Directed Acyclic Graph (DAG) Tracking** — A centralised SQLite database (`data/lineage.db`) tracks the exact ancestry of every dataset and model weight, enabling safe experiment branching and pruning.

- **Task-Driven PyTorch Factory** — Decouples `Dataset`, `Backbone`, `Head`, and `Task Objective` into reusable Lego-style blocks. No monolithic training loops — ever.

- **Cloud-Ready HDF5 Sharding** — Handles massive continuous time-series tensors safely by sharding data into manageable files and applying in-RAM decimation to prevent GPU OOM errors.

- **Schema Consistency** — Both synthetic and measured data are forced into an identical HDF5 scene-based schema, allowing models trained on synthetic data to validate against real-world measurements without architectural changes.

---

## 🗂️ Directory Architecture (Master Blueprint v3.0)

> **Note:** `data/` is strictly `.gitignore`'d. Only `src/` and `models/weights/` are version-controlled. Heavy data artifacts are stored on Kaggle Datasets; model weights are downloaded after cloud training and saved to `models/weights/`.

```
FYP/
├── data/
│   ├── touchstone_files/             # HFSS .s2p Touchstone channel files
│   ├── unprocessed_measured/         # Raw oscilloscope .wfm files (pre-ingestion)
│   ├── raw/
│   │   ├── synthesised/              # Birthplace of 'sy' HDF5 shards
│   │   └── measured/                 # Birthplace of 'ms' HDF5 shards
│   ├── isolation_output/             # Bounding box predictions (.h5)
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   ├── features/                     # Extracted feature matrices (e.g., CWT)
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   ├── classification_output/        # Class predictions (& boxes if joint task)
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   ├── tdoa/                         # Calculated inter-sensor time delays
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   ├── localisation_output/          # 3D spatial coordinates of PD sources
│   │   └── <method>/
│   │       └── YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   ├── performance_evaluation/       # Metrics, plots, confusion matrices (CSV/PNG)
│   │   ├── isolation/
│   │   │   └── <method>/YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   │   ├── classification/
│   │   │   └── <method>/YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   │   ├── tdoa/
│   │   │   └── <method>/YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   │   └── localisation/
│   │       └── <method>/YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]/
│   └── lineage.db                    # SQLite Master Tracking Database
│
├── models/                           # Saved Neural Network Artifacts
│   ├── weights/                      # .pt checkpoint files (Named by NodeID)
│   └── configuration_snapshots/      # config.yaml snapshots (Named by NodeID)
│
└── src/                              # All source code (version-controlled)
    ├── generation/                   # MATLAB HFSS synthesis scripts
    ├── ingestion/                    # .wfm / .csv → HDF5 conversion & augmentation
    ├── isolation/                    # Non-DL signal isolation algorithms
    │   └── <method>/
    ├── features/                     # Feature extraction algorithms
    │   └── <method>/
    ├── classification/               # Non-DL classification algorithms (e.g., DBSCAN)
    │   └── <method>/
    ├── obtain_tdoa/                  # TDOA math algorithms (e.g., Cross-Correlation)
    │   └── <method>/
    ├── localisation/                 # Spatial algorithms (e.g., PSO)
    │   └── <method>/
    ├── utils/                        # Shared utilities (lineage_tracker.py)
    └── models/                       # ⚡ DEEP LEARNING HUB
        ├── configs/                  # Experiment YAML configs (one per experiment)
        ├── data/                     # PyTorch Datasets & DataLoaders
        ├── models/
        │   ├── backbones/            # Pure feature extractor architectures
        │   └── heads/                # Task-specific output projectors
        ├── tasks/                    # Task logic: loss functions & training steps
        ├── utils/                    # PyTorch metrics helpers
        ├── train.py                  # Universal DAG training orchestrator
        └── predict.py                # Inference + automatic performance evaluation
```

---

## 🔄 Pipeline Stages

The data flows sequentially through 6 stages. Each stage reads from the previous stage's output folder and writes to its own, maintaining strict segregation.

```
[A] Data Generation / Ingestion
        │
        ▼
[B] Signal Isolation  ──────────────────────── src/isolation/<method>/
        │                                       src/models/  (DL methods)
        ▼                                       → data/isolation_output/<method>/
[C] Feature Extraction ─────────────────────── src/features/<method>/
        │                                       → data/features/<method>/
        ▼
[D] Signal Classification ──────────────────── src/classification/<method>/
        │                                       src/models/  (DL methods)
        ▼                                       → data/classification_output/<method>/
[E] TDOA Calculation ───────────────────────── src/obtain_tdoa/<method>/
        │                                       → data/tdoa/<method>/
        ▼
[F] Localisation ───────────────────────────── src/localisation/<method>/
                                                → data/localisation_output/<method>/
```

> **Note — "Detection" = Isolation + Classification.** A model that jointly performs both tasks (e.g., a detection model) outputs ALL its artifacts to `classification_output/` per the **Downstream Routing Rule**. Outputs are never scattered across multiple stage folders.

### Stage Details

| Stage | Input Source | Output Destination | Notes |
|---|---|---|---|
| **A1. Synthesis** | `data/touchstone_files/` | `data/raw/synthesised/` | MATLAB + HFSS; registers Root Node in SQLite |
| **A2. Ingestion** | `data/unprocessed_measured/` | `data/raw/measured/` | `.wfm` → HDF5; augmentation & scene assembly |
| **B. Isolation** | `data/raw/` | `data/isolation_output/<method>/` | Threshold-based or DL model |
| **C. Features** | `data/isolation_output/` | `data/features/<method>/` | CWT, STFT, etc. |
| **D. Classification** | `data/features/` | `data/classification_output/<method>/` | DBSCAN, DL models, etc. |
| **E. TDOA** | `data/classification_output/` | `data/tdoa/<method>/` | Cross-correlation per Pulse ID |
| **F. Localisation** | `data/tdoa/` | `data/localisation_output/<method>/` | Particle Swarm Optimization (PSO) |

---

## 📦 HDF5 Data Schema

All data (synthetic and measured) conforms to a shared **Scene-Based** schema, stored as sharded `.h5` files:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `/scenes` | `(num_scenes, 4, N_points)` | Multi-channel raw waveform windows |
| `/labels` | `(num_pulses, 7)` | Physics-tracking label matrix |

**Label columns:** `[Scene_ID, Channel_ID, Class_ID, Pulse_Instance_ID, TOA_Index, Start_Idx, End_Idx]`

Each HDF5 file also carries an `analysis_history` attribute (or a companion `analysis_history.txt`/`.json`), acting as a continuous audit log of every algorithm the data has been through since its birth.

---

## 🧬 Data Lineage & DAG Tracking

### Naming Convention

Every dynamically generated output folder uses a strict naming standard:

```
YYYYMMDD_HHMMSS_[Origin]-[RootID]-[NodeID]
```

| Part | Description |
|------|-------------|
| `YYYYMMDD_HHMMSS` | Birth datetime of the original raw dataset — **inherited by all downstream nodes** for chronological OS sorting |
| `Origin` | `sy` (Synthesised) or `ms` (Measured) |
| `RootID` | 4-character alphanumeric ID of the raw dataset root (e.g., `Xa3A`) |
| `NodeID` | 4-character alphanumeric ID of the specific process applied (e.g., `8Hh7`) |

**Example:** `20260416_175200_sy-Xa3A-8Hh7` → synthesised dataset born on April 16 2026, root `Xa3A`, currently at processing stage `8Hh7`.

Because each `NodeID` is unique per process run, output folders never collide — old results are never overwritten.

### SQLite Master Ledger (`src/utils/lineage.db`)

The `nodes` table schema:

| Column | Description |
|--------|-------------|
| `node_id` | **Primary Key.** 4-char alphanumeric unique to this process run |
| `parent_id` | `node_id` of the direct predecessor (`"NONE"` for root datasets) |
| `root_id` | `node_id` of the original raw dataset — enables instant family tree queries |
| `origin` | `"sy"` or `"ms"` |
| `stage` | Pipeline stage (e.g., `"generation"`, `"isolation"`, `"classification"`) |
| `method` | Specific algorithm/model (e.g., `"dynamic_threshold"`, `"cnn_yolo1d"`) |
| `folder_path` | Relative path to the physical output directory on disk |
| `nickname` | Human-readable label assigned at dataset birth; inherited by all children |
| `timestamp` | Node creation time (`YYYYMMDD_HHMMSS`) |
| `history_log` | Append-only full ancestry log |

### Branching Example

One raw dataset can spawn multiple parallel experiments simultaneously:

```
Xa3A (Root: Synthetic Dataset)
 ├── 2Jw9  [isolation]  dynamic_threshold
 ├── bYj7  [isolation]  wavelet
 └── xR42  [isolation]  deep_learning_model
```

### Pruning Protocol

Before any node is deleted, the ledger checks for downstream dependants:
- ✅ **Leaf node** (no children) → confirm `YES` → deletes physical folder → drops DB row.
- ❌ **Has children** → pruning is **blocked** to protect pipeline integrity.

### Registering a New Node from Python

```python
from src.utils.lineage_tracker import register_process

new_node = register_process(
    parent_id="Xa3A",
    stage="classification",
    method="cnn_yolo1d",
    folder_path="data/classification_output/cnn_yolo1d/20260416_175200_sy-Xa3A-8Hh7",
    appended_history="Isolation & Classification via YOLO1D at 2026-04-19 14:00:00"
)
```

---

## 🧠 Deep Learning Architecture (Task-Driven Factory)

All DL experiments live in `src/models/`. The architecture is strictly decomposed into 5 components — no monolithic scripts.

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **Config** | `src/models/configs/*.yaml` | Defines dataset, backbone, head, task, and all hyperparameters. No hardcoded values. |
| **Dataset** | `src/models/data/` | Modal `Dataset` classes per input type (raw canvas, pre-cut boxes, features). Shared base handles HDF5 I/O safely. |
| **Backbone** | `src/models/models/backbones/` | Pure feature extractors. Agnostic to the downstream task. |
| **Head** | `src/models/models/heads/` | Task-specific output projectors (detection, classification, TDOA regression, etc.). |
| **Task** | `src/models/tasks/` | Encapsulates the model, optimizer, and loss function. One `task_*.py` per objective (e.g., `task_detection.py`, `task_classification.py`). |

`train.py` is the **only** entry point — it reads the YAML, builds the factory components, runs the training loop, and saves weights + config snapshots under the generated `NodeID`.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

| Dependency | Version |
|------------|---------|
| `torch` | ≥ 2.1.0 |
| `numpy` | ≥ 1.26.0 |
| `h5py` | ≥ 3.10.0 |
| `PyYAML` | ≥ 6.0.0 |

> MATLAB (with the MATLAB Engine for Python) is additionally required for the synthetic data generation stage.

---

### Step 1 — Initialise the Master Ledger

```bash
python src/utils/lineage_tracker.py --action init
```

Run this **once** before generating any data. Creates `data/lineage.db`.

---

### Step 2 — Generate Synthetic Data (MATLAB)

Navigate to `src/generation/` and run the master orchestrator script in MATLAB:

```
signal_generation.m
```

This script reads HFSS `.s2p` Touchstone files, synthesises physics-distorted Gaussian PD pulses across 4 UHF channels, saves HDF5 shards to `data/raw/synthesised/`, and issues a Python system call to register the Root Node in the SQLite ledger.

---

### Step 3 — Ingest Measured Data (Python)

```bash
python src/ingestion/<script>.py --input data/unprocessed_measured/ --output data/raw/measured/
```

Cleans, normalises, augments, and converts raw `.wfm`/`.csv` oscilloscope files into the same scene-based HDF5 schema as synthetic data.

---

### Step 4 — Train a Deep Learning Model

Create or select a YAML config in `src/models/configs/`, then run:

```bash
python src/models/train.py --config configs/<experiment>.yaml
```

Weights are saved to `models/weights/model_<NodeID>.pt` and the config snapshot to `models/configuration_snapshots/config_<NodeID>.yaml`.

---

### Step 5 — Run Inference & Evaluation

```bash
python src/models/predict.py --config configs/<experiment>.yaml --node_id <NodeID>
```

Outputs (Precision, Recall, F1, mIoU, confusion matrices, plots) are saved to `data/performance_evaluation/<stage>/<method>/<NodeFolder>/` and registered in the lineage ledger.

---

## 🔍 Lineage Tracker CLI Reference

```bash
# Initialise the database
python src/utils/lineage_tracker.py --action init

# Visualise the experiment tree from a root dataset
python src/utils/lineage_tracker.py --action visualize --root_id Xa3A

# Safely prune a dead-end experiment node (leaf nodes only)
python src/utils/lineage_tracker.py --action prune --node_id 8Hh7
```

---

## 📐 Architectural Rules & Constraints

| Rule | Description |
|------|-------------|
| **Downstream Routing Rule** | A model performing multiple sequential tasks (e.g., Isolation → Classification) saves ALL its artifacts to the **most downstream** stage folder. No output scattering. |
| **No Monolithic Models** | PyTorch architectures must be built as decoupled components: `Dataset + Backbone + Head + Task`. The training loop in `train.py` must never change. |
| **DAG-Only Deletions** | Experiment artifacts must always be pruned via `lineage_tracker.py`, never deleted manually. |
| **Metadata Injection** | Every output folder carries an `analysis_history` attribute or companion file detailing the exact algorithmic ancestry of the data. |
| **No Hardcoded Hyperparameters** | All model and training parameters live in YAML config files under `src/models/configs/`. |
| **Data Gitignore** | `data/` and `models/weights/` are strictly `.gitignore`'d. Code is on GitHub; heavy artifacts are on Kaggle Datasets. |

---

## ☁️ Cloud Workflow

| Concern | Tool |
|---------|------|
| **Code versioning** | GitHub (`src/` only) |
| **Heavy data storage** | Kaggle Datasets (HDF5 shards) |
| **Model training** | Google Colab / cloud GPU instances |
| **Artifact return** | Only `models/weights/` and `models/configuration_snapshots/` are returned to local storage |

---

## 📄 License

This project is submitted in partial fulfilment of the requirements for the Bachelor of Engineering degree at the Faculty of Engineering, Universiti Malaya. All rights reserved.
