"""
tune_exp09.py
=============
Optuna hyperparameter tuning for Exp09:
  ViT + Instance-SupCon + 4-Domain DANN (GRL) + Semi-Supervised DEC
  on 2-Channel Complex Bispectrum features.

KEY DIFFERENCES vs tune_exp08.py
-----------------------------------
1.  Uses DECDataset_Exp09 (2-channel bispectra, cross-sensor pairing, domain labels).
2.  Uses SupConDECTask_Exp09 (Instance-SupCon + DANN + DEC).
3.  dann_weight added to the search space (log-uniform [0.01, 1.0]).
4.  Phase 1 batch unpacking updated for Exp09 tuple format.
5.  GRL lambda held at dann_lambda_max/2 during tuning (fixed, not ramped)
    to keep trials comparable to each other.

HARDCODED CONSTANTS (not adjustable at runtime)
------------------------------------------------
- Tuning epochs: Phase1=3, Phase2=3 (short for high trial throughput)
- Dataset subset: 300 train / 100 val samples per source
- GRL lambda during tuning: dann_lambda_max / 2 (half-ramp, stable estimate)

Usage:
    python src/models/hyperparam_tuning/tune_exp09.py \\
        --config src/models/configs/exp09_vit_dann.yaml
"""

import os
import sys
import copy
import json
import argparse
import random
import string
from datetime import datetime

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

try:
    import optuna
except ImportError:
    print("[Error] Optuna is not installed. Run: pip install optuna")
    sys.exit(1)

# Project root into sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.models.data.dataset_exp09 import DECDataset_Exp09
from src.models.tasks.task_exp09   import SupConDECTask_Exp09


# ---------------------------------------------------------------------------
# Hardcoded tuning constants
# ---------------------------------------------------------------------------

TUNE_TRAIN_MAX_PER_SOURCE = 300   # Samples per source for train trials
TUNE_VAL_MAX_PER_SOURCE   = 100   # Samples per source for val trials
TUNE_EPOCHS_P1            = 3     # Short Phase 1 to probe LR + dann_weight sensitivity
TUNE_EPOCHS_P2            = 3     # Short Phase 2 to measure DEC convergence tendency


# ---------------------------------------------------------------------------
# Subset helper
# ---------------------------------------------------------------------------

def subset_dataset_evenly(dataset, max_per_source: int):
    """Draw at most `max_per_source` samples per shard directory."""
    from collections import defaultdict
    source_to_indices = defaultdict(list)
    for i, item in enumerate(dataset.index):
        source_dir = os.path.dirname(item[0])   # shard_path parent dir
        source_to_indices[source_dir].append(i)

    selected = []
    for _, indices in source_to_indices.items():
        if len(indices) > max_per_source:
            selected.extend(
                np.random.choice(indices, max_per_source, replace=False).tolist()
            )
        else:
            selected.extend(indices)

    return torch.utils.data.Subset(dataset, selected)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_loaders(cfg: dict):
    """Build the four DataLoaders needed by an Optuna trial."""
    label_fraction = cfg["data"].get("label_fraction", 0.10)
    max_pulse_len  = cfg["data"].get("max_pulse_len",  4096)
    bs             = cfg["training"]["batch_size"]
    sources        = cfg["data"]["sources"]
    domain_map     = cfg["data"].get("domain_map", None)
    num_workers    = cfg["training"].get("num_workers", 0)
    pin_memory     = torch.cuda.is_available()

    loader_kwargs  = dict(num_workers=num_workers, pin_memory=pin_memory)

    # Phase 1 train (cross-sensor pairing, augmented)
    train_ds_p1 = DECDataset_Exp09(
        sources=sources, shard_key="train_shards",
        max_pulse_len=max_pulse_len, augment=True,
        label_fraction=label_fraction, domain_map=domain_map,
    )
    train_ds_p1 = subset_dataset_evenly(train_ds_p1, TUNE_TRAIN_MAX_PER_SOURCE)
    train_loader_p1 = DataLoader(train_ds_p1, batch_size=bs, shuffle=True,
                                 drop_last=True, **loader_kwargs)

    # Phase 2 train (single signal, no augmentation)
    train_ds_p2 = DECDataset_Exp09(
        sources=sources, shard_key="train_shards",
        max_pulse_len=max_pulse_len, augment=False,
        label_fraction=label_fraction, domain_map=domain_map,
    )
    train_ds_p2 = subset_dataset_evenly(train_ds_p2, TUNE_TRAIN_MAX_PER_SOURCE)
    train_loader_p2 = DataLoader(train_ds_p2, batch_size=bs, shuffle=True,
                                 drop_last=True, **loader_kwargs)

    # Phase 1 val (augmented pairs for validation_step)
    val_ds_p1 = DECDataset_Exp09(
        sources=sources, shard_key="val_shards",
        max_pulse_len=max_pulse_len, augment=True,
        label_fraction=label_fraction, domain_map=domain_map,
    )
    val_ds_p1 = subset_dataset_evenly(val_ds_p1, TUNE_VAL_MAX_PER_SOURCE)
    val_loader_p1 = DataLoader(val_ds_p1, batch_size=bs, shuffle=False, **loader_kwargs)

    # Phase 2 val (single signal)
    val_ds_p2 = DECDataset_Exp09(
        sources=sources, shard_key="val_shards",
        max_pulse_len=max_pulse_len, augment=False,
        label_fraction=label_fraction, domain_map=domain_map,
    )
    val_ds_p2 = subset_dataset_evenly(val_ds_p2, TUNE_VAL_MAX_PER_SOURCE)
    val_loader_p2 = DataLoader(val_ds_p2, batch_size=bs, shuffle=False, **loader_kwargs)

    return train_loader_p1, train_loader_p2, val_loader_p1, val_loader_p2


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial, base_config: dict) -> float:
    """Suggest hyperparameters, run a short trial, return Phase 2 val loss."""
    cfg          = copy.deepcopy(base_config)
    search_space = cfg.get("tune", {}).get("search_space", {})

    # Suggest learning rates (log-uniform)
    if "learning_rate_phase1" in search_space:
        lo, hi = [float(b) for b in search_space["learning_rate_phase1"]]
        cfg["training"]["learning_rate_phase1"] = trial.suggest_float(
            "learning_rate_phase1", lo, hi, log=True
        )
    if "learning_rate_phase2" in search_space:
        lo, hi = [float(b) for b in search_space["learning_rate_phase2"]]
        cfg["training"]["learning_rate_phase2"] = trial.suggest_float(
            "learning_rate_phase2", lo, hi, log=True
        )

    # Suggest dann_weight (log-uniform)
    if "dann_weight" in search_space:
        lo, hi = [float(b) for b in search_space["dann_weight"]]
        cfg["task"]["dann_weight"] = trial.suggest_float(
            "dann_weight", lo, hi, log=True
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader_p1, train_loader_p2, val_loader_p1, val_loader_p2 = build_loaders(cfg)

    if len(train_loader_p1) == 0:
        raise ValueError("[Tune] Train loader (P1) is empty. Check bispectra_v2 paths.")
    if len(train_loader_p2) == 0:
        raise ValueError("[Tune] Train loader (P2) is empty. Check bispectra_v2 paths.")

    task = SupConDECTask_Exp09(cfg).to(device)

    # GRL lambda fixed at half-maximum during tuning for stable comparison
    dann_lambda_tune = cfg.get("task", {}).get("dann_lambda_max", 1.0) / 2.0
    lr1 = cfg["training"]["learning_rate_phase1"]
    lr2 = cfg["training"]["learning_rate_phase2"]

    # ---------- Phase 1: Short SupCon + DANN ----------
    task.set_phase(1, lr=lr1, dann_lambda=dann_lambda_tune)
    task.train()
    for _ in range(TUNE_EPOCHS_P1):
        for batch in train_loader_p1:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            task.training_step_phase1(batch)

    # ---------- K-Means init ----------
    task.eval()
    all_embs = []
    with torch.no_grad():
        for batch in train_loader_p1:
            view1 = batch[0].to(device)   # (B, 2, 128, 128)
            z, _  = task.backbone(view1)
            all_embs.append(z.cpu().numpy())
    if not all_embs:
        raise RuntimeError("[Tune] No embeddings for K-Means.")
    task.init_cluster_centroids(np.concatenate(all_embs, axis=0))

    # ---------- Phase 2: Short DEC + DANN ----------
    task.set_phase(2, lr=lr2, dann_lambda=dann_lambda_tune)
    best_val_loss = float("inf")

    for epoch in range(TUNE_EPOCHS_P2):
        task.train()
        for batch in train_loader_p2:
            batch     = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            dec_batch = (batch[0], batch[1], batch[3])   # signal, reported_class, domain_label
            task.training_step_phase2(dec_batch)

        task.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader_p2:
                batch     = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                val_batch = (batch[0], batch[1], batch[3])
                res = task.validation_step(val_batch)
                val_losses.append(res["total"])

        epoch_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        best_val_loss  = min(best_val_loss, epoch_val_loss)

        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for Exp09 (ViT + Instance-SupCon + DANN + DEC)."
    )
    parser.add_argument(
        "--config", type=str,
        default="src/models/configs/exp09_vit_dann.yaml",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    n_trials = config.get("tune", {}).get("n_trials", 15)
    print(f"[Tuning] Experiment   : {config['experiment']['name']}")
    print(f"[Tuning] Trials       : {n_trials}")
    print(f"[Tuning] P1 epochs    : {TUNE_EPOCHS_P1} | P2 epochs: {TUNE_EPOCHS_P2}")
    print(f"[Tuning] Train subset : {TUNE_TRAIN_MAX_PER_SOURCE}/source | "
          f"Val subset: {TUNE_VAL_MAX_PER_SOURCE}/source")

    study = optuna.create_study(
        direction  = "minimize",
        study_name = config["experiment"]["name"],
        pruner     = optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)

    best = study.best_trial
    print(f"\n[Tuning Complete] Best trial #{best.number}:")
    print(f"  Validation Loss : {best.value:.6f}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    # -----------------------------------------------------------------------
    # Write best params back to YAML (preserve comments)
    # -----------------------------------------------------------------------
    import ruamel.yaml

    ryaml = ruamel.yaml.YAML()
    ryaml.preserve_quotes = True
    with open(config_path, "r") as f:
        doc = ryaml.load(f)

    tune_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_node_id = "".join(random.choices(string.ascii_letters + string.digits, k=4))

    if "learning_rate_phase1" in best.params:
        doc["training"]["learning_rate_phase1"] = round(best.params["learning_rate_phase1"], 10)
    if "learning_rate_phase2" in best.params:
        doc["training"]["learning_rate_phase2"] = round(best.params["learning_rate_phase2"], 10)
    if "dann_weight" in best.params:
        doc["task"]["dann_weight"] = round(best.params["dann_weight"], 8)

    doc["training"]["tuned_at"]        = tune_ts
    doc["training"]["tuning_node_id"]  = tune_node_id
    doc["training"]["tuning_val_loss"] = round(float(best.value), 8)

    with open(config_path, "w") as f:
        ryaml.dump(doc, f)
    print(f"\n[Saved] Best params written back to: {config_path}")

    # -----------------------------------------------------------------------
    # Lineage registration
    # -----------------------------------------------------------------------
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from src.utils.lineage_tracker import register_process

    parent_id = config["experiment"].get("parent_node_id", "NONE")
    register_process(
        parent_id        = parent_id,
        stage            = "hyperparameter_tuning",
        method           = "optuna_vit_dann_bispectrum_v2",
        folder_path      = config_path,
        appended_history = (
            f"Exp09 Optuna tuning ({n_trials} trials). "
            f"Best val loss: {best.value:.6f}. "
            f"Params: {json.dumps(best.params)}"
        ),
        force_node_id    = tune_node_id,
        force_timestamp  = tune_ts,
    )
    print(f"[Lineage] Registered tuning node: {tune_node_id}")

    utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
    with open(os.path.join(utils_dir, "latest_node.txt"), "w") as f:
        f.write(tune_node_id)
    print(f"[Lineage] Latest node ID written: {tune_node_id}")
