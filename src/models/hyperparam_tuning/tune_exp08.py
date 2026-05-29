"""
tune_exp08.py
=============
Optuna hyperparameter tuning for Exp08: ViT + SupCon & Semi-Supervised DEC
on 2D Bispectrum features.

KEY DIFFERENCES vs tune_exp07.py
----------------------------------
1.  Uses DECDataset_Exp08 (bispectra) instead of DECDataset_Exp07 (STFT).
2.  Uses SupConDECTask_Exp08 (ViT backbone) instead of SupConDECTask (CNN).
3.  Removed `base_channels` from the search space (ViT has no CNN base_channels).
4.  Added `pairwise_weight_gamma` to search space.

WHAT IS HARDCODED (not interactive, not tunable)
-------------------------------------------------
- Tuning epochs: Phase1 = min(3, config_value),  Phase2 = min(3, config_value)
  These short trials are intentionally short to maximize trial throughput.
- Dataset subset size: 300 samples per source for train, 100 for val.
  This keeps each trial fast while still being representative.

Usage:
    python src/models/hyperparam_tuning/tune_exp08.py \\
        --config src/models/configs/tune_exp08_dec.yaml
"""

import os
import sys
import yaml
import copy
import argparse
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader

try:
    import optuna
except ImportError:
    print("[Error] Optuna is not installed. Run: pip install optuna")
    sys.exit(1)

# Project root into sys.path (3 levels up from this script)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.models.data.dataset_exp08_dec import DECDataset_Exp08
from src.models.tasks.task_exp08_dec   import SupConDECTask_Exp08


# ---------------------------------------------------------------------------
# Subset helper — same as tune_exp07 (hardcoded sample count per source)
# ---------------------------------------------------------------------------

# HARDCODED: number of samples per dataset source for tuning
TUNE_TRAIN_MAX_PER_SOURCE = 300   # Kept small for speed; not adjustable at runtime
TUNE_VAL_MAX_PER_SOURCE   = 100   # Kept small for speed; not adjustable at runtime


def subset_dataset_evenly(dataset, max_per_source: int):
    """
    Draw at most `max_per_source` samples from each dataset source (by shard dir).
    Returns a torch.utils.data.Subset.
    """
    from collections import defaultdict
    source_to_indices = defaultdict(list)
    for i, item in enumerate(dataset.index):
        # item[0] is shard_path; use its parent directory as source key
        source_dir = os.path.dirname(item[0])
        source_to_indices[source_dir].append(i)

    selected_indices = []
    for source, indices in source_to_indices.items():
        if len(indices) > max_per_source:
            sampled = np.random.choice(indices, max_per_source, replace=False)
            selected_indices.extend(sampled.tolist())
        else:
            selected_indices.extend(indices)

    return torch.utils.data.Subset(dataset, selected_indices)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_loaders(cfg: dict):
    """Build the four DataLoaders needed by an Optuna trial."""
    label_fraction = cfg["data"].get("label_fraction", 0.10)
    max_pulse_len  = cfg["data"].get("max_pulse_len", 4096)
    bs             = cfg["training"]["batch_size"]
    sources        = cfg["data"]["sources"]

    # Phase 1 train (augmented pairs)
    train_ds_p1 = DECDataset_Exp08(
        sources=sources, shard_key="train_shards",
        max_pulse_len=max_pulse_len, augment=True,
        label_fraction=label_fraction,
    )
    train_ds_p1 = subset_dataset_evenly(train_ds_p1, TUNE_TRAIN_MAX_PER_SOURCE)
    train_loader_p1 = DataLoader(train_ds_p1, batch_size=bs, shuffle=True, drop_last=True)

    # Phase 2 train (no augmentation)
    train_ds_p2 = DECDataset_Exp08(
        sources=sources, shard_key="train_shards",
        max_pulse_len=max_pulse_len, augment=False,
        label_fraction=label_fraction,
    )
    train_ds_p2 = subset_dataset_evenly(train_ds_p2, TUNE_TRAIN_MAX_PER_SOURCE)
    train_loader_p2 = DataLoader(train_ds_p2, batch_size=bs, shuffle=True, drop_last=True)

    # Phase 1 val (augmented, required by validation_step when phase=1)
    val_ds_p1 = DECDataset_Exp08(
        sources=sources, shard_key="val_shards",
        max_pulse_len=max_pulse_len, augment=True,
        label_fraction=label_fraction,
    )
    val_ds_p1 = subset_dataset_evenly(val_ds_p1, TUNE_VAL_MAX_PER_SOURCE)
    val_loader_p1 = DataLoader(val_ds_p1, batch_size=bs, shuffle=False)

    # Phase 2 val (no augmentation)
    val_ds_p2 = DECDataset_Exp08(
        sources=sources, shard_key="val_shards",
        max_pulse_len=max_pulse_len, augment=False,
        label_fraction=label_fraction,
    )
    val_ds_p2 = subset_dataset_evenly(val_ds_p2, TUNE_VAL_MAX_PER_SOURCE)
    val_loader_p2 = DataLoader(val_ds_p2, batch_size=bs, shuffle=False)

    return train_loader_p1, train_loader_p2, val_loader_p1, val_loader_p2


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

# HARDCODED tuning epoch counts — not interactive, not adjustable
TUNE_EPOCHS_P1 = 3   # Short Phase 1 to probe LR sensitivity quickly
TUNE_EPOCHS_P2 = 3   # Short Phase 2 to measure DEC convergence tendency


def objective(trial, base_config: dict) -> float:
    """Suggest hyperparameters, train briefly, return Phase 2 val loss."""
    cfg          = copy.deepcopy(base_config)
    search_space = cfg.get("tune", {}).get("search_space", {})

    # Suggest learning rates (log-uniform)
    if "learning_rate_phase1" in search_space:
        bounds = search_space["learning_rate_phase1"]
        cfg["training"]["learning_rate_phase1"] = trial.suggest_float(
            "learning_rate_phase1", float(bounds[0]), float(bounds[1]), log=True
        )
    if "learning_rate_phase2" in search_space:
        bounds = search_space["learning_rate_phase2"]
        cfg["training"]["learning_rate_phase2"] = trial.suggest_float(
            "learning_rate_phase2", float(bounds[0]), float(bounds[1]), log=True
        )
    # Suggest pairwise gamma (log-uniform)
    if "pairwise_weight_gamma" in search_space:
        bounds = search_space["pairwise_weight_gamma"]
        cfg["task"]["pairwise_weight_gamma"] = trial.suggest_float(
            "pairwise_weight_gamma", float(bounds[0]), float(bounds[1]), log=True
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader_p1, train_loader_p2, val_loader_p1, val_loader_p2 = build_loaders(cfg)

    if len(train_loader_p1) == 0:
        raise ValueError("Train loader is empty. Check your bispectrum paths.")

    # Build model
    task = SupConDECTask_Exp08(cfg).to(device)

    # Phase 1: Short SupCon warmup
    task.set_phase(1, lr=cfg["training"]["learning_rate_phase1"])
    for _ in range(TUNE_EPOCHS_P1):
        task.train()
        for batch in train_loader_p1:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            task.training_step_phase1(batch)

    # Extract embeddings and init K-Means centroids
    task.eval()
    all_embs = []
    with torch.no_grad():
        for batch in train_loader_p1:
            view1 = batch[0].to(device)
            z     = task.backbone(view1)
            all_embs.append(z.cpu().numpy())
    if not all_embs:
        raise RuntimeError("No embeddings for K-Means init.")
    task.init_cluster_centroids(np.concatenate(all_embs, axis=0))

    # Phase 2: Short DEC trial
    task.set_phase(2, lr=cfg["training"]["learning_rate_phase2"])
    best_val_loss = float("inf")

    for epoch in range(TUNE_EPOCHS_P2):
        task.train()
        for batch in train_loader_p2:
            batch     = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            dec_batch = (batch[0], batch[1])   # (signal, reported_class)
            task.training_step_phase2(dec_batch)

        # Validation
        task.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader_p2:
                signal         = batch[0].to(device)
                reported_class = batch[1].to(device)
                res = task.validation_step((signal, reported_class))
                val_losses.append(res["total"])

        epoch_val_loss  = float(np.mean(val_losses)) if val_losses else float("inf")
        best_val_loss   = min(best_val_loss, epoch_val_loss)

        # Report to Optuna for pruning
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for Exp08 (ViT + Bispectrum DEC)."
    )
    parser.add_argument(
        "--config", type=str,
        default="src/models/configs/tune_exp08_dec.yaml",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    n_trials = config.get("tune", {}).get("n_trials", 10)
    print(f"[Tuning] Starting Optuna study for {config['experiment']['name']}")
    print(f"[Tuning] Trials: {n_trials} | P1 epochs: {TUNE_EPOCHS_P1} | P2 epochs: {TUNE_EPOCHS_P2}")
    print(f"[Tuning] Train subset: {TUNE_TRAIN_MAX_PER_SOURCE}/source | "
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
    # Write best params back into the YAML config (preserve comments)
    # -----------------------------------------------------------------------
    import json
    import random
    import string
    import ruamel.yaml

    ryaml = ruamel.yaml.YAML()
    ryaml.preserve_quotes = True
    with open(config_path, "r") as f:
        doc = ryaml.load(f)

    tune_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_node_id   = "".join(random.choices(string.ascii_letters + string.digits, k=4))

    if "learning_rate_phase1" in best.params:
        doc["training"]["learning_rate_phase1"] = round(best.params["learning_rate_phase1"], 10)
    if "learning_rate_phase2" in best.params:
        doc["training"]["learning_rate_phase2"] = round(best.params["learning_rate_phase2"], 10)
    if "pairwise_weight_gamma" in best.params:
        doc["task"]["pairwise_weight_gamma"] = round(best.params["pairwise_weight_gamma"], 6)

    doc["training"]["tuned_at"]       = tune_timestamp
    doc["training"]["tuning_node_id"] = tune_node_id
    doc["training"]["tuning_val_loss"] = round(float(best.value), 8)

    with open(config_path, "w") as f:
        ryaml.dump(doc, f)
    print(f"\n[Saved] Best params written back to: {config_path}")

    # -----------------------------------------------------------------------
    # Register in lineage DB
    # -----------------------------------------------------------------------
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from src.utils.lineage_tracker import register_process

    parent_id = config["experiment"].get("parent_node_id", "NONE")
    register_process(
        parent_id        = parent_id,
        stage            = "hyperparameter_tuning",
        method           = "optuna_vit_bispectrum",
        folder_path      = config_path,
        appended_history = (
            f"Exp08 Optuna tuning ({n_trials} trials). "
            f"Best val loss: {best.value:.6f}. "
            f"Params: {json.dumps(best.params)}"
        ),
        force_node_id    = tune_node_id,
        force_timestamp  = tune_timestamp,
    )
    print(f"[Lineage] Registered tuning node: {tune_node_id}")

    # Write latest node for Colab commit messages
    utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
    with open(os.path.join(utils_dir, "latest_node.txt"), "w") as f:
        f.write(tune_node_id)
    print(f"[Lineage] Latest node ID written: {tune_node_id}")
