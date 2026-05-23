"""
train.py
========
Universal DAG Orchestrator for the FYP PD Deep Learning Pipeline.

This is the ONLY script the user executes:

    python src/models/train.py --config src/models/configs/exp01_yolo1d.yaml

What this script does (and ONLY what it does)
---------------------------------------------
1. Parse the YAML config.
2. Select the compute device (CUDA / CPU).
3. Build the DataLoaders from the config.
4. Build the Task (model + optimizer + loss) from the config.
5. Run the training loop: for each epoch -> training_step -> validation_step.
6. Save the best checkpoint to models/weights/ and models/configuration_snapshots/.
7. Register the run to the SQLite lineage database (src/utils/lineage.db).

NOTE ON COLAB WORKFLOW
-----------------------
After training, commit and push all of the following from Colab:
  - models/weights/model_<NodeID>.pt
  - models/configuration_snapshots/config_<NodeID>.yaml
  - src/utils/lineage.db
  - data/performance_evaluation/training/<NodeID>_timing.json

What this script does NOT do
-----------------------------
- It does not define any model layers.
- It does not define any loss functions.
- It does not parse HDF5 files.
- It does not implement any algorithm.

All of the above live in their dedicated modules.  This file only orchestrates.
"""

import argparse
import json
import os
import sys
import random
import string
import time
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Make sure src/models is importable regardless of the working directory
# ---------------------------------------------------------------------------
# Insert the project root (two levels above this file) into sys.path
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))          # src/models
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))  # FYP/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Task factory — imports are deferred so unused modules are never loaded
# ---------------------------------------------------------------------------

def _build_datasets_and_task(config: dict, task_type: str):
    """
    Returns (train_dataset, val_dataset, task) for the requested task_type.
    Add new elif branches here as new task types are implemented.
    """
    data_cfg  = config["data"]
    
    if task_type == "detection":
        from src.models.data.dataset_exp01 import DetectionDataset
        from src.models.tasks.task_exp01   import DetectionTask
        root_path = os.path.abspath(data_cfg["root_path"])
        train_ds = DetectionDataset(
            root_path         = root_path,
            shard_ids         = data_cfg["train_shards"],
            decimation_factor = data_cfg["decimation_factor"],
            grid_cells        = data_cfg["grid_cells"],
        )
        val_ds = DetectionDataset(
            root_path         = root_path,
            shard_ids         = data_cfg["val_shards"],
            decimation_factor = data_cfg["decimation_factor"],
            grid_cells        = data_cfg["grid_cells"],
        )
        task = DetectionTask(config)

    elif task_type == "classification":
        from src.models.data.dataset_exp02 import ClassificationDataset
        from src.models.tasks.task_exp02   import ClassificationTask
        root_path = os.path.abspath(data_cfg["root_path"])
        train_ds = ClassificationDataset(
            root_path     = root_path,
            shard_ids     = data_cfg["train_shards"],
            max_pulse_len = data_cfg["max_pulse_len"],
        )
        val_ds = ClassificationDataset(
            root_path     = root_path,
            shard_ids     = data_cfg["val_shards"],
            max_pulse_len = data_cfg["max_pulse_len"],
        )
        task = ClassificationTask(config)

    elif task_type == "contrastive":
        from src.models.data.dataset_exp03_contrastive import ContrastiveDataset
        from src.models.tasks.task_exp03_contrastive   import ContrastiveTask
        root_path = os.path.abspath(data_cfg["root_path"])
        train_ds = ContrastiveDataset(
            root_path     = root_path,
            shard_ids     = data_cfg["train_shards"],
            max_pulse_len = data_cfg["max_pulse_len"],
        )
        val_ds = ContrastiveDataset(
            root_path     = root_path,
            shard_ids     = data_cfg["val_shards"],
            max_pulse_len = data_cfg["max_pulse_len"],
        )
        task = ContrastiveTask(config)

    elif task_type == "dec":
        from src.models.data.dataset_exp04_dec import DECDataset
        from src.models.tasks.task_exp04_dec   import DECTask
        sources = data_cfg["sources"]
        # Read wavelet denoising settings from config (with safe defaults)
        wavelet_kwargs = dict(
            denoise        = data_cfg.get("denoise",        True),
            wavelet        = data_cfg.get("wavelet",        "db4"),
            wavelet_level  = data_cfg.get("wavelet_level",  4),
            threshold_mode = data_cfg.get("threshold_mode", "soft"),
        )
        # Phase 1 uses augmented pairs (SimCLR)
        train_ds = DECDataset(
            sources       = sources,
            shard_key     = "train_shards",
            max_pulse_len = data_cfg["max_pulse_len"],
            augment       = True,
            **wavelet_kwargs,
        )
        val_ds = DECDataset(
            sources       = sources,
            shard_key     = "val_shards",
            max_pulse_len = data_cfg["max_pulse_len"],
            augment       = True,   # Symmetric validation during Phase 1
            **wavelet_kwargs,
        )
        task = DECTask(config)

    elif task_type == "dec_spherical":
        from src.models.data.dataset_exp05_dec import DECDataset
        from src.models.tasks.task_exp05_dec   import DECTask
        sources = data_cfg["sources"]
        wavelet_kwargs = dict(
            denoise        = data_cfg.get("denoise",        True),
            wavelet        = data_cfg.get("wavelet",        "db4"),
            wavelet_level  = data_cfg.get("wavelet_level",  4),
            threshold_mode = data_cfg.get("threshold_mode", "soft"),
        )
        train_ds = DECDataset(
            sources       = sources,
            shard_key     = "train_shards",
            max_pulse_len = data_cfg["max_pulse_len"],
            augment       = True,
            **wavelet_kwargs,
        )
        val_ds = DECDataset(
            sources       = sources,
            shard_key     = "val_shards",
            max_pulse_len = data_cfg["max_pulse_len"],
            augment       = True,
            **wavelet_kwargs,
        )
        task = DECTask(config)

    else:
        raise ValueError(f"[Error] Unknown task_type in config: '{task_type}'. "
                         f"Choose from: detection, classification, contrastive, dec")

    return train_ds, val_ds, task
from src.utils.lineage_tracker import register_process


def generate_node_id(length: int = 4) -> str:
    """Generate a random 4-character alphanumeric NodeID (matches lineage convention)."""
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def select_device(device_cfg: str) -> torch.device:
    """Resolve 'auto' / 'cuda' / 'cpu' to a torch.device."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def move_batch_to_device(
    batch: tuple,
    device: torch.device,
) -> tuple:
    """Move all elements of a batch tuple to the target device."""
    return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ----------------------------------------------------------------------- #
    # 1. Parse arguments                                                        #
    # ----------------------------------------------------------------------- #
    parser = argparse.ArgumentParser(
        description="FYP PD Pipeline - Universal Training Orchestrator"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the experiment YAML config file."
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run a quick smoke test: 2 shards, 2 epochs, batch_size=2."
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found: {config_path}")
        sys.exit(1)

    # ----------------------------------------------------------------------- #
    # 2. Load config                                                            #
    # ----------------------------------------------------------------------- #
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_cfg   = config["experiment"]
    data_cfg  = config["data"]
    train_cfg = config["training"]
    out_cfg   = config["output"]

    print("=" * 60)
    print(f"  Experiment : {exp_cfg['name']}")
    print(f"  Description: {exp_cfg['description']}")
    print(f"  Parent Node: {exp_cfg['parent_node_id']}")
    print("=" * 60)

    # ----------------------------------------------------------------------- #
    # 3. Smoke-test overrides                                                   #
    # ----------------------------------------------------------------------- #
    if args.smoke_test:
        if task_type in ["dec", "dec_spherical"]:
            # Override shards inside each source
            for src in data_cfg["sources"]:
                src["train_shards"] = [src["train_shards"][0]]
                src["val_shards"]   = [src["val_shards"][0]]
            train_cfg["phase1_epochs"] = 1
            train_cfg["phase2_epochs"] = 1
            train_cfg["epochs"]        = 2
            train_cfg["batch_size"]    = 4
            print("[Smoke Test] Overriding sources to 1 shard each, 1+1 epochs, batch=4")
        else:
            print("[Smoke Test] Overriding shards -> [1, 2], epochs -> 2, batch -> 2")
            data_cfg["train_shards"] = [1, 2]
            data_cfg["val_shards"]   = [3]
            train_cfg["epochs"]      = 2
            train_cfg["batch_size"]  = 2

    # ----------------------------------------------------------------------- #
    # 4. Device                                                                 #
    # ----------------------------------------------------------------------- #
    device = select_device(train_cfg["device"])
    print(f"[Device] Using: {device}")

    # ----------------------------------------------------------------------- #
    # 5. Generate lineage identifiers for this run                              #
    # ----------------------------------------------------------------------- #
    node_id   = generate_node_id()
    run_ts    = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Inherit RootID from the parent node.
    # The raw dataset folder name encodes it: ...sy-<RootID>-<NodeID>
    # Since the parent IS the root (no processing done yet), root_id == parent_id
    parent_id = exp_cfg["parent_node_id"]   # "ShmH" (example)
    root_id   = parent_id                   # ShmH is itself the root (example)

    weights_dir = os.path.abspath(out_cfg["weights_dir"])

    print(f"[Lineage] NodeID      : {node_id}")
    print(f"[Lineage] Weights dir : {weights_dir}")

    # ----------------------------------------------------------------------- #
    # 6. Build Datasets, DataLoaders, and Task                                  #
    # ----------------------------------------------------------------------- #
    task_type = exp_cfg.get("task_type", "detection")
    print(f"[Task] Type: {task_type}")

    train_dataset, val_dataset, task = _build_datasets_and_task(config, task_type)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = train_cfg["batch_size"],
        shuffle     = True,
        num_workers = 0,    # 0 = main process; safe for HDF5 on all platforms
        pin_memory  = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = train_cfg["batch_size"],
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )

    print(f"[Data] Train samples : {len(train_dataset)}")
    print(f"[Data] Val samples   : {len(val_dataset)}")

    # ----------------------------------------------------------------------- #
    # 7. Build Task (model + optimizer + loss)                                  #
    # ----------------------------------------------------------------------- #
    task = task.to(device)

    total_params = sum(p.numel() for p in task.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {total_params:,}")

    # Shared timing / tracking variables (used by both DEC and standard paths)
    best_val_loss  = float("inf")
    best_epoch     = -1
    epoch_times    = []
    train_start    = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ----------------------------------------------------------------------- #
    # 8. Training loop                                                          #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # Special two-phase DEC training path                                       #
    # ----------------------------------------------------------------------- #
    if task_type in ["dec", "dec_spherical"]:
        if task_type == "dec":
            from src.models.data.dataset_exp04_dec import DECDataset
        else:
            from src.models.data.dataset_exp05_dec import DECDataset
        import numpy as np

        phase1_epochs = train_cfg.get("phase1_epochs", 20)
        phase2_epochs = train_cfg.get("phase2_epochs", 15)
        lr1 = train_cfg.get("learning_rate_phase1", 0.001)
        lr2 = train_cfg.get("learning_rate_phase2", 0.0001)
        update_interval = config.get("task", {}).get("dec_update_interval", 140)

        # --- Phase 1: SimCLR --- #
        print(f"\n[DEC] === Phase 1: SimCLR Pre-Training ({phase1_epochs} epochs) ===")
        task.set_phase(1, lr1)
        best_val_loss = float("inf")
        best_epoch = -1

        for epoch in range(1, phase1_epochs + 1):
            epoch_start = time.time()
            task.train()

            train_losses = None
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                _, loss_dict = task.training_step_phase1(batch)
                if train_losses is None:
                    train_losses = {k: 0.0 for k in loss_dict}
                for k in loss_dict:
                    train_losses[k] += loss_dict[k]

            train_losses = {k: v / len(train_loader) for k, v in train_losses.items()}

            val_metrics = None
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                m = task.validation_step(batch)
                if val_metrics is None:
                    val_metrics = {k: 0.0 for k in m}
                for k in m:
                    val_metrics[k] += m[k]
            val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}

            elapsed = time.time() - epoch_start
            epoch_times.append(round(elapsed, 2))
            print(f"  P1 Epoch {epoch:03d}/{phase1_epochs} | "
                  f"Train: {train_losses['simclr']:.4f} | "
                  f"Val: {val_metrics['simclr']:.4f} | {elapsed:.1f}s")

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                best_epoch = epoch
                task.save_checkpoint(epoch, node_id,
                    os.path.abspath(out_cfg["weights_dir"]),
                    os.path.abspath(out_cfg["config_snapshot_dir"]),
                    config_path)
                print(f"    ^ New best Phase1 val loss: {best_val_loss:.4f}")

        # --- Centroid Initialization --- #
        print("\n[DEC] Extracting all embeddings for K-Means init...")
        unaugmented_train = DECDataset(
            sources       = data_cfg["sources"],
            shard_key     = "train_shards",
            max_pulse_len = data_cfg["max_pulse_len"],
            augment       = False,
        )
        init_loader = DataLoader(unaugmented_train, batch_size=train_cfg["batch_size"],
                                  shuffle=False, num_workers=0)
        all_embs = []
        task.eval()
        with torch.no_grad():
            for batch in init_loader:
                sig = batch[0].to(device)
                z = task.backbone(sig)
                all_embs.append(z.cpu().numpy())
        all_embs = np.concatenate(all_embs, axis=0)
        task.init_cluster_centroids(all_embs)

        # --- Phase 2: DEC --- #
        print(f"\n[DEC] === Phase 2: DEC Cluster Refinement ({phase2_epochs} epochs) ===")
        task.set_phase(2, lr2)

        # Rebuild loaders with un-augmented data for Phase 2
        train_loader_p2 = DataLoader(
            DECDataset(sources=data_cfg["sources"], shard_key="train_shards",
                       max_pulse_len=data_cfg["max_pulse_len"], augment=False),
            batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_loader_p2 = DataLoader(
            DECDataset(sources=data_cfg["sources"], shard_key="val_shards",
                       max_pulse_len=data_cfg["max_pulse_len"], augment=False),
            batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        best_val_loss_p2 = float("inf")
        for epoch in range(1, phase2_epochs + 1):
            epoch_start = time.time()
            task.train()

            train_losses = None
            for batch in train_loader_p2:
                batch = move_batch_to_device(batch, device)
                _, loss_dict = task.training_step_phase2(batch)
                if train_losses is None:
                    train_losses = {k: 0.0 for k in loss_dict}
                for k in loss_dict:
                    train_losses[k] += loss_dict[k]
            train_losses = {k: v / len(train_loader_p2) for k, v in train_losses.items()}

            val_metrics = None
            for batch in val_loader_p2:
                batch = move_batch_to_device(batch, device)
                m = task.validation_step(batch)
                if val_metrics is None:
                    val_metrics = {k: 0.0 for k in m}
                for k in m:
                    val_metrics[k] += m[k]
            val_metrics = {k: v / len(val_loader_p2) for k, v in val_metrics.items()}

            elapsed = time.time() - epoch_start
            epoch_times.append(round(elapsed, 2))
            print(f"  P2 Epoch {epoch:03d}/{phase2_epochs} | "
                  f"Train KL: {train_losses['kl_div']:.4f} | "
                  f"Val KL: {val_metrics['kl_div']:.4f} | {elapsed:.1f}s")

            # DEC Phase 2 Saving Logic
            if task_type == "dec":
                # Original DEC Phase 2: KL loss ALWAYS increases by design (see task_exp04_dec.py).
                # "Best checkpoint" logic based on lowest val KL is meaningless here —
                # it would always save epoch 1. Instead, save every epoch and keep the last.
                best_epoch = phase1_epochs + epoch
                task.save_checkpoint(best_epoch, node_id,
                    os.path.abspath(out_cfg["weights_dir"]),
                    os.path.abspath(out_cfg["config_snapshot_dir"]),
                    config_path)
                if epoch == phase2_epochs:
                    print(f"    ^ Final Phase2 checkpoint saved (epoch {best_epoch}).")
            else:
                # Spherical DEC Phase 2: KL loss should correctly decrease.
                if val_metrics["kl_div"] < best_val_loss_p2:
                    best_val_loss_p2 = val_metrics["kl_div"]
                    best_epoch = phase1_epochs + epoch
                    task.save_checkpoint(best_epoch, node_id,
                        os.path.abspath(out_cfg["weights_dir"]),
                        os.path.abspath(out_cfg["config_snapshot_dir"]),
                        config_path)
                    print(f"    ^ New best Phase2 val KL: {best_val_loss_p2:.4f}")

        if task_type == "dec_spherical":
            best_val_loss = best_val_loss_p2
        else:
            best_val_loss = val_metrics["total"]   # Report the final epoch's val KL

    # ----------------------------------------------------------------------- #
    # Standard single-phase training (detection / classification / contrastive)#
    # ----------------------------------------------------------------------- #
    else:
        for epoch in range(1, train_cfg["epochs"] + 1):
            epoch_start = time.time()

            # ---- Training ---- #
            train_losses = None
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                _, loss_dict = task.training_step(batch)
                if train_losses is None:
                    train_losses = {k: 0.0 for k in loss_dict}
                for k in loss_dict:
                    train_losses[k] += loss_dict[k]

            n_train = len(train_loader)
            train_losses = {k: v / n_train for k, v in train_losses.items()}

            # ---- Validation ---- #
            val_metrics = None
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                m = task.validation_step(batch)
                if val_metrics is None:
                    val_metrics = {k: 0.0 for k in m}
                for k in m:
                    val_metrics[k] += m[k]

            n_val = len(val_loader)
            val_metrics = {k: v / n_val for k, v in val_metrics.items()}

            epoch_elapsed = time.time() - epoch_start
            epoch_times.append(round(epoch_elapsed, 2))

            train_str = " ".join(f"{k}={v:.4f}" for k, v in train_losses.items())
            val_str   = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(
                f"Epoch {epoch:03d}/{train_cfg['epochs']:03d} | "
                f"Train: {train_str} | "
                f"Val: {val_str} | "
                f"{epoch_elapsed:.1f}s"
            )

            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                best_epoch    = epoch
                task.save_checkpoint(
                    epoch       = epoch,
                    node_id     = node_id,
                    weights_dir = os.path.abspath(out_cfg["weights_dir"]),
                    config_dir  = os.path.abspath(out_cfg["config_snapshot_dir"]),
                    config_path = config_path,
                )
                print(f"  ^ New best val loss: {best_val_loss:.4f} (epoch {best_epoch})")

    # ----------------------------------------------------------------------- #
    # 9. Save timing report to data/performance_evaluation/training/           #
    # ----------------------------------------------------------------------- #
    total_elapsed  = time.time() - train_start
    end_time_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Identify the exact GPU model Colab assigned (or "cpu" if no GPU)
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "cpu"

    timing_report = {
        "node_id"              : node_id,
        "experiment"           : exp_cfg["name"],
        "device"               : str(device),
        "gpu_name"             : gpu_name,
        "start_time"           : start_time_str,
        "end_time"             : end_time_str,
        "n_epochs_completed"   : len(epoch_times),
        "n_train_samples"      : len(train_dataset),
        "n_val_samples"        : len(val_dataset),
        "total_training_time_s": round(total_elapsed, 2),
        "mean_epoch_time_s"    : round(sum(epoch_times) / len(epoch_times), 2) if epoch_times else 0,
        "epoch_times_s"        : epoch_times,
        "best_epoch"           : best_epoch,
        "best_val_loss"        : round(best_val_loss, 6),
    }
    timing_dir  = os.path.abspath("data/performance_evaluation/training")
    os.makedirs(timing_dir, exist_ok=True)
    timing_path = os.path.join(timing_dir, f"{node_id}_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing_report, f, indent=2)
    print(f"[Timing] Total training time : {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)")
    print(f"[Timing] Report saved        -> {timing_path}")

    # ----------------------------------------------------------------------- #
    # 10. Register this run to the SQLite lineage database                     #
    # ----------------------------------------------------------------------- #
    history_line = exp_cfg.get("description", "No description provided")
    print("\n[Lineage] Registering to SQLite DAG...")
    new_node_id = register_process(
        parent_id        = parent_id,
        stage            = "classification",   # Routing Rule: most downstream task
        method           = task_type,
        folder_path      = weights_dir,         # actual artifact location
        appended_history = history_line,
        force_node_id    = node_id,
    )
    print(f"[Lineage] Registered as Node {new_node_id} (child of {parent_id})")
    print(f"\n[Done] Training complete. Best epoch: {best_epoch},"
          f" best val loss: {best_val_loss:.6f}")
    print(f"[Done] Weights -> {weights_dir}/model_{node_id}.pt")


if __name__ == "__main__":
    main()
