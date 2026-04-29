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
6. Save the best checkpoint.
7. Register the run to the SQLite lineage database.
8. Write output files to the correct DAG folder.

What this script does NOT do
-----------------------------
- It does not define any model layers.
- It does not define any loss functions.
- It does not parse HDF5 files.
- It does not implement any algorithm.

All of the above live in their dedicated modules.  This file only orchestrates.
"""

import argparse
import os
import sys
import random
import string
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

from src.models.data.dataset_detection import DetectionDataset
from src.models.tasks.task_detection   import DetectionTask
from src.utils.lineage_tracker         import register_process


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a (signal, target) batch tuple to the target device."""
    signal, target = batch
    return signal.to(device), target.to(device)


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
    parent_id = exp_cfg["parent_node_id"]   # "ShmH"
    root_id   = parent_id                   # ShmH is itself the root

    # Output folder name follows the DAG naming convention
    folder_name = f"{run_ts}_sy-{root_id}-{node_id}"

    # Per the ROUTING RULE: multi-task model (isolate + classify) -> goes into
    # classification_output/
    results_path = os.path.abspath(
        os.path.join(out_cfg["results_dir"], "cnn_yolo1d", folder_name)
    )
    eval_path = os.path.abspath(
        os.path.join(out_cfg["eval_dir"], "cnn_yolo1d", folder_name)
    )
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(eval_path,    exist_ok=True)

    print(f"[Lineage] NodeID     : {node_id}")
    print(f"[Lineage] Run folder : {results_path}")

    # ----------------------------------------------------------------------- #
    # 6. Build Datasets and DataLoaders                                         #
    # ----------------------------------------------------------------------- #
    root_path = os.path.abspath(data_cfg["root_path"])

    train_dataset = DetectionDataset(
        root_path         = root_path,
        shard_ids         = data_cfg["train_shards"],
        decimation_factor = data_cfg["decimation_factor"],
        grid_cells        = data_cfg["grid_cells"],
    )
    val_dataset = DetectionDataset(
        root_path         = root_path,
        shard_ids         = data_cfg["val_shards"],
        decimation_factor = data_cfg["decimation_factor"],
        grid_cells        = data_cfg["grid_cells"],
    )

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
    task = DetectionTask(config)
    task = task.to(device)

    total_params = sum(p.numel() for p in task.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {total_params:,}")

    # ----------------------------------------------------------------------- #
    # 8. Training loop                                                          #
    # ----------------------------------------------------------------------- #
    best_val_loss = float("inf")
    best_epoch    = -1

    for epoch in range(1, train_cfg["epochs"] + 1):

        # ---- Training ---- #
        train_losses = {"obj": 0.0, "box": 0.0, "cls": 0.0, "total": 0.0}
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            _, loss_dict = task.training_step(batch)
            for k in train_losses:
                train_losses[k] += loss_dict[k]

        n_train = len(train_loader)
        train_losses = {k: v / n_train for k, v in train_losses.items()}

        # ---- Validation ---- #
        val_metrics = {
            "obj": 0.0, "box": 0.0, "cls": 0.0, "total": 0.0,
            "iou": 0.0, "cls_acc": 0.0,
        }
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)
            m = task.validation_step(batch)
            for k in val_metrics:
                val_metrics[k] += m[k]

        n_val = len(val_loader)
        val_metrics = {k: v / n_val for k, v in val_metrics.items()}

        # ---- Logging ---- #
        print(
            f"Epoch {epoch:03d}/{train_cfg['epochs']:03d} | "
            f"Train loss: {train_losses['total']:.4f} "
            f"(obj={train_losses['obj']:.3f} box={train_losses['box']:.3f} cls={train_losses['cls']:.3f}) | "
            f"Val loss: {val_metrics['total']:.4f} "
            f"IoU={val_metrics['iou']:.3f} Acc={val_metrics['cls_acc']:.3f}"
        )

        # ---- Save best checkpoint ---- #
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
    # 9. Write analysis_history.txt into the results folder                     #
    # ----------------------------------------------------------------------- #
    history_line = (
        f"Detection & Classification via YOLO1D (cnn_yolo1d) at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, NodeID: {node_id}, "
        f"BestEpoch: {best_epoch}, BestValLoss: {best_val_loss:.6f}"
    )
    history_file = os.path.join(results_path, "analysis_history.txt")
    with open(history_file, "w") as f:
        f.write(history_line + "\n")

    # ----------------------------------------------------------------------- #
    # 10. Register this run to the SQLite lineage database                      #
    # ----------------------------------------------------------------------- #
    print("\n[Lineage] Registering to SQLite DAG...")
    new_node_id = register_process(
        parent_id       = parent_id,
        stage           = "classification",   # Routing Rule: most downstream task
        method          = "cnn_yolo1d",
        folder_path     = results_path,
        appended_history= history_line,
        force_node_id   = node_id,
    )
    print(f"[Lineage] Registered as Node {new_node_id} (child of {parent_id})")
    print(f"\n[Done] Training complete. Best epoch: {best_epoch},"
          f" best val loss: {best_val_loss:.6f}")
    print(f"[Done] Results -> {results_path}")


if __name__ == "__main__":
    main()
