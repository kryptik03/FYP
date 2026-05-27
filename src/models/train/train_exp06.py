"""
train_exp06.py
==============
Dedicated training orchestrator for Exp06:
  Semi-Supervised Deep Embedded Clustering on 2D STFT features.

Usage:
    python src/models/train/train_exp06.py --config src/models/configs/tune_exp06_dec.yaml

The script reads the same YAML as the tuning script. The training/model/task
fields in that YAML are already stamped with the best hyperparameters found by
tune_exp06.py, so no manual editing is needed.

After training completes:
  - Weights are saved to models/weights/model_<NodeID>.pt
  - Config snapshot is saved to models/configuration_snapshots/config_<NodeID>.yaml
  - A structured run folder is created under:
      data/performance_evaluation/training/<timestamp>_<origin>-<rootid>-<nodeid>/
    containing:
      * timing_and_metrics.json  — timing, hyperparams, dataset sizes, gradient norms
      * fig_loss_phase1.png      — Phase 1 SimCLR train/val loss curves
      * fig_loss_phase2.png      — Phase 2 KL-div / Pairwise / Total train/val curves
      * fig_epoch_times.png      — Per-epoch wall-clock times
  - The run is registered in src/utils/lineage.db
  - The node ID is written to src/utils/latest_node.txt for Colab commit messages
"""

import argparse
import json
import os
import sys
import random
import string
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))           # src/models/train
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))  # FYP/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.data.dataset_exp06_dec import DECDataset_Exp06
from src.models.tasks.task_exp06_dec   import SemiSupervisedDECTask
from src.utils.lineage_tracker         import register_process, get_node_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_node_id(length: int = 4) -> str:
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def move_to_device(batch: tuple, device: torch.device) -> tuple:
    return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)


def write_latest_node(node_id: str):
    """Writes the node ID to src/utils/latest_node.txt for Colab commit messages."""
    utils_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "utils"))
    path = os.path.join(utils_dir, "latest_node.txt")
    with open(path, "w") as f:
        f.write(node_id)
    print(f"[Lineage] Latest node ID written to: {path}")


def get_grad_norm(model: torch.nn.Module) -> float:
    """Compute the global L2 gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    return total_norm ** 0.5


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DARK_BG  = "#0F0F0F"
PANEL_BG = "#1A1A2E"
COLORS   = ["#4CC9F0", "#F72585", "#7209B7", "#4361EE", "#3A0CA3"]


def _apply_dark_style(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="#888888")
        ax.xaxis.label.set_color("#AAAAAA")
        ax.yaxis.label.set_color("#AAAAAA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")


def plot_phase1_loss(history: dict, out_dir: str, phase1_epochs: int) -> str:
    """Phase 1 SimCLR train/val loss curves."""
    epochs = list(range(1, phase1_epochs + 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_dark_style(fig, ax)

    ax.plot(epochs, history["p1_train_simclr"], color=COLORS[0], lw=2, label="Train SimCLR")
    ax.plot(epochs, history["p1_val_simclr"],   color=COLORS[1], lw=2, linestyle="--", label="Val SimCLR")

    best_ep = int(np.argmin(history["p1_val_simclr"])) + 1
    ax.axvline(best_ep, color="#FFBE0B", lw=1, linestyle=":", alpha=0.8, label=f"Best Val (ep {best_ep})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("SimCLR Loss")
    ax.set_title("Phase 1 — SimCLR Pre-Training Loss", color="white", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.3, labelcolor="white", facecolor=DARK_BG)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig_loss_phase1.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_phase2_loss(history: dict, out_dir: str, phase2_epochs: int) -> str:
    """Phase 2 KL / Pairwise / Total train+val loss curves."""
    epochs = list(range(1, phase2_epochs + 1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    _apply_dark_style(fig, axes)

    series = [
        ("p2_train_kl",       "p2_val_kl",       "KL Divergence",    COLORS[0]),
        ("p2_train_pairwise", "p2_val_pairwise",  "Pairwise Loss",    COLORS[1]),
        ("p2_train_total",    "p2_val_total",     "Total Loss",       COLORS[2]),
    ]
    for ax, (tr_key, vl_key, label, color) in zip(axes, series):
        ax.plot(epochs, history[tr_key], color=color,     lw=2, label="Train")
        ax.plot(epochs, history[vl_key], color=color,     lw=2, linestyle="--", alpha=0.7, label="Val")
        best_ep = int(np.argmin(history[vl_key])) + 1
        ax.axvline(best_ep, color="#FFBE0B", lw=1, linestyle=":", alpha=0.8, label=f"Best (ep {best_ep})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor=DARK_BG)

    fig.suptitle("Phase 2 — Semi-Supervised DEC Training Curves",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig_loss_phase2.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_epoch_times(epoch_times: list, phase1_epochs: int, out_dir: str) -> str:
    """Bar chart of wall-clock time per epoch, split by phase."""
    n = len(epoch_times)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.4 + 2), 4))
    _apply_dark_style(fig, ax)

    x = list(range(1, n + 1))
    colors = [COLORS[0] if i < phase1_epochs else COLORS[1] for i in range(n)]
    ax.bar(x, epoch_times, color=colors, width=0.7, alpha=0.9)

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLORS[0], label="Phase 1 (SimCLR)"),
        Patch(facecolor=COLORS[1], label="Phase 2 (DEC)"),
    ], fontsize=9, framealpha=0.3, labelcolor="white", facecolor=DARK_BG)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Per-Epoch Training Time", color="white", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig_epoch_times.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_grad_norms(history: dict, phase1_epochs: int, out_dir: str) -> str:
    """Line chart of per-epoch gradient norm, split by phase."""
    p1_norms = history.get("p1_grad_norms", [])
    p2_norms = history.get("p2_grad_norms", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _apply_dark_style(fig, axes)

    for ax, norms, label, color in zip(
        axes,
        [p1_norms, p2_norms],
        ["Phase 1 (SimCLR)", "Phase 2 (DEC)"],
        [COLORS[0], COLORS[1]],
    ):
        if norms:
            ax.plot(range(1, len(norms) + 1), norms, color=color, lw=2, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm (L2)")
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)

    fig.suptitle("Per-Epoch Gradient Norms", color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig_grad_norms.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Exp06: Semi-Supervised DEC on STFT features")
    parser.add_argument("--config", required=False,
                        default="src/models/configs/tune_exp06_dec.yaml",
                        help="Path to the exp06 YAML config (already populated with best tuned params).")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick sanity check: 1 shard per source, 1+1 epochs, batch=4.")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"[Error] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_cfg   = config["experiment"]
    data_cfg  = config["data"]
    train_cfg = config["training"]
    out_cfg   = config["output"]

    print("=" * 60)
    print(f"  Experiment  : {exp_cfg['name']}")
    print(f"  Description : {exp_cfg['description']}")
    print(f"  Parent Node : {exp_cfg['parent_node_id']}")
    if "tuning_node_id" in train_cfg:
        print(f"  Tuned by    : node {train_cfg['tuning_node_id']} @ {train_cfg.get('tuned_at', '?')}")
        print(f"  Tuned LR P1 : {train_cfg['learning_rate_phase1']}")
        print(f"  Tuned LR P2 : {train_cfg['learning_rate_phase2']}")
    print("=" * 60)

    # --- Smoke test overrides ---
    if args.smoke_test:
        for src in data_cfg["sources"]:
            src["train_shards"] = [src["train_shards"][0]]
            src["val_shards"]   = [src["val_shards"][0]]
        train_cfg["phase1_epochs"] = 1
        train_cfg["phase2_epochs"] = 1
        train_cfg["epochs"]        = 2
        train_cfg["batch_size"]    = 4
        print("[Smoke Test] 1 shard, 1+1 epochs, batch=4")

    # --- Device ---
    device = select_device(train_cfg["device"])
    print(f"[Device] {device}")

    # --- Lineage identifiers ---
    node_id  = generate_node_id()
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Prefer the tuning node as parent so the DAG reads: dataset -> tuning -> training.
    # Falls back to the raw dataset node if no tuning has been recorded yet.
    tuning_node_id = train_cfg.get("tuning_node_id")
    parent_id = tuning_node_id if tuning_node_id and tuning_node_id != "manual" \
                else exp_cfg["parent_node_id"]
    print(f"[Lineage] NodeID: {node_id}  |  Parent: {parent_id}")

    # Resolve origin + root_id for the output folder name
    # (pulled from the lineage DB via the parent node)
    try:
        from src.utils.lineage_tracker import get_node_history
        import sqlite3
        from src.utils.lineage_tracker import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT origin, root_id FROM nodes WHERE node_id=?", (parent_id,))
        row = cur.fetchone()
        conn.close()
        _origin  = row[0] if row else "unk"
        _root_id = row[1] if row else "unk"
    except Exception:
        _origin  = "unk"
        _root_id = "unk"

    # Create the structured run output directory
    run_folder_name = f"{run_ts}_{_origin}-{_root_id}-{node_id}"
    perf_base = os.path.abspath("data/performance_evaluation/training")
    run_dir   = os.path.join(perf_base, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Output] Run folder: {run_dir}")

    # -----------------------------------------------------------------------
    # Build Datasets (Phase 1 — augmented pairs for SimCLR)
    # -----------------------------------------------------------------------
    label_fraction = data_cfg.get("label_fraction", 0.10)

    train_ds_p1 = DECDataset_Exp06(
        sources       = data_cfg["sources"],
        shard_key     = "train_shards",
        max_pulse_len = data_cfg["max_pulse_len"],
        augment       = True,
        label_fraction= label_fraction,
    )
    val_ds_p1 = DECDataset_Exp06(
        sources       = data_cfg["sources"],
        shard_key     = "val_shards",
        max_pulse_len = data_cfg["max_pulse_len"],
        augment       = True,
        label_fraction= label_fraction,
    )

    bs = train_cfg["batch_size"]
    loader_kwargs = dict(num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader_p1 = DataLoader(train_ds_p1, batch_size=bs, shuffle=True,  drop_last=True, **loader_kwargs)
    val_loader_p1   = DataLoader(val_ds_p1,   batch_size=bs, shuffle=False, **loader_kwargs)

    print(f"[Data P1] Train={len(train_ds_p1):,}  Val={len(val_ds_p1):,}")

    # -----------------------------------------------------------------------
    # Build Model
    # -----------------------------------------------------------------------
    task = SemiSupervisedDECTask(config).to(device)
    total_params = sum(p.numel() for p in task.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {total_params:,}")

    best_val_loss = float("inf")
    best_epoch    = -1
    epoch_times   = []
    train_start   = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    phase1_epochs = train_cfg.get("phase1_epochs", 10)
    phase2_epochs = train_cfg.get("phase2_epochs", 10)
    lr1 = train_cfg.get("learning_rate_phase1", 0.001)
    lr2 = train_cfg.get("learning_rate_phase2", 0.0001)

    weights_dir     = os.path.abspath(out_cfg["weights_dir"])
    config_snap_dir = os.path.abspath(out_cfg["config_snapshot_dir"])

    # Loss history accumulators
    history = {
        "p1_train_simclr":   [],
        "p1_val_simclr":     [],
        "p1_grad_norms":     [],
        "p2_train_kl":       [],
        "p2_train_pairwise": [],
        "p2_train_total":    [],
        "p2_val_kl":         [],
        "p2_val_pairwise":   [],
        "p2_val_total":      [],
        "p2_grad_norms":     [],
    }

    # -----------------------------------------------------------------------
    # Phase 1 — SimCLR
    # -----------------------------------------------------------------------
    print(f"\n[P1] === SimCLR Pre-Training ({phase1_epochs} epochs) ===")
    task.set_phase(1, lr=lr1)

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()
        task.train()
        train_losses: dict = {}
        for batch in train_loader_p1:
            batch = move_to_device(batch, device)
            _, ld = task.training_step_phase1(batch)
            for k, v in ld.items():
                train_losses[k] = train_losses.get(k, 0.0) + v
        train_losses = {k: v / len(train_loader_p1) for k, v in train_losses.items()}
        epoch_grad_norm = get_grad_norm(task.backbone)

        task.eval()
        val_metrics: dict = {}
        with torch.no_grad():
            for batch in val_loader_p1:
                batch = move_to_device(batch, device)
                m = task.validation_step(batch)
                for k, v in m.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v
        val_metrics = {k: v / len(val_loader_p1) for k, v in val_metrics.items()}

        elapsed = time.time() - t0
        epoch_times.append(round(elapsed, 2))
        history["p1_train_simclr"].append(round(train_losses["simclr"], 6))
        history["p1_val_simclr"].append(round(val_metrics["simclr"], 6))
        history["p1_grad_norms"].append(round(epoch_grad_norm, 6))
        print(f"  P1 Epoch {epoch:03d}/{phase1_epochs} | "
              f"Train SimCLR: {train_losses['simclr']:.4f} | "
              f"Val SimCLR: {val_metrics['simclr']:.4f} | "
              f"GradNorm: {epoch_grad_norm:.4f} | {elapsed:.1f}s")

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch    = epoch
            task.save_checkpoint(epoch, node_id, weights_dir, config_snap_dir, config_path)
            print(f"    ^ Best P1 val loss: {best_val_loss:.4f}")

    # -----------------------------------------------------------------------
    # K-Means Centroid Initialization (between Phase 1 and 2)
    # -----------------------------------------------------------------------
    print("\n[Init] Extracting embeddings for K-Means...")
    kmeans_t0 = time.time()
    init_ds = DECDataset_Exp06(
        sources=data_cfg["sources"], shard_key="train_shards",
        max_pulse_len=data_cfg["max_pulse_len"], augment=False,
        label_fraction=label_fraction,
    )
    init_loader = DataLoader(init_ds, batch_size=bs, shuffle=False, **loader_kwargs)

    all_embs = []
    task.eval()
    with torch.no_grad():
        for batch in init_loader:
            sig = batch[0].to(device)
            z = task.backbone(sig)
            all_embs.append(z.cpu().numpy())
    task.init_cluster_centroids(np.concatenate(all_embs, axis=0))
    kmeans_time_s = round(time.time() - kmeans_t0, 2)
    print(f"[Init] K-Means init completed in {kmeans_time_s:.1f}s")

    # -----------------------------------------------------------------------
    # Phase 2 — Semi-Supervised DEC
    # -----------------------------------------------------------------------
    print(f"\n[P2] === Semi-Supervised DEC Refinement ({phase2_epochs} epochs) ===")
    task.set_phase(2, lr=lr2)

    train_ds_p2 = DECDataset_Exp06(
        sources=data_cfg["sources"], shard_key="train_shards",
        max_pulse_len=data_cfg["max_pulse_len"], augment=False,
        label_fraction=label_fraction,
    )
    val_ds_p2 = DECDataset_Exp06(
        sources=data_cfg["sources"], shard_key="val_shards",
        max_pulse_len=data_cfg["max_pulse_len"], augment=False,
        label_fraction=label_fraction,
    )
    train_loader_p2 = DataLoader(train_ds_p2, batch_size=bs, shuffle=True,  drop_last=True, **loader_kwargs)
    val_loader_p2   = DataLoader(val_ds_p2,   batch_size=bs, shuffle=False, **loader_kwargs)

    best_val_loss_p2 = float("inf")

    for epoch in range(1, phase2_epochs + 1):
        t0 = time.time()
        task.train()
        train_losses = {}
        for batch in train_loader_p2:
            batch = move_to_device(batch, device)
            dec_batch = (batch[0], batch[1])   # (signal, reported_class)
            _, ld = task.training_step_phase2(dec_batch)
            for k, v in ld.items():
                train_losses[k] = train_losses.get(k, 0.0) + v
        train_losses = {k: v / len(train_loader_p2) for k, v in train_losses.items()}
        epoch_grad_norm = get_grad_norm(task)

        task.eval()
        val_metrics = {}
        with torch.no_grad():
            for batch in val_loader_p2:
                batch = move_to_device(batch, device)
                dec_batch = (batch[0], batch[1])
                m = task.validation_step(dec_batch)
                for k, v in m.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v
        val_metrics = {k: v / len(val_loader_p2) for k, v in val_metrics.items()}

        elapsed = time.time() - t0
        epoch_times.append(round(elapsed, 2))
        history["p2_train_kl"].append(round(train_losses.get("kl_div", 0), 6))
        history["p2_train_pairwise"].append(round(train_losses.get("pairwise", 0), 6))
        history["p2_train_total"].append(round(train_losses.get("total", 0), 6))
        history["p2_val_kl"].append(round(val_metrics.get("kl_div", 0), 6))
        history["p2_val_pairwise"].append(round(val_metrics.get("pairwise", 0), 6))
        history["p2_val_total"].append(round(val_metrics.get("total", 0), 6))
        history["p2_grad_norms"].append(round(epoch_grad_norm, 6))
        print(f"  P2 Epoch {epoch:03d}/{phase2_epochs} | "
              f"KL: {train_losses.get('kl_div', 0):.4f} | "
              f"PW: {train_losses.get('pairwise', 0):.4f} | "
              f"Val KL: {val_metrics.get('kl_div', 0):.4f} | "
              f"GradNorm: {epoch_grad_norm:.4f} | {elapsed:.1f}s")

        if val_metrics["total"] < best_val_loss_p2:
            best_val_loss_p2 = val_metrics["total"]
            best_epoch = phase1_epochs + epoch
            task.save_checkpoint(best_epoch, node_id, weights_dir, config_snap_dir, config_path)
            print(f"    ^ Best P2 val loss: {best_val_loss_p2:.4f}")

    best_val_loss = best_val_loss_p2

    # -----------------------------------------------------------------------
    # Save timing + metrics JSON into the run folder
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - train_start
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    timing_report = {
        # ── Identity ────────────────────────────────────────────────────────
        "node_id"              : node_id,
        "parent_id"            : parent_id,
        "experiment"           : exp_cfg["name"],
        "run_folder"           : run_dir,
        # ── Hardware ────────────────────────────────────────────────────────
        "device"               : str(device),
        "gpu_name"             : gpu_name,
        # ── Timing ──────────────────────────────────────────────────────────
        "start_time"           : start_time_str,
        "end_time"             : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_time_s": round(total_elapsed, 2),
        "kmeans_init_time_s"   : kmeans_time_s,
        "mean_epoch_time_s"    : round(sum(epoch_times) / len(epoch_times), 2) if epoch_times else 0,
        "epoch_times_s"        : epoch_times,
        # ── Dataset ─────────────────────────────────────────────────────────
        "n_train_samples_p1"   : len(train_ds_p1),
        "n_val_samples_p1"     : len(val_ds_p1),
        "n_train_samples_p2"   : len(train_ds_p2),
        "n_val_samples_p2"     : len(val_ds_p2),
        "label_fraction"       : label_fraction,
        # ── Model ───────────────────────────────────────────────────────────
        "total_trainable_params": total_params,
        "n_clusters"           : config["model"]["n_clusters"],
        "embedding_dim"        : config["model"]["embedding_dim"],
        "base_channels"        : config["model"]["base_channels"],
        # ── Hyperparameters ──────────────────────────────────────────────────
        "learning_rate_phase1" : lr1,
        "learning_rate_phase2" : lr2,
        "batch_size"           : bs,
        "phase1_epochs"        : phase1_epochs,
        "phase2_epochs"        : phase2_epochs,
        "pairwise_weight_gamma": config.get("task", {}).get("pairwise_weight_gamma", None),
        "simclr_temperature"   : config.get("task", {}).get("simclr_temperature", None),
        # ── Results ─────────────────────────────────────────────────────────
        "best_epoch"           : best_epoch,
        "best_val_loss_p1"     : round(min(history["p1_val_simclr"]), 6) if history["p1_val_simclr"] else None,
        "best_val_loss_p2"     : round(best_val_loss, 6),
        # ── Loss History ────────────────────────────────────────────────────
        "loss_history"         : history,
    }
    timing_path = os.path.join(run_dir, "timing_and_metrics.json")
    with open(timing_path, "w") as f:
        json.dump(timing_report, f, indent=2)
    print(f"\n[Timing] {total_elapsed:.1f}s total. Report -> {timing_path}")

    # -----------------------------------------------------------------------
    # Generate and save plots
    # -----------------------------------------------------------------------
    print("[Plots] Generating training visualizations...")
    if history["p1_train_simclr"]:
        plot_phase1_loss(history, run_dir, phase1_epochs)
    if history["p2_train_kl"]:
        plot_phase2_loss(history, run_dir, phase2_epochs)
    plot_epoch_times(epoch_times, phase1_epochs, run_dir)
    plot_grad_norms(history, phase1_epochs, run_dir)

    # -----------------------------------------------------------------------
    # Lineage Registration
    # -----------------------------------------------------------------------
    history_line = exp_cfg.get("description", "No description.")
    tuning_note  = (f" | Tuned by node {train_cfg['tuning_node_id']} @ {train_cfg.get('tuned_at','?')}"
                    if "tuning_node_id" in train_cfg else "")
    perf_note = (
        f" | BestValP1: {min(history['p1_val_simclr']):.4f}"
        if history["p1_val_simclr"] else ""
    ) + f" | BestValP2: {best_val_loss:.4f} | BestEpoch: {best_epoch}"

    new_node = register_process(
        parent_id        = parent_id,
        stage            = "classification",
        method           = "dec_semi",
        folder_path      = (
            f"{os.path.join(config_snap_dir, f'config_{node_id}.yaml')};"
            f"{os.path.join(weights_dir, f'model_{node_id}.pt')};"
            f"{run_dir}"
        ),
        appended_history = history_line + tuning_note + perf_note,
        force_node_id    = node_id,
    )
    print(f"[Lineage] Registered as Node {new_node} (child of {parent_id})")

    # Write latest node ID for Colab commit messages
    write_latest_node(node_id)

    print(f"\n[Done] Best epoch: {best_epoch} | Best val loss: {best_val_loss:.6f}")
    print(f"[Done] Weights    -> {weights_dir}/model_{node_id}.pt")
    print(f"[Done] Perf data  -> {run_dir}")


if __name__ == "__main__":
    main()
