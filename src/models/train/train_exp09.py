"""
train_exp09.py
==============
Dedicated training orchestrator for Exp09:
  Vision Transformer (ViT) + Instance-Linked SupCon + 4-Domain DANN (GRL)
  + Semi-Supervised Spherical DEC on 2-Channel Complex Bispectrum features.

KEY DIFFERENCES vs train_exp08.py
-----------------------------------
1.  Imports DECDataset_Exp09 and SupConDECTask_Exp09.
2.  Batch tuple unpacking updated for the extended Exp09 format
    (includes global_inst_id and domain_label at indices 3 and 4).
3.  GRL Lambda Scheduling:
      At each epoch, lambda is computed via a sigmoid ramp:
          progress = epoch_idx / total_epochs   (0 → 1 over all training)
          lambda_  = dann_lambda_max * (2/(1+exp(-10*progress)) - 1)
      This keeps the GRL near 0 during early SupCon warm-up and ramps it
      to dann_lambda_max after the backbone has learned basic features.
4.  Domain loss tracked in history and timing report.
5.  Lineage method: "supcon_dec_dann_vit_bispectrum_v2".
6.  Output dir: data/classification_output/exp09_dec/.

Usage:
    python src/models/train/train_exp09.py \\
        --config src/models/configs/exp09_vit_dann.yaml

    # Skip interactive epoch prompt:
    python src/models/train/train_exp09.py --config ... --no_prompt

    # Quick sanity check (1 shard, 1+1 epochs, batch=4):
    python src/models/train/train_exp09.py --config ... --smoke_test

After training:
  - Weights:    models/weights/model_<NodeID>.pt
  - Config:     models/configuration_snapshots/config_<NodeID>.yaml
  - Run folder: data/performance_evaluation/training/<ts>_<origin>-<root>-<node>/
      * timing_and_metrics.json
      * fig_loss_phase1.png
      * fig_loss_phase2.png
      * fig_epoch_times.png
      * fig_grad_norms.png
"""

import argparse
import json
import math
import os
import random
import string
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.data.dataset_exp09       import DECDataset_Exp09
from src.models.tasks.task_exp09         import SupConDECTask_Exp09
from src.utils.lineage_tracker           import register_process, get_node_history


# ---------------------------------------------------------------------------
# Generic Helpers
# ---------------------------------------------------------------------------

def generate_node_id(length: int = 4) -> str:
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def move_to_device(batch: tuple, device: torch.device) -> tuple:
    """Move tensor elements to device; leave strings and non-tensors as-is."""
    return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)


def write_latest_node(node_id: str):
    utils_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "utils"))
    path      = os.path.join(utils_dir, "latest_node.txt")
    with open(path, "w") as f:
        f.write(node_id)
    print(f"[Lineage] Latest node ID written to: {path}")


def get_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    return total_norm ** 0.5


# ---------------------------------------------------------------------------
# Google Drive Intermediate Checkpointing (Colab)
# ---------------------------------------------------------------------------

def gdrive_save(
    node_id:        str,
    epoch:          int,
    phase:          int,
    weights_dir:    str,
    config_path:    str,
    gdrive_dir:     str = "/content/drive/MyDrive/Colab_Intermediate_Training",
):
    """
    Copies the latest checkpoint .pt and config .yaml to Google Drive.

    SAFETY DESIGN:
    - The entire function is wrapped in a broad try/except.
    - ANY failure (Drive not mounted, quota, network drop, permission error)
      prints a warning and returns immediately — training NEVER aborts.
    - Files are written to a temp name first, then renamed atomically
      (prevents a partial write from corrupting the Drive copy).

    Args:
        node_id     : Current training node ID (used in checkpoint filename).
        epoch       : Current epoch number (for log messages only).
        phase       : 1 or 2 (for log messages only).
        weights_dir : Local directory containing model_<node_id>.pt.
        config_path : Path to the YAML config to back up alongside the weights.
        gdrive_dir  : Destination folder on Google Drive.
    """
    try:
        import shutil

        src_weight = os.path.join(weights_dir, f"model_{node_id}.pt")
        if not os.path.exists(src_weight):
            # No checkpoint saved yet (e.g., val loss hasn't improved even once).
            print(f"[GDrive] Epoch {epoch} P{phase}: no checkpoint to copy yet, skipping.")
            return

        os.makedirs(gdrive_dir, exist_ok=True)

        # --- Copy weights (atomic: write to .tmp, then rename) ---
        dst_weight     = os.path.join(gdrive_dir, f"model_{node_id}.pt")
        dst_weight_tmp = dst_weight + ".tmp"
        shutil.copy2(src_weight, dst_weight_tmp)
        os.replace(dst_weight_tmp, dst_weight)

        # --- Copy config ---
        if os.path.exists(config_path):
            dst_config     = os.path.join(gdrive_dir, f"config_{node_id}.yaml")
            dst_config_tmp = dst_config + ".tmp"
            shutil.copy2(config_path, dst_config_tmp)
            os.replace(dst_config_tmp, dst_config)

        print(
            f"[GDrive] Epoch {epoch} P{phase}: checkpoint saved to "
            f"{gdrive_dir}/model_{node_id}.pt"
        )

    except Exception as e:
        # Non-fatal — warn and let training continue
        print(f"[GDrive] WARNING: Drive save FAILED at epoch {epoch} P{phase} — {e}")
        print("[GDrive] Training will continue unaffected.")


# ---------------------------------------------------------------------------
# Resume State — Save and Load for Colab Reconnect Recovery
# ---------------------------------------------------------------------------

def save_resume_state(
    node_id:           str,
    abs_epoch_done:    int,
    phase:             int,
    phase1_epochs:     int,
    phase2_epochs:     int,
    task,
    history:           dict,
    best_val_loss_p1:  float,
    best_val_loss_p2:  float,
    best_epoch:        int,
    epoch_times:       list,
    kmeans_done:       bool,
    local_weights_dir: str,
    gdrive_dir:        str,
):
    """
    Saves a full training-resumption checkpoint as resume_<node_id>.pt.
    Stored locally first, then mirrored to Google Drive (non-fatal on failure).

    NOTE: The optimizer is NOT saved — set_phase() creates a fresh Adam at the
    start of every epoch, so there are no accumulated momentum terms to restore.

    What IS saved:
        model_state_dict    — weights + cluster centroids (nn.Parameter, auto-included)
        history             — loss curves so plots are continuous after resume
        best_val_loss_p1/p2 — checkpoint-on-best-val still works after resume
        best_epoch, epoch_times — for lineage and timing plots
        kmeans_done         — skip K-Means re-run if resuming mid-Phase-2
        grl_lambda          — current GRL lambda for diagnostic accuracy
    """
    import shutil

    state = {
        "abs_epoch_done":   abs_epoch_done,
        "phase":            phase,
        "phase1_epochs":    phase1_epochs,
        "phase2_epochs":    phase2_epochs,
        "node_id":          node_id,
        "model_state":      task.state_dict(),
        "history":          history,
        "best_val_loss_p1": best_val_loss_p1,
        "best_val_loss_p2": best_val_loss_p2,
        "best_epoch":       best_epoch,
        "epoch_times":      epoch_times,
        "kmeans_done":      kmeans_done,
        "grl_lambda":       task.backbone.grl.lambda_,
        "dann_weight":      task.dann_weight,
    }

    # Atomic local write (.tmp then rename prevents corruption on mid-write crash)
    os.makedirs(local_weights_dir, exist_ok=True)
    local_path = os.path.join(local_weights_dir, f"resume_{node_id}.pt")
    tmp_path   = local_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, local_path)
    print(f"[Resume] State saved locally -> {local_path}")

    # Mirror to GDrive (non-fatal)
    try:
        os.makedirs(gdrive_dir, exist_ok=True)
        dst     = os.path.join(gdrive_dir, f"resume_{node_id}.pt")
        dst_tmp = dst + ".tmp"
        shutil.copy2(local_path, dst_tmp)
        os.replace(dst_tmp, dst)
        print(f"[Resume] Mirrored to GDrive -> {dst}")
    except Exception as e:
        print(f"[Resume] WARNING: GDrive mirror FAILED — {e}")
        print("[Resume] Local resume file is intact. Training will continue.")


def load_resume_state(
    resume_path:   str,
    task,
    device:        torch.device,
    phase1_epochs: int,
    phase2_epochs: int,
) -> dict:
    """
    Loads a resume_<node_id>.pt file and restores model + cluster-centroid state.

    Returns a dict with:
        p1_done, p2_done    — epochs already completed in each phase
        kmeans_done         — whether K-Means init has already run
        history             — loss history to extend (not overwrite)
        best_val_loss_p1/p2 — for correct checkpoint-on-best-val after resume
        best_epoch          — for lineage reporting
        epoch_times         — for timing plot continuity
        node_id             — original node_id (preserved for lineage identity)
    """
    if not os.path.exists(resume_path):
        raise FileNotFoundError(
            f"[Resume] File not found: {resume_path}\n"
            f"         Check --resume_from points to a valid resume_<node_id>.pt."
        )

    print(f"[Resume] Loading from: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    # Restores weights AND cluster_layer centroids (they are nn.Parameters)
    task.load_state_dict(ckpt["model_state"])

    abs_done = ckpt.get("abs_epoch_done", 0)
    p1_done  = min(abs_done, phase1_epochs)
    p2_done  = max(0, abs_done - phase1_epochs)

    # Restore GRL state so lambda schedule is consistent
    if "grl_lambda" in ckpt:
        task.backbone.set_dann_lambda(ckpt["grl_lambda"])
    if "dann_weight" in ckpt:
        task.dann_weight = ckpt["dann_weight"]

    print(
        f"[Resume] Restored: abs_epoch={abs_done} | "
        f"P1 done={p1_done}/{phase1_epochs} | "
        f"P2 done={p2_done}/{phase2_epochs} | "
        f"kmeans_done={ckpt.get('kmeans_done', False)}"
    )

    return {
        "p1_done":          p1_done,
        "p2_done":          p2_done,
        "kmeans_done":      ckpt.get("kmeans_done", False),
        "history":          ckpt.get("history"),
        "best_val_loss_p1": ckpt.get("best_val_loss_p1", float("inf")),
        "best_val_loss_p2": ckpt.get("best_val_loss_p2", float("inf")),
        "best_epoch":       ckpt.get("best_epoch", -1),
        "epoch_times":      ckpt.get("epoch_times", []),
        "node_id":          ckpt.get("node_id"),
    }


# ---------------------------------------------------------------------------
# GRL Lambda Schedule
# ---------------------------------------------------------------------------

def compute_dann_lambda(
    epoch_idx:       int,
    total_epochs:    int,
    dann_lambda_max: float,
) -> float:
    """
    Sigmoid ramp for the GRL lambda.

        progress = epoch_idx / max(total_epochs - 1, 1)   ∈ [0, 1]
        lambda_  = dann_lambda_max × (2/(1+exp(-10×progress)) - 1)

    Starts near 0, reaches ~0.76× max at the midpoint, and saturates at max.
    This prevents the GRL from destabilising the backbone before it has
    learned any useful features.
    """
    progress = epoch_idx / max(total_epochs - 1, 1)
    scale    = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return dann_lambda_max * scale


# ---------------------------------------------------------------------------
# Interactive Epoch Selection
# ---------------------------------------------------------------------------

def prompt_epochs(
    default_p1: int,
    default_p2: int,
    no_prompt:  bool = False,
) -> tuple[int, int]:
    if no_prompt:
        print(f"[Epochs] Using YAML defaults: Phase1={default_p1}, Phase2={default_p2}")
        return default_p1, default_p2

    print("\n" + "─" * 60)
    print("  INTERACTIVE EPOCH SELECTION")
    print("  Press Enter to accept the default value shown in [brackets].")
    print("─" * 60)

    for label, default in [("Phase 1 (SupCon+DANN)", default_p1), ("Phase 2 (DEC+DANN)", default_p2)]:
        while True:
            try:
                raw = input(f"  {label} epochs [{default}]: ").strip()
                val = int(raw) if raw else default
                if val < 1:
                    print("  Please enter a positive integer.")
                    continue
                if label.startswith("Phase 1"):
                    phase1_epochs = val
                else:
                    phase2_epochs = val
                break
            except ValueError:
                print("  Invalid input — please enter an integer.")

    print(f"\n[Epochs] Confirmed: Phase1={phase1_epochs}, Phase2={phase2_epochs}")
    print("─" * 60 + "\n")
    return phase1_epochs, phase2_epochs


# ---------------------------------------------------------------------------
# Dark-Theme Plotting
# ---------------------------------------------------------------------------

DARK_BG  = "#0F0F0F"
PANEL_BG = "#1A1A2E"
COLORS   = ["#4CC9F0", "#F72585", "#7209B7", "#4361EE", "#3A0CA3", "#FFBE0B"]


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
    epochs = list(range(1, phase1_epochs + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    _apply_dark_style(fig, axes)

    series = [
        ("p1_train_supcon",  "p1_val_supcon",  "SupCon Loss",  COLORS[0]),
        ("p1_train_domain",  "p1_val_domain",  "Domain Loss",  COLORS[1]),
        ("p1_train_total",   "p1_val_total",   "Total Loss",   COLORS[2]),
    ]
    for ax, (tr_k, vl_k, label, color) in zip(axes, series):
        if history.get(tr_k):
            ax.plot(epochs, history[tr_k], color=color, lw=2, label="Train")
        if history.get(vl_k):
            ax.plot(epochs, history[vl_k], color=color, lw=2, linestyle="--",
                    alpha=0.7, label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label)
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor=DARK_BG)

    fig.suptitle(
        "Phase 1 — Instance-SupCon + DANN Pre-Training (2-Ch Bispectrum ViT)",
        color="white", fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_loss_phase1.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved → {out_path}")
    return out_path


def plot_phase2_loss(history: dict, out_dir: str, phase2_epochs: int) -> str:
    epochs = list(range(1, phase2_epochs + 1))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    _apply_dark_style(fig, axes)

    series = [
        ("p2_train_kl",       "p2_val_kl",       "KL Divergence",  COLORS[0]),
        ("p2_train_pairwise", "p2_val_pairwise",  "Pairwise Loss",  COLORS[1]),
        ("p2_train_domain",   "p2_val_domain",    "Domain Loss",    COLORS[5]),
        ("p2_train_total",    "p2_val_total",     "Total Loss",     COLORS[2]),
    ]
    for ax, (tr_k, vl_k, label, color) in zip(axes, series):
        if history.get(tr_k):
            ax.plot(epochs, history[tr_k], color=color, lw=2, label="Train")
        if history.get(vl_k):
            ax.plot(epochs, history[vl_k], color=color, lw=2, linestyle="--",
                    alpha=0.7, label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label)
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor=DARK_BG)

    fig.suptitle(
        "Phase 2 — Semi-Supervised DEC + DANN (2-Ch Bispectrum ViT)",
        color="white", fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_loss_phase2.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved → {out_path}")
    return out_path


def plot_dann_lambda(history: dict, phase1_epochs: int, out_dir: str) -> str:
    lambdas = history.get("dann_lambda", [])
    if not lambdas:
        return ""
    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_dark_style(fig, ax)
    ax.plot(range(1, len(lambdas) + 1), lambdas, color=COLORS[5], lw=2, marker="o", markersize=3)
    ax.axvline(phase1_epochs + 0.5, color="#888888", lw=1, linestyle=":", alpha=0.7,
               label="Phase 1 → Phase 2")
    ax.set_xlabel("Epoch"); ax.set_ylabel("GRL Lambda (λ)")
    ax.set_title("DANN GRL Lambda Schedule", color="white", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.3, labelcolor="white", facecolor=DARK_BG)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_dann_lambda.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved → {out_path}")
    return out_path


def plot_epoch_times(epoch_times: list, phase1_epochs: int, out_dir: str) -> str:
    n = len(epoch_times)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.4 + 2), 4))
    _apply_dark_style(fig, ax)
    x      = list(range(1, n + 1))
    colors = [COLORS[0] if i < phase1_epochs else COLORS[1] for i in range(n)]
    ax.bar(x, epoch_times, color=colors, width=0.7, alpha=0.9)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS[0], label="Phase 1 (SupCon+DANN)"),
            Patch(facecolor=COLORS[1], label="Phase 2 (DEC+DANN)"),
        ],
        fontsize=9, framealpha=0.3, labelcolor="white", facecolor=DARK_BG,
    )
    ax.set_xlabel("Epoch"); ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Per-Epoch Training Time", color="white", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_epoch_times.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved → {out_path}")
    return out_path


def plot_grad_norms(history: dict, phase1_epochs: int, out_dir: str) -> str:
    p1_norms = history.get("p1_grad_norms", [])
    p2_norms = history.get("p2_grad_norms", [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _apply_dark_style(fig, axes)
    for ax, norms, label, color in zip(
        axes, [p1_norms, p2_norms],
        ["Phase 1 (SupCon+DANN)", "Phase 2 (DEC+DANN)"],
        [COLORS[0], COLORS[1]],
    ):
        if norms:
            ax.plot(range(1, len(norms) + 1), norms, color=color, lw=2, marker="o", markersize=3)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Gradient Norm (L2)")
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)
    fig.suptitle("Per-Epoch Gradient Norms", color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_grad_norms.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Exp09: ViT + Instance-SupCon + 4-Domain DANN + DEC on 2-Ch Bispectra."
    )
    parser.add_argument(
        "--config", required=False,
        default="src/models/configs/exp09_vit_dann.yaml",
        help="Path to the exp09 YAML config file.",
    )
    parser.add_argument("--smoke_test", action="store_true",
                        help="1 shard per source, 1+1 epochs, batch=4.")
    parser.add_argument("--no_prompt", action="store_true",
                        help="Skip interactive epoch prompt; use YAML values.")
    parser.add_argument(
        "--gdrive_interval", type=int, default=3,
        help="Save checkpoint to Google Drive every N epochs (0 = disabled). Default: 3.",
    )
    parser.add_argument(
        "--gdrive_dir", type=str,
        default="/content/drive/MyDrive/Colab_Intermediate_Training",
        help="Google Drive destination folder for intermediate checkpoints.",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        metavar="PATH",
        help=(
            "Path to a resume_<node_id>.pt saved by a previous interrupted run. "
            "The script will skip already-completed epochs and continue from the "
            "saved model weights, loss history, and best-val tracking. "
            "Example: --resume_from /content/drive/MyDrive/Colab_Intermediate_Training/resume_AbCd.pt"
        ),
    )
    parser.add_argument(
        "--skip_phase1", action="store_true",
        help=(
            "Skip Phase 1 entirely and jump straight to K-Means init + Phase 2. "
            "Must be used together with --phase1_weights."
        ),
    )
    parser.add_argument(
        "--phase1_weights", type=str, default=None,
        metavar="PATH",
        help=(
            "Path to a model_<node_id>.pt (best-checkpoint) to load as the "
            "Phase 1 result when using --skip_phase1. "
            "Example: --phase1_weights /content/drive/MyDrive/Colab_Intermediate_Training/model_aoiE.pt"
        ),
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"[Error] Config not found: {config_path}"); sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_cfg   = config["experiment"]
    data_cfg  = config["data"]
    train_cfg = config["training"]
    task_cfg  = config.get("task", {})
    out_cfg   = config["output"]

    print("=" * 60)
    print(f"  Experiment  : {exp_cfg['name']}")
    print(f"  Description : {exp_cfg['description']}")
    print(f"  Parent Node : {exp_cfg['parent_node_id']}")
    print("=" * 60)

    # Smoke test overrides
    if args.smoke_test:
        for src in data_cfg["sources"]:
            src["train_shards"] = [src["train_shards"][0]]
            src["val_shards"]   = [src["val_shards"][0]]
        train_cfg["phase1_epochs"] = 1
        train_cfg["phase2_epochs"] = 1
        train_cfg["batch_size"]    = 4
        print("[Smoke Test] 1 shard per source, 1+1 epochs, batch=4")

    device = select_device(train_cfg["device"])
    print(f"[Device] {device}")

    # Lineage identifiers
    node_id   = generate_node_id()
    run_ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_id = exp_cfg["parent_node_id"]
    print(f"[Lineage] NodeID: {node_id}  |  Parent: {parent_id}")

    # Run folder
    try:
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

    run_folder_name = f"{run_ts}_{_origin}-{_root_id}-{node_id}"
    perf_base = os.path.abspath("data/performance_evaluation/training")
    run_dir   = os.path.join(perf_base, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Output] Run folder: {run_dir}")

    # Interactive epoch selection
    yaml_p1 = train_cfg.get("phase1_epochs", 30)
    yaml_p2 = train_cfg.get("phase2_epochs", 20)
    phase1_epochs, phase2_epochs = prompt_epochs(
        default_p1 = yaml_p1,
        default_p2 = yaml_p2,
        no_prompt  = args.smoke_test or args.no_prompt,
    )
    total_epochs     = phase1_epochs + phase2_epochs
    dann_lambda_max  = task_cfg.get("dann_lambda_max", 1.0)
    dann_weight      = task_cfg.get("dann_weight", 0.5)

    lr1 = train_cfg.get("learning_rate_phase1", 1e-4)
    lr2 = train_cfg.get("learning_rate_phase2", 1e-4)
    bs  = train_cfg["batch_size"]

    weights_dir     = os.path.abspath(out_cfg["weights_dir"])
    config_snap_dir = os.path.abspath(out_cfg["config_snapshot_dir"])

    gdrive_interval = args.gdrive_interval
    gdrive_dir      = args.gdrive_dir
    if gdrive_interval > 0:
        print(f"[GDrive] Intermediate checkpointing ENABLED every {gdrive_interval} epoch(s) -> {gdrive_dir}")
    else:
        print("[GDrive] Intermediate checkpointing DISABLED (--gdrive_interval 0)")

    # Resume tracking — overwritten by load_resume_state() if --resume_from is set
    _resume_p1_done = 0     # Phase 1 epochs already completed before this run
    _resume_p2_done = 0     # Phase 2 epochs already completed before this run
    _kmeans_done    = False  # Whether K-Means init has already been run

    # Domain map from config
    domain_map = data_cfg.get("domain_map", None)

    # -----------------------------------------------------------------------
    # Build Phase 1 Datasets
    # -----------------------------------------------------------------------
    label_fraction = data_cfg.get("label_fraction", 0.10)

    train_ds_p1 = DECDataset_Exp09(
        sources        = data_cfg["sources"],
        shard_key      = "train_shards",
        max_pulse_len  = data_cfg.get("max_pulse_len", 4096),
        augment        = True,
        label_fraction = label_fraction,
        domain_map     = domain_map,
    )
    val_ds_p1 = DECDataset_Exp09(
        sources        = data_cfg["sources"],
        shard_key      = "val_shards",
        max_pulse_len  = data_cfg.get("max_pulse_len", 4096),
        augment        = True,
        label_fraction = label_fraction,
        domain_map     = domain_map,
    )

    num_workers = train_cfg.get("num_workers", 0)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"))
    train_loader_p1 = DataLoader(train_ds_p1, batch_size=bs, shuffle=True,
                                 drop_last=True, **loader_kwargs)
    val_loader_p1   = DataLoader(val_ds_p1,   batch_size=bs, shuffle=False, **loader_kwargs)
    print(f"[Data P1] Train={len(train_ds_p1):,}  Val={len(val_ds_p1):,}")

    # -----------------------------------------------------------------------
    # Build Model
    # -----------------------------------------------------------------------
    task = SupConDECTask_Exp09(config).to(device)
    total_params = sum(p.numel() for p in task.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {total_params:,}")

    # -----------------------------------------------------------------------
    # Initialise tracking variables (may be overwritten by resume loading)
    # -----------------------------------------------------------------------
    best_epoch       = -1
    best_val_loss_p1 = float("inf")
    best_val_loss_p2 = float("inf")
    epoch_times      = []

    _fresh_history = {
        "p1_train_supcon":   [],
        "p1_train_domain":   [],
        "p1_train_total":    [],
        "p1_val_supcon":     [],
        "p1_val_domain":     [],
        "p1_val_total":      [],
        "p1_grad_norms":     [],
        "p2_train_kl":       [],
        "p2_train_pairwise": [],
        "p2_train_domain":   [],
        "p2_train_total":    [],
        "p2_val_kl":         [],
        "p2_val_pairwise":   [],
        "p2_val_domain":     [],
        "p2_val_total":      [],
        "p2_grad_norms":     [],
        "dann_lambda":       [],
    }

    # -----------------------------------------------------------------------
    # Resume Loading
    # -----------------------------------------------------------------------
    if args.resume_from:
        print(f"\n[Resume] === Resuming interrupted run ===")
        rs = load_resume_state(
            resume_path   = args.resume_from,
            task          = task,
            device        = device,
            phase1_epochs = phase1_epochs,
            phase2_epochs = phase2_epochs,
        )
        _resume_p1_done  = rs["p1_done"]
        _resume_p2_done  = rs["p2_done"]
        _kmeans_done     = rs["kmeans_done"]
        best_val_loss_p1 = rs["best_val_loss_p1"]
        best_val_loss_p2 = rs["best_val_loss_p2"]
        best_epoch       = rs["best_epoch"]
        epoch_times      = rs["epoch_times"]
        # Restore history so loss curves are uninterrupted in plots
        history = rs["history"] if rs["history"] is not None else _fresh_history
        # Preserve the original node_id for lineage continuity
        if rs["node_id"] and rs["node_id"] != node_id:
            print(f"[Resume] Adopting original node_id {rs['node_id']} (discarding {node_id})")
            node_id = rs["node_id"]
        print(
            f"[Resume] Skipping P1 epochs 1-{_resume_p1_done}, "
            f"P2 epochs 1-{_resume_p2_done}.\n"
        )
    elif args.skip_phase1:
        # ------------------------------------------------------------------
        # Skip Phase 1: load best-checkpoint weights and jump to Phase 2
        # ------------------------------------------------------------------
        if not args.phase1_weights:
            raise ValueError(
                "[SkipP1] --skip_phase1 requires --phase1_weights PATH. "
                "Point it at the model_<node_id>.pt file you want to use as the Phase 1 result."
            )
        p1w_path = args.phase1_weights
        if not os.path.exists(p1w_path):
            raise FileNotFoundError(f"[SkipP1] Weights file not found: {p1w_path}")

        print(f"\n[SkipP1] === Skipping Phase 1 — loading weights from: {p1w_path} ===")
        ckpt = torch.load(p1w_path, map_location=device, weights_only=False)

        # model_<nodeid>.pt saves under the key 'model_state' (see task_exp09.py)
        raw = ckpt.get("model_state", ckpt)   # fall back to raw state dict
        task.load_state_dict(raw)

        # Adopt the original node_id so lineage is preserved
        if ckpt.get("node_id") and ckpt["node_id"] != node_id:
            print(f"[SkipP1] Adopting original node_id {ckpt['node_id']} (discarding {node_id})")
            node_id = ckpt["node_id"]

        # Mark all P1 epochs as done so the P1 loop is skipped entirely
        _resume_p1_done = phase1_epochs
        history = _fresh_history   # fresh history (P1 curve will be empty, P2 will fill in)

        print(
            f"[SkipP1] Weights loaded. Phase 1 ({phase1_epochs} epochs) will be skipped.\n"
            f"[SkipP1] Will proceed: K-Means init -> Phase 2 ({phase2_epochs} epochs)."
        )
    else:
        history = _fresh_history

    train_start    = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -----------------------------------------------------------------------
    # Phase 1 — Instance-SupCon + DANN Pre-Training
    # -----------------------------------------------------------------------
    print(f"\n[P1] === Instance-SupCon + DANN Pre-Training ({phase1_epochs} epochs) ===")
    if _resume_p1_done > 0:
        print(f"[P1] Resuming from epoch {_resume_p1_done + 1} (best val so far: {best_val_loss_p1:.6f})")

    for epoch in range(1, phase1_epochs + 1):
        # Skip epochs already completed in a previous interrupted run
        if epoch <= _resume_p1_done:
            continue

        # GRL lambda ramp over the full training timeline
        epoch_idx    = epoch - 1
        dann_lambda  = compute_dann_lambda(epoch_idx, total_epochs, dann_lambda_max)
        task.set_phase(1, lr=lr1, dann_weight=dann_weight, dann_lambda=dann_lambda)
        history["dann_lambda"].append(round(dann_lambda, 6))

        t0 = time.time()
        task.train()
        train_losses: dict = {}

        for batch in train_loader_p1:
            batch = move_to_device(batch, device)
            _, ld = task.training_step_phase1(batch)
            for k, v in ld.items():
                train_losses[k] = train_losses.get(k, 0.0) + v

        if len(train_loader_p1) > 0:
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
        if len(val_loader_p1) > 0:
            val_metrics = {k: v / len(val_loader_p1) for k, v in val_metrics.items()}

        elapsed = time.time() - t0
        epoch_times.append(round(elapsed, 2))
        history["p1_train_supcon"].append(round(train_losses.get("supcon", 0.0), 6))
        history["p1_train_domain"].append(round(train_losses.get("domain", 0.0), 6))
        history["p1_train_total"].append(round(train_losses.get("total",  0.0), 6))
        history["p1_val_supcon"].append(round(val_metrics.get("supcon", 0.0), 6))
        history["p1_val_domain"].append(round(val_metrics.get("domain", 0.0), 6))
        history["p1_val_total"].append(round(val_metrics.get("total",  0.0), 6))
        history["p1_grad_norms"].append(round(epoch_grad_norm, 6))

        print(
            f"  P1 Epoch {epoch:03d}/{phase1_epochs} | "
            f"SupCon: {train_losses.get('supcon', 0.0):.4f} | "
            f"Domain: {train_losses.get('domain', 0.0):.4f} | "
            f"Val SupCon: {val_metrics.get('supcon', 0.0):.4f} | "
            f"λ_grl: {dann_lambda:.4f} | "
            f"GradNorm: {epoch_grad_norm:.4f} | {elapsed:.1f}s"
        )

        # Checkpoint on best Phase 1 val SupCon loss (primary objective)
        if val_metrics.get("supcon", float("inf")) < best_val_loss_p1:
            best_val_loss_p1 = val_metrics["supcon"]
            best_epoch       = epoch
            task.save_checkpoint(epoch, node_id, weights_dir, config_snap_dir, config_path)
            print(f"    ^ Best P1 val SupCon: {best_val_loss_p1:.4f}")

        # --- Periodic Google Drive backup + resume state save ---
        if gdrive_interval > 0 and epoch % gdrive_interval == 0:
            gdrive_save(
                node_id     = node_id,
                epoch       = epoch,
                phase       = 1,
                weights_dir = weights_dir,
                config_path = config_path,
                gdrive_dir  = gdrive_dir,
            )
            save_resume_state(
                node_id           = node_id,
                abs_epoch_done    = epoch,
                phase             = 1,
                phase1_epochs     = phase1_epochs,
                phase2_epochs     = phase2_epochs,
                task              = task,
                history           = history,
                best_val_loss_p1  = best_val_loss_p1,
                best_val_loss_p2  = best_val_loss_p2,
                best_epoch        = best_epoch,
                epoch_times       = epoch_times,
                kmeans_done       = False,
                local_weights_dir = weights_dir,
                gdrive_dir        = gdrive_dir,
            )

    # -----------------------------------------------------------------------
    # K-Means Centroid Initialisation
    # -----------------------------------------------------------------------
    if _kmeans_done:
        print("\n[Init] K-Means SKIPPED — cluster centroids restored from resume checkpoint.")
        kmeans_time_s = 0.0
    else:
        print("\n[Init] Extracting embeddings for K-Means centroid init...")
        kmeans_t0 = time.time()

        init_ds = DECDataset_Exp09(
            sources        = data_cfg["sources"],
            shard_key      = "train_shards",
            max_pulse_len  = data_cfg.get("max_pulse_len", 4096),
            augment        = False,
            label_fraction = label_fraction,
            domain_map     = domain_map,
        )
        init_loader = DataLoader(init_ds, batch_size=bs, shuffle=False, **loader_kwargs)

        all_embs = []
        task.eval()
        with torch.no_grad():
            for batch in init_loader:
                sig = batch[0].to(device)   # (B, 2, 128, 128)
                z, _ = task.backbone(sig)
                all_embs.append(z.cpu().numpy())

        if not all_embs:
            raise RuntimeError("[Init] No embeddings extracted for K-Means.")

        task.init_cluster_centroids(np.concatenate(all_embs, axis=0))
        kmeans_time_s = round(time.time() - kmeans_t0, 2)
        print(f"[Init] K-Means init completed in {kmeans_time_s:.1f}s")

    # -----------------------------------------------------------------------
    # Phase 2 — Semi-Supervised Spherical DEC + DANN
    # -----------------------------------------------------------------------
    print(f"\n[P2] === Semi-Supervised DEC + DANN ({phase2_epochs} epochs) ===")

    train_ds_p2 = DECDataset_Exp09(
        sources        = data_cfg["sources"],
        shard_key      = "train_shards",
        max_pulse_len  = data_cfg.get("max_pulse_len", 4096),
        augment        = False,
        label_fraction = label_fraction,
        domain_map     = domain_map,
    )
    val_ds_p2 = DECDataset_Exp09(
        sources        = data_cfg["sources"],
        shard_key      = "val_shards",
        max_pulse_len  = data_cfg.get("max_pulse_len", 4096),
        augment        = False,
        label_fraction = label_fraction,
        domain_map     = domain_map,
    )
    train_loader_p2 = DataLoader(train_ds_p2, batch_size=bs, shuffle=True,
                                 drop_last=True, **loader_kwargs)
    val_loader_p2   = DataLoader(val_ds_p2,   batch_size=bs, shuffle=False, **loader_kwargs)

    if _resume_p2_done > 0:
        print(f"[P2] Resuming from epoch {_resume_p2_done + 1} (best val so far: {best_val_loss_p2:.6f})")

    for epoch in range(1, phase2_epochs + 1):
        # Skip epochs already completed in a previous interrupted run
        if epoch <= _resume_p2_done:
            continue

        # Continue GRL ramp from where Phase 1 left off
        epoch_idx   = (phase1_epochs - 1) + epoch
        dann_lambda = compute_dann_lambda(epoch_idx, total_epochs, dann_lambda_max)
        task.set_phase(2, lr=lr2, dann_weight=dann_weight, dann_lambda=dann_lambda)
        history["dann_lambda"].append(round(dann_lambda, 6))

        t0 = time.time()
        task.train()
        train_losses = {}

        for batch in train_loader_p2:
            batch     = move_to_device(batch, device)
            # Phase 2 step expects (signal, reported_class, domain_label)
            dec_batch = (batch[0], batch[1], batch[3])
            _, ld = task.training_step_phase2(dec_batch)
            for k, v in ld.items():
                train_losses[k] = train_losses.get(k, 0.0) + v

        if len(train_loader_p2) > 0:
            train_losses = {k: v / len(train_loader_p2) for k, v in train_losses.items()}

        epoch_grad_norm = get_grad_norm(task)

        task.eval()
        val_metrics = {}
        with torch.no_grad():
            for batch in val_loader_p2:
                batch     = move_to_device(batch, device)
                val_batch = (batch[0], batch[1], batch[3])
                m = task.validation_step(val_batch)
                for k, v in m.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v
        if len(val_loader_p2) > 0:
            val_metrics = {k: v / len(val_loader_p2) for k, v in val_metrics.items()}

        elapsed = time.time() - t0
        epoch_times.append(round(elapsed, 2))
        history["p2_train_kl"].append(round(train_losses.get("kl_div",   0), 6))
        history["p2_train_pairwise"].append(round(train_losses.get("pairwise", 0), 6))
        history["p2_train_domain"].append(round(train_losses.get("domain",   0), 6))
        history["p2_train_total"].append(round(train_losses.get("total",    0), 6))
        history["p2_val_kl"].append(round(val_metrics.get("kl_div",   0), 6))
        history["p2_val_pairwise"].append(round(val_metrics.get("pairwise", 0), 6))
        history["p2_val_domain"].append(round(val_metrics.get("domain",   0), 6))
        history["p2_val_total"].append(round(val_metrics.get("total",    0), 6))
        history["p2_grad_norms"].append(round(epoch_grad_norm, 6))

        print(
            f"  P2 Epoch {epoch:03d}/{phase2_epochs} | "
            f"KL: {train_losses.get('kl_div', 0):.4f} | "
            f"PW: {train_losses.get('pairwise', 0):.4f} | "
            f"Dom: {train_losses.get('domain', 0):.4f} | "
            f"Val KL: {val_metrics.get('kl_div', 0):.4f} | "
            f"λ_grl: {dann_lambda:.4f} | {elapsed:.1f}s"
        )

        # Checkpoint on best Phase 2 DEC val loss (kl + pairwise, excl. domain)
        dec_val = val_metrics.get("total", float("inf"))
        if dec_val < best_val_loss_p2:
            best_val_loss_p2 = dec_val
            best_epoch       = phase1_epochs + epoch
            task.save_checkpoint(best_epoch, node_id, weights_dir, config_snap_dir, config_path)
            print(f"    ^ Best P2 val DEC loss: {best_val_loss_p2:.4f}")

        # --- Periodic Google Drive backup + resume state save ---
        if gdrive_interval > 0 and epoch % gdrive_interval == 0:
            gdrive_save(
                node_id     = node_id,
                epoch       = phase1_epochs + epoch,
                phase       = 2,
                weights_dir = weights_dir,
                config_path = config_path,
                gdrive_dir  = gdrive_dir,
            )
            save_resume_state(
                node_id           = node_id,
                abs_epoch_done    = phase1_epochs + epoch,
                phase             = 2,
                phase1_epochs     = phase1_epochs,
                phase2_epochs     = phase2_epochs,
                task              = task,
                history           = history,
                best_val_loss_p1  = best_val_loss_p1,
                best_val_loss_p2  = best_val_loss_p2,
                best_epoch        = best_epoch,
                epoch_times       = epoch_times,
                kmeans_done       = True,   # K-Means is always done by Phase 2
                local_weights_dir = weights_dir,
                gdrive_dir        = gdrive_dir,
            )

    # -----------------------------------------------------------------------
    # Save timing + metrics JSON
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - train_start
    try:
        gpu_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    except Exception:
        gpu_name = "cuda"

    timing_report = {
        "node_id":               node_id,
        "parent_id":             parent_id,
        "experiment":            exp_cfg["name"],
        "run_folder":            run_dir,
        "device":                str(device),
        "gpu_name":              gpu_name,
        "start_time":            start_time_str,
        "end_time":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_time_s": round(total_elapsed, 2),
        "kmeans_init_time_s":    kmeans_time_s,
        "mean_epoch_time_s":     round(sum(epoch_times) / len(epoch_times), 2) if epoch_times else 0,
        "epoch_times_s":         epoch_times,
        "n_train_samples_p1":    len(train_ds_p1),
        "n_val_samples_p1":      len(val_ds_p1),
        "n_train_samples_p2":    len(train_ds_p2),
        "n_val_samples_p2":      len(val_ds_p2),
        "label_fraction":        label_fraction,
        "total_trainable_params": total_params,
        "n_clusters":            config["model"]["n_clusters"],
        "n_domains":             config["model"].get("n_domains", 4),
        "embedding_dim":         config["model"]["embedding_dim"],
        "in_channels":           config["model"].get("in_channels", 2),
        "learning_rate_phase1":  lr1,
        "learning_rate_phase2":  lr2,
        "batch_size":            bs,
        "phase1_epochs":         phase1_epochs,
        "phase2_epochs":         phase2_epochs,
        "dann_weight":           dann_weight,
        "dann_lambda_max":       dann_lambda_max,
        "pairwise_weight_gamma": task_cfg.get("pairwise_weight_gamma", None),
        "supcon_temperature":    task_cfg.get("simclr_temperature", None),
        "best_epoch":            best_epoch,
        "best_val_loss_p1":      round(best_val_loss_p1, 6),
        "best_val_loss_p2":      round(best_val_loss_p2, 6),
        "loss_history":          history,
    }
    timing_path = os.path.join(run_dir, "timing_and_metrics.json")
    with open(timing_path, "w") as f:
        json.dump(timing_report, f, indent=2)
    print(f"\n[Timing] {total_elapsed:.1f}s total. Report → {timing_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("[Plots] Generating training visualisations...")
    if history["p1_train_supcon"]:
        plot_phase1_loss(history, run_dir, phase1_epochs)
    if history["p2_train_kl"]:
        plot_phase2_loss(history, run_dir, phase2_epochs)
    plot_dann_lambda(history, phase1_epochs, run_dir)
    plot_epoch_times(epoch_times, phase1_epochs, run_dir)
    plot_grad_norms(history, phase1_epochs, run_dir)

    # -----------------------------------------------------------------------
    # Lineage Registration
    # -----------------------------------------------------------------------
    history_line = exp_cfg.get("description", "No description.")
    perf_note = (
        f" | BestValP1SupCon: {best_val_loss_p1:.4f}"
        f" | BestValP2DEC: {best_val_loss_p2:.4f}"
        f" | BestEpoch: {best_epoch}"
        f" | dann_weight: {dann_weight}"
        f" | dann_lambda_max: {dann_lambda_max}"
    )

    new_node = register_process(
        parent_id        = parent_id,
        stage            = "classification",
        method           = "supcon_dec_dann_vit_bispectrum_v2",
        folder_path      = (
            f"{os.path.join(config_snap_dir, f'config_{node_id}.yaml')};"
            f"{os.path.join(weights_dir, f'model_{node_id}.pt')};"
            f"{run_dir}"
        ),
        appended_history = history_line + perf_note,
        force_node_id    = node_id,
    )
    print(f"[Lineage] Registered as Node {new_node} (child of {parent_id})")

    write_latest_node(node_id)

    print(f"\n[Done] Best epoch: {best_epoch} | Best P2 val loss: {best_val_loss_p2:.6f}")
    print(f"[Done] Weights    → {weights_dir}/model_{node_id}.pt")
    print(f"[Done] Perf data  → {run_dir}")


if __name__ == "__main__":
    main()
