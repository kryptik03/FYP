"""
predict_exp09.py
================
Inference and Analytics Pipeline for Exp09:
  ViT + Instance-SupCon + 4-Domain DANN (GRL) + Semi-Supervised DEC
  on 2-Channel Complex Bispectrum features.

KEY DIFFERENCES vs predict_exp08.py
--------------------------------------
1.  UMAP instead of t-SNE
    UMAP is topology-preserving, faster, and can `transform()` new points
    (a prerequisite for Online Detection). t-SNE shreds high-dimensional
    manifolds into arbitrary blobs and misleads HDBSCAN.
    Install: pip install umap-learn

2.  Domain-Origin Visualisation (fig2)
    Extra plot: embeddings coloured by source domain
    (equation / synthesised / cwru / measured).
    Goal: complete visual overlap between domains = DANN succeeded.

3.  2-Channel Input
    The dataset and backbone now expect (B, 2, 128, 128) inputs.
    Single-channel (V1) feature shards will be rejected with a clear error.

4.  Maintained predictions.h5 Schema
    The output HDF5 still contains shard_path, scene_idx, ch_idx, start_idx,
    gt_class_id, pred_class_id, cluster_id, gt_inst_id, pred_inst_id, time_res,
    pulse_idx — identical to Exp08 format. localise.py requires no changes.

USAGE
-----
    python src/models/predictions/predict_exp09.py \\
        --checkpoint_id <node_id> \\
        --source "data/features/bispectra_v2/<dir>:measured:all_shards"

    # Multiple sources:
    python src/models/predictions/predict_exp09.py \\
        --checkpoint_id <node_id> \\
        --source "data/features/bispectra_v2/<dir1>:measured:all_shards" \\
        --source "data/features/bispectra_v2/<dir2>:equation:all_shards"

    # Use all shards in each split:
    --source "path:type:all_shards"

FIGURES PRODUCED
----------------
    fig1_umap_by_class.png       — UMAP coloured by ground-truth class
    fig2_umap_by_domain.png      — UMAP coloured by source domain (overlap = success)
    fig3_cluster_composition.png — HDBSCAN cluster composition by GT class (stacked bar)
    fig4_soft_q_heatmap.png      — DEC soft-assignment Q matrix
"""

import argparse
import glob
import json
import os
import random
import sqlite3
import string
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.data.dataset_exp09 import DECDataset_Exp09
from src.models.tasks.task_exp09   import SupConDECTask_Exp09
from src.utils.lineage_tracker     import register_process


# ---------------------------------------------------------------------------
# Class name mapping (shared with Exp08)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "Void Sim",     1: "Incision Sim",  2: "Void Meas",
    3: "Incision Meas",4: "Delamination",  5: "FeOx",
    6: "FeOx High",    7: "SEDO",          8: "DED",
    9: "DEDO",         10: "SMG",
    11: "CWRU Normal", 12: "CWRU B007",    13: "CWRU B014",
    14: "CWRU B021",   15: "CWRU IR007",   16: "CWRU IR014",
    17: "CWRU IR021",  18: "CWRU OR007",   19: "CWRU OR014",
    20: "CWRU OR021",
}

DOMAIN_NAMES = {
    0: "Equation (Math)",
    1: "Synthesised (HFSS)",
    2: "CWRU (Lab)",
    3: "Measured (UHF)",
   -1: "Unknown",
}

DARK_BG  = "#0F0F0F"
PANEL_BG = "#1A1A2E"


# ---------------------------------------------------------------------------
# Source Parsing
# ---------------------------------------------------------------------------

def parse_sources(source_strs: list[str], all_h5_in_dir: bool = False) -> list[dict]:
    """
    Parse --source arguments into dataset source dicts.

    Format: "path:type:all_shards"  or  "path:type:1,2,3,4"

    Example:
        "data/features/bispectra_v2/20260601-ms-abc-XYZ:measured:all_shards"
        "data/features/bispectra_v2/20260601-sy-def-ABC:equation:1,2,3"
    """
    sources = []
    for src_str in source_strs:
        parts = src_str.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid --source format: '{src_str}'. Expected path:type[:shards]")

        raw_path    = parts[0]
        source_type = parts[1]
        shard_spec  = parts[2] if len(parts) > 2 else "all_shards"

        abs_path = os.path.abspath(raw_path)
        if not os.path.isdir(abs_path):
            print(f"[Warning] Source directory not found, skipping: {abs_path}")
            continue

        # Discover shards
        h5_files = sorted(glob.glob(os.path.join(abs_path, "shard_*.h5")))
        if not h5_files:
            print(f"[Warning] No shard_*.h5 files found in: {abs_path}")
            continue

        def _shard_id(f):
            base = os.path.splitext(os.path.basename(f))[0]  # "shard_03"
            return int(base.split("_")[1])

        all_ids = [_shard_id(f) for f in h5_files]

        if shard_spec == "all_shards":
            shard_ids = all_ids
        else:
            shard_ids = [int(x) for x in shard_spec.split(",")]

        sources.append({
            "type":         source_type,
            "path":         abs_path,
            "train_shards": shard_ids,
            "val_shards":   shard_ids,
        })
        print(f"  Source: [{source_type}] {abs_path} — {len(shard_ids)} shard(s)")

    return sources


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_id: str,
    config:        dict,
    weights_dir:   str,
    device:        torch.device,
) -> SupConDECTask_Exp09:
    weight_path = os.path.join(weights_dir, f"model_{checkpoint_id}.pt")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    task = SupConDECTask_Exp09(config).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    task.load_state_dict(ckpt["model_state"])
    task.eval()
    print(f"[Checkpoint] Loaded → {weight_path}")
    print(f"             Saved at epoch : {ckpt.get('epoch', '?')}")
    print(f"             n_clusters     : {ckpt.get('n_clusters', '?')}")
    print(f"             n_domains      : {ckpt.get('n_domains', '?')}")
    print(f"             dann_weight    : {ckpt.get('dann_weight', '?')}")
    return task


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(
    task:       SupConDECTask_Exp09,
    loader:     DataLoader,
    device:     torch.device,
    dataset:    DECDataset_Exp09,
) -> dict:
    """
    Run the model on all samples. Returns a dict of numpy arrays:
        embs            : (N, embedding_dim)
        q_soft          : (N, K)
        domain_logits   : (N, n_domains)
        gt_class_ids    : (N,)
        domain_labels   : (N,)
        reported_classes: (N,)
        shard_paths     : list[str]   (N,)
        pulse_idxs      : (N,)  — index within the shard's labels array
        time_res_arr    : (N,)
    """
    all_embs          = []
    all_q             = []
    all_domain_logits = []
    all_gt_classes    = []
    all_domain_labels = []
    all_reported      = []
    all_shard_paths   = []
    all_pulse_idxs    = []
    all_time_res      = []

    task.eval()
    with torch.no_grad():
        for batch in loader:
            sig            = batch[0].to(device)   # (B, 2, 128, 128)
            reported_class = batch[1]              # (B,)
            global_inst_id = batch[2]              # (B,) unused here
            domain_label   = batch[3]              # (B,)
            shard_path     = batch[4]              # list[str]
            pulse_idx      = batch[5]              # (B,)
            time_res       = batch[6]              # (B,)
            actual_class   = batch[7]              # (B,)

            with torch.cuda.amp.autocast(enabled=getattr(task, "use_amp", False)):
                z, domain_logit = task.backbone(sig)
                q               = task.soft_assign(z)

            all_embs.append(z.cpu().numpy())
            all_q.append(q.cpu().numpy())
            all_domain_logits.append(domain_logit.cpu().numpy())
            all_gt_classes.extend(actual_class.numpy().tolist())
            all_domain_labels.extend(domain_label.numpy().tolist())
            all_reported.extend(reported_class.numpy().tolist())
            all_shard_paths.extend(list(shard_path))
            all_pulse_idxs.extend(pulse_idx.numpy().tolist())
            all_time_res.extend(time_res.numpy().tolist())

    return {
        "embs":             np.concatenate(all_embs,          axis=0),
        "q_soft":           np.concatenate(all_q,             axis=0),
        "domain_logits":    np.concatenate(all_domain_logits, axis=0),
        "gt_class_ids":     np.array(all_gt_classes,    dtype=np.int32),
        "domain_labels":    np.array(all_domain_labels, dtype=np.int32),
        "reported_classes": np.array(all_reported,      dtype=np.int32),
        "shard_paths":      all_shard_paths,
        "pulse_idxs":       np.array(all_pulse_idxs,    dtype=np.int32),
        "time_res_arr":     np.array(all_time_res,       dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# UMAP Projection
# ---------------------------------------------------------------------------

def umap_2d(embs: np.ndarray) -> np.ndarray:
    """
    Reduce embeddings to 2D using UMAP with cosine metric.

    Why UMAP over t-SNE:
    - Preserves global topological structure (manifold fidelity).
    - Scalable to large datasets via approximate nearest neighbours.
    - Parametric mode: can `transform()` new points in real-time.
    - t-SNE's stochastic layout shreds manifolds, misleading HDBSCAN.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required. Install via: pip install umap-learn"
        )
    print(f"[UMAP] Reducing {len(embs):,} × {embs.shape[1]}-D embeddings to 2D...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",   # L2-normalised → cosine is the natural metric
        random_state=42,
        verbose=False,
    )
    xy = reducer.fit_transform(embs)
    print(f"[UMAP] Done in {time.time() - t0:.1f}s")
    return xy.astype(np.float32)


# ---------------------------------------------------------------------------
# HDBSCAN Clustering
# ---------------------------------------------------------------------------

def hdbscan_cluster(xy: np.ndarray, min_cluster_size: int = 30) -> np.ndarray:
    """Run HDBSCAN on UMAP-projected 2D coordinates."""
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        raise ImportError("hdbscan is required. Install via: pip install hdbscan")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels    = clusterer.fit_predict(xy)
    n_found   = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise   = (labels == -1).sum()
    print(f"[HDBSCAN] Found {n_found} clusters | Noise: {n_noise} / {len(labels)}")
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Cluster-to-Class Alignment (Hungarian)
# ---------------------------------------------------------------------------

def align_clusters_to_classes(
    cluster_ids: np.ndarray,
    gt_class_ids: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Hungarian algorithm to align cluster labels to GT class labels.
    Returns (pred_class_ids, cluster_to_class_map, confusion_matrix).
    """
    from scipy.optimize import linear_sum_assignment

    cluster_labels = sorted(c for c in np.unique(cluster_ids) if c >= 0)
    n_clusters     = len(cluster_labels)

    cost = np.zeros((n_clusters, n_classes), dtype=np.int32)
    for ci, cid in enumerate(cluster_labels):
        mask = cluster_ids == cid
        for g in gt_class_ids[mask]:
            if 0 <= g < n_classes:
                cost[ci, g] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)
    cluster_to_class = {cluster_labels[r]: col_ind[r] for r in row_ind}

    pred_class_ids = np.full_like(cluster_ids, fill_value=-1)
    for cid, cls in cluster_to_class.items():
        pred_class_ids[cluster_ids == cid] = cls

    conf = np.zeros((n_classes, n_classes), dtype=np.int32)
    for gt, pred in zip(gt_class_ids, pred_class_ids):
        if 0 <= gt < n_classes and 0 <= pred < n_classes:
            conf[gt, pred] += 1

    return pred_class_ids, cluster_to_class, conf


# ---------------------------------------------------------------------------
# Instance Grouping
# ---------------------------------------------------------------------------

def group_by_instance(
    pulse_idxs:    np.ndarray,
    pred_class_ids: np.ndarray,
    shard_paths:   list,
    dataset:       DECDataset_Exp09,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign a predicted instance class by majority vote over all sensors
    that observed the same physical event (same Pulse_Instance_ID).
    Returns (gt_inst_ids, pred_inst_ids).
    """
    n = len(pulse_idxs)
    gt_inst_ids   = np.full(n, -1, dtype=np.int32)
    pred_inst_ids = np.full(n, -1, dtype=np.int32)

    # Group flat indices by (shard_path, global_inst_id)
    from collections import defaultdict
    group: dict[int, list[int]] = defaultdict(list)

    for i in range(n):
        # Recover global_inst_id from dataset index
        sp = shard_paths[i]
        for entry_flat, entry in enumerate(dataset.index):
            if entry[0] == sp and entry[1] == int(pulse_idxs[i]):
                gid = entry[4]  # global_inst_id
                gt_inst_ids[i] = gid
                group[gid].append(i)
                break

    for gid, idxs in group.items():
        preds = pred_class_ids[idxs]
        preds = preds[preds >= 0]
        if len(preds) > 0:
            majority = int(np.bincount(preds).argmax())
            for i in idxs:
                pred_inst_ids[i] = majority

    return gt_inst_ids, pred_inst_ids


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    gt_class_ids:   np.ndarray,
    pred_class_ids: np.ndarray,
    pred_inst_ids:  np.ndarray,
    gt_inst_ids:    np.ndarray,
    embs:           np.ndarray,
    cluster_ids:    np.ndarray,
    q_soft:         np.ndarray,
) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, silhouette_score,
        adjusted_rand_score, normalized_mutual_info_score,
    )

    valid_mask    = (pred_class_ids >= 0) & (gt_class_ids >= 0)
    gt_v          = gt_class_ids[valid_mask]
    pred_v        = pred_class_ids[valid_mask]

    cls_acc  = accuracy_score(gt_v, pred_v) if len(gt_v) > 0 else 0.0
    cls_f1   = f1_score(gt_v, pred_v, average="macro", zero_division=0) if len(gt_v) > 0 else 0.0
    ari      = adjusted_rand_score(gt_class_ids, cluster_ids)
    nmi      = normalized_mutual_info_score(gt_class_ids, cluster_ids)

    # Silhouette on embeddings
    uc = np.unique(cluster_ids[cluster_ids >= 0])
    if len(uc) > 1:
        sil_mask = cluster_ids >= 0
        sil = float(silhouette_score(embs[sil_mask], cluster_ids[sil_mask], metric="cosine"))
    else:
        sil = float("nan")

    # Instance grouping F1
    inst_valid = (pred_inst_ids >= 0) & (gt_inst_ids >= 0)
    gi, pi     = gt_inst_ids[inst_valid], pred_inst_ids[inst_valid]
    group_f1   = f1_score(gi, pi, average="macro", zero_division=0) if len(gi) > 0 else 0.0

    # DEC entropy (lower = more confident clusters)
    entropy = float(-np.mean(np.sum(q_soft * np.log(q_soft + 1e-9), axis=1)))

    return {
        "cls_accuracy":  round(cls_acc,  4),
        "cls_f1_macro":  round(cls_f1,   4),
        "grouping_f1":   round(group_f1, 4),
        "silhouette":    round(sil, 4) if not np.isnan(sil) else None,
        "ari":           round(ari, 4),
        "nmi":           round(nmi, 4),
        "dec_entropy":   round(entropy, 4),
        "n_clusters":    int(len(np.unique(cluster_ids[cluster_ids >= 0]))),
        "n_noise":       int((cluster_ids == -1).sum()),
        "n_total":       int(len(cluster_ids)),
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _dark_scatter_setup(fig, ax, title: str):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def plot_umap_by_class(xy: np.ndarray, gt_class_ids: np.ndarray, out_path: str):
    """Fig 1: UMAP coloured by ground-truth PD class."""
    unique_classes = np.unique(gt_class_ids)
    cmap   = plt.get_cmap("tab20", max(len(unique_classes), 1))
    colors = {cls: cmap(i) for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(12, 9))
    _dark_scatter_setup(fig, ax, "UMAP Projection — Coloured by PD Class (Exp09)")

    for cls in unique_classes:
        mask = gt_class_ids == cls
        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, alpha=0.65,
                   color=colors[cls], rasterized=True,
                   label=CLASS_NAMES.get(cls, f"Class {cls}"))

    legend = ax.legend(
        loc="best", fontsize=7.5, framealpha=0.25,
        labelcolor="white", facecolor=DARK_BG, ncol=2,
    )
    ax.set_xlabel("UMAP-1", color="#AAAAAA")
    ax.set_ylabel("UMAP-2", color="#AAAAAA")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] fig1 saved → {out_path}")


def plot_umap_by_domain(xy: np.ndarray, domain_labels: np.ndarray, out_path: str):
    """
    Fig 2: UMAP coloured by source domain.

    SUCCESS INDICATOR: if DANN worked, embeddings from all 4 domains
    should be COMPLETELY INTERLEAVED — you should not be able to see
    clean boundaries between the coloured regions.
    """
    unique_domains = sorted(set(domain_labels.tolist()))
    domain_colors  = {
        0:  "#4CC9F0",   # Equation (Math)       — cyan
        1:  "#7209B7",   # Synthesised (HFSS)    — purple
        2:  "#F72585",   # CWRU (Lab)            — pink/red
        3:  "#FFBE0B",   # Measured (UHF)        — gold
       -1:  "#555555",   # Unknown               — grey
    }

    fig, ax = plt.subplots(figsize=(12, 9))
    _dark_scatter_setup(
        fig, ax,
        "UMAP — Coloured by Source Domain (Exp09)\n"
        "↑ Complete overlap = DANN succeeded — domain-agnostic embeddings ↑",
    )

    for d in unique_domains:
        mask = domain_labels == d
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.6,
                   color=domain_colors.get(d, "#888888"),
                   rasterized=True, zorder=2,
                   label=DOMAIN_NAMES.get(d, f"Domain {d}"))

    legend = ax.legend(
        loc="best", fontsize=9, framealpha=0.25,
        labelcolor="white", facecolor=DARK_BG,
    )
    ax.set_xlabel("UMAP-1", color="#AAAAAA")
    ax.set_ylabel("UMAP-2", color="#AAAAAA")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] fig2 saved → {out_path}")


def plot_cluster_composition(
    cluster_ids:  np.ndarray,
    gt_class_ids: np.ndarray,
    n_classes:    int,
    out_path:     str,
):
    """Fig 3: Stacked bar — HDBSCAN cluster composition by GT class."""
    unique_clusters = sorted(c for c in np.unique(cluster_ids) if c >= 0)
    if not unique_clusters:
        print("[Plot] No valid clusters for composition plot.")
        return

    cmap         = plt.get_cmap("tab20", n_classes)
    class_colors = [cmap(c) for c in range(n_classes)]

    data = np.zeros((len(unique_clusters), n_classes), dtype=np.int32)
    for ci, cid in enumerate(unique_clusters):
        mask = cluster_ids == cid
        for g in gt_class_ids[mask]:
            if 0 <= g < n_classes:
                data[ci, g] += 1

    fig, ax = plt.subplots(figsize=(max(10, len(unique_clusters) * 0.6 + 3), 6))
    _dark_scatter_setup(fig, ax, "HDBSCAN Cluster Composition by GT Class (Exp09)")

    bottoms = np.zeros(len(unique_clusters))
    for g in range(n_classes):
        vals = data[:, g].astype(float)
        if vals.sum() > 0:
            ax.bar(unique_clusters, vals, bottom=bottoms,
                   color=class_colors[g], label=CLASS_NAMES.get(g, f"C{g}"),
                   edgecolor="#111111", linewidth=0.3)
            bottoms += vals

    ax.set_xlabel("HDBSCAN Cluster ID", color="#AAAAAA")
    ax.set_ylabel("Number of Samples",   color="#AAAAAA")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.25,
              labelcolor="white", facecolor=DARK_BG, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] fig3 saved → {out_path}")


def plot_soft_q_heatmap(q_soft: np.ndarray, gt_class_ids: np.ndarray, out_path: str):
    """Fig 4: Mean soft-assignment Q matrix per GT class (DEC cluster purity)."""
    n_classes  = max(gt_class_ids.max() + 1, 1)
    n_clusters = q_soft.shape[1]
    avg_q      = np.zeros((n_classes, n_clusters), dtype=np.float32)
    counts     = np.zeros(n_classes, dtype=np.int32)

    for g in range(n_classes):
        mask = gt_class_ids == g
        if mask.sum() > 0:
            avg_q[g]   = q_soft[mask].mean(axis=0)
            counts[g]  = mask.sum()

    fig, ax = plt.subplots(figsize=(max(12, n_clusters * 0.8), max(6, n_classes * 0.5)))
    _dark_scatter_setup(fig, ax, "DEC Soft-Assignment Q Matrix (mean per GT class)")

    im = ax.imshow(avg_q, aspect="auto", cmap="viridis", vmin=0.0)
    plt.colorbar(im, ax=ax, label="Mean Soft Assignment q[i,k]")

    ax.set_xlabel("DEC Cluster k",      color="#AAAAAA")
    ax.set_ylabel("Ground-Truth Class", color="#AAAAAA")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(
        [f"{CLASS_NAMES.get(g, str(g))} (n={counts[g]})" for g in range(n_classes)],
        fontsize=7.5, color="#CCCCCC",
    )
    ax.set_xticks(range(0, n_clusters, max(1, n_clusters // 10)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] fig4 saved → {out_path}")


# ---------------------------------------------------------------------------
# Save predictions.h5
# ---------------------------------------------------------------------------

def save_predictions_h5(
    out_path:      str,
    results:       dict,
    cluster_ids:   np.ndarray,
    pred_class_ids: np.ndarray,
    gt_inst_ids:   np.ndarray,
    pred_inst_ids: np.ndarray,
    dataset:       DECDataset_Exp09,
):
    """
    Saves predictions in the canonical schema expected by localise.py.

    Schema (identical to Exp08 predictions.h5):
        shard_path     : bytes (N,)
        scene_idx      : int32 (N,)
        ch_idx         : int32 (N,)
        start_idx      : int32 (N,)
        gt_class_id    : int32 (N,)
        pred_class_id  : int32 (N,)
        cluster_id     : int32 (N,)
        gt_inst_id     : int32 (N,)
        pred_inst_id   : int32 (N,)
        time_res       : float32 (N,)
        pulse_idx      : int32 (N,)
    """
    n           = len(results["gt_class_ids"])
    shard_paths = results["shard_paths"]
    pulse_idxs  = results["pulse_idxs"]

    # Recover scene_idx, ch_idx, start_idx from shard labels
    scene_idxs  = np.full(n, -1, dtype=np.int32)
    ch_idxs     = np.full(n, -1, dtype=np.int32)
    start_idxs  = np.full(n, -1, dtype=np.int32)

    shard_cache: dict = {}
    for i in range(n):
        sp  = shard_paths[i]
        pid = int(pulse_idxs[i])
        if sp not in shard_cache:
            with h5py.File(sp, "r") as f:
                shard_cache[sp] = f["labels"][:].astype(np.int32)
        lbs = shard_cache[sp]
        if 0 <= pid < lbs.shape[1]:
            scene_idxs[i] = lbs[0, pid]
            ch_idxs[i]    = lbs[1, pid]
            start_idxs[i] = lbs[5, pid]

    with h5py.File(out_path, "w") as f:
        f.create_dataset("shard_path",    data=np.array(shard_paths, dtype=h5py.string_dtype()))
        f.create_dataset("scene_idx",     data=scene_idxs)
        f.create_dataset("ch_idx",        data=ch_idxs)
        f.create_dataset("start_idx",     data=start_idxs)
        f.create_dataset("gt_class_id",   data=results["gt_class_ids"])
        f.create_dataset("pred_class_id", data=pred_class_ids)
        f.create_dataset("cluster_id",    data=cluster_ids)
        f.create_dataset("gt_inst_id",    data=gt_inst_ids)
        f.create_dataset("pred_inst_id",  data=pred_inst_ids)
        f.create_dataset("time_res",      data=results["time_res_arr"])
        f.create_dataset("pulse_idx",     data=pulse_idxs)
        f.attrs["n_samples"]   = n
        f.attrs["n_clusters"]  = int(len(set(cluster_ids.tolist())) - (1 if -1 in cluster_ids else 0))
        f.attrs["experiment"]  = "exp09"

    print(f"[H5] predictions saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp09 Inference + Analytics: UMAP + HDBSCAN + 4-Domain Visualisation."
    )
    parser.add_argument("--checkpoint_id", required=True,
                        help="Node ID of the saved checkpoint (e.g. 'Xk9Z').")
    parser.add_argument("--config", default="src/models/configs/exp09_vit_dann.yaml")
    parser.add_argument("--source", action="append", dest="sources", default=[],
                        help="Source spec: 'path:type:all_shards' or 'path:type:1,2,3'")
    parser.add_argument("--min_cluster_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    # Output directory
    results_dir = os.path.abspath(config["output"].get("results_dir", "data/classification_output/exp09_dec"))
    node_id     = args.checkpoint_id
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir     = os.path.join(results_dir, f"{run_ts}_inf-{node_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Output] {out_dir}")

    # Source parsing
    print("[Sources]")
    sources = parse_sources(args.sources)
    if not sources:
        # Fallback: use config sources (inference mode)
        sources = config["data"]["sources"]
        print("[Sources] Using config data.sources as fallback.")

    # Build dataset (all shards, no label masking, no augment)
    domain_map = config["data"].get("domain_map", None)
    infer_ds = DECDataset_Exp09(
        sources        = sources,
        shard_key      = "train_shards",   # all shards treated as inference
        max_pulse_len  = config["data"].get("max_pulse_len", 4096),
        augment        = False,
        label_fraction = 1.0,              # reveal all labels for evaluation
        domain_map     = domain_map,
    )
    print(f"[Data] Inference dataset: {len(infer_ds):,} pulses")

    num_workers = config.get("training", {}).get("num_workers", 0)
    pin_memory  = (device.type == "cuda")
    infer_loader = DataLoader(infer_ds, batch_size=args.batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=pin_memory)

    # Load model
    weights_dir = os.path.abspath(config["output"]["weights_dir"])
    task = load_model_from_checkpoint(node_id, config, weights_dir, device)

    # Extract embeddings
    print("[Inference] Extracting embeddings...")
    t0      = time.time()
    results = extract_features(task, infer_loader, device, infer_ds)
    embs    = results["embs"]
    print(f"[Inference] {len(embs):,} embeddings in {time.time()-t0:.1f}s")

    # UMAP
    xy = umap_2d(embs)

    # HDBSCAN
    cluster_ids = hdbscan_cluster(xy, min_cluster_size=args.min_cluster_size)

    # Alignment
    n_classes      = max(int(results["gt_class_ids"].max()) + 1, len(CLASS_NAMES))
    pred_class_ids, c2c, conf = align_clusters_to_classes(
        cluster_ids, results["gt_class_ids"], n_classes
    )

    # Instance grouping
    print("[Grouping] Computing per-instance majority-vote prediction...")
    gt_inst_ids, pred_inst_ids = group_by_instance(
        results["pulse_idxs"], pred_class_ids, results["shard_paths"], infer_ds
    )

    # Metrics
    print("[Metrics] Computing...")
    metrics = compute_metrics(
        results["gt_class_ids"], pred_class_ids,
        pred_inst_ids, gt_inst_ids,
        embs, cluster_ids, results["q_soft"],
    )
    print("\n[Metrics]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save metrics
    metrics["checkpoint_id"] = node_id
    metrics["timestamp"]     = run_ts
    metrics["n_sources"]     = len(sources)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved → {metrics_path}")

    # Save predictions.h5
    h5_path = os.path.join(out_dir, "predictions.h5")
    save_predictions_h5(h5_path, results, cluster_ids, pred_class_ids,
                        gt_inst_ids, pred_inst_ids, infer_ds)

    # Plots
    print("[Plots] Generating...")
    plot_umap_by_class(xy, results["gt_class_ids"],
                       os.path.join(out_dir, "fig1_umap_by_class.png"))
    plot_umap_by_domain(xy, results["domain_labels"],
                        os.path.join(out_dir, "fig2_umap_by_domain.png"))
    plot_cluster_composition(cluster_ids, results["gt_class_ids"], n_classes,
                             os.path.join(out_dir, "fig3_cluster_composition.png"))
    plot_soft_q_heatmap(results["q_soft"], results["gt_class_ids"],
                        os.path.join(out_dir, "fig4_soft_q_heatmap.png"))

    # Lineage
    parent_id = config["experiment"].get("parent_node_id", "NONE")
    inf_node  = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    register_process(
        parent_id        = parent_id,
        stage            = "inference",
        method           = "dann_vit_umap_hdbscan_exp09",
        folder_path      = out_dir,
        appended_history = (
            f"Exp09 inference (checkpoint={node_id}). "
            f"cls_acc={metrics['cls_accuracy']:.3f}, "
            f"grouping_f1={metrics['grouping_f1']:.3f}, "
            f"silhouette={metrics.get('silhouette')}, "
            f"n_clusters={metrics['n_clusters']}."
        ),
        force_node_id = inf_node,
    )
    print(f"[Lineage] Inference node registered: {inf_node}")

    print(f"\n[Done] All outputs → {out_dir}")
    print(f"  fig2_umap_by_domain.png — check for domain overlap (DANN quality)")


if __name__ == "__main__":
    main()
