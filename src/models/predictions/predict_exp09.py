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
import warnings
warnings.filterwarnings("ignore", message="Tight layout not applied")
from collections import defaultdict
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
from src.utils.lineage_tracker     import register_process, get_node_history


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
    task.load_state_dict(ckpt["model_state"], strict=False)
    task.eval()
    print(f"[Checkpoint] Loaded -> {weight_path}")
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
    all_scene_idxs    = []
    all_ch_idxs       = []
    all_start_idxs    = []

    shard_labels_cache = {}

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

            for i in range(len(shard_path)):
                s_path = shard_path[i]
                p_idx  = pulse_idx[i].item()
                if s_path not in shard_labels_cache:
                    import h5py
                    with h5py.File(s_path, "r") as f:
                        shard_labels_cache[s_path] = f["labels"][:]
                labels_arr = shard_labels_cache[s_path]
                all_scene_idxs.append(int(labels_arr[0, p_idx]))
                all_ch_idxs.append(int(labels_arr[1, p_idx]))
                all_start_idxs.append(int(labels_arr[5, p_idx]))

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
        "pulse_idxs":       np.array(all_pulse_idxs,          dtype=np.int32),
        "scene_idxs":       np.array(all_scene_idxs,          dtype=np.int32),
        "ch_idxs":          np.array(all_ch_idxs,             dtype=np.int32),
        "start_idxs":       np.array(all_start_idxs,          dtype=np.int32),
        "time_res_arr":     np.array(all_time_res,            dtype=np.float32) if all_time_res else np.array([]),
    }


# ---------------------------------------------------------------------------
# UMAP Projection
# ---------------------------------------------------------------------------

def umap_nd(embs: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce embeddings to n_components using UMAP with cosine metric.

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
    print(f"[UMAP] Reducing {len(embs):,} × {embs.shape[1]}-D embeddings to {n_components}D...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",   # L2-normalised -> cosine is the natural metric
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

    present_classes = sorted(c for c in np.unique(gt_class_ids) if 0 <= c < n_classes)
    n_present = len(present_classes)
    
    # Map index to actual class id
    class_idx_to_id = {i: cls_id for i, cls_id in enumerate(present_classes)}

    cost = np.zeros((n_clusters, n_present), dtype=np.int32)
    for ci, cid in enumerate(cluster_labels):
        mask = cluster_ids == cid
        for i, cls_id in class_idx_to_id.items():
            cost[ci, i] += np.sum(gt_class_ids[mask] == cls_id)

    row_ind, col_ind = linear_sum_assignment(-cost)
    cluster_to_class = {cluster_labels[r]: class_idx_to_id[col_ind[r]] for r in row_ind}

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

def group_pd_instances(
    embs:           np.ndarray,
    pred_class_ids: np.ndarray,
    shard_paths:    list,
    scene_idxs:     np.ndarray,
    start_idxs:     np.ndarray,
    time_res_arr:   np.ndarray,
    time_th:        float = 1e-5,
    dist_th:        float = 0.5,
    mode:           str   = "greedy",
    ch_idxs:        np.ndarray = None,
) -> tuple[np.ndarray, dict]:
    """
    Groups pulses that occur closely in time (within time_th seconds)
    and in embedding space (Euclidean distance < dist_th), and computes
    majority-vote predicted classes.

    Parameters
    ----------
    mode : 'greedy' or 'channel_capped'.
        greedy        : Any unassigned pulse within both thresholds joins the instance.
        channel_capped: At most one pulse per channel (max 4 per instance). Among
                        multiple candidates on the same channel, the one with the
                        smallest L2 embedding distance to the seed is chosen.
                        Requires ch_idxs to be provided.

    Returns
    -------
    (pred_inst_ids, inst_majority_class)
    """
    n = len(embs)
    pred_inst_ids = np.full(n, -1, dtype=np.int32)

    scene_groups = defaultdict(list)
    for i in range(n):
        scene_groups[(shard_paths[i], scene_idxs[i])].append(i)

    assigned_global: dict = {}
    inst_counter = 0

    for pulses_idxs in scene_groups.values():
        for i in pulses_idxs:
            if i in assigned_global:
                continue
            inst_counter += 1
            assigned_global[i] = inst_counter
            t_i = start_idxs[i] * time_res_arr[i]

            if mode == "greedy":
                for j in pulses_idxs:
                    if j == i or j in assigned_global:
                        continue
                    t_j = start_idxs[j] * time_res_arr[j]
                    if abs(t_i - t_j) > time_th:
                        continue
                    if np.linalg.norm(embs[i] - embs[j]) < dist_th:
                        assigned_global[j] = inst_counter

            elif mode == "channel_capped" and ch_idxs is not None:
                seed_ch = int(ch_idxs[i])
                by_channel: dict = defaultdict(list)  # ch -> [(j, dist), ...]
                for j in pulses_idxs:
                    if j == i or j in assigned_global:
                        continue
                    t_j = start_idxs[j] * time_res_arr[j]
                    if abs(t_i - t_j) > time_th:
                        continue
                    d = float(np.linalg.norm(embs[i] - embs[j]))
                    if d < dist_th:
                        by_channel[int(ch_idxs[j])].append((j, d))
                for ch, candidates in by_channel.items():
                    if ch == seed_ch:
                        continue  # seed already occupies this channel slot
                    best_j = min(candidates, key=lambda x: x[1])[0]
                    assigned_global[best_j] = inst_counter

    inst_majority: dict = {}
    inst_to_preds: dict = defaultdict(list)
    for i, inst_id in assigned_global.items():
        pred_inst_ids[i] = inst_id
        if pred_class_ids[i] >= 0:
            inst_to_preds[inst_id].append(pred_class_ids[i])

    for inst_id, preds in inst_to_preds.items():
        if len(preds) > 0:
            inst_majority[inst_id] = int(np.bincount(preds).argmax())

    return pred_inst_ids, inst_majority



# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    gt_class_ids:    np.ndarray,
    pred_class_ids:  np.ndarray,
    pred_inst_ids:   np.ndarray,
    gt_inst_ids:     np.ndarray,
    embs:            np.ndarray,
    cluster_ids:     np.ndarray,
    q_soft:          np.ndarray,
    shard_paths:     list,
    scene_idxs:      np.ndarray,
    class_scores:    np.ndarray = None,
    inference_time_s: float = 0.0,
    reduce_time_s:    float = 0.0,
    hdbscan_time_s:   float = 0.0
) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, silhouette_score,
        adjusted_rand_score, normalized_mutual_info_score,
        precision_recall_fscore_support,
        davies_bouldin_score, calinski_harabasz_score,
        roc_auc_score,
    )

    valid_mask = (pred_class_ids >= 0) & (gt_class_ids >= 0)
    gt_v       = gt_class_ids[valid_mask]
    pred_v     = pred_class_ids[valid_mask]

    cls_acc         = float(accuracy_score(gt_v, pred_v)) if len(gt_v) > 0 else 0.0
    cls_f1_macro    = float(f1_score(gt_v, pred_v, average="macro",    zero_division=0)) if len(gt_v) > 0 else 0.0
    cls_f1_weighted = float(f1_score(gt_v, pred_v, average="weighted", zero_division=0)) if len(gt_v) > 0 else 0.0
    ari             = float(adjusted_rand_score(gt_class_ids, cluster_ids))
    nmi             = float(normalized_mutual_info_score(gt_class_ids, cluster_ids))

    # Silhouette on cosine-metric embeddings (UMAP-aligned)
    uc = np.unique(cluster_ids[cluster_ids >= 0])
    if len(uc) > 1:
        sil_mask = cluster_ids >= 0
        sil = float(silhouette_score(embs[sil_mask], cluster_ids[sil_mask], metric="cosine"))
    else:
        sil = float("nan")

    # Davies-Bouldin & Calinski-Harabasz (Euclidean, non-noise only)
    dbi = float("nan")
    chi = float("nan")
    if len(uc) > 1:
        sil_mask = cluster_ids >= 0
        try:
            dbi = float(davies_bouldin_score(embs[sil_mask], cluster_ids[sil_mask]))
            chi = float(calinski_harabasz_score(embs[sil_mask], cluster_ids[sil_mask]))
        except Exception:
            pass

    # Cluster purity
    purity_num, purity_den = 0, 0
    for lbl in uc:
        mask = cluster_ids == lbl
        if mask.sum() == 0:
            continue
        dominant    = int(np.bincount(gt_class_ids[mask].astype(int)).max())
        purity_num += dominant
        purity_den += int(mask.sum())
    cluster_purity = purity_num / purity_den if purity_den > 0 else 0.0

    # AUC ROC (macro, one-vs-rest)
    auc_roc_macro  = float("nan")
    unique_classes = np.unique(gt_class_ids[gt_class_ids >= 0])
    n_cls = class_scores.shape[1] if class_scores is not None else 0
    if class_scores is not None and len(unique_classes) >= 2 and n_cls >= 2:
        try:
            labels_present = [int(c) for c in unique_classes if c < n_cls]
            cs_sub = class_scores[:, labels_present].copy()
            rs = cs_sub.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            cs_sub /= rs
            if len(labels_present) == 2:
                auc_roc_macro = float(roc_auc_score(gt_class_ids == labels_present[1], cs_sub[:, 1]))
            else:
                auc_roc_macro = float(roc_auc_score(
                    gt_class_ids, cs_sub, multi_class="ovr", average="macro",
                    labels=labels_present,
                ))
        except Exception as e:
            print(f"[Warning] AUC ROC: {e}")

    # Pairwise Instance grouping F1
    tp = total_gt = total_pred = 0
    scenes: dict = defaultdict(list)
    for i in range(len(embs)):
        scenes[(shard_paths[i], scene_idxs[i])].append(i)
    for pulses in scenes.values():
        n = len(pulses)
        for i in range(n):
            for j in range(i + 1, n):
                idx_i, idx_j = pulses[i], pulses[j]
                if gt_inst_ids[idx_i] == -1 or gt_inst_ids[idx_j] == -1:
                    continue
                gs = gt_inst_ids[idx_i]   == gt_inst_ids[idx_j]
                ps = pred_inst_ids[idx_i] == pred_inst_ids[idx_j]
                if gs: total_gt   += 1
                if ps: total_pred += 1
                if gs and ps: tp  += 1

    prec     = tp / total_pred if total_pred > 0 else 1.0
    rec      = tp / total_gt   if total_gt   > 0 else 1.0
    group_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # DEC entropy
    entropy = float(-np.mean(np.sum(q_soft * np.log(q_soft + 1e-9), axis=1)))

    # Per-class precision / recall / F1
    labels_sorted = sorted(set(gt_class_ids[gt_class_ids >= 0].tolist()))
    if valid_mask.sum() > 0:
        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
            gt_v, pred_v, labels=labels_sorted, zero_division=0, average=None,
        )
    else:
        z = [0.0] * len(labels_sorted)
        prec_arr, rec_arr, f1_arr = z, z, z

    class_stats: dict = {}
    for li, cid in enumerate(labels_sorted):
        mask      = gt_class_ids == cid
        class_acc = float((gt_class_ids[mask] == pred_class_ids[mask]).mean()) if mask.sum() > 0 else 0.0
        class_stats[int(cid)] = {
            "name":      CLASS_NAMES.get(int(cid), f"Class {cid}"),
            "count":     int(mask.sum()),
            "accuracy":  round(class_acc,           4),
            "precision": round(float(prec_arr[li]), 4),
            "recall":    round(float(rec_arr[li]),  4),
            "f1":        round(float(f1_arr[li]),   4),
        }

    n_samples  = len(gt_class_ids)

    return {
        "classification_accuracy":   round(cls_acc,          4),
        "cls_f1_macro":              round(cls_f1_macro,     4),
        "cls_f1_weighted":           round(cls_f1_weighted,  4),
        "auc_roc_macro":             (round(auc_roc_macro, 4) if not np.isnan(auc_roc_macro) else None),
        "silhouette_score":          (round(sil, 4) if not np.isnan(sil) else None),
        "davies_bouldin_index":      (round(dbi, 4) if not np.isnan(dbi) else None),
        "calinski_harabasz_index":   (round(chi, 4) if not np.isnan(chi) else None),
        "ari":                       round(ari, 4),
        "nmi":                       round(nmi, 4),
        "cluster_purity":            round(cluster_purity, 4),
        "grouping_precision":        round(prec, 4),
        "grouping_recall":           round(rec, 4),
        "grouping_f1":               round(group_f1, 4),
        "n_clusters_found":          int(len(uc)),
        "n_noise_points":            int((cluster_ids == -1).sum()),
        "n_gt_instances":            len(set(zip(shard_paths, gt_inst_ids))),
        "n_pred_instances":          len(set(pred_inst_ids)),
        "n_total":                   int(n_samples),
        "time_inference_s":          round(inference_time_s, 3),
        "time_reduce_s":             round(reduce_time_s, 3),
        "time_hdbscan_s":            round(hdbscan_time_s, 3),
        "throughput_inference_smp_s": round(n_samples / inference_time_s, 2) if inference_time_s > 0 else None,
        "throughput_reduce_smp_s":    round(n_samples / reduce_time_s, 2) if reduce_time_s > 0 else None,
        "throughput_hdbscan_smp_s":   round(n_samples / hdbscan_time_s, 2) if hdbscan_time_s > 0 else None,
        "per_class_stats":           class_stats,
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
    print(f"[Plot] fig1 saved -> {out_path}")


def plot_umap_by_cluster(xy: np.ndarray, cluster_ids: np.ndarray, out_path: str):
    """Fig 1b: UMAP coloured by HDBSCAN cluster."""
    unique_clusters = np.unique(cluster_ids)
    cmap   = plt.get_cmap("tab20", max(len(unique_clusters), 1))
    colors = {c: cmap(i) for i, c in enumerate(sorted(c for c in unique_clusters if c >= 0))}
    colors[-1] = (0.5, 0.5, 0.5, 1.0)  # Noise points as grey

    fig, ax = plt.subplots(figsize=(12, 9))
    _dark_scatter_setup(fig, ax, "UMAP Projection — Coloured by HDBSCAN Cluster (Exp09)")

    for c in unique_clusters:
        mask = cluster_ids == c
        label = "Noise" if c == -1 else f"Cluster {c}"
        color = colors.get(c, "#888888")
        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, alpha=0.65 if c != -1 else 0.3,
                   color=color, rasterized=True, label=label)

    legend = ax.legend(
        loc="best", fontsize=7.5, framealpha=0.25,
        labelcolor="white", facecolor=DARK_BG, ncol=3,
    )
    ax.set_xlabel("UMAP-1", color="#AAAAAA")
    ax.set_ylabel("UMAP-2", color="#AAAAAA")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] fig1b saved -> {out_path}")


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
    print(f"[Plot] fig2 saved -> {out_path}")


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
    print(f"[Plot] fig3 saved -> {out_path}")


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
    print(f"[Plot] fig4 saved -> {out_path}")


# ---------------------------------------------------------------------------
# Class probability scores from DEC q_soft  (exp09 array-based variant)
# ---------------------------------------------------------------------------

def build_class_scores_from_dec_arr(
    q_soft: np.ndarray, gt_class_ids: np.ndarray
) -> np.ndarray:
    """
    Build a (N, n_classes) probability matrix by aligning DEC internal clusters
    to GT classes via hard-assignment majority vote, then summing q_soft values
    for all DEC clusters that share the same semantic class.

    Used as a proxy confidence score for ROC / PR / calibration curves.
    Note: confidence scores may appear over-confident when few semantic classes
    are present; curves remain statistically valid.

    Returns
    -------
    class_scores : np.ndarray, shape (N, n_classes), rows normalised to sum to 1.
    """
    n_classes = int(gt_class_ids.max()) + 1
    q_matrix  = q_soft.astype(np.float32)

    dec_votes: dict = defaultdict(list)
    for i in range(len(gt_class_ids)):
        hard_dec = int(np.argmax(q_matrix[i]))
        dec_votes[hard_dec].append(int(gt_class_ids[i]))
    dec_map = {k: max(set(v), key=v.count) for k, v in dec_votes.items()}

    class_scores = np.zeros((len(gt_class_ids), n_classes), dtype=np.float32)
    for dec_k, cls_id in dec_map.items():
        if 0 <= cls_id < n_classes and dec_k < q_matrix.shape[1]:
            class_scores[:, cls_id] += q_matrix[:, dec_k]

    row_sums = class_scores.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    class_scores /= row_sums
    return class_scores


# ---------------------------------------------------------------------------
# New plots (exp09)
# ---------------------------------------------------------------------------

def _dark_ax09(ax, title: str):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def plot_confusion_matrix_09(
    gt_class_ids: np.ndarray, pred_class_ids: np.ndarray,
    out_dir: str, exp_tag: str = "exp09"
) -> str:
    """Normalised confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    valid  = (pred_class_ids >= 0) & (gt_class_ids >= 0)
    yt, yp = gt_class_ids[valid], pred_class_ids[valid]
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    cm     = confusion_matrix(yt, yp, labels=labels, normalize="true")
    ticks  = [CLASS_NAMES.get(l, f"C{l}") for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9 + 2),
                                     max(5, len(labels) * 0.8 + 2)))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax09(ax, f"Normalised Confusion Matrix ({exp_tag})")
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Proportion of true class")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=8, color="#CCCCCC")
    ax.set_yticklabels(ticks, fontsize=8, color="#CCCCCC")
    ax.set_xlabel("Predicted", color="#AAAAAA")
    ax.set_ylabel("Ground Truth", color="#AAAAAA")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] < 0.6 else "black")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_confusion_matrix.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_pr_curves_09(
    gt_class_ids: np.ndarray, class_scores: np.ndarray,
    out_dir: str, exp_tag: str = "exp09"
) -> str:
    """Per-class Precision-Recall curves using DEC q_soft as a proxy confidence score."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    unique_classes = sorted(set(gt_class_ids[gt_class_ids >= 0].tolist()))
    n_cls = class_scores.shape[1]
    cmap  = plt.get_cmap("tab20", max(len(unique_classes), 1))

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax09(ax, (f"Precision-Recall Curves ({exp_tag})\n"
                    "[Proxy confidence: DEC q_soft remapped via hard-assignment alignment]"))
    for i, cls_id in enumerate(unique_classes):
        if cls_id >= n_cls:
            continue
        y_bin  = (gt_class_ids == cls_id).astype(int)
        scores = class_scores[:, cls_id]
        try:
            prec, rec, _ = precision_recall_curve(y_bin, scores)
            ap = average_precision_score(y_bin, scores)
            ax.plot(rec, prec, color=cmap(i), lw=1.5,
                    label=f"{CLASS_NAMES.get(cls_id, f'C{cls_id}')} (AP={ap:.3f})")
        except Exception as e:
            print(f"[Warning] PR curve class {cls_id}: {e}")
    ax.set_xlabel("Recall", color="#AAAAAA")
    ax.set_ylabel("Precision", color="#AAAAAA")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white",
              facecolor=DARK_BG, loc="lower left")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_pr_curve.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_roc_curves_09(
    gt_class_ids: np.ndarray, class_scores: np.ndarray,
    out_dir: str, exp_tag: str = "exp09"
) -> str:
    """Per-class ROC curves using DEC q_soft as a proxy confidence score."""
    from sklearn.metrics import roc_curve, roc_auc_score
    unique_classes = sorted(set(gt_class_ids[gt_class_ids >= 0].tolist()))
    n_cls = class_scores.shape[1]
    cmap  = plt.get_cmap("tab20", max(len(unique_classes), 1))

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax09(ax, (f"ROC Curves ({exp_tag})\n"
                    "[Proxy confidence: DEC q_soft remapped via hard-assignment alignment]"))
    ax.plot([0, 1], [0, 1], "--", color="#555555", lw=1.5, label="Random (AUC=0.5)")
    for i, cls_id in enumerate(unique_classes):
        if cls_id >= n_cls:
            continue
        y_bin  = (gt_class_ids == cls_id).astype(int)
        scores = class_scores[:, cls_id]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        try:
            fpr, tpr, _ = roc_curve(y_bin, scores)
            auc = roc_auc_score(y_bin, scores)
            ax.plot(fpr, tpr, color=cmap(i), lw=1.5,
                    label=f"{CLASS_NAMES.get(cls_id, f'C{cls_id}')} (AUC={auc:.3f})")
        except Exception as e:
            print(f"[Warning] ROC curve class {cls_id}: {e}")
    ax.set_xlabel("False Positive Rate", color="#AAAAAA")
    ax.set_ylabel("True Positive Rate (Recall)", color="#AAAAAA")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white",
              facecolor=DARK_BG, loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_roc_curve.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_calibration_curve_09(
    gt_class_ids: np.ndarray, pred_class_ids: np.ndarray,
    class_scores: np.ndarray, out_dir: str, exp_tag: str = "exp09"
) -> str:
    """Reliability diagram: model confidence vs observed accuracy."""
    from sklearn.calibration import calibration_curve as sk_cal_curve
    valid  = (pred_class_ids >= 0) & (gt_class_ids >= 0)
    yt, yp = gt_class_ids[valid], pred_class_ids[valid]
    cs     = class_scores[valid]
    confidence = cs.max(axis=1)
    correct    = (yt == yp).astype(int)
    try:
        prob_true, prob_pred = sk_cal_curve(correct, confidence, n_bins=10,
                                             strategy="uniform")
    except Exception as e:
        print(f"[Warning] Calibration curve failed: {e}")
        return ""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax09(ax, (f"Confidence / Calibration Curve ({exp_tag})\n"
                    "[Proxy confidence: max DEC q_soft class score]"))
    ax.plot([0, 1], [0, 1], "--", color="#555555", lw=1.5, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color="#4CC9F0", lw=2, ms=6, label="Model")
    ax.set_xlabel("Mean Predicted Confidence", color="#AAAAAA")
    ax.set_ylabel("Fraction Correct (Accuracy)", color="#AAAAAA")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=9, framealpha=0.3, labelcolor="white", facecolor=DARK_BG)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_calibration_curve.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_time_vs_distance_09(
    embs: np.ndarray, gt_inst_ids: np.ndarray, shard_paths: list,
    scene_idxs: np.ndarray, start_idxs: np.ndarray, time_res_arr: np.ndarray,
    out_dir: str, exp_tag: str = "exp09", max_pairs: int = 10_000
) -> str:
    """
    Scatter of intra-scene pulse pairs: time difference (X) vs L2 embedding
    distance (Y). Green = same GT instance; Red = different.
    """
    n      = len(embs)
    scenes: dict = defaultdict(list)
    for i in range(n):
        scenes[(shard_paths[i], int(scene_idxs[i]))].append(i)

    rng       = np.random.default_rng(42)
    all_pairs: list = []
    for idxs in scenes.values():
        m = len(idxs)
        for ii in range(m):
            for jj in range(ii + 1, m):
                i, j = idxs[ii], idxs[jj]
                if gt_inst_ids[i] == -1 or gt_inst_ids[j] == -1:
                    continue
                td   = abs(float(start_idxs[i]) * float(time_res_arr[i]) -
                           float(start_idxs[j]) * float(time_res_arr[j]))
                ed   = float(np.linalg.norm(embs[i] - embs[j]))
                same = bool(gt_inst_ids[i] == gt_inst_ids[j])
                all_pairs.append((td, ed, same))

    if len(all_pairs) > max_pairs:
        idx       = rng.choice(len(all_pairs), max_pairs, replace=False)
        all_pairs = [all_pairs[k] for k in idx]

    same_td, same_ed, diff_td, diff_ed = [], [], [], []
    for td, ed, same in all_pairs:
        if same:
            same_td.append(td); same_ed.append(ed)
        else:
            diff_td.append(td); diff_ed.append(ed)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax09(ax, (f"Time Difference vs Embedding Distance ({exp_tag})\n"
                    "Green = Same GT Instance  |  Red = Different GT Instance"))
    if diff_td:
        ax.scatter(diff_td, diff_ed, s=4, alpha=0.25, color="#E63946", rasterized=True,
                   label=f"Different instance (n={len(diff_td):,})")
    if same_td:
        ax.scatter(same_td, same_ed, s=6, alpha=0.6, color="#06D6A0", rasterized=True,
                   label=f"Same instance (n={len(same_td):,})")
    ax.set_xlabel("Time Difference (s)", color="#AAAAAA")
    ax.set_ylabel("Embedding L2 Distance", color="#AAAAAA")
    ax.legend(fontsize=9, framealpha=0.3, labelcolor="white", facecolor=DARK_BG)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_time_vs_distance.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


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

    print(f"[H5] predictions saved -> {out_path}")


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
    parser.add_argument("--reduce_method", type=str, default="umap", choices=["none", "umap"],
                        help="Method to reduce embeddings before HDBSCAN.")
    parser.add_argument("--reduce_dims", type=int, default=2,
                        help="Number of dimensions for dimensionality reduction (if method is umap).")
    parser.add_argument("--time_threshold", type=float, default=1e-5,
                        help="Max time diff (s) to group pulses from different sensors.")
    parser.add_argument("--dist_threshold", type=float, default=0.5,
                        help="Max embedding distance to group pulses from different sensors.")
    parser.add_argument("--grouping_mode", type=str, default="greedy",
                        choices=["greedy", "channel_capped"],
                        help=("greedy: any pulse within thresholds joins the instance. "
                              "channel_capped: at most one pulse per channel (max 4)."))
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
    inf_id      = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    out_dir     = os.path.join(results_dir, f"{run_ts}_inf-{node_id}-{inf_id}")
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
    t_infer_start = time.time()
    results       = extract_features(task, infer_loader, device, infer_ds)
    inference_time_s = time.time() - t_infer_start
    embs    = results["embs"]
    print(f"[Inference] {len(embs):,} embeddings in {inference_time_s:.1f}s  "
          f"({len(embs) / inference_time_s:.0f} samples/s)")

    # Dimensionality reduction for clustering
    t_reduce = time.time()
    if args.reduce_method == "umap":
        if args.reduce_dims == 2:
            xy_cluster = xy_plot = umap_nd(embs, n_components=2)
        else:
            xy_cluster = umap_nd(embs, n_components=args.reduce_dims)
            xy_plot    = umap_nd(embs, n_components=2)
    else:
        xy_cluster = embs
        xy_plot    = umap_nd(embs, n_components=2)
    reduce_time_s = time.time() - t_reduce
    print(f"[Timing] Reduction: {reduce_time_s:.2f}s  ({len(embs) / reduce_time_s if reduce_time_s > 0 else 0:.1f} samples/s)")

    # HDBSCAN
    t_cluster = time.time()
    cluster_ids = hdbscan_cluster(xy_cluster, min_cluster_size=args.min_cluster_size)
    hdbscan_time_s = time.time() - t_cluster
    print(f"[Timing] HDBSCAN:   {hdbscan_time_s:.2f}s  ({len(embs) / hdbscan_time_s if hdbscan_time_s > 0 else 0:.1f} samples/s)")

    # Alignment
    n_classes      = max(int(results["gt_class_ids"].max()) + 1, len(CLASS_NAMES))
    pred_class_ids, c2c, conf = align_clusters_to_classes(
        cluster_ids, results["gt_class_ids"], n_classes
    )

    # Instance Grouping
    print("[Grouping] Computing per-instance majority-vote prediction with heuristics...")
    gt_inst_ids = np.array([entry[4] for entry in infer_ds.index], dtype=np.int32)
    # Build class probability scores from DEC q_soft
    class_scores = build_class_scores_from_dec_arr(
        results["q_soft"], results["gt_class_ids"]
    )

    pred_inst_ids, inst_majority = group_pd_instances(
        embs           = results["embs"],
        pred_class_ids = pred_class_ids,
        shard_paths    = results["shard_paths"],
        scene_idxs     = results["scene_idxs"],
        start_idxs     = results["start_idxs"],
        time_res_arr   = results["time_res_arr"],
        time_th        = args.time_threshold,
        dist_th        = args.dist_threshold,
        mode           = args.grouping_mode,
        ch_idxs        = results.get("ch_idxs"),
    )
    
    # Overwrite pulse-level predictions with their instance-level majority vote
    for i in range(len(pred_class_ids)):
        if pred_inst_ids[i] in inst_majority:
            pred_class_ids[i] = inst_majority[pred_inst_ids[i]]

    # Metrics
    print("[Metrics] Computing...")
    metrics = compute_metrics(
        gt_class_ids     = results["gt_class_ids"],
        pred_class_ids   = pred_class_ids,
        pred_inst_ids    = pred_inst_ids,
        gt_inst_ids      = gt_inst_ids,
        embs             = results["embs"],
        cluster_ids      = cluster_ids,
        q_soft           = results["q_soft"],
        shard_paths      = results["shard_paths"],
        scene_idxs       = results["scene_idxs"],
        class_scores     = class_scores,
        inference_time_s = inference_time_s,
        reduce_time_s    = reduce_time_s,
        hdbscan_time_s   = hdbscan_time_s,
    )
    
    metrics.update({
        "checkpoint_id":  args.checkpoint_id,
        "reduce_method":  args.reduce_method,
        "reduce_dims":    args.reduce_dims,
        "time_threshold": args.time_threshold,
        "dist_threshold": args.dist_threshold,
        "grouping_mode":  args.grouping_mode,
    })

    print("\n[Metrics]")
    for k, v in metrics.items():
        if k != "per_class_stats":
            print(f"  {k}: {v}")

    print("\n[Per-Class Stats]")
    for cid, stat in metrics["per_class_stats"].items():
        print(f"  {stat['name']:30s}  n={stat['count']:5d}  "
              f"acc={stat['accuracy']:.4f}  prec={stat.get('precision', 0):.4f}  "
              f"rec={stat.get('recall', 0):.4f}  f1={stat.get('f1', 0):.4f}")

    # Save metrics
    metrics["checkpoint_id"] = node_id
    metrics["timestamp"]     = run_ts
    metrics["n_sources"]     = len(sources)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved -> {metrics_path}")

    # Save predictions.h5
    h5_path = os.path.join(out_dir, "predictions.h5")
    save_predictions_h5(h5_path, results, cluster_ids, pred_class_ids,
                        gt_inst_ids, pred_inst_ids, infer_ds)

    # Original plots
    print("[Plots] Generating...")
    plot_umap_by_class(xy_plot, results["gt_class_ids"],
                       os.path.join(out_dir, "fig1_umap_by_class.png"))
    plot_umap_by_cluster(xy_plot, cluster_ids,
                         os.path.join(out_dir, "fig1b_umap_by_cluster.png"))
    plot_umap_by_domain(xy_plot, results["domain_labels"],
                        os.path.join(out_dir, "fig2_umap_by_domain.png"))
    plot_cluster_composition(cluster_ids, results["gt_class_ids"], n_classes,
                             os.path.join(out_dir, "fig3_cluster_composition.png"))
    plot_soft_q_heatmap(results["q_soft"], results["gt_class_ids"],
                        os.path.join(out_dir, "fig4_soft_q_heatmap.png"))

    # New plots
    plot_confusion_matrix_09(results["gt_class_ids"], pred_class_ids,
                             out_dir, exp_tag="exp09")
    plot_pr_curves_09(results["gt_class_ids"], class_scores,
                      out_dir, exp_tag="exp09")
    plot_roc_curves_09(results["gt_class_ids"], class_scores,
                       out_dir, exp_tag="exp09")
    plot_calibration_curve_09(results["gt_class_ids"], pred_class_ids,
                              class_scores, out_dir, exp_tag="exp09")
    plot_time_vs_distance_09(
        embs=results["embs"], gt_inst_ids=gt_inst_ids,
        shard_paths=results["shard_paths"], scene_idxs=results["scene_idxs"],
        start_idxs=results["start_idxs"], time_res_arr=results["time_res_arr"],
        out_dir=out_dir, exp_tag="exp09",
    )

    # Lineage
    register_process(
        parent_id        = node_id,
        stage            = "inference",
        method           = "dann_vit_umap_hdbscan_exp09",
        folder_path      = out_dir,
        appended_history = (
            f"Exp09 inference (checkpoint={node_id}). "
            f"cls_acc={metrics['classification_accuracy']:.3f}, "
            f"cls_f1_macro={metrics['cls_f1_macro']:.3f}, "
            f"grouping_f1={metrics['grouping_f1']:.3f}, "
            f"auc_roc={metrics.get('auc_roc_macro')}, "
            f"silhouette={metrics.get('silhouette_score')}, "
            f"grouping_mode={args.grouping_mode}, "
            f"n_clusters={metrics.get('n_clusters_found')}."
        ),
        force_node_id = inf_id,
    )
    print(f"[Lineage] Inference node registered: {inf_id}")

    with open(os.path.join(out_dir, "analysis_history.txt"), "w") as f:
        f.write(get_node_history(inf_id))

    print(f"\n[Done] All outputs -> {out_dir}")
    print(f"  fig2_umap_by_domain.png - check for domain overlap (DANN quality)")


if __name__ == "__main__":
    main()
