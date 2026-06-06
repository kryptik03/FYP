"""
predict_exp07.py
================
Inference + Visualization script for exp07 SupCon & Semi-Supervised DEC.

Usage:
    python src/models/predict/predict_exp07.py \\
        --checkpoint_id <NODE_ID> \\
        --source "data/features/stft_magnitude/20260524_234212-cw-TuKT-vraW:cwru:14,15,16" \\
        --source "data/features/stft_magnitude/20260524_234301-sy-0s6o-UI2r:equation:17,18,19,20" \\
        [--min_cluster_size 5] \\
        [--time_threshold 100e-9] \\
        [--dist_threshold 0.5] \\
        [--grouping_mode {greedy,channel_capped}]

Each --source is formatted as: path:type:shard1,shard2,...

Key differences from exp04/exp05:
  - Sources point to data/features/stft_magnitude/ (2D STFT features), NOT raw data.
  - No wavelet preprocessing step (already extracted at feature stage).
  - Ground-truth class comes from index 6 (actual_class_id), NOT index 0.
    The reported_class (index 1) may be -1 for unlabelled samples — we never
    use it for metrics.

New in this version:
  - channel_capped grouping mode: at most one pulse per channel (4 max) per instance.
  - Extended metrics: cls_f1_macro/weighted, ARI, NMI, Davies-Bouldin,
    Calinski-Harabasz, cluster purity, AUC ROC (macro), per-class
    precision/recall/F1, inference time & throughput.
  - New plots: confusion matrix, PR curves, ROC curves, calibration curve,
    time vs embedding distance scatter.
"""

import argparse
import json
import os
import random
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.tasks.task_exp07_dec   import SupConDECTask
from src.models.data.dataset_exp07_dec import DECDataset_Exp07
from src.utils.lineage_tracker         import register_process, get_node_history

# ---------------------------------------------------------------------------
# Class name mapping  (actual class_id -> human-readable label)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0:  "PD1 Void Simulated",
    1:  "PD2 Incision Simulated",
    2:  "PD1 Void Measured",
    3:  "PD2 Incision Measured",
    4:  "PD3 Delamination Measured",
    5:  "PD4 FeOx Measured",
    6:  "PD5 FeOx High Measured",
    7:  "SEDO",
    8:  "DED",
    9:  "DEDO",
    10: "SMG",
    11: "CWRU Normal",
    12: "CWRU B007",
    13: "CWRU B014",
    14: "CWRU B021",
    15: "CWRU IR007",
    16: "CWRU IR014",
    17: "CWRU IR021",
    18: "CWRU OR007",
    19: "CWRU OR014",
    20: "CWRU OR021",
}

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A0572", "#118AB2", "#06D6A0", "#FFD166", "#EF476F",
    "#8338EC", "#3A86FF", "#FB5607", "#FFBE0B", "#8AC926",
]

DARK_BG  = "#0F0F0F"
PANEL_BG = "#1A1A2E"


# ---------------------------------------------------------------------------
# Config / Weights / Device
# ---------------------------------------------------------------------------

def load_config(checkpoint_id: str) -> dict:
    import yaml
    p = os.path.abspath(f"models/configuration_snapshots/config_{checkpoint_id}.yaml")
    if not os.path.exists(p):
        print(f"[Error] Config not found: {p}"); sys.exit(1)
    with open(p) as f:
        return yaml.safe_load(f)


def load_weights(checkpoint_id: str, task: SupConDECTask):
    p = os.path.abspath(f"models/weights/model_{checkpoint_id}.pt")
    if not os.path.exists(p):
        print(f"[Error] Weights not found: {p}"); sys.exit(1)
    ckpt = torch.load(p, map_location="cpu")
    task.load_state_dict(ckpt["model_state"])
    print(f"[Checkpoint] Loaded {p}  (epoch {ckpt.get('epoch', '?')})")


def select_device(cfg: str) -> torch.device:
    if cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg)


def parse_sources(raw_list: list[str]) -> list[dict]:
    """Parse '--source path:type:shard1,shard2,...' arguments."""
    sources = []
    for raw in raw_list:
        tokens = raw.strip().split(":")
        if len(tokens) < 3:
            print(f"[Error] Bad source format: '{raw}'. Expected path:type:shards")
            sys.exit(1)
        path   = tokens[0]
        stype  = tokens[1]
        shards = [int(x) for x in tokens[2].split(",") if x.strip()]
        sources.append({
            "type":         stype,
            "path":         path,
            "train_shards": shards,
            "val_shards":   shards,
        })
    return sources


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(task: SupConDECTask, loader: DataLoader,
                     dataset: DECDataset_Exp07, device: torch.device) -> list[dict]:
    """
    Run inference over the dataset and collect per-pulse embeddings + soft assignments.

    Batch layout from DECDataset_Exp07 (augment=False):
        [0] signal          (B, 1, F, T)
        [1] reported_class  int  (may be -1 for unlabelled)
        [2] inst_id         int
        [3] shard_path      str
        [4] start_idx       int
        [5] time_res        float
        [6] actual_class    int  <- ground-truth label, always set
    """
    task.eval()
    records    = []
    global_idx = 0
    with torch.no_grad():
        for batch in loader:
            sig = batch[0].to(device)
            z, q = task(sig)
            z = z.cpu().numpy()
            q = q.cpu().numpy()
            for b in range(sig.shape[0]):
                if global_idx >= len(dataset.index):
                    break
                (shard_path, scene_idx, ch_idx, start_idx, end_idx,
                 reported_class, actual_class, inst_id, time_res, hop_length) = dataset.index[global_idx]

                records.append({
                    "shard_path":   shard_path,
                    "scene_idx":    scene_idx,
                    "ch_idx":       ch_idx,
                    "start_idx":    start_idx,
                    "gt_class_id":  actual_class,
                    "gt_inst_id":   inst_id,
                    "time_res":     time_res,
                    "emb":          z[b],
                    "soft_q":       q[b],
                })
                global_idx += 1
    print(f"[Inference] Extracted {len(records)} pulse embeddings.")
    return records


# ---------------------------------------------------------------------------
# HDBSCAN Clustering
# ---------------------------------------------------------------------------

def run_hdbscan(features: list[dict], min_cluster_size: int = 5,
                reduce_method: str = "tsne", reduce_dims: int = 2,
                epsilon: float = 0.0) -> tuple[np.ndarray, float, float, np.ndarray]:
    try:
        import hdbscan
    except ImportError:
        print("[Error] hdbscan not installed. Run: pip install hdbscan"); sys.exit(1)

    embs = np.array([f["emb"] for f in features])

    import time
    t_reduce = time.time()
    if reduce_method == "pca" and reduce_dims > 0 and embs.shape[1] > reduce_dims:
        from sklearn.decomposition import PCA
        print(f"[PCA] Reducing {embs.shape[1]} -> {reduce_dims}D before HDBSCAN...")
        embs = PCA(n_components=reduce_dims, random_state=42).fit_transform(embs)
    elif reduce_method == "tsne" and reduce_dims > 0 and embs.shape[1] > reduce_dims:
        from sklearn.manifold import TSNE as _TSNE
        if reduce_dims > 3:
            reduce_dims = 3
        print(f"[t-SNE] Reducing {embs.shape[1]} -> {reduce_dims}D before HDBSCAN...")
        perp = min(30, max(5, len(embs) // 10))
        embs = _TSNE(n_components=reduce_dims, perplexity=perp, random_state=42,
                     max_iter=1000).fit_transform(embs)
    reduce_time_s = time.time() - t_reduce

    t_hdbscan = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  cluster_selection_epsilon=epsilon, metric="euclidean")
    labels = clusterer.fit_predict(embs)
    hdbscan_time_s = time.time() - t_hdbscan

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[HDBSCAN] Clusters found: {n_clusters}  |  Noise points: {(labels == -1).sum()}")
    return labels, reduce_time_s, hdbscan_time_s, embs


# ---------------------------------------------------------------------------
# Post-hoc majority-vote cluster->class alignment
# ---------------------------------------------------------------------------

def align_clusters(features: list[dict], hdb_labels: np.ndarray) -> dict:
    votes = defaultdict(list)
    for feat, lbl in zip(features, hdb_labels):
        if lbl != -1:
            votes[lbl].append(feat["gt_class_id"])
    mapping = {k: max(set(v), key=v.count) for k, v in votes.items()}
    mapping[-1] = -1
    return mapping


# ---------------------------------------------------------------------------
# Class probability scores from DEC soft assignments
# ---------------------------------------------------------------------------

def build_class_scores_from_dec(results: list[dict]) -> tuple[np.ndarray, int]:
    """
    Build a (N, n_classes) probability matrix from DEC q_soft soft-assignments.

    Since q_soft addresses DEC internal clusters (not semantic classes), we
    perform a hard-assignment majority-vote alignment of DEC cluster indices to
    GT classes, then sum q_soft values for all DEC clusters that map to the same
    semantic class.

    This is used as a proxy confidence score for ROC / PR / calibration curves.
    Note: when many DEC clusters collapse to a small number of semantic classes
    (e.g. exp07 with 2 classes), individual class scores will be high;
    the resulting curves remain valid but may appear over-confident.

    Returns
    -------
    class_scores : np.ndarray of shape (N, n_classes), rows normalised to sum to 1.
    n_classes    : int
    """
    y_true   = np.array([r["gt_class_id"] for r in results])
    q_matrix = np.array([r["soft_q"]      for r in results], dtype=np.float32)
    n_classes = int(y_true.max()) + 1

    dec_votes: dict = defaultdict(list)
    for r in results:
        hard_dec = int(np.argmax(r["soft_q"]))
        dec_votes[hard_dec].append(r["gt_class_id"])
    dec_map = {k: max(set(v), key=v.count) for k, v in dec_votes.items()}

    class_scores = np.zeros((len(results), n_classes), dtype=np.float32)
    for dec_k, cls_id in dec_map.items():
        if 0 <= cls_id < n_classes and dec_k < q_matrix.shape[1]:
            class_scores[:, cls_id] += q_matrix[:, dec_k]

    row_sums = class_scores.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    class_scores /= row_sums
    return class_scores, n_classes


# ---------------------------------------------------------------------------
# PD Instance Grouping
# ---------------------------------------------------------------------------

def group_pd_instances(features: list[dict], hdb_labels: np.ndarray,
                       time_th: float, dist_th: float,
                       mode: str = "greedy") -> list[dict]:
    """
    Group pulses within the same scene into PD instances.

    Parameters
    ----------
    features   : per-pulse record dicts (must include 'ch_idx' for channel_capped).
    hdb_labels : HDBSCAN cluster assignments.
    time_th    : max time difference (s) for two pulses to be in the same instance.
    dist_th    : max L2 embedding distance for two pulses to be in the same instance.
    mode       : 'greedy' or 'channel_capped'.

    Modes
    -----
    greedy (default):
        Any unassigned pulse within time_th AND dist_th of the seed joins the
        instance. No cap on group size.

    channel_capped:
        At most one pulse per channel (max 4 pulses per instance total).
        The seed occupies its own channel slot. Among multiple candidates on
        the same channel, the one with the smallest L2 distance to the seed
        is chosen as the tiebreaker. Candidates that do not pass both thresholds
        are still excluded regardless of channel.
    """
    scene_groups: dict = defaultdict(list)
    for i, feat in enumerate(features):
        scene_groups[(feat["shard_path"], feat["scene_idx"])].append((i, feat))

    assigned_global: dict = {}
    inst_counter = 0

    for pulses in scene_groups.values():
        for idx, feat in pulses:
            if idx in assigned_global:
                continue
            inst_counter += 1
            assigned_global[idx] = inst_counter
            t_i = feat["start_idx"] * feat["time_res"]

            if mode == "greedy":
                for jdx, feat_j in pulses:
                    if jdx == idx or jdx in assigned_global:
                        continue
                    t_j = feat_j["start_idx"] * feat_j["time_res"]
                    if abs(t_i - t_j) > time_th:
                        continue
                    if np.linalg.norm(feat["emb"] - feat_j["emb"]) < dist_th:
                        assigned_global[jdx] = inst_counter

            elif mode == "channel_capped":
                seed_ch = feat["ch_idx"]
                by_channel: dict = defaultdict(list)
                for jdx, feat_j in pulses:
                    if jdx == idx or jdx in assigned_global:
                        continue
                    t_j = feat_j["start_idx"] * feat_j["time_res"]
                    if abs(t_i - t_j) > time_th:
                        continue
                    d = float(np.linalg.norm(feat["emb"] - feat_j["emb"]))
                    if d < dist_th:
                        by_channel[feat_j["ch_idx"]].append((jdx, d))
                for ch, candidates in by_channel.items():
                    if ch == seed_ch:
                        continue
                    best_jdx = min(candidates, key=lambda x: x[1])[0]
                    assigned_global[best_jdx] = inst_counter

    results = []
    for i, feat in enumerate(features):
        results.append({**feat,
                        "hdb_cluster":  int(hdb_labels[i]),
                        "pred_inst_id": assigned_global.get(i, -1)})
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict], hdb_labels: np.ndarray,
                    cluster_map: dict, class_scores: np.ndarray,
                    inference_time_s: float = 0.0,
                    reduce_time_s:    float = 0.0,
                    hdbscan_time_s:   float = 0.0) -> dict:
    from sklearn.metrics import (
        f1_score, precision_recall_fscore_support,
        adjusted_rand_score, normalized_mutual_info_score,
        davies_bouldin_score, calinski_harabasz_score,
        roc_auc_score,
    )

    embs   = np.array([r["emb"]         for r in results])
    y_true = np.array([r["gt_class_id"]  for r in results])
    y_pred = np.array([cluster_map.get(int(hdb_labels[i]), -1)
                       for i in range(len(results))])

    valid  = y_pred != -1
    yt_v, yp_v = y_true[valid], y_pred[valid]

    acc             = float((yt_v == yp_v).mean()) if valid.sum() > 0 else 0.0
    cls_f1_macro    = float(f1_score(yt_v, yp_v, average="macro",    zero_division=0)) if valid.sum() > 0 else 0.0
    cls_f1_weighted = float(f1_score(yt_v, yp_v, average="weighted", zero_division=0)) if valid.sum() > 0 else 0.0

    ari = float(adjusted_rand_score(y_true, hdb_labels))
    nmi = float(normalized_mutual_info_score(y_true, hdb_labels))

    unique_lbls = set(hdb_labels) - {-1}
    n_clusters  = len(unique_lbls)

    sil = float("nan")
    dbi = float("nan")
    chi = float("nan")
    if n_clusters > 1 and valid.sum() > 1:
        try:
            sil = float(silhouette_score(embs[valid], hdb_labels[valid]))
        except Exception:
            pass
        try:
            dbi = float(davies_bouldin_score(embs[valid], hdb_labels[valid]))
            chi = float(calinski_harabasz_score(embs[valid], hdb_labels[valid]))
        except Exception:
            pass

    purity_num, purity_den = 0, 0
    for lbl in unique_lbls:
        mask = hdb_labels == lbl
        if mask.sum() == 0:
            continue
        dominant    = int(np.bincount(y_true[mask].astype(int)).max())
        purity_num += dominant
        purity_den += int(mask.sum())
    cluster_purity = purity_num / purity_den if purity_den > 0 else 0.0

    auc_roc_macro  = float("nan")
    unique_classes = np.unique(y_true)
    n_cls = class_scores.shape[1] if class_scores is not None else 0
    if class_scores is not None and len(unique_classes) >= 2 and n_cls >= 2:
        try:
            labels_present = [int(c) for c in unique_classes if c < n_cls]
            cs_sub = class_scores[:, labels_present].copy()
            rs = cs_sub.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            cs_sub /= rs
            if len(labels_present) == 2:
                auc_roc_macro = float(roc_auc_score(y_true, cs_sub[:, 1]))
            else:
                auc_roc_macro = float(roc_auc_score(
                    y_true, cs_sub, multi_class="ovr", average="macro",
                    labels=labels_present,
                ))
        except Exception as e:
            print(f"[Warning] AUC ROC: {e}")

    tp = total_gt = total_pred = 0
    scenes: dict = defaultdict(list)
    for r in results:
        scenes[(r["shard_path"], r["scene_idx"])].append(r)
    for pulses in scenes.values():
        n = len(pulses)
        for i in range(n):
            for j in range(i + 1, n):
                gs = pulses[i]["gt_inst_id"]  == pulses[j]["gt_inst_id"]
                ps = pulses[i]["pred_inst_id"] == pulses[j]["pred_inst_id"]
                if gs: total_gt   += 1
                if ps: total_pred += 1
                if gs and ps: tp  += 1
    g_prec = tp / total_pred if total_pred > 0 else 1.0
    g_rec  = tp / total_gt   if total_gt   > 0 else 1.0
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0

    labels_sorted = sorted(set(y_true.tolist()))
    if valid.sum() > 0:
        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
            yt_v, yp_v, labels=labels_sorted, zero_division=0, average=None,
        )
    else:
        z = [0.0] * len(labels_sorted)
        prec_arr, rec_arr, f1_arr = z, z, z

    class_stats: dict = {}
    for li, cid in enumerate(labels_sorted):
        mask      = y_true == cid
        class_acc = float((y_true[mask] == y_pred[mask]).mean()) if mask.sum() > 0 else 0.0
        class_stats[int(cid)] = {
            "name":      CLASS_NAMES.get(int(cid), f"Class {cid}"),
            "count":     int(mask.sum()),
            "accuracy":  round(class_acc,           4),
            "precision": round(float(prec_arr[li]), 4),
            "recall":    round(float(rec_arr[li]),  4),
            "f1":        round(float(f1_arr[li]),   4),
        }

    n_samples  = len(results)

    return {
        "classification_accuracy":  round(acc,             4),
        "cls_f1_macro":             round(cls_f1_macro,    4),
        "cls_f1_weighted":          round(cls_f1_weighted,  4),
        "auc_roc_macro":            (round(auc_roc_macro, 4) if not np.isnan(auc_roc_macro) else None),
        "silhouette_score":         (round(sil, 4) if not np.isnan(sil) else None),
        "davies_bouldin_index":     (round(dbi, 4) if not np.isnan(dbi) else None),
        "calinski_harabasz_index":  (round(chi, 4) if not np.isnan(chi) else None),
        "ari":                      round(ari, 4),
        "nmi":                      round(nmi, 4),
        "cluster_purity":           round(cluster_purity, 4),
        "grouping_precision":       round(g_prec, 4),
        "grouping_recall":          round(g_rec,  4),
        "grouping_f1":              round(g_f1,   4),
        "n_pulses":                 n_samples,
        "n_clusters_found":         int(n_clusters),
        "n_noise_points":           int((hdb_labels == -1).sum()),
        "n_gt_instances":           len(set((r["shard_path"], r["gt_inst_id"]) for r in results)),
        "n_pred_instances":         len(set(r["pred_inst_id"] for r in results)),
        "time_inference_s":         round(inference_time_s, 3),
        "time_reduce_s":            round(reduce_time_s, 3),
        "time_hdbscan_s":           round(hdbscan_time_s, 3),
        "throughput_inference_smp_s": round(n_samples / inference_time_s, 2) if inference_time_s > 0 else None,
        "throughput_reduce_smp_s":    round(n_samples / reduce_time_s, 2) if reduce_time_s > 0 else None,
        "throughput_hdbscan_smp_s":   round(n_samples / hdbscan_time_s, 2) if hdbscan_time_s > 0 else None,
        "per_class_stats":          class_stats,
    }


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _dark_ax(ax, title: str):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def _tsne_2d(embs: np.ndarray) -> np.ndarray:
    print("[t-SNE] Reducing to 2D (this may take a moment)...")
    perp = min(30, max(5, len(embs) // 10))
    return TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000).fit_transform(embs)


# ---------------------------------------------------------------------------
# Original plots (structure unchanged)
# ---------------------------------------------------------------------------

def plot_embeddings(results: list[dict], hdb_labels: np.ndarray,
                    cluster_map: dict, out_dir: str, embs_reduced: np.ndarray = None) -> str:
    if embs_reduced is not None and embs_reduced.shape[1] == 2:
        xy = embs_reduced
    else:
        embs = np.array([r["emb"] for r in results])
        xy   = _tsne_2d(embs)
    gt_classes = np.array([r["gt_class_id"]  for r in results])
    hdb_arr    = np.array([r["hdb_cluster"]  for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    titles     = ["Ground-Truth Class Labels", "HDBSCAN Discovered Clusters"]
    label_arrs = [gt_classes, hdb_arr]

    for ax, arr, title in zip(axes, label_arrs, titles):
        ax.set_facecolor(PANEL_BG)
        for k, uid in enumerate(sorted(set(arr))):
            mask  = arr == uid
            color = "#888888" if uid == -1 else PALETTE[k % len(PALETTE)]
            label = "Noise" if uid == -1 else (
                CLASS_NAMES.get(uid, f"Class {uid}") if title.startswith("Ground")
                else f"Cluster {uid}"
            )
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=10,
                       alpha=0.7, label=label, linewidths=0)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.set_xlabel("t-SNE Dim 1", color="#888888", fontsize=9)
        ax.set_ylabel("t-SNE Dim 2", color="#888888", fontsize=9)
        ax.legend(fontsize=7.5, framealpha=0.3, labelcolor="white",
                  facecolor=DARK_BG, loc="best", markerscale=2)

    plt.suptitle("exp07 SupCon & Semi-Supervised DEC — Embedding Space (t-SNE)",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig1_tsne_embedding.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_cluster_composition(results: list[dict], hdb_labels: np.ndarray,
                              cluster_map: dict, out_dir: str) -> str:
    unique_clusters  = sorted(set(hdb_labels) - {-1})
    unique_class_ids = sorted(set(r["gt_class_id"] for r in results))
    comp = np.zeros((len(unique_clusters), len(unique_class_ids)), dtype=int)
    for r, lbl in zip(results, hdb_labels):
        if lbl == -1:
            continue
        ci = unique_clusters.index(lbl)
        if r["gt_class_id"] in unique_class_ids:
            gi = unique_class_ids.index(r["gt_class_id"])
            comp[ci, gi] += 1

    fig, ax = plt.subplots(figsize=(max(8, len(unique_clusters) * 0.9 + 2), 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    bottom = np.zeros(len(unique_clusters))
    x = np.arange(len(unique_clusters))
    for gi, cls_id in enumerate(unique_class_ids):
        vals = comp[:, gi]
        ax.bar(x, vals, bottom=bottom, color=PALETTE[gi % len(PALETTE)],
               label=CLASS_NAMES.get(cls_id, f"Class {cls_id}"), width=0.7, alpha=0.9)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in unique_clusters], color="white", fontsize=9)
    ax.set_ylabel("Pulse Count", color="#AAAAAA")
    ax.set_xlabel("HDBSCAN Discovered Cluster", color="#AAAAAA")
    ax.set_title("Cluster Composition by Ground-Truth PD Type",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="y", colors="#888888")
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white",
              facecolor=DARK_BG, loc="upper right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig2_cluster_composition.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_soft_assignment_heatmap(results: list[dict], out_dir: str) -> str:
    gt_classes = np.array([r["gt_class_id"] for r in results])
    q_matrix   = np.array([r["soft_q"]       for r in results])
    sort_idx   = np.argsort(gt_classes)
    q_sorted   = q_matrix[sort_idx]
    gt_sorted  = gt_classes[sort_idx]

    fig, ax = plt.subplots(figsize=(min(16, q_matrix.shape[1] * 1.2 + 2), 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    im = ax.imshow(q_sorted.T, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Soft Assignment q(i,k)")
    prev_cls = gt_sorted[0]
    for i, c in enumerate(gt_sorted):
        if c != prev_cls:
            ax.axvline(x=i - 0.5, color="white", lw=0.5, alpha=0.5)
            prev_cls = c
    ax.set_xlabel("Pulses (sorted by GT class)", color="#AAAAAA")
    ax.set_ylabel("DEC Cluster Index", color="#AAAAAA")
    ax.set_title("Soft Assignment Heatmap Q (sorted by GT class)",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#888888")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig3_soft_assignment_heatmap.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# New plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          out_dir: str, exp_tag: str = "exp07") -> str:
    from sklearn.metrics import confusion_matrix
    valid  = y_pred != -1
    yt, yp = y_true[valid], y_pred[valid]
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    cm     = confusion_matrix(yt, yp, labels=labels, normalize="true")
    ticks  = [CLASS_NAMES.get(l, f"C{l}") for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9 + 2),
                                     max(5, len(labels) * 0.8 + 2)))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax(ax, f"Normalised Confusion Matrix ({exp_tag})")
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


def plot_pr_curves(y_true: np.ndarray, class_scores: np.ndarray,
                   out_dir: str, exp_tag: str = "exp07") -> str:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    unique_classes = sorted(set(y_true.tolist()))
    n_cls = class_scores.shape[1]
    cmap  = plt.get_cmap("tab20", max(len(unique_classes), 1))

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax(ax, (f"Precision-Recall Curves ({exp_tag})\n"
                  "[Proxy confidence: DEC q_soft remapped via hard-assignment alignment]"))
    for i, cls_id in enumerate(unique_classes):
        if cls_id >= n_cls:
            continue
        y_bin  = (y_true == cls_id).astype(int)
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


def plot_roc_curves(y_true: np.ndarray, class_scores: np.ndarray,
                    out_dir: str, exp_tag: str = "exp07") -> str:
    from sklearn.metrics import roc_curve, roc_auc_score
    unique_classes = sorted(set(y_true.tolist()))
    n_cls = class_scores.shape[1]
    cmap  = plt.get_cmap("tab20", max(len(unique_classes), 1))

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax(ax, (f"ROC Curves ({exp_tag})\n"
                  "[Proxy confidence: DEC q_soft remapped via hard-assignment alignment]"))
    ax.plot([0, 1], [0, 1], "--", color="#555555", lw=1.5, label="Random (AUC=0.5)")
    for i, cls_id in enumerate(unique_classes):
        if cls_id >= n_cls:
            continue
        y_bin  = (y_true == cls_id).astype(int)
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


def plot_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray,
                           class_scores: np.ndarray,
                           out_dir: str, exp_tag: str = "exp07") -> str:
    from sklearn.calibration import calibration_curve as sk_cal_curve
    valid  = y_pred != -1
    yt, yp = y_true[valid], y_pred[valid]
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
    _dark_ax(ax, (f"Confidence / Calibration Curve ({exp_tag})\n"
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


def plot_time_vs_distance(results: list[dict], out_dir: str,
                          exp_tag: str = "exp07", max_pairs: int = 10_000) -> str:
    scenes: dict = defaultdict(list)
    for r in results:
        scenes[(r["shard_path"], r["scene_idx"])].append(r)

    rng       = np.random.default_rng(42)
    all_pairs: list = []
    for pulses in scenes.values():
        n = len(pulses)
        for i in range(n):
            for j in range(i + 1, n):
                t_i  = pulses[i]["start_idx"] * pulses[i]["time_res"]
                t_j  = pulses[j]["start_idx"] * pulses[j]["time_res"]
                td   = abs(t_i - t_j)
                ed   = float(np.linalg.norm(pulses[i]["emb"] - pulses[j]["emb"]))
                same = pulses[i]["gt_inst_id"] == pulses[j]["gt_inst_id"]
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
    _dark_ax(ax, (f"Time Difference vs Embedding Distance ({exp_tag})\n"
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="exp07 SupCon & Semi-Supervised DEC: Inference, HDBSCAN Clustering, Visualization"
    )
    parser.add_argument("--checkpoint_id", required=True,
                        help="NodeID of the trained checkpoint (e.g. M3Fu)")
    parser.add_argument("--source", action="append", dest="sources", default=[],
                        metavar="path:type:shards",
                        help="STFT feature source. Repeatable. Format: path:type:shard1,shard2,...")
    parser.add_argument("--time_threshold",  type=float, default=100e-9,
                        help="Max temporal distance (s) to group pulses into same instance.")
    parser.add_argument("--dist_threshold",  type=float, default=0.5,
                        help="Max L2 embedding distance to group pulses into same instance.")
    parser.add_argument("--min_cluster_size", type=int, default=5,
                        help="HDBSCAN min_cluster_size parameter.")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="HDBSCAN cluster_selection_epsilon parameter.")
    parser.add_argument("--reduce_method", type=str, default="tsne",
                        choices=["none", "pca", "tsne"],
                        help="Method to reduce embeddings before HDBSCAN.")
    parser.add_argument("--reduce_dims", type=int, default=2,
                        help="Dimensionality for reduction (default 2 for t-SNE).")
    parser.add_argument("--grouping_mode", type=str, default="greedy",
                        choices=["greedy", "channel_capped"],
                        help=("greedy: any pulse within thresholds joins the instance. "
                              "channel_capped: at most one pulse per channel (max 4)."))
    parser.add_argument("--smoke_test", action="store_true",
                        help="Limit to 1 shard per source for a quick sanity check.")
    args = parser.parse_args()

    if not args.sources:
        print("[Error] Provide at least one --source path:type:shards")
        sys.exit(1)

    # 1. Config + model
    config = load_config(args.checkpoint_id)
    device = select_device(config["training"].get("device", "auto"))
    print(f"[Device] {device}")
    print(f"[Checkpoint] {args.checkpoint_id}")

    task = SupConDECTask(config)
    load_weights(args.checkpoint_id, task)
    task = task.to(device)

    # 2. Dataset
    sources = parse_sources(args.sources)
    if args.smoke_test:
        for s in sources:
            s["train_shards"] = [s["train_shards"][0]]

    data_cfg = config["data"]
    dataset = DECDataset_Exp07(
        sources       = sources,
        shard_key     = "train_shards",
        max_pulse_len = data_cfg["max_pulse_len"],
        augment       = False,
        label_fraction= 1.0,
    )
    loader = DataLoader(dataset, batch_size=config["training"].get("batch_size", 128),
                        shuffle=False, num_workers=0)
    if not dataset.index:
        print("[Warning] No pulses found. Check your --source arguments.")
        sys.exit(0)
    print(f"[Data] {len(dataset):,} pulses loaded from {len(sources)} source(s).")

    # 3. Extract embeddings (timed)
    t_infer_start    = time.time()
    features         = extract_features(task, loader, dataset, device)
    inference_time_s = time.time() - t_infer_start
    print(f"[Timing] Inference: {inference_time_s:.2f}s  ({len(features) / inference_time_s:.1f} samples/s)")

    # 4. HDBSCAN + alignment
    hdb_labels, reduce_time_s, hdbscan_time_s, embs_reduced = run_hdbscan(
        features, min_cluster_size=args.min_cluster_size,
        reduce_method=args.reduce_method, reduce_dims=args.reduce_dims,
        epsilon=args.epsilon
    )
    print(f"[Timing] Reduction: {reduce_time_s:.2f}s  ({len(features) / reduce_time_s if reduce_time_s > 0 else 0:.1f} samples/s)")
    print(f"[Timing] HDBSCAN:   {hdbscan_time_s:.2f}s  ({len(features) / hdbscan_time_s if hdbscan_time_s > 0 else 0:.1f} samples/s)")
    
    cluster_map = align_clusters(features, hdb_labels)
    print(f"[Alignment] Cluster->Class: "
          f"{ {int(k): CLASS_NAMES.get(v, str(v)) for k, v in cluster_map.items()} }")

    # 5. Build class probability scores
    class_scores, _ = build_class_scores_from_dec(features)

    # 6. PD instance grouping
    results = group_pd_instances(features, hdb_labels,
                                  args.time_threshold, args.dist_threshold,
                                  mode=args.grouping_mode)

    # 7. Metrics
    metrics = compute_metrics(results, hdb_labels, cluster_map,
                              class_scores=class_scores,
                              inference_time_s=inference_time_s,
                              reduce_time_s=reduce_time_s,
                              hdbscan_time_s=hdbscan_time_s)
    metrics.update({
        "checkpoint_id":    args.checkpoint_id,
        "time_threshold":   args.time_threshold,
        "dist_threshold":   args.dist_threshold,
        "grouping_mode":    args.grouping_mode,
        "min_cluster_size": args.min_cluster_size,
        "epsilon":          args.epsilon,
        "reduce_method":    args.reduce_method,
        "reduce_dims":      args.reduce_dims,
        "sources":          args.sources,
    })

    print(f"\n[Results]  CLS Accuracy    : {metrics['classification_accuracy']:.4f}")
    print(f"[Results]  Macro F1        : {metrics['cls_f1_macro']:.4f}")
    print(f"[Results]  Weighted F1     : {metrics['cls_f1_weighted']:.4f}")
    print(f"[Results]  AUC ROC (macro) : {metrics.get('auc_roc_macro')}")
    print(f"[Results]  Silhouette      : {metrics.get('silhouette_score')}")
    print(f"[Results]  Davies-Bouldin  : {metrics.get('davies_bouldin_index')}")
    print(f"[Results]  Calinski-H      : {metrics.get('calinski_harabasz_index')}")
    print(f"[Results]  ARI             : {metrics['ari']:.4f}")
    print(f"[Results]  NMI             : {metrics['nmi']:.4f}")
    print(f"[Results]  Cluster Purity  : {metrics['cluster_purity']:.4f}")
    print(f"[Results]  Grouping F1     : {metrics['grouping_f1']:.4f}")
    print(f"[Results]  Clusters found  : {metrics['n_clusters_found']}")
    print(f"[Results]  Noise points    : {metrics['n_noise_points']}")
    print(f"[Results]  Inference time  : {metrics['time_inference_s']:.2f}s")
    print(f"[Results]  Reduction time  : {metrics['time_reduce_s']:.2f}s")
    print(f"[Results]  HDBSCAN time    : {metrics['time_hdbscan_s']:.2f}s")
    print(f"[Results]  Inf. Throughput : {metrics.get('throughput_inference_smp_s')} smp/s")
    print(f"[Results]  Red. Throughput : {metrics.get('throughput_reduce_smp_s')} smp/s")
    print(f"[Results]  HDB. Throughput : {metrics.get('throughput_hdbscan_smp_s')} smp/s")
    print("\n[Results]  Per-class:")
    for cid, stat in metrics["per_class_stats"].items():
        print(f"             {stat['name']:30s}  n={stat['count']:5d}  "
              f"acc={stat['accuracy']:.4f}  prec={stat['precision']:.4f}  "
              f"rec={stat['recall']:.4f}  f1={stat['f1']:.4f}")

    # 8. Output directory
    run_ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_id       = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    out_cfg      = config.get("output", {})
    results_base = os.path.abspath(out_cfg.get("results_dir", "data/classification_output/exp07_supcon_dec"))
    out_dir      = os.path.join(results_base, f"{run_ts}_inf-{args.checkpoint_id}-{inf_id}")
    os.makedirs(out_dir, exist_ok=True)

    # 9. Save metrics JSON
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 10. Save predictions.h5
    h5_path = os.path.join(out_dir, "predictions.h5")
    with h5py.File(h5_path, "w") as f:
        for r in results:
            r["cluster_id"]    = r.get("hdb_cluster", -1)
            r["pred_class_id"] = cluster_map.get(r["cluster_id"], -1)
        keys_to_save = ["shard_path", "scene_idx", "ch_idx", "start_idx",
                        "gt_class_id", "pred_class_id", "cluster_id",
                        "gt_inst_id", "pred_inst_id", "time_res"]
        for k in keys_to_save:
            values = [r.get(k, -1) for r in results]
            if not values:
                continue
            if isinstance(values[0], str):
                f.create_dataset(k, data=np.array(values, dtype=object),
                                 dtype=h5py.string_dtype(encoding="utf-8"))
            else:
                f.create_dataset(k, data=np.array(values))
    print(f"[Export] Saved pulse mappings -> {h5_path}")

    # 11. Original plots
    plot_embeddings(results, hdb_labels, cluster_map, out_dir, embs_reduced=embs_reduced)
    plot_cluster_composition(results, hdb_labels, cluster_map, out_dir)
    plot_soft_assignment_heatmap(results, out_dir)

    # 12. New plots
    y_true = np.array([r["gt_class_id"] for r in results])
    y_pred = np.array([cluster_map.get(r.get("hdb_cluster", -1), -1) for r in results])
    plot_confusion_matrix(y_true, y_pred, out_dir, exp_tag="exp07")
    plot_pr_curves(y_true, class_scores, out_dir, exp_tag="exp07")
    plot_roc_curves(y_true, class_scores, out_dir, exp_tag="exp07")
    plot_calibration_curve(y_true, y_pred, class_scores, out_dir, exp_tag="exp07")
    plot_time_vs_distance(results, out_dir, exp_tag="exp07")

    # 13. Lineage
    history = (
        f"SupCon & Semi-Supervised DEC Inference (exp07) | Checkpoint: {args.checkpoint_id} | "
        f"Sources: {args.sources} | Grouping: {args.grouping_mode} | "
        f"Clusters: {metrics['n_clusters_found']} | "
        f"CLS Acc: {metrics['classification_accuracy']:.4f} | "
        f"Macro F1: {metrics['cls_f1_macro']:.4f} | "
        f"Silhouette: {metrics.get('silhouette_score')} | "
        f"GroupF1: {metrics['grouping_f1']:.4f} | "
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("\n[Lineage] Registering prediction node...")
    register_process(
        parent_id        = args.checkpoint_id,
        stage            = "prediction",
        method           = "dec_semi_hdbscan",
        folder_path      = out_dir,
        appended_history = history,
        force_node_id    = inf_id,
    )
    print(f"[Lineage] Node {inf_id} registered (child of {args.checkpoint_id})")

    with open(os.path.join(out_dir, "analysis_history.txt"), "w") as f:
        f.write(get_node_history(inf_id))

    print(f"\n[Done] Results saved -> {out_dir}")
    return metrics, out_dir


if __name__ == "__main__":
    main()
