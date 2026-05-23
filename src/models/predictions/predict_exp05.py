"""
predict_exp05.py
================
Inference + Visualization script for exp05 Deep Embedded Clustering (DEC).

Usage:
    python src/models/predictions/predict_exp05.py \
        --checkpoint_id vWIh \
        --source "data/raw/synthesised/20260427_170034_sy-ShmH-ShmH:synthesised:17,18,19,20" \
        --source "data/raw/measured/20260517_004832_ms-K7FX-K7FX:measured:17,18,19,20" \
        [--time_threshold 100e-9] \
        [--dist_threshold 0.5] \
        [--min_cluster_size 5]

Each --source is formatted as: path:type:shard1,shard2,...
"""

import argparse
import json
import os
import random
import string
import sys
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

from src.models.tasks.task_exp05_dec import DECTask
from src.models.data.dataset_exp05_dec import DECDataset
from src.utils.lineage_tracker import register_process, get_node_history

# ---------------------------------------------------------------------------
# Class name mapping (PD type index -> human-readable label)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "PD1 Void Simulated",
    1: "PD2 Incision Simulated",
    3: "PD2 Incision Measured",
    4: "PD3 Delamination Measured",
    5: "PD4 FeOx Measured",
    6: "PD5 FeOx High Measured",
    7: "SEDO",
    8: "DED",
    9: "DEDO",
    10: "SMG",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(checkpoint_id: str) -> dict:
    import yaml
    p = os.path.abspath(f"models/configuration_snapshots/config_{checkpoint_id}.yaml")
    if not os.path.exists(p):
        print(f"[Error] Config not found: {p}"); sys.exit(1)
    with open(p) as f:
        return yaml.safe_load(f)

def load_weights(checkpoint_id: str, task: DECTask):
    p = os.path.abspath(f"models/weights/model_{checkpoint_id}.pt")
    if not os.path.exists(p):
        print(f"[Error] Weights not found: {p}"); sys.exit(1)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    task.load_state_dict(ckpt["model_state"])
    print(f"[Checkpoint] Loaded {p}  (epoch {ckpt.get('epoch','?')})")

def select_device(cfg: str) -> torch.device:
    if cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg)

def parse_sources(raw_list: list[str]) -> list[dict]:
    sources = []
    for raw in raw_list:
        tokens = raw.strip().split(":")
        if len(tokens) < 3:
            print(f"[Error] Bad source format: '{raw}'. Use path:type:shards")
            sys.exit(1)
        path   = tokens[0]
        stype  = tokens[1]
        shards = [int(x) for x in tokens[2].split(",") if x.strip()]
        sources.append({"type": stype, "path": path,
                        "train_shards": shards, "val_shards": shards})
    return sources

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(task, loader, dataset, device) -> list[dict]:
    task.eval()
    features = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            sig = batch[0].to(device)
            z, q = task(sig)
            z = z.cpu().numpy()
            q = q.cpu().numpy()
            for b in range(sig.shape[0]):
                global_idx = batch_idx * loader.batch_size + b
                if global_idx >= len(dataset.index):
                    break
                shard_path, scene_idx, ch_idx, start_idx, end_idx, class_id, inst_id, time_res = dataset.index[global_idx]
                features.append({
                    "shard_path": shard_path,
                    "scene_idx":  scene_idx,
                    "ch_idx":     ch_idx,
                    "start_idx":  start_idx,
                    "gt_class_id": class_id,
                    "gt_inst_id":  inst_id,
                    "time_res":   time_res,
                    "emb":        z[b],
                    "soft_q":     q[b],
                })
    print(f"[Inference] Extracted {len(features)} pulse embeddings.")
    return features

# ---------------------------------------------------------------------------
# HDBSCAN Clustering
# ---------------------------------------------------------------------------

def run_hdbscan(features, min_cluster_size=5) -> np.ndarray:
    try:
        import hdbscan
    except ImportError:
        print("[Error] hdbscan not installed. Run: pip install hdbscan"); sys.exit(1)
    embs = np.array([f["emb"] for f in features])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embs)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[HDBSCAN]   Clusters found: {n_clusters}  |  Noise points: {(labels==-1).sum()}")
    return labels

# ---------------------------------------------------------------------------
# Post-hoc cluster -> class alignment
# ---------------------------------------------------------------------------

def align_clusters(features, hdb_labels) -> dict:
    votes = defaultdict(list)
    for feat, lbl in zip(features, hdb_labels):
        if lbl != -1:
            votes[lbl].append(feat["gt_class_id"])
    mapping = {k: max(set(v), key=v.count) for k, v in votes.items()}
    mapping[-1] = -1
    return mapping

# ---------------------------------------------------------------------------
# PD Instance Grouping
# ---------------------------------------------------------------------------

def group_pd_instances(features, hdb_labels, time_th, dist_th) -> list[dict]:
    scene_groups = defaultdict(list)
    for i, feat in enumerate(features):
        scene_groups[(feat["shard_path"], feat["scene_idx"])].append((i, feat))

    results = []
    inst_counter = 0
    assigned_global = {}

    for pulses in scene_groups.values():
        for i, (idx, feat) in enumerate(pulses):
            if idx in assigned_global:
                continue
            inst_counter += 1
            assigned_global[idx] = inst_counter
            t_i = feat["start_idx"] * feat["time_res"]
            for j, (jdx, feat_j) in enumerate(pulses):
                if i == j or jdx in assigned_global:
                    continue
                t_j = feat_j["start_idx"] * feat_j["time_res"]
                if abs(t_i - t_j) > time_th:
                    continue
                if np.linalg.norm(feat["emb"] - feat_j["emb"]) < dist_th:
                    assigned_global[jdx] = inst_counter

    for i, feat in enumerate(features):
        results.append({**feat,
                        "hdb_cluster": int(hdb_labels[i]),
                        "pred_inst_id": assigned_global.get(i, -1)})
    return results

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results, hdb_labels, cluster_map) -> dict:
    embs   = np.array([r["emb"] for r in results])
    y_true = np.array([r["gt_class_id"] for r in results])
    y_pred = np.array([cluster_map.get(int(hdb_labels[i]), -1) for i in range(len(results))])

    valid = y_pred != -1
    acc   = float((y_true[valid] == y_pred[valid]).mean()) if valid.sum() > 0 else 0.0

    # Silhouette (only if >1 cluster and not all noise)
    sil = -1.0
    unique_lbls = set(hdb_labels) - {-1}
    if len(unique_lbls) > 1 and valid.sum() > 1:
        try:
            sil = float(silhouette_score(embs[valid], hdb_labels[valid]))
        except Exception:
            pass

    # Pair-based grouping
    tp = total_gt = total_pred = 0
    scenes = defaultdict(list)
    for r in results:
        scenes[(r["shard_path"], r["scene_idx"])].append(r)
    for pulses in scenes.values():
        n = len(pulses)
        for i in range(n):
            for j in range(i+1, n):
                gs = pulses[i]["gt_inst_id"]  == pulses[j]["gt_inst_id"]
                ps = pulses[i]["pred_inst_id"] == pulses[j]["pred_inst_id"]
                if gs: total_gt += 1
                if ps: total_pred += 1
                if gs and ps: tp += 1

    prec = tp / total_pred if total_pred > 0 else 1.0
    rec  = tp / total_gt   if total_gt   > 0 else 1.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

    return {
        "classification_accuracy":  acc,
        "silhouette_score":         sil,
        "grouping_precision":       prec,
        "grouping_recall":          rec,
        "grouping_f1":              f1,
        "n_pulses":                 len(results),
        "n_clusters_found":         int(n_clusters),
        "n_noise_points":           int((hdb_labels == -1).sum()),
        "n_gt_instances":           len(set((r["shard_path"], r["gt_inst_id"]) for r in results)),
        "n_pred_instances":         len(set(r["pred_inst_id"] for r in results)),
    }

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A0572", "#118AB2", "#06D6A0", "#FFD166", "#EF476F",
    "#8338EC", "#3A86FF", "#FB5607",
]

def _tsne_2d(embs: np.ndarray) -> np.ndarray:
    print("[t-SNE] Reducing to 2D (this may take a moment)...")
    perp = min(30, max(5, len(embs) // 10))
    return TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000).fit_transform(embs)

def plot_embeddings(results, hdb_labels, cluster_map, out_dir: str):
    """
    Figure 1: Two-panel t-SNE scatter.
      Left  — coloured by Ground-Truth Class  (what the labels say)
      Right — coloured by HDBSCAN Cluster     (what the model discovered)
    Suitable for the extended abstract.
    """
    embs = np.array([r["emb"] for r in results])
    xy   = _tsne_2d(embs)

    gt_classes  = np.array([r["gt_class_id"]    for r in results])
    hdb_arr     = np.array([r["hdb_cluster"]     for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0F0F0F")

    ax_titles = ["Ground-Truth Class Labels", "HDBSCAN Discovered Clusters"]
    label_arrays = [gt_classes, hdb_arr]

    for ax, arr, title in zip(axes, label_arrays, ax_titles):
        ax.set_facecolor("#1A1A2E")
        unique = sorted(set(arr))
        for k, uid in enumerate(unique):
            mask = arr == uid
            color = "#888888" if uid == -1 else PALETTE[k % len(PALETTE)]
            label = "Noise" if uid == -1 else (
                CLASS_NAMES.get(uid, f"Class {uid}") if title.startswith("Ground") else f"Cluster {uid}"
            )
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=10,
                       alpha=0.7, label=label, linewidths=0)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.set_xlabel("t-SNE Dim 1", color="#888888", fontsize=9)
        ax.set_ylabel("t-SNE Dim 2", color="#888888", fontsize=9)
        leg = ax.legend(fontsize=7.5, framealpha=0.3,
                        labelcolor="white", facecolor="#0F0F0F",
                        loc="best", markerscale=2)

    plt.suptitle("exp05 DEC — Embedding Space Visualisation (t-SNE)",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig1_tsne_embedding.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path

def plot_cluster_composition(results, hdb_labels, cluster_map, out_dir: str):
    """
    Figure 2: Stacked bar chart showing what GT classes make up each HDBSCAN cluster.
    Demonstrates how well the discovered clusters align with real PD types.
    """
    unique_clusters = sorted(set(hdb_labels) - {-1})
    unique_classes  = sorted(CLASS_NAMES.keys())

    # Build composition matrix: [cluster, class] -> count
    comp = np.zeros((len(unique_clusters), len(unique_classes)), dtype=int)
    for r, lbl in zip(results, hdb_labels):
        if lbl == -1:
            continue
        ci = unique_clusters.index(lbl)
        if r["gt_class_id"] in unique_classes:
            gi = unique_classes.index(r["gt_class_id"])
            comp[ci, gi] += 1

    fig, ax = plt.subplots(figsize=(max(8, len(unique_clusters)*0.9 + 2), 5))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#1A1A2E")

    bottom = np.zeros(len(unique_clusters))
    x = np.arange(len(unique_clusters))
    for gi, cls_id in enumerate(unique_classes):
        vals = comp[:, gi]
        bars = ax.bar(x, vals, bottom=bottom,
                      color=PALETTE[gi % len(PALETTE)],
                      label=CLASS_NAMES.get(cls_id, f"Class {cls_id}"),
                      width=0.7, alpha=0.9)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in unique_clusters],
                       color="white", fontsize=9)
    ax.set_ylabel("Pulse Count", color="#AAAAAA")
    ax.set_xlabel("HDBSCAN Discovered Cluster", color="#AAAAAA")
    ax.set_title("Cluster Composition by Ground-Truth PD Type",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="y", colors="#888888")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(fontsize=8, framealpha=0.3, labelcolor="white",
                    facecolor="#0F0F0F", loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig2_cluster_composition.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="exp05 DEC: Inference, HDBSCAN Clustering, Visualization"
    )
    parser.add_argument("--checkpoint_id", required=True,
                        help="NodeID of the trained checkpoint (e.g. vWIh)")
    parser.add_argument("--source", action="append", dest="sources", default=[],
                        metavar="path:type:shards",
                        help="Dataset source. Repeatable. Format: path:type:shard1,shard2,...")
    parser.add_argument("--time_threshold", type=float, default=100e-9)
    parser.add_argument("--dist_threshold",  type=float, default=0.5)
    parser.add_argument("--min_cluster_size", type=int,  default=5)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if not args.sources:
        print("[Error] Provide at least one --source path:type:shards")
        sys.exit(1)

    # 1. Config + model
    config   = load_config(args.checkpoint_id)
    device   = select_device(config["training"].get("device", "auto"))
    print(f"[Device] {device}")

    task = DECTask(config)
    load_weights(args.checkpoint_id, task)
    task = task.to(device)

    # 2. Dataset
    sources = parse_sources(args.sources)
    if args.smoke_test:
        for s in sources:
            s["train_shards"] = [s["train_shards"][0]]
            s["val_shards"]   = [s["val_shards"][0]]

    data_cfg = config["data"]
    wavelet_kwargs = dict(
        denoise        = data_cfg.get("denoise",        True),
        wavelet        = data_cfg.get("wavelet",        "db4"),
        wavelet_level  = data_cfg.get("wavelet_level",  4),
        threshold_mode = data_cfg.get("threshold_mode", "soft"),
    )
    dataset = DECDataset(sources=sources, shard_key="train_shards",
                         max_pulse_len=data_cfg["max_pulse_len"],
                         augment=False, **wavelet_kwargs)
    loader  = DataLoader(dataset, batch_size=config["training"].get("batch_size", 128),
                         shuffle=False, num_workers=0)
    if not dataset.index:
        print("[Warning] No pulses found. Check your --source arguments.")
        sys.exit(0)

    # 3. Extract embeddings
    features   = extract_features(task, loader, dataset, device)
    hdb_labels = run_hdbscan(features, min_cluster_size=args.min_cluster_size)
    cluster_map = align_clusters(features, hdb_labels)
    print(f"[Alignment] Cluster->Class map: {cluster_map}")

    # 4. PD instance grouping
    results = group_pd_instances(features, hdb_labels,
                                 args.time_threshold, args.dist_threshold)

    # 5. Metrics
    metrics = compute_metrics(results, hdb_labels, cluster_map)
    metrics.update({
        "checkpoint_id":   args.checkpoint_id,
        "time_threshold":  args.time_threshold,
        "dist_threshold":  args.dist_threshold,
        "min_cluster_size": args.min_cluster_size,
        "sources":         args.sources,
    })
    print(f"\n[Results]  CLS Accuracy : {metrics['classification_accuracy']:.4f}")
    print(f"[Results]  Silhouette   : {metrics['silhouette_score']:.4f}")
    print(f"[Results]  Grouping F1  : {metrics['grouping_f1']:.4f}")
    print(f"[Results]  Clusters     : {metrics['n_clusters_found']}")
    print(f"[Results]  Noise Points : {metrics['n_noise_points']}")

    # 6. Output directory
    run_ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_id     = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    out_cfg    = config.get("output", {})
    results_base = os.path.abspath(out_cfg.get("results_dir", "data/classification_output/exp05_dec"))
    method     = config["experiment"].get("name", "exp05_dec")
    out_dir    = os.path.join(results_base, method, f"{run_ts}_inf-{args.checkpoint_id}-{inf_id}")
    os.makedirs(out_dir, exist_ok=True)

    # 7. Save metrics + history
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    history = (
        f"DEC Inference | Checkpoint: {args.checkpoint_id} | "
        f"Sources: {args.sources} | "
        f"Clusters found: {metrics['n_clusters_found']} | "
        f"CLS Acc: {metrics['classification_accuracy']:.4f} | "
        f"Silhouette: {metrics['silhouette_score']:.4f} | "
        f"GroupF1: {metrics['grouping_f1']:.4f} | "
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    # 8. Plots
    plot_embeddings(results, hdb_labels, cluster_map, out_dir)
    plot_cluster_composition(results, hdb_labels, cluster_map, out_dir)

    # 9. Lineage
    print("\n[Lineage] Registering...")
    register_process(parent_id=args.checkpoint_id, stage="prediction",
                     method="dec_hdbscan", folder_path=out_dir,
                     appended_history=history, force_node_id=inf_id)
    print(f"[Lineage] Node {inf_id} (child of {args.checkpoint_id})")
    print(f"\n[Done] Results saved -> {out_dir}")

    with open(os.path.join(out_dir, "analysis_history.txt"), "w") as f:
        full_history = get_node_history(inf_id)
        f.write(full_history)
    return metrics, out_dir


if __name__ == "__main__":
    main()
