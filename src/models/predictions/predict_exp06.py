"""
predict_exp06.py
================
Inference + Visualization script for exp06 Semi-Supervised DEC.

Usage:
    python src/models/predictions/predict_exp06.py \\
        --checkpoint_id M3Fu \\
        --source "data/features/stft_magnitude/20260524_234212-cw-TuKT-vraW:cwru:14,15,16" \\
        --source "data/features/stft_magnitude/20260524_234301-sy-0s6o-UI2r:equation:17,18,19,20" \\
        [--min_cluster_size 5] \\
        [--time_threshold 100e-9] \\
        [--dist_threshold 0.5]

Each --source is formatted as: path:type:shard1,shard2,...

Key differences from exp04/exp05:
  - Sources point to data/features/stft_magnitude/ (2D STFT features), NOT raw data.
  - No wavelet preprocessing step (already extracted at feature stage).
  - Ground-truth class comes from index 6 (actual_class_id), NOT index 0.
    The reported_class (index 1) may be -1 for unlabelled samples — we never
    use it for metrics.
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

from src.models.tasks.task_exp06_dec   import SemiSupervisedDECTask
from src.models.data.dataset_exp06_dec import DECDataset_Exp06
from src.utils.lineage_tracker         import register_process, get_node_history

# ---------------------------------------------------------------------------
# Class name mapping  (actual class_id -> human-readable label)
# These IDs match the values stored in the 'labels' array of the STFT shards.
# CWRU classes start at 11+. Synthetic/Equation classes start at 0.
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A0572", "#118AB2", "#06D6A0", "#FFD166", "#EF476F",
    "#8338EC", "#3A86FF", "#FB5607", "#FFBE0B", "#8AC926",
]


def load_config(checkpoint_id: str) -> dict:
    import yaml
    p = os.path.abspath(f"models/configuration_snapshots/config_{checkpoint_id}.yaml")
    if not os.path.exists(p):
        print(f"[Error] Config not found: {p}"); sys.exit(1)
    with open(p) as f:
        return yaml.safe_load(f)


def load_weights(checkpoint_id: str, task: SemiSupervisedDECTask):
    p = os.path.abspath(f"models/weights/model_{checkpoint_id}.pt")
    if not os.path.exists(p):
        print(f"[Error] Weights not found: {p}"); sys.exit(1)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
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
            "train_shards": shards,   # DECDataset_Exp06 always reads via 'train_shards'
            "val_shards":   shards,
        })
    return sources


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(task: SemiSupervisedDECTask, loader: DataLoader,
                     dataset: DECDataset_Exp06, device: torch.device) -> list[dict]:
    """
    Run inference over the dataset and collect per-pulse embeddings + soft assignments.

    Batch layout from DECDataset_Exp06 (augment=False):
        [0] signal          (B, 1, F, T)
        [1] reported_class  int  (may be -1 for unlabelled)
        [2] inst_id         int
        [3] shard_path      str
        [4] start_idx       int
        [5] time_res        float
        [6] actual_class    int  ← ground-truth label, always set
    """
    task.eval()
    records = []
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
                (shard_path, scene_idx, ch_idx, start_idx, end_idx,
                 reported_class, actual_class, inst_id, time_res, hop_length) = dataset.index[global_idx]

                records.append({
                    "shard_path":   shard_path,
                    "scene_idx":    scene_idx,
                    "ch_idx":       ch_idx,
                    "start_idx":    start_idx,
                    "gt_class_id":  actual_class,     # true label — always present
                    "gt_inst_id":   inst_id,
                    "time_res":     time_res,
                    "emb":          z[b],
                    "soft_q":       q[b],
                })
    print(f"[Inference] Extracted {len(records)} pulse embeddings.")
    return records


# ---------------------------------------------------------------------------
# HDBSCAN Clustering
# ---------------------------------------------------------------------------

def run_hdbscan(features: list[dict], min_cluster_size: int = 5, reduce_method: str = "tsne", reduce_dims: int = 2, epsilon: float = 0.0) -> np.ndarray:
    try:
        import hdbscan
    except ImportError:
        print("[Error] hdbscan not installed. Run: pip install hdbscan"); sys.exit(1)
        
    embs = np.array([f["emb"] for f in features])
    
    if reduce_method == "pca" and reduce_dims > 0 and embs.shape[1] > reduce_dims:
        from sklearn.decomposition import PCA
        print(f"[PCA] Reducing dimensions from {embs.shape[1]} to {reduce_dims} before HDBSCAN...")
        embs = PCA(n_components=reduce_dims, random_state=42).fit_transform(embs)
    elif reduce_method == "tsne" and reduce_dims > 0 and embs.shape[1] > reduce_dims:
        from sklearn.manifold import TSNE
        if reduce_dims > 3:
            print(f"[Warning] t-SNE only supports up to 3 dimensions efficiently. Clamping reduce_dims from {reduce_dims} to 3.")
            reduce_dims = 3
        print(f"[t-SNE] Reducing dimensions from {embs.shape[1]} to {reduce_dims} before HDBSCAN...")
        perp = min(30, max(5, len(embs) // 10))
        embs = TSNE(n_components=reduce_dims, perplexity=perp, random_state=42, max_iter=1000).fit_transform(embs)
        
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=epsilon, metric="euclidean")
    labels = clusterer.fit_predict(embs)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[HDBSCAN] Clusters found: {n_clusters}  |  Noise points: {(labels == -1).sum()}")
    return labels


# ---------------------------------------------------------------------------
# Post-hoc majority-vote cluster→class alignment
# ---------------------------------------------------------------------------

def align_clusters(features: list[dict], hdb_labels: np.ndarray) -> dict:
    """
    For each HDBSCAN cluster, determine which actual class_id appears most often.
    This is a post-hoc alignment — the model never saw these labels during training.
    """
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

def group_pd_instances(features: list[dict], hdb_labels: np.ndarray,
                        time_th: float, dist_th: float) -> list[dict]:
    """
    Group pulses within the same scene into 'instances' based on temporal proximity
    and embedding distance. Mirrors the exp04/exp05 approach.
    """
    scene_groups = defaultdict(list)
    for i, feat in enumerate(features):
        scene_groups[(feat["shard_path"], feat["scene_idx"])].append((i, feat))

    assigned_global = {}
    inst_counter = 0
    for pulses in scene_groups.values():
        for i, (idx, feat) in enumerate(pulses):
            if idx in assigned_global:
                continue
            inst_counter += 1
            assigned_global[idx] = inst_counter
            t_i = feat["start_idx"] * feat["time_res"]
            for _, (jdx, feat_j) in enumerate(pulses):
                if jdx == idx or jdx in assigned_global:
                    continue
                t_j = feat_j["start_idx"] * feat_j["time_res"]
                if abs(t_i - t_j) > time_th:
                    continue
                if np.linalg.norm(feat["emb"] - feat_j["emb"]) < dist_th:
                    assigned_global[jdx] = inst_counter

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
                    cluster_map: dict) -> dict:
    embs   = np.array([r["emb"] for r in results])
    y_true = np.array([r["gt_class_id"] for r in results])
    y_pred = np.array([cluster_map.get(int(hdb_labels[i]), -1) for i in range(len(results))])

    valid = y_pred != -1
    acc   = float((y_true[valid] == y_pred[valid]).mean()) if valid.sum() > 0 else 0.0

    # Silhouette score
    sil = -1.0
    unique_lbls = set(hdb_labels) - {-1}
    if len(unique_lbls) > 1 and valid.sum() > 1:
        try:
            sil = float(silhouette_score(embs[valid], hdb_labels[valid]))
        except Exception:
            pass

    # Pair-based grouping metrics
    tp = total_gt = total_pred = 0
    scenes = defaultdict(list)
    for r in results:
        scenes[(r["shard_path"], r["scene_idx"])].append(r)
    for pulses in scenes.values():
        n = len(pulses)
        for i in range(n):
            for j in range(i + 1, n):
                gs = pulses[i]["gt_inst_id"]   == pulses[j]["gt_inst_id"]
                ps = pulses[i]["pred_inst_id"]  == pulses[j]["pred_inst_id"]
                if gs: total_gt += 1
                if ps: total_pred += 1
                if gs and ps: tp += 1

    prec = tp / total_pred if total_pred > 0 else 1.0
    rec  = tp / total_gt   if total_gt   > 0 else 1.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

    # Per-class breakdown
    class_stats = {}
    for cid in sorted(set(y_true)):
        mask = y_true == cid
        class_acc = float((y_true[mask] == y_pred[mask]).mean()) if mask.sum() > 0 else 0.0
        class_stats[int(cid)] = {
            "name":  CLASS_NAMES.get(int(cid), f"Class {cid}"),
            "count": int(mask.sum()),
            "accuracy": round(class_acc, 4),
        }

    return {
        "classification_accuracy": acc,
        "silhouette_score":        sil,
        "grouping_precision":      prec,
        "grouping_recall":         rec,
        "grouping_f1":             f1,
        "n_pulses":                len(results),
        "n_clusters_found":        int(n_clusters),
        "n_noise_points":          int((hdb_labels == -1).sum()),
        "n_gt_instances":          len(set((r["shard_path"], r["gt_inst_id"]) for r in results)),
        "n_pred_instances":        len(set(r["pred_inst_id"] for r in results)),
        "per_class_stats":         class_stats,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _tsne_2d(embs: np.ndarray) -> np.ndarray:
    print("[t-SNE] Reducing to 2D (this may take a moment)...")
    perp = min(30, max(5, len(embs) // 10))
    return TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000).fit_transform(embs)


def plot_embeddings(results: list[dict], hdb_labels: np.ndarray,
                    cluster_map: dict, out_dir: str) -> str:
    """
    Two-panel t-SNE scatter:
      Left  — coloured by Ground-Truth Class
      Right — coloured by HDBSCAN Discovered Cluster
    """
    embs = np.array([r["emb"] for r in results])
    xy   = _tsne_2d(embs)

    gt_classes = np.array([r["gt_class_id"] for r in results])
    hdb_arr    = np.array([r["hdb_cluster"]  for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0F0F0F")

    titles      = ["Ground-Truth Class Labels", "HDBSCAN Discovered Clusters"]
    label_arrs  = [gt_classes, hdb_arr]

    for ax, arr, title in zip(axes, label_arrs, titles):
        ax.set_facecolor("#1A1A2E")
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
                  facecolor="#0F0F0F", loc="best", markerscale=2)

    plt.suptitle("exp06 Semi-Supervised DEC — Embedding Space (t-SNE)",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig1_tsne_embedding.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_cluster_composition(results: list[dict], hdb_labels: np.ndarray,
                              cluster_map: dict, out_dir: str) -> str:
    """
    Stacked bar chart: for each HDBSCAN cluster, show which GT classes it captured.
    """
    unique_clusters = sorted(set(hdb_labels) - {-1})
    # Only include class IDs that actually appear in this inference set
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
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#1A1A2E")

    bottom = np.zeros(len(unique_clusters))
    x = np.arange(len(unique_clusters))
    for gi, cls_id in enumerate(unique_class_ids):
        vals = comp[:, gi]
        ax.bar(x, vals, bottom=bottom,
               color=PALETTE[gi % len(PALETTE)],
               label=CLASS_NAMES.get(cls_id, f"Class {cls_id}"),
               width=0.7, alpha=0.9)
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
              facecolor="#0F0F0F", loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig2_cluster_composition.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


def plot_soft_assignment_heatmap(results: list[dict], out_dir: str) -> str:
    """
    Figure 3 (exp06-specific):
    Heatmap of the soft cluster assignment matrix Q (n_samples × n_clusters).
    Rows sorted by GT class so you can visually assess if clusters track classes.
    This is particularly useful for exp06 since the DEC soft assignments are a
    direct output of the semi-supervised cluster layer.
    """
    gt_classes = np.array([r["gt_class_id"] for r in results])
    q_matrix   = np.array([r["soft_q"]       for r in results])

    sort_idx = np.argsort(gt_classes)
    q_sorted = q_matrix[sort_idx]
    gt_sorted = gt_classes[sort_idx]

    fig, ax = plt.subplots(figsize=(min(16, q_matrix.shape[1] * 1.2 + 2), 6))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#0F0F0F")

    im = ax.imshow(q_sorted.T, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Soft Assignment q(i,k)")

    # Mark GT class boundaries
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
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="exp06 Semi-Supervised DEC: Inference, HDBSCAN Clustering, Visualization"
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
                        help="HDBSCAN cluster_selection_epsilon parameter to merge close clusters.")
    parser.add_argument("--reduce_method", type=str, default="tsne", choices=["none", "pca", "tsne"],
                        help="Method to reduce embeddings before HDBSCAN.")
    parser.add_argument("--reduce_dims", type=int, default=2,
                        help="Number of dimensions for dimensionality reduction (default 2 for tsne).")
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

    task = SemiSupervisedDECTask(config)
    load_weights(args.checkpoint_id, task)
    task = task.to(device)

    # 2. Dataset  (no augmentation, all labels available for evaluation)
    sources = parse_sources(args.sources)
    if args.smoke_test:
        for s in sources:
            s["train_shards"] = [s["train_shards"][0]]

    data_cfg = config["data"]
    dataset = DECDataset_Exp06(
        sources       = sources,
        shard_key     = "train_shards",
        max_pulse_len = data_cfg["max_pulse_len"],
        augment       = False,
        label_fraction= 1.0,   # Expose ALL labels for evaluation (post-hoc alignment)
    )
    loader = DataLoader(dataset, batch_size=config["training"].get("batch_size", 128),
                        shuffle=False, num_workers=0)
    if not dataset.index:
        print("[Warning] No pulses found. Check your --source arguments.")
        sys.exit(0)
    print(f"[Data] {len(dataset):,} pulses loaded from {len(sources)} source(s).")

    # 3. Extract embeddings + soft assignments
    features    = extract_features(task, loader, dataset, device)
    hdb_labels  = run_hdbscan(features, min_cluster_size=args.min_cluster_size, 
                              reduce_method=args.reduce_method, reduce_dims=args.reduce_dims,
                              epsilon=args.epsilon)
    cluster_map = align_clusters(features, hdb_labels)
    print(f"[Alignment] Cluster→Class map: { {k: CLASS_NAMES.get(v, str(v)) for k,v in cluster_map.items()} }")

    # 4. PD instance grouping
    results = group_pd_instances(features, hdb_labels,
                                  args.time_threshold, args.dist_threshold)

    # 5. Metrics
    metrics = compute_metrics(results, hdb_labels, cluster_map)
    metrics.update({
        "checkpoint_id":    args.checkpoint_id,
        "time_threshold":   args.time_threshold,
        "dist_threshold":   args.dist_threshold,
        "min_cluster_size": args.min_cluster_size,
        "epsilon":          args.epsilon,
        "reduce_method":    args.reduce_method,
        "reduce_dims":      args.reduce_dims,
        "sources":          args.sources,
    })

    print(f"\n[Results]  CLS Accuracy    : {metrics['classification_accuracy']:.4f}")
    print(f"[Results]  Silhouette      : {metrics['silhouette_score']:.4f}")
    print(f"[Results]  Grouping F1     : {metrics['grouping_f1']:.4f}")
    print(f"[Results]  Clusters found  : {metrics['n_clusters_found']}")
    print(f"[Results]  Noise points    : {metrics['n_noise_points']}")
    print("\n[Results]  Per-class accuracy:")
    for cid, stat in metrics["per_class_stats"].items():
        print(f"             {stat['name']:30s}  n={stat['count']:5d}  acc={stat['accuracy']:.4f}")

    # 6. Output directory
    run_ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_id       = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    out_cfg      = config.get("output", {})
    results_base = os.path.abspath(out_cfg.get("results_dir", "data/classification_output/exp06_dec"))
    out_dir      = os.path.join(results_base, f"{run_ts}_inf-{args.checkpoint_id}-{inf_id}")
    os.makedirs(out_dir, exist_ok=True)

    # 7. Save metrics JSON
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 7.5 Save predictions to HDF5
    import h5py
    h5_path = os.path.join(out_dir, "predictions.h5")
    with h5py.File(h5_path, "w") as f:
        # Standardise keys
        for r in results:
            r["cluster_id"] = r.get("hdb_cluster", -1)
            r["pred_class_id"] = cluster_map.get(r["cluster_id"], -1)
            
        keys_to_save = ["shard_path", "scene_idx", "ch_idx", "start_idx", 
                        "gt_class_id", "pred_class_id", "cluster_id", 
                        "gt_inst_id", "pred_inst_id", "time_res"]
        for k in keys_to_save:
            values = [r.get(k, -1) for r in results]
            if not values:
                continue
            if isinstance(values[0], str):
                f.create_dataset(k, data=np.array(values, dtype=object), dtype=h5py.string_dtype(encoding='utf-8'))
            else:
                f.create_dataset(k, data=np.array(values))
    print(f"[Export] Saved pulse mappings -> {h5_path}")

    # 8. Plots
    plot_embeddings(results, hdb_labels, cluster_map, out_dir)
    plot_cluster_composition(results, hdb_labels, cluster_map, out_dir)
    plot_soft_assignment_heatmap(results, out_dir)

    # 9. Lineage registration
    history = (
        f"Semi-Supervised DEC Inference (exp06) | Checkpoint: {args.checkpoint_id} | "
        f"Sources: {args.sources} | "
        f"Clusters: {metrics['n_clusters_found']} | "
        f"CLS Acc: {metrics['classification_accuracy']:.4f} | "
        f"Silhouette: {metrics['silhouette_score']:.4f} | "
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
