"""
predict_exp04.py
================
Inference script for exp04 Deep Embedded Clustering (DEC).

What this script does:
1. Loads the trained DEC checkpoint (backbone + cluster centroids).
2. Extracts 128-D embeddings for every pulse in the dataset.
3. Runs HDBSCAN to discover the number of clusters automatically (no predefined K).
4. Performs post-hoc label alignment: maps HDBSCAN cluster IDs to real PD class names
   by majority voting against the ground-truth class_id in the dataset.
5. Groups pulses into PD instances (same physical discharge event) using:
   a) Temporal proximity: pulses within `time_threshold` of each other.
   b) Embedding proximity: L2 distance < `dist_threshold` in embedding space.
6. Saves results to the standardized output folder and registers in lineage.db.

Usage:
    python src/models/predictions/predict_exp04.py \\
        --checkpoint_id <NodeID> \\
        --input_sources "data/raw/measured/FOLDER:measured:1,2,3" \\
        [--time_threshold 100e-9] \\
        [--dist_threshold 0.5] \\
        [--min_cluster_size 5] \\
        [--smoke_test]

Note: --input_sources accepts semicolon-separated sources in the format:
    path:type:shard1,shard2,...
    Example: "data/raw/measured/FOLDER:measured:1,2,3;data/raw/synth/FOLDER:synthesized:17,18,19"
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
from torch.utils.data import DataLoader

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.tasks.task_exp04_dec import DECTask
from src.models.data.dataset_exp04_dec import DECDataset
from src.utils.lineage_tracker import register_process


# ---------------------------------------------------------------------------
# Checkpoint Loading
# ---------------------------------------------------------------------------

def load_config_for_checkpoint(checkpoint_id: str) -> dict:
    import yaml
    config_dir = os.path.abspath("models/configuration_snapshots")
    config_path = os.path.join(config_dir, f"config_{checkpoint_id}.yaml")
    if not os.path.exists(config_path):
        print(f"[Error] Config snapshot not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"[Checkpoint] Config  : {config_path}")
    return config


def load_weights(checkpoint_id: str, task: DECTask):
    weights_dir = os.path.abspath("models/weights")
    weight_path = os.path.join(weights_dir, f"model_{checkpoint_id}.pt")
    if not os.path.exists(weight_path):
        print(f"[Error] Weights not found: {weight_path}")
        sys.exit(1)
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    task.load_state_dict(ckpt["model_state"])
    print(f"[Checkpoint] Weights : {weight_path}")
    print(f"[Checkpoint] Epoch   : {ckpt.get('epoch', '?')}")


def select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(task: DECTask, loader: DataLoader, dataset: DECDataset,
                     device: torch.device) -> list[dict]:
    """Extract embeddings and soft cluster assignments for all pulses."""
    task.eval()
    features = []
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            signal = batch[0].to(device)
            z, q = task(signal)
            z = z.cpu().numpy()
            q = q.cpu().numpy()
            B = signal.shape[0]

            for b in range(B):
                flat_idx = sample_idx + b
                shard_path, scene_idx, ch_idx, start_idx, end_idx, class_id, inst_id, time_res = dataset.index[flat_idx]
                features.append({
                    "shard_path": shard_path,
                    "scene_idx":  scene_idx,
                    "ch_idx":     ch_idx,
                    "start_idx":  start_idx,
                    "end_idx":    end_idx,
                    "gt_class_id": class_id,
                    "gt_inst_id":  inst_id,
                    "time_res":   time_res,
                    "emb":        z[b],
                    "soft_q":     q[b],
                })
            sample_idx += B

    print(f"[Inference] Extracted features for {len(features)} pulses.")
    return features


# ---------------------------------------------------------------------------
# HDBSCAN Clustering
# ---------------------------------------------------------------------------

def run_hdbscan(features: list[dict], min_cluster_size: int = 5) -> np.ndarray:
    """Run HDBSCAN on embeddings. Returns cluster labels (-1 = noise)."""
    try:
        import hdbscan
    except ImportError:
        print("[Error] hdbscan not installed. Run: pip install hdbscan")
        sys.exit(1)

    embs = np.array([f["emb"] for f in features])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embs)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"[HDBSCAN] Found {n_clusters} clusters | Noise points: {n_noise}")
    return labels


# ---------------------------------------------------------------------------
# Post-Hoc Label Alignment
# ---------------------------------------------------------------------------

def align_clusters_to_classes(features: list[dict], hdb_labels: np.ndarray) -> dict[int, int]:
    """
    Map each HDBSCAN cluster ID to the most common ground-truth class_id
    in that cluster (majority voting). Labels are never used during training —
    this is a post-hoc evaluation and naming step only.
    Returns: {hdb_cluster_id -> gt_class_id}
    """
    cluster_votes = defaultdict(list)
    for feat, label in zip(features, hdb_labels):
        if label != -1:
            cluster_votes[label].append(feat["gt_class_id"])

    mapping = {}
    for cluster_id, votes in cluster_votes.items():
        # Majority vote
        mapping[cluster_id] = max(set(votes), key=votes.count)
    mapping[-1] = -1   # Noise points stay as -1
    return mapping


# ---------------------------------------------------------------------------
# PD Instance Grouping (same as exp03 logic)
# ---------------------------------------------------------------------------

def group_into_pd_instances(features: list[dict], hdb_labels: np.ndarray,
                             time_threshold: float, dist_threshold: float) -> list[dict]:
    """
    Two-step PD instance grouping:
    1. Temporal filter: only compare pulses within `time_threshold` of each other.
    2. Embedding filter: group if L2 distance in embedding space < `dist_threshold`.
    """
    results = []
    scene_groups = defaultdict(list)
    for i, feat in enumerate(features):
        key = (feat["shard_path"], feat["scene_idx"])
        scene_groups[key].append((i, feat, hdb_labels[i]))

    instance_id_counter = 0
    for scene_pulses in scene_groups.values():
        assigned = {}

        for i, (idx, feat, _) in enumerate(scene_pulses):
            if idx in assigned:
                continue
            instance_id_counter += 1
            group = [idx]
            assigned[idx] = instance_id_counter
            t_i = feat["start_idx"] * feat["time_res"]

            for j, (jdx, feat_j, _) in enumerate(scene_pulses):
                if i == j or jdx in assigned:
                    continue
                t_j = feat_j["start_idx"] * feat_j["time_res"]
                if abs(t_i - t_j) > time_threshold:
                    continue
                d = np.linalg.norm(feat["emb"] - feat_j["emb"])
                if d < dist_threshold:
                    group.append(jdx)
                    assigned[jdx] = instance_id_counter

        for idx, inst_id in assigned.items():
            feat = features[idx]
            results.append({**feat, "pred_inst_id": inst_id})

    return results


# ---------------------------------------------------------------------------
# Output Saving
# ---------------------------------------------------------------------------

def save_results(results: list[dict], hdb_labels: np.ndarray,
                 cluster_map: dict, out_dir: str, node_id: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, f"results_{node_id}.h5")

    dt = np.dtype([
        ("scene_idx",    np.int32),
        ("ch_idx",       np.int32),
        ("start_idx",    np.int32),
        ("end_idx",      np.int32),
        ("gt_class_id",  np.int32),
        ("gt_inst_id",   np.int32),
        ("hdb_cluster",  np.int32),
        ("pred_class_id",np.int32),
        ("pred_inst_id", np.int32),
    ])

    arr = np.empty(len(results), dtype=dt)
    for i, r in enumerate(results):
        hdb_lbl = hdb_labels[i] if i < len(hdb_labels) else -1
        pred_cls = cluster_map.get(hdb_lbl, -1)
        arr[i] = (
            r["scene_idx"], r["ch_idx"], r["start_idx"], r["end_idx"],
            r["gt_class_id"], r["gt_inst_id"],
            hdb_lbl, pred_cls, r["pred_inst_id"],
        )

    with h5py.File(out_h5, "w") as f:
        ds = f.create_dataset("predictions", data=arr, compression="gzip")
        ds.attrs["description"] = "HDBSCAN cluster IDs + post-hoc class alignment + PD instance grouping"

    print(f"[Output] Saved predictions to {out_h5}")
    return out_h5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict], hdb_labels: np.ndarray,
                    cluster_map: dict) -> dict:
    y_true = np.array([r["gt_class_id"] for r in results])
    y_pred = np.array([cluster_map.get(hdb_labels[i], -1) for i in range(len(results))])

    # Exclude noise points (-1) from classification accuracy
    valid = y_pred != -1
    acc = (y_true[valid] == y_pred[valid]).mean() if valid.sum() > 0 else 0.0

    # Grouping metrics (pair-based)
    tp_pairs = total_gt_pairs = total_pred_pairs = 0
    eval_scenes = defaultdict(list)
    for r in results:
        eval_scenes[(r["shard_path"], r["scene_idx"])].append(r)

    for pulses in eval_scenes.values():
        n = len(pulses)
        for i in range(n):
            for j in range(i + 1, n):
                gt_same   = (pulses[i]["gt_inst_id"]   == pulses[j]["gt_inst_id"])
                pred_same = (pulses[i]["pred_inst_id"]  == pulses[j]["pred_inst_id"])
                if gt_same:   total_gt_pairs += 1
                if pred_same: total_pred_pairs += 1
                if gt_same and pred_same: tp_pairs += 1

    group_recall    = tp_pairs / total_gt_pairs   if total_gt_pairs   > 0 else 1.0
    group_precision = tp_pairs / total_pred_pairs if total_pred_pairs > 0 else 1.0
    group_f1 = (2 * group_precision * group_recall /
                (group_precision + group_recall)) if (group_precision + group_recall) > 0 else 0.0

    n_clusters_found = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_noise = (hdb_labels == -1).sum()

    return {
        "classification_accuracy": float(acc),
        "grouping_recall":         float(group_recall),
        "grouping_precision":      float(group_precision),
        "grouping_f1":             float(group_f1),
        "n_pulses":                len(results),
        "n_clusters_found":        int(n_clusters_found),
        "n_noise_points":          int(n_noise),
        "n_gt_instances":          len(set((r["shard_path"], r["gt_inst_id"]) for r in results)),
        "n_pred_instances":        len(set(r["pred_inst_id"] for r in results)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_input_sources(raw: str) -> list[dict]:
    """
    Parse semicolon-separated sources of the form:
        path:type:shard1,shard2,...
    """
    sources = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        if len(tokens) < 3:
            print(f"[Error] Bad source spec: '{part}'. Expected path:type:shards")
            sys.exit(1)
        path     = tokens[0]
        src_type = tokens[1]
        shards   = [int(x) for x in tokens[2].split(",") if x.strip()]
        sources.append({"type": src_type, "path": path,
                        "train_shards": shards, "val_shards": shards})
    return sources


def main():
    parser = argparse.ArgumentParser(description="Exp04 DEC: Feature Extraction + HDBSCAN Clustering")
    parser.add_argument("--checkpoint_id",  required=True, help="NodeID of the trained DEC model")
    parser.add_argument("--input_sources",  required=True,
                        help="Semicolon-separated: path:type:shards. E.g. data/raw/measured/FOLDER:measured:1,2,3")
    parser.add_argument("--time_threshold", type=float, default=100e-9,
                        help="Max time delta (s) for PD instance grouping (default: 100ns)")
    parser.add_argument("--dist_threshold", type=float, default=0.5,
                        help="Max L2 embedding distance for PD instance grouping (default: 0.5)")
    parser.add_argument("--min_cluster_size", type=int, default=5,
                        help="HDBSCAN min_cluster_size (default: 5)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on first shard of first source only")
    args = parser.parse_args()

    # 1. Config + model
    config = load_config_for_checkpoint(args.checkpoint_id)
    train_cfg = config["training"]
    device = select_device(train_cfg.get("device", "auto"))
    print(f"[Device] Using: {device}")

    task = DECTask(config)
    load_weights(args.checkpoint_id, task)
    task = task.to(device)
    task.eval()

    # 2. Build dataset from CLI-specified sources
    sources = parse_input_sources(args.input_sources)
    if args.smoke_test:
        for s in sources:
            s["train_shards"] = [s["train_shards"][0]]
            s["val_shards"]   = [s["val_shards"][0]]
        print("[Smoke Test] Using first shard of each source.")

    dataset = DECDataset(
        sources       = sources,
        shard_key     = "train_shards",   # Uses the shards specified in --input_sources
        max_pulse_len = config["data"]["max_pulse_len"],
        augment       = False,
    )
    loader = DataLoader(dataset, batch_size=train_cfg.get("batch_size", 128),
                        shuffle=False, num_workers=0)

    if not dataset.index:
        print("[Warning] No pulses found. Check --input_sources paths and shard IDs.")
        sys.exit(0)

    # 3. Extract embeddings
    print("[Inference] Extracting features for all pulses...")
    features = extract_features(task, loader, dataset, device)

    # 4. HDBSCAN clustering (no predefined K)
    hdb_labels = run_hdbscan(features, min_cluster_size=args.min_cluster_size)

    # 5. Post-hoc label alignment (labels only used for naming, not training)
    cluster_map = align_clusters_to_classes(features, hdb_labels)
    print(f"[Alignment] Cluster -> Class mapping: {cluster_map}")

    # 6. PD instance grouping
    results = group_into_pd_instances(features, hdb_labels,
                                      args.time_threshold, args.dist_threshold)

    # 7. Compute metrics
    metrics = compute_metrics(results, hdb_labels, cluster_map)
    print(f"[Metrics] CLS Accuracy: {metrics['classification_accuracy']:.4f} | "
          f"Grouping F1: {metrics['grouping_f1']:.4f} | "
          f"Clusters Found: {metrics['n_clusters_found']}")

    # 8. Save output
    run_ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_node_id  = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    origin       = getattr(dataset, "origin", "ms")
    root_id      = getattr(dataset, "root_id", "UNKN")
    method       = config["experiment"].get("name", "exp04_dec")
    folder_name  = f"{run_ts}_{origin}-{root_id}-{inf_node_id}"

    out_dir = os.path.abspath(os.path.join(config["output"]["results_dir"], method, folder_name))
    save_results(results, hdb_labels, cluster_map, out_dir, inf_node_id)

    metrics["time_threshold"]  = args.time_threshold
    metrics["dist_threshold"]  = args.dist_threshold
    metrics["min_cluster_size"] = args.min_cluster_size
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    history_line = (
        f"DEC Inference on [{args.input_sources}] with model {args.checkpoint_id} "
        f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"Clusters found: {metrics['n_clusters_found']}, "
        f"ClsAcc: {metrics['classification_accuracy']:.4f}, "
        f"GroupF1: {metrics['grouping_f1']:.4f}."
    )
    with open(os.path.join(out_dir, "analysis_history.txt"), "w") as f:
        f.write(history_line + "\n")

    print("\n[Lineage] Registering prediction run...")
    register_process(
        parent_id        = args.checkpoint_id,
        stage            = "prediction",
        method           = "dec_hdbscan",
        folder_path      = out_dir,
        appended_history = history_line,
        force_node_id    = inf_node_id,
    )
    print(f"[Lineage] Registered as Node {inf_node_id} (child of {args.checkpoint_id})")
    print(f"\n[Done] Results -> {out_dir}")


if __name__ == "__main__":
    main()
