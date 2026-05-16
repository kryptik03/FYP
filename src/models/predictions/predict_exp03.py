"""
predict_exp03.py
================
Inference script for the Contrastive Embedding model (exp03).

This script performs two main steps:
1. Feature Extraction: Passes every isolated pulse through the model to get a 128-d embedding and classification logits.
2. Grouping/Clustering: For every scene, candidate pulses within a time threshold (default 100ns) are compared via Euclidean distance. If the distance is below the threshold, they are grouped into the same Predicted Pulse Instance.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Handle imports dynamically based on script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import networkx as nx

from src.models.data.dataset_exp02 import ClassificationDataset
from src.models.tasks.task_exp03_contrastive import ContrastiveTask
from src.utils.lineage_tracker import register_process


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[Error] Config not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[Checkpoint] Config  : {config_path}")
    return config


def load_weights(checkpoint_id: str, task: ContrastiveTask) -> int:
    weights_dir = "models/weights"
    weights_path = os.path.abspath(os.path.join(weights_dir, f"model_{checkpoint_id}.pt"))
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[Error] Weights not found: {weights_path}")
    
    state = torch.load(weights_path, map_location="cpu")
    task.load_state_dict(state["model_state"])
    epoch = state.get("epoch", "?")
    print(f"[Checkpoint] Weights : {weights_path}  (trained to epoch {epoch})")
    return epoch


@torch.no_grad()
def extract_features(
    task: ContrastiveTask,
    loader: DataLoader,
    dataset: ClassificationDataset,
    device: torch.device
) -> list[dict]:
    """Passes all pulses through the model and extracts embeddings + logits."""
    task.eval()
    features = []
    sample_idx = 0

    print("[Inference] Extracting features for all pulses...")
    for signals, _ in loader:
        signals = signals.to(device)
        
        # Forward pass through ContrastiveTask
        emb, logits = task.forward(signals)
        
        emb = emb.cpu().numpy()
        logits = logits.cpu().numpy()

        B = signals.shape[0]
        for b in range(B):
            flat_idx = sample_idx + b
            shard_path, scene_idx, ch_idx, start_idx, end_idx, class_id = dataset.index[flat_idx]

            features.append({
                "shard_path": shard_path,
                "scene_idx": scene_idx,
                "ch_idx": ch_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "gt_class_id": class_id,
                "emb": emb[b],
                "logits": logits[b]
            })
        sample_idx += B

    print(f"[Inference] Extracted features for {len(features)} pulses.")
    return features


def group_and_classify_pulses(
    features: list[dict],
    time_threshold: float,
    dist_threshold: float,
    time_resolution_s: float
) -> list[dict]:
    """
    Groups pulses using a time-domain heuristic followed by an embedding distance filter.
    Returns a list of prediction dictionaries.
    """
    print(f"[Clustering] Grouping pulses (Time TH: {time_threshold*1e9:.1f}ns, Dist TH: {dist_threshold})...")
    
    # Group by (shard_path, scene_idx) because instances cannot cross scenes
    scenes = defaultdict(list)
    for f in features:
        scenes[(f["shard_path"], f["scene_idx"])].append(f)

    grouped_results = []
    instance_id_counter = 1

    for (shard_path, scene_idx), pulses in scenes.items():
        # Build an adjacency graph for the pulses in this scene
        G = nx.Graph()
        n_pulses = len(pulses)
        for i in range(n_pulses):
            G.add_node(i)

        for i in range(n_pulses):
            for j in range(i + 1, n_pulses):
                # 1. Time Filter (Heuristic)
                time_diff_s = abs(pulses[i]["start_idx"] - pulses[j]["start_idx"]) * time_resolution_s
                if time_diff_s <= time_threshold:
                    # 2. Embedding Filter (Contrastive Distance)
                    dist = np.linalg.norm(pulses[i]["emb"] - pulses[j]["emb"])
                    if dist <= dist_threshold:
                        G.add_edge(i, j)

        # Find connected components (each component is a predicted PD instance)
        components = list(nx.connected_components(G))

        for comp in components:
            comp_pulses = [pulses[idx] for idx in comp]
            
            # Aggregate classification across the grouped instance
            avg_logits = np.mean([p["logits"] for p in comp_pulses], axis=0)
            pred_class = int(np.argmax(avg_logits))
            
            for p in comp_pulses:
                grouped_results.append({
                    "shard_path": p["shard_path"],
                    "scene_idx": p["scene_idx"],
                    "ch_idx": p["ch_idx"],
                    "start_idx": p["start_idx"],
                    "end_idx": p["end_idx"],
                    "gt_class_id": p["gt_class_id"],
                    "pred_inst_id": instance_id_counter,
                    "pred_class_id": pred_class,
                    "avg_logits": avg_logits.tolist(),
                    "emb": p["emb"].tolist()
                })
            
            instance_id_counter += 1

    print(f"[Clustering] Formed {instance_id_counter - 1} unique instances.")
    return grouped_results


def save_results(
    grouped_results: list[dict],
    out_dir: str,
    node_id: str,
    args: argparse.Namespace
):
    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, f"results_{node_id}.h5")

    # Serialize results to HDF5
    with h5py.File(out_h5, "w") as f:
        # Save metadata
        f.attrs["checkpoint_id"]  = args.checkpoint_id
        f.attrs["input_path"]     = args.input_path
        f.attrs["time_threshold"] = args.time_threshold
        f.attrs["dist_threshold"] = args.dist_threshold
        f.attrs["creation_date"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create structured array for the tabular results
        dt = np.dtype([
            ("scene_idx", np.int32),
            ("ch_idx", np.int32),
            ("start_idx", np.int32),
            ("end_idx", np.int32),
            ("gt_class_id", np.int32),
            ("pred_inst_id", np.int32),
            ("pred_class_id", np.int32)
        ])
        
        arr = np.empty(len(grouped_results), dtype=dt)
        for i, res in enumerate(grouped_results):
            arr[i] = (
                res["scene_idx"], res["ch_idx"], res["start_idx"], res["end_idx"],
                res["gt_class_id"], res["pred_inst_id"], res["pred_class_id"]
            )
            
        f.create_dataset("predictions", data=arr, compression="gzip")
        
    print(f"[Output] Saved clustering results to {out_h5}")
    return out_h5


def main():
    parser = argparse.ArgumentParser(description="Exp03: Inference + Pulse Grouping")
    parser.add_argument("--checkpoint_id", required=True, help="NodeID of the trained exp03 model")
    parser.add_argument("--input_path", required=True, help="Path to shard directory (measured or synthesized)")
    parser.add_argument("--time_threshold", type=float, default=100e-9, help="Max time delta (s) for grouping")
    parser.add_argument("--dist_threshold", type=float, default=0.5, help="Max L2 distance for grouping")
    parser.add_argument("--smoke_test", action="store_true", help="Run on a tiny subset for testing")
    args = parser.parse_args()

    # 1. Setup paths & config
    config_path = os.path.abspath(f"models/configuration_snapshots/config_{args.checkpoint_id}.yaml")
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Build task and load weights
    task = ContrastiveTask(config).to(device)
    epoch_trained = load_weights(args.checkpoint_id, task)

    # 3. Load dataset
    data_cfg = config["data"]
    shard_ids = data_cfg["val_shards"]
    if args.smoke_test:
        shard_ids = [shard_ids[0]]
        print(f"[Smoke Test] Overriding val_shards to {shard_ids}")

    dataset = ClassificationDataset(
        root_path=os.path.abspath(args.input_path),
        shard_ids=shard_ids,
        max_pulse_len=data_cfg["max_pulse_len"]
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=0
    )

    # 4. Extract embeddings
    features = extract_features(task, loader, dataset, device)

    if not features:
        print("[Warning] No pulses found in the dataset.")
        sys.exit(0)

    # 5. Group pulses
    time_res = getattr(dataset, "time_resolution_s", 1e-11)
    grouped_results = group_and_classify_pulses(
        features, 
        time_threshold=args.time_threshold, 
        dist_threshold=args.dist_threshold, 
        time_resolution_s=time_res
    )

    # 6. Save results
    node_id = f"pred_{args.checkpoint_id}"
    out_dir = os.path.abspath(config["output"]["results_dir"])
    out_h5  = save_results(grouped_results, out_dir, node_id, args)

    # 7. Lineage registration
    print("\n[Lineage] Registering prediction run to SQLite DAG...")
    history_line = (
        f"Contrastive Grouping Inference on dataset {args.input_path} "
        f"with model {args.checkpoint_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"Time TH: {args.time_threshold}, Dist TH: {args.dist_threshold}."
    )
    
    # We use args.checkpoint_id as parent because this data inherits from the model
    # (or you could argue it inherits from the dataset, but usually prediction is child of model)
    register_process(
        parent_id        = args.checkpoint_id,
        stage            = "prediction",
        method           = "contrastive_inference",
        folder_path      = out_h5,
        appended_history = history_line,
        force_node_id    = node_id
    )
    print(f"[Lineage] Registered prediction as Node {node_id}")

if __name__ == "__main__":
    main()
