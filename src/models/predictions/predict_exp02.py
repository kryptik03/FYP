"""
predict_exp02.py
==========
Inference + Evaluation script for the CNN 1D Classification model (exp02).

Usage
-----
    python src/models/predictions/predict_exp02.py \
        --checkpoint <NodeID> \
        --shards 17 18 19 20

Evaluation
----------
Evaluates the full-resolution classification model on isolated PD windows.
Metrics computed:
  Classification Accuracy (Overall)
  Per-class (PD1/PD2) Precision, Recall, and F1 Score

All metrics are printed to the console and saved to metrics.json in the
output folder.

Output files
------------
  predicted_classes.h5  shape (7, N)  rows: Scene,Ch,DetID,ClassID,ClsScore,Start,End
  metrics.json          evaluation report
  analysis_history.txt  one-line human summary
"""

import argparse
import json
import os
import random
import string
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.data.dataset_exp02 import ClassificationDataset
from src.models.tasks.task_exp02   import ClassificationTask
from src.utils.lineage_tracker     import register_process


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_config_for_checkpoint(checkpoint_id: str) -> tuple:
    """Return (config dict, config path) for a given NodeID."""
    config_dir  = os.path.join(_PROJECT_ROOT, "models/configuration_snapshots")
    config_path = os.path.abspath(os.path.join(config_dir, f"config_{checkpoint_id}.yaml"))
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[Error] Config snapshot not found: {config_path}\n"
            f"  Did you download it from Colab?"
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[Checkpoint] Config  : {config_path}")
    return config, config_path


def load_weights(checkpoint_id: str, task: ClassificationTask) -> int:
    """Load model weights in-place. Returns the epoch the model was saved at."""
    weights_dir  = os.path.join(_PROJECT_ROOT, "models/weights")
    weights_path = os.path.abspath(os.path.join(weights_dir, f"model_{checkpoint_id}.pt"))
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"[Error] Weights not found: {weights_path}\n"
            f"  Did you download the .pt file from Colab?"
        )
    state = torch.load(weights_path, map_location="cpu")
    task.load_state_dict(state["model_state"])
    epoch = state.get("epoch", "?")
    print(f"[Checkpoint] Weights : {weights_path}  (trained to epoch {epoch})")
    return epoch


# ---------------------------------------------------------------------------
# Inference + ground-truth collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    task:      ClassificationTask,
    loader:    DataLoader,
    dataset:   ClassificationDataset,
    device:    torch.device,
) -> list:
    """
    Run the model over all batches and simultaneously record ground-truth labels.

    Returns
    -------
    results : list[dict]
        dict keys: scene_id, channel_id, start_raw, end_raw,
                   pred_class, pred_score, gt_class
    """
    task.eval()
    results = []
    sample_idx = 0

    for signals, labels in loader:
        signals = signals.to(device)
        logits  = task.forward(signals)                          # (B, num_classes)
        probs   = F.softmax(logits, dim=1)                       # (B, num_classes)
        
        preds_cls   = torch.argmax(probs, dim=1).cpu().numpy()
        preds_score = torch.max(probs, dim=1)[0].cpu().numpy()
        labels      = labels.cpu().numpy()

        B = signals.shape[0]
        for b in range(B):
            flat_idx = sample_idx + b
            shard_path, scene_idx, ch_idx, start_idx, end_idx, gt_class_id = dataset.index[flat_idx]

            results.append({
                "scene_id"  : scene_idx,
                "channel_id": ch_idx,
                "start_raw" : start_idx,
                "end_raw"   : end_idx,
                "pred_class": int(preds_cls[b]),
                "pred_score": float(preds_score[b]),
                "gt_class"  : int(labels[b])
            })
            
        sample_idx += B

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(results: list) -> dict:
    """
    Compute accuracy and per-class precision/recall from inference results.
    """
    n_total = len(results)
    if n_total == 0:
        return {}
    
    correct = sum(1 for r in results if r["pred_class"] == r["gt_class"])
    accuracy = correct / n_total

    classes = [0, 1]
    per_class = {}
    for c in classes:
        tp = sum(1 for r in results if r["pred_class"] == c and r["gt_class"] == c)
        fp = sum(1 for r in results if r["pred_class"] == c and r["gt_class"] != c)
        fn = sum(1 for r in results if r["pred_class"] != c and r["gt_class"] == c)
        tn = sum(1 for r in results if r["pred_class"] != c and r["gt_class"] != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[f"PD{c+1}"] = {
            "n_gt": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
        
    return {
        "n_samples_evaluated": n_total,
        "accuracy": round(accuracy, 4),
        "per_class": per_class
    }


def print_and_save_metrics(metrics: dict, output_dir: str):
    """Print a formatted summary and save metrics.json."""
    sep = "=" * 56
    print("\n" + sep)
    print("  EVALUATION REPORT")
    print(sep)
    print(f"  Total samples     : {metrics.get('n_samples_evaluated', 0)}")
    print(f"  Overall Accuracy  : {metrics.get('accuracy', 0.0):.4f}")
    print("-" * 56)
    
    if "per_class" in metrics:
        for cls_label, cm in metrics["per_class"].items():
            print(f"  {cls_label}: {cm['tp']}/{cm['n_gt']} correct (TPs) | "
                  f"Precision={cm['precision']:.4f} | Recall={cm['recall']:.4f} | F1={cm['f1']:.4f}")
    
    print("-" * 56)
    if "inference_time_s" in metrics:
        print(f"  Inference time    : {metrics['inference_time_s']:.2f}s "
              f"({metrics.get('samples_per_second', 0):.1f} samples/sec)")
    print(sep)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved  -> {metrics_path}\n")


# ---------------------------------------------------------------------------
# HDF5 output writers
# ---------------------------------------------------------------------------

def save_predicted_classes(results: list, output_dir: str, checkpoint_id: str) -> str:
    path = os.path.join(output_dir, "predicted_classes.h5")
    N    = len(results)
    mat  = np.zeros((7, N), dtype=np.float64)
    for i, d in enumerate(results):
        mat[0, i] = d["scene_id"]
        mat[1, i] = d["channel_id"]
        mat[2, i] = i                  # Det_ID
        mat[3, i] = d["pred_class"]
        mat[4, i] = d["pred_score"]
        mat[5, i] = d["start_raw"]
        mat[6, i] = d["end_raw"]
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("predictions", data=mat)
        ds.attrs["row_0"] = "Scene_ID"
        ds.attrs["row_1"] = "Channel_ID"
        ds.attrs["row_2"] = "Det_ID"
        ds.attrs["row_3"] = "Class_ID (0=PD1, 1=PD2)"
        ds.attrs["row_4"] = "Cls_Score"
        ds.attrs["row_5"] = "Start_Idx (0-indexed raw sample)"
        ds.attrs["row_6"] = "End_Idx   (0-indexed raw sample)"
        f.attrs["checkpoint_id"] = checkpoint_id
        f.attrs["n_predictions"] = N
    print(f"[Output] {N} classes -> {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FYP PD Pipeline - CNN1D Classification Inference + Evaluation"
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="NodeID of the trained model (e.g. QIRE).")
    parser.add_argument("--input_path",    default=None,
                        help="Dataset folder. Defaults to root_path in config.")
    parser.add_argument("--shards",        nargs="+", type=int, default=None,
                        help="Shard numbers to run on. Defaults to all 20.")
    parser.add_argument("--batch_size",    type=int,   default=64,
                        help="Inference batch size (default 64).")
    args = parser.parse_args()

    checkpoint_id = args.checkpoint

    # ----------------------------------------------------------------------- #
    # 1. Config + model                                                         #
    # ----------------------------------------------------------------------- #
    config, _ = load_config_for_checkpoint(checkpoint_id)
    data_cfg  = config["data"]
    train_cfg = config["training"]

    device = select_device(train_cfg["device"])
    print(f"[Device] {device}")

    task = ClassificationTask(config)
    load_weights(checkpoint_id, task)
    task = task.to(device)
    task.eval()

    # ----------------------------------------------------------------------- #
    # 2. Dataset + loader                                                       #
    # ----------------------------------------------------------------------- #
    # If using project root path resolution instead of relative, we need to handle it.
    default_root = data_cfg["root_path"]
    if not os.path.isabs(default_root):
        default_root = os.path.join(_PROJECT_ROOT, default_root)
        
    root_path = os.path.abspath(args.input_path or default_root)
    shards    = args.shards or list(range(1, 21))

    print(f"[Data] Path  : {root_path}")
    print(f"[Data] Shards: {shards}")

    dataset = ClassificationDataset(
        root_path     = root_path,
        shard_ids     = shards,
        max_pulse_len = data_cfg.get("max_pulse_len", 4096),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    print(f"[Data] Total samples: {len(dataset)}")

    # ----------------------------------------------------------------------- #
    # 3. Inference                                                              #
    # ----------------------------------------------------------------------- #
    print(f"\n[Inference] Running classification...")
    infer_start = time.time()
    results = run_inference(task, loader, dataset, device)
    infer_elapsed = time.time() - infer_start
    
    samples_per_sec = len(dataset) / infer_elapsed if infer_elapsed > 0 else 0
    print(f"[Inference] {len(results)} predictions made")
    print(f"[Inference] Time: {infer_elapsed:.2f}s ({samples_per_sec:.1f} samples/sec)")

    # ----------------------------------------------------------------------- #
    # 4. Evaluation                                                             #
    # ----------------------------------------------------------------------- #
    print(f"\n[Evaluation] Computing metrics...")
    metrics = evaluate(results)
    
    # Attach timing info
    metrics["inference_time_s"]    = round(infer_elapsed, 3)
    metrics["samples_per_second"]  = round(samples_per_sec, 2)

    # ----------------------------------------------------------------------- #
    # 5. Output folder                                                          #
    # ----------------------------------------------------------------------- #
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_node_id = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    exp_cfg     = config.get("experiment", {})
    root_id     = exp_cfg.get("parent_node_id", "UNKN")   # e.g. "ShmH"
    origin      = exp_cfg.get("origin", "sy")              # "sy" or "ms"
    method      = exp_cfg.get("name", "unknown")           # e.g. "exp02_cnn_cls"
    
    folder_name = f"{run_ts}_{origin}-{root_id}-{inf_node_id}"
    out_cfg     = config.get("output", {})
    
    # Make sure output_dir is an absolute path from project root
    results_dir = out_cfg.get("results_dir", "data/classification_output")
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(_PROJECT_ROOT, results_dir)
        
    output_dir  = os.path.abspath(os.path.join(results_dir, method, folder_name))
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Output] Folder: {output_dir}")

    save_predicted_classes(results, output_dir, checkpoint_id)
    print_and_save_metrics(metrics, output_dir)

    history_line = (
        f"Inference+Eval via CNN1D Cls ({method}) at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
        f"Checkpoint: {checkpoint_id}, InfNodeID: {inf_node_id}, "
        f"Shards: {shards}, "
        f"N_samples: {len(results)}, Accuracy: {metrics.get('accuracy', 0.0):.4f}"
    )
    with open(os.path.join(output_dir, "analysis_history.txt"), "w") as f:
        f.write(history_line + "\n")

    # ----------------------------------------------------------------------- #
    # 6. Lineage registration                                                   #
    # ----------------------------------------------------------------------- #
    print("[Lineage] Registering ...")
    new_node = register_process(
        parent_id        = checkpoint_id,
        stage            = "classification",
        method           = method,
        folder_path      = output_dir,
        appended_history = history_line,
        force_node_id    = inf_node_id,
    )
    print(f"[Lineage] Node {new_node} registered (child of {checkpoint_id})")


if __name__ == "__main__":
    main()
