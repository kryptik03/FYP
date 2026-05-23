"""
predict_exp01.py
==========
Inference + Evaluation script for the YOLO1D PD detection model.

Usage
-----
    python src/models/predict_exp01.py \\
        --checkpoint QIRE \\
        --shards 17 18 19 20 \\
        --threshold 0.5 \\
        --iou_threshold 0.5

Evaluation
----------
Predictions are automatically matched against ground-truth labels from the
HDF5 shards using greedy 1D IoU matching. Metrics computed:

  Detection (class-agnostic):
    Precision, Recall, F1

  Localisation (matched TPs only):
    Mean IoU, Mean start/end index error (raw samples)

  Classification (matched TPs only):
    Overall class accuracy
    Per-class (PD1/PD2) Precision and Recall

All metrics are printed to the console and saved to metrics.json in the
output folder.

Output files
------------
  predicted_boxes.h5    shape (6, N_det)  rows: Scene,Ch,DetID,ObjScore,Start,End
  predicted_classes.h5  shape (5, N_det)  rows: Scene,Ch,DetID,ClassID,ClsScore
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
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.data.dataset_exp01 import DetectionDataset
from src.models.tasks.task_exp01   import DetectionTask
from src.utils.lineage_tracker         import register_process, get_node_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_config_for_checkpoint(checkpoint_id: str) -> tuple:
    """Return (config dict, config path) for a given NodeID."""
    config_dir  = "models/configuration_snapshots"
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


def load_weights(checkpoint_id: str, task: DetectionTask) -> int:
    """Load model weights in-place. Returns the epoch the model was saved at."""
    weights_dir  = "models/weights"
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
    task:      DetectionTask,
    loader:    DataLoader,
    dataset:   DetectionDataset,
    device:    torch.device,
    threshold: float,
) -> tuple:
    """
    Run the model over all batches and simultaneously read ground-truth labels.

    Returns
    -------
    preds_by_sample : dict[flat_idx -> list[det_dict]]
        det_dict keys: scene_id, channel_id, start_raw, end_raw,
                       class_id, obj_score, cls_score
    gt_by_sample    : dict[flat_idx -> list[gt_dict]]
        gt_dict keys: scene_id, channel_id, start_raw, end_raw, class_id
    """
    task.eval()
    preds_by_sample = {}
    gt_by_sample    = {}
    sample_idx = 0

    for signals, _ in loader:
        signals = signals.to(device)
        preds   = task.forward(signals)                          # (B, S, 5)
        batch_results = task.decode_predictions(
            preds, 
            seq_len=dataset.seq_len, 
            decimation_factor=dataset.decimation_factor, 
            threshold=threshold
        )

        B = signals.shape[0]
        for b in range(B):
            flat_idx = sample_idx + b
            shard_path, scene_idx, ch_idx = dataset.index[flat_idx]

            # --- Predictions for this sample ---
            dets = []
            for det in batch_results[b]:
                dets.append({
                    "scene_id"  : scene_idx,
                    "channel_id": ch_idx,
                    "start_raw" : det["start_raw"],
                    "end_raw"   : det["end_raw"],
                    "class_id"  : det["class_id"],
                    "obj_score" : det["obj_score"],
                    "cls_score" : det["cls_score"],
                })
            preds_by_sample[flat_idx] = dets

            # --- Ground truth for this sample (direct HDF5 read) ---
            _, ch_labels = dataset._get_raw(flat_idx)   # (7, K)
            gt_boxes = []
            for k in range(ch_labels.shape[1]):
                gt_boxes.append({
                    "scene_id"  : scene_idx,
                    "channel_id": ch_idx,
                    "start_raw" : int(ch_labels[dataset.ROW_START_IDX, k]),
                    "end_raw"   : int(ch_labels[dataset.ROW_END_IDX,   k]),
                    "class_id"  : int(ch_labels[dataset.ROW_CLASS_ID,  k]),
                })
            gt_by_sample[flat_idx] = gt_boxes

        sample_idx += B

    return preds_by_sample, gt_by_sample


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _iou_1d(p_s: int, p_e: int, g_s: int, g_e: int) -> float:
    """1D Intersection-over-Union between two sample-index intervals."""
    inter = max(0, min(p_e, g_e) - max(p_s, g_s))
    union = (p_e - p_s) + (g_e - g_s) - inter
    return inter / max(union, 1)


def evaluate(
    preds_by_sample: dict,
    gt_by_sample:    dict,
    iou_threshold:   float = 0.5,
) -> dict:
    """
    Greedy IoU matching (predictions sorted by obj_score descending).

    Returns a metrics dict with detection, localisation, and classification
    results.
    """
    TP = FP = FN = 0
    matched_ious  = []
    start_errors  = []
    end_errors    = []
    cls_correct   = 0

    # Per-class counters
    gt_total    = {0: 0, 1: 0}   # total GT boxes per class
    gt_detected = {0: 0, 1: 0}   # GT boxes matched by any prediction
    gt_cls_ok   = {0: 0, 1: 0}   # matched AND correct class

    for flat_idx in sorted(gt_by_sample.keys()):
        gt_list   = gt_by_sample[flat_idx]
        pred_list = preds_by_sample.get(flat_idx, [])

        for gt in gt_list:
            gt_total[gt["class_id"]] += 1

        matched_gt       = {}    # gt_idx -> pred_idx
        matched_pred_set = set()

        # Sort predictions by confidence descending
        ranked_preds = sorted(enumerate(pred_list),
                              key=lambda x: -x[1]["obj_score"])

        for pred_idx, pred in ranked_preds:
            best_iou    = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_list):
                if gt_idx in matched_gt:
                    continue
                iou = _iou_1d(pred["start_raw"], pred["end_raw"],
                              gt["start_raw"],  gt["end_raw"])
                if iou > best_iou:
                    best_iou    = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                TP += 1
                matched_gt[best_gt_idx] = pred_idx
                matched_pred_set.add(pred_idx)

                gt = gt_list[best_gt_idx]
                matched_ious.append(best_iou)
                start_errors.append(abs(pred["start_raw"] - gt["start_raw"]))
                end_errors.append(abs(pred["end_raw"]   - gt["end_raw"]))

                gt_detected[gt["class_id"]] += 1
                if pred["class_id"] == gt["class_id"]:
                    cls_correct += 1
                    gt_cls_ok[gt["class_id"]] += 1
            else:
                FP += 1

        # Unmatched GT = False Negatives
        FN += len(gt_list) - len(matched_gt)

    # ---- Aggregate metrics ---- #
    precision  = TP / (TP + FP)  if (TP + FP) > 0  else 0.0
    recall     = TP / (TP + FN)  if (TP + FN) > 0  else 0.0
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
    mean_iou   = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    cls_acc    = cls_correct / TP  if TP > 0 else 0.0
    mean_s_err = sum(start_errors) / len(start_errors) if start_errors else 0.0
    mean_e_err = sum(end_errors)   / len(end_errors)   if end_errors   else 0.0

    # Per-class detection recall and classification precision
    per_class = {}
    for c, label in enumerate(["PD1", "PD2"]):
        det_recall = (gt_detected[c] / gt_total[c]
                      if gt_total[c] > 0 else 0.0)
        cls_prec   = (gt_cls_ok[c] / gt_detected[c]
                      if gt_detected[c] > 0 else 0.0)
        per_class[label] = {
            "n_gt"           : gt_total[c],
            "n_detected"     : gt_detected[c],
            "detection_recall": round(det_recall, 4),
            "cls_precision"  : round(cls_prec,   4),
        }

    return {
        "iou_threshold"           : iou_threshold,
        "n_ground_truth"          : sum(gt_total.values()),
        "n_predictions"           : sum(len(v) for v in preds_by_sample.values()),
        "tp"                      : TP,
        "fp"                      : FP,
        "fn"                      : FN,
        "precision"               : round(precision,  4),
        "recall"                  : round(recall,     4),
        "f1"                      : round(f1,         4),
        "mean_iou_matched"        : round(mean_iou,   4),
        "cls_accuracy_at_tp"      : round(cls_acc,    4),
        "mean_start_error_samples": round(mean_s_err, 1),
        "mean_end_error_samples"  : round(mean_e_err, 1),
        "per_class"               : per_class,
    }


def print_and_save_metrics(metrics: dict, output_dir: str):
    """Print a formatted summary and save metrics.json."""
    sep = "=" * 56
    print("\n" + sep)
    print("  EVALUATION REPORT")
    print(sep)
    print(f"  IoU threshold     : {metrics['iou_threshold']}")
    print(f"  Ground-truth boxes: {metrics['n_ground_truth']}")
    print(f"  Predictions made  : {metrics['n_predictions']}")
    print(f"  True Positives    : {metrics['tp']}")
    print(f"  False Positives   : {metrics['fp']}")
    print(f"  False Negatives   : {metrics['fn']}")
    print("-" * 56)
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  F1 Score          : {metrics['f1']:.4f}")
    print("-" * 56)
    print(f"  Mean IoU (TPs)    : {metrics['mean_iou_matched']:.4f}")
    print(f"  Cls Accuracy (TPs): {metrics['cls_accuracy_at_tp']:.4f}")
    print(f"  Mean start error  : {metrics['mean_start_error_samples']:.0f} raw samples")
    print(f"  Mean end error    : {metrics['mean_end_error_samples']:.0f} raw samples")
    print("-" * 56)
    for cls_label, cm in metrics["per_class"].items():
        print(f"  {cls_label}: {cm['n_detected']}/{cm['n_gt']} detected "
              f"| Det Recall={cm['detection_recall']:.3f} "
              f"| Cls Prec={cm['cls_precision']:.3f}")
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

def _flatten(preds_by_sample: dict) -> list:
    """Convert per-sample dict to a flat list for HDF5 output."""
    flat = []
    for dets in preds_by_sample.values():
        flat.extend(dets)
    return flat


def save_predicted_boxes(detections: list, output_dir: str, checkpoint_id: str) -> str:
    path = os.path.join(output_dir, "predicted_boxes.h5")
    N    = len(detections)
    mat  = np.zeros((6, N), dtype=np.float64)
    for i, d in enumerate(detections):
        mat[0, i] = d["scene_id"]
        mat[1, i] = d["channel_id"]
        mat[2, i] = i                  # Det_ID
        mat[3, i] = d["obj_score"]
        mat[4, i] = d["start_raw"]
        mat[5, i] = d["end_raw"]
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("predictions", data=mat)
        ds.attrs["row_0"] = "Scene_ID"
        ds.attrs["row_1"] = "Channel_ID"
        ds.attrs["row_2"] = "Det_ID (links to predicted_classes.h5)"
        ds.attrs["row_3"] = "Obj_Score"
        ds.attrs["row_4"] = "Start_Idx (0-indexed raw sample)"
        ds.attrs["row_5"] = "End_Idx   (0-indexed raw sample)"
        f.attrs["checkpoint_id"] = checkpoint_id
        f.attrs["n_detections"]  = N
    print(f"[Output] {N} boxes  -> {path}")
    return path


def save_predicted_classes(detections: list, output_dir: str, checkpoint_id: str) -> str:
    path = os.path.join(output_dir, "predicted_classes.h5")
    N    = len(detections)
    mat  = np.zeros((5, N), dtype=np.float64)
    for i, d in enumerate(detections):
        mat[0, i] = d["scene_id"]
        mat[1, i] = d["channel_id"]
        mat[2, i] = i                  # Det_ID
        mat[3, i] = d["class_id"]
        mat[4, i] = d["cls_score"]
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("predictions", data=mat)
        ds.attrs["row_0"] = "Scene_ID"
        ds.attrs["row_1"] = "Channel_ID"
        ds.attrs["row_2"] = "Det_ID (links to predicted_boxes.h5)"
        ds.attrs["row_3"] = "Class_ID (0=PD1, 1=PD2)"
        ds.attrs["row_4"] = "Cls_Score"
        f.attrs["checkpoint_id"] = checkpoint_id
        f.attrs["n_detections"]  = N
    print(f"[Output] {N} classes -> {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FYP PD Pipeline - YOLO1D Inference + Evaluation"
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="NodeID of the trained model (e.g. QIRE).")
    parser.add_argument("--input_path",    default=None,
                        help="Dataset folder. Defaults to root_path in config.")
    parser.add_argument("--shards",        nargs="+", type=int, default=None,
                        help="Shard numbers to run on. Defaults to all 20.")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Objectness threshold (default 0.5).")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for TP matching in evaluation (default 0.5).")
    parser.add_argument("--batch_size",    type=int,   default=32,
                        help="Inference batch size (default 32).")
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

    task = DetectionTask(config)
    load_weights(checkpoint_id, task)
    task = task.to(device)
    task.eval()

    # ----------------------------------------------------------------------- #
    # 2. Dataset + loader                                                       #
    # ----------------------------------------------------------------------- #
    root_path = os.path.abspath(args.input_path or data_cfg["root_path"])
    shards    = args.shards or list(range(1, 21))

    print(f"[Data] Path  : {root_path}")
    print(f"[Data] Shards: {shards}")

    dataset = DetectionDataset(
        root_path         = root_path,
        shard_ids         = shards,
        decimation_factor = data_cfg["decimation_factor"],
        grid_cells        = data_cfg["grid_cells"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    print(f"[Data] Total samples: {len(dataset)}")

    # ----------------------------------------------------------------------- #
    # 3. Inference + GT collection                                              #
    # ----------------------------------------------------------------------- #
    print(f"\n[Inference] threshold={args.threshold} ...")
    infer_start = time.time()
    preds_by_sample, gt_by_sample = run_inference(
        task, loader, dataset, device, args.threshold
    )
    infer_elapsed = time.time() - infer_start
    total_preds = sum(len(v) for v in preds_by_sample.values())
    total_gt    = sum(len(v) for v in gt_by_sample.values())
    samples_per_sec = len(dataset) / infer_elapsed if infer_elapsed > 0 else 0
    print(f"[Inference] {total_preds} predictions | {total_gt} ground-truth boxes")
    print(f"[Inference] Time: {infer_elapsed:.2f}s ({samples_per_sec:.1f} samples/sec)")

    # ----------------------------------------------------------------------- #
    # 4. Evaluation                                                             #
    # ----------------------------------------------------------------------- #
    print(f"\n[Evaluation] IoU threshold={args.iou_threshold} ...")
    metrics = evaluate(preds_by_sample, gt_by_sample, args.iou_threshold)
    # Attach timing info to the metrics dict so it appears in metrics.json
    metrics["inference_time_s"]    = round(infer_elapsed, 3)
    metrics["samples_per_second"]  = round(samples_per_sec, 2)
    metrics["n_samples_evaluated"] = len(dataset)

    # ----------------------------------------------------------------------- #
    # 5. Output folder                                                          #
    # ----------------------------------------------------------------------- #
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_node_id = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    exp_cfg     = config["experiment"]
    root_id     = exp_cfg.get("parent_node_id", "UNKN")   # e.g. "ShmH"
    origin      = exp_cfg.get("origin", "sy")              # "sy" or "ms"
    method      = exp_cfg.get("name", "unknown")           # e.g. "exp01_yolo1d"
    folder_name = f"{run_ts}_{origin}-{root_id}-{inf_node_id}"
    out_cfg     = config["output"]
    output_dir  = os.path.abspath(
        os.path.join(out_cfg["results_dir"], method, folder_name)
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Output] Folder: {output_dir}")

    flat_dets = _flatten(preds_by_sample)
    save_predicted_boxes(flat_dets,   output_dir, checkpoint_id)
    save_predicted_classes(flat_dets, output_dir, checkpoint_id)
    print_and_save_metrics(metrics, output_dir)

    history_line = (
        f"Inference+Eval via YOLO1D (cnn_yolo1d) at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
        f"Checkpoint: {checkpoint_id}, InfNodeID: {inf_node_id}, "
        f"Shards: {shards}, ObjThresh: {args.threshold}, "
        f"IoUThresh: {args.iou_threshold}, "
        f"N_det: {total_preds}, F1: {metrics['f1']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}"
    )
    
    # ----------------------------------------------------------------------- #
    # 6. Lineage registration                                                   #
    # ----------------------------------------------------------------------- #
    print("[Lineage] Registering ...")
    new_node = register_process(
        parent_id        = checkpoint_id,
        stage            = "classification",
        method           = "cnn_yolo1d",
        folder_path      = output_dir,
        appended_history = history_line,
        force_node_id    = inf_node_id,
    )

    with open(os.path.join(output_dir, "analysis_history.txt"), "w") as f:
        full_history = get_node_history(inf_node_id)
        f.write(full_history)
    
    print(f"[Lineage] Node {new_node} registered (child of {checkpoint_id})")


if __name__ == "__main__":
    main()
