"""
predict.py
==========
Inference script for the YOLO1D joint PD detection and classification model.

Usage
-----
    python src/models/predict.py \\
        --checkpoint QIRE \\
        --input_path data/raw/synthesised/20260427_170034_sy-ShmH-ShmH \\
        --shards 1 2 3 4 5

What this script does
---------------------
1. Loads the trained model weights from models/weights/model_<NodeID>.pt
2. Loads the corresponding config from models/configuration_snapshots/config_<NodeID>.yaml
3. Runs inference on every (scene, channel) sample in the specified shards
4. Collects all detections above an objectness threshold
5. Saves predicted_boxes.h5 and predicted_classes.h5 to the correct DAG output folder
6. Registers the inference run to the SQLite lineage database

Output HDF5 format
------------------
Both output files share the same row-indexed column structure (mirrors the
ground-truth label format used during training):

predicted_boxes.h5   dataset '/predictions', shape (6, N_detections)
    Row 0  Scene_ID      (0-indexed)
    Row 1  Channel_ID    (0-indexed, 0-3)
    Row 2  Det_ID        (sequential unique ID for this prediction run)
    Row 3  Obj_Score     (objectness probability, 0-1)
    Row 4  Start_Idx     (0-indexed raw sample index)
    Row 5  End_Idx       (0-indexed raw sample index)

predicted_classes.h5  dataset '/predictions', shape (5, N_detections)
    Row 0  Scene_ID
    Row 1  Channel_ID
    Row 2  Det_ID        (same ID as predicted_boxes.h5 -- links the two files)
    Row 3  Class_ID      (0 = PD1, 1 = PD2)
    Row 4  Cls_Score     (class confidence, 0-1)
"""

import argparse
import os
import sys
from datetime import datetime

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup — works from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))       # src/models
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))  # FYP/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.data.dataset_detection import DetectionDataset
from src.models.tasks.task_detection   import DetectionTask
from src.utils.lineage_tracker         import register_process


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_id: str, out_cfg: dict) -> dict:
    """Load the saved state dict from models/weights/model_<ID>.pt."""
    weights_path = os.path.abspath(
        os.path.join(out_cfg["weights_dir"], f"model_{checkpoint_id}.pt")
    )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"[Error] Weights not found: {weights_path}\n"
            f"  Did you download the .pt file from Colab?"
        )
    state = torch.load(weights_path, map_location="cpu")
    print(f"[Checkpoint] Loaded weights from: {weights_path}")
    print(f"[Checkpoint] Trained to epoch   : {state.get('epoch', '?')}")
    return state


def load_config_for_checkpoint(checkpoint_id: str, out_cfg: dict) -> dict:
    """Load the YAML config snapshot that was saved alongside the weights."""
    config_path = os.path.abspath(
        os.path.join(out_cfg["config_snapshot_dir"], f"config_{checkpoint_id}.yaml")
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[Error] Config snapshot not found: {config_path}"
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[Checkpoint] Loaded config from : {config_path}")
    return config, config_path


def select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    task:       DetectionTask,
    loader:     DataLoader,
    dataset:    DetectionDataset,
    device:     torch.device,
    threshold:  float,
) -> list[dict]:
    """
    Iterate over all (scene, channel) samples and collect detections.

    Returns
    -------
    detections : list of dicts, one entry per detection above threshold:
        {
          'scene_id'  : int,
          'channel_id': int,
          'start_raw' : int,   # 0-indexed raw sample
          'end_raw'   : int,   # 0-indexed raw sample
          'class_id'  : int,
          'obj_score' : float,
          'cls_score' : float,
        }
    """
    task.eval()
    detections = []
    sample_idx = 0  # tracks position in the flat dataset index

    for signals, _ in loader:
        signals = signals.to(device)                         # (B, 1, seq_len)
        preds   = task.forward(signals)                      # (B, S, 5)
        batch_results = task.decode_predictions(preds, threshold=threshold)

        B = signals.shape[0]
        for b in range(B):
            flat_idx   = sample_idx + b
            shard_path, scene_idx, ch_idx = dataset.index[flat_idx]

            for det in batch_results[b]:
                detections.append({
                    "scene_id"  : scene_idx,
                    "channel_id": ch_idx,
                    "start_raw" : det["start_raw"],
                    "end_raw"   : det["end_raw"],
                    "class_id"  : det["class_id"],
                    "obj_score" : det["obj_score"],
                    "cls_score" : det["cls_score"],
                })

        sample_idx += B

    return detections


# ---------------------------------------------------------------------------
# HDF5 output writers
# ---------------------------------------------------------------------------

def save_predicted_boxes(detections: list[dict], output_dir: str, checkpoint_id: str):
    """
    Save predicted bounding boxes to predicted_boxes.h5.

    Shape: (6, N_detections)
    Rows : [Scene_ID, Channel_ID, Det_ID, Obj_Score, Start_Idx, End_Idx]
    """
    path = os.path.join(output_dir, "predicted_boxes.h5")
    N = len(detections)

    matrix = np.zeros((6, N), dtype=np.float64)
    for i, d in enumerate(detections):
        matrix[0, i] = d["scene_id"]
        matrix[1, i] = d["channel_id"]
        matrix[2, i] = i               # Det_ID
        matrix[3, i] = d["obj_score"]
        matrix[4, i] = d["start_raw"]
        matrix[5, i] = d["end_raw"]

    if os.path.exists(path):
        os.remove(path)

    with h5py.File(path, "w") as f:
        ds = f.create_dataset("predictions", data=matrix)
        ds.attrs["row_0"] = "Scene_ID (0-indexed)"
        ds.attrs["row_1"] = "Channel_ID (0-indexed)"
        ds.attrs["row_2"] = "Det_ID (sequential, links to predicted_classes.h5)"
        ds.attrs["row_3"] = "Obj_Score (objectness probability)"
        ds.attrs["row_4"] = "Start_Idx (0-indexed raw sample)"
        ds.attrs["row_5"] = "End_Idx (0-indexed raw sample)"
        f.attrs["checkpoint_id"]  = checkpoint_id
        f.attrs["n_detections"]   = N
        f.attrs["creation_date"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[Output] Saved {N} detections -> {path}")
    return path


def save_predicted_classes(detections: list[dict], output_dir: str, checkpoint_id: str):
    """
    Save predicted classes to predicted_classes.h5.

    Shape: (5, N_detections)
    Rows : [Scene_ID, Channel_ID, Det_ID, Class_ID, Cls_Score]
    """
    path = os.path.join(output_dir, "predicted_classes.h5")
    N = len(detections)

    matrix = np.zeros((5, N), dtype=np.float64)
    for i, d in enumerate(detections):
        matrix[0, i] = d["scene_id"]
        matrix[1, i] = d["channel_id"]
        matrix[2, i] = i               # Det_ID
        matrix[3, i] = d["class_id"]
        matrix[4, i] = d["cls_score"]

    if os.path.exists(path):
        os.remove(path)

    with h5py.File(path, "w") as f:
        ds = f.create_dataset("predictions", data=matrix)
        ds.attrs["row_0"] = "Scene_ID (0-indexed)"
        ds.attrs["row_1"] = "Channel_ID (0-indexed)"
        ds.attrs["row_2"] = "Det_ID (sequential, links to predicted_boxes.h5)"
        ds.attrs["row_3"] = "Class_ID (0=PD1, 1=PD2)"
        ds.attrs["row_4"] = "Cls_Score (class confidence)"
        f.attrs["checkpoint_id"] = checkpoint_id
        f.attrs["n_detections"]  = N
        f.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[Output] Saved {N} class predictions -> {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FYP PD Pipeline - YOLO1D Inference"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="NodeID of the trained model to load (e.g. QIRE). "
             "Looks for models/weights/model_<ID>.pt"
    )
    parser.add_argument(
        "--input_path",
        help="Path to the dataset folder to run inference on. "
             "Defaults to the root_path in the config snapshot."
    )
    parser.add_argument(
        "--shards", nargs="+", type=int,
        help="Which shard numbers to run inference on (e.g. --shards 1 2 3). "
             "Defaults to all 20 shards."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Objectness probability threshold for reporting a detection (default 0.5)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Inference batch size (default 32)."
    )
    args = parser.parse_args()

    checkpoint_id = args.checkpoint

    # ----------------------------------------------------------------------- #
    # 1. Load config snapshot (the config that was used during training)       #
    # ----------------------------------------------------------------------- #
    # Use a minimal out_cfg to locate the snapshot — paths are repo-relative
    _out_cfg_default = {
        "weights_dir":        "models/weights",
        "config_snapshot_dir": "models/configuration_snapshots",
        "results_dir":        "data/classification_output",
        "eval_dir":           "data/performance_evaluation/classification",
    }
    config, _ = load_config_for_checkpoint(checkpoint_id, _out_cfg_default)

    out_cfg   = config["output"]
    data_cfg  = config["data"]
    train_cfg = config["training"]

    # ----------------------------------------------------------------------- #
    # 2. Resolve device                                                         #
    # ----------------------------------------------------------------------- #
    device = select_device(train_cfg["device"])
    print(f"[Device] Using: {device}")

    # ----------------------------------------------------------------------- #
    # 3. Build model and load weights                                           #
    # ----------------------------------------------------------------------- #
    task  = DetectionTask(config)
    state = load_checkpoint(checkpoint_id, _out_cfg_default)
    task.load_state_dict(state["model_state"])
    task  = task.to(device)
    task.eval()

    total_params = sum(p.numel() for p in task.parameters())
    print(f"[Model] Parameters: {total_params:,}")

    # ----------------------------------------------------------------------- #
    # 4. Build Dataset and DataLoader                                           #
    # ----------------------------------------------------------------------- #
    root_path = os.path.abspath(args.input_path or data_cfg["root_path"])
    shards    = args.shards or list(range(1, 21))   # default: all 20 shards

    print(f"[Data] Input path : {root_path}")
    print(f"[Data] Shards     : {shards}")

    dataset = DetectionDataset(
        root_path         = root_path,
        shard_ids         = shards,
        decimation_factor = data_cfg["decimation_factor"],
        grid_cells        = data_cfg["grid_cells"],
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,            # preserve order so index maps correctly
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )
    print(f"[Data] Total samples: {len(dataset)}")

    # ----------------------------------------------------------------------- #
    # 5. Run inference                                                          #
    # ----------------------------------------------------------------------- #
    print(f"\n[Inference] Running with objectness threshold = {args.threshold}...")
    detections = run_inference(task, loader, dataset, device, args.threshold)
    print(f"[Inference] Found {len(detections)} total detections.")

    if len(detections) == 0:
        print("[Warning] No detections found. Try lowering --threshold.")

    # ----------------------------------------------------------------------- #
    # 6. Create output folder and save results                                  #
    # ----------------------------------------------------------------------- #
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    # The parent (QIRE) inherited root_id from ShmH -> root_id stays ShmH
    # We generate a new NodeID for this inference run
    import random, string
    inf_node_id = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    folder_name = f"{run_ts}_sy-ShmH-{inf_node_id}"

    output_dir = os.path.abspath(
        os.path.join(out_cfg["results_dir"], "cnn_yolo1d", folder_name)
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Output] Saving to: {output_dir}")

    boxes_path   = save_predicted_boxes(detections,   output_dir, checkpoint_id)
    classes_path = save_predicted_classes(detections, output_dir, checkpoint_id)

    # analysis_history.txt (created locally — this is an inference run, not Colab)
    history_line = (
        f"Inference via YOLO1D (cnn_yolo1d) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
        f"Checkpoint: {checkpoint_id}, InfNodeID: {inf_node_id}, "
        f"Shards: {shards}, Threshold: {args.threshold}, "
        f"N_detections: {len(detections)}"
    )
    with open(os.path.join(output_dir, "analysis_history.txt"), "w") as f:
        f.write(history_line + "\n")

    # ----------------------------------------------------------------------- #
    # 7. Register inference run to lineage DB                                   #
    # ----------------------------------------------------------------------- #
    print("\n[Lineage] Registering to SQLite DAG...")
    new_node = register_process(
        parent_id        = checkpoint_id,          # parent = trained model node
        stage            = "classification",        # still the most downstream task
        method           = "cnn_yolo1d",
        folder_path      = output_dir,
        appended_history = history_line,
        force_node_id    = inf_node_id,
    )
    print(f"[Lineage] Registered as Node {new_node} (child of {checkpoint_id})")
    print(f"\n[Done] Inference complete.")
    print(f"  predicted_boxes.h5   -> {boxes_path}")
    print(f"  predicted_classes.h5 -> {classes_path}")


if __name__ == "__main__":
    main()
