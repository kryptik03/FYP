"""
generate_measured_shards.py
===========================
Generates PyTorch-compatible HDF5 shards by taking isolated waveforms from
data/interim_measured/isolated_waveforms/ and injecting them into a 10us canvas.
Preserves multi-channel TDOA and supports real or synthetic noise injection.

Output: data/raw/measured/<timestamp>_ms-<node_id>-<node_id>/synth_shard_XX.h5
"""

import os
import re
import sys
import random
import string
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
from tqdm import tqdm

# Nickname
NICKNAME = "Measured Dataset Generation"

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.lineage_tracker import register_root_dataset

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────

INTERIM_ROOT = PROJECT_ROOT / "data" / "interim_measured" / "isolated_waveforms"
RAW_MEASURED_ROOT = PROJECT_ROOT / "data" / "raw" / "measured"
NOISE_DB_PATH = PROJECT_ROOT / "data" / "noise" / "real_noise_db.h5"

# Shard Generation Settings
N_SCENE_POINTS = 50000  # 10us at 5GS/s (dt = 0.2ns)
NUM_SENSORS = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def extract_class_id(pd_type_str: str) -> int:
    """PD1=0, PD2=1, PD3=2, PD4=3, PD5=4"""
    m = re.search(r"PD(\d+)", pd_type_str)
    if m:
        return int(m.group(1)) - 1
    return 0


# ─── Shard Generation ─────────────────────────────────────────────

def generate_shards(
    interim_dir: Path, output_dir: Path, root_id: str,
    noise_type: str, num_shards: int, scenes_per_shard: int
):
    # Collect all available isolated samples across all batches/voltages
    h5_files = list(interim_dir.rglob("*.h5"))
    samples = []
    
    log.info("Scanning interim isolated waveforms...")
    for h5p in h5_files:
        # Expected filename e.g. PD2_Incision_Batch1_14kv.h5
        with h5py.File(h5p, "r") as f:
            for voltage in f:
                for sample in f[voltage]:
                    samples.append((h5p, voltage, sample))
    
    if not samples:
        log.error("No samples found in %s to generate shards.", interim_dir)
        return

    log.info("Found %d isolated sample groups.", len(samples))

    noise_db = None
    noise_keys = []
    if noise_type == "real":
        if not NOISE_DB_PATH.exists():
            log.error("Real noise requested but %s not found. Run preprocess_noise.py first.", NOISE_DB_PATH)
            sys.exit(1)
        noise_db = h5py.File(NOISE_DB_PATH, "r")
        noise_keys = list(noise_db["traces"].keys())
        log.info("Loaded real noise database with %d traces.", len(noise_keys))

    output_dir.mkdir(parents=True, exist_ok=True)
    global_pulse_id = 0
    
    for shard_id in range(1, num_shards + 1):
        # Shape: (scenes_per_shard, channels, time)
        batch_scenes = np.zeros((scenes_per_shard, NUM_SENSORS, N_SCENE_POINTS), dtype=np.float64)
        batch_labels = []
        
        for scene_idx in tqdm(range(scenes_per_shard), desc=f"Generating Shard {shard_id}", leave=False):
            num_pulses = random.randint(1, 3)
            
            for p in range(num_pulses):
                h5p, voltage, sample_id = random.choice(samples)
                
                # Extract Class ID from the filename (e.g. PD2_Incision_...)
                pd_type_str = h5p.stem.split("_")[0]  # "PD2"
                class_id = extract_class_id(pd_type_str)
                
                with h5py.File(h5p, "r") as f:
                    grp = f[voltage][sample_id]
                    channels = list(grp.keys())
                    
                    signals = {}
                    starts = {}
                    for ch in channels:
                        ch_data = grp[ch]
                        signals[ch] = ch_data["iso_signal"][:]
                        starts[ch] = int(ch_data.attrs["start_idx"])
                        
                    if not signals: continue
                        
                    # Calculate relative TDOA shifts
                    global_s = min(starts.values())
                    max_len_with_shift = max(len(signals[ch]) + (starts[ch] - global_s) for ch in channels)
                    
                    if max_len_with_shift >= N_SCENE_POINTS - 1000:
                        continue # Pulse too long for canvas
                        
                    idx_in = random.randint(500, N_SCENE_POINTS - max_len_with_shift - 500)
                    global_pulse_id += 1
                    
                    for ch in channels:
                        ch_idx = int(ch.replace("Ch", "")) - 1
                        shift = starts[ch] - global_s
                        sig = signals[ch]
                        
                        start_ch = idx_in + shift
                        end_ch = start_ch + len(sig)
                        batch_scenes[scene_idx, ch_idx, start_ch:end_ch] += sig
                        
                        bbox_start = max(0, start_ch - 50)
                        bbox_end = min(N_SCENE_POINTS, end_ch + 50)
                        
                        # [Scene_ID, Channel_ID, Class_ID, Pulse_Instance_ID, TOA_Index, Start_Idx, End_Idx]
                        batch_labels.append([
                            scene_idx, ch_idx, class_id, global_pulse_id - 1, idx_in, bbox_start, bbox_end
                        ])
                        
            # Add Background Noise
            if noise_type == "synthetic":
                batch_scenes[scene_idx] += np.random.randn(NUM_SENSORS, N_SCENE_POINTS) * 0.05
                t = np.arange(N_SCENE_POINTS) * 0.2e-9
                fm_wave = 0.05 * np.sin(2 * np.pi * 100e6 * t)
                batch_scenes[scene_idx] += fm_wave
            elif noise_type == "real":
                for ch_idx in range(NUM_SENSORS):
                    trace_name = random.choice(noise_keys)
                    trace_ds = noise_db["traces"][trace_name]
                    L = trace_ds.attrs["length"]
                    if L > N_SCENE_POINTS:
                        start = random.randint(0, L - N_SCENE_POINTS - 1)
                        batch_scenes[scene_idx, ch_idx, :] += trace_ds[start:start+N_SCENE_POINTS]
            
        # Write to HDF5
        out_file = output_dir / f"synth_shard_{shard_id:02d}.h5"
        if out_file.exists(): out_file.unlink()
            
        with h5py.File(out_file, "w") as h5f:
            h5f.create_dataset("scenes", data=batch_scenes, dtype=np.float64, compression="gzip", compression_opts=4)
            if batch_labels:
                lbl_arr = np.array(batch_labels, dtype=np.float64).T
                h5f.create_dataset("labels", data=lbl_arr, dtype=np.float64, compression="gzip", compression_opts=4)
                
            dt_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            h5f.attrs["creation_date"] = dt_str
            h5f.attrs["sampling_frequency_Hz"] = 5e9
            h5f.attrs["time_resolution_s"] = 0.2e-9
            h5f.attrs["scene_duration_s"] = 10e-6
            h5f.attrs["num_scenes"] = scenes_per_shard
            h5f.attrs["num_sensors"] = 4
            h5f.attrs["shard_id"] = shard_id
            h5f.attrs["root_id"] = root_id
            h5f.attrs["node_id"] = root_id
            h5f.attrs["origin"] = "ms"
            h5f.attrs["noise_type"] = noise_type
            
            h5f["scenes"].attrs["python_h5py_shape"] = f"({scenes_per_shard}, 4, {N_SCENE_POINTS})"
            
            if batch_labels:
                h5f["labels"].attrs["column_1"] = "Scene_ID (0-indexed)"
                h5f["labels"].attrs["column_2"] = "Channel_ID (0-indexed)"
                h5f["labels"].attrs["column_3"] = "Class_ID (0=PD1, 1=PD2, 2=PD3, ...)"
                h5f["labels"].attrs["column_4"] = "Pulse_Instance_ID (0-indexed)"
                h5f["labels"].attrs["column_5"] = "TOA_Index"
                h5f["labels"].attrs["column_6"] = "Start_Idx"
                h5f["labels"].attrs["column_7"] = "End_Idx"

    if noise_db:
        noise_db.close()
    log.info("Generated %d shards in %s", num_shards, output_dir)


# ─── Main Pipeline ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch Shards from isolated waveforms")
    parser.add_argument("--noise_type", type=str, choices=["none", "synthetic", "real"], default="real")
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--scenes_per_shard", type=int, default=100)
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    node_id = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    folder_name = f"{run_ts}_ms-{node_id}-{node_id}"

    raw_output_dir = RAW_MEASURED_ROOT / folder_name
    
    log.info("Interim  : %s", INTERIM_ROOT)
    log.info("Shards   : %s", raw_output_dir)
    log.info("Noise    : %s", args.noise_type)

    if not INTERIM_ROOT.exists():
        log.error("Interim root directory not found. Run process_and_isolate.py first.")
        sys.exit(1)

    generate_shards(INTERIM_ROOT, raw_output_dir, node_id, args.noise_type, args.num_shards, args.scenes_per_shard)
    
    # Register in lineage
    history_log = f"Measured dataset [Nickname: {NICKNAME}] generated at {run_ts}, RootID: {node_id}"
    register_root_dataset(
        origin="ms",
        method="measurement",
        folder_path=str(raw_output_dir),
        nickname=NICKNAME,
        history_log=history_log,
        force_root_id=node_id,
        force_timestamp=run_ts
    )

if __name__ == "__main__":
    main()
