"""
equation_dataset_generate.py
============================
Generates PyTorch-compatible HDF5 shards from synthesized UHF PD waveforms using mathematical equations (SEDO, DED, DEDO, SMG).

Output format matches the FYP pipeline requirements (dynamic thresholding, TOA tracking, 7-column label matrix) 
and registers to the SQLite lineage DB.

Classes:
    - SEDO (Class ID: 5)
    - DED  (Class ID: 6)
    - DEDO (Class ID: 7)
    - SMG  (Class ID: 8)
"""

import argparse
import logging
import os
import random
import string
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.lineage_tracker import register_root_dataset

OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "eqn_generated"

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
N_SCENE_POINTS = 50_000   # 10 µs at 5 GS/s (dt = 0.2 ns)
NUM_SENSORS    = 4
NICKNAME       = "Equation Based Synthesised Data"
FS             = 5e9
DT             = 0.2e-9
SCENE_DURATION = 10e-6

# Physics constants
C = 3e8

# Base Coordinates (extracted from Generate_Waveforms.ipynb)
# Sensors
COORDS_S = np.array([
    [1.0, 3.0, 1.4], # S1
    [1.0, 5.0, 1.8], # S2
    [1.0, 7.0, 1.4], # S3
    [1.0, 9.0, 1.8], # S4
])

# PD Sources
COORDS_PD = np.array([
    [1.2, 2.3, 3.1], # PD1
    [2.9, 6.3, 5.4], # PD2
    [4.5, 7.8, 6.2], # PD3
    [6.1, 1.7, 3.5], # PD4
    [7.8, 1.2, 4.9], # PD5
    [9.2, 3.5, 2.8], # PD6
])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Signal Generators
# ──────────────────────────────────────────────────────────────────
def vary(val, limit=0.05):
    """Returns a value varied by up to +/- limit."""
    return val * random.uniform(1.0 - limit, 1.0 + limit)

def simulate_SEDO(t, V0, tau, fc, theta=0, td=0):
    V0 = vary(V0); tau = vary(tau); fc = vary(fc)
    signal = V0 * np.exp(-(t-td) / tau) * np.sin(2 * np.pi * fc * (t-td) + theta)
    signal[(t-td)<0] = 0
    return signal

def simulate_DED(t, V0, alpha, beta, td=0):
    V0 = vary(V0); alpha = vary(alpha); beta = vary(beta)
    signal = V0 * (np.exp(-alpha * (t-td)) - np.exp(-beta * (t-td)))
    signal[(t-td)<0] = 0
    return signal

def simulate_DEDO(t, V0, alpha, beta, fc, theta=0, td=0):
    V0 = vary(V0); alpha = vary(alpha); beta = vary(beta); fc = vary(fc)
    signal = V0 * (np.exp(-alpha * (t-td)) - np.exp(-beta * (t-td))) * np.sin(2 * np.pi * fc * (t-td) + theta)
    signal[(t-td)<0] = 0
    return signal

def simulate_SMG(t, A, t0, tau, fc, td=0):
    A = vary(A); tau = vary(tau); fc = vary(fc)
    exponent = -((t - t0 - td)**2) / (2 * (tau**2))
    signal = A * np.exp(exponent) * np.sin(2 * np.pi * fc * (t-t0-td))
    signal[(t-td)<0] = 0
    return signal

# ──────────────────────────────────────────────────────────────────
# Generation Loop
# ──────────────────────────────────────────────────────────────────
def generate_shards(
    output_dir: Path,
    root_id: str,
    noise_type: str,
    num_shards: int,
    scenes_per_shard: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    global_pulse_id = 0
    
    # 2 µs base pulse duration canvas for generating a clean pulse
    t_pulse = np.linspace(0, 2e-6, int(2e-6 * FS))
    
    buffer_idx = int(50e-9 / DT) # 50ns safe buffer
    label_buffer_idx = int(10e-9 / DT) # 10ns box buffer

    for shard_id in range(1, num_shards + 1):
        batch_scenes = np.zeros((scenes_per_shard, NUM_SENSORS, N_SCENE_POINTS), dtype=np.float64)
        batch_labels = []

        for scene_idx in tqdm(range(scenes_per_shard), desc=f"Shard {shard_id:02d}/{num_shards}", leave=False):
            num_pulses = random.randint(1, 3)
            
            for _ in range(num_pulses):
                global_pulse_id += 1
                
                # Pick equation and coords
                eq_type = random.randint(5, 8)  # 5=SEDO, 6=DED, 7=DEDO, 8=SMG
                pd_idx = random.randint(0, len(COORDS_PD) - 1)
                pd_loc = COORDS_PD[pd_idx]
                
                # Calculate distances to all 4 sensors
                distances = np.linalg.norm(COORDS_S - pd_loc, axis=1) # Shape: (4,)
                tof = distances / C
                tdoa_sec = tof - np.min(tof)
                tdoa_idx = np.round(tdoa_sec / DT).astype(int)
                
                # Attenuation (1/R) relative to the closest sensor
                min_dist = np.min(distances)
                attenuation = min_dist / distances # Closest is 1.0, further away < 1.0
                
                # Generate clean base pulse starting slightly delayed to allow smooth 0
                td_base = 200e-9
                if eq_type == 5:
                    base_sig = simulate_SEDO(t_pulse, 1.0, 0.4e-6, 15e6, td=td_base)
                elif eq_type == 6:
                    base_sig = simulate_DED(t_pulse, 2.0, 2e6, 10e6, td=td_base)
                elif eq_type == 7:
                    base_sig = simulate_DEDO(t_pulse, 2.0, 1.5e6, 15e6, 15e6, td=td_base)
                elif eq_type == 8:
                    base_sig = simulate_SMG(t_pulse, 1.0, 1e-6, 0.2e-6, 20e6, td=td_base)
                
                # Randomize injection index
                pulse_len = len(base_sig)
                max_tdoa_idx = np.max(tdoa_idx)
                # Keep it safe within the canvas
                if buffer_idx + 1 >= N_SCENE_POINTS - pulse_len - max_tdoa_idx - buffer_idx:
                    continue # Should not happen with 10us canvas and 2us pulse
                start_idx = random.randint(buffer_idx + 1, N_SCENE_POINTS - pulse_len - max_tdoa_idx - buffer_idx)
                
                for ch in range(NUM_SENSORS):
                    # TDOA Shift and Attenuation
                    ch_shift = tdoa_idx[ch]
                    sig = base_sig * attenuation[ch]
                    
                    idx_in = start_idx + ch_shift
                    idx_end = idx_in + pulse_len
                    
                    # Ensure dimensions
                    if idx_end > N_SCENE_POINTS:
                        continue
                        
                    batch_scenes[scene_idx, ch, idx_in:idx_end] += sig
                    
                    # Dynamic Thresholding for Bounding Box (5% of peak)
                    peak_val = np.max(np.abs(sig))
                    cutoff_threshold = 0.05 * peak_val
                    active_indices = np.where(np.abs(sig) > cutoff_threshold)[0]
                    
                    if len(active_indices) > 0:
                        local_start = active_indices[0]
                        local_end = active_indices[-1]
                    else:
                        local_start = 0
                        local_end = pulse_len - 1
                        
                    global_pulse_start = idx_in + local_start
                    global_pulse_end = idx_in + local_end
                    
                    # Apply label buffer
                    start_ch = max(0, global_pulse_start - label_buffer_idx)
                    end_ch = min(N_SCENE_POINTS - 1, global_pulse_end + label_buffer_idx)
                    
                    toa_idx = idx_in
                    
                    # [Scene_ID, Channel_ID, Class_ID, Pulse_Instance_ID, TOA_Index, Start_Idx, End_Idx]
                    batch_labels.append([
                        scene_idx, ch, eq_type, global_pulse_id - 1, toa_idx, start_ch, end_ch
                    ])
                    
            # Inject Noise
            if noise_type == "synthetic":
                batch_scenes[scene_idx] += np.random.randn(NUM_SENSORS, N_SCENE_POINTS) * 0.05
                t_arr = np.arange(N_SCENE_POINTS) * DT
                batch_scenes[scene_idx] += 0.05 * np.sin(2 * np.pi * 100e6 * t_arr)

        # Write shard
        out_file = output_dir / f"synth_shard_{shard_id:02d}.h5"
        if out_file.exists():
            out_file.unlink()

        with h5py.File(out_file, "w") as h5f:
            h5f.create_dataset("scenes", data=batch_scenes,
                               dtype=np.float64, compression="gzip", compression_opts=4)
            if batch_labels:
                lbl_arr = np.array(batch_labels, dtype=np.float64).T   # (7, N)
                h5f.create_dataset("labels", data=lbl_arr,
                                   dtype=np.float64, compression="gzip", compression_opts=4)
                h5f["labels"].attrs["column_1"] = "Scene_ID (0-indexed)"
                h5f["labels"].attrs["column_2"] = "Channel_ID (0-indexed)"
                h5f["labels"].attrs["column_3"] = "Class_ID (5=SEDO, 6=DED, 7=DEDO, 8=SMG)"
                h5f["labels"].attrs["column_4"] = "Pulse_Instance_ID (0-indexed)"
                h5f["labels"].attrs["column_5"] = "TOA_Index"
                h5f["labels"].attrs["column_6"] = "Start_Idx"
                h5f["labels"].attrs["column_7"] = "End_Idx"

            h5f.attrs["creation_date"]       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h5f.attrs["sampling_frequency_Hz"] = FS
            h5f.attrs["time_resolution_s"]   = DT
            h5f.attrs["scene_duration_s"]    = SCENE_DURATION
            h5f.attrs["num_scenes"]          = scenes_per_shard
            h5f.attrs["num_sensors"]         = NUM_SENSORS
            h5f.attrs["shard_id"]            = shard_id
            h5f.attrs["root_id"]             = root_id
            h5f.attrs["node_id"]             = root_id
            h5f.attrs["origin"]              = "sy"
            h5f.attrs["noise_type"]          = noise_type

    log.info("Done. Generated %d shards -> %s", num_shards, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate UHF synthetic shards using Equations (SEDO, DED, DEDO, SMG)."
    )
    parser.add_argument("--noise_type", choices=["none", "synthetic"], default="synthetic",
                        help="Noise type to inject (default: synthetic).")
    parser.add_argument("--num_shards", type=int, default=10,
                        help="Number of output shard files (default: 10).")
    parser.add_argument("--scenes_per_shard", type=int, default=100,
                        help="Scenes per shard (default: 100).")
    args = parser.parse_args()

    # Lineage setup
    node_id     = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{run_ts}_sy-{node_id}-{node_id}"
    output_dir  = OUTPUT_ROOT / folder_name

    history_log = (
        f"Equation based generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"Shards: {args.num_shards}, Scenes/shard: {args.scenes_per_shard}, Noise: {args.noise_type}."
    )

    generate_shards(
        output_dir       = output_dir,
        root_id          = node_id,
        noise_type       = args.noise_type,
        num_shards       = args.num_shards,
        scenes_per_shard = args.scenes_per_shard,
    )

    print(f"\nRegistering to lineage database...")
    register_root_dataset(
        origin          = "sy",
        method          = "equation_dataset_generate",
        folder_path     = str(output_dir),
        nickname        = NICKNAME,
        history_log     = history_log,
        force_root_id   = node_id,
        force_timestamp = run_ts,
    )
    print(f"[Lineage] Root Dataset Registered: {node_id}")
    print(f"\n[Done] Output -> {output_dir}")


if __name__ == "__main__":
    main()
