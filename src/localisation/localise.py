import os
import sys
import h5py
import numpy as np
import random
import string
import json
import argparse
from datetime import datetime
from collections import defaultdict
import scipy.signal
import time

# Ensure we can import from src/ even if run from inside src/localisation/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.lineage_tracker import register_process, get_node_history

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT = 299792458.0  # m/s in air/vacuum

# Sensor placements (meters)
RECEIVERS = [
    np.array([1.0, 3.0, 1.4]),
    np.array([1.0, 5.0, 1.8]),
    np.array([1.0, 7.0, 1.4]),
    np.array([1.0, 9.0, 1.8]),
]

# True locations by PD Type (Class ID).
# Users should edit these coordinates corresponding to the physical testing setup.
TRUE_LOCATIONS = {
    3: np.array([3.7, 7, 0.1]), # Example: PD2 Incision
    4: np.array([3.7, 7, 0.1]), # Example: PD3 Delamination
    5: np.array([7.000000, 3.0000000, 1.5]), # Example: FeOx
    6: np.array([9.000000, 4.0000000, 1.5]), # Example: FeOx_High
}

# ---------------------------------------------------------------------------
# ALGORITHMS
# ---------------------------------------------------------------------------

def compute_tdoa(signals, time_res):
    """
    Compute TDOA of channels 2, 3, 4 relative to channel 1.
    Signals must be a list or array of shape (4, N).
    Returns [T21, T31, T41] in seconds.
    """
    ref_sig = signals[0]
    tdoas = []
    
    for i in range(1, 4):
        sig = signals[i]
        # Cross correlation
        corr = scipy.signal.correlate(sig, ref_sig, mode='full')
        lags = scipy.signal.correlation_lags(len(sig), len(ref_sig), mode='full')
        
        peak_idx = np.argmax(np.abs(corr))
        shift_in_samples = lags[peak_idx]
        
        # shift > 0 means 'sig' is delayed relative to 'ref_sig'
        tdoa_seconds = shift_in_samples * time_res
        tdoas.append(tdoa_seconds)
        
    return np.array(tdoas)

def pso_localise(TDOA_N, receivers, speed_of_EM, max_iterations=10000, num_particles=1000):
    """
    Vectorized Particle Swarm Optimization to find the PD source location.
    """
    # Initialize particles within boundary
    particles_position = np.random.uniform(
        low=[0.5, 0.5, 0.1],
        high=[11.0, 11.0, 2.0],
        size=(num_particles, 3)
    )
    particles_velocity = np.random.rand(num_particles, 3)
    
    personal_best_positions = particles_position.copy()
    personal_best_values = np.ones(num_particles) * np.inf
    global_best_value = np.inf
    global_best_position = particles_position[0].copy()
    
    convergence_counter = 0
    prev_global_best_value = np.inf
    
    c1, c2, w = 3.0, 1.0, 0.1
    
    for _ in range(max_iterations):
        # Calculate distances for all particles at once. shape: (num_particles, 3)
        d1 = np.linalg.norm(particles_position - receivers[0], axis=1)
        d2 = np.linalg.norm(particles_position - receivers[1], axis=1)
        d3 = np.linalg.norm(particles_position - receivers[2], axis=1)
        d4 = np.linalg.norm(particles_position - receivers[3], axis=1)
        
        # Residuals
        r1 = d2 - d1 - (speed_of_EM * TDOA_N[0])
        r2 = d3 - d1 - (speed_of_EM * TDOA_N[1])
        r3 = d4 - d1 - (speed_of_EM * TDOA_N[2])
        
        # RMSE objective
        objective_value = np.sqrt((r1**2 + r2**2 + r3**2) / 3.0)
        
        # Update personal best
        better_mask = objective_value < personal_best_values
        personal_best_values[better_mask] = objective_value[better_mask]
        personal_best_positions[better_mask] = particles_position[better_mask]
        
        # Update global best
        min_idx = np.argmin(personal_best_values)
        if personal_best_values[min_idx] < global_best_value:
            global_best_value = personal_best_values[min_idx]
            global_best_position = personal_best_positions[min_idx].copy()
            
        # Velocity update
        r1_rand = np.random.rand(num_particles, 1)
        r2_rand = np.random.rand(num_particles, 1)
        
        particles_velocity = (w * particles_velocity +
                              c1 * r1_rand * (personal_best_positions - particles_position) +
                              c2 * r2_rand * (global_best_position - particles_position))
                              
        particles_position += particles_velocity
        particles_position = np.clip(particles_position, [0.5, 0.5, 0.1], [11.0, 11.0, 11.0])
        
        # Convergence check
        if abs(global_best_value - prev_global_best_value) < 1e-12:
            convergence_counter += 1
        else:
            convergence_counter = 0
            
        if convergence_counter >= 1000:
            break
            
        prev_global_best_value = global_best_value
        
    return global_best_position

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_h5", type=str, required=True,
                        help="Path to the predictions.h5 file generated by the classification stage.")
    parser.add_argument("--raw_data_dir", type=str, required=True,
                        help="Path to the raw measured dataset folder containing the original .h5 shards.")
    parser.add_argument("--cluster_threshold", type=float, default=0.05,
                        help="Minimum cluster size as a percentage of total dataset (default 0.05 = 5%%).")
    parser.add_argument("--pulse_len", type=int, default=4096,
                        help="Length of pulse to extract for cross-correlation.")
    parser.add_argument("--parent_node", type=str, required=True,
                        help="The node ID of the classification run that produced the predictions.h5")
    args = parser.parse_args()

    if not os.path.exists(args.predictions_h5):
        print(f"[Error] File not found: {args.predictions_h5}"); sys.exit(1)
    if not os.path.isdir(args.raw_data_dir):
        print(f"[Error] Directory not found: {args.raw_data_dir}"); sys.exit(1)

    # 1. Load predictions
    print(f"[Input] Loading {args.predictions_h5}...")
    pred_data = {}
    with h5py.File(args.predictions_h5, "r") as f:
        for k in f.keys():
            arr = f[k][:]
            if arr.dtype == object:
                # decode bytes back to string
                arr = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in arr])
            pred_data[k] = arr
    
    n_pulses = len(pred_data["cluster_id"])
    print(f"[Input] Loaded {n_pulses} pulses.")

    # 2. Filter Clusters
    unique_clusters, counts = np.unique(pred_data["cluster_id"], return_counts=True)
    valid_clusters = []
    
    for cid, count in zip(unique_clusters, counts):
        if cid == -1: continue # Skip noise
        if count / n_pulses >= args.cluster_threshold:
            valid_clusters.append(cid)
            
    print(f"[Filter] Found {len(valid_clusters)} clusters exceeding {args.cluster_threshold*100}% threshold.")

    if not valid_clusters:
        print("[Done] No valid clusters found. Exiting.")
        sys.exit(0)

    # Dictionary to keep open raw dataset files
    raw_files = {}
    def get_raw_shard(shard_path):
        basename = os.path.basename(shard_path)
        raw_path = os.path.join(args.raw_data_dir, basename)
        if raw_path not in raw_files:
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Missing raw shard {raw_path}")
            raw_files[raw_path] = h5py.File(raw_path, "r")
        return raw_files[raw_path]

    # 3. Process Clusters
    results = []
    t0 = time.time()
    
    for cid in valid_clusters:
        cluster_mask = (pred_data["cluster_id"] == cid)
        pulse_indices = np.where(cluster_mask)[0]
        
        # Majority predicted class for this cluster
        pred_classes = pred_data["pred_class_id"][pulse_indices]
        pred_maj_class = np.bincount(pred_classes[pred_classes >= 0]).argmax() if len(pred_classes[pred_classes >= 0]) > 0 else -1
        
        # Ground truth class for this cluster (>50% majority vote)
        gt_classes = pred_data["gt_class_id"][pulse_indices]
        gt_counts = np.bincount(gt_classes[gt_classes >= 0])
        gt_maj_class = -1
        if len(gt_counts) > 0:
            top_gt = gt_counts.argmax()
            if gt_counts[top_gt] / len(pulse_indices) > 0.5:
                gt_maj_class = top_gt
        
        # Group by instance ID
        instances = defaultdict(list)
        for idx in pulse_indices:
            inst_id = pred_data["pred_inst_id"][idx]
            if inst_id == -1: continue
            instances[inst_id].append(idx)
            
        print(f"\n[Cluster {cid}] Size: {len(pulse_indices)} | Pred Class: {pred_maj_class} | GT Class: {gt_maj_class} | Grouped into {len(instances)} instances.")
        
        cluster_locs = []
        compute_times = []
        errors = []
        
        for inst_id, idxs in instances.items():
            # Check if exactly 1 signal per channel (0, 1, 2, 3)
            channels = pred_data["ch_idx"][idxs]
            if len(channels) == 4 and set(channels) == {0, 1, 2, 3}:
                # Valid 4-channel instance
                # Sort indices by channel so they are [ch0, ch1, ch2, ch3]
                sorted_idxs = [idxs[i] for i in np.argsort(channels)]
                
                # Extract raw signals
                signals = []
                time_res = pred_data["time_res"][sorted_idxs[0]] # should be same for all
                
                try:
                    for s_idx in sorted_idxs:
                        shard = pred_data["shard_path"][s_idx]
                        scene = pred_data["scene_idx"][s_idx]
                        ch    = pred_data["ch_idx"][s_idx]
                        start = pred_data["start_idx"][s_idx]
                        
                        raw_f = get_raw_shard(shard)
                        sig = raw_f["scenes"][scene, ch, start:start+args.pulse_len]
                        signals.append(sig)
                except Exception as e:
                    print(f"  [Error] Failed reading raw signal for instance {inst_id}: {e}")
                    continue
                
                # Compute TDOA
                tdoa_n = compute_tdoa(signals, time_res)
                
                # PSO Localisation
                pso_t0 = time.time()
                est_loc = pso_localise(tdoa_n, RECEIVERS, SPEED_OF_LIGHT)
                pso_dur = time.time() - pso_t0
                
                cluster_locs.append(est_loc)
                compute_times.append(pso_dur)
                
        if not cluster_locs:
            print(f"  -> No valid 4-channel instances found in cluster {cid}.")
            continue
            
        # Average location for the cluster
        avg_loc = np.mean(cluster_locs, axis=0)
        
        # Calculate error if true location is known based on GT class
        true_loc = TRUE_LOCATIONS.get(gt_maj_class, None)
        if true_loc is not None:
            euclidean_distance = np.linalg.norm(true_loc - avg_loc)
            # Normalizer logic from deprecated script:
            sum_total = np.linalg.norm((np.abs(avg_loc - RECEIVERS[0]) + 
                                        np.abs(avg_loc - RECEIVERS[1]) + 
                                        np.abs(avg_loc - RECEIVERS[2]) + 
                                        np.abs(avg_loc - RECEIVERS[3])) / 4)
            percentage_error = (euclidean_distance / sum_total) * 100
        else:
            euclidean_distance = None
            percentage_error = None
            
        print(f"  -> Processed {len(cluster_locs)} valid instances.")
        print(f"  -> Estimated Loc: {np.round(avg_loc, 3)}")
        if true_loc is not None:
            print(f"  -> True Loc     : {np.round(true_loc, 3)}")
            print(f"  -> Euclid Dist  : {euclidean_distance:.3f} m")
            print(f"  -> % Error      : {percentage_error:.2f} %")
        else:
            print(f"  -> True Loc     : Unknown for GT Class {gt_maj_class}")
            
        results.append({
            "cluster_id": int(cid),
            "pred_class_id": int(pred_maj_class),
            "gt_class_id": int(gt_maj_class),
            "num_valid_instances": len(cluster_locs),
            "avg_compute_time_per_inst": round(np.mean(compute_times), 3),
            "est_x": float(avg_loc[0]),
            "est_y": float(avg_loc[1]),
            "est_z": float(avg_loc[2]),
            "true_x": float(true_loc[0]) if true_loc is not None else None,
            "true_y": float(true_loc[1]) if true_loc is not None else None,
            "true_z": float(true_loc[2]) if true_loc is not None else None,
            "error_m": round(float(euclidean_distance), 4) if euclidean_distance is not None else None,
            "error_pct": round(float(percentage_error), 4) if percentage_error is not None else None
        })

    # Close all raw datasets
    for f in raw_files.values():
        f.close()

    # 4. Output and Lineage
    parent_id = args.parent_node
    node_id = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir_name = f"{timestamp_str}_loc-{parent_id}-{node_id}"
    out_dir = os.path.abspath(os.path.join("data", "localisation_output", out_dir_name))
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "localisation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[Done] Processed in {time.time() - t0:.1f}s. Results saved -> {out_dir}")

    # Register in lineage
    
    history = (
        f"Localisation (PSO+TDOA) | Classes detected: {[r['pred_class_id'] for r in results]} | "
        f"Total evaluated clusters: {len(results)} | "
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    register_process(
        parent_id        = parent_id,
        stage            = "localisation",
        method           = "tdoa_pso_crosscorr",
        folder_path      = out_dir,
        appended_history = history,
        force_node_id    = node_id,
    )
    print(f"[Lineage] Node {node_id} registered (child of {parent_id})")

    with open(os.path.join(out_dir, "analysis_history.txt"), "w") as f:
        f.write(get_node_history(node_id))


if __name__ == "__main__":
    main()
