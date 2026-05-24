import os
import h5py
import numpy as np
import scipy.signal
import argparse
import sys
from datetime import datetime
import string
import random
import glob

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from src.utils.lineage_tracker import register_process
except ImportError:
    print("Warning: Could not import lineage_tracker.")
    register_process = None

def generate_short_id(length=4):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def process_shard(input_path, output_path, nperseg, noverlap):
    with h5py.File(input_path, 'r') as h5_in:
        scenes_1d = h5_in['scenes'][:] # Shape: (N_scenes, N_channels, T_samples)
        
        fs = h5_in.attrs.get('sampling_frequency_Hz', 1.0)
        
        # scipy.signal.stft can process along the last axis (-1)
        f, t, Zxx = scipy.signal.stft(scenes_1d, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
        
        # We only want the magnitude spectrogram
        magnitude = np.abs(Zxx).astype(np.float32) # Shape: (N, C, F, T_bins)
        
        # Save to output
        with h5py.File(output_path, 'w') as h5_out:
            h5_out.create_dataset('scenes_stft', data=magnitude)
            
            # Copy label dataset if it exists
            if 'labels' in h5_in:
                labels = h5_in['labels'][:]
                h5_out.create_dataset('labels', data=labels)
                for k, v in h5_in['labels'].attrs.items():
                    h5_out['labels'].attrs[k] = v
            
            # Copy all attributes from root
            for k, v in h5_in.attrs.items():
                h5_out.attrs[k] = v
                
            # Add STFT specific metadata
            h5_out.attrs['stft_nperseg'] = nperseg
            h5_out.attrs['stft_noverlap'] = noverlap
            h5_out.attrs['stft_hop_length'] = nperseg - noverlap
            h5_out.attrs['stft_fs'] = fs
            h5_out.attrs['stft_freq_bins'] = magnitude.shape[2]
            h5_out.attrs['stft_time_bins'] = magnitude.shape[3]
            h5_out.attrs['feature_type'] = "stft_magnitude"

def extract_features(input_dir, output_root, parent_node_id, nperseg=256, noverlap=128):
    os.makedirs(output_root, exist_ok=True)
    
    node_id = generate_short_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Query database for parent metadata to maintain folder structure 
    # data/features/<feature_name>/<timestamp>-<origin>-<rootid>-<nodeid>/
    import sqlite3
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils', 'lineage.db'))
    
    origin = "unknown"
    root_id = "unknown"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT origin, root_id FROM nodes WHERE node_id=?", (parent_node_id,))
        row = c.fetchone()
        if row:
            origin, root_id = row
        conn.close()
        
    folder_name = f"{timestamp}-{origin}-{root_id}-{node_id}"
    target_output_dir = os.path.join(output_root, "stft_magnitude", folder_name)
    os.makedirs(target_output_dir, exist_ok=True)
    
    h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {input_dir}")
        return
        
    print(f"Starting STFT Feature Extraction for {len(h5_files)} shards...")
    print(f"Output Directory: {target_output_dir}")
    
    for h5_file in h5_files:
        filename = os.path.basename(h5_file)
        out_file = os.path.join(target_output_dir, filename)
        
        print(f"Processing {filename} -> STFT...")
        process_shard(h5_file, out_file, nperseg, noverlap)
        
    history_log = f"Extracted STFT magnitude features. nperseg={nperseg}, noverlap={noverlap}."
    
    if register_process:
        register_process(
            parent_id=parent_node_id,
            stage="feature_extraction",
            method="stft",
            folder_path=target_output_dir,
            appended_history=history_log,
            force_node_id=node_id,
            force_timestamp=timestamp
        )
        print(f"Registered feature extraction node {node_id} successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw .h5 shards")
    parser.add_argument("--output_root", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'features')))
    parser.add_argument("--parent_node_id", type=str, required=True, help="Node ID of the raw dataset")
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--noverlap", type=int, default=128)
    
    args = parser.parse_args()
    
    extract_features(args.input_dir, args.output_root, args.parent_node_id, args.nperseg, args.noverlap)
