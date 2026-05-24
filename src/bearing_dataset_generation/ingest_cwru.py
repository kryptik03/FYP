import os
import h5py
import scipy.io
import numpy as np
import random
import string
import re
from datetime import datetime
import sys
import argparse

# Add the parent directory to the path so we can import the lineage tracker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from src.utils.lineage_tracker import register_root_dataset
except ImportError:
    print("Warning: Could not import lineage_tracker. Running in standalone mode.")

def generate_short_id(length=4):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def ingest_cwru_directory(
    input_dir, 
    output_dir, 
    chunk_size=2048, 
    group_by_fault_base=True, 
    max_scenes_per_shard=None,
    include_files=None,
    exclude_files=None
):
    """
    Crawls CWRU directory, parses fault strings, chunks into uniform scenes, 
    and exports to MLOps-compliant HDF5 shards.
    group_by_fault_base: If True, groups all faults with the same fault base (e.g., IR007_0 and IR007_3) into the same class.
    If False, groups by the full fault string (e.g., IR007_0 and IR007_3 into different classes).
    max_scenes_per_shard: If set, limits the number of scenes generated per shard.
    include_files: If set, a list of substrings. Only files matching any substring will be ingested.
    exclude_files: If set, a list of substrings. Files matching any substring will be excluded.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. MLOps Lineage Setup
    root_id = generate_short_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_str = "GroupedByFault" if group_by_fault_base else "DistinctHP"
    nickname = f"CWRU_Baseline_{group_str}"
    
    inc_str = ",".join(include_files) if include_files is not None else "None"
    exc_str = ",".join(exclude_files) if exclude_files is not None else "None"
    max_scenes_str = str(max_scenes_per_shard) if max_scenes_per_shard is not None else "All"
    history_log = (
        f"Measured dataset [{nickname}] ingested from CWRU .mat files at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"HP Grouping: {group_by_fault_base}, Max Scenes/Shard: {max_scenes_str}, "
        f"Include Filters: {inc_str}, Exclude Filters: {exc_str}"
    )
    
    folder_name = f"{timestamp}_cw-{root_id}-{root_id}"
    target_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(target_output_dir, exist_ok=True)

    with open(os.path.join(target_output_dir, "analysis_history.txt"), "w") as f:
        f.write(history_log + "\n")

    global_pulse_id = 0
    class_mapping = {}
    next_class_id = 11
    
    # Tracking for the Summary Report
    missing_channel_files = []
    processed_files_count = 0
    shard_id = 1

    # 2. Recursively find all .mat files
    for root, dirs, files in os.walk(input_dir):
        mat_files = [f for f in files if f.endswith('.mat')]
        
        for filename in sorted(mat_files):
            filepath = os.path.join(root, filename)
            parent_folder = os.path.basename(root) # e.g., '12k_Drive_End'
            
            # --- Include/Exclude Filter Check ---
            rel_path = f"{parent_folder}/{filename}"
            if include_files is not None:
                if not any(inc in filename or inc in rel_path for inc in include_files):
                    continue
            if exclude_files is not None:
                if any(exc in filename or exc in rel_path for exc in exclude_files):
                    continue
            
            # --- Dynamic Class Parsing ---
            # Matches strings like "IR007_0.mat" or "OR014@6_3.mat"
            # Group 1: Fault Base (e.g., IR007, OR014@6)
            # Group 2: HP (e.g., 0, 1, 3)
            match = re.match(r"(.*)_(\d+)\.mat", filename)
            
            if match:
                fault_base = match.group(1)
                hp = match.group(2)
            else:
                # Fallback if filename doesn't match convention
                fault_base = os.path.splitext(filename)[0]
                hp = "unknown"

            # Determine the Class Key based on user preference
            if group_by_fault_base:
                class_key = f"{parent_folder}_{fault_base}" 
            else:
                class_key = f"{parent_folder}_{fault_base}_HP{hp}"

            # Assign a permanent integer ID to this class key
            if class_key not in class_mapping:
                class_mapping[class_key] = next_class_id
                next_class_id += 1
            
            class_id = class_mapping[class_key]
            print(f"Processing: {parent_folder}/{filename} -> Class {class_id} ({class_key})")
            
            # 3. Load and Extract Arrays
            try:
                mat_data = scipy.io.loadmat(filepath)
            except Exception as e:
                print(f"  [!] Error reading {filename}: {e}")
                continue
                
            time_keys = [k for k in mat_data.keys() if k.endswith('_time')]
            
            # --- THE 3-CHANNEL SAFETY NET ---
            if len(time_keys) < 3:
                missing_channel_files.append((f"{parent_folder}/{filename}", len(time_keys)))
                
                # Zero-pad to ensure PyTorch doesn't crash on matrix shape mismatch
                reference_shape = mat_data[time_keys[0]].shape
                while len(time_keys) < 3:
                    pad_key = f"dummy_pad_{len(time_keys)}"
                    mat_data[pad_key] = np.zeros(reference_shape)
                    time_keys.append(pad_key)
            
            time_keys = time_keys[:3]
            arrays = [mat_data[k].flatten() for k in time_keys]
            min_length = min([len(arr) for arr in arrays])
            num_chunks = min_length // chunk_size
            
            if max_scenes_per_shard is not None:
                num_chunks = min(num_chunks, max_scenes_per_shard)
                
            if num_chunks == 0:
                print(f"  [-] Skipped: Not enough data points.")
                continue
                
            # 4. HDF5 Shard Construction
            # Initialized with 4 channels to match UHF PD setup
            batch_scenes = np.zeros((num_chunks, 4, chunk_size), dtype=np.float64)
            batch_labels = []
            
            for scene_idx in range(num_chunks):
                global_pulse_id += 1
                start_idx_raw = scene_idx * chunk_size
                end_idx_raw = start_idx_raw + chunk_size
                
                for ch_idx, arr in enumerate(arrays):
                    batch_scenes[scene_idx, ch_idx, :] = arr[start_idx_raw:end_idx_raw]
                    
                    batch_labels.append([
                        scene_idx, ch_idx, class_id, global_pulse_id, 
                        0, 0, chunk_size - 1
                    ])
                
                # Duplicate Channel 2 (Index 1) into Channel 4 (Index 3)
                batch_scenes[scene_idx, 3, :] = batch_scenes[scene_idx, 1, :]
                batch_labels.append([
                    scene_idx, 3, class_id, global_pulse_id, 
                    0, 0, chunk_size - 1
                ])

            h5_filename = f"shard_{shard_id:02d}.h5"
            h5_path = os.path.join(target_output_dir, h5_filename)
            shard_id += 1
            
            with h5py.File(h5_path, 'w') as h5f:
                h5f.create_dataset('scenes', data=batch_scenes)
                # Transpose labels to be (7, num_scenes) to match MATLAB format.
                h5f.create_dataset('labels', data=np.array(batch_labels, dtype=np.float64).T)
                
                # Append Metadata
                h5f.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                h5f.attrs['num_scenes'] = num_chunks
                h5f.attrs['num_sensors'] = 3
                h5f.attrs['chunk_size'] = chunk_size
                h5f.attrs['original_file'] = f"{parent_folder}/{filename}"
                h5f.attrs['assigned_class_name'] = class_key
                h5f.attrs['root_id'] = root_id
                h5f.attrs['node_id'] = root_id
                
                # Filter and limits tracking
                h5f.attrs['max_scenes_per_shard'] = max_scenes_per_shard if max_scenes_per_shard is not None else -1
                h5f.attrs['include_files'] = ",".join(include_files) if include_files is not None else ""
                h5f.attrs['exclude_files'] = ",".join(exclude_files) if exclude_files is not None else ""

                # Root Attributes to add:
                h5f.attrs['sampling_frequency_Hz'] = 12000.0 # Standard for CWRU drive end
                h5f.attrs['time_resolution_s'] = 1.0 / 12000.0

                # Label Definitions to add (matching MATLAB exactly):
                h5f['labels'].attrs['column_1'] = 'Scene_ID (0-indexed)'
                h5f['labels'].attrs['column_2'] = 'Channel_ID (0-indexed)'
                h5f['labels'].attrs['column_3'] = 'Class_ID'
                h5f['labels'].attrs['column_4'] = 'Pulse_Instance_ID (0-indexed)'
                h5f['labels'].attrs['column_5'] = 'TOA_Index'
                h5f['labels'].attrs['column_6'] = 'Start_Idx'
                h5f['labels'].attrs['column_7'] = 'End_Idx'

            processed_files_count += 1

    # 5. Database Registration
    print("\nRegistering CWRU dataset to SQLite Master Ledger...")
    try:
        register_root_dataset("cw", "ingestion_cwru", target_output_dir, nickname, history_log, force_root_id=root_id, force_timestamp=timestamp)
    except Exception as e:
        print(f"  [!] SQLite Registration Failed: {e}")

    # 6. Post-Processing Summary Report
    print("\n" + "="*50)
    print(" CWRU INGESTION SUMMARY REPORT")
    print("="*50)
    print(f"Total .mat Files Processed : {processed_files_count}")
    print(f"Total Unique Classes Found : {len(class_mapping)}")
    print(f"Output Directory           : {target_output_dir}")
    print(f"Grouping by Fault Type     : {group_by_fault_base}")
    print("\nClass ID Map:")
    for key, val in sorted(class_mapping.items(), key=lambda x: x[1]):
        print(f"  Class {val:02d} : {key}")
    
    print("\nChannel Integrity Check:")
    if not missing_channel_files:
        print("  [SUCCESS] All files contained 3 or more channels.")
    else:
        print(f"  [WARNING] {len(missing_channel_files)} files had fewer than 3 channels and were zero-padded:")
        for f_name, ch_count in missing_channel_files:
            print(f"    - {f_name} (Found {ch_count} channels)")
    print("="*50 + "\n")

if __name__ == "__main__":
    # USER CONFIGURATION
    INPUT_CWRU_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'unprocessed_bearing'))
    OUTPUT_H5_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cwru'))
    
    # TOGGLE THIS:
    # True = 'IR007_0' and 'IR007_3' become the same Class ID.
    # False = 'IR007_0' and 'IR007_3' become different Class IDs.
    GROUP_BY_FAULT_LOCATION = False
    
    # Filter settings:
    # Set to an integer to limit the number of scenes per file, or None to ingest all.
    MAX_SCENES_PER_SHARD = 5
    
    # Set to a list of substrings to match filenames/paths you WANT to ingest (e.g. ["IR007", "Normal"]) or None.
    INCLUDE_FILES = ["IR007", "IR014"]
    
    # Set to a list of substrings to match filenames/paths you want to EXCLUDE (e.g. ["OR014"]) or None.
    EXCLUDE_FILES = None
    
    parser = argparse.ArgumentParser(description="Ingest CWRU dataset.")
    parser.add_argument("--group_by_fault_location", type=str, choices=['True', 'False'], default=str(GROUP_BY_FAULT_LOCATION), help="Group by fault location")
    parser.add_argument("--max_scenes_per_shard", type=int, default=MAX_SCENES_PER_SHARD, help="Max scenes per shard (-1 for no limit)")
    args = parser.parse_args()
    
    group_by_fault_location = args.group_by_fault_location == 'True'
    max_scenes_per_shard = args.max_scenes_per_shard if args.max_scenes_per_shard > 0 else None
    
    ingest_cwru_directory(
        input_dir=INPUT_CWRU_DIR, 
        output_dir=OUTPUT_H5_DIR, 
        chunk_size=2048, 
        group_by_fault_base=group_by_fault_location,
        max_scenes_per_shard=max_scenes_per_shard,
        include_files=INCLUDE_FILES,
        exclude_files=EXCLUDE_FILES
    )


    """
    # Filter settings:
    # Set to an integer to limit the number of scenes per file, or None to ingest all.
    MAX_SCENES_PER_SHARD = 50  # <-- E.g., limit to 50 scenes per file
    
    # Set to a list of substrings to match filenames/paths you WANT to ingest or None.
    INCLUDE_FILES = ["IR007", "Normal"]  # <-- E.g., only ingest IR007 faults and Normal bearings
    
    # Set to a list of substrings to match filenames/paths you want to EXCLUDE or None.
    EXCLUDE_FILES = ["OR014"]  # <-- E.g., ignore all OR014 faults

    """