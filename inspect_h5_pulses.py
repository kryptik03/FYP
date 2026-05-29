import h5py
import numpy as np
import os

datasets = {
    "cwru": "data/features/stft_magnitude/20260524_234212-cw-TuKT-vraW/shard_01.h5",
    "equation": "data/features/stft_magnitude/20260524_234301-sy-0s6o-UI2r/shard_01.h5",
    "synthesised": "data/features/stft_magnitude/20260524_234445-sy-1yGk-N5ZR/shard_01.h5",
    "measured": "data/features/stft_magnitude/20260527_205109-ms-nQgf-LTaf/shard_01.h5"
}

ROW_START_IDX = 5
ROW_END_IDX = 6

for name, rel_path in datasets.items():
    path = os.path.join(r"d:\Zee_Documents\Studies\Uni\Sem_8\KIE4002_FYP\Git_Cloned_Code\FYP", rel_path)
    if not os.path.exists(path):
        print(f"{name}: File not found ({path})")
        continue
    
    try:
        with h5py.File(path, "r") as f:
            if "labels" not in f:
                print(f"{name}: No labels in file")
                continue
            labels = f["labels"][:]
            
            # length in indices
            lens = labels[ROW_END_IDX, :] - labels[ROW_START_IDX, :]
            avg_len_idx = np.mean(lens)
            min_len_idx = np.min(lens)
            max_len_idx = np.max(lens)
            
            # time resolution
            time_res = float(f.attrs.get("time_resolution_s", 1e-11))
            
            avg_time = avg_len_idx * time_res
            min_time = min_len_idx * time_res
            max_time = max_len_idx * time_res
            
            print(f"--- {name.upper()} ---")
            print(f"  Pulses Count: {labels.shape[1]}")
            print(f"  Time Resolution: {time_res:.2e} s")
            print(f"  Length (indices): Avg={avg_len_idx:.1f}, Min={min_len_idx}, Max={max_len_idx}")
            print(f"  Length (time):    Avg={avg_time*1e6:.2f} us, Min={min_time*1e6:.2f} us, Max={max_time*1e6:.2f} us")
    except Exception as e:
        print(f"{name}: Error reading file: {e}")

