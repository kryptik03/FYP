import h5py
import numpy as np

def print_h5_dataset(filepath, dataset_name="labels", num_rows=5):
    with h5py.File(filepath, "r") as f:
        if dataset_name not in f:
            print(f"Dataset '{dataset_name}' not found in the file.")
            return
            
        data = f[dataset_name]
        print(f"--- {dataset_name} (Shape: {data.shape}) ---")
        
        # Load the requested portion into memory
        # In MATLAB formatting, arrays are often transposed. 
        # (7, 32) means 7 columns/features and 32 entries, so we slice accordingly.
        if data.shape[0] < data.shape[1]:
            # This handles cases like (7, 32) where we want the first 'num_rows' columns
            subset = data[:, :num_rows] 
            print(np.array(subset))
        else:
            # This handles standard (N, M) shapes where we want the first 'num_rows' rows
            subset = data[:num_rows, :]
            print(np.array(subset))
            
        # Print column descriptions if they exist in the metadata
        print("\n--- Attributes (Column Meanings) ---")
        for key, val in data.attrs.items():
            print(f"  {key}: {val}")

# Run the function
print_h5_dataset(r"data\interim_measured\isolated_waveforms\PD3_Delam_Batch1_8kv.h5", "labels", 5)
