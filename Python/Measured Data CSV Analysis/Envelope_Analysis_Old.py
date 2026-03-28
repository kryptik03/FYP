import os
import glob
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import sys

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = 'Isolated_Waveforms'
# Threshold for Difference Coefficient (D). 
# The paper suggests margins exist, e.g., D < 0.35 or D < 2.3 depending on sampling.
# You may need to tune this value based on your specific data.
SIMILARITY_THRESHOLD = 0.51

# Gaussian Smoothing Sigma
# Paper suggests Window Width W=20ns. Sigma is roughly W/4 or W/6.
# Adjust 'SIGMA_SAMPLES' based on your sampling rate.
SIGMA_SAMPLES = 27

# ==========================================
# SECTION 3: ENVELOPE PROCESSING FUNCTIONS
# ==========================================

def load_waveform(filepath):
    """
    Placeholder function to load waveform data.
    Assumes simple text file or numpy file. Modify as needed for your format.
    """
    try:
        data = pd.read_csv(filepath, header=None)
        sig = np.array(data.iloc[:, 1])
        return sig
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_envelope(voltage_signal, sigma_samples = SIGMA_SAMPLES):
    
    # Calculate mean and subtract
    signal_no_dc = voltage_signal - voltage_signal.mean()
    
    # Square to produce unipolar waveform proportional to power
    signal_squared = signal_no_dc ** 2
    
    # Gaussian kernel smoothing. 
    envelope_smoothed = gaussian_filter1d(signal_squared, sigma=sigma_samples)
    
    # Normalize amplitude with signal energy
    avg_energy = envelope_smoothed.mean()
    envelope_norm = envelope_smoothed / avg_energy
    
    # envelope_norm = envelope_smoothed / envelope_smoothed.max()
    
    return envelope_norm

def align_signals(REF_ENV, target_env):
    # REF_ENV should not be modified
    # REF_ENV should ideally be slightly smaller or the same size as the target_env
    # Cross-correlate
    correlation = signal.correlate(REF_ENV, target_env, mode='full')
    lags = signal.correlation_lags(REF_ENV.size, target_env.size, mode='full')
    
    # Find lag with maximum correlation (restricted to middle 600 samples)
    mid = len(correlation) // 2
    start_idx = max(0, mid - 1000)
    end_idx = min(len(correlation), mid + 1000)
    lag = lags[start_idx:end_idx][np.argmax(correlation[start_idx:end_idx])]
    print(f"Lag: {lag}")
    
    # Shift target signal
    if lag > 0:
        # Target needs to shift right (pad left)
        aligned_env = np.pad(target_env, (lag, 0), 'constant')[:len(REF_ENV)]
        aligned_actual_idx = [lag, len(aligned_env)-1]
    elif lag < 0:
        # Target needs to shift left (pad right)
        aligned_env = np.pad(target_env, (0, -lag), 'constant')[-lag:]
        aligned_actual_idx = [0, len(aligned_env)+lag-1]
        if len(aligned_env) > len(REF_ENV):
            aligned_env = aligned_env[:len(REF_ENV)]
            if (aligned_actual_idx[1]+1) > len(aligned_env):
                aligned_actual_idx[1] = len(aligned_env)-1
    else:
        aligned_env = target_env
        if len(aligned_env) > len(REF_ENV):
            aligned_env = aligned_env[:len(REF_ENV)]
        aligned_actual_idx = [0, len(aligned_env)-1]
    
    if len(aligned_env) < len(REF_ENV):
        aligned_env = np.pad(aligned_env, (0, len(REF_ENV)-len(aligned_env)), 'constant')

    return REF_ENV, aligned_env, aligned_actual_idx

def calculate_similarity_D(REF_ENV, aligned_env, aligned_actual_idx):
    # Ensure they are aligned first (caller should handle alignment, but double check sizes)
    D = (REF_ENV[aligned_actual_idx[0]:(aligned_actual_idx[1]+1)] - 
               aligned_env[aligned_actual_idx[0]:(aligned_actual_idx[1]+1)])
    D = np.abs(D).mean()
    return D

# ==========================================
# MAIN GROUPING LOGIC
# ==========================================

def main():
    # 1. Get list of files
    search_path = os.path.join(INPUT_FOLDER, '*')
    files = sorted(glob.glob(search_path)) # Sort to ensure consistent order
    
    if not files:
        print(f"No files found in {INPUT_FOLDER}")
        sys.exit()

    # Dictionary Structure: 
    # { group_id : [ [list_of_filenames], average_envelope_array ] }
    groups = {}
    
    # 2. Process the first file to start Group 1
    print(f"Processing {files[0]}...")
    raw_sig_0 = load_waveform(files[0])
    env_0 = process_envelope(raw_sig_0)
    
    # Initialize Group 1
    # { 1 : [ ['file1'], env_data ] }
    groups[1] = [[files[0]], env_0]
    
    # 3. Iterative Comparison
    for i in range(1, len(files)):
        current_filename = files[i]
        print(f"Processing {current_filename}...")
        
        # Load and Create Envelope
        raw_sig = load_waveform(current_filename)
        current_env = process_envelope(raw_sig)
        
        matched_group_id = None
        
        # Compare against existing groups
        for g_id, group_data in groups.items():
            file_list = group_data[0]
            avg_envelope = group_data[1]
            
            _, curr_aligned, aligned_actual_idx = align_signals(avg_envelope, current_env)
            
            # Calculate D 
            D = calculate_similarity_D(avg_envelope, curr_aligned, aligned_actual_idx)
            print(f"  Comparing vs Group {g_id}: D = {D:.4f}")
            
            if D < SIMILARITY_THRESHOLD:
                matched_group_id = g_id
                
                # Logic: If similar, update average
                # Formula: (avg * n + new) / (n + 1)
                n = len(file_list)
                
                # Update the Average Envelope
                # Note: We use the *aligned* version of the new envelope to keep the average crisp
                if (aligned_actual_idx[1]-aligned_actual_idx[0]) >= 5800:
                    new_avg_envelope = (avg_envelope[aligned_actual_idx[0]:(aligned_actual_idx[1]+1)] * n 
                    + curr_aligned[aligned_actual_idx[0]:(aligned_actual_idx[1]+1)]) / (n + 1)
                    
                    # Only update envelope if it will not reduce the length to below 5800
                    groups[g_id][1] = new_avg_envelope
                
                # Update Dictionary
                groups[g_id][0].append(current_filename)
                
                print(f"  -> Match found! Added to Group {g_id}.")
                break # Stop checking other groups if match found
        
        # If deemed unsimilar to all existing groups
        if matched_group_id is None:
            new_group_id = len(groups) + 1
            groups[new_group_id] = [[current_filename], current_env]
            print(f"  -> No match. Created Group {new_group_id}.")

    # ==========================================
    # FINAL OUTPUT
    # ==========================================
    print("\n" + "="*30)
    print("FINAL GROUPING RESULTS")
    print("="*30)
    
    for g_id, data in groups.items():
        file_names = data[0]
        
        print(f"Group {g_id}: {len(file_names)} files")
        print(f"Files: {file_names}")
        print("-" * 20)
        
    # The actual object containing the data
    # final_dictionary = groups 

if __name__ == "__main__":
    main()