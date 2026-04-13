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
INPUT_FOLDER = 'Simulated_Waveforms'
INDICES_FILE = 'Waveform_Analysis.csv'
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

def load_waveform(filename, start_idx, end_idx, channel_dataframes):
    """
    Extracts a waveform segment from the pre-loaded channel dataframes.
    
    Args:
        filename (str): The name of the channel file (e.g., 'ch1.csv').
        start_idx (int): Starting row index of the segment.
        end_idx (int): Ending row index of the segment.
        channel_dataframes (dict): Dictionary mapping filenames to DataFrames.
    """
    try:
        df = channel_dataframes[filename]
        # Column 1 is expected to be Amplitude
        sig = df.iloc[start_idx:end_idx, 1].values
        return sig
    except Exception as e:
        print(f"Error extracting segment from {filename} ({start_idx}:{end_idx}): {e}")
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
    
    # Find lag with maximum correlation (restricted to middle 2000 samples)
    mid = len(correlation) // 2
    start_idx = max(0, mid - 1000)
    end_idx = min(len(correlation), mid + 1000)
    lag = lags[start_idx:end_idx][np.argmax(correlation[start_idx:end_idx])]
    # print(f"Lag: {lag}")
    
    # Shift target signal
    if lag > 0:
        # Target needs to shift right (pad left)
        aligned_env = np.pad(target_env, (lag, 0), 'constant')[:len(REF_ENV)]
        # ensures difference coefficient calculated at the portions where aligned_env is not 0
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

def cluster_waveforms(channel_dataframes):
    # 1. Load the indices summary file
    if not os.path.exists(INDICES_FILE):
        print(f"Error: {INDICES_FILE} not found. Run Waveform_Isolation.py first.")
        return {}
        
    indices_df = pd.read_csv(INDICES_FILE)
    
    if indices_df.empty:
        print("No isolated waveforms found in indices file.")
        return {}

    # Ensure needed columns exist
    if not all(col in indices_df.columns for col in ['input filename', 'start idx', 'end idx', 'count']):
        print(f"Error: Input CSV must contain ['input filename', 'start idx', 'end idx', 'count'] columns.")
        return

    # Dictionary Structure: 
    # { group_id : [ [list_of_metadata], average_envelope_array ] }
    groups = {}
    
    # 2. Process the first entry to start Group 1
    first_row = indices_df.iloc[0]
    filename = first_row['input filename']
    start_idx = first_row['start idx']
    end_idx = first_row['end idx']
    count = first_row['count']
    
    
    print(f"Processing {filename}_#{count}...")
    
    raw_sig_0 = load_waveform(filename, start_idx, end_idx, channel_dataframes)
    env_0 = process_envelope(raw_sig_0)
    
    if env_0 is None:
        print("Failed to process first envelope.")
        return {}

    # Initialize Group 1
    groups[1] = [[ [filename, count], ], env_0]

    grouping_list = [{'Grouping_Envelope_Analysis': 1},]
    
    # 3. Iterative Comparison
    for i in range(1, len(indices_df)):
        row = indices_df.iloc[i]
        filename = row['input filename']
        start_idx = row['start idx']
        end_idx = row['end idx']
        count = row['count']
        
        print(f"Processing {filename}_#{count}...")
        
        # Load and Create Envelope
        raw_sig = load_waveform(filename, start_idx, end_idx, channel_dataframes)
        if raw_sig is None: continue
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
                groups[g_id][0].append([filename, count])

                grouping_list.append({'Grouping_Envelope_Analysis': matched_group_id})
                
                print(f"  -> Match found! Added to Group {g_id}.")
                break # Stop checking other groups if match found
        
        # If deemed unsimilar to all existing groups
        if matched_group_id is None:
            new_group_id = len(groups) + 1
            groups[new_group_id] = [[[filename, count],], current_env]
            grouping_list.append({'Grouping_Envelope_Analysis': new_group_id})
            print(f"  -> No match. Created Group {new_group_id}.")

    gl = pd.DataFrame(grouping_list)
    indices_df = pd.concat([indices_df, gl], axis=1)
    indices_df.to_csv(INDICES_FILE, index=False)

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
    return groups



def cluster_waveforms_2(channel_dataframes):
    '''
    Clusters the waveforms, but finds one best match for each waveform in Ch1
    in each of the other channels.
    '''
    # 1. Load the indices summary file
    if not os.path.exists(INDICES_FILE):
        print(f"Error: {INDICES_FILE} not found. Run Waveform_Isolation.py first.")
        return {}
        
    indices_df = pd.read_csv(INDICES_FILE)
    
    if indices_df.empty:
        print("No isolated waveforms found in indices file.")
        return {}

    # Ensure needed columns exist
    if not all(col in indices_df.columns for col in ['input filename', 'start idx', 'end idx', 'count']):
        print(f"Error: Input CSV must contain ['input filename', 'start idx', 'end idx', 'count'] columns.")
        return

    indices_df['Grouping_Envelope_Analysis'] = 0

    envelopes_1 = pd.DataFrame()

    ch1_indices=indices_df[indices_df['input filename']=='Ch1.csv']

    # create a dataframe of envelopes for Ch1, columns are the count
    for i in range(len(ch1_indices)):
        row = ch1_indices.iloc[i]
        filename = row['input filename']
        start_idx = row['start idx']
        end_idx = row['end idx']
        count = row['count']
        
        sig = load_waveform(filename, start_idx, end_idx, channel_dataframes)
        env = process_envelope(sig)
        
        if env is None:
            print("Failed to process envelope.")
            continue
        
        # groupings for Ch1 are the same as the count
        indices_df.loc[(indices_df['input filename']=='Ch1.csv')&(indices_df['count']==count),'Grouping_Envelope_Analysis'] = count
        
        envelopes_1 = pd.concat([envelopes_1, pd.DataFrame(env, columns=[count])], axis=1)
    
    # for each channel, find the most similar envelope to each envelope in Ch1
    for ch in channel_dataframes.keys():
        if 'Ch1' in ch: continue
        envelopes_2 = pd.DataFrame()
        # extract all rows with the same filename
        ch_indices=indices_df[indices_df['input filename']==ch]
        
        # create a dataframe of envelopes for each channel, columns are the count
        for i in range(len(ch_indices)):
            row = ch_indices.iloc[i]
            filename = row['input filename']
            start_idx = row['start idx']
            end_idx = row['end idx']
            count = row['count']
            
            sig = load_waveform(filename, start_idx, end_idx, channel_dataframes)
            env = process_envelope(sig)
            
            if env is None:
                print("Failed to process envelope.")
                continue
            
            envelopes_2 = pd.concat([envelopes_2, pd.DataFrame(env, columns=[count])], axis=1)
        

        for group in envelopes_1.columns:
            minD=np.inf
            minCnt=np.inf
            ch1_env = envelopes_1[group].to_numpy()
            for cnt in envelopes_2.columns:
                if (indices_df.loc[(indices_df['input filename']==ch)&(indices_df['count']==cnt),'Grouping_Envelope_Analysis'] != 0).any():
                    continue
                ch2_env = envelopes_2[cnt].to_numpy()
                _, ch2_env_aligned, aligned_actual_idx = align_signals(ch1_env, ch2_env)
                D = calculate_similarity_D(ch1_env, ch2_env_aligned, aligned_actual_idx)
                if D<minD:
                    minD=D
                    minCnt=cnt
            if (minD != np.inf):
                indices_df.loc[(indices_df['input filename']==ch)&(indices_df['count']==minCnt),'Grouping_Envelope_Analysis'] = group
                
    indices_df.to_csv(INDICES_FILE, index=False)

    return

if __name__ == "__main__":
    # Recursively find all CSV files in the Simulated_Waveforms directory
    search_path = os.path.join(INPUT_FOLDER, '*.csv')
    channel_files = glob.glob(search_path)
    
    if not channel_files:
        print(f"No channel files found in {INPUT_FOLDER}")
        sys.exit()
        
    print(f"Loading {len(channel_files)} channel files into memory...")
    
    # dictionary containing the time and signal data for each channel
    # format: {filename: pd.DataFrame, shape = (N, 2), columns = ['time', 'signal']}
    channel_dataframes = {}
    for filepath in channel_files:
        filename = os.path.basename(filepath)
        try:
            # Load the CSV file without headers
            df = pd.read_csv(filepath, header=None)
            channel_dataframes[filename] = df
            print(f"  Loaded {filename}")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            
    if not channel_dataframes:
        print("No dataframes loaded. Exiting.")
        sys.exit()
        
    # cluster_waveforms(channel_dataframes)
    cluster_waveforms_2(channel_dataframes)


'''
# Gemini generated version
def cluster_waveforms_2(channel_dataframes):
    """
    Clusters waveforms by finding the best match for each reference waveform (Ch1)
    in every other channel, within the SIMILARITY_THRESHOLD.
    """
    if not os.path.exists(INDICES_FILE):
        print(f"Error: {INDICES_FILE} not found. Run Waveform_Isolation.py first.")
        return {}
        
    indices_df = pd.read_csv(INDICES_FILE)
    if indices_df.empty:
        print("No isolated waveforms found in indices file.")
        return {}

    # Initialize grouping column if it doesn't exist
    indices_df['Grouping_Envelope_Analysis'] = 0

    # 1. Prepare Reference Envelopes (Ch1)
    # Find which filename corresponds to Channel 1
    ref_filename = next((k for k in channel_dataframes.keys() if 'Ch1' in k), None)
    if not ref_filename:
        print("Error: Could not find Reference Channel (Ch1) in loaded files.")
        return
    
    print(f"Using {ref_filename} as reference for synchronization...")
    
    ch1_meta = indices_df[indices_df['input filename'] == ref_filename]
    envelopes_ref = {} # {count: envelope_array}

    for _, row in ch1_meta.iterrows():
        cnt = row['count']
        sig = load_waveform(ref_filename, row['start idx'], row['end idx'], channel_dataframes)
        env = process_envelope(sig)
        if env is not None:
            envelopes_ref[cnt] = env
            # Ch1 items group with themselves
            indices_df.loc[(indices_df['input filename'] == ref_filename) & (indices_df['count'] == cnt), 'Grouping_Envelope_Analysis'] = cnt

    # 2. Match other channels to Reference
    for ch_name in channel_dataframes.keys():
        if ch_name == ref_filename: continue
        
        print(f"Matching {ch_name} against reference groups...")
        
        # Get all metadata for this channel
        ch_meta = indices_df[indices_df['input filename'] == ch_name].copy()
        if ch_meta.empty: continue
        
        # Pre-calculate envelopes for this channel to avoid redundant processing
        envelopes_target = {} # {count: envelope_array}
        for _, row in ch_meta.iterrows():
            cnt = row['count']
            sig = load_waveform(ch_name, row['start idx'], row['end idx'], channel_dataframes)
            env = process_envelope(sig)
            if env is not None:
                envelopes_target[cnt] = env

        # Track which counts in this channel have been assigned
        assigned_counts = set()

        for ref_cnt, ref_env in envelopes_ref.items():
            best_D = np.inf
            best_target_cnt = None

            for target_cnt, target_env in envelopes_target.items():
                if target_cnt in assigned_counts:
                    continue
                
                # Align and calculate similarity
                _, target_aligned, aligned_idx = align_signals(ref_env, target_env)
                D = calculate_similarity_D(ref_env, target_aligned, aligned_idx)
                
                if D < best_D:
                    best_D = D
                    best_target_cnt = target_cnt

            # Only assign if it's the best match AND below the threshold
            if best_target_cnt is not None and best_D < SIMILARITY_THRESHOLD:
                indices_df.loc[
                    (indices_df['input filename'] == ch_name) & (indices_df['count'] == best_target_cnt),
                    'Grouping_Envelope_Analysis'
                ] = ref_cnt
                assigned_counts.add(best_target_cnt)
                # print(f"  Ref #{ref_cnt} -> {ch_name} #{best_target_cnt} (D={best_D:.4f})")

    # Save consolidated results
    indices_df.to_csv(INDICES_FILE, index=False)
    print(f"\nGrouping complete. Results saved to {INDICES_FILE}")
    return
'''