import pandas as pd
import numpy as np
import os
import glob
import shutil
import scipy.stats as stats

# Configuration
PRE_TRIGGER_SAMPLES = 299  # Samples to capture before threshold
POST_TRIGGER_SAMPLES = 6000   # Samples to capture after threshold
OUTPUT_FILE = 'Waveform_Analysis.csv'


def process_file(df, filepath, results_list, pre_trigger_samples=PRE_TRIGGER_SAMPLES, 
                 post_trigger_samples=POST_TRIGGER_SAMPLES):
    """
    Processes a waveform dataframe, isolates significant waveform segments 
    based on a dynamic threshold, and appends their indices to a results list.

    Args:
        df (pd.DataFrame): Dataframe containing 'Time' and 'Amplitude' columns.
        filepath (str): Path to the input CSV file.
        results_list (list): List to store metadata of isolated waveforms.
        pre_trigger_samples (int): Number of samples to include before the trigger point.
        post_trigger_samples (int): Number of samples to include after the trigger point.
    """
    try:
        # Center the amplitude around zero by subtracting the mean (DC offset removal)
        # Note: This modifies the dataframe passed by reference
        df[1] = df[1] - df[1].mean()
        
        # --- Dynamic Threshold Calculation ---
        # Create a linear space for the KDE evaluation, from 0 to the max amplitude
        x = np.linspace(0, df[1].max(), 500)
        # Calculate the Kernel Density Estimate (KDE) of the amplitude distribution
        kde = stats.gaussian_kde(df[1])
        y = kde(x)
        max_kde = np.max(y)
        
        # Determine the cutoff amplitude where the density drops to 0.4% of the peak
        # This helps identify the 'noise floor' or the boundary of normal variations
        indices = np.where(y >= (max_kde * 0.004))[0]
        idx = indices[-1] # Take the last index that meets the condition
        
        abs_amplitude = df[1].abs()
        # Set threshold to 1.6 times that boundary amplitude
        threshold = x[idx]*1.6
        
        # if the mean criteria needs to be used instead of the dynamic threshold
        # uncomment the following lines

        # mean method is more sensitive to the threshold
        # not as good as the dynamic threshold method, sometimes the mean is
        # unpredictable and can cause false positives
        '''
        abs_amplitude = df[1].abs()
        threshold = abs_amplitude.mean() * 6
        '''

        # Find all indices where amplitude magnitude exceeds the dynamic threshold
        trigger_indices = df.index[abs_amplitude > threshold].tolist()
        
        if not trigger_indices:
            print(f"Skipping {os.path.basename(filepath)}: Dynamic threshold {threshold:.6f} not reached.")
            return

        # Construct output filename
        original_filename = os.path.basename(filepath)
        
        saved_count = 0
        # Calculate average energy of the entire signal
        average_energy = (abs_amplitude**2).mean()
        
        # --- Waveform Isolation Loop ---
        while trigger_indices:
            first_trigger_idx = trigger_indices[0]
            
            # Define the sample window around the trigger point
            start_idx = max(0, first_trigger_idx - pre_trigger_samples)
            end_idx = min(len(df), first_trigger_idx + post_trigger_samples + 1) # +1 for inclusive slicing in iloc
            
            # Filter the dataframe to extract the isolated segment
            isolated_df = df.iloc[start_idx:end_idx]
            # Calculate energy of the isolated segment
            isolated_energy = (isolated_df[1]**2).mean()

            # Store indices only if the isolated segment has significantly higher energy than global average (1.1x)
            if isolated_energy > (average_energy * 1.1):
                saved_count += 1

                PD_number = isolated_df[2].value_counts().idxmax()
                
                # Store the metadata in the results list (passed by reference)
                results_list.append({
                    'input filename': original_filename,
                    'count': saved_count,
                    'PD number': PD_number,
                    'start idx': start_idx,
                    'end idx': end_idx
                })
            
            # Remove indices that are within the current window to avoid overlapping captures
            trigger_indices = [idx for idx in trigger_indices if idx >= end_idx]

        print(f"Processed {original_filename} -> Saved {saved_count} waveforms")
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    # Recursively find all CSV files in current and subdirectories
    files_to_process = glob.glob('Simulated_Waveforms/*.csv', recursive=True)
    
    print(f"Found {len(files_to_process)} CSV files to process.")
    
    results = [] # This list will be passed by reference to accumulate results
    
    for filepath in files_to_process:
        try:
            # Load the CSV file here and pass the dataframe by reference
            df = pd.read_csv(filepath, header=None)
            process_file(df, filepath, results, PRE_TRIGGER_SAMPLES, POST_TRIGGER_SAMPLES)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # After processing all files, save the accumulated results to a single CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccessfully saved {len(results)} waveform entries to {OUTPUT_FILE}")
    else:
        print("\nNo waveforms isolated. Summary file not created.")

