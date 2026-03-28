import pandas as pd
import numpy as np
import os
import glob
import shutil
import scipy.stats as stats

# Configuration
PRE_TRIGGER_SAMPLES = 299  # Samples to capture before threshold
POST_TRIGGER_SAMPLES = 6000   # Samples to capture after threshold
OUTPUT_DIR = 'Isolated_Waveforms'


def process_file(filepath, pre_trigger_samples=PRE_TRIGGER_SAMPLES, 
                 post_trigger_samples=POST_TRIGGER_SAMPLES, output_dir=OUTPUT_DIR):
    """
    Reads a CSV file containing waveform data, isolates significant waveform segments 
    based on a dynamic threshold derived from the amplitude distribution (KDE), 
    and saves them as separate files if they contain sufficient energy.

    Args:
        filepath (str): Path to the input CSV file.
        pre_trigger_samples (int): Number of samples to include before the trigger point.
        post_trigger_samples (int): Number of samples to include after the trigger point.
        output_dir (str): Directory to save the isolated waveform CSVs.
    """
    try:
        # Load the CSV file
        # Assuming no header, columns: Time, Amplitude
        df = pd.read_csv(filepath, header=None, names=['Time', 'Amplitude'])
        # Center the amplitude around zero by subtracting the mean (DC offset removal)
        df['Amplitude'] = df['Amplitude'] - df['Amplitude'].mean()
        
        # --- Dynamic Threshold Calculation ---
        # Create a linear space for the KDE evaluation, from 0 to the max amplitude
        x = np.linspace(0, df['Amplitude'].max(), 500)
        # Calculate the Kernel Density Estimate (KDE) of the amplitude distribution
        kde = stats.gaussian_kde(df['Amplitude'])
        y = kde(x)
        max_kde = np.max(y)
        
        # Determine the cutoff amplitude where the density drops to 0.4% of the peak
        # This helps identify the 'noise floor' or the boundary of normal variations
        indices = np.where(y >= (max_kde * 0.004))[0]   
        idx = indices[-1] # Take the last index that meets the condition
        
        abs_amplitude = df['Amplitude'].abs()
        # Set threshold to 1.6 times that boundary amplitude
        threshold = x[idx]*1.6
        
        # if the mean criteria needs to be used instead of the dynamic threshold
        # uncomment the following lines

        # mean method is more sensitive to the threshold
        # not as good as the dynamic threshold method, sometimes the mean is
        # unpredictable and can cause false positives
        '''
        abs_amplitude = df['Amplitude'].abs()
        threshold = abs_amplitude.mean() * 6
        '''

        # Find all indices where amplitude magnitude exceeds the dynamic threshold
        trigger_indices = df.index[abs_amplitude > threshold].tolist()
        
        if not trigger_indices:
            print(f"Skipping {os.path.basename(filepath)}: Dynamic threshold {threshold:.6f} not reached.")
            return

        # Construct output filename
        original_filename = os.path.basename(filepath)
        filename_no_ext = os.path.splitext(original_filename)[0]
        
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
            isolated_energy = (isolated_df['Amplitude']**2).mean()

            # Save only if the isolated segment has significantly higher energy than global average (1.1x)
            if isolated_energy > (average_energy * 1.1):
                saved_count += 1
                # Format filename with 3 digits for sorting, e.g., 001, 002
                output_filename = f"{filename_no_ext}_Isolated_{saved_count:03d}.csv"
                output_path = os.path.join(output_dir, output_filename)
                isolated_df.to_csv(output_path, index=False, header=False)
            
            # Remove indices that are within the current window to avoid overlapping captures
            trigger_indices = [idx for idx in trigger_indices if idx >= end_idx]

        print(f"Processed {original_filename} -> Saved {saved_count} waveforms")
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # Clear and recreate output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"Created/Cleared directory: {OUTPUT_DIR}")
    
    # Recursively find all CSV files in current and subdirectories
    # Excluding the output directory to avoid re-processing generated files if run multiple times
    all_files = glob.glob('Measured_Waveforms/*.csv', recursive=True)
    
    # Filter out files in OUTPUT_DIR
    files_to_process = [f for f in all_files if OUTPUT_DIR not in f]
    
    print(f"Found {len(files_to_process)} CSV files to process.")
    
    for filepath in files_to_process:
        process_file(filepath, PRE_TRIGGER_SAMPLES, POST_TRIGGER_SAMPLES, OUTPUT_DIR)

if __name__ == "__main__":
    main()
