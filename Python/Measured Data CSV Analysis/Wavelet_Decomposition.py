import os
import csv
import numpy as np
import pandas as pd
import pywt
import glob

# Configuration
INPUT_FOLDER = 'Simulated_Waveforms'
INDICES_FILE = 'Waveform_Analysis.csv'
OUTPUT_FILE = 'Waveform_Analysis.csv'

def calculate_wavelet_energy(signal, wavelet='sym9', level=5):
    """
    Performs wavelet decomposition on a signal array and returns energy ratios.
    Matches the paper's methodology: Symlet 9, Level 5.
    """
    try:
        # Perform Wavelet Decomposition
        # Returns list: [cA5, cD5, cD4, cD3, cD2, cD1]
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Calculate Energy for each level (Sum of squared coefficients)
        # Energy = Sum(coeff^2)
        raw_energies = np.array([np.sum(c**2) for c in coeffs])
        
        # Calculate Total Energy
        total_energy = np.sum(raw_energies)
        
        # Avoid division by zero for empty/silent signals
        if total_energy == 0:
            return [0.0] * 6
            
        # Calculate Relative Energy (Distribution) as per Equation (1) & (2)
        # This normalizes the data for PCA later
        rel_energies = raw_energies / total_energy
        
        # Unpack for clarity (wavedec output order is Approx -> Detail N -> Detail 1)
        # coeffs[0] = A5
        # coeffs[1] = D5
        # coeffs[2] = D4
        # coeffs[3] = D3
        # coeffs[4] = D2
        # coeffs[5] = D1
        
        return rel_energies

    except Exception as e:
        print(f"Error during wavelet decomposition: {e}")
        return None

def main():
    # 1. Load the indices summary file
    if not os.path.exists(INDICES_FILE):
        print(f"Error: {INDICES_FILE} not found. Run Waveform_Isolation.py first.")
        return
        
    indices_df = pd.read_csv(INDICES_FILE)
    if indices_df.empty:
        print("No isolated waveforms found in indices file.")
        return

    # 2. Pre-load channel files into memory
    search_path = os.path.join(INPUT_FOLDER, '*.csv')
    channel_files = glob.glob(search_path)
    
    if not channel_files:
        print(f"No channel files found in {INPUT_FOLDER}")
        return
        
    print(f"Loading {len(channel_files)} channel files into memory...")
    channel_dataframes = {}
    for filepath in channel_files:
        filename = os.path.basename(filepath)
        try:
            df = pd.read_csv(filepath, header=None)
            channel_dataframes[filename] = df
            print(f"  Loaded {filename}")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    if not channel_dataframes:
        print("No dataframes loaded. Exiting.")
        return

    # 3. Process each isolated waveform
    print(f"Extracting features for {len(indices_df)} waveforms...")
    
    results = []
    
    for i, row in indices_df.iterrows():
        filename = row['input filename']
        start_idx = int(row['start idx'])
        end_idx = int(row['end idx'])
        count = row['count']
        
        # Extract signal from the pre-loaded dataframes
        try:
            df = channel_dataframes[filename]
            signal_segment = df.iloc[start_idx:end_idx, 1].values
            
            # Extract features
            energies = calculate_wavelet_energy(signal_segment)
            
            if energies is not None:
                res_row = {
                    'input filename': filename,
                    'count': count,
                    'Energy_A5': energies[0],
                    'Energy_D5': energies[1],
                    'Energy_D4': energies[2],
                    'Energy_D3': energies[3],
                    'Energy_D2': energies[4],
                    'Energy_D1': energies[5]
                }
                results.append(res_row)
                
        except Exception as e:
            print(f"Error processing {filename} #{count}: {e}")

    # 4. Save results to CSV
    if results:
        new_df = pd.DataFrame(results)
        
        if os.path.exists(OUTPUT_FILE):
            existing_df = pd.read_csv(OUTPUT_FILE)
            # Merge new results as columns. 
            # We assume 'Filename' is the common identifier for rows.
            if 'input filename' in existing_df.columns and 'count' in existing_df.columns:
                # Merge on Filename and Count to ensure rows match correctly
                # 'outer' join ensures we don't lose rows
                final_df = pd.merge(existing_df, new_df, on=['input filename', 'count'], how='outer')
            else:
                # Fallback: if keys are missing, just concatenate columns
                final_df = pd.concat([existing_df, new_df], axis=1)
            
            final_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Processing complete. Data appended as columns to {OUTPUT_FILE}")
        else:
            new_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Processing complete. Data saved to {OUTPUT_FILE}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()