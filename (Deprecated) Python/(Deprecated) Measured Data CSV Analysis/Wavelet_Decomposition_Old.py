import os
import csv
import numpy as np
import pandas as pd
import pywt
import glob

def calculate_wavelet_energy(file_path, wavelet='sym9', level=5):
    """
    Reads a csv, performs wavelet decomposition, and returns energy ratios.
    Matches the paper's methodology: Symlet 9, Level 5.
    """
    try:
        # Fast load using pandas (assumes data is in the first column)
        # header=None assumes raw numbers; if headers exist, change to header=0
        df = pd.read_csv(file_path, header=None, dtype=np.float64)
        
        # Flatten to 1D array
        signal = df.iloc[:, 1].values

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
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    input_folder = 'Isolated_Waveforms'
    output_file = 'Wavelet_Features.csv'
    
    # Check if folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' not found.")
        return

    # Get list of all csv files
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    print(f"Found {len(csv_files)} files. Starting processing...")

    # Open output file in write mode
    with open(output_file, mode='w', newline='') as f_out: # opening the file in 'w' mode clears it
        writer = csv.writer(f_out)
        
        # Write Header
        # Columns: Filename, A5, D5, D4, D3, D2, D1
        header = ['Filename', 'Energy_A5', 'Energy_D5', 'Energy_D4', 
                  'Energy_D3', 'Energy_D2', 'Energy_D1']
        writer.writerow(header)

        # Process files iteratively (Low RAM usage)
        for i, file_path in enumerate(csv_files):
            filename = os.path.basename(file_path)
            
            # Extract features
            energies = calculate_wavelet_energy(file_path)
            
            if energies is not None:
                # Write row immediately to disk
                row = [filename] + list(energies)
                writer.writerow(row)

    print("Processing complete. Data saved to", output_file)

if __name__ == "__main__":
    main()