import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_wfm(wfm_file):
    # Construct the CSV filename
    csv_file = wfm_file.replace('.wfm', '.csv')
    
    # Run the converter
    try:
        print(f"Converting {wfm_file}...")
        # Check if converter exists
        if not os.path.exists('ConvertTekWfm.exe'):
            print("Error: ConvertTekWfm.exe not found in current directory.")
            return

        subprocess.run(['./ConvertTekWfm.exe', wfm_file, '/CSV'], check=True, capture_output=True)
        
        # Load the CSV
        # Data is in the 4th and 5th columns (index 3 and 4)
        data = np.genfromtxt(csv_file, delimiter=',', usecols=(3, 4))
        
        # Check if data was loaded correctly
        if data.size == 0:
            print(f"Warning: No data loaded from {csv_file}")
            return

        time = data[:, 0]
        voltage = data[:, 1]
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(time, voltage)
        plt.title(f"Waveform: {wfm_file}")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
        # Clean up the CSV file
        if os.path.exists(csv_file):
            os.remove(csv_file)
            
    except Exception as e:
        print(f"Error processing {wfm_file}: {e}")

def main():
    # Find all .wfm files
    wfm_files = glob.glob("*.wfm")
    wfm_files.sort()
    
    if not wfm_files:
        print("No .wfm files found in the current directory.")
        return
    
    print(f"Found {len(wfm_files)} .wfm files.")
    
    for i, wfm_file in enumerate(wfm_files):
        print(f"[{i+1}/{len(wfm_files)}] Processing {wfm_file}")
        plot_wfm(wfm_file)
        
        if i < len(wfm_files) - 1:
            resp = input("Press Enter for next waveform (or 'q' to quit): ")
            if resp.lower() == 'q':
                break

if __name__ == "__main__":
    main()
