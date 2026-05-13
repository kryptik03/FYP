import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Define the file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'fftData.txt')
output_file = os.path.join(script_dir, 'fft_plot.png')

# If the file doesn't exist in the same directory, fall back to the absolute path
if not os.path.exists(data_file):
    data_file = r'd:\Zee_Documents\Studies\Uni\Sem_8\KIE4002_FYP\Experiment 2 - 11 & 12 May 2026\PD3_Delamination\PD3_Delam_Batch1\Voltage_Waveform\fftData.txt'

print(f"Reading data from: {data_file}")

frequencies = []
amplitudes = []

# Load the data using csv module
try:
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # skip header: frequency (Hz),MPD 600 1.1 (dBm)
        for row in reader:
            if len(row) >= 2:
                try:
                    frequencies.append(float(row[0]))
                    amplitudes.append(float(row[1]))
                except ValueError:
                    continue
except FileNotFoundError:
    print(f"Error: Data file not found at {data_file}")
    exit(1)

if not frequencies:
    print("Error: No data found in the file.")
    exit(1)

frequencies = np.array(frequencies)
amplitudes = np.array(amplitudes)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(frequencies, amplitudes, color='#dc3545', linewidth=1.2, label='FFT Spectrum')

# Aesthetics
plt.title('Frequency Spectrum (PD3 Delamination 8kV)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Amplitude (dBm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig(output_file, dpi=300)
print(f"FFT plot successfully saved to: {output_file}")

# Optional: Set x-axis to log scale if frequency range is very wide
# plt.xscale('log')
# plt.savefig(output_file.replace('.png', '_log.png'), dpi=300)
