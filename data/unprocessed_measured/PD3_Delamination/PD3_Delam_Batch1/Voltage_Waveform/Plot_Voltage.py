import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Define the file path relative to this script's directory if possible, 
# but for now we'll use the absolute path provided earlier.
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'pdData_PD3_Delam_8kV.txt')
output_file = os.path.join(script_dir, 'voltage_vs_time_plot.png')

# If the file doesn't exist in the same directory, fall back to the absolute path
if not os.path.exists(data_file):
    data_file = r'd:\Zee_Documents\Studies\Uni\Sem_8\KIE4002_FYP\Experiment 2 - 11 & 12 May 2026\PD3_Delamination\PD3_Delam_Batch1\Voltage_Waveform\pdData_PD3_Delam_8kV.txt'

print(f"Reading data from: {data_file}")

times = []
voltages = []

# Load the data using csv module
try:
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # skip header: time (s), MPD 600 1.1 (V)
        for row in reader:
            if len(row) >= 2:
                try:
                    times.append(float(row[0]))
                    voltages.append(float(row[1]))
                except ValueError:
                    continue
except FileNotFoundError:
    print(f"Error: Data file not found at {data_file}")
    exit(1)

if not times:
    print("Error: No data found in the file.")
    exit(1)

times = np.array(times)
voltages = np.array(voltages)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(times, voltages, color='#007bff', linewidth=1.5, label='Voltage Waveform')

# Aesthetics
plt.title('Voltage vs Time (PD3 Delamination 8kV)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Voltage (V)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig(output_file, dpi=300)
print(f"Plot successfully saved to: {output_file}")

# Also show the plot if running in an interactive environment
# plt.show()
