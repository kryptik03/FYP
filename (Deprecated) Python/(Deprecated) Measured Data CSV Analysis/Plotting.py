import pandas as pd
import matplotlib.pyplot as plt
import os

# Folder path
folder = 'Chiang_Void'

# Channels
channels = ['Ch1', 'Ch2', 'Ch3', 'Ch4']

for ch in channels:
    # Create figure for each channel
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for num in ['000', '001', '002']:
        filename = f'8kV_Void{num}_{ch}.csv'
        filepath = os.path.join(folder, filename)
        
        # Read CSV without header
        df = pd.read_csv(filepath, header=None)
        df[1] = df[1] - df[1].mean()

        # Extract time and voltage (assuming first 3 columns already removed)
        time = df[0]
        voltage = df[1]
        
        # Filter data to time range -0.5E-7 to 6E-7
        mask = (time >= -0.5e-7) & (time <= 6e-7)
        time_filtered = time[mask]
        voltage_filtered = voltage[mask]
        
        # Plot
        ax.plot(time_filtered, voltage_filtered, label=f'Void{num}', linewidth=0.3)
    
    ax.set_title(f'Channel {ch}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'waveforms_Void_{ch}.png')
