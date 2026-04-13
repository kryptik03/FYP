import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import scipy.stats
import numpy as np

folder = 'Isolated_Waveforms'


# Channels
channels = {'Ch1':0, 'Ch2':1, 'Ch3':2, 'Ch4':3}

fig, ax = plt.subplots(4,1)

for ch,i in channels.items():
    for num in ['000', '001', '002']:
        filename = f'15kV_2PDs{num}_{ch}_Isolated.csv'
        filepath = os.path.join(folder, filename)
        
        # Read CSV without header
        df = pd.read_csv(filepath, header=None)
        
        # Extract time and voltage (assuming first 3 columns already removed)
        time = df[0]
        voltage = df[1]
        
        # Plot
        ax[i].plot(time, voltage, label=f'Incision{num}', linewidth=0.3)
    
    ax[i].set_title(f'Channel {ch}')
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel('Voltage (V)')
    ax[i].legend()
    ax[i].grid(True)

plt.tight_layout()
plt.show()


'''
fig, ax = plt.subplots(3, 1)

ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Energy')

ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Amplitude')

ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel('Gradient')

channels = ['Ch1', 'Ch2', 'Ch3', 'Ch4']


for ch in channels:
    for num in ['001']:
        filename = f'17kV_Incision{num}_{ch}.csv'
        data = pd.read_csv(os.path.join(folder, filename), header=None, names=['Time', 'Amplitude'])
        # Calculate Energy (Amplitude squared)
        data['Energy'] = data['Amplitude'] ** 2
        
        # Calculate Rolling Average Energy
        window_size = 500
        data['Rolling_Energy'] = data['Energy'].rolling(window=window_size).mean()
        data['Rolling_Energy_Gradient'] = data['Rolling_Energy'].diff()
        
        mask = (data['Time'] >= -3.5e-7) & (data['Time'] <= 9e-7)
        ax[0].plot(data['Time'][mask], data['Rolling_Energy'][mask], label=f'{ch}_{num}')
        ax[1].plot(data['Time'][mask], data['Amplitude'][mask], label=f'{ch}_{num}', linewidth=0.3)
        ax[2].plot(data['Time'][mask], data['Rolling_Energy_Gradient'][mask], label=f'{ch}_{num}', linewidth=0.3)

ax[0].legend()
ax[1].legend()
ax[2].legend()
'''
'''
# Channels
channels = ['Ch1']


for num in ['002']:
    # Create figure for each channel
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for ch in channels:
        filename = f'17kV_Incision{num}_{ch}.csv'
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
        ax.plot(time_filtered, voltage_filtered, label=f'{ch}', linewidth=0.3)
    
    ax.set_title(f'Incision{num}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    
fig, ax = plt.subplots(1, 1)
for ch in channels:
    for num in ['002']:
        filename = f'17kV_Incision{num}_{ch}.csv'
        data = pd.read_csv(os.path.join(folder, filename), header=None, names=['Time', 'Amplitude'])
        mask = (data['Time'] >= -0.5e-7) & (data['Time'] <= 6e-7)
        data = data[mask]
        data['Amplitude'] = data['Amplitude'] - data['Amplitude'].mean()
        sns.kdeplot(data['Amplitude'], ax=ax, label=f'{ch}_{num}')
    ax.legend()


fig, ax_fft = plt.subplots(1, 1)
for ch in channels:
    for num in ['002']:
        filename = f'17kV_Incision{num}_{ch}.csv'
        filepath = os.path.join(folder, filename)
        
        # Read CSV without header
        df = pd.read_csv(filepath, header=None)
        df[1] = df[1] - df[1].mean()
        
        # Extract time and voltage
        time = df[0]
        voltage = df[1]

        # Filter data to time range -0.5E-7 to 6E-7
        mask = (time >= 6e-7) & (time <= 12e-7)
        voltage_filtered = voltage[mask].values
        time_filtered = time[mask].values
        
        if len(time_filtered) > 1:
            dt = time_filtered[1] - time_filtered[0]
            n = len(voltage_filtered)
            freq = np.fft.rfftfreq(n, d=dt)
            fft_vals = np.abs(np.fft.rfft(voltage_filtered))
            
            ax_fft.plot(freq, fft_vals, label=f'{ch}', linewidth=0.5)

    ax_fft.set_title(f'Incision{num} - Frequency Domain')
    ax_fft.set_xlabel('Frequency (Hz)')
    ax_fft.set_ylabel('Magnitude')
    ax_fft.legend()
    ax_fft.grid(True)
    ax_fft.set_xlim(0, 5e8) # Limit x-axis to 500 MHz for better visibility


'''
'''
fig, ax = plt.subplots(figsize=(12, 8))


# Read CSV without header
time = np.linspace(-0.5e-7, 6e-7, 10000)
voltage = np.random.normal(0,1, 10000)

# Plot
ax.plot(time, voltage, label='Random Data', linewidth=0.3)

ax.set_title('Random Data')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.legend()
ax.grid(True)

plt.tight_layout()
    
fig, ax = plt.subplots(1, 1)


sns.kdeplot(voltage, ax=ax, label='Random Data')
ax.set_yscale('log')
ax.legend()


fig, ax_fft = plt.subplots(1, 1)


if len(time) > 1:
    dt = time[1] - time[0]
    n = len(voltage)
    freq = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.abs(np.fft.rfft(voltage))
    
    ax_fft.plot(freq, fft_vals, label='Random Data', linewidth=0.5)

ax_fft.set_title(f'Random Data - Frequency Domain')
ax_fft.set_xlabel('Frequency (Hz)')
ax_fft.set_ylabel('Magnitude')
ax_fft.legend()
ax_fft.grid(True)
ax_fft.set_xlim(0, 5e8) # Limit x-axis to 500 MHz for better visibility

voltage = pd.Series(voltage)
print(f'Random data mean: {(voltage).mean():.6f}')
plt.show()
'''