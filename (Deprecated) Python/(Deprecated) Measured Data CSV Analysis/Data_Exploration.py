import pandas as pd
import matplotlib.pyplot as plt
import os
'''
Void Ch1_000 from -0.5e-7 to 6e-7
Incision Ch1_001 from -0.5e-7 to 6e-7
2PD Ch1_000 from 203.2e-7 to 210.2e-7
'''

df = pd.read_csv(os.path.join('Chiang_Incision', '17kV_Incision001_Ch1.csv'), header=None)
df.columns = ['Time', 'Voltage_Incision']
df_temp = pd.read_csv(os.path.join('Chiang_Void', '8kV_Void000_Ch1.csv'), header=None)
df['Voltage_Void'] = df_temp[1]

df_temp = pd.read_csv(os.path.join('Chiang_2PD', '15kV_2PDs000_Ch1.csv'), header=None)
df_temp.columns = ['Time', 'Voltage_2PD_2']

# Filter data to time range -0.5E-7 to 6E-7
mask = (df['Time'] >= -0.5e-7) & (df['Time'] <= 6e-7)
df_filtered = df[mask].copy()

mask2 = (df_temp['Time'] >= 203.7e-7) & (df_temp['Time'] <= 210.2e-7)
df_temp_filtered = df_temp[mask2].copy()
df_filtered['Voltage_2PD_2'] = df_temp_filtered['Voltage_2PD_2'].values

fig, ax = plt.subplots(figsize=(12, 8))
# Plot
ax.plot(df_filtered['Time'], df_filtered['Voltage_Incision'], label=f'Incision001', linewidth=0.3)
# ax.plot(df_filtered['Time'], df_filtered['Voltage_Void'], label=f'Void000', linewidth=0.3)
ax.plot(df_filtered['Time'], df_filtered['Voltage_2PD_2'], label=f'2PD000_2', linewidth=0.3)

ax.set_title('Comparison of Waveforms Ch1')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('Comparison Of Waveforms_Ch1_2.png')


'''
Void Ch1_001 from -0.5e-7 to 6e-7
Incision Ch1_001 from -0.5e-7 to 6e-7
2PD Ch1_002 from -0.6e-7 to 5.9e-7
'''

df = pd.read_csv(os.path.join('Chiang_Incision', '17kV_Incision001_Ch1.csv'), header=None)
df.columns = ['Time', 'Voltage_Incision']
df_temp = pd.read_csv(os.path.join('Chiang_Void', '8kV_Void001_Ch1.csv'), header=None)
df['Voltage_Void'] = df_temp[1]

df_temp = pd.read_csv(os.path.join('Chiang_2PD', '15kV_2PDs002_Ch1.csv'), header=None)
df_temp.columns = ['Time', 'Voltage_2PD_1']

# Filter data to time range -0.5E-7 to 6E-7
mask = (df['Time'] >= -0.5e-7) & (df['Time'] <= 6e-7)
df_filtered = df[mask].copy()

mask2 = (df_temp['Time'] >= -0.6e-7) & (df_temp['Time'] <= 5.9e-7)
df_temp_filtered = df_temp[mask2].copy()
df_filtered['Voltage_2PD_1'] = df_temp_filtered['Voltage_2PD_1'].values

fig, ax = plt.subplots(figsize=(12, 8))
# Plot
# ax.plot(df_filtered['Time'], df_filtered['Voltage_Incision'], label=f'Incision001', linewidth=0.3)
ax.plot(df_filtered['Time'], df_filtered['Voltage_Void'], label=f'Void001', linewidth=0.3)
ax.plot(df_filtered['Time'], df_filtered['Voltage_2PD_1'], label=f'2PD002_1', linewidth=0.3)

ax.set_title('Comparison of Waveforms Ch1')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('Comparison Of Waveforms_Ch1_1.png')

'''
Void Ch1_001 from -0.5e-7 to 6e-7
Incision Ch1_001 from -0.5e-7 to 6e-7
2PD Ch1_001 from -0.6e-7 to 6e-7
'''

df = pd.read_csv(os.path.join('Chiang_Incision', '17kV_Incision001_Ch1.csv'), header=None)
df.columns = ['Time', 'Voltage_Incision']
df_temp = pd.read_csv(os.path.join('Chiang_Void', '8kV_Void000_Ch1.csv'), header=None)
df['Voltage_Void'] = df_temp[1]

df_temp = pd.read_csv(os.path.join('Chiang_2PD', '15kV_2PDs001_Ch1.csv'), header=None)
df_temp.columns = ['Time', 'Voltage_2PD_1']

# Filter data to time range -0.5E-7 to 6E-7
mask = (df['Time'] >= -0.5e-7) & (df['Time'] <= 6e-7)
df_filtered = df[mask].copy()

mask2 = (df_temp['Time'] >= -0.6e-7) & (df_temp['Time'] <= 5.9e-7)
df_temp_filtered = df_temp[mask2].copy()
df_filtered['Voltage_2PD_1'] = df_temp_filtered['Voltage_2PD_1'].values

fig, ax = plt.subplots(figsize=(12, 8))
# Plot
# ax.plot(df_filtered['Time'], df_filtered['Voltage_Incision'], label=f'Incision001', linewidth=0.3)
ax.plot(df_filtered['Time'], df_filtered['Voltage_Void'], label=f'Void000', linewidth=0.3)
ax.plot(df_filtered['Time'], df_filtered['Voltage_2PD_1'], label=f'2PD001_1', linewidth=0.3)

ax.set_title('Comparison of Waveforms Ch1')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('Comparison Of Waveforms_Ch1_3.png')