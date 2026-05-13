import wfm_oxide
import matplotlib.pyplot as plt
import numpy as np

file_path = "PD3_Delam_7kV_Batch1_000_Ch1.wfm"
try:
    # Based on dir(wfm_oxide), the class is WfmOxide
    wfm = wfm_oxide.WfmOxide(file_path)
    print(f"Model: {wfm.model}")
    print(f"Firmware: {wfm.firmware}")
    print(f"Sample rate: {wfm.sample_rate} Hz")
    print(f"X-origin: {wfm.x_origin} s")
    print(f"X-increment: {wfm.x_increment} s")
    print(f"Enabled channels: {wfm.enabled_channels}")
    
    # Get channel data. 
    # enabled_channels is likely a list or bitmask.
    # We'll try the first available channel.
    if wfm.enabled_channels:
        ch_idx = wfm.enabled_channels[0]
        channel_data = wfm.get_channel_data(ch_idx)
        times = wfm.get_time_axis()
        
        print(f"Data shape: {channel_data.shape}")
        
        plt.figure(figsize=(10, 4))
        plt.plot(times, channel_data)
        plt.title(f"Waveform: {file_path} (Channel {ch_idx})")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        # plt.show() # Uncomment for user execution
        plt.savefig("test_plot.png")
        print("Plot saved to test_plot.png")
    else:
        print("No enabled channels found.")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
