import os
import pandas as pd
import glob

def process_waveforms():
    # Define directories
    source_dir = 'Isolated_Waveforms'
    output_dir = 'Simulated_Waveforms'

    # 1. Check and create Simulated_Waveforms directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dictionary to map user specified channel search strings types
    channels = ['Ch1', 'Ch2', 'Ch3', 'Ch4']
    
    # Initialize list to store metadata dictionaries
    metadata_rows = []

    # 4. Iterate through each channel type
    for channel in channels:
        # Create a list to hold the signal data for this channel
        # We will collect [time, signal] pairs/rows
        channel_signal_data = []
        
        # Counter for files processed for this channel
        count = 0
        
        # Label to use in metadata
        output_filename = f"{channel}.csv"
        
        # Get list of all files in source directory
        # We use a case-insensitive search logic manually
        # Get all csv files first
        all_files = glob.glob(os.path.join(source_dir, '*.csv'))
        
        # Filter files that contain the channel name (case-insensitive)
        # It is good practice to sort them to ensure deterministic order
        channel_files = sorted([f for f in all_files if channel.lower() in os.path.basename(f).lower()])
        
        for file_path in channel_files:
            file_name = os.path.basename(file_path)
            
            # Read the file
            # Assuming no header, columns 0 and 1 are time and signal
            try:
                df = pd.read_csv(file_path, header=None)
                if df.shape[1] < 2:
                    print(f"Skipping {file_name}: Not enough columns.")
                    continue
                
                # Extract time and signal (values)
                # It's faster to work with list of lists or arrays until the end
                time_col = df.iloc[:, 0].values
                signal_col = df.iloc[:, 1].values
                
                current_len = len(time_col)
                if current_len == 0:
                    continue

                # record start index (current length of the accumulated data)
                start_idx = len(channel_signal_data)
                
                # Add data to the main list
                # extend is cleaner than looping append
                # Create pairs
                new_rows = list(zip(time_col, signal_col))
                channel_signal_data.extend(new_rows)
                
                # record end index (current length)
                end_idx = len(channel_signal_data)
                
                # Add buffer row [0, 0]
                channel_signal_data.append((0, 0))
                
                # Update count
                count += 1
                
                # Determine PD number
                # "if 'void' is in the filename, then PD number is 'void'. else, if 'incision', then PD number is 'incision"
                file_name_lower = file_name.lower()
                if 'void' in file_name_lower:
                    pd_number = 'void'
                elif 'incision' in file_name_lower:
                    pd_number = 'incision'
                else:
                    pd_number = 'unknown' # Handling case where neither is present
                
                # Add metadata
                metadata_rows.append({
                    'input filename': output_filename,
                    'count': count,
                    'PD number': pd_number,
                    'start idx': start_idx,
                    'end idx': end_idx
                })

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        
        # Save the collected signal dataframe for this channel
        if channel_signal_data:
            signal_df = pd.DataFrame(channel_signal_data) # columns will be 0, 1
            save_path = os.path.join(output_dir, output_filename)
            # Save without header and index as requested (implied by "input filename" structure and common practice for signal processing csvs here)
            signal_df.to_csv(save_path, header=False, index=False)
            print(f"Saved {output_filename} with {len(signal_df)} rows.")

    # 5. Save the metadata dataframe
    if metadata_rows:
        metadata_df = pd.DataFrame(metadata_rows)
        # Ensure column order
        cols = ['input filename', 'count', 'PD number', 'start idx', 'end idx']
        metadata_df = metadata_df[cols]
        # User said: "save the metadata dataframe as 'Waveform_Analysis.csv'"
        # Assuming inside Simulated_Waveforms
        metadata_save_path = 'Waveform_Analysis.csv'
        metadata_df.to_csv(metadata_save_path, index=False)
        print(f"Saved metadata to {metadata_save_path}")

if __name__ == "__main__":
    process_waveforms()
