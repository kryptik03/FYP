import pandas as pd
import os
'''
DO NOT RUN THIS FILE DIRECTLY
This file is used to remove the first 3 columns from all CSV files in the specified folder
You may accidentally run this file and lose data if you are not careful
'''
# Folder path
folder = 'Chiang_2PD'

# Get all CSV files in the folder
csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

for filename in csv_files:
    filepath = os.path.join(folder, filename)
    
    # Read CSV without header
    df = pd.read_csv(filepath, header=None)
    
    # Drop the first 3 columns
    df.drop(columns=[0, 1, 2], inplace=True)
    
    # Save back to the same file
    df.to_csv(filepath, index=False, header=False)
    
    print(f"Processed {filename}")