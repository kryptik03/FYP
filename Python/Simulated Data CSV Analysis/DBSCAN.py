import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = 'Waveform_Analysis.csv'
PCA_COLS = ['PC1', 'PC2', 'PC3']
EPSILON = 0.0002
MIN_SAMPLES = 5

def perform_dbscan_clustering(input_csv=INPUT_CSV):
    # 1. Load Data
    print(f"Loading data from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Please run Principal_Component_Analysis.py first.")
        return

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Ensure PCA columns exist
    if not all(col in df.columns for col in PCA_COLS):
        print(f"Error: Input CSV must contain {PCA_COLS} columns.")
        return

    X = df[PCA_COLS].values

    # 2. Configure and Run DBSCAN
    # The paper notes two degrees of freedom: epsilon (radius) and n_min (density) [cite: 216-217].
    # These parameters often need tuning based on your specific data scale.
    
    epsilon = EPSILON
    min_samples = MIN_SAMPLES
    
    print(f"Running DBSCAN with eps={epsilon} and min_samples={min_samples}...")
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    
    # Fit the model and get labels
    # Labels will be 0, 1, 2... for clusters.
    # Label -1 indicates NOISE (outliers).
    clusters = dbscan.fit_predict(X)
    
    # 3. Process Results
    # Append cluster IDs to the dataframe
    df['Cluster_ID_DBSCAN'] = clusters
    
    # Count distribution
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")

    # Save back to original CSV
    df.to_csv(input_csv, index=False)
    print(f"Clustered data saved back to {input_csv}")

    # 4. Visualization (3D Plot with Clusters)
    # This replicates the visual verification described in Section 2.2/2.3 [cite: 178-179].
    print("Generating 3D Cluster Plot...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get unique labels
    unique_labels = set(clusters)
    
    # Define colors (using a colormap)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise, similar to unclassified points in paper figures
            col = [0, 0, 0, 1]
            label = "Noise"
            marker = 'x'
            size = 10
        else:
            label = f"Cluster {k}"
            marker = 'o'
            size = 20

        # Select data for this cluster
        class_member_mask = (clusters == k)
        xyz = X[class_member_mask]
        
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                   c=[col], s=size, label=label, marker=marker, alpha=0.6)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'DBSCAN Clustering (eps={epsilon}, min_samples={min_samples})')
    ax.legend()

    plt.show()

def perform_simple_clustering(input_csv=INPUT_CSV):
    # 1. Load Data
    print(f"Loading data from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Please run Principal_Component_Analysis.py first.")
        return

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Ensure PCA columns exist
    if not all(col in df.columns for col in PCA_COLS):
        print(f"Error: Input CSV must contain {PCA_COLS} columns.")
        return

    ch1_df = df[df['input filename'].str.contains('ch1', case=False, na=False)]
    ch2_df = df[df['input filename'].str.contains('ch2', case=False, na=False)]
    ch3_df = df[df['input filename'].str.contains('ch3', case=False, na=False)]
    ch4_df = df[df['input filename'].str.contains('ch4', case=False, na=False)]
    
    ch1_df['Simple PCA Feature Cluster ID'] = ch1_df['count']
    ch2_df['Simple PCA Feature Cluster ID'] = 0
    ch3_df['Simple PCA Feature Cluster ID'] = 0
    ch4_df['Simple PCA Feature Cluster ID'] = 0

    for idx, ch1_row in ch1_df.iterrows():
        minDistance = np.inf
        minIdx = np.inf
        for idx2, ch2_row in ch2_df.iterrows():
            if (ch2_df.loc[idx2, 'Simple PCA Feature Cluster ID'] != 0):
                continue
            distance = np.linalg.norm(ch2_row[PCA_COLS].values - ch1_row[PCA_COLS].values)
            if distance < minDistance:
                minDistance = distance
                minIdx = idx2
        if minDistance != np.inf:
            ch2_df.loc[minIdx, 'Simple PCA Feature Cluster ID'] = ch1_row['Simple PCA Feature Cluster ID']
    
    for idx, ch1_row in ch1_df.iterrows():
        minDistance = np.inf
        minIdx = np.inf
        for idx2, ch3_row in ch3_df.iterrows():
            if (ch3_df.loc[idx2, 'Simple PCA Feature Cluster ID'] != 0):
                continue
            distance = np.linalg.norm(ch3_row[PCA_COLS].values - ch1_row[PCA_COLS].values)
            if distance < minDistance:
                minDistance = distance
                minIdx = idx2
        if minDistance != np.inf:
            ch3_df.loc[minIdx, 'Simple PCA Feature Cluster ID'] = ch1_row['Simple PCA Feature Cluster ID']

    for idx, ch1_row in ch1_df.iterrows():
        minDistance = np.inf
        minIdx = np.inf
        for idx2, ch4_row in ch4_df.iterrows():
            if (ch4_df.loc[idx2, 'Simple PCA Feature Cluster ID'] != 0):
                continue
            distance = np.linalg.norm(ch4_row[PCA_COLS].values - ch1_row[PCA_COLS].values)
            if distance < minDistance:
                minDistance = distance
                minIdx = idx2
        if minDistance != np.inf:
            ch4_df.loc[minIdx, 'Simple PCA Feature Cluster ID'] = ch1_row['Simple PCA Feature Cluster ID']
    
    df = pd.concat([ch1_df, ch2_df, ch3_df, ch4_df], ignore_index=True)
    df.to_csv(input_csv, index=False)
    print(f"Simple PCA clustering complete. Data saved back to {input_csv}")
    

        
    
    

if __name__ == "__main__":
    # perform_dbscan_clustering()
    perform_simple_clustering()