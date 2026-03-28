import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import os

def perform_dbscan_clustering(input_csv='Wavelet_Features.csv'):
    # 1. Load Data
    print(f"Loading data from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Ensure PCA columns exist
    pca_cols = ['PC1', 'PC2', 'PC3']
    if not all(col in df.columns for col in pca_cols):
        print(f"Error: Input CSV must contain {pca_cols} columns.")
        return

    X = df[pca_cols].values

    # 2. Configure and Run DBSCAN
    # The paper notes two degrees of freedom: epsilon (radius) and n_min (density) [cite: 216-217].
    # These parameters often need tuning based on your specific data scale.
    
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    epsilon = 0.07  # Trial value: Since inputs are normalized energies (0-1), PCA distances will be small.
    min_samples = 3 # Standard default, implies a cluster must have at least 5 waveforms.
    
    print(f"Running DBSCAN with eps={epsilon} and min_samples={min_samples}...")
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    
    # Fit the model and get labels
    # Labels will be 0, 1, 2... for clusters.
    # Label -1 indicates NOISE (outliers).
    clusters = dbscan.fit_predict(X)
    
    # 3. Process Results
    # Append cluster IDs to the dataframe
    df['Cluster_ID'] = clusters
    
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
    colors = plt.cm.Spectral(np.linspace(0, 5, len(unique_labels)*5))
    
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

if __name__ == "__main__":
    perform_dbscan_clustering()