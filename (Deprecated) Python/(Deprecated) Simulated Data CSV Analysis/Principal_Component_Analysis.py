import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = 'Waveform_Analysis.csv'
FEATURE_COLS = ['Energy_A5', 'Energy_D5', 'Energy_D4', 'Energy_D3', 'Energy_D2', 'Energy_D1']

def perform_pca_and_visualize(input_csv=INPUT_CSV):
    # 1. Load Data
    print(f"Loading data from {input_csv}...")
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Please run Wavelet_Decomposition.py first.")
        return

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Separate features (Energies) and identifiers (Filenames)
    # Check if columns exist
    if not all(col in df.columns for col in FEATURE_COLS):
        print(f"Error: Input CSV missing required energy columns.")
        return

    X = df[FEATURE_COLS].values

    # 2. PCA Implementation
    # The paper aims to reduce dimensionality to 3-D space
    # while "minimizing lost information".
    
    print("Computing Principal Components...")
    
    # Initialize PCA with 3 components
    pca = PCA(n_components=3)
    
    # Fit and Transform
    # This performs the eigenvalue problem solution described in equations (3) and (4) [cite: 118-175]
    # It automatically centers the observations (subtracts mean) as required [cite: 119]
    principal_components = pca.fit_transform(X)

    # 3. Analyze Variance
    # Check how much information is preserved
    explained_variance = pca.explained_variance_ratio_
    total_variance = np.sum(explained_variance) * 100
    print(f"Explained Variance by component: {explained_variance}")
    print(f"Total Information Retained: {total_variance:.2f}%")

    # 4. Store Results
    # Add the Principal Components as new columns to the original DataFrame
    df['PC1'] = principal_components[:, 0]
    df['PC2'] = principal_components[:, 1]
    df['PC3'] = principal_components[:, 2]

    # Save back to the original CSV
    df.to_csv(input_csv, index=False)
    print(f"PCA results appended to {input_csv}")

    # 5. Visualization (3D Plot)
    # Replicates the plotting style of Figure 6 and Figure 9 
    print("Generating 3D Plot...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    # s=1 makes points small/sparse similar to the paper's dense plots
    # c='k' makes points black, similar to paper figures
    ax.scatter(df['PC1'], df['PC2'], df['PC3'], s=2, c='k', alpha=0.5)

    # Label axes as per paper figures
    ax.set_xlabel('1st Principal component')
    ax.set_ylabel('2nd Principal component')
    ax.set_zlabel('3rd Principal component')
    ax.set_title(f'3D PCA Projection (Retained Variance: {total_variance:.1f}%)')

    plt.show()

if __name__ == "__main__":
    perform_pca_and_visualize()