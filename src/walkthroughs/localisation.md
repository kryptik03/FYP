# Localisation Pipeline Implementation Walkthrough

The classification output format has been successfully expanded, and a brand new localisation script using Particle Swarm Optimization (PSO) and Time Difference of Arrival (TDOA) is now in place. 

## 1. Classification Output HDF5 Export

Both `predict_exp07.py` and `predict_exp06.py` have been modified. After calculating the predictions and metrics, the scripts will now iterate over all evaluated pulses and generate a `predictions.h5` file in the prediction output folder.

> [!NOTE] 
> The `.h5` file maps each individual pulse exactly to its parent `shard_path`, `pred_class_id`, `cluster_id`, and `pred_inst_id`. This creates a seamless bridge between the abstract clustered features and the physical raw waveforms needed for TDOA.

## 2. Localisation Pipeline (`localise.py`)

A new script `src/models/predictions/localise.py` was created to perform the physical TDOA coordinate mapping. 

### TDOA Calculation
- **Signal Grouping:** The pipeline first identifies valid clusters (those representing > 5% of the total dataset by default). Inside these clusters, it groups the pulses by `pred_inst_id`. 
- **4-Channel Extraction:** It rigorously filters these instances to ensure they contain exactly 4 signals, guaranteeing that exactly 1 signal comes from each of the 4 physical channels (Ch 0 to Ch 3).
- **Cross-Correlation:** For each valid 4-channel instance, it accesses the raw `.h5` dataset, pulls the exact $4096$-sample wide waveforms, and computes the TDOA for Channels 2, 3, and 4 relative to Channel 1 using full Cross-Correlation via `scipy.signal.correlate()`.
- **Dimensionality:** Because the TDOA shift is identified in samples, the script automatically queries the `time_res` variable (which is already mapped into seconds like $10^{-11} \text{s}$) and multiplies it by the sample shift to get `TDOA_N` strictly in seconds.

### Vectorised Particle Swarm Optimization (PSO)
> [!TIP]
> The deprecated PSO script evaluated 1,000 particles over 50,000 iterations using Python `for` loops. Doing this over hundreds of individual pulses would have taken hours. 

To make this feasible, the PSO objective function and position updates were completely **vectorised using NumPy**. It evaluates the positions and distances for all 1,000 particles simultaneously.
- **Math Correction:** The objective formula applies `speed_of_EM * TDOA_N`, naturally multiplying $m/s \times s$ to produce the correct residual distances in meters.
- **Averaging:** The algorithm computes the estimated 3D position for each valid pulse instance. Finally, it averages all instance positions within the cluster to generate a single, highly accurate physical coordinate per cluster.

### Output and Configuration
- **Lineage DB Integration:** The results are logged directly to a new Lineage Node under `localisation_output/`, linking backward to the exact `inf-xxx` prediction node that generated the `.h5`.
- **Customizable Ground Truths:** At the very top of `localise.py`, there is a `TRUE_LOCATIONS` dictionary mapping the `pred_class_id` (e.g. 3 for PD2, 4 for PD3) to their physical `[x, y, z]` arrays. You can edit these manually for any future experimental setups!

## How to use:
To run the localisation pipeline on your new `.h5` predictions:
```bash
python src/models/predictions/localise.py \
  --predictions_h5 data/classification_output/.../predictions.h5 \
  --raw_data_dir data/raw/measured/...
```
