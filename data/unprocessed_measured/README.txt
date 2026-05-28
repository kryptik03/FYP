Unprocessed Measured PD Data
============================

Source: Lab measurements using Tektronix oscilloscopes and UHF sensors.
Format: Raw .wfm files organized by PD type, batch, and voltage.

Subdirectories:
- PD1_Void: Data for Void-type Partial Discharge.
- PD2_Incision: Data for Incision-type Partial Discharge.
- PD3_Delamination: Data for Delamination-type Partial Discharge.
- PD4_FeOx: Data for Iron Oxide contamination (Standard).
- PD5_FeOx_High: Data for Iron Oxide contamination (High Concentration).
- Noise / Noise_10M / Noise_RL12.5M: Various ambient noise measurements.

Pipeline Flow:
--------------
1. Isolation: `src/ingestion/process_and_isolate.py` processes these folders, performs wavelet denoising, and isolates individual PD pulses into .h5 files in `data/interim_measured/`.
2. Sharding: `src/ingestion/generate_measured_shards.py` takes the isolated pulses and superimposes them onto random noise to create deep learning shards in `data/raw/measured/`.

Sampling Info:
--------------
Sampled at 5 GHz (0.2ns resolution).

PD2, PD3 and PD5 at 3.7, 7, 0.1
PD4 at 3.7, 5, 0.1
