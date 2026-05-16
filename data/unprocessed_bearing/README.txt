CWRU Bearing Data Dataset
=========================

Source: Case Western Reserve University (CWRU) Bearing Data Center.
Link: https://engineering.case.edu/bearingdatacenter/download-data-file

Structure:
----------
This directory contains vibration data for bearings in .mat format.
The data is sampled at 12k samples per second for both Drive End and Fan End.

Subdirectories:
- 12k_Drive_End: Data collected from the drive-end bearing.
- 12k_Fan_End: Data collected from the fan-end bearing.

Naming Convention:
------------------
Files are named as [FaultType][FaultDiameter]_[Load].mat
- FaultType: IR (Inner Race), OR (Outer Race), B (Ball)
- FaultDiameter: 007, 014, 021 (inches)
- Load: 0, 1, 2, 3 (HP)

Preprocessing:
--------------
These files are used as input for `src/bearing_data_preprocessing/ingest_cwru.py`, which chunks the continuous signals into uniform 2048-sample scenes for deep learning classification.
