# Exp06 Architecture Walkthrough

We have successfully built and integrated the complete **Exp06** Semi-Supervised Deep Embedded Clustering pipeline. This architecture processes 2D Time-Frequency signals and utilizes a small fraction of your dataset's labels to steer the unsupervised clustering algorithm via pairwise constraints.

---

## 1. Feature Extraction (STFT)
Instead of feeding 1D waveforms directly into the neural network, Exp06 uses an explicit feature extraction phase to generate 2D magnitude spectrograms.

- **Script**: [extract_stft_features.py](file:///d:/Zee_Documents/Studies/Uni/Sem_8/KIE4002_FYP/Git_Cloned_Code/FYP/src/features/extract_stft_features.py)
- **Mechanism**: Reads the 1D `.h5` shards and processes them along the time axis using `scipy.signal.stft`. 
- **Metadata Integrity**: It strictly duplicates all root attributes, custom dataset properties, and label matrices into the newly generated files, appending specific STFT metadata (like `hop_length` and `nperseg`).
- **Lineage Registration**: It outputs the data to `data/features/stft_magnitude/<timestamp>-<origin>-<rootid>-<nodeid>` and seamlessly registers the `feature_extraction` stage in your SQLite database.

> [!TIP]
> Run `python src/features/extract_stft_features.py --input_dir <PATH_TO_RAW_H5> --parent_node_id <RAW_DATA_NODE>` to kick off the extraction process before attempting to train Exp06.

---

## 2. Dataset & Pairwise Constraints
The PyTorch Dataset is heavily modified to support both 2D imagery and Semi-Supervised pairwise logic.

- **Script**: [dataset_exp06_dec.py](file:///d:/Zee_Documents/Studies/Uni/Sem_8/KIE4002_FYP/Git_Cloned_Code/FYP/src/models/data/dataset_exp06_dec.py)
- **Time-Index Mapping**: It translates the 1D boundaries (`start_idx` to `end_idx`) into STFT frame bins by dividing by the `stft_hop_length`, effectively cropping the spectrogram to precisely the pulse duration.
- **Label Masking**: Based on the `label_fraction` configured in your `.yaml` file (default 10%), it artificially masks 90% of the dataset labels. Unlabelled items return a `reported_class_id` of `-1`, while labelled items return their actual `class_id`.

---

## 3. The 2D Backbone Model
Because the inputs are now 2D `(Batch, Channels, Freq, Time)` spectrograms, the 1D CNN used in Exp05 has been replaced.

- **Script**: [backbone_exp06.py](file:///d:/Zee_Documents/Studies/Uni/Sem_8/KIE4002_FYP/Git_Cloned_Code/FYP/src/models/models/backbones/backbone_exp06.py)
- **Architecture**: A lightweight 2D CNN with 3 convolutional blocks (Conv2d, BatchNorm, ReLU, MaxPool2d), culminating in an Adaptive Average Pool and a linear projection layer. It maintains the crucial L2-normalization for spherical cluster stability.

---

## 4. Semi-Supervised DEC Task
The task wrapper handles the two distinct phases of training, injecting supervised signals only where explicitly requested.

- **Script**: [task_exp06_dec.py](file:///d:/Zee_Documents/Studies/Uni/Sem_8/KIE4002_FYP/Git_Cloned_Code/FYP/src/models/tasks/task_exp06_dec.py)
- **Phase 1 (SimCLR)**: Continues to be entirely unsupervised. Uses 2D image augmentations (Gaussian noise, amplitude scaling, and SpecAugment-style time/frequency masking).
- **Phase 2 (DEC + Pairwise)**: Introduces the pairwise constraint loss. For any items in the batch where the `reported_class_id` is *not* `-1`:
  - **Must-Link**: If items share the same class ID, the loss actively pushes their soft-assignment vectors to be identical.
  - **Cannot-Link**: If items have different class IDs, the loss forces their soft-assignment dot product to zero, guaranteeing they end up in different clusters.

---

## 5. Hyperparameter Tuning (Optuna)
To optimize the new architecture, an Optuna script wraps the training loop.

- **Config**: [tune_exp06_dec.yaml](file:///d:/Zee_Documents/Studies/Uni/Sem_8/KIE4002_FYP/Git_Cloned_Code/FYP/src/models/configs/tune_exp06_dec.yaml)
- **Script**: [tune_exp06.py](file:///d:/Zee_Documents/Studies/Uni/Sem_8/KIE4002_FYP/Git_Cloned_Code/FYP/src/models/hyperparam_tuning/tune_exp06.py)
- **Tuning Space**: Dynamically optimizes `learning_rate_phase1`, `learning_rate_phase2`, `base_channels`, and importantly, the `pairwise_weight_gamma` (which dictates how strongly the Must-Link / Cannot-Link constraints pull against the unsupervised DEC loss).

> [!NOTE]
> The tune script fully integrates with Optuna's pruning mechanism. If an experiment performs poorly early on during the Phase 2 epochs, Optuna will prune it and skip to the next trial to save time.
