"""
dataset_exp07_dec.py
====================
Multi-Source PyTorch Dataset for Semi-Supervised Deep Embedded Clustering (exp07).

Aggregates 2D STFT features from multiple shard collections. 
Implements Semi-Supervised Pairwise Constraint logic by partially unmasking labels.
Includes Stratified Label Exposure: ensures exactly `label_fraction` of pulses 
within each individual shard are exposed.

Returned sample (for Phase 1):
    (view1, view2, reported_class_id, gt_inst_id, shard_path, start_idx, time_res, actual_class_id)

Returned sample (for Phase 2 / inference):
    (signal, reported_class_id, gt_inst_id, shard_path, start_idx, time_res, actual_class_id)
"""

import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 2D Signal Augmentation Utilities
# ---------------------------------------------------------------------------

def _augment_2d(spectrogram: np.ndarray) -> np.ndarray:
    """Apply a random chain of lightweight 2D augmentations to an STFT magnitude spectrogram."""
    aug = spectrogram.copy()
    
    # 1. Additive Gaussian noise
    sigma = aug.std() * random.uniform(0.01, 0.1)
    aug = aug + np.random.normal(0, sigma, aug.shape)
    
    # 2. Random amplitude scaling [0.9, 1.1]
    aug = aug * random.uniform(0.9, 1.1)
    
    # 3. Frequency / Time Masking (SpecAugment style)
    if random.random() < 0.5:
        # Time masking
        max_t = aug.shape[2]
        if max_t > 4:
            t_width = random.randint(1, max_t // 4)
            t_start = random.randint(0, max_t - t_width)
            aug[:, :, t_start:t_start+t_width] = 0
            
    if random.random() < 0.5:
        # Freq masking
        max_f = aug.shape[1]
        if max_f > 4:
            f_width = random.randint(1, max_f // 4)
            f_start = random.randint(0, max_f - f_width)
            aug[:, f_start:f_start+f_width, :] = 0

    return aug.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DECDataset_Exp07(Dataset):
    ROW_SCENE_ID   = 0
    ROW_CHANNEL_ID = 1
    ROW_CLASS_ID   = 2
    ROW_PULSE_ID   = 3
    ROW_TOA_IDX    = 4
    ROW_START_IDX  = 5
    ROW_END_IDX    = 6

    def __init__(
        self,
        sources: list,          # List of source dicts from YAML
        shard_key: str,         # "train_shards" or "val_shards"
        max_pulse_len: int = 4096, # 1D pulse length equivalent
        augment: bool = False,  # True for SupCon Phase 1
        label_fraction: float = 0.10, # Percentage of labels to unmask for Pairwise loss
    ):
        super().__init__()
        self.max_pulse_len = max_pulse_len
        self.augment       = augment
        self.label_fraction = label_fraction

        self.index: list[tuple] = []
        self._build_index(sources, shard_key)

    def _build_index(self, sources: list, shard_key: str):
        for source in sources:
            root_path  = os.path.abspath(source["path"])
            shard_ids  = source.get(shard_key, [])

            for shard_id in shard_ids:
                shard_path = os.path.join(root_path, f"shard_{shard_id:02d}.h5")
                if not os.path.exists(shard_path):
                    print(f"[DECDataset_Exp07] Warning: Shard not found, skipping: {shard_path}")
                    continue

                with h5py.File(shard_path, "r") as f:
                    if "labels" not in f or f["labels"].shape[1] == 0:
                        continue
                    
                    if "scenes_stft" not in f:
                        print(f"[DECDataset_Exp07] Warning: Shard does not contain STFT features: {shard_path}")
                        continue

                    labels = f["labels"][:]          # Expected (7, N_pulses)
                    time_res = float(np.array(f.attrs.get("time_resolution_s", 1e-11)).item())
                    hop_length = f.attrs.get("stft_hop_length", 128)

                num_pulses = labels.shape[1]
                num_labeled = int(num_pulses * self.label_fraction)
                
                # Create a list of boolean flags for unmasking (Stratified Label Exposure per shard)
                is_labeled_flags = [True] * num_labeled + [False] * (num_pulses - num_labeled)
                random.shuffle(is_labeled_flags)

                for k in range(num_pulses):
                    actual_class_id = int(labels[self.ROW_CLASS_ID, k])
                    is_labeled = is_labeled_flags[k]
                    reported_class_id = actual_class_id if is_labeled else -1
                    
                    self.index.append((
                        shard_path,
                        int(labels[self.ROW_SCENE_ID,   k]),
                        int(labels[self.ROW_CHANNEL_ID, k]),
                        int(labels[self.ROW_START_IDX,  k]),
                        int(labels[self.ROW_END_IDX,    k]),
                        reported_class_id,
                        actual_class_id,
                        int(labels[self.ROW_PULSE_ID,   k]),
                        time_res,
                        hop_length
                    ))

    def _pad_or_crop_2d(self, spectrogram: np.ndarray, target_time_bins: int) -> np.ndarray:
        F, T = spectrogram.shape
        if T >= target_time_bins:
            start = (T - target_time_bins) // 2
            return spectrogram[:, start : start + target_time_bins]
        
        pad_total = target_time_bins - T
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(spectrogram, ((0,0), (pad_left, pad_right)), mode="constant", constant_values=0.0)

    @staticmethod
    def _normalise(spectrogram: np.ndarray) -> np.ndarray:
        mu, std = spectrogram.mean(), spectrogram.std()
        return (spectrogram - mu) / std if std > 1e-9 else spectrogram - mu

    def _read_signal(self, shard_path, scene_idx, ch_idx, start_idx, end_idx, hop_length) -> np.ndarray:
        s_t = start_idx // hop_length
        e_t = end_idx // hop_length
        target_time_bins = self.max_pulse_len // hop_length
        
        with h5py.File(shard_path, "r") as f:
            n_t = f["scenes_stft"].shape[3]
            s = max(0, min(s_t, n_t - 1))
            e = max(s + 1, min(e_t + 1, n_t))
            stft_mag = f["scenes_stft"][scene_idx, ch_idx, :, s:e].astype(np.float32)

        stft_mag = self._pad_or_crop_2d(stft_mag, target_time_bins)
        stft_mag = self._normalise(stft_mag)
        
        return np.expand_dims(stft_mag, axis=0)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        shard_path, scene_idx, ch_idx, start_idx, end_idx, reported_class, actual_class, inst_id, time_res, hop_length = self.index[idx]
        
        sig = self._read_signal(shard_path, scene_idx, ch_idx, start_idx, end_idx, hop_length)

        if self.augment:
            view1 = torch.from_numpy(_augment_2d(sig))
            view2 = torch.from_numpy(_augment_2d(sig))
            return view1, view2, reported_class, inst_id, shard_path, start_idx, float(time_res), actual_class
        else:
            signal = torch.from_numpy(sig)
            return signal, reported_class, inst_id, shard_path, start_idx, float(time_res), actual_class
