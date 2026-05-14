"""
dataset_exp03_contrastive.py
============================
PyTorch Dataset for Contrastive Embedding + Classification (exp03).

Yields triplets of raw signal windows:
    - anchor_signal   : (1, max_pulse_len)
    - positive_signal : (1, max_pulse_len)
    - negative_signal : (1, max_pulse_len)
    - class_id        : Scalar tensor (long)
"""

import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    """
    Builds an index of pulses and groups them by Pulse_Instance_ID to
    enable on-the-fly Triplet generation for contrastive metric learning.
    """

    ROW_SCENE_ID   = 0
    ROW_CHANNEL_ID = 1
    ROW_CLASS_ID   = 2
    ROW_PULSE_ID   = 3
    ROW_START_IDX  = 5
    ROW_END_IDX    = 6

    def __init__(self, root_path: str, shard_ids: list, max_pulse_len: int = 4096):
        super().__init__()
        self.root_path     = root_path
        self.shard_ids     = shard_ids
        self.max_pulse_len = max_pulse_len

        # Flat list of all pulses (Anchors)
        # Entry: (shard_path, scene_idx, ch_idx, start_idx, end_idx, class_id, inst_id)
        self.index: list[tuple] = []
        
        # Group pulses by instance for fast Positive lookups
        # Key: (shard_path, inst_id) -> Value: List of flat index integers
        self.instance_groups: dict[tuple, list[int]] = {}

        self._build_index()

    def _build_index(self):
        """Scans shards, builds the flat index, and groups by instance."""
        idx_counter = 0

        for shard_id in self.shard_ids:
            shard_path = os.path.join(
                self.root_path, f"synth_shard_{shard_id:02d}.h5"
            )
            # Fallback to measured data format if synthetic not found
            if not os.path.exists(shard_path):
                shard_path = os.path.join(
                    self.root_path, f"measured_shard_{shard_id:02d}.h5"
                )
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"[ContrastiveDataset] Shard not found: {shard_path}")
                
            with h5py.File(shard_path, "r") as f:
                if "labels" not in f or f["labels"].shape[1] == 0:
                    continue
                labels = f["labels"][:]   # (7, N_pulses)

            for k in range(labels.shape[1]):
                inst_id = int(labels[self.ROW_PULSE_ID, k])
                
                self.index.append((
                    shard_path,
                    int(labels[self.ROW_SCENE_ID,   k]),
                    int(labels[self.ROW_CHANNEL_ID, k]),
                    int(labels[self.ROW_START_IDX,  k]),
                    int(labels[self.ROW_END_IDX,    k]),
                    int(labels[self.ROW_CLASS_ID,   k]),
                    inst_id
                ))
                
                group_key = (shard_path, inst_id)
                if group_key not in self.instance_groups:
                    self.instance_groups[group_key] = []
                self.instance_groups[group_key].append(idx_counter)
                
                idx_counter += 1

    # ------------------------------------------------------------------ #
    # Window helpers                                                       #
    # ------------------------------------------------------------------ #

    def _pad_or_crop(self, signal: np.ndarray) -> np.ndarray:
        L = len(signal)
        if L >= self.max_pulse_len:
            start = (L - self.max_pulse_len) // 2
            return signal[start : start + self.max_pulse_len]
        pad_total = self.max_pulse_len - L
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(signal, (pad_left, pad_right), mode="constant", constant_values=0.0)

    @staticmethod
    def _normalise(signal: np.ndarray) -> np.ndarray:
        mu  = signal.mean()
        std = signal.std()
        if std < 1e-9:
            return signal - mu
        return (signal - mu) / std

    def _get_signal(self, flat_idx: int) -> torch.Tensor:
        """Reads and formats a single signal window."""
        shard_path, scene_idx, ch_idx, start_idx, end_idx, _, _ = self.index[flat_idx]
        
        with h5py.File(shard_path, "r") as f:
            n_samples = f["scenes"].shape[2]
            start_idx = max(0, min(start_idx, n_samples - 1))
            end_idx   = max(start_idx + 1, min(end_idx + 1, n_samples))
            signal    = f["scenes"][scene_idx, ch_idx, start_idx:end_idx].astype(np.float32)

        signal = self._pad_or_crop(signal)
        signal = self._normalise(signal)
        return torch.from_numpy(signal).unsqueeze(0)   # (1, max_pulse_len)

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        # 1. Anchor
        shard_path, _, _, _, _, class_id, inst_id = self.index[idx]
        anc_sig = self._get_signal(idx)
        
        # 2. Positive
        # Choose a different pulse from the same instance if possible
        group_key = (shard_path, inst_id)
        group_indices = self.instance_groups[group_key]
        if len(group_indices) > 1:
            pos_candidates = [i for i in group_indices if i != idx]
            pos_idx = random.choice(pos_candidates)
        else:
            # Fallback: if no other channels captured this pulse, use the anchor
            # (In a true contrastive setup with data augmentations, we'd add noise here)
            pos_idx = idx
            
        pos_sig = self._get_signal(pos_idx)
        
        # 3. Negative
        # Randomly sample until we find a pulse from a DIFFERENT instance
        while True:
            neg_idx = random.randint(0, len(self.index) - 1)
            neg_shard, _, _, _, _, _, neg_inst_id = self.index[neg_idx]
            if not (neg_shard == shard_path and neg_inst_id == inst_id):
                break
                
        neg_sig = self._get_signal(neg_idx)
        
        label_tensor = torch.tensor(class_id, dtype=torch.long)
        
        return anc_sig, pos_sig, neg_sig, label_tensor
