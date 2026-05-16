"""
dataset_exp02.py
==========================
PyTorch Dataset for the classification-only pipeline.

Unlike DetectionDataset (which indexes by scene/channel and builds YOLO grids),
this dataset indexes by individual PD pulses. For each pulse in the HDF5 labels,
it cuts the full-resolution raw waveform at (Start_Idx, End_Idx) and returns a
fixed-length window ready for a 1D CNN classifier.

There is NO decimation — the full-resolution signal is used.

Index entry: (shard_path, scene_idx, ch_idx, start_idx, end_idx, class_id)
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """
    Cuts individual PD pulse windows from raw HDF5 shards and returns
    (signal_window, class_id) pairs.

    Args
    ----
    root_path     : Path to the dataset node folder (same as DetectionDataset).
    shard_ids     : List of shard numbers to include.
    max_pulse_len : Fixed output window length in raw samples.
                    Short windows are centre-padded with zeros.
                    Long windows are centre-cropped.
    """

    # Label row constants (must match BaseDataset / generation script)
    ROW_SCENE_ID   = 0
    ROW_CHANNEL_ID = 1
    ROW_CLASS_ID   = 2
    ROW_PULSE_ID   = 3
    ROW_TOA_IDX    = 4
    ROW_START_IDX  = 5
    ROW_END_IDX    = 6

    def __init__(self, root_path: str, shard_ids: list, max_pulse_len: int = 4096):
        super().__init__()
        self.root_path     = root_path
        self.shard_ids     = shard_ids
        self.max_pulse_len = max_pulse_len

        # Each entry: (shard_path, scene_idx, ch_idx, start_idx, end_idx, class_id)
        self.index: list[tuple] = []
        self._build_index()

    # ------------------------------------------------------------------ #
    # Index construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_index(self):
        """
        Reads labels from every requested shard and registers one index
        entry per PD pulse. Called once at construction — O(num_shards).
        first_file = True
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
                raise FileNotFoundError(
                    f"[ClassificationDataset] Shard not found: {shard_path}"
                )
            with h5py.File(shard_path, "r") as f:
                if first_file:
                    self.time_resolution_s = f.attrs.get("time_resolution_s", 1e-11)
                    first_file = False
                    
                if "labels" not in f or f["labels"].shape[1] == 0:
                    continue
                labels = f["labels"][:]   # (7, N_pulses)

            for k in range(labels.shape[1]):
                self.index.append((
                    shard_path,
                    int(labels[self.ROW_SCENE_ID,   k]),
                    int(labels[self.ROW_CHANNEL_ID, k]),
                    int(labels[self.ROW_START_IDX,  k]),
                    int(labels[self.ROW_END_IDX,    k]),
                    int(labels[self.ROW_CLASS_ID,   k]),
                ))

    # ------------------------------------------------------------------ #
    # Window helpers                                                       #
    # ------------------------------------------------------------------ #

    def _pad_or_crop(self, signal: np.ndarray) -> np.ndarray:
        """Centre-pad (zeros) or centre-crop to self.max_pulse_len."""
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
        """Per-sample z-score normalisation. Guards against zero-energy windows."""
        mu  = signal.mean()
        std = signal.std()
        if std < 1e-9:
            return signal - mu
        return (signal - mu) / std

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        shard_path, scene_idx, ch_idx, start_idx, end_idx, class_id = self.index[idx]

        with h5py.File(shard_path, "r") as f:
            # Clamp indices to valid range
            n_samples = f["scenes"].shape[2]
            start_idx = max(0, min(start_idx, n_samples - 1))
            end_idx   = max(start_idx + 1, min(end_idx + 1, n_samples))
            signal    = f["scenes"][scene_idx, ch_idx, start_idx:end_idx].astype(np.float32)

        signal = self._pad_or_crop(signal)
        signal = self._normalise(signal)
        signal_tensor = torch.from_numpy(signal).unsqueeze(0)   # (1, max_pulse_len)
        label_tensor  = torch.tensor(class_id, dtype=torch.long)
        return signal_tensor, label_tensor
