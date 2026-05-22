"""
dataset_exp04_dec.py
====================
Multi-Source PyTorch Dataset for Deep Embedded Clustering (exp04).

Aggregates pulse signals from any combination of synthesized and measured
HDF5 shard collections. Each source is specified as a dict with keys:
    - type: "synthesized" or "measured"
    - path: root directory containing the shard files
    - train_shards / val_shards: list of integer shard IDs

Returned sample (for SimCLR Phase 1):
    (view1, view2, class_id, gt_inst_id, shard_path, start_idx, time_res)
    - view1, view2: two independently augmented windows of the same pulse.
    - class_id / gt_inst_id: retained for post-hoc cluster alignment ONLY.
      The model NEVER uses these during forward passes.

Returned sample (for DEC Phase 2 / inference):
    (signal, class_id, gt_inst_id, shard_path, start_idx, time_res)
"""

import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Signal Augmentation Utilities (used in SimCLR Phase 1)
# ---------------------------------------------------------------------------

def _augment(signal: np.ndarray) -> np.ndarray:
    """Apply a random chain of lightweight augmentations to a 1-D signal."""
    # 1. Additive Gaussian noise (SNR-aware: scale proportional to signal energy)
    sigma = signal.std() * random.uniform(0.01, 0.15)
    signal = signal + np.random.normal(0, sigma, signal.shape)

    # 2. Random amplitude scaling [0.8, 1.2]
    signal = signal * random.uniform(0.8, 1.2)

    # 3. Random time-shift (circular shift up to 5% of window length)
    max_shift = max(1, int(len(signal) * 0.05))
    shift = random.randint(-max_shift, max_shift)
    signal = np.roll(signal, shift)

    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DECDataset(Dataset):
    """
    Flat-indexed multi-source dataset that aggregates pulses from multiple
    HDF5 shard collections (synthesized and/or measured).

    Label column indices (matching MATLAB and generate_measured_shards.py):
        ROW 0 = Scene_ID
        ROW 1 = Channel_ID
        ROW 2 = Class_ID
        ROW 3 = Pulse_Instance_ID
        ROW 4 = TOA_Index
        ROW 5 = Start_Idx
        ROW 6 = End_Idx
    """

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
        max_pulse_len: int = 4096,
        augment: bool = False,  # True for SimCLR Phase 1
        denoise: bool = True,   # Apply wavelet denoising before returning signal
        wavelet: str = "db4",   # PyWavelets wavelet family
        wavelet_level: int = 4, # Decomposition level
        threshold_mode: str = "soft",  # "soft" or "hard" thresholding
    ):
        super().__init__()
        self.max_pulse_len   = max_pulse_len
        self.augment         = augment
        self.denoise         = denoise and _PYWT_AVAILABLE
        self.wavelet         = wavelet
        self.wavelet_level   = wavelet_level
        self.threshold_mode  = threshold_mode

        if denoise and not _PYWT_AVAILABLE:
            print("[DECDataset] Warning: pywt not installed. Wavelet denoising disabled. "
                  "Run: pip install PyWavelets")

        # Flat index: (shard_path, scene_idx, ch_idx, start_idx, end_idx, class_id, inst_id, time_res)
        self.index: list[tuple] = []

        self._build_index(sources, shard_key)

    def _build_index(self, sources: list, shard_key: str):
        for source in sources:
            src_type   = source["type"]           # "synthesized" or "measured"
            root_path  = os.path.abspath(source["path"])
            shard_ids  = source.get(shard_key, [])

            for shard_id in shard_ids:
                shard_path = os.path.join(root_path, f"shard_{shard_id:02d}.h5")
                if not os.path.exists(shard_path):
                    print(f"[DECDataset] Warning: Shard not found, skipping: {shard_path}")
                    continue

                with h5py.File(shard_path, "r") as f:
                    if "labels" not in f or f["labels"].shape[1] == 0:
                        continue

                    labels = f["labels"][:]          # Expected (7, N_pulses) — MATLAB format
                    time_res = float(np.array(f.attrs.get("time_resolution_s", 1e-11)).item())

                for k in range(labels.shape[1]):
                    self.index.append((
                        shard_path,
                        int(labels[self.ROW_SCENE_ID,   k]),
                        int(labels[self.ROW_CHANNEL_ID, k]),
                        int(labels[self.ROW_START_IDX,  k]),
                        int(labels[self.ROW_END_IDX,    k]),
                        int(labels[self.ROW_CLASS_ID,   k]),
                        int(labels[self.ROW_PULSE_ID,   k]),
                        time_res,
                    ))

    # ----------------------------------------------------------------------- #
    # Signal Helpers                                                            #
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _wavelet_denoise(
        signal: np.ndarray,
        wavelet: str,
        level: int,
        mode: str,
    ) -> np.ndarray:
        """
        Denoise a 1-D signal using Discrete Wavelet Transform (DWT) thresholding.

        Steps:
          1. Decompose the signal into approximation + detail coefficients.
          2. Estimate the noise floor from the finest detail coefficients
             using the median absolute deviation (MAD) estimator.
          3. Apply soft/hard thresholding to suppress sub-threshold detail
             coefficients (noise), leaving only significant transients.
          4. Reconstruct the signal from the thresholded coefficients.
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Noise estimate from the finest-scale detail (coeffs[-1])
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745   # MAD estimator
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        # Threshold all detail coefficient arrays (leave approximation unchanged)
        denoised_coeffs = [coeffs[0]]  # Keep approximation as-is
        for detail in coeffs[1:]:
            denoised_coeffs.append(
                pywt.threshold(detail, value=threshold, mode=mode)
            )

        return pywt.waverec(denoised_coeffs, wavelet).astype(np.float32)

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
        mu, std = signal.mean(), signal.std()
        return (signal - mu) / std if std > 1e-9 else signal - mu

    def _read_signal(self, shard_path, scene_idx, ch_idx, start_idx, end_idx) -> np.ndarray:
        with h5py.File(shard_path, "r") as f:
            n = f["scenes"].shape[2]
            s = max(0, min(start_idx, n - 1))
            e = max(s + 1, min(end_idx + 1, n))
            sig = f["scenes"][scene_idx, ch_idx, s:e].astype(np.float32)

        # Wavelet denoising applied BEFORE padding/normalisation
        if self.denoise:
            sig = self._wavelet_denoise(sig, self.wavelet, self.wavelet_level, self.threshold_mode)

        sig = self._pad_or_crop(sig)
        sig = self._normalise(sig)
        return sig

    # ----------------------------------------------------------------------- #
    # Dataset Protocol                                                          #
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        shard_path, scene_idx, ch_idx, start_idx, end_idx, class_id, inst_id, time_res = self.index[idx]
        sig = self._read_signal(shard_path, scene_idx, ch_idx, start_idx, end_idx)

        if self.augment:
            # Return two independently augmented views (SimCLR)
            view1 = torch.from_numpy(_augment(sig.copy())).unsqueeze(0)   # (1, L)
            view2 = torch.from_numpy(_augment(sig.copy())).unsqueeze(0)   # (1, L)
            return view1, view2, class_id, inst_id, shard_path, start_idx, float(time_res)
        else:
            signal = torch.from_numpy(sig).unsqueeze(0)                   # (1, L)
            return signal, class_id, inst_id, shard_path, start_idx, float(time_res)
