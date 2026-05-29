"""
dataset_exp08_dec.py
====================
Multi-Source PyTorch Dataset for Semi-Supervised Deep Embedded Clustering (Exp08).

KEY CHANGES vs Exp07 (STFT-based dataset)
------------------------------------------
1.  Loads `scenes_bispectra` (pre-computed 2D bispectrum grids) instead of
    `scenes_stft` (short-time Fourier transform spectrograms).

2.  REMOVED: STFT-specific time-slicing logic (hop_length, s_t, e_t
    conversion, pad_or_crop_2d for time-axis).

3.  ADDED: *** AdaptiveMaxPool2d downsampling ***
    The bispectrum grid is (2049, 2049) — far too large to feed directly into
    a ViT.  We downsample via:
        torch.nn.functional.adaptive_max_pool2d(grid, (224, 224))
    This is the "Adaptive Max Pool" step described in the Exp08 strategy.
    Max pooling (vs average) is chosen to preserve the sharpest spectral
    hotspots / energy peaks in the bispectrum surface.

4.  UPDATED: `_augment_2d` — SpecAugment masking now targets BOTH frequency
    axes (ω₁ and ω₂) of the bispectrum instead of time/freq axes of an STFT.

Returned sample (Phase 1 — augmented):
    (view1, view2, reported_class_id, gt_inst_id, shard_path, scene_idx, time_res, actual_class_id)
    All tensors in view1/view2 have shape (1, 224, 224).

Returned sample (Phase 2 / inference — single):
    (signal, reported_class_id, gt_inst_id, shard_path, scene_idx, time_res, actual_class_id)
    signal has shape (1, 224, 224).
"""

import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 2D Augmentation for Bispectrum Grids
# ---------------------------------------------------------------------------

def _augment_2d(bispectrum: np.ndarray) -> np.ndarray:
    """
    Apply a random chain of lightweight augmentations to a bispectrum image.

    The input `bispectrum` has shape (1, H, W) — a single-channel 2D image —
    where H=W=224 after AdaptiveMaxPool downsampling.

    AUGMENTATION STRATEGY:
    1. Additive Gaussian Noise  — randomly perturbs spectral magnitudes
                                   (signal-level ±1–10% std dev).
    2. Amplitude Scaling        — random global gain in [0.9, 1.1].
    3. ω₁-axis masking          — zeros out a contiguous band of ROWS
                                   (first bispectral frequency axis).
    4. ω₂-axis masking          — zeros out a contiguous band of COLUMNS
                                   (second bispectral frequency axis).

    NOTE: ω₁ and ω₂ masking are the bispectrum equivalent of SpecAugment
    frequency masking applied to both axes of the 2D frequency grid, since
    the bispectrum has NO time axis (it is a purely spectral transform).

    Args:
        bispectrum : np.ndarray of shape (1, H, W), dtype float32.

    Returns:
        Augmented np.ndarray of shape (1, H, W), dtype float32.
    """
    aug = bispectrum.copy()   # Work on a copy — never mutate the cached grid

    # ------------------------------------------------------------------
    # 1. Additive Gaussian Noise
    #    Scale noise to the input's own std dev so it is signal-adaptive.
    # ------------------------------------------------------------------
    sigma = aug.std() * random.uniform(0.01, 0.10)
    aug = aug + np.random.normal(0.0, sigma, aug.shape).astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Random Amplitude Scaling   [0.9, 1.1]
    # ------------------------------------------------------------------
    aug = aug * random.uniform(0.9, 1.1)

    # ------------------------------------------------------------------
    # 3. ω₁-axis masking (ROW masking — first bispectrum frequency axis)
    #    Masks a random contiguous stripe of rows (up to 25% of H).
    # ------------------------------------------------------------------
    if random.random() < 0.5:
        H = aug.shape[1]   # height (ω₁ axis)
        if H > 4:
            w1_width = random.randint(1, H // 4)
            w1_start = random.randint(0, H - w1_width)
            # Zero the stripe across all columns for this ω₁ band
            aug[:, w1_start : w1_start + w1_width, :] = 0.0

    # ------------------------------------------------------------------
    # 4. ω₂-axis masking (COLUMN masking — second bispectrum frequency axis)
    #    Masks a random contiguous stripe of columns (up to 25% of W).
    # ------------------------------------------------------------------
    if random.random() < 0.5:
        W = aug.shape[2]   # width (ω₂ axis)
        if W > 4:
            w2_width = random.randint(1, W // 4)
            w2_start = random.randint(0, W - w2_width)
            # Zero the stripe across all rows for this ω₂ band
            aug[:, :, w2_start : w2_start + w2_width] = 0.0

    return aug.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DECDataset_Exp08(Dataset):
    """
    Multi-source bispectrum dataset for Exp08 (SupCon + Semi-Supervised DEC).

    *** KEY STEP — AdaptiveMaxPool2d ***
    Each bispectrum grid read from disk is (F, F) where F = n_fft//2+1 = 2049.
    Before returning to the DataLoader, we apply:
        F.adaptive_max_pool2d(tensor, (224, 224))
    to produce a fixed (1, 224, 224) image. This makes it compatible with
    torchvision ViT models while preserving the highest-energy frequency pairs.

    Label row indices (same schema as all previous Exp datasets):
    """
    ROW_SCENE_ID   = 0
    ROW_CHANNEL_ID = 1
    ROW_CLASS_ID   = 2
    ROW_PULSE_ID   = 3
    ROW_TOA_IDX    = 4
    ROW_START_IDX  = 5
    ROW_END_IDX    = 6

    # Target output resolution after AdaptiveMaxPool2d
    # 224 × 224 is the canonical input size for ViT-B/16 and ViT-B/32.
    TARGET_SIZE: int = 224

    def __init__(
        self,
        sources: list,             # List of source dicts from YAML (each has path + shard lists)
        shard_key: str,            # "train_shards" or "val_shards"
        max_pulse_len: int = 4096, # Kept for API consistency with Exp07; not used for time-slicing
        augment: bool = False,     # True → return (view1, view2, ...) for SupCon Phase 1
        label_fraction: float = 0.10,  # Fraction of labels exposed for pairwise constraints
    ):
        super().__init__()
        self.max_pulse_len  = max_pulse_len   # Stored for reference / config logging
        self.augment        = augment
        self.label_fraction = label_fraction

        # self.index entries are tuples:
        #   (shard_path, scene_idx, ch_idx, reported_class, actual_class, inst_id, time_res)
        self.index: list[tuple] = []
        self._build_index(sources, shard_key)

    def _build_index(self, sources: list, shard_key: str):
        """
        Walk every shard in every source, read the labels array, and build a
        flat list of (shard_path, scene_idx, ch_idx, ...) tuples.

        Stratified Label Exposure: exactly `label_fraction` of pulses within
        each individual shard have their ground-truth class revealed.  The rest
        receive reported_class = -1 (unlabeled, used in SupCon / pairwise logic).
        """
        for source in sources:
            root_path = os.path.abspath(source["path"])
            shard_ids = source.get(shard_key, [])

            for shard_id in shard_ids:
                shard_path = os.path.join(root_path, f"shard_{shard_id:02d}.h5")
                if not os.path.exists(shard_path):
                    print(
                        f"[DECDataset_Exp08] Warning: shard not found, skipping: {shard_path}"
                    )
                    continue

                with h5py.File(shard_path, "r") as f:
                    # Validate required datasets are present
                    if "labels" not in f or f["labels"].shape[1] == 0:
                        continue

                    if "scenes_bispectra" not in f:
                        print(
                            f"[DECDataset_Exp08] Warning: 'scenes_bispectra' not found "
                            f"in {shard_path}. Did you run extract_bispectra.py?"
                        )
                        continue

                    labels    = f["labels"][:]    # (7, N_pulses)
                    time_res  = float(
                        np.array(f.attrs.get("time_resolution_s", 1e-11)).item()
                    )

                num_pulses  = labels.shape[1]
                num_labeled = int(num_pulses * self.label_fraction)

                # Stratified label exposure: shuffle booleans so exposed pulses
                # are spread randomly within this shard.
                is_labeled_flags = [True] * num_labeled + [False] * (num_pulses - num_labeled)
                random.shuffle(is_labeled_flags)

                for k in range(num_pulses):
                    actual_class_id  = int(labels[self.ROW_CLASS_ID,  k])
                    is_labeled       = is_labeled_flags[k]
                    reported_class   = actual_class_id if is_labeled else -1

                    self.index.append((
                        shard_path,
                        int(labels[self.ROW_SCENE_ID,   k]),   # scene_idx
                        int(labels[self.ROW_CHANNEL_ID, k]),   # ch_idx
                        reported_class,                         # masked label
                        actual_class_id,                        # ground truth
                        int(labels[self.ROW_PULSE_ID,   k]),   # pulse / inst ID
                        time_res,
                    ))

    @staticmethod
    def _normalise(grid: np.ndarray) -> np.ndarray:
        """
        Zero-mean, unit-variance normalisation across the 2D grid.
        If std ≈ 0 (flat surface), only subtract the mean to avoid div-by-zero.
        """
        mu, std = grid.mean(), grid.std()
        return (grid - mu) / std if std > 1e-9 else grid - mu

    def _read_bispectrum(
        self,
        shard_path: str,
        scene_idx: int,
        ch_idx: int,
    ) -> np.ndarray:
        """
        Load the pre-computed bispectrum grid for a single (scene, channel) and
        apply AdaptiveMaxPool downsampling + normalisation.

        Returns:
            np.ndarray of shape (1, TARGET_SIZE, TARGET_SIZE) = (1, 224, 224), float32.
        """
        with h5py.File(shard_path, "r") as f:
            # scenes_bispectra has shape (N_scenes, N_channels, F, F)
            # We slice out one (F, F) grid.
            grid = f["scenes_bispectra"][scene_idx, ch_idx, :, :].astype(np.float32)
            # grid shape: (F, F) e.g. (2049, 2049)

        # Add channel + batch dims for adaptive_max_pool2d: (1, 1, F, F)
        grid_t = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)   # (1, 1, F, F)

        # *** KEY STEP — Adaptive Max Pool ***
        # Downsample from (F, F) → (224, 224) preserving highest-energy peaks.
        # This is memory-safe and device-agnostic (runs on CPU here, before batching).
        grid_t = F.adaptive_max_pool2d(grid_t, (self.TARGET_SIZE, self.TARGET_SIZE))
        # grid_t shape: (1, 1, 224, 224)

        # Remove the batch dim; keep the channel dim → (1, 224, 224)
        grid_np = grid_t.squeeze(0).numpy()   # (1, 224, 224)

        # Normalise to zero-mean unit-variance for stable training
        grid_np = self._normalise(grid_np)

        return grid_np   # float32, shape (1, 224, 224)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        (shard_path, scene_idx, ch_idx,
         reported_class, actual_class, inst_id, time_res) = self.index[idx]

        # Load and downsample the bispectrum grid
        grid = self._read_bispectrum(shard_path, scene_idx, ch_idx)
        # grid: np.float32, shape (1, 224, 224)

        if self.augment:
            # SupCon Phase 1: return two independently augmented views
            view1 = torch.from_numpy(_augment_2d(grid))  # (1, 224, 224)
            view2 = torch.from_numpy(_augment_2d(grid))  # (1, 224, 224)
            return (
                view1,
                view2,
                reported_class,
                inst_id,
                shard_path,
                scene_idx,
                float(time_res),
                actual_class,
            )
        else:
            # Phase 2 / Inference: return single clean signal
            signal = torch.from_numpy(grid)   # (1, 224, 224)
            return (
                signal,
                reported_class,
                inst_id,
                shard_path,
                scene_idx,
                float(time_res),
                actual_class,
            )
