"""
dataset_exp08_dec.py
====================
Multi-Source PyTorch Dataset for Semi-Supervised Deep Embedded Clustering (Exp08).

KEY CHANGES vs Exp07 (STFT-based dataset)
------------------------------------------
1.  Loads `pulses_bispectra` (pre-computed per-pulse 2D bispectrum grids) instead
    of `scenes_stft` (short-time Fourier transform spectrograms).

    CRITICAL DISTINCTION: Unlike the STFT, the Bispectrum destroys the time axis.
    It cannot be computed over a full scene and sliced afterwards. The feature
    extraction script (extract_bispectra.py) therefore computes one bispectrum per
    INDIVIDUAL ISOLATED PULSE, stored at `pulses_bispectra[k]` where `k` is the
    column index in the `labels` array. This dataset simply fetches by pulse index.

2.  REMOVED: STFT-specific time-slicing logic (hop_length, s_t, e_t conversion,
    pad_or_crop_2d for time-axis). The pulse is already a single compact grid.

3.  REMOVED: AdaptiveMaxPool2d. The Welch-method bispectrum grids are already
    compact: 129 × 129 pixels. We simply crop off one row and one column
    (grid[:128, :128]) to make the size exactly divisible by the ViT patch size
    of 16. No pooling is needed or applied.

4.  UPDATED: `_augment_2d` — SpecAugment masking now targets BOTH frequency
    axes (ω₁ and ω₂) of the bispectrum instead of time/freq axes of an STFT.

H5 Feature Shard Schema (produced by extract_bispectra.py):
    pulses_bispectra : float32  (N_pulses, 129, 129)
                       Row k = bispectrum of the pulse at labels[:, k].
    labels           : float32  (7, N_pulses) — same column-k correspondence.

Returned sample (Phase 1 — augmented):
    (view1, view2, reported_class_id, gt_inst_id, shard_path, pulse_idx, time_res, actual_class_id)
    All tensors in view1/view2 have shape (1, 128, 128).

Returned sample (Phase 2 / inference — single):
    (signal, reported_class_id, gt_inst_id, shard_path, pulse_idx, time_res, actual_class_id)
    signal has shape (1, 128, 128).
"""

import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 2D Augmentation for Bispectrum Grids
# ---------------------------------------------------------------------------

def _augment_2d(bispectrum: np.ndarray) -> np.ndarray:
    """
    Apply a random chain of lightweight augmentations to a bispectrum image.

    The input `bispectrum` has shape (1, H, W) — a single-channel 2D image —
    where H=W=128 after the 129→128 crop step.

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
            aug[:, :, w2_start : w2_start + w2_width] = 0.0

    return aug.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DECDataset_Exp08(Dataset):
    """
    Multi-source per-pulse bispectrum dataset for Exp08 (SupCon + Semi-Supervised DEC).

    *** KEY STEP — 129→128 Crop (NO pooling) ***
    The Welch-method bispectra stored on disk are (129, 129) grids.
    To satisfy the ViT patch-embedding requirement that image dimensions are
    divisible by patch_size=16, we crop to (128, 128) by slicing off the last
    row and column: grid = grid[:128, :128].

    This is a 1-pixel discard, not a downsampling. No information is lost in
    practice because the highest bispectral frequency bin (Nyquist) is near-zero
    energy for band-limited PD signals.

    *** KEY CHANGE FROM EXP07 ***
    The index stores `pulse_idx` (the column index into the labels array) rather
    than `scene_idx` + `ch_idx`. This is because the bispectrum grid is stored
    per-pulse in `pulses_bispectra[pulse_idx]`, not per-scene-channel pair.

    Label row indices (same schema as all previous Exp datasets):
    """
    ROW_SCENE_ID   = 0
    ROW_CHANNEL_ID = 1
    ROW_CLASS_ID   = 2
    ROW_PULSE_ID   = 3
    ROW_TOA_IDX    = 4
    ROW_START_IDX  = 5
    ROW_END_IDX    = 6

    # After the 129→128 crop: output grid shape for ViT input.
    # 128 is evenly divisible by patch_size=16 (gives 8×8=64 patches).
    TARGET_SIZE: int = 128

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
        #   (shard_path, pulse_idx, reported_class, actual_class, inst_id, time_res)
        self.index: list[tuple] = []
        self._build_index(sources, shard_key)

    def _build_index(self, sources: list, shard_key: str):
        """
        Walk every shard in every source, read the labels array, and build a
        flat list of (shard_path, pulse_idx, ...) tuples — one entry per pulse.

        Stratified Label Exposure: exactly `label_fraction` of pulses within
        each individual shard have their ground-truth class revealed. The rest
        receive reported_class = -1 (unlabeled).
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
                    if "labels" not in f or f["labels"].shape[1] == 0:
                        continue

                    if "pulses_bispectra" not in f:
                        print(
                            f"[DECDataset_Exp08] Warning: 'pulses_bispectra' not found "
                            f"in {shard_path}. Did you run extract_bispectra.py?"
                        )
                        continue

                    labels   = f["labels"][:]    # (7, N_pulses)
                    time_res = float(
                        np.array(f.attrs.get("time_resolution_s", 1e-11)).item()
                    )

                num_pulses  = labels.shape[1]
                num_labeled = int(num_pulses * self.label_fraction)

                # Stratified label exposure: randomly select which pulses are labeled
                is_labeled_flags = [True] * num_labeled + [False] * (num_pulses - num_labeled)
                random.shuffle(is_labeled_flags)

                for k in range(num_pulses):
                    actual_class_id = int(labels[self.ROW_CLASS_ID,  k])
                    is_labeled      = is_labeled_flags[k]
                    reported_class  = actual_class_id if is_labeled else -1

                    # *** KEY CHANGE: store pulse_idx (k) not scene_idx+ch_idx ***
                    self.index.append((
                        shard_path,
                        k,                                              # pulse_idx
                        reported_class,                                 # masked label
                        actual_class_id,                                # ground truth
                        int(labels[self.ROW_PULSE_ID, k]),              # pulse / inst ID
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
        pulse_idx: int,
    ) -> np.ndarray:
        """
        Load the pre-computed bispectrum grid for a single pulse,
        crop from 129×129 → 128×128, and normalise.

        *** KEY CHANGE FROM EXP07 ***
        Reads `pulses_bispectra[pulse_idx]` (shape 129×129) instead of
        `scenes_bispectra[scene_idx, ch_idx]`.

        *** KEY STEP — Crop, NO pooling ***
        The grid produced by extract_bispectra.py is (129, 129).
        We crop it to (128, 128) so the ViT's patch-embedding conv (patch=16)
        divides evenly: 128/16 = 8 patches per axis = 64 total patch tokens.
        Slicing grid[:128, :128] discards the single Nyquist-edge bin.

        Returns:
            np.ndarray of shape (1, 128, 128), dtype float32.
        """
        with h5py.File(shard_path, "r") as f:
            # pulses_bispectra has shape (N_pulses, 129, 129)
            grid = f["pulses_bispectra"][pulse_idx, :, :].astype(np.float32)
            # grid shape: (129, 129)

        # *** KEY STEP — Crop 129→128 (discard Nyquist-edge bin) ***
        grid = grid[:self.TARGET_SIZE, :self.TARGET_SIZE]   # (128, 128)

        # Add the channel dimension: (128, 128) → (1, 128, 128)
        grid = grid[np.newaxis, :, :]   # (1, 128, 128)

        # Normalise to zero-mean unit-variance for stable training
        grid = self._normalise(grid)

        return grid   # float32, shape (1, 128, 128)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        (shard_path, pulse_idx,
         reported_class, actual_class, inst_id, time_res) = self.index[idx]

        # Load, crop (129→128), and normalise the bispectrum grid
        grid = self._read_bispectrum(shard_path, pulse_idx)
        # grid: np.float32, shape (1, 128, 128)

        if self.augment:
            # SupCon Phase 1: return two independently augmented views
            view1 = torch.from_numpy(_augment_2d(grid))  # (1, 128, 128)
            view2 = torch.from_numpy(_augment_2d(grid))  # (1, 128, 128)
            return (
                view1,
                view2,
                reported_class,
                inst_id,
                shard_path,
                pulse_idx,
                float(time_res),
                actual_class,
            )
        else:
            # Phase 2 / Inference: return single clean signal
            signal = torch.from_numpy(grid)   # (1, 128, 128)
            return (
                signal,
                reported_class,
                inst_id,
                shard_path,
                pulse_idx,
                float(time_res),
                actual_class,
            )
