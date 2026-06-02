"""
dataset_exp09.py
================
Multi-Source PyTorch Dataset for Exp09: Physics-Invariant PD Classification.

KEY CHANGES vs DECDataset_Exp08
---------------------------------
1.  2-Channel Bispectra (V2 feature format)
    Reads `pulses_bispectra` of shape (N_pulses, 2, 129, 129):
      Channel 0 = magnitude (energy envelope)
      Channel 1 = phase     (physics fingerprint, attenuation-stable)
    Raises a clear error if V1 (single-channel) shards are given.

2.  Instance Index for Cross-Sensor Positive Pairing
    During index construction, a secondary lookup table is built:
        inst_index[(shard_path, raw_inst_id)] = [flat_idx_1, flat_idx_2, ...]
    This maps each physical event (identified by Pulse_Instance_ID in the
    labels array) to all its sensor observations within the same shard.

    In Phase 1 (__getitem__ with augment=True):
      - view1 = this sample's 2-channel bispectrum (lightly augmented)
      - view2 = bispectrum of a DIFFERENT pulse with the same inst_id
                (ideally a different sensor/channel of the same physical event)
      If only one pulse exists in the instance group, view2 falls back to
      an augmented copy of view1 (Exp08 behaviour; safe for isolated sources).

3.  Domain Label
    A `domain_label` (int) is assigned to each pulse based on its source type:
        equation    → 0
        synthesised → 1
        cwru        → 2
        measured    → 3
        unknown     → -1  (excluded from domain loss automatically)
    Any subset of source types works (1–4 types). Missing domain classes
    simply never appear in the batch; CrossEntropyLoss handles this gracefully.
    DANN effectively becomes a no-op when only 1 domain type is present.

4.  Global Instance ID
    Raw inst_ids (Pulse_Instance_ID in the labels) are only unique within a
    shard. Across shards, values may repeat. The dataset assigns a globally
    unique `global_inst_id` per (shard_path, raw_inst_id) pair to prevent
    false-positive matching in the SupCon batch.

5.  Augmentation — Magnitude Only
    `_augment_2ch` applies Gaussian noise, amplitude scaling, and ω₁/ω₂
    masking ONLY to channel 0 (magnitude). Channel 1 (phase) is left
    untouched — the phase-coupling fingerprint must not be corrupted.

H5 Feature Shard Schema (produced by extract_bispectra_v2.py):
    pulses_bispectra : float32  (N_pulses, 2, 129, 129)
                       [:, 0, :, :] = magnitude
                       [:, 1, :, :] = phase
    labels           : float32  (7, N_pulses) — same schema as all prior exps.

Returned sample (Phase 1 — cross-sensor paired):
    (view1, view2, reported_class, global_inst_id, domain_label,
     shard_path, pulse_idx, time_res, actual_class)
    view1/view2 have shape (2, 128, 128).

Returned sample (Phase 2 / inference — single):
    (signal, reported_class, global_inst_id, domain_label,
     shard_path, pulse_idx, time_res, actual_class)
    signal has shape (2, 128, 128).
"""

import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Default domain map (canonical 4-way split for Exp09)
# ---------------------------------------------------------------------------

_DEFAULT_DOMAIN_MAP = {
    "equation":    0,
    "synthesised": 1,
    "cwru":        2,
    "measured":    3,
}


# ---------------------------------------------------------------------------
# 2-Channel Augmentation — Magnitude Only
# ---------------------------------------------------------------------------

def _augment_2ch(bispectrum: np.ndarray) -> np.ndarray:
    """
    Apply lightweight augmentations to a 2-channel bispectrum (2, H, W).

    IMPORTANT: only channel 0 (magnitude) is modified.
    Channel 1 (phase) is left exactly as-is — it encodes the phase-coupling
    fingerprint that must remain stable to act as the physics invariant.

    Augmentations applied to channel 0:
        1. Additive Gaussian noise  (signal-adaptive std dev, 1–10%)
        2. Amplitude scaling        (random gain in [0.9, 1.1])
        3. ω₁-axis masking          (row band, ≤25% of H, p=0.5)
        4. ω₂-axis masking          (column band, ≤25% of W, p=0.5)

    Args:
        bispectrum : np.ndarray of shape (2, H, W), dtype float32.

    Returns:
        Augmented np.ndarray of shape (2, H, W), dtype float32.
    """
    aug = bispectrum.copy()       # never mutate the cached grid
    mag = aug[0]                  # view into channel 0 (magnitude)

    # 1. Additive Gaussian noise (signal-adaptive)
    sigma = mag.std() * random.uniform(0.01, 0.10)
    aug[0] = mag + np.random.normal(0.0, sigma, mag.shape).astype(np.float32)

    # 2. Amplitude scaling
    aug[0] = aug[0] * random.uniform(0.9, 1.1)

    # 3. ω₁-axis masking (rows)
    if random.random() < 0.5:
        H = aug.shape[1]
        if H > 4:
            w1_width = random.randint(1, H // 4)
            w1_start = random.randint(0, H - w1_width)
            aug[0, w1_start : w1_start + w1_width, :] = 0.0

    # 4. ω₂-axis masking (columns)
    if random.random() < 0.5:
        W = aug.shape[2]
        if W > 4:
            w2_width = random.randint(1, W // 4)
            w2_start = random.randint(0, W - w2_width)
            aug[0, :, w2_start : w2_start + w2_width] = 0.0

    # Channel 1 (phase) is intentionally not modified.
    return aug.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DECDataset_Exp09(Dataset):
    """
    Multi-source per-pulse 2-channel bispectrum dataset for Exp09.

    Supports cross-sensor positive pairing (view1 from Sensor A, view2 from
    Sensor B of the same physical event) for Instance-Linked SupCon training.

    See module docstring for full design rationale.
    """

    # Label row indices (shared schema across all experiments)
    ROW_SCENE_ID   = 0
    ROW_CHANNEL_ID = 1
    ROW_CLASS_ID   = 2
    ROW_PULSE_ID   = 3    # Pulse_Instance_ID — links multi-sensor observations
    ROW_TOA_IDX    = 4
    ROW_START_IDX  = 5
    ROW_END_IDX    = 6

    # After 129→128 crop
    TARGET_SIZE: int = 128

    def __init__(
        self,
        sources:        list,
        shard_key:      str,
        max_pulse_len:  int   = 4096,
        augment:        bool  = False,
        label_fraction: float = 0.10,
        domain_map:     dict  = None,
    ):
        """
        Args:
            sources       : List of source dicts from YAML config.
                            Each dict: {type, path, train_shards, val_shards}.
            shard_key     : "train_shards" or "val_shards".
            max_pulse_len : Kept for API compatibility; not used for time-slicing.
            augment       : If True, return (view1, view2, ...) for Phase 1 SupCon.
            label_fraction: Fraction of labels revealed for pairwise constraints.
            domain_map    : Mapping source-type string → int domain label.
                            Defaults to the canonical 4-way Exp09 map.
        """
        super().__init__()
        self.max_pulse_len  = max_pulse_len
        self.augment        = augment
        self.label_fraction = label_fraction
        self.domain_map     = domain_map if domain_map is not None else _DEFAULT_DOMAIN_MAP

        # Flat pulse index: one entry per pulse across all shards / sources.
        # Entry format: (shard_path, pulse_idx, reported_class, actual_class,
        #                global_inst_id, time_res, domain_label)
        self.index: list[tuple] = []

        # Instance-level secondary index:
        # (shard_path, raw_inst_id) → [flat_idx_1, flat_idx_2, ...]
        # Used to look up cross-sensor pair candidates in __getitem__.
        self.inst_index: dict[tuple, list[int]] = {}

        # Global instance counter — ensures inst_ids are unique across shards.
        self._global_inst_map: dict[tuple, int] = {}

        self._build_index(sources, shard_key)

    # -----------------------------------------------------------------------
    # Index Construction
    # -----------------------------------------------------------------------

    def _build_index(self, sources: list, shard_key: str):
        """
        Walk every shard in every source and build:
          - self.index      : flat list of (shard_path, pulse_idx, ...) tuples
          - self.inst_index : (shard_path, raw_inst_id) → [flat positions]

        Stratified Label Exposure:
            Exactly `label_fraction` of pulses within each shard have their
            ground-truth class revealed. The rest receive reported_class = -1.

        Domain Label:
            Derived from source["type"] via self.domain_map.
            Unknown types receive domain_label = -1.
        """
        for source in sources:
            root_path    = os.path.abspath(source["path"])
            shard_ids    = source.get(shard_key, [])
            domain_label = self.domain_map.get(source.get("type", ""), -1)

            for shard_id in shard_ids:
                shard_path = os.path.join(root_path, f"shard_{shard_id:02d}.h5")
                if not os.path.exists(shard_path):
                    print(
                        f"[DECDataset_Exp09] Warning: shard not found, skipping: {shard_path}"
                    )
                    continue

                with h5py.File(shard_path, "r") as f:
                    if "labels" not in f or f["labels"].shape[1] == 0:
                        continue

                    if "pulses_bispectra" not in f:
                        print(
                            f"[DECDataset_Exp09] Warning: 'pulses_bispectra' not found "
                            f"in {shard_path}. Did you run extract_bispectra_v2.py?"
                        )
                        continue

                    # Validate V2 shape: (N, 2, H, W)
                    shape = f["pulses_bispectra"].shape
                    if len(shape) != 4 or shape[1] != 2:
                        raise ValueError(
                            f"[DECDataset_Exp09] Expected V2 shape (N, 2, H, W) in "
                            f"{shard_path}, got {shape}. "
                            f"Please run extract_bispectra_v2.py to generate 2-channel features."
                        )

                    labels   = f["labels"][:]
                    time_res = float(
                        np.array(f.attrs.get("time_resolution_s", 1e-11)).item()
                    )

                num_pulses  = labels.shape[1]
                num_labeled = int(num_pulses * self.label_fraction)

                # Stratified label exposure
                is_labeled_flags = [True] * num_labeled + [False] * (num_pulses - num_labeled)
                random.shuffle(is_labeled_flags)

                for k in range(num_pulses):
                    actual_class_id = int(labels[self.ROW_CLASS_ID, k])
                    raw_inst_id     = int(labels[self.ROW_PULSE_ID,  k])
                    reported_class  = actual_class_id if is_labeled_flags[k] else -1

                    # Globally unique instance ID across shards
                    inst_key = (shard_path, raw_inst_id)
                    if inst_key not in self._global_inst_map:
                        self._global_inst_map[inst_key] = len(self._global_inst_map)
                    global_inst_id = self._global_inst_map[inst_key]

                    flat_pos = len(self.index)
                    self.index.append((
                        shard_path,
                        k,                  # pulse_idx
                        reported_class,     # masked label (-1 if unlabeled)
                        actual_class_id,    # ground truth
                        global_inst_id,     # globally unique across shards
                        time_res,
                        domain_label,       # 0–3 or -1
                    ))

                    # Register this pulse in the instance-level secondary index
                    self.inst_index.setdefault(inst_key, []).append(flat_pos)

    # -----------------------------------------------------------------------
    # Bispectrum I/O
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalise_2ch(grid: np.ndarray) -> np.ndarray:
        """
        Per-channel zero-mean, unit-variance normalisation for a (2, H, W) grid.
        Channel 0 (magnitude): normalised independently.
        Channel 1 (phase):     normalised independently.
        If std ≈ 0, only subtract the mean to avoid div-by-zero.
        """
        out = grid.copy()
        for c in range(grid.shape[0]):
            mu, std = grid[c].mean(), grid[c].std()
            out[c]  = (grid[c] - mu) / std if std > 1e-9 else grid[c] - mu
        return out.astype(np.float32)

    def _read_bispectrum_v2(self, shard_path: str, pulse_idx: int) -> np.ndarray:
        """
        Load the pre-computed 2-channel bispectrum grid for a single pulse,
        crop from 129×129 → 128×128 per channel, and normalise each channel.

        Returns:
            np.ndarray of shape (2, 128, 128), dtype float32.
        """
        with h5py.File(shard_path, "r") as f:
            # pulses_bispectra: (N_pulses, 2, 129, 129)
            grid = f["pulses_bispectra"][pulse_idx, :, :, :].astype(np.float32)
            # grid shape: (2, 129, 129)

        # Crop 129→128 per channel (discard single Nyquist-edge bin)
        grid = grid[:, : self.TARGET_SIZE, : self.TARGET_SIZE]   # (2, 128, 128)

        # Per-channel normalisation
        grid = self._normalise_2ch(grid)

        return grid   # float32, shape (2, 128, 128)

    # -----------------------------------------------------------------------
    # Dataset Protocol
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        (shard_path, pulse_idx, reported_class, actual_class,
         global_inst_id, time_res, domain_label) = self.index[idx]

        # Load, crop (129→128), normalise the 2-channel bispectrum grid
        grid1 = self._read_bispectrum_v2(shard_path, pulse_idx)
        # grid1: float32, shape (2, 128, 128)

        if self.augment:
            # -----------------------------------------------------------------
            # Phase 1 — Cross-Sensor Positive Pairing
            # -----------------------------------------------------------------
            # Find other pulses from the same physical event (same inst_id).
            # These are from different sensors (different channel IDs) of the
            # same discharge event, giving a naturally diverse positive pair.
            raw_inst_id = int(self.index[idx][4])   # global_inst_id used for key
            # Reconstruct the shard-local key for inst_index lookup
            # Note: _global_inst_map is keyed by (shard_path, raw_inst_id) but
            # we stored global_inst_id. We need the shard-local raw inst_id.
            # Retrieve it from the labels via a secondary approach:
            # We stored global_inst_id in index[4]; the inst_index is keyed by
            # (shard_path, raw_inst_id_from_labels). We build a reverse map.
            # Simpler: scan inst_index for this shard to find the right key.
            # *** Efficient approach: at build time we store (shard, raw_inst) ***
            # Since global_inst_id is the canonical key across shards, use it
            # to look up inst_index entries with the same global_inst_id.
            candidates = self._get_instance_candidates(idx)

            if candidates:
                pair_flat = random.choice(candidates)
                (pair_shard, pair_pulse_idx, *_rest) = self.index[pair_flat]
                grid2 = self._read_bispectrum_v2(pair_shard, pair_pulse_idx)
            else:
                # Fallback: augmented copy of the same pulse (single-sensor)
                grid2 = grid1.copy()

            view1 = torch.from_numpy(_augment_2ch(grid1))   # (2, 128, 128)
            view2 = torch.from_numpy(_augment_2ch(grid2))   # (2, 128, 128)

            return (
                view1,
                view2,
                reported_class,   # int → collated to (B,) tensor by DataLoader
                global_inst_id,   # int → collated to (B,) tensor
                domain_label,     # int → collated to (B,) tensor
                shard_path,       # str → kept as list by DataLoader
                pulse_idx,        # int → collated to (B,) tensor
                float(time_res),  # float → collated to (B,) tensor
                actual_class,     # int → collated to (B,) tensor
            )

        else:
            # -----------------------------------------------------------------
            # Phase 2 / Inference — single clean signal
            # -----------------------------------------------------------------
            signal = torch.from_numpy(grid1)   # (2, 128, 128)

            return (
                signal,
                reported_class,
                global_inst_id,
                domain_label,
                shard_path,
                pulse_idx,
                float(time_res),
                actual_class,
            )

    # -----------------------------------------------------------------------
    # Cross-Sensor Candidate Lookup
    # -----------------------------------------------------------------------

    def _get_instance_candidates(self, idx: int) -> list[int]:
        """
        Return the flat indices of all OTHER pulses sharing the same physical
        instance (Pulse_Instance_ID × shard) as the pulse at `idx`.

        Uses the `inst_index` secondary lookup table built during _build_index.
        Returns an empty list if no other pulses exist for this instance.
        """
        (shard_path, _, _, _, global_inst_id, *_) = self.index[idx]

        # Recover the (shard_path, raw_inst_id) key from the reverse map.
        # We stored the mapping _global_inst_map[(shard, raw_inst)] = global_id.
        # Build the reverse if not already cached.
        if not hasattr(self, "_reverse_inst_map"):
            self._reverse_inst_map: dict[int, tuple] = {
                v: k for k, v in self._global_inst_map.items()
            }

        inst_key = self._reverse_inst_map.get(global_inst_id)
        if inst_key is None:
            return []

        all_flat = self.inst_index.get(inst_key, [])
        # Exclude self
        return [f for f in all_flat if f != idx]
