"""
base_dataset.py
===============
Base PyTorch Dataset for the FYP PD pipeline.

Responsibilities
----------------
1. Scan a dataset folder for the requested HDF5 shards.
2. Build a flat index of every (shard_path, scene_idx, channel_idx) triple so
   that torch.utils.data.DataLoader can iterate over individual channel signals.
3. Open HDF5 files lazily inside __getitem__ and close them immediately after
   reading - this prevents file-handle leaks when num_workers > 0.
4. Parse the (7, N_pulses) label matrix that MATLAB writes in column-major order
   (h5py reads it transposed relative to how MATLAB stored it).

HDF5 label matrix layout (confirmed from generation code + attrs)
-----------------------------------------------------------------
MATLAB writes each label row as:
    [Scene_ID, Channel_ID, Class_ID, Pulse_Instance_ID,
     TOA_Index, Start_Idx, End_Idx]

Because MATLAB stores arrays in column-major order, h5py reads the
(N_pulses x 7) MATLAB matrix as shape (7, N_pulses) in Python.
So in Python:

    labels[0, :]  ->  Scene_ID           (0-indexed)
    labels[1, :]  ->  Channel_ID         (0-indexed, 0–3)
    labels[2, :]  ->  Class_ID           (0 = PD1,  1 = PD2)
    labels[3, :]  ->  Pulse_Instance_ID  (global 0-indexed counter)
    labels[4, :]  ->  TOA_Index          (0-indexed sample index)
    labels[5, :]  ->  Start_Idx          (0-indexed sample index)
    labels[6, :]  ->  End_Idx            (0-indexed sample index)

Note: TOA_Index / Start_Idx / End_Idx were previously 1-indexed in MATLAB.
The user has updated the MATLAB script so they are now 0-indexed - no
subtraction needed here.
"""

import os

import h5py
import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Builds a flat list index of (shard_path, scene_idx, channel_idx) triples
    and provides _get_raw() for subclasses to fetch raw signal + filtered labels.

    Subclasses must implement __getitem__.
    """

    # ------------------------------------------------------------------ #
    # Named constants for the 7 label rows - avoids magic numbers in code #
    # ------------------------------------------------------------------ #
    ROW_SCENE_ID   = 0   # Scene_ID          (0-indexed)
    ROW_CHANNEL_ID = 1   # Channel_ID        (0-indexed, 0–3)
    ROW_CLASS_ID   = 2   # Class_ID          (0=PD1, 1=PD2)
    ROW_PULSE_ID   = 3   # Pulse_Instance_ID (global 0-indexed)
    ROW_TOA_IDX    = 4   # TOA_Index         (0-indexed sample)
    ROW_START_IDX  = 5   # Start_Idx         (0-indexed sample)
    ROW_END_IDX    = 6   # End_Idx           (0-indexed sample)

    def __init__(self, root_path: str, shard_ids: list):
        """
        Args:
            root_path:  Path to the dataset node folder, e.g.
                        'data/raw/synthesised/20260427_170034_sy-ShmH-ShmH'.
            shard_ids:  List of integer shard numbers to include, e.g.
                        [1, 2, ..., 16] for training.
        """
        super().__init__()
        self.root_path = root_path
        self.shard_ids = shard_ids

        # Each entry: (shard_path: str, scene_idx: int, channel_idx: int)
        self.index: list[tuple[str, int, int]] = []
        self._build_index()

    # ------------------------------------------------------------------ #
    # Index construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_index(self):
        """
        Scans each requested shard file and enumerates all
        (shard_path, scene_idx, channel_idx) triples.

        Called once at construction time - O(num_shards) file opens.
        """
        NUM_CHANNELS = 4  # fixed by the generation script (num_sensors = 4)

        for shard_id in self.shard_ids:
            shard_path = os.path.join(
                self.root_path, f"synth_shard_{shard_id:02d}.h5"
            )
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"[BaseDataset] Shard file not found: {shard_path}\n"
                    f"  root_path = {self.root_path}\n"
                    f"  shard_id  = {shard_id}"
                )

            # Read only the scene count - no need to load the data yet
            with h5py.File(shard_path, "r") as f:
                num_scenes = f["scenes"].shape[0]   # (num_scenes, 4, 500001)

            for scene_idx in range(num_scenes):
                for ch_idx in range(NUM_CHANNELS):
                    self.index.append((shard_path, scene_idx, ch_idx))

    # ------------------------------------------------------------------ #
    # Raw data access                                                      #
    # ------------------------------------------------------------------ #

    def _get_raw(self, idx: int):
        """
        Opens the HDF5 file for sample `idx`, reads the signal and labels for
        the corresponding (scene, channel) pair, then closes the file.

        Returns
        -------
        signal : np.ndarray, shape (500_001,), dtype float32
            Raw single-channel waveform - 5 us at 100 GHz.

        ch_labels : np.ndarray, shape (7, K), dtype float32
            All label columns that belong to this (scene, channel) pair.
            K = 0 for empty scenes (no PD pulses in that scene).
        """
        shard_path, scene_idx, ch_idx = self.index[idx]

        with h5py.File(shard_path, "r") as f:
            # ------------------------------------------------------------- #
            # Signal                                                          #
            # scenes shape in h5py: (num_scenes, num_sensors, N_scene)       #
            #   = (20, 4, 500001)                                             #
            # ------------------------------------------------------------- #
            signal = f["scenes"][scene_idx, ch_idx, :].astype(np.float32)

            # ------------------------------------------------------------- #
            # Labels                                                          #
            # h5py shape: (7, N_total_pulses_in_shard)                       #
            # Filter to rows that match this (scene_idx, ch_idx) pair.       #
            # ------------------------------------------------------------- #
            if "labels" not in f:
                # Shard has no labels dataset at all (should not happen, but
                # guard against it gracefully)
                ch_labels = np.empty((7, 0), dtype=np.float32)
            else:
                all_labels = f["labels"][:]           # (7, N_total)

                scene_mask   = all_labels[self.ROW_SCENE_ID,   :] == scene_idx
                channel_mask = all_labels[self.ROW_CHANNEL_ID, :] == ch_idx
                combined     = scene_mask & channel_mask

                ch_labels = all_labels[:, combined].astype(np.float32)
                # Shape: (7, K) where K is the number of pulses in this channel

        return signal, ch_labels

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        """Subclasses must override this to apply their specific transforms."""
        raise NotImplementedError(
            "BaseDataset.__getitem__ must be implemented by subclasses."
        )
