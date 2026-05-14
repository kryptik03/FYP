"""
dataset_exp01.py
====================
PyTorch Dataset for simultaneous PD signal detection and classification.

Each sample is one (scene, channel) pair:

    signal  : (1, seq_len)   float32 tensor  - decimated single-channel waveform
    target  : (S, 4)         float32 tensor  - YOLO anchor-free target grid

Where:
    seq_len = 1000   (500 001 raw points decimated by factor 500)
    S       = 32     (grid cells)

Target grid columns
-------------------
Column 0  objectness   - 1.0 if a pulse centre falls in this cell, else 0.0
Column 1  centre_offset- fractional position of pulse centre within the cell,
                         in [0, 1].  0.0 = left edge,  1.0 = right edge.
Column 2  log_width    - natural log of the normalised pulse width.
                         Width is normalised by seq_len so it is in (0, 1].
                         The model predicts this directly (no sigmoid needed).
Column 3  class_id     - float(Class_ID): 0.0 = PD1,  1.0 = PD2.
                         Only meaningful where objectness == 1.0.

Coordinate conventions
-----------------------
All sample indices that come from the HDF5 labels (Start_Idx, End_Idx) are
already 0-indexed after the user's MATLAB update.

Decimated index = raw_index // decimation_factor

After decimation the sequence runs from index 0 to seq_len-1 (0..999).
Normalised position = decimated_index / seq_len  ->  [0, 1).

Cell assignment
---------------
cell_idx = min(int(centre_norm * S), S - 1)   in  {0, ..., S-1}

Collision handling
------------------
If two pulses in the same (scene, channel) have centres that fall in the same
grid cell, the last one in the label list overwrites the earlier one.  Given
the realistic minimum pulse separation enforced by the 50 ns injection buffer
(5 000 raw samples -> 10 decimated pts -> < 1 cell width), collisions are
extremely rare.
"""

import math

import numpy as np
import torch

from .base_dataset import BaseDataset
from .transforms import DecimateMaxPool1D


class DetectionDataset(BaseDataset):
    """
    Returns decimated per-channel signals paired with YOLO target grids.

    Usage
    -----
    >>> ds = DetectionDataset(
    ...     root_path="data/raw/synthesised/20260427_170034_sy-ShmH-ShmH",
    ...     shard_ids=list(range(1, 17)),
    ...     decimation_factor=500,
    ...     grid_cells=32,
    ... )
    >>> signal, target = ds[0]
    >>> signal.shape   # (1, 1000)
    >>> target.shape   # (32, 4)
    """

    def __init__(
        self,
        root_path: str,
        shard_ids: list,
        decimation_factor: int = 500,
        grid_cells: int = 32,
    ):
        """
        Args:
            root_path:         Path to the ShmH node folder.
            shard_ids:         Shard numbers to include.
            decimation_factor: Decimation stride (500 -> 1 000 pts output).
            grid_cells:        Number of YOLO grid cells S (default 32).
        """
        super().__init__(root_path, shard_ids)
        self.decimation_factor = decimation_factor
        self.grid_cells        = grid_cells
        self.decimator         = DecimateMaxPool1D(factor=decimation_factor)

        # Pre-compute output sequence length using dynamically loaded raw_len
        # F.max_pool1d formula: floor((N - kernel) / stride) + 1
        self.seq_len = (self.raw_len - decimation_factor) // decimation_factor + 1

    # ------------------------------------------------------------------ #
    # YOLO target builder                                                  #
    # ------------------------------------------------------------------ #

    def _build_yolo_target(self, ch_labels: np.ndarray) -> torch.Tensor:
        """
        Convert per-channel label rows into a (S, 4) YOLO target tensor.

        Args:
            ch_labels: shape (7, K) - labels for this (scene, channel).
                       K == 0 for empty scenes.

        Returns:
            target: (grid_cells, 4) float32 tensor, all-zeros for empty scenes.
        """
        S       = self.grid_cells
        seq_len = self.seq_len
        dec     = self.decimation_factor

        target = torch.zeros(S, 4, dtype=torch.float32)

        # Empty scene - no PD events present
        if ch_labels.shape[1] == 0:
            return target

        for k in range(ch_labels.shape[1]):
            start_idx = int(ch_labels[BaseDataset.ROW_START_IDX, k])
            end_idx   = int(ch_labels[BaseDataset.ROW_END_IDX,   k])
            class_id  = int(ch_labels[BaseDataset.ROW_CLASS_ID,  k])

            # ---------------------------------------------------------- #
            # Step 1 - Convert raw 0-indexed sample positions             #
            #          -> decimated sample positions                       #
            # ---------------------------------------------------------- #
            start_dec = start_idx // dec
            end_dec   = end_idx   // dec

            # Guard: decimation can collapse a very narrow pulse to 0 width
            if end_dec <= start_dec:
                end_dec = start_dec + 1

            # Clamp to valid decimated range [0, seq_len-1]
            start_dec = max(0, min(start_dec, seq_len - 1))
            end_dec   = max(0, min(end_dec,   seq_len - 1))

            # ---------------------------------------------------------- #
            # Step 2 - Normalise to [0, 1] over the full decimated window #
            # ---------------------------------------------------------- #
            center_dec  = (start_dec + end_dec) / 2.0
            width_dec   = float(end_dec - start_dec)

            center_norm = center_dec / seq_len            # in [0, 1)
            width_norm  = max(width_dec / seq_len, 1e-6) # in (0, 1]

            # ---------------------------------------------------------- #
            # Step 3 - Assign to a grid cell                              #
            # ---------------------------------------------------------- #
            # cell_idx in {0, ..., S-1}
            cell_idx = min(int(center_norm * S), S - 1)

            # Fractional offset of the pulse centre within the cell [0, 1]
            # cell_size_norm = 1/S
            # offset = (centre_norm - cell_idx/S) / (1/S)
            #        = centre_norm * S - cell_idx
            offset = center_norm * S - cell_idx
            offset = max(0.0, min(offset, 1.0))   # clamp numerical noise

            # Log-space width - stable for regression (avoids predicting
            # widths directly in [0,1] where gradient near 0 is very flat)
            log_width = math.log(width_norm)

            # ---------------------------------------------------------- #
            # Step 4 - Write into target tensor (last pulse wins on       #
            #          the rare collision)                                 #
            # ---------------------------------------------------------- #
            target[cell_idx, 0] = 1.0            # objectness
            target[cell_idx, 1] = offset         # centre offset in cell
            target[cell_idx, 2] = log_width      # log normalised width
            target[cell_idx, 3] = float(class_id)  # 0.0 or 1.0

        return target

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        signal : torch.Tensor, shape (1, seq_len), float32
            Decimated single-channel waveform ready for the CNN backbone.

        target : torch.Tensor, shape (grid_cells, 4), float32
            YOLO target grid for this (scene, channel) pair.
        """
        raw_signal, ch_labels = self._get_raw(idx)

        # numpy (500001,) -> torch (1, 500001) -> decimate -> (1, 1000)
        signal_tensor = torch.from_numpy(raw_signal).unsqueeze(0)  # (1, N)
        signal_tensor = self.decimator(signal_tensor)               # (1, seq_len)

        target = self._build_yolo_target(ch_labels)                 # (S, 4)

        return signal_tensor, target
