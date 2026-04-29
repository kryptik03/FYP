"""
cnn_1d.py
=========
1D CNN Backbone - pure feature extractor.

This module is completely task-agnostic.  It takes a raw (decimated) 1-D
signal and turns it into a spatial feature map.  It does NOT know whether the
downstream head will detect bounding boxes, classify PD types, estimate TDOA,
or do anything else.

Architecture
------------
Four convolutional blocks, each consisting of:
    Conv1d  ->  BatchNorm1d  ->  ReLU  ->  MaxPool1d(2)

Followed by AdaptiveAvgPool1d to fix the spatial dimension to exactly S cells
(matching the YOLO grid size set in the config).

    Input : (B, 1,   1000)
    Block1: (B, 32,   500)   - Conv(k=7) + BN + ReLU + MaxPool(2)
    Block2: (B, 64,   250)   - Conv(k=5) + BN + ReLU + MaxPool(2)
    Block3: (B, 128,  125)   - Conv(k=3) + BN + ReLU + MaxPool(2)
    Block4: (B, 256,  125)   - Conv(k=3) + BN + ReLU  (no pooling here)
    Pool  : (B, 256,   32)   - AdaptiveAvgPool1d(grid_cells)
    Output: (B, 256,   32)

Why these kernel sizes?
-----------------------
Larger kernels early on (k=7, k=5) give each neuron a wider receptive field
over the raw signal.  By the time we reach the later blocks, each feature-map
position already 'sees' a large portion of the waveform, so smaller kernels
(k=3) are sufficient.

Why AdaptiveAvgPool at the end?
--------------------------------
This decouples the backbone from the exact sequence length.  If you change the
decimation factor (and thus seq_len) in the YAML, the backbone still outputs
(B, 256, grid_cells) without any code changes.
"""

import torch
import torch.nn as nn


class CNN1DBackbone(nn.Module):
    """
    1-D CNN feature extractor.

    Args:
        in_channels:   Number of input signal channels.  1 for per-channel
                       single-waveform mode (the default).
        base_channels: Output channels of the first conv block.  Subsequent
                       blocks double this: base -> 2xbase -> 4xbase -> 8xbase.
                       Default 32 -> channel progression 32, 64, 128, 256.
        grid_cells:    Number of spatial positions in the output feature map.
                       Must match the YOLO head's grid_cells setting.

    Input:   (B, in_channels, seq_len)    e.g. (B, 1, 1000)
    Output:  (B, 8 x base_channels, grid_cells)  e.g. (B, 256, 32)
    """

    def __init__(
        self,
        in_channels:   int = 1,
        base_channels: int = 32,
        grid_cells:    int = 32,
    ):
        super().__init__()

        c1 = base_channels          #  32
        c2 = base_channels * 2     #  64
        c3 = base_channels * 4     # 128
        c4 = base_channels * 8     # 256

        # -------------------------------------------------------------- #
        # Convolutional blocks                                             #
        # Each block: Conv -> BN -> ReLU -> MaxPool(2)                       #
        # padding = kernel_size // 2  keeps spatial dim before pooling    #
        # -------------------------------------------------------------- #

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),        # (B, 32, seq//2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),        # (B, 64, seq//4)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),        # (B, 128, seq//8)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
            # No MaxPool here - let AdaptiveAvgPool handle the final size
        )

        # Fix spatial output to exactly grid_cells positions regardless of
        # the input sequence length
        self.pool = nn.AdaptiveAvgPool1d(grid_cells)   # (B, 256, grid_cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, seq_len)

        Returns:
            feature_map: (B, 8*base_channels, grid_cells)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return x
