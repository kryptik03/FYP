"""
cnn_1d_exp02.py
=============
1D CNN feature extractor backbone for the classification-only pipeline.

Operates on full-resolution, un-decimated pulse windows.

Architecture
------------
4 conv blocks (32 -> 64 -> 128 -> feature_dim channels).
Each block: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d(2).
After 4 blocks the temporal dimension is reduced by 16x.
AdaptiveAvgPool1d(1) collapses the remaining time axis to a single vector,
making this backbone independent of input length.

Output: (Batch, feature_dim)  — ready for a linear classifier head.
"""

import torch.nn as nn


class CNN1DClassifierBackbone(nn.Module):
    """
    Pure feature extractor for 1D PD pulse windows.

    Args
    ----
    in_channels  : Number of input channels (1 for single-channel signal).
    feature_dim  : Number of output feature channels (depth of final conv block).
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()

        self.blocks = nn.Sequential(
            self._block(in_channels, 32,          kernel_size=7),
            self._block(32,          64,          kernel_size=5),
            self._block(64,          128,         kernel_size=3),
            self._block(128,         feature_dim, kernel_size=3),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    @staticmethod
    def _block(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        """
        Args
        ----
        x : Tensor, shape (B, 1, L)

        Returns
        -------
        features : Tensor, shape (B, feature_dim)
        """
        x = self.blocks(x)          # (B, feature_dim, L // 16)
        x = self.global_pool(x)     # (B, feature_dim, 1)
        return x.squeeze(-1)        # (B, feature_dim)
