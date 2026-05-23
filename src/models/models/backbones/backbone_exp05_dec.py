"""
backbone_exp05_dec.py
=====================
1D CNN Backbone for Spherical Deep Embedded Clustering (exp05).

Architecture: 5-layer strided Conv1D → AdaptiveAvgPool → FC projection.
Uses AdaptiveAvgPool so it accepts any input length (sampling-rate agnostic
when signals are resampled to max_pulse_len by the Dataset).

Input:  (B, 1, max_pulse_len)
Output: (B, embedding_dim)  — L2-normalised embedding vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DECCNN1D(nn.Module):
    """
    1D CNN backbone for DEC.
    Slightly wider than exp03 (base_channels=32 default) to capture richer
    multi-scale patterns across both UHF PD and UHF noise signals.
    """

    def __init__(
        self,
        in_channels: int   = 1,
        base_channels: int = 32,
        embedding_dim: int = 128,
    ):
        super().__init__()

        C = base_channels

        # ---- Strided convolutional blocks ---------------------------------- #
        self.conv1 = nn.Conv1d(in_channels, C,      kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm1d(C)

        self.conv2 = nn.Conv1d(C,      C * 2, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm1d(C * 2)

        self.conv3 = nn.Conv1d(C * 2,  C * 4, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm1d(C * 4)

        self.conv4 = nn.Conv1d(C * 4,  C * 8, kernel_size=3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm1d(C * 8)

        self.conv5 = nn.Conv1d(C * 8,  C * 8, kernel_size=3, stride=2, padding=1)
        self.bn5   = nn.BatchNorm1d(C * 8)

        # ---- Global pooling: removes time dimension regardless of input len #
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Projection head (non-linear, better for SimCLR) --------------- #
        self.proj = nn.Sequential(
            nn.Linear(C * 8, C * 8),
            nn.ReLU(inplace=True),
            nn.Linear(C * 8, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, L)
        returns: (B, embedding_dim) — L2-normalised
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))     # (B, C*8, L')

        x = self.global_pool(x).squeeze(-1)      # (B, C*8)
        emb = self.proj(x)                        # (B, embedding_dim)
        return F.normalize(emb, p=2, dim=1)       # L2-normalise
