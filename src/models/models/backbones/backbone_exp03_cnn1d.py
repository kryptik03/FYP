"""
backbone_exp03_cnn1d.py
=======================
1D CNN backbone for Contrastive Embedding.
Takes a (B, 1, max_pulse_len) tensor and outputs a (B, embedding_dim) feature vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveCNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16, embedding_dim: int = 128):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm1d(base_channels)
        
        self.conv2 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm1d(base_channels * 2)
        
        self.conv3 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm1d(base_channels * 4)
        
        self.conv4 = nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm1d(base_channels * 8)
        
        # Collapse the sequence dimension entirely
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Project to the desired embedding dimension
        self.fc_embed = nn.Linear(base_channels * 8, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))   # (B, C, L')
        
        x = self.global_pool(x)               # (B, C, 1)
        x = x.squeeze(-1)                     # (B, C)
        
        emb = self.fc_embed(x)                # (B, embedding_dim)
        
        # L2 normalize embeddings for stable Triplet Margin Loss
        emb = F.normalize(emb, p=2, dim=1)
        
        return emb
