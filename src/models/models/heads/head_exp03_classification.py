"""
head_exp03_classification.py
============================
Classification head for the Contrastive Embedding architecture.
Projects the (B, embedding_dim) vector to (B, num_classes) logits.
"""

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        # Simple linear projection
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        # embedding shape: (B, embedding_dim)
        logits = self.fc(embedding)  # (B, num_classes)
        return logits
