"""
cls_head_exp02.py
===========
Linear classification head for the classification-only pipeline.

Receives a feature vector from the backbone and projects it to class logits.

Input : (Batch, feature_dim)
Output: (Batch, num_classes)   — raw logits (no softmax)
"""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Single linear layer projecting backbone features to class logits.

    Args
    ----
    feature_dim : Dimensionality of the backbone output vector.
    num_classes : Number of target classes (2 for PD1 vs PD2).
    dropout_p   : Dropout probability applied before the linear layer.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_classes: int = 2,
        dropout_p:   float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, features):
        """
        Args
        ----
        features : Tensor, shape (B, feature_dim)

        Returns
        -------
        logits : Tensor, shape (B, num_classes)
        """
        return self.head(features)
