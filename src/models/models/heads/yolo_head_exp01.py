"""
yolo_head_exp01.py
============
Anchor-Free YOLO Detection Head for 1-D PD signals.

This module is the "interpreter" - it takes the rich spatial feature map
produced by the backbone and projects it into per-cell predictions.

It knows nothing about the backbone internals; it only cares about the
feature-map shape (B, feat_channels, S) it receives.

Output layout  (per grid cell)
-------------------------------
Index 0  - raw objectness logit      (sigmoid -> probability of a pulse here)
Index 1  - raw centre-offset logit   (sigmoid -> fractional offset in cell [0,1])
Index 2  - raw log-width             (exp -> normalised width > 0)
Index 3  - raw class logit for PD0   }  (softmax / cross-entropy over indices 3:5)
Index 4  - raw class logit for PD1   }

Output tensor shape:  (B, grid_cells, 5)

Why raw logits?
---------------
We do NOT apply sigmoid/softmax here.  Instead, the loss functions receive raw
logits and apply the non-linearities internally.  This is numerically more
stable (BCEWithLogitsLoss, CrossEntropyLoss).
Only the decoding function in task_exp01.py applies sigmoid/softmax.
"""

import torch
import torch.nn as nn


class YOLOHead(nn.Module):
    """
    Projects a (B, feat_channels, S) backbone feature map into
    (B, S, num_preds) YOLO predictions.

    Args:
        feat_channels: Number of channels in the incoming feature map
                       (must match the backbone's output channels).
        num_classes:   Number of PD classes (2 for PD0 vs PD1).
        grid_cells:    Number of spatial grid cells S (must match backbone).

    num_preds = 3 + num_classes
        [obj_logit, centre_logit, log_width, cls_logit_0, cls_logit_1, ...]
    """

    def __init__(
        self,
        feat_channels: int = 256,
        num_classes:   int = 2,
        grid_cells:    int = 32,
    ):
        super().__init__()
        self.grid_cells = grid_cells
        self.num_preds  = 3 + num_classes   # 5 for num_classes=2

        # A 1x1 convolution acts as a learned linear projection at each of
        # the S spatial positions independently - no information mixing across
        # adjacent cells.
        self.conv = nn.Conv1d(
            in_channels=feat_channels,
            out_channels=self.num_preds,
            kernel_size=1,
            bias=True,
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (B, feat_channels, S)  - backbone output

        Returns:
            preds: (B, S, num_preds)  - raw predictions per grid cell
        """
        # (B, feat_channels, S) -> (B, num_preds, S)
        x = self.conv(feature_map)

        # Permute to put predictions last: (B, S, num_preds)
        # Just rearranges the data, nothing else done.
        x = x.permute(0, 2, 1).contiguous()

        return x
