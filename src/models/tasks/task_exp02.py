"""
task_exp02.py
=======================
Task logic for standalone PD pulse classification.

This class owns:
    - The assembled model  (CNN1DClassifierBackbone + ClassificationHead)
    - The optimizer        (Adam)
    - The loss function    (CrossEntropyLoss)
    - Training and validation step logic
    - Checkpoint saving

Input contract
--------------
    batch = (signals, labels)
    signals : (B, 1, max_pulse_len)  — full-resolution, normalised pulse window
    labels  : (B,)                   — integer class index (0=PD1, 1=PD2)

Return contract
---------------
    training_step   -> (None, {"total": float, "accuracy": float})
    validation_step -> {"total": float, "accuracy": float,
                        "cls_acc_pd1": float, "cls_acc_pd2": float}
"""

import os

import torch
import torch.nn as nn
import yaml

from src.models.models.backbones.cnn_1d_exp02 import CNN1DClassifierBackbone
from src.models.models.heads.cls_head_exp02   import ClassificationHead


class ClassificationTask(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        model_cfg    = config["model"]
        train_cfg    = config["training"]

        feature_dim  = model_cfg.get("feature_dim",  128)
        num_classes  = model_cfg.get("num_classes",  2)
        dropout_p    = model_cfg.get("dropout_p",    0.3)
        lr           = train_cfg.get("lr",           1e-3)

        self.backbone  = CNN1DClassifierBackbone(
            in_channels=1, feature_dim=feature_dim
        )
        self.head      = ClassificationHead(
            feature_dim=feature_dim, num_classes=num_classes, dropout_p=dropout_p
        )
        self.loss_fn   = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, signals):
        """signals: (B, 1, L) -> logits: (B, num_classes)"""
        features = self.backbone(signals)
        return self.head(features)

    # ------------------------------------------------------------------ #
    # Training step                                                        #
    # ------------------------------------------------------------------ #

    def training_step(self, batch):
        """
        One gradient update.

        Returns
        -------
        (None, {"total": float, "accuracy": float})
        """
        self.train()
        signals, labels = batch

        logits = self(signals)
        loss   = self.loss_fn(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds    = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()

        return None, {
            "total"   : loss.item(),
            "accuracy": accuracy,
        }

    # ------------------------------------------------------------------ #
    # Validation step                                                      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def validation_step(self, batch):
        """
        One forward pass over a validation batch, no gradient.

        Returns
        -------
        dict with keys: total, accuracy, cls_acc_pd1, cls_acc_pd2
        """
        self.eval()
        signals, labels = batch

        logits = self(signals)
        loss   = self.loss_fn(logits, labels)
        preds  = logits.argmax(dim=1)

        accuracy = (preds == labels).float().mean().item()

        # Per-class accuracy
        per_cls_acc = {}
        for c in range(2):
            mask = (labels == c)
            if mask.sum() > 0:
                per_cls_acc[c] = (preds[mask] == labels[mask]).float().mean().item()
            else:
                per_cls_acc[c] = 0.0

        return {
            "total"      : loss.item(),
            "accuracy"   : accuracy,
            "cls_acc_pd1": per_cls_acc[0],
            "cls_acc_pd2": per_cls_acc[1],
        }

    # ------------------------------------------------------------------ #
    # Checkpoint I/O                                                       #
    # ------------------------------------------------------------------ #

    def save_checkpoint(
        self,
        epoch:       int,
        node_id:     str,
        weights_dir: str,
        config_dir:  str,
        config_path: str,
    ):
        """
        Saves model weights and a frozen copy of the YAML config.

        Files written
        -------------
        models/weights/model_<node_id>.pt
        models/configuration_snapshots/config_<node_id>.yaml
        """
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(config_dir,  exist_ok=True)

        weights_path = os.path.join(weights_dir, f"model_{node_id}.pt")
        torch.save(
            {"epoch": epoch, "model_state": self.state_dict()},
            weights_path,
        )

        config_snap_path = os.path.join(config_dir, f"config_{node_id}.yaml")
        with open(config_path, "r") as src, open(config_snap_path, "w") as dst:
            dst.write(src.read())
