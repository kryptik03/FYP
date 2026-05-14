"""
task_exp03_contrastive.py
=========================
Joint Contrastive Embedding + Classification task logic.

Computes:
1. TripletMarginLoss on (Anchor, Positive, Negative) embeddings.
2. CrossEntropyLoss on the classification of the Anchor embedding.
"""

import os
import shutil

import torch
import torch.nn as nn

from ..models.backbones.backbone_exp03_cnn1d import ContrastiveCNN1D
from ..models.heads.head_exp03_classification import ClassificationHead


class ContrastiveTask(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        model_cfg   = config["model"]
        task_cfg    = config["task"]
        train_cfg   = config["training"]

        self.num_classes   = task_cfg["num_classes"]
        self.lambda_trip   = task_cfg.get("lambda_triplet", 1.0)
        self.lambda_cls    = task_cfg.get("lambda_cls", 1.0)
        
        # ------------------------------------------------------------------ #
        # Build model blocks                                                   #
        # ------------------------------------------------------------------ #
        self.backbone = ContrastiveCNN1D(
            in_channels   = model_cfg["in_channels"],
            base_channels = model_cfg["base_channels"],
            embedding_dim = model_cfg.get("embedding_dim", 128)
        )
        self.head = ClassificationHead(
            embedding_dim = model_cfg.get("embedding_dim", 128),
            num_classes   = self.num_classes
        )

        # ------------------------------------------------------------------ #
        # Loss functions                                                       #
        # ------------------------------------------------------------------ #
        margin = task_cfg.get("margin", 1.0)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        self.cls_loss     = nn.CrossEntropyLoss()

        # ------------------------------------------------------------------ #
        # Optimizer                                                            #
        # ------------------------------------------------------------------ #
        lr = train_cfg["learning_rate"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            emb:    (B, embedding_dim) L2-normalized feature vector
            logits: (B, num_classes) classification logits
        """
        emb = self.backbone(signal)
        logits = self.head(emb)
        return emb, logits

    def training_step(self, batch: tuple) -> tuple[torch.Tensor, dict]:
        self.train()
        anc_sig, pos_sig, neg_sig, class_id = batch

        self.optimizer.zero_grad()
        
        # Pass all three through the backbone
        emb_anc, logits_anc = self.forward(anc_sig)
        emb_pos, _          = self.forward(pos_sig)
        emb_neg, _          = self.forward(neg_sig)
        
        # 1. Contrastive Loss
        loss_trip = self.triplet_loss(emb_anc, emb_pos, emb_neg)
        
        # 2. Classification Loss (only on the anchor)
        loss_c = self.cls_loss(logits_anc, class_id)
        
        # 3. Joint Loss
        total_loss = self.lambda_trip * loss_trip + self.lambda_cls * loss_c
        
        total_loss.backward()
        self.optimizer.step()

        loss_dict = {
            "triplet": loss_trip.item(),
            "cls":     loss_c.item(),
            "total":   total_loss.item(),
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch: tuple) -> dict:
        self.eval()
        anc_sig, pos_sig, neg_sig, class_id = batch
        
        emb_anc, logits_anc = self.forward(anc_sig)
        emb_pos, _          = self.forward(pos_sig)
        emb_neg, _          = self.forward(neg_sig)
        
        loss_trip = self.triplet_loss(emb_anc, emb_pos, emb_neg)
        loss_c    = self.cls_loss(logits_anc, class_id)
        total_loss = self.lambda_trip * loss_trip + self.lambda_cls * loss_c
        
        # Classification accuracy on Anchor
        pred_classes = logits_anc.argmax(dim=-1)
        cls_acc = (pred_classes == class_id).float().mean().item()
        
        loss_dict = {
            "triplet": loss_trip.item(),
            "cls":     loss_c.item(),
            "total":   total_loss.item(),
            "cls_acc": cls_acc,
        }
        return loss_dict

    def save_checkpoint(
        self,
        epoch:       int,
        node_id:     str,
        weights_dir: str,
        config_dir:  str,
        config_path: str,
    ):
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(config_dir,  exist_ok=True)

        weights_path = os.path.join(weights_dir, f"model_{node_id}.pt")
        config_snap  = os.path.join(config_dir,  f"config_{node_id}.yaml")

        torch.save({
            "epoch":       epoch,
            "node_id":     node_id,
            "model_state": self.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }, weights_path)

        shutil.copy(config_path, config_snap)

        print(f"[Task] Checkpoint saved -> {weights_path}")
        print(f"[Task] Config snapshot  -> {config_snap}")
