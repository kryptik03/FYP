"""
task_exp01.py
=================
Task logic for joint PD detection (bounding boxes) and classification.

This class is the "brain" of the training loop.  It owns:
    - The assembled model (backbone + head)
    - The optimizer
    - The compound YOLO loss function
    - Training and validation step logic
    - Prediction decoding (raw logits -> human-readable boxes)
    - Checkpoint saving

Nothing in this file knows about HDF5 files, shard paths, or YAML — those
concerns belong to the dataset and train.py respectively.

Loss formulation
----------------
For each batch of shape (B, S, 5):

    preds[:, :, 0]   = raw objectness logit
    preds[:, :, 1]   = raw centre-offset logit  (target: sigmoid applied)
    preds[:, :, 2]   = raw log-width            (target: log applied)
    preds[:, :, 3:5] = raw class logits

For each batch of targets (B, S, 4):

    targets[:, :, 0] = objectness  (0.0 or 1.0)
    targets[:, :, 1] = centre_offset in [0, 1]
    targets[:, :, 2] = log_width
    targets[:, :, 3] = class_id (float, cast to long for CrossEntropy)

Total loss
----------
    L = λ_obj · BCE_obj  +  λ_box · SmoothL1_box  +  λ_cls · CE_cls

    BCE_obj  — applied to ALL S cells (objectness must be low everywhere
               except where a real pulse exists)
    SmoothL1_box — applied ONLY to positive cells (where target obj == 1)
    CE_cls   — applied ONLY to positive cells

Positive-cell weighting
-----------------------
~85 % of grid cells are background (obj=0).  Left unweighted, the model would
learn to predict "no pulse" everywhere.  We pass pos_weight to
BCEWithLogitsLoss so positive cells contribute proportionally more to the
objectness loss.

Coordinate decoding
-------------------
centre_norm(i) = (i + sigmoid(pred_centre)) / S
width_norm     = exp(pred_logwidth)
start_norm     = centre_norm - width_norm / 2
end_norm       = centre_norm + width_norm / 2

All normalised values ∈ [0, 1].  Multiply by seq_len to get decimated sample
indices, then multiply by decimation_factor to recover raw sample indices.
"""

import os
import shutil

import torch
import torch.nn as nn

from ..models.backbones.cnn_1d_exp01  import CNN1DBackbone
from ..models.heads.yolo_head_exp01   import YOLOHead


class DetectionTask(nn.Module):
    """
    Encapsulates the model, optimizer, and loss for the YOLO1D detection task.

    Args:
        config: The full experiment config dict parsed from the YAML.
    """

    def __init__(self, config: dict):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Unpack config sub-sections                                           #
        # ------------------------------------------------------------------ #
        model_cfg   = config["model"]
        task_cfg    = config["task"]
        train_cfg   = config["training"]
        data_cfg    = config["data"]

        self.num_classes       = task_cfg["num_classes"]        # 2
        self.grid_cells        = data_cfg["grid_cells"]          # 32
        self.seq_len           = (500_001 - data_cfg["decimation_factor"]) \
                                  // data_cfg["decimation_factor"] + 1  # 1000
        self.decimation_factor = data_cfg["decimation_factor"]   # 500

        # Loss weights
        self.lambda_obj = task_cfg["lambda_obj"]    # 1.0
        self.lambda_box = task_cfg["lambda_box"]    # 5.0
        self.lambda_cls = task_cfg["lambda_cls"]    # 1.0

        # ------------------------------------------------------------------ #
        # Build model: backbone -> head assembled into a single nn.Module      #
        # ------------------------------------------------------------------ #
        self.backbone = CNN1DBackbone(
            in_channels   = model_cfg["in_channels"],    # 1
            base_channels = model_cfg["base_channels"],  # 32
            grid_cells    = self.grid_cells,             # 32
        )
        self.head = YOLOHead(
            feat_channels = model_cfg["base_channels"] * 8,  # 256
            num_classes   = self.num_classes,                 # 2
            grid_cells    = self.grid_cells,                  # 32
        )

        # ------------------------------------------------------------------ #
        # Loss functions                                                       #
        # ------------------------------------------------------------------ #
        pos_weight_val = torch.tensor([task_cfg["pos_weight"]])  # e.g. 5.0
        # Initialise pos_weight for BCEWithLogitsLoss
        self.bce_obj   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)

        self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_ent = nn.CrossEntropyLoss()

        # ------------------------------------------------------------------ #
        # Optimizer                                                            #
        # ------------------------------------------------------------------ #
        lr = train_cfg["learning_rate"]   # 0.001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: (B, 1, seq_len)   — decimated single-channel waveform

        Returns:
            preds: (B, S, 5)          — raw per-cell predictions
        """
        features = self.backbone(signal)    # (B, 256, S)
        preds    = self.head(features)      # (B, S, 5)
        return preds

    # ------------------------------------------------------------------ #
    # Loss computation                                                     #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        preds:   torch.Tensor,   # (B, S, 5)  raw logits from forward()
        targets: torch.Tensor,   # (B, S, 4)  target grid from dataset
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the compound YOLO loss.

        Returns
        -------
        total_loss : scalar tensor (differentiable)
        loss_dict  : {'obj': float, 'box': float, 'cls': float, 'total': float}
                     for logging / printing
        """
        # ---------------------------------------------------------------- #
        # Split predictions and targets                                     #
        # ---------------------------------------------------------------- #
        pred_obj    = preds[..., 0]        # (B, S) raw objectness logit
        pred_centre = preds[..., 1]        # (B, S) raw centre-offset logit
        pred_logw   = preds[..., 2]        # (B, S) raw log-width
        pred_cls    = preds[..., 3:]       # (B, S, num_classes) raw class logits

        tgt_obj     = targets[..., 0]      # (B, S) 0.0 or 1.0
        tgt_centre  = targets[..., 1]      # (B, S) [0, 1]
        tgt_logw    = targets[..., 2]      # (B, S) log-space width
        tgt_class   = targets[..., 3].long()  # (B, S) 0 or 1  (int)

        # Boolean mask of cells that contain a real PD pulse
        pos_mask = tgt_obj > 0.5           # (B, S)

        # ---------------------------------------------------------------- #
        # Objectness loss — ALL cells                                        #
        # ---------------------------------------------------------------- #
        # Move pos_weight to same device as predictions
        # Remember, pos_weight is the weight for positive samples (the "1" in binary classification)
        # Defined in the config file.
        self.bce_obj.pos_weight = self.bce_obj.pos_weight.to(preds.device)
        # Calculates loss based on the BCEWithLogitsLoss formula.
        obj_loss = self.bce_obj(pred_obj, tgt_obj)

        # ---------------------------------------------------------------- #
        # Box regression loss — positive cells only                         #
        # ---------------------------------------------------------------- #
        if pos_mask.any():
            # Centre: the model predicts a raw logit; we apply sigmoid so
            # the prediction is bounded to [0, 1], matching the target.
            pred_centre_pos = torch.sigmoid(pred_centre[pos_mask])
            tgt_centre_pos  = tgt_centre[pos_mask]

            # Log-width: predicted directly in log space, target is also
            # in log space, so SmoothL1 operates in the same domain.
            pred_logw_pos   = pred_logw[pos_mask]
            tgt_logw_pos    = tgt_logw[pos_mask]

            box_loss = self.smooth_l1(pred_centre_pos, tgt_centre_pos) \
                     + self.smooth_l1(pred_logw_pos,   tgt_logw_pos)
        else:
            # No positive cells in this batch (rare for empty-scene batches)
            box_loss = torch.tensor(0.0, requires_grad=True,
                                    device=preds.device)

        # ---------------------------------------------------------------- #
        # Classification loss — positive cells only                         #
        # ---------------------------------------------------------------- #
        if pos_mask.any():
            # pred_cls[pos_mask]: (N_pos, num_classes)
            # tgt_class[pos_mask]: (N_pos,) long tensor
            cls_loss = self.cross_ent(pred_cls[pos_mask], tgt_class[pos_mask])
        else:
            cls_loss = torch.tensor(0.0, requires_grad=True,
                                    device=preds.device)

        # ---------------------------------------------------------------- #
        # Weighted total                                                     #
        # ---------------------------------------------------------------- #
        total_loss = (
            self.lambda_obj * obj_loss
            + self.lambda_box * box_loss
            + self.lambda_cls * cls_loss
        )

        loss_dict = {
            "obj":   obj_loss.item(),
            "box":   box_loss.item(),
            "cls":   cls_loss.item(),
            "total": total_loss.item(),
        }
        return total_loss, loss_dict

    # ------------------------------------------------------------------ #
    # Training step                                                         #
    # ------------------------------------------------------------------ #

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """
        One mini-batch forward + backward + optimiser step.

        Args:
            batch: (signal, target) tensors already on the correct device.

        Returns:
            total_loss, loss_dict
        """
        self.train()
        signal, target = batch

        self.optimizer.zero_grad()
        preds = self.forward(signal)                    # (B, S, 5)
        total_loss, loss_dict = self.compute_loss(preds, target)
        total_loss.backward()
        self.optimizer.step()

        return total_loss, loss_dict

    # ------------------------------------------------------------------ #
    # Validation step                                                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> dict:
        """
        One mini-batch forward pass for validation (no gradient computation).

        Computes:
            - All three loss components
            - Mean IoU over positive cells
            - Classification accuracy over positive cells

        Returns:
            metrics dict with keys: 'obj', 'box', 'cls', 'total', 'iou', 'cls_acc'
        """
        self.eval()
        signal, target = batch

        preds = self.forward(signal)                    # (B, S, 5)
        _, loss_dict = self.compute_loss(preds, target)

        # ---------------------------------------------------------------- #
        # Detection IoU (in normalised [0,1] coordinate space)              #
        # ---------------------------------------------------------------- #
        S       = self.grid_cells
        tgt_obj = target[..., 0]               # (B, S)
        pos_mask = tgt_obj > 0.5

        iou_mean = 0.0
        cls_acc  = 0.0

        if pos_mask.any():
            # --- Decode predicted boxes ---
            # Cell index tensor: shape (S,) -> broadcast to (B, S)
            cell_indices = torch.arange(S, device=preds.device).float()
            cell_indices = cell_indices.unsqueeze(0)               # (1, S)

            pred_centre_norm = (cell_indices + torch.sigmoid(preds[..., 1])) / S
            pred_width_norm  = torch.exp(preds[..., 2])
            pred_start_norm  = pred_centre_norm - pred_width_norm / 2
            pred_end_norm    = pred_centre_norm + pred_width_norm / 2

            # --- Decode target boxes ---
            tgt_centre_norm = (cell_indices + target[..., 1]) / S
            tgt_width_norm  = torch.exp(target[..., 2])
            tgt_start_norm  = tgt_centre_norm - tgt_width_norm / 2
            tgt_end_norm    = tgt_centre_norm + tgt_width_norm / 2

            # --- IoU for positive cells ---
            p_s = pred_start_norm[pos_mask]
            p_e = pred_end_norm[pos_mask]
            t_s = tgt_start_norm[pos_mask]
            t_e = tgt_end_norm[pos_mask]

            inter_start = torch.max(p_s, t_s)
            inter_end   = torch.min(p_e, t_e)
            inter       = (inter_end - inter_start).clamp(min=0)
            union       = (p_e - p_s) + (t_e - t_s) - inter
            iou         = inter / union.clamp(min=1e-6)
            iou_mean    = iou.mean().item()

            # --- Classification accuracy at positive cells ---
            pred_class_ids = preds[..., 3:][pos_mask].argmax(dim=-1)  # (N_pos,)
            true_class_ids = target[..., 3][pos_mask].long()
            cls_acc = (pred_class_ids == true_class_ids).float().mean().item()

        loss_dict["iou"]     = iou_mean
        loss_dict["cls_acc"] = cls_acc
        return loss_dict

    # ------------------------------------------------------------------ #
    # Prediction decoding                                                   #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def decode_predictions(
        self,
        preds:     torch.Tensor,   # (B, S, 5) — output of forward()
        threshold: float = 0.5,    # objectness probability cutoff
    ) -> list[list[dict]]:
        """
        Convert raw grid predictions into human-readable detection results.

        Returns a list (one entry per sample in the batch) of lists of dicts:
            {
              'start_raw'  : int,    raw sample index (0-indexed)
              'end_raw'    : int,    raw sample index (0-indexed)
              'class_id'   : int,    predicted PD class (0 or 1)
              'obj_score'  : float,  objectness probability
              'cls_score'  : float,  class confidence
            }
        """
        B, S, _ = preds.shape
        cell_indices = torch.arange(S, device=preds.device).float()

        obj_scores    = torch.sigmoid(preds[..., 0])          # (B, S)
        centre_norms  = (cell_indices + torch.sigmoid(preds[..., 1])) / S
        width_norms   = torch.exp(preds[..., 2])
        cls_probs     = torch.softmax(preds[..., 3:], dim=-1) # (B, S, 2)

        results = []
        for b in range(B):
            sample_results = []
            for s in range(S):
                obj_score = obj_scores[b, s].item()
                if obj_score < threshold:
                    continue

                centre_n = centre_norms[b, s].item()
                width_n  = width_norms[b, s].item()
                start_n  = centre_n - width_n / 2
                end_n    = centre_n + width_n / 2

                # Convert normalised -> decimated -> raw sample index
                start_dec = int(start_n * self.seq_len)
                end_dec   = int(end_n   * self.seq_len)
                start_raw = start_dec * self.decimation_factor
                end_raw   = end_dec   * self.decimation_factor

                cls_prob_arr = cls_probs[b, s].cpu().tolist()
                class_id     = int(torch.argmax(cls_probs[b, s]).item())
                cls_score    = cls_prob_arr[class_id]

                sample_results.append({
                    "start_raw":  start_raw,
                    "end_raw":    end_raw,
                    "class_id":   class_id,
                    "obj_score":  obj_score,
                    "cls_score":  cls_score,
                })
            results.append(sample_results)

        return results

    # ------------------------------------------------------------------ #
    # Checkpoint saving                                                     #
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
        Save model weights and a copy of the experiment config.

        Args:
            epoch:        Current training epoch number (for display only).
            node_id:      The lineage NodeID for this run (e.g. 'aB3x').
            weights_dir:  Directory where .pt files are saved.
            config_dir:   Directory where config snapshots are saved.
            config_path:  Path to the original YAML (to snapshot alongside weights).
        """
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(config_dir,  exist_ok=True)

        weights_path = os.path.join(weights_dir, f"model_{node_id}.pt")
        config_snap  = os.path.join(config_dir,  f"config_{node_id}.yaml")

        # Save full model state (backbone + head weights + optimizer state)
        torch.save({
            "epoch":       epoch,
            "node_id":     node_id,
            "model_state": self.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }, weights_path)

        # Snapshot the YAML that produced this run
        shutil.copy(config_path, config_snap)

        print(f"[Task] Checkpoint saved -> {weights_path}")
        print(f"[Task] Config snapshot  -> {config_snap}")
