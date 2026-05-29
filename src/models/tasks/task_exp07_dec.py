"""
task_exp07_dec.py
=================
Two-Phase Semi-Supervised Spherical Deep Embedded Clustering (DEC) Task.

PHASE 1  — SupCon Self-Supervised Pre-Training (2D STFT)
--------------------------------------------------------
Loss: Supervised Contrastive Learning (SupCon).
Input batches: (view1, view2, reported_class, ...)

PHASE 2  — Semi-Supervised Spherical DEC 
--------------------------------------------------------
Loss: KL-divergence (unsupervised) + Pairwise Constraint Loss (supervised).
Uses reported_class to identify Must-Link and Cannot-Link pairs.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.models.backbones.backbone_exp07 import DECCNN2D_Exp07

class SupConDECTask(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        task_cfg  = config.get("task", {})

        self.embedding_dim  = model_cfg["embedding_dim"]
        self.n_clusters     = model_cfg["n_clusters"]
        self.temperature    = task_cfg.get("simclr_temperature", 0.5) # Kept name for compatibility or use supcon_temperature
        self.pairwise_weight_gamma = task_cfg.get("pairwise_weight_gamma", 1.0)
        self.alpha          = 1.0

        # ---- Backbone ------------------------------------------------------ #
        self.backbone = DECCNN2D_Exp07(
            in_channels   = model_cfg.get("in_channels", 1),
            base_channels = model_cfg["base_channels"],
            embedding_dim = self.embedding_dim,
        )

        # ---- DEC Cluster Layer --------------------------------------------- #
        self.cluster_layer = nn.Parameter(
            torch.zeros(self.n_clusters, self.embedding_dim)
        )

        self._phase = 1
        self._optimizer = None

    def set_phase(self, phase: int, lr: float):
        self._phase = phase
        if phase == 1:
            self._optimizer = torch.optim.Adam(self.backbone.parameters(), lr=lr)
        else:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print(f"[SupConDECTask] Switched to Phase {phase} | lr={lr}")

    # ======================================================================= #
    # SupCon Phase 1                                                            #
    # ======================================================================= #

    def _supcon_loss(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = z1.device
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0) # Shape: (2N, D)
        y = torch.cat([labels, labels], dim=0) # Shape: (2N)

        # Compute similarity matrix
        sim = torch.mm(z, z.T) / self.temperature

        # Mask for self-similarity
        self_mask = torch.eye(2 * N, device=device, dtype=torch.bool)

        # Build positive pair mask as a writable float tensor (avoids in-place error on derived bool tensors)
        pos_mask = torch.zeros(2 * N, 2 * N, device=device, dtype=torch.bool)

        # For labeled data (y != -1): same label attracts all same-label embeddings in the batch
        labeled_mask = (y != -1)
        label_matrix = y.unsqueeze(0) == y.unsqueeze(1)
        labeled_pos = label_matrix & labeled_mask.unsqueeze(0) & labeled_mask.unsqueeze(1)
        pos_mask |= labeled_pos

        # For unlabeled data (y == -1): only the augmented view is a positive pair
        # The augmented view of sample i is at index (i + N) % (2N)
        aug_idx = (torch.arange(2 * N, device=device) + N) % (2 * N)
        unlabeled_indices = torch.where(y == -1)[0]
        if len(unlabeled_indices) > 0:
            pos_mask[unlabeled_indices, aug_idx[unlabeled_indices]] = True

        # Remove self-pairs from the positive mask
        pos_mask.masked_fill_(self_mask, False)

        # Numerically stable log-softmax over the similarity row (excluding self)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_centered = sim - sim_max.detach()

        exp_sim = torch.exp(sim_centered) * (~self_mask).float()
        log_prob = sim_centered - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positives
        pos_mask_f = pos_mask.float()
        num_pos = pos_mask_f.sum(dim=1)
        num_pos = torch.clamp(num_pos, min=1.0)

        loss = -(pos_mask_f * log_prob).sum(dim=1) / num_pos
        return loss.mean()

    def training_step_phase1(self, batch: tuple) -> tuple:
        view1 = batch[0]   # (B, 1, F, T)
        view2 = batch[1]   # (B, 1, F, T)
        reported_class = batch[2]

        self._optimizer.zero_grad()
        z1 = self.backbone(view1)
        z2 = self.backbone(view2)

        loss = self._supcon_loss(z1, z2, reported_class)
        loss.backward()
        self._optimizer.step()

        return z1, {"total": loss.item(), "supcon": loss.item()}

    # ======================================================================= #
    # Semi-Supervised DEC Phase 2                                               #
    # ======================================================================= #

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        centroids = F.normalize(self.cluster_layer, p=2, dim=1)
        diff = z.unsqueeze(1) - centroids.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        q = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        f = q.sum(dim=0)
        p = q ** 2 / f
        p = p / p.sum(dim=1, keepdim=True)
        return p

    def _pairwise_constraint_loss(self, q: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates Must-Link and Cannot-Link pairwise penalties.
        Labels of -1 are ignored.
        """
        valid_mask = labels != -1
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=q.device)
            
        q_valid = q[valid_indices]
        labels_valid = labels[valid_indices]
        
        # Similarity matrix between assignments
        sim_matrix = torch.matmul(q_valid, q_valid.T)
        
        # Generate ML and CL masks
        labels_matrix = labels_valid.unsqueeze(0) == labels_valid.unsqueeze(1)
        eye = torch.eye(len(valid_indices), dtype=torch.bool, device=q.device)
        
        ml_mask = labels_matrix & ~eye
        cl_mask = ~labels_matrix & ~eye
        
        loss_ml = -torch.log(sim_matrix[ml_mask] + 1e-8).mean() if ml_mask.any() else torch.tensor(0.0, device=q.device)
        loss_cl = -torch.log(1.0 - sim_matrix[cl_mask] + 1e-8).mean() if cl_mask.any() else torch.tensor(0.0, device=q.device)
        
        return loss_ml + loss_cl

    def training_step_phase2(self, batch: tuple) -> tuple:
        signal = batch[0]
        reported_class = batch[1]

        self._optimizer.zero_grad()
        z = self.backbone(signal)
        q = self.soft_assign(z)
        p = self.target_distribution(q)

        # Detach p when calculating KL divergence to avoid cluster collapse
        loss_kl = F.kl_div(q.log(), p.detach(), reduction="batchmean")
        loss_pw = self._pairwise_constraint_loss(q, reported_class)
        
        loss = loss_kl + self.pairwise_weight_gamma * loss_pw
        loss.backward()
        self._optimizer.step()

        return q, {"total": loss.item(), "kl_div": loss_kl.item(), "pairwise": loss_pw.item()}

    # ======================================================================= #
    # Shared Validation                                                         #
    # ======================================================================= #

    def validation_step(self, batch: tuple) -> dict:
        with torch.no_grad():
            if self._phase == 1:
                view1, view2, reported_class = batch[0], batch[1], batch[2]
                z1, z2 = self.backbone(view1), self.backbone(view2)
                loss = self._supcon_loss(z1, z2, reported_class)
                result = {"total": loss.item(), "supcon": loss.item()}
            else:
                signal, reported_class = batch[0], batch[1]
                z = self.backbone(signal)
                q = self.soft_assign(z)
                p = self.target_distribution(q)
                
                # Detach p for validation loss consistency
                loss_kl = F.kl_div(q.log(), p.detach(), reduction="batchmean")
                loss_pw = self._pairwise_constraint_loss(q, reported_class)
                loss = loss_kl + self.pairwise_weight_gamma * loss_pw
                result = {"total": loss.item(), "kl_div": loss_kl.item(), "pairwise": loss_pw.item()}
        return result

    # ======================================================================= #
    # Cluster Centroid Initialization                                           #
    # ======================================================================= #

    def init_cluster_centroids(self, all_embeddings: np.ndarray):
        from sklearn.cluster import KMeans
        print(f"[DEC] Running K-Means (K={self.n_clusters}) on {len(all_embeddings)} embeddings...")
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        km.fit(all_embeddings)
        centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        centroids = F.normalize(centroids, p=2, dim=1)
        self.cluster_layer.data = centroids.to(self.cluster_layer.device)
        print(f"[DEC] Cluster centroids initialized and L2-normalized.")

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        q = self.soft_assign(z)
        return z, q

    def save_checkpoint(self, epoch: int, node_id: str, weights_dir: str, config_dir: str, config_path: str):
        import shutil
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(config_dir,  exist_ok=True)

        weight_path = os.path.join(weights_dir, f"model_{node_id}.pt")
        torch.save({
            "epoch":           epoch,
            "node_id":         node_id,
            "model_state":     self.state_dict(),
            "n_clusters":      self.n_clusters,
            "embedding_dim":   self.embedding_dim,
        }, weight_path)

        config_snap = os.path.join(config_dir, f"config_{node_id}.yaml")
        shutil.copy2(config_path, config_snap)
        print(f"  [Checkpoint] Saved -> {weight_path}")
