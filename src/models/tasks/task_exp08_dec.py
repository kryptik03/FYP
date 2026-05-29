"""
task_exp08_dec.py
=================
Two-Phase Semi-Supervised Spherical Deep Embedded Clustering (DEC) Task — Exp08.

ARCHITECTURE CHANGE vs Exp07
-----------------------------
The backbone is replaced from `DECCNN2D_Exp07` (lightweight 2D CNN)
to `DECViT_Exp08` (Vision Transformer with 1→3 channel adapter + projector).

All other logic is IDENTICAL to Exp07:
  - Phase 1  : Supervised Contrastive Learning (SupCon) on augmented pairs.
  - Phase 2  : Semi-Supervised Spherical DEC with pairwise constraints.
  - Losses   : SupCon, KL-divergence, Pairwise Must-Link/Cannot-Link.
  - Cluster  : Soft Student-t assignments, K-Means centroid initialisation.
  - Embedding: L2-normalised 128-D (unit hypersphere geometry).

PHASE 1  — SupCon Self-Supervised Pre-Training (2D Bispectra)
-------------------------------------------------------------
Loss: Supervised Contrastive Learning (SupCon).
Input batches: (view1, view2, reported_class, ...)
For labeled samples (reported_class ≠ -1): all same-class pairs are attracted.
For unlabeled samples (reported_class == -1): only the augmented view is a positive.

PHASE 2  — Semi-Supervised Spherical DEC
-----------------------------------------
Loss: KL-divergence (unsupervised) + Pairwise Constraint Loss (supervised).
Uses reported_class to identify Must-Link and Cannot-Link pairs.
Cluster centroids are L2-normalised (spherical geometry).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.models.backbones.backbone_exp08 import DECViT_Exp08


class SupConDECTask_Exp08(nn.Module):
    """
    Thin wrapper that combines the DECViT_Exp08 backbone with the DEC cluster
    layer and exposes the standard Phase 1 / Phase 2 training API used by
    train_exp08.py.

    Config keys read from `config`:
      model.in_channels        : always 1 (bispectrum)
      model.embedding_dim      : 128
      model.n_clusters         : number of PD classes to discover
      model.vit_variant        : "vit_b_16" or "vit_b_32"
      model.pretrained         : bool (default True)
      task.simclr_temperature  : SupCon temperature τ (default 0.5)
      task.pairwise_weight_gamma: weight for pairwise loss in Phase 2
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        task_cfg  = config.get("task", {})

        self.embedding_dim         = model_cfg["embedding_dim"]
        self.n_clusters            = model_cfg["n_clusters"]
        self.temperature           = task_cfg.get("simclr_temperature", 0.5)
        self.pairwise_weight_gamma = task_cfg.get("pairwise_weight_gamma", 1.0)
        self.alpha                 = 1.0   # Student-t degree of freedom

        # ------------------------------------------------------------------ #
        # ViT Backbone (replaces the CNN of Exp07)                            #
        # ------------------------------------------------------------------ #
        self.backbone = DECViT_Exp08(
            in_channels   = model_cfg.get("in_channels", 1),
            embedding_dim = self.embedding_dim,
            vit_variant   = model_cfg.get("vit_variant", "vit_b_16"),
            pretrained    = model_cfg.get("pretrained", True),
        )

        # ------------------------------------------------------------------ #
        # DEC Cluster Layer                                                   #
        # ------------------------------------------------------------------ #
        self.cluster_layer = nn.Parameter(
            torch.zeros(self.n_clusters, self.embedding_dim)
        )

        self._phase     = 1
        self._optimizer = None

    # ======================================================================= #
    # Phase Management                                                         #
    # ======================================================================= #

    def set_phase(self, phase: int, lr: float):
        """Switch between Phase 1 (backbone only) and Phase 2 (full model)."""
        self._phase = phase
        if phase == 1:
            # Only update backbone parameters during SupCon pre-training
            self._optimizer = torch.optim.Adam(self.backbone.parameters(), lr=lr)
        else:
            # Update everything (backbone + cluster layer) during DEC
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print(f"[SupConDECTask_Exp08] Switched to Phase {phase} | lr={lr:.2e}")

    # ======================================================================= #
    # SupCon Phase 1                                                           #
    # ======================================================================= #

    def _supcon_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supervised Contrastive Loss (Khosla et al., 2020) with semi-supervised
        extension: unlabeled samples (label == -1) only attract their own
        augmented view.

        Args:
            z1, z2 : L2-normalised embeddings, shape (B, D).
            labels : reported class IDs, shape (B,); -1 = unlabeled.

        Returns:
            Scalar loss tensor.
        """
        device = z1.device
        N      = z1.shape[0]

        # Concatenate both views → (2N, D)
        z = torch.cat([z1, z2], dim=0)
        y = torch.cat([labels, labels], dim=0)   # (2N,)

        # Cosine similarity matrix (z is already L2-normalised)
        sim = torch.mm(z, z.T) / self.temperature   # (2N, 2N)

        # Self-similarity mask
        self_mask = torch.eye(2 * N, device=device, dtype=torch.bool)

        # Positive pair mask (writable float tensor to avoid in-place errors)
        pos_mask = torch.zeros(2 * N, 2 * N, device=device, dtype=torch.bool)

        # Labeled samples: all same-class pairs are positives
        labeled_mask  = (y != -1)
        label_matrix  = y.unsqueeze(0) == y.unsqueeze(1)
        labeled_pos   = label_matrix & labeled_mask.unsqueeze(0) & labeled_mask.unsqueeze(1)
        pos_mask |= labeled_pos

        # Unlabeled samples: only the augmented view (at index i+N mod 2N) is a positive
        aug_idx           = (torch.arange(2 * N, device=device) + N) % (2 * N)
        unlabeled_indices = torch.where(y == -1)[0]
        if len(unlabeled_indices) > 0:
            pos_mask[unlabeled_indices, aug_idx[unlabeled_indices]] = True

        # Remove self-pairs from positives
        pos_mask.masked_fill_(self_mask, False)

        # Numerically stable log-softmax (subtract row max before exp)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_c      = sim - sim_max.detach()
        exp_sim    = torch.exp(sim_c) * (~self_mask).float()
        log_prob   = sim_c - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positives
        pos_f   = pos_mask.float()
        num_pos = torch.clamp(pos_f.sum(dim=1), min=1.0)
        loss    = -(pos_f * log_prob).sum(dim=1) / num_pos
        return loss.mean()

    def training_step_phase1(self, batch: tuple) -> tuple:
        """
        One training step for SupCon Phase 1.

        Args:
            batch : (view1, view2, reported_class, ...)
                    view1/view2 have shape (B, 1, 224, 224).

        Returns:
            (z1, loss_dict)  where loss_dict = {"total": ..., "supcon": ...}
        """
        view1          = batch[0]   # (B, 1, 224, 224)
        view2          = batch[1]   # (B, 1, 224, 224)
        reported_class = batch[2]   # (B,)

        self._optimizer.zero_grad()
        z1 = self.backbone(view1)
        z2 = self.backbone(view2)

        loss = self._supcon_loss(z1, z2, reported_class)
        loss.backward()
        self._optimizer.step()

        return z1, {"total": loss.item(), "supcon": loss.item()}

    # ======================================================================= #
    # Semi-Supervised DEC Phase 2                                              #
    # ======================================================================= #

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """
        Student-t soft cluster assignment on the unit hypersphere.

        q[i,k] ∝ (1 + ||z_i − μ_k||² / α)^{-(α+1)/2}

        Centroids μ_k are L2-normalised to keep them on the sphere.
        """
        centroids = F.normalize(self.cluster_layer, p=2, dim=1)  # (K, D)
        diff      = z.unsqueeze(1) - centroids.unsqueeze(0)       # (B, K, D)
        dist_sq   = (diff ** 2).sum(dim=2)                        # (B, K)
        q         = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q         = q / q.sum(dim=1, keepdim=True)
        return q   # (B, K)

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Sharpen the soft assignments to produce a high-confidence target:
            p[i,k] = (q[i,k]² / f_k) / Σ_k (q[i,k]² / f_k)
        where f_k = Σ_i q[i,k] is the soft cluster frequency.
        """
        f = q.sum(dim=0)          # (K,)
        p = q ** 2 / f            # (B, K)
        p = p / p.sum(dim=1, keepdim=True)
        return p                  # (B, K)

    def _pairwise_constraint_loss(
        self,
        q: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Must-Link (ML) and Cannot-Link (CL) pairwise penalty on soft assignments.
        Samples with label == -1 are excluded (unlabeled → no constraint).

        ML loss: pull together soft assignments of same-class pairs.
        CL loss: push apart soft assignments of different-class pairs.
        """
        valid_mask    = labels != -1
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=q.device)

        q_valid      = q[valid_indices]                                     # (V, K)
        labels_valid = labels[valid_indices]                                # (V,)

        # Soft-assignment similarity matrix: S[i,j] = q_i · q_j
        sim_matrix = torch.matmul(q_valid, q_valid.T)                      # (V, V)

        # Build ML and CL boolean masks (exclude diagonal)
        labels_matrix = labels_valid.unsqueeze(0) == labels_valid.unsqueeze(1)  # (V, V)
        eye           = torch.eye(len(valid_indices), dtype=torch.bool, device=q.device)
        ml_mask       = labels_matrix & ~eye
        cl_mask       = ~labels_matrix & ~eye

        loss_ml = (
            -torch.log(sim_matrix[ml_mask] + 1e-8).mean()
            if ml_mask.any()
            else torch.tensor(0.0, device=q.device)
        )
        loss_cl = (
            -torch.log(1.0 - sim_matrix[cl_mask] + 1e-8).mean()
            if cl_mask.any()
            else torch.tensor(0.0, device=q.device)
        )

        return loss_ml + loss_cl

    def training_step_phase2(self, batch: tuple) -> tuple:
        """
        One training step for Semi-Supervised DEC Phase 2.

        Args:
            batch : (signal, reported_class)
                    signal has shape (B, 1, 224, 224).

        Returns:
            (q, loss_dict)  where loss_dict = {"total":..., "kl_div":..., "pairwise":...}
        """
        signal         = batch[0]   # (B, 1, 224, 224)
        reported_class = batch[1]   # (B,)

        self._optimizer.zero_grad()
        z = self.backbone(signal)
        q = self.soft_assign(z)
        p = self.target_distribution(q)

        # KL divergence loss (detach p to prevent cluster collapse)
        loss_kl = F.kl_div(q.log(), p.detach(), reduction="batchmean")
        loss_pw = self._pairwise_constraint_loss(q, reported_class)

        loss = loss_kl + self.pairwise_weight_gamma * loss_pw
        loss.backward()
        self._optimizer.step()

        return q, {
            "total":    loss.item(),
            "kl_div":   loss_kl.item(),
            "pairwise": loss_pw.item(),
        }

    # ======================================================================= #
    # Shared Validation                                                        #
    # ======================================================================= #

    def validation_step(self, batch: tuple) -> dict:
        with torch.no_grad():
            if self._phase == 1:
                view1, view2, reported_class = batch[0], batch[1], batch[2]
                z1 = self.backbone(view1)
                z2 = self.backbone(view2)
                loss   = self._supcon_loss(z1, z2, reported_class)
                result = {"total": loss.item(), "supcon": loss.item()}
            else:
                signal, reported_class = batch[0], batch[1]
                z  = self.backbone(signal)
                q  = self.soft_assign(z)
                p  = self.target_distribution(q)
                loss_kl = F.kl_div(q.log(), p.detach(), reduction="batchmean")
                loss_pw = self._pairwise_constraint_loss(q, reported_class)
                loss    = loss_kl + self.pairwise_weight_gamma * loss_pw
                result  = {
                    "total":    loss.item(),
                    "kl_div":   loss_kl.item(),
                    "pairwise": loss_pw.item(),
                }
        return result

    # ======================================================================= #
    # Cluster Centroid Initialisation                                          #
    # ======================================================================= #

    def init_cluster_centroids(self, all_embeddings: np.ndarray):
        """
        Fit K-Means on the collected embeddings and L2-normalise the centroids
        so they live on the unit hypersphere (required for Spherical DEC).
        """
        from sklearn.cluster import KMeans
        print(
            f"[DEC] Running K-Means (K={self.n_clusters}) "
            f"on {len(all_embeddings):,} embeddings..."
        )
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        km.fit(all_embeddings)
        centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        centroids = F.normalize(centroids, p=2, dim=1)   # Spherical constraint
        self.cluster_layer.data = centroids.to(self.cluster_layer.device)
        print("[DEC] Cluster centroids initialised and L2-normalised.")

    # ======================================================================= #
    # Forward & Checkpoint                                                     #
    # ======================================================================= #

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        q = self.soft_assign(z)
        return z, q

    def save_checkpoint(
        self,
        epoch: int,
        node_id: str,
        weights_dir: str,
        config_dir: str,
        config_path: str,
    ):
        import shutil
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(config_dir,  exist_ok=True)

        weight_path = os.path.join(weights_dir, f"model_{node_id}.pt")
        torch.save({
            "epoch":         epoch,
            "node_id":       node_id,
            "model_state":   self.state_dict(),
            "n_clusters":    self.n_clusters,
            "embedding_dim": self.embedding_dim,
        }, weight_path)

        config_snap = os.path.join(config_dir, f"config_{node_id}.yaml")
        shutil.copy2(config_path, config_snap)
        print(f"  [Checkpoint] Saved → {weight_path}")
