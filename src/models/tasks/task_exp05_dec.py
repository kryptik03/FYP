"""
task_exp05_dec.py
=================
Two-Phase Spherical Deep Embedded Clustering (DEC) Task.

PHASE 1  — SimCLR Self-Supervised Pre-Training
----------------------------------------------
Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy).
Input batches: (view1, view2, ...) — two augmented views of each pulse.
No labels used. The backbone learns a structured 128-D embedding space.

PHASE 2  — Spherical DEC Unsupervised Cluster Refinement
--------------------------------------------------------
Loss: KL-divergence between soft cluster assignments (Q) and the sharpened
      target distribution (P).
Input batches: (signal, ...) — single (un-augmented) view of each pulse.
No labels used. Cluster centroids are initialized from K-Means on Phase-1
embeddings, then refined by minimizing KL(P || Q).
Because embeddings are L2-normalized on the unit sphere, cluster centroids
are also constrained to the unit sphere during distance computation.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.models.backbones.backbone_exp05_dec import DECCNN1D


class DECTask(nn.Module):
    """
    Wraps the backbone and DEC cluster layer.
    Exposes:
        training_step_phase1(batch) -> (emb, loss_dict)   for SimCLR
        training_step_phase2(batch) -> (q, loss_dict)     for Spherical DEC
        validation_step(batch)      -> metric_dict
        init_cluster_centroids(all_embs) -> None
        save_checkpoint(...)
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        task_cfg  = config.get("task", {})

        self.embedding_dim  = model_cfg["embedding_dim"]
        self.n_clusters     = model_cfg["n_clusters"]
        self.temperature    = task_cfg.get("simclr_temperature", 0.5)
        self.alpha          = 1.0   # Degrees of freedom for Student's t in DEC

        # ---- Backbone ------------------------------------------------------ #
        self.backbone = DECCNN1D(
            in_channels   = model_cfg["in_channels"],
            base_channels = model_cfg["base_channels"],
            embedding_dim = self.embedding_dim,
        )

        # ---- DEC Cluster Layer (K centroids in embedding space) ------------ #
        # Initialized to zeros; call init_cluster_centroids() before Phase 2.
        self.cluster_layer = nn.Parameter(
            torch.zeros(self.n_clusters, self.embedding_dim)
        )

        # ---- Training Phase Tracker ---------------------------------------- #
        self._phase = 1   # 1 = SimCLR, 2 = DEC

        # ---- Optimizer: stored here so train.py can use set_phase() -------- #
        self._optimizer = None

    # ======================================================================= #
    # Phase Control                                                             #
    # ======================================================================= #

    def set_phase(self, phase: int, lr: float):
        """Switch between Phase 1 (SimCLR) and Phase 2 (DEC). Sets a fresh optimizer."""
        self._phase = phase
        if phase == 1:
            # Optimize backbone only (cluster layer not used yet)
            self._optimizer = torch.optim.Adam(self.backbone.parameters(), lr=lr)
        else:
            # Optimize backbone + cluster centroids together
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print(f"[DECTask] Switched to Phase {phase} | lr={lr}")

    # ======================================================================= #
    # SimCLR Phase 1                                                            #
    # ======================================================================= #

    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent loss for a batch of N augmented pairs.
        z1, z2: (N, D) — already L2-normalised.
        """
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)                     # (2N, D)

        # Cosine similarity matrix
        sim = torch.mm(z, z.T) / self.temperature           # (2N, 2N)

        # Mask out self-similarity
        mask = torch.eye(2 * N, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N,     device=z.device),
        ])                                                   # (2N,)

        loss = F.cross_entropy(sim, labels)
        return loss

    def training_step_phase1(self, batch: tuple) -> tuple:
        """
        batch: (view1, view2, class_id, inst_id, shard_path, start_idx, time_res)
        Returns: (embeddings_z1, loss_dict)
        """
        view1 = batch[0]   # (B, 1, L)
        view2 = batch[1]   # (B, 1, L)

        self._optimizer.zero_grad()
        z1 = self.backbone(view1)   # (B, D)
        z2 = self.backbone(view2)   # (B, D)

        loss = self._nt_xent_loss(z1, z2)
        loss.backward()
        self._optimizer.step()

        return z1, {"total": loss.item(), "simclr": loss.item()}

    # ======================================================================= #
    # Spherical DEC Phase 2                                                     #
    # ======================================================================= #

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """
        Student's t-distribution soft assignment.
        z: (B, D), centroids: (K, D)
        Returns q: (B, K) — soft cluster membership probabilities.
        """
        # Enforce L2 normalization on centroids so they lie on the unit sphere
        centroids = F.normalize(self.cluster_layer, p=2, dim=1)

        # Squared Euclidean distances (B, K) between normalized vectors
        diff = z.unsqueeze(1) - centroids.unsqueeze(0)   # (B, K, D)
        dist_sq = (diff ** 2).sum(dim=2)                 # (B, K)

        # Student's t-distribution with alpha=1
        q = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)   # Normalize to sum=1 over K
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Sharpen Q to produce target distribution P.
        p_ij = (q_ij^2 / f_j) / sum_j(q_ij^2 / f_j)
        where f_j = sum_i(q_ij)
        """
        f = q.sum(dim=0)            # (K,) soft cluster frequencies
        p = q ** 2 / f              # (B, K)
        p = p / p.sum(dim=1, keepdim=True)
        return p

    def training_step_phase2(self, batch: tuple) -> tuple:
        """
        batch: (signal, class_id, inst_id, shard_path, start_idx, time_res)
        Returns: (q, loss_dict)
        """
        signal = batch[0]   # (B, 1, L)

        self._optimizer.zero_grad()
        z = self.backbone(signal)    # (B, D)
        q = self.soft_assign(z)      # (B, K)

        # Target distribution P is computed from the current Q
        # Detach P so it's treated as a fixed target (standard DEC practice)
        p = self.target_distribution(q).detach()

        # KL-divergence: sum_ij p_ij * log(p_ij / q_ij)
        loss = F.kl_div(q.log(), p, reduction="batchmean")
        loss.backward()
        self._optimizer.step()

        return q, {"total": loss.item(), "kl_div": loss.item()}

    # ======================================================================= #
    # Shared Validation                                                         #
    # ======================================================================= #

    def validation_step(self, batch: tuple) -> dict:
        """
        Works for both phases. In Phase 1 returns SimCLR val loss.
        In Phase 2 returns KL-div val loss.
        """
        self.eval()
        with torch.no_grad():
            if self._phase == 1:
                view1 = batch[0]
                view2 = batch[1]
                z1 = self.backbone(view1)
                z2 = self.backbone(view2)
                loss = self._nt_xent_loss(z1, z2)
                result = {"total": loss.item(), "simclr": loss.item()}
            else:
                signal = batch[0]
                z = self.backbone(signal)
                q = self.soft_assign(z)
                p = self.target_distribution(q)
                loss = F.kl_div(q.log(), p, reduction="batchmean")
                result = {"total": loss.item(), "kl_div": loss.item()}
        self.train()
        return result

    # ======================================================================= #
    # Cluster Centroid Initialization (called between Phase 1 and 2)           #
    # ======================================================================= #

    def init_cluster_centroids(self, all_embeddings: np.ndarray):
        """
        Run K-Means on all Phase-1 embeddings to initialize cluster centroids.
        all_embeddings: (N, embedding_dim) numpy array.
        """
        from sklearn.cluster import KMeans
        print(f"[DEC] Running K-Means (K={self.n_clusters}) on {len(all_embeddings)} embeddings...")
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        km.fit(all_embeddings)
        centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        
        # Normalize the initialized centroids to lie on the unit sphere
        centroids = F.normalize(centroids, p=2, dim=1)
        
        self.cluster_layer.data = centroids.to(self.cluster_layer.device)
        print(f"[DEC] Cluster centroids initialized and L2-normalized.")

    # ======================================================================= #
    # Checkpoint I/O (compatible with train.py convention)                     #
    # ======================================================================= #

    def forward(self, x: torch.Tensor):
        """Inference-time forward pass: returns (embedding, soft_assignment)."""
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
            "epoch":           epoch,
            "node_id":         node_id,
            "model_state":     self.state_dict(),
            "n_clusters":      self.n_clusters,
            "embedding_dim":   self.embedding_dim,
        }, weight_path)

        config_snap = os.path.join(config_dir, f"config_{node_id}.yaml")
        shutil.copy2(config_path, config_snap)
        print(f"  [Checkpoint] Saved -> {weight_path}")
