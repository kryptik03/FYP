"""
task_exp09.py
=============
Two-Phase Physics-Invariant DEC Task for Exp09.

Combines:
  - Instance-Linked SupCon (Phase 1): positives are same Pulse_Instance_ID,
    not just augmented copies — forces Distance-Invariant embeddings.
  - Domain Adversarial Training (both phases): 4-class GRL-based DANN
    (equation / synthesised / cwru / measured) — forces Domain-Agnostic embeddings.
  - Semi-Supervised Spherical DEC (Phase 2): KL-divergence + Pairwise
    Must-Link/Cannot-Link — same as Exp07/08.

TOTAL LOSSES
------------
Phase 1:
    L_total = L_SupCon  +  dann_weight × L_domain

Phase 2:
    L_total = L_KL  +  pairwise_weight_gamma × L_pairwise  +  dann_weight × L_domain

BATCH TUPLE FORMATS
-------------------
Phase 1 (DECDataset_Exp09, augment=True):
    batch[0] = view1           (B, 2, 128, 128) tensor
    batch[1] = view2           (B, 2, 128, 128) tensor
    batch[2] = reported_class  (B,) long tensor  — -1 if unlabeled
    batch[3] = global_inst_id  (B,) long tensor  — always known
    batch[4] = domain_label    (B,) long tensor  — 0-3 or -1
    batch[5] = shard_path      list[str]
    batch[6] = pulse_idx       (B,) long tensor
    batch[7] = time_res        (B,) float tensor
    batch[8] = actual_class    (B,) long tensor

Phase 2 / Inference (DECDataset_Exp09, augment=False):
    batch[0] = signal          (B, 2, 128, 128) tensor
    batch[1] = reported_class  (B,) long tensor
    batch[2] = global_inst_id  (B,) long tensor
    batch[3] = domain_label    (B,) long tensor
    batch[4] = shard_path      list[str]
    batch[5] = pulse_idx       (B,) long tensor
    batch[6] = time_res        (B,) float tensor
    batch[7] = actual_class    (B,) long tensor
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.models.backbones.backbone_exp09 import DECViT_Exp09


class SupConDECTask_Exp09(nn.Module):
    """
    Thin wrapper combining DECViT_Exp09 with the DEC cluster layer.
    Exposes the Phase 1 / Phase 2 training API consumed by train_exp09.py.

    Config keys read from `config`:
        model.in_channels          : 2 (magnitude + phase)
        model.image_size           : 128
        model.patch_size           : 16
        model.d_model              : 384
        model.nhead                : 6
        model.depth                : 6
        model.embedding_dim        : 128
        model.n_clusters           : number of PD classes
        model.n_domains            : 4 (equation / synthesised / cwru / measured)
        task.simclr_temperature    : SupCon temperature τ (default 0.5)
        task.pairwise_weight_gamma : pairwise constraint weight in Phase 2
        task.dann_weight           : λ_dann scaling factor
        task.dann_lambda_max       : GRL lambda ceiling (ramped externally)
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        task_cfg  = config.get("task", {})

        self.embedding_dim         = model_cfg["embedding_dim"]
        self.n_clusters            = model_cfg["n_clusters"]
        self.n_domains             = model_cfg.get("n_domains", 4)
        self.temperature           = task_cfg.get("simclr_temperature", 0.5)
        self.pairwise_weight_gamma = task_cfg.get("pairwise_weight_gamma", 1.0)
        self.dann_weight           = task_cfg.get("dann_weight", 0.5)
        self.alpha                 = 1.0   # Student-t degrees of freedom

        # ------------------------------------------------------------------ #
        # ViT Backbone (2-channel input + GRL + Domain Head)                 #
        # ------------------------------------------------------------------ #
        self.backbone = DECViT_Exp09(
            in_channels   = model_cfg.get("in_channels",   2),
            image_size    = model_cfg.get("image_size",    128),
            patch_size    = model_cfg.get("patch_size",    16),
            d_model       = model_cfg.get("d_model",       384),
            nhead         = model_cfg.get("nhead",         6),
            depth         = model_cfg.get("depth",         6),
            embedding_dim = self.embedding_dim,
            n_domains     = self.n_domains,
            dann_lambda   = 0.0,   # starts at 0; train loop ramps it up
        )

        # ------------------------------------------------------------------ #
        # DEC Cluster Layer                                                   #
        # ------------------------------------------------------------------ #
        self.cluster_layer = nn.Parameter(
            torch.zeros(self.n_clusters, self.embedding_dim)
        )

        self._phase     = 1
        self._optimizer = None
        
        self.use_amp = config.get("training", {}).get("use_amp", False)
        try:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        except RuntimeError:
            # CUDA not available — AMP is a no-op on CPU regardless
            self.use_amp = False
            self.scaler  = torch.cuda.amp.GradScaler(enabled=False)

    # ======================================================================= #
    # Phase Management                                                         #
    # ======================================================================= #

    def set_phase(
        self,
        phase:       int,
        lr:          float,
        dann_weight: float = None,
        dann_lambda: float = None,
    ):
        """
        Switch between Phase 1 (backbone only) and Phase 2 (full model).
        Optionally update dann_weight and the GRL lambda at the same time.
        """
        self._phase = phase

        if dann_weight is not None:
            self.dann_weight = dann_weight
        if dann_lambda is not None:
            self.backbone.set_dann_lambda(dann_lambda)

        if phase == 1:
            self._optimizer = torch.optim.Adam(self.backbone.parameters(), lr=lr)
        else:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        print(
            f"[SupConDECTask_Exp09] Phase {phase} | lr={lr:.2e} | "
            f"dann_weight={self.dann_weight:.3f} | "
            f"grl_lambda={self.backbone.grl.lambda_:.4f}"
        )

    # ======================================================================= #
    # Domain Adversarial Loss                                                  #
    # ======================================================================= #

    def _domain_loss(
        self,
        domain_logit: torch.Tensor,
        domain_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        4-class domain adversarial loss via CrossEntropyLoss.

        Samples with domain_label == -1 (unknown source type) are excluded.
        If NO valid domain labels are present in the batch, returns 0.0 so
        training proceeds uninterrupted (safe when using a single source type).

        Args:
            domain_logit : (B, n_domains) — raw logits from the domain head.
            domain_label : (B,) long tensor — 0-3 or -1.

        Returns:
            Scalar loss tensor.
        """
        valid = domain_label >= 0
        if not valid.any():
            return torch.tensor(0.0, device=domain_logit.device)
        return F.cross_entropy(
            domain_logit[valid],
            domain_label[valid].long(),
        )

    # ======================================================================= #
    # Instance-Linked SupCon Loss (Phase 1)                                   #
    # ======================================================================= #

    def _instance_supcon_loss(
        self,
        z1:            torch.Tensor,
        z2:            torch.Tensor,
        global_inst_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supervised Contrastive Loss where POSITIVES are defined by
        Pulse_Instance_ID — i.e., different sensor observations of the
        SAME physical discharge event.

        Unlike Exp08 (which used reported_class for positive matching),
        instance IDs are ALWAYS known (never masked), so every sample
        contributes a positive pair. This achieves Distance-Invariant
        embeddings: the model is forced to learn that an attenuated
        far-sensor signal and a sharp near-sensor signal are equivalent.

        Args:
            z1, z2          : L2-normalised embeddings, each (B, D).
            global_inst_id  : Instance IDs, shape (B,); globally unique.

        Returns:
            Scalar SupCon loss tensor.
        """
        device = z1.device
        N      = z1.shape[0]

        # Concatenate both views → (2N, D)
        z = torch.cat([z1, z2], dim=0)
        y = torch.cat([global_inst_id, global_inst_id], dim=0)  # (2N,)

        # Cosine similarity (z is L2-normalised → dot product = cosine sim)
        sim = torch.mm(z, z.T) / self.temperature   # (2N, 2N)

        # Self-similarity mask
        self_mask = torch.eye(2 * N, device=device, dtype=torch.bool)

        # Positive mask: same Pulse_Instance_ID (cross-sensor positive pairs)
        inst_matrix = y.unsqueeze(0) == y.unsqueeze(1)   # (2N, 2N)
        pos_mask    = inst_matrix & ~self_mask

        # Numerically stable log-softmax
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_c      = sim - sim_max.detach()
        exp_sim    = torch.exp(sim_c) * (~self_mask).float()
        log_prob   = sim_c - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positives
        pos_f   = pos_mask.float()
        num_pos = torch.clamp(pos_f.sum(dim=1), min=1.0)
        loss    = -(pos_f * log_prob).sum(dim=1) / num_pos
        return loss.mean()

    # ======================================================================= #
    # Phase 1 Training Step                                                    #
    # ======================================================================= #

    def training_step_phase1(self, batch: tuple) -> tuple:
        """
        One Phase 1 training step: Instance-SupCon + Domain Adversarial Loss.

        Args:
            batch : (view1, view2, reported_class, global_inst_id,
                     domain_label, shard_path, pulse_idx, time_res, actual_class)

        Returns:
            (z1, loss_dict) where loss_dict = {
                "total":   L_supcon + dann_weight × L_domain,
                "supcon":  L_supcon,
                "domain":  L_domain,
            }
        """
        view1          = batch[0]    # (B, 2, 128, 128)
        view2          = batch[1]    # (B, 2, 128, 128)
        global_inst_id = batch[3]    # (B,) long
        domain_label   = batch[4]    # (B,) long, -1 if unknown

        self._optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            z1, d_logit1 = self.backbone(view1)   # z1:(B,128) d_logit1:(B,n_domains)
            z2, d_logit2 = self.backbone(view2)

            loss_supcon = self._instance_supcon_loss(z1, z2, global_inst_id)

            # Average domain logits from both views; both share the same source domain
            d_logits_avg = (d_logit1 + d_logit2) / 2.0
            loss_domain  = self._domain_loss(d_logits_avg, domain_label)

            loss = loss_supcon + self.dann_weight * loss_domain

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self._optimizer)   # unscale before grad-norm is read externally
        self.scaler.step(self._optimizer)
        self.scaler.update()

        return z1, {
            "total":  loss.item(),
            "supcon": loss_supcon.item(),
            "domain": loss_domain.item(),
        }

    # ======================================================================= #
    # Semi-Supervised Spherical DEC (Phase 2)                                 #
    # ======================================================================= #

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """
        Student-t soft cluster assignment on the unit hypersphere.
        q[i,k] ∝ (1 + ||z_i − μ_k||² / α)^{−(α+1)/2}
        Centroids μ_k are L2-normalised (spherical DEC).
        """
        centroids = F.normalize(self.cluster_layer, p=2, dim=1)   # (K, D)
        diff      = z.unsqueeze(1) - centroids.unsqueeze(0)        # (B, K, D)
        dist_sq   = (diff ** 2).sum(dim=2)                         # (B, K)
        q         = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q         = q / q.sum(dim=1, keepdim=True)
        return q   # (B, K)

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """Sharpen soft assignments: p[i,k] = (q²[i,k]/f_k) / Σ(q²/f)."""
        f = q.sum(dim=0)
        p = q ** 2 / f
        p = p / p.sum(dim=1, keepdim=True)
        return p

    def _pairwise_constraint_loss(
        self,
        q:      torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Must-Link (ML) + Cannot-Link (CL) pairwise penalty on soft assignments.
        Samples with label == -1 are excluded (unlabeled → no constraint).
        """
        valid_mask    = labels != -1
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=q.device)

        q_valid      = q[valid_indices]
        labels_valid = labels[valid_indices]

        sim_matrix    = torch.matmul(q_valid, q_valid.T)
        labels_matrix = labels_valid.unsqueeze(0) == labels_valid.unsqueeze(1)
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

    # ======================================================================= #
    # Phase 2 Training Step                                                    #
    # ======================================================================= #

    def training_step_phase2(self, batch: tuple) -> tuple:
        """
        One Phase 2 step: KL-div + Pairwise Constraints + Domain Adversarial.

        Args:
            batch : (signal, reported_class, domain_label)
                    signal : (B, 2, 128, 128)

        Returns:
            (q, loss_dict) where loss_dict = {
                "total":    L_kl + γ×L_pairwise + λ×L_domain,
                "kl_div":   L_kl,
                "pairwise": L_pairwise,
                "domain":   L_domain,
            }
        """
        signal         = batch[0]   # (B, 2, 128, 128)
        reported_class = batch[1]   # (B,) long
        domain_label   = batch[2]   # (B,) long

        self._optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            z, domain_logit = self.backbone(signal)
            q = self.soft_assign(z)
            p = self.target_distribution(q)

            loss_kl = F.kl_div(q.log(), p.detach(), reduction="batchmean")
            loss_pw = self._pairwise_constraint_loss(q, reported_class)
            loss_domain = self._domain_loss(domain_logit, domain_label)

            loss = loss_kl + self.pairwise_weight_gamma * loss_pw + self.dann_weight * loss_domain
            
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self._optimizer)   # unscale before grad-norm is read externally
        self.scaler.step(self._optimizer)
        self.scaler.update()

        return q, {
            "total":    loss.item(),
            "kl_div":   loss_kl.item(),
            "pairwise": loss_pw.item(),
            "domain":   loss_domain.item(),
        }

    # ======================================================================= #
    # Shared Validation Step                                                   #
    # ======================================================================= #

    def validation_step(self, batch: tuple) -> dict:
        """
        Validation step for both phases.

        Phase 1 batch: (view1, view2, reported_class, global_inst_id, domain_label, ...)
        Phase 2 batch: (signal, reported_class, domain_label)

        NOTE: Domain loss in validation is a DIAGNOSTIC — a higher domain loss
        means the backbone IS domain-agnostic (domain classifier is confused),
        which is what we WANT. It is NOT used for checkpoint selection.
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if self._phase == 1:
                    view1, view2 = batch[0], batch[1]
                    global_inst_id = batch[3]
                    domain_label   = batch[4]
                    z1, d_logit1 = self.backbone(view1)
                    z2, d_logit2 = self.backbone(view2)
                    loss_supcon  = self._instance_supcon_loss(z1, z2, global_inst_id)
                    d_logits_avg = (d_logit1 + d_logit2) / 2.0
                    loss_domain  = self._domain_loss(d_logits_avg, domain_label)
                    result = {
                        "total":  (loss_supcon + self.dann_weight * loss_domain).item(),
                        "supcon": loss_supcon.item(),
                        "domain": loss_domain.item(),
                    }
                else:
                    signal         = batch[0]
                    reported_class = batch[1]
                    domain_label   = batch[2]
                    z, domain_logit = self.backbone(signal)
                    q  = self.soft_assign(z)
                    p  = self.target_distribution(q)
                    loss_kl     = F.kl_div(q.log(), p.detach(), reduction="batchmean")
                    loss_pw     = self._pairwise_constraint_loss(q, reported_class)
                    loss_domain = self._domain_loss(domain_logit, domain_label)
                    result = {
                        "total":    (loss_kl + self.pairwise_weight_gamma * loss_pw).item(),
                        "kl_div":   loss_kl.item(),
                        "pairwise": loss_pw.item(),
                        "domain":   loss_domain.item(),
                    }
        return result

    # ======================================================================= #
    # Cluster Centroid Initialisation                                          #
    # ======================================================================= #

    def init_cluster_centroids(self, all_embeddings: np.ndarray):
        """
        K-Means on collected embeddings + L2-normalise centroids (spherical DEC).
        """
        from sklearn.cluster import KMeans
        print(
            f"[DEC] Running K-Means (K={self.n_clusters}) "
            f"on {len(all_embeddings):,} embeddings..."
        )
        km = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        km.fit(all_embeddings)
        centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        centroids = F.normalize(centroids, p=2, dim=1)
        self.cluster_layer.data = centroids.to(self.cluster_layer.device)
        print("[DEC] Cluster centroids initialised and L2-normalised.")

    # ======================================================================= #
    # Forward & Checkpoint                                                     #
    # ======================================================================= #

    def forward(self, x: torch.Tensor):
        """Returns (z, q) for inference. Domain logit is not used at inference."""
        z, _ = self.backbone(x)
        q    = self.soft_assign(z)
        return z, q

    def save_checkpoint(
        self,
        epoch:         int,
        node_id:       str,
        weights_dir:   str,
        config_dir:    str,
        config_path:   str,
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
            "n_domains":     self.n_domains,
            "embedding_dim": self.embedding_dim,
            "dann_weight":   self.dann_weight,
            "grl_lambda":    self.backbone.grl.lambda_,
        }, weight_path)

        config_snap = os.path.join(config_dir, f"config_{node_id}.yaml")
        shutil.copy2(config_path, config_snap)
        print(f"  [Checkpoint] Saved → {weight_path}")
