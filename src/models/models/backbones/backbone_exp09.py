"""
backbone_exp09.py
=================
Custom mini-Vision Transformer (ViT) backbone for Exp09 — Physics-Invariant
Partial Discharge classification via 2-channel Complex Bispectra and a
Domain Adversarial Neural Network (DANN) head with Gradient Reversal Layer.

KEY DIFFERENCES vs backbone_exp08.py
--------------------------------------
1.  in_channels = 2 (Magnitude + Phase channels, not single-channel magnitude)
    The PatchEmbedding Conv2d now maps 2 input channels to d_model, giving
    the ViT full access to both the energy envelope AND the phase-coupling
    fingerprint in every patch token.

2.  Gradient Reversal Layer (GRL)
    A parameter-free autograd function that:
      - Forward pass:  identity (x → x)
      - Backward pass: gradient is multiplied by −lambda_
    Inserted between the [CLS] token and the Domain Head.

3.  Domain Head (4-class linear classifier)
    Attached to the reversed CLS token. Outputs raw logits over n_domains
    classes for CrossEntropyLoss. During training:
      - The Domain Head tries to correctly predict which source domain
        (equation / synthesised / cwru / measured) the sample came from.
      - The GRL reverses gradients flowing back to the backbone, FORCING
        the backbone to produce domain-INDISTINGUISHABLE embeddings.

4.  forward() returns (z, domain_logit) — a 2-tuple — so callers must unpack
    both values. The task and train scripts are updated accordingly.

ARCHITECTURE OVERVIEW
----------------------
              ┌───────────────────────────────────────┐
 Input        │  (B, 2, 128, 128)  ← Magnitude+Phase  │
 Bispectrum   │  2-channel bispectrum (after 129→128) │
              └──────────────────┬────────────────────┘
                                 │
              ┌──────────────────▼────────────────────┐
 Patch        │  Conv2d(2, d_model, 16×16, stride=16) │
 Embedding    │  → (B, 64, d_model)                   │
              └──────────────────┬────────────────────┘
                                 │
              ┌──────────────────▼────────────────────┐
 [CLS]+Pos    │  Prepend CLS token + positional embed │
 Embedding    │  → (B, 65, d_model)                   │
              └──────────────────┬────────────────────┘
                                 │
              ┌──────────────────▼────────────────────┐
 Transformer  │  N=6 Pre-LN TransformerBlock layers   │
 Encoder      │  d_model=384, nhead=6                 │
              └──────────────────┬────────────────────┘
                                 │ [CLS] token → (B, 384)
              ┌──────────────────┴──────────────────┐
              │                                     │
    ┌─────────▼─────────┐              ┌────────────▼────────────┐
    │ Projector Head    │              │ GRL (λ)  ← reversed grad│
    │ Linear→ReLU→Linear│              │ Domain Head (4 logits)  │
    │ → (B, 128)        │              │ → (B, n_domains)        │
    │ L2-normalised     │              └─────────────────────────┘
    └───────────────────┘
        z (embedding)                       domain_logit
        for SupCon + DEC                    for CrossEntropyLoss
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """
    Autograd function implementing the Gradient Reversal Layer (GRL).

    Forward  : identity pass-through (x → x).
    Backward : gradient is scaled by −lambda_ (reversal + scaling).

    Reference: Ganin & Lempitsky, "Unsupervised Domain Adaptation by
    Backpropagation," ICML 2015.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lambda_, dtype=torch.float32))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (lambda_,) = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Thin nn.Module wrapper around GradientReversalFunction.

    Args:
        lambda_ : Initial GRL scaling factor (default 1.0).
                  Call set_lambda() from the train loop to ramp it up.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        """Update the GRL lambda at runtime (for scheduled ramp-up)."""
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def extra_repr(self) -> str:
        return f"lambda_={self.lambda_:.4f}"


# ---------------------------------------------------------------------------
# Mini-ViT building blocks (identical to Exp08 except in_channels)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project each to d_model dims.

    Implemented as a single Conv2d with kernel_size=patch_size, stride=patch_size.
    Now accepts in_channels=2 (magnitude + phase channels) for Exp09.
    """

    def __init__(
        self,
        in_channels: int = 2,
        image_size:  int = 128,
        patch_size:  int = 16,
        d_model:     int = 384,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size={image_size} must be divisible by patch_size={patch_size}."
        )
        self.num_patches = (image_size // patch_size) ** 2   # 64

        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)  with C=2 for Exp09
        Returns:
            tokens : (B, num_patches, d_model)
        """
        x = self.proj(x)        # (B, d_model, H/P, W/P)
        x = x.flatten(2)        # (B, d_model, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, d_model)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer block (identical to Exp08):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        d_model:   int   = 384,
        nhead:     int   = 6,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        ffn_drop:  float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn  = nn.MultiheadAttention(
            d_model, nhead,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm2   = nn.LayerNorm(d_model, eps=1e-6)
        mlp_hidden   = int(d_model * mlp_ratio)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(ffn_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full backbone: DECViT_Exp09
# ---------------------------------------------------------------------------

class DECViT_Exp09(nn.Module):
    """
    Custom mini-ViT backbone for Exp09.

    Extends DECViT_Exp08 with:
      - 2-channel input (magnitude + phase bispectra).
      - Gradient Reversal Layer (GRL) for domain adversarial training.
      - Domain Head: 4-class classifier (equation / synthesised / cwru / measured).

    The forward() method returns a 2-tuple:
        z             : (B, embedding_dim)  — L2-normalised on the unit hypersphere.
                        Used for SupCon pre-training and Spherical DEC clustering.
        domain_logit  : (B, n_domains)      — raw logits for CrossEntropyLoss.
                        GRL reversal occurs in the backward pass only.

    Args:
        in_channels   : Must be 2 (magnitude + phase).
        image_size    : Spatial input size (default 128).
        patch_size    : Patch size (default 16; 128/16=8 patches/axis).
        d_model       : Transformer hidden dimension (default 384, ViT-Small).
        nhead         : Attention heads (default 6; must divide d_model).
        depth         : Transformer encoder blocks (default 6).
        mlp_ratio     : FFN hidden dim multiplier (default 4.0).
        embedding_dim : Final L2-normalised embedding dimension (default 128).
        drop          : Dropout probability for attention and FFN (default 0.0).
        n_domains     : Number of domain classes for the adversarial head (default 4).
        dann_lambda   : Initial GRL lambda (ramped by the train loop, default 0.0).
    """

    def __init__(
        self,
        in_channels:   int   = 2,
        image_size:    int   = 128,
        patch_size:    int   = 16,
        d_model:       int   = 384,
        nhead:         int   = 6,
        depth:         int   = 6,
        mlp_ratio:     float = 4.0,
        embedding_dim: int   = 128,
        drop:          float = 0.0,
        n_domains:     int   = 4,
        dann_lambda:   float = 0.0,
    ):
        super().__init__()

        if in_channels != 2:
            raise ValueError(
                f"DECViT_Exp09 expects 2-channel bispectra (magnitude+phase), "
                f"got in_channels={in_channels}."
            )
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}."
            )

        self.embedding_dim = embedding_dim
        self.d_model       = d_model
        self.n_domains     = n_domains

        num_patches = (image_size // patch_size) ** 2   # 64
        seq_length  = num_patches + 1                   # +1 for [CLS]

        # ------------------------------------------------------------------
        # 1. Patch Embedding — Conv2d(2, d_model, 16×16, stride=16)
        # ------------------------------------------------------------------
        self.patch_embed = PatchEmbedding(
            in_channels = in_channels,
            image_size  = image_size,
            patch_size  = patch_size,
            d_model     = d_model,
        )

        # ------------------------------------------------------------------
        # 2. [CLS] token + Positional Embedding
        # ------------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length, d_model))
        self._init_pos_embed(seq_length, d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ------------------------------------------------------------------
        # 3. Transformer Encoder
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model   = d_model,
                nhead     = nhead,
                mlp_ratio = mlp_ratio,
                attn_drop = drop,
                ffn_drop  = drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # ------------------------------------------------------------------
        # 4. Projector: d_model → d_model → embedding_dim
        #    Produces the L2-normalised embedding for SupCon + DEC.
        # ------------------------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, embedding_dim),
        )

        # ------------------------------------------------------------------
        # 5. Gradient Reversal Layer
        #    Lambda starts at 0 and is ramped by the training loop.
        # ------------------------------------------------------------------
        self.grl = GradientReversalLayer(lambda_=dann_lambda)

        # ------------------------------------------------------------------
        # 6. Domain Head: d_model → 256 → n_domains (raw logits)
        #    Attached to the GRL output (reversed-gradient CLS token).
        #    Trained with CrossEntropyLoss; n_domains=4 for Exp09.
        # ------------------------------------------------------------------
        self.domain_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_domains),
        )

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------

    def _init_pos_embed(self, seq_length: int, d_model: int):
        """Sinusoidal positional embedding initialisation."""
        pe       = torch.zeros(seq_length, d_model)
        pos      = torch.arange(seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term[:d_model // 2])
        self.pos_embed.data.copy_(pe.unsqueeze(0))

    def _init_weights(self):
        """Standard ViT weight initialisations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def set_dann_lambda(self, lambda_: float):
        """Expose GRL lambda update to the training loop for scheduled ramp-up."""
        self.grl.set_lambda(lambda_)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 2, 128, 128) — 2-channel bispectrum (magnitude + phase)

        Returns:
            z            : (B, embedding_dim=128) — L2-normalised on unit hypersphere.
            domain_logit : (B, n_domains=4)       — raw logits for domain classification.
        """
        B = x.shape[0]

        # Step 1: Patch embedding → (B, 64, d_model)
        tokens = self.patch_embed(x)

        # Step 2: Prepend [CLS] token → (B, 65, d_model)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Step 3: Add positional embedding
        tokens = tokens + self.pos_embed

        # Step 4: Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Step 5: Final LayerNorm
        tokens = self.norm(tokens)

        # Step 6: Extract [CLS] token as global representation
        cls_out = tokens[:, 0, :]   # (B, d_model)

        # Step 7: Project to embedding_dim + L2-normalise  →  z
        z = self.projector(cls_out)         # (B, 128)
        z = F.normalize(z, p=2, dim=1)      # unit hypersphere

        # Step 8: Domain head via GRL
        #   Forward:  cls_out passes through unchanged.
        #   Backward: gradient is multiplied by −lambda_ (reversed).
        cls_reversed = self.grl(cls_out)                    # (B, d_model)
        domain_logit = self.domain_head(cls_reversed)       # (B, n_domains)

        return z, domain_logit
