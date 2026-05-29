"""
backbone_exp08.py
=================
Custom mini-Vision Transformer (ViT) backbone for Exp08 — Partial Discharge
classification using Welch-method 2D Bispectrum representations.

ARCHITECTURE OVERVIEW
---------------------
                ┌──────────────────────────────────────┐
  Input         │  (B, 1, 128, 128)  ← 1-channel       │
  Bispectrum    │  bispectrum image after 129→128 crop  │
                └───────────────┬──────────────────────┘
                                │
                ┌───────────────▼──────────────────────┐
  Patch         │  Conv2d(1, d_model, 16×16, stride=16) │
  Embedding     │  Flattens image → 64 patch tokens     │
                │  (8×8 grid of 16×16 patches)          │
                │  output: (B, 64, d_model)             │
                └───────────────┬──────────────────────┘
                                │
                ┌───────────────▼──────────────────────┐
  [CLS] token   │  Prepended learnable token            │
  + Pos Embed   │  + learnable 1D position embeddings  │
                │  sequence length: 64 + 1 = 65         │
                └───────────────┬──────────────────────┘
                                │
                ┌───────────────▼──────────────────────┐
  Transformer   │  N × TransformerEncoderLayer          │
  Encoder       │  (Multi-head self-attention + FFN)    │
                │  d_model=384, nhead=6, N=6 layers     │
                └───────────────┬──────────────────────┘
                                │ [CLS] token → (B, 384)
                ┌───────────────▼──────────────────────┐
  Projector     │  Linear(384, 384) → ReLU             │
                │  Linear(384, 128)                     │
                └───────────────┬──────────────────────┘
                                │ (B, 128)
                ┌───────────────▼──────────────────────┐
  L2 Norm       │  F.normalize(z, p=2, dim=1)          │
                │  Output lives on the unit hypersphere │
                └───────────────┬──────────────────────┘
                                │
                           (B, 128)  ← embedding_dim

WHY A CUSTOM MINI-ViT (not torchvision vit_b_16)?
--------------------------------------------------
torchvision's vit_b_16 HARDCODES image_size=224 inside its positional
embedding and _process_input method.  Feeding a 128×128 image raises
a runtime shape error because the sequence length (64 tokens for 128/16=8
patches per axis) does not match the pretrained positional embedding that
expects 196 tokens (14 patches per axis at 224/16).

Instead we build a custom mini-ViT that is configured from scratch for:
    image_size  = 128
    patch_size  = 16
    num_patches = (128/16)² = 64        ← 8×8 grid
    seq_length  = 64 + 1 (CLS) = 65

This avoids the image_size mismatch entirely and produces a model that is:
  - ~3× smaller than ViT-B/16 (~27 M params vs ~86 M)
  - Correct for 128×128 inputs
  - Still parameter-free pretrained initialisation → trained from scratch on bispectra

WHY d_model=384, nhead=6, N=6?
--------------------------------
The "ViT-Small" configuration from the original DeiT paper:
  d_model=384, nhead=6, mlp_ratio=4, depth=6
scales well to our 64-token sequences and has proven effective for compact
image domains.  It hits a good balance between expressiveness and GPU memory
during SupCon training with batch_size=32 on a single GPU.

WHY L2 NORMALIZE?
-----------------
Both SupCon and Spherical DEC operate on the unit hypersphere (cosine geometry).
L2-normalising the output embedding ensures:
  - SupCon: dot product = cosine similarity → stable, temperature-scaled loss.
  - DEC: cluster centroids are also L2-normalised → soft assignments use
         cosine distances.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mini-ViT building blocks
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project each to d_model dims.

    Implemented as a single Conv2d with kernel_size=patch_size and
    stride=patch_size, which is mathematically identical to splitting then
    applying a linear projection.

    Args:
        in_channels : input image channels (1 for grayscale bispectra).
        image_size  : spatial size of the input image (must be square).
        patch_size  : size of each square patch (default 16).
        d_model     : embedding dimension for each patch token.
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size:  int = 128,
        patch_size:  int = 16,
        d_model:     int = 384,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size={image_size} must be divisible by patch_size={patch_size}."
        )
        self.num_patches = (image_size // patch_size) ** 2   # 64 for 128/16=8

        # One conv = one learned linear projection per patch
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            tokens : (B, num_patches, d_model)
        """
        x = self.proj(x)        # (B, d_model, H/P, W/P)
        x = x.flatten(2)        # (B, d_model, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, d_model)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer block:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Pre-LN (normalise before attention) is more stable for training from
    scratch than the original Post-LN variant.

    Args:
        d_model    : token embedding dimension.
        nhead      : number of self-attention heads (must divide d_model evenly).
        mlp_ratio  : ratio of FFN hidden dim to d_model (default 4 → dim 4×d_model).
        attn_drop  : dropout inside scaled dot-product attention.
        ffn_drop   : dropout inside the FFN.
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
            batch_first=True,   # input: (B, T, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        mlp_hidden = int(d_model * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(ffn_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with Pre-LN and residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out

        # FFN with Pre-LN and residual
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------

class DECViT_Exp08(nn.Module):
    """
    Custom mini-ViT backbone for Exp08.

    Configured for 1-channel 128×128 bispectrum images.  Produces L2-normalised
    128-dimensional embeddings for SupCon pre-training and DEC clustering.

    Architecture: ViT-Small (d=384, nhead=6, depth=6), trained from scratch.

    Args:
        in_channels   : Input channels; must be 1 (grayscale bispectrum).
        image_size    : Spatial size of the input grid (default 128).
        patch_size    : Patch size (default 16); image_size must be divisible.
        d_model       : Internal transformer embedding dimension (default 384).
        nhead         : Number of attention heads (default 6, must divide d_model).
        depth         : Number of transformer encoder blocks (default 6).
        mlp_ratio     : FFN hidden dim multiplier (default 4.0).
        embedding_dim : Final L2-normalised embedding dimension (default 128).
        drop          : Dropout probability for attention and FFN (default 0.0).
    """

    def __init__(
        self,
        in_channels:   int   = 1,
        image_size:    int   = 128,
        patch_size:    int   = 16,
        d_model:       int   = 384,
        nhead:         int   = 6,
        depth:         int   = 6,
        mlp_ratio:     float = 4.0,
        embedding_dim: int   = 128,
        drop:          float = 0.0,
    ):
        super().__init__()

        if in_channels != 1:
            raise ValueError(
                f"DECViT_Exp08 expects 1-channel bispectra, got in_channels={in_channels}."
            )
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}."
            )

        self.embedding_dim = embedding_dim
        self.d_model       = d_model

        num_patches = (image_size // patch_size) ** 2   # 64 for 128/16=8
        seq_length  = num_patches + 1                   # +1 for [CLS] token

        # ------------------------------------------------------------------
        # 1. Patch Embedding
        #    Conv2d(1, d_model, 16×16, stride=16) → (B, 64, d_model)
        # ------------------------------------------------------------------
        self.patch_embed = PatchEmbedding(
            in_channels = in_channels,
            image_size  = image_size,
            patch_size  = patch_size,
            d_model     = d_model,
        )

        # ------------------------------------------------------------------
        # 2. [CLS] token + Positional Embedding
        #    [CLS]: a learnable token prepended to the patch sequence.
        #    Pos embed: one learnable vector per position (65 total).
        # ------------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length, d_model))

        # Initialise positional embedding with sinusoidal values for stability
        self._init_pos_embed(seq_length, d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ------------------------------------------------------------------
        # 3. Transformer Encoder
        #    N=6 Pre-LN TransformerBlock layers.
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
        #    Non-linear projector partially decouples the embedding space
        #    from the raw transformer representations, which is beneficial
        #    for contrastive learning (SimCLR / SupCon finding).
        # ------------------------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, embedding_dim),
        )

        # Weight initialisation
        self._init_weights()

    def _init_pos_embed(self, seq_length: int, d_model: int):
        """
        Sinusoidal positional embedding initialisation.
        Provides a good starting point that encodes spatial distance before
        the model has seen any data.
        """
        pe = torch.zeros(seq_length, d_model)
        pos = torch.arange(seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term[:d_model // 2])
        self.pos_embed.data.copy_(pe.unsqueeze(0))

    def _init_weights(self):
        """Apply standard ViT weight initialisations."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (B, 1, 128, 128)  ← 1-channel bispectrum

        Returns:
            z : L2-normalised embedding of shape (B, embedding_dim=128)
                Lives on the unit hypersphere (||z||₂ = 1 for each sample).
        """
        B = x.shape[0]

        # Step 1: Patch embedding → (B, 64, d_model)
        tokens = self.patch_embed(x)

        # Step 2: Prepend [CLS] token → (B, 65, d_model)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 65, d_model)

        # Step 3: Add positional embedding
        tokens = tokens + self.pos_embed          # (B, 65, d_model)

        # Step 4: Pass through N transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Step 5: Final LayerNorm
        tokens = self.norm(tokens)                # (B, 65, d_model)

        # Step 6: Extract [CLS] token (index 0) as the global representation
        cls_out = tokens[:, 0, :]                 # (B, d_model)

        # Step 7: Project to embedding_dim
        z = self.projector(cls_out)               # (B, 128)

        # Step 8: L2-normalise — project onto unit hypersphere
        # Required for both SupCon (cosine similarity) and Spherical DEC.
        z = F.normalize(z, p=2, dim=1)           # (B, 128),  ||z||₂ = 1

        return z
