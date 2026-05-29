"""
backbone_exp08.py
=================
Vision Transformer (ViT) backbone for Exp08 — Partial Discharge classification
using 2D Bispectrum representations.

ARCHITECTURE OVERVIEW
---------------------
                ┌──────────────────────────────────────┐
  Input         │  (B, 1, 224, 224)  ← 1-channel       │
  Bispectrum    │  bispectrum image after AdaptivePool  │
                └───────────────┬──────────────────────┘
                                │
                ┌───────────────▼──────────────────────┐
  Patch         │  1-channel → 3-channel               │
  Conversion    │  via Conv2d(1, 3, 1×1)               │
                │  (no pretrained RGB assumption)       │
                └───────────────┬──────────────────────┘
                                │
                ┌───────────────▼──────────────────────┐
  ViT Encoder   │  torchvision ViT-B/16 (pretrained)   │
                │  Patch size: 16×16                    │
                │  Sequence len: (224/16)² + 1 = 197    │
                │  d_model: 768                         │
                │  We keep ALL transformer layers.      │
                │  Classification head is STRIPPED.     │
                └───────────────┬──────────────────────┘
                                │ [CLS] token → shape (B, 768)
                ┌───────────────▼──────────────────────┐
  Projector     │  Linear(768, 768) → ReLU             │
                │  Linear(768, 128)                     │
                └───────────────┬──────────────────────┘
                                │ (B, 128)
                ┌───────────────▼──────────────────────┐
  L2 Norm       │  F.normalize(z, p=2, dim=1)          │
                │  Output lives on the unit hypersphere │
                └───────────────┬──────────────────────┘
                                │
                           (B, 128)  ← embedding_dim

WHY ViT?
--------
The bispectrum is a 2D frequency-frequency map with long-range phase coupling
encoded as spatial patterns. ViT's self-attention mechanism excels at capturing
these non-local relationships across the full (224, 224) grid without the
locality bias of CNNs. The [CLS] token aggregates global context across all
bispectral frequency pairs, which is exactly what we want for PD classification.

WHY L2 NORMALIZE?
-----------------
Both SupCon and Spherical DEC operate on the unit hypersphere (cosine geometry).
L2-normalising the output embedding ensures:
  - SupCon: dot product = cosine similarity → stable, temperature-scaled contrastive loss.
  - DEC: cluster centroids are also L2-normalised → soft assignments use cosine distances.

IN_CHANNELS = 1 (not 3)
------------------------
Bispectra are single-channel magnitude images (grayscale), not RGB.
We handle this by converting 1-ch → 3-ch with a learned 1×1 conv before
feeding the ViT, which allows us to use pretrained ImageNet weights for
the transformer blocks while adapting the first-pixel representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DECViT_Exp08(nn.Module):
    """
    Vision Transformer backbone for Exp08.

    Takes 1-channel 224×224 bispectrum images and outputs L2-normalised
    128-dimensional embeddings for SupCon pre-training and DEC clustering.

    Args:
        in_channels   : Number of input channels (must be 1 for bispectra).
        embedding_dim : Dimension of the final normalised embedding (default 128).
        vit_variant   : Which torchvision ViT to use. Supports:
                          "vit_b_16"  — 86M params, patch 16 (recommended)
                          "vit_b_32"  — 88M params, patch 32 (faster, less detail)
        pretrained    : Whether to load ImageNet-21k pretrained weights (True recommended).
    """

    def __init__(
        self,
        in_channels:   int  = 1,
        embedding_dim: int  = 128,
        vit_variant:   str  = "vit_b_16",
        pretrained:    bool = True,
    ):
        super().__init__()

        if in_channels != 1:
            raise ValueError(
                f"DECViT_Exp08 expects 1-channel bispectra, got in_channels={in_channels}."
            )

        self.embedding_dim = embedding_dim

        # ------------------------------------------------------------------
        # 1. 1-channel → 3-channel adapter
        #    A learned 1×1 convolution to project the single bispectrum channel
        #    into a 3-channel tensor that the pretrained ViT patch embedding
        #    can accept.  This is much lighter-weight than replacing the entire
        #    patch embedding and preserves pretrained representations.
        # ------------------------------------------------------------------
        self.channel_adapter = nn.Conv2d(
            in_channels  = 1,
            out_channels = 3,
            kernel_size  = 1,
            bias         = True,
        )
        nn.init.kaiming_normal_(self.channel_adapter.weight, mode="fan_out")
        nn.init.zeros_(self.channel_adapter.bias)

        # ------------------------------------------------------------------
        # 2. ViT backbone (torchvision)
        #    We load a pretrained ViT and strip the classification head.
        #    The [CLS] token output from the final encoder block serves as
        #    our global scene representation.
        # ------------------------------------------------------------------
        import torchvision.models as tv_models

        weights_map = {
            "vit_b_16": (tv_models.vit_b_16, tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None),
            "vit_b_32": (tv_models.vit_b_32, tv_models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None),
        }
        if vit_variant not in weights_map:
            raise ValueError(
                f"Unknown vit_variant '{vit_variant}'. "
                f"Choose from: {list(weights_map.keys())}"
            )

        vit_constructor, weights = weights_map[vit_variant]

        if pretrained and weights is not None:
            print(f"[DECViT_Exp08] Loading pretrained {vit_variant} weights "
                  f"({weights.__class__.__name__})...")
            vit = vit_constructor(weights=weights)
        else:
            print(f"[DECViT_Exp08] Initialising {vit_variant} from scratch (no pretrained weights).")
            vit = vit_constructor(weights=None)

        # The torchvision ViT's encoder produces hidden states; the 'heads'
        # attribute is the linear classification head — we replace it with
        # an identity to expose the [CLS] token directly.
        d_model = vit.hidden_dim   # 768 for ViT-B variants

        # Strip the classification head — we want the raw [CLS] token (768-D)
        vit.heads = nn.Identity()

        self.vit = vit
        self._d_model = d_model

        # ------------------------------------------------------------------
        # 3. Projector: Linear → ReLU → Linear → embedding_dim
        #    Maps the 768-D [CLS] token to a 128-D contrastive embedding.
        #    Two linear layers with ReLU allow the projector to learn a
        #    non-linear mapping that partially decouples the embedding space
        #    from the ViT's ImageNet representation.
        # ------------------------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (B, 1, 224, 224)  ← 1-channel bispectrum

        Returns:
            z : L2-normalised embedding of shape (B, embedding_dim=128)
                Lives on the unit hypersphere (||z||₂ = 1 for each sample).
        """
        # Step 1: 1-channel → 3-channel adapter
        x = self.channel_adapter(x)          # (B, 3, 224, 224)

        # Step 2: ViT encoder → [CLS] token
        # torchvision ViT._process_input returns patch tokens + [CLS].
        # vit.forward with heads=Identity returns the [CLS] representation.
        cls_token = self.vit(x)              # (B, 768)

        # Step 3: Project to embedding_dim
        z = self.projector(cls_token)        # (B, 128)

        # Step 4: L2-normalise — project onto unit hypersphere
        # Required for both SupCon (cosine similarity) and Spherical DEC.
        z = F.normalize(z, p=2, dim=1)      # (B, 128),  ||z||₂ = 1

        return z
