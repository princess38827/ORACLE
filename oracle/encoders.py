"""
Vision and Language encoders for the OvalHumanoidVLA model.

Both encoders project their respective modalities into a shared ``fused_dim``
space so they can be concatenated in the fusion module.

Swap-in points
--------------
* Replace ``VisionEncoder`` with a frozen ViT / DINOv2 / CLIP image tower and
  add a linear projection onto ``fused_dim``.
* Replace ``LanguageEncoder`` with a frozen LLM backbone (e.g. LLaMA / Mistral)
  and project the hidden states onto ``fused_dim``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import VLAConfig


class VisionEncoder(nn.Module):
    """ViT-style patch-embedding encoder.

    Args:
        cfg: Model configuration.

    Input:
        images: ``(B, C, H, W)``

    Output:
        ``(B, N_patches, fused_dim)``
    """

    def __init__(self, cfg: VLAConfig) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(
            cfg.vision_channels,
            cfg.vision_embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.n_patches, cfg.vision_embed_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.vision_embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.vision_embed_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.vision_n_layers, enable_nested_tensor=False
        )
        self.proj = nn.Linear(cfg.vision_embed_dim, cfg.fused_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into patch tokens.

        Args:
            images: ``(B, C, H, W)``

        Returns:
            ``(B, N_patches, fused_dim)``
        """
        x = self.patch_embed(images)        # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)    # (B, N, D)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.proj(x)                 # (B, N, fused_dim)


class LanguageEncoder(nn.Module):
    """Token-embedding + transformer language encoder.

    Args:
        cfg: Model configuration.

    Input:
        token_ids: ``(B, L)``
        attn_mask:  ``(B, L)`` float/bool mask; 1 = keep, 0 = pad (optional).

    Output:
        ``(B, L, fused_dim)``
    """

    def __init__(self, cfg: VLAConfig) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.text_embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.max_text_len, cfg.text_embed_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.text_embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.text_embed_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.language_n_layers, enable_nested_tensor=False
        )
        self.proj = nn.Linear(cfg.text_embed_dim, cfg.fused_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode tokenised instructions.

        Args:
            token_ids: ``(B, L)``
            attn_mask: ``(B, L)`` — 1 for real tokens, 0 for padding.

        Returns:
            ``(B, L, fused_dim)``
        """
        L = token_ids.shape[1]
        x = self.token_embed(token_ids) + self.pos_embed[:, :L]
        # PyTorch src_key_padding_mask convention: True = *ignore*
        pad_mask: torch.Tensor | None = (
            None if attn_mask is None else ~attn_mask.bool()
        )
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.proj(x)                 # (B, L, fused_dim)
