"""
Cross-modal fusion and autoregressive action decoding.

Design
------
1. Project proprioception to ``fused_dim``.
2. Add per-modality embeddings (vision / language / proprio).
3. Concatenate all tokens and run a joint transformer encoder (backbone).
4. Use a set of learned *action query* tokens — one per future timestep —
   to cross-attend into the fused memory (DETR-style decoder).
5. Apply a **causal mask** over the action queries so that query *t* can
   only attend to queries *0 … t* in the self-attention layers of the
   decoder, enabling autoregressive action prediction at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import VLAConfig


def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask for causal (autoregressive) self-attention.

    Returns a ``(size, size)`` boolean tensor where ``True`` means
    *block this position* (PyTorch ``attn_mask`` convention for
    ``nn.MultiheadAttention``).

    Args:
        size: Sequence length (number of action timesteps).
        device: Target device.

    Returns:
        ``(size, size)`` bool tensor, upper-triangular with diagonal False.
    """
    # torch.triu with diagonal=1 gives the strictly upper triangle
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class CrossModalFusion(nn.Module):
    """Fuse vision, language, and proprio tokens then decode an action chunk.

    The action decoder uses causal masking over its self-attention layers so
    that action query *t* cannot attend to future action queries *t+1 … T-1*.
    This makes the model compatible with autoregressive action generation.

    Args:
        cfg: Model configuration.

    Inputs:
        vision_tokens: ``(B, N_patches, fused_dim)``
        lang_tokens:   ``(B, L, fused_dim)``
        proprio:       ``(B, proprio_dim)``

    Output:
        ``(B, action_chunk_size, fused_dim)``
    """

    def __init__(self, cfg: VLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proprio_proj = nn.Linear(cfg.proprio_dim, cfg.fused_dim)
        # Learned modality-type embeddings: 0=vision, 1=language, 2=proprio
        self.modality_embed = nn.Embedding(3, cfg.fused_dim)

        backbone_layer = nn.TransformerEncoderLayer(
            d_model=cfg.fused_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.fused_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            backbone_layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )

        self.action_queries = nn.Parameter(
            torch.randn(1, cfg.action_chunk_size, cfg.fused_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.fused_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.fused_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.action_decoder_layers
        )

    def forward(
        self,
        vision_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Run cross-modal fusion and decode action chunk with causal masking.

        Args:
            vision_tokens: ``(B, N_patches, fused_dim)``
            lang_tokens:   ``(B, L, fused_dim)``
            proprio:       ``(B, proprio_dim)``

        Returns:
            ``(B, action_chunk_size, fused_dim)``
        """
        B = vision_tokens.shape[0]
        device = vision_tokens.device

        # Project proprioception to a single token
        proprio_tok = self.proprio_proj(proprio).unsqueeze(1)  # (B, 1, D)

        # Add modality-type embeddings (broadcast over sequence length)
        v = vision_tokens + self.modality_embed(
            torch.zeros(1, dtype=torch.long, device=device)
        )
        t = lang_tokens + self.modality_embed(
            torch.ones(1, dtype=torch.long, device=device)
        )
        p = proprio_tok + self.modality_embed(
            torch.full((1,), 2, dtype=torch.long, device=device)
        )

        # Concatenate all context tokens and run joint backbone
        tokens = torch.cat([v, t, p], dim=1)    # (B, N+L+1, D)
        memory = self.backbone(tokens)           # (B, N+L+1, D)

        # Causal mask: action query t cannot see query t+1 … T-1
        T = self.cfg.action_chunk_size
        causal = _causal_mask(T, device)

        queries = self.action_queries.expand(B, -1, -1)
        actions = self.action_decoder(
            tgt=queries, memory=memory, tgt_mask=causal
        )
        return actions  # (B, action_chunk_size, fused_dim)
