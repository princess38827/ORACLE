"""
Full OvalHumanoidVLA model assembly.

Wires together:
    VisionEncoder -> patch tokens         (B, N, fused_dim)
    LanguageEncoder -> language tokens    (B, L, fused_dim)
    CrossModalFusion -> action queries    (B, T, fused_dim)
    action_head -> joint-angle chunk      (B, T, n_joints)

The output is squashed through tanh and scaled to ``joint_limits`` so
predicted actions are always physically valid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import VLAConfig
from .encoders import LanguageEncoder, VisionEncoder
from .fusion import CrossModalFusion


class OvalHumanoidVLA(nn.Module):
    """Vision-Language-Action model for an oval-form-factor humanoid robot.

    Shape contract
    --------------
    * images:    ``(B, C, H, W)``
    * token_ids: ``(B, L)``
    * proprio:   ``(B, proprio_dim)``
    * output:    ``(B, action_chunk_size, n_joints)``

    Args:
        cfg: Model configuration.  Defaults to :class:`~oracle.config.VLAConfig`.
    """

    def __init__(self, cfg: VLAConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VLAConfig()
        self.vision   = VisionEncoder(self.cfg)
        self.language = LanguageEncoder(self.cfg)
        self.fusion   = CrossModalFusion(self.cfg)
        self.action_head = nn.Sequential(
            nn.Linear(self.cfg.fused_dim, self.cfg.fused_dim),
            nn.GELU(),
            nn.Linear(self.cfg.fused_dim, self.cfg.n_joints),
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a full forward pass and return a predicted action chunk.

        Args:
            images:    ``(B, C, H, W)`` — camera frame(s).
            token_ids: ``(B, L)`` — tokenised instruction.
            proprio:   ``(B, proprio_dim)`` — current joint state.
            text_mask: ``(B, L)`` — 1 for real tokens, 0 for padding (optional).

        Returns:
            ``(B, action_chunk_size, n_joints)`` predicted joint-angle trajectory,
            clipped to ``cfg.joint_limits`` via tanh scaling.
        """
        self._validate_inputs(images, token_ids, proprio)

        v = self.vision(images)
        t = self.language(token_ids, text_mask)
        fused = self.fusion(v, t, proprio)
        raw_actions = self.action_head(fused)

        _, hi = self.cfg.joint_limits
        return torch.tanh(raw_actions) * hi  # (B, T, n_joints)

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only the immediate next-step action (first of the chunk).

        Sets the model to eval mode.  Do not call during training.

        Args:
            images:    ``(B, C, H, W)``
            token_ids: ``(B, L)``
            proprio:   ``(B, proprio_dim)``
            text_mask: ``(B, L)`` optional padding mask.

        Returns:
            ``(B, n_joints)`` next-step joint-angle command.
        """
        self.eval()
        chunk = self.forward(images, token_ids, proprio, text_mask)
        return chunk[:, 0, :]   # (B, n_joints)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _validate_inputs(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
    ) -> None:
        cfg = self.cfg
        if images.ndim != 4:
            raise ValueError(
                f"images must be 4-D (B, C, H, W), got shape {tuple(images.shape)}."
            )
        if images.shape[1] != cfg.vision_channels:
            raise ValueError(
                f"images channel dim must be {cfg.vision_channels}, "
                f"got {images.shape[1]}."
            )
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be 2-D (B, L), got shape {tuple(token_ids.shape)}."
            )
        if token_ids.shape[1] > cfg.max_text_len:
            raise ValueError(
                f"token_ids sequence length {token_ids.shape[1]} exceeds "
                f"max_text_len {cfg.max_text_len}."
            )
        if proprio.ndim != 2:
            raise ValueError(
                f"proprio must be 2-D (B, proprio_dim), got shape {tuple(proprio.shape)}."
            )
        if proprio.shape[1] != cfg.proprio_dim:
            raise ValueError(
                f"proprio last dim must be {cfg.proprio_dim}, got {proprio.shape[1]}."
            )
