"""
Configuration dataclass for the OvalHumanoidVLA model.

All architecture hyperparameters live here so they can be serialised,
versioned, and swapped without touching model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VLAConfig:
    # ------------------------------------------------------------------
    # Vision encoder
    # ------------------------------------------------------------------
    image_size: int = 224
    patch_size: int = 16
    vision_channels: int = 3
    vision_embed_dim: int = 384
    vision_n_layers: int = 3

    # ------------------------------------------------------------------
    # Language encoder
    # ------------------------------------------------------------------
    vocab_size: int = 32000
    max_text_len: int = 32
    text_embed_dim: int = 384
    language_n_layers: int = 3

    # ------------------------------------------------------------------
    # Fusion / joint backbone
    # ------------------------------------------------------------------
    fused_dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # ------------------------------------------------------------------
    # Action space — tuned for a humanoid, e.g. O1-class rig:
    # 2×6 legs, 2×7 arms, 3 torso, 2×6 hands (coarse) = 32 DOF example
    # ------------------------------------------------------------------
    n_joints: int = 32
    action_chunk_size: int = 16         # predict this many future timesteps
    joint_limits: tuple = (-3.14, 3.14) # radians, symmetric default
    action_decoder_layers: int = 2

    # ------------------------------------------------------------------
    # Proprioception (joint state fed back in)
    # ------------------------------------------------------------------
    proprio_dim: int = 32

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})."
            )
        if self.fused_dim % self.n_heads != 0:
            raise ValueError(
                f"fused_dim ({self.fused_dim}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        if self.vision_embed_dim % self.n_heads != 0:
            raise ValueError(
                f"vision_embed_dim ({self.vision_embed_dim}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        if self.text_embed_dim % self.n_heads != 0:
            raise ValueError(
                f"text_embed_dim ({self.text_embed_dim}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        lo, hi = self.joint_limits
        if lo >= hi:
            raise ValueError(
                f"joint_limits lo ({lo}) must be strictly less than hi ({hi})."
            )
        if self.n_joints <= 0:
            raise ValueError(f"n_joints must be positive, got {self.n_joints}.")
        if self.action_chunk_size <= 0:
            raise ValueError(
                f"action_chunk_size must be positive, got {self.action_chunk_size}."
            )
        if self.proprio_dim <= 0:
            raise ValueError(f"proprio_dim must be positive, got {self.proprio_dim}.")

    @property
    def n_patches(self) -> int:
        """Number of vision patches per image."""
        return (self.image_size // self.patch_size) ** 2
