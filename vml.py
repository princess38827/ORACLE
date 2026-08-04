"""
Vision-Language-Action (VLA) model for an oval-form-factor humanoid robot.

Architecture (RT-2 / OpenVLA-style, simplified):
    Vision Encoder  -> patch tokens
    Language Encoder -> instruction tokens
    Cross-Modal Transformer -> fused representation
    Action Head -> chunk of future joint commands (position/torque)

Designed to be backbone-agnostic: swap VisionEncoder for a real ViT/DINOv2/
CLIP encoder and LanguageEncoder for a real LLM/tokenizer later. The scaffold,
shapes, and action-chunking logic are the part worth keeping stable.
"""

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class VLAConfig:
    # Vision
    image_size: int = 224
    patch_size: int = 16
    vision_channels: int = 3
    vision_embed_dim: int = 384

    # Language
    vocab_size: int = 32000
    max_text_len: int = 32
    text_embed_dim: int = 384

    # Fusion / backbone
    fused_dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # Action space — tuned for a humanoid, e.g. O1-class rig:
    # 2x6 legs, 2x7 arms, 3 torso, 2x6 hands (coarse) = 32 DOF example
    n_joints: int = 32
    action_chunk_size: int = 16          # predict this many future timesteps
    joint_limits: tuple = (-3.14, 3.14)  # radians, symmetric default

    # Proprioception (joint state fed back in)
    proprio_dim: int = 32


# --------------------------------------------------------------------------
# Vision encoder (ViT-style patch embedding + transformer)
# --------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            cfg.vision_channels, cfg.vision_embed_dim,
            kernel_size=cfg.patch_size, stride=cfg.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, cfg.vision_embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.vision_embed_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.vision_embed_dim * 4,
            dropout=cfg.dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.proj = nn.Linear(cfg.vision_embed_dim, cfg.fused_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W)
        x = self.patch_embed(images)                # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)             # (B, N, D)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.proj(x)                          # (B, N, fused_dim)


# --------------------------------------------------------------------------
# Language encoder (token embedding + transformer)
# --------------------------------------------------------------------------

class LanguageEncoder(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.text_embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_text_len, cfg.text_embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.text_embed_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.text_embed_dim * 4,
            dropout=cfg.dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.proj = nn.Linear(cfg.text_embed_dim, cfg.fused_dim)

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # token_ids: (B, L)
        L = token_ids.shape[1]
        x = self.token_embed(token_ids) + self.pos_embed[:, :L]
        pad_mask = None if attn_mask is None else ~attn_mask.bool()
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.proj(x)  # (B, L, fused_dim)


# --------------------------------------------------------------------------
# Cross-modal fusion + action decoding
# --------------------------------------------------------------------------

class CrossModalFusion(nn.Module):
    """Concatenate vision + language + proprio tokens, run a joint transformer,
    then use a set of learned 'action query' tokens (like DETR) to read out
    a chunk of future actions via cross-attention."""

    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.proprio_proj = nn.Linear(cfg.proprio_dim, cfg.fused_dim)
        self.modality_embed = nn.Embedding(3, cfg.fused_dim)  # 0=vision,1=lang,2=proprio

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.fused_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.fused_dim * 4,
            dropout=cfg.dropout, batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        self.action_queries = nn.Parameter(
            torch.randn(1, cfg.action_chunk_size, cfg.fused_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.fused_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.fused_dim * 4,
            dropout=cfg.dropout, batch_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def forward(self, vision_tokens, lang_tokens, proprio):
        B = vision_tokens.shape[0]
        proprio_tok = self.proprio_proj(proprio).unsqueeze(1)  # (B, 1, D)

        v = vision_tokens + self.modality_embed(torch.zeros(1, dtype=torch.long, device=vision_tokens.device))
        t = lang_tokens + self.modality_embed(torch.ones(1, dtype=torch.long, device=lang_tokens.device))
        p = proprio_tok + self.modality_embed(torch.full((1,), 2, dtype=torch.long, device=proprio.device))

        tokens = torch.cat([v, t, p], dim=1)
        fused = self.backbone(tokens)

        queries = self.action_queries.expand(B, -1, -1)
        actions = self.action_decoder(tgt=queries, memory=fused)
        return actions  # (B, action_chunk_size, fused_dim)


# --------------------------------------------------------------------------
# Full VLA model
# --------------------------------------------------------------------------

class OvalHumanoidVLA(nn.Module):
    def __init__(self, cfg: VLAConfig = None):
        super().__init__()
        self.cfg = cfg or VLAConfig()
        self.vision = VisionEncoder(self.cfg)
        self.language = LanguageEncoder(self.cfg)
        self.fusion = CrossModalFusion(self.cfg)
        self.action_head = nn.Sequential(
            nn.Linear(self.cfg.fused_dim, self.cfg.fused_dim),
            nn.GELU(),
            nn.Linear(self.cfg.fused_dim, self.cfg.n_joints),
        )

    def forward(self, images, token_ids, proprio, text_mask=None):
        """
        images:    (B, C, H, W)          camera frame
        token_ids: (B, L)                tokenized instruction
        proprio:   (B, proprio_dim)      current joint state
        returns:   (B, chunk, n_joints)  predicted joint-angle trajectory
        """
        v = self.vision(images)
        t = self.language(token_ids, text_mask)
        fused = self.fusion(v, t, proprio)
        raw_actions = self.action_head(fused)
        lo, hi = self.cfg.joint_limits
        return torch.tanh(raw_actions) * hi  # squash into joint limits (symmetric)

    @torch.no_grad()
    def act(self, images, token_ids, proprio, text_mask=None):
        """Inference helper: returns only the next-step action (first of chunk)."""
        self.eval()
        chunk = self.forward(images, token_ids, proprio, text_mask)
        return chunk[:, 0, :]  # (B, n_joints)


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = VLAConfig()
    model = OvalHumanoidVLA(cfg)

    B = 2
    images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    tokens = torch.randint(0, cfg.vocab_size, (B, 12))
    mask = torch.ones(B, 12)
    proprio = torch.randn(B, cfg.proprio_dim)

    out = model(images, tokens, proprio, mask)
    print("Action chunk shape:", out.shape)  # (B, action_chunk_size, n_joints)

    next_action = model.act(images, tokens, proprio, mask)
    print("Next-step action shape:", next_action.shape)  # (B, n_joints)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
