"""
vml.py — top-level entry point for the OvalHumanoidVLA package.

This module re-exports the public API from the ``oracle`` package and
provides a runnable smoke test demonstrating a forward pass, loss
computation, and preprocessing stubs.

Usage (module)
--------------
>>> from vml import OvalHumanoidVLA, VLAConfig, imitation_loss
>>> model = OvalHumanoidVLA(VLAConfig())

Usage (smoke test)
------------------
$ python vml.py
"""

# Re-export the full public API so callers can do ``from vml import ...``
from oracle import (  # noqa: F401
    ImagePreprocessor,
    LanguageEncoder,
    LossComponents,
    OvalHumanoidVLA,
    TextTokenizer,
    VisionEncoder,
    VLAConfig,
    action_huber,
    action_l2,
    imitation_loss,
    joint_limit_penalty,
)


# --------------------------------------------------------------------------
# Smoke test / example usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    cfg = VLAConfig()
    model = OvalHumanoidVLA(cfg)
    model.train()

    B = 2
    # ---- Preprocessing stubs ----
    img_pre = ImagePreprocessor(image_size=cfg.image_size)
    tokenizer = TextTokenizer(vocab_size=cfg.vocab_size, max_length=cfg.max_text_len)

    raw_images = torch.rand(B, 3, 256, 256)           # arbitrary input resolution
    images = img_pre(raw_images)                       # -> (B, 3, 224, 224)

    instructions = ["move left arm forward", "step right foot back"]
    token_ids, mask = tokenizer(instructions)          # (B, L), (B, L)

    proprio = torch.randn(B, cfg.proprio_dim)

    # ---- Forward pass ----
    pred_actions = model(images, token_ids, proprio, mask)
    print(f"Action chunk shape : {tuple(pred_actions.shape)}")
    # expected: (2, 16, 32)

    # ---- Inference (next-step only) ----
    next_action = model.act(images, token_ids, proprio, mask)
    print(f"Next-step action   : {tuple(next_action.shape)}")
    # expected: (2, 32)

    # ---- Imitation-learning loss ----
    target_actions = torch.randn_like(pred_actions) * 0.5   # fake demo targets
    model.train()
    pred_actions = model(images, token_ids, proprio, mask)   # re-run in train mode

    loss_out = imitation_loss(
        pred=pred_actions,
        target=target_actions,
        joint_limits=cfg.joint_limits,
    )
    print(f"Total loss         : {loss_out.total.item():.4f}")
    print(f"  L2               : {loss_out.l2.item():.4f}")
    print(f"  Huber            : {loss_out.huber.item():.4f}")
    print(f"  Joint penalty    : {loss_out.joint_penalty.item():.4f}")

    loss_out.total.backward()
    print("Backward pass      : OK")

    # ---- Model size ----
    print(f"Total params       : {model.n_parameters():,}")

