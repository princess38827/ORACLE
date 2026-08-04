"""
oracle — Vision-Language-Action package for the oval-form-factor humanoid robot.

Quick-start
-----------
>>> from oracle import OvalHumanoidVLA, VLAConfig
>>> model = OvalHumanoidVLA(VLAConfig())
>>> import torch
>>> out = model(
...     torch.randn(1, 3, 224, 224),
...     torch.randint(0, 32000, (1, 12)),
...     torch.randn(1, 32),
... )
>>> out.shape
torch.Size([1, 16, 32])
"""

from .config import VLAConfig
from .encoders import LanguageEncoder, VisionEncoder
from .fusion import CrossModalFusion
from .losses import LossComponents, action_huber, action_l2, imitation_loss, joint_limit_penalty
from .model import OvalHumanoidVLA
from .preprocessing import ImagePreprocessor, TextTokenizer

__all__ = [
    "VLAConfig",
    "VisionEncoder",
    "LanguageEncoder",
    "CrossModalFusion",
    "OvalHumanoidVLA",
    "imitation_loss",
    "action_l2",
    "action_huber",
    "joint_limit_penalty",
    "LossComponents",
    "ImagePreprocessor",
    "TextTokenizer",
]
