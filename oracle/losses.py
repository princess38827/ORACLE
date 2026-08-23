"""
Imitation-learning losses for the OvalHumanoidVLA model.

Losses
------
``action_l2``
    Mean squared error between predicted and target action chunks.

``action_huber``
    Huber (smooth-L1) loss — less sensitive to outliers than MSE, good for
    large joint-angle residuals.

``joint_limit_penalty``
    Soft penalty that pushes predicted actions away from joint limits.
    Useful as an auxiliary loss to discourage unsafe configurations.

``imitation_loss``
    Weighted combination of the above, returning both the scalar total loss
    and a dict of individual components for logging.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class LossComponents(NamedTuple):
    """Container returned by :func:`imitation_loss`."""

    total: torch.Tensor
    """Weighted sum of all active loss terms."""

    l2: torch.Tensor
    """Per-element MSE between predicted and target actions."""

    huber: torch.Tensor
    """Per-element Huber loss between predicted and target actions."""

    joint_penalty: torch.Tensor
    """Soft joint-limit penalty on predicted actions."""


def action_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over the full action chunk.

    Args:
        pred:    ``(B, T, n_joints)`` predicted joint angles.
        target:  ``(B, T, n_joints)`` ground-truth joint angles.
        weights: Optional ``(B, T)`` per-timestep importance weights.

    Returns:
        Scalar loss tensor.
    """
    loss = F.mse_loss(pred, target, reduction="none")  # (B, T, n_joints)
    loss = loss.mean(dim=-1)                            # (B, T)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def action_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Huber (smooth-L1) loss over the full action chunk.

    Args:
        pred:    ``(B, T, n_joints)`` predicted joint angles.
        target:  ``(B, T, n_joints)`` ground-truth joint angles.
        delta:   Huber threshold.
        weights: Optional ``(B, T)`` per-timestep importance weights.

    Returns:
        Scalar loss tensor.
    """
    loss = F.huber_loss(pred, target, delta=delta, reduction="none")  # (B, T, J)
    loss = loss.mean(dim=-1)                                           # (B, T)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def joint_limit_penalty(
    pred: torch.Tensor,
    joint_limits: tuple[float, float] = (-3.14, 3.14),
    margin: float = 0.1,
) -> torch.Tensor:
    """Soft penalty encouraging actions to stay within joint limits.

    Uses a ReLU barrier: zero inside ``[lo+margin, hi-margin]``, growing
    linearly outside.  This acts as a "soft wall" rather than a hard clamp.

    Args:
        pred:         ``(B, T, n_joints)`` predicted joint angles.
        joint_limits: ``(lo, hi)`` hard joint-angle limits in radians.
        margin:       Safety margin inside the hard limits.

    Returns:
        Scalar penalty tensor.
    """
    lo, hi = joint_limits
    lower_violation = F.relu((lo + margin) - pred)
    upper_violation = F.relu(pred - (hi - margin))
    return (lower_violation + upper_violation).mean()


def imitation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    joint_limits: tuple[float, float] = (-3.14, 3.14),
    l2_weight: float = 0.5,
    huber_weight: float = 0.5,
    joint_penalty_weight: float = 0.01,
    huber_delta: float = 1.0,
    weights: torch.Tensor | None = None,
) -> LossComponents:
    """Weighted imitation-learning loss combining L2, Huber, and joint penalty.

    Args:
        pred:                ``(B, T, n_joints)`` predicted joint angles.
        target:              ``(B, T, n_joints)`` ground-truth joint angles.
        joint_limits:        ``(lo, hi)`` hard joint-angle limits.
        l2_weight:           Weight for the L2 loss term.
        huber_weight:        Weight for the Huber loss term.
        joint_penalty_weight: Weight for the joint-limit penalty term.
        huber_delta:         Huber threshold.
        weights:             Optional ``(B, T)`` per-timestep importance weights.

    Returns:
        :class:`LossComponents` named-tuple with ``total``, ``l2``,
        ``huber``, and ``joint_penalty`` fields.
    """
    l2 = action_l2(pred, target, weights=weights)
    huber = action_huber(pred, target, delta=huber_delta, weights=weights)
    jpenalty = joint_limit_penalty(pred, joint_limits=joint_limits)

    total = (
        l2_weight * l2
        + huber_weight * huber
        + joint_penalty_weight * jpenalty
    )
    return LossComponents(total=total, l2=l2, huber=huber, joint_penalty=jpenalty)
