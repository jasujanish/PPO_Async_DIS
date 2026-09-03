"""Direct double-sided importance sampling hooks for SLIME."""

from __future__ import annotations

import os
from typing import Any


def _ratios(train_log_probs, rollout_log_probs, loss_masks):
    import torch

    current = torch.cat(train_log_probs, dim=0)
    behavior = torch.cat(rollout_log_probs, dim=0)
    mask = torch.cat(loss_masks, dim=0).to(dtype=torch.bool)
    if current.shape != behavior.shape or current.shape != mask.shape:
        raise RuntimeError("DIS log-probabilities and action masks are not token aligned")
    if not torch.isfinite(current).all() or not torch.isfinite(behavior).all():
        raise RuntimeError("DIS received non-finite log-probabilities")
    return torch.exp(current - behavior), mask


def observe_importance(
    args: Any,
    *,
    pg_loss,
    train_log_probs,
    rollout_log_probs,
    loss_masks,
    **_: Any,
):
    """Report behavior/current ratios without changing standard PPO."""
    import torch

    ratio, action_mask = _ratios(train_log_probs, rollout_log_probs, loss_masks)
    metrics = {
        "behavior_ratio": ratio.detach(),
        "behavior_ratio_abs_delta": (ratio - 1.0).abs().detach(),
        "behavior_action_token": action_mask.to(dtype=torch.float32).detach(),
    }
    return pg_loss, loss_masks, metrics


def apply_dis_mask(
    args: Any,
    *,
    pg_loss,
    train_log_probs,
    rollout_log_probs,
    loss_masks,
    **_: Any,
):
    """Zero PPO actor gradients outside the open DIS trust region."""
    import torch

    ratio, action_mask = _ratios(train_log_probs, rollout_log_probs, loss_masks)
    epsilon_low = float(os.environ["PPO_ASYNC_DIS_EPSILON_LOW"])
    epsilon_high = float(os.environ["PPO_ASYNC_DIS_EPSILON_HIGH"])
    if not 0.0 < epsilon_low < 1.0 or epsilon_high <= 0.0:
        raise RuntimeError("invalid DIS trust-region bounds")
    keep = (
        (ratio > 1.0 - epsilon_low)
        & (ratio < 1.0 + epsilon_high)
        & action_mask
    )
    if pg_loss.shape != keep.shape:
        raise RuntimeError("DIS mask and PPO loss are not token aligned")
    kept_loss = pg_loss * keep.to(dtype=pg_loss.dtype)
    metrics = {
        "dis_ratio": ratio.detach(),
        "dis_masked": (action_mask & ~keep).to(dtype=torch.float32).detach(),
        "dis_kept": keep.to(dtype=torch.float32).detach(),
        "dis_ratio_abs_delta": (ratio - 1.0).abs().detach(),
    }
    # Preserve the original denominator: excluded action tokens contribute a
    # zero numerator, which makes the masked-token fraction fully accountable.
    return kept_loss, loss_masks, metrics
