"""Baseline-policy and direct-ratio rejection masks for SLIME."""

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
    """Mask the supplied policy-to-rollout ratio; preserve PPO loss weights."""
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


def direct_dis_policy_loss(args, batch, logits, sum_of_sample_mean):
    """Run 0's PPO objective, rejecting divergent tokens before exponentiation.

    Use SLIME's log-probability kernel, clipping function and batch reducer.
    Only token-sized tensors are added; no baseline model forward is needed.
    """
    import math
    import torch
    from slime.backends.megatron_utils.loss import (
        _maybe_capture_log_probs,
        get_log_probs_and_entropy,
        get_rollout_top_p_logprob_kwargs,
    )
    from slime.utils.ppo_utils import compute_policy_loss

    if not args.use_rollout_logprobs or args.use_tis or args.use_kl_loss or args.use_opsm:
        raise ValueError("direct DIS masking requires rollout-ratio PPO without TIS/KL/OPSM")
    epsilon_low = float(os.environ["PPO_ASYNC_DIS_EPSILON_LOW"])
    epsilon_high = float(os.environ["PPO_ASYNC_DIS_EPSILON_HIGH"])
    if not 0 < epsilon_low < 1 or not math.isfinite(epsilon_high) or epsilon_high <= 0:
        raise ValueError("invalid DIS bounds")
    _, outputs = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
        with_entropy=True, **get_rollout_top_p_logprob_kwargs(args, batch),
    )
    _maybe_capture_log_probs(batch, outputs["log_probs"])
    current = torch.cat(outputs["log_probs"])
    behavior = torch.cat(batch["rollout_log_probs"]).detach()
    advantages = torch.cat(batch["advantages"]).detach()
    actions = torch.cat(batch["loss_masks"]).bool()
    if not (current.shape == behavior.shape == advantages.shape == actions.shape):
        raise ValueError("direct DIS inputs are not token aligned")
    if not all(torch.isfinite(x).all() for x in (current, behavior, advantages)):
        raise ValueError("direct DIS inputs must be finite")
    log_ratio = current - behavior
    keep = actions & (log_ratio > math.log1p(-epsilon_low)) & (log_ratio < math.log1p(epsilon_high))
    # Masking AFTER exp can leave inf * 0 in the loss or its backward pass.
    # Rejected positions use a constant zero log-ratio, then a zero loss.
    safe_kl = torch.where(keep, -log_ratio, 0.0)
    token_loss, clipfrac = compute_policy_loss(
        safe_kl, advantages, args.eps_clip, args.eps_clip_high, eps_clip_c=args.eps_clip_c,
    )
    pg_loss = sum_of_sample_mean(token_loss * keep)
    entropy = sum_of_sample_mean(torch.cat(outputs["entropy"]))
    loss = pg_loss - args.entropy_coef * entropy
    metrics = {
        "loss": loss.detach(), "pg_loss": pg_loss.detach(), "entropy_loss": entropy.detach(),
        "pg_clipfrac": sum_of_sample_mean(clipfrac * keep).detach(),
        "ppo_kl": sum_of_sample_mean(-log_ratio).detach(),
        "dis_masked": sum_of_sample_mean((actions & ~keep).float()).detach(),
        "dis_kept": sum_of_sample_mean(keep.float()).detach(),
        "dis_log_ratio_abs": sum_of_sample_mean(log_ratio.abs()).detach(),
    }
    return loss, metrics
