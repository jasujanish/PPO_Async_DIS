"""Small numerical contracts that are testable without SLIME or a GPU."""

from __future__ import annotations

import torch


def validate_aligned(*tensors: torch.Tensor) -> None:
    if not tensors or any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("trajectory tensors must be one-dimensional")
    if len({tensor.numel() for tensor in tensors}) != 1:
        raise ValueError("trajectory tensors must have the same length")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("trajectory tensors must contain only finite values")


def action_only_gae(
    values: torch.Tensor,
    terminal_reward: float,
    action_mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_aligned(values, action_mask)
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and lambda must be in [0, 1]")
    mask = action_mask.to(dtype=torch.bool)
    if not bool(mask.any()):
        zeros = torch.zeros_like(values, dtype=torch.float32)
        return zeros, zeros.clone()

    action_values = values[mask].float()
    rewards = torch.zeros_like(action_values)
    rewards[-1] = float(terminal_reward)
    advantages = torch.zeros_like(action_values)
    next_value = torch.zeros((), dtype=action_values.dtype, device=action_values.device)
    next_advantage = torch.zeros_like(next_value)
    for index in range(action_values.numel() - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - action_values[index]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = next_advantage
        next_value = action_values[index]

    returns = advantages + action_values
    scattered_advantages = torch.zeros_like(values, dtype=torch.float32)
    scattered_returns = torch.zeros_like(values, dtype=torch.float32)
    scattered_advantages[mask] = advantages
    scattered_returns[mask] = returns
    return scattered_advantages, scattered_returns


def direct_double_sided_mask(
    current_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    epsilon_low: float,
    epsilon_high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_aligned(current_log_probs, rollout_log_probs, action_mask)
    if not 0.0 < epsilon_low < 1.0 or epsilon_high <= 0.0:
        raise ValueError("invalid DIS bounds")
    ratio = torch.exp(current_log_probs - rollout_log_probs)
    keep = (
        (ratio > 1.0 - epsilon_low)
        & (ratio < 1.0 + epsilon_high)
        & action_mask.to(dtype=torch.bool)
    )
    return ratio, keep
