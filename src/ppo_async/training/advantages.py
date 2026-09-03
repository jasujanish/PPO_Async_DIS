"""Action-token-only GAE hook for SLIME PPO."""

from __future__ import annotations

from typing import Any

from ppo_async.training.numerics import action_only_gae


def compute_action_only_gae(args: Any, rollout_data: dict[str, Any]) -> None:
    values = rollout_data.get("values")
    masks = rollout_data.get("loss_masks")
    rewards = rollout_data.get("rewards")
    if values is None or masks is None or rewards is None:
        raise ValueError("PPO rollout data requires values, loss_masks, and rewards")
    if not (len(values) == len(masks) == len(rewards)):
        raise ValueError("PPO rollout fields have different sample counts")

    advantages = []
    returns = []
    for value, mask, reward in zip(values, masks, rewards, strict=True):
        advantage, result_return = action_only_gae(
            value,
            float(reward),
            mask,
            gamma=float(args.gamma),
            gae_lambda=float(args.lambd),
        )
        advantages.append(advantage)
        returns.append(result_return)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
