"""Trajectory-level policy-version and lag auditing."""

from __future__ import annotations

import os
from typing import Any, Iterable


def single_policy_version(versions: Iterable[str | int]) -> int:
    normalized = {int(version) for version in versions if str(version)}
    if len(normalized) != 1:
        raise ValueError(
            "a trajectory must use exactly one immutable behavior-policy version; "
            f"got {sorted(normalized)}"
        )
    return next(iter(normalized))


def policy_lag_for_rollout(rollout_id: int, publish_interval: int) -> int:
    if rollout_id < 0 or publish_interval < 1:
        raise ValueError("rollout_id must be non-negative and publish_interval positive")
    return rollout_id % publish_interval


def audit_policy_version(
    args: Any,
    sample: Any,
    *,
    rollout_id: int | None = None,
    evaluation: bool = False,
) -> Any:
    if evaluation:
        return sample
    if rollout_id is None:
        raise RuntimeError("SLIME did not provide a rollout_id to the policy-version hook")
    try:
        behavior_version = single_policy_version(sample.weight_versions)
    except ValueError as exc:
        sample.remove_sample = True
        sample.metadata["policy_version_error"] = str(exc)
        return sample

    interval = int(os.environ["PPO_ASYNC_PUBLISH_INTERVAL"])
    max_lag = int(os.environ["PPO_ASYNC_MAX_POLICY_LAG"])
    lag = policy_lag_for_rollout(int(rollout_id), interval)
    sample.metadata.update(
        behavior_policy_version=behavior_version,
        learner_rollout_id=int(rollout_id),
        policy_lag_updates=lag,
    )
    if lag > max_lag:
        sample.remove_sample = True
        sample.metadata["policy_lag_rejected"] = True
    return sample
