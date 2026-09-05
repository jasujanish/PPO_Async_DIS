"""Audit actual behavior versions against learner updates at batch consumption."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def single_policy_version(versions: Iterable[str | int]) -> int:
    normalized = {int(version) for version in versions if str(version)}
    if len(normalized) != 1:
        raise ValueError(
            "a trajectory must use exactly one immutable behavior-policy version; "
            f"got {sorted(normalized)}"
        )
    return next(iter(normalized))


def learner_updates_before(args: Any, rollout_id: int) -> int:
    """The driver performs one actor update per non-critic-only rollout."""
    if rollout_id < 0:
        raise ValueError("rollout_id must be non-negative")
    return max(0, rollout_id - int(getattr(args, "num_critic_only_steps", 0)))


def policy_lag(learner_updates: int, behavior_updates: int) -> int:
    if not 0 <= behavior_updates <= learner_updates:
        raise ValueError("behavior policy is from an invalid or future learner update")
    return learner_updates - behavior_updates


def write_policy_versions(versions: dict[int, int]) -> None:
    """Publish the mapping shared by the driver and rollout process atomically.

    Reset it on each container start: engine versions restart, learner updates
    resume. No generation may start before this write or overlap a transfer.
    """
    path = Path(os.environ["PPO_ASYNC_POLICY_VERSIONS"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(versions, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    if sample.metadata is None:
        sample.metadata = {}
    # Use the count when this batch WILL be consumed, not the wall-clock count
    # while generation overlaps the previous learner step.
    learner_updates = learner_updates_before(args, int(rollout_id))
    try:
        behavior_version = single_policy_version(sample.weight_versions)
        versions = json.loads(Path(os.environ["PPO_ASYNC_POLICY_VERSIONS"]).read_text())
        behavior_updates = int(versions[str(behavior_version)])
        lag = policy_lag(learner_updates, behavior_updates)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        sample.remove_sample = True
        sample.metadata["policy_version_error"] = str(exc)
        return sample

    max_lag = int(os.environ["PPO_ASYNC_MAX_POLICY_LAG"])
    sample.metadata.update(
        behavior_policy_version=behavior_version,
        behavior_learner_updates=behavior_updates,
        learner_updates_at_consumption=learner_updates,
        learner_rollout_id=int(rollout_id),
        policy_lag_updates=lag,
    )
    if lag > max_lag:
        sample.remove_sample = True
        sample.metadata["policy_lag_rejected"] = True
    return sample
