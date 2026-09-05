"""Structured, example-counted production tracking hooks for SLIME."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any


_LOCK = threading.Lock()
_STATE_BY_PATH: dict[str, dict[str, Any]] = {}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _state_path() -> Path:
    return Path(os.environ["PPO_ASYNC_TRACKING_STATE"])


def _metrics_path() -> Path:
    return Path(os.environ["PPO_ASYNC_METRICS_LOG"])


def _load_state(path: Path) -> dict[str, Any]:
    key = str(path)
    cached = _STATE_BY_PATH.get(key)
    if cached is not None:
        return cached
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {"next_example": 1, "pending": []}
    _STATE_BY_PATH[key] = state
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sample_scalar(args: Any, sample: Any) -> dict[str, Any]:
    metadata = sample.metadata or {}
    verification = metadata.get("verification") or {}
    try:
        reward = float(sample.get_reward_value(args))
    except (TypeError, ValueError):
        reward = float(sample.reward or 0.0)
    return {
        "reward": reward,
        "response_length": int(getattr(sample, "effective_response_length", sample.response_length)),
        "truncated": int(str(getattr(sample, "status", "")).endswith("TRUNCATED")),
        "removed": int(bool(getattr(sample, "remove_sample", False))),
        "source": str(metadata.get("source_name", "unknown")),
        "verification_status": str(verification.get("status", "unknown")),
        "policy_lag_updates": int(metadata.get("policy_lag_updates", 0)),
        "policy_lag_is_upper_bound": bool(metadata.get("policy_lag_is_upper_bound", False)),
    }


def _mean(items: list[dict[str, Any]], key: str) -> float:
    return sum(float(item[key]) for item in items) / len(items)


def _summarize_window(items: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    sources = sorted({item["source"] for item in items})
    verification_statuses = sorted({item["verification_status"] for item in items})
    return {
        "event": "scalar_metrics",
        "recorded_at": _timestamp(),
        "examples_start": start,
        "examples_end": end,
        "window_examples": len(items),
        "reward_mean": _mean(items, "reward"),
        "verified_rate": sum(item["verification_status"] == "verified" for item in items) / len(items),
        "response_length_mean": _mean(items, "response_length"),
        "truncated_rate": _mean(items, "truncated"),
        "removed_rate": _mean(items, "removed"),
        "policy_lag_updates_mean": _mean(items, "policy_lag_updates"),
        "policy_lag_upper_bound_fraction": _mean(items, "policy_lag_is_upper_bound"),
        "source_counts": {source: sum(item["source"] == source for item in items) for source in sources},
        "verification_counts": {
            status: sum(item["verification_status"] == status for item in items)
            for status in verification_statuses
        },
    }


def log_rollout_scalars(
    rollout_id: int,
    args: Any,
    samples: list[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Persist exact ten-example scalar windows and replace per-rollout logging."""
    del rollout_id, rollout_extra_metrics, rollout_time
    interval = int(os.environ["PPO_ASYNC_SCALAR_LOG_EVERY"])
    state_path = _state_path()
    with _LOCK:
        state = _load_state(state_path)
        state["pending"].extend(_sample_scalar(args, sample) for sample in samples)
        while len(state["pending"]) >= interval:
            window = state["pending"][:interval]
            del state["pending"][:interval]
            start = int(state["next_example"])
            end = start + interval - 1
            _append(_metrics_path(), _summarize_window(window, start, end))
            state["next_example"] = end + 1
        _save_state(state_path, state)
    return True


def log_eval_scalars(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Persist single-sample pass@1 for selection or final evaluation."""
    del extra_metrics
    batch_size = int(os.environ["PPO_ASYNC_BATCH_SIZE"])
    datasets: dict[str, Any] = {}
    for name, values in sorted(data.items()):
        rewards = [float(value) for value in values["rewards"]]
        datasets[name] = {
            "examples": len(rewards),
            "pass_at_1": sum(rewards) / len(rewards),
        }
    _append(
        Path(os.environ["PPO_ASYNC_EVALUATION_LOG"]),
        {
            "event": "evaluation",
            "recorded_at": _timestamp(),
            "rollout_id": int(rollout_id),
            "processed_examples": (int(rollout_id) + 1) * batch_size,
            "datasets": datasets,
        },
    )
    return True
