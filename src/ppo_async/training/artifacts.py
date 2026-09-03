"""Completion gates for paid remote training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_smoke_artifacts(run_root: Path, final_rollout_id: int) -> dict[str, Any]:
    """Fail closed unless training and every required export were persisted."""
    required = {
        "actor_checkpoint": run_root
        / "checkpoints"
        / "actor"
        / "latest_checkpointed_iteration.txt",
        "critic_checkpoint": run_root
        / "checkpoints"
        / "critic"
        / "latest_checkpointed_iteration.txt",
        "hf_export": run_root
        / f"hf-{final_rollout_id}"
        / "model.safetensors.index.json",
        "event_log": run_root / "events.jsonl",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"smoke run missing required artifacts: {', '.join(missing)}")

    events = [
        json.loads(line)
        for line in required["event_log"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_events = {
        "gpu_placement": lambda event: True,
        "learner_step": lambda event: event.get("rollout_id") == final_rollout_id,
        "checkpoint": lambda event: event.get("rollout_id") == final_rollout_id,
        "weight_publish": lambda event: event.get("learner_updates", 0) >= 1,
    }
    absent_events = [
        kind
        for kind, predicate in required_events.items()
        if not any(event.get("event") == kind and predicate(event) for event in events)
    ]
    if absent_events:
        raise RuntimeError(
            f"smoke run missing terminal events: {', '.join(absent_events)}"
        )
    return {
        "actor_checkpoint": str(required["actor_checkpoint"].parent),
        "critic_checkpoint": str(required["critic_checkpoint"].parent),
        "hf_export": str(required["hf_export"].parent),
        "events": len(events),
    }


def validate_production_artifacts(
    run_root: Path,
    *,
    final_rollout_id: int,
    processed_example_budget: int,
    scalar_interval: int,
    action_interval: int,
) -> dict[str, Any]:
    """Require terminal state, exact tracking windows, and all scheduled actions."""
    verified = validate_smoke_artifacts(run_root, final_rollout_id)
    event_path = run_root / "events.jsonl"
    metric_path = run_root / "metrics.jsonl"
    evaluation_path = run_root / "evaluations.jsonl"
    tracking_state_path = run_root / "tracking" / "state.json"
    missing = [
        str(path)
        for path in (metric_path, evaluation_path, tracking_state_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"production run missing tracking artifacts: {missing}")

    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = [json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluations = [
        json.loads(line)
        for line in evaluation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_scalar_windows = processed_example_budget // scalar_interval
    expected_actions = processed_example_budget // action_interval
    num_rollouts = final_rollout_id + 1
    if processed_example_budget % num_rollouts:
        raise RuntimeError("production budget is not divisible by rollout count")
    batch_size = processed_example_budget // num_rollouts
    expected_action_ids = [
        (threshold + batch_size - 1) // batch_size - 1
        for threshold in range(action_interval, processed_example_budget + 1, action_interval)
    ]
    scalar_ends = [record.get("examples_end") for record in metrics if record.get("event") == "scalar_metrics"]
    if scalar_ends != list(range(scalar_interval, processed_example_budget + 1, scalar_interval)):
        raise RuntimeError("production scalar metric windows are incomplete or out of order")
    checkpoint_events = [event for event in events if event.get("event") == "checkpoint"]
    evaluation_events = [event for event in events if event.get("event") == "evaluation"]
    if len(checkpoint_events) != expected_actions or len(evaluation_events) != expected_actions:
        raise RuntimeError(
            "production checkpoint/evaluation cadence mismatch: "
            f"checkpoints={len(checkpoint_events)}, evaluations={len(evaluation_events)}, expected={expected_actions}"
        )
    if [event.get("rollout_id") for event in checkpoint_events] != expected_action_ids:
        raise RuntimeError("production checkpoint rollout IDs do not match example thresholds")
    if [event.get("rollout_id") for event in evaluation_events] != expected_action_ids:
        raise RuntimeError("production evaluation rollout IDs do not match example thresholds")
    if len(evaluations) != expected_actions:
        raise RuntimeError(
            f"production evaluation records mismatch: {len(evaluations)} != {expected_actions}"
        )
    if any(set(record.get("datasets", {})) != {"gaokao-formal", "fate-m"} for record in evaluations):
        raise RuntimeError("production evaluation record is missing a required dataset")
    if [record.get("rollout_id") for record in evaluations] != expected_action_ids:
        raise RuntimeError("production evaluation records do not match example thresholds")
    missing_exports = [
        rollout_id
        for rollout_id in expected_action_ids
        if not (run_root / f"hf-{rollout_id}" / "model.safetensors.index.json").is_file()
    ]
    missing_commits = [
        rollout_id
        for rollout_id in expected_action_ids
        if not (run_root / "tracking" / "commits" / f"rollout_{rollout_id}.json").is_file()
    ]
    if missing_exports or missing_commits:
        raise RuntimeError(
            f"production cadence artifacts missing: hf={missing_exports}, commits={missing_commits}"
        )
    state = json.loads(tracking_state_path.read_text(encoding="utf-8"))
    if state.get("next_example") != processed_example_budget + 1 or state.get("pending"):
        raise RuntimeError("production scalar tracking state did not finish cleanly")
    return {
        **verified,
        "scalar_windows": expected_scalar_windows,
        "checkpoint_events": len(checkpoint_events),
        "evaluation_events": len(evaluation_events),
        "evaluation_records": len(evaluations),
    }
