"""Pinned SLIME training loops with non-overlapping four-GPU placement.

SLIME v0.3.2's stock placement helper gives the critic the actor placement
group. This driver deliberately partitions one four-GPU Ray placement group
into actor, critic, and rollout slices before delegating all model, SGLang,
rollout, and optimizer work to SLIME.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

from ppo_async.training.hooks import learner_updates_before, write_policy_versions


def role_slices(total_gpus: int = 4) -> dict[str, tuple[int, ...]]:
    if total_gpus != 4:
        raise ValueError("PPO_ASYNC requires exactly four GPUs")
    return {"actor": (0,), "critic": (1,), "rollout": (2, 3)}


def _placement_groups(args: Any) -> dict[str, Any]:
    from slime.ray.placement_group import _create_placement_group

    if args.actor_num_nodes != 1 or args.actor_num_gpus_per_node != 1:
        raise ValueError("actor must use exactly GPU 0")
    if args.critic_num_nodes != 1 or args.critic_num_gpus_per_node != 1:
        raise ValueError("critic must use exactly GPU 1")
    if args.rollout_num_gpus != 2 or args.rollout_num_gpus_per_engine != 1:
        raise ValueError("rollout must use two independent one-GPU engines")
    if args.colocate:
        raise ValueError("colocation would violate physical GPU isolation")

    placement_group, bundle_indices, gpu_ids = _create_placement_group(4)

    def subset(indices: tuple[int, ...]):
        return (
            placement_group,
            [bundle_indices[index] for index in indices],
            [gpu_ids[index] for index in indices],
        )

    slices = role_slices()
    return {role: subset(indices) for role, indices in slices.items()}


def _event(kind: str, **fields: Any) -> None:
    path = Path(os.environ.get("PPO_ASYNC_EVENT_LOG", "/tmp/ppo-async-events.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": kind,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _add_production_arguments(parser):
    parser.add_argument("--processed-example-budget", type=int, default=None)
    parser.add_argument("--scalar-log-every-examples", type=int, default=None)
    parser.add_argument("--checkpoint-every-examples", type=int, default=None)
    parser.add_argument("--evaluation-every-examples", type=int, default=None)
    return parser


def _processed_examples(args: Any, rollout_id: int) -> int:
    processed = (rollout_id + 1) * args.rollout_batch_size * args.n_samples_per_prompt
    budget = getattr(args, "processed_example_budget", None)
    return min(processed, budget) if budget is not None else processed


def _crosses_interval(args: Any, rollout_id: int, interval: int | None) -> bool:
    if interval is None:
        return False
    current = _processed_examples(args, rollout_id)
    previous = 0 if rollout_id == 0 else _processed_examples(args, rollout_id - 1)
    final = rollout_id == args.num_rollout - 1
    return final or current // interval > previous // interval


def _production_mode(args: Any) -> bool:
    return getattr(args, "processed_example_budget", None) is not None


def _snapshot_tracking_state(args: Any, rollout_id: int) -> None:
    source_value = os.environ.get("PPO_ASYNC_TRACKING_STATE")
    snapshot_root_value = os.environ.get("PPO_ASYNC_TRACKING_SNAPSHOTS")
    if not source_value or not snapshot_root_value:
        return
    source = Path(source_value)
    if not source.is_file():
        return
    snapshot_root = Path(snapshot_root_value)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snapshot_root / f"tracking_state_{rollout_id}.json")


def _write_commit_boundary(args: Any, rollout_id: int) -> None:
    snapshot_root_value = os.environ.get("PPO_ASYNC_TRACKING_SNAPSHOTS")
    if not snapshot_root_value:
        return
    root = Path(snapshot_root_value).parent / "commits"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"rollout_{rollout_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "rollout_id": rollout_id,
                "processed_examples": _processed_examples(args, rollout_id),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _initialize(args: Any):
    import ray
    from slime.observability.logging_utils import configure_logger, init_tracking
    from slime.ray.placement_group import create_rollout_manager, create_training_models

    configure_logger()
    pgs = _placement_groups(args)
    _event(
        "gpu_placement",
        actor=list(role_slices()["actor"]),
        critic=list(role_slices()["critic"]),
        rollout=list(role_slices()["rollout"]),
    )
    init_tracking(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    if critic_model is None:
        raise RuntimeError("PPO must create a critic model")
    actor_model.update_weights()
    version = int(getattr(args, "update_weight_start_version", 0)) + 1
    updates = learner_updates_before(args, args.start_rollout_id)
    write_policy_versions({version: updates})
    _event("weight_publish", published_version=version, learner_updates=updates)
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))
    return rollout_manager, num_rollout_per_epoch, actor_model, critic_model


def _train_one(args: Any, rollout_id: int, data, actor_model, critic_model) -> bool:
    import ray

    actor_trains = rollout_id >= args.num_critic_only_steps
    value_refs = critic_model.async_train(rollout_id, data)
    if actor_trains:
        ray.get(actor_model.async_train(rollout_id, data, external_data=value_refs))
    else:
        ray.get(value_refs)
    _event("learner_step", rollout_id=rollout_id, actor_trained=actor_trains)
    return actor_trains


def _save_if_due(
    args: Any,
    rollout_id: int,
    num_rollout_per_epoch: int | None,
    actor_model: Any,
    critic_model: Any,
    rollout_manager: Any,
    actor_trains: bool,
) -> bool:
    import ray
    from slime.utils.misc import should_run_periodic_action

    if _production_mode(args):
        due = _crosses_interval(args, rollout_id, args.checkpoint_every_examples)
    else:
        due = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
    if not due:
        return False
    # Production checkpoints are recovery boundaries. Wait for any asynchronous
    # distributed save to finish before asking Modal to persist the Volume.
    force_sync = _production_mode(args) or rollout_id == args.num_rollout - 1
    if actor_trains:
        actor_model.save_model(rollout_id, force_sync=force_sync)
    critic_model.save_model(rollout_id, force_sync=force_sync)
    if args.rollout_global_dataset:
        ray.get(rollout_manager.save.remote(rollout_id))
    _snapshot_tracking_state(args, rollout_id)
    processed_examples = _processed_examples(args, rollout_id)
    _event(
        "checkpoint",
        rollout_id=rollout_id,
        processed_examples=processed_examples,
        force_sync=force_sync,
    )
    _write_commit_boundary(args, rollout_id)
    # The Modal parent watches this marker and commits the mounted Volume while
    # the long-running child process is still alive. This makes checkpoints
    # recoverable even if the container is later interrupted.
    print(
        "PPO_ASYNC_COMMIT "
        + json.dumps({"rollout_id": rollout_id, "processed_examples": processed_examples}),
        flush=True,
    )
    return True


def _evaluation_due(args: Any, rollout_id: int) -> bool:
    return _production_mode(args) and _crosses_interval(
        args, rollout_id, args.evaluation_every_examples
    )


def _evaluate_if_due(args: Any, rollout_id: int, rollout_manager: Any) -> bool:
    if not _evaluation_due(args, rollout_id):
        return False
    import ray

    ray.get(rollout_manager.eval.remote(rollout_id))
    _event(
        "evaluation",
        rollout_id=rollout_id,
        processed_examples=_processed_examples(args, rollout_id),
    )
    return True


def train_synchronous(args: Any) -> None:
    import ray
    from slime.observability.logging_utils import finish_tracking

    rollout_manager, per_epoch, actor_model, critic_model = _initialize(args)
    published_version = int(getattr(args, "update_weight_start_version", 0)) + 1
    versions = {published_version: learner_updates_before(args, args.start_rollout_id)}
    try:
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            data = ray.get(rollout_manager.generate.remote(rollout_id))
            _event("rollout_batch_complete", rollout_id=rollout_id)
            actor_trains = _train_one(
                args, rollout_id, data, actor_model, critic_model
            )
            actor_model.update_weights()
            published_version += 1
            versions[published_version] = learner_updates_before(args, rollout_id + 1)
            write_policy_versions(versions)
            _event(
                "weight_publish",
                published_version=published_version,
                learner_updates=versions[published_version],
            )
            _evaluate_if_due(args, rollout_id, rollout_manager)
            # Checkpoint last: the dataset cursor, actor/critic/optimizer,
            # scalar state, evaluation record, and HF export become one
            # recoverable boundary when the parent commits the Modal Volume.
            _save_if_due(
                args,
                rollout_id,
                per_epoch,
                actor_model,
                critic_model,
                rollout_manager,
                actor_trains,
            )
    finally:
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)


def train_asynchronous(args: Any) -> None:
    import ray
    from slime.observability.logging_utils import finish_tracking

    rollout_manager, per_epoch, actor_model, critic_model = _initialize(args)
    publish_interval = int(os.environ["PPO_ASYNC_PUBLISH_INTERVAL"])
    max_lag = int(os.environ["PPO_ASYNC_MAX_POLICY_LAG"])
    published_version = int(getattr(args, "update_weight_start_version", 0)) + 1
    learner_updates = learner_updates_before(args, args.start_rollout_id)
    published_updates = learner_updates
    versions = {published_version: published_updates}
    # A retry after the final committed checkpoint must not consume a third
    # pass's first batch while merely rebuilding the completion report.
    next_future = (
        rollout_manager.generate.remote(args.start_rollout_id)
        if args.start_rollout_id < args.num_rollout else None
    )
    prefetched_data = None
    try:
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if prefetched_data is not None:
                current_data = prefetched_data
                prefetched_data = None
            else:
                if next_future is None:
                    next_future = rollout_manager.generate.remote(rollout_id)
                current_data = ray.get(next_future)
                next_future = None
            _event("rollout_batch_complete", rollout_id=rollout_id)

            action_due = _crosses_interval(
                args, rollout_id, getattr(args, "checkpoint_every_examples", None)
            ) or _evaluation_due(args, rollout_id)
            # Do not prefetch with weights that will be too old at consumption.
            # In that case, finish this update and publish before generating.
            next_lag = learner_updates_before(args, rollout_id + 1) - published_updates
            if rollout_id + 1 < args.num_rollout and not action_due and next_lag <= max_lag:
                next_future = rollout_manager.generate.remote(rollout_id + 1)
                _event("rollout_batch_started", rollout_id=rollout_id + 1)

            actor_trains = _train_one(
                args, rollout_id, current_data, actor_model, critic_model
            )
            learner_updates += int(actor_trains)
            final_step = rollout_id == args.num_rollout - 1
            if final_step or action_due or next_lag > max_lag or (rollout_id + 1) % publish_interval == 0:
                # SLIME must never mutate SGLang weights midway through a
                # trajectory. Finish the already-overlapped batch first.
                if next_future is not None:
                    prefetched_data = ray.get(next_future)
                    next_future = None
                actor_model.update_weights()
                published_version += 1
                published_updates = learner_updates
                versions[published_version] = published_updates
                write_policy_versions(versions)
                _event(
                    "weight_publish",
                    published_version=published_version,
                    learner_updates=learner_updates,
                )
            _evaluate_if_due(args, rollout_id, rollout_manager)
            _save_if_due(
                args,
                rollout_id,
                per_epoch,
                actor_model,
                critic_model,
                rollout_manager,
                actor_trains,
            )

            if (
                rollout_id + 1 < args.num_rollout
                and next_future is None
                and prefetched_data is None
            ):
                next_future = rollout_manager.generate.remote(rollout_id + 1)
                _event("rollout_batch_started", rollout_id=rollout_id + 1)
    finally:
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)


def main() -> None:
    from slime.utils.arguments import parse_args

    args = parse_args(add_custom_arguments=_add_production_arguments)
    if _production_mode(args):
        expected = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt
        if expected != args.processed_example_budget:
            raise ValueError(
                "production budget must equal num_rollout * rollout_batch_size * n_samples_per_prompt"
            )
        for name in (
            "scalar_log_every_examples",
            "checkpoint_every_examples",
            "evaluation_every_examples",
        ):
            value = getattr(args, name)
            if value is None or value <= 0:
                raise ValueError(f"{name} must be positive in production")
    if os.environ.get("PPO_ASYNC_PARSE_ONLY") == "1":
        print("PPO_ASYNC_PARSE_OK", flush=True)
        return
    arm = os.environ.get("PPO_ASYNC_ARM", "")
    if arm == "sync_ppo":
        train_synchronous(args)
    elif arm in {"async_ppo", "async_ppo_dis"}:
        train_asynchronous(args)
    else:
        raise ValueError(f"unknown PPO_ASYNC_ARM: {arm!r}")


if __name__ == "__main__":
    main()
