"""One-GPU synchronous PPO and two-GPU completion-driven asynchronous PPO."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from ppo_async.training.hooks import learner_updates_before, write_policy_versions


def role_slices(total_gpus: int = 2) -> dict[str, tuple[int, ...]]:
    if total_gpus not in (1, 2):
        raise ValueError("use one GPU for sync or two for async")
    return {"actor": (0,), "critic": (0,), "rollout": (total_gpus - 1,)}


def _placement_groups(args: Any) -> dict[str, Any]:
    from slime.ray.placement_group import _create_placement_group

    total = 1 if args.colocate else 2
    if args.offload_rollout and not args.sglang_enable_weights_cpu_backup:
        raise ValueError("rollout offload requires SGLang CPU weight backup for checkpoint resume")
    if not args.offload_train:
        raise ValueError("shared actor/critic GPU requires training offload")
    placement_group, bundle_indices, gpu_ids = _create_placement_group(total)
    return {
        role: (placement_group, [bundle_indices[i] for i in indices], [gpu_ids[i] for i in indices])
        for role, indices in role_slices(total).items()
    }


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
    parser.add_argument("--eval-concurrency", type=int, default=16)
    parser.add_argument("--final-eval-rollout", type=int, default=None)
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
        actor=list(role_slices(1 if args.colocate else 2)["actor"]),
        critic=list(role_slices(1 if args.colocate else 2)["critic"]),
        rollout=list(role_slices(1 if args.colocate else 2)["rollout"]),
    )
    init_tracking(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    if critic_model is None:
        raise RuntimeError("PPO must create a critic model")
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
    actor_model.update_weights()
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())
    version = int(getattr(args, "update_weight_start_version", 0)) + 1
    updates = learner_updates_before(args, args.start_rollout_id)
    write_policy_versions({version: updates})
    _event("weight_publish", published_version=version, learner_updates=updates)
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))
    return rollout_manager, num_rollout_per_epoch, actor_model, critic_model


def _train_one(args: Any, rollout_id: int, data, actor_model, critic_model) -> bool:
    import ray

    started = time.monotonic()
    actor_trains = rollout_id >= args.num_critic_only_steps
    value_refs = critic_model.async_train(rollout_id, data)
    # Both models occupy GPU 0. Critic must finish and offload before actor wakes.
    ray.get(value_refs)
    if actor_trains:
        ray.get(actor_model.async_train(rollout_id, data, external_data=value_refs))
    _event("learner_step", rollout_id=rollout_id, actor_trained=actor_trains,
           seconds=time.monotonic() - started)
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
    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())
    # Production checkpoints are recovery boundaries. Wait for any asynchronous
    # distributed save to finish before asking Modal to persist the Volume.
    force_sync = _production_mode(args) or rollout_id == args.num_rollout - 1
    if actor_trains:
        actor_model.save_model(rollout_id, force_sync=force_sync)
    critic_model.save_model(rollout_id, force_sync=force_sync)
    if args.rollout_global_dataset:
        ray.get(rollout_manager.save.remote(rollout_id))
    if args.offload_rollout:
        ray.get(rollout_manager.onload.remote())
    _snapshot_tracking_state(args, rollout_id)
    processed_examples = _processed_examples(args, rollout_id)
    _event(
        "checkpoint",
        rollout_id=rollout_id,
        processed_examples=processed_examples,
        force_sync=force_sync,
    )
    # Persist trained weights before evaluation can fail or the container ends.
    # On recovery, finish a missing selection eval before consuming more data.
    _commit_checkpoint(args, rollout_id)
    return True


def _commit_checkpoint(args, rollout_id):
    processed_examples = _processed_examples(args, rollout_id)
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
    _commit_checkpoint(args, rollout_id)
    return True


def _finish_pending_evaluation(args, rollout_manager):
    path = os.environ.get("PPO_ASYNC_EVALUATION_LOG")
    if not _production_mode(args) or args.start_rollout_id == 0 or not path:
        return
    rollout_id = args.start_rollout_id - 1
    log = Path(path)
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    if not any(record["rollout_id"] == rollout_id for record in records):
        _evaluate_if_due(args, rollout_id, rollout_manager)


def train_synchronous(args: Any) -> None:
    import ray
    from slime.observability.logging_utils import finish_tracking

    rollout_manager, per_epoch, actor_model, critic_model = _initialize(args)
    published_version = int(getattr(args, "update_weight_start_version", 0)) + 1
    versions = {published_version: learner_updates_before(args, args.start_rollout_id)}
    try:
        _finish_pending_evaluation(args, rollout_manager)
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            data = ray.get(rollout_manager.generate.remote(rollout_id))
            _event("rollout_batch_complete", rollout_id=rollout_id)
            ray.get(rollout_manager.offload.remote())
            actor_trains = _train_one(
                args, rollout_id, data, actor_model, critic_model
            )
            ray.get(rollout_manager.onload_weights.remote())
            actor_model.update_weights()
            ray.get(rollout_manager.onload_kv.remote())
            published_version += 1
            versions[published_version] = learner_updates_before(args, rollout_id + 1)
            write_policy_versions(versions)
            _event(
                "weight_publish",
                published_version=published_version,
                learner_updates=versions[published_version],
            )
            # Persist the trained checkpoint, then evaluate that exact policy.
            _save_if_due(
                args,
                rollout_id,
                per_epoch,
                actor_model,
                critic_model,
                rollout_manager,
                actor_trains,
            )
            _evaluate_if_due(args, rollout_id, rollout_manager)
    finally:
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)


def train_asynchronous(args: Any) -> None:
    import ray
    from slime.observability.logging_utils import finish_tracking

    rollout_manager, per_epoch, actor_model, critic_model = _initialize(args)
    published_version = int(getattr(args, "update_weight_start_version", 0)) + 1
    versions = {published_version: learner_updates_before(args, args.start_rollout_id)}
    try:
        _finish_pending_evaluation(args, rollout_manager)
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            # This dequeues completed proofs. The persistent producer continues
            # generating through learner updates, saves and evaluation.
            data = ray.get(rollout_manager.generate.remote(rollout_id))
            _event("rollout_batch_complete", rollout_id=rollout_id)
            actor_trains = _train_one(args, rollout_id, data, actor_model, critic_model)
            # SGLang aborts and drains active requests for transfer; the stream
            # resumes their partial tokens once admissions reopen.
            actor_model.update_weights()
            published_version += 1
            versions[published_version] = learner_updates_before(args, rollout_id + 1)
            write_policy_versions(versions)
            _event("weight_publish", published_version=published_version,
                   learner_updates=versions[published_version])
            _save_if_due(args, rollout_id, per_epoch, actor_model, critic_model,
                         rollout_manager, actor_trains)
            _evaluate_if_due(args, rollout_id, rollout_manager)
    finally:
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)


def main() -> None:
    from slime.utils.arguments import parse_args

    args = parse_args(add_custom_arguments=_add_production_arguments)
    if args.eval_concurrency <= 0:
        raise ValueError("eval concurrency must be positive")
    if _production_mode(args):
        if args.checkpoint_every_examples != args.evaluation_every_examples:
            raise ValueError("checkpoint and selection evaluation intervals must match")
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
    if args.final_eval_rollout is not None:
        from ppo_async.training.evaluation import evaluate_only
        evaluate_only(args)
        return
    arm = os.environ.get("PPO_ASYNC_ARM", "")
    if arm == "sync_ppo":
        train_synchronous(args)
    elif arm in {"async_ppo", "async_ppo_dis", "async_ppo_dis_masking"}:
        train_asynchronous(args)
    else:
        raise ValueError(f"unknown PPO_ASYNC_ARM: {arm!r}")


if __name__ == "__main__":
    main()
