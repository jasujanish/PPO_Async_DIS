"""Fixed checkpoint selection, independent eval concurrency, and best-model eval."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path


_EVALUATING = ContextVar("ppo_evaluating", default=False)


class EvaluationSemaphore:
    """Route eval tasks to their own limit without changing active training tasks."""

    def __init__(self, training, concurrency):
        self.training = training
        self.evaluation = asyncio.Semaphore(concurrency)

    def _current(self):
        return self.evaluation if _EVALUATING.get() else self.training

    async def __aenter__(self):
        return await self._current().__aenter__()

    async def __aexit__(self, *exc):
        return await self._current().__aexit__(*exc)


def selection_paths(paths, output_root: Path, per_dataset: int, seed: int):
    """Select by immutable statement hash, independent of input row ordering."""
    output_root.mkdir(parents=True, exist_ok=True)
    selected = {}
    for name, path in paths.items():
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: hashlib.sha256(
            f"{seed}:{r['metadata']['statement_hash']}".encode()
        ).hexdigest())
        if len(rows) < per_dataset:
            raise ValueError(f"{name} has fewer than {per_dataset} selection problems")
        destination = output_root / f"{name}.jsonl"
        destination.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows[:per_dataset]))
        selected[name] = str(destination)
    return selected


def best_checkpoint(records):
    """Highest aggregate pass@1; break ties in favor of the earlier checkpoint."""
    def score(record):
        datasets = record["datasets"].values()
        total = sum(d["examples"] for d in datasets)
        return sum(d["pass_at_1"] * d["examples"] for d in datasets) / total

    best = max(records, key=lambda r: (score(r), -r["rollout_id"]))
    return {"rollout_id": best["rollout_id"], "processed_examples": best["processed_examples"],
            "pass_at_1": score(best)}


class EvaluationClient:
    """SLIME also caps its HTTP pool at training concurrency; route eval separately."""

    def __init__(self, training, evaluation):
        self.training, self.evaluation = training, evaluation

    def __getattr__(self, name):
        return getattr(self.evaluation if _EVALUATING.get() else self.training, name)


async def _evaluate(args, rollout_id):
    from slime.rollout.sglang_rollout import GenerateState, eval_rollout

    state = GenerateState(args)
    if not isinstance(state.semaphore, EvaluationSemaphore):
        state.semaphore = EvaluationSemaphore(state.semaphore, args.eval_concurrency)
    import httpx
    from slime.utils import http_utils

    if args.use_distributed_post:
        raise ValueError("separate eval concurrency requires local HTTP dispatch")
    training_client = http_utils._http_client
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.eval_concurrency),
                                 timeout=httpx.Timeout(None), trust_env=False) as client:
        http_utils._http_client = EvaluationClient(training_client, client)
        token = _EVALUATING.set(True)
        try:
            output, _ = await eval_rollout(args, rollout_id)
            # Infrastructure failures must not affect checkpoint selection.
            for values in output.data.values():
                if any(sample.remove_sample for sample in values["samples"]):
                    raise RuntimeError("evaluation contains an invalid/infrastructure-error sample")
            return output
        finally:
            _EVALUATING.reset(token)
            http_utils._http_client = training_client


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.utils.async_utils import run

    if not evaluation:
        raise ValueError("evaluation hook cannot generate training batches")
    return run(_evaluate(args, rollout_id))


def evaluate_only(args):
    """Load the selected HF export directly; do not construct actor or critic."""
    import ray
    from slime.observability.logging_utils import configure_logger, init_tracking, finish_tracking
    from slime.ray.placement_group import _create_placement_group, create_rollout_manager

    configure_logger()
    init_tracking(args)
    args.offload_rollout = False
    # No learner is present, so inference owns GPU 0 in either arm.
    args.colocate = True
    args.num_gpus_per_node = 1
    manager, _ = create_rollout_manager(args, _create_placement_group(1))
    try:
        ray.get(manager.eval.remote(args.final_eval_rollout))
    finally:
        ray.get(manager.dispose.remote())
        finish_tracking(args)


def final_eval_command(command, best, selection, full):
    command = list(command)
    command[command.index("--hf-checkpoint") + 1] = best["hf_export"]
    for name, path in selection.items():
        command[command.index(str(path))] = str(full[name])
    return command + ["--final-eval-rollout", str(best["rollout_id"])]
