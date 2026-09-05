"""Bounded, completion-ordered rollouts with checkpointable outstanding prompts.

SLIME owns HTTP, generation, rewards and the background asyncio loop. This module
only schedules that work and records which logical attempts still need training.
"""

from __future__ import annotations

import asyncio
from collections import deque
import copy
import json
import os
from pathlib import Path
import threading
import time

from slime.rollout.data_source import RolloutDataSource
from slime.utils.types import Sample

from ppo_async.training.hooks import learner_updates_before, policy_lag


class StreamDataSource(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        if args.n_samples_per_prompt != 1:
            raise ValueError("streaming PPO requires one response per prompt")
        self.budget = args.num_rollout * args.rollout_batch_size
        self.lock = threading.RLock()
        self.pending = {}
        self.replay = deque()
        self.retries = {}
        self.worker = None

    def get_samples(self, num_samples):
        if num_samples != 1:
            raise ValueError("stream workers reserve one prompt at a time")
        with self.lock:
            if self.replay:
                return [copy.deepcopy(self.pending[self.replay.popleft()])]
            if self.sample_group_index >= self.budget:
                return []
            group = super().get_samples(1)[0]
            self.pending[group[0].index] = copy.deepcopy(group)
            return [group]

    def acknowledge(self, group):
        with self.lock:
            index = group[0].index
            del self.pending[index]
            self.retries.pop(index, None)

    def retry(self, group):
        with self.lock:
            index = group[0].index
            count = self.retries.get(index, 0) + 1
            if count > 3:
                raise RuntimeError(f"prompt {index} exceeded three policy-age retries")
            self.retries[index] = count
            self.replay.append(index)

    def save(self, rollout_id):
        # Reservation, acknowledgement and snapshot share one lock. The cursor
        # may be ahead of training, but every untrained attempt is saved with it.
        with self.lock:
            self.metadata["stream_pending"] = [
                [sample.to_dict() for sample in group] for group in self.pending.values()
            ]
            super().save(rollout_id)

    def load(self, rollout_id=None):
        with self.lock:
            super().load(rollout_id)
            groups = self.metadata.pop("stream_pending", [])
            self.pending = {group[0]["index"]: [Sample.from_dict(s) for s in group] for group in groups}
            self.replay = deque(self.pending)
            # Checkpoints contain original prompts, not stale model outputs.
            # Replay only outstanding attempts, preserving their original IDs.
            consumed = self.sample_group_index - len(self.pending)
            expected = (rollout_id + 1) * self.args.rollout_batch_size if rollout_id is not None else 0
            if consumed != expected:
                raise RuntimeError(f"stream checkpoint has {consumed} consumed attempts; expected {expected}")


class CompletionStream:
    def __init__(self, source, generate, *, concurrency, ready_capacity):
        self.source = source
        self.generate = generate
        self.ready = asyncio.Queue(maxsize=ready_capacity)
        self.tasks = [asyncio.create_task(self._produce()) for _ in range(concurrency)]

    async def _produce(self):
        try:
            while True:
                groups = self.source.get_samples(1)
                if not groups:
                    # Stay available for stale-sample retries, even at budget end.
                    await asyncio.sleep(0.01)
                    continue
                group = await self.generate(groups[0])
                await self.ready.put(group)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Fail the learner instead of silently losing a prompt or hanging.
            await self.ready.put(exc)

    async def take(self, size, accept):
        batch = []
        while len(batch) < size:
            group = await self.ready.get()
            if isinstance(group, Exception):
                raise group
            if accept(group):
                self.source.acknowledge(group)
                batch.append(group)
            else:
                self.source.retry(group)
        return batch

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def _published_updates():
    versions = json.loads(Path(os.environ["PPO_ASYNC_POLICY_VERSIONS"]).read_text())
    return max(int(updates) for updates in versions.values())


def _accept(args, group, rollout_id):
    sample = group[0]
    updates = learner_updates_before(args, rollout_id)
    oldest = sample.metadata["behavior_updates_at_submission"]
    age = policy_lag(updates, oldest)
    # A request may span publications. SGLang returns actual token logprobs but
    # only a final version label. Submission age is a conservative upper bound,
    # not an invented exact version for every token in the trajectory.
    sample.metadata.update(
        behavior_learner_updates=oldest,
        learner_updates_at_consumption=updates,
        learner_rollout_id=rollout_id,
        policy_lag_updates=age,
        policy_lag_is_upper_bound=True,
    )
    if sample.remove_sample:
        raise RuntimeError(f"invalid generated sample {sample.index}: {sample.metadata}")
    return age <= int(os.environ["PPO_ASYNC_MAX_POLICY_LAG"])


async def _next_batch(args, rollout_id, source):
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group

    if source.worker is None:
        state = GenerateState(args)

        async def generate(group):
            oldest = _published_updates()
            started = time.monotonic()
            result = await generate_and_rm_group(args, group, state.sampling_params.copy(), evaluation=False)
            if len(result) != 1 or result[0].status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
                raise RuntimeError("stream generation did not return one completed proof")
            sample = result[0]
            if len(sample.rollout_log_probs or []) != sample.response_length:
                raise RuntimeError("missing behavior log probabilities for generated tokens")
            sample.metadata = sample.metadata or {}
            sample.metadata.update(behavior_updates_at_submission=oldest,
                                   generation_seconds=time.monotonic() - started)
            return result

        source.worker = CompletionStream(
            source, generate, concurrency=args.sglang_server_concurrency,
            ready_capacity=args.rollout_batch_size,
        )
    started = time.monotonic()
    batch = await source.worker.take(args.rollout_batch_size, lambda group: _accept(args, group, rollout_id))
    from ppo_async.training.driver import _event
    _event("stream_batch", rollout_id=rollout_id, wait_seconds=time.monotonic() - started,
           ready=source.worker.ready.qsize(), sample_ids=[g[0].index for g in batch])
    if rollout_id == args.num_rollout - 1:
        await source.worker.close()
    return batch


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.utils.async_utils import run

    if evaluation:
        raise ValueError("use SLIME's separate evaluation function")
    return run(_next_batch(args, rollout_id, data_source))
