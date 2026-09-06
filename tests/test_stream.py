"""Deterministic concurrency tests: a straggler must not hold later prompts up."""

import asyncio
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def stream(monkeypatch):
    # Only the SLIME data/tokenizer boundary is faked; exercise the real stream,
    # bounded queue, acknowledgement, age check and checkpoint wrapper.
    class Sample:
        def __init__(self, index):
            self.index = index
            self.metadata = {}
            self.remove_sample = False
        def to_dict(self):
            return copy.deepcopy(vars(self))
        @staticmethod
        def from_dict(value):
            sample = Sample(value['index'])
            sample.__dict__.update(copy.deepcopy(value))
            return sample

    class Base:
        saved = {}
        def __init__(self, args):
            self.args = args
            self.metadata = {}
            self.sample_group_index = 0
        def get_samples(self, size):
            sample = Sample(self.sample_group_index)
            self.sample_group_index += 1
            return [[sample]]
        def save(self, rollout_id):
            self.saved[rollout_id] = copy.deepcopy((self.sample_group_index, self.metadata))
        def load(self, rollout_id):
            if rollout_id in self.saved:
                self.sample_group_index, self.metadata = copy.deepcopy(self.saved[rollout_id])

    for name, attr, value in [('slime.rollout.data_source', 'RolloutDataSource', Base),
                              ('slime.utils.types', 'Sample', Sample)]:
        module = ModuleType(name)
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / 'src/ppo_async/training/stream.py'
    spec = importlib.util.spec_from_file_location('stream_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(stream, budget=24):
    return stream.StreamDataSource(SimpleNamespace(n_samples_per_prompt=1, num_rollout=budget // 8,
                                                   rollout_batch_size=8, num_critic_only_steps=0))


def test_fast_completions_train_and_replenish_while_first_proof_is_blocked(stream):
    async def run():
        data = source(stream)
        release_slow = asyncio.Event()
        started, finished = [], []
        async def generate(group):
            index = group[0].index
            started.append(index)
            if index == 0:
                await release_slow.wait()
            finished.append(index)
            return group
        worker = stream.CompletionStream(data, generate, concurrency=8, ready_capacity=8)
        try:
            first = await asyncio.wait_for(worker.take(8, lambda _: True), 1)
            assert [g[0].index for g in first] == list(range(1, 9))
            assert 0 not in finished
            # Simulate learner compute: no further take() call is needed to
            # start and finish work beyond the initial eight-prompt cohort.
            await asyncio.sleep(0)
            assert 16 in started
            assert worker.ready.qsize() <= 8
            # At most eight completed queued + eight producing/blocking on put.
            assert len(data.pending) <= 16
            release_slow.set()
            rest = await asyncio.wait_for(worker.take(16, lambda _: True), 1)
            ids = [g[0].index for g in first + rest]
            assert sorted(ids) == list(range(24))
            assert not data.pending
            assert data.sample_group_index == 24
        finally:
            await worker.close()
    asyncio.run(run())


def test_checkpoint_replays_only_unconsumed_attempts_at_budget_end(stream):
    data = source(stream, 16)
    groups = [data.get_samples(1)[0] for _ in range(16)]
    consumed = [1, 3, 5, 7, 9, 11, 13, 15]
    for index in consumed:
        data.acknowledge(groups[index])
    # Mutating an in-flight response cannot contaminate its saved raw prompt.
    groups[0][0].metadata['generated_response'] = 'stale'
    data.save(0)
    resumed = source(stream, 16)
    resumed.load(0)
    replayed = [resumed.get_samples(1)[0] for _ in range(8)]
    assert [g[0].index for g in replayed] == list(range(0, 16, 2))
    assert all(not g[0].metadata for g in replayed)
    assert resumed.get_samples(1) == []
    for group in replayed:
        resumed.acknowledge(group)
    resumed.save(1)
    complete = source(stream, 16)
    complete.load(1)
    assert complete.get_samples(1) == []
    assert not complete.pending


def test_stale_proof_retries_same_prompt_after_budget_exhaustion(stream):
    async def run():
        data = source(stream, 8)
        counts = {}
        async def generate(group):
            i = group[0].index
            counts[i] = counts.get(i, 0) + 1
            group[0].metadata['attempt'] = counts[i]
            return group
        worker = stream.CompletionStream(data, generate, concurrency=8, ready_capacity=8)
        try:
            batch = await asyncio.wait_for(worker.take(8, lambda g: g[0].metadata['attempt'] == 2), 1)
            assert sorted(g[0].index for g in batch) == list(range(8))
            assert data.sample_group_index == 8
            assert set(counts.values()) == {2}
        finally:
            await worker.close()
    asyncio.run(run())


def test_generation_exception_reaches_learner(stream):
    async def run():
        async def broken(group):
            raise RuntimeError('verifier service unavailable')
        worker = stream.CompletionStream(source(stream), broken, concurrency=1, ready_capacity=8)
        try:
            with pytest.raises(RuntimeError, match='verifier service'):
                await asyncio.wait_for(worker.take(8, lambda _: True), 1)
        finally:
            await worker.close()
    asyncio.run(run())


def test_age_is_checked_at_dequeue_with_mixed_versions_allowed(stream, monkeypatch):
    monkeypatch.setenv('PPO_ASYNC_MAX_POLICY_LAG', '4')
    data = source(stream)
    group = data.get_samples(1)[0]
    sample = group[0]
    sample.weight_versions = ['7', '8']
    sample.metadata['behavior_updates_at_submission'] = 10
    assert stream._accept(data.args, group, 14)
    assert sample.metadata['policy_lag_updates'] == 4
    assert sample.metadata['policy_lag_is_upper_bound']
    assert not stream._accept(data.args, group, 15)
    assert sample.metadata['policy_lag_updates'] == 5


def test_retry_limit_prevents_unbounded_wasted_compute(stream):
    data = source(stream)
    group = data.get_samples(1)[0]
    for _ in range(3):
        data.retry(group)
        group = data.get_samples(1)[0]
    with pytest.raises(RuntimeError, match='three policy-age retries'):
        data.retry(group)


def test_publication_resumes_partial_proof_with_one_total_budget(stream, monkeypatch):
    statuses = SimpleNamespace(ABORTED='abort', COMPLETED='stop', TRUNCATED='length')
    stream.Sample.Status = statuses
    events = []
    driver = ModuleType('ppo_async.training.driver')
    driver._event = lambda *a, **kw: events.append((a, kw))
    monkeypatch.setitem(sys.modules, driver.__name__, driver)
    sample = stream.Sample(7)
    sample.status, sample.response_length = 'pending', 0
    sample.tokens, sample.rollout_log_probs = [99], []
    params = {'max_new_tokens': 6}
    remaining = []
    async def generate(args, group, sampling_params, evaluation):
        # SLIME's continuation contract: existing tokens are the next prompt,
        # and the old response length is subtracted from a fresh budget.
        assert group[0] is sample
        assert sample.tokens == [99] + list(range(sample.response_length))
        sampling_params['max_new_tokens'] -= sample.response_length
        remaining.append(sampling_params['max_new_tokens'])
        start = sample.response_length
        sample.tokens.extend(range(start, start + 2))
        sample.rollout_log_probs.extend([-0.1 * (start + 1)] * 2)
        sample.response_length += 2
        sample.status = statuses.TRUNCATED if sample.response_length == 6 else statuses.ABORTED
        return group
    result = asyncio.run(stream._complete_generation(SimpleNamespace(num_rollout=3), [sample], params, generate))
    assert result == [sample]
    assert remaining == [6, 4, 2]
    assert params == {'max_new_tokens': 6}
    assert sample.tokens == [99, 0, 1, 2, 3, 4, 5]
    assert sample.rollout_log_probs == pytest.approx([-.1, -.1, -.3, -.3, -.5, -.5])
    assert len(events) == 2


@pytest.mark.parametrize('failure', ['missing_logprobs', 'over_budget', 'unexpected_status', 'endless_abort'])
def test_partial_generation_fails_closed(stream, monkeypatch, failure):
    stream.Sample.Status = SimpleNamespace(ABORTED='abort', COMPLETED='stop', TRUNCATED='length')
    driver = ModuleType('ppo_async.training.driver')
    driver._event = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, driver.__name__, driver)
    sample = stream.Sample(0)
    sample.response_length = 2 if failure == 'over_budget' else 1
    sample.rollout_log_probs = [] if failure == 'missing_logprobs' else [-.1] * sample.response_length
    sample.status = 'pending' if failure == 'unexpected_status' else 'abort'
    async def generate(*a, **kw):
        return [sample]
    with pytest.raises(RuntimeError):
        asyncio.run(stream._complete_generation(SimpleNamespace(num_rollout=2), [sample], {'max_new_tokens': 1}, generate))
