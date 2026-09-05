"""Exercise the real driver with deterministic fake Ray and model actors."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ppo_async.training import driver
from ppo_async.training.hooks import audit_policy_version, learner_updates_before, write_policy_versions


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("start,critic_only", [(0, 0), (13, 0), (88, 0), (175, 0), (0, 3), (2, 3)])
def test_lag_and_dataset_cursor_across_publications_and_resume(
    asynchronous, start, critic_only, tmp_path, monkeypatch,
):
    args = SimpleNamespace(
        start_rollout_id=start, num_rollout=175, num_critic_only_steps=critic_only,
        rollout_batch_size=8, n_samples_per_prompt=1, processed_example_budget=1400,
        checkpoint_every_examples=100, evaluation_every_examples=100,
        offload_rollout=not asynchronous,
    )
    versions_path = tmp_path / "versions.json"
    # A restarted container must not inherit an old engine's version mapping.
    versions_path.write_text('{"99": 999}\n')
    monkeypatch.setenv("PPO_ASYNC_POLICY_VERSIONS", str(versions_path))
    monkeypatch.setenv("PPO_ASYNC_MAX_POLICY_LAG", "1" if asynchronous else "0")
    monkeypatch.setenv("PPO_ASYNC_PUBLISH_INTERVAL", "2" if asynchronous else "1")
    state = {
        "updates": learner_updates_before(args, start), "version": 1,
        "published_updates": learner_updates_before(args, start), "pending": 0,
    }
    generated, trained, checkpoints, overlaps = [], [], [], []

    class Future:
        def __init__(self, value):
            self.value = value
        def get(self):
            state["pending"] -= 1
            return self.value

    def generate(rollout_id):
        assert state["pending"] == 0
        state["pending"] += 1
        sample = SimpleNamespace(
            weight_versions=[str(state["version"])], metadata={}, remove_sample=False,
        )
        audit_policy_version(args, sample, rollout_id=rollout_id)
        assert not sample.remove_sample, sample.metadata
        generated.append(rollout_id)
        return Future((sample, state["published_updates"]))

    def publish():
        assert state["pending"] == 0, "weights changed while generation was outstanding"
        state["version"] += 1
        state["published_updates"] = state["updates"]

    manager = SimpleNamespace(
        generate=SimpleNamespace(remote=generate),
        dispose=SimpleNamespace(remote=lambda: None),
        offload=SimpleNamespace(remote=lambda: None),
        onload_weights=SimpleNamespace(remote=lambda: None),
        onload_kv=SimpleNamespace(remote=lambda: None),
    )
    actor = SimpleNamespace(update_weights=publish)

    def initialize(_args):
        write_policy_versions({1: state["updates"]})
        return manager, None, actor, None

    def train(_args, rollout_id, data, *_models):
        sample, behavior_updates = data
        actual_lag = state["updates"] - behavior_updates
        assert sample.metadata["policy_lag_updates"] == actual_lag
        assert actual_lag <= (1 if asynchronous else 0)
        assert sample.metadata["learner_updates_at_consumption"] == state["updates"]
        overlaps.append(state["pending"] > 0)
        trained.append(rollout_id)
        actor_trains = rollout_id >= critic_only
        state["updates"] += int(actor_trains)
        return actor_trains

    def save(_args, rollout_id, *_rest):
        if driver._crosses_interval(args, rollout_id, 100):
            assert state["pending"] == 0
            assert generated == trained, "checkpoint cursor is ahead of trained data"
            checkpoints.append(rollout_id)

    ray = ModuleType("ray")
    ray.get = lambda value: value.get() if isinstance(value, Future) else value
    logging = ModuleType("slime.observability.logging_utils")
    logging.finish_tracking = lambda _args: None
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "slime.observability.logging_utils", logging)
    monkeypatch.setattr(driver, "_initialize", initialize)
    monkeypatch.setattr(driver, "_train_one", train)
    monkeypatch.setattr(driver, "_save_if_due", save)
    monkeypatch.setattr(driver, "_evaluate_if_due", lambda *args: None)
    monkeypatch.setattr(driver, "_event", lambda *args, **kwargs: None)

    (driver.train_asynchronous if asynchronous else driver.train_synchronous)(args)

    assert generated == trained == list(range(start, 175))
    if start < 175:
        assert checkpoints[-1] == 174
    else:
        assert not checkpoints
    # Background producer overlap is exercised by test_stream; the driver
    # itself must not launch a competing batch-generation call.
    assert not any(overlaps)
    versions = json.loads(versions_path.read_text())
    assert versions["1"] == learner_updates_before(args, start)
    assert versions[str(state["version"])] == state["updates"]


def test_shared_gpu_critic_finishes_before_actor_is_submitted(monkeypatch):
    events = []
    ray = ModuleType('ray')
    def get(refs):
        events.append(('wait', refs))
    ray.get = get
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.setattr(driver, '_event', lambda *args, **kwargs: None)
    def critic_train(*args):
        events.append(('submit', 'critic'))
        return 'critic_done_and_offloaded'
    def actor_train(*args, external_data):
        assert events[-1] == ('wait', 'critic_done_and_offloaded')
        assert external_data == 'critic_done_and_offloaded'
        events.append(('submit', 'actor'))
        return 'actor_done_and_offloaded'
    driver._train_one(SimpleNamespace(num_critic_only_steps=0), 0, 'batch',
                      SimpleNamespace(async_train=actor_train), SimpleNamespace(async_train=critic_train))
    assert events == [('submit', 'critic'), ('wait', 'critic_done_and_offloaded'),
                      ('submit', 'actor'), ('wait', 'actor_done_and_offloaded')]
