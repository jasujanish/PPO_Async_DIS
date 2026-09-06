import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ppo_async.config import load_experiment
from ppo_async.training import evaluation as ev
from ppo_async.training.driver import _crosses_interval


def test_fixed_selection_is_balanced_repeatable_and_order_independent(tmp_path):
    root = Path(__file__).resolve().parents[1] / "prepared_data/balanced-400-v4"
    paths = {name: root / f"eval-{name}.jsonl" for name in ("gaokao-formal", "fate-m")}
    selected = ev.selection_paths(paths, tmp_path / "first", 100, 42)
    shuffled = {}
    for name, path in paths.items():
        shuffled[name] = tmp_path / f"reversed-{name}.jsonl"
        shuffled[name].write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    repeated = ev.selection_paths(shuffled, tmp_path / "second", 100, 42)
    for name in paths:
        assert Path(selected[name]).read_bytes() == Path(repeated[name]).read_bytes()
        rows = [json.loads(line) for line in Path(selected[name]).read_text().splitlines()]
        assert len({r["metadata"]["statement_hash"] for r in rows}) == 100


def test_select_best_and_load_that_export_not_last_checkpoint():
    records = [{"rollout_id": i, "processed_examples": (i + 1) * 8,
                "datasets": {"a": {"examples": 100, "pass_at_1": score},
                             "b": {"examples": 100, "pass_at_1": score}}}
               for i, score in [(24, .2), (49, .3), (74, .25)]]
    best = ev.best_checkpoint(records)
    assert best == {"rollout_id": 49, "processed_examples": 400, "pass_at_1": .3}
    records[-1]["datasets"] = records[1]["datasets"]
    assert ev.best_checkpoint(records)["rollout_id"] == 49
    best["hf_export"] = "/run/hf-49"
    command = ev.final_eval_command(
        ["python", "--hf-checkpoint", "/base", "--eval-prompt-data", "a", "/subset"],
        best, {"a": "/subset"}, {"a": "/full"},
    )
    assert command == ["python", "--hf-checkpoint", "/run/hf-49", "--eval-prompt-data",
                       "a", "/full", "--final-eval-rollout", "49"]


def test_training_and_eval_limits_are_independent_even_when_overlapping():
    async def check():
        semaphore = ev.EvaluationSemaphore(asyncio.Semaphore(8), 16)
        clients = ev.EvaluationClient(SimpleNamespace(kind="train"), SimpleNamespace(kind="eval"))
        active = {False: 0, True: 0}
        peak = dict(active)
        release = asyncio.Event()
        async def task(evaluating):
            token = ev._EVALUATING.set(evaluating)
            try:
                async with semaphore:
                    assert clients.kind == ("eval" if evaluating else "train")
                    active[evaluating] += 1
                    peak[evaluating] = max(peak[evaluating], active[evaluating])
                    await release.wait()
                    active[evaluating] -= 1
            finally:
                ev._EVALUATING.reset(token)
        tasks = [asyncio.create_task(task(kind)) for kind in [False, True] for _ in range(32)]
        await asyncio.sleep(.01)
        assert active == {False: 8, True: 16}
        # Cancellation must release the correct pool while the other is active.
        tasks[0].cancel()
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert peak == {False: 8, True: 16}
        assert semaphore.training._value == 8
        assert semaphore.evaluation._value == 16
        assert clients.kind == "train"
    asyncio.run(check())


def test_active_experiment_has_two_exact_checkpoint_boundaries():
    config = load_experiment()
    args = SimpleNamespace(num_rollout=config["production"]["num_rollouts"],
                           processed_example_budget=config["production"]["processed_example_budget"],
                           rollout_batch_size=8, n_samples_per_prompt=1)
    assert [i for i in range(args.num_rollout) if _crosses_interval(args, i, 200)] == [24, 49]


@pytest.mark.parametrize("failure", [False, True])
def test_eval_finishes_or_fails_before_checkpoint_is_committed(monkeypatch, failure):
    import sys
    from ppo_async.training import driver
    calls = []
    def evaluate(rollout_id):
        calls.append("eval")
        if failure:
            raise RuntimeError("verifier failed")
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda x: x))
    monkeypatch.setattr(driver, "_event", lambda *a, **kw: None)
    monkeypatch.setattr(driver, "_commit_checkpoint", lambda *a: calls.append("commit"))
    args = SimpleNamespace(num_rollout=75, processed_example_budget=600,
                           rollout_batch_size=8, n_samples_per_prompt=1, evaluation_every_examples=200)
    manager = SimpleNamespace(eval=SimpleNamespace(remote=evaluate))
    if failure:
        with pytest.raises(RuntimeError, match="verifier failed"):
            driver._evaluate_if_due(args, 24, manager)
        assert calls == ["eval"]
    else:
        driver._evaluate_if_due(args, 24, manager)
        assert calls == ["eval", "commit"]


def test_final_inference_loads_only_selected_model(monkeypatch):
    import sys
    calls = []
    args = SimpleNamespace(hf_checkpoint="/run/hf-49", final_eval_rollout=49,
                           offload_rollout=True, colocate=False, num_gpus_per_node=2)
    manager = SimpleNamespace(
        eval=SimpleNamespace(remote=lambda i: calls.append(("eval", i))),
        dispose=SimpleNamespace(remote=lambda: calls.append(("dispose",))),
    )
    def create(args, pg):
        assert args.hf_checkpoint == "/run/hf-49"
        assert args.offload_rollout is False
        assert args.colocate and args.num_gpus_per_node == 1
        assert pg == "one-gpu"
        return manager, None
    def placement(count):
        assert count == 1
        return "one-gpu"
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda x: x))
    monkeypatch.setitem(sys.modules, "slime.observability.logging_utils", SimpleNamespace(
        configure_logger=lambda: None, init_tracking=lambda a: None, finish_tracking=lambda a: None))
    monkeypatch.setitem(sys.modules, "slime.ray.placement_group", SimpleNamespace(
        _create_placement_group=placement, create_rollout_manager=create))
    ev.evaluate_only(args)
    assert calls == [("eval", 49), ("dispose",)]


def test_completion_gate_rejects_final_eval_of_wrong_checkpoint(tmp_path, monkeypatch):
    from ppo_async.training import artifacts
    monkeypatch.setattr(artifacts, "validate_smoke_artifacts", lambda *a: {})
    records = []
    events = []
    for i, score in [(24, .3), (49, .2), (74, .1)]:
        records.append({"rollout_id": i, "processed_examples": (i + 1) * 8,
                        "datasets": {name: {"examples": 100, "pass_at_1": score}
                                     for name in ["gaokao-formal", "fate-m"]}})
        events.extend({"event": kind, "rollout_id": i} for kind in ["checkpoint", "evaluation"])
        for path in [tmp_path / f"hf-{i}/model.safetensors.index.json",
                     tmp_path / f"tracking/commits/rollout_{i}.json"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    def write(name, rows):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    write("evaluations.jsonl", records)
    write("events.jsonl", events)
    write("metrics.jsonl", [{"event": "scalar_metrics", "examples_end": i} for i in range(10, 601, 10)])
    (tmp_path / "tracking/state.json").write_text('{"next_example": 601, "pending": []}')
    (tmp_path / "best-checkpoint.json").write_text(json.dumps(ev.best_checkpoint(records)))
    final = {"rollout_id": 24, "datasets": {name: {"examples": count, "pass_at_1": .2}
                                            for name, count in [("gaokao-formal", 495), ("fate-m", 150)]}}
    write("final-evaluation.jsonl", [final])
    kwargs = dict(final_rollout_id=74, processed_example_budget=600, scalar_interval=10, action_interval=200)
    assert artifacts.validate_production_artifacts(tmp_path, **kwargs)["best_checkpoint"]["rollout_id"] == 24
    final["rollout_id"] = 74
    write("final-evaluation.jsonl", [final])
    with pytest.raises(RuntimeError, match="selected checkpoint"):
        artifacts.validate_production_artifacts(tmp_path, **kwargs)


@pytest.mark.parametrize("arm", ["sync_ppo", "async_ppo_dis"])
@pytest.mark.parametrize("fails", [False, True])
def test_final_ray_workers_inherit_final_log_path(tmp_path, monkeypatch, arm, fails):
    from contextlib import nullcontext
    import subprocess
    import modal_train
    from ppo_async.training.tracking import log_eval_scalars

    selection_log = tmp_path / "evaluations.jsonl"
    selection_log.write_text('{"checkpoint": 0}\n{"checkpoint": 1}\n')
    original = selection_log.read_bytes()
    environment = {"PPO_ASYNC_EVALUATION_LOG": str(selection_log), "PPO_ASYNC_BATCH_SIZE": "8"}
    command = ["python", "-m", "ppo_async.training.driver", "--final-eval-rollout", "0"]
    calls, commits = [], []
    worker_environment = {}

    def run(cmd, *, env, check, **kwargs):
        calls.append(cmd)
        assert env["PPO_ASYNC_EVALUATION_LOG"] == str(tmp_path / "final-evaluation.jsonl")
        if cmd[:2] == ["ray", "start"]:
            worker_environment.update(env)
        elif cmd == command:
            assert env == worker_environment
            if fails:
                raise subprocess.CalledProcessError(1, cmd)
            # Model Ray inheritance: the scoring worker sees the head's env.
            for key, value in worker_environment.items():
                monkeypatch.setenv(key, value)
            log_eval_scalars(0, SimpleNamespace(), {
                name: {"rewards": [0.0] * 32} for name in ["fate-m", "gaokao-formal"]
            }, None)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(modal_train.subprocess, "run", run)
    monkeypatch.setattr(modal_train, "_gpu_utilization", lambda root: nullcontext())
    monkeypatch.setattr(modal_train, "artifact_volume", SimpleNamespace(commit=lambda: commits.append(True)))
    if fails:
        with pytest.raises(subprocess.CalledProcessError):
            modal_train._run_final_evaluation(command, tmp_path, arm, environment)
    else:
        modal_train._run_final_evaluation(command, tmp_path, arm, environment)
        final = [json.loads(line) for line in (tmp_path / "final-evaluation.jsonl").read_text().splitlines()]
        assert len(final) == 1 and final[0]["rollout_id"] == 0
        assert sum(d["examples"] for d in final[0]["datasets"].values()) == 64
    assert selection_log.read_bytes() == original
    assert environment["PPO_ASYNC_EVALUATION_LOG"] == str(selection_log)
    assert calls[-1] == ["ray", "stop", "--force"]
    assert commits == [True]


def test_base_eval_uses_pinned_model_and_full_suites_without_training(tmp_path, monkeypatch):
    import modal_train
    from ppo_async.training.tracking import log_eval_scalars

    real_path = Path
    monkeypatch.setattr(modal_train, "Path", lambda p: tmp_path if p == "/vol/artifacts" else real_path(p))
    root = real_path(__file__).resolve().parents[1] / "prepared_data/balanced-400-v4"
    prepared = {"report": json.loads((root / "data-report.json").read_text()),
                "paths": {"train": root / "train-final.jsonl", **{
                    name: root / f"eval-{name}.jsonl" for name in ["gaokao-formal", "fate-m"]}}}
    monkeypatch.setattr(modal_train, "_prepared_production_data", lambda: prepared)
    monkeypatch.setattr(modal_train, "_hardware_inventory", lambda arm: {"requested": "H200:1"})
    monkeypatch.setattr(modal_train, "_download_model", lambda: real_path("/pinned-base"))
    monkeypatch.setattr(modal_train, "artifact_volume", SimpleNamespace(commit=lambda: None))
    calls = []
    def evaluate(command, run_root, arm, env):
        calls.append(command)
        assert arm == "sync_ppo"
        assert command[command.index("--hf-checkpoint") + 1] == "/pinned-base"
        assert command[-2:] == ["--final-eval-rollout", "-1"]
        assert command[command.index("--eval-concurrency") + 1] == "16"
        for name in ["gaokao-formal", "fate-m"]:
            assert str(prepared["paths"][name]) in command
        monkeypatch.setenv("PPO_ASYNC_EVALUATION_LOG", str(run_root / "final-evaluation.jsonl"))
        monkeypatch.setenv("PPO_ASYNC_BATCH_SIZE", "8")
        log_eval_scalars(-1, SimpleNamespace(), {
            name: {"rewards": [0.0] * count} for name, count in prepared["report"]["evaluation_rows"].items()
        }, None)
    monkeypatch.setattr(modal_train, "_run_final_evaluation", evaluate)
    result = modal_train._run_base_evaluation("base-test")
    assert result["status"] == "complete"
    assert result["evaluation"]["processed_examples"] == 0
    assert modal_train._run_base_evaluation("base-test") == result
    assert len(calls) == 1
