from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
import pytest
import torch

from ppo_async.config import load_experiment, validate_experiment
from ppo_async.data import make_prompt, materialize, normalized_statement
from ppo_async.lean import extract_proof_expression, verify_proof
from ppo_async.training.dis import apply_dis_mask, observe_importance
from ppo_async.training.driver import _crosses_interval, _processed_examples, role_slices
from ppo_async.training.hooks import (
    audit_policy_version,
    policy_lag,
    write_policy_versions,
    single_policy_version,
)
from ppo_async.training.launcher import build_train_command
from ppo_async.training.numerics import action_only_gae, direct_double_sided_mask
from ppo_async.training.tracking import log_rollout_scalars


ROOT = Path(__file__).resolve().parents[1]
STATEMENT = "theorem example : True := by sorry"


def test_experiment_uses_shared_training_gpu() -> None:
    config = load_experiment(ROOT / "config/experiment.json")
    assert config["hardware"]["sync_gpus"] == 1
    assert config["hardware"]["async_gpus"] == 2
    assert role_slices(1) == {"actor": (0,), "critic": (0,), "rollout": (0,)}
    assert role_slices(2) == {"actor": (0,), "critic": (0,), "rollout": (1,)}
    broken = json.loads(json.dumps(config))
    broken["hardware"]["async_gpus"] = 4
    with pytest.raises(ValueError, match="hardware"):
        validate_experiment(broken)


def test_prompt_contains_no_placeholder_or_reference_proof() -> None:
    prompt = make_prompt(STATEMENT, "import Mathlib", "4.28.0", "v4.28.0")
    assert "theorem example : True" in prompt
    assert "by sorry" not in prompt
    assert "import Mathlib" in prompt


def test_normalization_ignores_name_comments_and_formatting_only() -> None:
    left = "theorem one (n : Nat) : n = n := by sorry"
    right = "-- comment\n theorem two\n(n : Nat):n=n := sorry"
    near_miss = "theorem three (n : Nat) : n + 0 = n := by sorry"
    assert normalized_statement(left) == normalized_statement(right)
    assert normalized_statement(left) != normalized_statement(near_miss)


def _write_dataset(root: Path, name: str, rows: list[dict[str, str]]) -> None:
    path = root / name
    Dataset.from_list(rows).save_to_disk(path)
    (path / "opsd_metadata.json").write_text(
        json.dumps({"filter": {"status": "proved"}}), encoding="utf-8"
    )


def _row(theorem_id: str, statement: str, *, preamble: str = "import Mathlib") -> dict[str, str]:
    return {
        "id": theorem_id,
        "status": "proved",
        "tactic": "REFERENCE PROOF MUST NOT LEAK",
        "state_before": preamble,
        "state_after": "no goals",
        "natural_language_statement": theorem_id,
        "answer": "REFERENCE ANSWER MUST NOT LEAK",
        "formal_statement": statement,
    }


def test_materialize_balances_sources_and_filters_eval_leakage(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_dataset(
        data,
        "lean-workbook-proved",
        [
            _row("lw_overlap", "theorem train_name : True := by sorry"),
            _row("lw_a", "theorem lw_a : 1 = 1 := by sorry"),
            _row("lw_b", "theorem lw_b : 2 = 2 := by sorry"),
        ],
    )
    _write_dataset(
        data,
        "proofnet-verified",
        [
            _row("pn_a", "theorem pn_a : 3 = 3 := by sorry"),
            _row("pn_b", "theorem pn_b : 4 = 4 := by sorry"),
        ],
    )
    _write_dataset(
        data,
        "gaokao-formal",
        [_row("g", "theorem eval_name : True := by sorry")],
    )
    _write_dataset(
        data,
        "fate-m",
        [_row("f", "theorem fate : 5 = 5 := by sorry")],
    )

    report = materialize(data, tmp_path / "out", training_examples=4, seed=3)
    records = [json.loads(line) for line in Path(report["train_path"]).read_text().splitlines()]

    assert report["training_by_source"] == {"lean-workbook": 2, "proofnet-verified": 2}
    assert report["excluded"]["eval_exact_match"] == 1
    assert all("tactic" not in record and "answer" not in record for record in records)
    assert "REFERENCE" not in json.dumps(records)


def test_action_only_gae_ignores_observation_values() -> None:
    mask = torch.tensor([1, 0, 1])
    values = torch.tensor([0.2, 999.0, 0.4])
    advantages, returns = action_only_gae(
        values, 1.0, mask, gamma=1.0, gae_lambda=1.0
    )
    torch.testing.assert_close(advantages, torch.tensor([0.8, 0.0, 0.6]))
    torch.testing.assert_close(returns, torch.tensor([1.0, 0.0, 1.0]))


def test_dis_uses_open_asymmetric_region() -> None:
    behavior = torch.zeros(4)
    current = torch.log(torch.tensor([0.69, 0.70, 1.5, 6.0]))
    ratio, keep = direct_double_sided_mask(
        current,
        behavior,
        torch.ones(4),
        epsilon_low=0.3,
        epsilon_high=5.0,
    )
    torch.testing.assert_close(ratio, torch.tensor([0.69, 0.70, 1.5, 6.0]))
    assert keep.tolist() == [False, False, True, False]


def test_slime_dis_hook_masks_loss_but_observer_does_not(monkeypatch) -> None:
    monkeypatch.setenv("PPO_ASYNC_DIS_EPSILON_LOW", "0.3")
    monkeypatch.setenv("PPO_ASYNC_DIS_EPSILON_HIGH", "5.0")
    current = [torch.log(torch.tensor([0.5, 1.0, 7.0]))]
    behavior = [torch.zeros(3)]
    masks = [torch.ones(3)]
    pg_loss = torch.ones(3)

    observed, _, _ = observe_importance(
        None,
        pg_loss=pg_loss,
        train_log_probs=current,
        rollout_log_probs=behavior,
        loss_masks=masks,
    )
    masked, original_masks, metrics = apply_dis_mask(
        None,
        pg_loss=pg_loss,
        train_log_probs=current,
        rollout_log_probs=behavior,
        loss_masks=masks,
    )
    assert torch.equal(observed, pg_loss)
    assert masked.tolist() == [0.0, 1.0, 0.0]
    assert original_masks is masks
    assert metrics["dis_masked"].tolist() == [1.0, 0.0, 1.0]


def test_policy_versions_and_lag_are_audited(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    assert single_policy_version(["3", "3"]) == 3
    with pytest.raises(ValueError, match="exactly one"):
        single_policy_version(["3", "4"])
    assert policy_lag(12, 10) == 2
    monkeypatch.setenv("PPO_ASYNC_POLICY_VERSIONS", str(tmp_path / "versions.json"))
    monkeypatch.setenv("PPO_ASYNC_MAX_POLICY_LAG", "1")
    write_policy_versions({7: 10, 8: 12})
    args = SimpleNamespace(num_critic_only_steps=0)
    stale = SimpleNamespace(weight_versions=["7"], metadata={}, remove_sample=False)
    audit_policy_version(args, stale, rollout_id=12)
    assert stale.metadata["behavior_policy_version"] == 7
    assert stale.metadata["policy_lag_updates"] == 2
    assert stale.remove_sample
    fresh = SimpleNamespace(weight_versions=["8"], metadata=None, remove_sample=False)
    audit_policy_version(args, fresh, rollout_id=12)
    assert fresh.metadata["policy_lag_updates"] == 0
    assert not fresh.remove_sample
    unknown = SimpleNamespace(weight_versions=["99"], metadata={}, remove_sample=False)
    audit_policy_version(args, unknown, rollout_id=12)
    assert unknown.remove_sample
    assert "policy_version_error" in unknown.metadata


def test_launcher_selects_distinct_objectives() -> None:
    config = load_experiment(ROOT / "config/experiment.json")

    def command(arm: str) -> list[str]:
        return build_train_command(
            arm,
            hf_checkpoint=Path("/model/hf"),
            torch_dist_checkpoint=Path("/model/torch-dist"),
            prompt_data=Path("/data/train.jsonl"),
            role_config=Path("/tmp/roles.json"),
            artifact_root=Path("/artifacts/run"),
            num_rollouts=2,
            config=config,
        )

    sync = command("sync_ppo")
    asynchronous = command("async_ppo")
    dis = command("async_ppo_dis")
    direct = command("async_ppo_dis_masking")
    for built in (sync, asynchronous, dis, direct):
        assert built[:4] == ["python3", "-m", "ppo_async.training.driver", "--train-backend"]
        assert not any(argument.endswith("driver.py") for argument in built)
        assert built[built.index("--actor-num-gpus-per-node") + 1] == "1"
        assert built[built.index("--rollout-num-gpus") + 1] == "1"
        assert "--offload-train" in built
        assert "Qwen3.5-4B" not in " ".join(built)  # model is supplied by checkpoint path
        assert built[built.index("--rollout-max-response-len") + 1] == "16384"
        assert built[built.index("--max-tokens-per-gpu") + 1] == "4096"
        assert built[built.index("--log-probs-chunk-size") + 1] == "256"
        assert "--recompute-loss-function" in built
        assert "ppo_async.training.memory.initialize" in built
        assert built[built.index("--save-hf") + 1] == "/artifacts/run/hf-{rollout_id}"
        assert "--optimizer-cpu-offload" in built
        assert "--overlap-cpu-optimizer-d2h-h2d" in built
        assert "--use-precision-aware-optimizer" in built
    assert "--rollout-function-path" not in sync
    assert "ppo_async.training.stream.generate_rollout" in asynchronous
    assert "ppo_async.training.stream.StreamDataSource" in asynchronous
    assert "--sglang-enable-weights-cpu-backup" in sync
    assert "--sglang-enable-weights-cpu-backup" not in asynchronous
    assert "--sglang-enable-weights-cpu-backup" not in dis
    assert "--colocate" in sync
    assert "--colocate" not in asynchronous
    assert "--rollout-sample-hook-path" not in asynchronous
    assert "--use-rollout-logprobs" in sync
    assert "--use-rollout-logprobs" in asynchronous
    assert "--use-tis" not in asynchronous
    assert "--use-tis" in dis
    assert "--use-rollout-logprobs" not in dis
    assert "ppo_async.training.dis.apply_dis_mask" in dis
    assert "--use-rollout-logprobs" in direct
    assert "--use-tis" not in direct
    assert "--get-mismatch-metrics" not in direct  # no extra baseline forward
    assert direct[direct.index("--loss-type") + 1] == "custom_loss"
    assert "ppo_async.training.dis.direct_dis_policy_loss" in direct
    assert "ppo_async.training.stream.generate_rollout" in direct


def test_production_launcher_uses_exact_data_cadences_and_bounded_async() -> None:
    config = load_experiment(ROOT / "config/experiment.json")
    command = build_train_command(
        "async_ppo_dis",
        hf_checkpoint=Path("/model/hf"),
        torch_dist_checkpoint=Path("/model/torch-dist"),
        prompt_data=Path("/data/train-final.jsonl"),
        role_config=Path("/tmp/roles.json"),
        artifact_root=Path("/artifacts/final"),
        num_rollouts=50,
        config=config,
        production=True,
        eval_paths={
            "gaokao-formal": "/data/eval-gaokao-formal.jsonl",
            "fate-m": "/data/eval-fate-m.jsonl",
        },
        load_checkpoint=Path("/artifacts/final/checkpoints/actor"),
    )
    assert command[command.index("--rollout-batch-size") + 1] == "8"
    assert command[command.index("--global-batch-size") + 1] == "8"
    assert command[command.index("--processed-example-budget") + 1] == "400"
    assert command[command.index("--scalar-log-every-examples") + 1] == "10"
    assert command[command.index("--checkpoint-every-examples") + 1] == "200"
    assert command[command.index("--evaluation-every-examples") + 1] == "200"
    assert command[command.index("--eval-max-response-len") + 1] == "16384"
    assert command[command.index("--load") + 1] == "/artifacts/final/checkpoints/actor"
    assert "--eval-prompt-data" in command
    assert "ppo_async.training.evaluation.generate_rollout" in command
    assert "slime.rollout.fully_async_rollout.generate_rollout_fully_async" not in command
    assert command[command.index("--save-interval") + 1] == "51"
    assert "--ci-test" not in command
    assert "--no-save-optim" not in command


def test_batch_eight_actions_cross_each_hundred_example_threshold() -> None:
    from types import SimpleNamespace

    args = SimpleNamespace(
        rollout_batch_size=8, n_samples_per_prompt=1,
        processed_example_budget=1400, num_rollout=175,
    )
    due = [i for i in range(175) if _crosses_interval(args, i, 100)]
    assert len(due) == 14
    assert due[:4] == [12, 24, 37, 49]
    assert due[-1] == 174
    assert [_processed_examples(args, i) for i in due[:4]] == [104, 200, 304, 400]
    assert _processed_examples(args, due[-1]) == 1400


def test_scalar_tracking_writes_exact_ten_example_windows(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "tracking/state.json"
    metrics = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("PPO_ASYNC_TRACKING_STATE", str(state))
    monkeypatch.setenv("PPO_ASYNC_METRICS_LOG", str(metrics))
    monkeypatch.setenv("PPO_ASYNC_SCALAR_LOG_EVERY", "10")

    class Sample:
        metadata = {
            "source_name": "lean-workbook",
            "verification": {"status": "verified"},
            "policy_lag_updates": 0,
        }
        effective_response_length = 12
        response_length = 12
        status = "COMPLETED"
        remove_sample = False
        reward = 1.0

        @staticmethod
        def get_reward_value(_args):
            return 1.0

    assert log_rollout_scalars(0, object(), [Sample() for _ in range(8)], None, 1.0)
    assert not metrics.exists()
    assert log_rollout_scalars(1, object(), [Sample() for _ in range(8)], None, 1.0)
    records = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [(record["examples_start"], record["examples_end"]) for record in records] == [(1, 10)]
    tracking = json.loads(state.read_text())
    assert tracking["next_example"] == 11
    assert len(tracking["pending"]) == 6


def test_modal_entrypoints_request_only_needed_gpus() -> None:
    source = (ROOT / "modal_train.py").read_text(encoding="utf-8")
    assert 'gpu="H200:1"' in source and 'gpu="H200:2"' in source
    assert '"--num-gpus", str(gpu_count(arm))' in source
    assert "training_examples <= maximum < 100" in source


def test_remote_completion_gate_requires_all_checkpoints(tmp_path: Path) -> None:
    from ppo_async.training.artifacts import validate_smoke_artifacts

    actor = tmp_path / "checkpoints/actor/latest_checkpointed_iteration.txt"
    critic = tmp_path / "checkpoints/critic/latest_checkpointed_iteration.txt"
    export = tmp_path / "hf-0/model.safetensors.index.json"
    for path in (actor, critic, export):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    events = [
        {"event": "gpu_placement"},
        {"event": "learner_step", "rollout_id": 0},
        {"event": "checkpoint", "rollout_id": 0},
        {"event": "weight_publish", "learner_updates": 1},
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    verified = validate_smoke_artifacts(tmp_path, 0)
    assert verified["events"] == 4
    export.unlink()
    with pytest.raises(RuntimeError, match="hf_export"):
        validate_smoke_artifacts(tmp_path, 0)


def test_proof_extraction_and_missing_lean_are_fail_closed(tmp_path: Path) -> None:
    assert extract_proof_expression("<think>x</think>\n```lean\nby trivial\n```") == "by trivial"
    assert extract_proof_expression("<think>draft\n```lean\nby trivial\n```") == ""
    rejected = verify_proof(
        STATEMENT,
        "import Mathlib",
        "by sorry",
        "lean-4.28.0",
        projects={"lean-4.28.0": tmp_path},
    )
    assert rejected.status == "rejected"
    infrastructure = verify_proof(
        STATEMENT,
        "import Mathlib",
        "by trivial",
        "lean-4.28.0",
        projects={},
    )
    assert infrastructure.status == "infrastructure_error"
    assert infrastructure.reward is None
