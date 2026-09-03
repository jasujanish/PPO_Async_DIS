"""Construct pinned SLIME commands for all three experiment arms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ppo_async.config import arm_config, load_experiment


MODEL_ARGS = (
    "--spec", "slime_plugins.models.qwen3_5", "get_qwen3_5_spec",
    "--disable-bias-linear",
    "--qk-layernorm",
    "--group-query-attention",
    "--num-attention-heads", "16",
    "--num-query-groups", "4",
    "--kv-channels", "256",
    "--num-layers", "32",
    "--hidden-size", "2560",
    "--ffn-hidden-size", "9216",
    "--use-gated-attention",
    "--normalization", "RMSNorm",
    "--apply-layernorm-1p",
    "--position-embedding-type", "rope",
    "--norm-epsilon", "1e-6",
    "--rotary-percent", "0.25",
    "--swiglu",
    "--vocab-size", "248320",
    "--rotary-base", "10000000",
    "--attention-output-gate",
)


def build_conversion_command(
    slime_root: Path, hf_checkpoint: Path, torch_dist_checkpoint: Path
) -> list[str]:
    return [
        "python3",
        str(slime_root / "tools" / "convert_hf_to_torch_dist.py"),
        *MODEL_ARGS,
        "--hf-checkpoint",
        str(hf_checkpoint),
        "--save",
        str(torch_dist_checkpoint),
    ]


def write_role_config(
    path: Path,
    *,
    initial_checkpoint: Path,
    actor_save: Path,
    critic_save: Path,
    actor_lr: float,
    critic_lr: float,
    actor_load: Path | None = None,
    critic_load: Path | None = None,
) -> None:
    payload = {
        "megatron": [
            {
                "name": "ppo-async-actor",
                "role": "actor",
                "overrides": {
                    "load": str(actor_load or initial_checkpoint),
                    "save": str(actor_save),
                    "lr": actor_lr,
                },
            },
            {
                "name": "ppo-async-critic",
                "role": "critic",
                "overrides": {
                    "load": str(critic_load or initial_checkpoint),
                    "save": str(critic_save),
                    "lr": critic_lr,
                    "disable_param_buffers_cpu_backup": False,
                },
            },
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_train_command(
    arm: str,
    *,
    hf_checkpoint: Path,
    torch_dist_checkpoint: Path,
    prompt_data: Path,
    role_config: Path,
    artifact_root: Path,
    num_rollouts: int,
    config: dict[str, Any] | None = None,
    production: bool = False,
    eval_paths: dict[str, str] | None = None,
    load_checkpoint: Path | None = None,
) -> list[str]:
    experiment = load_experiment() if config is None else config
    selected_arm = arm_config(experiment, arm)
    ppo = experiment["ppo"]
    rollout = experiment["rollout"]
    if num_rollouts < 1:
        raise ValueError("num_rollouts must be positive")

    actor_save = artifact_root / "checkpoints" / "actor"
    command = [
        # Module execution is required here. Executing training/driver.py as a
        # file puts its directory on sys.path, where our PPO-DIS hook (dis.py)
        # would shadow Python's standard-library dis module.
        "python3", "-m", "ppo_async.training.driver",
        "--train-backend", "megatron",
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", "1",
        "--rollout-num-gpus", "2",
        "--rollout-num-gpus-per-engine", "1",
        *MODEL_ARGS,
        "--hf-checkpoint", str(hf_checkpoint),
        "--ref-load", str(torch_dist_checkpoint),
        "--save", str(actor_save),
        # SLIME formats this path with ``format(rollout_id=...)``.  Keep the
        # named field literal: an anonymous ``{}`` raises IndexError only after
        # the otherwise-successful learner step and distributed checkpoint.
        "--save-hf", str(artifact_root / "hf-{rollout_id}"),
        "--megatron-config-path", str(role_config),
        "--prompt-data", str(prompt_data),
        "--input-key", "input",
        "--metadata-key", "metadata",
        "--apply-chat-template",
        "--apply-chat-template-kwargs", '{"enable_thinking": true}',
        "--rollout-shuffle",
        "--custom-rm-path", "ppo_async.training.reward.reward",
        "--rollout-sample-hook-path", "ppo_async.training.hooks.audit_policy_version",
        "--num-rollout", str(num_rollouts),
        "--rollout-batch-size", str(rollout["batch_size"]),
        "--n-samples-per-prompt", str(rollout["samples_per_prompt"]),
        "--global-batch-size", str(rollout["global_batch_size"]),
        "--num-steps-per-rollout", "1",
        "--rollout-max-response-len", str(rollout["max_response_tokens"]),
        "--rollout-max-prompt-len", str(rollout["max_prompt_tokens"]),
        "--rollout-temperature", str(rollout["temperature"]),
        "--rollout-top-p", str(rollout["top_p"]),
        "--rollout-top-k", str(rollout["top_k"]),
        "--rollout-seed", str(rollout["seed"]),
        "--seed", str(rollout["seed"]),
        "--rollout-skip-special-tokens",
        "--advantage-estimator", "ppo",
        "--custom-advantage-function-path", "ppo_async.training.advantages.compute_action_only_gae",
        "--gamma", str(ppo["gamma"]),
        "--lambd", str(ppo["lambda"]),
        "--normalize-advantages",
        "--eps-clip", str(ppo["clip_epsilon"]),
        "--value-clip", str(ppo["value_clip"]),
        "--entropy-coef", str(ppo["entropy_coefficient"]),
        "--kl-coef", "0.0",
        "--optimizer", "adam",
        # Qwen3.5's gated-delta-rule Triton backward autotuner otherwise peaks
        # above 79 GiB with full GPU-resident Adam states on an 80 GiB H100.
        # These are the optimizer settings used by SLIME's Qwen3.5 example.
        "--optimizer-cpu-offload",
        "--overlap-cpu-optimizer-d2h-h2d",
        "--use-precision-aware-optimizer",
        "--lr", str(ppo["actor_learning_rate"]),
        "--lr-decay-style", "constant",
        "--weight-decay", str(ppo["weight_decay"]),
        "--adam-beta1", str(ppo["adam_beta1"]),
        "--adam-beta2", str(ppo["adam_beta2"]),
        "--clip-grad", str(ppo["gradient_clip"]),
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1",
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(rollout["max_train_tokens_per_gpu"]),
        "--log-probs-chunk-size", str(rollout["log_probs_chunk_size"]),
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--loss-mask-type", "qwen3_5",
        "--bf16",
        "--sglang-mem-fraction-static", "0.75",
        "--sglang-server-concurrency", str(rollout["server_concurrency_per_engine"]),
        "--sglang-max-running-requests", str(2 * rollout["server_concurrency_per_engine"]),
        "--sglang-cuda-graph-max-bs", "16",
        "--sglang-enable-metrics",
        "--update-weights-interval", str(selected_arm["weight_publish_interval"]),
    ]
    if load_checkpoint is not None:
        command.extend(["--load", str(load_checkpoint)])
    if production:
        production_config = experiment["production"]
        evaluation = experiment["evaluation"]
        if eval_paths is None or set(eval_paths) != {"gaokao-formal", "fate-m"}:
            raise ValueError("production requires Gaokao-Formal and FATE-M evaluation paths")
        command.extend(
            [
                # Megatron requires a non-null save_interval whenever --save
                # is present. The production driver ignores this rollout-based
                # sentinel and saves only on its example-counted boundaries.
                "--save-interval", str(num_rollouts + 1),
                "--processed-example-budget", str(production_config["processed_example_budget"]),
                "--scalar-log-every-examples", str(production_config["scalar_log_every_examples"]),
                "--checkpoint-every-examples", str(production_config["checkpoint_every_examples"]),
                "--evaluation-every-examples", str(production_config["evaluation_every_examples"]),
                "--custom-rollout-log-function-path",
                "ppo_async.training.tracking.log_rollout_scalars",
                "--custom-eval-rollout-log-function-path",
                "ppo_async.training.tracking.log_eval_scalars",
                "--eval-function-path", "slime.rollout.sglang_rollout.generate_rollout",
                "--eval-prompt-data",
                "gaokao-formal", str(eval_paths["gaokao-formal"]),
                "fate-m", str(eval_paths["fate-m"]),
                "--skip-eval-before-train",
                "--n-samples-per-eval-prompt", str(evaluation["samples_per_prompt"]),
                "--eval-max-response-len", str(evaluation["max_response_tokens"]),
                "--eval-temperature", str(evaluation["temperature"]),
                "--eval-top-p", str(evaluation["top_p"]),
                "--eval-top-k", str(evaluation["top_k"]),
            ]
        )
    else:
        command.extend(
            [
                "--save-interval", "1",
                "--dump-details", str(artifact_root / "details"),
                "--ci-test",
                "--ci-disable-kl-checker",
            ]
        )
    # Production async runs use the driver's one-batch lookahead. Unlike
    # SLIME's continuously-prefetching worker, this gives checkpoint-aligned
    # dataset cursors and leaves the rollout engines quiescent for evaluation.
    if selected_arm["fully_async_rollout"] and not production:
        command.extend(
            [
                "--rollout-function-path",
                "slime.rollout.fully_async_rollout.generate_rollout_fully_async",
            ]
        )
    if selected_arm["dis"]:
        # SLIME's TIS path recomputes the learner-policy log-probabilities and
        # compares them with rollout_log_probs.  Its validator deliberately
        # rejects --use-rollout-logprobs together with --use-tis because that
        # would replace the PPO old-policy baseline with the behavior policy.
        command.extend(
            [
                "--use-tis",
                "--custom-tis-function-path",
                "ppo_async.training.dis.apply_dis_mask",
            ]
        )
    else:
        command.extend(
            [
                "--use-rollout-logprobs",
                "--get-mismatch-metrics",
                "--custom-tis-function-path",
                "ppo_async.training.dis.observe_importance",
            ]
        )
    return command
