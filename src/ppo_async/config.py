"""Immutable experiment configuration and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARMS = ("sync_ppo", "async_ppo", "async_ppo_dis")
MODEL_ID = "Qwen/Qwen3.5-4B"
GPU_REQUEST = "H100!:4"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "config" / "experiment.json"


def load_experiment(path: Path | None = None) -> dict[str, Any]:
    config_path = path or default_config_path()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_experiment(config)
    return config


def validate_experiment(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("experiment schema_version must be 1")
    if config.get("model", {}).get("id") != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID}")

    hardware = config.get("hardware", {})
    if hardware.get("request") != GPU_REQUEST:
        raise ValueError(f"GPU request must be {GPU_REQUEST}")
    if hardware.get("physical_gpu_count") != 4:
        raise ValueError("experiment must allocate exactly four physical GPUs")
    placement = hardware.get("placement", {})
    expected = {"actor": [0], "critic": [1], "rollout": [2, 3]}
    if placement != expected:
        raise ValueError(f"GPU placement must be {expected}")
    assigned = [gpu for role in expected for gpu in placement[role]]
    if sorted(assigned) != list(range(4)) or len(set(assigned)) != 4:
        raise ValueError("actor, critic, and rollout GPU assignments must not overlap")
    if hardware.get("rollout_engines") != 2:
        raise ValueError("the two rollout GPUs must host two independent engines")
    if hardware.get("rollout_gpus_per_engine") != 1:
        raise ValueError("each SGLang rollout engine must use exactly one GPU")

    arms = config.get("arms", {})
    if tuple(arms) != ARMS:
        raise ValueError(f"arms must be ordered exactly as {ARMS}")
    if arms["sync_ppo"] != {
        "driver": "synchronous",
        "fully_async_rollout": False,
        "dis": False,
        "weight_publish_interval": 1,
        "max_policy_lag": 0,
    }:
        raise ValueError("sync_ppo contract changed")
    for name in ("async_ppo", "async_ppo_dis"):
        arm = arms[name]
        if arm["driver"] != "asynchronous" or not arm["fully_async_rollout"]:
            raise ValueError(f"{name} must use the asynchronous driver and fully-async rollout")
        if arm["max_policy_lag"] >= arm["weight_publish_interval"]:
            raise ValueError(f"{name} lag cap must be below its publish interval")
    if arms["async_ppo"]["dis"] or not arms["async_ppo_dis"]["dis"]:
        raise ValueError("DIS must be enabled only in async_ppo_dis")

    data = config.get("data", {})
    if data.get("training") != [
        "internlm/Lean-Workbook:proved-only",
        "marcusm117/ProofNet-Verified",
    ]:
        raise ValueError("training sources must be proved Lean-Workbook and ProofNet-Verified")
    if data.get("evaluation") != [
        "Huawei-AI4Math/Mathesis:Gaokao-Formal",
        "frenzymath/FATE-M",
    ]:
        raise ValueError("evaluation sources must be Gaokao-Formal and FATE-M")
    if data.get("proofs_in_prompt") is not False:
        raise ValueError("reference proofs must never appear in prompts")

    rollout = config.get("rollout", {})
    if rollout.get("batch_size") != 8 or rollout.get("global_batch_size") != 8:
        raise ValueError("production rollout and global batch sizes must both be 8")
    if rollout.get("max_response_tokens") != 16384:
        raise ValueError("training responses must allow 16384 tokens")
    if rollout.get("max_train_tokens_per_gpu") != 4096:
        raise ValueError("Qwen3.5-4B must use the 4096-token packing target (longer samples are packed alone)")
    if rollout.get("log_probs_chunk_size") != 1024:
        raise ValueError("Qwen3.5-4B must compute log probabilities in 1024-token chunks")

    production = config.get("production", {})
    expected_production = {
        "prepared_data_version": "balanced-700-v2",
        "lean_workbook_examples": 350,
        "proofnet_verified_examples": 350,
        "passes_per_dataset": 2,
        "processed_example_budget": 1400,
        "num_rollouts": 175,
        "scalar_log_every_examples": 10,
        "checkpoint_every_examples": 100,
        "evaluation_every_examples": 100,
        "timeout_seconds": 86400,
    }
    if production != expected_production:
        raise ValueError(f"production contract must be {expected_production}")
    if production["num_rollouts"] * rollout["batch_size"] != production["processed_example_budget"]:
        raise ValueError("production rollouts must process exactly the configured example budget")

    unique_examples = production["lean_workbook_examples"] + production["proofnet_verified_examples"]
    if unique_examples * production["passes_per_dataset"] != production["processed_example_budget"]:
        raise ValueError("production budget must equal unique examples times passes")

    evaluation = config.get("evaluation", {})
    if evaluation != {
        "samples_per_prompt": 1,
        "max_response_tokens": 16384,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }:
        raise ValueError("production evaluation sampling contract changed")

    smoke = config.get("smoke", {})
    count = smoke.get("training_examples")
    maximum = smoke.get("maximum_training_examples")
    if not isinstance(count, int) or not isinstance(maximum, int):
        raise ValueError("smoke example limits must be integers")
    if not 0 < count <= maximum < 100:
        raise ValueError("smoke training data must contain between 1 and 99 examples")


def arm_config(config: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    return config["arms"][arm]
