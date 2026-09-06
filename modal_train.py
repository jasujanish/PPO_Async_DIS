"""Serialized Modal runs: one H200 for sync, two for async."""

from __future__ import annotations

from datetime import datetime, timezone
import copy
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import modal
from contextlib import contextmanager


APP_NAME = "ppo-async-qwen35-4b"
PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/workspace/PPO_ASYNC")
CONFIG_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "config/experiment.json").is_file() else REMOTE_ROOT
sys.path.insert(0, str(CONFIG_ROOT / "src"))

from ppo_async.config import arm_config, load_experiment, gpu_count, gpu_request, implementation_digest  # noqa: E402
from ppo_async.training.driver import role_slices  # noqa: E402


EXPERIMENT = load_experiment(CONFIG_ROOT / "config/experiment.json")
MODEL_ID = EXPERIMENT["model"]["id"]
MODEL_REVISION = EXPERIMENT["model"]["revision"]
SLIME_REVISION = EXPERIMENT["runtime"]["slime_revision"]
BASE_IMAGE = EXPERIMENT["runtime"]["base_container"]

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(BASE_IMAGE)
    .apt_install("ca-certificates", "curl", "git", "zstd")
    .run_commands(
        f"git -C /root/slime fetch --depth=1 origin {SLIME_REVISION}",
        f"git -C /root/slime checkout {SLIME_REVISION}",
        f"test $(git -C /root/slime rev-parse HEAD) = {SLIME_REVISION}",
        "pip install --no-deps -e /root/slime",
        "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | "
        "sh -s -- -y --default-toolchain leanprover/lean4:v4.28.0",
    )
    .env(
        {
            "PATH": "/root/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HOME": "/vol/model-cache/huggingface",
            "HF_HUB_CACHE": "/vol/model-cache/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_dir(PROJECT_ROOT / "lean_env", remote_path="/lean-projects/v48", copy=True)
    .add_local_dir(PROJECT_ROOT / "lean_env_v427", remote_path="/lean-projects/v427", copy=True)
    .add_local_dir(PROJECT_ROOT / "lean_env_v428", remote_path="/lean-projects/v428", copy=True)
    .run_commands(
        "cd /lean-projects/v48 && lake update && lake exe cache get && lake build Mathlib",
        "cd /lean-projects/v427 && lake update && lake exe cache get && lake build Mathlib",
        "cd /lean-projects/v428 && lake update && lake exe cache get && lake build Mathlib",
        "cd /lean-projects/v48 && lake env lean Smoke.lean",
        "cd /lean-projects/v427 && lake env lean Smoke.lean",
        "cd /lean-projects/v428 && lake env lean Smoke.lean",
    )
    .add_local_dir(PROJECT_ROOT / "src", remote_path=REMOTE_ROOT / "src", copy=True)
    .run_commands(f"PYTHONPATH={REMOTE_ROOT / 'src'} python3 -m ppo_async.training.runtime_patch")
    .add_local_dir(PROJECT_ROOT / "config", remote_path=REMOTE_ROOT / "config", copy=True)
    .add_local_dir(PROJECT_ROOT / "data", remote_path=REMOTE_ROOT / "data", copy=True)
    .add_local_dir(
        PROJECT_ROOT / "prepared_data" / EXPERIMENT["production"]["prepared_data_version"],
        remote_path=REMOTE_ROOT / "prepared_data" / EXPERIMENT["production"]["prepared_data_version"],
        copy=True,
    )
)

model_cache = modal.Volume.from_name("ppo-async-model-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name("ppo-async-artifacts", create_if_missing=True)


def _safe_run_name(run_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError(f"invalid run name: {run_name!r}")
    return run_name


def _hardware_inventory(arm: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != gpu_count(arm):
        raise RuntimeError(
            f"expected {gpu_count(arm)} visible CUDA GPUs, found {torch.cuda.device_count()}"
        )
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != gpu_count(arm) or any("H200" not in row for row in rows):
        raise RuntimeError(f"strict H200 hardware attestation failed: {rows}")
    return {
        "requested": gpu_request(arm),
        "cuda_device_count": torch.cuda.device_count(),
        "gpus": rows,
        "roles": role_slices(gpu_count(arm)),
    }


@contextmanager
def _gpu_utilization(run_root: Path):
    with (run_root / "gpu-utilization.csv").open("a") as output:
        monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits", "--loop=1"],
            stdout=output, stderr=subprocess.DEVNULL,
        )
        try:
            yield
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)


def _download_model() -> Path:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir="/vol/model-cache/huggingface",
    )
    model_cache.commit()
    return Path(snapshot)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_production_data() -> dict[str, Any]:
    root = REMOTE_ROOT / "prepared_data" / EXPERIMENT["production"]["prepared_data_version"]
    report = json.loads((root / "data-report.json").read_text(encoding="utf-8"))
    production = EXPERIMENT["production"]
    expected_counts = {
        "lean-workbook": production["lean_workbook_examples"],
        "proofnet-verified": production["proofnet_verified_examples"],
    }
    if report.get("mode") != "production" or report.get("training_by_source") != expected_counts:
        raise RuntimeError("embedded production data does not match the experiment contract")
    if report.get("training_examples") != sum(expected_counts.values()):
        raise RuntimeError("embedded production data has the wrong training size")
    if (
        report.get("selection", {}).get("passes_per_dataset") != production["passes_per_dataset"]
        or report.get("processed_example_budget") != production["processed_example_budget"]
    ):
        raise RuntimeError("embedded production data has the wrong pass/budget contract")
    paths = {
        "train": root / "train-final.jsonl",
        "gaokao-formal": root / "eval-gaokao-formal.jsonl",
        "fate-m": root / "eval-fate-m.jsonl",
    }
    for name, path in paths.items():
        expected = report["sha256"][name]
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"embedded production data hash mismatch: {name}")
    return {"report": report, "paths": paths}


def _ensure_run_contract(run_root: Path, arm: str, data_report: dict[str, Any], config=None) -> None:
    """Prevent resuming old data, token limits, or a different arm by run name."""
    expected = {"experiment": config or EXPERIMENT, "arm": arm, "data_sha256": data_report["sha256"]}
    path = run_root / "run-contract.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("run contract changed; use a fresh run name")
        return
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError("existing run has no compatible contract; use a fresh run name")
    run_root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tracker_rollout_id(path: Path) -> int | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _resume_boundary(run_root: Path, scalar_interval: int | None = None) -> int | None:
    """Restore the newest complete transaction and discard a failed tail."""
    actor_root = run_root / "checkpoints" / "actor"
    critic_root = run_root / "checkpoints" / "critic"
    actor_latest = _tracker_rollout_id(actor_root / "latest_checkpointed_iteration.txt")
    critic_latest = _tracker_rollout_id(critic_root / "latest_checkpointed_iteration.txt")
    common_latest = min(actor_latest, critic_latest) if actor_latest is not None and critic_latest is not None else None
    candidates: list[int] = []
    for marker in (run_root / "tracking" / "commits").glob("rollout_*.json"):
        match = re.fullmatch(r"rollout_(\d+)\.json", marker.name)
        if match and common_latest is not None and int(match.group(1)) <= common_latest:
            candidates.append(int(match.group(1)))

    boundary = max(candidates) if candidates else None
    if boundary is None:
        # There was no complete production transaction. All paths are scoped
        # to this generated run, so resetting them is safe and deterministic.
        import shutil

        for path in (
            actor_root,
            critic_root,
            run_root / "tracking",
            run_root / "events.jsonl",
            run_root / "metrics.jsonl",
            run_root / "evaluations.jsonl",
        ):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        for path in run_root.glob("hf-*"):
            if path.is_dir():
                shutil.rmtree(path)
        return None

    snapshot = run_root / "tracking" / "snapshots" / f"tracking_state_{boundary}.json"
    dataset_state = actor_root / "rollout" / f"global_dataset_state_dict_{boundary}.pt"
    hf_index = run_root / f"hf-{boundary}" / "model.safetensors.index.json"
    if not snapshot.is_file() or not dataset_state.is_file() or not hf_index.is_file():
        raise RuntimeError(f"transaction marker {boundary} is missing a required artifact")

    # A failed save may have advanced one tracker past the last transaction.
    # Point both roles back to the complete actor/critic/optimizer checkpoint.
    for root in (actor_root, critic_root):
        (root / "latest_checkpointed_iteration.txt").write_text(f"{boundary}\n", encoding="utf-8")
    tracking_state = run_root / "tracking" / "state.json"
    tracking_state.write_bytes(snapshot.read_bytes())

    processed = (boundary + 1) * int(EXPERIMENT["rollout"]["batch_size"])
    scalar_interval = scalar_interval or int(EXPERIMENT["production"]["scalar_log_every_examples"])
    metric_limit = processed // scalar_interval * scalar_interval
    _write_jsonl(
        run_root / "metrics.jsonl",
        [record for record in _read_jsonl(run_root / "metrics.jsonl") if int(record["examples_end"]) <= metric_limit],
    )
    _write_jsonl(
        run_root / "evaluations.jsonl",
        [record for record in _read_jsonl(run_root / "evaluations.jsonl") if int(record["rollout_id"]) <= boundary],
    )
    _write_jsonl(
        run_root / "events.jsonl",
        [
            record
            for record in _read_jsonl(run_root / "events.jsonl")
            if record.get("rollout_id") is None or int(record["rollout_id"]) <= boundary
        ],
    )
    return boundary


def _persist_and_compact_boundary(run_root: Path, rollout_id: int) -> None:
    """Commit the new checkpoint, then retain only its resumable train state."""
    import shutil

    artifact_volume.commit()
    keep = f"iter_{rollout_id:07d}"
    for role in ("actor", "critic"):
        root = run_root / "checkpoints" / role
        for path in root.glob("iter_*"):
            if path.is_dir() and re.fullmatch(r"iter_\d{7}", path.name) and path.name != keep:
                shutil.rmtree(path)
    rollout_root = run_root / "checkpoints" / "actor" / "rollout"
    for path in rollout_root.glob("global_dataset_state_dict_*.pt"):
        if path.name != f"global_dataset_state_dict_{rollout_id}.pt":
            path.unlink()
    artifact_volume.commit()


@app.function(image=image, cpu=1, memory=512, timeout=12 * 60 * 60, max_containers=1)
def train_smoke(arm="sync_ppo", training_examples=16, num_rollouts=1, run_name="smoke-sync-ppo-16-v3"):
    arm_config(EXPERIMENT, arm)
    worker = train_sync if arm == "sync_ppo" else train_async
    return worker.remote("smoke", arm, run_name, training_examples, num_rollouts)


def _run_smoke(
    arm: str = "sync_ppo",
    training_examples: int = 16,
    num_rollouts: int = 1,
    run_name: str = "smoke-sync-ppo-16-v1",
) -> dict[str, Any]:
    """Run a bounded GPU integration smoke test."""
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, "/root/slime")
    from ppo_async.data import materialize
    from ppo_async.training.launcher import (
        build_conversion_command,
        build_train_command,
        write_role_config,
    )
    from ppo_async.training.artifacts import validate_smoke_artifacts

    selected_arm = arm_config(EXPERIMENT, arm)
    maximum = int(EXPERIMENT["smoke"]["maximum_training_examples"])
    if not 0 < training_examples <= maximum < 100:
        raise ValueError(f"training_examples must be between 1 and {maximum}")
    if not 0 < num_rollouts <= 4:
        raise ValueError("smoke num_rollouts must be between 1 and 4")
    run_name = _safe_run_name(run_name)
    run_root = Path("/vol/artifacts") / run_name
    if (run_root / "RUN_COMPLETE.json").exists():
        raise RuntimeError(f"run is already complete: {run_name}")
    run_root.mkdir(parents=True, exist_ok=True)

    hardware = _hardware_inventory(arm)
    preparation = materialize(
        REMOTE_ROOT / "data",
        run_root / "prompts",
        training_examples=training_examples,
        seed=int(EXPERIMENT["rollout"]["seed"]),
    )
    artifact_volume.commit()

    hf_checkpoint = _download_model()
    converted = Path("/vol/model-cache/torch-dist") / MODEL_REVISION
    tracker = converted / "latest_checkpointed_iteration.txt"
    environment = {
        **os.environ,
        "PYTHONPATH": f"/root/Megatron-LM:/root/slime:{REMOTE_ROOT / 'src'}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "PYTHONUNBUFFERED": "1",
        "PPO_ASYNC_ARM": arm,
        "PPO_ASYNC_MAX_POLICY_LAG": str(selected_arm["max_policy_lag"]),
        "PPO_ASYNC_DIS_EPSILON_LOW": str(EXPERIMENT["ppo"]["dis_epsilon_low"]),
        "PPO_ASYNC_DIS_EPSILON_HIGH": str(EXPERIMENT["ppo"]["dis_epsilon_high"]),
        "PPO_ASYNC_EVENT_LOG": str(run_root / "events.jsonl"),
        "PPO_ASYNC_POLICY_VERSIONS": str(run_root / "tracking" / "policy-versions.json"),
    }
    if not tracker.is_file():
        converted.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            build_conversion_command(Path("/root/slime"), hf_checkpoint, converted),
            cwd="/root/slime",
            env=environment,
            check=True,
        )
        model_cache.commit()

    role_config = run_root / "roles.json"
    write_role_config(
        role_config,
        initial_checkpoint=converted,
        actor_save=run_root / "checkpoints" / "actor",
        critic_save=run_root / "checkpoints" / "critic",
        actor_lr=float(EXPERIMENT["ppo"]["actor_learning_rate"]),
        critic_lr=float(EXPERIMENT["ppo"]["critic_learning_rate"]),
    )
    command = build_train_command(
        arm,
        hf_checkpoint=hf_checkpoint,
        torch_dist_checkpoint=converted,
        prompt_data=Path(preparation["train_path"]),
        role_config=role_config,
        artifact_root=run_root,
        num_rollouts=num_rollouts,
        config=EXPERIMENT,
    )
    (run_root / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    artifact_volume.commit()

    subprocess.run(
        [
            "ray", "start", "--head", "--node-ip-address", "127.0.0.1",
            "--num-gpus", str(gpu_count(arm)), "--num-cpus", "32", "--disable-usage-stats",
            "--dashboard-host", "127.0.0.1", "--dashboard-port", "8265",
        ],
        env=environment,
        check=True,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    with _gpu_utilization(run_root):
        try:
            subprocess.run(command, cwd="/root/slime", env=environment, check=True)
        finally:
            subprocess.run(["ray", "stop", "--force"], env=environment, check=False)
            artifact_volume.commit()

    artifacts = validate_smoke_artifacts(run_root, num_rollouts - 1)

    report = {
        "status": "complete",
        "arm": arm,
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "training_examples": training_examples,
        "num_rollouts": num_rollouts,
        "data": preparation,
        "verified_artifacts": artifacts,
        "artifact_root": str(run_root),
    }
    (run_root / "RUN_COMPLETE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_volume.commit()
    return report


@app.function(
    image=image,
    cpu=8,
    memory=32_768,
    timeout=30 * 60,
    volumes={"/vol/model-cache": model_cache},
)
def preflight_final_commands() -> dict[str, Any]:
    """Run all pinned SLIME/Megatron argument validators without allocating GPUs."""
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, "/root/slime")
    from ppo_async.training.launcher import build_train_command, write_role_config

    prepared = _prepared_production_data()
    hf_checkpoint = _download_model()
    converted = Path("/vol/model-cache/torch-dist") / MODEL_REVISION
    if not (converted / "latest_checkpointed_iteration.txt").is_file():
        raise RuntimeError("converted Qwen checkpoint is missing")
    environment = {
        **os.environ,
        "PYTHONPATH": f"/root/Megatron-LM:/root/slime:{REMOTE_ROOT / 'src'}",
        "PPO_ASYNC_PARSE_ONLY": "1",
    }
    validated = []
    for arm in EXPERIMENT["arms"]:
        root = Path("/tmp/ppo-async-preflight") / arm
        role_config = root / "roles.json"
        write_role_config(
            role_config,
            initial_checkpoint=converted,
            actor_save=root / "actor",
            critic_save=root / "critic",
            actor_lr=float(EXPERIMENT["ppo"]["actor_learning_rate"]),
            critic_lr=float(EXPERIMENT["ppo"]["critic_learning_rate"]),
        )
        command = build_train_command(
            arm,
            hf_checkpoint=hf_checkpoint,
            torch_dist_checkpoint=converted,
            prompt_data=prepared["paths"]["train"],
            role_config=role_config,
            artifact_root=root,
            num_rollouts=int(EXPERIMENT["production"]["num_rollouts"]),
            config=EXPERIMENT,
            production=True,
            eval_paths={
                "gaokao-formal": str(prepared["paths"]["gaokao-formal"]),
                "fate-m": str(prepared["paths"]["fate-m"]),
            },
        )
        completed = subprocess.run(
            command,
            cwd="/root/slime",
            env={**environment, "PPO_ASYNC_ARM": arm},
            check=True,
            capture_output=True,
            text=True,
        )
        if "PPO_ASYNC_PARSE_OK" not in completed.stdout:
            raise RuntimeError(f"{arm} did not reach the parse-only completion gate")
        validated.append(arm)
    return {"status": "valid", "arms": validated, "implementation_sha256": implementation_digest()}


@app.function(image=image, cpu=1, memory=512,
              timeout=int(EXPERIMENT["production"]["timeout_seconds"]), max_containers=1, retries=2)
def train_final(arm: str, run_name: str):
    arm_config(EXPERIMENT, arm)
    worker = train_sync if arm == "sync_ppo" else train_async
    return worker.remote("final", arm, run_name)


def _run_final_evaluation(command, run_root: Path, arm: str, environment: dict[str, str]) -> None:
    # Ray workers inherit the head's environment, not just the driver subprocess's.
    final_environment = {**environment, "PPO_ASYNC_EVALUATION_LOG": str(run_root / "final-evaluation.jsonl")}
    try:
        subprocess.run(
            ["ray", "start", "--head", "--node-ip-address", "127.0.0.1",
             "--num-gpus", str(gpu_count(arm)), "--num-cpus", "32", "--disable-usage-stats"],
            env=final_environment, check=True,
        )
        with _gpu_utilization(run_root):
            subprocess.run(command, cwd="/root/slime", check=True, env=final_environment)
    finally:
        subprocess.run(["ray", "stop", "--force"], env=final_environment, check=False)
        artifact_volume.commit()


def _run_base_evaluation(run_name: str) -> dict[str, Any]:
    """Evaluate the pinned base model with the same final-evaluation path."""
    from ppo_async.training.launcher import build_train_command, write_role_config

    run_root = Path("/vol/artifacts") / _safe_run_name(run_name)
    prepared = _prepared_production_data()
    _ensure_run_contract(run_root, "base_model", prepared["report"])
    completion = run_root / "RUN_COMPLETE.json"
    if completion.is_file():
        return json.loads(completion.read_text())
    hardware = _hardware_inventory("sync_ppo")
    hf_checkpoint = _download_model()
    converted = Path("/vol/model-cache/torch-dist") / MODEL_REVISION
    roles = run_root / "roles.json"
    write_role_config(roles, initial_checkpoint=converted,
                      actor_save=run_root / "unused-actor", critic_save=run_root / "unused-critic",
                      actor_lr=EXPERIMENT["ppo"]["actor_learning_rate"],
                      critic_lr=EXPERIMENT["ppo"]["critic_learning_rate"])
    command = build_train_command(
        "sync_ppo", hf_checkpoint=hf_checkpoint, torch_dist_checkpoint=converted,
        prompt_data=prepared["paths"]["train"], role_config=roles, artifact_root=run_root,
        num_rollouts=EXPERIMENT["production"]["num_rollouts"], config=EXPERIMENT, production=True,
        eval_paths={name: str(prepared["paths"][name]) for name in ("gaokao-formal", "fate-m")},
    ) + ["--final-eval-rollout", "-1"]
    (run_root / "command-final-eval.json").write_text(json.dumps(command, indent=2) + "\n")
    (run_root / "final-evaluation.jsonl").unlink(missing_ok=True)
    environment = {
        **os.environ, "PYTHONPATH": f"/root/Megatron-LM:/root/slime:{REMOTE_ROOT / 'src'}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "1", "PYTHONUNBUFFERED": "1",
        "PPO_ASYNC_ARM": "sync_ppo", "PPO_ASYNC_BATCH_SIZE": str(EXPERIMENT["rollout"]["batch_size"]),
    }
    started = datetime.now(timezone.utc).isoformat()
    _run_final_evaluation(command, run_root, "sync_ppo", environment)
    results = _read_jsonl(run_root / "final-evaluation.jsonl")
    if len(results) != 1 or results[0]["rollout_id"] != -1 or {
        name: value["examples"] for name, value in results[0]["datasets"].items()
    } != prepared["report"]["evaluation_rows"]:
        raise RuntimeError("base evaluation must cover both complete benchmarks exactly once")
    report = {"status": "complete", "mode": "base-eval", "run_name": run_name,
              "model": EXPERIMENT["model"], "hardware": hardware, "evaluation": results[0],
              "data_sha256": prepared["report"]["sha256"], "started_at": started,
              "finished_at": datetime.now(timezone.utc).isoformat(), "artifact_root": str(run_root)}
    completion.write_text(json.dumps(report, indent=2) + "\n")
    artifact_volume.commit()
    return report


def _run_final(
    arm: str,
    run_name: str,
    *, integration_smoke: bool = False,
) -> dict[str, Any]:
    """Run the production path, optionally with a small integration-test budget."""
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, "/root/slime")
    from ppo_async.training.artifacts import validate_production_artifacts
    from ppo_async.training.launcher import (
        build_train_command,
        write_role_config,
    )

    experiment = copy.deepcopy(EXPERIMENT)
    if integration_smoke:
        experiment["production"].update(
            processed_example_budget=16, num_rollouts=2, scalar_log_every_examples=8,
            checkpoint_every_examples=8, evaluation_every_examples=8,
            lean_workbook_examples=8, proofnet_verified_examples=8,
        )
        experiment["evaluation"]["selection_problems_per_dataset"] = 16
    selected_arm = arm_config(experiment, arm)
    production = experiment["production"]
    run_name = _safe_run_name(run_name)
    run_root = Path("/vol/artifacts") / run_name
    completion_path = run_root / "RUN_COMPLETE.json"
    prepared = _prepared_production_data()
    _ensure_run_contract(run_root, arm, prepared["report"], experiment)
    if completion_path.is_file():
        return json.loads(completion_path.read_text(encoding="utf-8"))
    run_root.mkdir(parents=True, exist_ok=True)

    hardware = _hardware_inventory(arm)
    hf_checkpoint = _download_model()
    converted = Path("/vol/model-cache/torch-dist") / MODEL_REVISION
    if not (converted / "latest_checkpointed_iteration.txt").is_file():
        raise RuntimeError(
            "the validated Qwen3.5-4B torch-distributed checkpoint cache is missing; "
            "run a conversion preflight before concurrent production jobs"
        )

    from ppo_async.training.evaluation import selection_paths, best_checkpoint, final_eval_command

    full_eval_paths = {name: prepared["paths"][name] for name in ("gaokao-formal", "fate-m")}
    checkpoint_eval_paths = selection_paths(
        full_eval_paths, run_root / "selection-prompts",
        experiment["evaluation"]["selection_problems_per_dataset"], experiment["rollout"]["seed"],
    )
    if integration_smoke:
        smoke_train = run_root / "train-integration.jsonl"
        rows = _read_jsonl(prepared["paths"]["train"])
        selected_rows = [r for source in ("lean-workbook", "proofnet-verified")
                         for r in [r for r in rows if r["metadata"]["source_name"] == source][:8]]
        _write_jsonl(smoke_train, selected_rows)
        prepared["paths"]["train"] = smoke_train
        full_eval_paths = selection_paths(full_eval_paths, run_root / "final-smoke-prompts", 32, 42)
    boundary = _resume_boundary(run_root, production["scalar_log_every_examples"])
    actor_root = run_root / "checkpoints" / "actor"
    critic_root = run_root / "checkpoints" / "critic"
    role_config = run_root / "roles.json"
    write_role_config(
        role_config,
        initial_checkpoint=converted,
        actor_save=actor_root,
        critic_save=critic_root,
        actor_lr=float(experiment["ppo"]["actor_learning_rate"]),
        critic_lr=float(experiment["ppo"]["critic_learning_rate"]),
        actor_load=actor_root if boundary is not None else None,
        critic_load=critic_root if boundary is not None else None,
    )
    command = build_train_command(
        arm,
        hf_checkpoint=hf_checkpoint,
        torch_dist_checkpoint=converted,
        prompt_data=prepared["paths"]["train"],
        role_config=role_config,
        artifact_root=run_root,
        num_rollouts=int(production["num_rollouts"]),
        config=experiment,
        production=True,
        eval_paths=checkpoint_eval_paths,
        load_checkpoint=actor_root if boundary is not None else None,
    )
    attempt = len(list(run_root.glob("command-attempt-*.json"))) + 1
    (run_root / f"command-attempt-{attempt:03d}.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    (run_root / "data-report.json").write_text(
        json.dumps(prepared["report"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_volume.commit()

    environment = {
        **os.environ,
        "PYTHONPATH": f"/root/Megatron-LM:/root/slime:{REMOTE_ROOT / 'src'}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "PYTHONUNBUFFERED": "1",
        "PPO_ASYNC_ARM": arm,
        "PPO_ASYNC_MAX_POLICY_LAG": str(selected_arm["max_policy_lag"]),
        "PPO_ASYNC_DIS_EPSILON_LOW": str(experiment["ppo"]["dis_epsilon_low"]),
        "PPO_ASYNC_DIS_EPSILON_HIGH": str(experiment["ppo"]["dis_epsilon_high"]),
        "PPO_ASYNC_EVENT_LOG": str(run_root / "events.jsonl"),
        "PPO_ASYNC_POLICY_VERSIONS": str(run_root / "tracking" / "policy-versions.json"),
        "PPO_ASYNC_METRICS_LOG": str(run_root / "metrics.jsonl"),
        "PPO_ASYNC_EVALUATION_LOG": str(run_root / "evaluations.jsonl"),
        "PPO_ASYNC_TRACKING_STATE": str(run_root / "tracking" / "state.json"),
        "PPO_ASYNC_TRACKING_SNAPSHOTS": str(run_root / "tracking" / "snapshots"),
        "PPO_ASYNC_SCALAR_LOG_EVERY": str(production["scalar_log_every_examples"]),
        "PPO_ASYNC_BATCH_SIZE": str(experiment["rollout"]["batch_size"]),
    }
    subprocess.run(
        [
            "ray", "start", "--head", "--node-ip-address", "127.0.0.1",
            "--num-gpus", str(gpu_count(arm)), "--num-cpus", "32", "--disable-usage-stats",
            "--dashboard-host", "127.0.0.1", "--dashboard-port", "8265",
        ],
        env=environment,
        check=True,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    process: subprocess.Popen[str] | None = None
    with _gpu_utilization(run_root):
        try:
            process = subprocess.Popen(
                command,
                cwd="/root/slime",
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                marker = re.search(r"PPO_ASYNC_COMMIT (\{.*\})", line)
                if marker:
                    payload = json.loads(marker.group(1))
                    _persist_and_compact_boundary(run_root, int(payload["rollout_id"]))
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
            subprocess.run(["ray", "stop", "--force"], env=environment, check=False)
            artifact_volume.commit()

    best = best_checkpoint(_read_jsonl(run_root / "evaluations.jsonl"))
    best["hf_export"] = str(run_root / f"hf-{best['rollout_id']}")
    (run_root / "best-checkpoint.json").write_text(json.dumps(best, indent=2) + "\n")
    final_command = final_eval_command(command, best, checkpoint_eval_paths, full_eval_paths)
    (run_root / "command-final-eval.json").write_text(json.dumps(final_command, indent=2) + "\n")
    final_log = run_root / "final-evaluation.jsonl"
    final_log.unlink(missing_ok=True)
    artifact_volume.commit()
    _run_final_evaluation(final_command, run_root, arm, environment)

    artifacts = validate_production_artifacts(
        run_root,
        final_rollout_id=int(production["num_rollouts"]) - 1,
        processed_example_budget=int(production["processed_example_budget"]),
        scalar_interval=int(production["scalar_log_every_examples"]),
        action_interval=int(production["checkpoint_every_examples"]),
        selection_per_dataset=experiment["evaluation"]["selection_problems_per_dataset"],
        final_counts={"gaokao-formal": 32, "fate-m": 32} if integration_smoke else None,
    )
    report = {
        "status": "complete",
        "arm": arm,
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "resumed_after_rollout": boundary,
        "hardware": hardware,
        "mode": "integration-smoke" if integration_smoke else "final",
        "training_examples": production["processed_example_budget"],
        "processed_examples": production["processed_example_budget"],
        "passes_per_dataset": production["passes_per_dataset"],
        "data_sha256": prepared["report"]["sha256"],
        "verified_artifacts": artifacts,
        "artifact_root": str(run_root),
    }
    completion_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_volume.commit()
    return report


# Sharing the implementation keeps GPU allocation explicit without duplicating
# training, checkpointing or artifact code. Production calls remain serialized
# by train_final, including worker retries.
_worker_options = dict(
    image=image, cpu=32, memory=196_608,
    timeout=int(EXPERIMENT["production"]["timeout_seconds"]),
    startup_timeout=2 * 60 * 60, max_containers=1,
    volumes={"/vol/model-cache": model_cache, "/vol/artifacts": artifact_volume},
)


@app.function(gpu="H200:1", **_worker_options)
def train_sync(mode, arm, run_name, training_examples=16, num_rollouts=1):
    if mode == "base-eval":
        return _run_base_evaluation(run_name)
    if arm != "sync_ppo":
        raise ValueError("one-GPU worker requires sync_ppo")
    if mode in {"final", "integration-smoke"}:
        return _run_final(arm, run_name, integration_smoke=mode == "integration-smoke")
    return _run_smoke(arm, training_examples, num_rollouts, run_name)


@app.function(gpu="H200:2", **_worker_options)
def train_async(mode, arm, run_name, training_examples=16, num_rollouts=1):
    if arm not in {"async_ppo", "async_ppo_dis", "async_ppo_dis_masking"}:
        raise ValueError("two-GPU worker requires an async arm")
    if mode in {"final", "integration-smoke"}:
        return _run_final(arm, run_name, integration_smoke=mode == "integration-smoke")
    return _run_smoke(arm, training_examples, num_rollouts, run_name)


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    arm: str = "sync-ppo",
    training_examples: int = 16,
    num_rollouts: int = 1,
    run_name: str = "smoke-sync-ppo-16-v1",
) -> None:
    normalized_arm = arm.replace("-", "_")
    if mode == "final":
        if normalized_arm == "all":
            raise ValueError(
                "submit all persistent calls with launch_final.py after modal deploy"
            )
        arms = [normalized_arm]
        launches = []
        for selected in arms:
            selected_run_name = f"{run_name}-{selected.replace('_', '-')}" if len(arms) > 1 else run_name
            worker = train_sync if selected == "sync_ppo" else train_async
            call = worker.spawn("final", selected, selected_run_name)
            launches.append(
                {
                    "arm": selected,
                    "run_name": selected_run_name,
                    "function_call_id": call.object_id,
                }
            )
        print(
            json.dumps(
                {
                    "status": "launched",
                    "gpu_concurrency": "per run: sync=1 H200, async=2 H200",
                    "calls": launches,
                },
                indent=2,
            )
        )
    elif mode == "base-eval":
        print(json.dumps(train_sync.remote(mode, "sync_ppo", run_name), indent=2))
    elif mode == "integration-smoke":
        worker = train_sync if normalized_arm == "sync_ppo" else train_async
        print(json.dumps(worker.remote(mode, normalized_arm, run_name), indent=2))
    elif mode == "smoke":
        worker = train_sync if normalized_arm == "sync_ppo" else train_async
        call = worker.spawn("smoke", normalized_arm, run_name, training_examples, num_rollouts)
        print(json.dumps({"status": "launched", "run_name": run_name,
                          "function_call_id": call.object_id}, indent=2))
    else:
        raise ValueError("mode must be 'smoke', 'integration-smoke', 'base-eval', or 'final'")
