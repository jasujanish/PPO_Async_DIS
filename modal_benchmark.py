"""Run a single-sample Qwen3.5-4B Lean-Workbook benchmark on Modal."""

from __future__ import annotations

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from typing import Any
from urllib.request import urlopen

import modal

APP_NAME = "opsd-qwen35-lean-workbook"
DATASET_ID = "internlm/Lean-Workbook"
DATASET_REVISION = "2e066e310b2c6d2c27616927ae131f82901c8f1c"
EXPECTED_SOURCE_ROWS = 25_214
EXPECTED_PROVED_ROWS = 18_985
EXPECTED_ARTIFACT_SHA256 = {
    "data-00000-of-00001.arrow": (
        "2b72aff0202205d9f2404dc0db228f78d1516b785010556553e5488b09cf34d9"
    )
}
CANONICAL_DATASET_COLUMNS = (
    "id",
    "status",
    "tactic",
    "state_before",
    "state_after",
    "natural_language_statement",
    "answer",
    "formal_statement",
)
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "lean-workbook": {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "path": "/benchmark-data/lean-workbook-proved",
        "source_rows": EXPECTED_SOURCE_ROWS,
        "proved_rows": EXPECTED_PROVED_ROWS,
        "unique_theorems": 10_434,
        "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
        "source_normalizations": {
            "qualified_unresolved_real_trig_identifiers": 1,
            "inserted_missing_proposition_colon": 1,
        },
        "lean_version": "4.8.0-rc1",
        "mathlib_version": "v4.8.0-rc1",
        "preamble_field": None,
        "lean_environment": "v4.8.0-rc1",
    },
    "fate-m": {
        "dataset_id": "frenzymath/FATE-M",
        "revision": "4eb33c8ccd0ff058b461cd763cc406509129743f",
        "path": "/benchmark-data/fate-m",
        "source_rows": 150,
        "proved_rows": 150,
        "unique_theorems": 150,
        "source_sha256": (
            "39c13a39f82bb2f39fc42d0a0efb41c6b0ef16b8194c8c8e0ae281c5cf0276fb"
        ),
        "artifact_sha256": {
            "data-00000-of-00001.arrow": (
                "a10cd6261939c2cdf786c9d98132f204c370c65db7c4fc043cc6b4991f79e6fe"
            )
        },
        "source_context": {
            "revision": "4eb33c8ccd0ff058b461cd763cc406509129743f",
            "official_exercise_files_with_directives": 22,
        },
        "lean_version": "4.28.0",
        "mathlib_version": "v4.28.0",
        "preamble_field": "state_before",
        "lean_environment": "v4.28.0",
    },
    "proofnet-verified": {
        "dataset_id": "marcusm117/ProofNet-Verified",
        "revision": "160414332dc196583f6c37c310b420d2a3b07c58",
        "path": "/benchmark-data/proofnet-verified",
        "source_rows": 367,
        "proved_rows": 367,
        "unique_theorems": 367,
        "source_sha256": (
            "381f4a06548a4ff6d9b923633c94a97b9c70f41033e13023aae31e1161b7f142"
        ),
        "artifact_sha256": {
            "data-00000-of-00001.arrow": (
                "6311757209d1dcf653fe5e2ebeb5fa49b09134566e6594216c749b1bc9ef7b05"
            )
        },
        "lean_version": "4.28.0",
        "mathlib_version": "v4.28.0",
        "preamble_field": "state_before",
        "lean_environment": "v4.28.0",
    },
    "gaokao-formal": {
        "dataset_id": "Huawei-AI4Math/Mathesis:Gaokao-Formal",
        "revision": "48c72753b7c7a7e0b07fe89bd97cf1a4899bab79",
        "path": "/benchmark-data/gaokao-formal",
        "source_rows": 495,
        "proved_rows": 495,
        "unique_theorems": 495,
        "source_sha256": (
            "89bd19c12f19f00d2f510598f765e911782028e0c0be0bccdf3cdc53081f4aaf"
        ),
        "artifact_sha256": {
            "data-00000-of-00001.arrow": (
                "160f19a2eb35cc63363f70c493b06da05a89f3556d68adf6d5968b638dd61f89"
            )
        },
        "source_normalizations": {
            "big_operator_ascii_in_to_membership": 144,
        },
        "lean_version": "4.27.0",
        "mathlib_version": "v4.27.0",
        "preamble_field": "state_before",
        "lean_environment": "v4.27.0",
    },
}

LEAN_WORKBOOK_STATEMENT_REPAIRS = {
    "lean_workbook_plus_56": {
        "original_sha256": (
            "3dc013f6e5642df3338bf76df883eda6c5c754a8ed23472473ab83b699f7729f"
        ),
        "statement": (
            "theorem lean_workbook_plus_56 : Real.sin (π / 4) = Real.cos (π / 4) ∧ "
            "Real.sin (π / 4) = 1 / Real.sqrt 2 ∧ Real.cos (π / 4) = "
            "1 / Real.sqrt 2 := by sorry"
        ),
    },
    "lean_workbook_plus_246": {
        "original_sha256": (
            "7474c77364febbfe909341148b6af6a7e9dcca0c042c77930b5c1e4c09926f09"
        ),
        "statement": (
            "theorem lean_workbook_plus_246 : "
            "(Nat.choose (4 + 218 - 1) 218) = 1774630 := by sorry"
        ),
    },
}
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SGLANG_VERSION = "0.5.18"
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.18-cu130-runtime"
RUN_NAME = "qwen35-4b-lean-workbook-thinking-16k-pass1"

GPU = "H100"
GENERATION_CPU = 8
GENERATION_MEMORY_MB = 65_536
MAX_RUNNING_REQUESTS = 256
CLIENT_CONCURRENCY = 512
MAX_MODEL_LEN = 32_768
DTYPE = "bfloat16"
TENSOR_PARALLEL_SIZE = 1
MEM_FRACTION_STATIC = 0.90
CHUNKED_PREFILL_SIZE = 8_192
MAX_PREFILL_TOKENS = 32_768
REASONING_PARSER = "qwen3"
MAX_NEW_TOKENS = 16_384
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 0.0
REPETITION_PENALTY = 1.0
ENABLE_THINKING = True
SEED = 42
REQUEST_TIMEOUT_SECONDS = 1800
# A cold Mathlib import is memory- and I/O-heavy. Running 16 independent Lean
# processes at once caused every process to miss a 30-second wall-clock limit on
# Modal. Four workers keeps the CPU verifier parallel without creating a cold
# import stampede; heartbeats remain the primary proof-complexity limit.
VERIFY_WORKERS = 4
VERIFY_MEMORY_MB = 32_768
LEAN_TIMEOUT_SECONDS = 180
LEAN_MAX_HEARTBEATS = 400_000
PREFLIGHT_CHUNK_SIZE = 500
PREFLIGHT_TIMEOUT_SECONDS = 600

RESULTS_ROOT = Path("/results")
app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name("opsd-benchmark-results", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("opsd-huggingface-cache", create_if_missing=True)

sglang_image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .pip_install("datasets==4.8.5")
    .add_local_dir(
        "data",
        remote_path="/benchmark-data",
        copy=True,
    )
    .env(
        {
            "HF_HOME": "/vol/huggingface",
            "HF_HUB_CACHE": "/vol/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

lean_image_v48 = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "curl", "git", "zstd")
    .pip_install("datasets==4.8.5")
    .run_commands(
        "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | "
        "sh -s -- -y --default-toolchain leanprover/lean4:v4.8.0-rc1"
    )
    .env({"PATH": "/root/.elan/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir("lean_env", remote_path="/lean-project", copy=True)
    .run_commands(
        "cd /lean-project && lake update",
        "cd /lean-project && lake exe cache get",
        # The cache contains Mathlib's dependencies but this old Mathlib release
        # does not provide the aggregate Mathlib.olean object. Build that root
        # module explicitly, then exercise the exact verifier import/tactic path.
        "cd /lean-project && lake build Mathlib",
        "cd /lean-project && lake env lean Smoke.lean",
    )
    # Dataset changes now invalidate only this final, inexpensive image layer.
    .add_local_dir("data", remote_path="/benchmark-data", copy=True)
)

lean_image_v428 = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "curl", "git", "zstd")
    .pip_install("datasets==4.8.5")
    .run_commands(
        "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | "
        "sh -s -- -y --default-toolchain leanprover/lean4:v4.28.0"
    )
    .env({"PATH": "/root/.elan/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir("lean_env_v428", remote_path="/lean-project", copy=True)
    .run_commands(
        "cd /lean-project && lake update",
        "cd /lean-project && lake exe cache get",
        "cd /lean-project && lake build Mathlib",
        "cd /lean-project && lake env lean Smoke.lean",
    )
    .add_local_dir("data", remote_path="/benchmark-data", copy=True)
)

lean_image_v427 = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "curl", "git", "zstd")
    .pip_install("datasets==4.8.5")
    .run_commands(
        "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | "
        "sh -s -- -y --default-toolchain leanprover/lean4:v4.27.0"
    )
    .env({"PATH": "/root/.elan/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir("lean_env_v427", remote_path="/lean-project", copy=True)
    .run_commands(
        "cd /lean-project && lake update",
        "cd /lean-project && lake exe cache get",
        "cd /lean-project && lake build Mathlib",
        "cd /lean-project && lake env lean Smoke.lean",
    )
    .add_local_dir("data", remote_path="/benchmark-data", copy=True)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_paths(run_name: str) -> tuple[Path, Path, Path, Path]:
    """Return isolated result paths after rejecting unsafe run names."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError(f"invalid run name: {run_name!r}")
    run_dir = RESULTS_ROOT / run_name
    return (
        run_dir,
        run_dir / "raw.jsonl",
        run_dir / "verified.jsonl",
        run_dir / "gpu_metrics.csv",
    )


def select_theorems(
    theorems: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Select a deterministic prefix; zero means the full benchmark."""
    if limit < 0 or limit > len(theorems):
        raise ValueError(f"limit must be between 0 and {len(theorems)}, got {limit}")
    return theorems if limit == 0 else theorems[:limit]


def theorem_signature(formal_statement: str) -> str:
    """Remove Lean-Workbook's placeholder proof from a formal statement."""
    signature, replacements = re.subn(
        r"\s*:=\s*by\s+sorry\s*$", "", formal_statement, count=1
    )
    if replacements != 1:
        raise ValueError("formal statement does not end in ':= by sorry'")
    return signature.strip()


def make_prompt(
    formal_statement: str,
    preamble: str = "",
    lean_version: str = "4.8.0-rc1",
    mathlib_version: str = "v4.8.0-rc1",
) -> str:
    signature = theorem_signature(formal_statement)
    context = f"\nLean context:\n{preamble.strip()}\n" if preamble.strip() else ""
    return f"""You are an expert theorem prover using Lean {lean_version} and Mathlib {mathlib_version}.
Complete the theorem below. Return only the Lean proof expression that belongs
after `:=`. A tactic proof beginning with `by` or a direct proof term is valid.
Do not restate the theorem or use Markdown fences. Do not use `sorry`, `admit`,
or introduce axioms.
{context}

{signature}
"""


def sampling_config() -> dict[str, Any]:
    """Return Qwen's thinking-mode profile for precise coding tasks."""
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": MIN_P,
        "presence_penalty": PRESENCE_PENALTY,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }


def server_config() -> dict[str, Any]:
    return {
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "dtype": DTYPE,
        "context_length": MAX_MODEL_LEN,
        "memory_fraction_static": MEM_FRACTION_STATIC,
        "requested_max_running_requests": MAX_RUNNING_REQUESTS,
        "chunked_prefill_size": CHUNKED_PREFILL_SIZE,
        "max_prefill_tokens": MAX_PREFILL_TOKENS,
        "reasoning_parser": REASONING_PARSER,
        "random_seed": SEED,
    }


def verifier_config(dataset_name: str = "lean-workbook") -> dict[str, Any]:
    dataset_config = DATASET_CONFIGS[dataset_name]
    return {
        "lean_version": dataset_config["lean_version"],
        "mathlib_version": dataset_config["mathlib_version"],
        "workers": VERIFY_WORKERS,
        "memory_mb": VERIFY_MEMORY_MB,
        "timeout_seconds": LEAN_TIMEOUT_SECONDS,
        "max_heartbeats": LEAN_MAX_HEARTBEATS,
    }


def validate_verification_config(
    config: dict[str, Any],
    theorems: list[dict[str, Any]],
    run_name: str,
    dataset_name: str,
) -> None:
    expected = base_run_config(
        theorems,
        run_name,
        int(config.get("benchmark_limit", 0)),
        dataset_name,
    )
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"verification config mismatch: {mismatches}")


def make_request_payload(
    formal_statement: str,
    preamble: str = "",
    lean_version: str = "4.8.0-rc1",
    mathlib_version: str = "v4.8.0-rc1",
) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": make_prompt(
                    formal_statement, preamble, lean_version, mathlib_version
                ),
            }
        ],
        **sampling_config(),
    }


def extract_proof_expression(content: str, formal_statement: str | None = None) -> str:
    """Extract a proof expression while accepting common Markdown wrapping."""
    text = content.strip()
    fenced = re.search(r"```(?:lean4?|Lean4?)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    elif text.startswith("`") and text.endswith("`"):
        text = text.strip("`").strip()

    if text.startswith("theorem ") and formal_statement is not None:
        signature = theorem_signature(formal_statement)
        flexible_signature = r"\s+".join(
            re.escape(fragment) for fragment in signature.split()
        )
        full_theorem = re.match(
            rf"{flexible_signature}\s*:=\s*(.*)\Z", text, re.DOTALL
        )
        if full_theorem:
            text = full_theorem.group(1).strip()
    elif text.startswith("theorem ") and ":=" in text:
        text = text.split(":=", 1)[1].strip()
    elif text.startswith(":="):
        text = text[2:].strip()

    return text


def load_theorems(dataset_name: str = "lean-workbook") -> list[dict[str, Any]]:
    from datasets import load_from_disk

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; choose from {sorted(DATASET_CONFIGS)}"
        )
    dataset_config = DATASET_CONFIGS[dataset_name]
    dataset_path = Path(dataset_config["path"])
    metadata_path = dataset_path / "opsd_metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"benchmark dataset has no provenance metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "dataset_id": dataset_config["dataset_id"],
        "dataset_revision": dataset_config["revision"],
        "source_rows": dataset_config["source_rows"],
        "proved_rows": dataset_config["proved_rows"],
        "filter": {"status": "proved"},
        "artifact_sha256": dataset_config["artifact_sha256"],
    }
    if "source_sha256" in dataset_config:
        expected_metadata["source_sha256"] = dataset_config["source_sha256"]
    if "source_normalizations" in dataset_config:
        expected_metadata["source_normalizations"] = dataset_config[
            "source_normalizations"
        ]
    if "source_context" in dataset_config:
        expected_metadata["source_context"] = dataset_config["source_context"]
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"benchmark dataset provenance mismatch: {mismatches}")

    for filename, expected_sha256 in metadata["artifact_sha256"].items():
        artifact_path = dataset_path / filename
        if not artifact_path.is_file():
            raise ValueError(f"benchmark dataset artifact is missing: {artifact_path}")
        actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"benchmark dataset checksum mismatch for {filename}: "
                f"{actual_sha256} != {expected_sha256}"
            )

    dataset = load_from_disk(str(dataset_path))
    if len(dataset) != dataset_config["proved_rows"]:
        raise ValueError(
            f"expected {dataset_config['proved_rows']} proved rows, found {len(dataset)}"
        )
    if tuple(dataset.column_names) != CANONICAL_DATASET_COLUMNS:
        raise ValueError(
            "benchmark schema differs from Lean-Workbook: "
            f"{dataset.column_names!r} != {list(CANONICAL_DATASET_COLUMNS)!r}"
        )

    steps_per_id: Counter[str] = Counter()
    terminal_rows_per_id: Counter[str] = Counter()
    canonical: dict[str, tuple[str, str, str]] = {}
    theorems: list[dict[str, Any]] = []
    for row in dataset:
        theorem_id = row["id"]
        formal_statement = row["formal_statement"]
        if (
            dataset_name == "lean-workbook"
            and theorem_id in LEAN_WORKBOOK_STATEMENT_REPAIRS
        ):
            repair = LEAN_WORKBOOK_STATEMENT_REPAIRS[theorem_id]
            original_sha256 = hashlib.sha256(formal_statement.encode()).hexdigest()
            if original_sha256 != repair["original_sha256"]:
                raise ValueError(
                    f"Lean-Workbook repair source mismatch for {theorem_id}: "
                    f"{original_sha256} != {repair['original_sha256']}"
                )
            formal_statement = repair["statement"]
        natural_statement = row["natural_language_statement"]
        preamble = (
            row[dataset_config["preamble_field"]]
            if dataset_config["preamble_field"]
            else ""
        )
        if row["status"] != "proved":
            raise ValueError(f"non-proved row survived filtering: {theorem_id}")
        if not theorem_id or not formal_statement or not natural_statement:
            raise ValueError(f"empty required field for theorem {theorem_id!r}")
        theorem_signature(formal_statement)

        steps_per_id[theorem_id] += 1
        if row["state_after"] == "no goals":
            terminal_rows_per_id[theorem_id] += 1

        statement_pair = (formal_statement, natural_statement, preamble)
        if theorem_id in canonical:
            if canonical[theorem_id] != statement_pair:
                raise ValueError(f"inconsistent statements for theorem {theorem_id}")
            continue
        canonical[theorem_id] = statement_pair
        theorems.append(
            {
                "id": theorem_id,
                "formal_statement": formal_statement,
                "natural_language_statement": natural_statement,
                "preamble": preamble,
            }
        )

    invalid_terminal_counts = {
        theorem_id: terminal_rows_per_id[theorem_id]
        for theorem_id in canonical
        if terminal_rows_per_id[theorem_id] != 1
    }
    if invalid_terminal_counts:
        sample = list(invalid_terminal_counts.items())[:10]
        raise ValueError(f"expected one terminal row per theorem; examples: {sample}")

    for theorem in theorems:
        theorem["reference_steps"] = steps_per_id[theorem["id"]]
    if len(theorems) != dataset_config["unique_theorems"]:
        raise ValueError(
            f"expected {dataset_config['unique_theorems']} unique theorems, "
            f"found {len(theorems)}"
        )
    return theorems


def wait_for_server(process: subprocess.Popen[Any], timeout_seconds: int = 1800) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = "http://127.0.0.1:30000/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited during startup with {process.returncode}")
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("SGLang did not become healthy within 30 minutes")


def get_server_info() -> dict[str, Any]:
    with urlopen("http://127.0.0.1:30000/get_server_info", timeout=10) as response:
        info = json.loads(response.read())
    if info.get("version") != SGLANG_VERSION:
        raise RuntimeError(
            f"SGLang version mismatch: {info.get('version')!r} != {SGLANG_VERSION!r}"
        )
    if info.get("context_length") != MAX_MODEL_LEN:
        raise RuntimeError(
            f"SGLang context mismatch: {info.get('context_length')!r} != {MAX_MODEL_LEN}"
        )
    return info


async def generate_all(
    indexed_theorems: list[tuple[int, dict[str, Any]]],
    *,
    total_theorems: int,
    initial_completed: int,
    raw_results_path: Path,
    dataset_name: str,
) -> dict[str, Any]:
    import aiohttp

    semaphore = asyncio.Semaphore(CLIENT_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    connector = aiohttp.TCPConnector(limit=CLIENT_CONCURRENCY)
    completed = 0
    api_errors = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    started = time.monotonic()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def generate_one(index: int, theorem: dict[str, Any]) -> dict[str, Any]:
            request_started = time.monotonic()
            dataset_config = DATASET_CONFIGS[dataset_name]
            payload = make_request_payload(
                theorem["formal_statement"],
                theorem["preamble"],
                dataset_config["lean_version"],
                dataset_config["mathlib_version"],
            )
            record: dict[str, Any] = {
                "index": index,
                **theorem,
                "request": sampling_config(),
            }
            async with semaphore:
                try:
                    async with session.post(
                        "http://127.0.0.1:30000/v1/chat/completions", json=payload
                    ) as response:
                        body = await response.text()
                        if response.status != 200:
                            record["error"] = f"HTTP {response.status}: {body[:2000]}"
                            return record
                        data = json.loads(body)
                    choice = data["choices"][0]
                    message = choice["message"]
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    return record

            record.update(
                {
                    "content": message.get("content") or "",
                    "reasoning_content": message.get("reasoning_content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage", {}),
                    "latency_seconds": time.monotonic() - request_started,
                }
            )
            return record

        tasks = [
            asyncio.create_task(generate_one(index, theorem))
            for index, theorem in indexed_theorems
        ]
        with raw_results_path.open("a", encoding="utf-8") as output:
            for future in asyncio.as_completed(tasks):
                record = await future
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed += 1
                if "error" in record:
                    api_errors += 1
                usage = record.get("usage", {})
                total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
                total_completion_tokens += usage.get("completion_tokens", 0) or 0
                overall_completed = initial_completed + completed
                if overall_completed % 100 == 0:
                    output.flush()
                    results_volume.commit()
                    elapsed = time.monotonic() - started
                    print(
                        f"completed={overall_completed}/{total_theorems} "
                        f"errors={api_errors} elapsed={elapsed:.1f}s"
                    )

            output.flush()

    return {
        "completed": completed,
        "api_errors": api_errors,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "generation_seconds": time.monotonic() - started,
    }


def base_run_config(
    theorems: list[dict[str, Any]],
    run_name: str = RUN_NAME,
    benchmark_limit: int = 0,
    dataset_name: str = "lean-workbook",
) -> dict[str, Any]:
    dataset_config = DATASET_CONFIGS[dataset_name]
    return {
        "app": APP_NAME,
        "run_name": run_name,
        "dataset_name": dataset_name,
        "dataset_id": dataset_config["dataset_id"],
        "dataset_revision": dataset_config["revision"],
        "source_normalizations": dataset_config.get("source_normalizations", {}),
        "dataset_rows": sum(theorem["reference_steps"] for theorem in theorems),
        "unique_theorems": len(theorems),
        "benchmark_limit": benchmark_limit,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sglang_version": SGLANG_VERSION,
        "sglang_image": SGLANG_IMAGE,
        "gpu": GPU,
        "generation_cpu": GENERATION_CPU,
        "generation_memory_mb": GENERATION_MEMORY_MB,
        "client_concurrency": CLIENT_CONCURRENCY,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "server": server_config(),
        "sampling": sampling_config(),
        "verifier": verifier_config(dataset_name),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def validate_existing_records(
    records: list[dict[str, Any]], theorems: list[dict[str, Any]]
) -> set[int]:
    completed_indices: set[int] = set()
    for record in records:
        index = record.get("index")
        if not isinstance(index, int) or not 0 <= index < len(theorems):
            raise ValueError(f"invalid result index: {index!r}")
        if index in completed_indices:
            raise ValueError(f"duplicate result index: {index}")
        theorem = theorems[index]
        if record.get("id") != theorem["id"]:
            raise ValueError(f"result/theorem ID mismatch at index {index}")
        if record.get("formal_statement") != theorem["formal_statement"]:
            raise ValueError(f"result/theorem statement mismatch at index {index}")
        if record.get("preamble", "") != theorem.get("preamble", ""):
            raise ValueError(f"result/theorem preamble mismatch at index {index}")
        if record.get("request") != sampling_config():
            raise ValueError(f"result sampling config mismatch at index {index}")
        completed_indices.add(index)
    return completed_indices


def start_gpu_metrics(
    gpu_metrics_path: Path,
) -> tuple[subprocess.Popen[Any] | None, Any | None, str | None]:
    """Sample GPU metrics with negligible overhead during model generation."""
    is_new_file = not gpu_metrics_path.exists() or gpu_metrics_path.stat().st_size == 0
    output = gpu_metrics_path.open("a", encoding="utf-8")
    if is_new_file:
        output.write(
            "timestamp,utilization_gpu_percent,power_draw_watts,"
            "memory_used_mib,temperature_c\n"
        )
        output.flush()
    try:
        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,power.draw,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
                "--loop=5",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        output.close()
        return None, None, f"{type(exc).__name__}: {exc}"
    return process, output, None


def stop_gpu_metrics(process: subprocess.Popen[Any] | None, output: Any | None) -> None:
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if output is not None:
        output.close()


def summarize_gpu_metrics(path: Path) -> dict[str, Any]:
    samples: list[tuple[float, float, float, float]] = []
    if path.exists():
        with path.open(encoding="utf-8") as source:
            next(source, None)
            for line in source:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 5:
                    continue
                try:
                    samples.append(tuple(float(value) for value in fields[1:]))
                except ValueError:
                    continue
    if not samples:
        return {"samples": 0}

    names = (
        "gpu_utilization_percent",
        "power_draw_watts",
        "memory_used_mib",
        "temperature_c",
    )
    summary: dict[str, Any] = {"samples": len(samples), "sample_interval_seconds": 5}
    for column, name in enumerate(names):
        values = [sample[column] for sample in samples]
        summary[name] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    return summary


@app.function(
    image=sglang_image,
    gpu=GPU,
    cpu=GENERATION_CPU,
    memory=GENERATION_MEMORY_MB,
    volumes={
        RESULTS_ROOT: results_volume,
        "/vol/huggingface": hf_cache_volume,
    },
    max_containers=1,
    retries=0,
    timeout=6 * 60 * 60,
    startup_timeout=60 * 60,
)
def generate(
    limit: int = 0,
    run_name: str = RUN_NAME,
    dataset_name: str = "lean-workbook",
) -> dict[str, Any]:
    """Generate one sample per theorem, resuming without repeating completed indices."""
    all_theorems = load_theorems(dataset_name)
    theorems = select_theorems(all_theorems, limit)
    run_dir, raw_results_path, _, gpu_metrics_path = run_paths(run_name)
    expected_config = base_run_config(theorems, run_name, limit, dataset_name)
    generation_marker = run_dir / "GENERATION_COMPLETE"
    canary_failure_marker = run_dir / "GENERATION_CANARY_FAILED"
    if generation_marker.exists():
        raise RuntimeError("generation is already complete; refusing to repeat it")
    if canary_failure_marker.exists():
        raise RuntimeError(
            "the request-schema canary previously failed; refusing to submit more requests"
        )

    config_path = run_dir / "config.json"
    if run_dir.exists():
        if not config_path.exists():
            raise RuntimeError(f"existing run directory has no config: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected_config.items()
            if config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"refusing to resume with a different config: {mismatches}")
    else:
        run_dir.mkdir(parents=True)
        config = {**expected_config, "started_at": utc_now(), "generation_attempts": []}

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    results_volume.commit()

    existing_records = read_jsonl(raw_results_path)
    completed_indices = validate_existing_records(existing_records, theorems)
    missing = [
        (index, theorem)
        for index, theorem in enumerate(theorems)
        if index not in completed_indices
    ]
    if completed_indices:
        print(
            f"Resuming with {len(completed_indices)}/{len(theorems)} existing results; "
            f"submitting only {len(missing)} missing requests"
        )

    attempt_metrics = {
        "completed": 0,
        "api_errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "generation_seconds": 0.0,
    }
    telemetry_error: str | None = None
    if missing:
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--tp-size",
            str(TENSOR_PARALLEL_SIZE),
            "--dtype",
            DTYPE,
            "--context-length",
            str(MAX_MODEL_LEN),
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
            "--max-running-requests",
            str(MAX_RUNNING_REQUESTS),
            "--chunked-prefill-size",
            str(CHUNKED_PREFILL_SIZE),
            "--max-prefill-tokens",
            str(MAX_PREFILL_TOKENS),
            "--reasoning-parser",
            REASONING_PARSER,
            "--random-seed",
            str(SEED),
        ]
        print("Starting SGLang:", " ".join(command))
        process = subprocess.Popen(command)
        gpu_metrics_process: subprocess.Popen[Any] | None = None
        gpu_metrics_output: Any | None = None
        try:
            wait_for_server(process)
            config["sglang_server_info"] = get_server_info()
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            results_volume.commit()
            gpu_metrics_process, gpu_metrics_output, telemetry_error = start_gpu_metrics(
                gpu_metrics_path
            )
            print("SGLang ready; validating the request schema with one benchmark sample")
            canary_metrics = asyncio.run(
                generate_all(
                    missing[:1],
                    total_theorems=len(theorems),
                    initial_completed=len(completed_indices),
                    raw_results_path=raw_results_path,
                    dataset_name=dataset_name,
                )
            )
            results_volume.commit()
            canary_record = next(
                record
                for record in read_jsonl(raw_results_path)
                if record["index"] == missing[0][0]
            )
            if "error" in canary_record:
                failure = canary_record["error"]
                canary_failure_marker.write_text(failure + "\n", encoding="utf-8")
                results_volume.commit()
                raise RuntimeError(f"generation canary failed; batch not submitted: {failure}")

            remaining = missing[1:]
            batch_metrics = {
                "completed": 0,
                "api_errors": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "generation_seconds": 0.0,
            }
            if remaining:
                print(f"Canary passed; submitting {len(remaining)} remaining requests")
                batch_metrics = asyncio.run(
                    generate_all(
                        remaining,
                        total_theorems=len(theorems),
                        initial_completed=len(completed_indices) + 1,
                        raw_results_path=raw_results_path,
                        dataset_name=dataset_name,
                    )
                )
            attempt_metrics = {
                key: canary_metrics[key] + batch_metrics[key]
                for key in canary_metrics
            }
        finally:
            stop_gpu_metrics(gpu_metrics_process, gpu_metrics_output)
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)

    final_records = read_jsonl(raw_results_path)
    final_indices = validate_existing_records(final_records, theorems)
    if len(final_indices) != len(theorems):
        raise RuntimeError(
            f"generation ended with {len(final_indices)}/{len(theorems)} results"
        )

    prior_seconds = float(config.get("generation_seconds", 0))
    config.update(
        {
            "completed": len(final_records),
            "api_errors": sum("error" in record for record in final_records),
            "prompt_tokens": sum(
                (record.get("usage") or {}).get("prompt_tokens", 0) or 0
                for record in final_records
            ),
            "completion_tokens": sum(
                (record.get("usage") or {}).get("completion_tokens", 0) or 0
                for record in final_records
            ),
            "generation_seconds": prior_seconds + attempt_metrics["generation_seconds"],
            "gpu_metrics": summarize_gpu_metrics(gpu_metrics_path),
        }
    )
    if telemetry_error:
        config["gpu_metrics_error"] = telemetry_error
    config.setdefault("generation_attempts", []).append(
        {
            "finished_at": utc_now(),
            "requests_submitted": len(missing),
            **attempt_metrics,
        }
    )
    config["finished_at"] = utc_now()
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    generation_marker.write_text(config["finished_at"] + "\n")
    results_volume.commit()
    return config


FORBIDDEN_PROOF_TOKEN = re.compile(r"\b(?:sorry|admit|axiom)\b", re.IGNORECASE)


def verify_one(record: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    result = dict(record)
    if record.get("error"):
        result.update({"verified": False, "verification_status": "generation_error"})
        return result

    proof = extract_proof_expression(
        record.get("content", ""), record.get("formal_statement")
    )
    result["extracted_proof"] = proof
    if not proof:
        result.update({"verified": False, "verification_status": "empty_proof"})
        return result
    if FORBIDDEN_PROOF_TOKEN.search(proof):
        result.update({"verified": False, "verification_status": "forbidden_token"})
        return result

    try:
        signature = theorem_signature(record["formal_statement"])
    except ValueError as exc:
        result.update(
            {
                "verified": False,
                "verification_status": "invalid_statement",
                "lean_output": str(exc),
            }
        )
        return result

    preamble = record.get("preamble", "").strip() or "import Mathlib"
    source = (
        f"{preamble}\n\n"
        f"set_option maxHeartbeats {LEAN_MAX_HEARTBEATS} in\n"
        f"{signature} := {proof}\n"
    )
    source_path = work_dir / f"candidate_{record['index']:05d}.lean"
    source_path.write_text(source, encoding="utf-8")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            ["lake", "env", "lean", str(source_path)],
            cwd="/lean-project",
            capture_output=True,
            text=True,
            timeout=LEAN_TIMEOUT_SECONDS,
        )
        lean_output = (completed.stdout + completed.stderr)[-8000:]
        verified = completed.returncode == 0
        status = "verified" if verified else "lean_error"
    except subprocess.TimeoutExpired as exc:
        verified = False
        status = "timeout"
        lean_output = str(exc)
    finally:
        source_path.unlink(missing_ok=True)

    result.update(
        {
            "verified": verified,
            "verification_status": status,
            "verification_seconds": time.monotonic() - started,
            "lean_output": lean_output,
        }
    )
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return (center - margin, center + margin)


def write_summary(
    records: list[dict[str, Any]], config: dict[str, Any], run_name: str, run_dir: Path
) -> dict[str, Any]:
    total = len(records)
    verified = sum(bool(record.get("verified")) for record in records)
    statuses = Counter(record.get("verification_status", "unknown") for record in records)
    finish_reasons = Counter(record.get("finish_reason", "missing") for record in records)
    generated = total - statuses.get("generation_error", 0)
    generation_errors = statuses.get("generation_error", 0)
    benchmark_valid = generation_errors == 0
    observed_success_fraction = verified / total if total else 0.0
    confidence_interval = wilson_interval(verified, total)
    generation_seconds = float(config.get("generation_seconds", 0))
    completion_tokens = int(config.get("completion_tokens", 0))
    summary = {
        "run_name": run_name,
        "dataset_name": config["dataset_name"],
        "dataset_id": config["dataset_id"],
        "dataset_revision": config["dataset_revision"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "gpu": GPU,
        "sglang_version": SGLANG_VERSION,
        "total_theorems": total,
        "generated_responses": generated,
        "verified_proofs": verified,
        "benchmark_valid": benchmark_valid,
        "pass_at_1": observed_success_fraction if benchmark_valid else None,
        "observed_success_fraction": observed_success_fraction,
        "pass_at_1_wilson_95_interval": list(confidence_interval),
        "verification_statuses": dict(statuses),
        "finish_reasons": dict(finish_reasons),
        "empty_proofs": statuses.get("empty_proof", 0),
        "empty_proof_rate": statuses.get("empty_proof", 0) / total if total else 0.0,
        "truncated_responses": finish_reasons.get("length", 0),
        "truncation_rate": finish_reasons.get("length", 0) / total if total else 0.0,
        "prompt_tokens": config.get("prompt_tokens", 0),
        "completion_tokens": completion_tokens,
        "generation_seconds": generation_seconds,
        "completion_tokens_per_second": (
            completion_tokens / generation_seconds if generation_seconds else 0.0
        ),
        "requested_max_running_requests": MAX_RUNNING_REQUESTS,
        "client_concurrency": CLIENT_CONCURRENCY,
        "max_model_len": MAX_MODEL_LEN,
        "sampling": sampling_config(),
        "gpu_metrics": config.get("gpu_metrics", {}),
        "finished_at": utc_now(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pass_at_1_text = (
        f"{observed_success_fraction:.2%}"
        if benchmark_valid
        else f"INVALID ({generation_errors} generation errors)"
    )
    markdown = f"""# Qwen3.5-4B on {config['dataset_name']}

- Theorems: {total:,}
- Generated responses: {generated:,}
- Lean-verified proofs: {verified:,}
- Benchmark valid: {benchmark_valid}
- Pass@1: {pass_at_1_text}
- Observed success fraction: {observed_success_fraction:.2%}
- Wilson 95% interval: [{confidence_interval[0]:.2%}, {confidence_interval[1]:.2%}]
- Completion tokens: {summary['completion_tokens']:,}
- Generation time: {summary['generation_seconds']:.1f} seconds
- Completion throughput: {summary['completion_tokens_per_second']:.1f} tokens/second
- Requested SGLang running-request limit: {MAX_RUNNING_REQUESTS}
- Client concurrency: {CLIENT_CONCURRENCY}
- Max model length: {MAX_MODEL_LEN:,}
- Max completion tokens: {MAX_NEW_TOKENS:,}
- Thinking mode: {ENABLE_THINKING}

## Verification statuses

```json
{json.dumps(dict(statuses), indent=2)}
```

## Finish reasons

```json
{json.dumps(dict(finish_reasons), indent=2)}
```
"""
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    return summary


def verify_impl(run_name: str, required_environment: str) -> dict[str, Any]:
    """Verify candidates with Lean, resuming without repeating completed indices."""
    run_dir, raw_results_path, verified_results_path, _ = run_paths(run_name)
    if not (run_dir / "GENERATION_COMPLETE").exists():
        raise RuntimeError("generation is not complete")
    verification_marker = run_dir / "VERIFICATION_COMPLETE"
    if verification_marker.exists():
        raise RuntimeError("verification is already complete; refusing to repeat it")

    # Fail once, before scheduling 10k candidates, if the Lean image is broken.
    smoke = subprocess.run(
        ["lake", "env", "lean", "Smoke.lean"],
        cwd="/lean-project",
        capture_output=True,
        text=True,
    )
    if smoke.returncode != 0:
        raise RuntimeError(
            "Lean verifier smoke test failed:\n" + smoke.stdout + smoke.stderr
        )

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dataset_name = config.get("dataset_name", "lean-workbook")
    dataset_config = DATASET_CONFIGS.get(dataset_name)
    if dataset_config is None:
        raise RuntimeError(f"unknown dataset in run config: {dataset_name!r}")
    if dataset_config["lean_environment"] != required_environment:
        raise RuntimeError(
            f"wrong verifier image for {dataset_name}: expected "
            f"{dataset_config['lean_environment']}, got {required_environment}"
        )
    all_theorems = load_theorems(dataset_name)
    theorems = select_theorems(all_theorems, int(config.get("benchmark_limit", 0)))
    validate_verification_config(config, theorems, run_name, dataset_name)
    raw_records = read_jsonl(raw_results_path)
    if len(raw_records) != config["unique_theorems"]:
        raise RuntimeError(
            f"expected {config['unique_theorems']} raw records, found {len(raw_records)}"
        )

    raw_by_index = {record["index"]: record for record in raw_records}
    if len(raw_by_index) != len(raw_records):
        raise RuntimeError("raw results contain duplicate indices")

    verified_records = read_jsonl(verified_results_path)
    completed_indices: set[int] = set()
    for record in verified_records:
        index = record.get("index")
        if index not in raw_by_index:
            raise RuntimeError(f"verified result has unknown index: {index!r}")
        if index in completed_indices:
            raise RuntimeError(f"duplicate verified result index: {index}")
        if record.get("id") != raw_by_index[index].get("id"):
            raise RuntimeError(f"verified/raw ID mismatch at index {index}")
        completed_indices.add(index)

    missing_records = [
        record for record in raw_records if record["index"] not in completed_indices
    ]
    if completed_indices:
        print(
            f"Resuming with {len(completed_indices)}/{len(raw_records)} verified records; "
            f"checking only {len(missing_records)} missing candidates"
        )

    with tempfile.TemporaryDirectory(prefix="opsd-lean-") as temporary_directory:
        work_dir = Path(temporary_directory)
        with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as executor:
            futures = [
                executor.submit(verify_one, record, work_dir)
                for record in missing_records
            ]
            with verified_results_path.open("a", encoding="utf-8") as output:
                for progress, future in enumerate(as_completed(futures), start=1):
                    record = future.result()
                    verified_records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    overall_progress = len(completed_indices) + progress
                    if overall_progress % 100 == 0:
                        output.flush()
                        results_volume.commit()
                        successes = sum(
                            item.get("verified", False) for item in verified_records
                        )
                        print(
                            f"verified={overall_progress}/{len(raw_records)} "
                            f"successes={successes}"
                        )
                output.flush()

    verified_records.sort(key=lambda record: record["index"])
    if len(verified_records) != len(raw_records):
        raise RuntimeError(
            f"verification ended with {len(verified_records)}/{len(raw_records)} records"
        )
    summary = write_summary(verified_records, config, run_name, run_dir)
    verification_marker.write_text(summary["finished_at"] + "\n")
    results_volume.commit()
    return summary


def theorem_declaration_name(formal_statement: str) -> str:
    """Return the declared theorem name used to avoid grouped-file collisions."""
    match = re.search(r"(?m)^theorem\s+([^\s({:]+)", formal_statement)
    if match is None:
        raise ValueError("formal statement has no parseable theorem declaration name")
    return match.group(1)


def make_preflight_chunks(
    theorems: list[dict[str, Any]], chunk_size: int = PREFLIGHT_CHUNK_SIZE
) -> list[dict[str, Any]]:
    """Group statements by exact preamble and make bounded collision-free chunks."""
    if chunk_size <= 0:
        raise ValueError("preflight chunk size must be positive")
    groups: dict[str, list[dict[str, Any]]] = {}
    for theorem in theorems:
        preamble = theorem["preamble"].strip() or "import Mathlib"
        groups.setdefault(preamble, []).append(theorem)

    chunks: list[dict[str, Any]] = []
    for preamble, group in groups.items():
        current: list[dict[str, Any]] = []
        current_names: set[str] = set()
        for theorem in group:
            declaration_name = theorem_declaration_name(theorem["formal_statement"])
            if len(current) >= chunk_size or declaration_name in current_names:
                chunks.append({"preamble": preamble, "theorems": current})
                current = []
                current_names = set()
            current.append(theorem)
            current_names.add(declaration_name)
        if current:
            chunks.append({"preamble": preamble, "theorems": current})
    return chunks


def render_preflight_chunk(
    chunk: dict[str, Any], source_path: Path
) -> tuple[str, list[tuple[int, int, str]]]:
    """Render a grouped source file and retain line ranges for error attribution."""
    lines = chunk["preamble"].splitlines() + [""]
    ranges: list[tuple[int, int, str]] = []
    for theorem_index, theorem in enumerate(chunk["theorems"]):
        # Elaborated declarations remain visible to later declarations. Give
        # each statement a namespace so theorem names such as `sin` cannot
        # shadow Mathlib identifiers in the next statement in the chunk.
        namespace = f"OPSDPreflightStatement{theorem_index}"
        lines.append(f"namespace {namespace}")
        statement_lines = theorem["formal_statement"].splitlines()
        start_line = len(lines) + 1
        lines.extend(statement_lines)
        ranges.append((start_line, len(lines), theorem["id"]))
        lines.append(f"end {namespace}")
        lines.append("")
    return "\n".join(lines), ranges


def preflight_impl(
    dataset_name: str, limit: int, required_environment: str
) -> dict[str, Any]:
    """Compile grouped placeholder statements before spending H100 time."""
    dataset_config = DATASET_CONFIGS[dataset_name]
    if dataset_config["lean_environment"] != required_environment:
        raise RuntimeError(
            f"wrong preflight image for {dataset_name}: expected "
            f"{dataset_config['lean_environment']}, got {required_environment}"
        )
    theorems = select_theorems(load_theorems(dataset_name), limit)

    chunks = make_preflight_chunks(theorems)

    def compile_one(
        chunk_index: int, chunk: dict[str, Any], work_dir: Path
    ) -> dict[str, Any]:
        source_path = work_dir / f"preflight_chunk_{chunk_index:04d}.lean"
        source, line_ranges = render_preflight_chunk(chunk, source_path)
        source_path.write_text(source, encoding="utf-8")
        theorem_ids = [theorem["id"] for theorem in chunk["theorems"]]
        try:
            completed = subprocess.run(
                ["lake", "env", "lean", str(source_path)],
                cwd="/lean-project",
                capture_output=True,
                text=True,
                timeout=PREFLIGHT_TIMEOUT_SECONDS,
            )
            output = completed.stdout + completed.stderr
            failed_ids: list[str] = []
            if completed.returncode != 0:
                # Do not treat the expected `declaration uses 'sorry'`
                # diagnostics as failures. Lean prints warnings with the same
                # path/line prefix as errors.
                error_lines = {
                    int(match.group(1))
                    for match in re.finditer(
                        rf"{re.escape(str(source_path))}:(\d+):\d+: error:",
                        output,
                    )
                }
                failed_ids = sorted(
                    {
                        theorem_id
                        for line in error_lines
                        for start, end, theorem_id in line_ranges
                        if start <= line <= end
                    }
                )
                if not failed_ids:
                    failed_ids = theorem_ids
            return {
                "ok": completed.returncode == 0,
                "checked": len(theorem_ids),
                "ids": failed_ids,
                "output": output[-16000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "checked": len(theorem_ids),
                "ids": theorem_ids,
                "output": str(exc),
            }
        finally:
            source_path.unlink(missing_ok=True)

    failures: list[dict[str, Any]] = []
    completed_statements = 0
    next_progress = 100
    started_at = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="opsd-preflight-") as temporary_directory:
        work_dir = Path(temporary_directory)
        with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as executor:
            futures = [
                executor.submit(compile_one, index, chunk, work_dir)
                for index, chunk in enumerate(chunks)
            ]
            for future in as_completed(futures):
                result = future.result()
                if not result["ok"]:
                    failures.append(result)
                completed_statements += result["checked"]
                while next_progress <= completed_statements:
                    print(
                        f"preflight={next_progress}/{len(theorems)} "
                        f"failed_chunks={len(failures)}"
                    )
                    next_progress += 100
    if completed_statements % 100:
        print(
            f"preflight={completed_statements}/{len(theorems)} "
            f"failed_chunks={len(failures)}"
        )
    if failures:
        raise RuntimeError(
            "statement preflight failed:\n" + json.dumps(failures[:10], indent=2)
        )
    return {
        "dataset_name": dataset_name,
        "checked_statements": len(theorems),
        "lean_environment": required_environment,
        "unique_preambles": len({chunk["preamble"] for chunk in chunks}),
        "compile_chunks": len(chunks),
        "chunk_size": PREFLIGHT_CHUNK_SIZE,
        "elapsed_seconds": time.monotonic() - started_at,
        "failures": 0,
    }


@app.function(
    image=lean_image_v48,
    cpu=VERIFY_WORKERS,
    memory=VERIFY_MEMORY_MB,
    max_containers=1,
    retries=0,
    timeout=6 * 60 * 60,
)
def preflight_v48(dataset_name: str, limit: int) -> dict[str, Any]:
    return preflight_impl(dataset_name, limit, "v4.8.0-rc1")


@app.function(
    image=lean_image_v427,
    cpu=VERIFY_WORKERS,
    memory=VERIFY_MEMORY_MB,
    max_containers=1,
    retries=0,
    timeout=6 * 60 * 60,
)
def preflight_v427(dataset_name: str, limit: int) -> dict[str, Any]:
    return preflight_impl(dataset_name, limit, "v4.27.0")


@app.function(
    image=lean_image_v428,
    cpu=VERIFY_WORKERS,
    memory=VERIFY_MEMORY_MB,
    max_containers=1,
    retries=0,
    timeout=6 * 60 * 60,
)
def preflight_v428(dataset_name: str, limit: int) -> dict[str, Any]:
    return preflight_impl(dataset_name, limit, "v4.28.0")


@app.function(
    image=lean_image_v48,
    cpu=VERIFY_WORKERS,
    memory=VERIFY_MEMORY_MB,
    volumes={RESULTS_ROOT: results_volume},
    max_containers=1,
    retries=0,
    timeout=6 * 60 * 60,
)
def verify(run_name: str = RUN_NAME) -> dict[str, Any]:
    return verify_impl(run_name, "v4.8.0-rc1")


@app.function(
    image=lean_image_v428,
    cpu=VERIFY_WORKERS,
    memory=VERIFY_MEMORY_MB,
    volumes={RESULTS_ROOT: results_volume},
    max_containers=1,
    retries=0,
    timeout=6 * 60 * 60,
)
def verify_v428(run_name: str) -> dict[str, Any]:
    return verify_impl(run_name, "v4.28.0")


@app.function(
    image=lean_image_v427,
    cpu=VERIFY_WORKERS,
    memory=VERIFY_MEMORY_MB,
    volumes={RESULTS_ROOT: results_volume},
    max_containers=1,
    retries=0,
    timeout=6 * 60 * 60,
)
def verify_v427(run_name: str) -> dict[str, Any]:
    return verify_impl(run_name, "v4.27.0")


@app.local_entrypoint()
def main(
    dataset_name: str = "lean-workbook",
    limit: int = 0,
    run_name: str = RUN_NAME,
    preflight_only: bool = False,
) -> None:
    """Run generation and verification exactly once, sequentially."""
    scope = "all tasks" if limit == 0 else f"the first {limit} tasks"
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; choose from {sorted(DATASET_CONFIGS)}"
        )
    print(f"Run {run_name!r}: evaluating {scope} from {dataset_name}")
    lean_environment = DATASET_CONFIGS[dataset_name]["lean_environment"]
    print("Compiling every selected benchmark statement before H100 generation...")
    if lean_environment == "v4.28.0":
        preflight = preflight_v428.remote(dataset_name, limit)
    elif lean_environment == "v4.27.0":
        preflight = preflight_v427.remote(dataset_name, limit)
    else:
        preflight = preflight_v48.remote(dataset_name, limit)
    print(json.dumps(preflight, indent=2))
    if preflight_only:
        print("Preflight-only run complete; no H100 generation was requested.")
        return
    print("Launching the single H100 generation run...")
    generation = generate.remote(limit, run_name, dataset_name)
    print(json.dumps(generation, indent=2))
    print("Generation finished. Launching one Lean verification pass on CPU...")
    if lean_environment == "v4.28.0":
        summary = verify_v428.remote(run_name)
    elif lean_environment == "v4.27.0":
        summary = verify_v427.remote(run_name)
    else:
        summary = verify.remote(run_name)
    print(json.dumps(summary, indent=2))
