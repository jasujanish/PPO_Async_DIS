import json
import hashlib

from datasets import Dataset
import pytest

import modal_benchmark
from modal_benchmark import (
    DATASET_ID,
    DATASET_REVISION,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    extract_proof_expression,
    load_theorems,
    make_preflight_chunks,
    make_prompt,
    make_request_payload,
    render_preflight_chunk,
    run_paths,
    sampling_config,
    select_theorems,
    server_config,
    summarize_gpu_metrics,
    theorem_signature,
    validate_existing_records,
    wilson_interval,
)


STATEMENT = "theorem example : (1 : ℕ) = 1 := by sorry"


def test_theorem_signature_removes_only_placeholder() -> None:
    assert theorem_signature(STATEMENT) == "theorem example : (1 : ℕ) = 1"


def test_theorem_signature_rejects_unexpected_statement() -> None:
    with pytest.raises(ValueError):
        theorem_signature("theorem example : True := True.intro")


def test_prompt_does_not_leak_reference_proof() -> None:
    prompt = make_prompt(STATEMENT)
    assert "theorem example : (1 : ℕ) = 1" in prompt
    assert "by sorry" not in prompt


def test_prompt_preserves_dataset_version_and_lean_preamble() -> None:
    prompt = make_prompt(
        STATEMENT,
        "import Mathlib\nopen Set",
        "4.28.0",
        "v4.28.0",
    )

    assert "Lean 4.28.0 and Mathlib v4.28.0" in prompt
    assert "import Mathlib\nopen Set" in prompt
    assert "by sorry" not in prompt


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("by\n  norm_num", "by\n  norm_num"),
        ("```lean\nby\n  norm_num\n```", "by\n  norm_num"),
        ("`by exact True.intro`", "by exact True.intro"),
        ("theorem example : True := by trivial", "by trivial"),
        (":= by simp", "by simp"),
    ],
)
def test_extract_proof_expression(content: str, expected: str) -> None:
    assert extract_proof_expression(content) == expected


def test_unfinished_thinking_is_not_treated_as_a_submitted_proof() -> None:
    assert extract_proof_expression("<think>draft\n```lean\nby trivial\n```") == ""


def test_extract_full_theorem_with_internal_assignment() -> None:
    statement = "theorem example : (let x := 1; x) = 1 := by sorry"
    content = """```lean4
theorem example : (let x := 1; x) = 1 := by rfl
```"""

    assert extract_proof_expression(content, statement) == "by rfl"


def test_extract_full_theorem_tolerates_whitespace_changes() -> None:
    statement = "theorem example (n : ℕ) : n = n := by sorry"
    content = "theorem   example\n(n : ℕ) : n = n := by rfl"

    assert extract_proof_expression(content, statement) == "by rfl"


def test_qwen_precise_coding_sampling_config_is_explicit() -> None:
    assert MAX_NEW_TOKENS == 16_384
    assert MAX_MODEL_LEN == 32_768
    assert sampling_config() == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": 16_384,
        "seed": 42,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_request_payload_uses_the_audited_sampling_config() -> None:
    payload = make_request_payload(STATEMENT)

    assert payload["model"] == "Qwen/Qwen3.5-4B"
    assert payload["messages"][0]["content"] == make_prompt(STATEMENT)
    for key, value in sampling_config().items():
        assert payload[key] == value


def test_smoke_limit_selects_deterministic_prefix() -> None:
    theorems = [{"id": str(index)} for index in range(600)]

    assert select_theorems(theorems, 500) == theorems[:500]
    assert select_theorems(theorems, 0) is theorems
    with pytest.raises(ValueError, match="between 0 and 600"):
        select_theorems(theorems, 601)


def test_preflight_groups_exact_preambles_and_bounds_chunks(tmp_path) -> None:
    theorems = [
        {
            "id": f"same_{index}",
            "preamble": "import Mathlib",
            "formal_statement": f"theorem same_{index} : True := by sorry",
        }
        for index in range(5)
    ] + [
        {
            "id": "other",
            "preamble": "import Mathlib\nopen Set",
            "formal_statement": "theorem other : True := by sorry",
        }
    ]

    chunks = make_preflight_chunks(theorems, chunk_size=2)

    assert [len(chunk["theorems"]) for chunk in chunks] == [2, 2, 1, 1]
    assert len({chunk["preamble"] for chunk in chunks}) == 2
    source, ranges = render_preflight_chunk(chunks[0], tmp_path / "chunk.lean")
    assert source.count("import Mathlib") == 1
    assert source.count(": True := by sorry") == 2
    assert source.count("namespace OPSDPreflightStatement") == 2
    assert source.count("end OPSDPreflightStatement") == 2
    assert [theorem_id for _, _, theorem_id in ranges] == ["same_0", "same_1"]


def test_preflight_splits_duplicate_declaration_names() -> None:
    theorems = [
        {
            "id": f"row_{index}",
            "preamble": "import Mathlib",
            "formal_statement": "theorem duplicate : True := by sorry",
        }
        for index in range(2)
    ]

    chunks = make_preflight_chunks(theorems)

    assert [len(chunk["theorems"]) for chunk in chunks] == [1, 1]


def test_run_name_cannot_escape_results_volume() -> None:
    run_dir, raw_path, _, _ = run_paths("smoke-500")

    assert run_dir.name == "smoke-500"
    assert raw_path == run_dir / "raw.jsonl"
    with pytest.raises(ValueError, match="invalid run name"):
        run_paths("../other")


def test_sglang_server_config_is_explicit() -> None:
    assert server_config() == {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "context_length": 32_768,
        "memory_fraction_static": 0.9,
        "requested_max_running_requests": 256,
        "chunked_prefill_size": 8_192,
        "max_prefill_tokens": 32_768,
        "reasoning_parser": "qwen3",
        "random_seed": 42,
    }


def test_existing_records_must_match_index_and_sampling_config() -> None:
    theorem = {
        "id": "example",
        "formal_statement": STATEMENT,
        "natural_language_statement": "Example",
        "reference_steps": 1,
    }
    record = {"index": 0, **theorem, "request": sampling_config()}

    assert validate_existing_records([record], [theorem]) == {0}
    with pytest.raises(ValueError, match="duplicate"):
        validate_existing_records([record, record], [theorem])


def test_gpu_metric_summary(tmp_path) -> None:
    metrics = tmp_path / "gpu_metrics.csv"
    metrics.write_text(
        "timestamp,utilization_gpu_percent,power_draw_watts,memory_used_mib,temperature_c\n"
        "2026-01-01 00:00:00,90,600,70000,70\n"
        "2026-01-01 00:00:05,100,650,71000,72\n",
        encoding="utf-8",
    )

    summary = summarize_gpu_metrics(metrics)

    assert summary["samples"] == 2
    assert summary["gpu_utilization_percent"] == {
        "mean": 95.0,
        "median": 95.0,
        "max": 100.0,
    }


def test_wilson_interval_contains_observed_fraction() -> None:
    lower, upper = wilson_interval(250, 1000)

    assert lower < 0.25 < upper
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_load_theorems_validates_and_deduplicates_dataset(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset"
    Dataset.from_dict(
        {
            "id": ["one", "one", "two"],
            "status": ["proved", "proved", "proved"],
            "tactic": ["", "", ""],
            "state_before": ["⊢ True", "⊢ True", "⊢ True"],
            "state_after": ["⊢ True", "no goals", "no goals"],
            "natural_language_statement": ["One", "One", "Two"],
            "answer": ["", "", ""],
            "formal_statement": [
                "theorem one : True := by sorry",
                "theorem one : True := by sorry",
                "theorem two : True := by sorry",
            ],
        }
    ).save_to_disk(dataset_path)
    arrow_path = dataset_path / "data-00000-of-00001.arrow"
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "source_rows": 3,
        "proved_rows": 3,
        "filter": {"status": "proved"},
        "artifact_sha256": {
            arrow_path.name: hashlib.sha256(arrow_path.read_bytes()).hexdigest()
        },
    }
    (dataset_path / "opsd_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    config = dict(modal_benchmark.DATASET_CONFIGS["lean-workbook"])
    config.pop("source_normalizations")
    config.update(
        {
            "path": str(dataset_path),
            "source_rows": 3,
            "proved_rows": 3,
            "unique_theorems": 2,
            "artifact_sha256": metadata["artifact_sha256"],
        }
    )
    monkeypatch.setitem(modal_benchmark.DATASET_CONFIGS, "lean-workbook", config)

    theorems = load_theorems()

    assert [theorem["id"] for theorem in theorems] == ["one", "two"]
    assert [theorem["reference_steps"] for theorem in theorems] == [2, 1]
