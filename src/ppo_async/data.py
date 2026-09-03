"""Build leakage-filtered SLIME prompt files for training and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from datasets import load_from_disk


LEAN_WORKBOOK_REPAIRS = {
    "lean_workbook_plus_56": (
        "3dc013f6e5642df3338bf76df883eda6c5c754a8ed23472473ab83b699f7729f",
        "theorem lean_workbook_plus_56 : Real.sin (π / 4) = Real.cos (π / 4) ∧ "
        "Real.sin (π / 4) = 1 / Real.sqrt 2 ∧ Real.cos (π / 4) = "
        "1 / Real.sqrt 2 := by sorry",
    ),
    "lean_workbook_plus_246": (
        "7474c77364febbfe909341148b6af6a7e9dcca0c042c77930b5c1e4c09926f09",
        "theorem lean_workbook_plus_246 : "
        "(Nat.choose (4 + 218 - 1) 218) = 1774630 := by sorry",
    ),
}

DATASETS = {
    "lean-workbook": {
        "path": "lean-workbook-proved",
        "role": "train",
        "environment": "lean-4.8.0-rc1",
        "lean_version": "4.8.0-rc1",
        "mathlib_version": "v4.8.0-rc1",
        "preamble_field": None,
    },
    "proofnet-verified": {
        "path": "proofnet-verified",
        "role": "train",
        "environment": "lean-4.28.0",
        "lean_version": "4.28.0",
        "mathlib_version": "v4.28.0",
        "preamble_field": "state_before",
    },
    "gaokao-formal": {
        "path": "gaokao-formal",
        "role": "eval",
        "environment": "lean-4.27.0",
        "lean_version": "4.27.0",
        "mathlib_version": "v4.27.0",
        "preamble_field": "state_before",
    },
    "fate-m": {
        "path": "fate-m",
        "role": "eval",
        "environment": "lean-4.28.0",
        "lean_version": "4.28.0",
        "mathlib_version": "v4.28.0",
        "preamble_field": "state_before",
    },
}

_COMMENT = re.compile(r"/\-.*?\-/|--[^\n]*", re.DOTALL)
_DECLARATION_NAME = re.compile(r"^(theorem|lemma)\s+[^\s({:]+", re.DOTALL)
_PLACEHOLDER = re.compile(r"\s*:=\s*(?:by\s+)?sorry\s*$", re.DOTALL)


def theorem_signature(formal_statement: str) -> str:
    signature, replacements = _PLACEHOLDER.subn("", formal_statement, count=1)
    if replacements != 1:
        raise ValueError("formal statement must end in a single sorry placeholder")
    return signature.strip()


def normalized_statement(formal_statement: str) -> str:
    """Normalize only syntax text for exact deduplication/leakage filtering."""
    signature = _COMMENT.sub("", theorem_signature(formal_statement)).strip()
    signature, replacements = _DECLARATION_NAME.subn(r"\1 _", signature, count=1)
    if replacements != 1:
        raise ValueError("formal statement must start with theorem or lemma")
    return "".join(signature.split())


def statement_hash(formal_statement: str) -> str:
    return hashlib.sha256(normalized_statement(formal_statement).encode()).hexdigest()


def make_prompt(
    formal_statement: str,
    preamble: str,
    lean_version: str,
    mathlib_version: str,
) -> str:
    context = f"\nLean context:\n{preamble.strip()}\n" if preamble.strip() else ""
    return (
        f"You are an expert theorem prover using Lean {lean_version} and Mathlib "
        f"{mathlib_version}.\nComplete the theorem below. Return only the Lean proof "
        "expression that belongs after `:=`. A tactic proof beginning with `by` or "
        "a direct proof term is valid. Do not restate the theorem or use Markdown "
        "fences. Do not use `sorry`, `admit`, `unsafe`, or introduce axioms.\n"
        f"{context}\n{theorem_signature(formal_statement)}\n"
    )


def _repair_lean_workbook(theorem_id: str, statement: str) -> str:
    repair = LEAN_WORKBOOK_REPAIRS.get(theorem_id)
    if repair is None:
        return statement
    expected_hash, replacement = repair
    actual_hash = hashlib.sha256(statement.encode()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"repair source mismatch for {theorem_id}")
    return replacement


def load_canonical_rows(data_root: Path, name: str) -> list[dict[str, Any]]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}")
    spec = DATASETS[name]
    dataset_path = data_root / spec["path"]
    metadata_path = dataset_path / "opsd_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing dataset provenance: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("filter") != {"status": "proved"}:
        raise ValueError(f"{name} was not prepared with the proved-only contract")

    dataset = load_from_disk(str(dataset_path))
    canonical: dict[str, dict[str, Any]] = {}
    for row in dataset:
        if row["status"] != "proved":
            raise ValueError(f"non-proved row survived in {name}: {row['id']}")
        theorem_id = str(row["id"])
        statement = str(row["formal_statement"])
        if name == "lean-workbook":
            statement = _repair_lean_workbook(theorem_id, statement)
        preamble_field = spec["preamble_field"]
        preamble = str(row[preamble_field]) if preamble_field else ""
        candidate = {
            "id": theorem_id,
            "dataset": name,
            "formal_statement": statement,
            "preamble": preamble,
            "lean_environment": spec["environment"],
            "lean_version": spec["lean_version"],
            "mathlib_version": spec["mathlib_version"],
            "statement_hash": statement_hash(statement),
        }
        existing = canonical.get(theorem_id)
        if existing is not None and existing != candidate:
            raise ValueError(f"inconsistent repeated theorem {theorem_id!r} in {name}")
        canonical.setdefault(theorem_id, candidate)
    return list(canonical.values())


def _rank(row: dict[str, Any], seed: int) -> str:
    key = f"{seed}:{row['dataset']}:{row['id']}:{row['statement_hash']}"
    return hashlib.sha256(key.encode()).hexdigest()


def _deduplicate_and_filter(
    rows: Iterable[dict[str, Any]], eval_hashes: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    retained: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {"eval_exact_match": 0, "train_exact_duplicate": 0}
    for row in rows:
        digest = row["statement_hash"]
        if digest in eval_hashes:
            counts["eval_exact_match"] += 1
            continue
        if digest in seen:
            counts["train_exact_duplicate"] += 1
            continue
        seen.add(digest)
        retained.append(row)
    return retained, counts


def _slime_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": [
            {
                "role": "user",
                "content": make_prompt(
                    row["formal_statement"],
                    row["preamble"],
                    row["lean_version"],
                    row["mathlib_version"],
                ),
            }
        ],
        "metadata": {
            "problem_id": row["id"],
            "source_name": row["dataset"],
            "formal_statement": row["formal_statement"],
            "preamble": row["preamble"],
            "lean_environment": row["lean_environment"],
            "statement_hash": row["statement_hash"],
        },
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_filtered_corpora(
    data_root: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, int],
]:
    eval_rows = {
        name: load_canonical_rows(data_root, name)
        for name in ("gaokao-formal", "fate-m")
    }
    eval_hashes = {
        row["statement_hash"] for rows in eval_rows.values() for row in rows
    }
    train_by_source = {
        name: load_canonical_rows(data_root, name)
        for name in ("lean-workbook", "proofnet-verified")
    }
    merged = [row for rows in train_by_source.values() for row in rows]
    filtered, exclusions = _deduplicate_and_filter(merged, eval_hashes)
    return eval_rows, train_by_source, filtered, exclusions


def _write_materialized_data(
    output_root: Path,
    *,
    selected: list[dict[str, Any]],
    eval_rows: dict[str, list[dict[str, Any]]],
    train_by_source: dict[str, list[dict[str, Any]]],
    exclusions: dict[str, int],
    seed: int,
    train_filename: str,
    mode: str,
) -> dict[str, Any]:
    train_path = output_root / train_filename
    hashes = {"train": _write_jsonl(train_path, map(_slime_record, selected))}
    eval_paths: dict[str, str] = {}
    for name, rows in eval_rows.items():
        path = output_root / f"eval-{name}.jsonl"
        hashes[name] = _write_jsonl(path, map(_slime_record, rows))
        eval_paths[name] = str(path)

    report = {
        "schema_version": 1,
        "mode": mode,
        "seed": seed,
        "training_examples": len(selected),
        "training_by_source": {
            name: sum(row["dataset"] == name for row in selected)
            for name in ("lean-workbook", "proofnet-verified")
        },
        "source_unique_rows": {
            name: len(rows) for name, rows in train_by_source.items()
        },
        "evaluation_rows": {name: len(rows) for name, rows in eval_rows.items()},
        "excluded": exclusions,
        "reference_proof_fields_written": [],
        "train_path": str(train_path),
        "eval_paths": eval_paths,
        "sha256": hashes,
    }
    (output_root / "data-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def materialize(
    data_root: Path,
    output_root: Path,
    *,
    training_examples: int,
    seed: int = 42,
) -> dict[str, Any]:
    if not 0 < training_examples < 100:
        raise ValueError("smoke training_examples must be between 1 and 99")

    eval_rows, train_by_source, filtered, exclusions = _load_filtered_corpora(data_root)

    # Select each source independently so the small pilot cannot collapse to
    # Lean-Workbook merely because that corpus is much larger.
    quotas = {
        "lean-workbook": (training_examples + 1) // 2,
        "proofnet-verified": training_examples // 2,
    }
    selected: list[dict[str, Any]] = []
    for source, quota in quotas.items():
        candidates = [row for row in filtered if row["dataset"] == source]
        candidates.sort(key=lambda row: _rank(row, seed))
        if len(candidates) < quota:
            raise ValueError(f"not enough retained {source} rows for quota {quota}")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: _rank(row, seed + 1))

    return _write_materialized_data(
        output_root,
        selected=selected,
        eval_rows=eval_rows,
        train_by_source=train_by_source,
        exclusions=exclusions,
        seed=seed,
        train_filename="train-smoke.jsonl",
        mode="smoke",
    )


def materialize_final(
    data_root: Path,
    output_root: Path,
    *,
    lean_workbook_examples: int = 200,
    proofnet_verified_examples: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Write the deterministic, one-pass production curriculum and full eval sets.

    "First" means canonical source order after proved-only validation, exact
    statement deduplication, and evaluation-leakage filtering. Selection is
    performed before the chosen rows are deterministically interleaved.
    """
    if lean_workbook_examples != 200 or proofnet_verified_examples != 200:
        raise ValueError("final curriculum must contain exactly 200 Lean-Workbook and 200 ProofNet rows")

    eval_rows, train_by_source, filtered, exclusions = _load_filtered_corpora(data_root)
    candidates = {
        source: [row for row in filtered if row["dataset"] == source]
        for source in ("lean-workbook", "proofnet-verified")
    }
    requested = {
        "lean-workbook": lean_workbook_examples,
        "proofnet-verified": proofnet_verified_examples,
    }
    selected: list[dict[str, Any]] = []
    for source, count in requested.items():
        if len(candidates[source]) < count:
            raise ValueError(f"not enough retained {source} rows: need {count}, found {len(candidates[source])}")
        selected.extend(candidates[source][:count])

    if len({row["statement_hash"] for row in selected}) != sum(requested.values()):
        raise RuntimeError("final curriculum is not statement-unique")
    selected.sort(key=lambda row: _rank(row, seed))

    report = _write_materialized_data(
        output_root,
        selected=selected,
        eval_rows=eval_rows,
        train_by_source=train_by_source,
        exclusions=exclusions,
        seed=seed,
        train_filename="train-final.jsonl",
        mode="production",
    )
    report["selection"] = {
        "contract": "first-retained-canonical-source-order",
        "passes_per_dataset": 1,
        "lean_workbook_examples": lean_workbook_examples,
        "proofnet_verified_examples": proofnet_verified_examples,
    }
    (output_root / "data-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/prompts"))
    parser.add_argument("--training-examples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "final":
        report = materialize_final(args.data_root, args.output_root, seed=args.seed)
    else:
        report = materialize(
            args.data_root,
            args.output_root,
            training_examples=args.training_examples,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
