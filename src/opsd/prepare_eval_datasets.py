"""Prepare FATE-M and ProofNet-Verified in the Lean-Workbook row schema."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import re
import ssl
from urllib.request import urlopen

import certifi
from datasets import Dataset, Features, Value


CANONICAL_COLUMNS = (
    "id",
    "status",
    "tactic",
    "state_before",
    "state_after",
    "natural_language_statement",
    "answer",
    "formal_statement",
)
CANONICAL_FEATURES = Features({column: Value("string") for column in CANONICAL_COLUMNS})

SOURCES = {
    "fate-m": {
        "revision": "4eb33c8ccd0ff058b461cd763cc406509129743f",
        "url": (
            "https://raw.githubusercontent.com/frenzymath/FATE-M/"
            "4eb33c8ccd0ff058b461cd763cc406509129743f/FATE-M.json"
        ),
        "sha256": "39c13a39f82bb2f39fc42d0a0efb41c6b0ef16b8194c8c8e0ae281c5cf0276fb",
        "rows": 150,
        "lean_version": "4.28.0",
        "mathlib_version": "v4.28.0",
    },
    "proofnet-verified": {
        "revision": "160414332dc196583f6c37c310b420d2a3b07c58",
        "url": (
            "https://raw.githubusercontent.com/marcusm117/ProofNet-Verified/"
            "160414332dc196583f6c37c310b420d2a3b07c58/"
            "data/proofnet-verified.jsonl"
        ),
        "sha256": "381f4a06548a4ff6d9b923633c94a97b9c70f41033e13023aae31e1161b7f142",
        "rows": 367,
        "lean_version": "4.28.0",
        "mathlib_version": "v4.28.0",
    },
    "gaokao-formal": {
        "revision": "48c72753b7c7a7e0b07fe89bd97cf1a4899bab79",
        "url": (
            "https://raw.githubusercontent.com/Huawei-AI4Math/Mathesis/"
            "48c72753b7c7a7e0b07fe89bd97cf1a4899bab79/"
            "Gaokao-formal-2025.json"
        ),
        "sha256": "89bd19c12f19f00d2f510598f765e911782028e0c0be0bccdf3cdc53081f4aaf",
        "rows": 495,
        "lean_version": "4.27.0",
        "mathlib_version": "v4.27.0",
    },
}

# Exact namespace/scope directives from the pinned official `FATEM/<id>.lean`
# files. The JSON contains theorem declarations but omits this required context.
FATE_M_PREAMBLE_DIRECTIVES = {
    27: "open Polynomial",
    30: "open Classical",
    35: "open ComplexConjugate",
    38: "open Pointwise",
    46: "open Polynomial",
    60: "open scoped Pointwise\n\nopen MulOpposite",
    62: "open Pointwise",
    70: "open Classical",
    76: "open Pointwise",
    85: "open Classical",
    92: "open Polynomial",
    95: "open MulOpposite Pointwise",
    98: "open Classical",
    102: "open Classical",
    106: "open Classical",
    108: "open Equiv Equiv.Perm",
    113: "open Polynomial",
    133: "open Polynomial",
    135: "open BigOperators",
    140: "open Pointwise",
    144: "open Classical",
    146: "open Polynomial",
}


def _download(url: str) -> bytes:
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=120, context=context) as response:
        return response.read()


def _target_statement(source: str) -> str:
    """Keep one theorem and normalize its placeholder to Lean-Workbook form."""
    match = re.search(r"(?m)^theorem\s", source)
    if match is None:
        raise ValueError("formal source has no top-level theorem")
    statement = source[match.start() :].strip()
    statement, replacements = re.subn(
        r"\s*:=\s*(?:by\s+)?sorry\s*$", " := by sorry", statement, count=1
    )
    if replacements != 1:
        raise ValueError("target theorem does not end in exactly one `by sorry`")
    if re.search(r"\bsorry\b", statement.removesuffix(" := by sorry")):
        raise ValueError("target theorem contains an unexpected additional `sorry`")
    return statement


def _canonical_row(
    *, theorem_id: str, natural_statement: str, formal_statement: str, preamble: str
) -> dict[str, str]:
    if not theorem_id or not natural_statement or not preamble:
        raise ValueError(f"empty required field for {theorem_id!r}")
    return {
        "id": theorem_id,
        "status": "proved",
        "tactic": "",
        "state_before": preamble.strip(),
        "state_after": "no goals",
        "natural_language_statement": natural_statement,
        "answer": "",
        "formal_statement": _target_statement(formal_statement),
    }


def normalize_fate_m(payload: bytes) -> list[dict[str, str]]:
    records = json.loads(payload)
    if not isinstance(records, list):
        raise ValueError("FATE-M source must be a JSON array")
    rows = []
    for record in records:
        exercise_id = int(record["id"])
        directives = FATE_M_PREAMBLE_DIRECTIVES.get(exercise_id, "")
        preamble = "import Mathlib" + (f"\n\n{directives}" if directives else "")
        rows.append(
            _canonical_row(
                theorem_id=f"fate_m_{exercise_id:03d}",
                natural_statement=record["informal_statement"],
                formal_statement=record["formal_statement"],
                preamble=preamble,
            )
        )
    expected_ids = list(range(1, 151))
    if [int(record["id"]) for record in records] != expected_ids:
        raise ValueError("FATE-M IDs are not exactly the ordered range 1..150")
    return rows


def normalize_proofnet_verified(payload: bytes) -> list[dict[str, str]]:
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    rows = []
    for record in records:
        preamble_parts = [record["header"].strip()]
        if record["helper"].strip():
            preamble_parts.append(record["helper"].strip())
        rows.append(
            _canonical_row(
                theorem_id=f"proofnet_verified_{int(record['index']):03d}_{record['name']}",
                natural_statement=record["informal_stmt"],
                formal_statement=record["formal_stmt"],
                preamble="\n\n".join(preamble_parts),
            )
        )
    expected_indices = list(range(1, 368))
    if [int(record["index"]) for record in records] != expected_indices:
        raise ValueError("ProofNet-Verified indices are not exactly the ordered range 1..367")
    return rows


def normalize_gaokao_formal(payload: bytes) -> list[dict[str, str]]:
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    rows = []
    for record in records:
        source = record["formal_statement"]
        # The published source uses Lean-3-style ASCII `in` in finite-set big
        # operator binders (for example, `∑ i in s, f i`). Mathlib 4.27's
        # grammar accepts `∑ i ∈ s, f i`. Restrict the repair to the binder of
        # a sum/product so ordinary occurrences of the word `in` are untouched.
        source = re.sub(
            r"([∑∏]\s+[^,\n]*?)\s+in\s+",
            r"\1 ∈ ",
            source,
        )
        theorem_match = re.search(r"(?m)^theorem\s", source)
        if theorem_match is None:
            raise ValueError(f"Gaokao-Formal {record['id']} has no theorem")
        preamble = source[: theorem_match.start()]
        # Natural-language comments are not Lean context and would make the
        # proof prompt inconsistent with Lean-Workbook's formal-only setup.
        preamble = re.sub(r"/-.*?-/", "", preamble, flags=re.DOTALL).strip()
        rows.append(
            _canonical_row(
                theorem_id=f"gaokao_formal_{record['id']}",
                natural_statement=record["NL_English"],
                formal_statement=source,
                preamble=preamble,
            )
        )
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Gaokao-Formal source IDs are not unique")
    return rows


def validate_rows(rows: Iterable[dict[str, str]], expected_rows: int) -> list[dict[str, str]]:
    materialized = list(rows)
    if len(materialized) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(materialized)}")
    if len({row["id"] for row in materialized}) != expected_rows:
        raise ValueError("canonical theorem IDs are not unique")
    for row in materialized:
        if tuple(row) != CANONICAL_COLUMNS:
            raise ValueError(f"canonical schema/order mismatch for {row.get('id')!r}")
        if row["status"] != "proved" or row["state_after"] != "no goals":
            raise ValueError(f"invalid proved-row markers for {row['id']}")
        if row["tactic"] or row["answer"]:
            raise ValueError(f"proof-bearing field is not blank for {row['id']}")
        _target_statement(row["formal_statement"])
    return materialized


def prepare_dataset(name: str, output_root: Path, payload: bytes | None = None) -> Path:
    spec = SOURCES[name]
    source = _download(spec["url"]) if payload is None else payload
    digest = hashlib.sha256(source).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"{name} source SHA256 mismatch: {digest}")
    normalizers = {
        "fate-m": normalize_fate_m,
        "proofnet-verified": normalize_proofnet_verified,
        "gaokao-formal": normalize_gaokao_formal,
    }
    normalizer = normalizers[name]
    rows = validate_rows(normalizer(source), int(spec["rows"]))
    dataset = Dataset.from_list(rows, features=CANONICAL_FEATURES)
    if tuple(dataset.column_names) != CANONICAL_COLUMNS:
        raise RuntimeError("saved dataset schema/order differs from Lean-Workbook")

    output_dir = output_root / name
    dataset.save_to_disk(output_dir)
    artifact_sha256 = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.glob("*.arrow"))
    }
    metadata = {
        "dataset_name": name,
        "dataset_id": (
            "frenzymath/FATE-M"
            if name == "fate-m"
            else (
                "marcusm117/ProofNet-Verified"
                if name == "proofnet-verified"
                else "Huawei-AI4Math/Mathesis:Gaokao-Formal"
            )
        ),
        "dataset_revision": spec["revision"],
        "source_url": spec["url"],
        "source_sha256": digest,
        "source_rows": len(rows),
        "proved_rows": len(rows),
        "filter": {"status": "proved"},
        "schema_columns": list(CANONICAL_COLUMNS),
        "lean_version": spec["lean_version"],
        "mathlib_version": spec["mathlib_version"],
        "preamble_field": "state_before",
        "artifact_sha256": artifact_sha256,
    }
    if name == "gaokao-formal":
        metadata["source_normalizations"] = {
            "big_operator_ascii_in_to_membership": 144,
        }
    if name == "fate-m":
        metadata["source_context"] = {
            "revision": SOURCES["fate-m"]["revision"],
            "official_exercise_files_with_directives": len(
                FATE_M_PREAMBLE_DIRECTIVES
            ),
        }
    (output_dir / "opsd_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=[*SOURCES, "all"])
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    for name in names:
        path = prepare_dataset(name, args.output_root)
        print(f"Prepared {SOURCES[name]['rows']} {name} theorems at {path.resolve()}")


if __name__ == "__main__":
    main()
