import json

import pytest

from opsd.prepare_eval_datasets import (
    CANONICAL_COLUMNS,
    normalize_fate_m,
    normalize_gaokao_formal,
    normalize_proofnet_verified,
    validate_rows,
)


def test_fate_m_normalizes_to_lean_workbook_schema() -> None:
    records = [
        {
            "id": index,
            "informal_statement": f"Problem {index}",
            "formal_statement": (
                "import Mathlib\n\n/-- Problem. -/\n"
                f"theorem fate_{index} : True := by\n  sorry"
            ),
        }
        for index in range(1, 151)
    ]

    rows = validate_rows(normalize_fate_m(json.dumps(records).encode()), 150)

    assert tuple(rows[0]) == CANONICAL_COLUMNS
    assert rows[0]["id"] == "fate_m_001"
    assert rows[0]["state_before"] == "import Mathlib"
    assert rows[0]["formal_statement"] == "theorem fate_1 : True := by sorry"
    assert rows[0]["tactic"] == rows[0]["answer"] == ""


def test_fate_m_restores_official_per_exercise_context() -> None:
    records = [
        {
            "id": index,
            "informal_statement": f"Problem {index}",
            "formal_statement": f"theorem fate_{index} : True := by sorry",
        }
        for index in range(1, 151)
    ]

    rows = normalize_fate_m(json.dumps(records).encode())

    assert rows[26]["state_before"] == "import Mathlib\n\nopen Polynomial"
    assert rows[34]["state_before"] == "import Mathlib\n\nopen ComplexConjugate"
    assert rows[59]["state_before"] == (
        "import Mathlib\n\nopen scoped Pointwise\n\nopen MulOpposite"
    )
    assert rows[84]["state_before"] == "import Mathlib\n\nopen Classical"
    assert rows[99]["state_before"] == "import Mathlib"


def test_proofnet_preserves_header_and_helper_but_not_proof() -> None:
    records = [
        {
            "index": index,
            "name": f"example_{index}",
            "header": "import Mathlib\nopen Set",
            "helper": "def helper : Nat := 1" if index == 1 else "",
            "informal_stmt": f"Problem {index}",
            "formal_stmt": f"theorem example_{index} : True := by\n  sorry",
        }
        for index in range(1, 368)
    ]
    payload = b"\n".join(json.dumps(record).encode() for record in records)

    rows = validate_rows(normalize_proofnet_verified(payload), 367)

    assert rows[0]["state_before"] == (
        "import Mathlib\nopen Set\n\ndef helper : Nat := 1"
    )
    assert "formal_proof" not in rows[0]
    assert rows[-1]["id"] == "proofnet_verified_367_example_367"


def test_validation_rejects_proof_leakage() -> None:
    row = dict.fromkeys(CANONICAL_COLUMNS, "")
    row.update(
        {
            "id": "bad",
            "status": "proved",
            "state_before": "import Mathlib",
            "state_after": "no goals",
            "natural_language_statement": "Bad",
            "tactic": "by trivial",
            "formal_statement": "theorem bad : True := by sorry",
        }
    )

    with pytest.raises(ValueError, match="proof-bearing"):
        validate_rows([row], 1)


def test_proofnet_normalizes_direct_sorry_placeholder() -> None:
    records = [
        {
            "index": index,
            "name": f"example_{index}",
            "header": "import Mathlib",
            "helper": "",
            "informal_stmt": f"Problem {index}",
            "formal_stmt": f"theorem example_{index} : True :=\n  sorry",
        }
        for index in range(1, 368)
    ]
    payload = b"\n".join(json.dumps(record).encode() for record in records)

    rows = normalize_proofnet_verified(payload)

    assert rows[0]["formal_statement"] == "theorem example_1 : True := by sorry"


def test_gaokao_keeps_lean_preamble_and_removes_natural_language_comment() -> None:
    record = {
        "id": "g_0",
        "NL_English": "Prove True.",
        "formal_statement": (
            "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n"
            "open BigOperators Real Nat Topology Rat\n\n"
            "/-Prove True.-/\ntheorem gaokaoformal_g_0 : True := by sorry"
        ),
    }

    rows = normalize_gaokao_formal((json.dumps(record) + "\n").encode())

    assert tuple(rows[0]) == CANONICAL_COLUMNS
    assert rows[0]["id"] == "gaokao_formal_g_0"
    assert "Prove True" not in rows[0]["state_before"]
    assert rows[0]["state_before"].startswith("import Mathlib\nimport Aesop")
    assert rows[0]["formal_statement"] == (
        "theorem gaokaoformal_g_0 : True := by sorry"
    )


def test_gaokao_repairs_only_ascii_big_operator_binders() -> None:
    record = {
        "id": "g_1",
        "NL_English": "A sum in a finite interval.",
        "formal_statement": (
            "import Mathlib\nopen BigOperators\n"
            "theorem gaokaoformal_g_1 (n : ℕ) : "
            "(∑ i in Finset.Icc 1 n, i) = ∑ i ∈ Finset.Icc 1 n, i := by sorry"
        ),
    }

    row = normalize_gaokao_formal((json.dumps(record) + "\n").encode())[0]

    assert "∑ i in" not in row["formal_statement"]
    assert row["formal_statement"].count("∑ i ∈ Finset.Icc 1 n, i") == 2
