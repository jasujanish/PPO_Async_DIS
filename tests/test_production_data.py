import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from ppo_async.config import load_experiment
from ppo_async.data import materialize_final


ROOT = Path(__file__).resolve().parents[1]


def test_prepared_curriculum_has_600_unique_problems_and_one_pass_budget():
    config = load_experiment()
    production = config["production"]
    root = ROOT / "prepared_data" / production["prepared_data_version"]
    report = json.loads((root / "data-report.json").read_text())
    rows = [json.loads(line) for line in (root / "train-final.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["metadata"]["statement_hash"] for r in rows}) == 600
    assert Counter(r["metadata"]["source_name"] for r in rows) == {
        "lean-workbook": 300, "proofnet-verified": 300,
    }
    assert report["training_examples"] == 600
    assert report["selection"]["passes_per_dataset"] == 1
    assert report["processed_example_budget"] == 600
    assert production["num_rollouts"] * config["rollout"]["batch_size"] == len(rows)
    hashes = set()
    for dataset in ("gaokao-formal", "fate-m"):
        path = root / f"eval-{dataset}.jsonl"
        eval_rows = [json.loads(line) for line in path.read_text().splitlines()]
        hashes.update(r["metadata"]["statement_hash"] for r in eval_rows)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report["sha256"][dataset]
    assert hashes.isdisjoint(r["metadata"]["statement_hash"] for r in rows)
    assert hashlib.sha256((root / "train-final.jsonl").read_bytes()).hexdigest() == report["sha256"]["train"]
    assert all(set(r) == {"input", "metadata"} for r in rows)
    assert all("tactic" not in r["metadata"] and "answer" not in r["metadata"] for r in rows)


def test_materialize_final_selects_first_300_per_source_before_shuffle(tmp_path, monkeypatch):
    from ppo_async import data

    sources = {}
    for source in ("lean-workbook", "proofnet-verified"):
        sources[source] = [
            dict(id=f"{source}-{i}", dataset=source,
                 formal_statement=f"theorem t{i} : {i} = {i} := by sorry",
                 preamble="import Mathlib", lean_environment="lean-4.28.0",
                 lean_version="4.28.0", mathlib_version="v4.28.0",
                 statement_hash=f"{source}-{i}")
            for i in range(367)
        ]
    monkeypatch.setattr(data, "_load_filtered_corpora", lambda _root: (
        {"gaokao-formal": [], "fate-m": []}, sources,
        [r for rows in sources.values() for r in rows], {},
    ))
    report = materialize_final(tmp_path / "source", tmp_path / "output")
    rows = [json.loads(line) for line in Path(report["train_path"]).read_text().splitlines()]
    assert {r["metadata"]["problem_id"] for r in rows} == {
        f"{source}-{i}" for source in sources for i in range(300)
    }
    assert report["processed_example_budget"] == 600


def test_production_rejects_resume_with_changed_data_or_arm(tmp_path):
    from modal_train import _ensure_run_contract

    root = tmp_path / "run"
    report = {"sha256": {"train": "original"}}
    _ensure_run_contract(root, "sync_ppo", report)
    (root / "RUN_COMPLETE.json").write_text("{}")
    _ensure_run_contract(root, "sync_ppo", report)
    with pytest.raises(RuntimeError, match="fresh run name"):
        _ensure_run_contract(root, "async_ppo", report)
    with pytest.raises(RuntimeError, match="fresh run name"):
        _ensure_run_contract(root, "sync_ppo", {"sha256": {"train": "changed"}})
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "events.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="fresh run name"):
        _ensure_run_contract(legacy, "sync_ppo", report)
