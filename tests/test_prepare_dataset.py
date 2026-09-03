from datasets import Dataset
import pytest

import json

import ppo_async.prepare_dataset as prepare_module
from ppo_async.prepare_dataset import (
    DATASET_REVISION,
    build_parser,
    keep_only_proved,
    prepare_dataset,
)


def test_keep_only_proved_removes_disproved_rows() -> None:
    source = Dataset.from_dict(
        {
            "id": ["theorem-1", "theorem-2", "theorem-3"],
            "status": ["proved", "disproved", "proved"],
            "formal_statement": ["first", "second", "third"],
        }
    )

    result = keep_only_proved(source)

    assert result["id"] == ["theorem-1", "theorem-3"]
    assert result["status"] == ["proved", "proved"]
    assert result.column_names == source.column_names


def test_keep_only_proved_requires_status_column() -> None:
    source = Dataset.from_dict({"id": ["theorem-1"]})

    with pytest.raises(ValueError, match="status"):
        keep_only_proved(source)


def test_dataset_revision_is_pinned_by_default() -> None:
    assert build_parser().parse_args([]).revision == DATASET_REVISION


def test_prepare_dataset_records_provenance(tmp_path, monkeypatch) -> None:
    source = Dataset.from_dict(
        {
            "id": ["one", "two"],
            "status": ["proved", "disproved"],
        }
    )
    calls = []

    def fake_load_dataset(dataset_id, **kwargs):
        calls.append((dataset_id, kwargs))
        return source

    monkeypatch.setattr(prepare_module, "load_dataset", fake_load_dataset)
    output = tmp_path / "prepared"

    assert prepare_dataset(output) == (2, 1)
    assert calls == [
        (
            "internlm/Lean-Workbook",
            {"split": "train", "revision": DATASET_REVISION},
        )
    ]
    metadata = json.loads((output / "opsd_metadata.json").read_text())
    assert metadata["dataset_revision"] == DATASET_REVISION
    assert metadata["filter"] == {"status": "proved"}
    assert list(metadata["artifact_sha256"]) == ["data-00000-of-00001.arrow"]
    assert len(metadata["artifact_sha256"]["data-00000-of-00001.arrow"]) == 64
