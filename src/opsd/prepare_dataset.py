"""Download Lean-Workbook and retain only proved theorem rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

DATASET_ID = "internlm/Lean-Workbook"
DATASET_REVISION = "2e066e310b2c6d2c27616927ae131f82901c8f1c"
PROVED_STATUS = "proved"


def keep_only_proved(dataset: Dataset) -> Dataset:
    """Return rows whose status is exactly ``proved`` and validate the result."""
    if "status" not in dataset.column_names:
        raise ValueError("Lean-Workbook dataset does not contain a 'status' column")

    filtered = dataset.filter(
        lambda status: status == PROVED_STATUS,
        input_columns=["status"],
        desc="Keeping proved theorems",
    )

    remaining_statuses = set(filtered.unique("status"))
    if remaining_statuses - {PROVED_STATUS}:
        raise RuntimeError(
            f"Filtered dataset contains unexpected statuses: {remaining_statuses}"
        )

    return filtered


def prepare_dataset(
    output_dir: Path,
    *,
    split: str = "train",
    revision: str = DATASET_REVISION,
) -> tuple[int, int]:
    """Download, filter, and persist Lean-Workbook.

    Returns the source and filtered row counts.
    """
    load_kwargs: dict[str, Any] = {"split": split, "revision": revision}

    source = load_dataset(DATASET_ID, **load_kwargs)
    filtered = keep_only_proved(source)
    filtered.save_to_disk(output_dir)
    artifact_sha256 = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.glob("*.arrow"))
    }
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_revision": revision,
        "split": split,
        "source_rows": len(source),
        "proved_rows": len(filtered),
        "filter": {"status": PROVED_STATUS},
        "artifact_sha256": artifact_sha256,
    }
    (output_dir / "opsd_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return len(source), len(filtered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Lean-Workbook and keep only proved theorem rows."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/lean-workbook-proved"),
        help="Directory for the filtered Hugging Face dataset",
    )
    parser.add_argument("--split", default="train", help="Dataset split to prepare")
    parser.add_argument(
        "--revision",
        default=DATASET_REVISION,
        help="Hugging Face dataset revision (defaults to the audited commit)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_count, proved_count = prepare_dataset(
        args.output_dir,
        split=args.split,
        revision=args.revision,
    )
    removed_count = source_count - proved_count
    print(f"Source rows: {source_count:,}")
    print(f"Proved rows written: {proved_count:,}")
    print(f"Disproved rows removed: {removed_count:,}")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
