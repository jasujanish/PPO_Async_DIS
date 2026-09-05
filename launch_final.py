"""Submit persistent, serialized production calls to the deployed Modal app."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import modal


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ppo_async.config import ARMS  # noqa: E402


APP_NAME = "ppo-async-qwen35-4b"


def _safe_prefix(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"invalid run prefix: {value!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", default="balanced600-1pass-qwen35-4b-h200-b8-v4")
    parser.add_argument(
        "--arm",
        choices=("selected", "all", "sync-ppo", "async-ppo", "async-ppo-dis"),
        default="selected",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the CPU-only pinned SLIME/Megatron command validation.",
    )
    args = parser.parse_args()
    prefix = _safe_prefix(args.run_prefix)
    if args.arm == "selected":
        selected = ["async_ppo_dis", "sync_ppo"]
    elif args.arm == "all":
        selected = list(ARMS)
    else:
        selected = [args.arm.replace("-", "_")]
    preflight = None
    if not args.skip_preflight:
        preflight = modal.Function.from_name(APP_NAME, "preflight_final_commands").remote()
    function = modal.Function.from_name(APP_NAME, "train_final")
    calls = []
    for arm in selected:
        run_name = f"{prefix}-{arm.replace('_', '-')}" if len(selected) > 1 else prefix
        call = function.spawn(arm, run_name)
        calls.append(
            {
                "arm": arm,
                "run_name": run_name,
                "function_call_id": call.object_id,
            }
        )
    print(
        json.dumps(
            {
                "status": "launched",
                "deployment": APP_NAME,
                "preflight": preflight,
                "gpu_concurrency": "one production run at a time; sync=1 H200, async=2 H200",
                "calls": calls,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
