"""Sparse Lean kernel reward for SLIME rollouts."""

from __future__ import annotations

import asyncio
from typing import Any

from ppo_async.lean import verify_proof


async def reward(args: Any, sample: Any, **_: Any) -> float:
    metadata = sample.metadata or {}
    required = ("formal_statement", "preamble", "lean_environment", "statement_hash")
    missing = [key for key in required if key not in metadata]
    if missing:
        sample.remove_sample = True
        metadata["verification"] = {
            "status": "infrastructure_error",
            "diagnostics": f"missing immutable metadata: {missing}",
        }
        return 0.0

    result = await asyncio.to_thread(
        verify_proof,
        str(metadata["formal_statement"]),
        str(metadata["preamble"]),
        str(sample.response),
        str(metadata["lean_environment"]),
    )
    metadata["verification"] = result.to_dict()
    if result.reward is None:
        sample.remove_sample = True
        return 0.0
    return result.reward
