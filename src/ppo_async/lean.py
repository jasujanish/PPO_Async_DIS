"""Fail-closed Lean proof extraction and kernel verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

from ppo_async.data import theorem_signature


FORBIDDEN_PROOF_TOKEN = re.compile(
    r"\b(?:sorry|sorryAx|admit|axiom|unsafe|native_decide|run_tac|implemented_by|extern)\b",
    re.IGNORECASE,
)
DECLARATION_NAME = re.compile(r"^(?:theorem|lemma)\s+([^\s({:]+)", re.DOTALL)
LEAN_PROJECTS = {
    "lean-4.8.0-rc1": Path("/lean-projects/v48"),
    "lean-4.27.0": Path("/lean-projects/v427"),
    "lean-4.28.0": Path("/lean-projects/v428"),
}


@dataclass(frozen=True)
class VerificationResult:
    status: str
    proof: str
    diagnostics: str
    elapsed_seconds: float

    @property
    def reward(self) -> float | None:
        if self.status == "verified":
            return 1.0
        if self.status == "rejected":
            return 0.0
        return None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_proof_expression(content: str, formal_statement: str | None = None) -> str:
    text = content.strip()
    if "<think>" in text and "</think>" not in text:
        # Never reward a draft proof copied from an unfinished hidden-reasoning
        # block. A truncated Qwen response has not actually submitted an answer.
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    fenced = re.search(r"```(?:lean4?|Lean4?)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    elif len(text) >= 2 and text.startswith("`") and text.endswith("`"):
        text = text.strip("`").strip()

    if text.startswith("theorem ") and formal_statement is not None:
        signature = theorem_signature(formal_statement)
        flexible = r"\s+".join(re.escape(fragment) for fragment in signature.split())
        match = re.match(rf"{flexible}\s*:=\s*(.*)\Z", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif text.startswith("theorem ") and ":=" in text:
        text = text.split(":=", 1)[1].strip()
    elif text.startswith(":="):
        text = text[2:].strip()
    return text


def theorem_name(formal_statement: str) -> str:
    match = DECLARATION_NAME.match(theorem_signature(formal_statement))
    if match is None:
        raise ValueError("formal statement has no theorem/lemma declaration name")
    return match.group(1)


def render_candidate(
    formal_statement: str, preamble: str, proof: str, *, nonce: str = "test"
) -> str:
    context = preamble.strip() or "import Mathlib"
    name = theorem_name(formal_statement)
    return (
        f"import Lean\n{context}\n\n"
        + Path(__file__).with_name("Verifier.lean").read_text(encoding="utf-8")
        + "\n\n"
        "set_option maxHeartbeats 400000 in\n"
        f"{theorem_signature(formal_statement)} := ppo_proof {json.dumps(proof, ensure_ascii=False)}\n\n"
        f"ppo_audit {name} {json.dumps(nonce)}\n"
    )


def verify_proof(
    formal_statement: str,
    preamble: str,
    content: str,
    lean_environment: str,
    *,
    projects: dict[str, Path] | None = None,
    timeout_seconds: int = 180,
) -> VerificationResult:
    started = time.monotonic()
    proof = extract_proof_expression(content, formal_statement)
    if not proof:
        return VerificationResult("rejected", proof, "empty proof", time.monotonic() - started)
    forbidden = FORBIDDEN_PROOF_TOKEN.search(proof)
    if forbidden:
        return VerificationResult(
            "rejected",
            proof,
            f"forbidden proof token: {forbidden.group(0)}",
            time.monotonic() - started,
        )
    nonce = uuid.uuid4().hex
    try:
        source = render_candidate(formal_statement, preamble, proof, nonce=nonce)
    except (ValueError, OSError) as exc:
        return VerificationResult("infrastructure_error", proof, str(exc), time.monotonic() - started)

    project_map = LEAN_PROJECTS if projects is None else projects
    project = project_map.get(lean_environment)
    if project is None or not project.is_dir():
        return VerificationResult(
            "infrastructure_error",
            proof,
            f"Lean project unavailable for {lean_environment}: {project}",
            time.monotonic() - started,
        )

    source_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", prefix="ppo-async-", encoding="utf-8", delete=False
        ) as handle:
            handle.write(source)
            source_path = Path(handle.name)
        completed = subprocess.run(
            ["lake", "env", "lean", str(source_path)],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        full_output = completed.stdout + completed.stderr
        output = full_output[-12_000:]
        audited = f"PPO_ASYNC_VERIFIED {nonce}" in full_output
        status = "verified" if completed.returncode == 0 and audited else "rejected"
        if completed.returncode == 0 and not audited:
            output += "\nmissing successful theorem axiom audit"
        return VerificationResult(status, proof, output, time.monotonic() - started)
    except subprocess.TimeoutExpired as exc:
        return VerificationResult("rejected", proof, f"Lean timeout: {exc}", time.monotonic() - started)
    except OSError as exc:
        return VerificationResult(
            "infrastructure_error", proof, f"failed to execute Lean: {exc}", time.monotonic() - started
        )
    finally:
        if source_path is not None:
            source_path.unlink(missing_ok=True)
