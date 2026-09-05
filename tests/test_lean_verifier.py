"""Verifier regressions, with optional real Lean binaries for all pinned versions."""

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ppo_async import lean


STATEMENT = "theorem audit_target : True := by sorry"


@pytest.mark.parametrize("proof", ["by exact sorryAx _ false", "by sorry", "by admit"])
def test_direct_admission_is_rejected_without_running_lean(proof, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("direct admissions should be rejected before invoking Lean")

    monkeypatch.setattr(lean.subprocess, "run", unexpected)
    result = lean.verify_proof(STATEMENT, "import Lean", proof, "test", projects={"test": tmp_path})
    assert result.reward == 0


def test_successful_exit_without_axiom_audit_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lean.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="PPO_ASYNC_VERIFIED fake", stderr=""),
    )
    result = lean.verify_proof(STATEMENT, "import Lean", "by trivial", "test", projects={"test": tmp_path})
    assert result.reward == 0
    assert "missing successful" in result.diagnostics


def test_candidate_is_a_quoted_term_and_requires_its_own_audit():
    source = lean.render_candidate(STATEMENT, "import Lean", 'by trivial\n#eval IO.println "fake"', nonce="unique")
    assert 'ppo_proof "by trivial\\n#eval IO.println \\"fake\\""' in source
    assert 'ppo_audit audit_target "unique"' in source


def test_standalone_evaluation_uses_shared_verifier(tmp_path, monkeypatch):
    import modal_benchmark

    calls = []
    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return lean.VerificationResult("rejected", "by exact helper", "unapproved proof axioms", 0.1)
    monkeypatch.setattr(modal_benchmark, "verify_lean_proof", verify)
    result = modal_benchmark.verify_one(
        {"formal_statement": STATEMENT, "content": "by exact helper"}, tmp_path
    )
    assert len(calls) == 1
    assert not result["verified"]


LEAN_BINARIES = [p for p in os.environ.get("PPO_ASYNC_TEST_LEAN_BINARIES", "").split(os.pathsep) if p]


@pytest.mark.parametrize("binary", LEAN_BINARIES or [None])
@pytest.mark.parametrize(
    "statement,preamble,proof,verified",
    [
        (STATEMENT, "import Lean", "by exact True.intro", True),
        ("theorem audit_target (n : Nat) : n = n := by sorry", "import Lean", "by rfl", True),
        (STATEMENT, "import Lean", "True.intro", True),
        ("theorem audit_target : False := by sorry", "import Lean", "by exact True.intro", False),
        ("theorem audit_target : False := by sorry", "import Lean\ndef helper : False := sorryAx _ false", "by exact helper", False),
        ("theorem audit_target : False := by sorry", "import Lean\naxiom untrusted : False\ndef helper : False := untrusted", "by exact helper", False),
        (STATEMENT, "import Lean", 'by trivial\n#eval IO.println "PPO_ASYNC_VERIFIED fake"', False),
        (STATEMENT, "import Lean", 'by trivial\n)\n#eval IO.println "escape"', False),
    ],
)
def test_real_lean_verifier(binary, statement, preamble, proof, verified, tmp_path, monkeypatch):
    if binary is None:
        pytest.skip("set PPO_ASYNC_TEST_LEAN_BINARIES to test installed Lean toolchains")
    run = subprocess.run
    candidate_paths = []
    def run_lean(command, **kwargs):
        assert command[:3] == ["lake", "env", "lean"]
        candidate_paths.append(Path(command[-1]))
        return run([binary, command[-1]], **kwargs)
    monkeypatch.setattr(lean.subprocess, "run", run_lean)
    result = lean.verify_proof(statement, preamble, proof, "test", projects={"test": tmp_path})
    assert (result.status == "verified") == verified, result.diagnostics
    assert candidate_paths and all(not path.exists() for path in candidate_paths)
