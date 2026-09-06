"""Small, checked compatibility patch for the pinned SLIME/SGLang pair."""

from pathlib import Path


_PAUSE = 'response = requests.post(f"http://{self.server_host}:{self.server_port}/pause_generation", json={})'
_ABORT = _PAUSE.replace('json={}', 'json={"mode": "abort"}')


def patch_slime(root: Path) -> None:
    path = root / "slime/backends/sglang_utils/sglang_engine.py"
    source = path.read_text()
    if source.count(_ABORT) == 1 and _PAUSE not in source:
        return
    if source.count(_PAUSE) != 1:
        raise RuntimeError("pinned SLIME pause implementation changed; review the abort patch")
    # Abort gates new admissions and drains requests before cache flushing.
    # The completion stream resumes partial samples after publication.
    path.write_text(source.replace(_PAUSE, _ABORT))


if __name__ == "__main__":
    patch_slime(Path("/root/slime"))
