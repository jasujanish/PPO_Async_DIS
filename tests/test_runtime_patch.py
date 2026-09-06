import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from ppo_async.training.runtime_patch import patch_slime, _PAUSE


def test_pause_drains_requests_and_leaves_resume_unchanged(tmp_path):
    path = tmp_path / "slime/backends/sglang_utils/sglang_engine.py"
    path.parent.mkdir(parents=True)
    path.write_text('''class SGLangEngine:
    def pause_generation(self):
        if self.node_rank != 0:
            return
        ''' + _PAUSE + '''
        response.raise_for_status()
        return response

    def continue_generation(self):
        return requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
''')
    before = path.read_text()
    patch_slime(tmp_path)
    after = path.read_text()
    assert before.split("    def continue_generation")[1] == after.split("    def continue_generation")[1]
    patch_slime(tmp_path)
    assert path.read_text() == after
    state = {"active": ["proof-a", "proof-b"], "paused": False}
    def post(url, json):
        if url.endswith("/pause_generation"):
            assert json == {"mode": "abort"}
            state["active"].clear()
            state["paused"] = True
        else:
            state["paused"] = False
        return SimpleNamespace(raise_for_status=lambda: None)
    namespace = {"requests": SimpleNamespace(post=post)}
    exec(compile(ast.parse(after), str(path), "exec"), namespace)
    engine = namespace["SGLangEngine"]()
    engine.node_rank, engine.server_host, engine.server_port = 0, "localhost", 1234
    for _ in range(5):
        engine.pause_generation()
        assert state == {"active": [], "paused": True}
        engine.continue_generation()
        assert state == {"active": [], "paused": False}
    engine.node_rank = 1
    engine.pause_generation()
    assert not state["paused"]


def test_patch_fails_if_upstream_contract_changes(tmp_path):
    path = tmp_path / "slime/backends/sglang_utils/sglang_engine.py"
    path.parent.mkdir(parents=True)
    path.write_text("# changed implementation\n")
    with pytest.raises(RuntimeError, match="implementation changed"):
        patch_slime(tmp_path)
