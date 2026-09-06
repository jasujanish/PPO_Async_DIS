import sys
from types import SimpleNamespace

import pytest

import launch_final


@pytest.mark.parametrize('deployed_arms', [['async_ppo'], ['async_ppo_dis_masking']])
def test_default_launch_selects_only_experiment2_and_checks_deployment(monkeypatch, deployed_arms):
    calls = []
    def lookup(app, name):
        if name == 'preflight_final_commands':
            return SimpleNamespace(remote=lambda: {'status': 'valid', 'arms': deployed_arms, 'implementation_sha256': launch_final.implementation_digest()})
        def spawn(arm, name):
            calls.append((arm, name))
            return SimpleNamespace(object_id='fake-call')
        return SimpleNamespace(spawn=spawn)
    monkeypatch.setattr(launch_final.modal.Function, 'from_name', lookup)
    monkeypatch.setattr(sys, 'argv', ['launch_final.py'])
    if 'async_ppo_dis_masking' not in deployed_arms:
        with pytest.raises(RuntimeError, match='deploy the current'):
            launch_final.main()
        assert not calls
    else:
        launch_final.main()
        assert calls == [('async_ppo_dis_masking', 'experiment2-dis-masking-balanced400-h200-v1')]


def test_launch_rejects_stale_deployed_code_before_allocating_gpu(monkeypatch):
    def lookup(app, name):
        assert name == 'preflight_final_commands'
        return SimpleNamespace(remote=lambda: dict(status='valid', arms=['async_ppo_dis_masking'],
                                                   implementation_sha256='old-code'))
    monkeypatch.setattr(launch_final.modal.Function, 'from_name', lookup)
    monkeypatch.setattr(sys, 'argv', ['launch_final.py'])
    with pytest.raises(RuntimeError, match='differs from this checkout'):
        launch_final.main()
