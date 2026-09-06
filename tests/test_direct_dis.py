"""CPU checks of the direct mask, including the pinned SLIME loss when available."""
import ast
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from ppo_async.training.dis import apply_dis_mask, direct_dis_policy_loss


@pytest.mark.parametrize("reject_all", [False, True])
def test_direct_mask_preserves_retained_ppo_gradients(monkeypatch, reject_all):
    monkeypatch.setenv("PPO_ASYNC_DIS_EPSILON_LOW", "0.3")
    monkeypatch.setenv("PPO_ASYNC_DIS_EPSILON_HIGH", "5.0")
    ratios = torch.tensor([0.5, 0.8, 1.1, 1.5, 7.0], dtype=torch.float64)
    if reject_all:
        ratios.fill_(7.0)
    log_probs = ratios.log().requires_grad_()
    advantages = torch.tensor([1., 1., -1., -1., -1.])
    ratio = log_probs.exp()
    ppo_loss = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)
    masks = [torch.ones(5)]
    masked, returned_masks, _ = apply_dis_mask(
        None, pg_loss=ppo_loss, train_log_probs=[log_probs.detach()],
        rollout_log_probs=[torch.zeros(5)], loss_masks=masks,
    )
    original_grad, = torch.autograd.grad(ppo_loss.mean(), log_probs, retain_graph=True)
    masked_grad, = torch.autograd.grad(masked.mean(), log_probs)
    keep = (ratios > 0.7) & (ratios < 6.0)
    torch.testing.assert_close(masked_grad, original_grad * keep)
    assert returned_masks is masks
    assert torch.isfinite(masked).all()


def _load_function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    # Avoid importing GPU-only Megatron dependencies to test its real loss on CPU.
    source = 'from __future__ import annotations\n' + ast.unparse(node)
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace[name]


@pytest.fixture
def native_loss(monkeypatch):
    root = os.environ.get("SLIME_ROOT")
    if not root:
        pytest.skip("set SLIME_ROOT to the pinned SLIME checkout for CPU integration")
    root = Path(root)
    from torch.utils.checkpoint import checkpoint
    from ppo_async.training.memory import checkpoint_chunks

    ns = dict(torch=torch, dist=torch.distributed, F=torch.nn.functional,
              checkpoint=checkpoint, _LOG_PROB_CAPTURE=None)
    ns["mpu"] = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_rank=lambda: 0,
        get_data_parallel_world_size=lambda **kw: 1,
    )
    # Execute the real pinned kernels, packing, reducers and dispatcher on CPU;
    # only the single-rank Megatron process-group queries are substituted.
    for relative in ('slime/utils/ppo_utils.py',
                     'slime/backends/megatron_utils/cp_utils.py',
                     'slime/backends/megatron_utils/loss.py'):
        path = root / relative
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        for node in nodes:
            node.decorator_list = []
        exec(compile('from __future__ import annotations\n' +
                     ast.unparse(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), ns)
    ns['_calculate_log_probs_and_entropy_chunk'] = checkpoint_chunks(ns['_calculate_log_probs_and_entropy_chunk'])
    ns['load_function'] = lambda path: (direct_dis_policy_loss if path.endswith('direct_dis_policy_loss')
                                       else apply_dis_mask)
    for name in ('slime.backends.megatron_utils.loss', 'slime.utils.ppo_utils'):
        module = ModuleType(name)
        module.__dict__.update(ns)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("PPO_ASYNC_DIS_EPSILON_LOW", "0.3")
    monkeypatch.setenv("PPO_ASYNC_DIS_EPSILON_HIGH", "5.0")
    return ns


def make_batch(native, differences):
    args = SimpleNamespace(
        use_rollout_logprobs=True, use_tis=False, get_mismatch_metrics=False,
        use_opsm=False, advantage_estimator='ppo', eps_clip=0.2,
        eps_clip_high=0.2, eps_clip_c=None, custom_tis_function_path=None,
        calculate_per_token_loss=False, entropy_coef=0., use_kl_loss=False,
        rollout_temperature=0.8, rollout_top_p=0.95, log_probs_chunk_size=2,
        allgather_cp=False, loss_type='custom_loss', recompute_loss_function=True,
        custom_loss_function_path='ppo_async.training.dis.direct_dis_policy_loss',
    )
    torch.manual_seed(7)
    logits = torch.randn(1, 9, 8, requires_grad=True)
    # Two differently sized packed proofs, with an observation token excluded.
    batch = dict(unconcat_tokens=[torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3, 4])],
                 total_lengths=[4, 5], response_lengths=[2, 3],
                 loss_masks=[torch.ones(2), torch.tensor([1., 0., 1.])],
                 rollout_mask_sums=[torch.tensor(2.), torch.tensor(2.)],
                 advantages=[torch.tensor([1., -1.]), torch.tensor([-1., 1., -1.])],
                 rollout_top_p_token_ids=[list(range(6))*2, list(range(6))*3],
                 rollout_top_p_token_offsets=[[0, 6, 12], [0, 6, 12, 18]])
    _, outputs = native['get_log_probs_and_entropy'](
        logits, args=args, unconcat_tokens=batch['unconcat_tokens'],
        total_lengths=batch['total_lengths'], response_lengths=batch['response_lengths'],
        **native['get_rollout_top_p_logprob_kwargs'](args, batch))
    current = torch.cat(outputs['log_probs']).detach()
    batch['rollout_log_probs'] = list((current - torch.tensor(differences)).split([2, 3]))
    batch['log_probs'] = [torch.full((2,), -10.), torch.full((3,), -10.)]
    reducer = native['get_sum_of_sample_mean'](
        batch['total_lengths'], batch['response_lengths'], batch['loss_masks'], batch['rollout_mask_sums'])
    return args, batch, logits, reducer


@pytest.mark.parametrize('differences', [
    [-0.1, 0.1, 0.2, 0., -0.2],  # all action tokens retained
    [-0.7, 0.1, 0.4, 0., 2.],    # mask complements PPO clipping
    [-1000., 1000., 1000., 0., -1000.],  # all action tokens rejected, exp would overflow
])
def test_native_packed_loss_gradients_and_checkpoint_recompute(native_loss, differences):
    native = native_loss
    args, batch, logits, reducer = make_batch(native, differences)
    original_baseline = batch['log_probs']
    loss, metrics = direct_dis_policy_loss(args, batch, logits, reducer)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
    assert torch.isfinite(gradient).all()
    # Independent reference: explicitly sum only retained token contributions.
    _, outputs = native['get_log_probs_and_entropy'](
        logits, args=args, unconcat_tokens=batch['unconcat_tokens'],
        total_lengths=batch['total_lengths'], response_lengths=batch['response_lengths'],
        **native['get_rollout_top_p_logprob_kwargs'](args, batch))
    reference = logits.sum() * 0
    for lp, old, advantage, mask in zip(outputs['log_probs'], batch['rollout_log_probs'],
                                       batch['advantages'], batch['loss_masks']):
        for token, behavior, adv, action in zip(lp, old, advantage, mask):
            delta = token - behavior
            if action and -0.3566749439 < delta.item() < 1.7917594692:
                ratio = delta.exp()
                reference = reference - torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv) / mask.sum()
    expected_grad, = torch.autograd.grad(reference, logits, retain_graph=True)
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(gradient, expected_grad)
    # Real dispatch + outer loss checkpoint + chunk checkpoints, scaled for 8 proofs.
    scaled, _, log = native['loss_function'](args, batch, 4, 8, logits)
    scaled.backward()
    torch.testing.assert_close(logits.grad, gradient * 0.5)
    assert 'dis_masked' in log['keys']
    assert batch['log_probs'] is original_baseline
    assert not args.get_mismatch_metrics
    assert args.custom_tis_function_path is None


def test_pinned_parser_accepts_new_objective_and_isolates_role_args(native_loss):
    import argparse
    import copy
    import json
    import logging
    from ppo_async.training.launcher import build_train_command

    path = Path(os.environ['SLIME_ROOT']) / 'slime/utils/arguments.py'
    ns = dict(argparse=argparse, copy=copy, json=json, os=os, logger=logging.getLogger(__name__))
    for name in ('reset_arg', 'get_slime_extra_args_provider', '_apply_megatron_role_overrides'):
        _load_function(path, name, ns)
    parser = ns['get_slime_extra_args_provider']()(argparse.ArgumentParser())
    kwargs = dict(hf_checkpoint=Path('/model/hf'), torch_dist_checkpoint=Path('/model/dist'),
                  prompt_data=Path('/data/train'), role_config=Path('/roles.json'),
                  artifact_root=Path('/artifacts'), num_rollouts=50)
    base, base_unknown = parser.parse_known_args(build_train_command('async_ppo', **kwargs)[3:])
    new, new_unknown = parser.parse_known_args(build_train_command('async_ppo_dis_masking', **kwargs)[3:])
    assert new_unknown == base_unknown  # unchanged Megatron/SGLang flags
    assert {key for key in vars(new) if getattr(new, key) != getattr(base, key)} == {
        'loss_type', 'custom_loss_function_path',
    }
    assert new.loss_type == 'custom_loss' and new.use_rollout_logprobs
    assert not new.use_tis and not new.get_mismatch_metrics
    actor = ns['_apply_megatron_role_overrides'](new, {}, 'actor')
    critic = ns['_apply_megatron_role_overrides'](new, {}, 'critic')
    critic.loss_type = 'value_loss'  # native train_critic sets this before train()
    assert actor.loss_type == new.loss_type == 'custom_loss'
