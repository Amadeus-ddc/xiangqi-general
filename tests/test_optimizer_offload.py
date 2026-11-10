"""Actual AdamW arithmetic, portable resume and opt-in memory contracts."""
from copy import deepcopy
import json
import sys

import pytest
import torch
from torch import nn

from xqgeneral import training
from xqgeneral.optimizer_offload import CPUOffloadedAdamW


class Groups(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(3, 4)
        self.bridges = nn.ModuleList([nn.Linear(4, 2)])
        self.board_weight = nn.Parameter(torch.ones(2))
        self.unused = nn.Parameter(torch.ones(2))
        self.frozen = nn.Parameter(torch.ones(3), requires_grad=False)

    def forward(self, x):
        return self.bridges[0](self.decoder(x)) + self.board_weight


def configuration(offload=True):
    return dict(learning_rate=.013, decoder_learning_rate=.002,
                token_learning_rate=.005, weight_decay=.1, optimizer_foreach=False,
                optimizer_state_offload='cpu' if offload else 'none')


def update(model, optimizer, step):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * (1 - step / 10)
    x = torch.arange(9, dtype=torch.float32).reshape(3, 3) / (step + 3)
    loss = model(x).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()


def assert_same(a, b):
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_same(x, y)
    else:
        assert a == b


def test_actual_native_adamw_all_groups_and_portable_4_vs_2plus2_resume():
    torch.manual_seed(19)
    reference = Groups(); offloaded = deepcopy(reference)
    native = training.training_optimizer(reference, configuration(False))
    staged = training.training_optimizer(offloaded, configuration())
    assert isinstance(staged, CPUOffloadedAdamW)
    initial = deepcopy(reference.state_dict())
    for step in range(4):
        update(reference, native, step); update(offloaded, staged, step)
        assert_same(reference.state_dict(), offloaded.state_dict())
        assert_same(native.state_dict(), staged.state_dict())
        assert staged.placement_summary()['all_states_on_CPU']
        if step == 1:
            saved = deepcopy(dict(model=offloaded.state_dict(), optimizer=staged.state_dict()))
    resumed = Groups(); resumed.load_state_dict(saved['model'])
    opt = training.training_optimizer(resumed, configuration())
    training.restore_optimizer_state(opt, saved['optimizer'], configuration(), 'cpu')
    for step in (2, 3): update(resumed, opt, step)
    assert_same(offloaded.state_dict(), resumed.state_dict())
    assert_same(staged.state_dict(), opt.state_dict())
    assert {g['name'] for g in opt.param_groups} == {'decoder', 'bridge', 'board'}
    assert torch.equal(initial['unused'], resumed.unused)
    assert torch.equal(initial['frozen'], resumed.frozen)
    assert opt.placement_summary()['all_moments_FP32']
    assert 0 < opt.peak_active_moment_bytes < opt.placement_summary()['total_state_bytes']


@pytest.mark.parametrize('change', [
    {'optimizer_state_offload': True}, {'optimizer_state_offload': 'disk'},
    {'optimizer_foreach': True}, {'optimizer_foreach': None},
    {'optimizer_state_sharding': 'zero1'}, {'ddp_world_size': 2},
    {'distributed_strategy': 'fsdp2'},
])
def test_unsupported_offload_contract_rejected(change):
    with pytest.raises(ValueError):
        training.training_memory_settings(dict(configuration(), **change))


def test_legacy_default_and_exact_resume_contract_preserved():
    assert training.training_memory_settings({}) == training.training_memory_settings({'optimizer_state_offload': 'none'})
    with pytest.raises(ValueError, match='Resume optimizer/DDP memory settings'):
        training.compatible_resume(configuration(False), configuration())
    training.compatible_resume(configuration(), configuration())


@pytest.mark.parametrize('bad_state', ['dtype', 'shape', 'unknown_owner', 'repeated_owner', 'missing_moment', 'foreach'])
def test_bad_checkpoint_refused_without_changing_live_optimizer(bad_state):
    model = Groups(); opt = training.training_optimizer(model, configuration())
    update(model, opt, 0); before = deepcopy(opt.state_dict()); saved = deepcopy(before)
    first = next(iter(saved['state']))
    if bad_state == 'dtype': saved['state'][first]['exp_avg'] = saved['state'][first]['exp_avg'].to(torch.bfloat16)
    elif bad_state == 'shape': saved['state'][first]['exp_avg_sq'] = torch.zeros(123)
    elif bad_state == 'unknown_owner': saved['state'][999999] = saved['state'][first]
    elif bad_state == 'missing_moment': del saved['state'][first]['exp_avg']
    elif bad_state == 'foreach': saved['param_groups'][0]['foreach'] = True
    else:
        group = next(g for g in saved['param_groups'] if len(g['params']) > 1)
        group['params'][1] = group['params'][0]
    with pytest.raises(ValueError): opt.load_state_dict(saved)
    assert_same(before, opt.state_dict())


def test_actual_multi_rank_launch_rejected_before_cuda(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'; path.write_text(json.dumps(configuration()))
    monkeypatch.setattr(sys, 'argv', ['training', '--config', str(path)])
    monkeypatch.setenv('WORLD_SIZE', '2'); monkeypatch.setenv('RANK', '0')
    def unexpected(*args, **kwargs): pytest.fail('Reserved-card CUDA initialization reached')
    monkeypatch.setattr(torch.cuda, 'set_device', unexpected)
    monkeypatch.setattr(training, 'load_model', unexpected)
    with pytest.raises(ValueError, match='one actual rank before CUDA'):
        training.main()


def test_only_full_fp32_parameters_are_accepted():
    with pytest.raises(ValueError, match='full FP32'):
        CPUOffloadedAdamW([nn.Parameter(torch.ones(2, dtype=torch.bfloat16))], foreach=False)


def test_closure_executes_once_and_groups_restore_after_error(monkeypatch):
    p = nn.Parameter(torch.ones(2)); opt = CPUOffloadedAdamW([p], foreach=False)
    calls = []
    def closure():
        calls.append(1); opt.zero_grad(); loss = p.square().sum(); loss.backward(); return loss
    assert float(opt.step(closure).detach()) == 2 and calls == [1]
    original = opt.param_groups[0]['params']
    def fail(*args, **kwargs): raise RuntimeError('CPU control intentional native error')
    monkeypatch.setattr(torch.optim.AdamW, 'step', fail)
    with pytest.raises(RuntimeError, match='intentional'): opt.step()
    assert opt.param_groups[0]['params'] is original and opt.placement_summary()['all_states_on_CPU']
