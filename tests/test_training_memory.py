"""Explicit optimizer/DDP memory settings, without loading model weights."""
from copy import deepcopy
import json
import sys

import pytest
import torch
from torch import nn

from xqgeneral import training


def test_defaults_preserve_automatic_optimizer_and_separate_gradient_buckets():
    expected = {'optimizer_foreach': None, 'ddp_gradient_as_bucket_view': False}
    assert training.training_memory_settings({}) == expected
    assert training.training_memory_settings(expected) == expected
    training.compatible_resume({}, expected)


@pytest.mark.parametrize('change', [
    {'optimizer_foreach': 0}, {'optimizer_foreach': 1}, {'optimizer_foreach': 'false'},
    {'optimizer_foreach': []}, {'ddp_gradient_as_bucket_view': 0},
    {'ddp_gradient_as_bucket_view': 1}, {'ddp_gradient_as_bucket_view': None},
    {'ddp_gradient_as_bucket_view': 'true'},
])
def test_invalid_memory_settings_fail_before_cuda_or_model_loading(tmp_path, monkeypatch, change):
    path = tmp_path / 'config.json';path.write_text(json.dumps(change))
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(path)])
    monkeypatch.setenv('WORLD_SIZE', '4');monkeypatch.setenv('LOCAL_RANK', '0')
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid configuration reached CUDA/model loading')
    monkeypatch.setattr(torch.cuda, 'set_device', unexpected)
    monkeypatch.setattr(training, 'load_model', unexpected)
    with pytest.raises(ValueError, match='must be a boolean'):
        training.main()


@pytest.mark.parametrize('change', [
    {'optimizer_foreach': False}, {'optimizer_foreach': True},
    {'ddp_gradient_as_bucket_view': True},
])
def test_exact_resume_cannot_change_memory_settings(change):
    with pytest.raises(ValueError, match='Resume optimizer/DDP memory settings'):
        training.compatible_resume({}, change)
    with pytest.raises(ValueError, match='Resume optimizer/DDP memory settings'):
        training.compatible_resume(change, {})
    training.compatible_resume(change, deepcopy(change))


class TrainableGroups(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.base.board_weight = nn.Parameter(torch.eye(4))
        self.bridges = nn.ModuleList([nn.Linear(4, 4)])
        self.frozen = nn.Linear(4, 4).requires_grad_(False)

    def forward(self, x):
        return self.bridges[0](self.base(x) + x @ self.base.board_weight + self.frozen(x))


def test_scalar_adamw_updates_all_trainable_groups_and_resumes_bitwise():
    torch.manual_seed(19)
    continuous = TrainableGroups();resumed = deepcopy(continuous)
    config = {'learning_rate': .01, 'decoder_learning_rate': .001,
              'token_learning_rate': .002, 'weight_decay': .1, 'optimizer_foreach': False}
    optimizer = training.training_optimizer(continuous, config)
    resume_optimizer = training.training_optimizer(resumed, config)
    initial = deepcopy(continuous.state_dict())
    groups = {g['name']: g for g in optimizer.param_groups}
    assert {name: g['lr'] for name, g in groups.items()} == {'bridge': .01, 'decoder': .001, 'board': .002}
    assert all(g['foreach'] is False for g in groups.values())
    frozen = {id(p) for p in continuous.frozen.parameters()}
    assert not frozen & {id(p) for g in groups.values() for p in g['params']}
    def update(model, opt, step):
        opt.zero_grad(set_to_none=True)
        inputs = torch.arange(16, dtype=torch.float32).reshape(4, 4) / (step + 10)
        model(inputs).square().mean().backward()
        opt.step()
    for step in range(8):
        update(continuous, optimizer, step)
        if step < 4:update(resumed, resume_optimizer, step)
    checkpoint = {'model': deepcopy(resumed.state_dict()), 'optimizer': deepcopy(resume_optimizer.state_dict())}
    resumed = TrainableGroups();resumed.load_state_dict(checkpoint['model'])
    resume_optimizer = training.training_optimizer(resumed, config)
    resume_optimizer.load_state_dict(checkpoint['optimizer'])
    assert all(g['foreach'] is False for g in resume_optimizer.param_groups)
    for step in range(4, 8):update(resumed, resume_optimizer, step)
    for name, value in continuous.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
        assert torch.equal(value, initial[name]) if name.startswith('frozen.') else not torch.equal(value, initial[name])
    left, right = optimizer.state_dict(), resume_optimizer.state_dict()
    assert left['param_groups'] == right['param_groups']
    for index, state in left['state'].items():
        for name, value in state.items():assert torch.equal(value, right['state'][index][name])
