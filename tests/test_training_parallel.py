"""Collective FSDP2 checkpoint behavior on real CPU process groups."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from xqgeneral import training
from xqgeneral import training_parallel as parallel


def configuration():
    return dict(distributed_strategy='fsdp2', fsdp_shard_paths=['base', 'bridges.*'],
                learning_rate=.003, decoder_learning_rate=.001, optimizer_foreach=False)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(8, 12), nn.Dropout(.1), nn.Linear(12, 8))
        self.bridges = nn.ModuleList([nn.Linear(8, 8), nn.LayerNorm(8)])

    def forward(self, values):
        value = self.base(values)
        for bridge in self.bridges:
            value = bridge(value)
        return value


@pytest.mark.parametrize('value', [None, True, 'zero3'])
def test_invalid_strategy(value):
    with pytest.raises(ValueError, match='distributed_strategy'):
        parallel.parallel_settings({'distributed_strategy': value})


@pytest.mark.parametrize('paths', [None, [], 'base', [''], ['base', 'base'], ['base.**'], ['base..0'], [False]])
def test_invalid_shard_paths(paths):
    with pytest.raises(ValueError, match='fsdp_shard_paths'):
        parallel.parallel_settings(dict(distributed_strategy='fsdp2', fsdp_shard_paths=paths))


def test_legacy_defaults_and_exact_resume_contract():
    assert parallel.parallel_settings({}) == parallel.parallel_settings({'distributed_strategy': 'ddp'}) == {}
    with pytest.raises(ValueError, match='requires distributed_strategy'):
        parallel.parallel_settings({'fsdp_shard_paths': ['base']})
    config = configuration()
    with pytest.raises(ValueError, match='already shards'):
        parallel.parallel_settings(dict(config, optimizer_state_sharding='zero1'))
    with pytest.raises(ValueError, match='memory settings'):
        training.compatible_resume({}, config)
    with pytest.raises(ValueError, match='memory settings'):
        training.compatible_resume(config, dict(config, fsdp_shard_paths=['base.*', 'base', 'bridges.*']))
    training.compatible_resume(config, deepcopy(config))


def test_module_selection_is_complete_bottom_up_and_single_level():
    model = Network()
    found = parallel.shard_modules(model, ['base.*', 'base', 'bridges.*'])
    assert [name for name, _ in found] == ['base.0', 'base.1', 'base.2', 'bridges.0', 'bridges.1', 'base']
    with pytest.raises(ValueError, match='no modules'):
        parallel.shard_modules(model, ['missing', 'base', 'bridges.*'])
    with pytest.raises(ValueError, match='more than once'):
        parallel.shard_modules(model, ['base.*', 'base.0', 'bridges.*'])
    with pytest.raises(ValueError, match='every model parameter'):
        parallel.shard_modules(model, ['base'])
    model.base.scalar = nn.Parameter(torch.tensor(1.))
    with pytest.raises(ValueError, match='at least one dimension'):
        parallel.shard_modules(model, ['base', 'bridges.*'])


def test_fsdp_requires_a_multi_rank_group():
    assert not dist.is_initialized()
    with pytest.raises(ValueError, match='initialized multi-rank'):
        parallel.shard_training_model(Network(), configuration())


def test_native_index_mapping_preserves_groups_and_moments():
    model = Network()
    optimizer = training.training_optimizer(model, configuration())
    model(torch.randn(2, 8)).square().mean().backward()
    optimizer.step()
    indexed = deepcopy(optimizer.state_dict())
    named = parallel.named_optimizer_state(model, optimizer, indexed)
    rebuilt = parallel.indexed_optimizer_state(model, optimizer, named)
    assert rebuilt['param_groups'] == indexed['param_groups']
    assert rebuilt['state'].keys() == indexed['state'].keys()
    for index, values in indexed['state'].items():
        for name, value in values.items():
            assert torch.equal(value, rebuilt['state'][index][name])
    reordered = deepcopy(named)
    reordered['param_groups'][0]['params'].reverse()
    with pytest.raises(ValueError, match='ordering'):
        parallel.indexed_optimizer_state(model, optimizer, reordered)
    malformed = deepcopy(indexed)
    malformed['state'][999] = next(iter(malformed['state'].values()))
    with pytest.raises(ValueError, match='indices'):
        parallel.named_optimizer_state(model, optimizer, malformed)
    prefix = parallel.parameter_prefix(next(model.parameters()), 3)
    assert prefix.numel() == 3 and not prefix.requires_grad


def same(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def worker(rank, home):
    from torch.distributed.tensor import DTensor
    root = Path(home)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method=(root / 'group').as_uri(), timeout=timedelta(seconds=60))
    try:
        def build():
            torch.manual_seed(231)
            model = parallel.shard_training_model(Network(), configuration())
            assert all(isinstance(p, DTensor) and p.dtype == torch.float32 for p in model.parameters())
            return model, training.training_optimizer(model, configuration())
        def update(model, optimizer, step):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                gen = torch.Generator().manual_seed(137 + step * 4 + micro + 17 * rank)
                loss = model(torch.randn(3, 8, generator=gen)).square().mean() / 2
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).full_tensor()
            assert torch.isfinite(norm)
            optimizer.step()
        model, optimizer = build()
        torch.manual_seed(211 + rank)
        for step in range(4):
            update(model, optimizer, step)
            if step == 1:
                weights, moments = parallel.gather_sharded_state(model, optimizer)
                states = [None] * 2
                dist.all_gather_object(states, torch.get_rng_state())
                if rank == 0:
                    assert all(type(v) is torch.Tensor and v.device.type == 'cpu' for v in weights.values())
                    torch.save(dict(trainable=weights, optimizer=moments, rngs=states), root / 'saved.pt')
                dist.barrier()
        expected_model, expected_optimizer = parallel.gather_sharded_state(model, optimizer)
        expected_rng = torch.get_rng_state().clone()
        resumed, opt = build()
        saved = torch.load(root / 'saved.pt', weights_only=True)
        parallel.load_sharded_trainable(resumed, saved)
        parallel.restore_sharded_optimizer(resumed, opt, saved['optimizer'])
        torch.set_rng_state(saved['rngs'][rank])
        for step in range(2, 4):
            update(resumed, opt, step)
        actual_model, actual_optimizer = parallel.gather_sharded_state(resumed, opt)
        assert torch.equal(expected_rng, torch.get_rng_state())
        if rank == 0:
            assert same(expected_model, actual_model)
            assert same(expected_optimizer, actual_optimizer)
            assert all(type(i) is int for i in actual_optimizer['state'])
        local = parallel.parameter_prefix(next(resumed.parameters()))
        assert type(local) is torch.Tensor
    finally:
        dist.destroy_process_group()


def test_real_two_rank_cpu_fsdp_full_state_and_exact_resume(tmp_path):
    mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=True)
