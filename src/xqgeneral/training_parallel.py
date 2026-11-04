"""Opt-in parameter/gradient sharding with portable, full CPU checkpoints.

All FSDP state operations are collective. The caller must enter them on every
rank, including validation, best-model reload and generation.
"""
from copy import deepcopy
import re

import torch
import torch.distributed as dist


def parallel_settings(config):
    strategy = config.get('distributed_strategy', 'ddp')
    if not isinstance(strategy, str) or strategy not in {'ddp', 'fsdp2'}:
        raise ValueError('distributed_strategy must be ddp or fsdp2')
    paths = config.get('fsdp_shard_paths')
    if strategy == 'ddp':
        if paths is not None:
            raise ValueError('fsdp_shard_paths requires distributed_strategy fsdp2')
        return {}
    if (not isinstance(paths, list) or not paths or
            any(not isinstance(p, str) or not p or
                any(not re.fullmatch(r'(?:[A-Za-z_]\w*|\d+|\*)', part) for part in p.split('.')) for p in paths) or
            len(set(paths)) != len(paths)):
        raise ValueError('fsdp_shard_paths must contain unique module paths with optional single-level stars')
    if config.get('optimizer_state_sharding', 'none') != 'none':
        raise ValueError('fsdp2 already shards Adam states; optimizer_state_sharding must be none')
    return {'distributed_strategy': strategy, 'fsdp_shard_paths': list(paths)}


def shard_modules(model, paths):
    """Resolve exact modules before mutation, preserving configuration order."""
    selected = {}
    for pattern in paths:
        matches = [('', model)]
        for part in pattern.split('.'):
            children = []
            for prefix, parent in matches:
                for name, module in parent.named_children():
                    if part == '*' or name == part:
                        children.append((f'{prefix}.{name}'.lstrip('.'), module))
            matches = children
        if not matches:
            raise ValueError(f'FSDP module path matched no modules: {pattern}')
        for name, module in matches:
            if name in selected or any(module is other for other in selected.values()):
                raise ValueError(f'FSDP module is selected more than once: {name}')
            selected[name] = module
    covered = {id(p) for module in selected.values() for p in module.parameters()}
    if covered != {id(p) for p in model.parameters()}:
        raise ValueError('FSDP paths must cover every model parameter')
    if any(p.ndim == 0 for p in model.parameters()):
        raise ValueError('FSDP parameter sharding requires tensors with at least one dimension')
    return sorted(selected.items(), key=lambda item: -item[0].count('.'))


def shard_training_model(model, config):
    settings = parallel_settings(config)
    if not settings:
        return model
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError('fsdp2 requires an initialized multi-rank process group')
    modules = shard_modules(model, settings['fsdp_shard_paths'])
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1:
        raise ValueError('FSDP setup requires one parameter device per rank')
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor
    device = next(iter(devices))
    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    for _, module in modules:
        fully_shard(module, mesh=mesh, reshard_after_forward=True)
    if not all(isinstance(p, DTensor) for p in model.parameters()):
        raise RuntimeError('FSDP left an unsharded parameter')
    return model


def parameter_prefix(parameter, length=1024):
    """A local diagnostic sample; never trigger a rank-zero-only all-gather."""
    from torch.distributed.tensor import DTensor
    value = parameter.detach()
    if isinstance(value, DTensor):
        value = value.to_local()
    return value.flatten()[:length].clone()


def _optimizer_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def indexed_optimizer_state(model, optimizer, named):
    """Retain the native integer-index Adam checkpoint format for consumers."""
    names = _optimizer_names(model, optimizer)
    if len(names) != len(named['param_groups']):
        raise ValueError('Optimizer checkpoint group count differs')
    result = {'state': {}, 'param_groups': []}
    next_index = 0
    seen = set()
    for expected, group in zip(names, named['param_groups'], strict=True):
        if group['params'] != expected:
            raise ValueError('Optimizer named parameter ordering differs')
        indices = list(range(next_index, next_index + len(expected)))
        next_index += len(expected)
        result['param_groups'].append({**{k: deepcopy(v) for k, v in group.items() if k != 'params'}, 'params': indices})
        for name, index in zip(expected, indices, strict=True):
            if name in named['state']:
                result['state'][index] = named['state'][name]
                seen.add(name)
    if seen != set(named['state']):
        raise ValueError('Optimizer state contains parameters outside the groups')
    return result


def named_optimizer_state(model, optimizer, indexed):
    names = _optimizer_names(model, optimizer)
    if len(names) != len(indexed['param_groups']):
        raise ValueError('Optimizer checkpoint group count differs')
    result = {'state': {}, 'param_groups': []}
    seen = set()
    all_indices = []
    for expected, group in zip(names, indexed['param_groups'], strict=True):
        if len(group['params']) != len(expected):
            raise ValueError('Optimizer checkpoint parameter count differs')
        all_indices.extend(group['params'])
        result['param_groups'].append({**{k: deepcopy(v) for k, v in group.items() if k != 'params'}, 'params': expected})
        for name, index in zip(expected, group['params'], strict=True):
            if index in indexed['state']:
                result['state'][name] = indexed['state'][index]
                seen.add(index)
    if (len(set(all_indices)) != len(all_indices) or seen != set(indexed['state']) or
            any(type(index) is not int for index in all_indices)):
        raise ValueError('Optimizer checkpoint indices are invalid')
    return result


def gather_sharded_state(model, optimizer=None):
    """All ranks enter; only rank zero receives full ordinary CPU tensors."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, StateDictOptions
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state = get_model_state_dict(model, options=options)
    trainable = {name: state[name] for name, p in model.named_parameters()
                 if p.requires_grad and name in state}
    if dist.get_rank() == 0 and set(trainable) != {n for n, p in model.named_parameters() if p.requires_grad}:
        raise ValueError('FSDP export is missing trainable parameters')
    moments = None
    if optimizer is not None:
        named = get_optimizer_state_dict(model, optimizer, options=options)
        if dist.get_rank() == 0:
            moments = indexed_optimizer_state(model, optimizer, named)
    return trainable, moments


def restore_sharded_optimizer(model, optimizer, indexed):
    """Load through DCP so full CPU moments never become full CUDA replicas."""
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict, StateDictOptions
    named = named_optimizer_state(model, optimizer, indexed)
    set_optimizer_state_dict(model, optimizer, named if dist.get_rank() == 0 else {},
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True))
    # DCP's broadcast flattens tuple-valued hyperparameters into lists. Preserve
    # the exact native checkpoint metadata as well as every moment value.
    for group, saved in zip(optimizer.param_groups, indexed['param_groups'], strict=True):
        for key, value in saved.items():
            if key != 'params':
                group[key] = deepcopy(value)


def load_sharded_trainable(model, checkpoint):
    from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    saved = checkpoint['trainable']
    if set(saved) != set(expected) or any(saved[n].shape != p.shape for n, p in expected.items()):
        raise ValueError('Checkpoint trainable parameter contract differs from model configuration')
    set_model_state_dict(model, saved if dist.get_rank() == 0 else {},
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True, strict=False))
