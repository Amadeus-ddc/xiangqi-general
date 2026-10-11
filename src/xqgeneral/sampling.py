"""Explicit stochastic decoding and scoped, resumable per-request seeds."""
from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math

import torch

SAMPLING_KEYS = {'do_sample', 'temperature', 'top_k', 'top_p'}
SEED_PROFILE = 'sha256_source_root_and_generation_ordinal_v1'


def sampling_options(value):
    if value is None:
        return {'do_sample': False}
    if not isinstance(value, dict) or set(value) != SAMPLING_KEYS or value.get('do_sample') is not True:
        raise ValueError('Explicit stochastic generation needs exactly do_sample, temperature, top_k and top_p')
    temperature, top_p, top_k = value['temperature'], value['top_p'], value['top_k']
    if (type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0 or
            type(top_p) not in (int, float) or not math.isfinite(top_p) or not 0 < top_p <= 1 or
            type(top_k) is not int or top_k < 0):
        raise ValueError('Invalid temperature, top-k or nucleus probability')
    return dict(do_sample=True, temperature=float(temperature), top_k=top_k, top_p=float(top_p))


def search_generation_policy(value):
    if not isinstance(value, dict) or set(value) != SAMPLING_KEYS | {'max_new_tokens'}:
        raise ValueError('Search generation requires explicit sampling and output-token settings')
    budget = value['max_new_tokens']
    if type(budget) is not int or budget <= 0:
        raise ValueError('Output-token budget must be a positive integer')
    return dict(sampling_options({k: value[k] for k in SAMPLING_KEYS}), max_new_tokens=budget)


def checked_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Generation seed must be a nonnegative signed 64-bit integer')
    return seed


def generation_seed(base_seed, source_root, call_index):
    checked_seed(base_seed)
    if (type(call_index) is not int or call_index < 0 or
            any(not isinstance(source_root.get(k), str) or not source_root[k] for k in ['id', 'feature_key'])):
        raise ValueError('A generation needs a source-root identity and nonnegative call ordinal')
    body = {'profile': SEED_PROFILE, 'seed': base_seed, 'root_id': source_root['id'],
            'root_feature_key': source_root['feature_key'], 'call_index': call_index}
    raw = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') & (2**63 - 1)


@contextmanager
def sample_rng(device, seed):
    if seed is None:
        yield
        return
    checked_seed(seed)
    device = torch.device(device)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Scoped sampling supports CPU or a single CUDA device')
    index = (torch.cuda.current_device() if device.index is None else device.index) if device.type == 'cuda' else None
    with torch.random.fork_rng(devices=[] if index is None else [index], device_type='cuda'):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        if index is not None:
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed)
        yield


def generation_software():
    try:
        transformers = version('transformers')
    except PackageNotFoundError:
        transformers = None
    return {'torch': str(torch.__version__), 'transformers': transformers}
