"""Immutable mmap feature shards behind the existing cache lookup interface."""
from bisect import bisect_right
from collections import OrderedDict
import json
import operator
from pathlib import Path

import torch

from .evidence import atomic_json, digest, file_signature, manifest

PROFILE = 'sharded_expert_features_xiangqi_v1'
KIND = 'sharded_expert_feature_cache'


def is_feature_store(path):
    return Path(path).suffix == '.json'


def feature_proof_path(path):
    path = Path(path)
    return path if is_feature_store(path) else path.with_suffix('.manifest.json')


def feature_paths(path):
    """Every storage file needed by a checkpoint, including the complete catalog."""
    original, path = path, Path(path)
    if not is_feature_store(path):
        return [original]
    proof = json.loads(path.read_text())
    if (proof.get('status') != 'complete' or proof.get('kind') != KIND or
            proof.get('config', {}).get('feature_store_profile') != PROFILE):
        raise ValueError('Feature store is incomplete or has an unsupported profile')
    return [original, Path(proof['catalog']['keys']), *[Path(s['path']) for s in proof['catalog']['shards']]]


def _validate(cache, depths, width):
    keys = cache['keys']
    if (not keys or any(not isinstance(k, str) or not k for k in keys) or
            len(keys) != len(set(keys)) or cache['depths'] != depths or
            len(cache['features']) != len(depths)):
        raise ValueError('Feature shard keys or depth contract differ')
    if any(f.shape != (len(keys), 90, width) or f.dtype != torch.float16 for f in cache['features']):
        raise ValueError('Feature shards require matching FP16 90-square tensors')
    if cache['wdl'].shape != (len(keys), 3) or cache['wdl'].dtype != torch.float32:
        raise ValueError('Feature shard WDL dimensions or precision differ')


def _binding(path):
    return {'sha256': digest(path), 'bytes': Path(path).stat().st_size}


class FeatureStoreWriter:
    """Append bounded batches or reference an already verified immutable cache."""
    def __init__(self, output, depths, expert_sha256, *, expert_dim=512, config=None):
        if (not depths or any(type(d) is not int or d < 0 for d in depths) or
                sorted(set(depths)) != list(depths) or type(expert_dim) is not int or expert_dim < 1):
            raise ValueError('Ordered distinct depths and a positive expert width are required')
        if not isinstance(expert_sha256, str) or len(expert_sha256) != 64:
            raise ValueError('A pinned expert SHA256 is required')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.depths, self.width, self.expert_sha = list(depths), expert_dim, expert_sha256
        self.keys, self.seen, self.shards = [], set(), []
        self.inputs, self.outputs, self.signatures = {}, {}, {}
        self.config, self.base_roots, self.finished = dict(config or {}), 0, False
        atomic_json(self.output / 'progress.json', {'status': 'building', 'roots': 0, 'shards': 0})

    def _add(self, path, cache, *, existing=False):
        if self.finished:
            raise ValueError('A completed feature store is immutable')
        _validate(cache, self.depths, self.width)
        if self.seen.intersection(cache['keys']):
            raise ValueError('Feature shards contain overlapping history keys')
        path = Path(path).resolve()
        binding = _binding(path)
        self.shards.append({'path': str(path), 'start': len(self.keys), 'rows': len(cache['keys'])})
        self.keys.extend(cache['keys']); self.seen.update(cache['keys'])
        (self.inputs if existing else self.outputs)[str(path)] = binding
        self.signatures[str(path)] = file_signature(path)
        atomic_json(self.output / 'progress.json', {'status': 'building', 'roots': len(self.keys),
                                                   'shards': len(self.shards)})

    def add_existing(self, path, proof_path):
        if self.shards or self.finished:
            raise ValueError('The preserved base cache must precede every new shard')
        path, proof_path = Path(path).resolve(), Path(proof_path).resolve()
        proof = json.loads(proof_path.read_text())
        bindings = {str(Path(p).resolve()): b for p, b in proof.get('outputs', {}).items()}
        if (proof.get('status') != 'complete' or
                proof.get('verification', {}).get('expert_sha256') != self.expert_sha or
                proof.get('code', {}).get('source_sha256', {}).get('src/xqgeneral/expert.py') !=
                digest(Path(__file__).with_name('expert.py'))):
            raise ValueError('Preserved expert weights or encoder contract differ')
        before = file_signature(path)
        cache = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        self._add(path, cache, existing=True)
        if self.inputs[str(path)] != bindings.get(str(path)) or file_signature(path) != before:
            raise ValueError('Preserved base cache bytes changed')
        self.inputs[str(proof_path)] = _binding(proof_path)
        self.signatures[str(proof_path)] = file_signature(proof_path)
        self.base_roots = len(self.keys)
        self.config.update(base_cache=str(path), base_proof=str(proof_path))

    def append(self, cache):
        if self.finished:
            raise ValueError('A completed feature store is immutable')
        _validate(cache, self.depths, self.width)
        if (self.seen.intersection(cache['keys']) or
                any(not torch.isfinite(f).all().item() for f in [*cache['features'], cache['wdl']])):
            raise ValueError('New feature shard contains duplicate histories or nonfinite values')
        path = self.output / f'shard-{len(self.shards):06d}.pt'
        partial = path.with_suffix('.pt.partial')
        if path.exists() or partial.exists():
            raise FileExistsError('Preserve existing feature shards; use a fresh store')
        torch.save(cache, partial); partial.replace(path)
        self._add(path, cache)

    def finish(self, *, source_bindings=None, source_signatures=None):
        if self.finished or not self.keys:
            raise ValueError('Cannot complete an empty or already completed feature store')
        for name, expected in (source_signatures or {}).items():
            if file_signature(name) != tuple(expected):
                raise ValueError('Feature source changed during extraction')
        for name, expected in self.signatures.items():
            if file_signature(name) != expected:
                raise ValueError('Feature shard or preserved proof changed during construction')
        keys_path = self.output / 'keys.json'
        atomic_json(keys_path, self.keys)
        self.outputs[str(keys_path)] = _binding(keys_path)
        self.signatures[str(keys_path)] = file_signature(keys_path)
        config = {**self.config, 'feature_store_profile': PROFILE, 'depths': self.depths,
                  'expert_dim': self.width}
        verification = {'status': 'complete', 'roots': len(self.keys), 'base_roots': self.base_roots,
                        'new_roots': len(self.keys) - self.base_roots, 'depths': self.depths,
                        'expert_sha256': self.expert_sha, 'all_history_keys_unique': True,
                        'existing_base_cache_referenced_without_tensor_copy': bool(self.base_roots),
                        'stored_feature_dtype': 'float16', 'stored_wdl_dtype': 'float32',
                        'storage_signatures': {p: list(s) for p, s in self.signatures.items()},
                        'student_training_executed': False}
        proof = manifest(KIND, config, verification=verification)
        proof['inputs'] = {**(source_bindings or {}), **self.inputs}
        proof['outputs'] = self.outputs
        proof['catalog'] = {'keys': str(keys_path), 'shards': self.shards}
        for name, expected in {**(source_signatures or {}), **self.signatures}.items():
            if file_signature(name) != tuple(expected):
                raise ValueError('Feature source or shard changed before publication')
        atomic_json(self.output / 'manifest.json', proof)
        atomic_json(self.output / 'progress.json', {'status': 'complete', 'roots': len(self.keys),
                                                   'shards': len(self.shards)})
        self.finished = True
        return verification


class _Level:
    def __init__(self, store, level):
        self.store, self.level = store, level
        self.dtype = torch.float32 if level is None else torch.float16
        self.shape = (len(store.keys), 3) if level is None else (len(store.keys), 90, store.width)

    def __getitem__(self, indices):
        return self.store._select(self.level, indices)


class FeatureStore:
    """Cache mapping with bounded mmap handles and original ordered tensor lookup."""
    def __init__(self, path, *, verified_hashes=None, max_open_shards=8):
        path = Path(path).resolve()
        before = file_signature(path)
        paths = feature_paths(path)
        if type(max_open_shards) is not int or max_open_shards < 1:
            raise ValueError('Positive mmap shard handle budget required')
        hashes = None if verified_hashes is None else {str(Path(p).resolve()): h for p, h in verified_hashes.items()}
        if hashes is not None and hashes.get(str(path)) != digest(path):
            raise ValueError('Feature store manifest is unverified or changed')
        self.proof = json.loads(path.read_text())
        p = self.proof
        self._signatures = {str(f.resolve()): file_signature(f) for f in paths}
        if self._signatures[str(path)] != before:
            raise ValueError('Feature store manifest changed while opening')
        expected = {str(Path(n).resolve()): b for n, b in {**p['inputs'], **p['outputs']}.items()}
        if len(set(str(f.resolve()) for f in paths)) != len(paths):
            raise ValueError('Feature store contains duplicate storage paths')
        for f in paths[1:]:
            name = str(f.resolve()); binding = expected.get(name)
            if binding is None:
                raise ValueError('Feature storage is missing its byte binding')
            actual = digest(f) if hashes is None else hashes.get(name)
            if (hashes is not None and list(self._signatures[name]) !=
                    p['verification'].get('storage_signatures', {}).get(name)):
                actual = digest(f)
            if actual != binding['sha256'] or f.stat().st_size != binding['bytes']:
                raise ValueError('Feature storage bytes are changed or unverified')
        self.keys = json.loads(Path(p['catalog']['keys']).read_text())
        self.depths, self.width = p['config']['depths'], p['config']['expert_dim']
        if (not isinstance(self.keys, list) or not self.keys or
                any(not isinstance(k, str) or not k for k in self.keys) or len(set(self.keys)) != len(self.keys) or
                not self.depths or any(type(d) is not int or d < 0 for d in self.depths) or
                sorted(set(self.depths)) != self.depths or type(self.width) is not int or self.width < 1 or
                len(self.keys) != p['verification']['roots']):
            raise ValueError('Feature store keys, depths or width contract differ')
        self.shards, self.starts, count = p['catalog']['shards'], [], 0
        for shard in self.shards:
            if (type(shard['start']) is not int or shard['start'] != count or
                    type(shard['rows']) is not int or shard['rows'] < 1):
                raise ValueError('Feature shard row ranges are noncontiguous')
            self.starts.append(count); count += shard['rows']
        if count != len(self.keys):
            raise ValueError('Feature shard ranges do not cover the full key catalog')
        self.features = [_Level(self, i) for i in range(len(self.depths))]
        self.wdl = _Level(self, None)
        self._open, self.max_open = OrderedDict(), max_open_shards
        self.check_unchanged()

    def __getitem__(self, name):
        return {'keys': self.keys, 'depths': self.depths, 'features': self.features, 'wdl': self.wdl}[name]

    def check_unchanged(self, paths=None):
        paths = self._signatures if paths is None else paths
        if any(file_signature(p) != self._signatures[str(p)] for p in paths):
            raise ValueError('Feature store changed after verification')

    def _cache(self, number):
        if number not in self._open:
            shard = self.shards[number]
            cache = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)
            _validate(cache, self.depths, self.width)
            if cache['keys'] != self.keys[shard['start']:shard['start'] + shard['rows']]:
                raise ValueError('Feature shard keys differ from the full ordered catalog')
            self._open[number] = cache
        self._open.move_to_end(number)
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return self._open[number]

    def select(self, indices, levels=None):
        """Gather every requested depth while opening each relevant shard only once."""
        levels = list(range(len(self.depths))) if levels is None else levels
        scalar = False
        if isinstance(indices, slice):
            indices = list(range(*indices.indices(len(self.keys))))
        else:
            try:
                indices = [operator.index(indices)]; scalar = True
            except TypeError:
                indices = list(indices)
        groups = {}
        for position, index in enumerate(indices):
            index = operator.index(index)
            if index < 0: index += len(self.keys)
            if not 0 <= index < len(self.keys):
                raise IndexError('Feature history index is out of range')
            number = bisect_right(self.starts, index) - 1
            positions, rows = groups.setdefault(number, ([], []))
            positions.append(position); rows.append(index - self.starts[number])
        paths = [*list(self._signatures)[:2], *[self.shards[n]['path'] for n in groups]]
        self.check_unchanged(paths)
        results = [torch.empty((len(indices), 3) if level is None else (len(indices), 90, self.width),
                               dtype=torch.float32 if level is None else torch.float16) for level in levels]
        for number, (positions, rows) in groups.items():
            cache = self._cache(number)
            for level, result in zip(levels, results, strict=True):
                value = cache['wdl'] if level is None else cache['features'][level]
                result[positions] = value[rows]
        self.check_unchanged(paths)
        return [result[0] if scalar else result for result in results]

    def _select(self, level, indices):
        return self.select(indices, [level])[0]


def open_feature_cache(path, *, mmap=False, verified_hashes=None):
    if is_feature_store(path):
        return FeatureStore(path, verified_hashes=verified_hashes)
    return torch.load(path, map_location='cpu', weights_only=True, mmap=mmap)


def gather_features(cache, indices, device):
    features = cache.select(indices) if isinstance(cache, FeatureStore) else [f[indices] for f in cache['features']]
    return [f.to(device=device, dtype=torch.bfloat16) for f in features]
