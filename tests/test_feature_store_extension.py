import json
from pathlib import Path

import pytest
import torch

from xqgeneral import cache_feature_store
from xqgeneral.evidence import digest
from xqgeneral.feature_store import FeatureStore, FeatureStoreWriter, feature_paths
from test_feature_store import block, fake_expert, histories, prepared


@pytest.mark.parametrize('legacy_ancestor', [False, True])
def test_two_rounds_reference_all_old_shards_and_preserve_every_value(tmp_path, legacy_ancestor):
    base, path, _, original = prepared(tmp_path, base=legacy_ancestor)
    before = {str(p): digest(p) for p in feature_paths(path)}
    expected = original
    for number, keys in enumerate([['new-1', 'new-2'], ['new-3']]):
        writer = FeatureStoreWriter(tmp_path / f'round-{number}', [0, 3], 'a' * 64, expert_dim=8)
        old_paths = feature_paths(path)
        writer.add_existing(path, path)
        added = block(keys, 31 + number)
        writer.append(added)
        result = writer.finish()
        assert result['base_roots'] == len(expected['keys'])
        assert result['new_roots'] == len(keys)
        expected = {'keys': [*expected['keys'], *keys], 'depths': [0, 3],
                    'features': [torch.cat([a, b]) for a, b in zip(expected['features'], added['features'])],
                    'wdl': torch.cat([expected['wdl'], added['wdl']])}
        new_path = writer.output / 'manifest.json'
        store = FeatureStore(new_path, max_open_shards=1)
        assert store.keys == expected['keys']
        assert [s['path'] for s in store.shards[:-1]] == [str(p) for p in old_paths[2:]]
        assert len(list(writer.output.glob('*.pt'))) == 1
        assert digest(path) == writer.inputs[str(path)]['sha256']
        assert all(torch.equal(a[:], b) for a, b in zip(store.features, expected['features']))
        assert torch.equal(store.wdl[:], expected['wdl'])
        indices = [len(store.keys) - 1, 0, 3, 0, len(store.keys) - 2]
        for a, b in zip(store.features, expected['features']):
            assert torch.equal(a[indices], b[indices])
        assert len(store._open) == 1
        path = new_path
    assert all(digest(p) == sha for p, sha in before.items())


@pytest.mark.parametrize('problem', ['expert', 'encoder', 'depth', 'width', 'status', 'separate_proof'])
def test_extension_rejects_incompatible_or_unfinished_base(tmp_path, problem):
    _, path, _, _ = prepared(tmp_path)
    proof = json.loads(path.read_text())
    before = digest(path)
    proof_path = path
    if problem == 'expert':
        proof['verification']['expert_sha256'] = 'b' * 64
    elif problem == 'encoder':
        proof['code']['source_sha256']['src/xqgeneral/expert.py'] = 'b' * 64
    elif problem == 'status':
        proof['status'] = 'building'
    elif problem == 'separate_proof':
        proof_path = tmp_path / 'copied-proof.json'
        proof_path.write_text(path.read_text())
    if problem in {'expert', 'encoder', 'status'}:
        path.write_text(json.dumps(proof))
    writer = FeatureStoreWriter(tmp_path / 'extension', [0, 4] if problem == 'depth' else [0, 3],
                                'a' * 64, expert_dim=9 if problem == 'width' else 8)
    with pytest.raises(ValueError):
        writer.add_existing(path, proof_path)
    assert not (writer.output / 'manifest.json').exists()
    if problem not in {'expert', 'encoder', 'status'}:
        assert digest(path) == before


@pytest.mark.parametrize('artifact', ['keys', 'shard', 'manifest'])
def test_extension_rejects_base_changed_before_or_after_import(tmp_path, artifact):
    _, path, store, _ = prepared(tmp_path)
    target = path if artifact == 'manifest' else Path(store.proof['catalog']['keys']) if artifact == 'keys' else Path(store.shards[1]['path'])
    writer = FeatureStoreWriter(tmp_path / 'extension', [0, 3], 'a' * 64, expert_dim=8)
    writer.add_existing(path, path)
    raw = bytearray(target.read_bytes())
    raw[-2] ^= 1
    target.write_bytes(raw)
    with pytest.raises(ValueError):
        writer.finish()
    assert not (writer.output / 'manifest.json').exists()
    another = FeatureStoreWriter(tmp_path / 'another', [0, 3], 'a' * 64, expert_dim=8)
    with pytest.raises(ValueError):
        another.add_existing(path, path)


def test_extension_checks_actual_shard_keys_even_with_fresh_byte_bindings(tmp_path):
    _, path, store, _ = prepared(tmp_path)
    shard = Path(store.shards[1]['path'])
    value = torch.load(shard, weights_only=True)
    value['keys'] = ['wrong-history']
    torch.save(value, shard)
    proof = json.loads(path.read_text())
    proof['outputs'][str(shard)] = {'sha256': digest(shard), 'bytes': shard.stat().st_size}
    path.write_text(json.dumps(proof))
    writer = FeatureStoreWriter(tmp_path / 'extension', [0, 3], 'a' * 64, expert_dim=8)
    with pytest.raises(ValueError, match='ordered catalog'):
        writer.add_existing(path, path)
    assert not (writer.output / 'manifest.json').exists()


def test_cached_source_extension_never_reloads_expert_or_copies_tensor_files(tmp_path, monkeypatch):
    rows, source, weights, proof = histories(tmp_path)
    calls = []
    fake_expert(monkeypatch, calls)
    cache_feature_store.build([source], proof, weights, [0, 3], tmp_path / 'first',
                              batch_size=2, shard_rows=3, device='cpu', cpu_threads=1)
    path = tmp_path / 'first/manifest.json'
    before = {str(p): digest(p) for p in feature_paths(path)}
    calls.clear()
    result = cache_feature_store.build([source], proof, weights, [0, 3], tmp_path / 'second',
                                       base_cache=path, base_proof=path, device='cpu', cpu_threads=1)
    assert result['base_roots'] == len(rows) and result['new_roots'] == 0
    assert not result['expert_loaded'] and calls == []
    assert not list((tmp_path / 'second').glob('*.pt'))
    assert all(digest(p) == sha for p, sha in before.items())
    first, second = FeatureStore(path), FeatureStore(tmp_path / 'second/manifest.json')
    assert first.keys == second.keys
    assert all(torch.equal(a[:], b[:]) for a, b in zip(first.features, second.features))
