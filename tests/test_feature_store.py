import json
from pathlib import Path

import pytest
import torch

from xqgeneral.evidence import digest, history_key, manifest, write_jsonl
from xqgeneral.feature_store import (
    FeatureStore, FeatureStoreWriter, feature_paths, feature_proof_path, gather_features, open_feature_cache,
)
from xqgeneral.foundation_readback import checked_preserved_cache
from xqgeneral.inference import Predictor
from xqgeneral.training import select_features
from xqgeneral import cache_feature_store
from xqgeneral.rules import START_FEN, replay


def block(keys, start=0, width=8):
    f = (torch.arange(len(keys) * 90 * width).reshape(len(keys), 90, width).float() / 137 + start).half()
    return {'keys': list(keys), 'depths': [0, 3], 'features': [f, (f.float() / 3).half()],
            'wdl': torch.arange(len(keys) * 3).reshape(len(keys), 3).float() / 9 + start}


def prepared(tmp_path, *, base=False, width=8):
    writer = FeatureStoreWriter(tmp_path / 'store', [0, 3], 'a' * 64, expert_dim=width)
    blocks = [block(['旧局面', 'a'], width=width), block(['b'], 4, width), block(['c', 'd', 'e'], 9, width)]
    if base:
        path = tmp_path / 'base.pt'; torch.save(blocks[0], path)
        proof = manifest('expert_feature_cache', {}, [Path(cache_feature_store.__file__).with_name('expert.py')],
                         [path], {'expert_sha256': 'a' * 64})
        proof_path = tmp_path / 'base.manifest.json'; proof_path.write_text(json.dumps(proof))
        writer.add_existing(path, proof_path)
    else:
        writer.append(blocks[0])
    for value in blocks[1:]: writer.append(value)
    writer.finish()
    expected = {'keys': sum((v['keys'] for v in blocks), []), 'depths': [0, 3],
                'features': [torch.cat([v['features'][i] for v in blocks]) for i in range(2)],
                'wdl': torch.cat([v['wdl'] for v in blocks])}
    path = writer.output / 'manifest.json'
    return writer, path, FeatureStore(path, max_open_shards=1), expected


@pytest.mark.parametrize('base', [False, True])
def test_every_feature_and_wdl_value_matches_original_cache_across_shards(tmp_path, base):
    writer, path, store, expected = prepared(tmp_path, base=base)
    assert store['keys'] == expected['keys'] and store['depths'] == [0, 3]
    for level, value in zip(store['features'], expected['features'], strict=True):
        assert level.shape == value.shape and level.dtype == value.dtype
        for indices in [[5, 0, 3, 0, 2], slice(None), slice(None, None, -2), -1, [], torch.tensor([4, 1])]:
            assert torch.equal(level[indices], value[indices] if not isinstance(indices, slice) or indices.step != -2
                               else value[torch.arange(len(value)-1, -1, -2)])
    assert torch.equal(store['wdl'][:], expected['wdl'])
    assert len(store._open) <= 1
    assert feature_proof_path(path) == path
    if base:
        assert not (writer.output / 'shard-000000.pt').exists()
        assert store.proof['verification']['base_roots'] == 2
        assert checked_preserved_cache(store, block(['旧局面', 'a'])) == 2


@pytest.mark.parametrize('zero,rotate', [(False, False), (True, False), (False, True)])
def test_training_and_inference_feature_lookup_are_bitwise_equal(tmp_path, zero, rotate):
    writer, path, store, expected = prepared(tmp_path)
    indices = {key: i for i, key in enumerate(expected['keys'])}
    rows = [{'feature_key': k} for k in ['e', '旧局面', 'b', '旧局面']]
    actual = select_features(store, indices, rows, 'cpu', zero=zero, rotate=rotate)
    reference = select_features(expected, indices, rows, 'cpu', zero=zero, rotate=rotate)
    assert all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))


@pytest.mark.parametrize('include_history', [False, True])
def test_cached_predictor_reads_shards_without_calling_live_expert(tmp_path, include_history):
    history = replay(START_FEN, ['b0c2']);key = history_key(history)
    value = block(['unrelated', key], width=512)
    writer = FeatureStoreWriter(tmp_path/'store', [0,3], 'a'*64)
    writer.append(value);writer.finish()
    predictor = Predictor.__new__(Predictor)
    predictor.cache = FeatureStore(writer.output/'manifest.json')
    predictor.indices = {k:i for i,k in enumerate(value['keys'])}
    predictor.device = 'cpu';predictor.config = {'expert_feature_depths':[0,3]}
    class Expert:
        def __call__(self,*args,**kwargs):pytest.fail('Cached inference unnecessarily loaded live expert features')
    predictor.expert = Expert()
    row = {'initial_fen':START_FEN,'moves':['b0c2'],'fen':history[-1],'feature_key':key}
    if include_history:row['history']=history
    expected = select_features(value,predictor.indices,[row],'cpu')
    assert all(torch.equal(a,b) for a,b in zip(predictor.features(row),expected,strict=True))


def test_all_depths_gather_each_relevant_shard_once_and_only_check_used_storage(tmp_path, monkeypatch):
    writer, path, store, expected = prepared(tmp_path)
    calls, checked = [], []
    original = store._cache
    monkeypatch.setattr(store, '_cache', lambda i: (calls.append(i), original(i))[1])
    guard = store.check_unchanged
    monkeypatch.setattr(store, 'check_unchanged', lambda paths=None: (checked.append(paths), guard(paths))[1])
    actual = gather_features(store, [3, 0, 3], 'cpu')
    assert calls == [2, 0]
    assert all(len(paths) == 4 for paths in checked)
    assert all(torch.equal(a, v[[3, 0, 3]].bfloat16()) for a, v in zip(actual, expected['features']))


@pytest.mark.parametrize('index', [6, -7])
def test_out_of_range_history_lookup_fails(tmp_path, index):
    writer, path, store, expected = prepared(tmp_path)
    with pytest.raises(IndexError): store['features'][0][index]


@pytest.mark.parametrize('artifact', ['keys', 'shard', 'manifest'])
@pytest.mark.parametrize('provided_hashes', [False, True])
def test_same_length_storage_change_rejects_even_stale_caller_hashes(tmp_path, artifact, provided_hashes):
    writer, path, store, expected = prepared(tmp_path)
    hashes = {str(p): digest(p) for p in feature_paths(path)}
    target = path if artifact == 'manifest' else Path(store.proof['catalog']['keys']) if artifact == 'keys' else Path(store.shards[1]['path'])
    raw = bytearray(target.read_bytes()); raw[-2] ^= 1; target.write_bytes(raw)
    with pytest.raises(ValueError):
        FeatureStore(path, verified_hashes=hashes if provided_hashes else None)
    with pytest.raises(ValueError): store.check_unchanged()


def test_changes_to_requested_shards_reject_before_returning_a_batch(tmp_path):
    writer, path, store, expected = prepared(tmp_path)
    p = Path(store.shards[2]['path']); raw = bytearray(p.read_bytes());raw[-1] ^= 1;p.write_bytes(raw)
    with pytest.raises(ValueError, match='changed after verification'): gather_features(store, [4], 'cpu')


@pytest.mark.parametrize('field,value', [('start', 1), ('rows', 0), ('rows', 7)])
def test_corrupt_catalog_cannot_cover_false_row_ranges(tmp_path, field, value):
    writer, path, store, expected = prepared(tmp_path)
    p = json.loads(path.read_text());p['catalog']['shards'][0][field] = value;path.write_text(json.dumps(p))
    with pytest.raises(ValueError): FeatureStore(path)


def test_incomplete_and_empty_stores_cannot_be_used(tmp_path):
    writer = FeatureStoreWriter(tmp_path / 'store', [0], 'a' * 64)
    with pytest.raises(ValueError): writer.finish()
    with pytest.raises(FileNotFoundError): open_feature_cache(writer.output / 'manifest.json')
    with pytest.raises(FileExistsError): FeatureStoreWriter(tmp_path / 'store', [0], 'a' * 64)


@pytest.mark.parametrize('field', ['duplicate', 'depth', 'feature_shape', 'feature_dtype', 'wdl', 'nan', 'inf'])
def test_invalid_feature_batches_never_publish_a_complete_store(tmp_path, field):
    writer = FeatureStoreWriter(tmp_path / 'store', [0, 3], 'a' * 64, expert_dim=8)
    value = block(['a', 'b'])
    if field == 'duplicate': value['keys'] = ['a', 'a']
    elif field == 'depth': value['depths'] = [0, 2]
    elif field == 'feature_shape': value['features'][0] = value['features'][0][:, :89]
    elif field == 'feature_dtype': value['features'][0] = value['features'][0].float()
    elif field == 'wdl': value['wdl'] = value['wdl'].half()
    else: value['features'][0][0, 0, 0] = float(field)
    with pytest.raises(ValueError): writer.append(value)
    assert not (writer.output / 'manifest.json').exists()


def test_duplicate_shards_and_finished_writers_are_immutable(tmp_path):
    writer, path, store, expected = prepared(tmp_path)
    before = digest(path)
    with pytest.raises(ValueError): writer.append(block(['new']))
    with pytest.raises(ValueError): writer.finish()
    assert digest(path) == before
    writer2 = FeatureStoreWriter(tmp_path / 'second', [0, 3], 'a' * 64, expert_dim=8)
    writer2.append(block(['a']))
    with pytest.raises(ValueError): writer2.append(block(['a']))
    assert len(writer2.shards) == 1 and not (writer2.output / 'manifest.json').exists()


def test_legacy_cache_loading_and_sidecar_identity_are_unchanged(tmp_path):
    path = tmp_path / 'legacy.pt';value = block(['a', 'b']);torch.save(value, path)
    assert feature_paths(path) == [path]
    assert feature_proof_path(path) == tmp_path / 'legacy.manifest.json'
    loaded = open_feature_cache(path, mmap=True)
    assert loaded['keys'] == value['keys']
    assert all(torch.equal(a, b) for a, b in zip(loaded['features'], value['features']))


@pytest.mark.parametrize('count', [1, 2, 511, 512, 513, 1027])
def test_prefix_readback_stops_at_exact_old_count_and_checks_every_tail_value(tmp_path, count):
    previous = block([str(i) for i in range(count)])
    writer = FeatureStoreWriter(tmp_path / 'store', [0, 3], 'a' * 64, expert_dim=8)
    writer.append(previous);writer.append(block(['new'], 41));writer.finish()
    store = FeatureStore(writer.output / 'manifest.json')
    assert checked_preserved_cache(store, previous) == count
    previous['features'][1][-1, -1, -1] = -1
    with pytest.raises(ValueError, match='layer 1'): checked_preserved_cache(store, previous)


def histories(tmp_path):
    moves = ['h2e2', 'h7e7', 'b0c2', 'b9c7', 'a0b0']
    rows = []
    for i in range(5):
        h = replay(START_FEN, moves[:i])
        rows.append({'id': str(i), 'split': ['train','validation','test'][i % 3],
                     'history': h, 'feature_key': history_key(h), 'fen': h[-1]})
    path = tmp_path / 'questions.jsonl';write_jsonl(path, [*rows, *rows])
    weights = tmp_path / 'weights.bin';weights.write_bytes(b'pinned-test-weights')
    proof = tmp_path / 'source.json';proof.write_text(json.dumps(manifest('questions', {}, outputs=[path])))
    return rows, path, weights, proof


def fake_expert(monkeypatch, calls):
    class Expert(torch.nn.Module):
        dim, depth = 512, 20
        def __init__(self, path): super().__init__();calls.append('load')
        def forward(self, planes, *, depths):
            calls.append(len(planes));v = planes.flatten(1).sum(1).reshape(-1, 1, 1).expand(-1, 90, 512)
            return [v.float() + d / 7 for d in depths], torch.ones((len(planes), 3)) / 3
    monkeypatch.setattr(cache_feature_store, 'FrozenPx0', Expert)


def test_stream_extraction_deduplicates_contexts_and_keeps_bounded_shards(tmp_path, monkeypatch):
    rows, source, weights, proof = histories(tmp_path);calls = [];fake_expert(monkeypatch, calls)
    result = cache_feature_store.build([source], proof, weights, [0, 3], tmp_path / 'features',
                                       batch_size=2, shard_rows=3, device='cpu', cpu_threads=1)
    store = open_feature_cache(tmp_path / 'features/manifest.json')
    assert result['question_rows_streamed'] == 10 and result['new_roots'] == 5
    assert calls == ['load', 2, 1, 2]
    assert [s['rows'] for s in store.shards] == [3, 2]
    assert store['keys'] == [r['feature_key'] for r in rows]
    assert torch.isfinite(store['features'][0][:]).all()
    assert digest(source) == json.loads(proof.read_text())['outputs'][str(source)]['sha256']


@pytest.mark.parametrize('corruption', ['key', 'board', 'split', 'source_hash', 'unbound_source'])
def test_stream_extraction_rejects_unverified_or_inconsistent_source(tmp_path, monkeypatch, corruption):
    rows, source, weights, proof = histories(tmp_path);calls = [];fake_expert(monkeypatch, calls)
    if corruption in ['source_hash','unbound_source']:
        if corruption == 'source_hash': source.write_text(source.read_text() + '\n')
        else:
            d=json.loads(proof.read_text());d['outputs']={};proof.write_text(json.dumps(d))
    else:
        row=rows[0];row[{'key':'feature_key','board':'fen','split':'split'}[corruption]]='incorrect'
        write_jsonl(source, rows);proof.write_text(json.dumps(manifest('questions', {}, outputs=[source])))
    with pytest.raises(ValueError):
        cache_feature_store.build([source], proof, weights, [0, 3], tmp_path / 'features', device='cpu')
    assert not (tmp_path / 'features/manifest.json').exists() and calls == []


@pytest.mark.parametrize('problem', ['status', 'sha', 'expert', 'encoder', 'depth'])
def test_existing_cache_requires_matching_production_bytes_and_expert_contract(tmp_path, problem):
    value = block(['base'])
    path = tmp_path / 'base.pt';torch.save(value, path)
    proof = manifest('cache', {}, outputs=[path], verification={'expert_sha256': 'a' * 64})
    if problem == 'status':proof['status']='building'
    elif problem == 'sha':proof['outputs'][str(path)]['sha256']='b'*64
    elif problem == 'expert':proof['verification']['expert_sha256']='b'*64
    elif problem == 'encoder':proof['code']['source_sha256']['src/xqgeneral/expert.py']='b'*64
    else:value['depths']=[0,2];torch.save(value,path);proof['outputs'][str(path)]['sha256']=digest(path)
    p = tmp_path / 'proof.json';p.write_text(json.dumps(proof))
    writer=FeatureStoreWriter(tmp_path/'store',[0,3],'a'*64,expert_dim=8)
    with pytest.raises(ValueError):writer.add_existing(path,p)
    assert not (writer.output/'manifest.json').exists()


def test_fp32_finite_but_fp16_overflowing_expert_values_are_rejected(tmp_path, monkeypatch):
    rows,source,weights,proof=histories(tmp_path)
    class Overflow(torch.nn.Module):
        dim,depth=512,20
        def __init__(self,path):super().__init__()
        def forward(self,x,*,depths):
            return [torch.full((len(x),90,512),70000.) for d in depths],torch.ones((len(x),3))
    monkeypatch.setattr(cache_feature_store,'FrozenPx0',Overflow)
    with pytest.raises(ValueError,match='nonfinite'):
        cache_feature_store.build([source],proof,weights,[0,3],tmp_path/'features',device='cpu')
    assert not (tmp_path/'features/manifest.json').exists()


@pytest.mark.parametrize('artifact',['source','weights'])
def test_source_mutation_during_extraction_never_publishes_complete_store(tmp_path,monkeypatch,artifact):
    rows,source,weights,proof=histories(tmp_path)
    fake_expert(monkeypatch,[])
    cls=cache_feature_store.FrozenPx0
    forward=cls.forward
    def changed(self,*args,**kwargs):
        p=source if artifact=='source' else weights
        p.write_bytes(p.read_bytes()+b'\n')
        return forward(self,*args,**kwargs)
    monkeypatch.setattr(cls,'forward',changed)
    with pytest.raises(ValueError,match='source changed'):
        cache_feature_store.build([source],proof,weights,[0,3],tmp_path/'features',device='cpu')
    assert not (tmp_path/'features/manifest.json').exists()


def test_all_existing_contexts_skip_expert_loading_and_preserve_base_values(tmp_path,monkeypatch):
    rows,source,weights,proof=histories(tmp_path)
    value=block([r['feature_key'] for r in rows],width=512)
    path=tmp_path/'base.pt';torch.save(value,path)
    p=tmp_path/'base-proof.json';p.write_text(json.dumps(manifest('cache',{},outputs=[path],
        verification={'expert_sha256':digest(weights)})))
    calls=[];fake_expert(monkeypatch,calls)
    result=cache_feature_store.build([source],proof,weights,[0,3],tmp_path/'features',
                                   base_cache=path,base_proof=p,device='cpu')
    assert result['expert_loaded'] is False and result['new_roots']==0 and calls==[]
    store=FeatureStore(tmp_path/'features/manifest.json')
    assert len(store.shards)==1 and checked_preserved_cache(store,value)==len(rows)


def test_exact_training_inputs_include_all_storage_and_row_indexes(tmp_path):
    from xqgeneral.finite_training import FINITE_PROFILE
    from xqgeneral.training import training_input_paths
    from xqgeneral.training_index import build_index,index_paths
    writer,path,store,expected=prepared(tmp_path)
    data=tmp_path/'data'
    records=[{'id':str(i),'stage':'explanation','feature_key':k,'split':'train'} for i,k in enumerate(expected['keys'])]
    write_jsonl(data/'train.jsonl',records)
    write_jsonl(data/'validation.jsonl',[dict(r,split='validation') for r in records])
    index=tmp_path/'index';build_index([data/'train.jsonl'],index)
    config={'feature_path':str(path),'data_path':str(data),'init_from':'selected.pt',
            'training_data_profile':FINITE_PROFILE,'row_index_paths':{'train':str(index)}}
    assert {str(p) for p in training_input_paths(config)} == {
        str(path),str(writer.output/'keys.json'),*[s['path'] for s in store.shards],
        str(data/'train.jsonl'),str(data/'validation.jsonl'),'selected.pt',*[str(p) for p in index_paths(index)]}
    config={'feature_path':'./legacy.pt','data_path':str(data)}
    assert training_input_paths(config)[0]=='./legacy.pt'


@pytest.mark.parametrize('changed',[False,True])
def test_sft_probe_uses_actual_subset_offsets_and_rejects_changed_subset_on_resume(tmp_path,changed):
    from xqgeneral.finite_training import FINITE_PROFILE
    from xqgeneral.sft_preflight import probe_row_indexes
    from xqgeneral.training_index import IndexedTrainingRows,build_index
    original,subset=tmp_path/'original',tmp_path/'subset'
    declared={}
    for split in ['train','validation']:
        rows=[{'id':str(i),'stage':'explanation','feature_key':str(i),'split':split} for i in range(10)]
        write_jsonl(original/(split+'.jsonl'),rows)
        write_jsonl(subset/(split+'.jsonl'),rows[4:7])
        directory=tmp_path/(split+'-original-index');build_index([original/(split+'.jsonl')],directory,split=split)
        declared[split]=str(directory)
    config={'data_path':str(subset),'training_data_profile':FINITE_PROFILE,
            'row_index_paths':declared,'mixture':{'explanation':1}}
    fresh=probe_row_indexes(config,tmp_path/'probe')
    assert fresh['row_index_paths']!=declared and config['row_index_paths']==declared
    reader=IndexedTrainingRows(fresh['row_index_paths']['train'],[subset/'train.jsonl'],{'explanation'})
    assert [r['id'] for r in reader]==['4','5','6']
    if changed:
        p=subset/'train.jsonl';p.write_text(p.read_text().replace('"4"','"X"'))
        with pytest.raises(ValueError):probe_row_indexes(config,tmp_path/'probe')
    else:assert probe_row_indexes(config,tmp_path/'probe')==fresh
