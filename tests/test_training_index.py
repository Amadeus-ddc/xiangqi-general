import json
from pathlib import Path

import numpy as np
import pytest

from xqgeneral.evidence import digest, write_jsonl
from xqgeneral.finite_training import FINITE_PROFILE, FiniteMixtureEpochs
from xqgeneral.training import load_training_rows, select_validation_rows, training_metadata, compatible_resume
from xqgeneral.training_index import (
    DTYPE, IndexedTrainingRows, build_index, index_paths, training_row_indexes,
)


def record(i, stage='new', split='train'):
    return {'id': f'实战/{i}', 'stage': stage, 'split': split, 'feature_key': f'history-{i % 3}',
            'question': '局面问题：' + '车炮馬\n' * (i % 5), 'answer': str(i), 'unmodified': {'i': i}}


def corpus(tmp_path, *, split='train'):
    first = [record(i, 'new' if i % 2 else 'old', split) for i in range(19)]
    second = [record(i, 'new' if i % 3 else 'unused', split) for i in range(19, 40)]
    paths = [tmp_path / 'first.jsonl', tmp_path / 'second.jsonl']
    # Byte offsets must survive Unicode, empty lines, CRLF and a final line without LF.
    paths[0].write_bytes(b'\r\n' + b'\r\n'.join(json.dumps(r, ensure_ascii=False).encode() for r in first))
    write_jsonl(paths[1], second)
    out = tmp_path / 'index'
    build_index(paths, out, split=split)
    rows = IndexedTrainingRows(out, paths, {'new', 'old'}, split=split)
    return paths, out, rows, load_training_rows(paths, {'new', 'old'})


def test_unicode_filtered_original_order_and_lookup(tmp_path):
    paths, out, rows, expected = corpus(tmp_path)
    assert rows.records.dtype == DTYPE and DTYPE.itemsize == 40
    assert isinstance(rows.records, np.memmap)
    assert list(rows) == expected
    assert rows[-1] == expected[-1]
    assert rows[2:9:2] == expected[2:9:2]
    assert rows[::-3] == expected[::-3]
    assert rows.get_batch([4, 1, 4]) == [expected[4], expected[1], expected[4]]
    with pytest.raises(IndexError):
        rows[len(rows)]
    with pytest.raises(IndexError):
        rows[-len(rows)-1]
    rows.close()


@pytest.mark.parametrize('world,micro', [(1, 2), (2, 1), (4, 2)])
@pytest.mark.parametrize('epoch', [0, 1, 4])
def test_finite_selection_batches_and_state_match_memory_backend(tmp_path, world, micro, epoch):
    paths, out, rows, expected = corpus(tmp_path)
    options = {'seed': 41, 'mixture_seed': 7, 'global_batch_size': 16, 'world': world, 'micro_batch_size': micro}
    memory = FiniteMixtureEpochs(expected, {'new': .8, 'old': .2}, **options)
    indexed = FiniteMixtureEpochs(rows, {'new': .8, 'old': .2}, **options)
    assert memory.contract() == indexed.contract()
    for step in range(indexed.steps_per_epoch):
        offset = epoch * indexed.steps_per_epoch + step
        assert memory.state_at(offset) == indexed.state_at(offset)
        for rank in range(world):
            assert memory.batch(offset, rank) == indexed.batch(offset, rank)


def test_sampling_and_schema_scans_do_not_read_question_payloads(tmp_path, monkeypatch):
    paths, out, rows, expected = corpus(tmp_path)
    def forbidden(*args):
        pytest.fail('A metadata-only preparation scan read question payloads')
    monkeypatch.setattr(rows, 'get_batch', forbidden)
    plan = FiniteMixtureEpochs(rows, {'new': .8, 'old': .2}, seed=4, global_batch_size=8)
    metadata = list(training_metadata(rows))
    assert len(metadata) == len(expected)
    assert all('question' not in r and 'answer' not in r for r in metadata)
    assert plan.batch_indices(0)[0]


@pytest.mark.parametrize('maximum', [1, 8, 1000])
def test_validation_reads_only_selected_rows_without_changing_ids(tmp_path, monkeypatch, maximum):
    paths, out, rows, expected = corpus(tmp_path, split='validation')
    read = []
    original = rows.get_batch
    def lookup(indices):
        read.extend(indices)
        return original(indices)
    monkeypatch.setattr(rows, 'get_batch', lookup)
    actual = select_validation_rows(rows, ['new'], 17, maximum)
    assert actual == select_validation_rows(expected, ['new'], 17, maximum)
    assert len(read) == len(actual) == min(maximum, sum(r['stage'] == 'new' for r in expected))


def test_single_batch_reads_only_requested_original_payloads(tmp_path, monkeypatch):
    paths, out, rows, expected = corpus(tmp_path)
    read = []
    original = rows.get_batch
    monkeypatch.setattr(rows, 'get_batch', lambda indices: (read.extend(indices), original(indices))[1])
    plan = FiniteMixtureEpochs(rows, {'new': 1}, seed=41, global_batch_size=8)
    assert not read
    batch, report = plan.batch(0)
    assert len(read) == len(batch) == report['global_valid_rows'] == 8
    assert batch == [expected[i] for i in read]


@pytest.mark.parametrize('field,value', [('id', ''), ('id', 3), ('stage', ''), ('feature_key', None),
                                         ('split', 'test'), ('task_profile', 'unknown')])
def test_invalid_rows_never_publish_complete_index(tmp_path, field, value):
    path = tmp_path / 'source.jsonl'
    write_jsonl(path, [dict(record(0), **{field: value})])
    with pytest.raises(ValueError):
        build_index([path], tmp_path / 'bad')
    assert not (tmp_path / 'bad/manifest.json').exists()


def test_duplicate_ids_across_sources_are_rejected(tmp_path):
    paths = [tmp_path / 'a.jsonl', tmp_path / 'b.jsonl']
    for path in paths:
        write_jsonl(path, [record(0)])
    with pytest.raises(ValueError, match='duplicate ID'):
        build_index(paths, tmp_path / 'bad')
    assert not (tmp_path / 'bad/manifest.json').exists()


@pytest.mark.parametrize('split', ['test', None, 'invalid'])
def test_independent_test_cannot_become_training_index(tmp_path, split):
    with pytest.raises(ValueError):
        build_index([tmp_path / 'not-read'], tmp_path / 'bad', split=split)
    assert not (tmp_path / 'bad').exists()


def test_existing_output_and_duplicate_source_are_preserved(tmp_path):
    paths, out, rows, expected = corpus(tmp_path)
    original = (out / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        build_index(paths, out)
    assert (out / 'manifest.json').read_bytes() == original
    with pytest.raises(ValueError):
        build_index([paths[0], paths[0]], tmp_path / 'repeat')


@pytest.mark.parametrize('change', ['order', 'source', 'split'])
def test_index_must_match_original_source_order_and_split(tmp_path, change):
    paths, out, rows, expected = corpus(tmp_path)
    requested = list(paths)
    split = 'train'
    if change == 'order':
        requested.reverse()
    elif change == 'source':
        other = tmp_path / 'copy.jsonl';other.write_bytes(paths[0].read_bytes());requested[0] = other
    else:
        split = 'validation'
    with pytest.raises(ValueError):
        IndexedTrainingRows(out, requested, {'new', 'old'}, split=split)


@pytest.mark.parametrize('artifact', ['source', 'records.bin', 'identities.bin', 'metadata.json', 'manifest.json'])
def test_mutation_after_open_rejects_next_batch(tmp_path, artifact):
    paths, out, rows, expected = corpus(tmp_path)
    path = paths[0] if artifact == 'source' else out / artifact
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='changed after verification'):
        rows.get_batch([0])


def test_stale_already_hashed_source_cannot_bypass_index_guard(tmp_path):
    paths, out, rows, expected = corpus(tmp_path)
    hashes = {str(p): digest(p) for p in [*paths, *index_paths(out)]}
    paths[0].write_bytes(paths[0].read_bytes().replace(b'"answer": "0"', b'"answer": "Z"'))
    with pytest.raises(ValueError, match='changed since index construction'):
        IndexedTrainingRows(out, paths, {'new', 'old'}, verified_hashes=hashes)


@pytest.mark.parametrize('artifact', ['records.bin', 'identities.bin', 'metadata.json', 'manifest.json'])
def test_stale_already_hashed_index_cannot_bypass_same_size_edit(tmp_path, artifact):
    paths, out, rows, expected = corpus(tmp_path)
    hashes = {str(p): digest(p) for p in [*paths, *index_paths(out)]}
    path = out / artifact
    data = bytearray(path.read_bytes())
    if artifact in ['records.bin', 'identities.bin']:
        data[-1] ^= 1
    else:
        data[data.index(b'"profile"') + 2] = ord('X')
    path.write_bytes(data)
    with pytest.raises(ValueError):
        IndexedTrainingRows(out, paths, {'new', 'old'}, verified_hashes=hashes)


def test_fresh_hashes_required_for_every_index_artifact(tmp_path):
    paths, out, rows, expected = corpus(tmp_path)
    hashes = {str(p): digest(p) for p in [*paths, *index_paths(out)]}
    assert IndexedTrainingRows(out, paths, {'new'}, verified_hashes=hashes)[0]['stage'] == 'new'
    hashes.pop(str(out / 'identities.bin'))
    with pytest.raises(ValueError, match='unverified'):
        IndexedTrainingRows(out, paths, {'new'}, verified_hashes=hashes)


@pytest.mark.parametrize('field,value', [('source', 2**32-1), ('stage', 2**16-1), ('feature', 2**32-1),
                                         ('id_offset', 2**64-1), ('offset', 2**64-1), ('bytes', 0)])
def test_corrupt_offsets_rejected_even_with_rebound_artifact_hash(tmp_path, field, value):
    paths, out, rows, expected = corpus(tmp_path)
    rows.close()
    record_path = out / 'records.bin'
    data = np.fromfile(record_path, dtype=DTYPE)
    data[0][field] = value
    data.tofile(record_path)
    p = out / 'manifest.json';receipt = json.loads(p.read_text())
    receipt['outputs'][str(record_path)]['sha256'] = digest(record_path)
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        IndexedTrainingRows(out, paths, {'new', 'old'})


def test_empty_and_unselected_corpora_have_zero_rows(tmp_path):
    path = tmp_path / 'empty.jsonl';path.write_text('\n')
    build_index([path], tmp_path / 'empty-index')
    rows = IndexedTrainingRows(tmp_path / 'empty-index', [path], ['new'])
    assert len(rows) == 0 and list(rows) == []
    assert select_validation_rows(rows, ['new'], 17, 8) == []
    paths, out, rows, expected = corpus(tmp_path)
    assert len(IndexedTrainingRows(out, paths, ['absent'])) == 0


@pytest.mark.parametrize('value', [None, {}, {'validation': 'index'}, {'train': ''}, {'train': 'index', 'test': 'test'}])
def test_index_recipe_requires_explicit_finite_train_contract(value):
    with pytest.raises(ValueError):
        training_row_indexes({'training_data_profile': FINITE_PROFILE, 'row_index_paths': value})


def test_index_recipe_default_and_resume_path_guard():
    assert training_row_indexes({}) == {}
    config = {'training_data_profile': FINITE_PROFILE, 'row_index_paths': {'train': 'index'}}
    assert training_row_indexes(config) == {'train': Path('index')}
    with pytest.raises(ValueError):
        training_row_indexes({'row_index_paths': {'train': 'index'}})
    with pytest.raises(ValueError, match='row index paths differ'):
        compatible_resume(config, dict(config, row_index_paths={'train': 'other'}))


def test_relative_index_output_is_resolved_and_reusable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_jsonl('source.jsonl', [record(0)])
    build_index(['source.jsonl'], 'relative-index')
    assert IndexedTrainingRows('relative-index', ['source.jsonl'], ['new'])[0] == record(0)
