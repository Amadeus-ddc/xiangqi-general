import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from xqgeneral.evidence import digest, iter_jsonl, load_jsonl, write_jsonl
from xqgeneral.search_inputs import prepare, prepared_training_roots, training_roots


def dataset(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    rows = [
        {'id': 'a-first', 'feature_key': 'a', 'split': 'train', 'game_id': 'g-a', 'answer': '红车'},
        {'id': 'b', 'feature_key': 'b', 'split': 'train', 'game_id': 'g-b', 'answer': '黑马',
         'augmentation_parent': 'b-original'},
        {'id': 'a-later', 'feature_key': 'a', 'split': 'train', 'game_id': 'g-a', 'answer': 'another task'},
        {'id': 'c', 'feature_key': 'c', 'split': 'train', 'game_id': 'g-c', 'answer': '空'},
    ]
    write_jsonl(data / 'train.jsonl', rows)
    write_jsonl(data / 'validation.jsonl', [{'split': 'validation', 'answer': 'protected'}])
    write_jsonl(data / 'test.jsonl', [{'split': 'test', 'answer': 'protected'}])
    return data, rows


def identity(data):
    path = data / 'train.jsonl'
    return {'sha256': digest(path), 'bytes': path.stat().st_size}


def test_jsonl_iterator_is_lazy_and_preserves_unicode_inside_json_strings(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text('\n' + json.dumps({'text': '红车\u2028黑马'}, ensure_ascii=False) + '\ninvalid-json\n')
    rows = iter_jsonl(path)
    assert next(rows) == {'text': '红车\u2028黑马'}
    with pytest.raises(json.JSONDecodeError):
        next(rows)
    write_jsonl(path, [{'text': '红车\u2028黑马'}, {'n': 2}])
    assert load_jsonl(path) == [{'text': '红车\u2028黑马'}, {'n': 2}]


@pytest.mark.parametrize('seed,limit', [(1, None), (2, 1), (20261013, 2), (33, 99)])
def test_streamed_roots_match_the_previous_first_record_and_seeded_shuffle(tmp_path, monkeypatch, seed, limit):
    data, rows = dataset(tmp_path)
    unique = {}
    for row in rows:
        unique.setdefault(row['feature_key'], row)
    expected = list(unique.values())
    random.Random(seed).shuffle(expected)
    if limit is not None:
        expected = expected[:limit]

    def no_whole_file_read(*args, **kwargs):
        raise AssertionError('Training selection must stream JSONL')

    monkeypatch.setattr(Path, 'read_text', no_whole_file_read)
    assert training_roots(data, limit, seed) == expected


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_search_limits_fail_before_scanning_data(tmp_path, limit):
    with pytest.raises(ValueError, match='positive'):
        training_roots(tmp_path / 'absent', limit, 1)


def test_late_nontraining_row_is_rejected_even_when_only_one_root_is_requested(tmp_path):
    data, rows = dataset(tmp_path)
    rows.append(dict(rows[-1], split='test'))
    write_jsonl(data / 'train.jsonl', rows)
    with pytest.raises(ValueError, match='training split'):
        training_roots(data, 1, 1)


def test_preparation_preserves_all_first_records_and_never_reads_heldout_answers(tmp_path, monkeypatch):
    data, _ = dataset(tmp_path)
    expected = training_roots(data, None, 3)
    original = identity(data)
    original_open = Path.open

    def protected_open(path, *args, **kwargs):
        assert path.name not in {'validation.jsonl', 'test.jsonl'}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', protected_open)
    output = tmp_path / 'pool'
    counts = prepare(data, output, seed=3)
    assert load_jsonl(output / 'roots.jsonl') == expected
    assert identity(data) == original
    assert counts['candidate_training_histories'] == 3
    assert counts['existing_color_derived_candidates'] == 1
    assert counts['validation_or_test_answers_read'] is False
    assert counts['new_distillation_targets_generated'] is False
    assert prepared_training_roots(output, data, 2, 3, original) == expected[:2]
    with pytest.raises(FileExistsError):
        prepare(data, output, seed=3)


def test_completed_source_preflight_is_checked_and_bound(tmp_path):
    data, _ = dataset(tmp_path)
    preflight = tmp_path / 'preflight.json'
    preflight.write_text(json.dumps({'status': 'complete', 'inputs': {str(data / 'train.jsonl'): identity(data)}}))
    output = tmp_path / 'pool'
    assert prepare(data, output, source_preflight=preflight)['source_preflight_identity_verified']
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['inputs'][str(preflight)]['sha256'] == digest(preflight)
    preflight.write_text(json.dumps({'status': 'complete', 'inputs': {}}))
    rejected = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='preflight'):
        prepare(data, rejected, source_preflight=preflight)
    assert not rejected.exists()


@pytest.mark.parametrize('artifact', ['roots.jsonl', 'counts.json'])
def test_prepared_output_tampering_is_rejected(tmp_path, artifact):
    data, _ = dataset(tmp_path)
    pool = tmp_path / 'pool'
    prepare(data, pool)
    with (pool / artifact).open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='outputs changed'):
        prepared_training_roots(pool, data, 1, 20261013, identity(data))


def test_source_seed_and_completion_changes_are_rejected(tmp_path):
    data, rows = dataset(tmp_path)
    pool = tmp_path / 'pool'
    prepare(data, pool)
    with pytest.raises(ValueError, match='seed'):
        prepared_training_roots(pool, data, 1, 9, identity(data))
    write_jsonl(data / 'train.jsonl', rows + [dict(rows[0], id='new', feature_key='new')])
    with pytest.raises(ValueError, match='training source'):
        prepared_training_roots(pool, data, 1, 20261013, identity(data))
    proof = json.loads((pool / 'manifest.json').read_text())
    proof['status'] = 'partial'
    (pool / 'manifest.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='completion'):
        prepared_training_roots(pool, data, 1, 20261013, identity(data))


def test_search_cli_consumes_prepared_roots_and_refuses_a_changed_execution_source(tmp_path, monkeypatch):
    from xqgeneral import search_distillation as search

    data, _ = dataset(tmp_path)
    pool = tmp_path / 'pool'
    prepare(data, pool)
    expected = load_jsonl(pool / 'roots.jsonl')[:2]
    resources = []
    for name in ['checkpoint.pt', 'features.pt', 'engine', 'weights.nnue']:
        path = tmp_path / name
        path.write_bytes(b'controlled-test-resource')
        resources.append(path)
    generated = []
    loaded = []

    class Miner:
        counts = {}

        def __init__(self, *args, **kwargs):
            pass

        def mine(self, row):
            generated.append(row)
            return None, [], 'controlled_no_improvement'

    def predictor(*args, **kwargs):
        loaded.append(True)
        return object()

    monkeypatch.setitem(sys.modules, 'xqgeneral.inference', SimpleNamespace(Predictor=predictor))
    monkeypatch.setattr(search, 'SearchMiner', Miner)
    monkeypatch.setattr(search, 'Pikafish', lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(search, 'reserved_positions', lambda *args: set())
    monkeypatch.setattr(search, 'training_roots', lambda *args: pytest.fail('Prepared search must use its completed pool'))
    output = tmp_path / 'search'
    argv = ['search', '--checkpoint', str(resources[0]), '--features', str(resources[1]),
        '--executable', str(resources[2]), '--weights', str(resources[3]), '--data', str(data),
        '--prepared-inputs', str(pool), '--limit', '2', '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', argv)
    search.main()
    assert generated == expected and len(loaded) == 1
    assert json.loads((output / 'contract.json').read_text())['code']
    (output / 'manifest.json').rename(output / 'preserved-controlled-complete.json')
    monkeypatch.setattr(search, 'code_identity', lambda: {'revision': 'changed', 'source_sha256': {}})
    with pytest.raises(ValueError, match='execution source'):
        search.main()
    assert len(loaded) == 1


@pytest.mark.parametrize('option,value', [('--limit', '0'), ('--nodes', '0'), ('--max-depth', '-1')])
def test_search_cli_rejects_invalid_budgets_before_reading_model_resources(monkeypatch, option, value):
    from xqgeneral import search_distillation as search

    monkeypatch.setattr(sys, 'argv', ['search', '--checkpoint', 'absent-model', '--output', 'absent-output', option, value])
    with pytest.raises(ValueError, match='budgets'):
        search.main()
