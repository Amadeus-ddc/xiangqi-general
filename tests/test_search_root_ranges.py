"""Real pool selection/ownership with controlled CLI workers, never label quality."""
import copy
import json
import sys

import pytest

from test_recorded_search_inputs import fixture, identities
from test_search_collection import recorded_collection
from test_search_pilot import ControlledChain, setup_pilot
from xqgeneral import search_distillation as search, search_pilot
from xqgeneral.collect_search import mining_reservations
from xqgeneral.evidence import atomic_json, digest, load_jsonl, write_jsonl
from xqgeneral.recorded_search_inputs import BoundInputs, prepare, prepared_roots


@pytest.mark.parametrize('offset', [-1, True, 1.5, '1', None])
def test_invalid_offset_fails_before_opening_missing_inputs(offset):
    with pytest.raises(ValueError, match='offset'):
        prepared_roots('missing-pool', 'missing-data', 1, 20261013, {}, offset=offset)


def test_adjacent_ranges_exactly_partition_the_original_stable_prefix(tmp_path):
    args, _ = fixture(tmp_path)
    pool = tmp_path / 'pool'
    prepare(**args, output=pool, min_ply=1, per_game=3)
    whole, protected = prepared_roots(pool, args['data'], 6, 20261013, identities(args))
    pieces = []
    for offset in [0, 2, 4]:
        selected, positions = prepared_roots(pool, args['data'], 2, 20261013, identities(args), offset=offset)
        assert positions == protected
        pieces.extend(selected)
    assert pieces == whole
    assert len({row['id'] for row in pieces}) == 6
    assert prepared_roots(pool, args['data'], 6, 20261013, identities(args), offset=0)[0] == whole


def test_nonzero_range_cannot_run_past_pool_coverage(tmp_path):
    args, _ = fixture(tmp_path)
    pool = tmp_path / 'pool'
    counts = prepare(**args, output=pool, min_ply=1, per_game=3)
    with pytest.raises(ValueError, match='range'):
        prepared_roots(pool, args['data'], counts['candidate_training_histories'],
                       20261013, identities(args), offset=1)


@pytest.mark.parametrize('corrupt_index', [0, -1])
def test_unselected_prefix_and_suffix_still_require_original_training_ownership(tmp_path, corrupt_index):
    args, _ = fixture(tmp_path)
    pool = tmp_path / 'pool'
    prepare(**args, output=pool, min_ply=1, per_game=3)
    path = pool / 'roots.jsonl'
    rows = load_jsonl(path);assert len(rows) > 3
    rows[corrupt_index]['split'] = 'validation'
    write_jsonl(path, rows)
    proof = json.loads((pool / 'manifest.json').read_text())
    proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(pool / 'manifest.json', proof)
    with pytest.raises(ValueError, match='ownership or order'):
        prepared_roots(pool, args['data'], 1, 20261013, identities(args), offset=1)


def test_collection_requires_the_actual_ranged_original_root(tmp_path):
    args, pool, miner, query = recorded_collection(tmp_path)
    second = load_jsonl(pool / 'roots.jsonl')[1]
    query = copy.deepcopy(query)
    query.update(id=second['id'] + '-search', source_root_id=second['id'], record=second, depth=0)
    queries = miner / 'queries.jsonl'
    write_jsonl(queries, [query])
    proof_path = miner / 'manifest.json'
    proof = json.loads(proof_path.read_text());proof['config']['root_offset'] = 1
    proof['outputs'][str(queries)] = {'sha256': digest(queries), 'bytes': queries.stat().st_size}
    atomic_json(proof_path, proof)
    reserved, info = mining_reservations(queries, [query], args['data'], args['heldout_data'], BoundInputs())
    assert reserved and info['recorded_root_offset'] == 1
    assert info['original_selected_root_ownership_and_descendants_verified']
    proof['config']['root_offset'] = 0;atomic_json(proof_path, proof)
    with pytest.raises(ValueError, match='original selected training root'):
        mining_reservations(queries, [query], args['data'], args['heldout_data'], BoundInputs())


def test_pilot_forwards_range_and_resume_rejects_a_changed_offset(tmp_path, monkeypatch):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    saved = json.loads(config.read_text());saved['root_offset'] = 1
    atomic_json(config, saved)
    query = copy.deepcopy(query)
    query['source_root_id'] = load_jsonl(saved['recorded_inputs'] + '/roots.jsonl')[1]['id']
    chain = ControlledChain(monkeypatch, query, empty=True)
    output = tmp_path / 'pilot'
    result = search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    assert result['mined_queries'] == 0
    command = next(value for value in chain.commands if 'xqgeneral.search_distillation' in value)
    assert command[command.index('--root-offset') + 1] == '1'
    mining = json.loads((output / 'mining/manifest.json').read_text())
    assert mining['config']['root_offset'] == mining['verification']['recorded_root_offset'] == 1
    (output / 'mining/manifest.json').unlink()
    changed = command[command.index('xqgeneral.search_distillation'):]
    changed[changed.index('--root-offset') + 1] = '0'
    monkeypatch.setattr(sys, 'argv', changed)
    with pytest.raises(ValueError, match='continuation inputs'):
        search.main()
    assert chain.student_loads == 1


@pytest.mark.parametrize('offset', [True, 1000000])
def test_invalid_or_uncovered_pilot_range_fails_before_models(tmp_path, monkeypatch, offset):
    pipeline, validation, config, query, _ = setup_pilot(tmp_path, monkeypatch)
    saved = json.loads(config.read_text());saved['root_offset'] = offset
    atomic_json(config, saved)
    chain = ControlledChain(monkeypatch, query)
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    assert not output.exists() and not chain.commands and chain.student_loads == chain.teacher_loads == 0


def test_offset_requires_recorded_pool_before_reading_model_assets(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['search', '--checkpoint', 'missing.pt', '--output', 'missing-output',
                                     '--root-offset', '1'])
    with pytest.raises(ValueError, match='recorded'):
        search.main()
