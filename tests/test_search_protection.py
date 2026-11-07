"""Actual raw-reference reconstruction; controlled corpus provenance, no GPU/query."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import test_oracle_policy_corpus as inputs
from xqgeneral import oracle_policy_corpus as corpus, search_protection as protection
from xqgeneral.evidence import atomic_json, load_jsonl, position_key, write_jsonl
from xqgeneral.oracle import parse_analysis
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.rules import legal_moves, play


def setup(tmp_path):
    args = inputs.fixture(tmp_path)
    policy_contract = args['policies'] / 'contract.json'
    contract = json.loads(policy_contract.read_text())
    shared = contract['shared_isolation_authority']
    authority = Path(shared['authority'])
    atomic_json(authority / 'manifest.json', dict(status='complete',
        kind='prepared_unused_recorded_search_roots', controlled_upstream_source=True))
    shared['authority_manifest_binding'] = inputs.binding(authority / 'manifest.json')
    atomic_json(policy_contract, contract)
    p = args['policies'] / 'manifest.json';proof = json.loads(p.read_text())
    proof['outputs'][str(policy_contract)] = inputs.binding(policy_contract);atomic_json(p, proof)
    path = args['references'] / 'roots.jsonl';record = load_jsonl(path)[0]
    original_positions = set(json.loads((authority / 'heldout-positions.json').read_text()))
    lines = []
    for rank, candidate in enumerate(record['response']['candidates'], 1):
        pv = list(candidate['pv']);fen = record['record']['fen']
        for move in pv:fen = play(fen, move)
        # New raw suffixes beyond the numerical answer's six-ply budget.
        while len(pv) < 10:
            moves = sorted(legal_moves(fen))
            if not moves:break
            outside = [move for move in moves if position_key(play(fen, move)) not in original_positions and
                       corpus._mirror(position_key(play(fen, move))) not in original_positions]
            move = (outside or moves)[-1];pv.append(move);fen = play(fen, move)
        lines.append(f'info depth 20 multipv {rank} score cp {25-rank} nodes 1000000 pv ' + ' '.join(pv))
    lines.append('bestmove ' + record['response']['best_move'])
    record['response'] = parse_analysis(record['record']['fen'], lines, 1000000)
    write_jsonl(path, [record])
    p = args['references'] / 'manifest.json';proof = json.loads(p.read_text())
    proof['outputs'][str(path)] = inputs.binding(path);atomic_json(p, proof)
    counts = corpus.prepare(**args)
    return args, counts, shared


def test_all_raw_candidate_suffixes_and_mirrors_survive_export_and_reopen(tmp_path, monkeypatch):
    args, counts, shared = setup(tmp_path)
    output = tmp_path / 'protection'
    original = Path.open
    def no_test(path, *more, **kwargs):
        if path.name == 'test.jsonl':pytest.fail('Protection must not open independent-test labels')
        return original(path, *more, **kwargs)
    monkeypatch.setattr(Path, 'open', no_test)
    result = protection.prepare(args['output'], output)
    assert result['protected_positions'] == counts['extended_protected_positions']
    assert result['newly_reserved_positions'] > 0 and result['new_engine_or_model_queries'] == 0
    positions, note = protection.checked_protection(output / 'manifest.json', BoundInputs(),
        base_data=Path(next(iter(shared['base_data_identities']))).parent, source_pool=shared['authority'])
    reference = load_jsonl(args['references'] / 'roots.jsonl')[0]
    new_end = reference['record']['fen']
    for move in reference['response']['candidates'][-1]['pv']:new_end = play(new_end, move)
    key = position_key(new_end)
    assert key in positions and corpus._mirror(key) in positions
    assert key not in json.loads((Path(shared['authority']) / 'heldout-positions.json').read_text())
    assert note['newly_reserved_positions'] == result['newly_reserved_positions']
    before = inputs.snapshot(output)
    assert protection.prepare(args['output'], output, resume=True) == result
    assert inputs.snapshot(output) == before


@pytest.mark.parametrize('damage', ['validation', 'reference', 'authority', 'counts', 'closure', 'original_order', 'course_duplicate'])
def test_changed_used_sources_cannot_create_protection(tmp_path, damage):
    args, _, shared = setup(tmp_path)
    if damage == 'validation':path = args['output'] / 'validation.jsonl';path.write_bytes(path.read_bytes() + b'changed')
    elif damage == 'reference':path = args['references'] / 'roots.jsonl';path.write_bytes(path.read_bytes() + b'changed')
    elif damage == 'authority':
        path = Path(shared['authority']) / 'heldout-positions.json';atomic_json(path, [])
    elif damage in ['original_order', 'course_duplicate']:
        path = args['output'] / 'validation.jsonl' if damage == 'original_order' else args['replay'][0] / 'validation.jsonl'
        rows = load_jsonl(path)
        if damage == 'original_order':rows[2], rows[3] = rows[3], rows[2]
        else:rows.append(deepcopy(rows[0]))
        write_jsonl(path, rows)
        p = args['output'] / 'manifest.json';proof = json.loads(p.read_text())
        proof['outputs' if damage == 'original_order' else 'inputs'][str(path)] = inputs.binding(path)
        atomic_json(p, proof)
    else:
        path = args['output'] / ('counts.json' if damage == 'counts' else 'contract.json')
        value = json.loads(path.read_text())
        if damage == 'counts':value['extended_protected_positions'] += 2
        else:value['closure']['oracle.py'] = '0' * 64
        atomic_json(path, value)
        p = args['output'] / 'manifest.json';proof = json.loads(p.read_text())
        proof['outputs'][str(path)] = inputs.binding(path)
        if damage == 'counts':proof['verification'] = value
        atomic_json(p, proof)
    output = tmp_path / 'protection'
    with pytest.raises(ValueError):protection.prepare(args['output'], output)
    assert not output.exists()


@pytest.mark.parametrize('damage', ['positions', 'input', 'native_count'])
def test_consistent_rehashing_cannot_hide_a_changed_native_footprint(tmp_path, damage):
    args, _, _ = setup(tmp_path);output = tmp_path / 'protection'
    protection.prepare(args['output'], output)
    path = output / 'manifest.json';proof = json.loads(path.read_text())
    if damage == 'input':proof['inputs'].pop(next(iter(proof['inputs'])))
    elif damage == 'native_count':proof['verification']['newly_reserved_positions'] += 1
    else:
        p = output / 'protected-positions.json';rows = json.loads(p.read_text());rows.pop()
        atomic_json(p, rows);proof['outputs'][str(p)] = inputs.binding(p)
    atomic_json(path, proof);before = inputs.snapshot(output)
    with pytest.raises(ValueError):protection.checked_protection(path, BoundInputs())
    with pytest.raises(ValueError):protection.prepare(args['output'], output, resume=True)
    assert inputs.snapshot(output) == before


@pytest.mark.parametrize('damage', ['base', 'recorded_pool', 'student_authority', 'incomplete_pool', 'parent_corpus'])
def test_certificate_cannot_move_to_a_different_search_scope(tmp_path, damage):
    args, _, shared = setup(tmp_path);output = tmp_path / 'protection'
    protection.prepare(args['output'], output)
    base = Path(next(iter(shared['base_data_identities']))).parent
    options = dict(base_data=base, source_pool=shared['authority'], corpus=args['output'])
    if damage == 'base':options['base_data'] = tmp_path / 'other-base'
    elif damage == 'parent_corpus':options['corpus'] = tmp_path / 'other-corpus'
    else:
        pool = tmp_path / 'other-pool';pool.mkdir();options['source_pool'] = pool
        value = dict(kind='prepared_unused_recorded_search_roots', status='complete')
        if damage in ['student_authority', 'incomplete_pool']:
            wrong = deepcopy(shared);wrong['reserved_positions'] += 2
            value = dict(kind='prepared_student_training_game_roots',status='complete',
                config=dict(shared_isolation_authority=wrong))
            if damage == 'incomplete_pool':value['status'] = 'running'
        atomic_json(pool / 'manifest.json', value)
    with pytest.raises(ValueError):protection.checked_protection(output / 'manifest.json', BoundInputs(), **options)


def test_student_pool_with_the_same_original_authority_can_use_protection(tmp_path):
    args, _, shared = setup(tmp_path);output = tmp_path / 'protection'
    protection.prepare(args['output'], output)
    pool = tmp_path / 'student-pool';pool.mkdir()
    atomic_json(pool / 'manifest.json', dict(kind='prepared_student_training_game_roots',status='complete',
        config=dict(shared_isolation_authority=shared)))
    positions, note = protection.checked_protection(output / 'manifest.json', BoundInputs(), source_pool=pool)
    assert len(positions) == note['protected_positions']
