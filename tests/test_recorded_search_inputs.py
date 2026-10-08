from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from xqgeneral.evidence import atomic_json, digest, history_key, load_jsonl, manifest, position_key, write_jsonl
from xqgeneral.human_games import assigned_split
from xqgeneral.recorded_search_inputs import prepare, prepared_roots
from xqgeneral.rules import START_FEN, adjudicate, legal_moves, play, replay
from xqgeneral.search_inputs import prepare as prepare_used
from xqgeneral.symmetry import mirror_fen


def canonical_games():
    selected, counts = [], Counter()
    for index, first in enumerate(sorted(legal_moves(START_FEN))):
        rng = random.Random(index + 55)
        moves, history = [first], [START_FEN, play(START_FEN, first)]
        while len(moves) < 16 and not adjudicate(START_FEN, moves)['ended']:
            move = rng.choice(sorted(legal_moves(history[-1])))
            moves.append(move)
            history.append(play(history[-1], move))
        identity = 'recorded-' + hashlib.sha256(json.dumps([START_FEN, moves], separators=(',', ':')).encode()).hexdigest()
        split = assigned_split(identity, 20261051)
        if counts[split] >= (3 if split == 'train' else 1):
            continue
        selected.append({'game_id': identity, 'split': split, 'initial_fen': START_FEN,
            'moves': moves, 'history': history, 'native_terminal': adjudicate(START_FEN, moves),
            'source_kind': 'recorded_human_match', 'headers': {'Red': '甲', 'Black': '乙'},
            'source': {'declared_license': 'test_fixture'}, 'provenance': 'native_test_fixture'})
        counts[split] += 1
        if counts == {'train': 3, 'validation': 1, 'test': 1}:
            return selected
    raise AssertionError('Fixture lacks all game splits')


def fixture(tmp_path, reserved=(), used_games=()):
    data = tmp_path / 'data'
    data.mkdir()
    for split in ('train', 'validation', 'test'):
        write_jsonl(data / f'{split}.jsonl', [{'id': 'old', 'game_id': 'old-game',
            'split': 'train', 'feature_key': 'old-key', 'answer': '旧题'}] if split == 'train' else [])
    atomic_json(data / 'manifest.json', manifest('controlled_base', {}, [],
        [data / f'{s}.jsonl' for s in ('train', 'validation', 'test')]))
    footprints = tmp_path / 'footprints'
    footprints.mkdir()
    atomic_json(footprints / 'footprints.json', {'combined_games': {'train': list(used_games)},
        'combined_positions': {'validation': list(reserved)}})
    base = json.loads((data / 'manifest.json').read_text())
    proof = manifest('recorded_coach_footprints', {'data': str(data)}, [], [footprints / 'footprints.json'])
    proof['inputs'] = base['outputs']
    atomic_json(footprints / 'manifest.json', proof)
    used = tmp_path / 'used'
    prepare_used(data, used)
    labels = tmp_path / 'labels'
    labels.mkdir()
    for split in ('train', 'validation', 'test'):
        write_jsonl(labels / f'{split}.jsonl', [])
    atomic_json(labels / 'manifest.json', manifest('controlled_labels', {}, [],
        [labels / f'{s}.jsonl' for s in ('train', 'validation', 'test')]))
    source = tmp_path / 'canonical'
    source.mkdir()
    games = canonical_games()
    write_jsonl(source / 'games.jsonl', games)
    atomic_json(source / 'manifest.json', manifest('recorded_games_native_import',
        {'seed': 20261051}, [], [source / 'games.jsonl']))
    return {'games': [source], 'data': data, 'footprints': footprints / 'manifest.json',
        'used_inputs': used, 'heldout_data': labels}, games


def identities(args):
    return json.loads((args['data'] / 'manifest.json').read_text())['outputs']


def footprint(row):
    states = replay(row['fen'], row['future_moves'])
    return {position_key(s) for s in states} | {position_key(mirror_fen(s)) for s in states}


def test_original_unused_roots_replay_preserve_splits_and_have_no_teacher_answers(tmp_path):
    args, games = fixture(tmp_path)
    out = tmp_path / 'pool'
    counts = prepare(**args, output=out, min_ply=1, per_game=3)
    rows = load_jsonl(out / 'roots.jsonl')
    originals = {g['game_id']: g for g in games}
    assert counts['canonical_game_splits'] == {'train': 3, 'validation': 1, 'test': 1}
    assert counts['new_color_derived_candidates'] == 0
    assert counts['new_distillation_targets_generated'] is False
    assert max(Counter(r['game_id'] for r in rows).values()) <= 3
    heldout = set(json.loads((out / 'heldout-positions.json').read_text()))
    for row in rows:
        game = originals[row['game_id']]
        assert row['split'] == game['split'] == 'train'
        assert replay(row['initial_fen'], row['moves']) == row['history']
        assert history_key(row['history']) == row['feature_key']
        assert 'answer' not in row and 'augmentation_parent' not in row
        assert row['moves'] == game['moves'][:row['ply']]
        assert row['future_moves'] == game['moves'][row['ply']:row['ply'] + len(row['future_moves'])]
        assert not footprint(row) & heldout
    selected, reserved = prepared_roots(out, args['data'], 2, 20261013, identities(args))
    assert selected == rows[:2] and reserved == heldout
    other = tmp_path / 'other'
    prepare(**args, output=other, min_ply=1, per_game=3)
    assert (other / 'roots.jsonl').read_bytes() == (out / 'roots.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        prepare(**args, output=out)


def test_prior_future_and_color_positions_are_excluded_and_whole_used_game_is_skipped(tmp_path):
    args, games = fixture(tmp_path)
    initial = tmp_path / 'initial'
    prepare(**args, output=initial, min_ply=1, per_game=3)
    example = load_jsonl(initial / 'roots.jsonl')[0]
    protected = position_key(mirror_fen(replay(example['fen'], example['future_moves'])[1]))
    prior_path = args['footprints'].parent / 'footprints.json'
    prior = json.loads(prior_path.read_text())
    prior['combined_positions']['validation'] = [protected]
    used_game = next(g for g in games if g['split'] == 'train' and g['game_id'] != example['game_id'])
    prior['combined_games']['train'] = [used_game['game_id']]
    atomic_json(prior_path, prior)
    proof = json.loads(args['footprints'].read_text())
    proof['outputs'][str(prior_path)] = {'sha256': digest(prior_path), 'bytes': prior_path.stat().st_size}
    atomic_json(args['footprints'], proof)
    output = tmp_path / 'filtered'
    counts = prepare(**args, output=output, min_ply=1, per_game=3)
    rows = load_jsonl(output / 'roots.jsonl')
    assert all(row['game_id'] != used_game['game_id'] and protected not in footprint(row) for row in rows)
    assert counts['excluded']['already_used_or_reserved_game'] >= 1
    assert counts['excluded']['heldout_root_or_recorded_future'] >= 1


def test_cross_source_game_duplicates_retain_first_attribution(tmp_path):
    args, games = fixture(tmp_path)
    extra = tmp_path / 'extra'
    extra.mkdir()
    changed = [dict(g, source_kind='recorded_computer_match', provenance='later_duplicate') for g in games]
    write_jsonl(extra / 'games.jsonl', changed)
    atomic_json(extra / 'manifest.json', manifest('recorded_games_native_import',
        {'seed': 20261051}, [], [extra / 'games.jsonl']))
    args['games'].append(extra)
    output = tmp_path / 'pool'
    counts = prepare(**args, output=output, min_ply=1)
    assert counts['duplicate_source_games'] == len(games)
    assert all(row['recorded_source_kind'] == 'recorded_human_match' and
               row['provenance'] != 'later_duplicate' for row in load_jsonl(output / 'roots.jsonl'))


@pytest.mark.parametrize('value', ['absent', None])
def test_missing_optional_platform_provenance_is_preserved_as_missing(tmp_path, value):
    args, games = fixture(tmp_path)
    for game in games:
        if value == 'absent': game.pop('provenance')
        else: game['provenance'] = None
    source = args['games'][0]
    write_jsonl(source / 'games.jsonl', games)
    atomic_json(source / 'manifest.json', manifest('public_platform_recorded_games_native_import',
        {'seed': 20261051}, [], [source / 'games.jsonl']))
    output = tmp_path / 'pool'
    prepare(**args, output=output, min_ply=1)
    rows = load_jsonl(output / 'roots.jsonl')
    assert rows and all(row['recorded_source_provenance'] is None and
                        row['provenance'] == 'unused_recorded_search_input' for row in rows)
    assert prepared_roots(output, args['data'], 1, 20261013, identities(args))[0] == rows[:1]


def test_explanation_candidate_branch_and_its_color_positions_are_reserved(tmp_path):
    args, games = fixture(tmp_path)
    game = next(g for g in games if g['split'] == 'validation')
    ply = 4
    root = game['history'][ply]
    alternative = next(move for move in legal_moves(root) if move != game['moves'][ply])
    row = {'id': 'protected-explanation', 'game_id': game['game_id'], 'split': 'validation',
        'stage': 'explanation', 'initial_fen': game['initial_fen'], 'moves': game['moves'][:ply],
        'history': game['history'][:ply + 1], 'fen': root,
        'feature_key': history_key(game['history'][:ply + 1]), 'future_moves': [],
        'answer': json.dumps({'pv': [game['moves'][ply]], 'branches': [{'pv': [alternative]}]})}
    labels = args['heldout_data']
    write_jsonl(labels / 'validation.jsonl', [row])
    atomic_json(labels / 'manifest.json', manifest('controlled_labels', {}, [],
        [labels / f'{s}.jsonl' for s in ('train', 'validation', 'test')]))
    output = tmp_path / 'pool'
    prepare(**args, output=output, min_ply=1)
    positions = set(json.loads((output / 'heldout-positions.json').read_text()))
    child = play(root, alternative)
    assert {position_key(child), position_key(mirror_fen(child))} <= positions


def test_late_nontraining_root_is_rejected_even_for_limit_one(tmp_path):
    args, _ = fixture(tmp_path)
    output = tmp_path / 'pool'
    prepare(**args, output=output, min_ply=1)
    rows = load_jsonl(output / 'roots.jsonl')
    assert len(rows) > 1
    rows[-1]['split'] = 'test'
    write_jsonl(output / 'roots.jsonl', rows)
    proof = json.loads((output / 'manifest.json').read_text())
    path = output / 'roots.jsonl'
    proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(output / 'manifest.json', proof)
    with pytest.raises(ValueError, match='ownership'):
        prepared_roots(output, args['data'], 1, 20261013, identities(args))


@pytest.mark.parametrize('change', ['split', 'identity', 'history_dimensions'])
def test_wrong_canonical_contract_is_rejected_even_with_updated_file_hash(tmp_path, change):
    args, games = fixture(tmp_path)
    if change == 'split': games[0]['split'] = 'test' if games[0]['split'] == 'train' else 'train'
    elif change == 'identity': games[0]['game_id'] = 'recorded-wrong'
    else: games[0]['history'].pop()
    source = args['games'][0]
    write_jsonl(source / 'games.jsonl', games)
    atomic_json(source / 'manifest.json', manifest('recorded_games_native_import',
        {'seed': 20261051}, [], [source / 'games.jsonl']))
    with pytest.raises(ValueError, match='Canonical'):
        prepare(**args, output=tmp_path / 'rejected', min_ply=1)
    assert not (tmp_path / 'rejected').exists()


@pytest.mark.parametrize('name,value', [('per_game', 0), ('per_game', True), ('min_ply', -1), ('workers', 0)])
def test_bad_budget_is_rejected_before_opening_any_inputs(tmp_path, name, value):
    with pytest.raises(ValueError, match='budgets'):
        prepare(['absent'], 'absent', 'absent', 'absent', 'absent', tmp_path / 'out', **{name: value})


@pytest.mark.parametrize('artifact', ['roots.jsonl', 'heldout-positions.json', 'counts.json'])
def test_changed_recorded_outputs_and_changed_upstream_are_rejected(tmp_path, artifact):
    args, _ = fixture(tmp_path)
    out = tmp_path / 'pool'
    prepare(**args, output=out, min_ply=1)
    with (out / artifact).open('a') as handle: handle.write(' ')
    with pytest.raises(ValueError, match='source changed'):
        prepared_roots(out, args['data'], 1, 20261013, identities(args))


def test_changed_data_seed_and_canonical_upstream_are_rejected(tmp_path):
    args, _ = fixture(tmp_path)
    out = tmp_path / 'pool'
    prepare(**args, output=out, min_ply=1)
    wrong = copy.deepcopy(identities(args));next(iter(wrong.values()))['sha256'] = 'wrong'
    for seed, ids in [(4, identities(args)), (20261013, wrong)]:
        with pytest.raises(ValueError, match='contract differs'):
            prepared_roots(out, args['data'], 1, seed, ids)
    with (args['games'][0] / 'games.jsonl').open('a') as handle: handle.write(' ')
    with pytest.raises(ValueError, match='source changed'):
        prepared_roots(out, args['data'], 1, 20261013, identities(args))


def test_search_cli_receives_reserved_canonical_future_positions_and_selected_roots(tmp_path, monkeypatch):
    from xqgeneral import search_distillation as search
    args, _ = fixture(tmp_path)
    pool = tmp_path / 'pool'
    prepare(**args, output=pool, min_ply=1)
    selected, additional = prepared_roots(pool, args['data'], 2, 20261013, identities(args))
    resources = []
    for name in ('checkpoint.pt', 'features.pt', 'engine', 'weights.nnue'):
        path = tmp_path / name;path.write_bytes(b'controlled-no-model-resource');resources.append(path)
    mined, observed = [], []
    class Miner:
        counts = {}
        def __init__(self, predictor, oracle, reserved, *a, **k): observed.append(reserved)
        def mine(self, row): mined.append(row);return None, [], 'controlled_no_improvement'
    monkeypatch.setitem(sys.modules, 'xqgeneral.inference', SimpleNamespace(Predictor=lambda *a, **k: object()))
    monkeypatch.setattr(search, 'Pikafish', lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(search, 'SearchMiner', Miner)
    monkeypatch.setattr(search, 'training_roots', lambda *a: pytest.fail('Unused-game search must consume its pool'))
    monkeypatch.setattr(search, 'reserved_positions', lambda *a: {'additional-existing-reservation'})
    output = tmp_path / 'search'
    monkeypatch.setattr(sys, 'argv', ['search', '--checkpoint', str(resources[0]), '--features', str(resources[1]),
        '--executable', str(resources[2]), '--weights', str(resources[3]), '--data', str(args['data']),
        '--recorded-inputs', str(pool), '--limit', '2', '--output', str(output)])
    search.main()
    assert mined == selected and observed == [additional | {'additional-existing-reservation'}]
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['verification']['unused_recorded_training_inputs'] is True
    assert proof['verification']['additional_reserved_canonical_and_explanation_positions'] == len(additional)
    assert all(str(pool / name) in proof['inputs'] for name in
        ('manifest.json', 'roots.jsonl', 'counts.json', 'heldout-positions.json'))
