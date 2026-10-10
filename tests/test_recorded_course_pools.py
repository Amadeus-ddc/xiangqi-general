from copy import deepcopy
import json

import pytest

from xqgeneral.course_tasks import native_tag, paper_records, validate_context
from xqgeneral.course_sampling import FrequencySampler
from xqgeneral.evidence import atomic_json, digest, history_key, load_jsonl, manifest, position_key, write_jsonl
from xqgeneral.recorded_course_pools import (
    POOL_PROFILE, build, native_classes, readback, root_footprint, terminal_candidates,
)
from xqgeneral.rules import START_FEN, adjudicate, replay
from xqgeneral.symmetry import mirror_fen


def game(fen, moves, name='constructed-native-game', split='train'):
    return {'game_id': name, 'initial_fen': fen, 'moves': list(moves), 'history': replay(fen, moves),
            'native_terminal': adjudicate(fen, moves), 'source': 'constructed_test_probe',
            'headers': {'Red': 'controlled-red', 'Black': 'controlled-black'},
            'players': ['controlled-red', 'controlled-black'], 'source_kind': 'constructed_native_probe',
            'provenance': 'constructed_test_probe;not_real_recorded_game', 'split': split}


CHECK = '4k4/9/9/9/4r4/9/9/9/9/4K4 w - - 0 1'
STALEMATE = '4k4/3R5/5R3/9/9/9/9/9/9/5K3 b - - 0 1'
MATE_LINE = ['e0d0', 'e5d5']


def fixture(tmp_path):
    games, owners = [], []
    for split, rank in [('train', 0), ('validation', 1), ('test', 2)]:
        rows = CHECK.split()[0].split('/')
        rows[9 - rank] = 'H3K4' if rank == 0 else 'H8'
        fen = '/'.join(rows) + ' w - - 0 1'
        source = game(fen, MATE_LINE, name=f'{split}-constructed-native-probe')
        games.append(source)
        # The archive's default train split does not replace the previous ownership.
        history = source['history'][:2]
        owners.append({'game_id': source['game_id'], 'split': split, 'recorded_source_game_id': source['game_id'],
                       'initial_fen': fen, 'moves': MATE_LINE[:1], 'history': history, 'fen': history[-1],
                       'feature_key': history_key(history), 'future_moves': [], 'future_branches': [],
                       'provenance': 'constructed_prior_owner_probe'})
    games.append(game(CHECK, MATE_LINE, name='unassigned-game'))
    paths = {'games': tmp_path / 'source/games.jsonl', 'owners': tmp_path / 'source/owners.jsonl'}
    proofs = {}
    for kind, values in [('games', games), ('owners', owners)]:
        write_jsonl(paths[kind], values)
        proofs[kind] = paths[kind].parent / f'{kind}-manifest.json'
        atomic_json(proofs[kind], manifest('constructed_native_probe', {'fixture': True}, [], [paths[kind]], {}))
    reference = tmp_path / 'prior'
    reference_paths = []
    for split in ('train', 'validation', 'test'):
        path = reference / f'{split}.jsonl'
        write_jsonl(path, [row for row in owners if row['split'] == split])
        reference_paths.append(path)
    atomic_json(reference / 'manifest.json', manifest('constructed_prior_probe', {}, [], reference_paths, {}))
    return paths, proofs, reference, games, owners


def produce(tmp_path, output=None, workers=1):
    paths, proofs, reference, games, owners = fixture(tmp_path)
    output = output or tmp_path / 'pools'
    result = build([paths['games']], [proofs['games']], paths['owners'], proofs['owners'],
                   [reference], output, workers)
    return output, result, paths, proofs, reference, games, owners


def test_actual_terminal_and_near_mate_targets_preserve_each_native_horizon():
    source = game(CHECK, MATE_LINE)
    result = terminal_candidates(source, 'validation')
    assert result['kind'] == 'mate'
    roots = result['roots']
    assert len(roots) == 3
    first, _, last = roots
    assert first['split'] == 'validation' and first['recorded_source_split'] == 'train'
    assert first['moves'] == [] and first['paper_recorded_mate_witness'] == MATE_LINE
    targets = first['paper_source_targets']
    assert [(t['horizon'], t['plies_before_recorded_mate']) for t in targets] == [(0, 2), (1, 1), (2, 0)]
    assert 'near_mate_check' in targets[0]['classes'] and 'near_mate_check' not in targets[1]['classes']
    assert 'mate' in targets[2]['classes'] and 'near_mate_check' not in targets[2]['classes']
    assert last['future_moves'] == [] and last['moves'] == MATE_LINE
    for root in roots:
        validate_context(root, check_future=True)
        assert root['history'] == source['history'][:root['ply'] + 1]
        assert root['recorded_source_headers'] == source['headers']
        assert root['recorded_continuation_is_best_move_label'] is False


def test_stalemate_is_an_actual_negative_mate_example_and_near_mate_requires_source_witness():
    result = terminal_candidates(game(STALEMATE, []), 'train')
    assert result['kind'] == 'stalemate'
    root = result['roots'][0]
    assert root['paper_recorded_mate_witness'] == []
    assert root['paper_source_targets'][0]['classes'] == ['generic', 'stalemate']
    import random
    record = next(r for r in paper_records(root, random.Random(7), FrequencySampler()) if r['task_type'] == 'mate')
    assert native_tag(record) == '否'
    assert 'near_mate_check' not in native_classes(CHECK)
    assert 'near_mate_check' not in native_classes(CHECK, 7)


@pytest.mark.parametrize('kind', ['terminal_metadata', 'history', 'crossed_history_ending', 'nonterminal'])
def test_forged_terminal_or_history_is_rejected(kind):
    source = game(CHECK, MATE_LINE)
    if kind == 'terminal_metadata':
        source['native_terminal']['winner'] = 'red'
    elif kind == 'history':
        source['history'][-1] = CHECK
    elif kind == 'crossed_history_ending':
        source = game(START_FEN, ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 3)
    else:
        source = game(CHECK, MATE_LINE[:1])
    with pytest.raises(ValueError, match='metadata|history|terminal prefix'):
        terminal_candidates(source, 'train')


def test_native_history_ending_with_legal_moves_is_counted_but_not_mislabelled_mate():
    source = game(START_FEN, ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2)
    assert terminal_candidates(source, 'train') == {'kind': 'history_ending_with_legal_moves', 'roots': []}


def test_complete_producer_and_source_readback_preserve_ownership_and_exclude_old_roots(tmp_path):
    output, result, paths, proofs, reference, games, owners = produce(tmp_path)
    source_hashes = {name: digest(path) for name, path in paths.items()}
    roots = load_jsonl(output / 'roots.jsonl')
    assert len(roots) == 6 and result['new_roots_by_split'] == {'train': 2, 'validation': 2, 'test': 2}
    assert result['excluded_candidates_by_reason'] == {'duplicate_full_history_root': 3}
    assert result['preassigned_source_games'] == 3
    assert not any(r['game_id'] == 'unassigned-game' for r in roots)
    assert all(r['paper_source_pool_profile'] == POOL_PROFILE for r in roots)
    assert all(r['split'] == r['game_id'].split('-')[0] for r in roots)
    assert result['prior_root_future_witness_color_isolation_verified'] is True
    assert result['paper_difficult_source_mix_applied'] is False
    proof = readback(output, tmp_path / 'readback')
    assert proof['records_regenerated_and_compared'] == {'roots': 6, 'excluded': 3}
    assert proof['source_selection_algorithm_independently_rewritten'] is False
    assert source_hashes == {name: digest(path) for name, path in paths.items()}
    with pytest.raises(ValueError, match='fresh'):
        readback(output, tmp_path / 'readback')
    with pytest.raises(ValueError, match='fresh'):
        build([paths['games']], [proofs['games']], paths['owners'], proofs['owners'], [reference], output)


def test_parallel_workers_produce_the_same_root_and_exclusion_bytes(tmp_path):
    output, _, paths, proofs, reference, _, _ = produce(tmp_path)
    parallel = tmp_path / 'parallel'
    build([paths['games']], [proofs['games']], paths['owners'], proofs['owners'], [reference], parallel, workers=2)
    for name in ('roots.jsonl', 'excluded.jsonl'):
        assert (parallel / name).read_bytes() == (output / name).read_bytes()
    assert readback(parallel, tmp_path / 'parallel-readback', workers=2)['status'] == 'complete'


def test_root_future_and_full_mate_witness_colors_cannot_cross_prior_splits(tmp_path):
    paths, proofs, reference, games, owners = fixture(tmp_path)
    future = games[0]['history'][-1]
    # The held-out prior corpus reserves only a color counterpart of the terminal.
    extra = dict(owners[1], game_id='prior-heldout-terminal', fen=mirror_fen(future), future_moves=[])
    path = reference / 'validation.jsonl'
    write_jsonl(path, [owners[1], extra])
    atomic_json(reference / 'manifest.json', manifest('constructed_prior_probe', {}, [],
                [reference / f'{s}.jsonl' for s in ('train', 'validation', 'test')], {}))
    result = build([paths['games']], [proofs['games']], paths['owners'], proofs['owners'], [reference], tmp_path / 'pools')
    assert result['new_roots_by_split'].get('train', 0) == 0
    assert result['excluded_candidates_by_reason']['root_future_witness_or_color_overlaps_another_split'] == 2
    assert mirror_fen(future).split()[:2] in [p.split()[:2] for p in root_footprint(terminal_candidates(games[0], 'train')['roots'][0])]


def test_mate_witness_reservation_extends_beyond_the_eight_queried_future_plies():
    moves = ['e0d0', 'e5e6', 'd0d1', 'e6e7', 'd1d2', 'e7e8',
             'd2d1', 'e8e6', 'd1d0', 'e6e7', 'd0d1', 'e7d7']
    source = game(CHECK, moves)
    root = terminal_candidates(source, 'train')['roots'][0]
    assert len(root['future_moves']) == 8 and len(root['paper_recorded_mate_witness']) == 12
    end = position_key(source['history'][-1])
    assert end not in {position_key(f) for f in replay(root['fen'], root['future_moves'])}
    assert end in root_footprint(root) and position_key(mirror_fen(source['history'][-1])) in root_footprint(root)
    assert [t['horizon'] for t in root['paper_source_targets']] == list(range(9))


@pytest.mark.parametrize('branch_only', [False, True])
def test_structured_prior_explanation_pv_and_branches_reserve_native_terminal_future(tmp_path, branch_only):
    paths, proofs, reference, games, owners = fixture(tmp_path)
    # Reserve unrelated old training geometry, allowing a validation explanation
    # to own the intermediate board and its PV even though future_moves is empty.
    replacement = dict(owners[0], game_id='old-training-other-game', initial_fen=START_FEN,
                       fen=START_FEN, moves=[], history=[START_FEN], feature_key=history_key([START_FEN]))
    write_jsonl(reference / 'train.jsonl', [replacement])
    atomic_json(reference / 'manifest.json', manifest('constructed_prior_probe', {}, [],
                [reference / f'{s}.jsonl' for s in ('train', 'validation', 'test')], {}))
    labels, label_paths = tmp_path / 'prior-explanations', []
    for split, row in [('train', replacement), ('validation', owners[0]), ('test', owners[2])]:
        value = {'pv': [], 'branches': []}
        if split == 'validation':
            if branch_only:
                value['branches'] = [{'pv': MATE_LINE[1:]}]
            else:
                value['pv'] = MATE_LINE[1:]
        record = dict(row, game_id=f'prior-explanation-{split}', split=split, answer=json.dumps(value))
        path = labels / f'{split}.jsonl'
        write_jsonl(path, [record])
        label_paths.append(path)
    atomic_json(labels / 'manifest.json', manifest('constructed_explanation_probe', {}, [], label_paths, {}))
    output = tmp_path / 'pools'
    result = build([paths['games']], [proofs['games']], paths['owners'], proofs['owners'], [reference], output,
                   explanation_reference_data=[labels])
    assert result['new_roots_by_split'].get('train', 0) == 0
    assert result['prior_structured_explanation_pv_and_branches_reserved'] is True
    assert result['excluded_candidates_by_reason']['root_future_witness_or_color_overlaps_another_split'] == 2
    assert readback(output, tmp_path / 'readback')['prior_structured_explanation_pv_and_branches_reserved'] is True


@pytest.mark.parametrize('kind', ['changed_source', 'missing_game', 'conflicting_owner', 'conflicting_game', 'no_prior', 'workers'])
def test_preparation_rejects_missing_changed_or_conflicting_source_contracts(tmp_path, kind):
    paths, proofs, reference, games, owners = fixture(tmp_path)
    refs, workers = [reference], 1
    if kind == 'changed_source':
        with paths['games'].open('a') as handle:
            handle.write('\n')
    elif kind in ('missing_game', 'conflicting_game'):
        if kind == 'missing_game':
            games.pop(0)
        else:
            duplicate = deepcopy(games[0])
            duplicate['native_terminal']['winner'] = 'red'
            games.append(duplicate)
        write_jsonl(paths['games'], games)
        atomic_json(proofs['games'], manifest('constructed_native_probe', {}, [], [paths['games']], {}))
    elif kind == 'conflicting_owner':
        owners.append(dict(owners[0], split='test'))
        write_jsonl(paths['owners'], owners)
        atomic_json(proofs['owners'], manifest('constructed_native_probe', {}, [], [paths['owners']], {}))
    elif kind == 'no_prior':
        refs = []
    else:
        workers = True
    with pytest.raises(ValueError, match='Source root|source game|ownership|source ID|reservation|worker'):
        build([paths['games']], [proofs['games']], paths['owners'], proofs['owners'], refs, tmp_path / 'invalid', workers)


@pytest.mark.parametrize('kind', ['native_target', 'mate_witness', 'extra_root', 'stored_counts', 'changed_source'])
def test_full_source_readback_rejects_tampered_roots_witnesses_counts_and_tail(tmp_path, kind):
    output, _, paths, _, _, _, _ = produce(tmp_path)
    proof_path = output / 'manifest.json'
    proof = json.loads(proof_path.read_text())
    if kind == 'changed_source':
        with paths['games'].open('a') as handle:
            handle.write('\n')
    elif kind == 'stored_counts':
        proof['verification']['new_roots_by_split']['train'] += 1
    else:
        path = output / 'roots.jsonl'
        rows = load_jsonl(path)
        if kind == 'extra_root':
            rows.append(rows[0])
        elif kind == 'mate_witness':
            rows[0]['paper_recorded_mate_witness'] = []
        else:
            rows[0]['paper_source_targets'][0]['classes'] = ['generic', 'stalemate']
        write_jsonl(path, rows)
        proof['outputs'][str(path)] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    atomic_json(proof_path, proof)
    with pytest.raises(ValueError, match='bytes changed|differs from|extra records|counts or isolation'):
        readback(output, tmp_path / 'invalid-readback')
