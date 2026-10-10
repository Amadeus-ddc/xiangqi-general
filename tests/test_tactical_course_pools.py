from copy import deepcopy
import hashlib
import json

import pytest

from xqgeneral.course_tasks import validate_context
from xqgeneral.evidence import atomic_json, history_key, load_jsonl, manifest, position_key, write_jsonl
from xqgeneral.human_games import assigned_split
from xqgeneral.recorded_course_pools import root_footprint, terminal_candidates
from xqgeneral.rules import START_FEN, adjudicate, in_check, legal_moves, replay
from xqgeneral.symmetry import mirror_fen
from xqgeneral.tactical_course_pools import (
    EXTENSION_ORIGIN, POOL_PROFILE, bound_forecasts, build, readback, source_candidates,
)
from xqgeneral.recorded_search_inputs import BoundInputs, TACTICAL_CONTEXT_FIELDS

CHECK = '4k4/9/9/9/4r4/9/9/9/9/4K4 w - - 0 1'
MATE_LINE = ['e0d0', 'e5d5']
PRE_STALEMATE = '4k4/3R5/5R3/9/9/9/9/9/9/5K3 w - - 0 1'


def game(fen, moves, seed, position=False):
    identity = hashlib.sha256(json.dumps([fen, moves], separators=(',', ':')).encode()).hexdigest()
    name = 'recorded-' + identity
    standard = fen.split()[:2] == START_FEN.split()[:2] and fen.split()[4:] == ['0', '1']
    return {'game_id': name, 'initial_fen': fen, 'source_initial_fen': fen, 'moves': moves,
            'history': replay(fen, moves), 'native_terminal': adjudicate(fen, moves),
            'source_kind': 'recorded_tactical_position' if position else 'recorded_tactical_line',
            'source': {'fixture': True}, 'headers': {'FEN': fen}, 'players': [],
            'declared_participants': {'Red': None, 'Black': None},
            'supplied_history_starts_at_standard_initial_position': standard,
            'pre_fragment_game_history_available': standard and not position,
            'provided_line_is_best_move_label': False,
            'source_comments_or_analysis_branches_used_as_labels': False,
            'provenance': 'constructed_test_probe;not_real_games', 'split': assigned_split(name, seed)}


def fixture(tmp_path):
    fens = []
    for rank in (0, 1, 2):
        rows = CHECK.split()[0].split('/')
        rows[9 - rank] = 'H3K4' if rank == 0 else 'H8'
        fens.append('/'.join(rows) + ' w - - 0 1')
    seed = 0
    while len({game(f, MATE_LINE, seed)['split'] for f in fens}) != 3:
        seed += 1
    games = [game(f, MATE_LINE, seed) for f in fens]
    source = tmp_path / 'tactical-lines'
    write_jsonl(source / 'games.jsonl', games)
    atomic_json(source / 'manifest.json', manifest('recorded_tactical_lines_native_import',
        {'seed': seed}, [], [source / 'games.jsonl'], {}))
    directories = []
    fields = {name: {s: [] for s in ('train', 'validation', 'test')} for name in (
        'course_games', 'course_positions', 'prior_label_games', 'prior_label_positions',
        'combined_games', 'combined_positions')}
    for directory, prefix in [(tmp_path / 'prior-course', 'course'), (tmp_path / 'prior-labels', 'prior_label')]:
        paths = []
        for g in games:
            split = g['split']
            row = {'game_id': f'prior-{prefix}-{split}', 'split': split, 'fen': g['initial_fen'],
                   'future_moves': [], 'future_branches': []}
            path = directory / f'{split}.jsonl'
            write_jsonl(path, [row]); paths.append(path)
            fields[prefix + '_games'][split] = [row['game_id']]
            fields[prefix + '_positions'][split] = sorted({position_key(row['fen']), position_key(mirror_fen(row['fen']))})
        atomic_json(directory / 'manifest.json', manifest('constructed_prior', {}, [], paths, {}))
        directories.append(directory)
    for split in ('train', 'validation', 'test'):
        for kind in ('games', 'positions'):
            fields['combined_' + kind][split] = sorted(set(fields['course_' + kind][split]) |
                                                     set(fields['prior_label_' + kind][split]))
    cached = tmp_path / 'forecasts'
    atomic_json(cached / 'footprints.json', fields)
    summary = {'status': 'complete', 'recorded_futures_and_prior_label_structured_answer_pvs_included': True,
               'course_game_counts': {s: 1 for s in ('train', 'validation', 'test')},
               'course_root_and_forecast_position_counts': {s: 2 for s in ('train', 'validation', 'test')}}
    atomic_json(cached / 'verification.json', summary)
    inputs = [p for d in directories for p in [d / 'manifest.json', *(d / f'{s}.jsonl' for s in ('train', 'validation', 'test'))]]
    atomic_json(cached / 'manifest.json', manifest('recorded_coach_footprints',
        {'data': str(directories[0]), 'prior_labels': str(directories[1])}, inputs,
        [cached / 'footprints.json', cached / 'verification.json'], summary))
    return source, cached / 'manifest.json', seed, games, directories


def test_recorded_tactical_terminal_retains_the_supplied_history_boundary():
    source = game(CHECK, MATE_LINE, 7)
    roots = source_candidates(source)['roots']
    assert len(roots) == 3
    for root in roots:
        assert root['paper_terminal_line_origin'] == 'actual_recorded_source_moves'
        assert {k: root[k] for k in TACTICAL_CONTEXT_FIELDS} == {k: source[k] for k in TACTICAL_CONTEXT_FIELDS}
        assert root['pre_fragment_game_history_available'] is False
        assert root['recorded_continuation_is_best_move_label'] is False
        validate_context(root, check_future=True)
    assert roots[0]['paper_recorded_mate_witness'] == MATE_LINE


def test_native_legal_extensions_never_become_recorded_moves_or_new_game_ids():
    source = game(PRE_STALEMATE, [], 7, position=True)
    before = deepcopy(source)
    result = source_candidates(source)
    assert result['extensions']['stalemate'] > 0 and result['extensions']['mate'] > 0
    assert source == before
    for root in result['roots']:
        assert root['game_id'] == source['game_id'] and root['split'] == source['split']
        assert root['extension_is_recorded_source_move'] is False
        assert root['recorded_future_moves_available'] is False
        assert root['pre_fragment_game_history_available'] is False
        assert root['paper_terminal_line_origin'] == EXTENSION_ORIGIN
        assert root['paper_recorded_mate_witness'] == []
        assert root['paper_generated_terminal_witness'] == ([root['generated_terminal_move']] if root['ply'] == 0 else [])
        assert all('plies_before_recorded_mate' not in target for target in root['paper_source_targets'])
        validate_context(root, check_future=True)
        for target in root['paper_source_targets']:
            if 'stalemate' in target['classes']:
                assert not legal_moves(target['fen']) and not in_check(target['fen'])
            if 'mate' in target['classes']:
                assert not legal_moves(target['fen']) and in_check(target['fen'])
        assert {position_key(f) for f in replay(root['fen'], root['paper_generated_terminal_witness'])} <= root_footprint(root)


def test_nonterminal_lines_are_not_promoted_to_terminal_examples():
    source = game(START_FEN, ['b0c2', 'b9c7'], 7)
    assert source_candidates(source)['roots'] == []
    assert source_candidates(source)['kind'] == 'nonterminal_recorded_line'


def test_history_rule_endings_with_legal_moves_are_counted_separately():
    source = game(START_FEN, ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2, 7)
    assert source_candidates(source)['kind'] == 'history_ending_with_legal_moves'
    assert source_candidates(source)['roots'] == []


@pytest.mark.parametrize('field,value', [
    ('pre_fragment_game_history_available', True), ('provided_line_is_best_move_label', True),
    ('source_initial_fen', START_FEN), ('declared_participants', {'Red': 'invented', 'Black': None}),
    ('source_comments_or_analysis_branches_used_as_labels', True),
])
def test_forged_tactical_context_is_rejected_before_using_recorded_terminal(field, value):
    source = game(CHECK, MATE_LINE, 7)
    source[field] = value
    with pytest.raises(ValueError, match='context'):
        terminal_candidates(source, source['split'])


def test_complete_source_production_and_readback_match_each_native_target(tmp_path):
    source, forecasts, seed, games, _ = fixture(tmp_path)
    output = tmp_path / 'pools'
    result = build([source], forecasts, output, split_seed=seed)
    roots = load_jsonl(output / 'roots.jsonl')
    assert len(roots) == 9
    assert result['new_roots_by_split'] == {'validation': 3, 'test': 3, 'train': 3}
    assert all(r['paper_source_pool_profile'] == POOL_PROFILE for r in roots)
    assert result['generated_legal_one_ply_terminal_moves_by_split_class'] == {}
    proof = readback(output, tmp_path / 'readback')
    assert proof['records_regenerated_and_compared'] == {'roots': 9}
    assert proof['source_selection_algorithm_independently_rewritten'] is False
    assert proof['cached_forecast_native_parser_reexecuted'] is False


def test_parallel_source_workers_match_serial_root_and_exclusion_bytes(tmp_path):
    source, forecasts, seed, _, _ = fixture(tmp_path)
    build([source], forecasts, tmp_path / 'serial', split_seed=seed)
    build([source], forecasts, tmp_path / 'parallel', split_seed=seed, workers=2)
    for name in ('roots.jsonl', 'excluded.jsonl'):
        assert (tmp_path / 'serial' / name).read_bytes() == (tmp_path / 'parallel' / name).read_bytes()
    assert readback(tmp_path / 'parallel', tmp_path / 'readback', workers=2)['status'] == 'complete'


def test_generated_source_pool_retains_all_terminal_descendants_and_counts_repeated_parents(tmp_path):
    lines, forecasts, seed, _, _ = fixture(tmp_path)
    source = game(PRE_STALEMATE, [], seed, position=True)
    positions = tmp_path / 'tactical-positions'
    write_jsonl(positions / 'games.jsonl', [source])
    atomic_json(positions / 'manifest.json', manifest('isolated_tactical_positions_native_import',
        {'seed': seed}, [], [positions / 'games.jsonl'], {}))
    output = tmp_path / 'pools'
    result = build([lines, positions], forecasts, output, split_seed=seed, workers=2)
    generated = [r for r in load_jsonl(output / 'roots.jsonl') if r['game_id'] == source['game_id']]
    extensions = source_candidates(source)['extensions']
    assert len(generated) == 1 + sum(extensions.values())
    assert sum(r['ply'] == 0 for r in generated) == 1
    assert result['excluded_candidates_by_reason']['duplicate_full_history_root'] == sum(extensions.values()) - 1
    assert result['generated_extensions_are_additional_independent_games'] is False
    assert result['new_source_ids_by_split'][source['split']] == 2
    for root in generated:
        assert position_key(source['initial_fen']) in root_footprint(root)
        assert position_key(mirror_fen(source['initial_fen'])) in root_footprint(root)
    assert readback(output, tmp_path / 'readback', workers=2)['status'] == 'complete'


def test_prior_full_history_roots_are_excluded_without_reassigning_owners(tmp_path):
    source, forecasts, seed, games, _ = fixture(tmp_path)
    path = tmp_path / 'owners.jsonl'
    owners = [{'game_id': g['game_id'], 'recorded_source_game_id': g['game_id'], 'split': g['split'],
               'feature_key': history_key(g['history'][:1])} for g in games]
    write_jsonl(path, owners)
    proof = tmp_path / 'owners-manifest.json'
    atomic_json(proof, manifest('constructed_owner_proof', {}, [], [path], {}))
    result = build([source], forecasts, tmp_path / 'pools', split_seed=seed, owners=path, owner_manifest=proof)
    assert result['new_roots_by_split'] == {'validation': 2, 'test': 2, 'train': 2}
    assert result['excluded_candidates_by_reason'] == {'duplicate_full_history_root': 3}


def test_future_terminal_color_overlap_with_other_source_split_is_excluded(tmp_path):
    source, forecasts, seed, games, _ = fixture(tmp_path)
    train = next(g for g in games if g['split'] == 'train')
    heldout = deepcopy(next(g for g in games if g['split'] == 'validation'))
    heldout.update(game_id='heldout-reservation-only', initial_fen=mirror_fen(train['history'][-1]),
                   history=[mirror_fen(train['history'][-1])], moves=[])
    extra = tmp_path / 'extra-source'
    write_jsonl(extra / 'games.jsonl', [heldout])
    atomic_json(extra / 'manifest.json', manifest('constructed_complete_source', {}, [], [extra / 'games.jsonl'], {}))
    # Reservation split is assigned before questions, without copying an archive default.
    heldout['game_id'] = next('heldout-' + str(i) for i in range(1000) if assigned_split('heldout-' + str(i), seed) == 'validation')
    write_jsonl(extra / 'games.jsonl', [heldout])
    atomic_json(extra / 'manifest.json', manifest('constructed_complete_source', {}, [], [extra / 'games.jsonl'], {}))
    result = build([source], forecasts, tmp_path / 'pools', split_seed=seed, reservation_games=[source, extra])
    assert result['new_roots_by_split'].get('train', 0) == 0
    assert result['excluded_candidates_by_reason']['root_future_witness_or_color_overlaps_another_split'] == 3


@pytest.mark.parametrize('target', ['history_boundary', 'target_class', 'extra_tail', 'verification'])
def test_semantically_forged_outputs_rejected_even_after_rebinding_manifest(tmp_path, target):
    source, forecasts, seed, _, _ = fixture(tmp_path)
    output = tmp_path / 'pools'
    build([source], forecasts, output, split_seed=seed)
    if target == 'verification':
        path = output / 'verification.json'
        value = json.loads(path.read_text()); value['missing_pre_fragment_history_preserved'] = False
        atomic_json(path, value)
    else:
        rows = load_jsonl(output / 'roots.jsonl')
        if target == 'history_boundary': rows[0]['pre_fragment_game_history_available'] = True
        elif target == 'target_class': rows[0]['paper_source_targets'][0]['classes'] = ['mate']
        else: rows.append(rows[-1])
        write_jsonl(output / 'roots.jsonl', rows)
    proof = json.loads((output / 'manifest.json').read_text())
    atomic_json(output / 'manifest.json', manifest(proof['kind'], proof['config'], proof['inputs'], proof['outputs'], proof['verification']))
    with pytest.raises(ValueError, match='differs|extra|verification'):
        readback(output, tmp_path / 'readback')


@pytest.mark.parametrize('kind', ['union', 'color_conflict', 'counts', 'source_bytes'])
def test_completed_forecast_reuse_checks_union_colors_counts_and_original_bytes(tmp_path, kind):
    _, forecasts, _, _, directories = fixture(tmp_path)
    proof = json.loads(forecasts.read_text())
    if kind == 'source_bytes':
        with (directories[0] / 'train.jsonl').open('a') as f: f.write('\n')
    else:
        path = forecasts.parent / 'footprints.json'
        value = json.loads(path.read_text())
        if kind == 'union': value['combined_games']['train'].append('unbound-game')
        elif kind == 'color_conflict':
            train = value['course_positions']['train'][0]
            for field in ('course_positions', 'prior_label_positions', 'combined_positions'):
                value[field]['train'] = [train]
                value[field]['validation'] = sorted({position_key(mirror_fen(train))})
            proof['verification']['course_root_and_forecast_position_counts'].update(train=1, validation=1)
        else: proof['verification']['course_game_counts']['train'] = 2
        atomic_json(path, value)
        atomic_json(forecasts.parent / 'verification.json', proof['verification'])
        atomic_json(forecasts, manifest(proof['kind'], proof['config'], proof['inputs'], proof['outputs'], proof['verification']))
    with pytest.raises(ValueError, match='union|color|counts|changed'):
        bound_forecasts(forecasts, BoundInputs())


def test_actual_cached_forecasts_need_not_already_contain_all_color_counterparts(tmp_path):
    _, forecasts, _, _, _ = fixture(tmp_path)
    proof = json.loads(forecasts.read_text())
    path = forecasts.parent / 'footprints.json'
    value = json.loads(path.read_text())
    for field in ('course_positions', 'prior_label_positions', 'combined_positions'):
        for split in ('train', 'validation', 'test'):
            value[field][split] = value[field][split][:1]
    proof['verification']['course_root_and_forecast_position_counts'] = {s: 1 for s in ('train', 'validation', 'test')}
    atomic_json(path, value)
    atomic_json(forecasts.parent / 'verification.json', proof['verification'])
    atomic_json(forecasts, manifest(proof['kind'], proof['config'], proof['inputs'], proof['outputs'], proof['verification']))
    _, positions = bound_forecasts(forecasts, BoundInputs())
    assert len(positions) == 6
    assert all(positions[position_key(mirror_fen(k))] == split for k, split in positions.items())


def test_mutated_tactical_archive_and_split_seed_are_rejected(tmp_path):
    source, forecasts, seed, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='seed'):
        build([source], forecasts, tmp_path / 'wrong-seed', split_seed=seed + 1)
    with (source / 'games.jsonl').open('a') as f: f.write('\n')
    with pytest.raises(ValueError, match='changed'):
        build([source], forecasts, tmp_path / 'changed-source', split_seed=seed)


def test_fresh_output_and_worker_contract_are_required(tmp_path):
    source, forecasts, seed, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='fresh'):
        build([source], forecasts, tmp_path, split_seed=seed)
    with pytest.raises(ValueError, match='positive'):
        build([source], forecasts, tmp_path / 'invalid', split_seed=seed, workers=True)
