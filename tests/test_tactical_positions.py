"""Real native position checks and complete search/collection boundary fixtures."""
import hashlib
import json

import pytest

from xqgeneral.evidence import atomic_json, digest, history_key, load_jsonl, manifest, write_jsonl
from xqgeneral.human_games import assigned_split
from xqgeneral.recorded_search_inputs import TACTICAL_CONTEXT_FIELDS, checked_game, prepare, prepared_roots, root_order
from xqgeneral.rules import START_FEN, adjudicate, legal_moves, play
from xqgeneral.symmetry import mirror_fen
from xqgeneral.tactical_positions import ACQUISITION_KIND, REPOSITORY, import_positions, native_position, position_orbit
from test_recorded_search_inputs import fixture, identities
from test_search_distillation import ControlledOracle, ControlledPredictor, root_record
from test_search_collection import TEACHER, response
from xqgeneral.search_distillation import SearchMiner
from xqgeneral.collect_search import consolidated_label


def origin():
    return {'repository': 'controlled position fixture', 'revision': '0' * 40, 'row_index': 0}


def acquisition(tmp_path, rows):
    cache = tmp_path / 'cache'
    cache.mkdir()
    path = cache / 'endgames_all.json'
    path.write_text(json.dumps(rows, ensure_ascii=False))
    raw = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    root = tmp_path / 'acquisition'
    root.mkdir()
    atomic_json(root / 'pinned-source.json', {'repository': REPOSITORY, 'revision': '0' * 40,
        'actual_git_tree_sha': '1' * 40, 'checked_against_actual_tree_object': True,
        'selected_blobs': [{'path': path.name, 'mode': '100644', 'type': 'blob', 'sha': blob, 'size': len(raw)}]})
    atomic_json(root / 'cached-source-files.json', [{'source_path': path.name, 'local_path': str(path),
        'git_blob_sha1': blob, 'sha256': digest(path), 'bytes': len(raw)}])
    rebind_acquisition(root)
    return cache, root


def rebind_acquisition(root):
    atomic_json(root / 'manifest.json', manifest(ACQUISITION_KIND,
        {'repository': REPOSITORY, 'revision': '0' * 40}, [root / 'pinned-source.json'],
        [root / 'cached-source-files.json']))


@pytest.mark.parametrize('fen', [None, '', '9/9 w', START_FEN.replace(' w ', ' r '),
    '9/9/9/9/9/9/9/9/9/9 w - - 0 1', START_FEN.rsplit(' ', 1)[0] + ' 0',
    START_FEN.rsplit(' ', 2)[0] + ' -1 1'])
def test_bad_position_inputs_fail_before_unsafe_native_loading(fen):
    with pytest.raises(ValueError):
        native_position({'fen': fen}, origin())


@pytest.mark.parametrize('two_fields', [False, True])
def test_standalone_initial_board_keeps_unknown_past_and_does_not_use_source_answer(two_fields):
    raw = ' '.join(START_FEN.split()[:2]) if two_fields else START_FEN
    game = native_position({'fen': raw, 'bestMove': 'a0a9\0', 'name': 'unverified source title'}, origin())
    game['split'] = assigned_split(game['game_id'], 20261051)
    checked_game(game, 20261051)
    assert game['initial_fen'] == START_FEN and game['source_initial_fen'] == raw
    assert game['moves'] == [] and game['history'] == [START_FEN]
    assert game['pre_fragment_game_history_available'] is False
    assert game['supplied_history_starts_at_standard_initial_position'] is (not two_fields)
    assert not game['provided_line_is_best_move_label'] and not game['source_move_optimality_proven']
    assert game['declared_participants'] == {'Red': None, 'Black': None}
    assert game['source']['missing_fen_counters_use_native_defaults'] is two_fields
    assert 'bestMove' not in game['source'] and 'name' not in game['headers']
    assert list(root_order(game, 20261013, 0)) == [(0, 0)]
    assert list(root_order(game, 20261013, 1)) == []


@pytest.mark.parametrize('kind', ['recorded_human_match', 'published_recorded_match',
    'recorded_computer_match', 'recorded_human_computer_match', 'recorded_tactical_line'])
def test_position_support_does_not_allow_empty_recorded_full_games_or_fragments(kind):
    game = native_position({'fen': START_FEN}, origin())
    game.update(source_kind=kind, split=assigned_split(game['game_id'], 20261051))
    with pytest.raises(ValueError, match='history dimensions'):
        checked_game(game, 20261051)


@pytest.mark.parametrize('past', [True, 0])
def test_standalone_context_cannot_invent_prior_history_or_reencode_false(past):
    game = native_position({'fen': START_FEN}, origin())
    game.update(split=assigned_split(game['game_id'], 20261051), pre_fragment_game_history_available=past)
    with pytest.raises(ValueError, match='Tactical source context'):
        checked_game(game, 20261051)


def test_canonical_standalone_position_cannot_acquire_a_recorded_move():
    game = native_position({'fen': START_FEN}, origin())
    game['moves'] = ['b0c2']
    game['history'].append(play(START_FEN, 'b0c2'))
    game['game_id'] = 'recorded-' + hashlib.sha256(json.dumps(
        [game['initial_fen'], game['moves']], separators=(',', ':')).encode()).hexdigest()
    game['split'] = assigned_split(game['game_id'], 20261051)
    with pytest.raises(ValueError, match='cannot invent'):
        checked_game(game, 20261051)


def test_native_import_deduplicates_source_and_colors_before_partitioning_and_preserves_rejections(tmp_path):
    rows = [{'fen': START_FEN, 'bestMove': 'a0a9\0'}, {'fen': mirror_fen(START_FEN)},
            {'fen': ' '.join(START_FEN.split()[:2])},
            {'fen': play(START_FEN, 'b0c2')}, {'fen': '9/9 w'}]
    cache, source = acquisition(tmp_path, rows)
    output = tmp_path / 'import'
    result = import_positions(cache, source, output)
    games = load_jsonl(output / 'games.jsonl')
    assert result['source_position_rows'] == 5 and len(games) == 2
    assert result['duplicate_positions'] == 2 and result['quarantined_positions'] == 1
    assert len({position_orbit(g['initial_fen']) for g in games}) == len(games)
    for game in games:
        checked_game(game, 20261051)
        assert game['moves'] == [] and game['history'] == [game['initial_fen']]
    first = next(g for g in games if g['initial_fen'] == START_FEN)
    assert {r['retained_split'] for r in load_jsonl(output / 'duplicates.jsonl')} == {first['split']}
    assert result['native_verified_recorded_plies'] == 0
    assert result['heldout_course_or_explanation_branch_isolation_completed_by_import'] is False
    with pytest.raises(FileExistsError):
        import_positions(cache, source, output)


def test_completed_previous_native_histories_and_their_colors_are_excluded(tmp_path):
    args, games = fixture(tmp_path)
    previous = args['games'][0]
    atomic_json(previous / 'manifest.json', manifest('recorded_games_native_import',
        {'seed': 20261051}, [], [previous / 'games.jsonl'],
        {'all_retained_moves_native_legal': True, 'full_history_terminal_checks_passed': True,
         'game_split_assigned_before_questions': True}))
    old = games[0]['history'][3]
    new = next(play(START_FEN, m) for m in legal_moves(START_FEN)
               if position_orbit(play(START_FEN, m)) not in {
                   position_orbit(fen) for game in games for fen in game['history']})
    cache, source = acquisition(tmp_path, [{'fen': mirror_fen(old)}, {'fen': new}])
    result = import_positions(cache, source, tmp_path / 'import', previous=[previous])
    assert result['new_unique_native_source_positions'] == 1 and result['duplicate_positions'] == 1
    assert load_jsonl(tmp_path / 'import/games.jsonl')[0]['initial_fen'] == new


@pytest.mark.parametrize('change', ['bytes', 'path', 'tree_mode'])
def test_pinned_cache_changes_fail_before_creating_an_import(tmp_path, change):
    cache, source = acquisition(tmp_path, [{'fen': START_FEN}])
    if change == 'bytes':
        (cache / 'endgames_all.json').write_text('[]')
    elif change == 'path':
        values = json.loads((source / 'cached-source-files.json').read_text())
        values[0]['local_path'] = str(source / 'pinned-source.json')
        atomic_json(source / 'cached-source-files.json', values)
        rebind_acquisition(source)
    else:
        values = json.loads((source / 'pinned-source.json').read_text())
        values['selected_blobs'][0]['mode'] = '120000'
        atomic_json(source / 'pinned-source.json', values)
        rebind_acquisition(source)
    with pytest.raises(ValueError):
        import_positions(cache, source, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def position_pool(tmp_path):
    args, _ = fixture(tmp_path)
    rows = [{'fen': play(START_FEN, move)} for move in sorted(legal_moves(START_FEN))]
    cache, source = acquisition(tmp_path, rows)
    imported = tmp_path / 'positions'
    import_positions(cache, source, imported)
    args['games'].append(imported)
    pool = tmp_path / 'pool'
    counts = prepare(**args, output=pool, min_ply=0, candidate_kinds=['recorded_tactical_position'])
    return args, pool, counts


def test_real_preparation_and_consumption_keep_zero_move_roots_and_global_color_reservations(tmp_path):
    args, pool, counts = position_pool(tmp_path)
    rows = load_jsonl(pool / 'roots.jsonl')
    assert len(rows) > 1
    assert counts['isolated_source_position_candidates'] == len(rows)
    for row in rows:
        assert row['ply'] == 0 and row['moves'] == row['future_moves'] == row['future_branches'] == []
        assert row['history'] == [row['initial_fen']] and history_key(row['history']) == row['feature_key']
        assert row['recorded_source_context']['pre_fragment_game_history_available'] is False
    selected, reservations = prepared_roots(pool, args['data'], 1, 20261013, identities(args))
    assert selected == rows[:1]
    assert all(position_orbit(row['fen']) not in {position_orbit(p + ' - - 0 1') for p in reservations}
               for row in rows)


@pytest.mark.parametrize('change', ['past', 'move', 'future', 'ply_bool', 'kind'])
def test_whole_position_pool_tail_is_checked_when_only_one_root_is_requested(tmp_path, change):
    args, pool, _ = position_pool(tmp_path)
    rows = load_jsonl(pool / 'roots.jsonl')
    row = rows[-1]
    if change == 'past': row['recorded_source_context']['pre_fragment_game_history_available'] = 0
    elif change == 'move': row['moves'] = ['b0c2']
    elif change == 'future': row['future_moves'] = ['b0c2']
    elif change == 'ply_bool': row['ply'] = False
    else: row['recorded_source_kind'] = 'recorded_human_match'
    write_jsonl(pool / 'roots.jsonl', rows)
    proof = json.loads((pool / 'manifest.json').read_text())
    proof['outputs'][str(pool / 'roots.jsonl')] = {'sha256': digest(pool / 'roots.jsonl'),
                                                'bytes': (pool / 'roots.jsonl').stat().st_size}
    atomic_json(pool / 'manifest.json', proof)
    with pytest.raises(ValueError):
        prepared_roots(pool, args['data'], 1, 20261013, identities(args))


def test_native_terminal_source_positions_never_become_search_roots():
    before = '4k4/3R5/5R3/9/9/9/9/9/9/5K3 w - - 0 1'
    terminal = play(before, 'f0f1')
    game = native_position({'fen': terminal}, origin())
    assert adjudicate(game['initial_fen'], [])['ended'] is True
    assert list(root_order(game, 20261013, 0)) == []


def test_actual_search_packets_and_labels_preserve_position_only_unknown_past():
    game = native_position({'fen': START_FEN}, origin())
    context = {key: game[key] for key in TACTICAL_CONTEXT_FIELDS}
    row = dict(root_record(), recorded_source_kind=game['source_kind'], recorded_source_context=context)
    query, trace, reason = SearchMiner(ControlledPredictor(), ControlledOracle(), set()).mine(row)
    assert reason == 'accepted_for_consolidation'
    packet = json.loads(query['messages'][1]['content'])
    assert packet['recorded_source_context'] == context
    assert '起点之前的历史未提供' in query['messages'][0]['content']
    assert all(child['record']['recorded_source_context'] == context for child in trace[0]['children'])
    answer = response()
    answer['id'] = query['id']
    assert consolidated_label(query, answer, TEACHER)['recorded_source_context'] == context
