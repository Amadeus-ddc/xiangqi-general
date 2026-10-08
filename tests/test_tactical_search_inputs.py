"""Tactical roots keep their supplied-history boundary through search and labels."""
import copy
import json
from pathlib import Path
import sys

import pytest

from test_recorded_search_inputs import fixture, footprint, identities
from test_search_collection import TEACHER, collect_args, recorded_collection, response
from test_search_distillation import ControlledOracle, ControlledPredictor, root_record
from test_tactical_games import origin, pgn
from xqgeneral.collect_search import consolidated_label
from xqgeneral.evidence import atomic_json, digest, history_key, load_jsonl, manifest, position_key, write_jsonl
from xqgeneral.human_games import assigned_split
from xqgeneral.recorded_search_inputs import TACTICAL_CONTEXT_FIELDS, prepare, prepared_roots, root_order
from xqgeneral.rules import START_FEN, adjudicate, legal_moves, replay
from xqgeneral.search_distillation import SearchMiner
from xqgeneral.symmetry import mirror_fen
from xqgeneral.tactical_games import native_line


def training_fragment(game):
    for ply in range(4, len(game['moves']) - 2):
        raw = pgn(game['history'][ply], game['moves'][ply:ply + 2], players=False)
        value, error = native_line((raw, origin(raw)))
        assert error is None
        value['split'] = assigned_split(value['game_id'], 20261051)
        if value['split'] == 'train':
            return value
    raise AssertionError('Controlled source has no train-owned two-ply fragment')


def tactical_fixture(tmp_path, include_heldout_fragment=False):
    args, games = fixture(tmp_path)
    values = [training_fragment(next(g for g in games if g['split'] == 'train'))]
    if include_heldout_fragment:
        values.append(training_fragment(next(g for g in games if g['split'] == 'validation')))
    source = tmp_path / 'tactical'
    source.mkdir()
    write_jsonl(source / 'games.jsonl', values)
    atomic_json(source / 'manifest.json', manifest('recorded_tactical_lines_native_import',
        {'seed': 20261051}, [], [source / 'games.jsonl'],
        {'all_retained_moves_native_legal': True, 'full_history_terminal_checks_passed': True,
         'game_split_assigned_before_questions': True}))
    args['games'].append(source)
    return args, values


def prepare_tactical(args, output):
    return prepare(**args, output=output, min_ply=0, per_game=6,
                   candidate_kinds=['recorded_tactical_line'])


def rebind_pool(pool):
    path = pool / 'roots.jsonl'
    proof = json.loads((pool / 'manifest.json').read_text())
    proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(pool / 'manifest.json', proof)


def test_short_fragments_include_the_given_root_and_preserve_missing_names_and_past(tmp_path):
    args, values = tactical_fixture(tmp_path)
    output = tmp_path / 'pool'
    counts = prepare_tactical(args, output)
    rows = load_jsonl(output / 'roots.jsonl')
    assert rows and all(row['recorded_source_kind'] == 'recorded_tactical_line' for row in rows)
    first = next(row for row in rows if row['ply'] == 0)
    assert first['moves'] == [] and first['history'] == [values[0]['initial_fen']]
    assert first['recorded_source_context']['pre_fragment_game_history_available'] is False
    assert first['recorded_source_context']['declared_participants'] == {'Red': None, 'Black': None}
    assert first['recorded_source_headers'] == values[0]['headers']
    assert counts['distinct_canonical_training_lines'] == 1
    assert counts['tactical_candidate_counts_by_supplied_history_boundary'] == {
        'pre_fragment_history_unavailable': len(rows)}
    for row in rows:
        assert replay(row['initial_fen'], row['moves']) == row['history']
        assert row['recorded_source_context'] == {k: values[0][k] for k in TACTICAL_CONTEXT_FIELDS}
    selected, _ = prepared_roots(output, args['data'], 1, 20261013, identities(args))
    assert selected == rows[:1]
    with pytest.raises(ValueError, match='No isolated'):
        prepare(**args, output=tmp_path / 'too-late', min_ply=12,
                candidate_kinds=['recorded_tactical_line'])


def test_unselected_full_match_sources_still_reserve_all_heldout_histories_and_colors(tmp_path):
    args, values = tactical_fixture(tmp_path, include_heldout_fragment=True)
    output = tmp_path / 'pool'
    counts = prepare_tactical(args, output)
    rows = load_jsonl(output / 'roots.jsonl')
    reserved = set(json.loads((output / 'heldout-positions.json').read_text()))
    assert all(row['game_id'] != values[1]['game_id'] for row in rows)
    assert {position_key(f) for f in values[1]['history']} <= reserved
    assert {position_key(mirror_fen(f)) for f in values[1]['history']} <= reserved
    assert counts['excluded']['unselected_source_kind'] == 3
    assert counts['excluded']['heldout_root_or_recorded_future'] >= 2
    assert all(not footprint(row) & reserved for row in rows)


def test_the_legal_decision_before_a_one_move_terminal_is_not_removed(tmp_path):
    args, _ = tactical_fixture(tmp_path)
    base = '4k4/3R5/5R3/9/9/9/9/9/9/5K3 w - -'
    for clock in range(10):
        raw = pgn(base + f' {clock} 1', ['f0f1'], players=False)
        game, error = native_line((raw, origin(raw)))
        assert error is None and game['native_terminal']['ended'] is True
        game['split'] = assigned_split(game['game_id'], 20261051)
        if game['split'] == 'train':
            break
    else:
        raise AssertionError('Controlled terminal line has no training split')
    assert not adjudicate(game['initial_fen'], [])['ended']
    assert list(root_order(game, 20261013, 0)) == [(0, 1)]
    source = args['games'][-1]
    write_jsonl(source / 'games.jsonl', [game])
    atomic_json(source / 'manifest.json', manifest('recorded_tactical_lines_native_import',
        {'seed': 20261051}, [], [source / 'games.jsonl']))
    output = tmp_path / 'pool'
    prepare_tactical(args, output)
    row, = load_jsonl(output / 'roots.jsonl')
    assert row['ply'] == 0 and row['future_moves'] == ['f0f1']
    assert adjudicate(row['initial_fen'], row['future_moves'])['ended']
    assert prepared_roots(output, args['data'], 1, 20261013, identities(args))[0] == [row]


@pytest.mark.parametrize('kinds', [[], False, 'recorded_tactical_line', ['unknown'], [None]])
def test_invalid_candidate_kind_selection_fails_before_opening_inputs(tmp_path, kinds):
    with pytest.raises(ValueError, match='source kinds'):
        prepare(['absent'], 'absent', 'absent', 'absent', 'absent', tmp_path / 'out',
                candidate_kinds=kinds)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('change', ['invent_past', 'missing', 'participant', 'source_fen'])
def test_source_context_corruption_fails_before_creating_a_pool(tmp_path, change):
    args, values = tactical_fixture(tmp_path)
    if change == 'invent_past': values[0]['pre_fragment_game_history_available'] = True
    elif change == 'missing': values[0].pop('pre_fragment_game_history_available')
    elif change == 'participant': values[0]['declared_participants']['Red'] = '杜撰棋手'
    else: values[0]['source_initial_fen'] = START_FEN
    source = args['games'][-1]
    write_jsonl(source / 'games.jsonl', values)
    atomic_json(source / 'manifest.json', manifest('recorded_tactical_lines_native_import',
        {'seed': 20261051}, [], [source / 'games.jsonl']))
    with pytest.raises(ValueError, match='Tactical source context'):
        prepare_tactical(args, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


@pytest.mark.parametrize('change', ['invent_past', 'missing', 'participant', 'source_fen', 'kind'])
def test_the_entire_pool_tail_keeps_context_even_when_only_one_root_is_requested(tmp_path, change):
    args, _ = tactical_fixture(tmp_path)
    output = tmp_path / 'pool'
    prepare_tactical(args, output)
    rows = load_jsonl(output / 'roots.jsonl')
    assert len(rows) > 1
    context = rows[-1]['recorded_source_context']
    if change == 'invent_past': context['pre_fragment_game_history_available'] = True
    elif change == 'missing': rows[-1].pop('recorded_source_context')
    elif change == 'participant': context['declared_participants']['Black'] = '杜撰棋手'
    elif change == 'source_fen': context['source_initial_fen'] = START_FEN
    else: rows[-1]['recorded_source_kind'] = 'recorded_human_match'
    write_jsonl(output / 'roots.jsonl', rows)
    rebind_pool(output)
    with pytest.raises(ValueError, match='Tactical source context|ownership'):
        prepared_roots(output, args['data'], 1, 20261013, identities(args))


def test_actual_miner_packets_and_consolidated_labels_preserve_unknown_clock_history():
    initial = ' '.join(START_FEN.split()[:4] + ['8', '12'])
    raw = pgn(initial, players=False)
    game, error = native_line((raw, origin(raw)))
    assert error is None and game['pre_fragment_game_history_available'] is False
    context = {k: game[k] for k in TACTICAL_CONTEXT_FIELDS}
    row = dict(root_record(), initial_fen=game['initial_fen'], fen=game['initial_fen'],
        history=[game['initial_fen']], feature_key=history_key([game['initial_fen']]),
        recorded_source_kind='recorded_tactical_line', recorded_source_context=context)
    query, trace, reason = SearchMiner(ControlledPredictor(), ControlledOracle(), set()).mine(row)
    assert reason == 'accepted_for_consolidation'
    packet = json.loads(query['messages'][1]['content'])
    assert packet['recorded_source_context'] == context
    assert '起点之前的历史未提供' in query['messages'][0]['content']
    assert all(child['record']['recorded_source_context'] == context for child in trace[0]['children'])
    answer = response(); answer['id'] = query['id']
    label = consolidated_label(query, answer, TEACHER)
    assert label['recorded_source_context'] == context and label['initial_fen'] == game['initial_fen']
    assert row['recorded_source_context'] == context


@pytest.mark.parametrize('change', ['kind', 'headers', 'context', 'boolean_encoding'])
def test_collection_rejects_changed_source_context_even_after_query_hash_rebinding(tmp_path, monkeypatch, change):
    from xqgeneral.collect_search import main
    args, pool, miner, query = recorded_collection(tmp_path, descendant=True)
    if change == 'kind': query['record']['recorded_source_kind'] = 'recorded_tactical_line'
    elif change == 'headers': query['record']['recorded_source_headers']['Red'] = '杜撰棋手'
    elif change == 'context': query['record']['recorded_source_context'] = {'pre_fragment_game_history_available': True}
    else:
        path = args['games'][0] / 'games.jsonl'
        games = load_jsonl(path)
        game = next(g for g in games if g['game_id'] == query['record']['game_id'])
        game['source_kind'] = 'recorded_tactical_line'
        game['headers']['FEN'] = game['initial_fen']
        standard = game['initial_fen'].split()[:2] == START_FEN.split()[:2]
        game.update(source_initial_fen=game['initial_fen'],
            declared_participants={k: game['headers'][k] for k in ['Red', 'Black']},
            supplied_history_starts_at_standard_initial_position=standard,
            pre_fragment_game_history_available=standard,
            provided_line_is_best_move_label=False,
            source_comments_or_analysis_branches_used_as_labels=False)
        write_jsonl(path, games)
        proof = json.loads((path.parent / 'manifest.json').read_text())
        proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        atomic_json(path.parent / 'manifest.json', proof)
        new_pool = tmp_path / 'tactical-pool'
        prepare(**args, output=new_pool, min_ply=1, per_game=3)
        original = load_jsonl(new_pool / 'roots.jsonl')[0]
        assert original['id'] == query['source_root_id']
        for field in ['recorded_source_kind', 'recorded_source_headers', 'recorded_source_context']:
            query['record'][field] = copy.deepcopy(original[field])
        query['record']['recorded_source_context']['pre_fragment_game_history_available'] = int(standard)
        proof = json.loads((miner / 'manifest.json').read_text())
        proof['config']['recorded_inputs'] = str(new_pool)
        for name in ['manifest.json', 'roots.jsonl', 'counts.json', 'heldout-positions.json']:
            proof['inputs'].pop(str(pool / name))
            path = new_pool / name
            proof['inputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        atomic_json(miner / 'manifest.json', proof)
    write_jsonl(miner / 'queries.jsonl', [query])
    proof = json.loads((miner / 'manifest.json').read_text())
    path = miner / 'queries.jsonl'
    proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(miner / 'manifest.json', proof)
    monkeypatch.setattr(sys, 'argv', collect_args(tmp_path, args, miner))
    with pytest.raises(ValueError, match='supplied-history context'):
        main()
    assert not (tmp_path / 'accepted').exists()
