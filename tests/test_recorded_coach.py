from collections import Counter
import copy
import json

import pytest

from xqgeneral.evidence import manifest, position_key, write_jsonl
from xqgeneral.human_games import assigned_split, parse_game
from xqgeneral.recorded_coach import (check_artifacts, curate_original_roots, extract_footprints,
                                    footprint_records, isolate_queries, query_footprint,
                                    verify_original_root)
from xqgeneral.recorded_curriculum import root_candidates
from xqgeneral.recorded_coach_readback import read_query, verify_rejection
from xqgeneral.recorded_coach_isolation_readback import geometric_forecasts
from xqgeneral.explanations import line_facts, move_facts
from xqgeneral.oracle import parse_analysis
from xqgeneral.rules import legal_moves, piece_map, piece_name, play, side
from xqgeneral.symmetry import mirror_fen


def game(first='H2-E2'):
    value = parse_game((f'[Game "Chinese Chess"]\n[Date "2021-01-01"]\n'
        f'[Red "棋手甲"]\n[Black "棋手乙"]\n[Result "1/2-1/2"]\n\n'
        f'1. {first} H9-G7 2. H0-G2 I9-H9 3. B0-C2 B7-E7 1/2-1/2').encode())
    value.update(split=assigned_split(value['game_id'], 20261051), players=['棋手甲', '棋手乙'],
                 source_kind='recorded_human_match', provenance='recorded_source',
                 source={'declared_license': 'test_fixture'})
    return value


def original_roots(value):
    return [root for root, _ in root_candidates(value, seed=7, min_ply=1)]


def query(first, turn, identity):
    root = next(row for row in original_roots(game(first)) if row['fen'].split()[1] == turn)
    # Isolation preserves assigned input splits; these isolated test records use
    # the same train split to exercise both colors without depending on hash luck.
    root = dict(root, split='train')
    moves = legal_moves(root['fen'])[:2]
    return {'id': identity, 'feature_key': root['feature_key'], 'record': root,
            'oracle': {'best_move': moves[0], 'candidates': [{'move': m, 'pv': [m]} for m in moves]}}


def audits(queries):
    return {q['id']: {'id': q['id'], 'feature_key': q['feature_key'], 'native_forecasts_valid': True}
            for q in queries}


def empty_footprints():
    return {k: {s: [] for s in ['train', 'validation', 'test']}
            for k in ['combined_positions', 'combined_games']}


QUOTAS = {'train': 2, 'validation': 0, 'test': 0}


def test_curator_preserves_original_histories_and_both_colors_without_derived_roots():
    source = game()
    roots = original_roots(source)
    quotas = {s: 2 if s == source['split'] else 0 for s in QUOTAS}
    chosen, proof = curate_original_roots(iter(roots), {source['game_id']: source}, quotas)
    assert len(chosen) == 2 and {r['fen'].split()[1] for r in chosen} == {'w', 'b'}
    assert all(r['split'] == source['split'] and r['feature_key'] in {x['feature_key'] for x in roots}
               and 'augmentation_parent' not in r for r in chosen)
    assert proof['maximum_selected_roots_per_original_game'] == 2
    excluded = {chosen[0]['feature_key']}
    fresh, _ = curate_original_roots(iter(roots), {source['game_id']: source}, quotas, excluded)
    assert not {r['feature_key'] for r in fresh} & excluded


def test_curator_rejects_source_metadata_and_native_history_changes():
    source = game()
    roots = original_roots(source)
    quotas = {s: 2 if s == source['split'] else 0 for s in QUOTAS}
    changed = copy.deepcopy(roots)
    changed[0]['recorded_source_headers']['Red'] = '不同署名'
    with pytest.raises(ValueError, match='selected game metadata'):
        curate_original_roots(iter(changed), {source['game_id']: source}, quotas)
    changed = copy.deepcopy(roots[0])
    changed['history'][-1] = changed['history'][0]
    with pytest.raises(ValueError, match='native full history'):
        verify_original_root((changed, 20261051))


def test_query_footprint_reserves_every_candidate_recorded_future_and_color_counterpart():
    q = query('H2-E2', 'b', 'fixture-black')
    positions = query_footprint(q)
    root = q['record']
    expected = {position_key(root['fen']), position_key(play(root['fen'], root['future_moves'][0]))}
    expected.update(position_key(play(root['fen'], c['pv'][0])) for c in q['oracle']['candidates'])
    assert expected <= positions
    assert {position_key(mirror_fen(fen)) for fen in expected} <= positions
    changed = copy.deepcopy(q)
    changed['oracle']['candidates'][1]['pv'] = ['a0a0']
    with pytest.raises(ValueError, match='Illegal'):
        query_footprint(changed)


def test_independent_geometric_readback_matches_completed_full_history_footprints():
    for q in [query('H2-E2', 'b', 'fixture-black'), query('B2-E2', 'w', 'fixture-red')]:
        assert geometric_forecasts(q) == query_footprint(q)
        changed = copy.deepcopy(q)
        changed['oracle']['candidates'][1]['pv'] = ['a0a0']
        with pytest.raises(ValueError, match='Illegal'):
            geometric_forecasts(changed)


def test_isolation_serial_and_spawned_workers_select_identical_unchanged_queries():
    queries = [query('H2-E2', 'b', 'fixture-black'), query('B2-E2', 'w', 'fixture-red')]
    original = copy.deepcopy(queries)
    a = isolate_queries(queries, empty_footprints(), audits(queries), QUOTAS, workers=1)
    b = isolate_queries(queries, empty_footprints(), audits(queries), QUOTAS, workers=2)
    assert a == b and queries == original
    assert a[0] == sorted(queries, key=lambda q: q['id'])
    assert a[3]['new_query_root_and_forecast_game_split_overlap'] == 0


@pytest.mark.parametrize('color_counterpart', [False, True])
def test_isolation_rejects_other_split_candidate_future_even_if_roots_differ(color_counterpart):
    queries = [query('H2-E2', 'b', 'fixture-black'), query('B2-E2', 'w', 'fixture-red')]
    footprints = empty_footprints()
    q = queries[0]
    fen = play(q['record']['fen'], q['oracle']['candidates'][1]['pv'][0])
    assert position_key(fen) != position_key(q['record']['fen'])
    footprints['combined_positions']['test'] = [position_key(mirror_fen(fen) if color_counterpart else fen)]
    with pytest.raises(ValueError, match='Insufficient isolated') as failure:
        isolate_queries(queries, footprints, audits(queries), QUOTAS)
    assert failure.value.quota_report['train']['unfilled_side_quotas'] == {'w': 0, 'b': 1}


def test_isolation_rejects_shared_game_identity_and_incomplete_or_wrong_native_audits():
    queries = [query('H2-E2', 'b', 'fixture-black'), query('B2-E2', 'w', 'fixture-red')]
    footprint = empty_footprints()
    footprint['combined_games']['validation'] = [queries[0]['record']['game_id']]
    with pytest.raises(ValueError, match='Insufficient isolated'):
        isolate_queries(queries, footprint, audits(queries), QUOTAS)
    with pytest.raises(ValueError, match='coverage'):
        isolate_queries(queries, empty_footprints(), audits(queries[:1]), QUOTAS)
    changed = audits(queries)
    changed[queries[0]['id']]['feature_key'] = 'different-history'
    with pytest.raises(ValueError, match='another history'):
        isolate_queries(queries, empty_footprints(), changed, QUOTAS)


def test_new_teacher_forecasts_cannot_overlap_across_selected_splits():
    train = [query('H2-E2', 'b', 'train-black'), query('B2-E2', 'w', 'train-red')]
    heldout = copy.deepcopy(train)
    for q in heldout:
        q['id'] = q['id'].replace('train', 'test')
        q['record']['split'] = 'test'
        q['record']['game_id'] += '-heldout-fixture'
    quotas = dict(QUOTAS, test=2)
    with pytest.raises(ValueError, match='Insufficient isolated') as failure:
        isolate_queries([*train, *heldout], empty_footprints(), audits([*train, *heldout]), quotas)
    assert failure.value.quota_report['test']['unfilled_side_quotas'] == {'w': 0, 'b': 0}
    assert failure.value.quota_report['train']['unfilled_side_quotas'] == {'w': 1, 'b': 1}


def test_streamed_prior_footprints_preserve_recorded_future_and_structured_answer_pvs(tmp_path):
    q = query('H2-E2', 'b', 'fixture-black')
    row = dict(q['record'], answer=json.dumps({'pv': q['oracle']['candidates'][0]['pv'],
               'branches': [{'pv': q['oracle']['candidates'][1]['pv']}]}))
    path = tmp_path / 'train.jsonl'
    write_jsonl(path, [row, row])
    counts = Counter()
    rows = footprint_records([path], counts, explanation=True)
    next_row = next(rows)
    assert counts['train'] == 1 and next_row['future_moves'] == row['future_moves']
    games, roots, positions, contexts = extract_footprints(iter([next_row, *rows]))
    assert contexts == 1 and counts['train'] == 2
    assert games['train'] == {row['game_id']} and roots['train'] == {position_key(row['fen'])}
    expected = {position_key(play(row['fen'], row['future_moves'][0]))}
    expected.update(position_key(play(row['fen'], c['pv'][0])) for c in q['oracle']['candidates'])
    assert expected <= positions['train']


def test_preparation_rejects_changed_completed_artifact_bytes(tmp_path):
    path = tmp_path / 'train.jsonl'
    write_jsonl(path, [{'id': 'fixture'}])
    proof = manifest('fixture', {}, outputs=[path])
    check_artifacts(proof, [path])
    path.write_text('{"id": "changed"}\n')
    with pytest.raises(ValueError, match='identity differs'):
        check_artifacts(proof, [path])


def scored_query():
    row = original_roots(game())[0]
    row = dict(row, id='fixture-' + row['feature_key'])
    moves = legal_moves(row['fen'])[:2]
    raw = [f'info depth 1 multipv {i + 1} score cp {50 - i} pv {m}' for i, m in enumerate(moves)]
    raw.append('bestmove ' + moves[0])
    result = parse_analysis(row['fen'], raw, 100000)
    facts = {'side_to_move': side(row['fen']), 'recommended': moves[0],
             'board': {s: piece_name(p) for s, p in sorted(piece_map(row['fen']).items())},
             'move_facts': move_facts(row['fen'], moves[0]),
             'branches': [{k: c[k] for k in ['move', 'score_type', 'score', 'perspective']} |
                          {'pv': c['pv'], 'line_facts': line_facts(row['fen'], c['pv'])}
                          for c in result['candidates']]}
    return {'id': row['id'] + '-explanation', 'record': row, 'feature_key': row['feature_key'],
            'oracle': result, 'verified_facts': facts}


def test_native_readback_rejects_scores_or_board_facts_changed_after_engine_search():
    q = scored_query()
    assert read_query((q, 100000, 20261051))['native_forecasts_valid']
    changed = copy.deepcopy(q)
    changed['oracle']['candidates'][0]['score'] += 1
    with pytest.raises(ValueError, match='original raw output'):
        read_query((changed, 100000, 20261051))
    changed = copy.deepcopy(q)
    square = next(iter(changed['verified_facts']['board']))
    changed['verified_facts']['board'][square] = '空'
    with pytest.raises(ValueError, match='native facts'):
        read_query((changed, 100000, 20261051))


def test_native_readback_preserves_game_assignment_and_reproduces_unscored_rejection():
    q = scored_query()
    changed = copy.deepcopy(q)
    changed['record']['split'] = 'test' if q['record']['split'] != 'test' else 'train'
    with pytest.raises(ValueError, match='original assignment'):
        read_query((changed, 100000, 20261051))
    rejection = {'record': q['record'], 'engine_output': ['bestmove ' + q['oracle']['best_move']],
                 'reason': 'Engine completed without a scored best-move candidate'}
    verify_rejection(rejection, {q['feature_key']: q['record']}, 100000)
    rejection['reason'] = 'different failure'
    with pytest.raises(ValueError, match='reason differs'):
        verify_rejection(rejection, {q['feature_key']: q['record']}, 100000)
