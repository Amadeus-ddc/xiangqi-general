import random

import pytest

from xqgeneral.evidence import history_key, position_key
from xqgeneral.rules import START_FEN, adjudicate, legal_moves, play, replay
from xqgeneral.selfplay_grounding import grounding_candidates, grounding_pair
from xqgeneral.symmetry import mirror_fen


def query(moves=(), pv=('b0c2', 'b9c7', 'h0g2', 'h9g7', 'a0a1', 'a9a8')):
    history = replay(START_FEN, moves)
    row = {'id': 'root', 'feature_key': history_key(history), 'initial_fen': START_FEN,
           'moves': list(moves), 'history': history, 'fen': history[-1], 'split': 'train',
           'game_id': 'engine-selfplay-training', 'provenance': 'engine_generated_game'}
    return {'id': 'root-query', 'feature_key': row['feature_key'], 'record': row,
            'oracle': {'best_move': pv[0], 'candidates': [{'move': pv[0], 'pv': list(pv)}]}}


def test_grounding_keeps_histories_and_answers_rules_on_the_requested_future_board():
    q = query()
    rows, reason = grounding_pair(q, random.Random(12))
    assert reason is None and len(rows) == 44
    assert {r['stage'] for r in rows} == {'static_current', 'dynamic_current', 'static_future', 'dynamic_future'}
    original = [r for r in rows if 'augmentation_parent' not in r]
    assert all(r['history'] == q['record']['history'] and r['fen'] == START_FEN for r in original)
    for row in original:
        target = row['fen']
        for move in row['future_moves']:
            target = play(target, move)
        assert not adjudicate(row['initial_fen'], row['moves'] + row['future_moves'])['ended']
        if row['task_type'] in {'legal', 'illegal'}:
            valid = row['query']['move'] in legal_moves(target)
            assert row['answer'] == ('合法' if valid else '不合法')
        if row['task_type'] == 'moves':
            assert row['answer'].split() == sorted(m for m in legal_moves(target)
                                                  if m[:2] == row['query']['source'])
    assert len({r['feature_key'] for r in rows}) == 2
    assert all(not r['teacher']['neural_prose_generated'] for r in rows)


def test_grounding_refuses_historical_terminal_futures_and_inconsistent_inputs():
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9']
    assert grounding_pair(query(cycle * 2), random.Random(1)) == ([], 'terminal_root')
    before_end = query((cycle * 2)[:-1], ('c7b9', 'a0a0'))
    assert grounding_pair(before_end, random.Random(1)) == ([], 'no_nonterminal_future')
    bad = query()
    bad['record']['history'] = [play(START_FEN, 'h0g2')]
    with pytest.raises(ValueError, match='full history'):
        grounding_pair(bad, random.Random(1))
    heldout = query()
    heldout['record']['split'] = 'validation'
    with pytest.raises(ValueError, match='training games'):
        grounding_pair(heldout, random.Random(1))


def test_grounding_excludes_deep_future_and_mirrored_reserved_positions():
    q = query()
    seed = 12
    rows, _ = grounding_pair(q, random.Random(seed))
    future_row = next(r for r in rows if r['stage'] == 'dynamic_future' and 'augmentation_parent' not in r)
    target = q['record']['fen']
    for move in future_row['future_moves']:
        target = play(target, move)
    assert position_key(target) != position_key(START_FEN)
    for reserved in [{position_key(target)}, {position_key(mirror_fen(target))}]:
        assert grounding_pair(q, random.Random(seed), reserved) == ([], 'heldout_root_or_future')
    assert grounding_pair(q, random.Random(seed), available_keys={q['feature_key']}) == (
        [], 'root_missing_from_verified_policy_data')


def test_parallel_grounding_preserves_order_labels_and_terminal_rejections():
    import json
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9']
    queries = [query(), query(['b0c2'], ('b9c7', 'h0g2', 'h9g7', 'a0a1')),
               query(cycle * 2), query((cycle * 2)[:-1], ('c7b9', 'a0a0'))]
    for index, q in enumerate(queries):
        q['id'] = f'query-{index}'
    serial = list(grounding_candidates(queries, frozenset(), None, 42, workers=1))
    parallel = list(grounding_candidates(queries, frozenset(), None, 42, workers=2, chunk_size=1))
    assert json.dumps(serial, ensure_ascii=False) == json.dumps(parallel, ensure_ascii=False)
    assert [reason for _, _, reason in serial] == [None, None, 'terminal_root', 'no_nonterminal_future']
