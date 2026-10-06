import random
import pytest
from xqgeneral.curriculum_data import illegal_probe, make_records, verify_splits
from xqgeneral.rules import START_FEN, legal_moves, piece_map, replay, adjudicate


def test_all_courses_have_counterexamples_and_independent_splits():
    rows = make_records(games=10, seed=19, plies=24)
    proof = verify_splits(rows)
    assert set(proof['games']) == {'train', 'validation', 'test'}
    for stage in ['dynamic_current', 'dynamic_future']:
        assert {r['answer'] for r in rows if r['stage'] == stage and r['task_type'] in {'legal', 'illegal'}} == {'合法', '不合法'}
    for row in rows:
        query = row['query']
        fen = replay(row['fen'], row['future_moves'])[-1]
        board = piece_map(fen)
        if row['task_type'] in {'legal', 'illegal'}:
            assert (query['move'] in legal_moves(fen)) == (row['answer'] == '合法')
        if row['task_type'] == 'count':
            assert row['answer'] == str(sum(p == query['symbol'] for p in board.values()))
        if row['task_type'] == 'locate':
            assert row['answer'] == (' '.join(sorted(s for s, p in board.items() if p == query['symbol'])) or '无')
    bad = dict(rows[0], split='test', id='leaked')
    with pytest.raises(ValueError, match='leakage'):
        verify_splits([rows[0], bad])


def test_negative_probe_uses_movers_piece():
    move = illegal_probe(START_FEN, random.Random(7))
    assert move not in legal_moves(START_FEN)
    assert piece_map(START_FEN)[move[:2]].isupper()
    assert move[:2] != move[2:]


def test_full_history_repetition_and_perpetual_check():
    assert adjudicate(START_FEN, [])['ended'] is False
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2
    assert adjudicate(START_FEN, cycle)['winner'] is None
    assert adjudicate(START_FEN, cycle)['ended'] is True
    fen = '4k4/3R5/9/9/4P4/9/9/9/9/4K4 w - - 0 1'
    result = adjudicate(fen, ['d8e8', 'e9d9', 'e8d8', 'd9e9'] * 2)
    assert result['ended'] and result['winner'] == 'black'


def test_perpetual_chase_loses_for_chaser():
    fen = '4k4/9/9/1n7/4P4/3R5/9/9/9/4K4 w - - 0 1'
    result = adjudicate(fen, ['d4b4', 'b6d7', 'b4d4', 'd7b6'] * 2)
    assert result['ended'] and result['winner'] == 'black'
