import pytest

from xqgeneral.evidence import history_key, position_key
from xqgeneral.planning_data import isolate_planning_rows, planning_label, planning_lessons
from xqgeneral.rules import START_FEN, play, replay
from xqgeneral.symmetry import mirror_move, mirrored_qa


def query(moves=(), split='train'):
    history = replay(START_FEN, moves)
    row = {'id': 'root', 'feature_key': history_key(history), 'initial_fen': START_FEN,
           'moves': list(moves), 'history': history, 'fen': history[-1], 'split': split,
           'game_id': split+'-game', 'provenance': 'real_engine_search', 'query': {}, 'future_moves': []}
    return {'id': 'root-query', 'record': row}


def test_planning_preserves_candidate_order_symmetry_and_excludes_heldout_future():
    q = query(); pv = ['b0c2', 'b9c7', 'h0g2', 'h9g7', 'a0a1', 'a9a8']
    candidate = {'move': pv[0], 'pv': pv}
    q['oracle'] = {'best_move': pv[0], 'candidates': [candidate]}
    row = planning_label(q, candidate, True)
    mirror = mirrored_qa(row)
    assert row['answer'] == ' '.join(pv) and row['future_moves'] == pv
    assert mirror['query']['move'] == mirror_move(pv[0])
    assert mirror['answer'] == ' '.join(mirror_move(m) for m in pv)
    labels, rejected = planning_lessons([q], {row['feature_key'], mirror['feature_key']})
    assert len(labels) == 4 and not rejected
    assert all(r['game_id']=='train-game' and not r['teacher']['neural_prose_generated'] for r in labels)
    future = play(play(START_FEN, pv[0]), pv[1])
    heldout = dict(row, split='validation', game_id='validation-game', fen=future, future_moves=[])
    kept, dropped = isolate_planning_rows([heldout], [row])
    assert kept == [heldout] and dropped == {'move_planning': 1}
    assert position_key(future) != position_key(row['fen'])


def test_planning_stops_at_full_history_repetition_before_the_next_move():
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9']
    q = query((cycle * 2)[:-1])
    candidate = {'move': 'c7b9', 'pv': ['c7b9', 'a0a0']}
    row = planning_label(q, candidate)
    assert row['answer'] == 'c7b9' and row['future_moves'] == ['c7b9']
    assert planning_label(query(cycle * 2), {'move':'b0c2','pv':['b0c2']}) is None
    with pytest.raises(ValueError, match='first PV'):
        planning_label(query(), {'move': 'h0g2', 'pv': ['b0c2']})
