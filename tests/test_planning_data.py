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


def test_parallel_planning_is_byte_identical_including_terminal_rejections():
    import json
    queries = []
    for index, moves in enumerate([[], ['b0c2'], [], ['b0c2', 'b9c7']]):
        q = query(moves)
        q['id'] = f'query-{index}'
        pv = ['b0c2', 'b9c7', 'h0g2', 'h9g7'][len(moves):]
        q['oracle'] = {'best_move': pv[0], 'candidates': [{'move': pv[0], 'pv': pv}]}
        queries.append(q)
    ended = query(['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2)
    ended['oracle'] = {'best_move': 'b0c2', 'candidates': [{'move': 'b0c2', 'pv': ['b0c2']}]}
    queries.append(ended)
    keys = {k for q in queries for k in [q['record']['feature_key'],
        mirrored_qa(dict(q['record'], task_type='best_line', question='', answer=''))['feature_key']]}
    serial = planning_lessons(queries, keys, chunk_size=2)
    parallel = planning_lessons(queries, keys, workers=2, chunk_size=1)
    assert json.dumps(serial, ensure_ascii=False) == json.dumps(parallel, ensure_ascii=False)
    assert serial[1] == {'terminal_root': 2}


def test_isolation_includes_nonprincipal_branch_futures():
    row = dict(query()['record'], stage='move_planning', future_moves=['h0g2'],
               future_branches=[['b0c2', 'b9c7']])
    heldout = dict(query(['b0c2', 'b9c7'], split='validation')['record'],
                   stage='move_planning', task_type='best_line', question='', answer='',
                   future_moves=[], future_branches=[])
    assert isolate_planning_rows([heldout], [row]) == ([heldout], {'move_planning': 1})


def test_parallel_isolation_preserves_bytes_order_counts_and_mirrored_branch_futures():
    import json
    row = dict(query()['record'], stage='move_planning', task_type='best_line',
               question='', answer='h0g2', future_moves=['h0g2'],
               future_branches=[['b0c2', 'b9c7']])
    mirror = mirrored_qa(row)
    heldout = dict(query(['b0c2', 'b9c7'], split='validation')['record'],
                   stage='move_planning', task_type='best_line', question='', answer='',
                   future_moves=[], future_branches=[])
    mirror_target = mirrored_qa(heldout)
    mirror_target.update(split='test', game_id='test-game')
    control = dict(query(['h0g2'])['record'], stage='move_quality', future_moves=['a9a8'])
    overlapping = dict(row, stage='move_quality')
    base = [control, heldout, overlapping, mirror_target, dict(control, id='duplicate')]
    serial = isolate_planning_rows(base, [row, mirror])
    progress = []
    parallel = isolate_planning_rows(base, [row, mirror], workers=2, chunk_size=1,
                                     progress=progress.append)
    assert serial == ([control, heldout, mirror_target, base[-1]],
                      {'move_quality': 1, 'move_planning': 2})
    assert json.dumps(serial, ensure_ascii=False) == json.dumps(parallel, ensure_ascii=False)
    assert progress and progress[-1] == 5


def test_parallel_isolation_propagates_illegal_future_instead_of_keeping_it():
    row = dict(query()['record'], stage='move_planning', future_branches=[['a0a0']])
    for workers in [1, 2]:
        with pytest.raises(ValueError, match='Illegal move a0a0'):
            isolate_planning_rows([], [row], workers=workers, chunk_size=1)
    with pytest.raises(ValueError, match='must be positive'):
        isolate_planning_rows([], [], workers=0)
