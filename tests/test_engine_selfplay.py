import random

import pytest

from xqgeneral.engine_selfplay import analyze_selfplay, choose_move, generate_game, select_training_contexts
from xqgeneral.evidence import history_key, position_key
from xqgeneral.rules import START_FEN, legal_moves, play, replay


def test_selfplay_exploration_excludes_bad_moves_and_rejects_illegal_best():
    moves = ['b0c2', 'h0g2', 'a0a1']
    result = {'best_move': moves[0], 'candidates': [
        {'move': move, 'wdl': wdl, 'score_type': 'cp', 'score': 0}
        for move, wdl in zip(moves, [[500, 500, 0], [480, 520, 0], [0, 0, 1000]])]}
    rng = random.Random(4)
    selected = {choose_move(START_FEN, result, rng, True) for _ in range(100)}
    assert selected == set(moves[:2])
    assert choose_move(START_FEN, result, rng, False) == moves[0]
    with pytest.raises(ValueError, match='illegal'):
        choose_move(START_FEN, dict(result, best_move='a0a9'), rng, False)


def test_selfplay_preserves_history_and_censors_without_a_draw():
    class Oracle:
        def analyze(self, fen, nodes, initial_fen, moves):
            assert replay(initial_fen, moves)[-1] == fen
            move = sorted(legal_moves(fen))[0]
            return {'best_move': move, 'candidates': [
                {'move': move, 'wdl': [0, 1000, 0], 'score_type': 'cp', 'score': 0}]}
    game = generate_game(Oracle(), 0, 1, 100, 12, 8, 4, 8)
    assert game['status'] == 'censored' and game['winner'] is None
    assert game['reason'] == 'ply_limit' and len(game['moves']) == 12
    assert game['split'] == 'train' and {r['fen'].split()[1] for r in game['contexts']} == {'w', 'b'}
    for row in game['contexts']:
        assert replay(row['initial_fen'], row['moves']) == row['history']
        assert row['feature_key'] == history_key(row['history'])
        assert row['future_moves'] == game['moves'][row['ply']:row['ply'] + 5]
        assert not row['answer']
    forbidden = {position_key(play(game['contexts'][0]['fen'], game['contexts'][0]['future_moves'][0]))}
    selected, rejected = select_training_contexts([game, game], forbidden)
    assert rejected['heldout_root_or_future'] >= 2
    assert rejected['duplicate_history'] == len(selected)
    with pytest.raises(ValueError, match='training games only'):
        select_training_contexts([dict(game, split='test')], set())


def test_selfplay_retries_only_unscored_search_and_preserves_failed_trace():
    class Oracle:
        calls = []
        last_output = ['info score cp 32 upperbound pv b0c2', 'bestmove b0c2']
        def analyze(self, fen, nodes, initial_fen, moves):
            self.calls.append(nodes)
            if nodes == 1000:
                raise RuntimeError('Engine completed without a scored best-move candidate')
            return {'best_move': 'b0c2', 'requested_nodes': nodes}
    oracle = Oracle()
    result = analyze_selfplay(oracle, START_FEN, 1000, [], 10000)
    assert oracle.calls == [1000, 10000] and result['requested_nodes'] == 10000
    assert result['retried_unscored_search']['raw_output'] == oracle.last_output
    with pytest.raises(RuntimeError, match='scored'):
        analyze_selfplay(oracle, START_FEN, 1000, [])
    class Broken:
        def analyze(self, *args):
            raise RuntimeError('Oracle returned an illegal or terminal move: a0a9')
    with pytest.raises(RuntimeError, match='illegal'):
        analyze_selfplay(Broken(), START_FEN, 1000, [], 10000)
