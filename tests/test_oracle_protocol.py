import pytest
from xqgeneral.oracle import Pikafish, parse_analysis
from xqgeneral.rules import START_FEN


def test_node_stop_preserves_exact_best_score_before_rank_changes():
    lines = ['info depth 8 multipv 1 score cp 20 wdl 10 985 5 pv b0c2 b9c7',
             'info depth 8 multipv 2 score cp 18 pv h0g2 h9g7',
             'info depth 9 multipv 1 score cp 22 pv h0g2 h9g7',
             'info depth 9 multipv 2 score cp 24 lowerbound pv b0c2 b9c7',
             'bestmove b0c2 ponder b9c7']
    result = parse_analysis(START_FEN, lines, 100)
    assert [c['move'] for c in result['candidates']] == ['b0c2', 'h0g2']
    assert result['candidates'][0]['score'] == 20
    assert result['candidates'][0]['wdl'] == [10, 985, 5]
    assert result['candidates'][1]['depth'] == 9
    with pytest.raises(RuntimeError, match='scored'):
        parse_analysis(START_FEN, [lines[3], lines[4]], 100)


def test_play_query_preserves_node_budget_without_inventing_exact_scores():
    oracle = Pikafish.__new__(Pikafish)
    output = ['info depth 1 score cp 32 upperbound pv b0c2', 'bestmove b0c2']
    budgets = []
    def search(fen, nodes, initial_fen=None, moves=(), searchmoves=None):
        budgets.append(nodes)
        return output
    oracle.search = search
    result = oracle.choose_move(START_FEN, 100, START_FEN, [])
    assert result['best_move'] == 'b0c2' and result['requested_nodes'] == 100
    assert result['raw_output'] == output and result['score_consumed'] is False
    assert budgets == [100]
    with pytest.raises(RuntimeError, match='scored'):
        oracle.analyze(START_FEN, 100, START_FEN, [])
    output[-1] = 'bestmove a0a9'
    with pytest.raises(RuntimeError, match='illegal'):
        oracle.choose_move(START_FEN, 100, START_FEN, [])
