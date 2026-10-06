import pytest
from xqgeneral.oracle import parse_analysis
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
