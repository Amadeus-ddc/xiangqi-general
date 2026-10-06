import json

import pytest

from xqgeneral.coach import advise
from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN, replay


@pytest.mark.parametrize('principal,branch', [
    (['c7b9', 'b0c2'], ['c7b9']),
    (['c7b9'], ['c7b9', 'b0c2']),
])
def test_coach_rejects_continuation_after_complete_history_ends(principal, branch):
    moves = (['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2)[:-1]
    fen = replay(START_FEN, moves)[-1]
    score = {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'}
    value = {'move': 'c7b9', 'pv': principal, 'candidates': ['c7b9'],
             'branches': [{'move': 'c7b9', 'pv': branch, 'evaluation': score}],
             'evaluation': score, 'facts': move_facts(fen, 'c7b9'),
             'explanation': '保留原始分析；变化必须遵守完整棋局历史。'}
    raw = json.dumps(value, ensure_ascii=False)

    class Predictor:
        def generate(self, record, max_new_tokens):
            assert record['moves'] == moves and record['history'][-1] == fen
            return raw

    result = advise(Predictor(), START_FEN, moves)
    assert not result['verification']['valid']
    assert 'invalid_history_continuation' in result['verification']['errors']
    assert result['raw'] == raw and result['analysis'] == value
    assert not result['external_oracle_used'] and not result['answer_repaired']


def test_coach_accepts_legal_line_that_reaches_history_terminal():
    moves = (['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2)[:-1]
    fen = replay(START_FEN, moves)[-1]
    score = {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'}
    value = {'move': 'c7b9', 'pv': ['c7b9'], 'candidates': ['c7b9'],
             'branches': [{'move': 'c7b9', 'pv': ['c7b9'], 'evaluation': score}],
             'evaluation': score, 'facts': move_facts(fen, 'c7b9'),
             'explanation': '本手使完整棋局历史达到重复终局。'}

    class Predictor:
        def generate(self, record, max_new_tokens):
            return json.dumps(value, ensure_ascii=False)

    assert advise(Predictor(), START_FEN, moves)['verification']['valid']
