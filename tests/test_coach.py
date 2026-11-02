import json
import sys
from types import SimpleNamespace

import pytest

from xqgeneral.coach import advise
from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN, replay


def run_resumed_session(monkeypatch, path, commands, output=None):
    class Predictor:
        def __init__(self, checkpoint, weights):
            assert checkpoint == 'chosen-model' and weights is None

        def generate(self, record, max_new_tokens):
            assert record['history'] == replay(START_FEN, record['moves'])
            assert max_new_tokens == 768
            return '保留模型的原始格式错误'

    arguments = ['xqgeneral-coach', '--checkpoint', 'chosen-model',
                 '--resume', str(path), '--interactive']
    if output is not None:
        arguments.extend(['--output', str(output)])
    inputs = iter(commands)
    monkeypatch.setattr(sys, 'argv', arguments)
    monkeypatch.setattr('builtins.input', lambda prompt: next(inputs))
    monkeypatch.setitem(sys.modules, 'xqgeneral.inference', SimpleNamespace(Predictor=Predictor))
    from xqgeneral.coach import main
    main()


@pytest.mark.parametrize('separate_output', [False, True])
def test_resumed_coach_persists_new_moves_and_unmodified_advice(tmp_path, monkeypatch, separate_output):
    original = tmp_path / 'study.json'
    original.write_text(json.dumps({'initial_fen': START_FEN, 'moves': ['b0c2'],
                                   'analyses': [{'raw': '原有分析'}], 'checkpoint': 'previous-model'}))
    before = original.read_bytes()
    destination = tmp_path / 'branch.json' if separate_output else original
    run_resumed_session(monkeypatch, original, ['b9c7', 'quit'],
                        output=destination if separate_output else None)
    saved = json.loads(destination.read_text())
    assert saved['initial_fen'] == START_FEN
    assert saved['moves'] == ['b0c2', 'b9c7']
    assert saved['checkpoint'] == 'chosen-model'
    assert len(saved['analyses']) == 2
    assert saved['analyses'][0] == {'raw': '原有分析'}
    analysis = saved['analyses'][1]
    assert analysis['raw'] == '保留模型的原始格式错误'
    assert analysis['record']['moves'] == ['b0c2']
    assert analysis['verification'] == {'valid': False, 'errors': ['invalid_json']}
    assert not analysis['external_oracle_used'] and not analysis['answer_repaired']
    if separate_output:
        assert original.read_bytes() == before


def test_resumed_coach_persists_undo_without_explicit_output(tmp_path, monkeypatch):
    path = tmp_path / 'study.json'
    path.write_text(json.dumps({'initial_fen': START_FEN, 'moves': ['b0c2'], 'analyses': []}))
    run_resumed_session(monkeypatch, path, ['undo', 'quit'])
    saved = json.loads(path.read_text())
    assert saved['moves'] == []
    assert saved['analyses'][0]['record']['moves'] == ['b0c2']


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
