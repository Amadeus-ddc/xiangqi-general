import json
import pytest
from xqgeneral.collect_teacher import assemble_label, mirror_label
from xqgeneral.evidence import history_key
from xqgeneral.rules import START_FEN, replay


def sample():
    history = replay(START_FEN, ['b0c2'])
    row = {'id': 'q', 'fen': history[-1], 'initial_fen': START_FEN, 'moves': ['b0c2'],
           'history': history, 'feature_key': history_key(history), 'game_id': 'g', 'split': 'train',
           'query': {}, 'provenance': 'test', 'future_moves': []}
    query = {'id': 'q-explanation', 'record': row,
             'oracle': {'best_move': 'b9c7', 'candidates': [
                 {'move': 'b9c7', 'pv': ['b9c7'], 'score_type': 'cp', 'score': 25}]}}
    annotation = {'id': query['id'], 'teacher_model': 'gpt-6-astra',
                  'reasoning_effort': 'low', 'backend': 'codex_subagent',
                  'explanation': '黑方可走b9c7，把马从底线调出来。这一步没有吃子，也没有将军。'
                                 '评分为黑方视角的正25厘兵，仍应留意红方后续出子和炮的调动，'
                                 '这个数字不能证明已经获胜，更不能当作强制杀棋。'}
    return query, annotation


def test_teacher_contract_and_color_derived_facts():
    query, annotation = sample()
    original = assemble_label(query, annotation)
    derived = mirror_label(original)
    value = json.loads(derived['answer'])
    assert value['move'] == 'b0c2'
    assert value['facts']['piece'] == '红马'
    assert value['evaluation']['value'] == 25
    assert derived['split'] == original['split']
    assert derived['teacher']['independent_teacher_call'] is False
    with pytest.raises(ValueError, match='identity'):
        assemble_label(query, dict(annotation, teacher_model='different-teacher'))
