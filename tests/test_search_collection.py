import json
import pytest

from xqgeneral.collect_search import consolidated_label
from xqgeneral.curriculum_data import verify_splits
from xqgeneral.explanations import continuation_positions, move_facts
from xqgeneral.rules import START_FEN, play, replay
from xqgeneral.search_distillation import reserved_positions
from xqgeneral.evidence import position_key, write_jsonl


TEACHER = {'repository':'Qwen/Qwen3.8-27B', 'revision':'pinned', 'quantization':'none',
           'inference_dtype':'bfloat16'}


def query():
    primary, alternate = ['b0c2', 'b9c7'], ['h0g2', 'h9g7']
    score = {'type':'cp', 'value':0, 'perspective':'side_to_move'}
    return {'id':'q-search', 'source_root_id':'q', 'depth':0,
            'record':{'fen':START_FEN, 'initial_fen':START_FEN, 'moves':[], 'split':'train',
                      'game_id':'train-game', 'provenance':'controlled', 'feature_key':'root'},
            'target_fields':{'move':primary[0], 'pv':primary, 'candidates':[primary[0], alternate[0]],
                'branches':[{'move':line[0], 'pv':line, 'evaluation':score} for line in [primary,alternate]],
                'facts':move_facts(START_FEN, primary[0]), 'evaluation':score}}


def response():
    return {'id':'q-search', 'hit_generation_limit':False,
            'teacher':{**TEACHER, 'parameter_elements_by_dtype':{'torch.bfloat16':27}},
            'text':json.dumps({'explanation':'只按给定事实解释出马和对方应对，分清根行棋方和对手的评分视角；'
                '两种变化都须核对吃子和将军。评分不能证明强制获胜，要根据后续局面比较子力活动及风险，'
                '避免把未给出的变化或进攻效果当成已经发生的结果。'}, ensure_ascii=False)}


def test_collected_search_preserves_every_branch_for_split_verification(tmp_path):
    q = query()
    row = consolidated_label(q, response(), TEACHER)
    assert row['future_moves'] == ['b0c2', 'b9c7']
    assert row['future_branches'] == [['b0c2', 'b9c7'], ['h0g2', 'h9g7']]
    forbidden_fen = replay(START_FEN, ['h0g2', 'h9g7'])[-1]
    assert position_key(forbidden_fen) in continuation_positions(row, json.loads(row['answer']))
    heldout = {'fen':forbidden_fen, 'split':'validation', 'game_id':'heldout', 'future_moves':[]}
    # Identical roots and primary PVs must not cause a distinct branch to be skipped.
    plain = dict(row, future_branches=[])
    with pytest.raises(ValueError, match='leakage'):
        verify_splits([plain, row, heldout])
    source = dict(row, split='validation', future_moves=[], future_branches=[])
    write_jsonl(tmp_path/'validation.jsonl', [source]);write_jsonl(tmp_path/'test.jsonl', [])
    assert position_key(forbidden_fen) in reserved_positions(tmp_path)


def test_search_collection_refuses_history_terminal_continuation_and_missing_branches():
    q = query()
    q['target_fields'].pop('branches')
    with pytest.raises(ValueError, match='missing_branches'):
        consolidated_label(q, response(), TEACHER)
    q = query()
    moves = (['b0c2','b9c7','c2b0','c7b9'] * 2)[:-1]
    q['record'].update(moves=moves, fen=replay(START_FEN,moves)[-1])
    target = q['target_fields'];line=['c7b9','b0c2']
    target.update(move=line[0], pv=line, candidates=[line[0]], facts=move_facts(q['record']['fen'],line[0]))
    target['branches']=[{'move':line[0], 'pv':line, 'evaluation':target['evaluation']}]
    with pytest.raises(ValueError, match='terminal history'):
        consolidated_label(q, response(), TEACHER)
