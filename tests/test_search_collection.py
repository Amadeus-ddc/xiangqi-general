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


def mined_capture_query():
    fen = '3nkab2/4a1c2/9/8N/4P1b2/P4R2P/9/2N1B4/5p3/2rAKAB2 w - - 1 37'
    lines = [['e2c0', 'g8g0', 'f0e1', 'd9c7', 'c0e2', 'c7b5'],
             ['i6g5', 'c0c2', 'f4c4', 'c2b2', 'c4c0', 'f1g1'],
             ['f0e1', 'c0c2', 'f4c4', 'c2b2', 'c4b4', 'g8i8']]
    q = query()
    q['record'].update(fen=fen, initial_fen=fen, moves=[])
    q['target_fields'].update(move=lines[0][0], pv=lines[0],
        candidates=[line[0] for line in lines], facts=move_facts(fen, lines[0][0]),
        branches=[{'move':line[0], 'pv':line,
                   'evaluation':{'type':'cp', 'value':score, 'perspective':'side_to_move'}}
                  for line, score in zip(lines, [623, -199, -199])])
    q['target_fields']['evaluation']['value'] = 677
    return q


@pytest.mark.parametrize('prose', [
    '红方推荐相七进九吃车，评分677，确立优势。黑方应以炮8平1将军，红仕五进六挡将。'
    '随后黑马9进7，红相九进七回防，黑马7进5。此变化中红方虽失一相，但成功消除黑车威胁，'
    '且子力位置更协调。备选马八进七吃象评分-199，黑车2进4捉马后红方陷入被动。',
    '推荐e2c0（相七进五），之后按给定变化分析对方的炮将军和红方的应对。两条备选的评分'
    '都比推荐着法低，具体吃子和将军须以每步棋盘事实为准。该有限变化与评分不能证明强制'
    '获胜，也不能把没有给出的后续攻击当作已经发生的结果。',
    '推荐e2c0吃车，再考虑f4f5加强进攻。两条备选的评分都比推荐着法低，具体吃子和将军'
    '须以每步棋盘事实为准。该有限变化与评分不能证明强制获胜，也不能把没有给出的后续'
    '攻击当作已经发生的结果，讲解必须保持与实际提供的变化一致。',
])
def test_search_collection_rejects_false_or_ungrounded_prose_moves(prose):
    raw = response()
    raw['text'] = json.dumps({'explanation':prose}, ensure_ascii=False)
    with pytest.raises(ValueError, match='prose[ _]move'):
        consolidated_label(mined_capture_query(), raw, TEACHER)


def test_search_collection_accepts_paired_native_notation_without_changing_prose():
    prose = ('推荐e2c0（相五退七）吃掉黑车。主线g8g0（炮7进8）吃红相并将军，'
             '红方f0e1（仕四进五）移开炮与帅之间的炮架，解除将军。备选i6g5吃象，'
             '黑方c0c2吃马，评分较低。根评分677是红方视角；给定变化不能证明强制获胜。')
    raw = response()
    raw['text'] = json.dumps({'explanation':prose}, ensure_ascii=False)
    row = consolidated_label(mined_capture_query(), raw, TEACHER)
    assert json.loads(row['answer'])['explanation'] == prose
