import json
import pytest
from xqgeneral.review_explanations import blinded_query, parse_rating
from xqgeneral.rules import START_FEN


def test_neural_review_hides_identity_and_marks_illegal_continuation():
    row = {'id': 'q', 'record': {'fen': START_FEN}, 'checkpoint': 'secret-model-name',
           'raw': json.dumps({'pv': ['b0c2','a0a1']}),
           'judgment': {'errors': ['illegal_pv'], 'first_move_legal': True}}
    query = blinded_query(row)
    content = json.loads(query['messages'][1]['content'])
    assert 'secret-model-name' not in json.dumps(query)
    assert content['root_side'] == '红方'
    assert content['principal_variation_facts']['error'] == 'illegal_move'
    assert len(content['principal_variation_facts']['facts']) == 1


def test_neural_review_requires_full_teacher_and_integer_rating():
    teacher = {'repository':'Qwen/Qwen3.8-27B','revision':'pinned','quantization':'none',
               'inference_dtype':'bfloat16'}
    identity = dict(teacher, parameter_elements_by_dtype={'torch.bfloat16':27})
    rating = {'factual_correctness':4,'strategic_reasoning':4,'clarity':5,'instruction_adherence':4,
              'unsupported_claims':[], 'reason':'棋盘事实与变化对应，推荐和备选差异明确。'}
    response = {'teacher':identity,'text':json.dumps(rating),'hit_generation_limit':False}
    assert parse_rating(response,teacher) == rating
    with pytest.raises(ValueError, match='BF16'):
        parse_rating(dict(response,teacher=dict(identity,quantization='q4')),teacher)
    rating['factual_correctness'] = True
    with pytest.raises(ValueError, match='schema'):
        parse_rating(dict(response,text=json.dumps(rating)),teacher)
