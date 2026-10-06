import json
from xqgeneral.explanations import choose_child, move_facts, oracle_explanation, parse_explanation, validate_explanation
from xqgeneral.rules import START_FEN


def valid_example():
    return {'move': 'b0c2', 'pv': ['b0c2', 'b9c7'], 'candidates': ['b0c2', 'h0g2'],
            'facts': move_facts(START_FEN, 'b0c2'),
            'evaluation': {'type': 'cp', 'value': 12, 'perspective': 'side_to_move'},
            'explanation': '红马从b0到c2，双方随后发展马。'}


def test_reject_false_facts_illegal_variations_and_wrong_perspective():
    row = valid_example()
    assert validate_explanation(START_FEN, parse_explanation(json.dumps(row)))['valid']
    row['facts']['piece'] = '黑车'
    assert 'incorrect_move_facts' in validate_explanation(START_FEN, row)['errors']
    row = valid_example(); row['pv'].append('a3b3')
    assert 'illegal_pv' in validate_explanation(START_FEN, row)['errors']
    row = valid_example(); row['evaluation']['perspective'] = 'red'
    assert 'invalid_evaluation_contract' in validate_explanation(START_FEN, row)['errors']


def test_consolidation_negates_child_scores_including_mate():
    def child(move, value, kind='cp'):
        return {'move': move, 'analysis': {'evaluation': {'type': kind, 'value': value, 'perspective': 'side_to_move'}}}
    assert choose_child([child('a', 100), child('b', -20)])['move'] == 'b'
    assert choose_child([child('a', -20), child('b', 2, 'mate')])['move'] == 'a'
    assert choose_child([child('a', -20), child('b', -2, 'mate')])['move'] == 'b'


def test_alternative_prose_moves_need_a_legal_declared_branch():
    result = {'best_move': 'b0c2', 'candidates': [
        {'move': 'b0c2', 'pv': ['b0c2', 'b9c7'], 'score_type': 'cp', 'score': 20},
        {'move': 'h0g2', 'pv': ['h0g2', 'h9g7'], 'score_type': 'cp', 'score': 10}]}
    row = oracle_explanation(START_FEN, result, '推荐b0c2。备选h0g2后，对方可走h9g7。')
    assert validate_explanation(START_FEN, row)['valid']
    row['branches'][1]['pv'][1] = 'a3b3'
    assert 'illegal_branch_pv' in validate_explanation(START_FEN, row)['errors']
    assert 'unsupported_prose_move' in validate_explanation(START_FEN, row)['errors']
