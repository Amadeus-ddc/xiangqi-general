import json
import pytest
from xqgeneral.calibration import evaluation_probability
from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN, play, replay
from xqgeneral.evidence import history_key, position_key
from xqgeneral.search_distillation import SearchMiner, descend, first_divergence, root_evaluation


def test_pv_divergence_uses_the_actual_mover_after_the_shared_prefix():
    result = first_divergence(START_FEN, ['b0c2', 'b9c7'], ['b0c2', 'h9g7'])
    assert result == (play(START_FEN, 'b0c2'), 'b9c7', 'h9g7', ['b0c2'])
    assert first_divergence(START_FEN, ['b0c2'], ['b0c2', 'b9c7']) is None
    row = {'id': 'r', 'fen': START_FEN, 'initial_fen': START_FEN, 'moves': [], 'history': [START_FEN]}
    child = descend(row, 'b0c2')
    assert child['moves'] == ['b0c2'] and child['history'][-1] == result[0]
    assert root_evaluation({'evaluation': {'type': 'cp', 'value': 30, 'perspective': 'side_to_move'}})['value'] == -30
    assert root_evaluation({'evaluation': {'type': 'mate', 'value': -2, 'perspective': 'side_to_move'}})['value'] == 3


def test_engine_score_calibration_respects_symmetry_and_rejects_wrong_units():
    def prob(cp):
        return evaluation_probability(START_FEN, {'type': 'cp', 'value': cp, 'perspective': 'side_to_move'})
    assert prob(0) == .5
    assert prob(50) < prob(200) < prob(500)
    assert prob(-200) == pytest.approx(1-prob(200))
    with pytest.raises(ValueError):
        evaluation_probability(START_FEN, {'type': 'cp', 'value': 20, 'perspective': 'red'})


class ControlledPredictor:
    def __init__(self, inject=False, incorrect_child=False):
        self.inject, self.incorrect_child = inject, incorrect_child

    def generate(self, row, question, max_new_tokens):
        if not row['moves']:
            candidates = ['b0c2', 'c3c4', 'e3e4'] if self.inject else ['b0c2', 'h0g2', 'c3c4']
            move, pv, score = 'b0c2', ['b0c2', 'b9c7'], -200
        else:
            candidates = ['b9c7']
            move, pv = 'b9c7', ['b9c7']
            score = {'b0c2': 200, 'h0g2': -200, 'c3c4': 300, 'e3e4': 250}[row['moves'][0]]
        facts = move_facts(row['fen'], move)
        if row['moves'] == ['b0c2'] and self.incorrect_child:
            facts['piece'] = '黑炮'
        return json.dumps({'move': move, 'pv': pv, 'candidates': candidates, 'facts': facts,
                           'evaluation': {'type': 'cp', 'value': score, 'perspective': 'side_to_move'},
                           'explanation': '按提供的局面先出马，注意对方后续调动。'}, ensure_ascii=False)


class ControlledOracle:
    def analyze(self, fen, nodes, initial_fen, moves, searchmoves=None):
        if not moves:
            scores = {'b0c2': -200, 'h0g2': 200, 'c3c4': -300, 'e3e4': -250}
            candidates = searchmoves or ['h0g2', 'b0c2', 'c3c4']
        else:
            scores = {'b9c7': {'b0c2': 200, 'h0g2': -200, 'c3c4': 300, 'e3e4': 250}[moves[0]]}
            candidates = ['b9c7']
        return {'best_move': candidates[0], 'candidates': [
            {'move': m, 'score_type': 'cp', 'score': scores[m], 'pv': [m], 'wdl': None} for m in candidates]}


def root_record():
    return {'id': 'r', 'game_id': 'g', 'split': 'train', 'fen': START_FEN, 'initial_fen': START_FEN,
            'moves': [], 'history': [START_FEN], 'feature_key': 'initial', 'provenance': 'controlled_test'}


def test_actual_mining_contract_improves_pv_and_only_injects_when_all_candidates_are_bad():
    for inject in [False, True]:
        miner = SearchMiner(ControlledPredictor(inject=inject), ControlledOracle(), set())
        query, trace, reason = miner.mine(root_record())
        assert reason == 'accepted_for_consolidation'
        assert query['target_fields']['move'] == 'h0g2'
        assert query['target_fields']['evaluation']['value'] == 200
        assert trace[0]['injected_oracle_move'] == inject
        assert query['selected_root_move_loss'] == 0
        content = json.loads(query['messages'][1]['content'])
        assert content['root_fen'] == START_FEN and content['root_board']['e0'] == '红帅'
        first = content['root_line_facts'][0]['facts'][0]
        assert first['fen_before'] == START_FEN and first['fen_after'] == play(START_FEN, first['move'])


def test_incorrect_child_forces_recursion_instead_of_oracle_replacing_its_analysis():
    miner = SearchMiner(ControlledPredictor(incorrect_child=True), ControlledOracle(), set(), max_depth=0)
    query, trace, reason = miner.mine(root_record())
    assert query is None and reason == 'recursion_limit'
    assert trace[0]['children'][0]['verification']['errors'] == ['incorrect_move_facts']
    assert miner.counts['recursive_descents'] == 1


def test_search_rejects_heldout_continuation_beyond_the_immediate_child():
    reserved = {position_key(replay(START_FEN, ['b0c2', 'b9c7'])[-1])}
    miner = SearchMiner(ControlledPredictor(), ControlledOracle(), reserved)
    query, trace, reason = miner.mine(root_record())
    assert query is None and reason == 'reserved_child_continuation'


def test_child_analysis_cannot_extend_a_terminal_repetition():
    moves = (['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2)[:-1]
    history = replay(START_FEN, moves)
    row = dict(root_record(), moves=moves, history=history, fen=history[-1], feature_key=history_key(history))

    class Predictor:
        def generate(self, record, question, max_new_tokens):
            return json.dumps({'move': 'c7b9', 'pv': ['c7b9', 'b0c2'], 'candidates': ['c7b9'],
                'facts': move_facts(record['fen'], 'c7b9'),
                'evaluation': {'type':'cp', 'value':0, 'perspective':'side_to_move'},
                'explanation':'按原始棋盘检查变化，不得在完整历史终局以后继续。'}, ensure_ascii=False)

    result = SearchMiner(Predictor(), None, set()).model_analysis(row)
    assert not result['verification']['valid']
    assert 'invalid_history_continuation' in result['verification']['errors']
