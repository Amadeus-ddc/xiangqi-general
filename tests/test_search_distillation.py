import json
import pytest
from xqgeneral.calibration import evaluation_probability
from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN, play
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


def test_incorrect_child_forces_recursion_instead_of_oracle_replacing_its_analysis():
    miner = SearchMiner(ControlledPredictor(incorrect_child=True), ControlledOracle(), set(), max_depth=0)
    query, trace, reason = miner.mine(root_record())
    assert query is None and reason == 'recursion_limit'
    assert trace[0]['children'][0]['verification']['errors'] == ['incorrect_move_facts']
    assert miner.counts['recursive_descents'] == 1
