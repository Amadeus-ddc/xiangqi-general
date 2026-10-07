import json
import pytest
from xqgeneral.calibration import evaluation_probability
from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN, play, replay
from xqgeneral.evidence import history_key, position_key
from xqgeneral.search_distillation import SearchMiner, descend, first_divergence, root_evaluation


def test_forced_move_quality_query_is_preserved_before_cached_reuse(tmp_path):
    class Oracle:
        def analyze(self, fen, nodes, initial_fen, moves, searchmoves=None):
            move = searchmoves[0] if searchmoves else 'b0c2'
            return {'best_move':move, 'requested_nodes':nodes,
                    'candidates':[{'move':move, 'score_type':'cp', 'score':0, 'wdl':[0,1000,0]}],
                    'raw_output':[f'info depth 1 score cp 0 wdl 0 1000 0 nodes {nodes} pv {move}',
                                  f'bestmove {move}']}

    path = tmp_path / 'oracle.jsonl'
    def preserve(item):
        with path.open('a') as handle:
            handle.write(json.dumps(item) + '\n')

    row = root_record()
    miner = SearchMiner(None, Oracle(), set(), record_oracle_query=preserve)
    root = miner.analyze(row)
    # This move is absent from root MultiPV; its probability needs another actual search.
    assert miner.move_probability(row, 'a0a1', root) == 0.5
    assert miner.move_probability(row, 'a0a1', root) == 0.5
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert [q['searchmoves'] for q in saved] == [[], ['a0a1']]
    assert all(q['record']['moves'] == row['moves'] for q in saved)
    assert saved[1]['response']['raw_output'][-1] == 'bestmove a0a1'


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


class DefectiveUnusedBranch(ControlledPredictor):
    def generate(self, row, question, max_new_tokens):
        value = json.loads(super().generate(row, question, max_new_tokens))
        if row['moves']:
            value['branches'] = [{'move': value['move'], 'pv': [value['move'], 'a0a0'],
                                  'evaluation': value['evaluation']}]
        return json.dumps(value, ensure_ascii=False)


def test_move_eval_child_contract_retains_raw_errors_and_only_consolidates_used_lines():
    predictor = DefectiveUnusedBranch()
    strict = SearchMiner(predictor, ControlledOracle(), set(), max_depth=0)
    assert strict.mine(root_record())[2] == 'recursion_limit'
    miner = SearchMiner(predictor, ControlledOracle(), set(), max_depth=0, child_contract='move_eval')
    query, trace, reason = miner.mine(root_record())
    assert reason == 'accepted_for_consolidation'
    assert query['target_fields']['move'] == 'h0g2'
    for child in trace[0]['children']:
        assert not child['verification']['valid']
        assert 'illegal_branch_pv' in child['verification']['errors']
        assert child['child_acceptance']['valid']
        assert json.loads(child['raw'])['branches'][0]['pv'][-1] == 'a0a0'
    packet = json.loads(query['messages'][1]['content'])
    assert 'verified_child_analyses' not in packet
    assert all('branches' not in child and 'facts' not in child for child in packet['child_analyses'])
    assert all(child['used_pv'] == ['b9c7'] for child in packet['child_analyses'])
    assert all(branch['pv'][-1] == 'b9c7' for branch in query['target_fields']['branches'])


def test_move_eval_contract_does_not_silently_correct_original_child_facts():
    miner = SearchMiner(ControlledPredictor(incorrect_child=True), ControlledOracle(), set(),
                        max_depth=0, child_contract='move_eval')
    query, trace, reason = miner.mine(root_record())
    assert reason == 'accepted_for_consolidation'
    child = next(item for item in trace[0]['children'] if item['move'] == 'b0c2')
    assert json.loads(child['raw'])['facts']['piece'] == '黑炮'
    assert child['analysis']['facts']['piece'] == '黑炮'
    assert child['verification']['errors'] == ['incorrect_move_facts']
    packet = json.loads(query['messages'][1]['content'])
    source = next(item for item in packet['child_analyses'] if item['move'] == child['move'])
    assert 'unverified_explanation' not in source
    assert packet['unverified_child_prose_omitted']
    assert child['analysis']['explanation'] == json.loads(child['raw'])['explanation']
    assert source['full_output_verification'] == child['verification']


@pytest.mark.parametrize('pv', [[], ['b9c7', 'a0a0']])
def test_move_eval_acceptance_still_rejects_unusable_inferred_pv(pv):
    class Predictor(ControlledPredictor):
        def generate(self, row, question, max_new_tokens):
            value = json.loads(super().generate(row, question, max_new_tokens))
            if row['moves']:
                value['pv'] = pv
            return json.dumps(value, ensure_ascii=False)

    query, trace, reason = SearchMiner(Predictor(), ControlledOracle(), set(), max_depth=0,
        child_contract='move_eval').mine(root_record())
    assert query is None and reason == 'invalid_inferred_continuation'
    assert all(child['child_acceptance']['valid'] for child in trace[0]['children'])


@pytest.mark.parametrize('defect', ['illegal_move', 'wrong_score_perspective'])
def test_move_eval_contract_recurses_on_bad_recommendation_or_evaluation(defect):
    class Predictor(ControlledPredictor):
        def generate(self, row, question, max_new_tokens):
            value = json.loads(super().generate(row, question, max_new_tokens))
            if row['moves']:
                if defect == 'illegal_move':
                    value['move'] = value['candidates'][0] = 'a0a0'
                else:
                    value['evaluation']['perspective'] = 'red'
            return json.dumps(value, ensure_ascii=False)

    query, trace, reason = SearchMiner(Predictor(), ControlledOracle(), set(), max_depth=0,
        child_contract='move_eval').mine(root_record())
    assert query is None and reason == 'recursion_limit'
    assert all(not child['child_acceptance']['valid'] for child in trace[0]['children'])


def test_move_eval_contract_excludes_reachable_heldout_prefix_of_invalid_unused_branch():
    class Predictor(DefectiveUnusedBranch):
        def generate(self, row, question, max_new_tokens):
            value = json.loads(super().generate(row, question, max_new_tokens))
            if row['moves']:
                value['branches'][0]['pv'] = ['b9c7', 'e3e4', 'a0a0']
            return json.dumps(value, ensure_ascii=False)

    heldout = {position_key(replay(START_FEN, ['b0c2', 'b9c7', 'e3e4'])[-1])}
    query, trace, reason = SearchMiner(Predictor(), ControlledOracle(), heldout,
        child_contract='move_eval').mine(root_record())
    assert query is None and reason == 'reserved_child_continuation'


@pytest.mark.parametrize('contract', ['full', 'move_eval'])
def test_engine_quality_failure_keeps_structural_student_verification_unchanged(contract):
    class Predictor(ControlledPredictor):
        def generate(self, row, question, max_new_tokens):
            value = json.loads(super().generate(row, question, max_new_tokens))
            if row['moves']:
                value['evaluation']['value'] *= -1
            return json.dumps(value, ensure_ascii=False)

    query, trace, reason = SearchMiner(Predictor(), ControlledOracle(), set(),
        max_depth=0, child_contract=contract).mine(root_record())
    assert query is None and reason == 'recursion_limit'
    for child in trace[0]['children']:
        assert child['verification']['valid'] and child['verification']['errors'] == []
        assert child['child_acceptance']['errors'] == ['evaluation_error_exceeds_threshold']
        assert not child['child_acceptance']['valid']


def test_move_eval_inferred_target_cannot_continue_after_history_terminal():
    moves = (['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2)[:-2]
    history = replay(START_FEN, moves)
    row = dict(root_record(), moves=moves, history=history, fen=history[-1], feature_key=history_key(history))

    class Predictor:
        def generate(self, record, question, max_new_tokens):
            move = 'c2b0' if len(record['moves']) == len(moves) else 'c7b9'
            pv = [move] if move == 'c2b0' else ['c7b9', 'b0c2']
            return json.dumps({'move': move, 'pv': pv, 'candidates': [move],
                'facts': move_facts(record['fen'], move),
                'evaluation': {'type':'cp','value':0,'perspective':'side_to_move'},
                'explanation':'依据完整历史检查变化，不在终局之后继续走棋。'}, ensure_ascii=False)

    class Oracle:
        def analyze(self, fen, nodes, initial_fen, played, searchmoves=None):
            move = 'c2b0' if len(played) == len(moves) else 'c7b9'
            return {'best_move':move,'candidates':[{'move':move,'score_type':'cp','score':0,'pv':[move],'wdl':None}]}

    query, trace, reason = SearchMiner(Predictor(), Oracle(), set(),
        child_contract='move_eval').mine(row)
    assert query is None and reason == 'invalid_inferred_continuation'
