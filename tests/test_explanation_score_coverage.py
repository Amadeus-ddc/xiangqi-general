import json

import pytest

from xqgeneral.evaluate_explanations import explanation_summary, judgment
from xqgeneral.explanations import move_facts
from xqgeneral.oracle import parse_analysis
from xqgeneral.review_explanations import blinded_query
from xqgeneral.rules import START_FEN, legal_moves


def raw_answer(pv=None):
    return json.dumps({'move': 'b0c2', 'pv': pv or ['b0c2', 'b9c7'],
                      'candidates': ['b0c2'], 'facts': move_facts(START_FEN, 'b0c2'),
                      'evaluation': {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'},
                      'explanation': '先出马，继续发展子力。'}, ensure_ascii=False)


class BoundOracle:
    def __init__(self, fail_ply):
        self.fail_ply = fail_ply
        self.calls = []

    def analyze(self, fen, nodes, initial_fen, moves, searchmoves=None):
        self.calls.append((list(moves), nodes, searchmoves))
        move = searchmoves[0] if searchmoves else ('b0c2' if not moves else 'b9c7')
        if move not in legal_moves(fen):
            move = legal_moves(fen)[0]
        bound = ' lowerbound' if len(moves) == self.fail_ply else ''
        # The actual incident was a legal final bestmove with no exact score.
        return parse_analysis(fen, [f'info depth 23 score cp 2{bound} pv {move}',
                                    f'bestmove {move}'], nodes)


@pytest.mark.parametrize('failed_ply', [0, 1])
def test_bound_only_engine_score_preserves_model_answer_and_pv_legality(failed_ply):
    record = {'fen': START_FEN, 'initial_fen': START_FEN, 'moves': []}
    oracle = BoundOracle(failed_ply)
    raw = raw_answer()
    result = judgment(oracle, record, raw, 1000000)
    assert result['parse_valid'] and result['first_move_legal'] and result['pv_legal']
    assert len(result['oracle_scoring_errors']) == 1
    assert result['oracle_scoring_errors'][0]['requested_nodes'] == 1000000
    assert result['oracle_scoring_errors'][0]['raw_output'][-1].startswith('bestmove ')
    assert len(result['pv_move_losses']) == 1
    assert len(oracle.calls) == 2  # Do not silently spend extra nodes retrying.
    rows = [{'id': 'control', 'record': record, 'raw': raw, 'judgment': result}]
    summary = explanation_summary(rows)
    assert summary['oracle_scoring_coverage']['unscored_engine_queries'] == 1
    assert summary['oracle_scoring_coverage']['legal_pv_plies_unscored'] == 1
    assert summary['oracle_scoring_coverage']['legal_first_moves_unscored'] == (failed_ply == 0)
    assert summary['oracle_scoring_coverage']['first_move_no_mistake_rate_is_lower_bound'] == (failed_ply == 0)
    if failed_ply == 0:
        assert 'first_move_expected_score_loss' not in result
        assert result['first_move_no_mistake'] is False
        assert summary['legal_scored_first_moves'] == 0
    else:
        assert result['first_move_expected_score_loss'] == 0
    evidence = json.loads(blinded_query(rows[0])['messages'][1]['content'])
    assert evidence['oracle_scoring_unavailable'][0]['reason'] == 'missing_exact_best_move_score'
    assert '未测量' in evidence['oracle_score_coverage_note']
    assert rows[0]['raw'] == raw


def test_missing_engine_score_does_not_hide_later_illegal_model_move():
    record = {'fen': START_FEN, 'initial_fen': START_FEN, 'moves': []}
    result = judgment(BoundOracle(0), record, raw_answer(['b0c2', 'a0a0']), 1000000)
    assert result['first_move_legal'] and not result['pv_legal']
    assert not result['contract_valid'] and 'illegal_pv' in result['errors']
    assert result['oracle_scoring_errors']


def test_other_engine_failures_still_abort_instead_of_becoming_missing_scores():
    class Broken:
        def analyze(self, *args, **kwargs):
            raise RuntimeError('Pikafish exited before finishing the request')
    with pytest.raises(RuntimeError, match='exited'):
        judgment(Broken(), {'fen': START_FEN, 'initial_fen': START_FEN, 'moves': []}, raw_answer(), 100)


def test_move_and_plan_reports_identify_unscored_quality_instead_of_claiming_model_errors():
    from xqgeneral.evaluate_explanations import judge_move
    from xqgeneral.evaluate_moves import move_summary
    from xqgeneral.evaluate_plans import judge_plan, plan_summary
    record = {'fen': START_FEN, 'initial_fen': START_FEN, 'moves': [],
              'answer': 'b0c2 b9c7', 'teacher': {}, 'query': {}}
    verdict = judge_move(BoundOracle(0), record, 'b0c2', 100)
    moves = move_summary([{'record': dict(record, answer='b0c2'), 'move': 'b0c2', 'judgment': verdict}])
    assert moves['legal_rate'] == 1 and moves['legal_scored_moves'] == 0
    assert moves['legal_unscored_moves'] == 1 and moves['no_mistake_rate_is_lower_bound']
    plan = judge_plan(BoundOracle(1), record, 'b0c2 b9c7', 100)
    summary = plan_summary([{'record': record, 'raw': 'b0c2 b9c7', 'judgment': plan}])
    assert summary['full_line_legal_rate'] == 1 and summary['contract_valid_rate'] == 1
    assert summary['quality_plies_unscored'] == 1 and summary['quality_ply_no_mistake_rate_is_lower_bound']


def test_restricted_root_missing_score_is_not_retried_for_the_same_pv_move():
    class Restricted(BoundOracle):
        def analyze(self, fen, nodes, initial_fen, moves, searchmoves=None):
            if moves:
                return super().analyze(fen, nodes, initial_fen, moves, searchmoves)
            self.calls.append((list(moves), nodes, searchmoves))
            move = searchmoves[0] if searchmoves else 'h0g2'
            bound = ' upperbound' if searchmoves else ''
            return parse_analysis(fen, [f'info depth 8 score cp 10{bound} pv {move}', f'bestmove {move}'], nodes)
    oracle = Restricted(-1)
    result = judgment(oracle, {'fen': START_FEN, 'initial_fen': START_FEN, 'moves': []}, raw_answer(), 100)
    assert result['first_move_legal'] and result['pv_legal']
    assert result['first_move_score_unavailable']
    assert result['oracle_scoring_errors'][0]['phase'] == 'first_move_restricted'
    assert len(result['oracle_scoring_errors']) == 1 and len(oracle.calls) == 3
    assert len(result['pv_move_losses']) == 1
