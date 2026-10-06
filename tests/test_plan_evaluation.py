from xqgeneral.evaluate_plans import judge_plan, parsed_line, plan_summary
from xqgeneral.rules import START_FEN, legal_moves, replay


class Oracle:
    def __init__(self):
        self.calls = []

    def analyze(self, fen, nodes, initial_fen, moves, searchmoves=None):
        self.calls.append((fen, initial_fen, list(moves)))
        candidates = [{'move': m, 'wdl': [0, 0, 1000] if m == 'h0g2' else [1000, 0, 0]}
                      for m in legal_moves(fen)]
        best = next(c['move'] for c in candidates if c['wdl'][0])
        return {'best_move': best, 'candidates': candidates}


def record(moves=(), answer='b0c2 b9c7', forced=None):
    return {'initial_fen': START_FEN, 'moves': list(moves), 'fen': replay(START_FEN, moves)[-1],
            'answer': answer, 'query': {'move': forced} if forced else {}}


def test_raw_plan_contract_and_forced_quality_use_each_plys_complete_history():
    assert parsed_line(' b0c2\nb9c7 ') == ['b0c2', 'b9c7']
    assert parsed_line('建议 b0c2') is None
    assert parsed_line(' '.join(['b0c2'] * 7)) is None
    assert not judge_plan(None, record(), 'a0a0', 100)['full_line_legal']
    oracle = Oracle()
    row = record(answer='h0g2 b9c7', forced='h0g2')
    result = judge_plan(oracle, row, 'h0g2 b9c7', 100)
    assert result['contract_valid'] and result['full_line_legal']
    assert [call[2] for call in oracle.calls] == [[], ['h0g2']]
    assert oracle.calls[1][0] == replay(START_FEN, ['h0g2'])[-1]
    assert all(call[1] == START_FEN for call in oracle.calls)
    assert result['plies'][0]['first_move_expected_score_loss'] == 1
    summary = plan_summary([{'record': row, 'raw': 'h0g2 b9c7', 'judgment': result}])
    assert summary['quality_plies_attempted'] == 1
    assert summary['quality_ply_no_mistake_rate_all_attempted'] == 1
    assert summary['mean_expected_score_loss_conditional'] == 0 and summary['oracle_repairs'] == 0
    short = judge_plan(Oracle(), record(), 'b0c2', 100)
    assert short['full_line_legal'] and not short['contract_valid']
    assert short['errors'] == ['shorter_than_reference_horizon']
    mismatch = judge_plan(Oracle(), record(forced='h0g2'), 'b0c2 b9c7', 100)
    assert mismatch['full_line_legal'] and not mismatch['conditional_first_move_matches']


def test_repetition_terminal_cannot_be_extended_or_sent_to_the_oracle():
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2
    root = record(cycle[:-1], answer='c7b9 b0c2')
    oracle = Oracle()
    valid = judge_plan(oracle, root, 'c7b9', 100)
    assert valid['contract_valid'] and valid['length_sufficient']
    assert len(oracle.calls) == 1
    oracle = Oracle()
    extended = judge_plan(oracle, root, 'c7b9 b0c2', 100)
    assert extended['errors'] == ['move_after_terminal']
    assert not extended['full_line_legal'] and len(oracle.calls) == 1
    terminal = judge_plan(None, record(cycle), 'b0c2', 100)
    assert terminal['errors'] == ['move_after_terminal']
