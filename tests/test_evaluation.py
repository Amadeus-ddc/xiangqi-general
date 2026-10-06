import json
from xqgeneral.evaluate_explanations import judgment
from xqgeneral.evaluate_games import match_summary, play_game
from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN


class FixedOracle:
    def analyze(self, fen, nodes, initial_fen, moves, searchmoves=None):
        return {'best_move': 'h0g2', 'candidates': [
            {'move': 'h0g2', 'score_type': 'cp', 'score': 300, 'wdl': None},
            {'move': 'b0c2', 'score_type': 'cp', 'score': -300, 'wdl': None}]}


def test_move_quality_is_measured_even_when_the_generated_pv_is_invalid():
    row = {'fen': START_FEN, 'initial_fen': START_FEN, 'moves': []}
    value = {'move': 'b0c2', 'pv': ['a0a0'], 'candidates': ['b0c2'], 'facts': move_facts(START_FEN, 'b0c2'),
             'evaluation': {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'}, 'explanation': '先出马。'}
    result = judgment(FixedOracle(), row, json.dumps(value), 100)
    assert result['first_move_legal'] and not result['first_move_no_mistake']
    assert result['first_move_expected_score_loss'] > .9
    assert not result['pv_legal'] and not result['contract_valid']


def test_model_error_is_a_forfeit_and_ply_limit_never_becomes_a_draw():
    class InvalidModel:
        def generate(self, *args, **kwargs):
            return '{"move":"a0a0"}'
    game = play_game(InvalidModel(), None, ('initial', []), 'red', 100, 8)
    assert game['winner'] == 'black' and game['reason'] == 'raw_model_invalid_move_forfeit'
    censored = play_game(None, None, ('initial', []), 'red', 100, 0)
    summary = match_summary([game, censored])
    assert summary['losses'] == 1 and summary['censored'] == 1 and summary['draws'] == 0
    assert summary['elo_estimate'] is None and summary['model_oracle_repairs'] == 0
