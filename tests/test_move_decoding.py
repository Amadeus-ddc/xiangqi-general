import pytest
import torch
from xqgeneral.move_decoding import MoveTokenConstraint, legal_options
from xqgeneral.rules import START_FEN, replay
from xqgeneral.evaluate_games import play_game


class ByteTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def test_legal_tokens_exclude_high_scoring_illegal_move_and_force_completion():
    options = ['a0a1', 'a0b0']
    trie = MoveTokenConstraint(ByteTokenizer(), options)
    answer = []
    for _ in range(trie.max_tokens):
        scores = torch.zeros(256)
        scores[ord('z')] = 100  # An illegal model preference must not win.
        scores[0] = 50  # EOS is illegal before the entire move is complete.
        allowed = torch.tensor(trie.allowed(answer))
        token = int(allowed[scores[allowed].argmax()])
        answer.append(token)
    assert answer[-1] == 0
    assert bytes(answer[:-1]).decode() in options
    assert trie.allowed([255]) == [0]  # Discarded beam cannot generate another move.


def test_move_constraints_require_matching_nonterminal_full_history():
    row = {'fen': START_FEN, 'initial_fen': START_FEN, 'moves': [], 'history': [START_FEN]}
    assert 'b0c2' in legal_options(row) and 'a0a0' not in legal_options(row)
    with pytest.raises(ValueError, match='history differ'):
        legal_options(dict(row, moves=['b0c2']))
    repeated = ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2
    history = replay(START_FEN, repeated)
    with pytest.raises(ValueError, match='terminal'):
        legal_options(dict(row, moves=repeated, fen=history[-1], history=history))


def test_constrained_move_matches_keep_censoring_and_forfeit_semantics():
    class Model:
        def generate_moves(self, records, beams):
            assert len(records[0]['history']) == len(records[0]['moves']) + 1
            return ['b0c2']
    game = play_game(Model(), None, ('initial', []), 'red', 100, 1, 'legal_move')
    assert game['status'] == 'censored'
    assert game['turns'][0]['action_mode'] == 'legal_move'
    assert not game['turns'][0]['model_oracle_used']
    class InvalidModel:
        def generate_moves(self, *args, **kwargs):
            return ['a0a0']
    game = play_game(InvalidModel(), None, ('initial', []), 'red', 100, 1, 'legal_move')
    assert game['reason'] == 'raw_model_invalid_move_forfeit' and game['winner'] == 'black'
