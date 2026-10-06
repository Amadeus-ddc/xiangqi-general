"""Rule-only move constraints; no engine scores or replacement moves."""
from .board_tokens import encode_board_text
from .rules import adjudicate, legal_moves, replay


def legal_options(record):
    history = replay(record['initial_fen'], record['moves'])
    if history[-1] != record['fen'] or record.get('history', history) != history:
        raise ValueError('Move decoding board and complete history differ')
    if adjudicate(record['initial_fen'], record['moves'])['ended']:
        raise ValueError('Cannot predict a move after a terminal history')
    options = legal_moves(record['fen'])
    if not options:
        raise ValueError('Move decoding has no legal options')
    return options


class MoveTokenConstraint:
    def __init__(self, tokenizer, moves, board_tokens=False):
        self.moves = set(moves)
        if not self.moves or tokenizer.eos_token_id is None:
            raise ValueError('Move constraints require legal moves and an EOS token')
        self.eos = tokenizer.eos_token_id
        self.tree = {}
        self.max_tokens = 0
        for move in sorted(self.moves):
            text = encode_board_text(move) if board_tokens else move
            tokens = tokenizer.encode(text, add_special_tokens=False) + [self.eos]
            self.max_tokens = max(self.max_tokens, len(tokens))
            node = self.tree
            for token in tokens:
                node = node.setdefault(token, {})

    def allowed(self, generated):
        node = self.tree
        for token in generated:
            if token not in node:
                # Beam search can query a discarded path with score -inf.
                return [self.eos]
            node = node[token]
        return sorted(node) if node else [self.eos]
