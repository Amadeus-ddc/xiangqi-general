"""Dedicated board vocabulary while preserving every pretrained weight row."""
import re
import torch
from torch import nn
from torch.nn import functional as F

SQUARES = tuple(f'{f}{r}' for r in range(10) for f in 'abcdefghi')
PIECES = ('红车', '红马', '红炮', '红仕', '红相', '红兵', '红帅',
          '黑车', '黑马', '黑炮', '黑士', '黑象', '黑卒', '黑将', '空')
PIECE_CODES = ('RED_ROOK', 'RED_HORSE', 'RED_CANNON', 'RED_ADVISOR', 'RED_ELEPHANT', 'RED_PAWN', 'RED_GENERAL',
               'BLACK_ROOK', 'BLACK_HORSE', 'BLACK_CANNON', 'BLACK_ADVISOR', 'BLACK_ELEPHANT', 'BLACK_PAWN', 'BLACK_GENERAL', 'EMPTY')
PAIRS = tuple((s, f'<XQ_{s}>') for s in SQUARES) + tuple((p, f'<XQ_{c}>') for p, c in zip(PIECES, PIECE_CODES))
RAW_TO_TOKEN = dict(PAIRS)
TOKEN_TO_RAW = {v: k for k, v in PAIRS}
RAW_PATTERN = re.compile('|'.join(re.escape(s) for s in sorted(RAW_TO_TOKEN, key=len, reverse=True)))
TOKEN_PATTERN = re.compile('|'.join(re.escape(s) for s in TOKEN_TO_RAW))


def encode_board_text(text):
    return RAW_PATTERN.sub(lambda m: RAW_TO_TOKEN[m.group()], text)


def decode_board_text(text):
    return TOKEN_PATTERN.sub(lambda m: TOKEN_TO_RAW[m.group()], text)


class BoardInputEmbedding(nn.Module):
    def __init__(self, frozen, token_ids, initial, vocab_size):
        super().__init__()
        self.frozen = frozen
        self.frozen.requires_grad_(False)
        self.board_weight = nn.Parameter(initial.clone())
        slots = torch.full((vocab_size,), -1, dtype=torch.long, device=initial.device)
        slots[torch.tensor(token_ids, device=initial.device)] = torch.arange(len(token_ids), device=initial.device)
        self.register_buffer('slots', slots)
        self.num_embeddings, self.embedding_dim = vocab_size, initial.shape[1]

    @property
    def weight(self):
        return self.frozen.weight

    def forward(self, ids):
        slots = self.slots[ids]
        # Added IDs may occupy unused padded rows or extend the original vocabulary.
        old = self.frozen(ids.clamp_max(self.frozen.num_embeddings - 1))
        new = F.embedding(slots.clamp_min(0), self.board_weight)
        return torch.where((slots >= 0).unsqueeze(-1), new, old)


class BoardOutputHead(nn.Module):
    def __init__(self, frozen, token_ids, initial, vocab_size):
        super().__init__()
        self.frozen = frozen
        self.frozen.requires_grad_(False)
        self.board_weight = nn.Parameter(initial.clone())
        self.register_buffer('token_ids', torch.tensor(token_ids, device=initial.device))
        self.out_features = vocab_size

    @property
    def weight(self):
        return self.frozen.weight

    def forward(self, hidden):
        logits = self.frozen(hidden)
        if logits.shape[-1] < self.out_features:
            logits = F.pad(logits, (0, self.out_features - logits.shape[-1]), value=-10000)
        added = F.linear(hidden, self.board_weight)
        return logits.index_copy(-1, self.token_ids, added)


def install_board_tokens(base, tokenizer):
    core = base.get_base_model() if hasattr(base, 'get_base_model') else base
    original_input, original_output = core.get_input_embeddings(), core.get_output_embeddings()
    if isinstance(original_input, BoardInputEmbedding):
        raise ValueError('Board vocabulary is already installed')
    encodings = [tokenizer.encode(raw, add_special_tokens=False) for raw, _ in PAIRS]
    with torch.no_grad():
        initial_input = torch.stack([original_input.weight[ids].mean(0) for ids in encodings])
        initial_output = torch.stack([original_output.weight[ids].mean(0) for ids in encodings])
    added = tokenizer.add_tokens([token for _, token in PAIRS], special_tokens=False)
    if added != len(PAIRS):
        raise ValueError('Unexpected pre-existing board vocabulary')
    ids = tokenizer.convert_tokens_to_ids([token for _, token in PAIRS])
    vocab_size = max(original_input.num_embeddings, max(ids) + 1)
    core.set_input_embeddings(BoardInputEmbedding(original_input, ids, initial_input, vocab_size))
    core.set_output_embeddings(BoardOutputHead(original_output, ids, initial_output, vocab_size))
    core.config.vocab_size = vocab_size
    return {'tokens': len(PAIRS), 'ids': ids, 'vocab_size': vocab_size,
            'pretrained_rows_modified': False}
