"""Native, factual move narration for explicitly selected future-question data."""
from dataclasses import dataclass
import re

from .rules import in_check, legal_moves, piece_map, piece_name, play, validate_position


FORWARD_MOVES_PROFILE = 'paper_forward_moves_v1'
PARRY_REASONING_PROFILE = 'paper_parry_reasons_v1'
FORWARD_PARRY_PROFILE = 'paper_forward_parry_v1'
RATIONALE_PROFILES = {
    FORWARD_MOVES_PROFILE: (True, False),
    PARRY_REASONING_PROFILE: (False, True),
    FORWARD_PARRY_PROFILE: (True, True),
}


def checked_rationale_profile(profile):
    if profile is not None and (not isinstance(profile, str) or profile not in RATIONALE_PROFILES):
        raise ValueError('Unknown curriculum rationale profile')
    return profile


def rationale_features(profile):
    checked_rationale_profile(profile)
    return RATIONALE_PROFILES.get(profile, (False, False))


def rationale_applies(row, profile):
    forward, parry = rationale_features(profile)
    return (forward and row['stage'].endswith('_future')) or (parry and row['task_type'] == 'parries')


@dataclass(frozen=True)
class ForwardMoveFact:
    move: str
    piece: str
    captured: str | None
    check: bool
    checkmate: bool
    stalemate: bool


def forward_move_facts(row):
    """Replay the supplied line from its presented board, retaining each mover."""
    if row.get('stage') not in ('static_future', 'dynamic_future'):
        raise ValueError('Forward rationale requires a future-stage question')
    moves = row.get('future_moves')
    if (not isinstance(moves, list) or not 1 <= len(moves) <= 8 or
            any(not isinstance(move, str) or re.fullmatch(r'[a-i][0-9][a-i][0-9]', move) is None
                for move in moves)):
        raise ValueError('Forward rationale requires one to eight supplied native moves')
    fen = validate_position(row['fen'])
    facts = []
    for move in moves:
        board = piece_map(fen)
        following = validate_position(play(fen, move))
        checked = in_check(following)
        no_moves = not legal_moves(following)
        facts.append(ForwardMoveFact(move, board[move[:2]], board.get(move[2:]),
                                     checked, checked and no_moves, not checked and no_moves))
        fen = following
    return tuple(facts)


def forward_move_narration(row):
    """Describe only verified supplied moves; keep final-task reasoning separate."""
    variant = row.get('rationale_variant', 0)
    if type(variant) is not int or variant not in (0, 1):
        raise ValueError('Forward rationale variant must be zero or one')
    facts = forward_move_facts(row)
    clauses = []
    for index, fact in enumerate(facts):
        connector = (f'第{index + 1}着，' if variant else
                     '首先，' if index == 0 else '最后，' if index == len(facts) - 1 else '随后，')
        text = connector + f'{piece_name(fact.piece)}从{fact.move[:2]}走到{fact.move[2:]}'
        if fact.captured is not None:
            text += f'，吃掉{piece_name(fact.captured)}'
        if fact.checkmate:
            text += '，将死对方'
        elif fact.check:
            text += '，形成将军'
        elif fact.stalemate:
            text += '，形成困毙'
        clauses.append(text + '。')
    return ''.join(clauses)
