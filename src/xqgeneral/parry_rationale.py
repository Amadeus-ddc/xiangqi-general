"""Chinese prose from native checker and check-evasion facts."""
from collections import defaultdict

from .rules import check_evasion_facts, king_checkers, piece_map, piece_name


MECHANISMS = {
    'capture_checker': '吃掉将军的棋子',
    'block_rook_line': '挡住车的将军线路',
    'remove_cannon_screen': '撤走炮架',
    'add_cannon_screen': '添子阻隔炮线',
    'block_horse_leg': '塞住马腿',
    'king_move': '移动将帅避将',
}


def parry_rationale(fen, expected_tag):
    """Name the checked general, checkers and all verified evasion mechanisms."""
    facts = check_evasion_facts(fen)
    tag = ' '.join(fact.move for fact in facts) or '无'
    if tag != expected_tag:
        raise ValueError('Parry rationale and canonical native move tag differ')
    board = piece_map(fen)
    king = 'K' if fen.split()[1]=='w' else 'k'
    square = next(s for s,p in board.items() if p==king)
    checkers = king_checkers(fen)
    names = '、'.join(f'{s}的{piece_name(p)}' for s,p in checkers)
    intro = f'{piece_name(king)}在{square}，正被{names}{"同时" if len(checkers)>1 else ""}将军。'
    if not facts:
        return intro + '没有合法解将着法，已被将死。'
    groups = defaultdict(list)
    for fact in facts:
        kinds = {mechanism for _,mechanism in fact.checker_mechanisms}
        label = '并'.join(text for kind,text in MECHANISMS.items() if kind in kinds)
        text = f'{piece_name(fact.piece)}从{fact.move[:2]}到{fact.move[2:]}'
        if fact.captured is not None:
            text += f'吃{piece_name(fact.captured)}'
        if fact.checkmate:
            text += '并将死对方'
        elif fact.check:
            text += '并将军'
        groups[label].append(text)
    return intro + ''.join(f'可通过{kind}解将：'+'；'.join(moves)+'。' for kind,moves in groups.items())
