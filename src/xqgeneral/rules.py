"""Pinned Fairy-Stockfish xiangqi rules with full-history AXF adjudication."""
from functools import lru_cache
from dataclasses import dataclass
import re
import pyffish

VARIANT = "xiangqi"
START_FEN = pyffish.start_fen(VARIANT)
ALIASES = {"n": "h", "b": "e"}
PIECES = {"r": "车", "a": "仕士", "c": "炮", "p": "兵卒", "h": "马", "e": "相象", "k": "帅将"}


def piece_map(fen):
    result = {}
    rows = fen.split()[0].split("/")
    if len(rows) != 10:
        raise ValueError("Xiangqi FEN must contain 10 ranks")
    for row, text in enumerate(rows):
        file = 0
        for symbol in text:
            if symbol.isdigit():
                file += int(symbol)
                continue
            if file >= 9:
                raise ValueError("Too many files")
            kind = ALIASES.get(symbol.lower(), symbol.lower())
            if kind not in PIECES:
                raise ValueError(f"Unsupported piece: {symbol}")
            result[f"{chr(97 + file)}{9 - row}"] = kind.upper() if symbol.isupper() else kind
            file += 1
        if file != 9:
            raise ValueError("Xiangqi FEN must contain 9 files")
    return result


def piece_name(symbol):
    if not symbol:
        return "空"
    names = {"R": "红车", "A": "红仕", "C": "红炮", "P": "红兵", "H": "红马", "E": "红相", "K": "红帅",
             "r": "黑车", "a": "黑士", "c": "黑炮", "p": "黑卒", "h": "黑马", "e": "黑象", "k": "黑将"}
    return names[symbol]


def side(fen):
    return "红方" if fen.split()[1] == "w" else "黑方"


def from_fairy(move):
    # pyffish uses ranks 1..10; Px0/Pikafish and this project use ranks 0..9.
    parts = re.fullmatch(r"([a-i])(10|[1-9])([a-i])(10|[1-9])", move)
    if not parts:
        raise ValueError(f"Unexpected Fairy-Stockfish move: {move}")
    a, r, b, s = parts.groups()
    return f"{a}{int(r)-1}{b}{int(s)-1}"


def to_fairy(move):
    if not re.fullmatch(r"[a-i][0-9][a-i][0-9]", move):
        raise ValueError(f"Unexpected UCCI move: {move}")
    return f"{move[0]}{int(move[1])+1}{move[2]}{int(move[3])+1}"


@lru_cache(maxsize=16384)
def legal_moves(fen):
    return tuple(from_fairy(m) for m in pyffish.legal_moves(VARIANT, fen, []))


@lru_cache(maxsize=131072)
def play(fen, move):
    if move not in legal_moves(fen):
        raise ValueError(f"Illegal move {move} for {fen}")
    return pyffish.get_fen(VARIANT, fen, [to_fairy(move)])


def replay(initial_fen, moves):
    history = [initial_fen]
    for move in moves:
        history.append(play(history[-1], move))
    return history


def move_notation(fen, move):
    """Native WXF and Chinese notation; unusual pawn disambiguation keeps WXF only."""
    if move not in legal_moves(fen):
        raise ValueError(f'Cannot name an illegal move: {move}')
    wxf = pyffish.get_san(VARIANT, fen, to_fairy(move), False, pyffish.NOTATION_XIANGQI_WXF)
    chinese = None
    parts = re.fullmatch(r'[KAEHRCP]([1-9+\-=])([+\-=])([1-9])', wxf)
    if parts:
        origin, direction, destination = parts.groups()
        piece = piece_map(fen)[move[:2]]
        name = piece_name(piece)[1:]
        number = lambda value: '一二三四五六七八九'[int(value)-1] if piece.isupper() else value
        prefix = (name + number(origin) if origin.isdigit() else
                  {'+':'前', '-':'后', '=':'中'}[origin] + name)
        chinese = prefix + {'+':'进', '-':'退', '=':'平'}[direction] + number(destination)
    return {'move':move, 'side':side(fen), 'wxf':wxf, 'chinese':chinese}


@lru_cache(maxsize=131072)
def future_fens(fen, moves):
    """Cache immutable, legally replayed continuations for repeated split checks."""
    return tuple(replay(fen, moves))


def gives_check(fen, move):
    if move not in legal_moves(fen):
        raise ValueError("Cannot check an illegal move")
    return bool(pyffish.gives_check(VARIANT, fen, [to_fairy(move)]))


@lru_cache(maxsize=16384)
def validate_position(fen):
    """Reject malformed/illegal supplied positions before calling move functions."""
    if not isinstance(fen, str) or len(fen.split()) != 6 or fen.split()[1] not in ('w', 'b'):
        raise ValueError('A complete native Xiangqi FEN is required')
    board = piece_map(fen)
    if sum(p == 'K' for p in board.values()) != 1 or sum(p == 'k' for p in board.values()) != 1:
        raise ValueError('Native Xiangqi positions require both generals')
    # validate_fen takes FEN before variant (unlike the move APIs).
    if pyffish.validate_fen(fen, VARIANT) != pyffish.FEN_OK:
        raise ValueError('Supplied Xiangqi FEN fails native position validation')
    return fen


@lru_cache(maxsize=16384)
def in_check(fen):
    return bool(pyffish.gives_check(VARIANT, fen, []))


def _controls(board, source, piece, target):
    """Capture geometry without king-safety filtering, including cannon screens."""
    x, y = ord(source[0]) - 97, int(source[1])
    tx, ty = ord(target[0]) - 97, int(target[1])
    dx, dy = tx - x, ty - y
    if dx == dy == 0:
        return False
    red, kind = piece.isupper(), piece.lower()
    if kind in ('r', 'c'):
        if dx and dy:
            return False
        step_x, step_y = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
        distance = max(abs(dx), abs(dy))
        blockers = sum(f'{chr(97 + x + step_x * n)}{y + step_y * n}' in board
                       for n in range(1, distance))
        return blockers == (1 if kind == 'c' else 0)
    if kind == 'h':
        if (abs(dx), abs(dy)) not in ((1, 2), (2, 1)):
            return False
        leg = (x + ((dx > 0) - (dx < 0)), y) if abs(dx) == 2 else (x, y + ((dy > 0) - (dy < 0)))
        return f'{chr(97 + leg[0])}{leg[1]}' not in board
    if kind == 'e':
        return (abs(dx) == abs(dy) == 2 and (ty <= 4 if red else ty >= 5) and
                f'{chr(97 + x + dx // 2)}{y + dy // 2}' not in board)
    if kind in ('a', 'k'):
        palace = 3 <= tx <= 5 and (0 <= ty <= 2 if red else 7 <= ty <= 9)
        return palace and (abs(dx) == abs(dy) == 1 if kind == 'a' else abs(dx) + abs(dy) == 1)
    if kind == 'p':
        return (dx == 0 and dy == (1 if red else -1)) or (
            dy == 0 and abs(dx) == 1 and (y >= 5 if red else y <= 4))
    raise ValueError('Unknown Xiangqi controller')


@lru_cache(maxsize=16384)
def square_controllers(fen, square):
    """Return attackers/defenders as immutable (square, symbol) pairs.

    Like Pikafish attackers_to, this is geometric control, not legal moves.
    Pinned pieces count. Empty targets have attackers from both colors and no
    defenders; cannon control requires one screen, even for an empty target.
    Flying-general check detection belongs to in_check, not this SEE relation.
    """
    if not re.fullmatch(r'[a-i][0-9]', square):
        raise ValueError('Controller query requires a Xiangqi square')
    board = piece_map(fen)
    occupant = board.get(square)
    attackers, defenders = [], []
    for source, piece in sorted(board.items()):
        if _controls(board, source, piece, square):
            destination = defenders if occupant and occupant.isupper() == piece.isupper() else attackers
            destination.append((source, piece))
    return tuple(attackers), tuple(defenders)


@lru_cache(maxsize=16384)
def king_checkers(fen, color=None):
    """Capture-geometry checkers for one general under the native board profile.

    Facing-general restrictions remain enforced by native legal_moves; they do
    not add a synthetic checker to the pinned engine's standing-check labels.
    """
    validate_position(fen)
    color = fen.split()[1] if color is None else color
    if color not in ('w', 'b'):
        raise ValueError('General color must be w or b')
    symbol = 'K' if color == 'w' else 'k'
    square = next(s for s,p in piece_map(fen).items() if p == symbol)
    return square_controllers(fen, square)[0]


def _between_squares(source, target):
    x,y,tx,ty = ord(source[0])-97,int(source[1]),ord(target[0])-97,int(target[1])
    if x != tx and y != ty or source == target:
        raise ValueError('A checking line must be orthogonal')
    dx,dy = (tx>x)-(tx<x),(ty>y)-(ty<y)
    return tuple(f'{chr(97+x+n*dx)}{y+n*dy}' for n in range(1,max(abs(tx-x),abs(ty-y))))


def _horse_leg(source, target):
    x,y,tx,ty = ord(source[0])-97,int(source[1]),ord(target[0])-97,int(target[1])
    dx,dy = tx-x,ty-y
    if (abs(dx),abs(dy)) not in ((1,2),(2,1)):
        raise ValueError('A checking horse must attack the general by a horse step')
    lx,ly = (x+((dx>0)-(dx<0)),y) if abs(dx)==2 else (x,y+((dy>0)-(dy<0)))
    return f'{chr(97+lx)}{ly}'


@dataclass(frozen=True)
class CheckEvasionFact:
    move: str
    piece: str
    captured: str | None
    checker_mechanisms: tuple[tuple[str, str], ...]
    check: bool
    checkmate: bool


@lru_cache(maxsize=16384)
def check_evasion_facts(fen):
    """Explain every native legal parry, including cannon and horse mechanisms."""
    validate_position(fen)
    if not in_check(fen):
        raise ValueError('Check-evasion facts require a native in-check position')
    before = piece_map(fen)
    color = fen.split()[1]
    king = 'K' if color == 'w' else 'k'
    king_square = next(s for s,p in before.items() if p==king)
    checkers = king_checkers(fen)
    if not checkers:
        raise ValueError('Native check has no supported capture-geometry checker')
    facts = []
    for move in sorted(legal_moves(fen)):
        following = validate_position(play(fen,move))
        after = piece_map(following)
        if king_checkers(following,color):
            raise ValueError('Native legal parry still leaves a geometric checker')
        mechanisms = []
        for square,piece in checkers:
            if move[2:]==square:
                mechanism = 'capture_checker'
            elif before[move[:2]]==king:
                mechanism = 'king_move'
            elif piece.lower()=='r':
                line = _between_squares(square,king_square)
                if move[2:] not in line or not any(s in after for s in line):
                    raise ValueError('Rook-check evasion does not block its checking line')
                mechanism = 'block_rook_line'
            elif piece.lower()=='c':
                line = _between_squares(square,king_square)
                old_screens = [s for s in line if s in before]
                new_screens = [s for s in line if s in after]
                if len(old_screens)!=1:
                    raise ValueError('A checking cannon must have exactly one screen')
                if not new_screens and move[:2]==old_screens[0]:
                    mechanism = 'remove_cannon_screen'
                elif len(new_screens)==2 and move[2:] in line and move[:2] not in line:
                    mechanism = 'add_cannon_screen'
                else:
                    raise ValueError('Cannon-check evasion has no supported screen transition')
            elif piece.lower()=='h':
                leg = _horse_leg(square,king_square)
                if leg in before or move[2:]!=leg or leg not in after:
                    raise ValueError('Horse-check evasion does not block its leg')
                mechanism = 'block_horse_leg'
            else:
                raise ValueError('Unsupported remaining checker after a non-general parry')
            mechanisms.append((square,mechanism))
        checked = gives_check(fen,move)
        facts.append(CheckEvasionFact(move,before[move[:2]],before.get(move[2:]),
            tuple(mechanisms),checked,checked and not legal_moves(following)))
    return tuple(facts)


def adjudicate(initial_fen, moves):
    """Use full history and claim optional engine-rule outcomes in the match protocol."""
    ended, winner, reason = _adjudication(initial_fen, tuple(moves))
    return {'ended': ended, 'winner': winner, 'reason': reason,
            'rule_profile': 'pyffish-0.0.90-xiangqi-AXF'}


@lru_cache(maxsize=16384)
def _adjudication(initial_fen, moves):
    """Cache immutable outcomes by the initial FEN and entire move history."""
    history = replay(initial_fen, moves)
    converted = [to_fairy(move) for move in moves]
    for kind, function in [("variant_end", pyffish.is_immediate_game_end),
                           ("AXF_repetition_or_move_limit", pyffish.is_optional_game_end)]:
        ended, value = function(VARIANT, initial_fen, converted)
        if ended:
            # The result integer is undefined whenever the boolean flag is false.
            winner = None if value == 0 else ("red" if (value > 0) == (history[-1].split()[1] == "w") else "black")
            return True, winner, kind
    if not legal_moves(history[-1]):
        return True, "black" if history[-1].split()[1] == "w" else "red", "checkmate_or_stalemate_loss"
    return False, None, None


def prompt(record):
    return f"坐标 a0 至 i9，a0 在红方左下角。当前轮到{side(record['fen'])}。{record['question']}"
