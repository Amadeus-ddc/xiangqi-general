"""Pinned Fairy-Stockfish xiangqi rules with full-history AXF adjudication."""
from functools import lru_cache
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


@lru_cache(maxsize=131072)
def future_fens(fen, moves):
    """Cache immutable, legally replayed continuations for repeated split checks."""
    return tuple(replay(fen, moves))


def gives_check(fen, move):
    if move not in legal_moves(fen):
        raise ValueError("Cannot check an illegal move")
    return bool(pyffish.gives_check(VARIANT, fen, [to_fairy(move)]))


def adjudicate(initial_fen, moves):
    """Use full history and claim optional engine-rule outcomes in the match protocol."""
    history = replay(initial_fen, moves)
    converted = [to_fairy(move) for move in moves]
    for kind, function in [("variant_end", pyffish.is_immediate_game_end),
                           ("AXF_repetition_or_move_limit", pyffish.is_optional_game_end)]:
        ended, value = function(VARIANT, initial_fen, converted)
        if ended:
            # The result integer is undefined whenever the boolean flag is false.
            winner = None if value == 0 else ("red" if (value > 0) == (history[-1].split()[1] == "w") else "black")
            return {"ended": True, "winner": winner, "reason": kind, "rule_profile": "pyffish-0.0.90-xiangqi-AXF"}
    if not legal_moves(history[-1]):
        return {"ended": True, "winner": "black" if history[-1].split()[1] == "w" else "red",
                "reason": "checkmate_or_stalemate_loss", "rule_profile": "pyffish-0.0.90-xiangqi-AXF"}
    return {"ended": False, "winner": None, "reason": None, "rule_profile": "pyffish-0.0.90-xiangqi-AXF"}


def prompt(record):
    return f"坐标 a0 至 i9，a0 在红方左下角。当前轮到{side(record['fen'])}。{record['question']}"
