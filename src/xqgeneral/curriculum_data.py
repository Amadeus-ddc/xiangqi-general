"""Seeded, game-disjoint four-course data with difficult negative probes."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import random
from .evidence import atomic_json, history_key, manifest, position_key, write_jsonl
from .rules import START_FEN, gives_check, legal_moves, play, piece_map, piece_name

STAGES = ("static_current", "dynamic_current", "static_future", "dynamic_future")
SQUARES = tuple(f"{file}{rank}" for rank in range(10) for file in "abcdefghi")
SYMBOLS = tuple("RACHEPKrachepk")
MATERIAL = {"r": 9, "h": 4, "c": 4, "a": 2, "e": 2, "p": 1, "k": 0}


def illegal_probe(fen, rng):
    legal = legal_moves(fen)
    board = piece_map(fen)
    sources = sorted(s for s, p in board.items() if p.isupper() == (fen.split()[1] == "w"))
    if legal:
        anchor = rng.choice(legal)
        file, rank = ord(anchor[2]) - 97, int(anchor[3])
        nearby = [f"{anchor[:2]}{chr(97 + f)}{r}" for f in range(max(0, file - 1), min(9, file + 2))
                  for r in range(max(0, rank - 1), min(10, rank + 2))]
        nearby = [m for m in nearby if m not in legal and m[:2] != m[2:]]
        if nearby:
            return rng.choice(nearby)
    choices = [f"{s}{d}" for s in sources for d in SQUARES if s != d and f"{s}{d}" not in legal]
    return rng.choice(choices) if choices else "a0a0"


def static_tasks(fen, rng):
    board = piece_map(fen)
    occupied = rng.choice(sorted(board))
    symbol = rng.choice(SYMBOLS)
    locations = sorted(s for s, p in board.items() if p == symbol)
    tasks = [("piece", f"{occupied} 上是什么棋子？只回答棋子名称。", piece_name(board[occupied]), {"square": occupied}),
             ("count", f"当前有几个{piece_name(symbol)}？只回答数字。", str(len(locations)), {"symbol": symbol}),
             ("locate", f"列出所有{piece_name(symbol)}所在格，按坐标字典序用空格分隔，没有则回答无。",
              " ".join(locations) or "无", {"symbol": symbol})]
    empty = sorted(set(SQUARES) - board.keys())
    if empty:
        square = rng.choice(empty)
        tasks.append(("empty", f"{square} 上是什么棋子？只回答棋子名称。", "空", {"square": square}))
    red = rng.choice([True, False])
    value = sum(MATERIAL[p.lower()] for p in board.values() if p.isupper() == red)
    tasks.append(("material", f"按车9马4炮4仕士2相象2兵卒1帅将0计，{'红' if red else '黑'}方子力总值是多少？只回答数字。",
                  str(value), {"red": red}))
    rank = rng.randrange(10)
    pieces = [f"{s}:{piece_name(board[s])}" for s in SQUARES if s[1] == str(rank) and s in board]
    tasks.append(("rank", f"列出第{rank}行棋子，以坐标:棋子名表示，按坐标字典序用空格分隔，没有则回答无。",
                  " ".join(pieces) or "无", {"rank": rank}))
    return tasks


def dynamic_tasks(fen, rng):
    legal = legal_moves(fen)
    if not legal:
        return [("terminal", "当前还有合法走法吗？只回答有或无。", "无", {})]
    move, illegal = rng.choice(legal), illegal_probe(fen, rng)
    source = rng.choice(legal)[:2]
    board = piece_map(fen)
    captures = sorted(m for m in legal if m[2:] in board)
    checks = sorted(m for m in legal if gives_check(fen, m))
    return [("legal", f"走法 {move} 合法吗？只回答合法或不合法。", "合法", {"move": move}),
            ("illegal", f"走法 {illegal} 合法吗？只回答合法或不合法。", "不合法", {"move": illegal}),
            ("moves", f"列出 {source} 上棋子的所有合法走法，按字典序用空格分隔，没有则回答无。",
             " ".join(sorted(m for m in legal if m[:2] == source)) or "无", {"source": source}),
            ("captures", "列出当前所有合法吃子走法，按字典序用空格分隔，没有则回答无。",
             " ".join(captures) or "无", {}),
            ("checks", "列出当前所有合法将军走法，按字典序用空格分隔，没有则回答无。",
             " ".join(checks) or "无", {})]


def make_records(games=80, seed=20261006, plies=64, test_fraction=0.15):
    if games < 3 or not 0 <= test_fraction < 0.4:
        raise ValueError("At least three games and a valid test fraction are required")
    rows, ownership = [], {}
    train_end, validation_end = int(games * (0.8 - test_fraction)), int(games * (1 - test_fraction))
    for game in range(games):
        split = "train" if game < train_end else "validation" if game < validation_end else "test"
        rng = random.Random(seed + game)
        fen, history, moves = START_FEN, [START_FEN], []
        for ply in range(plies):
            legal = legal_moves(fen)
            if not legal:
                break
            chosen = rng.choice(legal)
            fen = play(fen, chosen)
            moves.append(chosen)
            history.append(fen)
            if ply < 8 or ply % 4 or position_key(fen) in ownership:
                continue
            future, future_moves, future_keys = fen, [], set()
            for _ in range(rng.randint(1, 8)):
                choices = legal_moves(future)
                if not choices:
                    break
                chosen = rng.choice(choices)
                future = play(future, chosen)
                future_moves.append(chosen)
                future_keys.add(position_key(future))
            keys = {position_key(fen), *future_keys}
            if any(k in ownership and ownership[k] != split for k in keys):
                continue
            ownership.update({k: split for k in keys})
            tasks = [("static_current", static_tasks(fen, rng)), ("dynamic_current", dynamic_tasks(fen, rng)),
                     ("static_future", static_tasks(future, rng)), ("dynamic_future", dynamic_tasks(future, rng))]
            for stage, entries in tasks:
                for task, question, answer, query in entries:
                    if "future" in stage:
                        question = f"依次走 {' '.join(future_moves)} 后，" + question
                    identity = hashlib.sha256(f"{seed}/{game}/{ply}/{stage}/{question}".encode()).hexdigest()[:20]
                    rows.append({"id": identity, "game_id": f"synthetic-{seed}-{game}", "split": split,
                                 "stage": stage, "task_type": task, "query": query,
                                 "initial_fen": START_FEN, "moves": moves.copy(), "fen": fen,
                                 "history": history.copy(), "feature_key": history_key(history),
                                 "question": question, "answer": answer,
                                 "future_moves": future_moves if "future" in stage else [],
                                 "provenance": "seeded_random_legal_play;not_human_games"})
    return rows


def verify_splits(rows):
    games, roots, future, seen = {}, {}, {}, set()
    for row in rows:
        s = row["split"]
        games.setdefault(s, set()).add(row["game_id"])
        roots.setdefault(s, set()).add(position_key(row["fen"]))
        key = (s, row['fen'], tuple(row.get('future_moves', [])))
        if key in seen:
            continue
        seen.add(key)
        positions = future.setdefault(s, set())
        positions.add(position_key(row["fen"]))
        fen = row["fen"]
        for move in row.get("future_moves", []):
            fen = play(fen, move)
            positions.add(position_key(fen))
    splits = sorted(games)
    for i, s in enumerate(splits):
        for t in splits[i + 1:]:
            if games[s] & games[t] or future[s] & future[t]:
                raise ValueError(f"Data leakage between {s} and {t}")
    return {"games": {s: len(v) for s, v in games.items()}, "roots": {s: len(v) for s, v in roots.items()},
            "game_overlap": 0, "root_and_future_overlap": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/research-v1")
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    dest = Path(args.output)
    if (dest / "manifest.json").exists():
        raise FileExistsError("Completed dataset exists; select a new output directory")
    rows = make_records(games=args.games, seed=args.seed)
    proof, outputs = verify_splits(rows), []
    for split in ["train", "validation", "test"]:
        path = dest / f"{split}.jsonl"
        write_jsonl(path, [r for r in rows if r["split"] == split])
        outputs.append(path)
    summary = {"records": len(rows), "counts": dict(Counter(f"{r['split']}/{r['stage']}/{r['task_type']}" for r in rows)),
               "source_games": args.games, "seed": args.seed, "provenance": "synthetic_legal_games",
               "split_verification": proof, "future_negative_examples": sum(r["stage"] == "dynamic_future" and
                                                                          r["task_type"] == "illegal" for r in rows)}
    atomic_json(dest / "manifest.json", manifest("curriculum_data", vars(args), outputs=outputs, verification=summary))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
