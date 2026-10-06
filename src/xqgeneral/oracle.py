"""A CPU Pikafish oracle for verified seed data. It is separate from LM inference."""
from pathlib import Path
import argparse
import hashlib
import json
import queue
import re
import subprocess
import threading
import time
from .rules import legal_moves, play, piece_map, piece_name, side


def parse_analysis(fen, output, nodes):
    best = output[-1].split()[1]
    ranks, scored = {}, {}
    for line in output:
        if not line.startswith('info ') or ' pv ' not in line:
            continue
        score = re.search(r' score (cp|mate) (-?\d+)', line)
        if not score or 'upperbound' in line or 'lowerbound' in line:
            continue
        rank = re.search(r' multipv (\d+)', line)
        depth = re.search(r' depth (\d+)', line)
        wdl = re.search(r' wdl (\d+) (\d+) (\d+)', line)
        pv = line.split(' pv ', 1)[1].split()
        if not pv:
            continue
        ranks[int(rank.group(1)) if rank else 1] = pv[0]
        scored[pv[0]] = {'move': pv[0], 'pv': pv[:12], 'score_type': score.group(1),
                         'score': int(score.group(2)), 'perspective': 'side_to_move',
                         'depth': int(depth.group(1)) if depth else None,
                         'wdl': [int(x) for x in wdl.groups()] if wdl else None}
    if best not in legal_moves(fen):
        raise RuntimeError(f'Oracle returned an illegal or terminal move: {best}')
    if best not in scored:
        raise RuntimeError('Engine completed without a scored best-move candidate')
    # Preserve previous exact scores by move when a stop interrupts a MultiPV iteration.
    ordered_moves = list(dict.fromkeys([best, *[ranks[k] for k in sorted(ranks)]]))
    ordered = [scored[m] for m in ordered_moves[:3]]
    for candidate in ordered:
        position = fen
        for move in candidate['pv']:
            position = play(position, move)
    return {'best_move': best, 'candidates': ordered, 'requested_nodes': nodes,
            'score_perspective': 'side_to_move', 'raw_output': output}


class Pikafish:
    def __init__(self, executable, weights, threads=2, multipv=3):
        self.process = subprocess.Popen([str(Path(executable).resolve())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
        self.lines = queue.Queue()
        def reader():
            for line in self.process.stdout:
                self.lines.put(line.strip())
            self.lines.put(None)
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        self.send("uci")
        self.until("uciok")
        self.send(f"setoption name Threads value {threads}")
        self.send("setoption name Hash value 128")
        self.send(f"setoption name EvalFile value {Path(weights).resolve()}")
        self.send(f"setoption name MultiPV value {multipv}")
        self.send("setoption name UCI_ShowWDL value true")
        self.send("isready")
        self.until("readyok")

    def send(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def until(self, prefix, timeout=90):
        lines = []
        deadline = time.monotonic() + timeout
        while True:
            line = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            if line is None:
                raise RuntimeError("Pikafish exited before finishing the request")
            lines.append(line)
            if line.startswith(prefix):
                return lines
            if time.monotonic() > deadline:
                raise TimeoutError(f"Pikafish did not return {prefix}")

    def search(self, fen, nodes=20000, initial_fen=None, moves=(), searchmoves=None):
        if nodes <= 0:
            raise ValueError("Search node budget must be positive")
        self.send("ucinewgame")
        self.send("isready")
        self.until("readyok")
        if initial_fen is not None:
            from .rules import replay
            if replay(initial_fen, moves)[-1] != fen:
                raise ValueError("Oracle root differs from the supplied history")
        position = f"position fen {initial_fen or fen}"
        if initial_fen is not None and moves:
            position += " moves " + " ".join(moves)
        if searchmoves and any(m not in legal_moves(fen) for m in searchmoves):
            raise ValueError("searchmoves contains an illegal move")
        self.send(position)
        search = f"go nodes {nodes}"
        if searchmoves:
            search += " searchmoves " + " ".join(searchmoves)
        self.send(search)
        output = self.until("bestmove")
        self.last_output = output
        return output

    def analyze(self, fen, nodes=20000, initial_fen=None, moves=(), searchmoves=None):
        output = self.search(fen, nodes, initial_fen, moves, searchmoves)
        return parse_analysis(fen, output, nodes)

    def choose_move(self, fen, nodes=20000, initial_fen=None, moves=()):
        """Play at the fixed node budget; exact scores are not needed for a move."""
        output = self.search(fen, nodes, initial_fen, moves)
        fields = output[-1].split()
        if len(fields) < 2 or fields[0] != 'bestmove' or fields[1] not in legal_moves(fen):
            raise RuntimeError('Oracle returned an illegal or terminal move')
        return {'best_move': fields[1], 'requested_nodes': nodes, 'raw_output': output,
                'score_consumed': False}

    def close(self):
        if self.process.poll() is None:
            self.send("quit")
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)


def template_explanation(fen, result):
    """Facts only: a pilot HCE seed, not a neural or strategic explanation."""
    move = result["best_move"]
    board = piece_map(fen)
    source = piece_name(board[move[:2]])
    victim = piece_name(board.get(move[2:]))
    first = next(c for c in result["candidates"] if c["move"] == move)
    destination = f"吃掉{victim}" if victim != "空" else "落到空格"
    return (f"建议走 {move}：{source}从 {move[:2]} 到 {move[2:]}，{destination}。"
            f"皮卡鱼主要变化为 {' '.join(first['pv'][:6])}。"
            f"评分从当前{side(fen)}角度为 {first['score_type']} {first['score']}。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", default="vendor/pikafish/src/pikafish")
    parser.add_argument("--weights", default="vendor/pikafish/src/pikafish.nnue")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--output", default="data/oracle_seed.jsonl")
    args = parser.parse_args()
    selected = []
    for split in ["train", "validation"]:
        unique = {}
        for line in Path(f"data/{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            unique.setdefault(row["fen"], row)
        target = args.limit * 3 // 4 if split == "train" else args.limit - args.limit * 3 // 4
        candidates = list(unique.values())
        # Spread selections through multiple source games instead of taking the first game only.
        selected += [candidates[int(i * len(candidates)/target)] for i in range(target)]
    started = time.time()
    oracle = Pikafish(args.executable, args.weights)
    count = 0
    try:
        with Path(args.output).open("w") as output:
            for record in selected:
                result = oracle.analyze(record["fen"], args.nodes)
                item = {**record, "stage": "engine_seed_explanation",
                        "question": "请给出建议走法和可核对的变化。",
                        "answer": template_explanation(record["fen"], result),
                        "oracle": result, "teacher_type": "deterministic_HCE_template",
                        "neural_teacher": False, "pv_legality_verified": True}
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                output.flush()
                count += 1
    finally:
        oracle.close()
    manifest = {"records": count, "nodes_per_root": args.nodes, "threads": 2,
                "seconds": time.time() - started, "all_best_moves_and_pvs_legal": True,
                "nnue_sha256": hashlib.file_digest(Path(args.weights).open("rb"), "sha256").hexdigest(),
                "executable_sha256": hashlib.file_digest(Path(args.executable).open("rb"), "sha256").hexdigest(),
                "teacher_type": "deterministic_HCE_template", "not_neural_model_explanations": True}
    Path("runs/oracle_seed.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
