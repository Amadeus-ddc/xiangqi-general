"""Search-generated game contexts for training; never neural prose labels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from .calibration import candidate_probability
from .evaluate_games import OPENINGS
from .evidence import atomic_json, code_identity, digest, history_key, manifest, position_key, write_jsonl
from .oracle import Pikafish
from .rules import START_FEN, adjudicate, future_fens, legal_moves, play, replay
from .search_distillation import reserved_positions


def choose_move(fen, result, rng, explore, tolerance=.03):
    candidates = result['candidates']
    legal = legal_moves(fen)
    if (not candidates or result['best_move'] not in legal or
            any(c['move'] not in legal for c in candidates)):
        raise ValueError('Self-play engine supplied illegal candidates')
    if not explore:
        return result['best_move']
    values = [candidate_probability(fen, c) for c in candidates]
    best = max(values)
    eligible = [c for c, value in zip(candidates, values) if best - value <= tolerance]
    return rng.choices(eligible, weights=[.5 ** i for i in range(len(eligible))], k=1)[0]['move']


def analyze_selfplay(oracle, fen, nodes, moves, retry_nodes=None):
    try:
        return oracle.analyze(fen, nodes, START_FEN, moves)
    except RuntimeError as error:
        if str(error) != 'Engine completed without a scored best-move candidate' or not retry_nodes:
            raise
        failed = {'requested_nodes': nodes, 'error': str(error),
                  'raw_output': list(getattr(oracle, 'last_output', []))}
        result = oracle.analyze(fen, retry_nodes, START_FEN, moves)
        return dict(result, retried_unscored_search=failed)


def generate_game(oracle, index, seed, nodes, max_plies, explore_plies, min_ply, max_roots,
                  retry_nodes=None):
    rng = random.Random(seed + index)
    opening, book = OPENINGS[index % len(OPENINGS)]
    moves, history = list(book), replay(START_FEN, book)
    game_id = f'engine-selfplay-{seed}-{index:05d}'
    contexts, turns = [], []
    while True:
        outcome = adjudicate(START_FEN, moves)
        if outcome['ended']:
            status = 'completed'
            break
        if len(moves) >= max_plies:
            status = 'censored'
            outcome = {'winner': None, 'reason': 'ply_limit'}
            break
        fen, ply = history[-1], len(moves)
        if ply >= min_ply and (ply - min_ply) % 4 < 2:
            key = history_key(history)
            contexts.append({'id': key[:20], 'feature_key': key, 'game_id': game_id, 'split': 'train',
                             'initial_fen': START_FEN, 'moves': list(moves), 'history': list(history),
                             'fen': fen, 'ply': ply, 'stage': 'source_context', 'task_type': 'engine_context',
                             'query': {}, 'question': '', 'answer': '', 'future_moves': [],
                             'provenance': 'pinned_Pikafish_selfplay;bounded_early_multipv_exploration'})
        result = analyze_selfplay(oracle, fen, nodes, moves, retry_nodes)
        explore = ply < explore_plies
        move = choose_move(fen, result, rng, explore)
        turns.append({'ply': ply, 'move': move, 'exploration_enabled': explore, 'oracle': result})
        moves.append(move)
        history.append(play(fen, move))
    selected = []
    for side in ('w', 'b'):
        pool = [r for r in contexts if r['fen'].split()[1] == side]
        selected.extend(rng.sample(pool, min(len(pool), max_roots // 2)))
    selected.sort(key=lambda r: r['ply'])
    for row in selected:
        row['future_moves'] = moves[row['ply']:row['ply'] + 5]
    return {'game_id': game_id, 'split': 'train', 'opening': opening, 'status': status,
            'winner': outcome['winner'], 'reason': outcome['reason'], 'moves': moves,
            'turns': turns, 'contexts': selected}


def context_positions(row):
    positions = {position_key(row['fen'])}
    for line in [row.get('future_moves', []), *row.get('future_branches', [])]:
        positions.update(position_key(fen) for fen in future_fens(row['fen'], tuple(line)))
    return positions


def select_training_contexts(games, forbidden):
    records, seen = [], set()
    rejected = Counter()
    for game in games:
        if game['split'] != 'train':
            raise ValueError('Self-play supplement must contain training games only')
        for row in game['contexts']:
            if context_positions(row) & forbidden:
                rejected['heldout_root_or_future'] += 1
                continue
            if row['feature_key'] in seen:
                rejected['duplicate_history'] += 1
                continue
            seen.add(row['feature_key'])
            records.append(row)
    return records, dict(rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=128)
    parser.add_argument('--nodes', type=int, default=1000)
    parser.add_argument('--retry-nodes', type=int, default=10000,
                        help='One larger search when the node stop leaves the best move unscored')
    parser.add_argument('--max-plies', type=int, default=192)
    parser.add_argument('--explore-plies', type=int, default=24)
    parser.add_argument('--min-ply', type=int, default=8)
    parser.add_argument('--max-roots', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--reserved-data', default='data/move-quality-v1')
    parser.add_argument('--output', default='data/engine-selfplay-train-v2')
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    args = parser.parse_args()
    if (min(args.games, args.nodes, args.workers) < 1 or args.max_plies <= 4 or
            not 0 <= args.min_ply < args.max_plies or args.explore_plies < 0 or
            args.max_roots < 2 or args.max_roots % 2 or args.retry_nodes <= args.nodes):
        raise ValueError('Invalid self-play game or context budget')
    root = Path(args.output)
    inputs = [args.executable, args.weights,
              *[Path(args.reserved_data) / f'{s}.jsonl' for s in ('validation', 'test')]]
    code = code_identity()
    contract = {'arguments': vars(args), 'code': code,
                'input_hashes': {str(p): digest(p) for p in inputs}}
    contract_path = root / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Self-play continuation configuration, source or inputs changed')
    if (root / 'manifest.json').exists():
        raise FileExistsError('Completed self-play dataset exists; use a new output')
    atomic_json(contract_path, contract)

    def worker(worker_id):
        oracle = Pikafish(args.executable, args.weights, threads=1, multipv=3)
        games = []
        try:
            for index in range(worker_id, args.games, args.workers):
                path = root / 'games' / f'game-{index:05d}.json'
                if path.exists():
                    game = json.loads(path.read_text())
                    if game['game_id'] != f'engine-selfplay-{args.seed}-{index:05d}':
                        raise ValueError('Completed self-play game identity changed')
                else:
                    game = generate_game(oracle, index, args.seed, args.nodes, args.max_plies,
                                         args.explore_plies, args.min_ply, args.max_roots, args.retry_nodes)
                    atomic_json(path, game)
                games.append(game)
                print(json.dumps({'game': index, 'status': game['status'], 'plies': len(game['moves']),
                                  'contexts': len(game['contexts'])}), flush=True)
        finally:
            oracle.close()
        return games

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(worker, i) for i in range(args.workers)]
        games = [game for job in jobs for game in job.result()]
    games.sort(key=lambda game: game['game_id'])
    forbidden = reserved_positions(args.reserved_data)
    records, rejected = select_training_contexts(games, forbidden)
    write_jsonl(root / 'contexts.jsonl', records)
    write_jsonl(root / 'games.jsonl', games)
    overlap = sum(bool(context_positions(row) & forbidden) for row in records)
    proof = {'games': len(games), 'game_statuses': dict(Counter(g['status'] for g in games)),
             'termination_reasons': dict(Counter(g['reason'] for g in games)),
             'contexts': len(records), 'by_side': dict(Counter(r['fen'].split()[1] for r in records)),
             'rejected': rejected, 'heldout_root_or_future_overlap': overlap,
             'unscored_search_retries': sum('retried_unscored_search' in turn['oracle']
                                           for game in games for turn in game['turns']),
             'full_histories_preserved': True, 'all_contexts_training_only': True,
             'neural_prose_generated': False, 'training_labels_generated': False,
             'censored_games_counted_as_draws': False}
    if overlap or not records:
        raise ValueError('Self-play contexts are empty or overlap heldout positions')
    atomic_json(root / 'manifest.json', manifest('engine_selfplay_training_contexts', vars(args),
                inputs, [root / 'contexts.jsonl', root / 'games.jsonl'], proof, code))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
