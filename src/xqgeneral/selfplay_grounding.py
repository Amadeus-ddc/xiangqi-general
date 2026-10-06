"""Four-course rule questions from completed, train-only engine selfplay searches."""
import argparse
from collections import Counter
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import random

from .curriculum_data import STAGES, dynamic_tasks, static_tasks, verify_splits
from .engine_selfplay import context_positions
from .evidence import atomic_json, digest, history_key, load_jsonl, manifest, write_jsonl
from .planning_data import planning_label
from .policy_data import heldout_contract
from .rules import adjudicate, play, replay
from .symmetry import mirror_fen, mirror_move, mirrored_qa


def grounding_pair(query, rng, reserved=frozenset(), available_keys=None):
    """Return original/mirrored questions only for nonterminal, isolated histories."""
    source = query['record']
    if source['split'] != 'train':
        raise ValueError('Supplementary grounding queries must belong to training games')
    history = replay(source['initial_fen'], source['moves'])
    if (history != source['history'] or history[-1] != source['fen'] or
            history_key(history) != source['feature_key']):
        raise ValueError('Grounding query has inconsistent full history')
    if adjudicate(source['initial_fen'], source['moves'])['ended']:
        return [], 'terminal_root'
    oracle = query['oracle']
    best = next(c for c in oracle['candidates'] if c['move'] == oracle['best_move'])
    plan = planning_label(query, best)
    future, past, prefixes = source['fen'], list(source['moves']), []
    for move in plan['future_moves']:
        future = play(future, move)
        past.append(move)
        if adjudicate(source['initial_fen'], past)['ended']:
            break
        prefixes.append(future)
    if not prefixes:
        return [], 'no_nonterminal_future'
    horizon = rng.randrange(1, len(prefixes) + 1)
    future_moves = plan['future_moves'][:horizon]
    future = prefixes[horizon - 1]
    tasks = [('static_current', static_tasks(source['fen'], rng)),
             ('dynamic_current', dynamic_tasks(source['fen'], rng)),
             ('static_future', static_tasks(future, rng)),
             ('dynamic_future', dynamic_tasks(future, rng))]
    rows = []
    for stage, entries in tasks:
        for task, question, answer, fields in entries:
            continuation = future_moves if 'future' in stage else []
            if continuation:
                question = f"依次走 {' '.join(continuation)} 后，" + question
            identity = hashlib.sha256((query['id'] + '/' + stage + '/' + question).encode()).hexdigest()[:20]
            rows.append(dict(source, id=identity, stage=stage, task_type=task, query=fields,
                             question=question, answer=answer, future_moves=continuation,
                             future_branches=[], grounding_source_query_id=query['id'],
                             teacher={'provider': 'pinned_pyffish_rules', 'neural_prose_generated': False},
                             provenance=source['provenance'] + ';selfplay_rule_grounding'))
    initial = mirror_fen(source['initial_fen'])
    moves = [mirror_move(m) for m in source['moves']]
    mirror_context = initial, moves, replay(initial, moves)
    rows += [mirrored_qa(row, mirror_context) for row in list(rows)]
    if available_keys is not None and any(r['feature_key'] not in available_keys for r in rows):
        return [], 'root_missing_from_verified_policy_data'
    if any(context_positions(row) & reserved for row in rows):
        return [], 'heldout_root_or_future'
    return rows, None


def _sample_seed(seed, query):
    return int(hashlib.sha256(f"{seed}/{query['feature_key']}".encode()).hexdigest(), 16)


def _worker_init(reserved, available, seed):
    global _reserved, _available, _seed
    _reserved, _available, _seed = reserved, available, seed


def _worker(chunk):
    return [grounding_pair(q, random.Random(_sample_seed(_seed, q)), _reserved, _available)
            for q in chunk]


def grounding_candidates(queries, reserved, available, seed, workers=1, chunk_size=16):
    """Preserve query order with bounded parallel work; callers may stop early."""
    if min(workers, chunk_size) < 1:
        raise ValueError('Grounding workers and chunk size must be positive')
    if workers == 1:
        for query in queries:
            pair, reason = grounding_pair(query, random.Random(_sample_seed(seed, query)), reserved, available)
            yield query, pair, reason
        return
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=_worker_init, initargs=(reserved, available, seed))
    chunks = iter(queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size))
    pending = deque()
    try:
        for _ in range(workers):
            chunk = next(chunks, None)
            if chunk is not None:
                pending.append((chunk, executor.submit(_worker, chunk)))
        while pending:
            chunk, job = pending.popleft()
            results = job.result()
            next_chunk = next(chunks, None)
            if next_chunk is not None:
                pending.append((next_chunk, executor.submit(_worker, next_chunk)))
            for query, (pair, reason) in zip(chunk, results, strict=True):
                yield query, pair, reason
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/move-quality-selfplay-v3')
    parser.add_argument('--reserved-data', nargs='*', default=[])
    parser.add_argument('--game-prefix', default='engine-selfplay-')
    parser.add_argument('--train-roots', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=20261039)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--output', default='data/research-selfplay-v1')
    args = parser.parse_args()
    if args.train_roots < 2 or args.train_roots % 2:
        raise ValueError('Grounding root count must be positive, even and at least two')
    source, output = Path(args.data), Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh selfplay grounding output')
    proof_path = source / 'manifest.json'
    proof = json.loads(proof_path.read_text())
    paths = [source / f'{s}.jsonl' for s in ['train', 'validation', 'test']]
    query_path = source / 'engine-queries.jsonl'
    if proof['status'] != 'complete':
        raise ValueError('Grounding requires completed independent engine searches')
    for path in [*paths, query_path]:
        if digest(path) != proof['outputs'][str(path)]['sha256']:
            raise ValueError('Grounding source data changed')
    print(json.dumps({'phase': 'read_verified_source'}), flush=True)
    source_rows = [r for p in paths for r in load_jsonl(p)]
    base_rows = [r for r in source_rows if r['stage'] in STAGES]
    available = {r['feature_key'] for r in source_rows}
    print(json.dumps({'phase': 'heldout_rule_contract'}), flush=True)
    reserved, heldout_games, reserved_paths = heldout_contract([args.data, *args.reserved_data])
    queries = [q for q in load_jsonl(query_path) if q['record']['split'] == 'train'
               and q['record']['game_id'].startswith(args.game_prefix)]
    random.Random(args.seed).shuffle(queries)
    print(json.dumps({'phase': 'four_course_generation', 'candidate_queries': len(queries),
                      'workers': args.workers}), flush=True)
    rows, counts, rejected, roots, games = [], Counter(), Counter(), set(), set()
    if any(q['record']['game_id'] in heldout_games for q in queries):
        raise ValueError('Supplementary grounding query belongs to a held-out game')
    candidates = grounding_candidates(queries, reserved, available, args.seed, args.workers, args.chunk_size)
    try:
        for query, pair, reason in candidates:
            record = query['record']
            color = record['fen'].split()[1]
            if record['feature_key'] in roots or counts[color] >= args.train_roots // 2:
                continue
            if reason:
                rejected[reason] += 1
                continue
            rows.extend(pair)
            roots.add(record['feature_key'])
            games.add(record['game_id'])
            counts[color] += 1
            if len(roots) % 128 == 0:
                print(json.dumps({'grounded_roots': len(roots), 'goal': args.train_roots}), flush=True)
            if len(roots) == args.train_roots:
                break
    finally:
        candidates.close()
    if len(roots) != args.train_roots:
        raise ValueError(f'Insufficient isolated roots: {dict(counts)}')
    records = [*base_rows, *rows]
    split_proof = verify_splits(records)
    outputs = []
    for split in ['train', 'validation', 'test']:
        path = output / f'{split}.jsonl'
        selected = [r for r in records if r['split'] == split]
        if split != 'train' and selected != [r for r in base_rows if r['split'] == split]:
            raise ValueError('Grounding changed held-out course labels')
        write_jsonl(path, selected)
        outputs.append(path)
    summary = {**split_proof, 'new_original_training_roots': len(roots),
               'new_training_games': len(games), 'original_roots_by_side': dict(counts),
               'new_rule_question_records': len(rows), 'records': len(records),
               'by_split_stage': dict(Counter(f"{r['split']}/{r['stage']}" for r in records)),
               'rejected': dict(rejected), 'original_course_rows_preserved': True,
               'heldout_course_rows_preserved': True, 'all_queried_futures_and_mirrors_isolated': True,
               'historical_terminal_roots_and_futures_forbidden': True,
               'feature_keys_reusable_from_source': True, 'neural_prose_generated': False,
               'new_engine_searches': 0, 'student_training_executed': False,
               'independent_test_used_for_training_or_selection': False}
    atomic_json(output / 'manifest.json', manifest('selfplay_four_course_rule_grounding', vars(args),
                [proof_path, *paths, query_path, *reserved_paths], outputs, summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
