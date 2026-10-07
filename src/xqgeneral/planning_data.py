"""Engine PV planning lessons with full-history termination and heldout isolation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path

from .curriculum_data import verify_splits
from .engine_selfplay import context_positions
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from .rules import adjudicate, play, replay
from .symmetry import mirror_fen, mirror_move, mirrored_qa

BEST_QUESTION = '给出当前最佳走法的主变化，最多六步。只输出 UCCI 走法，按顺序用空格分隔，不写说明。'
BRANCH_QUESTION = '以 {move} 为首着给出分析变化，最多六步。只输出 UCCI 走法，按顺序用空格分隔，不写说明。'


@lru_cache(maxsize=65536)
def planning_ended(initial_fen, moves):
    return adjudicate(initial_fen, moves)['ended']


def planning_label(query, candidate, forced=False, max_plies=6):
    row = query['record']
    if not candidate['pv'] or candidate['pv'][0] != candidate['move']:
        raise ValueError('Planning candidate differs from its first PV move')
    position, history_moves, pv = row['fen'], list(row['moves']), []
    for move in candidate['pv'][:max_plies]:
        if planning_ended(row['initial_fen'], tuple(history_moves)):
            break
        position = play(position, move)
        history_moves.append(move)
        pv.append(move)
    if not pv:
        return None
    question = BRANCH_QUESTION.format(move=candidate['move']) if forced else BEST_QUESTION
    return dict(row, id=query['id'] + ('-plan-' + candidate['move'] if forced else '-plan-best'),
                stage='move_planning', task_type='conditional_line' if forced else 'best_line',
                query={'move': candidate['move']} if forced else {}, question=question,
                answer=' '.join(pv), future_moves=pv,
                teacher={'provider': 'pinned_Pikafish', 'neural_prose_generated': False},
                provenance=row['provenance'] + ';full_history_engine_pv_planning')


def _planning_chunk(queries, available_keys, max_plies):
    labels, rejected = [], Counter()
    for query in queries:
        candidates = query['oracle']['candidates']
        best = next(c for c in candidates if c['move'] == query['oracle']['best_move'])
        context = None
        for candidate, forced in [(best, False), *[(c, True) for c in candidates]]:
            row = planning_label(query, candidate, forced, max_plies)
            if row is None:
                rejected['terminal_root'] += 1
                continue
            if context is None:
                initial = mirror_fen(row['initial_fen'])
                moves = [mirror_move(m) for m in row['moves']]
                context = initial, moves, replay(initial, moves)
            for item in (row, mirrored_qa(row, context)):
                if item['feature_key'] not in available_keys:
                    rejected['root_not_in_verified_policy_data'] += 1
                    continue
                if 'augmentation_parent' in item:
                    item['teacher'] = dict(item['teacher'], independent_engine_query=False,
                                           derivation='color_rank_symmetry')
                labels.append(item)
    return labels, dict(rejected)


def _planning_worker_init(available_keys, max_plies):
    global _worker_keys, _worker_plies
    _worker_keys, _worker_plies = available_keys, max_plies


def _planning_worker(queries):
    return _planning_chunk(queries, _worker_keys, _worker_plies)


def planning_lessons(queries, available_keys, max_plies=6, progress=None, workers=1, chunk_size=64):
    if min(workers, chunk_size) < 1:
        raise ValueError('Planning workers and chunk size must be positive')
    chunks = [queries[i:i+chunk_size] for i in range(0, len(queries), chunk_size)]
    executor = None
    if workers == 1:
        results = (_planning_chunk(chunk, available_keys, max_plies) for chunk in chunks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=_planning_worker_init,
            initargs=(available_keys, max_plies))
        results = executor.map(_planning_worker, chunks)
    labels, rejected, completed, reported = [], Counter(), 0, 0
    try:
        for chunk, (rows, errors) in zip(chunks, results):
            labels.extend(rows)
            rejected.update(errors)
            completed += len(chunk)
            if progress is not None and (completed - reported >= 500 or completed == len(queries)):
                progress(completed)
                reported = completed
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return labels, dict(rejected)


def _position_signature(row):
    return (row['fen'], tuple(row.get('future_moves', [])),
            tuple(tuple(line) for line in row.get('future_branches', [])))


def _isolation_positions(signatures):
    return [context_positions({'fen': fen, 'future_moves': moves,
                               'future_branches': branches})
            for fen, moves, branches in signatures]


def isolate_planning_rows(base_rows, labels, workers=1, chunk_size=256, progress=None):
    if min(workers, chunk_size) < 1:
        raise ValueError('Isolation workers and chunk size must be positive')
    combined = [*base_rows, *labels]
    positions = None
    if workers > 1:
        # Keep heldout-first validation order; workers need only future context,
        # not prompts, answers or the complete source-game histories.
        ordered = [r for r in combined if r['split'] in {'validation', 'test'}]
        ordered.extend(r for r in combined if r['split'] == 'train')
        signatures = list(dict.fromkeys(_position_signature(r) for r in ordered))
        chunks = [signatures[i:i+chunk_size] for i in range(0, len(signatures), chunk_size)]
        positions, completed, reported = {}, 0, 0
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn')) as executor:
            for chunk, result in zip(chunks, executor.map(_isolation_positions, chunks)):
                positions.update(zip(chunk, result))
                completed += len(chunk)
                if progress is not None and (completed - reported >= 500 or completed == len(signatures)):
                    progress(completed)
                    reported = completed

    def row_positions(row):
        return context_positions(row) if positions is None else positions[_position_signature(row)]

    heldout = set()
    for row in combined:
        if row['split'] in {'validation', 'test'}:
            heldout.update(row_positions(row))
    rows, rejected = [], Counter()
    for row in combined:
        if row['split'] == 'train' and row_positions(row) & heldout:
            rejected[row['stage']] += 1
            continue
        rows.append(row)
    return rows, dict(rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/move-quality-selfplay-v2')
    parser.add_argument('--output', default='data/move-planning-v1')
    parser.add_argument('--max-plies', type=int, default=6, choices=[6])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--isolation-workers', type=int, default=1,
                        help='Separate process budget for heldout/future isolation')
    args = parser.parse_args()
    source, root = Path(args.data), Path(args.output)
    if root.exists():
        raise FileExistsError('Use a fresh planning dataset output')
    proof_path = source / 'manifest.json'
    proof = json.loads(proof_path.read_text())
    inputs = [source / f'{s}.jsonl' for s in ('train', 'validation', 'test')]
    query_path = source / 'engine-queries.jsonl'
    if proof['status'] != 'complete':
        raise ValueError('Planning requires completed independent engine searches')
    for path in [*inputs, query_path]:
        if digest(path) != proof['outputs'][str(path)]['sha256']:
            raise ValueError('Planning source data changed')
    base_rows = [r for path in inputs for r in load_jsonl(path)]
    queries = load_jsonl(query_path)
    print(json.dumps({'phase': 'full_history_planning', 'source_queries': len(queries)}), flush=True)
    labels, rejected = planning_lessons(queries, {r['feature_key'] for r in base_rows}, args.max_plies,
        progress=lambda count: print(json.dumps({'planned_roots': count}), flush=True),
        workers=args.workers, chunk_size=args.chunk_size)
    print(json.dumps({'phase': 'heldout_isolation', 'labels': len(labels)}), flush=True)
    rows, isolation_rejected = isolate_planning_rows(base_rows, labels,
        workers=args.isolation_workers,
        progress=lambda count: print(json.dumps({'isolated_contexts': count}), flush=True))
    print(json.dumps({'phase': 'verify_splits', 'records': len(rows)}), flush=True)
    split_proof = verify_splits(rows)
    print(json.dumps({'phase': 'write_artifacts', 'records': len(rows)}), flush=True)
    outputs = []
    for split in ('train', 'validation', 'test'):
        path = root / f'{split}.jsonl'
        write_jsonl(path, [r for r in rows if r['split'] == split])
        outputs.append(path)
    summary = {**split_proof, 'records': len(rows),
               'by_split_stage': dict(Counter(f"{r['split']}/{r['stage']}" for r in rows)),
               'planning_lessons': sum(r['stage'] == 'move_planning' for r in rows),
               'source_context_rejections': rejected, 'heldout_future_rejections': isolation_rejected,
               'all_plans_replayed_with_full_history': True, 'continuation_after_terminal_forbidden': True,
               'neural_prose_generated': False, 'feature_cache_reusable_from_source': True,
               'test_used_for_training_or_checkpoint_selection': False}
    atomic_json(root / 'manifest.json', manifest('engine_pv_planning_curriculum', vars(args),
                [proof_path, *inputs, query_path], outputs, summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
