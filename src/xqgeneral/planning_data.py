"""Engine PV planning lessons with full-history termination and heldout isolation."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

from .curriculum_data import verify_splits
from .engine_selfplay import context_positions
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from .rules import adjudicate, play
from .symmetry import mirrored_qa

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


def planning_lessons(queries, available_keys, max_plies=6, progress=None):
    labels, rejected = [], Counter()
    for index, query in enumerate(queries):
        candidates = query['oracle']['candidates']
        best = next(c for c in candidates if c['move'] == query['oracle']['best_move'])
        for candidate, forced in [(best, False), *[(c, True) for c in candidates]]:
            row = planning_label(query, candidate, forced, max_plies)
            if row is None:
                rejected['terminal_root'] += 1
                continue
            for item in (row, mirrored_qa(row)):
                if item['feature_key'] not in available_keys:
                    rejected['root_not_in_verified_policy_data'] += 1
                    continue
                if 'augmentation_parent' in item:
                    item['teacher'] = dict(item['teacher'], independent_engine_query=False,
                                           derivation='color_rank_symmetry')
                labels.append(item)
        if progress is not None and (index + 1) % 500 == 0:
            progress(index + 1)
    return labels, dict(rejected)


def isolate_planning_rows(base_rows, labels):
    heldout = set()
    for row in [*base_rows, *labels]:
        if row['split'] in {'validation', 'test'}:
            heldout.update(context_positions(row))
    rows, rejected = [], Counter()
    for row in [*base_rows, *labels]:
        if row['split'] == 'train' and context_positions(row) & heldout:
            rejected[row['stage']] += 1
            continue
        rows.append(row)
    return rows, dict(rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/move-quality-selfplay-v2')
    parser.add_argument('--output', default='data/move-planning-v1')
    parser.add_argument('--max-plies', type=int, default=6, choices=[6])
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
                                        progress=lambda count: print(json.dumps({'planned_roots': count}), flush=True))
    rows, isolation_rejected = isolate_planning_rows(base_rows, labels)
    split_proof = verify_splits(rows)
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
