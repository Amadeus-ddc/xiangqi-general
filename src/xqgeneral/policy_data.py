"""Engine-supervised move lessons from roots and real searched continuations."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from .curriculum_data import verify_splits
from .evidence import atomic_json, digest, load_jsonl, manifest, position_key, write_jsonl
from .prepare_explanations import prepare_shard
from .rules import adjudicate, play
from .search_distillation import descend, reserved_positions
from .symmetry import mirrored_qa

QUESTION = '请推荐当前局面最好的走法。只回答一个 UCCI 走法，不写说明。'


def descendant_contexts(queries, reserved, limit, seed):
    roots = {q['feature_key'] for q in queries}
    unique = {}
    for query in queries:
        if query['record']['split'] != 'train':
            continue
        for candidate in query['oracle']['candidates']:
            row = query['record']
            for move in candidate['pv'][:5]:
                row = descend(row, move)
                if adjudicate(row['initial_fen'], row['moves'])['ended']:
                    break
                if row['feature_key'] not in roots and position_key(row['fen']) not in reserved:
                    unique.setdefault(row['feature_key'], dict(row,
                        provenance=row['provenance']+';searched_engine_pv_context', source_root=query['id']))
    rows = list(unique.values()); random.Random(seed).shuffle(rows)
    return rows[:limit]


def move_label(query):
    row = query['record']; move = query['oracle']['best_move']
    return dict(row, id=query['id']+'-best-move', stage='move_quality', task_type='best_move',
                query={}, question=QUESTION, answer=move, future_moves=[move],
                teacher={'provider':'pinned_Pikafish', 'neural_prose_generated':False},
                provenance=row['provenance']+';independently_searched_best_move')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/astra-seed-full-v1')
    parser.add_argument('--replay-data', default='data/research-balanced-v1')
    parser.add_argument('--output', default='data/move-quality-v1')
    parser.add_argument('--limit', type=int, default=8192, help='Maximum original training roots including descendants')
    parser.add_argument('--nodes', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    args = parser.parse_args()
    root = Path(args.output)
    if (root/'manifest.json').exists():
        raise FileExistsError('Completed policy dataset exists')
    query_paths = [Path(args.input)/f'{s}.queries.jsonl' for s in ('train','validation','test')]
    replay_paths = [Path(args.replay_data)/f'{s}.jsonl' for s in ('train','validation','test')]
    inputs = [*query_paths,*replay_paths,args.executable,args.weights]
    contract = {'arguments':vars(args),'input_hashes':{str(p):digest(p) for p in inputs}}
    if (root/'contract.json').exists() and json.loads((root/'contract.json').read_text()) != contract:
        raise ValueError('Policy-data continuation inputs changed')
    atomic_json(root/'contract.json',contract)
    queries = [q for p in query_paths for q in load_jsonl(p)]
    forbidden = reserved_positions(args.replay_data)
    train_roots = sum(q['record']['split']=='train' for q in queries)
    if args.limit < train_roots or args.workers < 1 or args.nodes < 1:
        raise ValueError('Invalid policy-data budget')
    descendants = descendant_contexts(queries,forbidden,args.limit-train_roots,args.seed)
    print(json.dumps({'original_train_roots':train_roots,'new_engine_contexts':len(descendants)}),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(prepare_shard,descendants[i::args.workers],args,i) for i in range(args.workers)]
        queries += [q for job in jobs for q in job.result()]
    labels, rejected = [], Counter()
    for query in queries:
        row = move_label(query)
        if row['split']=='train' and (position_key(row['fen']) in forbidden or
                                     position_key(play(row['fen'],row['answer'])) in forbidden):
            rejected['heldout_root_or_answer_position'] += 1
            continue
        labels.append(row)
        derived = mirrored_qa(row)
        derived['teacher'] = dict(row['teacher'],independent_engine_query=False,derivation='color_rank_symmetry')
        if derived['split']=='train' and (position_key(derived['fen']) in forbidden or
                                         position_key(play(derived['fen'],derived['answer'])) in forbidden):
            rejected['heldout_mirrored_position'] += 1
            continue
        labels.append(derived)
    # The original course rows provide replay and ensure every old feature context remains present.
    rows = [r for p in replay_paths for r in load_jsonl(p)] + labels
    proof = verify_splits(rows)
    outputs = []
    for split in ('train','validation','test'):
        path=root/f'{split}.jsonl';write_jsonl(path,[r for r in rows if r['split']==split]);outputs.append(path)
    path=root/'engine-queries.jsonl';write_jsonl(path,queries);outputs.append(path)
    summary = {**proof,'move_lessons':len(labels),'by_split':dict(Counter(r['split'] for r in labels)),
               'new_independently_searched_contexts':len(queries)-sum(len(load_jsonl(p)) for p in query_paths),
               'rejected':dict(rejected),'root_and_answer_positions_checked':True,
               'neural_prose_generated':False,'original_course_replay_preserved':True}
    atomic_json(root/'manifest.json',manifest('engine_move_quality_curriculum',vars(args),inputs,outputs,summary))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
