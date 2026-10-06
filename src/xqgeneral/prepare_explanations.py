"""Build immutable engine facts for the explicitly selected explanation teacher."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from .explanations import line_facts, move_facts
from .oracle import Pikafish
from .rules import adjudicate, piece_map, piece_name, side


def select_roots(path, count, seed):
    unique = {}
    for row in load_jsonl(path):
        unique.setdefault(row['feature_key'], row)
    rows = list(unique.values())
    random.Random(seed).shuffle(rows)
    return rows[:min(count, len(rows))]


def prepare_shard(records, args, worker):
    oracle = Pikafish(args.executable, args.weights, threads=1)
    dest = Path(args.output) / 'preparation'
    dest.mkdir(parents=True, exist_ok=True)
    partial = dest / f'worker-{worker}.jsonl'
    outputs = load_jsonl(partial) if partial.exists() else []
    existing = {r['feature_key'] for r in outputs}
    try:
        for row in records:
            if row['feature_key'] in existing:
                continue
            if adjudicate(row['initial_fen'], row['moves'])['ended']:
                continue
            try:
                result = oracle.analyze(row['fen'], args.nodes, row['initial_fen'], row['moves'])
            except RuntimeError as error:
                atomic_json(dest / f"rejected-{row['id']}.json", {'record': row, 'reason': str(error),
                            'engine_output': getattr(oracle, 'last_output', [])})
                print(json.dumps({'rejected': row['id'], 'reason': str(error)}), flush=True)
                continue
            facts = {'side_to_move': side(row['fen']), 'recommended': result['best_move'],
                     'board': {square: piece_name(piece) for square, piece in sorted(piece_map(row['fen']).items())},
                     'move_facts': move_facts(row['fen'], result['best_move']),
                     'branches': [{k: c[k] for k in ['move', 'score_type', 'score', 'perspective']} |
                                  {'pv': c['pv'][:6], 'line_facts': line_facts(row['fen'], c['pv'][:6])}
                                  for c in result['candidates']]}
            query = {'id': row['id'] + '-explanation', 'feature_key': row['feature_key'],
                     'record': row, 'oracle': result, 'verified_facts': facts}
            with partial.open('a') as handle:
                handle.write(json.dumps(query, ensure_ascii=False) + '\n')
            outputs.append(query)
            print(json.dumps({'prepared': row['split'], 'id': row['id']}), flush=True)
    finally:
        oracle.close()
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/research-v1')
    parser.add_argument('--output', default='data/astra-seed-v1')
    parser.add_argument('--train-roots', type=int, default=512)
    parser.add_argument('--validation-roots', type=int, default=96)
    parser.add_argument('--test-roots', type=int, default=128)
    parser.add_argument('--nodes', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    args = parser.parse_args()
    root = Path(args.output)
    if (root / 'queries.manifest.json').exists():
        raise FileExistsError('Completed teacher inputs already exist')
    contract = {'arguments': vars(args), 'engine_sha256': digest(args.executable),
                'weights_sha256': digest(args.weights),
                'data_sha256': {s: digest(Path(args.data) / f'{s}.jsonl') for s in ['train', 'validation', 'test']}}
    contract_path = root / 'preparation.contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Preparation continuation inputs changed')
    atomic_json(contract_path, contract)
    records = []
    for split, n in [('train', args.train_roots), ('validation', args.validation_roots), ('test', args.test_roots)]:
        records += select_roots(Path(args.data) / f'{split}.jsonl', n, args.seed)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        work = [pool.submit(prepare_shard, records[i::args.workers], args, i) for i in range(args.workers)]
        rows = [r for result in work for r in result.result()]
    rows.sort(key=lambda r: (r['record']['split'], r['id']))
    outputs = []
    for split in ['train', 'validation', 'test']:
        path = root / f'{split}.queries.jsonl'
        write_jsonl(path, [r for r in rows if r['record']['split'] == split])
        outputs.append(path)
    for i in range(3):
        path = root / 'shards' / f'queries-{i}.jsonl'
        write_jsonl(path, rows[i::3])
        outputs.append(path)
    proof = {'counts': dict(Counter(r['record']['split'] for r in rows)), 'seconds': time.monotonic() - start,
             'oracle_threads_per_worker': 1, 'teacher_model': 'gpt-6-astra', 'reasoning_effort': 'low',
             'teacher_backend': 'codex_subagent_authorized_by_user', 'all_branches_legally_replayed': True,
             'rejected_engine_queries': len(list((root / 'preparation').glob('rejected-*.json')))}
    atomic_json(root / 'queries.manifest.json', manifest('initial_teacher_queries', vars(args),
                [args.executable, args.weights, *[Path(args.data) / f'{s}.jsonl' for s in ['train', 'validation', 'test']]],
                outputs, proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
