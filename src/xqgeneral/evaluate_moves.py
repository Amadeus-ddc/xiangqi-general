"""Raw move-course inference and independent engine judgments without repairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import time

from .evaluate_explanations import judge_move
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from .oracle import Pikafish


def parsed_move(raw):
    value = raw.strip()
    return value if re.fullmatch(r'[a-i][0-9][a-i][0-9]', value) else None


def move_summary(rows):
    if not rows:
        raise ValueError('No raw move predictions were evaluated')
    losses = [r['judgment']['first_move_expected_score_loss'] for r in rows
              if 'first_move_expected_score_loss' in r['judgment']]
    return {'examples': len(rows), 'format_valid_rate': sum(r['move'] is not None for r in rows)/len(rows),
            'legal_rate': sum(r['judgment']['first_move_legal'] for r in rows)/len(rows),
            'no_mistake_rate': sum(r['judgment']['first_move_no_mistake'] for r in rows)/len(rows),
            'teacher_best_move_exact_rate': sum(r['move'] == r['record']['answer'] for r in rows)/len(rows),
            'legal_scored_moves': len(losses),
            'mean_expected_score_loss_conditional': sum(losses)/len(losses) if losses else None,
            'terminal_roots': sum('terminal_root' in r['judgment'] for r in rows),
            'original_examples': sum(not r['record']['teacher'].get('derivation') for r in rows),
            'color_derived_examples': sum(bool(r['record']['teacher'].get('derivation')) for r in rows),
            'raw_generation': True, 'oracle_repairs': 0, 'score_model': 'pinned_Pikafish_material_WDL'}


def judge_shard(rows, args):
    oracle = Pikafish(args.executable, args.weights, threads=1)
    try:
        return [dict(r, judgment=judge_move(oracle, r['record'], r['move'], args.nodes)) for r in rows]
    finally:
        oracle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/move-quality-v1')
    parser.add_argument('--features', default='data/move-quality-v1/features-16.pt')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--limit', type=int, default=192)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--memory', choices=['normal', 'zero', 'shuffled'], default='normal')
    parser.add_argument('--decoding', choices=['raw', 'legal'], default='raw')
    parser.add_argument('--beams', type=int, default=4)
    parser.add_argument('--nodes', type=int, default=1000000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh raw move evaluation output')
    if min(args.limit, args.batch_size, args.nodes, args.workers, args.beams) < 1:
        raise ValueError('Move evaluation budgets must be positive')
    data = Path(args.data)/f'{args.split}.jsonl'
    rows = [r for r in load_jsonl(data) if r['stage'] == 'move_quality']
    if not rows or any(r['split'] != args.split for r in rows):
        raise ValueError('Move evaluation split contract differs')
    random.Random(args.seed).shuffle(rows); rows = rows[:args.limit]
    from .inference import Predictor
    started = time.monotonic()
    predictor = Predictor(args.checkpoint, feature_cache=args.features)
    raw = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset+args.batch_size]
        if args.memory == 'shuffled' and len(batch) == 1:
            raise ValueError('A singleton batch cannot shuffle expert memory')
        answers = (predictor.generate_moves(batch, args.memory, args.beams) if args.decoding == 'legal' else
                   predictor.generate_batch(batch, args.max_new_tokens, args.memory))
        raw.extend({'id': r['id'], 'record': r, 'raw': a, 'move': parsed_move(a)} for r,a in zip(batch,answers))
        print(json.dumps({'generated': len(raw), 'requested': len(rows)}), flush=True)
    dest.mkdir(parents=True)
    write_jsonl(dest/'raw-predictions.jsonl', raw)
    del predictor
    import torch
    torch.cuda.empty_cache()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(judge_shard, raw[i::args.workers], args) for i in range(args.workers)]
        judged = [r for job in jobs for r in job.result()]
    judged.sort(key=lambda r:r['id'])
    write_jsonl(dest/'judged-predictions.jsonl', judged)
    proof = {**move_summary(judged), 'split': args.split, 'memory': args.memory,
             'decoding': args.decoding, 'raw_generation': args.decoding == 'raw',
             'rule_legal_constraints': args.decoding == 'legal',
             'legality_is_imposed_by_decoding': args.decoding == 'legal',
             'judge_nodes_per_position': args.nodes, 'engine_sha256': digest(args.executable),
             'engine_weights_sha256': digest(args.weights), 'seconds': time.monotonic()-started}
    atomic_json(dest/'metrics.json', proof)
    atomic_json(dest/'manifest.json', manifest('independent_move_evaluation',vars(args),
                [args.checkpoint,data,args.features,args.executable,args.weights],
                [dest/'raw-predictions.jsonl',dest/'judged-predictions.jsonl',dest/'metrics.json'],proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
