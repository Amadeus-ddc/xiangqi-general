"""Evaluate raw planning answers against full-history rules and an independent engine."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import time

from .evaluate_explanations import judge_move
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from .oracle import Pikafish
from .rules import adjudicate, play


def parsed_line(raw):
    moves = raw.strip().split()
    return moves if 1 <= len(moves) <= 6 and all(re.fullmatch(r'[a-i][0-9][a-i][0-9]', m) for m in moves) else None


def judge_plan(oracle, record, raw, nodes):
    moves = parsed_line(raw)
    result = {'format_valid': moves is not None, 'full_line_legal': False,
              'conditional_first_move_matches': False, 'length_sufficient': False,
              'contract_valid': False, 'errors': [], 'plies': []}
    if moves is None:
        result['errors'].append('invalid_line_format')
        return result
    forced = record.get('query', {}).get('move')
    result['conditional_first_move_matches'] = forced is None or moves[0] == forced
    if not result['conditional_first_move_matches']:
        result['errors'].append('conditional_first_move_mismatch')
    current = dict(record, moves=list(record['moves']))
    for index, move in enumerate(moves):
        verdict = judge_move(oracle, current, move, nodes)
        result['plies'].append({'move': move, 'exclude_from_quality': forced is not None and index == 0,
                                **verdict})
        if not verdict['first_move_legal']:
            result['errors'].append('move_after_terminal' if 'terminal_root' in verdict else 'illegal_move')
            return result
        current['fen'] = play(current['fen'], move)
        current['moves'] = [*current['moves'], move]
    result['full_line_legal'] = True
    ended = adjudicate(current['initial_fen'], current['moves'])['ended']
    result['length_sufficient'] = len(moves) >= min(6, len(record['answer'].split())) or ended
    if not result['length_sufficient']:
        result['errors'].append('shorter_than_reference_horizon')
    result['contract_valid'] = not result['errors']
    return result


def plan_summary(rows):
    if not rows:
        raise ValueError('No planning answers were evaluated')
    verdicts = [r['judgment'] for r in rows]
    plies = [p for v in verdicts for p in v['plies'] if not p['exclude_from_quality']]
    losses = [p['first_move_expected_score_loss'] for p in plies if 'first_move_expected_score_loss' in p]
    return {'examples': len(rows),
            **{key + '_rate': sum(v[key] for v in verdicts) / len(rows)
               for key in ('format_valid', 'full_line_legal', 'conditional_first_move_matches',
                           'length_sufficient', 'contract_valid')},
            'line_lengths': dict(Counter(len(parsed_line(r['raw']) or []) for r in rows)),
            'errors': dict(Counter(e for v in verdicts for e in v['errors'])),
            'quality_plies_attempted': len(plies), 'legal_scored_quality_plies': len(losses),
            'quality_ply_no_mistake_rate_all_attempted':
                sum(p['first_move_no_mistake'] for p in plies) / len(plies) if plies else None,
            'mean_expected_score_loss_conditional': sum(losses) / len(losses) if losses else None,
            'teacher_line_exact_rate': sum(r['raw'].strip() == r['record']['answer'] for r in rows) / len(rows),
            'forced_root_excluded_from_quality': True, 'full_history_termination_checked': True,
            'raw_generation': True, 'oracle_repairs': 0}


def judge_shard(rows, args):
    oracle = Pikafish(args.executable, args.weights, threads=1)
    try:
        return [dict(r, judgment=judge_plan(oracle, r['record'], r['raw'], args.nodes)) for r in rows]
    finally:
        oracle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/move-planning-v1')
    parser.add_argument('--features', default='data/move-quality-selfplay-v2/features-16.pt')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--limit', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--nodes', type=int, default=1000000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh planning evaluation output')
    if min(args.limit, args.batch_size, args.max_new_tokens, args.nodes, args.workers) < 1:
        raise ValueError('Planning evaluation budgets must be positive')
    path = Path(args.data) / f'{args.split}.jsonl'
    rows = [r for r in load_jsonl(path) if r['stage'] == 'move_planning']
    if not rows or any(r['split'] != args.split for r in rows):
        raise ValueError('Planning evaluation split differs')
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    from .inference import Predictor
    started = time.monotonic()
    predictor = Predictor(args.checkpoint, feature_cache=args.features)
    raw = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        answers = predictor.generate_batch(batch, args.max_new_tokens)
        raw.extend({'id': r['id'], 'record': r, 'raw': a} for r, a in zip(batch, answers))
        print(json.dumps({'generated': len(raw), 'requested': len(rows)}), flush=True)
    write_jsonl(dest / 'raw-predictions.jsonl', raw)
    del predictor
    import torch
    torch.cuda.empty_cache()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(judge_shard, raw[i::args.workers], args) for i in range(args.workers)]
        judged = [r for job in jobs for r in job.result()]
    judged.sort(key=lambda r: r['id'])
    write_jsonl(dest / 'judged-predictions.jsonl', judged)
    proof = {**plan_summary(judged), 'split': args.split, 'judge_nodes_per_position': args.nodes,
             'engine_sha256': digest(args.executable), 'engine_weights_sha256': digest(args.weights),
             'seconds': time.monotonic() - started}
    atomic_json(dest / 'metrics.json', proof)
    atomic_json(dest / 'manifest.json', manifest('independent_raw_planning_evaluation', vars(args),
                [args.checkpoint, path, args.features, args.executable, args.weights],
                [dest / 'raw-predictions.jsonl', dest / 'judged-predictions.jsonl', dest / 'metrics.json'], proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
