"""Independent engine judgments of raw model explanations, without repairs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time

from .calibration import candidate_probability, evaluation_probability
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from .explanations import EXPLANATION_QUESTION, parse_explanation, validate_explanation
from .oracle import Pikafish
from .rules import adjudicate, legal_moves, play


def judge_move(oracle, record, move, nodes):
    """Score a raw root move with a separate engine and the complete history."""
    result = {'first_move_legal': False, 'first_move_no_mistake': False}
    outcome = adjudicate(record['initial_fen'], record['moves'])
    if outcome['ended']:
        return dict(result, terminal_root=outcome)
    if move not in legal_moves(record['fen']):
        return result
    result['first_move_legal'] = True
    engine = oracle.analyze(record['fen'], nodes, record['initial_fen'], record['moves'])
    best = next(c for c in engine['candidates'] if c['move'] == engine['best_move'])
    chosen = next((c for c in engine['candidates'] if c['move'] == move), None)
    if chosen is None:
        restricted = oracle.analyze(record['fen'], nodes, record['initial_fen'], record['moves'],
                                     searchmoves=[move])
        chosen = next(c for c in restricted['candidates'] if c['move'] == move)
        result['first_move_restricted_oracle'] = restricted
    loss = max(0, candidate_probability(record['fen'], best) - candidate_probability(record['fen'], chosen))
    result.update(first_move_expected_score_loss=loss, first_move_no_mistake=loss < .10,
                  first_move_oracle=engine)
    return result


def judgment(oracle, record, raw, nodes):
    try:
        analysis = parse_explanation(raw)
    except (ValueError, TypeError):
        return {'parse_valid': False, 'first_move_legal': False, 'contract_valid': False,
                'errors': ['invalid_json'], 'first_move_no_mistake': False, 'prose_semantic_rating': 'unmeasured'}
    proof = validate_explanation(record['fen'], analysis, require_branches=True)
    result = {'parse_valid': True, 'contract_valid': proof['valid'], 'errors': proof['errors'],
              'pv_move_losses': [], 'prose_semantic_rating': 'unmeasured',
              **judge_move(oracle, record, analysis.get('move'), nodes)}
    if 'terminal_root' in result:
        result.update(contract_valid=False, pv_legal=False)
        result['errors'].append('move_after_terminal_root')
        return result
    root_engine = result.get('first_move_oracle')
    if result['first_move_legal']:
        best = next(c for c in root_engine['candidates'] if c['move'] == root_engine['best_move'])
        try:
            result['evaluation_probability_error'] = abs(evaluation_probability(record['fen'], analysis.get('evaluation')) -
                                                          candidate_probability(record['fen'], best))
        except (ValueError, TypeError, AttributeError):
            pass
    pv = analysis.get('pv')
    if not isinstance(pv, list) or not pv or len(pv) > 32:
        result['pv_legal'] = False
        return result
    position, moves = record['fen'], list(record['moves'])
    engine_judgments = []
    for i, move in enumerate(pv):
        if adjudicate(record['initial_fen'], moves)['ended']:
            result['pv_legal'] = False
            result['errors'].append('pv_continues_after_game_end')
            break
        if not isinstance(move, str) or move not in legal_moves(position):
            result['pv_legal'] = False
            break
        engine = root_engine if i == 0 and root_engine is not None else oracle.analyze(position, nodes, record['initial_fen'], moves)
        best = next(c for c in engine['candidates'] if c['move'] == engine['best_move'])
        candidate = next((c for c in engine['candidates'] if c['move'] == move), None)
        restricted = None
        if candidate is None:
            restricted = oracle.analyze(position, nodes, record['initial_fen'], moves, searchmoves=[move])
            candidate = next(c for c in restricted['candidates'] if c['move'] == move)
        loss = max(0, candidate_probability(position, best) - candidate_probability(position, candidate))
        result['pv_move_losses'].append(loss)
        engine_judgments.append({'fen': position, 'move': move, 'oracle': engine, 'restricted': restricted})
        position = play(position, move); moves.append(move)
    else:
        result['pv_legal'] = True
    result['engine_judgments'] = engine_judgments
    return result


def explanation_summary(rows):
    if not rows:
        raise ValueError('No raw explanation predictions were evaluated')
    verdicts = [r['judgment'] for r in rows]
    move_losses = [v['first_move_expected_score_loss'] for v in verdicts if 'first_move_expected_score_loss' in v]
    pv_losses = [loss for v in verdicts for loss in v.get('pv_move_losses', [])]
    errors = Counter(e for v in verdicts for e in v.get('errors', []))
    return {'examples': len(rows), 'parse_valid_rate': sum(v['parse_valid'] for v in verdicts) / len(rows),
            'first_move_legal_rate': sum(v['first_move_legal'] for v in verdicts) / len(rows),
            'structured_contract_valid_rate': sum(v['contract_valid'] for v in verdicts) / len(rows),
            'first_move_no_mistake_rate': sum(v['first_move_no_mistake'] for v in verdicts) / len(rows),
            'legal_scored_first_moves': len(move_losses),
            'mean_first_move_expected_score_loss_conditional': sum(move_losses)/len(move_losses) if move_losses else None,
            'pv_legal_rate': sum(v.get('pv_legal', False) for v in verdicts) / len(rows),
            'judged_legal_pv_plies': len(pv_losses),
            'pv_no_mistake_fraction_conditional': sum(v < .10 for v in pv_losses)/len(pv_losses) if pv_losses else None,
            'errors': dict(errors), 'raw_generation': True, 'oracle_repairs': 0,
            'score_model': 'pinned_Pikafish_material_WDL', 'prose_semantic_rating': 'unmeasured'}


def judge_shard(rows, args):
    oracle = Pikafish(args.executable, args.weights, threads=1)
    try:
        return [dict(row, judgment=judgment(oracle, row['record'], row['raw'], args.nodes)) for row in rows]
    finally:
        oracle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/astra-explanations-v1')
    parser.add_argument('--features', default='data/research-balanced-v1/features-16.pt')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=768)
    parser.add_argument('--nodes', type=int, default=1000000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh raw explanation evaluation output')
    from .inference import Predictor
    rows = load_jsonl(Path(args.data) / f'{args.split}.jsonl')
    random.Random(args.seed).shuffle(rows); rows = rows[:args.limit]
    if not rows or any(r['split'] != args.split for r in rows):
        raise ValueError('Evaluation split contract differs')
    started = time.monotonic()
    predictor = Predictor(args.checkpoint, feature_cache=args.features)
    raw = []
    for offset in range(0, len(rows), args.batch_size):
        batch = [dict(r, question=EXPLANATION_QUESTION) for r in rows[offset:offset + args.batch_size]]
        answers = predictor.generate_batch(batch, args.max_new_tokens)
        raw.extend({'id': r['id'], 'record': r, 'raw': a} for r, a in zip(batch, answers))
        print(json.dumps({'generated': len(raw), 'requested': len(rows)}), flush=True)
    dest.mkdir(parents=True)
    write_jsonl(dest / 'raw-predictions.jsonl', raw)
    # Release the inference model before the independent CPU judgments.
    del predictor
    import torch
    torch.cuda.empty_cache()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(judge_shard, raw[i::args.workers], args) for i in range(args.workers)]
        judged = [r for job in jobs for r in job.result()]
    judged.sort(key=lambda r: r['id'])
    write_jsonl(dest / 'judged-predictions.jsonl', judged)
    proof = {**explanation_summary(judged), 'split': args.split, 'judge_nodes_per_position': args.nodes,
             'engine_sha256': digest(args.executable), 'engine_weights_sha256': digest(args.weights),
             'seconds': time.monotonic() - started}
    atomic_json(dest / 'metrics.json', proof)
    atomic_json(dest / 'manifest.json', manifest('independent_explanation_evaluation', vars(args),
                [args.checkpoint, Path(args.data)/f'{args.split}.jsonl', args.features, args.executable, args.weights],
                [dest/'raw-predictions.jsonl', dest/'judged-predictions.jsonl', dest/'metrics.json'], proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
