"""CPU scoring of preserved raw explanations, without loading or querying a model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from .evaluate_explanations import explanation_summary, judgment
from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest, write_jsonl
from .explanations import EXPLANATION_QUESTION
from .oracle import Pikafish
from .recorded_search_inputs import BoundInputs
from .verified_inputs import input_session_from_environment

KIND = 'preserved_raw_explanation_generation'


def checked_generation(receipt_path, guard):
    receipt = guard.json(receipt_path)
    if (receipt.get('kind') != KIND or receipt.get('status') != 'complete_raw_answers_only' or
            not isinstance(receipt.get('origin_artifacts'), dict) or not receipt['origin_artifacts'] or
            not isinstance(receipt.get('generation_code'), dict)):
        raise ValueError('Recovery requires an explicit receipt for the preserved raw generation')
    config = receipt['generation_config']
    if (config['split'] != 'validation' or config['memory'] not in {'normal', 'zero', 'shuffled'} or
            any(type(config[k]) is not int or config[k] <= 0
                for k in ('limit', 'batch_size', 'max_new_tokens', 'nodes', 'workers')) or
            type(config['seed']) is not int):
        raise ValueError('Preserve the original validation and positive generation/scoring budgets')
    for path, binding in receipt['origin_artifacts'].items():
        guard.bind(path, binding)
    validation = Path(config['data']) / 'validation.jsonl'
    if receipt['validation']['path'] != str(validation):
        raise ValueError('The original validation file must match the generation configuration')
    guard.bind(validation, {k: receipt['validation'][k] for k in ('sha256', 'bytes')})
    raw = receipt['raw_predictions']
    guard.bind(raw['path'], {k: raw[k] for k in ('sha256', 'bytes')})
    rows = load_jsonl(validation)
    random.Random(config['seed']).shuffle(rows)
    rows = rows[:config['limit']]
    expected = [{'id': row['id'], 'record': dict(row, question=EXPLANATION_QUESTION)} for row in rows]
    predictions = load_jsonl(raw['path'])
    if (not expected or len(expected) != config['limit'] or
            len({row['id'] for row in expected}) != len(expected) or
            any(row['record']['split'] != 'validation' for row in expected) or
            len(predictions) != len(expected) or
            any(set(row) != {'id', 'record', 'raw'} or not isinstance(row['raw'], str) or
                {k: row[k] for k in ('id', 'record')} != target
                for row, target in zip(predictions, expected, strict=True))):
        raise ValueError('Keep every original raw answer and exact ordered validation question')
    guard.unchanged()
    return receipt, predictions


def score_shard(rows, config, destination, index):
    oracle = Pikafish(config['executable'], config['weights'], threads=1)
    path = destination / f'judged-shard-{index}.jsonl'
    scored = []
    try:
        with path.open('x') as stream:
            for row in rows:
                value = dict(row, judgment=judgment(oracle, row['record'], row['raw'], config['nodes']))
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                scored.append(value)
        return scored
    finally:
        oracle.close()


def run_scoring(receipt_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous scoring attempts; choose a fresh output')
    guard = BoundInputs()
    receipt, raw = checked_generation(receipt_path, guard)
    config, code = receipt['generation_config'], code_identity()
    for path, binding in receipt['engine_artifacts'].items():
        guard.bind(path, binding)
    if set(receipt['engine_artifacts']) != {config['executable'], config['weights']}:
        raise ValueError('Score with the original explicit engine and weight identities')
    output.mkdir(parents=True)
    contract = {'receipt': str(receipt_path), 'generation_config': config, 'code': code}
    atomic_json(output / 'contract.json', contract)
    # A byte-exact copy keeps the failed attempt immutable and independently usable.
    (output / 'raw-predictions.jsonl').write_bytes(Path(receipt['raw_predictions']['path']).read_bytes())
    if digest(output / 'raw-predictions.jsonl') != receipt['raw_predictions']['sha256']:
        raise ValueError('Preserved raw answers changed during recovery')
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        jobs = [pool.submit(score_shard, raw[i::config['workers']], config, output, i)
                for i in range(config['workers'])]
        judged = [row for job in jobs for row in job.result()]
    judged.sort(key=lambda row: row['id'])
    write_jsonl(output / 'judged-predictions.jsonl', judged)
    guard.unchanged()
    if code_identity() != code:
        raise ValueError('Scoring source changed; preserve the attempted outputs')
    metrics = {**explanation_summary(judged), 'split': 'validation', 'memory': config['memory'],
               'judge_nodes_per_position': config['nodes'], 'engine_sha256': digest(config['executable']),
               'engine_weights_sha256': digest(config['weights']), 'new_model_or_teacher_queries': 0,
               'generation_input_bytes_rehashed_by_scoring': False,
               'saved_raw_generation_receipt_sha256': digest(receipt_path),
               'original_generation_code': receipt['generation_code']}
    atomic_json(output / 'metrics.json', metrics)
    proof = manifest('recovered_independent_explanation_evaluation', contract, (),
                     [output / name for name in ('raw-predictions.jsonl', 'judged-predictions.jsonl', 'metrics.json')],
                     metrics, code=code)
    proof['inputs'] = guard.artifacts
    guard.unchanged()
    atomic_json(output / 'manifest.json', proof)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with input_session_from_environment():
        result = run_scoring(args.receipt, args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
