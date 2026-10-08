"""Prepare all existing training histories as search candidates, without new labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .evidence import atomic_json, digest, file_signature, iter_jsonl, manifest, write_jsonl


def training_roots(data, limit, seed):
    """Keep the first record of each history and the original seeded shuffle."""
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('A search limit must be positive, or None for all training histories')
    unique = {}
    for row in iter_jsonl(Path(data) / 'train.jsonl'):
        if row.get('split') != 'train' or not isinstance(row.get('feature_key'), str) or not row['feature_key']:
            raise ValueError('Search roots require named full histories from the training split')
        unique.setdefault(row['feature_key'], row)
    records = list(unique.values())
    random.Random(seed).shuffle(records)
    return records if limit is None else records[:limit]


def prepared_training_roots(pool, data, limit, seed, source_identity):
    """Read an unchanged, complete root pool with the original data and ordering."""
    if type(limit) is not int or limit <= 0:
        raise ValueError('Search limits must be positive')
    pool, data = Path(pool), Path(data)
    metadata = pool / 'manifest.json'
    before = file_signature(metadata)
    saved = json.loads(metadata.read_text())
    if (saved.get('kind') != 'prepared_search_training_roots' or saved.get('status') != 'complete' or
            not Path(saved['config']['data']).samefile(data) or saved['config']['seed'] != seed):
        raise ValueError('Prepared search inputs differ from the data, seed or completion contract')
    train = str(Path(saved['config']['data']) / 'train.jsonl')
    if saved['inputs'].get(train) != source_identity:
        raise ValueError('Prepared search inputs have different training source bytes')
    expected = {(pool / 'roots.jsonl').resolve(), (pool / 'counts.json').resolve()}
    if {Path(name).resolve() for name in saved['outputs']} != expected or len(saved['outputs']) != 2:
        raise ValueError('Prepared search inputs lack their exact root and count outputs')
    signatures = {}
    for path, artifact in saved['outputs'].items():
        signatures[path] = file_signature(path)
        if signatures[path][2] != artifact['bytes'] or digest(path) != artifact['sha256']:
            raise ValueError('Prepared search outputs changed')
    counts = json.loads((pool / 'counts.json').read_text())
    rows = list(iter_jsonl(pool / 'roots.jsonl'))
    if (counts != saved['verification'] or not rows or len(rows) != counts['candidate_training_histories'] or
            len({r['feature_key'] for r in rows}) != len(rows) or
            any(r.get('split') != 'train' for r in rows) or
            counts.get('original_seeded_search_shuffle_preserved') is not True or
            counts.get('first_original_record_of_each_history_preserved') is not True or
            counts.get('new_distillation_targets_generated') is not False):
        raise ValueError('Prepared search root coverage or training-only contract changed')
    if (file_signature(metadata) != before or
            any(file_signature(path) != signature for path, signature in signatures.items())):
        raise ValueError('Prepared search inputs changed while being read')
    return rows[:limit]


def prepare(data, output, seed=20261013, source_preflight=None):
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError('Preserve prepared search inputs; use a fresh output directory')
    train = data / 'train.jsonl'
    before = file_signature(train)
    identity = {'sha256': digest(train), 'bytes': train.stat().st_size}
    if file_signature(train) != before:
        raise ValueError('Training input changed while hashing')
    inputs = {str(train): identity}
    preflight_signature = None
    if source_preflight is not None:
        path = Path(source_preflight)
        preflight_signature = file_signature(path)
        raw = path.read_bytes()
        saved = json.loads(raw)
        if file_signature(path) != preflight_signature:
            raise ValueError('Source preflight changed while being read')
        if saved.get('status') != 'complete' or saved.get('inputs', {}).get(str(train)) != identity:
            raise ValueError('Training input differs from its completed source preflight')
        inputs[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    rows = training_roots(data, None, seed)
    if not rows:
        raise ValueError('No training histories are available for search')
    if file_signature(train) != before:
        raise ValueError('Training input changed while preparing search roots')
    output.mkdir(parents=True)
    roots = output / 'roots.jsonl'
    write_jsonl(roots, rows)
    by_source = Counter(row.get('recorded_source_kind', 'previous_engine_or_seeded_rules') for row in rows)
    derived = sum(bool(row.get('augmentation_parent')) for row in rows)
    proof = {'candidate_training_histories': len(rows),
        'distinct_full_history_keys': len({r['feature_key'] for r in rows}),
        'distinct_stored_game_identifiers': len({r['game_id'] for r in rows}),
        'candidate_counts_by_source_declared_kind': dict(by_source),
        'existing_color_derived_candidates': derived,
        'first_original_record_of_each_history_preserved': True,
        'original_seeded_search_shuffle_preserved': True,
        'training_file_streamed': True, 'validation_or_test_answers_read': False,
        'gpu_or_model_loaded': False, 'teacher_consolidation_executed': False,
        'new_distillation_targets_generated': False, 'search_acceptance_yield_measured': False,
        'trained_search_distillation_rounds': 0, 'student_benefit_proven': False,
        'source_preflight_identity_verified': source_preflight is not None}
    atomic_json(output / 'counts.json', proof)
    saved = manifest('prepared_search_training_roots',
        {'data': str(data), 'seed': seed, 'source_preflight': str(source_preflight) if source_preflight else None},
        [], [roots, output / 'counts.json'], proof)
    saved['inputs'] = inputs
    if (file_signature(train) != before or (source_preflight is not None and
            file_signature(source_preflight) != preflight_signature)):
        raise ValueError('Search source inputs changed before completion')
    atomic_json(output / 'manifest.json', saved)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-preflight')
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.output, args.seed, args.source_preflight), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
