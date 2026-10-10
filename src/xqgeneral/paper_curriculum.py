"""Native thirteen-task questions from completed, preassigned source-root pools."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .course_tasks import PAPER_PROFILE, PAPER_TASKS, paper_records, validate_context
from .course_sampling import SAMPLING_PROFILE, make_sampler
from .curriculum_data import verify_splits
from .engine_selfplay import context_positions
from .evidence import atomic_json, digest, iter_jsonl, manifest, position_key
from .symmetry import mirror_fen, mirrored_qa

DATA_KIND = 'paper_xiangqi_four_course_questions'
SPLITS = ('train', 'validation', 'test')


def checked_sources(roots, proofs):
    if not roots or len(roots) != len(proofs):
        raise ValueError('Each supplied root file needs its completed source manifest')
    inputs = []
    for name, proof_name in zip(roots, proofs, strict=True):
        path, proof_path = Path(name), Path(proof_name)
        proof = json.loads(proof_path.read_text())
        expected = [item for source, item in proof.get('outputs', {}).items()
                    if Path(source).name == path.name]
        if (proof.get('status') != 'complete' or len(expected) != 1 or
                digest(path) != expected[0]['sha256'] or path.stat().st_size != expected[0]['bytes']):
            raise ValueError('Source root file differs from its completed producer binding')
        inputs.extend([path, proof_path])
    return inputs


def reference_contract(datasets):
    """Keep every prior game, queried future and color counterpart in its split."""
    inputs, games, positions, seen = [], {}, {}, set()
    for directory in datasets:
        data = Path(directory)
        proof_path = data / 'manifest.json'
        proof = json.loads(proof_path.read_text())
        inputs.append(proof_path)
        for split in SPLITS:
            path = data / f'{split}.jsonl'
            checked_sources([path], [proof_path])
            inputs.append(path)
            count = 0
            for row in iter_jsonl(path):
                count += 1
                game = row['game_id']
                if row['split'] != split or games.setdefault(game, split) != split:
                    raise ValueError('Prior corpus has conflicting game split ownership')
                context = (split, row['fen'], tuple(row.get('future_moves', [])),
                           tuple(tuple(line) for line in row.get('future_branches', [])))
                if context in seen:
                    continue
                seen.add(context)
                footprint = context_positions(row)
                footprint.update(position_key(mirror_fen(fen)) for fen in list(footprint))
                for key in footprint:
                    if positions.setdefault(key, split) != split:
                        raise ValueError('Prior corpus has overlapping root, future or color splits')
            if not count:
                raise ValueError('Prior reference corpus must contain every preassigned split')
    return inputs, games, positions


def question_groups(roots, seed, color_mirror=True, prior_games=None, prior_positions=None, sampler=None):
    """Deterministic questions without assigning or changing a source game's split."""
    features, games = set(), {}
    for name in roots:
        for root in iter_jsonl(name):
            validate_context(root, check_future=True)
            key, split, game = root['feature_key'], root['split'], root['game_id']
            if key in features:
                raise ValueError('Source pools contain a duplicate full-history root')
            if split not in SPLITS or games.setdefault(game, split) != split:
                raise ValueError('Source game has an invalid or conflicting preassigned split')
            if prior_games and game in prior_games and prior_games[game] != split:
                raise ValueError('New source game differs from its prior corpus split')
            footprint = context_positions(root)
            footprint.update(position_key(mirror_fen(fen)) for fen in list(footprint))
            if prior_positions and any(key in prior_positions and prior_positions[key] != split for key in footprint):
                raise ValueError('New root, maximum future or color overlaps another prior corpus split')
            features.add(key)
            rng = random.Random(int(hashlib.sha256(f'{seed}/{key}'.encode()).hexdigest(), 16))
            selected = dict(root)
            line = root.get('future_moves', [])
            if line:
                selected['future_moves'] = line[:rng.randrange(1, len(line) + 1)]
            rows = paper_records(selected, rng, sampler)
            if color_mirror:
                rows += [mirrored_qa(row) for row in list(rows)]
            if sampler is not None:
                sampler.observe(rows)
            yield root, rows


def build(roots, source_manifests, output, seed=20261071, color_mirror=True, public_evidence=None,
          reference_data=(), sampling_profile=None):
    output = Path(output)
    if output.exists() or type(color_mirror) is not bool:
        raise ValueError('Preserve previous corpora; use a fresh output and explicit mirror setting')
    sampler = make_sampler(sampling_profile)
    inputs = checked_sources(roots, source_manifests)
    reference_inputs, prior_games, prior_positions = reference_contract(reference_data)
    inputs += reference_inputs
    source_hashes = {str(p): digest(p) for p in inputs}
    output.mkdir(parents=True)
    paths = {split: output / f'{split}.jsonl' for split in SPLITS}
    handles = {split: path.open('x', encoding='utf-8') for split, path in paths.items()}
    counts, task_counts, root_counts, horizons = Counter(), Counter(), Counter(), Counter()
    ids, games = set(), set()
    try:
        for root, records in question_groups(roots, seed, color_mirror, prior_games, prior_positions, sampler):
            root_counts[root['split']] += 1
            games.add(root['game_id'])
            for row in records:
                if row['id'] in ids:
                    raise ValueError('Paper curriculum produced a duplicate question identity')
                ids.add(row['id'])
                handles[row['split']].write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[row['split']] += 1
                task_counts[f"{row['split']}/{row['stage']}/{row['task_type']}"] += 1
                if row['future_moves']:
                    horizons[len(row['future_moves'])] += 1
            if sum(root_counts.values()) % 1024 == 0:
                print(json.dumps({'native_source_roots': sum(root_counts.values()),
                                  'questions_written': len(ids)}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    if any(not counts[split] for split in SPLITS):
        raise ValueError('Paper curriculum requires nonempty preassigned train, validation and test sources')
    if any(digest(path) != source_hashes[str(path)] for path in inputs):
        raise ValueError('A root pool or completed producer manifest changed during generation')
    split_proof = verify_splits(row for path in paths.values() for row in iter_jsonl(path))
    coverage = {split: all(task_counts[f'{split}/{stage}/{task}'] > 0
                           for stage, tasks in PAPER_TASKS.items() for task in tasks) for split in SPLITS}
    result = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
              'task_profile': PAPER_PROFILE, 'counts': dict(counts), 'source_roots_by_split': dict(root_counts),
              'source_games': len(games), 'questions': len(ids), 'question_counts_by_task': dict(task_counts),
              'actual_future_horizons': dict(horizons), 'split_verification': split_proof,
              'all_declared_tasks_present_by_split': coverage,
              'all_supplied_roots_and_futures_natively_verified': True,
              'prior_corpus_isolation_verified': bool(reference_data),
              'prior_corpus_games': len(prior_games), 'prior_corpus_root_future_color_positions': len(prior_positions),
              'game_split_assigned_before_questions': True, 'native_gold_regenerated': True,
              'root_source_is_a_best_move_label': False, 'source_and_previous_corpora_modified': False,
              'global_answer_class_frequency_shaping_applied': False,
              'paper_difficult_source_mix_applied': False, 'neural_prose_generated': False,
              'feature_cache_and_token_preflight_complete': False, 'student_training_executed': False,
              'student_strength_measured': False, 'independent_test_used_for_training_or_selection': False}
    config = {'roots': list(map(str, roots)), 'source_manifests': list(map(str, source_manifests)),
              'output': str(output), 'seed': seed, 'color_mirror': color_mirror, 'task_profile': PAPER_PROFILE,
              'reference_data': list(map(str, reference_data))}
    if sampler is not None:
        result.update(sampler.verification())
        config['sampling_profile'] = sampling_profile
    atomic_json(output / 'manifest.json', manifest(DATA_KIND, config, inputs, paths.values(), result))
    if public_evidence:
        atomic_json(public_evidence, dict(result, manifest_sha256=digest(output / 'manifest.json')))
    return result


def readback(data, output):
    """Re-read source bytes, regenerate every native label and check exact output order."""
    data, output = Path(data), Path(output)
    if output.exists():
        raise ValueError('Use a fresh complete-source readback output')
    proof_path = data / 'manifest.json'
    proof = json.loads(proof_path.read_text())
    if (proof['status'] != 'complete' or proof['kind'] != DATA_KIND or
            proof['config']['task_profile'] != PAPER_PROFILE or
            Path(proof['config']['output']).resolve() != data.resolve()):
        raise ValueError('Readback requires a completed declared paper question corpus')
    for name, item in {**proof['inputs'], **proof['outputs']}.items():
        if digest(name) != item['sha256'] or Path(name).stat().st_size != item['bytes']:
            raise ValueError('Completed paper source or question bytes changed')
    config = proof['config']
    sampler = make_sampler(config.get('sampling_profile'))
    checked_sources(config['roots'], config['source_manifests'])
    _, prior_games, prior_positions = reference_contract(config['reference_data'])
    paths = {split: data / f'{split}.jsonl' for split in SPLITS}
    handles = {split: path.open(encoding='utf-8') for split, path in paths.items()}
    checked, counts = 0, Counter()
    try:
        for _, records in question_groups(config['roots'], config['seed'], config['color_mirror'],
                                         prior_games, prior_positions, sampler):
            for row in records:
                if handles[row['split']].readline() != json.dumps(row, ensure_ascii=False) + '\n':
                    raise ValueError('Question bytes differ from their complete native source root')
                checked += 1
                counts[row['split']] += 1
        if any(handle.read(1) for handle in handles.values()):
            raise ValueError('Completed question corpus has extra rows')
    finally:
        for handle in handles.values():
            handle.close()
    split_proof = verify_splits(row for path in paths.values() for row in iter_jsonl(path))
    if counts != proof['verification']['counts'] or split_proof != proof['verification']['split_verification']:
        raise ValueError('Source readback counts or complete future isolation changed')
    adaptive = sampler is not None
    if proof['verification'].get('global_answer_class_frequency_shaping_applied') is not adaptive:
        raise ValueError('Stored answer frequency shaping differs from the declared sampling profile')
    if adaptive and any(proof['verification'].get(k) != v for k, v in sampler.verification().items()):
        raise ValueError('Stored answer frequencies or distinct-query policy differ from native source readback')
    result = {'status': 'complete', 'task_profile': PAPER_PROFILE, 'questions_regenerated_and_compared': checked,
              'counts': dict(counts), 'split_verification': split_proof, 'all_source_and_question_bytes_checked': True,
              'student_model_loaded': False, 'student_training_executed': False, 'student_strength_measured': False}
    if adaptive:
        result.update(sampler.verification())
    output.mkdir(parents=True)
    atomic_json(output / 'verification.json', result)
    atomic_json(output / 'manifest.json', manifest('paper_curriculum_complete_source_readback',
        {'data': str(data), 'output': str(output)}, [proof_path, *proof['inputs'], *proof['outputs']],
        [output / 'verification.json'], result))
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--roots', nargs='+')
    group.add_argument('--readback', help='Regenerate every label of a completed paper corpus')
    parser.add_argument('--source-manifests', nargs='+')
    parser.add_argument('--reference-data', nargs='+', default=[], help='Prior complete corpora whose splits remain reserved')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261071)
    parser.add_argument('--no-color-mirror', action='store_true')
    parser.add_argument('--public-evidence')
    parser.add_argument('--sampling-profile', choices=[SAMPLING_PROFILE])
    args = parser.parse_args()
    if args.readback and args.sampling_profile is not None:
        parser.error('Readback uses its saved sampling profile; do not override it')
    result = (readback(args.readback, args.output) if args.readback else
              build(args.roots, args.source_manifests or [], args.output, args.seed,
                    not args.no_color_mirror, args.public_evidence, args.reference_data, args.sampling_profile))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
