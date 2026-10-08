"""Isolated search inputs from unused canonical games; never emit teaching labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from tempfile import TemporaryDirectory

from .evidence import atomic_json, digest, file_signature, history_key, iter_jsonl, manifest, position_key
from .human_games import assigned_split
from .recorded_coach import extract_footprints, footprint_records
from .symmetry import mirror_fen


KINDS = {'recorded_human_match', 'published_recorded_match',
         'recorded_computer_match', 'recorded_human_computer_match'}
ARTIFACTS = ('manifest.json', 'roots.jsonl', 'counts.json', 'heldout-positions.json')


class BoundInputs:
    def __init__(self):
        self.artifacts, self.signatures = {}, {}

    def bind(self, path, expected=None):
        path = str(path)
        before = file_signature(path)
        value = self.artifacts.get(path)
        if value is None:
            value = {'sha256': digest(path), 'bytes': before[2]}
        if (file_signature(path) != before or
                path in self.signatures and self.signatures[path] != before or
                expected is not None and value != expected):
            raise ValueError('A completed recorded search source changed')
        self.artifacts[path], self.signatures[path] = value, before
        return value

    def json(self, path, expected=None):
        self.bind(path, expected)
        value = json.loads(Path(path).read_text())
        self.unchanged()
        return value

    def unchanged(self):
        if any(file_signature(p) != s for p, s in self.signatures.items()):
            raise ValueError('Recorded search inputs changed during preparation')


def checked_game(game, split_seed):
    if (not {'game_id', 'initial_fen', 'moves', 'history', 'split', 'source_kind',
             'source', 'headers', 'native_terminal'} <= game.keys() or
            game.get('provenance') is not None and not isinstance(game['provenance'], str)):
        raise ValueError('Canonical required fields or optional provenance changed')
    identity = hashlib.sha256(json.dumps([game['initial_fen'], game['moves']],
        separators=(',', ':')).encode()).hexdigest()
    if (game['game_id'] != 'recorded-' + identity or
            game['split'] != assigned_split(game['game_id'], split_seed) or
            game['source_kind'] not in KINDS or not game['moves'] or
            len(game['history']) != len(game['moves']) + 1 or
            game['history'][0] != game['initial_fen']):
        raise ValueError('Canonical game identity, history dimensions or split changed')


def root_order(game, seed, min_ply):
    end = len(game['moves']) - int(game['native_terminal']['ended'])
    pools = [[], [], []]
    for ply in range(min_ply, end):
        pools[min(2, 3 * ply // max(1, end))].append(ply)
    rng = random.Random(int(hashlib.sha256(f"{seed}/{game['game_id']}".encode()).hexdigest(), 16))
    for pool in pools:
        rng.shuffle(pool)
    for offset in range(max(map(len, pools), default=0)):
        for pool in pools:
            if offset < len(pool):
                yield pool[offset], end


def prepare(games, data, footprints, used_inputs, heldout_data, output, *,
            seed=20261013, split_seed=20261051, per_game=6, min_ply=12, workers=1):
    if (type(per_game) is not int or per_game < 1 or type(min_ply) is not int or min_ply < 0 or
            type(workers) is not int or workers < 1 or not games):
        raise ValueError('Positive recorded search budgets and sources are required')
    output, data, footprints, used_inputs, heldout_data = map(Path,
        (output, data, footprints, used_inputs, heldout_data))
    if output.exists():
        raise FileExistsError('Preserve recorded search inputs; use a fresh output')
    bound = BoundInputs()
    base = bound.json(data / 'manifest.json')
    if base.get('status') != 'complete':
        raise ValueError('The base curriculum must be complete')
    prior = bound.json(footprints)
    if (prior.get('kind') != 'recorded_coach_footprints' or prior.get('status') != 'complete' or
            not Path(prior['config']['data']).samefile(data)):
        raise ValueError('Completed matching course footprints are required')
    for split in ('train', 'validation', 'test'):
        name = str(data / f'{split}.jsonl')
        if prior['inputs'].get(name) != base['outputs'].get(name):
            raise ValueError('Course footprint identity differs from the base data')
        signature = file_signature(name)
        if signature[2] != base['outputs'][name]['bytes']:
            raise ValueError('Completed base data size changed')
        bound.signatures[name] = signature
    prior_path = footprints.parent / 'footprints.json'
    footprint = bound.json(prior_path, prior['outputs'][str(prior_path)])
    used_games = set(footprint['combined_games'].get('train', []))
    heldout_games = set().union(*(set(footprint['combined_games'].get(s, []))
                                 for s in ('validation', 'test')))
    reserved = set().union(*(set(footprint['combined_positions'].get(s, []))
                            for s in ('validation', 'test')))
    used = bound.json(used_inputs / 'manifest.json')
    train = str(data / 'train.jsonl')
    if (used.get('status') != 'complete' or used.get('kind') != 'prepared_search_training_roots' or
            not Path(used['config']['data']).samefile(data) or
            used['inputs'].get(train) != base['outputs'].get(train)):
        raise ValueError('Completed used-root identities differ from the base training file')
    used_path = used_inputs / 'roots.jsonl'
    bound.bind(used_path, used['outputs'][str(used_path)])
    used_keys = set()
    for row in iter_jsonl(used_path):
        if row['split'] != 'train':
            raise ValueError('Used search roots must belong to training')
        used_keys.add(row['feature_key'])
        used_games.add(row.get('recorded_source_game_id', row['game_id']))
    if len(used_keys) != used['verification']['candidate_training_histories']:
        raise ValueError('Used-root coverage changed')
    labels = bound.json(heldout_data / 'manifest.json')
    if labels.get('status') != 'complete':
        raise ValueError('The protected explanation dataset must be complete')
    label_paths = [heldout_data / f'{s}.jsonl' for s in ('train', 'validation', 'test')]
    for path in label_paths:
        bound.bind(path, labels['outputs'][str(path)])
    for row in iter_jsonl(label_paths[0]):
        if row['split'] != 'train':
            raise ValueError('Explanation source file split changed')
        used_games.add(row.get('recorded_source_game_id', row['game_id']))
        used_keys.add(row['feature_key'])
    extra_games, _, extra_positions, _ = extract_footprints(
        footprint_records(label_paths[1:], Counter(), explanation=True), workers)
    for split in ('validation', 'test'):
        heldout_games.update(extra_games.get(split, []))
        reserved.update(extra_positions.get(split, []))
    # The footprint extractor returns the orientations present in source records.
    # New original-only explanation branches also reserve their color geometry.
    reserved.update(position_key(mirror_fen(p + ' - - 0 1')) for p in tuple(reserved))
    sources = []
    for directory in games:
        directory = Path(directory)
        proof = bound.json(directory / 'manifest.json')
        if (proof.get('status') != 'complete' or not proof.get('kind', '').endswith('_native_import') or
                proof['config']['seed'] != split_seed):
            raise ValueError('Completed canonical native imports with the same split seed are required')
        path = directory / 'games.jsonl'
        bound.bind(path, proof['outputs'][str(path)])
        sources.append(path)
    unique, source_counts, duplicate_games = {}, Counter(), 0
    for path in sources:
        for game in iter_jsonl(path):
            checked_game(game, split_seed)
            if game['game_id'] in unique:
                duplicate_games += 1
                continue
            unique[game['game_id']] = str(path)
            source_counts[game['split']] += 1
            if game['split'] != 'train':
                heldout_games.add(game['game_id'])
                for fen in game['history']:
                    reserved.add(position_key(fen))
                    reserved.add(position_key(mirror_fen(fen)))
    bound.unchanged()
    output.mkdir(parents=True)
    atomic_json(output / 'state.json', {'status': 'extracting_unused_training_game_roots',
        'unique_source_games': len(unique), 'reserved_positions': len(reserved), 'gpu_model_loaded': False})
    excluded, accepted_kind, accepted_side, accepted_phase = Counter(), Counter(), Counter(), Counter()
    per_source, seen, accepted_games = Counter(), set(), set()
    total = 0
    with TemporaryDirectory(prefix='ordering-', dir=output) as scratch:
        database = sqlite3.connect(str(Path(scratch) / 'roots.sqlite'))
        try:
            database.execute('CREATE TABLE roots (key TEXT PRIMARY KEY, priority TEXT NOT NULL, record TEXT NOT NULL)')
            for path in sources:
                for game in iter_jsonl(path):
                    identity = game['game_id']
                    if identity in seen:
                        continue
                    seen.add(identity)
                    if game['split'] != 'train':
                        continue
                    if identity in used_games or identity in heldout_games:
                        excluded['already_used_or_reserved_game'] += 1
                        continue
                    accepted = 0
                    for ply, end in root_order(game, seed, min_ply):
                        future = game['moves'][ply:min(ply + 8, end)]
                        history = game['history'][:ply + 1]
                        key = history_key(history)
                        if key in used_keys:
                            excluded['already_used_full_history'] += 1
                            continue
                        future_fens = game['history'][ply:ply + len(future) + 1]
                        footprint = {position_key(f) for f in future_fens}
                        footprint.update(position_key(mirror_fen(f)) for f in future_fens)
                        if footprint & reserved:
                            excluded['heldout_root_or_recorded_future'] += 1
                            continue
                        row = {'id': 'recorded-search-' + key, 'game_id': identity, 'split': 'train',
                            'initial_fen': game['initial_fen'], 'moves': game['moves'][:ply],
                            'history': history, 'fen': history[-1], 'feature_key': key, 'ply': ply,
                            'future_moves': future, 'future_branches': [],
                            'recorded_source_game_id': identity, 'recorded_source_kind': game['source_kind'],
                            'recorded_source': game['source'], 'recorded_source_headers': game['headers'],
                            'recorded_source_provenance': game.get('provenance'),
                            'provenance': ((game.get('provenance') + ';') if game.get('provenance') else '') +
                                          'unused_recorded_search_input',
                            'recorded_continuation_is_best_move_label': False}
                        priority = hashlib.sha256(f'{seed}/{key}'.encode()).hexdigest()
                        cursor = database.execute('INSERT OR IGNORE INTO roots VALUES (?, ?, ?)',
                            (key, priority, json.dumps(row, ensure_ascii=False)))
                        if cursor.rowcount == 0:
                            excluded['duplicate_full_history'] += 1
                            continue
                        total += 1
                        accepted += 1
                        accepted_games.add(identity)
                        accepted_kind[game['source_kind']] += 1
                        accepted_side['red' if row['fen'].split()[1] == 'w' else 'black'] += 1
                        accepted_phase[str(min(2, 3 * ply // max(1, end)))] += 1
                        per_source[str(path)] += 1
                        if accepted == per_game:
                            break
                    if len(seen) % 2000 == 0:
                        database.commit()
                        atomic_json(output / 'state.json', {'status': 'extracting_unused_training_game_roots',
                            'source_games_seen': len(seen), 'candidate_roots': total, 'gpu_model_loaded': False})
            if total == 0:
                raise ValueError('No isolated unused recorded training roots remain')
            database.commit()
            with (output / 'roots.jsonl').open('x', encoding='utf-8') as handle:
                for (record,) in database.execute('SELECT record FROM roots ORDER BY priority, key'):
                    handle.write(record + '\n')
        finally:
            database.close()
    counts = {'candidate_training_histories': total, 'distinct_original_training_games': len(accepted_games),
        'unique_canonical_source_games': len(unique), 'canonical_game_splits': dict(source_counts),
        'duplicate_source_games': duplicate_games, 'candidate_counts_by_source_declared_kind': dict(accepted_kind),
        'candidate_counts_by_canonical_file': dict(per_source), 'candidate_counts_by_side': dict(accepted_side),
        'candidate_counts_by_game_phase': dict(accepted_phase), 'excluded': dict(excluded),
        'reserved_game_identifiers': len(heldout_games), 'reserved_root_and_future_positions': len(reserved),
        'existing_used_full_history_keys': len(used_keys), 'existing_used_game_identifiers': len(used_games),
        'new_color_derived_candidates': 0, 'at_most_original_roots_per_game': per_game,
        'complete_canonical_native_import_reused': True, 'every_history_natively_replayed_again': False,
        'all_canonical_heldout_histories_and_color_positions_reserved': True,
        'all_existing_course_and_explanation_heldout_futures_reserved': True,
        'heldout_moves_and_structured_answers_used_for_isolation_only': True,
        'teacher_inference_executed': False, 'gpu_or_model_loaded': False,
        'new_distillation_targets_generated': False, 'student_benefit_proven': False}
    atomic_json(output / 'counts.json', counts)
    atomic_json(output / 'heldout-positions.json', sorted(reserved))
    config = {'games': list(map(str, games)), 'data': str(data), 'footprints': str(footprints),
        'used_inputs': str(used_inputs), 'heldout_data': str(heldout_data), 'seed': seed,
        'split_seed': split_seed, 'per_game': per_game, 'min_ply': min_ply, 'workers': workers,
        'base_data_identities_reused_from_completed_footprints': {
            str(data / f'{s}.jsonl'): base['outputs'][str(data / f'{s}.jsonl')]
            for s in ('train', 'validation', 'test')}}
    saved = manifest('prepared_unused_recorded_search_roots', config, [],
        [output / name for name in ARTIFACTS[1:]], counts)
    saved['inputs'] = bound.artifacts
    bound.unchanged()
    atomic_json(output / 'manifest.json', saved)
    atomic_json(output / 'state.json', {'status': 'complete', **counts})
    return counts


def prepared_roots(pool, data, limit, seed, base_identities):
    """Validate a complete unused-game pool before any student model is loaded."""
    if type(limit) is not int or limit < 1:
        raise ValueError('Recorded search limits must be positive')
    pool, data = Path(pool), Path(data)
    bound = BoundInputs()
    saved = bound.json(pool / 'manifest.json')
    if (saved.get('kind') != 'prepared_unused_recorded_search_roots' or
            saved.get('status') != 'complete' or not Path(saved['config']['data']).samefile(data) or
            saved['config']['seed'] != seed or
            saved['config']['base_data_identities_reused_from_completed_footprints'] != base_identities):
        raise ValueError('Recorded pool data, seed or completion contract differs')
    expected = {(pool / name).resolve() for name in ARTIFACTS[1:]}
    if {Path(p).resolve() for p in saved['outputs']} != expected or len(saved['outputs']) != 3:
        raise ValueError('The recorded pool must contain its exact roots, counts and reserved positions')
    for group in ('inputs', 'outputs'):
        for path, identity in saved[group].items():
            bound.bind(path, identity)
    counts = bound.json(pool / 'counts.json')
    if (counts != saved['verification'] or counts.get('new_color_derived_candidates') != 0 or
            counts.get('new_distillation_targets_generated') is not False or
            counts.get('all_canonical_heldout_histories_and_color_positions_reserved') is not True or
            counts.get('all_existing_course_and_explanation_heldout_futures_reserved') is not True):
        raise ValueError('Recorded pool counts or isolation contract differs')
    selected, keys, games, kinds, previous = [], set(), Counter(), Counter(), None
    for row in iter_jsonl(pool / 'roots.jsonl'):
        key = row['feature_key']
        ordered = hashlib.sha256(f'{seed}/{key}'.encode()).hexdigest(), key
        if (row.get('split') != 'train' or row.get('augmentation_parent') or key in keys or
                row.get('id') != 'recorded-search-' + key or
                row['game_id'] != row['recorded_source_game_id'] or
                assigned_split(row['game_id'], saved['config']['split_seed']) != 'train' or
                row['recorded_source_kind'] not in KINDS or
                row.get('recorded_continuation_is_best_move_label') is not False or
                'answer' in row or previous is not None and ordered <= previous):
            raise ValueError('Recorded roots changed their original training ownership or order')
        keys.add(key)
        games[row['game_id']] += 1
        kinds[row['recorded_source_kind']] += 1
        previous = ordered
        if len(selected) < limit:
            if history_key(row['history']) != key:
                raise ValueError('Selected recorded roots differ from their full-history keys')
            selected.append(row)
    positions = bound.json(pool / 'heldout-positions.json')
    if (not isinstance(positions, list) or any(not isinstance(p, str) or len(p.split()) != 2 for p in positions) or
            not selected or len(keys) != counts['candidate_training_histories'] or
            len(games) != counts['distinct_original_training_games'] or
            max(games.values()) > saved['config']['per_game'] or
            dict(kinds) != counts['candidate_counts_by_source_declared_kind'] or
            len(set(positions)) != len(positions) or
            len(positions) != counts['reserved_root_and_future_positions']):
        raise ValueError('Recorded pool coverage or reserved positions changed')
    bound.unchanged()
    return selected, set(positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', required=True, nargs='+')
    parser.add_argument('--data', required=True)
    parser.add_argument('--footprints', required=True)
    parser.add_argument('--used-inputs', required=True)
    parser.add_argument('--heldout-data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--split-seed', type=int, default=20261051)
    parser.add_argument('--per-game', type=int, default=6)
    parser.add_argument('--min-ply', type=int, default=12)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(prepare(args.games, args.data, args.footprints, args.used_inputs,
        args.heldout_data, args.output, seed=args.seed, split_seed=args.split_seed,
        per_game=args.per_game, min_ply=args.min_ply, workers=args.workers)), flush=True)


if __name__ == '__main__':
    main()
