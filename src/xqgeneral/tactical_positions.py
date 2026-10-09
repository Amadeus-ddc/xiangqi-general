"""Native import of pinned position-only JSON; source move text is never a label."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import pyffish

from .evidence import atomic_json, iter_jsonl, manifest, position_key, write_jsonl
from .human_games import assigned_split
from .recorded_search_inputs import BoundInputs, checked_game
from .rules import START_FEN, VARIANT, adjudicate, piece_map
from .symmetry import mirror_fen


REPOSITORY = 'dffge552/xiangqi-pwa-offline'
ACQUISITION_KIND = 'actual_pinned_public_tactical_source_acquisition'


def position_orbit(fen):
    return min(position_key(fen), position_key(mirror_fen(fen)))


def native_position(row, origin):
    """Check supplied geometry and counters before entering the native binding."""
    if not isinstance(row, dict) or not isinstance(row.get('fen'), str):
        raise ValueError('An isolated position requires its original string FEN')
    source_fen = row['fen']
    fields = source_fen.split()
    if (len(source_fen) > 256 or '\0' in source_fen or len(fields) not in (2, 6) or
            fields[1] not in ('w', 'b') or len(fields) == 6 and (
                fields[2:4] != ['-', '-'] or
                not all(re.fullmatch(r'[0-9]{1,6}', s) for s in fields[4:]) or
                int(fields[5]) < 1)):
        raise ValueError('Unsupported isolated-position FEN fields or counters')
    pieces = piece_map(source_fen)
    if any(sum(p == king for p in pieces.values()) != 1 for king in ('K', 'k')):
        raise ValueError('An isolated position requires one general per side')
    # Unlike get_fen/other APIs, this upstream binding takes FEN before variant.
    if pyffish.validate_fen(source_fen, VARIANT) != pyffish.FEN_OK:
        raise ValueError('The source FEN failed native position validation')
    initial = pyffish.get_fen(VARIANT, source_fen, [])
    identity = hashlib.sha256(json.dumps([initial, []], separators=(',', ':')).encode()).hexdigest()
    standard = (initial.split()[:2] == START_FEN.split()[:2] and fields[4:] == ['0', '1'])
    return {'game_id': 'recorded-' + identity, 'initial_fen': initial,
        'source_initial_fen': source_fen, 'moves': [], 'history': [initial],
        'headers': {'FEN': source_fen}, 'native_terminal': adjudicate(initial, []),
        'source_kind': 'recorded_tactical_position',
        'source': dict(origin, source_fen_counters_present=len(fields) == 6,
                       missing_fen_counters_use_native_defaults=len(fields) == 2),
        'declared_participants': {'Red': None, 'Black': None}, 'players': [],
        'supplied_history_starts_at_standard_initial_position': standard,
        'pre_fragment_game_history_available': False,
        'provided_line_is_best_move_label': False,
        'source_comments_or_analysis_branches_used_as_labels': False,
        'provenance': 'source_declared_tactical_position;native_supplied_position_verified;'
                      'prehistory_unavailable;unverified_attribution',
        'source_move_optimality_proven': False, 'neural_explanations_generated': False}


def checked_source(source, acquisition, bound):
    source, acquisition = Path(source), Path(acquisition)
    proof = bound.json(acquisition / 'manifest.json')
    if (proof.get('status') != 'complete' or proof.get('kind') != ACQUISITION_KIND or
            proof.get('config', {}).get('repository') != REPOSITORY):
        raise ValueError('A completed pinned public tactical acquisition is required')
    for group in ('inputs', 'outputs'):
        for name, expected in proof[group].items():
            if not Path(name).resolve().is_relative_to(acquisition.resolve()):
                raise ValueError('Acquisition metadata escapes its declared output directory')
            bound.bind(name, expected)
    pinned = bound.json(acquisition / 'pinned-source.json')
    inventory = bound.json(acquisition / 'cached-source-files.json')
    revision = proof['config']['revision']
    if (not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision) or
            pinned.get('repository') != REPOSITORY or pinned.get('revision') != revision or
            not re.fullmatch('[0-9a-f]{40}', pinned.get('actual_git_tree_sha', '')) or
            pinned.get('checked_against_actual_tree_object') is not True or
            not isinstance(inventory, list) or not inventory):
        raise ValueError('Pinned tactical acquisition identity or inventory changed')
    blobs = pinned.get('selected_blobs')
    if not isinstance(blobs, list) or len(blobs) != len(inventory):
        raise ValueError('Pinned tactical blob coverage changed')
    by_path = {}
    for blob in blobs:
        name = blob.get('path', '')
        parts = PurePosixPath(name).parts
        if (not parts or PurePosixPath(name).is_absolute() or '..' in parts or
                blob.get('mode') != '100644' or blob.get('type') != 'blob' or
                not re.fullmatch('[0-9a-f]{40}', blob.get('sha', '')) or
                type(blob.get('size')) is not int or blob['size'] < 1 or name in by_path):
            raise ValueError('Unsupported pinned tactical source blob')
        by_path[name] = blob
    if {item.get('source_path') for item in inventory} != set(by_path):
        raise ValueError('Cached tactical source membership changed')
    for item in inventory:
        name = item['source_path']
        path = Path(item['local_path'])
        expected_path = source / name
        if (path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or
                path.resolve() != expected_path.resolve()):
            raise ValueError('Cached tactical source path escapes its declared cache')
        expected = {'sha256': item['sha256'], 'bytes': item['bytes']}
        bound.bind(path, expected)
        blob = by_path[name]
        raw = path.read_bytes()
        sha1 = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if sha1 != blob['sha'] or len(raw) != blob['size'] or item['git_blob_sha1'] != sha1:
            raise ValueError('Cached tactical source bytes differ from their pinned Git blob')
    bound.unchanged()
    return revision, sorted(inventory, key=lambda item: item['source_path'])


def import_positions(source, acquisition, output, *, previous=(), seed=20261051):
    if type(seed) is not int:
        raise ValueError('An integer position split seed is required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve isolated-position imports; use a fresh output')
    bound = BoundInputs()
    revision, inventory = checked_source(source, acquisition, bound)
    previous_orbits = set()
    for directory in map(Path, previous):
        proof = bound.json(directory / 'manifest.json')
        verified = proof.get('verification', {})
        native = ((verified.get('all_retained_moves_native_legal') is True and
                   verified.get('full_history_terminal_checks_passed') is True) or
                  verified.get('all_retained_native_full_histories_validated') is True or
                  verified.get('all_retained_supplied_positions_native_validated') is True)
        if (proof.get('status') != 'complete' or not proof.get('kind', '').endswith('_native_import') or
                proof.get('config', {}).get('seed') != seed or not native or
                verified.get('game_split_assigned_before_questions') is not True):
            raise ValueError('Previous canonical native imports require the same split seed')
        path = directory / 'games.jsonl'
        bound.bind(path, proof['outputs'][str(path)])
        for game in iter_jsonl(path):
            checked_game(game, seed)
            previous_orbits.update(position_orbit(fen) for fen in game['history'])
    output.mkdir(parents=True)
    accepted, duplicates, quarantine, counts, seen = [], [], [], Counter(), {}
    for item in inventory:
        path = Path(item['local_path'])
        if path.suffix != '.json':
            continue
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(rows, list):
            if item['source_path'] != 'patterns_index.json':
                raise ValueError('A tactical position source changed from an array')
            counts['source_metadata_files_skipped'] += 1
            continue
        for index, row in enumerate(rows):
            counts['source_position_rows'] += 1
            origin = {'repository': REPOSITORY, 'revision': revision, 'path': item['source_path'],
                'row_index': index, 'blob_sha1': item['git_blob_sha1'],
                'content_sha256': item['sha256'], 'declared_license': 'MIT',
                'row_content_sha256': hashlib.sha256(json.dumps(row, ensure_ascii=False,
                    sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                'source_authenticity_independently_proven': False}
            try:
                game = native_position(row, origin)
            except ValueError as error:
                quarantine.append(dict(origin, source_initial_fen=row.get('fen') if isinstance(row, dict) else None,
                                       error=str(error)))
                continue
            orbit = position_orbit(game['initial_fen'])
            if orbit in previous_orbits:
                duplicates.append(dict(origin, source_initial_fen=row['fen'],
                    reason='position_or_color_seen_in_previous_canonical_supplied_history'))
            elif orbit in seen:
                kept = seen[orbit]
                duplicates.append(dict(origin, source_initial_fen=row['fen'],
                    reason='same_source_position_or_color_duplicate',
                    retained_canonical_id=kept['game_id'], retained_split=kept['split']))
            else:
                game['split'] = assigned_split(game['game_id'], seed)
                seen[orbit] = game
                accepted.append(game)
            if counts['source_position_rows'] % 1024 == 0:
                print(json.dumps({'source_positions_checked': counts['source_position_rows'],
                    'retained_positions': len(accepted), 'quarantined': len(quarantine),
                    'duplicates': len(duplicates)}), flush=True)
    if not accepted:
        raise ValueError('No new native isolated source positions remain')
    bound.unchanged()
    for name, rows in [('games.jsonl', accepted), ('duplicates.jsonl', duplicates), ('quarantine.jsonl', quarantine)]:
        write_jsonl(output / name, rows)
    result = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
        'source_revision': revision, 'source_files': len(inventory), **dict(counts),
        'new_unique_native_source_positions': len(accepted), 'duplicate_positions': len(duplicates),
        'quarantined_positions': len(quarantine), 'prior_canonical_history_color_orbits': len(previous_orbits),
        'positions_by_split': dict(Counter(g['split'] for g in accepted)),
        'native_terminal_positions': sum(g['native_terminal']['ended'] for g in accepted),
        'all_retained_supplied_positions_native_validated': True,
        'full_history_terminal_checks_passed': True, 'native_verified_recorded_plies': 0,
        'game_split_assigned_before_questions': True,
        'source_and_color_duplicates_removed_before_split_assignment': True,
        'every_position_has_explicitly_unavailable_pre_fragment_history': True,
        'supplied_move_text_used_as_recorded_moves_or_best_move_labels': False,
        'source_comments_or_titles_used_as_neural_labels': False,
        'source_move_optimality_proven': False, 'native_proof_reuses_same_rules_implementation': True,
        'heldout_course_or_explanation_branch_isolation_completed_by_import': False,
        'training_questions_or_neural_labels_generated': False,
        'student_teacher_engine_or_feature_cache_loaded': False}
    atomic_json(output / 'counts.json', result)
    saved = manifest('isolated_tactical_positions_native_import',
        {'source': str(source), 'acquisition': str(acquisition), 'previous': list(map(str, previous)),
         'seed': seed, 'revision': revision}, [],
        [output / name for name in ['games.jsonl', 'duplicates.jsonl', 'quarantine.jsonl', 'counts.json']], result)
    saved['inputs'] = bound.artifacts
    bound.unchanged()
    atomic_json(output / 'manifest.json', saved)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--acquisition', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous', nargs='*', default=[])
    parser.add_argument('--seed', type=int, default=20261051)
    args = parser.parse_args()
    print(json.dumps(import_positions(args.source, args.acquisition, args.output,
        previous=args.previous, seed=args.seed), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
