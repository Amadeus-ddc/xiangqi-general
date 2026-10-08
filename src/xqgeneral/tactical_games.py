"""Import source-declared tactical PGNs without inventing players or best moves."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path, PurePosixPath
import re
import subprocess

from .bundled_games import previous_games
from .evidence import atomic_json, file_signature, manifest, write_jsonl
from .human_games import assigned_split, parse_game
from .recorded_sources import participant_label
from .rules import START_FEN


REPOSITORY = 'Yvonne761/Chinese-Chess-Practical-Dataset'
CATEGORIES = {'中局': 'midgame', '殘局': 'endgame', '全盤戰術': 'full_game_tactics',
              '殺局_殺法_練習題': 'mate_exercises'}


def source_inventory(tree):
    blobs, counts = {}, Counter()
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        descriptor, raw_path = entry.split(b'\t', 1)
        path = raw_path.decode('utf-8')
        parts = PurePosixPath(path).parts
        if (len(parts) < 3 or parts[0] != 'Dataset' or parts[1] not in CATEGORIES or
                PurePosixPath(path).suffix != '.pgn'):
            continue
        mode, kind, blob = descriptor.decode('ascii').split()
        if mode != '100644' or kind != 'blob' or not re.fullmatch('[0-9a-f]{40}', blob):
            raise ValueError('Tactical source must contain ordinary pinned PGN blobs')
        blobs.setdefault(blob, []).append(path)
        counts[CATEGORIES[parts[1]]] += 1
    if not blobs:
        raise ValueError('The pinned source contains no supported tactical PGNs')
    return blobs, dict(counts)


def native_line(item):
    raw, origin = item
    if digest_bytes(raw) != origin['content_sha256']:
        raise ValueError('Tactical source bytes changed before native import')
    try:
        game = parse_game(raw, require_players=False)
    except ValueError as error:
        return None, dict(origin, error=str(error))
    standard = (game['initial_fen'].split()[:2] == START_FEN.split()[:2] and
                game['source_initial_fen'].split()[4:] == ['0', '1'])
    people = {key: participant_label(game['headers'][key]) if game['headers'].get(key) else None
              for key in ['Red', 'Black']}
    return dict(game, source_kind='recorded_tactical_line', source=origin,
        players=[label for label in people.values() if label], declared_participants=people,
        supplied_history_starts_at_standard_initial_position=standard,
        pre_fragment_game_history_available=standard,
        provided_line_is_best_move_label=False,
        source_comments_or_analysis_branches_used_as_labels=False,
        provenance='source_declared_tactical_line;native_supplied_history_verified;unverified_attribution'), None


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def ordered_lines(items, workers):
    if workers == 1:
        yield from map(native_line, items)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending, source = deque(), iter(items)
        for _ in range(workers * 2):
            item = next(source, None)
            if item is None:
                break
            pending.append(pool.submit(native_line, item))
        while pending:
            yield pending.popleft().result()
            item = next(source, None)
            if item is not None:
                pending.append(pool.submit(native_line, item))


def import_tactics(source, revision, output, *, previous=(), workers=8, seed=20261051):
    if type(workers) is not int or workers < 1 or type(seed) is not int or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('A pinned revision, integer split seed and positive worker count are required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve completed and interrupted tactical imports; use a fresh output')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != revision:
        raise ValueError('Tactical source differs from the pinned revision')
    tree = subprocess.check_output(['git', '-C', str(source), 'ls-tree', '-r', '-z', revision])
    blobs, categories = source_inventory(tree)
    prior_signatures = {str(Path(p) / name): file_signature(Path(p) / name)
                        for p in previous for name in ['manifest.json', 'games.jsonl']}
    seen, inputs = previous_games(previous, seed)
    if any(file_signature(p) != signature for p, signature in prior_signatures.items()):
        raise ValueError('A previous canonical source changed during tactical deduplication')
    prior_games = len(seen)
    output.mkdir(parents=True)
    atomic_json(output / 'source-paths-by-blob.json', blobs)
    objects = output / 'objects.bin'
    with objects.open('xb') as handle:
        extracted = subprocess.run(['git', '-C', str(source), 'cat-file', '--batch'],
            input=('\n'.join(blobs) + '\n').encode('ascii'), stdout=handle,
            stderr=subprocess.PIPE, timeout=600)
    if extracted.returncode:
        raise RuntimeError('Pinned tactical source extraction failed')
    license_path = output / 'SOURCE_LICENSE'
    license_path.write_bytes(subprocess.check_output(['git', '-C', str(source), 'show', revision + ':LICENSE']))

    def items():
        with objects.open('rb') as handle:
            for blob, paths in blobs.items():
                header = handle.readline().decode('ascii').split()
                if header[:2] != [blob, 'blob'] or len(header) != 3 or not 0 < int(header[2]) <= 2_000_000:
                    raise ValueError('Unexpected tactical Git object descriptor')
                raw = handle.read(int(header[2]))
                if (handle.read(1) != b'\n' or hashlib.sha1(
                        b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != blob):
                    raise ValueError('Tactical Git object bytes differ from the pinned tree')
                raw_path = output / 'sources' / (blob + '.pgn')
                raw_path.parent.mkdir(exist_ok=True)
                raw_path.write_bytes(raw)
                inputs.append(raw_path)
                origin = {'repository': REPOSITORY, 'revision': revision, 'paths': paths,
                    'categories': sorted({CATEGORIES[PurePosixPath(p).parts[1]] for p in paths}),
                    'blob_sha1': blob, 'content_sha256': digest_bytes(raw), 'declared_license': 'CC-BY-4.0',
                    'source_authenticity_independently_proven': False}
                yield raw, origin
            if handle.read(1):
                raise ValueError('Unexpected trailing tactical Git objects')

    accepted, quarantine, duplicates = [], [], []
    for index, (game, error) in enumerate(ordered_lines(items(), workers), 1):
        if error:
            quarantine.append(error)
        elif game['game_id'] in seen:
            duplicates.append({'game_id': game['game_id'], 'source': game['source'],
                               'original_split': seen[game['game_id']]['split']})
        else:
            game['split'] = assigned_split(game['game_id'], seed)
            seen[game['game_id']] = {'split': game['split']}
            accepted.append(game)
        if index % 128 == 0:
            print(json.dumps({'native_blobs_processed': index, 'retained_lines': len(accepted),
                              'quarantined': len(quarantine), 'duplicates': len(duplicates)}), flush=True)
    if not accepted:
        raise ValueError('No new source-declared tactical lines passed native import')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != revision:
        raise ValueError('Pinned tactical source changed during import')
    if any(file_signature(p) != signature for p, signature in prior_signatures.items()):
        raise ValueError('A previous canonical source changed during tactical import')
    for name, rows in [('games.jsonl', accepted), ('quarantine.jsonl', quarantine), ('duplicates.jsonl', duplicates)]:
        write_jsonl(output / name, rows)
    result = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
        'source_revision': revision, 'declared_source_license': 'CC-BY-4.0',
        'source_category_file_counts': categories, 'source_category_files': sum(categories.values()),
        'unique_git_blobs_examined': len(blobs), 'previous_unique_games': prior_games,
        'new_unique_tactical_lines': len(accepted), 'quarantined_blobs': len(quarantine),
        'duplicate_canonical_lines': len(duplicates), 'native_verified_plies': sum(len(g['moves']) for g in accepted),
        'games_by_split': dict(Counter(g['split'] for g in accepted)),
        'lines_by_primary_declared_category': dict(Counter(g['source']['categories'][0] for g in accepted)),
        'nonstandard_initial_position_fragments': sum(not g['pre_fragment_game_history_available'] for g in accepted),
        'lines_missing_one_or_both_declared_participants': sum(
            any(value is None for value in g['declared_participants'].values()) for g in accepted),
        'all_retained_moves_native_legal': True, 'full_history_terminal_checks_passed': True,
        'game_split_assigned_before_questions': True, 'provided_fragment_history_preserved': True,
        'unknown_pre_fragment_history_reconstructed': False, 'source_authenticity_independently_proven': False,
        'provided_moves_are_oracle_best_move_labels': False, 'source_prose_used_as_neural_labels': False,
        'training_questions_or_neural_labels_generated': False, 'gpu_or_teacher_loaded': False}
    atomic_json(output / 'manifest.json', manifest('recorded_tactical_lines_native_import',
        {'source': str(source), 'revision': revision, 'previous': list(map(str, previous)),
         'workers': workers, 'seed': seed, 'output': str(output)},
        [*inputs, objects, license_path, output / 'source-paths-by-blob.json'],
        [output / name for name in ['games.jsonl', 'quarantine.jsonl', 'duplicates.jsonl']], result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous', nargs='*', default=[])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261051)
    args = parser.parse_args()
    print(json.dumps(import_tactics(args.source, args.revision, args.output,
        previous=args.previous, workers=args.workers, seed=args.seed)), flush=True)


if __name__ == '__main__':
    main()
