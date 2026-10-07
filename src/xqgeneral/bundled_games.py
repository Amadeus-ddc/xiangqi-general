"""Import hash-pinned multi-game PGN archives with native history and provenance."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import zipfile

from .evidence import atomic_json, digest, manifest
from .human_games import assigned_split, parse_game, recorded_year
from .recorded_sources import participant_label
from .rules import START_FEN


ATTRIBUTION = ('Event', 'Date', 'Red', 'Black', 'Result')


def pgn_records(stream, max_record_bytes=2_000_000):
    """Preserve record bytes; a header inside a comment cannot start a game."""
    if max_record_bytes < 1:
        raise ValueError('A positive PGN record byte limit is required')
    lines, size, braces, variations = [], 0, 0, 0
    for line in stream:
        stripped = line.removeprefix(b'\xef\xbb\xbf').lstrip()
        header = not braces and not variations and re.match(rb'^\[Game\b', stripped)
        if header:
            if lines:
                yield b''.join(lines)
            lines, size = [], 0
        if not lines and not header:
            if stripped.strip():
                raise ValueError('Bundled PGN contains content before its first Game header')
            continue
        lines.append(line)
        size += len(line)
        if size > max_record_bytes:
            raise ValueError('Bundled PGN record exceeds the declared byte limit')
        if not braces and not variations and stripped.startswith(b'['):
            continue
        for char in line:
            if braces:
                braces += (char == 123) - (char == 125)
            elif char == 59:
                break  # PGN semicolon comments end at the newline.
            elif char == 123:
                braces = 1
            elif char == 40:
                variations += 1
            elif char == 41:
                variations = max(0, variations - 1)
    if lines:
        yield b''.join(lines)


def native_record(item):
    raw, origin = item
    try:
        game = parse_game(raw)
        if game['initial_fen'].split()[:2] != START_FEN.split()[:2]:
            raise ValueError('Recorded full match has a nonstandard starting position')
    except ValueError as error:
        return None, dict(origin, error=str(error))
    return dict(game, players=[participant_label(game['headers'][key]) for key in ['Red', 'Black']],
                source_kind='published_recorded_match', source=origin,
                provenance='public_recorded_mainline;native_full_history_verified;unverified_attribution'), None


def ordered_native_results(items, workers):
    """Keep source order and at most two queued records per CPU worker."""
    if workers < 1:
        raise ValueError('A positive CPU worker count is required')
    if workers == 1:
        yield from map(native_record, items)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        queue, source = deque(), iter(items)
        for _ in range(workers * 2):
            item = next(source, None)
            if item is None:
                break
            queue.append(pool.submit(native_record, item))
        while queue:
            yield queue.popleft().result()
            item = next(source, None)
            if item is not None:
                queue.append(pool.submit(native_record, item))


def verify_manifest(path, include_inputs=True):
    proof = json.loads(Path(path).read_text())
    if proof.get('status') != 'complete':
        raise ValueError('A completed, hash-pinned source manifest is required')
    artifacts = {**(proof['inputs'] if include_inputs else {}), **proof['outputs']}
    for name, item in artifacts.items():
        if digest(name) != item['sha256'] or Path(name).stat().st_size != item['bytes']:
            raise ValueError('Pinned source artifact changed: ' + name)
    return proof


def attribution(game):
    return {key: game['headers'].get(key) for key in ATTRIBUTION}


def previous_games(data_roots, seed):
    seen, inputs = {}, []
    for root in map(Path, data_roots):
        proof_path, path = root / 'manifest.json', root / 'games.jsonl'
        # Completed native imports are reused through their exact output bytes;
        # do not re-open tens of thousands of original upstream PGN files here.
        proof = verify_manifest(proof_path, include_inputs=False)
        verified = proof['verification']
        native = (verified.get('all_retained_moves_native_legal') is True and
                  verified.get('full_history_terminal_checks_passed') is True)
        if (proof['config'].get('seed') != seed or not verified.get('game_split_assigned_before_questions')
                or not (native or verified.get('all_retained_native_full_histories_validated') is True)):
            raise ValueError('Previous import must prove native full histories and the same declared split seed')
        if str(path) not in proof['outputs']:
            raise ValueError('Previous game file is absent from its completed source manifest')
        for line in path.open():
            game = json.loads(line)
            identity = hashlib.sha256(json.dumps([game['initial_fen'], game['moves']],
                                      separators=(',', ':')).encode()).hexdigest()
            if game['game_id'] != 'recorded-' + identity or game['split'] != assigned_split(game['game_id'], seed):
                raise ValueError('Previous canonical game identity or split differs from the declared contract')
            seen.setdefault(game['game_id'], {'headers': attribution(game), 'split': game['split'],
                                             'source': game.get('source'), 'existing_collection': str(root)})
        inputs.extend([proof_path, path])
    return seen, inputs


def import_bundles(acquisition_path, output, previous=(), workers=8, seed=20261051,
                   limit_records=None, max_record_bytes=2_000_000):
    if workers < 1 or max_record_bytes < 1 or (limit_records is not None and limit_records < 1):
        raise ValueError('Positive worker, record-byte and optional record-count limits are required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve completed or interrupted imports; use a fresh output directory')
    acquisition = verify_manifest(acquisition_path)
    source = acquisition['config']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', source['repository']) or not re.fullmatch(
            r'[0-9a-f]{40}', source['revision']):
        raise ValueError('Source repository and pinned Git revision are required')
    inventory = [item for item in acquisition['verification']['actual_downloads']
                 if Path(item['path']).suffix.lower() == '.zip']
    if not inventory:
        raise ValueError('Source acquisition contains no PGN archives')
    for item in inventory:
        if item['path'] not in acquisition['inputs'] or digest(item['path']) != item['sha256']:
            raise ValueError('Archive is absent from the pinned acquisition inputs')
    seen, prior_inputs = previous_games(previous, seed)
    prior_unique = len(seen)
    output.mkdir(parents=True)
    counts = Counter(dict.fromkeys(['candidate_records_natively_examined', 'quarantined_records',
        'records_passing_native_full_history', 'duplicate_records',
        'duplicate_records_with_differing_attribution', 'new_unique_recorded_games', 'new_recorded_plies'], 0))
    by_kind, by_year, by_split, labels, errors = Counter(), Counter(), Counter(), Counter(), Counter()
    archives_processed = []

    def candidates():
        ordinal = 0
        for item in inventory:
            path = Path(item['path'])
            archives_processed.append(str(path))
            with zipfile.ZipFile(path) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                if sum(member.file_size for member in members) > 2_000_000_000:
                    raise ValueError('Archive exceeds the supported uncompressed-byte budget')
                for member in members:
                    if Path(member.filename).suffix.lower() not in ('.pgn', '.pgns'):
                        raise ValueError('Only PGN data members are supported; no archived code is executed')
                    with archive.open(member) as handle:
                        for member_ordinal, raw in enumerate(pgn_records(handle, max_record_bytes), 1):
                            if limit_records is not None and ordinal >= limit_records:
                                return
                            ordinal += 1
                            origin = {'repository': source['repository'], 'revision': source['revision'],
                                      'collection_reference': source.get('upstream_collection'),
                                      'archive_path': str(path), 'archive_sha256': item['sha256'],
                                      'archive_git_blob_sha1': item['git_blob_sha1'],
                                      'archive_member': member.filename, 'record_ordinal': member_ordinal,
                                      'content_sha256': hashlib.sha256(raw).hexdigest(),
                                      'declared_license': source.get('declared_source_license', 'unspecified'),
                                      'public_redistribution_permission_established': False,
                                      'source_authenticity_independently_proven': False}
                            yield raw, origin

    def write(handle, value):
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')

    paths = {name: output / (name + '.jsonl') for name in ['games', 'quarantine', 'duplicates', 'attribution-differences']}
    with paths['games'].open('w') as accepted, paths['quarantine'].open('w') as rejected, \
            paths['duplicates'].open('w') as duplicates, paths['attribution-differences'].open('w') as differences:
        for game, failure in ordered_native_results(candidates(), workers):
            counts['candidate_records_natively_examined'] += 1
            if failure:
                counts['quarantined_records'] += 1
                errors[failure['error']] += 1
                write(rejected, failure)
            else:
                counts['records_passing_native_full_history'] += 1
                key, split, headers = game['game_id'], assigned_split(game['game_id'], seed), attribution(game)
                if key in seen:
                    first = seen[key]
                    if first['split'] != split:
                        raise ValueError('A duplicate canonical game cannot change split')
                    counts['duplicate_records'] += 1
                    write(duplicates, {'game_id': key, 'first': first, 'duplicate_source': game['source']})
                    if first['headers'] != headers:
                        counts['duplicate_records_with_differing_attribution'] += 1
                        write(differences, {'game_id': key, 'first': first, 'duplicate_headers': headers,
                                           'duplicate_source': game['source'], 'attributions_authenticated': False})
                else:
                    game['split'] = split
                    seen[key] = {'headers': headers, 'split': split, 'source': game['source'],
                                 'existing_collection': str(output)}
                    counts['new_unique_recorded_games'] += 1
                    counts['new_recorded_plies'] += len(game['moves'])
                    by_kind[game['source_kind']] += 1
                    by_year[str(recorded_year(game['headers']) or 'unknown')] += 1
                    by_split[split] += 1
                    labels.update(game['players'])
                    write(accepted, game)
            if counts['candidate_records_natively_examined'] % 1024 == 0:
                print(json.dumps(dict(counts)), flush=True)
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline', **dict(counts),
               'prior_unique_recorded_games': prior_unique, 'combined_unique_recorded_games': len(seen),
               'games_by_source_declared_kind': dict(by_kind), 'games_by_year': dict(by_year),
               'games_by_split': dict(by_split), 'distinct_participant_labels': len(labels),
               'quarantine_reasons': dict(errors), 'archives_processed': archives_processed,
               'record_budget_limited': limit_records is not None,
               'game_split_assigned_before_questions': True, 'all_retained_native_full_histories_validated': True,
               'source_authenticity_or_participant_identities_authenticated': False,
               'recorded_move_optimality_proven': False, 'site_prose_or_variations_used_as_neural_labels': False,
               'source_license_or_redistribution_permission_established': False,
               'raw_records_or_archives_bundled_with_code': False, 'training_data_or_neural_labels_generated': False}
    atomic_json(output / 'verification.json', summary)
    config = {'acquisition': str(acquisition_path), 'previous': list(map(str, previous)), 'seed': seed,
              'workers': workers, 'limit_records': limit_records, 'max_record_bytes': max_record_bytes}
    atomic_json(output / 'manifest.json', manifest('bundled_recorded_games_native_import', config,
                [acquisition_path, *[item['path'] for item in inventory], *prior_inputs],
                [*paths.values(), output / 'verification.json'], summary))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acquisition', required=True)
    parser.add_argument('--previous-data', nargs='*', default=[])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261051)
    parser.add_argument('--limit-records', type=int)
    parser.add_argument('--max-record-bytes', type=int, default=2_000_000)
    parser.add_argument('--output', required=True)
    parser.add_argument('--public-evidence')
    args = parser.parse_args()
    summary = import_bundles(args.acquisition, args.output, args.previous_data, args.workers,
                             args.seed, args.limit_records, args.max_record_bytes)
    if args.public_evidence:
        atomic_json(args.public_evidence, dict(summary, manifest_sha256=digest(Path(args.output) / 'manifest.json')))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
