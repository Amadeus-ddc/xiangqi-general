"""Native-admit a pinned anonymous elephantchess PvP CSV export."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import re
from urllib.parse import urlsplit
import zipfile

from .bundled_games import attribution, previous_games
from .evidence import atomic_json, digest, manifest
from .human_games import assigned_split, parse_game, recorded_year
from .recorded_sources import participant_label


PAGE = 'https://elephantchess.io/about/datasets'
COLUMNS = ('timestamp', 'move_index', 'move', 'game_id', 'red_player', 'black_player',
           'red_elo_before', 'red_elo_after', 'black_elo_before', 'black_elo_after',
           'time_control', 'time_control_category', 'rating_mode', 'game_status',
           'outcome', 'game_join_source', 'analysis', 'cpl')
STABLE = tuple(key for key in COLUMNS if key not in
               ('timestamp', 'move_index', 'move', 'analysis', 'cpl'))
RESULTS = {'RED_WINS': '1-0', 'BLACK_WINS': '0-1', 'DRAW': '1/2-1/2'}
DECISIVE = {'CHECKMATED', 'STALEMATED', 'PERPETUAL_CHECKING', 'RESIGNED', 'FLAGGED'}


def checked_acquisition(path):
    proof = json.loads(Path(path).read_text())
    if (proof.get('status') != 'complete_acquisition_and_structural_inventory_only'
            or proof.get('kind') != 'published_anonymous_xiangqi_pvp_export'
            or proof.get('dataset_page') != PAGE):
        raise ValueError('A completed pinned anonymous Xiangqi PvP acquisition is required')
    parsed = urlsplit(proof.get('archive_url', ''))
    if (parsed.scheme != 'https' or parsed.netloc != 'cdn.elephantchess.io'
            or not re.fullmatch(r'/data/pvp_game_moves_xiangqi_\d{4}-\d{2}\.zip', parsed.path)
            or parsed.query or parsed.fragment):
        raise ValueError('Archive is outside the explicitly published Xiangqi export contract')
    if (proof.get('distinct_archive_urls') != 1 or not proof.get('all_listed_zip_links')
            or set(proof['all_listed_zip_links']) != {proof['archive_url']}):
        raise ValueError('Month labels cannot be counted as disjoint exports')
    archive = Path(proof['archive_path'])
    page = archive.parent / 'datasets.html'
    if (archive.name != Path(parsed.path).name or digest(archive) != proof['archive_sha256']
            or archive.stat().st_size != proof['archive_bytes']
            or digest(page) != proof['dataset_page_sha256']):
        raise ValueError('Pinned archive or dataset-page bytes changed')
    cutoff = datetime.fromisoformat(proof['observed_at_utc'])
    if cutoff.tzinfo is None:
        raise ValueError('Acquisition cutoff must have an explicit timezone')
    return proof, archive, page, cutoff


def csv_groups(path, max_record_bytes=5_000_000):
    """Stream supplied row order, keeping one game across adjacent CSV members."""
    if type(max_record_bytes) is not int or max_record_bytes < 1:
        raise ValueError('A positive per-game CSV byte budget is required')
    seen, current, rows, spans, size = set(), None, [], [], 0
    with zipfile.ZipFile(path) as packed:
        members = [item for item in packed.infolist() if not item.is_dir()]
        if (not members or len(members) > 100000
                or sum(item.file_size for item in members) > 4_294_967_296
                or len({item.filename for item in members}) != len(members)):
            raise ValueError('Unsupported or duplicate archive member inventory')
        for item in members:
            if not re.fullmatch(r'pvp_game_moves_xiangqi_\d+\.csv', item.filename):
                raise ValueError('Only anonymous Xiangqi CSV members are supported')
            with packed.open(item) as source:
                reader = csv.DictReader(io.TextIOWrapper(source, encoding='utf-8-sig', newline=''),
                                        strict=True)
                if reader.fieldnames != list(COLUMNS):
                    raise ValueError('CSV columns differ from the pinned export schema')
                for ordinal, row in enumerate(reader, 1):
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError('CSV row does not supply exactly the declared fields')
                    identity = row['game_id']
                    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identity):
                        raise ValueError('CSV row lacks a supported anonymous game identifier')
                    if current is not None and identity != current:
                        yield rows, spans
                        seen.add(current)
                        rows, spans, size = [], [], 0
                    if identity != current:
                        if identity in seen:
                            raise ValueError('A game reappeared after another game; never reorder or join fragments')
                        current = identity
                    size += len(json.dumps(row, ensure_ascii=False).encode())
                    if size > max_record_bytes:
                        raise ValueError('A game exceeded the bounded CSV byte budget')
                    rows.append(row)
                    if not spans or spans[-1]['member'] != item.filename:
                        spans.append({'member': item.filename, 'first_data_row': ordinal,
                                      'last_data_row': ordinal})
                    else:
                        spans[-1]['last_data_row'] = ordinal
        if rows:
            yield rows, spans


def native_game(rows, cutoff, minimum_plies=20):
    """Check every CSV index and fact, then replay the full standard-start game."""
    if (type(minimum_plies) is not int or minimum_plies < 1
            or not isinstance(cutoff, datetime) or cutoff.tzinfo is None):
        raise ValueError('A positive minimum-ply budget and timezone-aware capture cutoff are required')
    if not rows or len(rows) < minimum_plies:
        raise ValueError('CSV game is below the minimum-ply budget')
    first, previous_time, moves = rows[0], None, []
    if any(set(row) != set(COLUMNS) or any(not isinstance(v, str) for v in row.values()) for row in rows):
        raise ValueError('Every supplied CSV field must be a string in the declared schema')
    status, result = first['game_status'], RESULTS.get(first['outcome'])
    if (not result or status not in DECISIVE | {'DRAW_ACCEPTED'}
            or (status == 'DRAW_ACCEPTED') != (result == '1/2-1/2')):
        raise ValueError('CSV status and outcome do not describe a supported finished game')
    if not first['red_player'].strip() or not first['black_player'].strip():
        raise ValueError('Both anonymous player labels are required')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', first['game_id']):
        raise ValueError('A supported anonymous source game identifier is required')
    for index, row in enumerate(rows):
        if row['move_index'] != str(index):
            raise ValueError('CSV move indices must start at zero and remain contiguous in source order')
        if any(row[key] != first[key] for key in STABLE):
            raise ValueError('Game-level CSV metadata changed within the supplied main line')
        try:
            when = datetime.fromisoformat(row['timestamp'])
        except ValueError as error:
            raise ValueError('CSV move timestamp is malformed') from error
        if (when.tzinfo is None or when > cutoff or
                (previous_time is not None and when < previous_time)):
            raise ValueError('CSV move timestamps must be aware, chronological and captured in the past')
        if not re.fullmatch(r'[a-i][0-9][a-i][0-9]', row['move']):
            raise ValueError('CSV moves must preserve the supplied zero-based ICCS coordinates')
        moves.append(row['move'])
        previous_time = when
    # These headers are factual CSV conversion, never an allegedly original PGN.
    headers = {'Game': 'Chinese Chess', 'Event': 'elephantchess.io anonymous PvP',
               'Date': datetime.fromisoformat(first['timestamp']).astimezone(timezone.utc).strftime('%Y.%m.%d'),
               'Red': first['red_player'], 'Black': first['black_player'], 'Result': result}
    body = '\n'.join(f'[{key} {json.dumps(value, ensure_ascii=False)}]' for key, value in headers.items())
    game = parse_game((body + '\n\n' + ' '.join(moves) + ' ' + result).encode())
    if status in {'CHECKMATED', 'STALEMATED', 'PERPETUAL_CHECKING'} and not game['native_terminal']['ended']:
        raise ValueError('A claimed board-rule terminal is not terminal under native full-history rules')
    return dict(game, source_format='ElephantchessAnonymousXiangqiCSV',
                platform_match_id=first['game_id'], date_precision='first_recorded_move',
                original_csv_game_metadata={key: first[key] for key in STABLE},
                first_recorded_move_utc=datetime.fromisoformat(first['timestamp']).astimezone(timezone.utc).isoformat(),
                last_recorded_move_utc=previous_time.astimezone(timezone.utc).isoformat(),
                declared_nonterminal_result_reason=status if not game['native_terminal']['ended'] else None,
                source_kind='recorded_human_match', source_kind_is_platform_pvp_assertion=True,
                real_world_player_identity_or_unassisted_play_authenticated=False,
                csv_analysis_or_cpl_used_for_labels=False,
                players=[participant_label(first[key]) for key in ['red_player', 'black_player']])


def native_record(item):
    rows, origin, cutoff, minimum = item
    try:
        game = native_game(rows, cutoff, minimum)
    except (ValueError, UnicodeError) as error:
        return None, dict(origin, error=str(error))
    return dict(game, source=origin), None


def native_results(items, workers):
    if workers == 1:
        yield from map(native_record, items)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending, source = deque(), iter(items)
        for _ in range(workers * 2):
            item = next(source, None)
            if item is None:
                break
            pending.append(pool.submit(native_record, item))
        while pending:
            yield pending.popleft().result()
            item = next(source, None)
            if item is not None:
                pending.append(pool.submit(native_record, item))


def import_csv(acquisition_path, output, previous=(), workers=8, seed=20261051,
               minimum_plies=20, max_record_bytes=5_000_000):
    if any(type(value) is not int or value < 1 for value in [workers, minimum_plies, max_record_bytes]):
        raise ValueError('Positive CPU worker, minimum-ply and CSV byte budgets are required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve an existing import; use a fresh output directory')
    proof, archive, page, cutoff = checked_acquisition(acquisition_path)
    source_inputs = [Path(acquisition_path), archive, page]
    before = {str(path): digest(path) for path in source_inputs}
    seen, prior_inputs = previous_games(previous, seed)
    prior_count = len(seen)
    counts = Counter(dict.fromkeys(['candidate_records_natively_examined', 'quarantined_records',
        'records_passing_native_full_history', 'duplicate_records',
        'duplicate_records_with_differing_attribution', 'new_unique_recorded_games', 'new_recorded_plies'], 0))
    kinds, splits, years, labels, errors = Counter(), Counter(), Counter(), Counter(), Counter()

    def candidates():
        for rows, spans in csv_groups(archive, max_record_bytes):
            origin = {'url': proof['archive_url'], 'dataset_page': PAGE,
                      'archive_sha256': proof['archive_sha256'], 'csv_spans': spans,
                      'source_game_id': rows[0]['game_id'], 'source_move_rows': len(rows),
                      'row_values_sha256': hashlib.sha256(json.dumps(rows, ensure_ascii=False,
                            separators=(',', ':')).encode()).hexdigest(),
                      'declared_license': 'GPL-3.0 (dataset-page declaration)',
                      'source_authenticity_independently_proven': False,
                      'public_redistribution_permission_established': False}
            yield rows, origin, cutoff, minimum_plies

    output.mkdir(parents=True)
    paths = {name: output / (name + '.jsonl') for name in
             ['games', 'quarantine', 'duplicates', 'attribution-differences']}
    def write(handle, row):
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    with paths['games'].open('w') as accepted, paths['quarantine'].open('w') as rejected, \
            paths['duplicates'].open('w') as duplicates, paths['attribution-differences'].open('w') as differences:
        for game, failure in native_results(candidates(), workers):
            counts['candidate_records_natively_examined'] += 1
            if failure:
                counts['quarantined_records'] += 1
                errors[failure['error']] += 1
                write(rejected, failure)
            else:
                counts['records_passing_native_full_history'] += 1
                key, split, facts = game['game_id'], assigned_split(game['game_id'], seed), attribution(game)
                if key in seen:
                    first = seen[key]
                    if first['split'] != split:
                        raise ValueError('Canonical duplicates cannot change their preassigned split')
                    counts['duplicate_records'] += 1
                    write(duplicates, {'game_id': key, 'first': first, 'duplicate_source': game['source']})
                    if facts != first['headers']:
                        counts['duplicate_records_with_differing_attribution'] += 1
                        write(differences, {'game_id': key, 'first': first, 'duplicate_headers': facts,
                                           'duplicate_source': game['source'], 'attributions_authenticated': False})
                else:
                    game['split'] = split
                    seen[key] = {'headers': facts, 'split': split, 'source': game['source'],
                                 'existing_collection': str(output)}
                    counts['new_unique_recorded_games'] += 1
                    counts['new_recorded_plies'] += len(game['moves'])
                    kinds[game['source_kind']] += 1
                    splits[split] += 1
                    years[str(recorded_year(game['headers']) or 'unknown')] += 1
                    labels.update(game['players'])
                    write(accepted, game)
            if counts['candidate_records_natively_examined'] % 1024 == 0:
                print(json.dumps(dict(counts)), flush=True)
    if any(digest(path) != before[str(path)] for path in source_inputs):
        raise ValueError('Pinned acquisition inputs changed during native import')
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline', **dict(counts),
               'prior_unique_recorded_games': prior_count, 'combined_unique_recorded_games': len(seen),
               'games_by_source_declared_kind': dict(kinds), 'games_by_split': dict(splits),
               'games_by_year': dict(years), 'distinct_anonymous_participant_labels': len(labels),
               'quarantine_reasons': dict(errors), 'game_split_assigned_before_questions': True,
               'all_retained_native_full_histories_validated': True,
               'original_csv_order_indices_timestamps_and_metadata_checked': True,
               'zero_based_coordinates_preserved_without_conversion': True,
               'source_authenticity_or_unassisted_human_play_authenticated': False,
               'source_license_declaration': 'GPL-3.0 (dataset-page declaration)',
               'public_redistribution_permission_established': False,
               'raw_records_or_archives_bundled_with_code': False,
               'csv_analysis_or_cpl_used_as_neural_labels': False,
               'recorded_move_optimality_proven': False, 'heldout_position_isolation_complete': False,
               'accepted_search_roots': 0, 'training_data_or_neural_labels_generated': False}
    atomic_json(output / 'verification.json', summary)
    config = {'acquisition': str(acquisition_path), 'previous': list(map(str, previous)),
              'workers': workers, 'seed': seed, 'minimum_plies': minimum_plies,
              'max_record_bytes': max_record_bytes}
    atomic_json(output / 'manifest.json', manifest('public_elephantchess_recorded_games_native_import', config,
        [*source_inputs, *prior_inputs], [*paths.values(), output / 'verification.json'], summary))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acquisition', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous', action='append', default=[])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261051)
    parser.add_argument('--minimum-plies', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(import_csv(args.acquisition, args.output, args.previous, args.workers,
                              args.seed, args.minimum_plies)), flush=True)


if __name__ == '__main__':
    main()
