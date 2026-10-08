"""Import immutable PlayStrategy exports through native Xiangqi rules."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from .bundled_games import attribution, previous_games, verify_manifest
from .evidence import atomic_json, manifest
from .human_games import assigned_split, mainline_text, parse_game, recorded_year
from .recorded_sources import participant_label
from .rules import START_FEN, from_fairy

PLATFORM_TAG = re.compile(r'^\[([A-Za-z][A-Za-z0-9_]*)\s+"((?:\\.|[^"\\])*)"\]\s*$', re.M)
FINISHED = {'mate', 'resign', 'stalemate', 'timeout', 'draw', 'outoftime', 'variantEnd'}


def native_export(value, cutoff_ms, minimum_plies=20):
    """Check original JSON/PGN parity before native full-history admission."""
    if type(cutoff_ms) is not int or cutoff_ms < 1 or type(minimum_plies) is not int or minimum_plies < 1:
        raise ValueError('Positive capture cutoff and minimum-ply budget are required')
    if not isinstance(value, dict) or value.get('variant') != 'xiangqi':
        raise ValueError('Export record is not standard Xiangqi')
    if not isinstance(value.get('id'), str) or not re.fullmatch('[A-Za-z0-9]{8}', value['id']):
        raise ValueError('A platform game identifier is required')
    if not isinstance(value.get('status'), str) or value['status'] not in FINISHED:
        raise ValueError('Export does not declare a finished substantive game')
    created, ended = value.get('createdAt'), value.get('lastMoveAt')
    if type(created) is not int or type(ended) is not int or not 0 < created <= ended <= cutoff_ms:
        raise ValueError('Export timestamps do not describe a completed past game')
    if not isinstance(value.get('moves'), str) or not isinstance(value.get('pgn'), str):
        raise ValueError('Complete JSON coordinates and original PGN are required')
    text = value['pgn'].encode().decode('utf-8-sig')
    entries = PLATFORM_TAG.findall(text)
    headers = {k: re.sub(r'\\(["\\])', r'\1', v) for k, v in entries}
    if len(headers) != len(entries):
        raise ValueError('Duplicate original platform PGN headers')
    if any(line.lstrip().startswith('[') and not PLATFORM_TAG.fullmatch(line) for line in text.splitlines()):
        raise ValueError('Malformed original platform PGN header')
    if headers.get('Variant') != 'Xiangqi' or headers.get('SetUp', '0') != '0' or 'FEN' in headers or value.get('initialFen'):
        raise ValueError('Only full standard-start platform games are admitted')
    expected_date = datetime.fromtimestamp(created / 1000, timezone.utc).strftime('%Y.%m.%d')
    if headers.get('Date') != expected_date or ('UTCDate' in headers and headers['UTCDate'] != expected_date):
        raise ValueError('Original PGN date differs from the exported UTC creation date')
    winner = value.get('winner')
    if winner not in ('p1', 'p2', None):
        raise ValueError('Unexpected platform winner')
    result = {'p1': '1-0', 'p2': '0-1', None: '1/2-1/2'}[winner]
    if result != headers.get('Result') or (winner is None and value['status'] != 'draw'):
        raise ValueError('Original PGN and platform result disagree')
    source_moves = value['moves'].split()
    if mainline_text(PLATFORM_TAG.sub('', text)).split() != [*source_moves, result]:
        raise ValueError('Original PGN and JSON main line disagree')
    moves = [from_fairy(token) for token in source_moves]
    if len(moves) < minimum_plies:
        raise ValueError('Platform game is below the declared minimum-ply budget')
    players = value.get('players')
    if not isinstance(players, dict):
        raise ValueError('Exported player identities are required')
    computers = []
    for slot in ('p1', 'p2'):
        player = players.get(slot)
        user = player.get('user') if isinstance(player, dict) else None
        if not isinstance(user, dict) or not isinstance(user.get('id'), str) or not user['id'] or not isinstance(user.get('name'), str) or not user['name']:
            raise ValueError('Both exported registered player identities are required')
        if user['name'] != headers.get(slot.upper()) or user.get('title') != headers.get(slot.upper() + 'Title'):
            raise ValueError('Original PGN and exported player identity or title differ')
        if 'aiLevel' in player and (type(player['aiLevel']) is not int or player['aiLevel'] < 1):
            raise ValueError('An explicitly declared AI level must be a positive integer')
        computers.append(user.get('title') == 'BOT' or 'aiLevel' in player)
    canonical = {**{k: v for k, v in headers.items() if re.fullmatch('[A-Za-z]+', k)},
                 'Game': 'Chinese Chess', 'Red': headers['P1'], 'Black': headers['P2']}
    body = '\n'.join(f'[{k} {json.dumps(v, ensure_ascii=False)}]' for k, v in canonical.items())
    game = parse_game((body + '\n\n' + ' '.join(moves) + ' ' + result).encode())
    if game['moves'] != moves or game['initial_fen'].split()[:2] != START_FEN.split()[:2]:
        raise ValueError('Native history differs from the exported standard-start coordinates')
    kind = ('recorded_computer_match' if all(computers) else
            'recorded_human_computer_match' if any(computers) else 'recorded_human_match')
    return dict(game, source_format='PlayStrategyUCI1', platform_match_id=value['id'],
                original_platform_headers=headers, source_kind=kind,
                source_kind_is_platform_user_or_bot_assertion=True,
                real_world_player_identity_or_unassisted_play_authenticated=False,
                players=[participant_label(headers[k]) for k in ['P1', 'P2']])


def native_record(item):
    raw, origin, cutoff, minimum = item
    try:
        game = native_export(json.loads(raw), cutoff, minimum)
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
            if item is None: break
            pending.append(pool.submit(native_record, item))
        while pending:
            yield pending.popleft().result()
            item = next(source, None)
            if item is not None: pending.append(pool.submit(native_record, item))


def capture_inventory(proof):
    """Accept only completed public captures, with exact file and time identities."""
    if proof.get('kind') != 'public_platform_recorded_game_acquisition':
        raise ValueError('A completed public-platform acquisition manifest is required')
    items = proof['verification'].get('captures', [])
    if not isinstance(items, list) or not items:
        raise ValueError('Acquisition contains no captured exports')
    completed = datetime.fromisoformat(proof['created_utc'])
    if completed.tzinfo is None:
        raise ValueError('Acquisition completion timestamp requires an explicit timezone')
    seen = set()
    for item in items:
        bound = proof['outputs'].get(item['path'])
        if item['path'] in seen or bound != {k: item[k] for k in ['sha256', 'bytes']}:
            raise ValueError('Capture is duplicated or absent from its exact acquisition outputs')
        seen.add(item['path'])
        captured = datetime.fromisoformat(item['captured_utc'])
        if captured.tzinfo is None or captured > completed:
            raise ValueError('Capture must be timezone-aware and precede acquisition completion')
        parsed = urlsplit(item['url'])
        if parsed.scheme != 'https' or parsed.netloc != 'playstrategy.org' or parsed.fragment or not re.fullmatch(
                r'/api/(?:games/user/[A-Za-z0-9_-]{1,64}|tournament/[A-Za-z0-9]{8}/games)', parsed.path):
            raise ValueError('Capture URL is outside the public export endpoint contract')
        allowed = {'max','perfType','pgnInJson','moves','clocks','ongoing','until','since','rated',
                   'tags','evals','opening','vs','playerIndex','analysed'}
        query = parse_qs(parsed.query, keep_blank_values=True)
        if set(query) - allowed or any(len(v) != 1 for v in query.values()) or query.get('pgnInJson') != ['true'] or query.get('moves') != ['true']:
            raise ValueError('Capture query must declare original PGN and moves without unrecognized parameters')
        if type(item.get('records')) is not int or item['records'] < 0:
            raise ValueError('Capture requires its exact exported-record count')
    return items


def import_exports(acquisition_path, output, previous=(), workers=8, seed=20261051,
                   minimum_plies=20, max_record_bytes=5_000_000):
    """Native-admit registered-account exports and deduplicate before questions."""
    if any(type(n) is not int or n < 1 for n in [workers, minimum_plies, max_record_bytes]):
        raise ValueError('Positive worker, minimum-ply and record-byte budgets are required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve completed or interrupted imports; use a fresh output directory')
    acquisition = verify_manifest(acquisition_path)
    inventory = capture_inventory(acquisition)
    seen, prior_inputs = previous_games(previous, seed)
    prior_unique = len(seen)
    counts = Counter(dict.fromkeys(['candidate_records_natively_examined','quarantined_records',
        'records_passing_native_full_history','duplicate_records','duplicate_records_with_differing_attribution',
        'new_unique_recorded_games','new_recorded_plies'], 0))
    by_kind, by_split, by_year, labels, errors = Counter(), Counter(), Counter(), Counter(), Counter()

    def candidates():
        for item in inventory:
            count = 0; cutoff = int(datetime.fromisoformat(item['captured_utc']).timestamp() * 1000)
            with Path(item['path']).open('rb') as handle:
                ordinal = 0
                while True:
                    raw = handle.readline(max_record_bytes + 1)
                    if not raw: break
                    ordinal += 1
                    if len(raw) > max_record_bytes:
                        raise ValueError('Export record exceeds the declared byte budget')
                    if not raw.strip(): continue
                    count += 1
                    origin = {'url':item['url'],'export_sha256':item['sha256'],'ordinal':ordinal,
                        'record_bytes_sha256':hashlib.sha256(raw).hexdigest(),'captured_utc':item['captured_utc'],
                        'declared_license':item.get('source_license','unspecified'),
                        'public_redistribution_permission_established':False,
                        'source_authenticity_independently_proven':False}
                    yield raw, origin, cutoff, minimum_plies
            if count != item['records']:
                raise ValueError('Actual export-record count differs from the acquisition contract')

    output.mkdir(parents=True)
    paths = {name:output/(name + '.jsonl') for name in ['games','quarantine','duplicates','attribution-differences']}
    def write(handle, row): handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    with paths['games'].open('w') as accepted, paths['quarantine'].open('w') as rejected, \
            paths['duplicates'].open('w') as duplicates, paths['attribution-differences'].open('w') as differences:
        for game, failure in native_results(candidates(), workers):
            counts['candidate_records_natively_examined'] += 1
            if failure:
                counts['quarantined_records'] += 1; errors[failure['error']] += 1; write(rejected,failure)
            else:
                counts['records_passing_native_full_history'] += 1
                key, split, headers = game['game_id'], assigned_split(game['game_id'],seed), attribution(game)
                if key in seen:
                    first = seen[key]
                    if first['split'] != split: raise ValueError('A duplicate canonical game cannot change split')
                    counts['duplicate_records'] += 1
                    write(duplicates, {'game_id':key,'first':first,'duplicate_source':game['source']})
                    if headers != first['headers']:
                        counts['duplicate_records_with_differing_attribution'] += 1
                        write(differences, {'game_id':key,'first':first,'duplicate_headers':headers,
                            'duplicate_source':game['source'],'attributions_authenticated':False})
                else:
                    game['split'] = split
                    seen[key] = {'headers':headers,'split':split,'source':game['source'],'existing_collection':str(output)}
                    counts['new_unique_recorded_games'] += 1; counts['new_recorded_plies'] += len(game['moves'])
                    by_kind[game['source_kind']] += 1; by_split[split] += 1
                    by_year[str(recorded_year(game['headers']) or 'unknown')] += 1
                    labels.update(game['players']); write(accepted,game)
            if counts['candidate_records_natively_examined'] % 1024 == 0:
                print(json.dumps(dict(counts)),flush=True)
    summary = {'status':'complete','evidence_state':'reconstructed_baseline',**dict(counts),
        'prior_unique_recorded_games':prior_unique,'combined_unique_recorded_games':len(seen),
        'games_by_source_declared_kind':dict(by_kind),'games_by_split':dict(by_split),'games_by_year':dict(by_year),
        'distinct_participant_labels':len(labels),'quarantine_reasons':dict(errors),'captured_exports':len(inventory),
        'original_json_pgn_mainlines_results_and_registered_player_metadata_checked':True,
        'explicit_one_based_platform_rank_conversion':True,'game_split_assigned_before_questions':True,
        'all_retained_native_full_histories_validated':True,'platform_source_kind_is_not_identity_or_unassisted_play_proof':True,
        'source_authenticity_or_participant_identities_authenticated':False,'recorded_move_optimality_proven':False,
        'site_prose_or_variations_used_as_neural_labels':False,'source_license_or_redistribution_permission_established':False,
        'raw_records_or_archives_bundled_with_code':False,'training_data_or_neural_labels_generated':False}
    atomic_json(output/'verification.json',summary)
    config = {'acquisition':str(acquisition_path),'previous':list(map(str,previous)),'workers':workers,
              'seed':seed,'minimum_plies':minimum_plies,'max_record_bytes':max_record_bytes}
    atomic_json(output/'manifest.json',manifest('public_platform_recorded_games_native_import',config,
        [acquisition_path,*[item['path'] for item in inventory],*prior_inputs], [*paths.values(),output/'verification.json'],summary))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acquisition',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--previous',action='append',default=[])
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261051)
    parser.add_argument('--minimum-plies',type=int,default=20)
    args = parser.parse_args()
    print(json.dumps(import_exports(args.acquisition,args.output,args.previous,args.workers,args.seed,args.minimum_plies)),flush=True)


if __name__ == '__main__': main()
