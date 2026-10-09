"""Fixed recorded-game opening suites for validation or final test matches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .evidence import atomic_json, history_key, iter_jsonl, manifest, position_key, write_jsonl
from .human_games import assigned_split
from .recorded_search_inputs import BoundInputs
from .rules import adjudicate, replay


MATCH_KINDS = {'recorded_human_match', 'published_recorded_match',
               'recorded_computer_match', 'recorded_human_computer_match'}
PACK_KIND = 'heldout_recorded_match_openings'


def checked_opening(row):
    """Check the supplied past without using recorded answers or future moves."""
    if (not isinstance(row, dict) or
            any(not isinstance(row.get(k), str) or not row[k]
                for k in ['id', 'game_id', 'initial_fen', 'fen', 'feature_key']) or
            row.get('split') not in {'validation', 'test'} or
            not isinstance(row.get('moves'), list) or
            any(not isinstance(m, str) for m in row['moves']) or
            not isinstance(row.get('history'), list)):
        raise ValueError('Opening requires a named protected split and complete supplied history')
    history = replay(row['initial_fen'], row['moves'])
    if (history != row['history'] or history[-1] != row['fen'] or
            history_key(history) != row['feature_key']):
        raise ValueError('Opening differs from its native complete history')
    return row


def prepare_openings(data, output, *, split='validation', count=32, plies=12, seed=20261079):
    if (split not in {'validation', 'test'} or type(count) is not int or count <= 0 or
            type(plies) is not int or plies <= 0 or type(seed) is not int):
        raise ValueError('Require a protected split and positive integer opening budgets')
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError('Preserve opening suites; use a fresh output directory')
    bound = BoundInputs()
    parent_path, source = data / 'manifest.json', data / f'{split}.jsonl'
    parent = bound.json(parent_path)
    if (parent.get('status') != 'complete' or
            parent.get('kind') != 'isolated_recorded_engine_four_course_questions' or
            type(parent.get('config', {}).get('split_seed')) is not int):
        raise ValueError('Require the completed isolated recorded curriculum source')
    expected = [item for name, item in parent.get('outputs', {}).items()
                if Path(name).resolve() == source.resolve()]
    if len(expected) != 1:
        raise ValueError('Protected source is absent from the completed curriculum output bindings')
    bound.bind(source, expected[0])
    config = {'data': str(data), 'split': split, 'count': count, 'plies': plies, 'seed': seed}
    output.mkdir(parents=True)
    games, rows_read, excluded = {}, 0, Counter()
    try:
        for row in iter_jsonl(source):
            rows_read += 1
            if row.get('split') != split:
                raise ValueError('Protected source contains a different split')
            if (row.get('augmentation_parent') or row.get('stage') != 'static_current' or
                    row.get('recorded_source_kind') not in MATCH_KINDS):
                excluded['derived_or_noncurrent_or_nonmatch_rows'] += 1
                continue
            game = row.get('game_id')
            if (not isinstance(game, str) or not game.startswith('recorded-') or
                    row.get('recorded_source_game_id') != game or
                    assigned_split(game, parent['config']['split_seed']) != split or
                    not isinstance(row.get('id'), str) or not row['id'] or
                    not isinstance(row.get('moves'), list) or
                    any(not isinstance(m, str) for m in row['moves']) or
                    not isinstance(row.get('history'), list) or
                    len(row['history']) != len(row['moves']) + 1 or
                    not isinstance(row.get('initial_fen'), str) or
                    row['history'][0] != row['initial_fen'] or
                    row['history'][-1] != row.get('fen') or
                    history_key(row['history']) != row.get('feature_key')):
                raise ValueError('Original recorded-game identity or history binding changed')
            if len(row['moves']) < plies:
                excluded['short_supplied_histories'] += 1
                continue
            history = row['history'][:plies + 1]
            key = history_key(history)
            opening = {'id': 'match-opening-' + key, 'game_id': game, 'split': split,
                       'initial_fen': row['initial_fen'], 'moves': row['moves'][:plies],
                       'history': history, 'fen': history[-1], 'feature_key': key,
                       'recorded_source_kind': row['recorded_source_kind'],
                       'source_row_id': row['id'], 'source_root_feature_key': row['feature_key']}
            previous = games.get(game)
            if previous is not None:
                if any(previous[k] != opening[k] for k in
                       ['initial_fen', 'moves', 'history', 'recorded_source_kind']):
                    raise ValueError('One recorded game declares inconsistent opening prefixes')
                if (previous['source_root_feature_key'], previous['source_row_id']) <= (
                        opening['source_root_feature_key'], opening['source_row_id']):
                    continue
            games[game] = opening
        ordered = sorted(games.values(), key=lambda row: (
            hashlib.sha256(f"{seed}/{row['game_id']}".encode()).hexdigest(), row['game_id']))
        selected, positions, histories = [], set(), set()
        for row in ordered:
            checked_opening(row)
            if adjudicate(row['initial_fen'], row['moves'])['ended']:
                excluded['terminal_opening_histories'] += 1
                continue
            position = position_key(row['fen'])
            if position in positions or row['feature_key'] in histories:
                excluded['duplicate_opening_positions_or_histories'] += 1
                continue
            selected.append(row);positions.add(position);histories.add(row['feature_key'])
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError('Insufficient distinct active recorded openings for the requested budget')
        bound.unchanged()
        proof = {'status': 'complete', 'split': split, 'source_rows_read': rows_read,
                 'eligible_original_recorded_games': len(games), 'openings': len(selected),
                 'distinct_source_games': len({r['game_id'] for r in selected}),
                 'distinct_opening_histories': len(histories), 'distinct_opening_positions': len(positions),
                 'opening_plies': plies, 'by_source_declared_kind': dict(Counter(
                     r['recorded_source_kind'] for r in selected)), 'excluded': dict(excluded),
                 'selected_full_prefixes_native_checked': True, 'answers_or_recorded_future_used': False,
                 'source_import_and_global_split_proof_reused': True,
                 'opening_geometry_unseen_during_training_proven': False,
                 'source_player_identities_authenticated': False, 'new_training_labels': 0,
                 'student_or_teacher_model_loaded': False}
        write_jsonl(output / 'openings.jsonl', selected)
        atomic_json(output / 'counts.json', proof)
        completed = manifest(PACK_KIND, config, (),
                             [output / 'openings.jsonl', output / 'counts.json'], proof)
        completed['inputs'] = bound.artifacts
        bound.unchanged();atomic_json(output / 'manifest.json', completed)
        return proof
    except Exception as error:
        atomic_json(output / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                            'preserve_partial_outputs': True})
        raise


def load_openings(directory):
    directory = Path(directory)
    bound = BoundInputs();proof = bound.json(directory / 'manifest.json')
    if (proof.get('status') != 'complete' or proof.get('kind') != PACK_KIND or
            proof.get('config', {}).get('split') not in {'validation', 'test'} or
            not isinstance(proof['config'].get('data'), str)):
        raise ValueError('Require a completed protected recorded opening suite')
    data = Path(proof['config']['data'])
    source, parent_path = data / f"{proof['config']['split']}.jsonl", data / 'manifest.json'
    if set(proof.get('inputs', {})) != {str(source), str(parent_path)}:
        raise ValueError('Opening suite lost its original protected source bindings')
    for name, item in proof.get('inputs', {}).items():
        bound.bind(name, item)
    rows_path, counts_path = directory / 'openings.jsonl', directory / 'counts.json'
    for path in [rows_path, counts_path]:
        if str(path) not in proof.get('outputs', {}):
            raise ValueError('Completed suite output bindings are missing')
        bound.bind(path, proof['outputs'][str(path)])
    rows = list(iter_jsonl(rows_path))
    counts = bound.json(counts_path)
    if (counts != proof.get('verification') or type(proof['config'].get('count')) is not int or
            len(rows) != proof['config']['count'] or len(rows) != counts.get('openings') or not rows or
            type(proof['config'].get('plies')) is not int or proof['config']['plies'] <= 0):
        raise ValueError('Opening suite count or verification changed')
    for row in rows:
        checked_opening(row)
        if (row['split'] != proof['config']['split'] or len(row['moves']) != proof['config']['plies'] or
                row.get('recorded_source_kind') not in MATCH_KINDS or
                adjudicate(row['initial_fen'], row['moves'])['ended']):
            raise ValueError('Opening suite lost its protected split, prefix budget or active history')
    if any(len({r[k] for r in rows}) != len(rows) for k in ['id', 'game_id', 'feature_key']) or \
            len({position_key(r['fen']) for r in rows}) != len(rows):
        raise ValueError('Opening suite repeats a game, history, position or identifier')
    parent = bound.json(parent_path)
    expected = [item for name, item in parent.get('outputs', {}).items()
                if Path(name).resolve() == source.resolve()]
    if (parent.get('status') != 'complete' or
            parent.get('kind') != 'isolated_recorded_engine_four_course_questions' or
            expected != [proof['inputs'][str(source)]]):
        raise ValueError('Opening suite differs from its completed curriculum source')
    by_source = {r.get('source_row_id'): r for r in rows}
    if len(by_source) != len(rows) or any(not isinstance(k, str) or not k for k in by_source):
        raise ValueError('Opening suite lost its selected original row identities')
    found, scanned = set(), 0
    for original in iter_jsonl(source):
        scanned += 1
        if original.get('split') != proof['config']['split']:
            raise ValueError('Protected source contains a different split')
        key = original.get('id')
        if key not in by_source:
            continue
        row, plies = by_source[key], proof['config']['plies']
        if (key in found or original.get('augmentation_parent') or
                original.get('stage') != 'static_current' or
                original.get('recorded_source_kind') != row['recorded_source_kind'] or
                original.get('game_id') != row['game_id'] or
                original.get('recorded_source_game_id') != row['game_id'] or
                original.get('feature_key') != row.get('source_root_feature_key') or
                original.get('initial_fen') != row['initial_fen'] or
                original['moves'][:plies] != row['moves'] or original['history'][:plies + 1] != row['history']):
            raise ValueError('Opening prefix or provenance differs from its exact protected source row')
        found.add(key)
    if found != set(by_source) or scanned != counts.get('source_rows_read'):
        raise ValueError('Opening suite does not cover its declared source rows')
    bound.unchanged()
    return rows, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True);parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--count', type=int, default=32);parser.add_argument('--plies', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261079)
    args = parser.parse_args()
    print(json.dumps(prepare_openings(args.data, args.output, split=args.split,
        count=args.count, plies=args.plies, seed=args.seed)), flush=True)


if __name__ == '__main__':main()
