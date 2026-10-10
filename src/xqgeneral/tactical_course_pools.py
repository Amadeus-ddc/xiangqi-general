"""Bound tactical terminal roots, separating source moves from legal extensions."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from .course_tasks import validate_context
from .evidence import atomic_json, digest, history_key, iter_jsonl, manifest, position_key
from .human_games import assigned_split
from .paper_curriculum import SPLITS, checked_sources, reference_contract
from .recorded_course_pools import (
    reference_explanations, root_footprint, source_owners, terminal_candidates,
)
from .recorded_search_inputs import BoundInputs, TACTICAL_KINDS, checked_game
from .rules import adjudicate, legal_moves, replay
from .symmetry import mirror_fen

DATA_KIND = 'tactical_native_terminal_course_root_pools'
POOL_PROFILE = 'tactical_terminal_targets_xiangqi_v1'
EXTENSION_ORIGIN = 'generated_legal_one_ply_terminal_extension'


def source_candidates(game):
    """Do not read a puzzle's proposed solution or invent its missing history."""
    context = {'initial_fen': game['initial_fen'], 'moves': game['moves'], 'history': game['history'],
               'fen': game['history'][-1], 'feature_key': history_key(game['history']), 'future_moves': []}
    if adjudicate(game['initial_fen'], game['moves']) != game['native_terminal']:
        raise ValueError('Tactical source outcome differs from its supplied native history')
    if game['source_kind'] == 'recorded_tactical_line':
        if not game['native_terminal']['ended']:
            validate_context(context, check_future=True)
            return {'kind': 'nonterminal_recorded_line', 'roots': [], 'extensions': Counter()}
        result = terminal_candidates(game, game['split'])
        for root in result['roots']:
            root['paper_terminal_line_origin'] = 'actual_recorded_source_moves'
        return dict(result, extensions=Counter())
    if game['native_terminal']['ended']:
        result = terminal_candidates(game, game['split'])
        for root in result['roots']:
            root['paper_terminal_line_origin'] = 'actual_supplied_terminal_position'
        return dict(result, extensions=Counter())
    validate_context(context, check_future=True)
    roots, extensions = [], Counter()
    for move in sorted(legal_moves(game['initial_fen'])):
        history = replay(game['initial_fen'], [move])
        if legal_moves(history[-1]):
            continue
        native = adjudicate(game['initial_fen'], [move])
        winner = 'red' if game['initial_fen'].split()[1] == 'w' else 'black'
        if not native['ended'] or native['winner'] != winner:
            raise ValueError('Generated legal terminal extension has an inconsistent native winner')
        generated = dict(game, moves=[move], history=history, native_terminal=native,
                         provenance=game['provenance'] + ';' + EXTENSION_ORIGIN)
        candidates = terminal_candidates(generated, game['split'])
        extensions[candidates['kind']] += 1
        for root in candidates['roots']:
            root['paper_recorded_mate_witness'] = []
            root['paper_generated_terminal_witness'] = [move] if root['ply'] == 0 else []
            root['paper_terminal_line_origin'] = EXTENSION_ORIGIN
            root['generated_terminal_move'] = move
            root['extension_is_recorded_source_move'] = False
            root['recorded_future_moves_available'] = False
            root['provenance'] = game['provenance'] + ';' + EXTENSION_ORIGIN
            for target in root['paper_source_targets']:
                target['plies_before_generated_mate'] = target.pop('plies_before_recorded_mate')
            roots.append(root)
    return {'kind': 'isolated_position', 'roots': roots, 'extensions': extensions}


def candidate_groups(games, workers):
    if workers == 1:
        for game in games:
            yield game, source_candidates(game)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        for game in games:
            pending.append((game, pool.submit(source_candidates, game)))
            if len(pending) >= workers * 2:
                source, job = pending.popleft()
                yield source, job.result()
        for source, job in pending:
            yield source, job.result()


def bound_forecasts(proof_path, bound):
    """Reuse completed native forecasts while freshly binding all original bytes."""
    proof_path = Path(proof_path)
    proof = bound.json(proof_path)
    if (proof.get('kind') != 'recorded_coach_footprints' or proof.get('status') != 'complete' or
            proof['verification'].get('recorded_futures_and_prior_label_structured_answer_pvs_included') is not True):
        raise ValueError('Tactical pools require completed native course and structured-label forecasts')
    for name, expected in {**proof['inputs'], **proof['outputs']}.items():
        bound.bind(Path(name), expected)
    summary_path = proof_path.parent / 'verification.json'
    if (not any(Path(k).resolve() == summary_path.resolve() for k in proof['outputs']) or
            bound.json(summary_path) != proof['verification']):
        raise ValueError('Completed native forecast verification changed')
    for directory in (proof['config']['data'], proof['config']['prior_labels']):
        for name in ('manifest.json', *(f'{s}.jsonl' for s in SPLITS)):
            if str(Path(directory) / name) not in proof['inputs']:
                raise ValueError('Completed forecasts omitted an original course or label input')
    path = proof_path.parent / 'footprints.json'
    expected = proof['outputs'].get(str(path))
    if expected is None:
        # Relative manifests may be invoked using an absolute manifest path.
        matches = [v for k, v in proof['outputs'].items() if Path(k).resolve() == path.resolve()]
        if len(matches) != 1:
            raise ValueError('Native forecast output is not bound by its completed manifest')
        expected = matches[0]
    value = bound.json(path, expected)
    fields = ('course_games', 'course_positions', 'prior_label_games', 'prior_label_positions',
              'combined_games', 'combined_positions')
    if set(value) != set(fields):
        raise ValueError('Native forecast set contract changed')
    sets = {}
    for field in fields:
        if set(value[field]) != set(SPLITS):
            raise ValueError('Native forecast split contract changed')
        sets[field] = {}
        for split, rows in value[field].items():
            if (not isinstance(rows, list) or not rows or
                    any(not isinstance(v, str) or not v for v in rows) or len(set(rows)) != len(rows)):
                raise ValueError('Native forecasts contain empty, duplicate or malformed split members')
            sets[field][split] = set(rows)
    games, positions = {}, {}
    for split in SPLITS:
        for kind in ('games', 'positions'):
            if sets['combined_' + kind][split] != (sets['course_' + kind][split] |
                                                    sets['prior_label_' + kind][split]):
                raise ValueError('Completed combined forecasts differ from their component union')
        if (len(sets['course_games'][split]) != proof['verification']['course_game_counts'][split] or
                len(sets['course_positions'][split]) !=
                proof['verification']['course_root_and_forecast_position_counts'][split]):
            raise ValueError('Completed forecast counts changed')
        for key in sets['combined_games'][split]:
            if games.setdefault(key, split) != split:
                raise ValueError('Native forecast game owners overlap')
        for key in sets['combined_positions'][split]:
            if position_key(key) != key or positions.setdefault(key, split) != split:
                raise ValueError('Native forecast position owners overlap or are malformed')
    # The completed extractor stores actual contexts, not a color closure.
    # Derive missing counterparts here and reject conflicting existing owners.
    for key, split in list(positions.items()):
        color = position_key(mirror_fen(key))
        if positions.setdefault(color, split) != split:
            raise ValueError('Native forecast color counterpart overlaps another split')
    return games, positions


def imported_games(directories, bound, *, tactical_only=False, split_seed=20261051):
    seen = {}
    for directory in directories:
        directory = Path(directory)
        proof_path, path = directory / 'manifest.json', directory / 'games.jsonl'
        proof = bound.json(proof_path)
        matches = [v for k, v in proof.get('outputs', {}).items() if Path(k).resolve() == path.resolve()]
        if proof.get('status') != 'complete' or len(matches) != 1:
            raise ValueError('Source import is not a complete bound canonical game archive')
        bound.bind(path, matches[0])
        if tactical_only and (proof.get('kind') not in ('recorded_tactical_lines_native_import',
                                                      'isolated_tactical_positions_native_import') or
                              proof['config'].get('seed') != split_seed):
            raise ValueError('Tactical import kind or preassigned split seed changed')
        for game in iter_jsonl(path):
            identity = (game['initial_fen'], tuple(game['moves']), tuple(game['history']))
            key = game['game_id']
            if key in seen:
                if seen[key] != identity:
                    raise ValueError('Duplicate source game identifier has conflicting supplied history')
                continue
            seen[key] = identity
            if tactical_only:
                checked_game(game, split_seed)
                expected_kind = ('recorded_tactical_line' if proof['kind'] == 'recorded_tactical_lines_native_import'
                                 else 'recorded_tactical_position')
                if game['source_kind'] not in TACTICAL_KINDS or game['source_kind'] != expected_kind:
                    raise ValueError('Tactical pool source has an unexpected declared source kind')
            yield game


def prepare(config):
    bound = BoundInputs()
    games, positions = bound_forecasts(config['footprints'], bound)
    references, other_games, other_positions = reference_contract(config['reference_data'])
    for key, split in other_games.items():
        if games.setdefault(key, split) != split:
            raise ValueError('Additional prior corpus has conflicting game ownership')
    for key, split in other_positions.items():
        if positions.setdefault(key, split) != split:
            raise ValueError('Additional prior corpus has conflicting future or color ownership')
    references += reference_explanations(config['explanation_reference_data'], games, positions)
    for path in references:
        bound.bind(path)
    features = set()
    if config['owners']:
        paths = checked_sources([config['owners']], [config['owner_manifest']])
        for path in paths:
            bound.bind(path)
        owners, features = source_owners(config['owners'])
        for key, split in owners.items():
            if games.setdefault(key, split) != split:
                raise ValueError('Existing root owners conflict with complete prior forecasts')
    # Original imports can share opening geometry. Block ambiguous positions;
    # do not move a recorded game to another split or silently pick one owner.
    for game in imported_games(config['reservation_games'], bound):
        split = games.get(game['game_id'], assigned_split(game['game_id'], config['split_seed']))
        if len(game['history']) != len(game['moves']) + 1 or game['history'][0] != game['initial_fen']:
            raise ValueError('Canonical reservation history dimensions changed')
        for fen in game['history']:
            for key in (position_key(fen), position_key(mirror_fen(fen))):
                if positions.setdefault(key, split) != split:
                    positions[key] = 'ambiguous_source_splits'
    selected = list(imported_games(config['games'], bound, tactical_only=True, split_seed=config['split_seed']))
    selected.sort(key=lambda g: (('validation', 'test', 'train').index(g['split']), g['game_id']))
    if not selected or any(not any(g['split'] == split for g in selected) for split in SPLITS):
        raise ValueError('Complete tactical sources require every preassigned split')
    counters = {k: Counter() for k in ('sources', 'extensions', 'roots', 'excluded', 'targets', 'missing_history')}
    counters['games'] = {s: set() for s in SPLITS}
    return bound, selected, games, positions, features, counters


def records(selected, games, positions, features, counters, workers):
    for game, result in candidate_groups(selected, workers):
        split, key = game['split'], game['game_id']
        if key in games and games[key] != split:
            raise ValueError('Tactical source ownership conflicts with its prior corpus')
        counters['sources'][f'{split}/{result["kind"]}'] += 1
        for kind, n in result['extensions'].items():
            counters['extensions'][f'{split}/{kind}'] += n
        for root in result['roots']:
            root['paper_source_pool_profile'] = POOL_PROFILE
            footprint = root_footprint(root)
            reason = ('duplicate_full_history_root' if root['feature_key'] in features else
                      'root_future_witness_or_color_overlaps_another_split'
                      if any(p in positions and positions[p] != split for p in footprint) else None)
            if reason:
                counters['excluded'][reason] += 1
                yield 'excluded', {'game_id': key, 'split': split, 'feature_key': root['feature_key'],
                    'ply': root['ply'], 'line_origin': root['paper_terminal_line_origin'],
                    'generated_terminal_move': root.get('generated_terminal_move'), 'reason': reason}
                continue
            features.add(root['feature_key'])
            positions.update({p: split for p in footprint})
            counters['roots'][split] += 1
            counters['games'][split].add(key)
            counters['missing_history'][f'{split}/{root["pre_fragment_game_history_available"]}'] += 1
            for target in root['paper_source_targets']:
                stage = 'current' if target['horizon'] == 0 else 'future'
                for kind in target['classes']:
                    counters['targets'][f'{split}/{stage}/{kind}'] += 1
            yield 'roots', root


def verification(counters, config):
    return {'status': 'complete', 'evidence_state': 'reconstructed_baseline', 'pool_profile': POOL_PROFILE,
            'supplied_sources_by_split_kind': dict(counters['sources']),
            'generated_legal_one_ply_terminal_moves_by_split_class': dict(counters['extensions']),
            'new_roots_by_split': dict(counters['roots']),
            'new_source_ids_by_split': {s: len(v) for s, v in counters['games'].items()},
            'root_counts_by_supplied_prehistory': dict(counters['missing_history']),
            'target_class_observations_by_split_stage': dict(counters['targets']),
            'excluded_candidates_by_reason': dict(counters['excluded']),
            'all_supplied_tactical_histories_and_generated_extensions_natively_checked': True,
            'recorded_and_generated_terminal_lines_explicitly_distinguished': True,
            'missing_pre_fragment_history_preserved': True, 'source_game_ids_and_splits_preserved': True,
            'prior_completed_native_forecasts_reused_with_all_original_bytes_freshly_bound': True,
            'prior_forecast_native_parser_reexecuted_for_cached_sources': False,
            'cached_forecast_color_counterparts_derived_and_reserved_by_this_producer': True,
            'canonical_source_history_and_color_reservations_applied': bool(config['reservation_games']),
            'prior_root_future_witness_color_isolation_verified': True,
            'generated_extensions_are_additional_independent_games': False,
            'mate_distance_proves_forced_mate_under_optimal_defense': False,
            'target_observations_are_independent_positions': False,
            'paper_difficult_source_mix_applied': False, 'question_corpus_generated': False,
            'feature_cache_extended': False, 'student_or_teacher_loaded': False,
            'student_training_executed': False, 'student_strength_or_prose_quality_measured': False}


def build(games, footprints, output, *, reference_data=(), explanation_reference_data=(),
          reservation_games=(), owners=None, owner_manifest=None, workers=1, split_seed=20261051):
    output = Path(output)
    if (output.exists() or type(workers) is not int or workers < 1 or type(split_seed) is not int or
            not games or bool(owners) != bool(owner_manifest)):
        raise ValueError('Use a fresh tactical pool, bound sources and a positive worker count')
    config = {'games': list(map(str, games)), 'footprints': str(footprints), 'output': str(output),
              'reference_data': list(map(str, reference_data)),
              'explanation_reference_data': list(map(str, explanation_reference_data)),
              'reservation_games': list(map(str, reservation_games or games)),
              'owners': str(owners) if owners else None, 'owner_manifest': str(owner_manifest) if owners else None,
              'workers': workers, 'split_seed': split_seed, 'pool_profile': POOL_PROFILE}
    bound, selected, prior_games, positions, features, counters = prepare(config)
    output.mkdir(parents=True)
    paths = {kind: output / f'{kind}.jsonl' for kind in ('roots', 'excluded')}
    with paths['roots'].open('x', encoding='utf-8') as roots, paths['excluded'].open('x', encoding='utf-8') as excluded:
        handles = {'roots': roots, 'excluded': excluded}
        for kind, row in records(selected, prior_games, positions, features, counters, workers):
            handles[kind].write(json.dumps(row, ensure_ascii=False) + '\n')
    bound.unchanged()
    result = verification(counters, config)
    atomic_json(output / 'verification.json', result)
    atomic_json(output / 'manifest.json', manifest(DATA_KIND, config, bound.artifacts,
                                                 [*paths.values(), output / 'verification.json'], result))
    return result


def readback(data, output, workers=1):
    data, output = Path(data), Path(output)
    if output.exists() or type(workers) is not int or workers < 1:
        raise ValueError('Use a fresh tactical pool readback and a positive worker count')
    bound = BoundInputs()
    proof_path = data / 'manifest.json'
    proof = bound.json(proof_path)
    if (proof.get('kind') != DATA_KIND or proof.get('status') != 'complete' or
            proof['config'].get('pool_profile') != POOL_PROFILE or
            Path(proof['config']['output']).resolve() != data.resolve()):
        raise ValueError('Readback requires a completed declared tactical terminal pool')
    for name, expected in {**proof['inputs'], **proof['outputs']}.items():
        bound.bind(name, expected)
    prepared, selected, games, positions, features, counters = prepare(proof['config'])
    checked = Counter()
    with (data / 'roots.jsonl').open(encoding='utf-8') as roots, (data / 'excluded.jsonl').open(encoding='utf-8') as excluded:
        handles = {'roots': roots, 'excluded': excluded}
        for kind, row in records(selected, games, positions, features, counters, workers):
            if handles[kind].readline() != json.dumps(row, ensure_ascii=False) + '\n':
                raise ValueError('Tactical terminal root or exclusion differs from its native source')
            checked[kind] += 1
        if any(handle.read(1) for handle in handles.values()):
            raise ValueError('Tactical terminal pool has extra records')
    if (verification(counters, proof['config']) != proof['verification'] or
            json.loads((data / 'verification.json').read_text()) != proof['verification']):
        raise ValueError('Tactical terminal verification counts or source claims changed')
    bound.unchanged()
    prepared.unchanged()
    result = {'status': 'complete', 'records_regenerated_and_compared': dict(checked),
              'all_source_and_output_bytes_checked': True, 'all_tactical_native_targets_and_exclusions_rederived': True,
              'prior_root_future_witness_color_isolation_rechecked': True,
              'prior_completed_native_forecasts_reused': True, 'cached_forecast_native_parser_reexecuted': False,
              'source_selection_algorithm_independently_rewritten': False,
              'student_or_teacher_loaded': False, 'student_training_executed': False}
    output.mkdir(parents=True)
    atomic_json(output / 'verification.json', result)
    atomic_json(output / 'manifest.json', manifest('tactical_terminal_pool_complete_source_readback',
        {'data': str(data), 'output': str(output), 'workers': workers}, bound.artifacts,
        [output / 'verification.json'], result))
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--games', nargs='+')
    group.add_argument('--readback')
    parser.add_argument('--footprints')
    parser.add_argument('--reference-data', nargs='+', default=[])
    parser.add_argument('--explanation-reference-data', nargs='+', default=[])
    parser.add_argument('--reservation-games', nargs='+', default=[])
    parser.add_argument('--owners')
    parser.add_argument('--owner-manifest')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--split-seed', type=int, default=20261051)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.readback:
        if any((args.footprints, args.reference_data, args.explanation_reference_data,
                args.reservation_games, args.owners, args.owner_manifest)) or args.split_seed != 20261051:
            parser.error('Readback uses its saved source and split contracts')
        result = readback(args.readback, args.output, args.workers)
    else:
        if not args.footprints:
            parser.error('Production requires completed native course and structured-label footprints')
        result = build(args.games, args.footprints, args.output, reference_data=args.reference_data,
            explanation_reference_data=args.explanation_reference_data, reservation_games=args.reservation_games,
            owners=args.owners, owner_manifest=args.owner_manifest, workers=args.workers, split_seed=args.split_seed)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
