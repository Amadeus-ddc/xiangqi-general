"""All supplied prefixes from bound recorded games, with prior split ownership."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from .evidence import atomic_json, digest, history_key, iter_jsonl, manifest, position_key
from .paper_curriculum import SPLITS, checked_sources, reference_contract
from .recorded_course_pools import reference_explanations, source_owners
from .recorded_search_inputs import BoundInputs, checked_game, tactical_source_context
from .rules import START_FEN, adjudicate, in_check, legal_moves, replay
from .tactical_course_pools import bound_forecasts, imported_games
from .symmetry import mirror_fen

DATA_KIND = 'recorded_full_native_course_root_pools'
POOL_PROFILE = 'all_recorded_history_prefixes_xiangqi_v1'
FULL_KINDS = {'recorded_human_match', 'published_recorded_match',
              'recorded_computer_match', 'recorded_human_computer_match'}


def game_roots(game, split, min_ply=0):
    """Replay each complete source once; stop roots before any AXF history ending."""
    if type(min_ply) is not int or min_ply < 0 or split not in SPLITS or game['source_kind'] not in FULL_KINDS:
        raise ValueError('Full recorded roots require their native source kind, split and nonnegative starting ply')
    moves, history = game['moves'], game['history']
    if replay(game['initial_fen'], moves) != history or adjudicate(game['initial_fen'], moves) != game['native_terminal']:
        raise ValueError('Bound recorded game differs from its complete native history')
    end, first_end, history_ending = len(moves), None, False
    for ply in range(end + 1):
        if adjudicate(game['initial_fen'], moves[:ply])['ended']:
            first_end = ply
            history_ending = bool(legal_moves(history[ply]))
            break
    last = end if first_end is None else first_end - int(history_ending)
    mate = (first_end is not None and not history_ending and in_check(history[first_end]))
    original = game.get('source_initial_fen', game['headers'].get('FEN', START_FEN))
    if original != game['headers'].get('FEN', START_FEN):
        raise ValueError('Recorded source lost its original initial FEN')
    context = tactical_source_context({
        'source_initial_fen': original,
        'declared_participants': _participants(game['headers']),
        'supplied_history_starts_at_standard_initial_position': _standard(game),
        'pre_fragment_game_history_available': _standard(game),
        'provided_line_is_best_move_label': False,
        'source_comments_or_analysis_branches_used_as_labels': False,
    }, game['initial_fen'], game['headers'])
    roots = []
    for ply in range(min_ply, last + 1):
        past, prefix = moves[:ply], history[:ply + 1]
        future = moves[ply:min(last, ply + 8)]
        witness = moves[ply:first_end] if mate and first_end - ply <= 14 else []
        target_end = ply + len(witness or future)
        fens = history[ply:target_end + 1]
        footprint = {position_key(fen) for fen in fens}
        footprint.update(position_key(mirror_fen(fen)) for fen in fens)
        root = {'game_id': game['game_id'], 'split': split, 'source_archive_split': game['split'],
                'initial_fen': game['initial_fen'], 'moves': past, 'history': prefix,
                'fen': history[ply], 'feature_key': history_key(prefix), 'ply': ply,
                'future_moves': future, 'future_branches': [], 'paper_recorded_mate_witness': witness,
                'paper_source_pool_profile': POOL_PROFILE,
                'recorded_source_game_id': game['game_id'], 'recorded_source': game['source'],
                'recorded_source_kind': game['source_kind'], 'recorded_source_headers': game['headers'],
                'recorded_source_players': game.get('players', []),
                'recorded_source_provenance': game.get('provenance'),
                'source_first_native_terminal_ply': first_end,
                'source_first_native_terminal_has_legal_moves': history_ending,
                'recorded_future_moves_available': bool(future),
                'provenance': (game.get('provenance') or 'bound_canonical_source') + ';all_recorded_history_prefixes',
                **context}
        roots.append((root, footprint))
    return {'roots': roots, 'source_last_eligible_ply': last,
            'source_first_native_terminal_ply': first_end,
            'source_history_ending_with_legal_moves': history_ending,
            'source_board_terminal_kind': 'mate' if mate else 'stalemate' if first_end is not None and not history_ending else None}


def _participants(headers):
    from .recorded_sources import participant_label
    return {key: participant_label(headers[key]) if headers.get(key) else None for key in ('Red', 'Black')}


def _standard(game):
    original = game['headers'].get('FEN', START_FEN)
    return game['initial_fen'].split()[:2] == START_FEN.split()[:2] and original.split()[4:] == ['0', '1']


def _game_worker(item):
    return game_roots(*item)


def candidate_groups(games, owners, workers, min_ply):
    if workers == 1:
        for game in games:
            yield game, game_roots(game, owners[game['game_id']], min_ply)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        for game in games:
            pending.append((game, pool.submit(_game_worker, (game, owners[game['game_id']], min_ply))))
            if len(pending) >= workers * 2:
                source, job = pending.popleft()
                yield source, job.result()
        for source, job in pending:
            yield source, job.result()


def prepare(config):
    bound = BoundInputs()
    games, positions = bound_forecasts(config['footprints'], bound) if config['footprints'] else ({}, {})
    references, old_games, old_positions = reference_contract(config['reference_data'])
    for key, split in old_games.items():
        if games.setdefault(key, split) != split:
            raise ValueError('Prior recorded course game ownership conflicts')
    for key, split in old_positions.items():
        if positions.setdefault(key, split) != split:
            raise ValueError('Prior course root, future or color ownership conflicts')
    references += reference_explanations(config['explanation_reference_data'], games, positions)
    for path in references:
        bound.bind(path)
    if config['owners']:
        for path in checked_sources([config['owners']], [config['owner_manifest']]):
            bound.bind(path)
        owners, _ = source_owners(config['owners'])
        for key, split in owners.items():
            if games.setdefault(key, split) != split:
                raise ValueError('Existing recorded root owner conflicts with prior corpora')
    archives = [*config['games'], *config['reservation_games']]
    for game in imported_games(archives, bound):
        checked_game(game, config['split_seed'])
        split = games.get(game['game_id'], game['split'])
        for fen in game['history']:
            for key in (position_key(fen), position_key(mirror_fen(fen))):
                if positions.setdefault(key, split) != split:
                    positions[key] = 'ambiguous_source_splits'
    selected = []
    for game in imported_games(config['games'], bound):
        checked_game(game, config['split_seed'])
        if game['source_kind'] in FULL_KINDS:
            games.setdefault(game['game_id'], game['split'])
            selected.append(game)
    selected.sort(key=lambda g: (SPLITS.index(games[g['game_id']]), g['game_id']))
    if not selected or any(not any(games[g['game_id']] == split for g in selected) for split in SPLITS):
        raise ValueError('Full recorded root sources require every preassigned split')
    counters = {k: Counter() for k in ('sources', 'endings', 'roots', 'excluded', 'witnesses', 'sides', 'missing_history')}
    counters['games'] = {split: set() for split in SPLITS}
    return bound, selected, games, positions, counters


def records(selected, games, positions, counters, workers, min_ply):
    features = set()
    for game, result in candidate_groups(selected, games, workers, min_ply):
        split, key = games[game['game_id']], game['game_id']
        counters['sources'][f'{split}/{game["source_kind"]}'] += 1
        ending = 'history_ending_with_legal_moves' if result['source_history_ending_with_legal_moves'] else result['source_board_terminal_kind'] or 'legal_moves_remain'
        counters['endings'][f'{split}/{ending}'] += 1
        for root, footprint in result['roots']:
            reason = ('duplicate_full_history_root' if root['feature_key'] in features else
                      'root_future_witness_or_color_overlaps_another_split'
                      if any(p in positions and positions[p] != split for p in footprint) else None)
            if reason:
                counters['excluded'][reason] += 1
                yield 'excluded', {'game_id': key, 'split': split, 'feature_key': root['feature_key'],
                                   'ply': root['ply'], 'reason': reason}
                continue
            features.add(root['feature_key'])
            for p in footprint:
                positions.setdefault(p, split)
            counters['roots'][split] += 1
            counters['games'][split].add(key)
            counters['sides'][f'{split}/{root["fen"].split()[1]}'] += 1
            counters['missing_history'][f'{split}/{not root["pre_fragment_game_history_available"]}'] += 1
            if root['paper_recorded_mate_witness']:
                counters['witnesses'][split] += 1
            yield 'root', root


def verification(counters):
    return {'status': 'complete', 'evidence_state': 'reconstructed_baseline', 'pool_profile': POOL_PROFILE,
            'source_games_by_split_kind': dict(counters['sources']), 'source_endings_by_split_kind': dict(counters['endings']),
            'roots_by_split': dict(counters['roots']), 'games_contributing_roots_by_split': {s: len(v) for s, v in counters['games'].items()},
            'excluded_candidates_by_reason': dict(counters['excluded']), 'recorded_mate_witness_roots_by_split': dict(counters['witnesses']),
            'root_sides_by_split': dict(counters['sides']), 'missing_history_roots_by_split': dict(counters['missing_history']),
            'complete_source_histories_and_all_native_prefix_endings_recomputed': True,
            'all_eligible_source_prefixes_examined_without_per_game_sampling_cap': True,
            'actual_future_terminal_roots_and_fourteen_ply_mate_witnesses_retained': True,
            'prior_root_future_witness_color_and_all_source_history_ownership_reserved': True,
            'prior_structured_explanation_pvs_and_branches_reserved': True,
            'prior_contexts_may_be_reused_only_in_their_existing_split': True,
            'root_counts_are_distinct_query_boards': False, 'recorded_moves_used_as_best_move_labels': False,
            'source_history_rewritten_or_missing_past_invented': False, 'paper_difficult_source_mix_applied': False,
            'question_corpus_generated': False, 'feature_cache_extended': False, 'models_loaded': False,
            'student_training_executed': False, 'student_strength_or_prose_quality_measured': False,
            'independent_test_used_for_training_or_selection': False}


def build(games, output, *, footprints=None, reference_data=(), explanation_reference_data=(),
          owners=None, owner_manifest=None, reservation_games=(), split_seed=20261051, min_ply=0, workers=8):
    output = Path(output)
    if (output.exists() or not games or type(workers) is not int or workers < 1 or
            type(min_ply) is not int or min_ply < 0 or type(split_seed) is not int or
            bool(owners) != bool(owner_manifest)):
        raise ValueError('Use fresh full-recorded pools, complete sources, owner binding and positive workers')
    config = {'games': [str(p) for p in games], 'output': str(output), 'footprints': str(footprints) if footprints else None,
              'reference_data': [str(p) for p in reference_data], 'explanation_reference_data': [str(p) for p in explanation_reference_data],
              'owners': str(owners) if owners else None, 'owner_manifest': str(owner_manifest) if owner_manifest else None,
              'reservation_games': [str(p) for p in reservation_games], 'split_seed': split_seed, 'min_ply': min_ply, 'workers': workers}
    bound, selected, game_owners, positions, counters = prepare(config)
    output.mkdir(parents=True)
    paths = {'root': output / 'roots.jsonl', 'excluded': output / 'excluded.jsonl'}
    handles = {kind: path.open('x', encoding='utf-8') for kind, path in paths.items()}
    try:
        for kind, row in records(selected, game_owners, positions, counters, workers, min_ply):
            handles[kind].write(json.dumps(row, ensure_ascii=False) + '\n')
    finally:
        for handle in handles.values():
            handle.close()
    bound.unchanged()
    result = verification(counters)
    atomic_json(output / 'manifest.json', manifest(DATA_KIND, config, bound.artifacts, paths.values(), result))
    return result


def readback(data, output, *, workers=None):
    data, output = Path(data), Path(output)
    proof_path = data / 'manifest.json'
    proof = json.loads(proof_path.read_text())
    if output.exists() or proof.get('status') != 'complete' or proof.get('kind') != DATA_KIND:
        raise ValueError('Use a fresh full-source readback and complete producer')
    for name, expected in {**proof['inputs'], **proof['outputs']}.items():
        if digest(name) != expected['sha256'] or Path(name).stat().st_size != expected['bytes']:
            raise ValueError('Recorded full-source input or root pool changed')
    config = proof['config']
    count = config['workers'] if workers is None else workers
    if type(count) is not int or count < 1:
        raise ValueError('Positive full-source native readback workers required')
    bound, selected, games, positions, counters = prepare(config)
    handles = {kind: (data / filename).open(encoding='utf-8') for kind, filename in
               [('root', 'roots.jsonl'), ('excluded', 'excluded.jsonl')]}
    checked = Counter()
    try:
        for kind, row in records(selected, games, positions, counters, count, config['min_ply']):
            if handles[kind].readline() != json.dumps(row, ensure_ascii=False) + '\n':
                raise ValueError('Full native source root or exclusion bytes differ from regeneration')
            checked[kind] += 1
        if any(handle.readline() for handle in handles.values()):
            raise ValueError('Full-source pool contains an unbound tail')
    finally:
        for handle in handles.values():
            handle.close()
    if verification(counters) != proof['verification']:
        raise ValueError('Full-source root counts or native provenance policy changed')
    bound.unchanged()
    result = {'status': 'complete', 'pool_profile': POOL_PROFILE, 'roots_and_exclusions_regenerated': dict(checked),
              'all_source_input_and_pool_bytes_checked': True, 'source_native_rules_and_root_selection_shared_with_producer': True,
              'models_loaded': False, 'student_training_executed': False, 'student_strength_or_prose_quality_measured': False}
    output.mkdir(parents=True)
    atomic_json(output / 'verification.json', result)
    atomic_json(output / 'manifest.json', manifest('recorded_full_native_course_root_readback',
        {'data': str(data), 'output': str(output), 'workers': count}, [proof_path, *proof['inputs'], *proof['outputs']],
        [output / 'verification.json'], result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+', default=[])
    parser.add_argument('--footprints')
    parser.add_argument('--reference-data', nargs='*', default=[])
    parser.add_argument('--explanation-reference-data', nargs='*', default=[])
    parser.add_argument('--reservation-games', nargs='*', default=[])
    parser.add_argument('--owners')
    parser.add_argument('--owner-manifest')
    parser.add_argument('--split-seed', type=int, default=20261051)
    parser.add_argument('--min-ply', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--readback')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = (readback(args.readback, args.output, workers=args.workers) if args.readback else
              build(args.games, args.output, footprints=args.footprints, reference_data=args.reference_data,
                    explanation_reference_data=args.explanation_reference_data, owners=args.owners,
                    owner_manifest=args.owner_manifest, reservation_games=args.reservation_games,
                    split_seed=args.split_seed, min_ply=args.min_ply, workers=args.workers))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
