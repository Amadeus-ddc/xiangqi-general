"""Recover real recorded terminal and near-mate roots without changing splits."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from .course_tasks import validate_context
from .evidence import atomic_json, digest, history_key, iter_jsonl, manifest, position_key
from .paper_curriculum import SPLITS, checked_sources, reference_contract
from .recorded_coach import footprint_records
from .rules import adjudicate, gives_check, in_check, legal_moves, replay
from .symmetry import mirror_fen

DATA_KIND = 'recorded_native_terminal_course_root_pools'
POOL_PROFILE = 'recorded_terminal_targets_xiangqi_v1'
MAX_FUTURE = 8
NEAR_MATE_PLIES = 6


def native_classes(fen, mate_distance=None):
    """Classes describe the queried board; near-mate needs an actual source witness."""
    moves, check = legal_moves(fen), in_check(fen)
    classes = ['generic']
    if any(gives_check(fen, move) for move in moves):
        classes.append('have_check')
    if check:
        classes.append('in_check')
    if not moves:
        classes.append('mate' if check else 'stalemate')
    if check and mate_distance is not None and 1 <= mate_distance <= NEAR_MATE_PLIES:
        classes.append('near_mate_check')
    return classes


def terminal_candidates(game, split):
    """Use complete actual histories, including their final native terminal board."""
    moves, history = game['moves'], game['history']
    if (split not in SPLITS or len(history) != len(moves) + 1 or
            not history or history[0] != game['initial_fen']):
        raise ValueError('Recorded terminal source has an invalid history or assigned split')
    if replay(game['initial_fen'], moves) != history:
        raise ValueError('Recorded terminal history differs from its actual native moves')
    native = adjudicate(game['initial_fen'], moves)
    if native != game['native_terminal'] or not native['ended']:
        raise ValueError('Recorded terminal metadata differs from the native full history')
    if legal_moves(history[-1]):
        for index in range(len(moves)):
            if adjudicate(game['initial_fen'], moves[:index])['ended']:
                raise ValueError('Recorded history crosses a terminal prefix')
        return {'kind': 'history_ending_with_legal_moves', 'roots': []}
    # Validate every past prefix once, before deriving any current/future target.
    end = len(moves)
    complete = {'initial_fen': game['initial_fen'], 'moves': moves, 'history': history,
                'fen': history[-1], 'feature_key': history_key(history), 'future_moves': []}
    validate_context(complete, check_future=True)
    mate = in_check(history[-1])
    kind = 'mate' if mate else 'stalemate'
    roots = []
    # Eight future plies can reach a board at most six plies before actual mate.
    for ply in range(max(0, end - MAX_FUTURE - NEAR_MATE_PLIES), end + 1):
        past, prefix = moves[:ply], history[:ply + 1]
        future = moves[ply:min(end, ply + MAX_FUTURE)]
        witness = moves[ply:] if mate else []
        targets = []
        for horizon in range(len(future) + 1):
            distance = end - ply - horizon if mate else None
            targets.append({'horizon': horizon, 'fen': history[ply + horizon],
                            'classes': native_classes(history[ply + horizon], distance),
                            'plies_before_recorded_mate': distance})
        root = {'game_id': game['game_id'], 'split': split, 'initial_fen': game['initial_fen'],
                'moves': past, 'history': prefix, 'fen': history[ply], 'feature_key': history_key(prefix),
                'ply': ply, 'future_moves': future, 'future_branches': [],
                'recorded_source': game['source'], 'recorded_source_headers': game['headers'],
                'recorded_source_kind': game['source_kind'], 'recorded_source_players': game['players'],
                'recorded_source_game_id': game['game_id'], 'recorded_source_split': game['split'],
                'provenance': game['provenance'] + ';actual_recorded_native_terminal_course_pool',
                'paper_source_pool_profile': POOL_PROFILE, 'paper_source_targets': targets,
                'paper_recorded_mate_witness': witness,
                'recorded_continuation_is_best_move_label': False}
        if replay(root['fen'], future) != history[ply:ply + len(future) + 1]:
            raise ValueError('Recorded terminal future differs from the actual game')
        if witness and replay(root['fen'], witness) != history[ply:]:
            raise ValueError('Recorded near-mate witness differs from the actual game')
        roots.append(root)
    return {'kind': kind, 'roots': roots}


def source_owners(roots):
    owners, features = {}, set()
    for root in iter_jsonl(roots):
        game, split = root['game_id'], root['split']
        if (split not in SPLITS or not isinstance(game, str) or not game or
                root['recorded_source_game_id'] != game or owners.setdefault(game, split) != split):
            raise ValueError('Preassigned recorded game ownership is inconsistent')
        features.add(root['feature_key'])
    if not owners:
        raise ValueError('Preassigned recorded roots cannot be empty')
    return owners, features


def source_games(paths, owners):
    """Archive defaults never override ownership assigned by the previous corpus."""
    terminal, seen, found = {}, {}, set()
    for path in paths:
        for game in iter_jsonl(path):
            key = game['game_id']
            if key not in owners:
                continue
            found.add(key)
            identity = (game['initial_fen'], tuple(game['moves']), tuple(game['history']),
                        json.dumps(game['native_terminal'], sort_keys=True))
            if key in seen and seen[key] != identity:
                raise ValueError('Duplicate recorded source ID has conflicting actual history')
            seen[key] = identity
            if game['native_terminal']['ended']:
                terminal.setdefault(key, game)
    if found != set(owners):
        raise ValueError('Complete recorded archives do not contain every preassigned source game')
    return [terminal[key] for split in ('validation', 'test', 'train')
            for key in sorted(terminal) if owners[key] == split]


def _candidate_worker(item):
    return terminal_candidates(*item)


def candidate_groups(games, owners, workers):
    """Ordered, bounded workers; source bytes do not depend on the worker count."""
    if type(workers) is not int or workers < 1:
        raise ValueError('Positive native source worker count required')
    if workers == 1:
        for game in games:
            yield game, terminal_candidates(game, owners[game['game_id']])
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        for game in games:
            pending.append((game, pool.submit(_candidate_worker, (game, owners[game['game_id']]))))
            if len(pending) >= workers * 2:
                source, job = pending.popleft()
                yield source, job.result()
        for source, job in pending:
            yield source, job.result()


def root_footprint(root):
    """Reserve the maximum future, the entire actual mate witness, and both colors."""
    line = root['paper_recorded_mate_witness'] or root['future_moves']
    fens = replay(root['fen'], line)
    return {position_key(fen) for fen in fens} | {position_key(mirror_fen(fen)) for fen in fens}


def prepared_records(games, owners, old_features, prior_games, prior_positions, workers, counters):
    features, positions = set(old_features), dict(prior_positions)
    for game, candidates in candidate_groups(games, owners, workers):
        split, key = owners[game['game_id']], game['game_id']
        if key in prior_games and prior_games[key] != split:
            raise ValueError('Recorded game ownership conflicts with its prior complete corpus')
        counters['source_endings'][f"{split}/{candidates['kind']}"] += 1
        for root in candidates['roots']:
            reason = None
            footprint = root_footprint(root)
            if root['feature_key'] in features:
                reason = 'duplicate_full_history_root'
            elif any(p in positions and positions[p] != split for p in footprint):
                reason = 'root_future_witness_or_color_overlaps_another_split'
            if reason:
                counters['excluded'][reason] += 1
                yield 'excluded', {'game_id': key, 'split': split, 'feature_key': root['feature_key'],
                                   'ply': root['ply'], 'reason': reason}
                continue
            features.add(root['feature_key'])
            positions.update({p: split for p in footprint})
            counters['roots'][split] += 1
            counters['games'][split].add(key)
            for target in root['paper_source_targets']:
                stage = 'current' if target['horizon'] == 0 else 'future'
                for label in target['classes']:
                    counters['targets'][f'{split}/{stage}/{label}'] += 1
            yield 'roots', root


def reference_explanations(datasets, games, positions):
    """Also reserve structured answer PVs/branches, which are absent from future_moves."""
    inputs = []
    for directory in datasets:
        data = Path(directory)
        proof = data / 'manifest.json'
        paths = [data / f'{split}.jsonl' for split in SPLITS]
        inputs.append(proof)
        inputs += checked_sources(paths, [proof] * len(paths))
        counts = Counter()
        for row in footprint_records(paths, counts, explanation=True):
            game, split = row['game_id'], row['split']
            if games.setdefault(game, split) != split:
                raise ValueError('Prior explanation has conflicting game split ownership')
            fens = [row['fen']]
            for line in [row.get('future_moves', []), *row['future_branches']]:
                fens += replay(row['fen'], line)
            footprint = {position_key(fen) for fen in fens} | {position_key(mirror_fen(fen)) for fen in fens}
            for key in footprint:
                if positions.setdefault(key, split) != split:
                    raise ValueError('Prior explanation PV or branch overlaps another corpus split')
        if any(not counts[split] for split in SPLITS):
            raise ValueError('Prior explanation reference must contain every preassigned split')
    return inputs


def preparation(games, game_manifests, owners, owner_manifest, reference_data, explanation_reference_data):
    if not reference_data:
        raise ValueError('Terminal source pools require prior complete corpus split reservation')
    inputs = checked_sources(games, game_manifests) + checked_sources([owners], [owner_manifest])
    refs, prior_games, prior_positions = reference_contract(reference_data)
    inputs += refs
    inputs += reference_explanations(explanation_reference_data, prior_games, prior_positions)
    input_sha = {str(path): digest(path) for path in inputs}
    assignments, features = source_owners(owners)
    selected = source_games(games, assignments)
    counters = {name: Counter() for name in ('source_endings', 'excluded', 'roots', 'targets')}
    counters['games'] = {split: set() for split in SPLITS}
    return inputs, input_sha, selected, assignments, features, prior_games, prior_positions, counters


def verification(assignments, prior_games, prior_positions, counters, explanation_reference_data):
    return {'status': 'complete', 'evidence_state': 'reconstructed_baseline', 'pool_profile': POOL_PROFILE,
            'preassigned_source_games': len(assignments), 'source_endings_by_split_kind': dict(counters['source_endings']),
            'new_roots_by_split': dict(counters['roots']),
            'new_source_games_by_split': {k: len(v) for k, v in counters['games'].items()},
            'target_class_observations_by_split_stage': dict(counters['targets']),
            'excluded_candidates_by_reason': dict(counters['excluded']),
            'prior_corpus_games': len(prior_games), 'prior_root_future_color_positions': len(prior_positions),
            'all_declared_terminal_histories_and_prefixes_natively_verified': True,
            'actual_mate_witnesses_and_target_horizons_preserved': True,
            'prior_root_future_witness_color_isolation_verified': True,
            'prior_structured_explanation_pv_and_branches_reserved': bool(explanation_reference_data),
            'target_observations_are_independent_positions': False,
            'game_splits_reassigned': False, 'source_and_prior_corpora_modified': False,
            'paper_difficult_source_mix_applied': False, 'question_corpus_generated': False,
            'feature_cache_extended': False, 'student_or_teacher_loaded': False,
            'student_training_executed': False, 'student_strength_or_prose_quality_measured': False}


def build(games, game_manifests, owners, owner_manifest, reference_data, output, workers=1,
          explanation_reference_data=()):
    output = Path(output)
    if output.exists() or type(workers) is not int or workers < 1:
        raise ValueError('Use a fresh source pool output and a positive worker count')
    inputs, hashes, selected, assignments, features, prior_games, prior_positions, counters = preparation(
        games, game_manifests, owners, owner_manifest, reference_data, explanation_reference_data)
    output.mkdir(parents=True)
    paths = {kind: output / f'{kind}.jsonl' for kind in ('roots', 'excluded')}
    handles = {kind: path.open('x', encoding='utf-8') for kind, path in paths.items()}
    try:
        for kind, row in prepared_records(selected, assignments, features, prior_games, prior_positions, workers, counters):
            handles[kind].write(json.dumps(row, ensure_ascii=False) + '\n')
    finally:
        for handle in handles.values():
            handle.close()
    if any(digest(path) != hashes[str(path)] for path in inputs):
        raise ValueError('A recorded source or prior corpus changed during native pool production')
    result = verification(assignments, prior_games, prior_positions, counters, explanation_reference_data)
    config = {'games': list(map(str, games)), 'game_manifests': list(map(str, game_manifests)),
              'owners': str(owners), 'owner_manifest': str(owner_manifest), 'reference_data': list(map(str, reference_data)),
              'output': str(output), 'workers': workers, 'pool_profile': POOL_PROFILE,
              'explanation_reference_data': list(map(str, explanation_reference_data))}
    atomic_json(output / 'verification.json', result)
    atomic_json(output / 'manifest.json', manifest(DATA_KIND, config, inputs, [*paths.values(), output / 'verification.json'], result))
    return result


def readback(data, output, workers=1):
    data, output = Path(data), Path(output)
    if output.exists() or type(workers) is not int or workers < 1:
        raise ValueError('Use a fresh source pool readback and a positive worker count')
    proof_path = data / 'manifest.json'
    proof = json.loads(proof_path.read_text())
    config = proof['config']
    if (proof['status'] != 'complete' or proof['kind'] != DATA_KIND or config['pool_profile'] != POOL_PROFILE or
            Path(config['output']).resolve() != data.resolve()):
        raise ValueError('Readback requires a completed declared terminal source pool')
    for name, item in {**proof['inputs'], **proof['outputs']}.items():
        if Path(name).stat().st_size != item['bytes'] or digest(name) != item['sha256']:
            raise ValueError('Completed terminal source or root pool bytes changed')
    inputs, hashes, selected, assignments, features, prior_games, prior_positions, counters = preparation(
        config['games'], config['game_manifests'], config['owners'], config['owner_manifest'], config['reference_data'],
        config['explanation_reference_data'])
    handles = {kind: (data / f'{kind}.jsonl').open(encoding='utf-8') for kind in ('roots', 'excluded')}
    checked = Counter()
    try:
        for kind, row in prepared_records(selected, assignments, features, prior_games, prior_positions, workers, counters):
            if handles[kind].readline() != json.dumps(row, ensure_ascii=False) + '\n':
                raise ValueError('Terminal root or exclusion differs from its complete native source history')
            checked[kind] += 1
        if any(handle.read(1) for handle in handles.values()):
            raise ValueError('Completed terminal source pool has extra records')
    finally:
        for handle in handles.values():
            handle.close()
    if verification(assignments, prior_games, prior_positions, counters, config['explanation_reference_data']) != proof['verification']:
        raise ValueError('Terminal source pool counts or isolation proof changed')
    if (json.loads((data / 'verification.json').read_text()) != proof['verification'] or
            any(digest(path) != hashes[str(path)] for path in inputs)):
        raise ValueError('Terminal verification or complete source bytes changed during readback')
    result = {'status': 'complete', 'pool_profile': POOL_PROFILE, 'records_regenerated_and_compared': dict(checked),
              'all_input_output_bytes_checked': True, 'all_actual_histories_targets_and_exclusions_regenerated': True,
              'prior_root_future_witness_color_isolation_rechecked': True,
              'prior_structured_explanation_pv_and_branches_reserved': bool(config['explanation_reference_data']),
              'source_selection_algorithm_independently_rewritten': False,
              'student_or_teacher_loaded': False, 'student_training_executed': False}
    output.mkdir(parents=True)
    atomic_json(output / 'verification.json', result)
    atomic_json(output / 'manifest.json', manifest('recorded_terminal_pool_complete_source_readback',
        {'data': str(data), 'output': str(output), 'workers': workers},
        [proof_path, *proof['inputs'], *proof['outputs']], [output / 'verification.json'], result))
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--games', nargs='+')
    group.add_argument('--readback')
    parser.add_argument('--game-manifests', nargs='+')
    parser.add_argument('--owners')
    parser.add_argument('--owner-manifest')
    parser.add_argument('--reference-data', nargs='+', default=[])
    parser.add_argument('--explanation-reference-data', nargs='+', default=[])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.readback and any((args.game_manifests, args.owners, args.owner_manifest,
                              args.reference_data, args.explanation_reference_data)):
        parser.error('Readback uses its saved complete source and split contracts')
    if not args.readback and (not args.owners or not args.owner_manifest):
        parser.error('Production requires preassigned owner roots and their completed manifest')
    result = (readback(args.readback, args.output, args.workers) if args.readback else
              build(args.games, args.game_manifests or [], args.owners, args.owner_manifest,
                    args.reference_data, args.output, args.workers, args.explanation_reference_data))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
