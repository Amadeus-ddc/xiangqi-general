"""Four-course questions from recorded continuations, isolated before question generation."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import shutil

from .curriculum_data import dynamic_tasks, static_tasks, verify_splits
from .engine_selfplay import context_positions
from .evidence import atomic_json, digest, history_key, manifest, position_key, write_jsonl
from .human_games import assigned_split, recorded_year
from .recorded_sources import participant_label, select_diverse_games
from .rules import adjudicate, replay
from .selfplay_grounding import question_variant
from .symmetry import mirror_fen, mirror_move, mirrored_qa


def root_candidates(game, seed, min_ply=12, max_future_plies=8):
    """Interleave early/middle/late recorded prefixes; never create extra game moves."""
    if not 1 <= max_future_plies <= 8 or min_ply < 0:
        raise ValueError('Invalid recorded root sampling contract')
    moves, history = game['moves'], game['history']
    if len(history) != len(moves) + 1 or history[0] != game['initial_fen']:
        raise ValueError('Recorded game history dimensions differ')
    end = len(moves) - int(game['native_terminal']['ended'])
    rng = random.Random(int(hashlib.sha256(f"{seed}/{game['game_id']}".encode()).hexdigest(), 16))
    pools = [[], [], []]
    for ply in range(min_ply, end):
        pools[min(2, 3 * ply // max(1, end))].append(ply)
    for pool in pools:
        rng.shuffle(pool)
    for offset in range(max(map(len, pools), default=0)):
        for pool in pools:
            if offset >= len(pool):
                continue
            ply = pool[offset]
            future = moves[ply:ply + min(max_future_plies, end - ply)]
            if not future:
                continue
            past, prefix = moves[:ply], history[:ply + 1]
            actual_future = replay(history[ply], future)
            if actual_future != history[ply:ply + len(future) + 1]:
                raise ValueError('Recorded future history differs from its native continuation')
            row = {'game_id': game['game_id'], 'split': game['split'], 'initial_fen': game['initial_fen'],
                   'moves': past, 'history': prefix, 'fen': history[ply], 'feature_key': history_key(prefix),
                   'ply': ply, 'future_moves': future, 'future_branches': [],
                   'recorded_source': game['source'], 'recorded_source_headers': game['headers'],
                   'recorded_source_kind': game['source_kind'], 'recorded_source_players': game['players'],
                   'recorded_source_game_id': game['game_id'],
                   'provenance': game['provenance'] + ';recorded_mainline_rule_grounding'}
            # Reserve every possible sampled future prefix and its color counterpart.
            footprint = {position_key(fen) for fen in actual_future}
            footprint.update(position_key(mirror_fen(fen)) for fen in actual_future)
            yield row, footprint


def recorded_pair(root, seed):
    """Generate native answers and a color mirror from a fixed recorded root."""
    history = replay(root['initial_fen'], root['moves'])
    if history != root['history'] or history[-1] != root['fen'] or history_key(history) != root['feature_key']:
        raise ValueError('Recorded root differs from its native full history')
    if adjudicate(root['initial_fen'], root['moves'])['ended']:
        raise ValueError('Recorded curriculum root is terminal')
    rng = random.Random(int(hashlib.sha256(f"{seed}/{root['feature_key']}".encode()).hexdigest(), 16))
    all_moves = root['future_moves']
    if not 1 <= len(all_moves) <= 8:
        raise ValueError('Recorded future must have one to eight actual moves')
    full_future = replay(root['fen'], all_moves)
    for index in range(1, len(all_moves) + 1):
        if adjudicate(root['initial_fen'], [*root['moves'], *all_moves[:index]])['ended']:
            raise ValueError('Recorded future contains a terminal prefix')
    horizon = rng.randrange(1, len(all_moves) + 1)
    future_moves, future = all_moves[:horizon], full_future[horizon]
    tasks = [('static_current', static_tasks(root['fen'], rng)),
             ('dynamic_current', dynamic_tasks(root['fen'], rng)),
             ('static_future', static_tasks(future, rng)),
             ('dynamic_future', dynamic_tasks(future, rng))]
    rows = []
    for stage, entries in tasks:
        for task, question, answer, fields in entries:
            continuation = future_moves if 'future' in stage else []
            identity = hashlib.sha256(f"{seed}/{root['feature_key']}/{stage}/{task}/{question}".encode()).hexdigest()[:20]
            row = dict(root, id=identity, stage=stage, task_type=task, query=fields,
                       question=(f"依次走 {' '.join(continuation)} 后，" if continuation else '') + question,
                       answer=answer, future_moves=continuation, future_branches=[],
                       teacher={'provider': 'pinned_pyffish_rules', 'neural_prose_generated': False},
                       recorded_continuation_is_best_move_label=False)
            variant = rng.randrange(3)
            row.update(question=question_variant(row, variant), question_variant=variant)
            row['id'] = hashlib.sha256(f'{identity}/question-format-{variant}'.encode()).hexdigest()[:20]
            rows.append(row)
    initial, moves = mirror_fen(root['initial_fen']), [mirror_move(m) for m in root['moves']]
    mirror_context = initial, moves, replay(initial, moves)
    rows += [mirrored_qa(row, mirror_context) for row in list(rows)]
    allowed_footprint = {position_key(fen) for fen in [*full_future, *[mirror_fen(f) for f in full_future]]}
    # Mirrored metadata retains its real recorded source identity, not a fictitious match.
    for row in rows:
        if row.get('augmentation_parent'):
            row['question'] = question_variant(row, row['question_variant'])
        if context_positions(row) - allowed_footprint:
            raise ValueError('Question future escaped the recorded continuation')
    return rows


def _pair_worker(item):
    root, seed = item
    return recorded_pair(root, seed)


def question_groups(roots, seed, workers):
    """Bounded, ordered workers; label output does not depend on process count."""
    if workers < 1:
        raise ValueError('Positive question worker count required')
    if workers == 1:
        for root in roots:
            yield recorded_pair(root, seed)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending, iterator = deque(), iter(roots)
        for root in iterator:
            pending.append(pool.submit(_pair_worker, (root, seed)))
            if len(pending) >= workers * 2:
                yield pending.popleft().result()
        for job in pending:
            yield job.result()


def iter_records(paths):
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def build(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve completed recorded curriculum; use a fresh output')
    output.mkdir(parents=True)
    footprint_root = Path(args.footprints)
    footprint_proof = json.loads((footprint_root / 'manifest.json').read_text())
    if footprint_proof['status'] != 'complete':
        raise ValueError('Prior corpus footprint preparation is incomplete')
    footprint_path = footprint_root / 'positions.json'
    if digest(footprint_path) != footprint_proof['outputs'][str(footprint_path)]['sha256']:
        raise ValueError('Reserved prior footprint bytes changed')
    footprints = {k: set(v) for k, v in json.loads(footprint_path.read_text()).items()}
    prior_train, prior_heldout = footprints['train'], footprints['heldout']
    inputs = [footprint_root / 'manifest.json', footprint_path]
    source_games, seen_games, duplicates = [], set(), 0
    for directory in args.games:
        root = Path(directory)
        proof = json.loads((root / 'manifest.json').read_text())
        path = root / 'games.jsonl'
        if proof['status'] != 'complete' or proof['config']['seed'] != args.split_seed:
            raise ValueError('Recorded source is incomplete or uses a different pre-question game split')
        if digest(path) != proof['outputs'][str(path)]['sha256']:
            raise ValueError('Recorded source game bytes changed')
        inputs.extend([root / 'manifest.json', path])
        for game in iter_records([path]):
            if game['split'] != assigned_split(game['game_id'], args.split_seed):
                raise ValueError('Recorded source game split changed')
            if game['game_id'] in seen_games:
                duplicates += 1
                continue
            seen_games.add(game['game_id'])
            game['players'] = [participant_label(game['headers'][key]) for key in ['Red', 'Black']]
            source_games.append(game)
    games, diversity_rejections = select_diverse_games(
        source_games, args.game_budget, args.participant_cap, args.event_cap, args.seed, args.modern_weight)
    roots, exclusions, owned_features, new_heldout = [], Counter(), set(), set()
    root_counts, game_counts, sides, horizon_capacity = Counter(), Counter(), Counter(), Counter()
    for split in ['validation', 'test', 'train']:
        forbidden = prior_heldout | new_heldout
        for game in games:
            if game['split'] != split:
                continue
            accepted = 0
            for root, footprint in root_candidates(game, args.seed, args.min_ply):
                if root['feature_key'] in owned_features:
                    exclusions['duplicate_full_history_root'] += 1
                    continue
                if split != 'train' and footprint & prior_train:
                    exclusions['new_heldout_overlaps_prior_training'] += 1
                    continue
                if footprint & forbidden:
                    exclusions['heldout_root_or_recorded_future'] += 1
                    continue
                roots.append(root)
                owned_features.add(root['feature_key'])
                if split != 'train':
                    new_heldout.update(footprint)
                    forbidden.update(footprint)
                accepted += 1
                root_counts.update([split])
                sides.update([root['fen'].split()[1]])
                horizon_capacity.update([len(root['future_moves'])])
                if accepted == args.roots_per_game:
                    break
            if accepted:
                game_counts.update([split])
    if any(root_counts[s] == 0 for s in ['train', 'validation', 'test']):
        raise ValueError('Recorded curriculum needs isolated roots in all three preassigned game splits')
    print(json.dumps({'phase': 'isolated_recorded_roots', 'roots': dict(root_counts),
                      'games': dict(game_counts), 'excluded': dict(exclusions)}), flush=True)
    write_jsonl(output / 'recorded-roots.jsonl', roots)
    atomic_json(output / 'selected-game-metadata.json', [
        {'game_id': g['game_id'], 'split': g['split'], 'players': g['players'], 'headers': g['headers'],
         'source': g['source'], 'source_kind': g['source_kind']} for g in games])
    base = Path(args.base_data)
    base_proof = json.loads((base / 'manifest.json').read_text())
    paths = {s: output / (s + '.jsonl') for s in ['train', 'validation', 'test']}
    base_counts, added, stages, future_lengths, formats = {}, Counter(), Counter(), Counter(), Counter()
    for split, path in paths.items():
        source = base / path.name
        if digest(source) != base_proof['outputs'][str(source)]['sha256']:
            raise ValueError('Base engine question bytes changed')
        inputs.append(source)
        shutil.copyfile(source, path)
        base_counts[split] = base_proof['verification']['counts'][split] if 'counts' in base_proof['verification'] else sum(1 for _ in source.open())
    handles = {s: p.open('a') for s, p in paths.items()}
    try:
        for index, rows in enumerate(question_groups(roots, args.seed, args.workers)):
            for row in rows:
                handles[row['split']].write(json.dumps(row, ensure_ascii=False) + '\n')
                added.update([row['split']])
                stages.update([row['stage']])
                formats.update([row['question_variant']])
                if row['future_moves']:
                    future_lengths.update([len(row['future_moves'])])
            if (index + 1) % 1024 == 0:
                print(json.dumps({'recorded_roots_converted': index + 1, 'added_rule_questions': sum(added.values())}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    split_proof = verify_splits(iter_records(paths.values()), workers=args.split_workers)
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
               'source_unique_recorded_games': len(source_games), 'duplicate_source_game_records': duplicates,
               'selected_recorded_games_before_root_isolation': len(games),
               'selected_game_participant_cap': args.participant_cap, 'selected_source_event_group_cap': args.event_cap,
               'event_grouping_is_source_metadata_heuristic': True,
               'modern_decade_sampling_weight': args.modern_weight,
               'selected_games_by_source_kind': dict(Counter(g['source_kind'] for g in games)),
               'selected_games_by_recorded_year': dict(Counter(str(recorded_year(g['headers'])) for g in games)),
               'distinct_normalized_participant_labels': len({p for g in games for p in g['players']}),
               'participant_identity_independently_verified': False,
               'games_contributing_isolated_roots_by_split': dict(game_counts),
               'original_recorded_roots_by_split': dict(root_counts), 'original_root_sides': dict(sides),
               'original_recorded_roots': len(roots), 'augmented_recorded_roots': 2 * len(roots),
               'maximum_recorded_future_horizon_capacity': dict(horizon_capacity),
               'actual_future_question_horizons': dict(future_lengths), 'question_formats': dict(formats),
               'base_engine_question_counts': base_counts, 'added_recorded_question_counts': dict(added),
               'added_recorded_questions': sum(added.values()),
               'counts': {s: base_counts[s] + added[s] for s in paths}, 'stages': dict(stages),
               'root_or_future_exclusions': dict(exclusions), 'diversity_exclusions': diversity_rejections,
               'new_human_validation_test_overlap_prior_training': 0,
               'game_split_assigned_before_questions': True, 'split_verification': split_proof,
               'base_file_bytes_preserved_as_prefixes': True, 'old_experiments_modified': False,
               'recorded_moves_used_as_best_move_labels': False, 'site_prose_used_as_neural_labels': False,
               'feature_cache_and_token_preflight_complete': False, 'new_curriculum_training_executed': False}
    atomic_json(output / 'manifest.json', manifest('isolated_recorded_engine_four_course_questions', vars(args),
        [*inputs, base / 'manifest.json'], [*paths.values(), output / 'recorded-roots.jsonl',
                                           output / 'selected-game-metadata.json'], summary))
    atomic_json(args.public_evidence, dict(summary, manifest_sha256=digest(output / 'manifest.json')))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+', required=True)
    parser.add_argument('--footprints', required=True)
    parser.add_argument('--base-data', default='data/research-selfplay-v2')
    parser.add_argument('--game-budget', type=int, default=6144)
    parser.add_argument('--roots-per-game', type=int, default=8)
    parser.add_argument('--participant-cap', type=int, default=256)
    parser.add_argument('--event-cap', type=int, default=128)
    parser.add_argument('--modern-weight', type=int, default=3)
    parser.add_argument('--min-ply', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261054)
    parser.add_argument('--split-seed', type=int, default=20261051)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--split-workers', type=int, default=16)
    parser.add_argument('--output', required=True)
    parser.add_argument('--public-evidence', required=True)
    args = parser.parse_args()
    if min(args.game_budget, args.roots_per_game, args.participant_cap, args.event_cap, args.workers, args.split_workers, args.modern_weight) < 1:
        raise ValueError('Recorded course sampling and worker budgets must be positive')
    build(args)


if __name__ == '__main__':
    main()
