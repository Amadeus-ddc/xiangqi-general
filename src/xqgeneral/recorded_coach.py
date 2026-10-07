"""Prepare original recorded-game teacher inputs without loading course QA in RAM.

The curator preserves source game splits. Qualification reserves every candidate
PV, the recorded continuation and their color counterparts against existing
course and explanation footprints before any neural annotation is requested.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import islice
import json
import multiprocessing
from pathlib import Path

from .curriculum_data import _future_footprints, _split_contexts, verify_splits
from .evidence import atomic_json, digest, history_key, load_jsonl, manifest, position_key, write_jsonl
from .explanations import continuation_positions
from .human_games import assigned_split, recorded_year
from .recorded_sources import select_diverse_games
from .rules import adjudicate, piece_map, replay
from .symmetry import mirror_fen


SPLITS = ('train', 'validation', 'test')
RECORDED_KINDS = {'recorded_human_match', 'published_recorded_match'}


def stream_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def check_artifacts(proof, paths, section='outputs'):
    if proof.get('status') != 'complete':
        raise ValueError('A completed upstream manifest is required')
    for path in paths:
        path = Path(path)
        expected = proof.get(section, {}).get(str(path))
        if (expected is None or path.stat().st_size != expected['bytes'] or
                digest(path) != expected['sha256']):
            raise ValueError(f'Upstream artifact identity differs: {path}')


def checked_manifest(path):
    proof = json.loads(Path(path).read_text())
    for section in ('inputs', 'outputs'):
        check_artifacts(proof, proof.get(section, {}), section)
    return proof


def validate_quotas(quotas):
    if (set(quotas) != set(SPLITS) or
            any(type(n) is not int or n < 0 or n % 2 for n in quotas.values()) or
            not sum(quotas.values())):
        raise ValueError('Specify nonnegative even side-balanced quotas for all three splits')


def ordered_native_map(function, rows, workers):
    """Keep at most two outstanding jobs per spawned process, in input order."""
    if type(workers) is not int or workers < 1:
        raise ValueError('Workers must be a positive integer')
    if workers == 1:
        yield from map(function, rows)
        return
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        for row in rows:
            pending.append(pool.submit(function, row))
            if len(pending) >= workers * 2:
                yield pending.popleft().result()
        for job in pending:
            yield job.result()


def verify_original_root(item):
    row, split_seed = item
    history = replay(row['initial_fen'], row['moves'])
    if (history != row['history'] or history[-1] != row['fen'] or
            history_key(history) != row['feature_key'] or row['ply'] != len(row['moves'])):
        raise ValueError('Original recorded root differs from its native full history')
    if (row['split'] != assigned_split(row['game_id'], split_seed) or
            row['game_id'] != row['recorded_source_game_id'] or
            row['recorded_source_kind'] not in RECORDED_KINDS):
        raise ValueError('Original recorded root source or game split differs')
    if (adjudicate(row['initial_fen'], row['moves'])['ended'] or
            not 1 <= len(row['future_moves']) <= 8 or row.get('future_branches')):
        raise ValueError('An original nonterminal recorded continuation is required')
    replay(row['fen'], row['future_moves'])
    for offset in range(1, len(row['future_moves']) + 1):
        if adjudicate(row['initial_fen'], [*row['moves'], *row['future_moves'][:offset]])['ended']:
            raise ValueError('Original recorded continuation reaches a terminal history')
    return row['feature_key']


def curate_original_roots(rows, profiles, quotas, excluded_keys=(), *, seed=20261053,
                          split_seed=20261051, max_year=2026, participant_cap=128,
                          event_cap=128, modern_weight=2, workers=1):
    validate_quotas(quotas)
    grouped, scanned, excluded = defaultdict(list), 0, set(excluded_keys)
    for row in rows:
        scanned += 1
        if row['recorded_source_kind'] not in RECORDED_KINDS or row['feature_key'] in excluded:
            continue
        year = recorded_year(row['recorded_source_headers'])
        if year is not None and year > max_year:
            continue
        profile = profiles[row['game_id']]
        for key, original in [('split', 'split'), ('source_kind', 'recorded_source_kind'),
                              ('headers', 'recorded_source_headers'),
                              ('players', 'recorded_source_players'), ('source', 'recorded_source')]:
            if profile[key] != row[original]:
                raise ValueError(f'Recorded root differs from selected game metadata: {key}')
        grouped[row['game_id']].append(row)
    chosen, used, selector_results = [], set(), {}
    for offset, split in enumerate(SPLITS):
        if not quotas[split]:
            continue
        candidates = [dict(profiles[identity], declared_result=profiles[identity]['headers']['Result'])
                      for identity in grouped if profiles[identity]['split'] == split]
        ordered, rejected = select_diverse_games(candidates, len(candidates), participant_cap,
                                                event_cap, seed + offset, modern_weight)
        selector_results[split] = {'eligible_game_profiles': len(candidates),
                                  'capped_ordered_profiles': len(ordered), 'excluded_by_caps': rejected}
        needed, game_counts = {'w': quotas[split] // 2, 'b': quotas[split] // 2}, Counter()
        for _ in range(2):
            for profile in ordered:
                identity = profile['game_id']
                if game_counts[identity] >= 2:
                    continue
                ordered_rows = sorted(grouped[identity], key=lambda row: hashlib.sha256(
                    f"{seed}/{identity}/{row['feature_key']}".encode()).hexdigest())
                eligible = [row for row in ordered_rows if needed[row['fen'].split()[1]] > 0
                            and row['feature_key'] not in used]
                if not eligible:
                    continue
                row = eligible[0]
                chosen.append(dict(row, id='recorded-coach-' + row['feature_key']))
                used.add(row['feature_key'])
                needed[row['fen'].split()[1]] -= 1
                game_counts[identity] += 1
                if not sum(needed.values()):
                    break
            if not sum(needed.values()):
                break
        if sum(needed.values()):
            raise ValueError(f'Insufficient capped original roots for {split}: {needed}')
    keys = list(ordered_native_map(verify_original_root,
                                  ((row, split_seed) for row in chosen), workers))
    if len(set(keys)) != sum(quotas.values()):
        raise ValueError('Duplicate original histories were selected')
    proof = verify_splits(chosen, workers=workers)
    phases = Counter()
    for row in chosen:
        phase = ('reduced_material' if len(piece_map(row['fen'])) <= 16 else
                 ('early' if row['ply'] <= 24 else 'later'))
        phases[f"{row['split']}/{phase}"] += 1
    return chosen, {'original_recorded_roots_scanned': scanned,
        'selected_original_roots': len(chosen), 'by_split': dict(Counter(r['split'] for r in chosen)),
        'by_split_and_side': dict(Counter(f"{r['split']}/{r['fen'].split()[1]}" for r in chosen)),
        'by_source_kind': dict(Counter(r['recorded_source_kind'] for r in chosen)),
        'heuristic_material_and_ply_groups': dict(phases),
        'distinct_source_participant_labels': len({p for r in chosen for p in r['recorded_source_players']}),
        'maximum_selected_roots_per_original_game': max(Counter(r['game_id'] for r in chosen).values()),
        'source_profile_participant_game_cap': participant_cap, 'source_profile_event_game_cap': event_cap,
        'selector_results': selector_results, 'split_verification_for_original_recorded_futures': proof,
        'all_selected_full_native_histories_replayed': True, 'existing_teacher_feature_keys_reused': False,
        'color_derived_roots_created': False, 'original_game_splits_preserved': True}


def footprint_records(paths, counts, explanation=False):
    for path in paths:
        for row in stream_jsonl(path):
            if row['split'] != Path(path).stem:
                raise ValueError('A footprint row differs from its source file split')
            counts[row['split']] += 1
            if explanation:
                value = json.loads(row['answer'])
                # Preserve any recorded future as well as the structured answer PV.
                row = dict(row, future_branches=[*row.get('future_branches', []), value['pv'],
                            *[branch['pv'] for branch in value['branches']]])
            yield row


def extract_footprints(rows, workers=1):
    games, roots, futures, context_count = {}, {}, {}, 0
    contexts = _split_contexts(rows, games, roots)

    def batches():
        nonlocal context_count
        while batch := list(islice(contexts, 128)):
            context_count += len(batch)
            yield batch

    for result in ordered_native_map(_future_footprints, batches(), workers):
        for split, positions in result.items():
            futures.setdefault(split, set()).update(positions)
    return games, roots, futures, context_count


def overlaps(values):
    return {f'{left}/{right}': len(values.get(left, set()) & values.get(right, set()))
            for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:]}


def query_footprint(query):
    row, result = query['record'], query['oracle']
    best = next(branch for branch in result['candidates'] if branch['move'] == result['best_move'])
    value = {'pv': best['pv'][:6], 'branches': [{'pv': c['pv'][:6]} for c in result['candidates']]}
    positions = continuation_positions(row, value)
    positions.update(position_key(fen) for fen in replay(row['fen'], row['future_moves']))
    # Geometry is reserved only; scores are not independently recomputed or copied
    # into a derived query or label by this operation.
    positions.update(position_key(mirror_fen(fen)) for fen in list(positions))
    return positions


def isolate_queries(queries, footprints, audits, quotas, *, seed=20261054, workers=1):
    validate_quotas(quotas)
    ids = [query['id'] for query in queries]
    if len(set(ids)) != len(ids) or set(audits) != set(ids):
        raise ValueError('Native query audit coverage must be complete and unique')
    reserved_positions = {s: set(footprints['combined_positions'].get(s, [])) for s in SPLITS}
    reserved_games = {s: set(footprints['combined_games'].get(s, [])) for s in SPLITS}
    if any(overlaps(reserved_positions).values()) or any(overlaps(reserved_games).values()):
        raise ValueError('Reserved course or prior-label footprints overlap across splits')
    eligible, rejected, candidate_positions = {s: [] for s in SPLITS}, [], {}
    native = []
    for query in queries:
        identity, split = query['id'], query['record']['split']
        audit = audits[identity]
        if audit['feature_key'] != query['feature_key']:
            raise ValueError('Native query audit belongs to another history')
        if not audit['native_forecasts_valid']:
            rejected.append({'id': identity, 'split': split, 'reason': 'native_full_history_forecast',
                             'detail': audit['reason']})
        else:
            native.append(query)
    for query, positions in zip(native, ordered_native_map(query_footprint, native, workers)):
        split, identity = query['record']['split'], query['id']
        others = [other for other in SPLITS if other != split]
        reason = None
        if any(query['record']['game_id'] in reserved_games[other] for other in others):
            reason = 'reserved_other_split_game'
        elif any(positions & reserved_positions[other] for other in others):
            reason = 'reserved_other_split_root_or_forecast'
        if reason:
            rejected.append({'id': identity, 'split': split, 'reason': reason})
        else:
            eligible[split].append(query)
            candidate_positions[identity] = positions
    chosen, selected_positions, selected_games, quota_report = [], {}, {}, {}
    for split in ('test', 'validation', 'train'):
        needed = {'w': quotas[split] // 2, 'b': quotas[split] // 2}
        ordered = sorted(eligible[split], key=lambda query: hashlib.sha256(
            f"{seed}/{query['id']}".encode()).hexdigest())
        selected_positions[split], selected_games[split] = set(), set()
        for query in ordered:
            turn, identity = query['record']['fen'].split()[1], query['id']
            if not needed[turn]:
                continue
            positions = candidate_positions[identity]
            if any(positions & selected_positions[other] for other in selected_positions if other != split):
                reason = 'new_selected_other_split_forecast'
            elif query['record']['game_id'] in selected_games[split]:
                reason = 'second_root_same_source_game'
            else:
                chosen.append(query)
                selected_positions[split].update(positions)
                selected_games[split].add(query['record']['game_id'])
                needed[turn] -= 1
                if not sum(needed.values()):
                    break
                continue
            rejected.append({'id': identity, 'split': split, 'reason': reason})
        quota_report[split] = {'eligible_candidates': len(eligible[split]), 'unfilled_side_quotas': needed}
    if any(sum(item['unfilled_side_quotas'].values()) for item in quota_report.values()):
        error = ValueError(f'Insufficient isolated side-balanced queries: {quota_report}')
        error.quota_report = quota_report
        raise error
    if any(overlaps(selected_positions).values()) or any(overlaps(selected_games).values()):
        raise ValueError('Selected query footprints overlap')
    chosen.sort(key=lambda query: (query['record']['split'], query['id']))
    if len({q['feature_key'] for q in chosen}) != sum(quotas.values()):
        raise ValueError('Selected original query histories are not unique')
    summary = {'actual_candidate_engine_queries': len(queries),
        'selected_original_teacher_queries': len(chosen),
        'by_split': dict(Counter(q['record']['split'] for q in chosen)),
        'by_split_and_side': dict(Counter(f"{q['record']['split']}/{q['record']['fen'].split()[1]}" for q in chosen)),
        'by_source_kind': dict(Counter(q['record']['recorded_source_kind'] for q in chosen)),
        'distinct_normalized_source_participant_labels': len({
            p for q in chosen for p in q['record']['recorded_source_players']}),
        'distinct_source_games': len({q['record']['game_id'] for q in chosen}),
        'maximum_original_roots_per_game': 1, 'selection_seed': seed,
        'qualified_candidate_counts': {s: len(v) for s, v in eligible.items()},
        'isolation_quarantine_by_reason': dict(Counter(r['reason'] for r in rejected)),
        'all_source_queries_copied_without_rewriting': True,
        'all_selected_forecasts_complete_native_history_replayed': True,
        'all_current_course_and_prior_label_root_and_answer_pv_footprints_protected': True,
        'new_query_root_and_forecast_game_split_overlap': 0,
        'color_counterpart_geometry_reserved_but_no_derived_queries_created': True,
        'game_splits_preserved_before_teacher_annotation': True,
        'independent_test_used_for_isolation_only_not_checkpoint_or_quality_selection': True}
    return chosen, rejected, quota_report, summary


def preparation_summary(summary):
    return dict(summary, status='complete', evidence_state='reconstructed_baseline',
                teacher_model_contract='gpt-6-astra', reasoning_effort_contract='low',
                teacher_backend_contract='codex_subagent_authorized_by_user',
                teacher_annotations_generated=False, neural_explanation_labels_or_training_generated=False,
                source_metadata_or_participant_identities_authenticated=False,
                current_foundation_inputs_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    curate = commands.add_parser('curate', help='Select original roots from the recorded-root ledger')
    curate.add_argument('--data', required=True)
    curate.add_argument('--source-readback', required=True)
    curate.add_argument('--prior-labels', required=True)
    curate.add_argument('--split-seed', type=int, default=20261051)
    curate.add_argument('--max-year', type=int, default=2026)
    curate.add_argument('--participant-cap', type=int, default=128)
    curate.add_argument('--event-cap', type=int, default=128)
    curate.add_argument('--modern-weight', type=int, default=2)
    footprint = commands.add_parser('footprints', help='Stream full course and structured-label forecasts')
    footprint.add_argument('--data', required=True)
    footprint.add_argument('--prior-labels', required=True)
    footprint.add_argument('--source-readback', required=True)
    isolate = commands.add_parser('isolate', help='Qualify immutable teacher queries before annotation')
    isolate.add_argument('--queries', required=True)
    isolate.add_argument('--query-readback', required=True)
    isolate.add_argument('--footprints', required=True, help='Completed footprint manifest')
    for command in (curate, footprint, isolate):
        command.add_argument('--output', required=True)
        command.add_argument('--workers', type=int, default=8)
    for command, seed in ((curate, 20261053), (isolate, 20261054)):
        command.add_argument('--train-roots', type=int, default=2048)
        command.add_argument('--validation-roots', type=int, default=192)
        command.add_argument('--test-roots', type=int, default=256)
        command.add_argument('--seed', type=int, default=seed)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Preserve every preparation batch; use a fresh output directory')
    if args.workers < 1:
        raise ValueError('Workers must be positive')
    quotas = {s: getattr(args, s + '_roots') for s in SPLITS} if args.operation != 'footprints' else None
    if quotas is not None:
        validate_quotas(quotas)
    inputs, outputs, repeat_checks = [Path(__file__)], [], []

    def bind(proof, paths, section='outputs'):
        paths = list(paths)
        check_artifacts(proof, paths, section)
        repeat_checks.append((proof, paths, section))

    if args.operation in ('curate', 'footprints'):
        data, prior = Path(args.data), Path(args.prior_labels)
        reader_path = Path(args.source_readback)
        reader = json.loads(reader_path.read_text())
        bind(reader, [data / 'manifest.json'], 'inputs')
        producer = json.loads((data / 'manifest.json').read_text())
        prior_proof = json.loads((prior / 'manifest.json').read_text())
        prior_paths = [prior / f'{s}.jsonl' for s in SPLITS]
        bind(prior_proof, prior_paths)
        inputs += [reader_path, data / 'manifest.json', prior / 'manifest.json', *prior_paths]
    if args.operation == 'curate':
        source, metadata = data / 'recorded-roots.jsonl', data / 'selected-game-metadata.json'
        bind(reader, [source], 'inputs')
        bind(producer, [metadata])
        profiles = {g['game_id']: g for g in json.loads(metadata.read_text())}
        excluded = {r['feature_key'] for p in prior_paths for r in stream_jsonl(p)}
        rows, summary = curate_original_roots(stream_jsonl(source), profiles, quotas, excluded,
            seed=args.seed, split_seed=args.split_seed, max_year=args.max_year,
            participant_cap=args.participant_cap, event_cap=args.event_cap,
            modern_weight=args.modern_weight, workers=args.workers)
        inputs += [source, metadata]
        for split in SPLITS:
            path = dest / f'{split}.jsonl'
            write_jsonl(path, (row for row in rows if row['split'] == split))
            outputs.append(path)
    elif args.operation == 'footprints':
        course_paths = [data / f'{s}.jsonl' for s in SPLITS]
        bind(producer, course_paths)
        course_counts, prior_counts = Counter(), Counter()
        games, roots, future, contexts = extract_footprints(
            footprint_records(course_paths, course_counts), args.workers)
        old_games, _, old_future, old_contexts = extract_footprints(
            footprint_records(prior_paths, prior_counts, explanation=True), args.workers)
        expected = producer['verification']['split_verification']
        if (dict(course_counts) != producer['verification']['counts'] or
                {s: len(v) for s, v in games.items()} != expected['games'] or
                {s: len(v) for s, v in roots.items()} != expected['roots']):
            raise ValueError('Streamed full course counts differ from the producer')
        combined_games = {s: games.get(s, set()) | old_games.get(s, set()) for s in SPLITS}
        combined = {s: future.get(s, set()) | old_future.get(s, set()) for s in SPLITS}
        if any(overlaps(combined_games).values()) or any(overlaps(combined).values()):
            raise ValueError('Full reserved game or forecast footprints overlap')
        value = {'course_games': games, 'course_positions': future, 'prior_label_games': old_games,
                 'prior_label_positions': old_future, 'combined_games': combined_games, 'combined_positions': combined}
        path = dest / 'footprints.json'
        atomic_json(path, {kind: {s: sorted(v) for s, v in sets.items()} for kind, sets in value.items()})
        inputs += course_paths
        outputs.append(path)
        summary = {'actual_course_rows': dict(course_counts), 'actual_prior_label_rows': dict(prior_counts),
            'unique_course_forecast_contexts': contexts, 'unique_prior_label_forecast_contexts': old_contexts,
            'course_game_counts': {s: len(v) for s, v in games.items()},
            'course_root_counts': {s: len(v) for s, v in roots.items()},
            'course_root_and_forecast_position_counts': {s: len(v) for s, v in future.items()},
            'combined_game_overlap': overlaps(combined_games), 'combined_root_and_forecast_overlap': overlaps(combined),
            'recorded_futures_and_prior_label_structured_answer_pvs_included': True,
            'shared_pinned_native_future_parser_used': True,
            'prior_label_complete_history_terminal_checks_repeated_by_this_extractor': False,
            'full_course_files_materialized_in_memory': False,
            'full_feature_cache_loaded_or_rehashed': False, 'independent_test_read_for_isolation_only': True}
    else:
        queries_root = Path(args.queries)
        readback_path, footprint_path = Path(args.query_readback), Path(args.footprints)
        checked = {p: checked_manifest(p) for p in
                   (readback_path, footprint_path, queries_root / 'queries.manifest.json')}
        readback, footprint_proof = checked[readback_path], checked[footprint_path]
        query_paths = [queries_root / f'{s}.queries.jsonl' for s in SPLITS]
        bind(readback, [queries_root / 'queries.manifest.json'], 'inputs')
        bind(checked[queries_root / 'queries.manifest.json'], query_paths)
        audits_path = readback_path.parent / 'query-audits.jsonl'
        footprints_path = footprint_path.parent / 'footprints.json'
        bind(readback, [audits_path])
        bind(footprint_proof, [footprints_path])
        queries = [q for p in query_paths for q in stream_jsonl(p)]
        audit_rows = load_jsonl(audits_path)
        audits = {r['id']: r for r in audit_rows}
        if len(audits) != len(audit_rows):
            raise ValueError('Duplicate native query audits')
        try:
            rows, rejected, quota_report, summary = isolate_queries(queries,
                json.loads(footprints_path.read_text()), audits, quotas, seed=args.seed, workers=args.workers)
        except ValueError as error:
            if hasattr(error, 'quota_report'):
                atomic_json(dest / 'selection-progress.json', error.quota_report)
            raise
        inputs += [*checked, *query_paths, audits_path, footprints_path]
        for split in SPLITS:
            path = dest / f'{split}.queries.jsonl'
            write_jsonl(path, (q for q in rows if q['record']['split'] == split))
            outputs.append(path)
        for shard in range(3):
            path = dest / 'shards' / f'queries-{shard}.jsonl'
            write_jsonl(path, rows[shard::3])
            outputs.append(path)
        atomic_json(dest / 'selection-progress.json', quota_report)
        write_jsonl(dest / 'isolation-quarantine.jsonl', rejected)
        outputs += [dest / 'selection-progress.json', dest / 'isolation-quarantine.jsonl']
        for path, proof in checked.items():
            if checked_manifest(path) != proof:
                raise ValueError('Completed source manifest changed during query isolation')
    for proof, paths, section in repeat_checks:
        check_artifacts(proof, paths, section)
    summary = preparation_summary(summary)
    atomic_json(dest / 'verification.json', summary)
    outputs.append(dest / 'verification.json')
    name = 'queries.manifest.json' if args.operation == 'isolate' else 'manifest.json'
    atomic_json(dest / name, manifest('recorded_coach_' + args.operation, vars(args), inputs, outputs, summary))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
