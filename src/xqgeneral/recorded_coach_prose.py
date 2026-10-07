"""Freeze, resolve and collect independently reviewed recorded-game teacher prose.

Preparation requires a complete authored shard and a bound query-isolation audit.
Resolution preserves rejected ancestors. Collection requires accepted coverage of
the entire original query batch; it never starts student training.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .collect_teacher import assemble_label
from .curriculum_data import verify_splits
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from .explanations import continuation_positions, line_facts, parse_explanation
from .recorded_coach import checked_manifest, check_artifacts, ordered_native_map, SPLITS
from .revise_prose import annotation_hash, TEACHER


def source_batch(root, isolation, author_plan):
    root, isolation, author_plan = Path(root), Path(isolation), Path(author_plan)
    source_path = root / 'queries.manifest.json'
    source, audit = checked_manifest(source_path), checked_manifest(isolation)
    query_paths = [root / f'{s}.queries.jsonl' for s in SPLITS]
    shard_paths = [root / 'shards' / f'queries-{i}.jsonl' for i in range(3)]
    check_artifacts(audit, [source_path, *query_paths, *shard_paths], 'inputs')
    if (audit['kind'] != 'recorded_coach_selected_query_isolation_readback' or
            not audit['verification'].get('all_queries_and_annotation_shards_match_original_candidates_exactly') or
            any(audit['verification']['selected_game_overlap'].values()) or
            any(audit['verification']['selected_forecast_overlap'].values())):
        raise ValueError('A completed original-query isolation audit is required')
    queries = []
    for split, path in zip(SPLITS, query_paths):
        batch = load_jsonl(path)
        if any(q['record']['split'] != split for q in batch):
            raise ValueError('Original query file changes its assigned game split')
        queries.extend(batch)
    counts = dict(Counter(q['record']['split'] for q in queries))
    if (not queries or len({q['id'] for q in queries}) != len(queries) or
            len({q['feature_key'] for q in queries}) != len(queries) or
            len({q['record']['game_id'] for q in queries}) != len(queries) or
            any(q['record'].get('augmentation_parent') for q in queries) or
            len(queries) != audit['verification']['actual_original_teacher_queries'] or
            counts != audit['verification']['by_split']):
        raise ValueError('Original query coverage differs from the isolation audit')
    plan = json.loads(author_plan.read_text())
    if (any(plan.get(k) != v for k, v in TEACHER.items()) or
            plan.get('source_queries') != str(root) or
            plan.get('qualified_readback_manifest_sha256') != digest(isolation) or
            plan.get('expected_original_annotations') != len(queries) or
            plan.get('by_split') != {s: counts.get(s, 0) for s in SPLITS}):
        raise ValueError('Author plan differs from the qualified original query batch')
    by_path = {a['input']: a for a in plan['agents']}
    if len(by_path) != 3 or set(by_path) != {str(p) for p in shard_paths}:
        raise ValueError('Author plan must bind all three immutable query shards')
    order = sorted(queries, key=lambda q: (q['record']['split'], q['id']))
    for index, path in enumerate(shard_paths):
        agent = by_path[str(path)]
        if (load_jsonl(path) != order[index::3] or digest(path) != agent['input_sha256'] or
                agent['expected'] != len(order[index::3]) or
                not isinstance(agent['task_name'], str) or not agent['task_name']):
            raise ValueError('Author shard differs from its complete original query assignment')
    check_artifacts(source, [*query_paths, *shard_paths])
    return queries, by_path, [source_path, isolation, author_plan, *query_paths, *shard_paths]


def input_bindings(paths):
    return {str(path): digest(path) for path in paths}


def require_unchanged(bindings):
    if any(digest(path) != sha for path, sha in bindings.items()):
        raise ValueError('A bound input changed while preparing or collecting prose')


def authored_shard(root, index, agents):
    path = Path(root) / 'shards' / f'queries-{index}.jsonl'
    agent = agents[str(path)]
    annotations_path, progress_path = Path(agent['annotations_output']), Path(agent['progress'])
    queries, annotations = load_jsonl(path), load_jsonl(annotations_path)
    progress = json.loads(progress_path.read_text())
    if (progress.get('status') != 'complete' or progress.get('completed') != len(queries) or
            progress.get('expected') != len(queries) or progress.get('input_sha256') != digest(path) or
            progress.get('output_sha256') != digest(annotations_path) or
            [a['id'] for a in annotations] != [q['id'] for q in queries]):
        raise ValueError('Teacher shard must be complete, unique and bound before review')
    return queries, annotations, agent['task_name'], [annotations_path, progress_path]


def checked_label(query, annotation):
    row = assemble_label(query, annotation)
    if len(re.findall(r'[\u4e00-\u9fff]', annotation['explanation'])) < 80:
        raise ValueError('Recorded-game teacher prose requires at least 80 Chinese characters')
    value = parse_explanation(row['answer'])
    continuation_positions(row, value)
    # Preserve the recorded continuation as well as every supplied answer line.
    row['future_moves'] = list(query['record']['future_moves'])
    row['future_branches'] = [value['pv'], *[b['pv'] for b in value['branches']]]
    return row


def review_packet(query, annotation, author):
    checked_label(query, annotation)
    facts = query['verified_facts']
    branches = [dict(b, line_facts=line_facts(query['record']['fen'], b['pv'], include_positions=True))
                for b in facts['branches']]
    return dict(facts, branches=branches, id=query['id'], feature_key=query['feature_key'],
                root_fen=query['record']['fen'], teacher_annotation=annotation,
                reviewed_annotation_sha256=annotation_hash(annotation), teacher_author_agent=author,
                source_teacher_query_id=query['id'], source_is_color_derived=False,
                score_perspective='side_to_move_at_root', engine_scores_are_search_snapshots=True)


def review_packet_item(item):
    return review_packet(*item)


def prepare_review(root, isolation, author_plan, output, shard=None, workers=1):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior review preparations; use a fresh output')
    all_queries, agents, inputs = source_batch(root, isolation, author_plan)
    bindings = input_bindings(inputs)
    if shard is not None and (type(shard) is not int or not 0 <= shard < 3):
        raise ValueError('Author shard must be 0, 1 or 2')
    prepared, rows = {}, []
    for index in range(3) if shard is None else [shard]:
        queries, annotations, author, paths = authored_shard(root, index, agents)
        bindings.update(input_bindings(paths))
        prepared[index] = list(ordered_native_map(review_packet_item,
            ((q, a, author) for q, a in zip(queries, annotations)), workers))
        rows.extend(queries); inputs.extend(paths)
    require_unchanged(bindings)
    outputs = []
    for index, packets in prepared.items():
        path = output / f'review-{index}.jsonl'
        write_jsonl(path, packets); outputs.append(path)
    summary = {'prepared_original_annotations': len(rows), 'original_batch_queries': len(all_queries),
               'by_split': dict(Counter(q['record']['split'] for q in rows)),
               'complete_authored_shards': list(prepared), 'all_annotations_hash_bound': True,
               'full_history_answer_lines_checked': True, 'actual_recorded_future_hidden_from_reviewer': True,
               'unverified_player_event_date_metadata_hidden_from_reviewer': True,
               'independent_semantic_review_complete': False, 'human_rating': False,
               'student_training_started': False}
    atomic_json(output / 'manifest.json', manifest('recorded_coach_prose_review_inputs',
        {'queries': str(root), 'isolation': str(isolation), 'author_plan': str(author_plan),
         'shard': shard, 'workers': workers},
        [Path(__file__), *inputs], outputs, summary))
    return summary


def resolved_packets(prepared, review_paths, repair_packet_paths=()):
    prepared = Path(prepared)
    proof = checked_manifest(prepared / 'manifest.json')
    bindings = {name: artifact['sha256'] for section in ('inputs', 'outputs')
                for name, artifact in proof[section].items()}
    bindings.update(input_bindings([prepared / 'manifest.json', *review_paths, *repair_packet_paths]))
    if proof['kind'] != 'recorded_coach_prose_review_inputs':
        raise ValueError('Recorded-game prose review preparation is required')
    original_paths = [Path(p) for p in proof['outputs'] if Path(p).suffix == '.jsonl']
    originals = [p for path in original_paths for p in load_jsonl(path)]
    original_by_id = {p['id']: p for p in originals}
    if not originals or len(original_by_id) != len(originals):
        raise ValueError('Review packets must contain unique original annotations')
    packet_files, annotations = {}, {}
    fixed_keys = set(originals[0]) - {'teacher_annotation', 'reviewed_annotation_sha256', 'teacher_author_agent'}
    for path in [*original_paths, *map(Path, repair_packet_paths)]:
        packets = load_jsonl(path)
        by_id = {}
        for packet in packets:
            identity, annotation = packet['id'], packet['teacher_annotation']
            sha = annotation_hash(annotation)
            if (identity in by_id or identity not in original_by_id or annotation.get('id') != identity or
                    any(annotation.get(k) != v for k, v in TEACHER.items()) or
                    not isinstance(annotation.get('explanation'), str) or
                    len(re.findall(r'[\u4e00-\u9fff]', annotation['explanation'])) < 80 or
                    packet['reviewed_annotation_sha256'] != sha or
                    not isinstance(packet.get('teacher_author_agent'), str) or not packet['teacher_author_agent'] or
                    {k: packet.get(k) for k in fixed_keys} !=
                    {k: original_by_id[identity].get(k) for k in fixed_keys}):
                raise ValueError('Review or repair packet changes its original facts or annotation binding')
            by_id[identity] = packet; annotations[sha] = annotation
        sha = digest(path)
        if not packets or sha in packet_files:
            raise ValueError('Review packet files must be nonempty and distinct')
        packet_files[sha] = by_id
    decisions, review_digests = defaultdict(list), set()
    for path in map(Path, review_paths):
        document, doc_sha = json.loads(path.read_text()), digest(path)
        if (doc_sha in review_digests or document.get('reviewer_model') != TEACHER['teacher_model'] or
                any(document.get(k) != TEACHER[k] for k in ('reasoning_effort', 'backend')) or
                document.get('human_rating') is not False or document.get('source_labels_modified') is not False or
                document.get('input_packet_sha256') not in packet_files or not document.get('results')):
            raise ValueError('Independent neural decisions must bind an actual immutable packet file')
        review_digests.add(doc_sha); seen = set()
        packets = packet_files[document['input_packet_sha256']]
        for decision in document['results']:
            identity = decision['id']; packet = packets.get(identity, {})
            actor = decision.get('reviewer_agent', document.get('reviewer_agent'))
            if (identity in seen or identity not in packets or
                    decision.get('reviewed_annotation_sha256') != packet.get('reviewed_annotation_sha256') or
                    decision.get('verdict') not in {'accept', 'reject'} or
                    not isinstance(decision.get('reason'), str) or not decision['reason'].strip() or
                    not isinstance(decision.get('issues'), list) or
                    any(not isinstance(x, str) for x in decision['issues']) or
                    (decision['verdict'] == 'accept' and decision['issues']) or
                    any(k in decision and decision[k] != expected for k, expected in
                        {'reviewer_model': TEACHER['teacher_model'], 'reasoning_effort': TEACHER['reasoning_effort'],
                         'backend': TEACHER['backend'], 'human_rating': False}.items()) or
                    not isinstance(actor, str) or not actor or actor == packet.get('teacher_author_agent')):
                raise ValueError('Review decision must independently bind its exact annotation and verdict')
            seen.add(identity)
            decisions[(identity, decision['reviewed_annotation_sha256'])].append((doc_sha, decision['verdict']))
    selected = []
    for packet in originals:
        identity, original_sha = packet['id'], packet['reviewed_annotation_sha256']
        accepted = []
        for sha, annotation in annotations.items():
            verdicts = decisions.get((identity, sha), [])
            if annotation['id'] != identity or not verdicts or any(v != 'accept' for _, v in verdicts):
                continue
            current, visited = sha, set()
            while current != original_sha:
                if current in visited or current not in annotations:
                    raise ValueError('Revision ancestry is cyclic or incomplete')
                visited.add(current); child = annotations[current]
                parent, rejection = child.get('corrected_from_annotation_sha256'), child.get('correction_review_sha256')
                if (parent not in annotations or annotations[parent]['id'] != identity or
                        (rejection, 'reject') not in decisions.get((identity, parent), [])):
                    raise ValueError('Revision must bind the exact rejected ancestor')
                current = parent
            accepted.append(annotation)
        if len(accepted) != 1:
            raise ValueError('Every original annotation requires exactly one independently accepted final revision')
        selected.append(accepted[0])
    require_unchanged(bindings)
    return selected, originals, proof, [prepared / 'manifest.json', *original_paths,
                                      *map(Path, review_paths), *map(Path, repair_packet_paths)]


def resolve_review(prepared, review_paths, repair_packet_paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve review decisions and revisions; use a fresh output')
    selected, originals, proof, inputs = resolved_packets(prepared, review_paths, repair_packet_paths)
    path = output / 'annotations.jsonl'; write_jsonl(path, selected)
    summary = {'resolved_original_annotations': len(selected),
               'accepted_unchanged_prose': sum(annotation_hash(a) == p['reviewed_annotation_sha256']
                                               for a, p in zip(selected, originals)),
               'all_original_rejections_and_exact_revision_ancestry_preserved': True,
               'independent_semantic_review_complete_for_prepared_shards': True,
               'human_rating': False, 'student_training_benefit_measured': False}
    atomic_json(output / 'manifest.json', manifest('recorded_coach_resolved_prose',
        {'prepared': str(prepared), 'reviews': list(map(str, review_paths)),
         'repair_packets': list(map(str, repair_packet_paths))}, [Path(__file__), *inputs], [path], summary))
    return summary


def reviewed_label_item(item):
    query, annotation, review_hash, original_hash = item
    row = checked_label(query, annotation)
    row['prose_review'] = {'source_annotation_sha256': original_hash,
        'reviewed_annotation_sha256': annotation_hash(annotation), 'review_manifest_sha256': review_hash,
        'changed': original_hash != annotation_hash(annotation), 'human_rating': False}
    return row


def collect_reviewed(root, isolation, author_plan, reviews, output, workers=1):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve collected labels; use a fresh output')
    queries, agents, inputs = source_batch(root, isolation, author_plan)
    bindings = input_bindings(inputs)
    original_annotations = {}
    for index in range(3):
        _, annotations, _, paths = authored_shard(root, index, agents)
        original_annotations.update({a['id']: a for a in annotations}); inputs.extend(paths)
        bindings.update(input_bindings(paths))
    accepted = {}
    for root_path in map(Path, reviews):
        path = root_path / 'manifest.json'; proof = checked_manifest(path)
        if proof['kind'] != 'recorded_coach_resolved_prose':
            raise ValueError('A resolved recorded-game prose review is required')
        cfg = proof['config']
        selected, originals, preparation, _ = resolved_packets(cfg['prepared'], cfg['reviews'], cfg['repair_packets'])
        for bound_proof in (proof, preparation):
            bindings.update({name: artifact['sha256'] for section in ('inputs', 'outputs')
                             for name, artifact in bound_proof[section].items()})
        if (preparation['config']['queries'] != str(root) or
                preparation['inputs'].get(str(Path(root) / 'queries.manifest.json'), {}).get('sha256') !=
                digest(Path(root) / 'queries.manifest.json') or
                selected != load_jsonl(root_path / 'annotations.jsonl')):
            raise ValueError('Resolved prose differs from its original prepared query batch')
        for annotation, packet in zip(selected, originals):
            identity = annotation['id']
            if (identity in accepted or identity not in original_annotations or
                    packet['reviewed_annotation_sha256'] != annotation_hash(original_annotations[identity])):
                raise ValueError('Resolved prose repeats or changes an original authored annotation')
            accepted[identity] = annotation, digest(path), packet['reviewed_annotation_sha256']
        inputs.extend([path, root_path / 'annotations.jsonl'])
        bindings.update(input_bindings([path, root_path / 'annotations.jsonl']))
    if set(accepted) != {q['id'] for q in queries}:
        raise ValueError('Independent semantic acceptance must cover the entire original query batch')
    rows = list(ordered_native_map(reviewed_label_item,
        ((query, *accepted[query['id']]) for query in queries), workers))
    split_proof = verify_splits(rows, workers=workers)
    require_unchanged(bindings)
    outputs = []
    for split in SPLITS:
        path = output / f'{split}.jsonl'; write_jsonl(path, [r for r in rows if r['split'] == split]); outputs.append(path)
    summary = {**split_proof, 'original_teacher_annotations': len(rows),
        'independently_accepted_original_annotations': len(rows),
        'by_split': dict(Counter(r['split'] for r in rows)),
        'by_side': dict(Counter(r['fen'].split()[1] for r in rows)),
        'structured_engine_answers_and_original_game_histories_preserved': True,
        'recorded_futures_and_all_answer_lines_checked_for_split_isolation': True,
        'derived_color_mirror_records': 0, 'human_rating': False,
        'source_metadata_or_participant_identities_authenticated': False,
        'student_training_started': False, 'student_training_benefit_measured': False,
        'four_new_courses_required_before_sft': True}
    atomic_json(output / 'manifest.json', manifest('reviewed_recorded_coach_initial_teacher_labels',
        {'queries': str(root), 'isolation': str(isolation), 'author_plan': str(author_plan),
         'reviews': list(map(str, reviews)), 'workers': workers}, [Path(__file__), *inputs], outputs, summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    prepare = actions.add_parser('prepare')
    prepare.add_argument('--shard', type=int, choices=range(3))
    prepare.add_argument('--workers', type=int, default=1)
    collect = actions.add_parser('collect')
    collect.add_argument('--reviews', nargs='+', required=True)
    collect.add_argument('--workers', type=int, default=1)
    for action in (prepare, collect):
        action.add_argument('--queries', required=True)
        action.add_argument('--isolation', required=True)
        action.add_argument('--author-plan', required=True)
    resolve = actions.add_parser('resolve')
    resolve.add_argument('--prepared', required=True)
    resolve.add_argument('--reviews', nargs='+', required=True)
    resolve.add_argument('--repair-packets', nargs='*', default=[])
    for action in (prepare, collect, resolve):
        action.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        summary = prepare_review(args.queries, args.isolation, args.author_plan, args.output, args.shard, args.workers)
    elif args.action == 'resolve':
        summary = resolve_review(args.prepared, args.reviews, args.repair_packets, args.output)
    else:
        summary = collect_reviewed(args.queries, args.isolation, args.author_plan, args.reviews, args.output, args.workers)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
