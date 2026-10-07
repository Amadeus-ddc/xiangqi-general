"""Read back selected teacher queries against completed native audits and forecasts.

This observer reuses the completed full-history query audit, checks its artifact
bindings, and independently recomputes geometric footprints from legal FEN play.
It does not claim to repeat full-history terminal adjudication or engine search.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .evidence import atomic_json, digest, load_jsonl, manifest, position_key, write_jsonl
from .recorded_coach import checked_manifest, check_artifacts, ordered_native_map, overlaps, SPLITS
from .rules import future_fens
from .symmetry import mirror_fen


def geometric_forecasts(query):
    row, result = query['record'], query['oracle']
    best = next(c for c in result['candidates'] if c['move'] == result['best_move'])
    lines = [best['pv'][:6], *[c['pv'][:6] for c in result['candidates']], row['future_moves']]
    positions = {position_key(row['fen'])}
    for line in lines:
        positions.update(position_key(fen) for fen in future_fens(row['fen'], tuple(line)))
    positions.update(position_key(mirror_fen(fen)) for fen in list(positions))
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--native-readback', required=True)
    parser.add_argument('--footprints', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--public-evidence')
    args = parser.parse_args()
    dest, selected_root, candidate_root = Path(args.output), Path(args.selected), Path(args.candidates)
    if dest.exists():
        raise FileExistsError('Preserve every isolation observer; use a fresh output')
    if args.workers < 1:
        raise ValueError('Isolation readback workers must be positive')
    paths = [selected_root / 'queries.manifest.json', candidate_root / 'queries.manifest.json',
             Path(args.native_readback), Path(args.footprints)]
    proofs = {p: checked_manifest(p) for p in paths}
    selected_proof, candidate_proof, native_proof, footprint_proof = (proofs[p] for p in paths)
    check_artifacts(native_proof, [paths[1]], 'inputs')
    # The selected files must be bound to these exact completed native/footprint
    # manifests; a similarly shaped unrelated audit is not sufficient.
    check_artifacts(selected_proof, [paths[1], paths[2], paths[3]], 'inputs')
    selected_paths = [selected_root / f'{s}.queries.jsonl' for s in SPLITS]
    candidate_paths = [candidate_root / f'{s}.queries.jsonl' for s in SPLITS]
    check_artifacts(selected_proof, selected_paths)
    check_artifacts(candidate_proof, candidate_paths)
    selected, candidates = [], {}
    for split, path in zip(SPLITS, selected_paths):
        rows = load_jsonl(path)
        if any(q['record']['split'] != split for q in rows):
            raise ValueError('Selected query file changes the source game split')
        selected.extend(rows)
    for path in candidate_paths:
        for q in load_jsonl(path):
            if q['id'] in candidates:
                raise ValueError('Duplicate source candidate query ID')
            candidates[q['id']] = q
    if (len({q['id'] for q in selected}) != len(selected) or
            len({q['feature_key'] for q in selected}) != len(selected) or
            len({q['record']['game_id'] for q in selected}) != len(selected)):
        raise ValueError('Selected original roots, games or histories are not unique')
    audit_path, footprint_path = paths[2].parent / 'query-audits.jsonl', paths[3].parent / 'footprints.json'
    check_artifacts(native_proof, [audit_path])
    check_artifacts(footprint_proof, [footprint_path])
    audit_rows = load_jsonl(audit_path)
    audits = {a['id']: a for a in audit_rows}
    if len(audits) != len(audit_rows) or set(audits) != set(candidates):
        raise ValueError('Completed native candidate audit coverage differs')
    for query in selected:
        audit = audits.get(query['id'], {})
        if (query != candidates.get(query['id']) or not audit.get('native_forecasts_valid') or
                audit.get('feature_key') != query['feature_key']):
            raise ValueError('Selected query differs from an accepted native-audited original')
    expected = selected_proof['verification']['by_split']
    counts = dict(Counter(q['record']['split'] for q in selected))
    if counts != expected or len(selected) != selected_proof['verification']['selected_original_teacher_queries']:
        raise ValueError('Selected query counts differ from their completed producer')
    sides = Counter(f"{q['record']['split']}/{q['record']['fen'].split()[1]}" for q in selected)
    if any(sides[f'{s}/w'] != counts.get(s, 0) // 2 or sides[f'{s}/b'] != counts.get(s, 0) // 2 for s in SPLITS):
        raise ValueError('Selected teacher queries are not side-balanced')
    order = sorted(selected, key=lambda q: (q['record']['split'], q['id']))
    shard_paths = [selected_root / 'shards' / f'queries-{i}.jsonl' for i in range(3)]
    check_artifacts(selected_proof, shard_paths)
    for i, path in enumerate(shard_paths):
        if load_jsonl(path) != order[i::3]:
            raise ValueError('Annotation shard differs from the actual selected queries')
    footprint = json.loads(footprint_path.read_text())
    reserved_games = {s: set(footprint['combined_games'][s]) for s in SPLITS}
    reserved_positions = {s: set(footprint['combined_positions'][s]) for s in SPLITS}
    games, positions, observations = {s: set() for s in SPLITS}, {s: set() for s in SPLITS}, []
    for query, keys in zip(selected, ordered_native_map(geometric_forecasts, selected, args.workers)):
        split = query['record']['split']
        if any(query['record']['game_id'] in reserved_games[other] or keys & reserved_positions[other]
               for other in SPLITS if other != split):
            raise ValueError('Selected query overlaps another reserved game or forecast split')
        games[split].add(query['record']['game_id'])
        positions[split].update(keys)
        observations.append({'id': query['id'], 'feature_key': query['feature_key'], 'split': split,
                             'source_query_unchanged': True, 'native_audit_accepted': True,
                             'legal_geometric_forecast_and_counterpart_positions': len(keys)})
    if any(overlaps(games).values()) or any(overlaps(positions).values()):
        raise ValueError('Selected query forecasts overlap each other across splits')
    for path, proof in proofs.items():
        if checked_manifest(path) != proof:
            raise ValueError('Completed selection input changed during independent readback')
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
        'actual_original_teacher_queries': len(selected), 'by_split': counts, 'by_split_and_side': dict(sides),
        'distinct_original_games': len(selected), 'maximum_original_roots_per_game': 1,
        'all_queries_and_annotation_shards_match_original_candidates_exactly': True,
        'completed_full_native_history_forecast_audit_reused_and_hash_bound': True,
        'complete_history_terminal_adjudication_repeated_by_this_observer': False,
        'all_geometric_forecasts_and_counterparts_independently_recomputed_with_native_fen_play': True,
        'all_current_course_and_prior_label_root_and_answer_pv_footprints_protected': True,
        'selected_game_overlap': overlaps(games), 'selected_forecast_overlap': overlaps(positions),
        'all_completed_manifests_and_declared_inputs_outputs_freshly_rehashed': True,
        'independent_test_used_for_isolation_only': True,
        'teacher_annotations_generated_by_this_observer': False, 'explanation_training_started': False,
        'source_metadata_or_participant_identities_authenticated': False,
        'current_foundation_inputs_changed': False, 'full_feature_cache_loaded_or_rehashed': False}
    write_jsonl(dest / 'selected-query-isolation.jsonl', observations)
    atomic_json(dest / 'verification.json', summary)
    atomic_json(dest / 'manifest.json', manifest('recorded_coach_selected_query_isolation_readback', vars(args),
        [Path(__file__), *paths, audit_path, footprint_path, *selected_paths, *shard_paths],
        [dest / 'verification.json', dest / 'selected-query-isolation.jsonl'], summary))
    if args.public_evidence:
        atomic_json(args.public_evidence, dict(summary, manifest_sha256=digest(dest / 'manifest.json')))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
