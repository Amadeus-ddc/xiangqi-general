"""Independently read back original recorded roots and raw engine fact queries."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .curriculum_data import verify_splits
from .evidence import atomic_json, history_key, load_jsonl, manifest, write_jsonl
from .explanations import continuation_positions, line_facts, move_facts
from .human_games import assigned_split
from .oracle import parse_analysis
from .recorded_coach import checked_manifest, ordered_native_map, SPLITS
from .rules import adjudicate, piece_map, piece_name, replay, side


def read_query(item):
    query, nodes, split_seed = item
    row = query['record']
    if query['id'] != row['id'] + '-explanation':
        raise ValueError('Query ID differs from its original root')
    history = replay(row['initial_fen'], row['moves'])
    if (history != row['history'] or history[-1] != row['fen'] or
            history_key(history) != query['feature_key'] or query['feature_key'] != row['feature_key']):
        raise ValueError('Query differs from its original complete history')
    if row['split'] != assigned_split(row['game_id'], split_seed):
        raise ValueError('Query game split differs from the original assignment')
    if adjudicate(row['initial_fen'], row['moves'])['ended']:
        raise ValueError('A query cannot start from a terminal history')
    actual = parse_analysis(row['fen'], query['oracle']['raw_output'], nodes)
    if actual != query['oracle']:
        raise ValueError('Query engine result differs from its original raw output')
    expected = {'side_to_move': side(row['fen']), 'recommended': actual['best_move'],
        'board': {square: piece_name(piece) for square, piece in sorted(piece_map(row['fen']).items())},
        'move_facts': move_facts(row['fen'], actual['best_move']),
        'branches': [{key: branch[key] for key in ['move', 'score_type', 'score', 'perspective']} |
            {'pv': branch['pv'][:6], 'line_facts': line_facts(row['fen'], branch['pv'][:6])}
            for branch in actual['candidates']]}
    if query['verified_facts'] != expected:
        raise ValueError('Query board, move or continuation facts differ from native facts')
    best = next(branch for branch in actual['candidates'] if branch['move'] == actual['best_move'])
    value = {'pv': best['pv'][:6], 'branches': [{'pv': c['pv'][:6]} for c in actual['candidates']]}
    base = {'id': query['id'], 'feature_key': query['feature_key'], 'candidates': len(actual['candidates']),
            'recorded_move_matches_first_choice': actual['best_move'] == row['future_moves'][0]}
    try:
        continuation_positions(row, value)
    except ValueError as error:
        return dict(base, native_forecasts_valid=False, reason=str(error))
    return dict(base, native_forecasts_valid=True,
        forecast=dict(row, future_moves=value['pv'],
                      future_branches=[*[branch['pv'] for branch in value['branches']], row['future_moves']]))


def verify_rejection(rejection, originals, nodes):
    row = rejection['record']
    if row != originals.get(row['feature_key']):
        raise ValueError('Rejected engine query differs from its original root')
    try:
        parse_analysis(row['fen'], rejection['engine_output'], nodes)
    except RuntimeError as error:
        if str(error) != rejection['reason']:
            raise ValueError('Original engine rejection reason differs') from error
    else:
        raise ValueError('Original engine query rejection did not reproduce')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--split-seed', type=int, default=20261051)
    parser.add_argument('--public-evidence')
    args = parser.parse_args()
    root, queries_root, dest = Path(args.roots), Path(args.queries), Path(args.output)
    if dest.exists():
        raise FileExistsError('Preserve completed or failed observers; use a fresh output')
    if args.workers < 1:
        raise ValueError('Readback workers must be positive')
    root_manifest, query_manifest = root / 'manifest.json', queries_root / 'queries.manifest.json'
    root_proof, query_proof = checked_manifest(root_manifest), checked_manifest(query_manifest)
    nodes = query_proof['config']['nodes']
    if type(nodes) is not int or nodes < 1:
        raise ValueError('A positive requested engine node budget is required')
    originals = [row for split in SPLITS for row in load_jsonl(root / f'{split}.jsonl')]
    by_key = {row['feature_key']: row for row in originals}
    queries = [q for split in SPLITS for q in load_jsonl(queries_root / f'{split}.queries.jsonl')]
    if (len(by_key) != len(originals) or len(originals) != root_proof['verification']['selected_original_roots'] or
            len({q['id'] for q in queries}) != len(queries) or
            len({q['feature_key'] for q in queries}) != len(queries)):
        raise ValueError('Original root or query coverage is not unique or complete')
    rejected_paths = sorted((queries_root / 'preparation').glob('rejected-*.json'))
    rejections = [json.loads(path.read_text()) for path in rejected_paths]
    missing = set(by_key) - {q['feature_key'] for q in queries}
    if (missing != {r['record']['feature_key'] for r in rejections} or len(missing) != len(rejections) or
            len(rejections) != query_proof['verification']['rejected_engine_queries'] or
            len(queries) + len(rejections) != len(originals)):
        raise ValueError('Missing engine queries are not explained by unique recorded rejections')
    for rejection in rejections:
        verify_rejection(rejection, by_key, nodes)
    for query in queries:
        if query['record'] != by_key.get(query['feature_key']):
            raise ValueError('Engine query rewrites its original recorded root')
    audits = []
    for result in ordered_native_map(read_query,
            ((query, nodes, args.split_seed) for query in queries), args.workers):
        audits.append(result)
        if len(audits) % 128 == 0:
            print(json.dumps({'queries_independently_replayed': len(audits)}), flush=True)
    forecasts = [audit['forecast'] for audit in audits if audit['native_forecasts_valid']]
    native_rejected = [{k: v for k, v in audit.items() if k != 'forecast'}
                       for audit in audits if not audit['native_forecasts_valid']]
    try:
        split_proof = verify_splits(forecasts, workers=args.workers)
    except ValueError as error:
        split_proof = {'passed': False, 'reason': str(error),
                       'candidate_filtering_required_before_annotation': True}
    if checked_manifest(root_manifest) != root_proof or checked_manifest(query_manifest) != query_proof:
        raise ValueError('Completed source artifact identity changed during native readback')
    write_jsonl(dest / 'query-audits.jsonl', [{k: v for k, v in audit.items() if k != 'forecast'} for audit in audits])
    write_jsonl(dest / 'native-forecast-quarantine.jsonl', native_rejected)
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
        'actual_original_recorded_roots': len(originals), 'actual_engine_fact_queries': len(queries),
        'queries_by_split': dict(Counter(q['record']['split'] for q in queries)),
        'queries_by_split_and_side': dict(Counter(
            f"{q['record']['split']}/{q['record']['fen'].split()[1]}" for q in queries)),
        'queries_by_source_kind': dict(Counter(q['record']['recorded_source_kind'] for q in queries)),
        'actual_engine_candidate_counts': dict(Counter(a['candidates'] for a in audits)),
        'recorded_next_move_matches_engine_first_choice': sum(a['recorded_move_matches_first_choice'] for a in audits),
        'recorded_move_disagreement_is_not_a_proven_mistake': True,
        'requested_nodes_per_actual_query': nodes, 'rejected_engine_queries': len(rejections),
        'all_missing_queries_explained_by_reproduced_engine_rejections': True,
        'all_engine_results_reparsed_from_actual_raw_output': True,
        'all_verified_board_move_and_line_facts_recomputed': True,
        'actual_queries_with_native_full_history_forecasts': len(forecasts),
        'native_full_history_forecast_rejected_queries': len(native_rejected),
        'original_game_splits_preserved': True, 'all_completed_artifacts_freshly_rehashed': True,
        'selected_query_forecast_split_verification': split_proof,
        'full_current_course_and_prior_data_forecast_isolation_pending': True,
        'shared_pinned_engine_result_parser_and_native_rules_used': True,
        'teacher_annotations_generated': False, 'recorded_move_optimality_proven': False,
        'source_metadata_or_participant_identities_authenticated': False,
        'new_sft_labels_or_training_generated': False, 'current_foundation_inputs_changed': False}
    atomic_json(dest / 'verification.json', summary)
    atomic_json(dest / 'manifest.json', manifest('recorded_coach_engine_query_native_readback', vars(args),
        [Path(__file__), root_manifest, query_manifest, *rejected_paths],
        [dest / 'verification.json', dest / 'query-audits.jsonl', dest / 'native-forecast-quarantine.jsonl'], summary))
    if args.public_evidence:
        from .evidence import digest
        atomic_json(args.public_evidence, dict(summary, manifest_sha256=digest(dest / 'manifest.json')))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
