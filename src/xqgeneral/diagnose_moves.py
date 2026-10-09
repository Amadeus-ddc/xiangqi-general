"""Diagnose preserved move-list QA errors on unchanged validation questions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .evaluate_qa import normalized
from .evidence import atomic_json, file_signature, manifest, write_jsonl
from .foundation_preflight import native_answer, native_context
from .rules import legal_moves, piece_map, replay
from .selfplay_grounding import question_variant

IDENTITY = ('id', 'game_id', 'stage', 'task_type', 'question', 'question_variant', 'expected')
STAGES = ('dynamic_current', 'dynamic_future')


def move_tokens(answer):
    fields = answer.split()
    if fields == ['无']:
        return []
    if not fields or any(re.fullmatch(r'[a-i][0-9][a-i][0-9]', field) is None for field in fields):
        return None
    return fields


def summarize(items):
    parsed = [item for item in items if item['parseable']]
    true_positive = sum(item['true_positive_unique_moves'] for item in parsed)
    predicted = sum(item['predicted_unique_moves'] for item in parsed)
    gold = sum(item['gold_moves'] for item in parsed)
    correct = sum(item['raw_correct'] for item in items)
    return {'examples': len(items), 'raw_correct': correct, 'raw_accuracy': correct / len(items),
            'error_categories': dict(Counter(item['category'] for item in items)),
            'parseable_answers': len(parsed),
            'exact_move_set_answers': sum(item['set_exact'] for item in parsed),
            'answers_with_duplicate_tokens': sum(item['duplicate_tokens'] > 0 for item in parsed),
            'answers_with_missing_moves': sum(item['missing_moves'] > 0 for item in parsed),
            'answers_with_extra_moves': sum(item['extra_moves'] > 0 for item in parsed),
            'unique_missing_moves': sum(item['missing_moves'] for item in parsed),
            'unique_extra_moves': sum(item['extra_moves'] for item in parsed),
            'unique_illegal_extra_moves': sum(item['illegal_extra_moves'] for item in parsed),
            'unique_legal_moves_from_wrong_source': sum(item['legal_wrong_source'] for item in parsed),
            'diagnostic_unique_move_precision': true_positive / predicted if predicted else None,
            'diagnostic_unique_move_recall': true_positive / gold if gold else None,
            'diagnostic_metrics_denominator': 'fully parseable answers only; malformed answers remain raw errors'}


def _predictions(path, stage):
    selected, digest, total = {}, hashlib.sha256(), 0
    with path.open('rb') as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row['stage'] != stage or row['task_type'] != 'moves':
                continue
            if (not isinstance(row['generated'], str) or not isinstance(row['expected'], str) or
                    type(row['correct']) is not bool or type(row['question_variant']) is not int or
                    row['question_variant'] not in (0, 1, 2)):
                raise ValueError('Move predictions require text, raw boolean correctness and a question variant')
            if row['correct'] != (normalized(row['generated']) == normalized(row['expected'])):
                raise ValueError('Preserved correctness differs from strict raw exact match')
            if row['id'] in selected:
                raise ValueError('Duplicate move prediction ID')
            selected[row['id']] = row
    if not selected:
        raise ValueError('No move-list predictions for the requested stage')
    return selected, {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}, total


def _validation(path, wanted, stage):
    found, digest, total = {}, hashlib.sha256(), 0
    with path.open('rb') as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row['split'] != 'validation':
                raise ValueError('Move diagnosis accepts validation data only')
            if row['id'] not in wanted:
                continue
            if row['id'] in found:
                raise ValueError('Duplicate selected validation ID')
            if row['stage'] != stage or row['task_type'] != 'moves':
                raise ValueError('Prediction ID differs from the validation task')
            native_context(row)
            if native_answer(row) != row['answer']:
                raise ValueError('Validation move-list answer differs from native rules')
            target = replay(row['fen'], row.get('future_moves', []))[-1]
            source = row['query']['source']
            if not isinstance(source, str) or re.fullmatch(r'[a-i][0-9]', source) is None:
                raise ValueError('Invalid queried source square')
            legal = set(legal_moves(target))
            gold = sorted(move for move in legal if move[:2] == source)
            if move_tokens(row['answer']) != gold:
                raise ValueError('Validation move-list order differs from native rules')
            found[row['id']] = (row, gold, legal, piece_map(target).get(source))
    if set(found) != wanted:
        raise ValueError('Move predictions are missing from the validation data')
    return found, {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}, total


def _diagnose(prediction, native):
    row, gold, legal, symbol = native
    if (prediction['game_id'] != row['game_id'] or prediction['expected'] != row['answer'] or
            prediction['question'] != question_variant(row, prediction['question_variant'])):
        raise ValueError('Prediction question or gold differs from its native validation row')
    actual = move_tokens(prediction['generated'])
    parsed = actual is not None
    actual_set = set(actual) if parsed else set()
    gold_set = set(gold)
    missing, extra = gold_set - actual_set, actual_set - gold_set
    duplicates = len(actual) - len(actual_set) if parsed else 0
    if prediction['correct']:
        category = 'raw_correct'
    elif not parsed:
        category = 'malformed_output'
    elif missing or extra:
        category = 'missing_and_extra' if missing and extra else 'missing_only' if missing else 'extra_only'
    else:
        wrong_order = list(dict.fromkeys(actual)) != gold
        category = 'order_and_duplicates' if duplicates and wrong_order else 'duplicates_only' if duplicates else 'order_only'
    size = len(gold)
    return {'id': row['id'], 'game_id': row['game_id'], 'stage': row['stage'],
            'feature_key': row['feature_key'], 'query_source': row['query']['source'],
            'piece_kind': symbol.lower() if symbol else 'empty',
            'piece_side': 'red' if symbol and symbol.isupper() else 'black' if symbol else 'empty',
            'gold_move_count_bucket': '0' if size == 0 else '1-3' if size <= 3 else '4-7' if size <= 7 else '8+',
            'gold_moves': size, 'question_variant': prediction['question_variant'],
            'expected': prediction['expected'], 'generated': prediction['generated'],
            'raw_correct': prediction['correct'], 'category': category, 'parseable': parsed,
            'set_exact': parsed and actual_set == gold_set, 'duplicate_tokens': duplicates,
            'predicted_unique_moves': len(actual_set) if parsed else 0,
            'true_positive_unique_moves': len(actual_set & gold_set) if parsed else 0,
            'missing_moves': len(missing) if parsed else 0, 'extra_moves': len(extra) if parsed else 0,
            'illegal_extra_moves': len(extra - legal) if parsed else 0,
            'legal_wrong_source': len(extra & legal) if parsed else 0}


def _groups(items):
    result = {}
    for field in ('piece_kind', 'gold_move_count_bucket'):
        grouped = defaultdict(list)
        for item in items:
            grouped[item[field]].append(item)
        result[field] = {key: summarize(value) for key, value in sorted(grouped.items())}
    return result


def _paired(before, after):
    first, last = summarize(before), summarize(after)
    return {'examples': len(before), 'before_raw_correct': first['raw_correct'],
            'after_raw_correct': last['raw_correct'],
            'corrected': sum(not left['raw_correct'] and right['raw_correct'] for left, right in zip(before, after)),
            'newly_wrong': sum(left['raw_correct'] and not right['raw_correct'] for left, right in zip(before, after)),
            'net_correct_change': last['raw_correct'] - first['raw_correct'],
            'before_missing_unique_moves': first['unique_missing_moves'],
            'after_missing_unique_moves': last['unique_missing_moves'],
            'before_illegal_extra_unique_moves': first['unique_illegal_extra_moves'],
            'after_illegal_extra_unique_moves': last['unique_illegal_extra_moves']}


def diagnose_predictions(validation_path, prediction_paths, output, *, stage='dynamic_current'):
    """Recheck native gold and compare raw errors; reject changed questions or inputs."""
    validation_path, output = Path(validation_path), Path(output)
    prediction_paths = [Path(path) for path in prediction_paths]
    if stage not in STAGES or not prediction_paths:
        raise ValueError('Choose a dynamic stage and at least one prediction file')
    if len({path.resolve() for path in prediction_paths}) != len(prediction_paths):
        raise ValueError('Duplicate prediction file')
    if output.exists():
        raise FileExistsError('Use a fresh diagnosis output')
    inputs = [validation_path, *prediction_paths]
    signatures = {str(path): file_signature(path) for path in inputs}
    predictions, input_hashes, totals = [], {}, []
    for path in prediction_paths:
        rows, identity, total = _predictions(path, stage)
        predictions.append(rows)
        input_hashes[str(path)] = identity
        totals.append(total)
    baseline = predictions[0]
    for rows in predictions[1:]:
        if set(rows) != set(baseline) or any(tuple(rows[key][field] for field in IDENTITY) !=
                tuple(baseline[key][field] for field in IDENTITY) for key in baseline):
            raise ValueError('Compare only identical move questions, variants and gold by ID')
    native, identity, validation_rows = _validation(validation_path, set(baseline), stage)
    input_hashes[str(validation_path)] = identity
    candidates, details, by_candidate = [], [], []
    for path, rows, total in zip(prediction_paths, predictions, totals):
        items = [dict(_diagnose(rows[key], native[key]), candidate=str(path)) for key in sorted(baseline)]
        details.extend(items)
        by_candidate.append(items)
        candidates.append({'predictions': str(path), 'prediction_file_rows': total,
                           'summary': summarize(items), 'groups': _groups(items)})
    pairs = [(index - 1, index) for index in range(1, len(candidates))]
    if len(candidates) > 2:
        pairs.append((0, len(candidates) - 1))
    comparisons = []
    for first, last in pairs:
        before, after = by_candidate[first], by_candidate[last]
        grouped = {}
        for field in ('piece_kind', 'gold_move_count_bucket'):
            grouped[field] = {key: _paired([item for item in before if item[field] == key],
                                          [item for item in after if item[field] == key])
                              for key in sorted({item[field] for item in before})}
        comparisons.append({'before': str(prediction_paths[first]), 'after': str(prediction_paths[last]),
                            'summary': _paired(before, after), 'groups': grouped})
    for path in inputs:
        if file_signature(path) != signatures[str(path)]:
            raise ValueError('Diagnosis input changed during readback')
    scope = {'raw_exact_match_scores_preserved': True, 'raw_predictions_repaired': False,
             'move_set_metrics_are_diagnostic_only': True, 'independent_test_used': False,
             'new_model_or_engine_search_executed': False, 'training_config_changed': False,
             'checkpoint_training_provenance_checked': False, 'piece_groups_prove_causation': False}
    report = {'status': 'complete', 'evidence_state': 'reconstructed_baseline', 'stage': stage,
              'candidates': candidates, 'paired_same_wording_diagnostics': comparisons,
              'same_ids_questions_variants_and_gold_verified': True,
              'native_move_list_answers_recomputed': len(native),
              'native_full_histories_replayed': len({value[0]['feature_key'] for value in native.values()}),
              'native_future_sequences_replayed': len(native) if stage == 'dynamic_future' else 0,
              'validation_file_rows_scanned': validation_rows, 'scope_limits': scope}
    output.mkdir(parents=True)
    atomic_json(output / 'analysis.json', report)
    write_jsonl(output / 'item-diagnostics.jsonl', details)
    proof = manifest('preserved_move_enumeration_error_diagnostic',
                     {'validation': str(validation_path), 'predictions': [str(path) for path in prediction_paths],
                      'stage': stage}, outputs=[output / 'analysis.json', output / 'item-diagnostics.jsonl'],
                     verification={**scope, 'native_move_list_answers_recomputed': len(native),
                                   'same_ids_questions_variants_and_gold_verified': True})
    proof['inputs'] = input_hashes
    for path in inputs:
        if file_signature(path) != signatures[str(path)]:
            raise ValueError('Diagnosis input changed before manifest completion')
    atomic_json(output / 'manifest.json', proof)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--predictions', nargs='+', required=True, help='Prediction files in checkpoint order')
    parser.add_argument('--stage', choices=STAGES, default='dynamic_current')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = diagnose_predictions(args.validation, args.predictions, args.output, stage=args.stage)
    print(json.dumps({'status': result['status'], 'candidates': len(result['candidates']),
                      'native_move_lists': result['native_move_list_answers_recomputed']}))


if __name__ == '__main__':
    main()
