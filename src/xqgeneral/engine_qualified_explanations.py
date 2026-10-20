"""Select unchanged reviewed training labels using a completed engine reference."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

from .evidence import atomic_json, digest, manifest
from .explanations import parse_explanation
from .recorded_coach import SPLITS, check_artifacts, checked_manifest
from .reviewed_explanations import (ENGINE_QUALIFIED_KIND, REFERENCE_CORRECTION_KIND,
                                    SOURCE_KINDS, source_contract)


def qualification_rule():
    return {'first_move_no_mistake_required': True,
            'all_main_variation_plies_no_mistake_required': True,
            'expected_score_loss_strictly_less_than': 0.10,
            'original_structured_and_legal_contract_required': True}


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _bound(proof, path, root, section):
    path, root = Path(path), Path(root)
    keys = [str(path)]
    if path.is_relative_to(root):
        keys.append(str(path.relative_to(root)))
    matches = [proof.get(section, {})[key] for key in keys if key in proof.get(section, {})]
    if len(matches) != 1:
        raise ValueError('Reference artifact has no unique exact path binding')
    return matches[0]


def _check_bound(proof, path, root, section, raw=None):
    path = Path(path)
    raw = path.read_bytes() if raw is None else raw
    if _bound(proof, path, root, section) != {'sha256': _hash(raw), 'bytes': len(raw)}:
        raise ValueError('Reference artifact bytes differ from their completed producer')
    return raw


def _source(data):
    proof = checked_manifest(data / 'manifest.json')
    if proof.get('kind') not in SOURCE_KINDS | {REFERENCE_CORRECTION_KIND}:
        raise ValueError('A supported original reviewed source, before engine selection, is required')
    paths = [data / f'{split}.jsonl' for split in SPLITS]
    check_artifacts(proof, paths)
    blobs = {split: path.read_bytes() for split, path in zip(SPLITS, paths)}
    for raw in blobs.values():
        if raw and (not raw.endswith(b'\n') or any(not line.strip() for line in raw.splitlines())):
            raise ValueError('Source split files require complete nonempty record lines')
    counts = {split: len(raw.splitlines()) for split, raw in blobs.items()}
    source_contract(proof, counts)
    if not counts['train']:
        raise ValueError('The original training corpus is empty')
    return proof, blobs, counts


def _reference(reference, data, root, examples):
    deep_path = reference / 'full/manifest.json'
    readback_path = reference / 'full/independent-readback.json'
    deep, readback = _read(deep_path), _read(readback_path)
    metrics = _read(reference / 'full/metrics.json')
    pipeline = _read(reference / 'full-pipeline-exit.json')
    producer_exit = _read(reference / 'full/exit.json')
    reader_exit = _read(reference / 'full-readback-exit.json')
    if (deep.get('status') != 'complete' or
            deep.get('kind') != 'existing_training_teacher_labels_deep_engine_reference' or
            deep.get('config', {}).get('scope') != 'full' or
            deep.get('config', {}).get('indices') != list(range(examples)) or
            deep.get('loss_cutoff') != 'expected_score_loss_strictly_less_than_0.10' or
            deep.get('metrics') != metrics or
            producer_exit.get('returncode') != 0 or producer_exit.get('manifest_sha256') != digest(deep_path) or
            reader_exit.get('returncode') != 0 or pipeline.get('returncode') != 0 or
            pipeline.get('independent_readback_complete') is not True or
            pipeline.get('stages') != [{'stage': 'producer', 'returncode': 0}, {'stage': 'readback', 'returncode': 0}]):
        raise ValueError('A completed full reference producer and independent readback are required')
    required = ['all_original_label_records_and_full_native_histories_verified',
                'all_stored_engine_responses_reparsed', 'all_complete_pv_loss_aggregates_recomputed',
                'original_unscored_and_same_best_additional_responses_verified']
    if (readback.get('status') != 'complete_training_label_reference_independent_readback' or
            readback.get('scope') != 'full' or readback.get('examples') != examples or
            not all(readback.get(key) is True for key in required) or
            readback.get('labels_or_predictions_rewritten') is not False or
            readback.get('validation_or_independent_test_answers_read') is not False or
            readback.get('producer_manifest_sha256') != digest(deep_path) or
            readback.get('plan_sha256') != digest(reference / 'plan.json') or
            readback.get('script_sha256') != digest(reference / 'readback.py')):
        raise ValueError('The full train-only retained-response reference contract differs')
    for key, value in readback.get('recomputed_metrics', {}).items():
        if metrics.get(key) != value:
            raise ValueError('Independent engine-reference metrics differ')
    if metrics.get('examples') != examples:
        raise ValueError('Reference coverage differs from the original training corpus')
    _check_bound(deep, data / 'manifest.json', root, 'inputs')
    _check_bound(deep, data / 'train.jsonl', root, 'inputs')
    paths = [deep_path, readback_path, reference / 'full/metrics.json',
             reference / 'full-pipeline-exit.json', reference / 'full/exit.json',
             reference / 'full-readback-exit.json', reference / 'plan.json', reference / 'readback.py']
    return deep, readback, paths


def _decision(index, row, line, scored_raw):
    scored = json.loads(scored_raw)
    if (scored.get('row_index') != index or scored.get('id') != row['id'] or
            scored.get('record') != row or scored.get('raw') != row['answer']):
        raise ValueError('Engine-reference row differs from the exact original training record')
    verdict = scored.get('judgment', {})
    if not all(verdict.get(key) is True for key in ['parse_valid', 'contract_valid', 'first_move_legal', 'pv_legal']):
        raise ValueError('A complete legal and structured reference is required before score qualification')
    first, losses = verdict.get('first_move_expected_score_loss'), verdict.get('pv_move_losses')
    def score(value):
        return type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1
    variation = parse_explanation(row['answer']).get('pv')
    judgments = verdict.get('engine_judgments')
    if (not score(first) or not isinstance(losses, list) or not losses or
            not all(score(value) for value in losses) or not isinstance(variation, list) or
            len(losses) != len(variation) or not isinstance(judgments, list) or
            not all(isinstance(item, dict) for item in judgments) or
            [item.get('move') for item in judgments] != variation):
        raise ValueError('Every main-variation ply needs a finite bound loss for the same move')
    first_passed, full_passed = first < .10, all(value < .10 for value in losses)
    if verdict.get('first_move_no_mistake') is not first_passed:
        raise ValueError('Recommended-move verdict differs from its retained loss')
    reasons = []
    if not first_passed:
        reasons.append('first_move_expected_score_loss_at_or_above_0.10')
    if not full_passed:
        reasons.append('main_variation_ply_expected_score_loss_at_or_above_0.10')
    return {'row_index': index, 'id': row['id'], 'source_line_sha256': _hash(line),
            'reference_row_sha256': _hash(scored_raw), 'first_move_expected_score_loss': first,
            'maximum_main_variation_ply_expected_score_loss': max(losses),
            'main_variation_plies': len(losses), 'first_move_passed': first_passed,
            'full_main_variation_passed': full_passed, 'eligible': first_passed and full_passed,
            'rejection_reasons': reasons}


def qualify(data, reference, output, *, artifact_root=None, workers=2):
    """Select original train lines; copy held-out bytes without parsing their records."""
    if type(workers) is not int or workers < 1:
        raise ValueError('A positive CPU reader budget is required')
    data, reference, output = (Path(p).resolve() for p in [data, reference, output])
    root = Path(artifact_root or Path.cwd()).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing qualified data; use a fresh output directory')
    source, blobs, original_counts = _source(data)
    lines = blobs['train'].splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if (any(not isinstance(row, dict) or row.get('split') != 'train' or row.get('stage') != 'explanation' or
            not isinstance(row.get('id'), str) or not row['id'] or
            not isinstance(row.get('feature_key'), str) or not row['feature_key'] for row in rows) or
            len({row['id'] for row in rows}) != len(rows) or
            len({row['feature_key'] for row in rows}) != len(rows)):
        raise ValueError('Original training labels must have unique identities and protected split membership')
    deep, readback, reference_paths = _reference(reference, data, root, len(rows))
    def one(index):
        path = reference / 'full/rows' / f'{index:06}.json'
        raw = _check_bound(deep, path, root, 'outputs')
        return _decision(index, rows[index], lines[index], raw)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        decisions = list(pool.map(one, range(len(rows))))
    counts = Counter(examples=len(rows), eligible=0, rejected=0, first_move_passed=0,
                     full_main_variation_passed=0, judged_main_variation_plies=0)
    for item in decisions:
        counts['eligible' if item['eligible'] else 'rejected'] += 1
        counts['first_move_passed'] += item['first_move_passed']
        counts['full_main_variation_passed'] += item['full_main_variation_passed']
        counts['judged_main_variation_plies'] += item['main_variation_plies']
    expected = readback.get('actual_retained_metric_counts', {})
    if (counts['first_move_passed'] != expected.get('first_move_no_mistake') or
            counts['full_main_variation_passed'] != expected.get('full_pv_no_mistake') or
            counts['judged_main_variation_plies'] != readback['recomputed_metrics'].get('judged_legal_pv_plies') or
            not counts['eligible']):
        raise ValueError('Qualification does not reproduce the complete bound reference population')
    if checked_manifest(data / 'manifest.json') != source:
        raise ValueError('The original reviewed dataset changed during qualification')
    retained = b''.join(line for line, decision in zip(lines, decisions) if decision['eligible'])
    output.mkdir(parents=True)
    paths = []
    for split in SPLITS:
        path = output / f'{split}.jsonl'
        with path.open('xb') as stream:
            stream.write(retained if split == 'train' else blobs[split])
        paths.append(path)
    witness = output / 'qualification.json'
    atomic_json(witness, {'rule': qualification_rule(), 'counts': dict(counts), 'decisions': decisions})
    by_split = dict(original_counts, train=counts['eligible'])
    summary = {'status': 'complete', 'by_split': by_split, 'original_by_split': original_counts,
        'qualification_counts': dict(counts), 'qualification_rule': qualification_rule(),
        'all_retained_training_record_bytes_preserved': True, 'original_split_membership_preserved': True,
        'heldout_files_byte_identical': True, 'validation_or_independent_test_records_parsed': False,
        'completed_parent_native_and_isolation_proofs_reused': True,
        'completed_semantic_acceptance_proofs_reused': True,
        'all_training_rows_have_inline_or_supplemental_acceptance': True,
        'all_training_first_moves_and_main_variations_passed_bound_reference': True,
        'all_original_training_records_and_reference_rows_hash_bound': True,
        'reference_full_independent_readback_complete': True,
        'reference_independent_readback_sha256': digest(reference / 'full/independent-readback.json'),
        'training_labels_rewritten': False, 'student_predictions_seen': False,
        'independent_test_used_for_training_or_selection': False, 'new_teacher_records_generated': 0,
        'model_weights_loaded_or_training_started': False, 'engine_search_started': False,
        'waiting_training_recipe_or_queue_modified': False, 'four_new_courses_required_before_sft': True,
        'student_strength_or_coaching_benefit_proven': False}
    atomic_json(output / 'verification.json', summary)
    inputs = [data / 'manifest.json', *[data / f'{s}.jsonl' for s in SPLITS], *reference_paths]
    proof = manifest(ENGINE_QUALIFIED_KIND,
        {'data': str(data), 'reference': str(reference), 'output': str(output),
         'artifact_root': str(root), 'workers': workers, 'qualification_rule': qualification_rule()},
        inputs, [*paths, witness, output / 'verification.json'], summary)
    atomic_json(output / 'manifest.json', proof)
    return summary


def qualified_source_contract(proof, counts):
    """Retain semantic lineage and the original held-out population at admission."""
    config, verified = proof.get('config', {}), proof.get('verification', {})
    required = ['all_retained_training_record_bytes_preserved', 'original_split_membership_preserved',
                'heldout_files_byte_identical', 'completed_parent_native_and_isolation_proofs_reused',
                'completed_semantic_acceptance_proofs_reused',
                'all_training_rows_have_inline_or_supplemental_acceptance',
                'all_training_first_moves_and_main_variations_passed_bound_reference',
                'all_original_training_records_and_reference_rows_hash_bound',
                'reference_full_independent_readback_complete']
    if (not all(verified.get(key) is True for key in required) or
            any(verified.get(key) is not False for key in ['training_labels_rewritten', 'student_predictions_seen',
                'independent_test_used_for_training_or_selection', 'validation_or_independent_test_records_parsed']) or
            config.get('qualification_rule') != qualification_rule() or verified.get('qualification_rule') != qualification_rule()):
        raise ValueError('Completed engine qualification and unchanged semantic acceptance are required')
    data = Path(config['data'])
    parent_path = data / 'manifest.json'
    parent = checked_manifest(parent_path)
    if parent.get('kind') not in SOURCE_KINDS | {REFERENCE_CORRECTION_KIND}:
        raise ValueError('Qualified labels require the original reviewed parent, without recursive qualification')
    check_artifacts(proof, [parent_path], 'inputs')
    original = {split: parent.get('verification', {}).get('by_split', {}).get(split, 0) for split in SPLITS}
    source_contract(parent, original)
    if (verified.get('original_by_split') != original or not 0 < counts['train'] <= original['train'] or
            any(counts[split] != original[split] for split in ['validation', 'test']) or
            verified.get('qualification_counts', {}).get('eligible') != counts['train']):
        raise ValueError('Qualified data must preserve the original held-out population and exact train subset count')
    reference = Path(config['reference'])
    _, readback, paths = _reference(reference, data, Path(config['artifact_root']), original['train'])
    check_artifacts(proof, paths, 'inputs')
    if verified.get('reference_independent_readback_sha256') != digest(reference / 'full/independent-readback.json'):
        raise ValueError('Engine qualification differs from the completed full readback')
    for split in ['validation', 'test']:
        selected = Path(config['output']) / f'{split}.jsonl'
        check_artifacts(proof, [selected])
        if selected.read_bytes() != (data / f'{split}.jsonl').read_bytes():
            raise ValueError('Qualified data changed the protected held-out bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--artifact-root')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(qualify(args.data, args.reference, args.output,
        artifact_root=args.artifact_root, workers=args.workers), ensure_ascii=False))


if __name__ == '__main__':
    main()
