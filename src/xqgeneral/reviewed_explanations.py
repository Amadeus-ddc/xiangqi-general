"""Combine completed reviewed labels while preserving bytes and held-out splits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .evidence import atomic_json, digest, history_key, manifest, position_key
from .explanations import continuation_positions, parse_explanation, validate_explanation
from .recorded_coach import SPLITS, checked_manifest, ordered_native_map
from .revise_prose import TEACHER
from .rules import adjudicate, replay
from .symmetry import mirror_fen


SOURCE_KINDS = {'fully_reviewed_selfplay_teacher_append',
                'reviewed_recorded_coach_initial_teacher_labels',
                'combined_reviewed_explanation_dataset'}


def checked_label(row):
    """Validate native history, answer facts and every continuation footprint."""
    if (row.get('stage') != 'explanation' or row.get('split') not in SPLITS or
            not isinstance(row.get('id'), str) or not row['id'] or
            not isinstance(row.get('game_id'), str) or not row['game_id']):
        raise ValueError('A preassigned explanation label identity and split are required')
    if any(row.get('teacher', {}).get(k) != v for k, v in TEACHER.items()):
        raise ValueError('Reviewed label teacher identity differs')
    review = row.get('prose_review')
    if review is not None and (not isinstance(review, dict) or
            any(not isinstance(review.get(k), str) or not re.fullmatch('[0-9a-f]{64}', review[k])
                for k in ['source_annotation_sha256', 'reviewed_annotation_sha256', 'review_manifest_sha256'])
            or type(review.get('changed')) is not bool):
        raise ValueError('Reviewed label is missing its exact upstream acceptance identities')
    history = replay(row['initial_fen'], row['moves'])
    if (history != row['history'] or history[-1] != row['fen'] or
            history_key(history) != row['feature_key'] or
            ('ply' in row and row['ply'] != len(row['moves']))):
        raise ValueError('Reviewed label differs from its native full-history context')
    if adjudicate(row['initial_fen'], row['moves'])['ended']:
        raise ValueError('Reviewed explanation root is terminal')
    value = parse_explanation(row['answer'])
    verdict = validate_explanation(row['fen'], value, require_facts=True, require_branches=True)
    if not verdict['valid']:
        raise ValueError(f'Reviewed explanation native answer contract failed: {verdict["errors"]}')
    lines = {tuple(line) for line in [value['pv'], *[b['pv'] for b in value['branches']],
                                      row.get('future_moves', []), *row.get('future_branches', [])]}
    positions = continuation_positions(row, {'pv': [], 'branches': [{'pv': list(line)} for line in lines]})
    positions.update(position_key(mirror_fen(fen)) for fen in list(positions))
    return {'id': row['id'], 'feature_key': row['feature_key'], 'game_id': row['game_id'],
            'split': row['split'], 'positions': sorted(positions)}


def source_contract(proof, counts):
    """Reuse exact completed semantic acceptance; do not invent a fresh review."""
    kind, verification = proof.get('kind'), proof.get('verification', {})
    if kind not in SOURCE_KINDS:
        raise ValueError('A supported completed reviewed-label dataset is required')
    if ({s: verification.get('by_split', {}).get(s, 0) for s in SPLITS} != counts or
            set(verification.get('by_split', {})) - set(SPLITS)):
        raise ValueError('Actual reviewed label counts differ from their completed producer')
    accepted = (
        kind == 'fully_reviewed_selfplay_teacher_append' and
        verification.get('all2048_exact_independent_neural_acceptance_chains_verified') is True and
        verification.get('all6662_prior_train_records_and_byte_prefix_preserved') is True and
        verification.get('heldout_files_byte_identical') is True
    ) or (
        kind == 'reviewed_recorded_coach_initial_teacher_labels' and
        verification.get('original_teacher_annotations') == sum(counts.values()) and
        verification.get('independently_accepted_original_annotations') == sum(counts.values())
    ) or (
        kind == 'combined_reviewed_explanation_dataset' and
        verification.get('source_label_bytes_and_original_splits_preserved') is True and
        verification.get('completed_semantic_acceptance_proofs_reused') is True and
        verification.get('all_training_rows_have_inline_or_supplemental_acceptance') is True
    )
    if not accepted:
        raise ValueError('Completed dataset does not establish full upstream semantic acceptance')


def legacy_training_reviews(roots):
    """Bind older reviewed training rows that predate inline review metadata."""
    labels, proofs = {}, []
    for name in roots:
        root = Path(name); path = root / 'manifest.json'; proof = checked_manifest(path)
        verified = proof.get('verification', {})
        if (proof.get('kind') != 'reviewed_selfplay_explanation_replay_pool' or
                verified.get('all_384_annotations_cross_reviewed') is not True or
                verified.get('all_39_revisions_independently_accepted') is not True or
                verified.get('changed_field_only') != 'stage' or
                verified.get('prompts_and_answers_byte_values_preserved') is not True):
            raise ValueError('A completed exact legacy training acceptance pool is required')
        for split in SPLITS:
            p = root / f'{split}.jsonl'
            if proof['outputs'].get(str(p)) != {'sha256': digest(p), 'bytes': p.stat().st_size}:
                raise ValueError('Legacy acceptance split is absent from its exact producer outputs')
            rows = [json.loads(line) for line in p.read_bytes().splitlines() if line.strip()]
            if len(rows) != verified.get(split, 0) or (split != 'train' and rows):
                raise ValueError('Legacy acceptance pool must contain only its declared training rows')
            for row in rows:
                if row['split'] != 'train' or row['stage'] != 'selfplay_explanation' or row['id'] in labels:
                    raise ValueError('Legacy acceptance identities or stages differ')
                labels[row['id']] = dict(row, stage='explanation')
        proofs.append((path, proof))
    return labels, proofs


def merge_reviewed(inputs, output, workers=8, legacy_review_data=()):
    """Concatenate exact source split files after native cross-split isolation."""
    if not inputs or type(workers) is not int or workers < 1:
        raise ValueError('Reviewed sources and a positive worker budget are required')
    output, roots = Path(output), [Path(p) for p in inputs]
    if output.exists():
        raise FileExistsError('Preserve existing label outputs; use a fresh dataset directory')
    if len({p.resolve() for p in roots}) != len(roots):
        raise ValueError('A reviewed source dataset cannot be supplied twice')
    legacy, source_proofs = legacy_training_reviews(legacy_review_data)
    rows, blobs, inventory, ids, keys = [], [], [], set(), set()
    legacy_counts, supplemental = Counter(), 0
    for root in roots:
        proof_path = root / 'manifest.json'
        proof = checked_manifest(proof_path)
        if proof.get('kind') not in SOURCE_KINDS:
            raise ValueError('A supported completed reviewed-label dataset is required')
        counts, blocks = dict.fromkeys(SPLITS, 0), {}
        for split in SPLITS:
            path = root / f'{split}.jsonl'
            raw = path.read_bytes()
            expected = proof['outputs'].get(str(path))
            if expected != {'sha256': digest(path), 'bytes': len(raw)}:
                raise ValueError('Reviewed split file is absent from its exact producer outputs')
            if raw and not raw.endswith(b'\n'):
                raise ValueError('Original label files must end with a record newline')
            for line in raw.splitlines():
                if not line.strip():
                    raise ValueError('Reviewed labels cannot contain empty record lines')
                row = json.loads(line)
                if row['split'] != split:
                    raise ValueError('Reviewed label file changes the original game split')
                if row.get('prose_review') is None:
                    if proof['kind'] == 'reviewed_recorded_coach_initial_teacher_labels':
                        raise ValueError('New recorded labels require exact inline acceptance metadata')
                    legacy_counts[split] += 1
                    if split == 'train':
                        if legacy.get(row['id']) != row:
                            raise ValueError('Legacy training label is missing its exact supplemental acceptance')
                        supplemental += 1
                if row['id'] in ids or row['feature_key'] in keys:
                    raise ValueError('Reviewed datasets overlap by label ID or board-history context')
                ids.add(row['id']); keys.add(row['feature_key']); rows.append(row)
                counts[split] += 1
            blocks[split] = raw
        source_contract(proof, counts)
        inventory.append({'source': str(root), 'manifest_sha256': digest(proof_path), 'by_split': counts,
                          'source_kind': proof['kind']})
        blobs.append(blocks); source_proofs.append((proof_path, proof))
    if not any(r['split'] == 'train' for r in rows):
        raise ValueError('The combined reviewed training split is empty')
    owners, game_owners = {}, {}
    for item in ordered_native_map(checked_label, rows, workers):
        split = item['split']
        if game_owners.setdefault(item['game_id'], split) != split:
            raise ValueError('Reviewed game identity overlaps held-out splits')
        for position in item['positions']:
            if owners.setdefault(position, split) != split:
                raise ValueError('Reviewed root, continuation or color counterpart overlaps held-out splits')
    for path, proof in source_proofs:
        if checked_manifest(path) != proof:
            raise ValueError('Reviewed source changed during native merge checks')
    output.mkdir(parents=True)
    paths = []
    for split in SPLITS:
        path = output / f'{split}.jsonl'
        path.write_bytes(b''.join(block[split] for block in blobs)); paths.append(path)
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
        'by_split': dict(Counter(r['split'] for r in rows)), 'reviewed_label_rows': len(rows),
        'distinct_full_history_feature_keys': len(keys), 'source_datasets': inventory,
        'source_label_bytes_and_original_splits_preserved': True,
        'all_source_train_validation_test_segments_byte_preserved': True,
        'complete_native_histories_facts_and_all_continuations_revalidated': True,
        'all_color_corresponding_geometric_forecasts_reserved': True,
        'game_overlap': 0, 'root_and_future_and_color_overlap': 0,
        'completed_semantic_acceptance_proofs_reused': True,
        'legacy_rows_without_inline_review_metadata_by_split': dict(legacy_counts),
        'legacy_training_rows_bound_to_exact_supplemental_acceptance': supplemental,
        'all_training_rows_have_inline_or_supplemental_acceptance': True,
        'legacy_heldout_semantic_acceptance_not_newly_established': True,
        'existing_color_derived_rows_preserved': sum(r['teacher'].get('independent_teacher_call') is False
                                                  for r in rows),
        'fresh_neural_or_human_semantic_review_performed': False,
        'teacher_annotations_generated_or_repaired': False, 'derived_teacher_records_generated': 0,
        'independent_test_used_for_isolation_only': True, 'student_training_started': False,
        'four_new_courses_required_before_sft': True, 'active_training_inputs_changed': False,
        'student_training_benefit_measured': False}
    atomic_json(output / 'verification.json', summary)
    atomic_json(output / 'manifest.json', manifest('combined_reviewed_explanation_dataset',
        {'inputs': list(map(str, roots)), 'workers': workers,
         'legacy_review_data': list(map(str, legacy_review_data))},
        [p for root in [*roots, *map(Path, legacy_review_data)]
         for p in [root / 'manifest.json', *[root / f'{s}.jsonl' for s in SPLITS]]],
        [*paths, output / 'verification.json'], summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--public-evidence')
    parser.add_argument('--legacy-review-data', action='append', default=[])
    args = parser.parse_args()
    summary = merge_reviewed(args.inputs, args.output, args.workers, args.legacy_review_data)
    if args.public_evidence:
        atomic_json(args.public_evidence, dict(summary, manifest_sha256=digest(Path(args.output) / 'manifest.json')))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
