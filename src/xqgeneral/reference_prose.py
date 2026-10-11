"""Correct audited held-out reference prose before a new student experiment."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .evidence import atomic_json, digest, load_jsonl, manifest
from .explanations import parse_explanation
from .recorded_coach import SPLITS, checked_manifest, ordered_native_map
from .reviewed_explanations import REFERENCE_CORRECTION_KIND, checked_label, source_contract
from .revise_prose import TEACHER

AUDIT_KIND = 'independently_reviewed_heldout_reference_prose'
SCORES = ('factual_correctness', 'strategic_reasoning', 'clarity', 'instruction_adherence')


def teacher_annotation(row):
    """Hash teacher content separately from color-context and review ancestry."""
    return {'id': row['id'], 'explanation': parse_explanation(row['answer'])['explanation'], **TEACHER}


def annotation_hash(annotation):
    return hashlib.sha256(json.dumps(annotation, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def checked_annotation(annotation):
    if (set(annotation) != {'id', 'explanation', *TEACHER} or
            not isinstance(annotation['id'], str) or not annotation['id'] or
            not isinstance(annotation['explanation'], str) or
            any(annotation[k] != v for k, v in TEACHER.items())):
        raise ValueError('Reference teacher content or identity differs')
    return annotation


def checked_decision(decision):
    claims = decision.get('unsupported_claims')
    if (any(type(decision.get(k)) is not int or not 1 <= decision[k] <= 5 for k in SCORES) or
            not isinstance(claims, list) or any(not isinstance(c, str) for c in claims) or
            not isinstance(decision.get('reason'), str) or not decision['reason'].strip() or
            not isinstance(decision.get('reviewer'), str) or not decision['reviewer'] or
            any(decision.get(k) != v for k, v in TEACHER.items())):
        raise ValueError('A complete individual Astra Low reference decision is required')
    accepted = all(decision[k] >= 4 for k in SCORES) and not claims
    if decision.get('verdict') != ('accept' if accepted else 'reject'):
        raise ValueError('Reference verdict differs from the individual scores and claims')


def checked_outputs(proof, paths):
    """Match actual files when older producer manifests use relative names."""
    for path in paths:
        path = Path(path)
        bindings = [v for name, v in proof.get('outputs', {}).items() if Path(name).samefile(path)]
        if (len(bindings) != 1 or bindings[0] != {'sha256': digest(path), 'bytes': path.stat().st_size}):
            raise ValueError('Required reference artifact is not uniquely bound to its exact bytes')


def read_audit(root):
    """Resolve exact accepted bodies through independently reviewed rejection chains."""
    root = Path(root); path = root / 'manifest.json'; proof = checked_manifest(path)
    verification = proof.get('verification', {})
    if (proof.get('kind') != AUDIT_KIND or verification.get('human_rating') is not False or
            verification.get('student_predictions_seen') is not False or
            verification.get('test_used_for_training_or_selection') is not False):
        raise ValueError('Completed student-blind reference audit is required')
    resolved_path = root / 'references.jsonl'; checked_outputs(proof, [resolved_path])
    annotations, originals, revisions, decisions = {}, {}, {}, defaultdict(list)
    for name, artifact in {**proof['inputs'], **proof['outputs']}.items():
        if Path(name).suffix != '.jsonl' or Path(name).samefile(resolved_path):
            continue
        for row in load_jsonl(name):
            if 'teacher_annotation' in row:
                annotation = checked_annotation(row['teacher_annotation']); sha = annotation_hash(annotation)
                if sha != row.get('reviewed_annotation_sha256'):
                    raise ValueError('Reference packet content hash differs')
                annotations[sha] = annotation
                if 'corrected_from_annotation_sha256' not in row:
                    binding = sha, row['source_label_line_sha256']
                    if row['id'] != annotation['id'] or originals.setdefault(row['id'], binding) != binding:
                        raise ValueError('Original reference bindings conflict')
            elif 'explanation' in row and 'author_actor' in row:
                annotation = checked_annotation({k: row[k] for k in ('id', 'explanation', *TEACHER)})
                sha = annotation_hash(annotation); annotations[sha] = annotation
                binding = row, artifact['sha256']
                if revisions.setdefault(sha, binding) != binding:
                    raise ValueError('A revised body has conflicting authors or ancestry')
            elif 'verdict' in row:
                checked_decision(row)
                decisions[row['id'], row['reviewed_annotation_sha256']].append((artifact['sha256'], row))
    selected = load_jsonl(resolved_path)
    if not selected or len({r['id'] for r in selected}) != len(selected) or {r['id'] for r in selected} != set(originals):
        raise ValueError('Reference resolution must cover every audited original exactly once')
    result = {}
    for row in selected:
        identity = row['id']; original_sha, line_sha = originals[identity]; final_sha = row['reviewed_annotation_sha256']
        if (row.get('source_annotation_sha256') != original_sha or
                row.get('source_label_line_sha256') != line_sha or final_sha not in annotations or
                annotations[final_sha]['id'] != identity):
            raise ValueError('Resolved reference differs from its original content or label line')
        current, seen = final_sha, set()
        while current != original_sha:
            if current in seen or current not in revisions:
                raise ValueError('Reference revision ancestry is incomplete or cyclic')
            seen.add(current); child, child_file_sha = revisions[current]
            parent = child.get('corrected_from_annotation_sha256'); rejection = child.get('correction_review_sha256')
            if (parent not in annotations or annotations[parent]['id'] != identity or
                    child.get('source_label_line_sha256') != line_sha or
                    not isinstance(child.get('author_actor'), str) or not child['author_actor'] or
                    annotations[current]['explanation'] == annotations[parent]['explanation'] or
                    not 80 <= len(re.findall(r'[\u4e00-\u9fff]', child['explanation'])) <= 250):
                raise ValueError('Reference correction requires a substantive body and exact parent')
            prior_author = revisions[parent][0]['author_actor'] if parent in revisions else None
            if not any(file_sha == rejection and d['verdict'] == 'reject' and
                       d.get('source_label_line_sha256') == line_sha and d['reviewer'] != prior_author
                       for file_sha, d in decisions[identity, parent]):
                raise ValueError('Reference correction must bind the exact independently rejected parent')
            current = parent
        final_author, final_file = revisions[final_sha] if final_sha in revisions else (None, None)
        accepted = []
        for file_sha, d in decisions[identity, final_sha]:
            if d['verdict'] != 'accept' or d.get('source_label_line_sha256') != line_sha:
                continue
            if final_author is not None and (
                    d['reviewer'] == final_author['author_actor'] or
                    d.get('corrected_from_annotation_sha256') != final_author['corrected_from_annotation_sha256'] or
                    d.get('correction_review_sha256') != final_author['correction_review_sha256'] or
                    d.get('author_annotation_file_sha256') != final_file):
                continue
            accepted.append(file_sha)
        if not accepted:
            raise ValueError('Final reference body has no exact independent acceptance')
        result[identity] = {**row, 'annotation': annotations[final_sha],
                            'acceptance_file_sha256': sorted(accepted)[0],
                            'review_manifest_sha256': digest(path), 'revision_count': len(seen)}
    if checked_manifest(path) != proof:
        raise ValueError('Reference audit changed while resolving acceptance')
    return result


def correct_references(data, audit, output, workers=8):
    """Preserve training bytes and all question/engine facts; revise only audited prose."""
    data, audit, output = Path(data), Path(audit), Path(output)
    if output.exists():
        raise FileExistsError('Preserve old datasets; use a fresh corrected-reference output')
    if type(workers) is not int or workers < 1:
        raise ValueError('A positive native worker budget is required')
    source_path = data / 'manifest.json'; proof = checked_manifest(source_path)
    if proof.get('kind') != 'combined_reviewed_explanation_dataset':
        raise ValueError('Reference correction requires the original native-checked combined dataset')
    paths = [data / f'{s}.jsonl' for s in SPLITS]; checked_outputs(proof, paths)
    selected = read_audit(audit); blocks, labels, counts, ids = {}, {}, {}, set()
    for split, path in zip(SPLITS, paths):
        raw = path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            raise ValueError('Original reference label records require a final newline')
        blocks[split] = raw.splitlines(keepends=True); counts[split] = len(blocks[split])
        for line in blocks[split]:
            row = json.loads(line)
            if row['id'] in ids or row['split'] != split or row['stage'] != 'explanation':
                raise ValueError('Original label identity, stage or split differs')
            ids.add(row['id']);labels[row['id']] = row, line
    source_contract(proof, counts)
    original_proof = proof['verification']
    if (original_proof.get('complete_native_histories_facts_and_all_continuations_revalidated') is not True or
            original_proof.get('game_overlap') != 0 or original_proof.get('root_and_future_and_color_overlap') != 0):
        raise ValueError('Completed native histories and protected-split isolation are required')
    required = {i for i, (r, _) in labels.items() if r['split'] != 'train' and r.get('prose_review') is None}
    if set(selected) != required:
        raise ValueError('Audit must cover all previously unrated held-out references, and no training rows')
    corrected, changed_counts = {}, Counter()
    for identity, review in selected.items():
        row, line = labels[identity]; original_sha = annotation_hash(teacher_annotation(row))
        if (review['source_label_line_sha256'] != hashlib.sha256(line).hexdigest() or
                review['source_annotation_sha256'] != original_sha):
            raise ValueError('Reviewed reference no longer matches the exact original label')
        value = parse_explanation(row['answer']); prose = review['annotation']['explanation']
        if value['explanation'] == prose:
            continue
        value['explanation'] = prose
        revised = dict(row, answer=json.dumps(value, ensure_ascii=False), prose_review={
            'source_annotation_sha256': original_sha,
            'reviewed_annotation_sha256': review['reviewed_annotation_sha256'],
            'review_manifest_sha256': review['review_manifest_sha256'],
            'acceptance_file_sha256': review['acceptance_file_sha256'], 'changed': True,
            'kind': 'heldout_reference_correction', 'human_rating': False,
            'independent_teacher_revision': True, 'annotation_hash_profile': 'sha256_utf8_sorted_compact_json',
            **TEACHER})
        corrected[identity] = revised;changed_counts[row['split']] += 1
    for _ in ordered_native_map(checked_label, list(corrected.values()), workers):
        pass
    if checked_manifest(source_path) != proof:
        raise ValueError('Original dataset changed during reference correction')
    if read_audit(audit) != selected:
        raise ValueError('Reference acceptance changed during native checks')
    output.mkdir(parents=True); outputs = []
    for split in SPLITS:
        path = output / f'{split}.jsonl'
        path.write_bytes(b''.join((json.dumps(corrected[row['id']], ensure_ascii=False) + '\n').encode()
                         if row['id'] in corrected else line
                         for line in blocks[split] for row in [json.loads(line)]));outputs.append(path)
    assert (data / 'train.jsonl').read_bytes() == (output / 'train.jsonl').read_bytes()
    summary = {'status': 'complete', 'by_split': counts, 'audited_heldout_references': len(selected),
        'revised_heldout_prose': len(corrected), 'revised_heldout_prose_by_split': dict(changed_counts),
        'training_file_byte_identical': True, 'unchanged_heldout_record_bytes_preserved': True,
        'heldout_only_prose_and_review_metadata_changed': True,
        'identities_structured_labels_histories_and_original_splits_preserved': True,
        'completed_parent_native_and_isolation_proofs_reused': True,
        'revised_bodies_complete_native_histories_and_continuations_revalidated': True,
        'all_original_unrated_heldout_annotations_independently_accepted': True,
        'all_training_rows_have_inline_or_supplemental_acceptance': True,
        'completed_semantic_acceptance_proofs_reused': True,
        'fresh_heldout_neural_semantic_review_performed': True, 'human_rating': False,
        'source_label_bytes_and_original_splits_preserved': not corrected,
        'student_predictions_seen': False, 'independent_test_used_for_training_or_selection': False,
        'model_or_gpu_loaded': False, 'active_training_inputs_changed': False,
        'four_new_courses_required_before_sft': True, 'student_training_benefit_measured': False}
    atomic_json(output / 'verification.json', summary)
    atomic_json(output / 'manifest.json', manifest(REFERENCE_CORRECTION_KIND,
        {'data': str(data), 'audit': str(audit), 'workers': workers},
        [source_path, *paths, audit / 'manifest.json'], [*outputs, output / 'verification.json'], summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(correct_references(args.data, args.audit, args.output, args.workers), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
