"""Apply hash-bound, independently accepted train prose in a fresh dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from .curriculum_data import verify_splits
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from .explanations import continuation_positions, parse_explanation, validate_explanation

TEACHER = {'teacher_model': 'gpt-6-astra', 'reasoning_effort': 'low', 'backend': 'codex_subagent'}


def annotation_hash(annotation):
    return hashlib.sha256(json.dumps(annotation, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_annotation(row):
    return {'id': row['id'], 'explanation': parse_explanation(row['answer'])['explanation'], **row['teacher']}


def revise_row(row, annotation, source_hash, reviewed_hash, review_manifest_hash):
    if row['split'] != 'train' or row['stage'] != 'explanation':
        raise ValueError('Only train explanation prose can be revised')
    if annotation.get('id') != row['id'] or any(annotation.get(k) != v for k, v in TEACHER.items()):
        raise ValueError('Revision identity differs from the authorized initial teacher')
    if annotation_hash(source_annotation(row)) != source_hash or annotation_hash(annotation) != reviewed_hash:
        raise ValueError('Source prose or reviewed annotation hash differs')
    prose = annotation.get('explanation')
    if not isinstance(prose, str) or len(prose) < 80 or not re.search(r'[\u4e00-\u9fff]', prose):
        raise ValueError('Revision must contain substantive Chinese teacher prose')
    value = parse_explanation(row['answer'])
    changed = value['explanation'] != prose
    value['explanation'] = prose
    proof = validate_explanation(row['fen'], value, require_branches=True)
    if not proof['valid']:
        raise ValueError(f'Revised explanation violates structured facts: {proof["errors"]}')
    continuation_positions(row, value)
    return dict(row, answer=json.dumps(value, ensure_ascii=False) if changed else row['answer'],
                prose_review={'source_annotation_sha256': source_hash, 'reviewed_annotation_sha256': reviewed_hash,
                              'review_manifest_sha256': review_manifest_hash, 'changed': changed,
                              'teacher_model': 'gpt-6-astra', 'reasoning_effort': 'low',
                              'backend': 'codex_subagent', 'human_rating': False})


def read_review_bundle(root):
    root = Path(root)
    path = root / 'manifest.json'
    proof = json.loads(path.read_text())
    if proof['status'] != 'complete' or proof['verification'].get('human_rating') is not False:
        raise ValueError('Completed neural review bundle is required')
    files = {**proof['inputs'], **proof['outputs']}
    if str(root / 'annotations.jsonl') not in proof['outputs']:
        raise ValueError('Resolved annotations are not a declared review output')
    for name, artifact in files.items():
        if digest(name) != artifact['sha256']:
            raise ValueError('Review bundle input or output changed')
    annotations, originals, decisions = {}, {}, {}
    for name in files:
        if Path(name).suffix == '.jsonl':
            for row in load_jsonl(name):
                if 'teacher_annotation' in row:
                    annotation = row['teacher_annotation']
                    originals[annotation['id']] = annotation_hash(annotation)
                elif 'explanation' in row and 'teacher_model' in row:
                    annotation = row
                else:
                    continue
                annotations[annotation_hash(annotation)] = annotation
        elif Path(name).suffix == '.json':
            doc = json.loads(Path(name).read_text())
            for decision in doc.get('results', []):
                key = decision['id'], decision['reviewed_annotation_sha256']
                decisions.setdefault(key, []).append((files[name]['sha256'], decision['verdict']))
    selected = load_jsonl(root / 'annotations.jsonl')
    if not selected or len({a['id'] for a in selected}) != len(selected):
        raise ValueError('Resolved review annotations must be nonempty and unique')
    result = {}
    for annotation in selected:
        key, final_hash = annotation['id'], annotation_hash(annotation)
        if not any(verdict == 'accept' for _, verdict in decisions.get((key, final_hash), [])):
            raise ValueError('Resolved annotation has no independent acceptance')
        current, seen = final_hash, set()
        while current != originals.get(key):
            if current in seen or current not in annotations:
                raise ValueError('Revision ancestry is cyclic or incomplete')
            seen.add(current)
            child = annotations[current]
            parent, rejection = child.get('corrected_from_annotation_sha256'), child.get('correction_review_sha256')
            if (parent not in annotations or annotations[parent]['id'] != key or
                    (rejection, 'reject') not in decisions.get((key, parent), [])):
                raise ValueError('Revision must bind the exact rejected parent annotation')
            current = parent
        result[key] = annotation, originals[key], final_hash, digest(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--reviews', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.data), Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh revised dataset output')
    source_proof = json.loads((source / 'manifest.json').read_text())
    if source_proof['status'] != 'complete':
        raise ValueError('Source dataset is incomplete')
    for name, artifact in source_proof['outputs'].items():
        if digest(name) != artifact['sha256']:
            raise ValueError('Source dataset output changed')
    selected = {}
    for root in args.reviews:
        revised = read_review_bundle(root)
        if set(selected) & set(revised):
            raise ValueError('Review bundles repeat annotation IDs')
        selected.update(revised)
    rows = load_jsonl(source / 'train.jsonl')
    if not set(selected) <= {r['id'] for r in rows}:
        raise ValueError('Review bundle contains an unknown or heldout label')
    train = [revise_row(row, *selected[row['id']]) if row['id'] in selected else row for row in rows]
    validation, test = [load_jsonl(source / (s + '.jsonl')) for s in ('validation', 'test')]
    split_proof = verify_splits([*train, *validation, *test])
    output.mkdir(parents=True)
    write_jsonl(output / 'train.jsonl', train)
    for split in ('validation', 'test'):
        shutil.copyfile(source / (split + '.jsonl'), output / (split + '.jsonl'))
        assert (source / (split + '.jsonl')).read_bytes() == (output / (split + '.jsonl')).read_bytes()
    verification = {**split_proof, 'reviewed_train_labels': len(selected),
                    'revised_train_prose': sum(r.get('prose_review', {}).get('changed', False)
                                              for r in train if r['id'] in selected),
                    'by_split': {'train': len(train), 'validation': len(validation), 'test': len(test)},
                    'by_side': dict(Counter(r['fen'].split()[1] for r in train)),
                    'structured_labels_and_histories_preserved': True, 'heldout_files_byte_identical': True,
                    'human_rating': False, 'training_benefit_measured': False}
    atomic_json(output / 'manifest.json', manifest('reviewed_prose_revision', vars(args),
                [source / 'manifest.json', *[source / (s + '.jsonl') for s in ('train','validation','test')],
                 *[Path(root) / 'manifest.json' for root in args.reviews]],
                [output / (s + '.jsonl') for s in ('train','validation','test')], verification))
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
