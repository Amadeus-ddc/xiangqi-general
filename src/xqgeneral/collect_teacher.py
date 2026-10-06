"""Validate explicitly authorized teacher annotations without a prose fallback."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .curriculum_data import verify_splits
from .evidence import atomic_json, load_jsonl, manifest, write_jsonl
from .explanations import EXPLANATION_QUESTION, oracle_explanation, validate_explanation
from .symmetry import mirrored_qa


def assemble_label(query, annotation):
    if annotation.get('id') != query['id']:
        raise ValueError('Teacher annotation belongs to a different query')
    contract = {'teacher_model': 'gpt-6-astra', 'reasoning_effort': 'low', 'backend': 'codex_subagent'}
    if any(annotation.get(k) != v for k, v in contract.items()):
        raise ValueError('Initial teacher identity differs from the authorized contract')
    prose = annotation.get('explanation')
    if not isinstance(prose, str) or len(prose) < 80 or not re.search(r'[\u4e00-\u9fff]', prose):
        raise ValueError('Teacher must supply a substantive Chinese explanation')
    source = query['record']
    value = oracle_explanation(source['fen'], query['oracle'], prose)
    return dict(source, id=query['id'], stage='explanation', task_type='explanation', query={},
                question=EXPLANATION_QUESTION, answer=json.dumps(value, ensure_ascii=False), future_moves=[],
                teacher=contract, teacher_query_id=query['id'],
                provenance=source['provenance'] + ';pikafish_grounded;gpt-6-astra_low_codex_subagent')


def mirror_label(row):
    derived = mirrored_qa(row)
    proof = validate_explanation(derived['fen'], json.loads(derived['answer']))
    if not proof['valid']:
        raise ValueError(f'Color-symmetric explanation violates facts: {proof["errors"]}')
    derived['teacher'] = dict(row['teacher'], derivation='color_rank_symmetry_of_original_annotation',
                              independent_teacher_call=False, engine_scores_recomputed=False)
    return derived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/astra-seed-v1')
    parser.add_argument('--output', default='data/astra-explanations-v1')
    parser.add_argument('--color-mirror', action='store_true')
    args = parser.parse_args()
    root, dest = Path(args.input), Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh annotation dataset output')
    query_paths = [root / f'{s}.queries.jsonl' for s in ['train', 'validation', 'test']]
    queries = [q for p in query_paths for q in load_jsonl(p)]
    annotation_paths = sorted((root / 'shards').glob('annotations-*.jsonl'))
    annotations = [a for p in annotation_paths for a in load_jsonl(p)]
    by_id = {a['id']: a for a in annotations}
    if (len(by_id) != len(annotations) or len({q['id'] for q in queries}) != len(queries) or
            set(by_id) != {q['id'] for q in queries}):
        raise ValueError('Teacher coverage must be complete, unique and contain no foreign IDs')
    review_paths = sorted(root.glob('review-*.json'))
    review_items = [r for p in review_paths for r in json.loads(p.read_text())['results']]
    if any(r['id'] not in by_id for r in review_items):
        raise ValueError('Review references an unknown annotation')
    rejected = {r['id'] for r in review_items if r['verdict'] == 'reject'}
    rows = [assemble_label(q, by_id[q['id']]) for q in queries if q['id'] not in rejected]
    if args.color_mirror:
        rows += [mirror_label(row) for row in list(rows)]
    proof = verify_splits(rows)
    outputs = []
    for split in ['train', 'validation', 'test']:
        path = dest / f'{split}.jsonl'
        write_jsonl(path, [r for r in rows if r['split'] == split])
        outputs.append(path)
    summary = {**proof, 'original_teacher_annotations': len(annotations),
               'accepted_original_annotations': len(queries) - len(rejected),
               'rejected_by_review': sorted(rejected), 'records': len(rows),
               'by_split': dict(Counter(r['split'] for r in rows)),
               'by_side': dict(Counter(r['fen'].split()[1] for r in rows)),
               'teacher_model': 'gpt-6-astra', 'reasoning_effort': 'low', 'backend': 'codex_subagent',
               'legality_and_structured_facts_checked': True,
               'independent_neural_review_samples': len(review_items),
               'review_is_human_rating': False, 'strategic_prose_accuracy_fully_proven': False,
               'derived_color_mirror_records': sum('augmentation_parent' in r for r in rows),
               'derived_scores_independently_recomputed': False}
    atomic_json(dest / 'manifest.json', manifest('initial_teacher_annotations', vars(args),
                [*query_paths, *annotation_paths, *review_paths], outputs, summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
