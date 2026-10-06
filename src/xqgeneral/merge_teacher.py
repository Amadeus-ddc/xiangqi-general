"""Combine completed initial-teacher batches without changing preserved labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .collect_teacher import assemble_label
from .evidence import atomic_json, load_jsonl, manifest, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh combined teacher batch')
    queries, annotations, reviews, inputs = [], [], [], []
    for index, name in enumerate(args.inputs):
        root = Path(name)
        source_manifest = root / 'queries.manifest.json'
        if not source_manifest.exists():
            source_manifest = root / 'manifest.json'
        if json.loads(source_manifest.read_text())['status'] != 'complete':
            raise ValueError('Teacher query preparation is incomplete')
        paths = [root / f'{s}.queries.jsonl' for s in ('train', 'validation', 'test')]
        batch_queries = [q for p in paths for q in load_jsonl(p)]
        if index and any(q['record']['split'] != 'train' for q in batch_queries):
            raise ValueError('Additional batches must preserve the original held-out sets')
        annotation_paths = sorted((root / 'shards').glob('annotations-*.jsonl'))
        batch_annotations = [a for p in annotation_paths for a in load_jsonl(p)]
        by_id = {a['id']: a for a in batch_annotations}
        query_ids = {q['id'] for q in batch_queries}
        if (len(query_ids) != len(batch_queries) or len(by_id) != len(batch_annotations)
                or set(by_id) != query_ids):
            raise ValueError(f'Incomplete or duplicate teacher coverage in {name}')
        for query in batch_queries:
            assemble_label(query, by_id[query['id']])
        review_paths = sorted(root.glob('review-*.json'))
        reviews += [json.loads(p.read_text()) for p in review_paths]
        queries += batch_queries
        annotations += batch_annotations
        inputs += [*paths, *annotation_paths, *review_paths,
                   *sorted(root.glob('corrections-*.jsonl')), source_manifest]
    if (len({q['id'] for q in queries}) != len(queries)
            or len({q['feature_key'] for q in queries}) != len(queries)):
        raise ValueError('Teacher batches overlap by ID or board-history context')
    outputs = []
    for split in ('train', 'validation', 'test'):
        path = dest / f'{split}.queries.jsonl'
        write_jsonl(path, [q for q in queries if q['record']['split'] == split])
        outputs.append(path)
    path = dest / 'shards/annotations-0.jsonl'
    write_jsonl(path, annotations)
    outputs.append(path)
    for index, review in enumerate(reviews):
        path = dest / f'review-{index}.json'
        atomic_json(path, review)
        outputs.append(path)
    proof = {'queries': len(queries), 'annotations': len(annotations),
             'by_split': dict(Counter(q['record']['split'] for q in queries)),
             'original_queries_and_annotations_preserved': True,
             'held_out_sets_unchanged': True, 'teacher_identity_checked': True}
    atomic_json(dest / 'manifest.json', manifest('combined_initial_teacher_batches',
                vars(args), inputs, outputs, proof))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
