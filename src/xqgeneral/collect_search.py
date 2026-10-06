"""Accept real Qwen consolidation prose with verified improved-root fields."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .curriculum_data import verify_splits
from .evidence import atomic_json, digest, load_jsonl, manifest, write_jsonl
from .explanations import EXPLANATION_QUESTION, continuation_positions, parse_explanation, validate_explanation
from .search_distillation import reserved_positions


def consolidated_label(query, response, teacher):
    identity = response.get('teacher', {})
    if (response.get('id') != query['id'] or response.get('hit_generation_limit', True) or
            any(identity.get(k) != teacher[k] for k in ['repository', 'revision', 'quantization', 'inference_dtype']) or
            identity.get('parameter_elements_by_dtype', {}).get('torch.bfloat16', 0) <= 0):
        raise ValueError('Response does not prove the authorized full BF16 teacher inference')
    prose = parse_explanation(response['text'])
    if set(prose) != {'explanation'} or not isinstance(prose['explanation'], str) or len(prose['explanation']) < 80:
        raise ValueError('Consolidator must supply Chinese prose, without replacing verified root fields')
    value = dict(query['target_fields'], explanation=prose['explanation'])
    row = query['record']
    proof = validate_explanation(row['fen'], value, require_branches=True)
    if not proof['valid']:
        raise ValueError(f'Consolidation violates grounded facts: {proof["errors"]}')
    if row['split'] != 'train':
        raise ValueError('Search training must never use a heldout root')
    continuation_positions(row, value)
    return dict(row, id=query['id'], stage='explanation', task_type='explanation', query={},
                future_moves=list(value['pv']), future_branches=[list(b['pv']) for b in value['branches']],
                question=EXPLANATION_QUESTION, answer=json.dumps(value, ensure_ascii=False),
                teacher=identity, search_source_root=query['source_root_id'], search_depth=query['depth'],
                provenance=row['provenance'] + ';student_recursive_search;verified_pv_improvement;Qwen3.8-27B_BF16_consolidation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', required=True)
    parser.add_argument('--responses', required=True)
    parser.add_argument('--config', default='configs/teachers.json')
    parser.add_argument('--heldout-data', default='data/research-balanced-v1')
    parser.add_argument('--validation-data', default='data/astra-explanations-v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh consolidated dataset output')
    teacher = json.loads(Path(args.config).read_text())['search_consolidator']
    queries = load_jsonl(args.queries)
    responses = load_jsonl(args.responses)
    by_id = {r['id']: r for r in responses}
    if (len(by_id) != len(responses) or len({q['id'] for q in queries}) != len(queries) or
            set(by_id) != {q['id'] for q in queries}):
        raise ValueError('Consolidator response coverage differs from mined queries')
    forbidden = reserved_positions(args.heldout_data) | reserved_positions(args.validation_data)
    train, rejected, keys = [], [], set()
    for query in queries:
        try:
            row = consolidated_label(query, by_id[query['id']], teacher)
            if continuation_positions(row, parse_explanation(row['answer'])) & forbidden:
                raise ValueError('Consolidated root or branch overlaps a heldout position')
            if row['feature_key'] in keys:
                raise ValueError('Duplicate mined history context')
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({'id': query['id'], 'reason': str(error)})
            continue
        keys.add(row['feature_key']); train.append(row)
    dest.mkdir(parents=True)
    atomic_json(dest / 'rejected.json', rejected)
    if not train:
        raise RuntimeError('No real search-consolidated explanations passed; an empty dataset is not a completed distillation')
    rows = list(train)
    for split in ['validation', 'test']:
        rows += load_jsonl(Path(args.validation_data) / f'{split}.jsonl')
    proof = verify_splits(rows)
    outputs = [dest / 'rejected.json']
    for split in ['train', 'validation', 'test']:
        path = dest / f'{split}.jsonl'
        write_jsonl(path, [r for r in rows if r['split'] == split]); outputs.append(path)
    summary = {**proof, 'mined_queries': len(queries), 'accepted_training_roots': len(train),
               'rejected_consolidations': len(rejected), 'teacher': teacher,
               'search_depths': dict(Counter(r['search_depth'] for r in train)),
               'neural_consolidation_executed': True, 'structured_facts_verified': True,
               'all_branch_positions_isolated': True, 'full_history_termination_checked': True,
               'validation_source_unchanged': args.validation_data, 'strategic_prose_human_rating': False}
    atomic_json(dest / 'manifest.json', manifest('search_consolidated_dataset', vars(args),
                [args.queries, args.responses, args.config,
                 *[Path(args.validation_data) / f'{s}.jsonl' for s in ['validation', 'test']],
                 *[Path(args.heldout_data) / f'{s}.jsonl' for s in ['validation', 'test']]], outputs, summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
