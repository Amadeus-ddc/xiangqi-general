"""Accept real Qwen consolidation prose with verified improved-root fields."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from .curriculum_data import verify_splits
from .evidence import atomic_json, history_key, load_jsonl, manifest, write_jsonl
from .explanations import EXPLANATION_QUESTION, continuation_positions, parse_explanation, validate_explanation
from .recorded_search_inputs import ARTIFACTS, BoundInputs, prepared_roots
from .rules import replay
from .search_distillation import reserved_positions
from .prose_moves import validate_prose_moves


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
    prose_proof = validate_prose_moves(row['fen'], value, prose['explanation'])
    if not prose_proof['valid']:
        raise ValueError(f'Consolidation violates grounded prose moves: {prose_proof["errors"]}')
    if row['split'] != 'train':
        raise ValueError('Search training must never use a heldout root')
    continuation_positions(row, value)
    return dict(row, id=query['id'], stage='explanation', task_type='explanation', query={},
                future_moves=list(value['pv']), future_branches=[list(b['pv']) for b in value['branches']],
                question=EXPLANATION_QUESTION, answer=json.dumps(value, ensure_ascii=False),
                teacher=identity, search_source_root=query['source_root_id'], search_depth=query['depth'],
                provenance=row['provenance'] + ';student_recursive_search;verified_pv_improvement;Qwen3.8-27B_BF16_consolidation')


def mining_reservations(queries_path, queries, data, validation_data, bound,
                        mining_manifest=None, recorded_inputs=None):
    """Carry a completed recorded miner's source ownership and isolation into labels."""
    tagged_recorded = any(isinstance(query.get('record'), dict) and
        query['record'].get('recorded_continuation_is_best_move_label') is False for query in queries)
    source = Path(mining_manifest) if mining_manifest else Path(queries_path).parent / 'manifest.json'
    if not source.exists():
        if mining_manifest or recorded_inputs or tagged_recorded:
            raise ValueError('Recorded consolidation requires a completed mining manifest')
        return set(), {}
    saved = bound.json(source)
    config = saved.get('config', {})
    declared_pool = config.get('recorded_inputs')
    if not declared_pool and (tagged_recorded or
            saved.get('verification', {}).get('unused_recorded_training_inputs') is True):
        raise ValueError('Recorded mining lost its declared input pool')
    if not declared_pool and not mining_manifest and not recorded_inputs:
        return set(), {}
    if (saved.get('kind') != 'search_distillation_mining' or saved.get('status') != 'complete' or
            not isinstance(config.get('data'), str) or not Path(config['data']).samefile(data)):
        raise ValueError('Mining completion or source data differs')
    outputs = saved.get('outputs', {})
    query_outputs = [value for name, value in outputs.items() if Path(name).name == 'queries.jsonl']
    if len(query_outputs) != 1:
        raise ValueError('Mining must bind exactly one original query output')
    bound.bind(queries_path, query_outputs[0])
    if saved.get('verification', {}).get('accepted_for_consolidation') != len(queries):
        raise ValueError('Mining query coverage differs')
    info = {'source_mining_manifest': str(source), 'source_mining_queries_byte_verified': True}
    if not declared_pool:
        if recorded_inputs:
            raise ValueError('Mining did not use the requested recorded pool')
        return set(), info
    pool = Path(declared_pool)
    if recorded_inputs and not pool.samefile(recorded_inputs):
        raise ValueError('Recorded consolidation pool differs from mining')
    proof = saved.get('verification', {})
    required = ['unused_recorded_training_inputs', 'training_roots_only',
                'heldout_positions_excluded', 'all_child_branch_positions_isolated',
                'raw_child_reachable_prefixes_isolated', 'all_inferred_target_lines_history_validated',
                'strict_pv_improvement_required', 'all_successful_oracle_queries_preserved']
    if any(proof.get(name) is not True for name in required):
        raise ValueError('Recorded mining lacks its complete isolation and search contract')
    inputs = saved.get('inputs', {})

    def declared(path):
        values = [identity for name, identity in inputs.items() if Path(name).resolve() == Path(path).resolve()]
        if len(values) != 1:
            raise ValueError('Mining does not bind the complete recorded pool and base data')
        return values[0]

    for name in ARTIFACTS:
        bound.bind(pool / name, declared(pool / name))
    identities = {str(Path(data) / f'{split}.jsonl'):
        bound.bind(Path(data) / f'{split}.jsonl', declared(Path(data) / f'{split}.jsonl'))
        for split in ('train', 'validation', 'test')}
    selected, additional = prepared_roots(pool, data, config['limit'], config['seed'], identities)
    if proof.get('additional_reserved_canonical_and_explanation_positions') != len(additional):
        raise ValueError('Mining reserved-position coverage differs from the recorded pool')
    pool_proof = bound.json(pool / 'manifest.json')
    for name, identity in pool_proof['inputs'].items():
        bound.bind(name, identity)
    labels = Path(pool_proof['config']['heldout_data'])
    for split in ('validation', 'test'):
        identity = pool_proof['inputs'][str(labels / f'{split}.jsonl')]
        bound.bind(Path(validation_data) / f'{split}.jsonl', identity)
    by_id = {row['id']: row for row in selected}
    for query in queries:
        original = by_id.get(query.get('source_root_id'))
        row, depth = query.get('record', {}), query.get('depth')
        if (original is None or type(depth) is not int or not 0 <= depth <= config['max_depth'] or
                row.get('split') != 'train' or row.get('game_id') != original['game_id'] or
                row.get('initial_fen') != original['initial_fen'] or
                row.get('moves', [])[:len(original['moves'])] != original['moves'] or
                len(row.get('moves', [])) != len(original['moves']) + depth):
            raise ValueError('Consolidation root is not an original selected training root or its descendant')
        for field in ['recorded_source_kind', 'recorded_source', 'recorded_source_headers',
                      'recorded_source_provenance', 'recorded_source_context']:
            if json.dumps(row.get(field), sort_keys=True) != json.dumps(original.get(field), sort_keys=True):
                raise ValueError('Consolidation changed original recorded source or supplied-history context')
        history = replay(row['initial_fen'], row['moves'])
        suffix = ''.join('-' + move for move in row['moves'][len(original['moves']):])
        if (row.get('history') != history or row.get('fen') != history[-1] or
                row.get('feature_key') != history_key(history) or
                row.get('id') != original['id'] + suffix or query.get('id') != row['id'] + '-search'):
            raise ValueError('Consolidation root history, cache key or identity differs from mining')
    bound.unchanged()
    info.update(recorded_inputs=str(pool), additional_reserved_canonical_and_explanation_positions=len(additional),
                original_selected_root_ownership_and_descendants_verified=True,
                validation_and_test_labels_byte_preserved_from_recorded_pool=True,
                recorded_mining_isolation_rechecked_during_collection=True,
                student_search_or_oracle_comparisons_recomputed=False)
    return additional, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', required=True)
    parser.add_argument('--responses', required=True)
    parser.add_argument('--config', default='configs/teachers.json')
    parser.add_argument('--heldout-data', default='data/research-balanced-v1')
    parser.add_argument('--validation-data', default='data/astra-explanations-v1')
    parser.add_argument('--mining-manifest', help='Completed miner manifest; defaults to the queries sibling')
    parser.add_argument('--recorded-inputs', help='Require this exact unused-recorded pool from the miner')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a fresh consolidated dataset output')
    bound = BoundInputs()
    inputs = [args.queries, args.responses, args.config,
              *[Path(args.validation_data) / f'{s}.jsonl' for s in ['validation', 'test']],
              *[Path(args.heldout_data) / f'{s}.jsonl' for s in ['validation', 'test']]]
    for path in inputs:
        bound.bind(path)
    teacher = bound.json(args.config)['search_consolidator']
    queries = load_jsonl(args.queries)
    responses = load_jsonl(args.responses)
    by_id = {r['id']: r for r in responses}
    if (len(by_id) != len(responses) or len({q['id'] for q in queries}) != len(queries) or
            set(by_id) != {q['id'] for q in queries}):
        raise ValueError('Consolidator response coverage differs from mined queries')
    additional, source_proof = mining_reservations(args.queries, queries, args.heldout_data,
        args.validation_data, bound, args.mining_manifest, args.recorded_inputs)
    forbidden = reserved_positions(args.heldout_data) | reserved_positions(args.validation_data) | additional
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
        if split == 'train':
            write_jsonl(path, train)
        else:
            partial = path.with_suffix(path.suffix + '.partial')
            shutil.copyfile(Path(args.validation_data) / f'{split}.jsonl', partial)
            partial.replace(path)
        outputs.append(path)
    summary = {**proof, 'mined_queries': len(queries), 'accepted_training_roots': len(train),
               'rejected_consolidations': len(rejected), 'teacher': teacher,
               'search_depths': dict(Counter(r['search_depth'] for r in train)),
               'neural_consolidation_executed': True, 'structured_facts_verified': True,
               'explicit_prose_moves_natively_verified': True,
               'all_strategic_prose_semantics_verified': False,
               'all_branch_positions_isolated': True, 'full_history_termination_checked': True,
               'validation_source_unchanged': args.validation_data, 'strategic_prose_human_rating': False,
               **source_proof}
    saved = manifest('search_consolidated_dataset', vars(args), [], outputs, summary)
    saved['inputs'] = bound.artifacts
    bound.unchanged()
    atomic_json(dest / 'manifest.json', saved)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
