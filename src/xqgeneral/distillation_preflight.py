"""Check actual mined labels, the selected full-decoder parent and new cache."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from .course_tasks import validate_context
from .collect_search import EVALUATION_FILTER_PROFILE, search_evaluation_filter
from .curriculum_data import verify_splits
from .evidence import atomic_json, digest, iter_jsonl, manifest
from .explanation_preflight import token_chunk, token_initializer
from .explanations import continuation_positions, parse_explanation, validate_explanation
from .extend_features import validate_cache
from .feature_store import FeatureStore, feature_paths
from .foundation_preflight import bounded_results
from .prose_moves import validate_prose_moves
from .recorded_coach import ordered_native_map
from .recorded_search_inputs import BoundInputs
from .sft import checked_parent


TOKEN_KIND = 'latent_only_search_distillation_data_preflight'
ROUND_KIND = 'full_decoder_search_distillation_sft_round'


def completed_parent(source, completion_path, bound=None):
    """Accept the selected initial SFT or a completed descendant SFT round."""
    source, completion_path = Path(source), Path(completion_path)
    bound = bound or BoundInputs()
    proof = bound.json(completion_path)
    required = {
        'clean_curriculum_to_initial_sft_pipeline': [
            'four_courses_completed_before_any_gpu_sft_work', 'clean_handoff_verified',
            'actual_four_gpu_full_decoder_preflight_passed', 'initial_explanation_sft_completed'],
        ROUND_KIND: ['initial_explanation_sft_completed_before_round',
            'actual_four_gpu_full_decoder_preflight_passed', 'actual_distillation_sft_completed'],
    }
    result = proof.get('verification', {})
    if (proof.get('status') != 'complete' or proof.get('kind') not in required or
            any(result.get(k) is not True for k in required.get(proof.get('kind'), [])) or
            result.get('complete_shapes_and_finite_fp32_verified') is not True or
            type(result.get('trainable_parameters')) is not int or result['trainable_parameters'] <= 0 or
            result.get('world_size') != 4 or result.get('selected_checkpoint') != str(source) or
            result.get('independent_test_used_for_training_or_selection') is not False):
        raise ValueError('Require the actual completed selected full-decoder SFT parent')
    for section in ['inputs', 'outputs']:
        for path, expected in proof[section].items():
            bound.bind(path, expected)
    selected_binding = proof['outputs'].get(str(source))
    if selected_binding is None:
        raise ValueError('Parent completion must bind its actual selected checkpoint')
    bound.bind(source, selected_binding)
    training = bound.json(source.parent / 'manifest.json')
    saved = bound.json(source.parent / 'config.json')
    metrics = training.get('verification', {})
    if (training.get('status') != 'complete' or training.get('kind') != 'model_training' or
            training.get('config') != saved or training['outputs'].get(str(source)) != selected_binding or
            saved.get('decoder_training') != 'full' or saved.get('trainable_parameter_dtype') != 'float32' or
            saved.get('ddp_world_size') != 4 or metrics.get('world_size') != 4 or
            metrics.get('decoder_training') != 'full' or metrics.get('selected_step', 0) <= 0 or
            result.get('trainable_parameters') != metrics.get('trainable_parameters') or
            result.get('selected_sft_step', result.get('selected_step')) != metrics['selected_step']):
        raise ValueError('Selected parent differs from its completed full-decoder training')
    if checked_parent(source, False) != saved:
        raise ValueError('Selected parent configuration changed')
    bound.unchanged()
    return saved, proof


def collected_dataset(data, source, bound):
    data = Path(data)
    proof = bound.json(data / 'manifest.json')
    result = proof.get('verification', {})
    if (proof.get('status') != 'complete' or proof.get('kind') != 'search_consolidated_dataset' or
            type(result.get('accepted_training_roots')) is not int or result['accepted_training_roots'] < 32 or
            any(result.get(k) is not True for k in ['neural_consolidation_executed',
                'structured_facts_verified', 'explicit_prose_moves_natively_verified',
                'all_branch_positions_isolated', 'full_history_termination_checked',
                'recorded_mining_isolation_rechecked_during_collection',
                'selected_branch_evaluation_filter_applied']) or
            result.get('evaluation_filter_profile') != EVALUATION_FILTER_PROFILE or
            result.get('max_absolute_winrate_error_exclusive') != 0.075 or
            result.get('teacher', {}).get('quantization') != 'none' or
            result.get('teacher', {}).get('inference_dtype') != 'bfloat16'):
        raise ValueError('Require at least 32 actually mined, consolidated and isolated training labels')
    for section in ['inputs', 'outputs']:
        for path, expected in proof[section].items():
            bound.bind(path, expected)
    config = proof['config']
    mining_path = Path(config.get('mining_manifest') or Path(config['queries']).parent / 'manifest.json')
    mining = bound.json(mining_path)
    for section in ['inputs', 'outputs']:
        for path, expected in mining[section].items():
            bound.bind(path, expected)
    if (mining.get('kind') != 'search_distillation_mining' or mining.get('status') != 'complete' or
            mining['config']['checkpoint'] != str(source) or
            proof['inputs'].get(str(mining_path)) != bound.artifacts[str(mining_path)] or
            mining['inputs'].get(str(source)) != bound.bind(source) or
            mining['verification'].get('unused_recorded_training_inputs') is not True):
        raise ValueError('Actual search labels must come from this selected parent and unused recorded roots')
    validation_data = Path(result['validation_source_unchanged'])
    for split in ['validation', 'test']:
        original, copied = validation_data / f'{split}.jsonl', data / f'{split}.jsonl'
        if (proof['inputs'].get(str(original)) is None or
                proof['outputs'].get(str(copied)) != proof['inputs'][str(original)]):
            raise ValueError('Distillation must preserve the original protected validation and test bytes')
    return proof


def native_label(row):
    validate_context(row)
    value = parse_explanation(row['answer'])
    facts = validate_explanation(row['fen'], value, require_branches=True)
    prose = validate_prose_moves(row['fen'], value, value['explanation'])
    if not facts['valid'] or not prose['valid']:
        raise ValueError('A consolidated label has invalid native facts or explicit prose moves')
    if row['split'] == 'train':
        rechecked = search_evaluation_filter({'target_fields': value, 'record': row})
        if row.get('search_evaluation_filter') != rechecked:
            raise ValueError('A training label changed its original student evaluation or exclusive filter')
    continuation_positions(row, value)
    return row['feature_key']


def extended_cache(features, saved, bound):
    features, previous = Path(features), Path(saved['feature_path'])
    if features.resolve() == previous.resolve():
        raise ValueError('Distillation requires its completed extended feature store')
    for path in feature_paths(features):
        bound.bind(path)
    cache = FeatureStore(features, verified_hashes={p: b['sha256'] for p, b in bound.artifacts.items()})
    base_cache = cache.proof['config'].get('base_cache')
    if not isinstance(base_cache, str) or not Path(base_cache).exists() or not Path(base_cache).samefile(previous):
        raise ValueError('Extended cache must inherit the actual selected parent feature store')
    for path in feature_paths(previous):
        bound.bind(path)
    old = FeatureStore(previous, verified_hashes={p: b['sha256'] for p, b in bound.artifacts.items()})
    if (cache.keys[:len(old.keys)] != old.keys or cache.shards[:len(old.shards)] != old.shards or
            cache.proof['verification']['expert_sha256'] != old.proof['verification']['expert_sha256']):
        raise ValueError('Extended cache must preserve all original ordered history keys and tensor references')
    validate_cache(cache, saved['expert_feature_depths'])
    cache.check_unchanged(); old.check_unchanged()
    return cache


def preflight(source, parent_completion, data, features, foundation_preflight, output, *, max_tokens=2048, workers=8):
    if type(workers) is not int or workers < 1 or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('Positive native/token worker and length budgets are required')
    root, data, source = Path(output), Path(data), Path(source)
    if root.exists():
        raise FileExistsError('Preserve completed and interrupted preflights; use a fresh output')
    bound = BoundInputs()
    saved, parent = completed_parent(source, parent_completion, bound)
    labels = collected_dataset(data, source, bound)
    cache = extended_cache(features, saved, bound)
    rows, ids, keys, counts = [], set(), set(), Counter()
    for split in ['train', 'validation', 'test']:
        path = data / f'{split}.jsonl'
        bound.bind(path, labels['outputs'].get(str(path)))
        for row in iter_jsonl(path):
            if (row['split'] != split or row['stage'] != 'explanation' or
                    row['id'] in ids or row['feature_key'] in keys):
                raise ValueError('Distillation labels have changed stages, splits or duplicate identities')
            ids.add(row['id']); keys.add(row['feature_key']); counts[split] += 1; rows.append(row)
    if counts['train'] != labels['verification']['accepted_training_roots'] or not counts['validation']:
        raise ValueError('Actual label coverage differs from completed collection')
    cached_keys = set(cache.keys)
    if not keys.issubset(cached_keys):
        raise ValueError('A distillation label is missing its expert history features')
    native_split_rows = []
    for row in rows:
        value = parse_explanation(row['answer'])
        native_split_rows.append(dict(row, future_moves=list(value['pv']),
            future_branches=[list(branch['pv']) for branch in value.get('branches', [])]))
    split_proof = verify_splits(native_split_rows, workers=workers)
    if list(ordered_native_map(native_label, rows, workers)) != [r['feature_key'] for r in rows]:
        raise ValueError('Native full-history label coverage differs')
    foundation = bound.json(foundation_preflight)
    if (foundation.get('status') != 'complete' or
            foundation.get('kind') != 'recorded_engine_clean_latent_foundation_preflight'):
        raise ValueError('Require the completed pinned-base foundation preflight')
    model_path = Path(saved['model_path'])
    assets = {p: b for p, b in foundation['inputs'].items() if Path(p).resolve().parent == model_path.resolve()}
    if not any(p.endswith('.safetensors') for p in assets):
        raise ValueError('The full official pretrained base must remain hash-bound')
    for p, expected in assets.items():
        bound.bind(p, expected)
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json',
                 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt']:
        p = model_path / name
        if p.exists():
            bound.bind(p)
    eligible = [row for row in rows if row['split'] != 'test']
    root.mkdir(parents=True)
    observed = root / 'token-lengths.jsonl'
    maxima, token_counts = Counter(), Counter()
    with observed.open('x') as handle:
        def emit(results):
            for item in results:
                maxima[item['split']] = max(maxima[item['split']], item['tokens'])
                token_counts[item['split']] += 1
                handle.write(json.dumps(item) + '\n')
        if workers == 1:
            token_initializer(str(model_path), max_tokens)
            emit(token_chunk(eligible))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=token_initializer, initargs=(str(model_path), max_tokens)) as pool:
                for result in bounded_results(pool, token_chunk, eligible, workers, 128):
                    emit(result)
    if dict(token_counts) != {s: counts[s] for s in ['train', 'validation']}:
        raise ValueError('Every actual train/validation label must fit without truncation')
    cache.check_unchanged(); bound.unchanged()
    result = {'status': 'complete', 'by_split': dict(counts), 'token_limit': max_tokens,
        'maximum_tokens_by_split': dict(maxima), 'train_validation_rows_tokenized': len(eligible),
        'all_train_validation_supervision_fits_without_truncation': True,
        'all_full_native_histories_facts_branches_and_explicit_prose_moves_checked': True,
        'original_selected_branch_evaluation_filter_rechecked': True,
        'protected_validation_and_test_bytes_unchanged': True, 'independent_test_rows_tokenized': 0,
        'independent_test_used_for_training_or_selection': False, 'split_verification': split_proof,
        'all_label_feature_contexts_present': True, 'parent_full_decoder_completion_verified': True,
        'all_original_feature_keys_and_tensor_references_preserved': True,
        'model_weights_loaded_or_training_started': False, 'strong_play_or_coaching_proven': False}
    atomic_json(root / 'verification.json', result)
    proof = manifest(TOKEN_KIND, {'data': str(data), 'source': str(source),
        'parent_completion': str(parent_completion), 'foundation_preflight': str(foundation_preflight),
        'evaluation_filter_profile': EVALUATION_FILTER_PROFILE,
        'max_tokens': max_tokens, 'workers': workers}, outputs=[observed, root / 'verification.json'], verification=result)
    proof['inputs'] = bound.artifacts
    proof['cache_header_only_input'] = {'path': str(features), **bound.artifacts[str(features)],
                                       'fresh_full_bytes_hash_verified_by_this_preflight': True}
    atomic_json(root / 'manifest.json', proof)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['init', 'parent-completion', 'data', 'features', 'foundation-preflight', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(preflight(args.init, args.parent_completion, args.data, args.features,
        args.foundation_preflight, args.output, max_tokens=args.max_tokens, workers=args.workers)), flush=True)


if __name__ == '__main__':
    main()
