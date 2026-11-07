"""Check actual mined labels, the selected full-decoder parent and new cache."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from .course_tasks import PAPER_PROFILE, grade_answer, row_profile, validate_context
from .collect_search import EVALUATION_FILTER_PROFILE
from .merge_search import collected_dataset, native_label
from .curriculum_data import STAGES, verify_splits
from .evidence import atomic_json, digest, iter_jsonl, manifest
from .explanation_preflight import token_chunk, token_initializer
from .explanations import parse_explanation
from .extend_features import validate_cache
from .feature_store import FeatureStore, feature_paths
from .foundation_preflight import bounded_results, native_answer, validate_task
from .recorded_coach import ordered_native_map
from .recorded_search_inputs import BoundInputs
from .sft import checked_parent
from .auxiliary_search_replay import (PROFILE as AUXILIARY_REPLAY_PROFILE,
    LEGACY_PROFILE as LEGACY_REPLAY_PROFILE)


TOKEN_KIND = 'latent_only_search_distillation_data_preflight'
ROUND_KIND = 'full_decoder_search_distillation_sft_round'
AUXILIARY_ROUND_KIND = 'full_decoder_auxiliary_policy_sft_round'


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
        AUXILIARY_ROUND_KIND: ['initial_explanation_sft_completed_before_round',
            'actual_four_gpu_full_decoder_preflight_passed','actual_auxiliary_policy_sft_completed'],
    }
    result = proof.get('verification', {})
    if proof.get('kind')==AUXILIARY_ROUND_KIND:
        from .full_decoder_topology import completed_auxiliary_topology
        topology=completed_auxiliary_topology(proof)
        required[AUXILIARY_ROUND_KIND]=['initial_explanation_sft_completed_before_round','actual_auxiliary_policy_sft_completed']
        world=topology['world_size']
    elif proof.get('kind') == ROUND_KIND:
        from .full_decoder_topology import completed_search_topology
        topology = completed_search_topology(proof)
        required[ROUND_KIND] = ['initial_explanation_sft_completed_before_round', 'actual_distillation_sft_completed']
        world = topology['world_size']
    else:world=4
    if (proof.get('status') != 'complete' or proof.get('kind') not in required or
            any(result.get(k) is not True for k in required.get(proof.get('kind'), [])) or
            result.get('complete_shapes_and_finite_fp32_verified') is not True or
            type(result.get('trainable_parameters')) is not int or result['trainable_parameters'] <= 0 or
            result.get('world_size') != world or result.get('selected_checkpoint') != str(source) or
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
    if proof.get('kind')==AUXILIARY_ROUND_KIND:
        from .full_decoder_topology import auxiliary_topology
        if (auxiliary_topology(saved)!=topology or
                metrics.get('global_batch_size')!=topology['global_batch_size'] or
                result.get('global_batch_size')!=topology['global_batch_size']):
            raise ValueError('Auxiliary parent geometry differs from its actual completed training')
    elif proof.get('kind') == ROUND_KIND and world == 3:
        from .full_decoder_topology import search_topology
        if (search_topology(saved) != topology or
                metrics.get('global_batch_size') != topology['global_batch_size'] or
                result.get('global_batch_size') != topology['global_batch_size']):
            raise ValueError('Search parent geometry differs from its actual completed training')
    if (training.get('status') != 'complete' or training.get('kind') != 'model_training' or
            training.get('config') != saved or training['outputs'].get(str(source)) != selected_binding or
            saved.get('decoder_training') != 'full' or saved.get('trainable_parameter_dtype') != 'float32' or
            saved.get('ddp_world_size') != world or metrics.get('world_size') != world or
            metrics.get('decoder_training') != 'full' or metrics.get('selected_step', 0) <= 0 or
            result.get('trainable_parameters') != metrics.get('trainable_parameters') or
            result.get('selected_sft_step', result.get('selected_step')) != metrics['selected_step']):
        raise ValueError('Selected parent differs from its completed full-decoder training')
    if checked_parent(source, False) != saved:
        raise ValueError('Selected parent configuration changed')
    bound.unchanged()
    return saved, proof


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


def course_replay_sources(paths, bound, *, profile=LEGACY_REPLAY_PROFILE):
    """Reopen certified original-course subsets; keep tests out of replay."""
    if not isinstance(paths, (list, tuple)):
        raise ValueError('Course replay data paths must be an explicit sequence')
    if profile == AUXILIARY_REPLAY_PROFILE:
        if len(paths) != 1 or not isinstance(paths[0], (str, Path)) or not str(paths[0]):
            raise ValueError('Six-stage replay requires exactly one declared original-corpus subset')
        from .auxiliary_search_replay import checked_subset
        rows, result = checked_subset(paths[0], bound)
        return rows, dict(data_paths=[str(Path(paths[0]).resolve())], by_split_and_stage=result['by_split_and_stage'])
    if profile != LEGACY_REPLAY_PROFILE:
        raise ValueError('Unknown explicit replay source profile')
    directories, rows, seen, counts = [], [], set(), {split: Counter() for split in ['train', 'validation']}
    for path in paths:
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError('Course replay paths must name data directories')
        directory = Path(path)
        resolved = directory.resolve()
        if resolved in seen:
            raise ValueError('Course replay directories must be distinct')
        seen.add(resolved); directories.append(str(directory))
        proof = bound.json(directory / 'manifest.json')
        result = proof.get('verification', {})
        if (proof.get('status') != 'complete' or proof.get('kind') != 'latent_only_course_replay_data_preflight' or
                any(result.get(key) is not True for key in [
                    'all_selected_train_and_validation_answers_natively_regraded',
                    'all_selected_full_histories_and_futures_natively_checked',
                    'training_rows_drawn_only_from_original_train_split',
                    'original_training_rows_preserved_byte_for_byte']) or
                result.get('independent_test_used_for_training_or_selection') is not False or
                result.get('by_split', {}).get('test') != 0):
            raise ValueError('Require a certified original-training-only course replay subset')
        for section in ['inputs', 'outputs']:
            for name, binding in proof[section].items():
                bound.bind(name, binding)
        for split in ['train', 'validation']:
            filename = directory / (split + '.jsonl')
            if str(filename) not in proof['outputs']:
                raise ValueError('Course replay completion must bind its raw rows')
            selected = list(iter_jsonl(filename))
            if len(selected) != result['by_split'].get(split):
                raise ValueError('Actual course replay coverage differs from its completion')
            for row in selected:
                if row.get('split') != split or row.get('stage') not in STAGES or row_profile(row) != PAPER_PROFILE:
                    raise ValueError('Course replay must retain the declared paper task profile and split')
                counts[split][row['stage']] += 1
            rows.extend(selected)
    return rows, {'data_paths': directories, 'by_split_and_stage': {
        split: {stage: counts[split][stage] for stage in STAGES} for split in counts}}


def native_replay_label(row):
    if row.get('stage') in ('engine_numeric', 'move_quality'):
        from .auxiliary_search_replay import native_label
        return native_label(row)
    validate_task(row, PAPER_PROFILE)
    validate_context(row, check_future=True)
    if not grade_answer(native_answer(row), row['answer'], row['task_type'])['correct']:
        raise ValueError('Course replay answer differs from native QA rules')
    return row['feature_key']


def preflight(source, parent_completion, data, features, foundation_preflight, output, *, max_tokens=2048,
              workers=8, replay_data_paths=(), replay_profile=LEGACY_REPLAY_PROFILE):
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
    replay, replay_result = course_replay_sources(replay_data_paths, bound, profile=replay_profile)
    for row in replay:
        if row['id'] in ids:
            raise ValueError('Course replay and explanation labels must have distinct IDs')
        ids.add(row['id']); keys.add(row['feature_key'])
    cached_keys = set(cache.keys)
    if not keys.issubset(cached_keys):
        raise ValueError('A distillation label is missing its expert history features')
    native_split_rows = []
    for row in [*rows, *replay]:
        if row['stage'] in ('explanation', 'engine_numeric'):
            value = parse_explanation(row['answer'])
            native_split_rows.append(dict(row, future_moves=list(value['pv']),
                future_branches=[list(branch['pv']) for branch in value.get('branches', [])]))
        else:
            native_split_rows.append(row)
    split_proof = verify_splits(native_split_rows, workers=workers)
    if list(ordered_native_map(native_label, rows, workers)) != [r['feature_key'] for r in rows]:
        raise ValueError('Native full-history label coverage differs')
    if replay and list(ordered_native_map(native_replay_label, replay, workers)) != [r['feature_key'] for r in replay]:
        raise ValueError('Native course replay coverage differs')
    foundation = bound.json(foundation_preflight)
    if (foundation.get('status') != 'complete' or
            foundation.get('kind') != 'recorded_engine_clean_latent_foundation_preflight'):
        raise ValueError('Require the completed pinned-base foundation preflight')
    model_path = Path(saved['model_path'])
    from .sft_preflight import pinned_decoder_assets
    assets, base_reference = pinned_decoder_assets(saved, saved, foundation, bound)
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json',
                 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt']:
        p = model_path / name
        if p.exists():
            bound.bind(p)
    eligible = [row for row in [*rows, *replay] if row['split'] != 'test']
    eligible_by_id = {row['id']: row for row in eligible}
    root.mkdir(parents=True)
    observed = root / 'token-lengths.jsonl'
    maxima, token_counts, token_ids = Counter(), Counter(), set()
    with observed.open('x') as handle:
        def emit(results):
            for item in results:
                identity = item['id']
                if (identity not in eligible_by_id or identity in token_ids or
                        item['split'] != eligible_by_id[identity]['split']):
                    raise ValueError('Token checks must cover every distinct actual supervision row')
                token_ids.add(identity)
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
    expected_tokens = Counter(row['split'] for row in eligible)
    if token_counts != expected_tokens or token_ids != set(eligible_by_id):
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
    if replay:
        result['course_replay'] = dict(replay_result,
            all_raw_train_validation_answers_natively_regraded=True, all_features_present=True,
            combined_split_isolation_verified=True, train_validation_rows_tokenized=len(replay),
            independent_test_rows_tokenized=0)
    if base_reference is not None:
        result['pinned_base_model_manifest'] = base_reference
    atomic_json(root / 'verification.json', result)
    proof = manifest(TOKEN_KIND, {'data': str(data), 'source': str(source),
        'parent_completion': str(parent_completion), 'foundation_preflight': str(foundation_preflight),
        'evaluation_filter_profile': EVALUATION_FILTER_PROFILE,
        'max_tokens': max_tokens, 'workers': workers}, outputs=[observed, root / 'verification.json'], verification=result)
    proof['inputs'] = bound.artifacts
    if base_reference is not None:
        proof['config']['pinned_base_model_manifest'] = base_reference
    if replay_result['data_paths']:
        proof['config']['replay_data_paths'] = replay_result['data_paths']
    if replay_profile != LEGACY_REPLAY_PROFILE:
        proof['config']['replay_profile'] = replay_profile
    proof['cache_header_only_input'] = {'path': str(features), **bound.artifacts[str(features)],
                                       'fresh_full_bytes_hash_verified_by_this_preflight': True}
    atomic_json(root / 'manifest.json', proof)
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['init', 'parent-completion', 'data', 'features', 'foundation-preflight', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--replay-data', nargs='+', default=[], help='Certified original-course replay subsets')
    parser.add_argument('--replay-profile', choices=[LEGACY_REPLAY_PROFILE, AUXILIARY_REPLAY_PROFILE],
                        default=LEGACY_REPLAY_PROFILE)
    args = parser.parse_args()
    print(json.dumps(preflight(args.init, args.parent_completion, args.data, args.features,
        args.foundation_preflight, args.output, max_tokens=args.max_tokens, workers=args.workers,
        replay_data_paths=args.replay_data, replay_profile=args.replay_profile)), flush=True)


def main():
    from .verified_inputs import input_session_from_environment
    with input_session_from_environment():
        _main()


if __name__ == '__main__':
    main()
