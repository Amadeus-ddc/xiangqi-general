"""CPU token and feature-index checks for latent-only reviewed explanation data."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import torch

from .board_tokens import PAIRS
from .evidence import atomic_json, digest, manifest
from .extend_features import validate_cache
from .feature_store import feature_paths, feature_proof_path, is_feature_store, open_feature_cache
from .foundation_preflight import bounded_results, clean_recipe
from .recorded_coach import SPLITS, checked_manifest, check_artifacts
from .reviewed_explanations import source_contract
from .training import encode_record, messages

_TOKENIZER = None
_TOKEN_LIMIT = None


def encoded_lengths(tokenizer, row, token_limit):
    prompt = messages(row, 'bridge', None)[1]['content']
    if row['fen'] in prompt or '棋盘FEN：' in prompt or '棋盘各格：' in prompt:
        raise ValueError('Latent-only explanation prompt contains literal board state')
    tokens, labels = encode_record(tokenizer, row, token_limit, 'bridge', True, None)
    supervised = sum(label != -100 for label in labels)
    if not supervised:
        raise ValueError('Explanation has no supervised answer tokens')
    return {'id': row['id'], 'split': row['split'], 'tokens': len(tokens),
            'supervised_tokens': supervised, 'prompt_tokens': len(tokens) - supervised}


def token_initializer(model_path, token_limit):
    from transformers import AutoTokenizer
    global _TOKENIZER, _TOKEN_LIMIT
    torch.set_num_threads(1)
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    _TOKENIZER.pad_token = _TOKENIZER.eos_token
    if _TOKENIZER.add_tokens([t for _, t in PAIRS], special_tokens=False) != len(PAIRS):
        raise ValueError('Pinned base tokenizer already contains trained board tokens')
    _TOKEN_LIMIT = token_limit


def token_chunk(rows):
    return [encoded_lengths(_TOKENIZER, row, _TOKEN_LIMIT) for row in rows]


def preflight(data, foundation_config, foundation_preflight, output, *, max_tokens=1024, workers=8):
    """Precheck immutable labels without loading a model or starting training."""
    if type(workers) is not int or workers < 1 or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('Positive worker and token budgets are required')
    root, data = Path(output), Path(data)
    if root.exists():
        raise FileExistsError('Preserve every completed or interrupted preflight; use a fresh output')
    config_path, completed_path = Path(foundation_config), Path(foundation_preflight)
    recipe = json.loads(config_path.read_text());clean_recipe(recipe)
    data_manifest = data / 'manifest.json';data_proof = checked_manifest(data_manifest)
    if data_proof.get('kind') != 'combined_reviewed_explanation_dataset':
        raise ValueError('Explanation preflight requires a completed native-checked combined dataset')
    paths = [data / f'{s}.jsonl' for s in SPLITS];check_artifacts(data_proof, paths)
    rows, counts, ids, history_keys = [], dict.fromkeys(SPLITS, 0), set(), set()
    for split, path in zip(SPLITS, paths):
        for line in path.open():
            row = json.loads(line)
            if row['split'] != split or row['stage'] != 'explanation':
                raise ValueError('Explanation stage or protected split changed')
            if row['id'] in ids or row['feature_key'] in history_keys:
                raise ValueError('Duplicate explanation label or full-history feature identity')
            ids.add(row['id']);history_keys.add(row['feature_key']);counts[split] += 1;rows.append(row)
    source_contract(data_proof, counts)
    if data_proof['verification'].get('all_training_rows_have_inline_or_supplemental_acceptance') is not True:
        raise ValueError('All training labels require completed upstream acceptance')
    completed = json.loads(completed_path.read_text())
    cache_path = Path(recipe['feature_path']);cache_manifest = feature_proof_path(cache_path)
    cache_proof = json.loads(cache_manifest.read_text())
    if (completed.get('status') != 'complete' or
            completed.get('kind') != 'recorded_engine_clean_latent_foundation_preflight' or
            cache_proof.get('status') != 'complete'):
        raise ValueError('Completed foundation preflight and cache production proofs are required')
    check_artifacts(completed, [config_path, cache_manifest], 'inputs')
    expected = ({'sha256': digest(cache_path), 'bytes': cache_path.stat().st_size}
                if is_feature_store(cache_path) else cache_proof['outputs'].get(str(cache_path)))
    if (expected is None or completed['inputs'].get(str(cache_path)) != expected or
            completed['verification'].get('cache_sha256') != expected['sha256'] or
            completed['verification'].get('source_recipe_sha256') != digest(config_path) or
            cache_path.stat().st_size != expected['bytes']):
        raise ValueError('Feature cache identity differs from completed production and full preflight')
    cache = open_feature_cache(cache_path, mmap=True)
    validate_cache(cache, recipe['expert_feature_depths'])
    if len(cache['features']) != len(recipe['decoder_bridge_positions']):
        raise ValueError('Expert depths and latent bridge positions differ')
    cached_keys = set(cache['keys'])
    if any(row['feature_key'] not in cached_keys for row in rows):
        raise ValueError('Reviewed explanation is missing an existing expert feature context')
    model_path = Path(recipe['model_path'])
    tokenizer_paths = [model_path / n for n in ['config.json', 'tokenizer_config.json', 'tokenizer.json']]
    tokenizer_paths += [model_path / n for n in ['special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt']
                        if (model_path / n).exists()]
    tokenizer_hashes = {str(p): digest(p) for p in tokenizer_paths}
    eligible = [row for row in rows if row['split'] != 'test']
    root.mkdir(parents=True);observations = root / 'token-lengths.jsonl';maximum = Counter();token_counts = Counter()
    def save(results, handle):
        for item in results:
            maximum[item['split']] = max(maximum[item['split']], item['tokens'])
            token_counts[item['split']] += 1;handle.write(json.dumps(item) + '\n')
    with observations.open('x') as handle:
        if workers == 1:
            token_initializer(str(model_path), max_tokens)
            save(token_chunk(eligible), handle)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=token_initializer, initargs=(str(model_path), max_tokens)) as pool:
                for result in bounded_results(pool, token_chunk, eligible, workers, 128):
                    save(result, handle)
    if dict(token_counts) != {s: n for s, n in counts.items() if s != 'test' and n}:
        raise ValueError('Actual token coverage differs from the protected train/validation splits')
    if checked_manifest(data_manifest) != data_proof:
        raise ValueError('Explanation labels changed during token preflight')
    for path, sha in tokenizer_hashes.items():
        if digest(path) != sha:raise ValueError('Local tokenizer assets changed during preflight')
    if hasattr(cache, 'check_unchanged'):
        cache.check_unchanged()
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
        'reviewed_label_rows': len(rows), 'by_split': counts,
        'train_validation_rows_tokenized': len(eligible), 'tokenized_rows_by_split': dict(token_counts),
        'maximum_tokens_by_split': dict(maximum), 'token_limit': max_tokens,
        'all_train_validation_supervision_fits_without_truncation': True,
        'independent_test_rows_tokenized': 0, 'independent_test_used_for_training_or_selection': False,
        'feature_contexts_checked': len(history_keys), 'feature_contexts_missing': 0,
        'existing_cache_contexts': len(cache['keys']), 'cache_depths': cache['depths'],
        'existing_cache_header_dimensions_and_precision_checked': True,
        'complete_cache_producer_and_full_foundation_preflight_reused': True,
        'full_feature_cache_rehashed_again_by_this_preflight': is_feature_store(cache_path),
        'feature_tensor_values_recomputed_or_verified_by_this_preflight': False,
        'local_tokenizer_assets_freshly_hash_bound': True, 'pinned_model_revision': recipe['model_revision'],
        'decoder_literal_board_text': None, 'model_weights_loaded_or_training_started': False,
        'new_neural_labels_generated': False, 'active_foundation_inputs_changed': False,
        'four_new_raw_course_gates_required_before_sft': True, 'ready_to_start_sft': False,
        'student_benefit_proven': False}
    atomic_json(root / 'verification.json', summary)
    proof = manifest('latent_only_reviewed_explanation_data_preflight',
        {'data': str(data), 'foundation_config': str(config_path), 'foundation_preflight': str(completed_path),
         'workers': workers, 'max_tokens': max_tokens},
        [data_manifest, *paths, config_path, completed_path, cache_manifest,
         *(feature_paths(cache_path) if is_feature_store(cache_path) else []), *tokenizer_paths],
        [observations, root / 'verification.json'], summary)
    proof['cache_header_only_input'] = {'path': str(cache_path), **expected,
                                       'fresh_full_bytes_hash_verified_by_this_preflight': is_feature_store(cache_path)}
    atomic_json(root / 'manifest.json', proof)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--foundation-config', required=True)
    parser.add_argument('--foundation-preflight', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--public-evidence')
    args = parser.parse_args()
    summary = preflight(args.data, args.foundation_config, args.foundation_preflight, args.output,
                        max_tokens=args.max_tokens, workers=args.workers)
    if args.public_evidence:
        atomic_json(args.public_evidence, dict(summary, manifest_sha256=digest(Path(args.output) / 'manifest.json')))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
