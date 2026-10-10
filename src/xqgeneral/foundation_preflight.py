"""Full immutable-corpus checks before fresh, latent-only foundation training."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path
import random

import torch

from .board_tokens import PAIRS
from .curriculum_data import MATERIAL, verify_splits
from .course_tasks import PAPER_PROFILE, paper_answer, recipe_profile, row_profile, task_groups, validate_context, validate_query
from .evidence import atomic_json, digest, history_key, manifest
from .extend_features import validate_cache
from .gated_curriculum import TASKS, validate_recipe
from .rules import adjudicate, gives_check, legal_moves, piece_map, piece_name, replay
from .selfplay_grounding import question_variant
from .training import encode_record

_TOKENIZER = None
_RECIPE = None


def clean_recipe(recipe):
    validate_recipe(recipe)
    if recipe.get('init_from') or recipe.get('board_text') is not None:
        raise ValueError('Fresh latent foundation must not inherit trained weights or literal board text')
    if not recipe.get('board_tokens') or not recipe.get('feature_cache_mmap'):
        raise ValueError('Production foundation requires board tokens and mapped expert features')


def corpus_preserves_prefix(data_proof, profile):
    """Require the declared producer contract before accepting a new task corpus."""
    if profile == PAPER_PROFILE:
        from .paper_curriculum import DATA_KIND
        if (data_proof.get('kind') != DATA_KIND or row_profile(data_proof['config']) != profile or
                data_proof['verification'].get('task_profile') != profile or
                data_proof['verification'].get('all_supplied_roots_and_futures_natively_verified') is not True or
                data_proof['verification'].get('prior_corpus_isolation_verified') is not True):
            raise ValueError('Paper recipe requires its completed declared native question producer')
        return False
    if row_profile(data_proof['config']) != profile or not data_proof['config'].get('base_data'):
        raise ValueError('Legacy recorded curriculum requires its preserved base-data prefix')
    return True


def native_context(row):
    if row_profile(row) == PAPER_PROFILE:
        return validate_context(row)
    history = replay(row['initial_fen'], row['moves'])
    if (history != row['history'] or history[-1] != row['fen'] or
            history_key(history) != row['feature_key']):
        raise ValueError('Course root differs from its native full-history feature key')
    if row.get('recorded_source_kind') and adjudicate(row['initial_fen'], row['moves'])['ended']:
        raise ValueError('Recorded course contains a terminal root')
    return row['feature_key']


@lru_cache(maxsize=4096)
def checked_moves(fen):
    return tuple(sorted(move for move in legal_moves(fen) if gives_check(fen, move)))


def native_answer(row):
    """Recompute an answer from rule state, independently of stored labels."""
    if row_profile(row) == PAPER_PROFILE:
        return paper_answer(row)
    target = replay(row['fen'], row.get('future_moves', []))[-1]
    board, query, task = piece_map(target), row['query'], row['task_type']
    if task in ('piece', 'empty'):
        return piece_name(board.get(query['square']))
    if task == 'count':
        return str(sum(piece == query['symbol'] for piece in board.values()))
    if task == 'locate':
        return ' '.join(sorted(square for square, piece in board.items() if piece == query['symbol'])) or '无'
    if task == 'material':
        return str(sum(MATERIAL[p.lower()] for p in board.values() if p.isupper() == query['red']))
    if task == 'rank':
        return ' '.join(f'{s}:{piece_name(p)}' for s, p in sorted(board.items())
                        if int(s[1]) == query['rank']) or '无'
    legal = legal_moves(target)
    if task in ('legal', 'illegal'):
        return '合法' if query['move'] in legal else '不合法'
    if task == 'moves':
        return ' '.join(sorted(move for move in legal if move[:2] == query['source'])) or '无'
    if task == 'captures':
        return ' '.join(sorted(move for move in legal if move[2:] in board)) or '无'
    if task == 'checks':
        return ' '.join(checked_moves(target)) or '无'
    if task == 'terminal':
        return '有' if legal else '无'
    raise ValueError(f'Unknown foundation task: {task}')


def validate_task(row, expected_profile=None):
    profile = row_profile(row)
    if expected_profile is not None and profile != expected_profile:
        raise ValueError('Course row differs from the recipe task profile')
    if profile == PAPER_PROFILE:
        validate_query(row)
        return
    if row['stage'] in TASKS and row['task_type'] in TASKS[row['stage']]:
        return
    # Preserve the four historical independent-test terminal probes. They never
    # enter training/validation sampling or the 22-task course mastery gates.
    if (row['split'] == 'test' and row['stage'] in ('dynamic_current', 'dynamic_future') and
            row['task_type'] == 'terminal'):
        if native_answer(row) != row['answer']:
            raise ValueError('Independent terminal probe answer differs from native rules')
        return
    raise ValueError('Course task differs from the declared raw QA contract')


def token_initializer(recipe):
    from transformers import AutoTokenizer
    global _TOKENIZER, _RECIPE
    torch.set_num_threads(1)
    _RECIPE = recipe
    _TOKENIZER = AutoTokenizer.from_pretrained(recipe['model_path'], local_files_only=True)
    _TOKENIZER.pad_token = _TOKENIZER.eos_token
    if _TOKENIZER.add_tokens([token for _, token in PAIRS], special_tokens=False) != len(PAIRS):
        raise ValueError('Foundation base tokenizer already contains trained board tokens')


def token_chunk(rows):
    lengths, extra, checked = {}, 0, 0
    for row in rows:
        selected = [row]
        if row['split'] == 'validation':
            # Always check both explicit alternate forms as well as the stored request.
            selected += [dict(row, question=question_variant(row, variant)) for variant in (1, 2)]
            extra += 2
        for record in selected:
            tokens, labels = encode_record(_TOKENIZER, record, _RECIPE['max_tokens'], 'bridge', True, None)
            if not any(label != -100 for label in labels):
                raise ValueError('Course record has no supervised answer')
            lengths[row['stage']] = max(lengths.get(row['stage'], 0), len(tokens))
        checked += 1
    return {'checked': checked, 'extra_validation_formats': extra, 'maximum_tokens_by_stage': lengths}


def native_chunk(rows):
    return [native_context(row) for row in rows]


def bounded_results(pool, function, rows, workers, chunk_size):
    pending, chunk = deque(), []
    for row in rows:
        chunk.append(row)
        if len(chunk) == chunk_size:
            pending.append(pool.submit(function, chunk))
            chunk = []
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
    if chunk:
        pending.append(pool.submit(function, chunk))
    for job in pending:
        yield job.result()


def iter_rows(data):
    for split in ('train', 'validation', 'test'):
        with (data / f'{split}.jsonl').open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row['split'] != split:
                        raise ValueError('Course row differs from its preassigned game split')
                    yield row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--public-evidence', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--answer-samples-per-task', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261055)
    args = parser.parse_args()
    root, recipe_path = Path(args.output), Path(args.config)
    if root.exists() or min(args.workers, args.answer_samples_per_task) < 1:
        raise ValueError('Use a fresh preflight output and positive worker/sample budgets')
    recipe = json.loads(recipe_path.read_text())
    clean_recipe(recipe)
    profile = recipe_profile(recipe)
    data = Path(recipe['data_path'])
    data_proof_path, cache_proof_path = data / 'manifest.json', Path(recipe['feature_path']).with_suffix('.manifest.json')
    data_proof, cache_proof = [json.loads(p.read_text()) for p in (data_proof_path, cache_proof_path)]
    if data_proof['status'] != 'complete' or cache_proof['status'] != 'complete':
        raise ValueError('Production data and expert feature extension must both be complete')
    preserves_prefix = corpus_preserves_prefix(data_proof, profile)
    root.mkdir(parents=True)
    paths = [data / f'{s}.jsonl' for s in ('train', 'validation', 'test')]
    source_hashes = {}
    for path in paths:
        source_hashes[str(path)] = digest(path)
        if source_hashes[str(path)] != data_proof['outputs'][str(path)]['sha256']:
            raise ValueError('Completed course question bytes changed')
    feature_sha = digest(recipe['feature_path'])
    if feature_sha != cache_proof['outputs'][recipe['feature_path']]['sha256']:
        raise ValueError('Completed production feature cache bytes changed')
    weights = cache_proof['config']['weights']
    if digest(weights) != cache_proof['verification']['expert_sha256']:
        raise ValueError('Pinned expert weights changed')
    cache = torch.load(recipe['feature_path'], map_location='cpu', weights_only=True, mmap=True)
    validate_cache(cache, recipe['expert_feature_depths'])
    if len(cache['features']) != len(recipe['decoder_bridge_positions']):
        raise ValueError('Bridge and expert-cache layer dimensions differ')
    keys = set(cache['keys'])
    ids, histories, counts, task_counts, horizons, source_kinds = set(), set(), Counter(), Counter(), Counter(), Counter()
    pools, pool_counts, rng = {}, Counter(), random.Random(args.seed)
    contexts_path = root / 'unique-contexts.jsonl'
    with contexts_path.open('x') as contexts:
        def inspected_rows():
            for row in iter_rows(data):
                if row['id'] in ids or row['feature_key'] not in keys:
                    raise ValueError('Duplicate label identity or missing expert full-history key')
                validate_task(row, profile)
                ids.add(row['id'])
                counts[f"{row['split']}/{row['stage']}"] += 1
                task_counts[f"{row['split']}/{row['stage']}/{row['task_type']}"] += 1
                if row['future_moves']:
                    horizons[len(row['future_moves'])] += 1
                source_kinds[row.get('recorded_source_kind', 'previous_engine_or_seeded_rules')] += 1
                if row['feature_key'] not in histories:
                    if history_key(row['history']) != row['feature_key']:
                        raise ValueError('Feature key differs from the actual full FEN history')
                    histories.add(row['feature_key'])
                    identity = {k: row[k] for k in ('initial_fen', 'moves', 'history', 'fen', 'feature_key')}
                    if row.get('recorded_source_kind'):
                        identity['recorded_source_kind'] = row['recorded_source_kind']
                    if profile == PAPER_PROFILE:
                        identity['task_profile'] = profile
                    contexts.write(json.dumps(identity, ensure_ascii=False) + '\n')
                if row['split'] == 'train':
                    kind = 'recorded' if row.get('recorded_source_kind') else 'previous_engine_or_seeded'
                    group = kind, row['stage'], row['task_type']
                    pool_counts[group] += 1
                    pool = pools.setdefault(group, [])
                    if len(pool) < args.answer_samples_per_task:
                        pool.append(row)
                    else:
                        index = rng.randrange(pool_counts[group])
                        if index < args.answer_samples_per_task:
                            pool[index] = row
                if row['split'] != 'test':
                    yield row
        print(json.dumps({'phase': 'full_train_validation_token_preflight', 'workers': args.workers}), flush=True)
        checked, extra, lengths = 0, 0, {}
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=token_initializer, initargs=(recipe,)) as pool:
            for result in bounded_results(pool, token_chunk, inspected_rows(), args.workers, 256):
                checked += result['checked']
                extra += result['extra_validation_formats']
                for stage, value in result['maximum_tokens_by_stage'].items():
                    lengths[stage] = max(lengths.get(stage, 0), value)
                if checked % 25600 == 0:
                    print(json.dumps({'train_validation_records_tokenized': checked,
                                      'extra_validation_formats': extra}), flush=True)
    expected_counts = data_proof['verification']['counts']
    if len(ids) != sum(expected_counts.values()) or checked != expected_counts['train'] + expected_counts['validation']:
        raise ValueError('Preflight record counts differ from the completed course manifest')
    if any(sum(n for group, n in counts.items() if group.startswith(split + '/')) != number
           for split, number in expected_counts.items()):
        raise ValueError('Preflight split counts differ from the completed course manifest')
    for stage, tasks in task_groups(profile).items():
        for task in tasks:
            if task_counts[f'validation/{stage}/{task}'] < recipe['raw_qa_gates']['per_task']:
                raise ValueError('Validation lacks enough records for a declared raw QA task')
            if profile == PAPER_PROFILE and not task_counts[f'train/{stage}/{task}']:
                raise ValueError('Training lacks records for a declared paper QA task')
    tokens = {'status': 'complete', 'train_validation_records_tokenized': checked,
              'extra_validation_format_encodings': extra, 'maximum_tokens_by_stage': lengths,
              'source_data_sha256': source_hashes, 'source_recipe_sha256': digest(recipe_path)}
    atomic_json(root / 'token-verification.json', tokens)
    print(json.dumps({'phase': 'all_unique_root_native_replay', 'histories': len(histories)}), flush=True)
    def context_rows():
        with contexts_path.open() as handle:
            for line in handle:
                yield json.loads(line)
    natively_verified = set()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for verified in bounded_results(pool, native_chunk, context_rows(), args.workers, 128):
            natively_verified.update(verified)
    if histories != natively_verified:
        raise ValueError('Native root replay coverage differs from all question histories')
    samples, native_counts = [], Counter()
    for group in sorted(pools):
        if len(pools[group]) != args.answer_samples_per_task:
            raise ValueError('Native answer sampling lacks a source/task group')
        for row in pools[group]:
            if native_answer(row) != row['answer']:
                raise ValueError(f"Native answer mismatch: {row['id']}")
            native_counts['/'.join(group)] += 1
            samples.append(row)
    atomic_json(root / 'native-answer-sample.json', {'seed': args.seed, 'records': samples})
    print(json.dumps({'phase': 'all_root_and_future_split_verification', 'workers': args.workers}), flush=True)
    split_proof = verify_splits(iter_rows(data), workers=args.workers)
    if split_proof != data_proof['verification']['split_verification']:
        raise ValueError('Independent root/future split verification differs from production generation')
    # The combined corpus preserves every prior engine question byte as a prefix.
    if preserves_prefix:
        base = Path(data_proof['config']['base_data'])
        for path in paths:
            original = base / path.name
            with original.open('rb') as source, path.open('rb') as destination:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                    if destination.read(len(block)) != block:
                        raise ValueError('Combined course changed a preserved engine question byte')
            if digest(original) != data_proof['inputs'][str(original)]['sha256']:
                raise ValueError('Preserved prior engine corpus changed')
    result = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
              'total_rule_records': len(ids), 'records_by_split_stage': dict(counts),
              'records_by_source_kind': dict(source_kinds), 'actual_future_horizons': dict(horizons),
              **tokens, 'actual_unique_histories_natively_replayed': len(histories),
              'all_question_history_keys_cached': True, 'cache_keys': len(keys), 'cache_depths': cache['depths'],
              'cache_sha256': feature_sha, 'native_answer_samples': len(samples),
              'native_answer_samples_by_source_and_task': dict(native_counts),
              'native_sample_is_not_full_label_accuracy': True, 'split_verification': split_proof,
              'previous_engine_file_bytes_preserved_as_prefix': preserves_prefix,
              'decoder_literal_board_text': None, 'fresh_bridge_and_token_initialization_required': True,
              'independent_test_used_for_training_or_selection': False,
              'actual_student_training_started': False, 'student_improvement_measured': False}
    if profile == PAPER_PROFILE:
        result['task_profile'] = profile
    atomic_json(root / 'verification.json', result)
    inputs = [recipe_path, data_proof_path, cache_proof_path, *paths, recipe['feature_path'], weights]
    outputs = [root / 'verification.json', root / 'token-verification.json', contexts_path, root / 'native-answer-sample.json']
    atomic_json(root / 'manifest.json', manifest('recorded_engine_clean_latent_foundation_preflight', vars(args), inputs, outputs, result))
    atomic_json(args.public_evidence, dict(result, readback_manifest_sha256=digest(root / 'manifest.json')))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
