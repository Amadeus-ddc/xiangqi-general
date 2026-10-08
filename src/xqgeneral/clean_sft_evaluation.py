"""Wait for clean initial SFT, then evaluate paired raw outputs and blinded prose."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest
from .evaluate_explanations import explanation_summary
from .explanations import EXPLANATION_QUESTION
from .foundation_handoff import Artifacts
from .review_explanations import DIMENSIONS, blinded_query, parse_rating
from .sft import checked_parent, prepare_config
from .sft_preflight import checked_artifacts, full_parameter_summary


def validation_spec(config):
    if config['split'] != 'validation' or config['memories'] != ['normal', 'zero', 'shuffled']:
        raise ValueError('Development evaluation requires all three paired validation memories')
    for key in ['validation_examples', 'batch_size', 'max_new_tokens', 'nodes', 'workers', 'review_max_new_tokens']:
        if type(config[key]) is not int or config[key] <= 0:raise ValueError('Evaluation budgets must be positive integers')
    token = json.loads(Path(config['token_preflight']).read_text());checked_artifacts(token)
    recipe = json.loads(Path(config['sft_recipe']).read_text())
    if (token.get('kind') != 'latent_only_reviewed_explanation_data_preflight' or
            token['config']['data'] != config['data'] or recipe['data_path'] != config['data'] or
            recipe.get('require_clean_foundation_handoff') is not True or
            recipe['feature_path'] != config['features'] or token['cache_header_only_input']['path'] != config['features']):
        raise ValueError('Evaluation must use the exact reviewed clean SFT data and cache')
    path = Path(config['data']) / 'validation.jsonl';rows = load_jsonl(path)
    if (len(rows) != config['validation_examples'] or len({row['id'] for row in rows}) != len(rows) or
            len({row['feature_key'] for row in rows}) != len(rows) or
            any(row['split'] != 'validation' or row['stage'] != 'explanation' for row in rows)):
        raise ValueError('Evaluate every distinct validation label without changing its protected split')
    if config['batch_size'] < 2 or len(rows) < 2 or len(rows) % config['batch_size'] == 1:
        raise ValueError('Every paired shuffled batch needs at least two histories')
    return rows, token, recipe


def wait_for_sft(pipeline, producer_session, output, poll_seconds=30):
    if not 0 < poll_seconds <= 60:raise ValueError('Use a positive polling interval of at most 60 seconds')
    path = Path(pipeline) / 'manifest.json'
    while not path.exists():
        if subprocess.run(['tmux', 'has-session', '-t', '=' + producer_session], capture_output=True).returncode:
            if path.exists():break
            raise RuntimeError('The owned clean SFT pipeline stopped before a completed model was available')
        atomic_json(Path(output) / 'state.json', {'status': 'waiting_for_completed_clean_sft',
            'pipeline': str(pipeline), 'producer_session': producer_session, 'gpu_model_or_reviewer_loaded': False})
        time.sleep(poll_seconds)
    return path


def checked_sft(pipeline, config, recipe):
    root = Path(pipeline);proof = json.loads((root / 'manifest.json').read_text());checked_artifacts(proof)
    result = proof['verification'];source = root / 'sft/adapter.pt'
    if (proof.get('kind') != 'clean_curriculum_to_initial_sft_pipeline' or
            any(result.get(key) is not True for key in ['four_courses_completed_before_any_gpu_sft_work',
                'clean_handoff_verified', 'actual_four_gpu_full_decoder_preflight_passed',
                'initial_explanation_sft_completed', 'complete_shapes_and_finite_fp32_verified']) or
            result.get('world_size') != 4 or result.get('selected_sft_step', 0) <= 0 or
            result['selected_checkpoint'] != str(source) or
            proof['config']['recipe'] != config['sft_recipe'] or
            proof['config']['recipe_sha256'] != digest(config['sft_recipe']) or
            proof['config']['token_preflight'] != config['token_preflight'] or
            proof['config']['token_preflight_sha256'] != digest(config['token_preflight']) or
            result.get('independent_test_used_for_training_or_selection') is not False):
        raise ValueError('Require the actual completed clean SFT pipeline and its exact selected model')
    trained = json.loads((root / 'sft/manifest.json').read_text());checked_artifacts(trained)
    saved = json.loads((root / 'sft/config.json').read_text())
    expected = prepare_config(checked_parent(root / 'handoff/adapter.pt', True), recipe,
                              root / 'handoff/adapter.pt', root / 'sft')
    checkpoint = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    preflight = json.loads((root / 'preflight/manifest.json').read_text())
    initial_path = root / 'preflight/zero-init.pt'
    checked_artifacts({'status': 'complete', 'inputs': {},
                       'outputs': {str(initial_path): preflight['outputs'][str(initial_path)]}})
    reference = torch.load(root / 'preflight/zero-init.pt', map_location='cpu', weights_only=True, mmap=True)['trainable']
    summary = full_parameter_summary(checkpoint['trainable'], reference)
    inputs = {path: item['sha256'] for path, item in trained['inputs'].items()}
    inputs[config['features']] = preflight['fresh_feature_cache_input']['sha256']
    if (trained.get('kind') != 'model_training' or saved != expected or trained['config'] != expected or
            checkpoint['config'] != expected or checkpoint['code'] != proof['code'] or trained['code'] != proof['code'] or
            checkpoint['input_hashes'] != inputs or
            checkpoint['selected_step'] != result['selected_sft_step'] or
            trained['verification']['selected_step'] != result['selected_sft_step'] or
            trained['verification']['steps_completed'] != result['actual_sft_steps'] or
            summary['trainable_parameters'] != result['trainable_parameters'] or
            summary['trainable_tensors'] != result['trainable_tensors'] or
            expected['decoder_training'] != 'full' or expected['feature_path'] != config['features']):
        raise ValueError('The actual selected full-decoder state differs from clean SFT completion')
    del checkpoint, reference
    return source, proof


def teacher_artifacts(path):
    teacher = json.loads(Path(path).read_text())['search_consolidator'];root = Path(teacher['model_path'])
    guard = Artifacts();proof = guard.json(root / 'weights.manifest.json')
    if (teacher.get('quantization') != 'none' or teacher.get('inference_dtype') != 'bfloat16' or
            any(proof.get(key) != teacher[key] for key in ['repository', 'revision', 'quantization']) or
            proof.get('all_official_lfs_sha256_matched') is not True or not proof.get('shards') or
            set(proof.get('stored_tensor_dtypes', {})) != {'BF16'} or
            sum(item['bytes'] for item in proof['shards'].values()) != proof['total_weight_bytes']):
        raise ValueError('The prose reviewer requires the pinned full BF16 teacher weights')
    for name, item in proof['shards'].items():
        if Path(name).name != name or not name.endswith('.safetensors'):raise ValueError('Invalid declared teacher shard')
        guard.add(root / name, item)
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        p = root / name;guard.add(p, {'sha256': digest(p), 'bytes': p.stat().st_size})
    return teacher, guard


def checked_raw(root, memory, rows, args, core):
    proof = json.loads((root / 'manifest.json').read_text())
    if (proof.get('status') != 'complete' or proof.get('kind') != 'independent_explanation_evaluation' or
            proof['config'] != args or proof['inputs'] != core.expected or proof['code'] != code_identity()):
        raise ValueError('Raw evaluation differs from the paired model, inputs or generation budget')
    core.unchanged();checked_artifacts({'status': 'complete', 'inputs': {}, 'outputs': proof['outputs']})
    raw = load_jsonl(root / 'raw-predictions.jsonl');judged = load_jsonl(root / 'judged-predictions.jsonl')
    expected = {row['id']: dict(row, question=EXPLANATION_QUESTION) for row in rows}
    if len(raw) != len(expected) or len(judged) != len(expected) or len({r['id'] for r in raw}) != len(raw):
        raise ValueError('Raw validation must preserve every requested answer, including errors')
    by_id = {row['id']: row for row in raw}
    if (set(by_id) != set(expected) or len({r['id'] for r in judged}) != len(judged) or
            any(row['record'] != expected[row['id']] for row in raw) or
            any(row['id'] not in by_id or {k: row[k] for k in ['id', 'record', 'raw']} != by_id[row['id']] for row in judged)):
        raise ValueError('Engine judgments must bind the exact unmodified raw answers and protected records')
    metrics = json.loads((root / 'metrics.json').read_text());recomputed = explanation_summary(judged)
    if (metrics != proof['verification'] or any(metrics.get(k) != v for k, v in recomputed.items()) or
            metrics.get('split') != 'validation' or metrics.get('memory') != memory or
            metrics.get('judge_nodes_per_position') != args['nodes'] or
            metrics.get('engine_sha256') != core.expected[args['executable']]['sha256'] or
            metrics.get('engine_weights_sha256') != core.expected[args['weights']]['sha256']):
        raise ValueError('Raw aggregate metrics differ from their complete judgment rows')
    return metrics, judged


def run_evaluation(pipeline, producer_session, config_path, output, *, resume=False, poll_seconds=30):
    config_path = Path(config_path);root = Path(output);config = json.loads(config_path.read_text())
    rows, token, recipe = validation_spec(config)
    teacher, teacher_guard = teacher_artifacts(config['teacher_config'])
    contract = {'pipeline': str(pipeline), 'producer_session': producer_session,
        'config': str(config_path), 'config_sha256': digest(config_path), 'code': code_identity(),
        'prerequisite_sha256': {p: digest(p) for p in [config['sft_recipe'], config['token_preflight'],
                                                    config['teacher_config']]}, 'teacher_assets': teacher_guard.expected,
        'engine_artifacts': {p: {'sha256': digest(p), 'bytes': Path(p).stat().st_size}
                             for p in [config['executable'], config['weights']]}}
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Resume only the exact preserved evaluation contract; otherwise use a fresh output')
        if (root / 'manifest.json').exists():
            proof = json.loads((root / 'manifest.json').read_text());checked_artifacts(proof);return proof['verification']
    else:root.mkdir(parents=True);atomic_json(root / 'contract.json', contract)
    stage = 'waiting_for_completed_clean_sft'
    def unchanged():
        if digest(config_path) != contract['config_sha256'] or code_identity() != contract['code']:
            raise ValueError('Evaluation configuration or preserved source changed')
        if any(digest(p) != sha for p, sha in contract['prerequisite_sha256'].items()):
            raise ValueError('Evaluation prerequisite changed')
        checked_artifacts(token)
    def execute(command):
        with (root / 'commands.jsonl').open('a') as log:log.write(json.dumps(command) + '\n')
        with (root / 'log.txt').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=config['cuda_visible_devices']))
    try:
        source_manifest = wait_for_sft(pipeline, producer_session, root, poll_seconds)
        unchanged();source, parent = checked_sft(pipeline, config, recipe)
        core = Artifacts()
        core.add(source, parent['outputs'][str(source)])
        validation = str(Path(config['data']) / 'validation.jsonl');core.add(validation, token['inputs'][validation])
        cache = token['cache_header_only_input'];core.add(config['features'], {k: cache[k] for k in ['sha256', 'bytes']})
        for path, item in contract['engine_artifacts'].items():core.add(path, item)
        core.verify();teacher_guard.verify()
        metrics, normal_rows, proofs = {}, None, []
        for memory in config['memories']:
            stage = 'raw_validation_' + memory;atomic_json(root / 'state.json', {'status': stage})
            dest = root / memory
            args = {'checkpoint': str(source), 'data': config['data'], 'features': config['features'],
                    'split': 'validation', 'limit': len(rows), 'memory': memory, 'output': str(dest),
                    **{key: config[key] for key in ['batch_size', 'max_new_tokens', 'nodes', 'workers', 'seed', 'executable', 'weights']}}
            if not (dest / 'manifest.json').exists():
                if dest.exists():raise ValueError('Preserve incomplete raw evaluation; use a fresh output for that experiment')
                command = [sys.executable, '-u', '-m', 'xqgeneral.evaluate_explanations']
                for key, value in args.items():command += ['--' + key.replace('_', '-'), str(value)]
                execute(command)
            metrics[memory], judged = checked_raw(dest, memory, rows, args, core)
            if memory == 'normal':normal_rows = judged
            proofs.append(dest / 'manifest.json');unchanged();teacher_guard.unchanged()
        stage = 'blinded_prose_review';atomic_json(root / 'state.json', {'status': stage})
        queries, inference, ratings = [root / name for name in ['queries', 'reviewer', 'ratings']]
        if not (queries / 'manifest.json').exists():
            execute([sys.executable, '-u', '-m', 'xqgeneral.review_explanations', 'prepare', '--predictions',
                     str(root / 'normal/judged-predictions.jsonl'), '--limit', str(len(rows)), '--seed',
                     str(config['seed']), '--output', str(queries)])
        query_proof = json.loads((queries / 'manifest.json').read_text());checked_artifacts(query_proof)
        query_rows = load_jsonl(queries / 'queries.jsonl');expected_queries = {r['id']: blinded_query(r) for r in normal_rows}
        if (query_proof.get('kind') != 'blinded_neural_explanation_review' or query_proof['code'] != contract['code'] or
                query_proof['verification'].get('split') != 'validation' or len(query_rows) != len(rows) or
                len({q['id'] for q in query_rows}) != len(rows) or
                any(q != expected_queries.get(q['id']) for q in query_rows)):
            raise ValueError('Blinded review must cover the exact normal raw validation and grounded queries')
        if not (inference / 'manifest.json').exists():
            execute([sys.executable, '-u', '-m', 'xqgeneral.local_teacher', '--config', config['teacher_config'],
                     '--input', str(queries / 'queries.jsonl'), '--max-new-tokens', str(config['review_max_new_tokens']),
                     '--output', str(inference)])
        inference_proof = json.loads((inference / 'manifest.json').read_text());checked_artifacts(inference_proof)
        if (inference_proof.get('kind') != 'local_teacher_inference' or inference_proof['code'] != contract['code'] or
                inference_proof['config']['input'] != str(queries / 'queries.jsonl') or
                inference_proof['config']['config'] != config['teacher_config'] or
                inference_proof['config']['max_new_tokens'] != config['review_max_new_tokens']):
            raise ValueError('Reviewer inference differs from the actual grounded queries and pinned teacher')
        if not (ratings / 'manifest.json').exists():
            execute([sys.executable, '-u', '-m', 'xqgeneral.review_explanations', 'collect', '--config', config['teacher_config'],
                     '--queries', str(queries / 'queries.jsonl'), '--responses', str(inference / 'responses.jsonl'), '--output', str(ratings)])
        rating_proof = json.loads((ratings / 'manifest.json').read_text());checked_artifacts(rating_proof)
        if (rating_proof.get('kind') != 'blinded_neural_explanation_review' or rating_proof['code'] != contract['code'] or
                rating_proof['config']['queries'] != str(queries / 'queries.jsonl') or
                rating_proof['config']['responses'] != str(inference / 'responses.jsonl') or
                rating_proof['config']['config'] != config['teacher_config']):
            raise ValueError('Collected ratings differ from the actual neural review inputs')
        responses = load_jsonl(inference / 'responses.jsonl');accepted, rejected = {}, set()
        if len(responses) != len(rows) or len({r['id'] for r in responses}) != len(rows) or {r['id'] for r in responses} != set(expected_queries):
            raise ValueError('Neural review response coverage differs from the actual blinded queries')
        for response in responses:
            try:accepted[response['id']] = dict(id=response['id'], **parse_rating(response, teacher))
            except (ValueError, KeyError, TypeError):rejected.add(response['id'])
        actual_ratings = load_jsonl(ratings / 'ratings.jsonl');actual_rejections = json.loads((ratings / 'rejected.json').read_text())
        if (len(actual_ratings) != len(accepted) or {r['id']: r for r in actual_ratings} != accepted or
                len(actual_rejections) != len(rejected) or {r['id'] for r in actual_rejections} != rejected):
            raise ValueError('Collected ratings must preserve every accepted and rejected raw teacher answer')
        summary = rating_proof['verification']
        means = {k: sum(r[k] for r in accepted.values()) / len(accepted) if accepted else None for k in DIMENSIONS}
        fraction = sum(all(r[k] >= 4 for k in DIMENSIONS) for r in accepted.values()) / len(rows)
        if (summary.get('requested') != len(rows) or summary.get('valid_ratings') != len(accepted) or
                summary.get('rejected') != len(rejected) or summary.get('mean_scores_conditional_on_valid_ratings') != means or
                summary.get('all_dimensions_at_least_4_fraction_all_queries') != fraction or
                summary.get('ratings_with_unsupported_claims') != sum(bool(r['unsupported_claims']) for r in accepted.values())):
            raise ValueError('Neural aggregates differ from actual accepted and rejected ratings')
        unchanged();core.unchanged();teacher_guard.unchanged()
        final = {'status': 'complete', 'evidence_state': 'reconstructed_baseline', 'split': 'validation',
            'actual_clean_initial_sft_completed_before_model_or_reviewer_load': True,
            'all_validation_examples': len(rows), 'raw_validation_by_memory': metrics,
            'paired_memory_rate_differences_normal_minus_ablation': {memory: {k: metrics['normal'][k] - metrics[memory][k]
                for k in ['first_move_no_mistake_rate', 'structured_contract_valid_rate', 'pv_legal_rate']}
                for memory in ['zero', 'shuffled']}, 'blinded_prose_review': summary,
            'raw_answers_repaired': False, 'neural_ratings_are_human_ratings': False,
            'independent_test_used_for_training_or_selection': False, 'strong_play_or_reliable_coaching_proven': False}
        atomic_json(root / 'verification.json', final)
        proofs += [queries / 'manifest.json', inference / 'manifest.json', ratings / 'manifest.json']
        completed = manifest('clean_initial_sft_capability_validation', contract,
            [config_path, source_manifest, *[Path(p) for p in contract['prerequisite_sha256']], *proofs],
            [root / 'verification.json'], final)
        completed['inputs'].update(core.expected);completed['inputs'].update(teacher_guard.expected)
        for path in proofs:completed['outputs'].update(json.loads(path.read_text())['outputs'])
        atomic_json(root / 'manifest.json', completed);atomic_json(root / 'state.json', final)
        return final
    except Exception as error:
        atomic_json(root / 'state.json', {'status': 'failed', 'stage': stage, 'error_type': type(error).__name__,
            'preserve_partial_outputs': True, 'capability_evaluation_completion_proven': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', required=True);parser.add_argument('--producer-session', required=True)
    parser.add_argument('--config', default='configs/evaluation-clean-sft-v1.json');parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true');parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args();torch.set_num_threads(8)
    print(json.dumps(run_evaluation(args.pipeline, args.producer_session, args.config, args.output,
        resume=args.resume, poll_seconds=args.poll_seconds)), flush=True)


if __name__ == '__main__':main()
