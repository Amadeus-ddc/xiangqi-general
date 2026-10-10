"""Measure clean-model search yield after the existing SFT validation completes."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .clean_sft_evaluation import checked_sft, teacher_artifacts
from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest
from .recorded_search_inputs import ARTIFACTS, BoundInputs


def pilot_spec(config, bound):
    for key in ['limit', 'nodes', 'teacher_max_new_tokens']:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError('Positive integer search pilot budgets are required')
    if type(config.get('teacher_batch_size', 1)) is not int or config.get('teacher_batch_size', 1) <= 0:
        raise ValueError('A positive integer teacher batch size is required')
    if (type(config['max_depth']) is not int or config['max_depth'] < 0 or
            type(config['seed']) is not int or config['child_contract'] not in {'full', 'move_eval'}):
        raise ValueError('Invalid recursive search pilot configuration')
    evaluation = bound.json(config['evaluation_config'])
    recipe = bound.json(evaluation['sft_recipe'])
    token = bound.json(evaluation['token_preflight'])
    pool = bound.json(Path(config['recorded_inputs']) / 'manifest.json')
    counts = bound.json(Path(config['recorded_inputs']) / 'counts.json')
    if (pool.get('status') != 'complete' or pool.get('kind') != 'prepared_unused_recorded_search_roots' or
            pool['config']['data'] != config['data'] or pool['config']['seed'] != config['seed'] or
            pool['config']['heldout_data'] != evaluation['data'] or
            counts['candidate_training_histories'] < config['limit'] or
            evaluation.get('split') != 'validation' or evaluation.get('memories') != ['normal', 'zero', 'shuffled'] or
            recipe.get('require_clean_foundation_handoff') is not True or
            recipe['data_path'] != evaluation['data'] or recipe['feature_path'] != evaluation['features'] or
            token.get('status') != 'complete' or
            token.get('kind') != 'latent_only_reviewed_explanation_data_preflight' or
            token['config']['data'] != evaluation['data'] or
            token['cache_header_only_input']['path'] != evaluation['features']):
        raise ValueError('The pilot requires the original isolated pool and exact clean SFT validation inputs')
    teacher, teacher_guard = teacher_artifacts(evaluation['teacher_config'])
    bound.bind(evaluation['teacher_config'])
    return evaluation, recipe, token, pool, teacher, teacher_guard


def wait_for_validation(validation, producer_session, output, poll_seconds=30):
    if not 0 < poll_seconds <= 60:
        raise ValueError('Use a positive polling interval of at most 60 seconds')
    path = Path(validation) / 'manifest.json'
    while not path.exists():
        if subprocess.run(['tmux', 'has-session', '-t', '=' + producer_session], capture_output=True).returncode:
            if path.exists():
                break
            raise RuntimeError('The owned clean SFT validation stopped before completion')
        atomic_json(Path(output) / 'state.json', {'status': 'waiting_for_completed_clean_sft_validation',
            'validation': str(validation), 'producer_session': producer_session,
            'student_or_teacher_loaded': False, 'new_distillation_labels': 0})
        time.sleep(poll_seconds)
    return path


def checked_proof(path, kind, bound, *, arguments=None, code=None):
    proof = bound.json(path)
    if (proof.get('status') != 'complete' or proof.get('kind') != kind or
            arguments is not None and proof.get('config') != arguments or
            code is not None and proof.get('code') != code):
        raise ValueError('Search pilot stage differs from its completed execution contract')
    for section in ['inputs', 'outputs']:
        for name, expected in proof[section].items():
            bound.bind(name, expected)
    return proof


def run_pilot(pipeline, validation, producer_session, config_path, output, *, resume=False, poll_seconds=30):
    root, config_path = Path(output), Path(config_path)
    bound = BoundInputs()
    config = bound.json(config_path)
    evaluation, recipe, token, pool, teacher, teacher_guard = pilot_spec(config, bound)
    contract = {'pipeline': str(pipeline), 'validation': str(validation),
        'producer_session': producer_session, 'config': str(config_path),
        'config_sha256': digest(config_path), 'code': code_identity(),
        'prerequisite_inputs': dict(bound.artifacts), 'teacher_assets': teacher_guard.expected}
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Resume only the exact preserved pilot contract; otherwise use a fresh output')
        if (root / 'manifest.json').exists():
            return checked_proof(root / 'manifest.json', 'clean_recorded_search_yield_pilot', bound,
                                 arguments=contract, code=contract['code'])['verification']
    else:
        root.mkdir(parents=True)
        atomic_json(root / 'contract.json', contract)
    stage = 'waiting_for_completed_clean_sft_validation'

    def unchanged():
        bound.unchanged()
        teacher_guard.unchanged()
        if code_identity() != contract['code']:
            raise ValueError('Preserved search pilot execution source changed')

    def execute(module, arguments):
        command = [sys.executable, '-u', '-m', 'xqgeneral.' + module]
        for key, value in arguments.items():
            if value is not None:
                command += ['--' + key.replace('_', '-'), str(value)]
        with (root / 'commands.jsonl').open('a') as handle:
            handle.write(json.dumps(command) + '\n')
        with (root / 'log.txt').open('a') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=config['cuda_visible_devices']))

    def completed_stage(module, arguments, kind, destination, resumable=False):
        path = destination / 'manifest.json'
        if not path.exists():
            if destination.exists() and not resumable:
                raise ValueError('Preserve incomplete collection; use a fresh pilot output')
            execute(module, arguments)
        return checked_proof(path, kind, bound, arguments=arguments, code=contract['code'])

    try:
        validation_path = wait_for_validation(validation, producer_session, root, poll_seconds)
        unchanged()
        capability = checked_proof(validation_path, 'clean_initial_sft_capability_validation', bound)
        result = capability['verification']
        if (capability['config']['pipeline'] != str(pipeline) or
                capability['config']['config'] != config['evaluation_config'] or
                capability['config']['config_sha256'] != digest(config['evaluation_config']) or
                result.get('actual_clean_initial_sft_completed_before_model_or_reviewer_load') is not True or
                result.get('split') != 'validation' or result.get('all_validation_examples') != evaluation['validation_examples'] or
                result.get('independent_test_used_for_training_or_selection') is not False or
                result.get('raw_answers_repaired') is not False or
                set(result.get('raw_validation_by_memory', {})) != {'normal', 'zero', 'shuffled'}):
            raise ValueError('Complete paired validation of the exact clean SFT must precede search')
        source, parent = checked_sft(pipeline, evaluation, recipe)
        if capability['inputs'].get(str(source)) != parent['outputs'][str(source)]:
            raise ValueError('The search parent differs from the model actually evaluated')
        bound.bind(source, parent['outputs'][str(source)])
        for name in ARTIFACTS:
            path = Path(config['recorded_inputs']) / name
            bound.bind(path, pool['outputs'].get(str(path)))
        for path, expected in teacher_guard.expected.items():
            bound.bind(path, expected)
        for section in ['inputs', 'outputs']:
            for path, expected in token[section].items():
                bound.bind(path, expected)
        unchanged()

        stage = 'student_recursive_search'
        atomic_json(root / 'state.json', {'status': stage})
        mining = root / 'mining'
        arguments = {'checkpoint': str(source), 'data': config['data'], 'features': evaluation['features'],
            'prepared_inputs': None, 'recorded_inputs': config['recorded_inputs'], 'output': str(mining),
            **{key: config[key] for key in ['limit', 'nodes', 'max_depth', 'child_contract', 'seed']},
            **{key: evaluation[key] for key in ['executable', 'weights']}}
        mined = completed_stage('search_distillation', arguments, 'search_distillation_mining', mining, True)
        queries = load_jsonl(mining / 'queries.jsonl')
        attempts = load_jsonl(mining / 'results.partial.jsonl')
        actual_queries = [row['query'] for row in attempts if row['query'] is not None]
        if (len(attempts) != config['limit'] or len({row['source_root_id'] for row in attempts}) != len(attempts) or
                actual_queries != queries or mined['verification']['processed'] != len(attempts) or
                mined['verification']['accepted_for_consolidation'] != len(queries) or
                mined['verification']['reasons'] != dict(Counter(row['reason'] for row in attempts)) or
                mined['verification'].get('unused_recorded_training_inputs') is not True):
            raise ValueError('Actual mining coverage and yield differ from the preserved raw attempts')
        if any(mined['verification'].get(key) is not True for key in [
                'training_roots_only', 'heldout_positions_excluded', 'all_child_branch_positions_isolated',
                'raw_child_reachable_prefixes_isolated', 'all_inferred_target_lines_history_validated',
                'strict_pv_improvement_required', 'all_successful_oracle_queries_preserved']):
            raise ValueError('The actual miner must preserve its complete search and isolation contract')
        proofs = [validation_path, Path(pipeline) / 'manifest.json', mining / 'manifest.json']
        labels = []
        rejections = []
        if queries:
            stage = 'full_bf16_consolidation'
            atomic_json(root / 'state.json', {'status': stage, 'mined_queries': len(queries)})
            consolidation = root / 'consolidation'
            args = {'config': evaluation['teacher_config'], 'input': str(mining / 'queries.jsonl'),
                    'output': str(consolidation), 'max_new_tokens': config['teacher_max_new_tokens']}
            if config.get('teacher_batch_size', 1) != 1:
                args['batch_size'] = config['teacher_batch_size']
            teacher_proof = completed_stage('local_teacher', args, 'local_teacher_inference', consolidation, True)
            responses = load_jsonl(consolidation / 'responses.jsonl')
            if (len(responses) != len(queries) or len({row['id'] for row in responses}) != len(responses) or
                    {row['id'] for row in responses} != {row['id'] for row in queries} or
                    teacher_proof['verification'].get('neural_teacher_inference_executed') is not True):
                raise ValueError('Actual BF16 responses must cover every original mined query')
            unchanged()
            stage = 'isolated_label_collection'
            atomic_json(root / 'state.json', {'status': stage})
            data = root / 'labels'
            args = {'queries': str(mining / 'queries.jsonl'), 'responses': str(consolidation / 'responses.jsonl'),
                    'config': evaluation['teacher_config'], 'heldout_data': config['data'],
                    'validation_data': evaluation['data'], 'mining_manifest': str(mining / 'manifest.json'),
                    'recorded_inputs': config['recorded_inputs'], 'output': str(data)}
            collected = completed_stage('collect_search', args, 'search_consolidated_dataset', data)
            labels = load_jsonl(data / 'train.jsonl')
            rejections = json.loads((data / 'rejected.json').read_text())
            if (len(labels) != collected['verification']['accepted_training_roots'] or
                    len(labels) + len(rejections) != len(queries) or
                    collected['verification'].get('recorded_mining_isolation_rechecked_during_collection') is not True):
                raise ValueError('Final label counts and complete recorded isolation differ')
            proofs += [consolidation / 'manifest.json', data / 'manifest.json']
        unchanged()
        final = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
            'clean_sft_and_paired_validation_completed_before_search': True,
            'actual_selected_checkpoint': str(source), 'actual_selected_step': parent['verification']['selected_sft_step'],
            'attempted_original_training_roots': len(attempts), 'mined_queries': len(queries),
            'mining_reasons': mined['verification']['reasons'], 'accepted_structured_training_labels': len(labels),
            'rejected_consolidations': len(rejections), 'mining_yield': len(queries) / len(attempts),
            'final_structured_label_yield': len(labels) / len(attempts),
            'neural_consolidation_executed': bool(queries), 'empty_distillation_training_started': False,
            'strategic_prose_fully_verified': False, 'trained_search_distillation_rounds': 0,
            'independent_test_used_for_training_or_selection': False,
            'strong_play_or_reliable_coaching_proven': False,
            'next_action': 'Evaluate accepted prose and scale useful sources before creating a distillation training run'
                           if labels else 'Inspect preserved raw failures before spending more teacher or training budget'}
        atomic_json(root / 'verification.json', final)
        completed = manifest('clean_recorded_search_yield_pilot', contract, proofs,
                             [root / 'verification.json'], final)
        completed['inputs'].update(bound.artifacts)
        for path in proofs:
            completed['outputs'].update(json.loads(path.read_text())['outputs'])
        atomic_json(root / 'manifest.json', completed)
        atomic_json(root / 'state.json', final)
        return final
    except Exception as error:
        atomic_json(root / 'state.json', {'status': 'failed', 'stage': stage,
            'error_type': type(error).__name__, 'preserve_partial_outputs': True,
            'distillation_training_started': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--producer-session', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.pipeline, args.validation, args.producer_session, args.config,
        args.output, resume=args.resume, poll_seconds=args.poll_seconds)), flush=True)


if __name__ == '__main__':
    main()
