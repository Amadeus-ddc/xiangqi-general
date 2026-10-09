"""Run a protected full-match baseline after clean SFT and paired validation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .clean_sft_evaluation import checked_sft, validation_spec
from .evaluate_games import match_summary, restore_game
from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest
from .match_openings import load_openings
from .recorded_search_inputs import BoundInputs
from .search_pilot import checked_proof, wait_for_validation


def baseline_spec(config, bound):
    nodes = config['nodes']
    if (not isinstance(nodes, list) or not nodes or
            any(type(n) is not int or n <= 0 for n in nodes) or len(set(nodes)) != len(nodes) or
            any(type(config[k]) is not int or config[k] <= 0
                for k in ['opening_count', 'max_plies', 'move_beams']) or
            config['action_mode'] != 'explanation' or
            not isinstance(config['cuda_visible_devices'], str) or
            re.fullmatch(r'\d+', config['cuda_visible_devices']) is None):
        raise ValueError('Raw explanation matches require positive, distinct budgets and one declared device')
    evaluation = bound.json(config['evaluation_config'])
    _, token, recipe = validation_spec(evaluation)
    bound.bind(evaluation['sft_recipe'])
    bound.bind(evaluation['token_preflight'])
    for section in ['inputs', 'outputs']:
        for path, expected in token[section].items():bound.bind(path, expected)
    openings, proof = load_openings(config['opening_suite'])
    if (proof['config']['split'] != 'validation' or len(openings) != config['opening_count'] or
            bound.json(Path(config['opening_suite']) / 'manifest.json') != proof):
        raise ValueError('The baseline requires its complete protected validation opening suite')
    for section in ['inputs', 'outputs']:
        for path, expected in proof[section].items():bound.bind(path, expected)
    for key in ['executable', 'weights', 'expert_weights']:bound.bind(config[key])
    return evaluation, recipe, openings


def execute_matches(command, log, device):
    with Path(log).open('a') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=device))


def checked_matches(root, arguments, openings, bound, code):
    proof = checked_proof(Path(root) / 'manifest.json', 'raw_model_full_matches', bound,
                          arguments=arguments, code=code)
    games = load_jsonl(Path(root) / 'games.jsonl')
    schedule = [(opening, color, n) for n in arguments['nodes'] for opening in openings
                for color in ['red', 'black']]
    if len(games) != len(schedule):raise ValueError('Full matches must cover every opening, color and node budget')
    required = {str(Path(root) / name) for name in ['games.jsonl', 'metrics.json', 'contract.json']}
    required.update(str(Path(root) / f'game-{i:03d}.json') for i in range(1, len(schedule) + 1))
    required.update(str(Path(root) / 'progress' / f'game-{i:03d}.json') for i in range(1, len(schedule) + 1))
    if set(proof['outputs']) != required:raise ValueError('Baseline must bind every raw game and turn checkpoint')
    for game, (opening, color, nodes) in zip(games, schedule):
        checked = restore_game(opening, color, nodes, arguments['max_plies'], game,
                               arguments['action_mode'], arguments['move_beams'])
        if checked['status'] == 'playing':raise ValueError('A saved partial game is not a completed baseline')
    metrics = proof['verification']
    if (any(metrics.get(k) != value for k, value in match_summary(games).items()) or
            metrics.get('opening_split') != 'validation' or metrics.get('openings') != len(openings) or
            metrics.get('action_mode') != 'explanation' or metrics.get('rule_legal_constraints') is not False or
            metrics.get('by_opponent_nodes') != {str(n): match_summary([g for g in games if g['opponent_nodes'] == n])
                                                 for n in arguments['nodes']} or
            metrics.get('both_model_colors_at_each_opening_and_budget') is not True):
        raise ValueError('Baseline metrics differ from its raw full-history matches')
    return proof, games


def run_baseline(pipeline, validation, producer_session, config_path, output, *, resume=False, poll_seconds=30):
    root, config_path = Path(output), Path(config_path)
    bound = BoundInputs()
    config = bound.json(config_path)
    evaluation, recipe, openings = baseline_spec(config, bound)
    contract = {'pipeline': str(pipeline), 'validation': str(validation), 'producer_session': producer_session,
                'config': str(config_path), 'config_sha256': digest(config_path), 'code': code_identity(),
                'prerequisite_inputs': dict(bound.artifacts)}
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Resume only the original baseline contract and frozen source')
        if (root / 'manifest.json').exists():
            return checked_proof(root / 'manifest.json', 'clean_initial_sft_full_match_baseline', bound,
                                 arguments=contract, code=contract['code'])['verification']
    else:
        root.mkdir(parents=True)
        atomic_json(root / 'contract.json', contract)
    stage = 'waiting_for_completed_clean_sft_validation'

    def unchanged():
        bound.unchanged()
        if code_identity() != contract['code']:raise ValueError('Preserved baseline execution source changed')

    try:
        path = wait_for_validation(validation, producer_session, root, poll_seconds)
        unchanged()
        capability = checked_proof(path, 'clean_initial_sft_capability_validation', bound)
        result = capability['verification']
        if (capability['config']['pipeline'] != str(pipeline) or
                capability['config']['config'] != config['evaluation_config'] or
                capability['config']['config_sha256'] != digest(config['evaluation_config']) or
                result.get('actual_clean_initial_sft_completed_before_model_or_reviewer_load') is not True or
                result.get('split') != 'validation' or result.get('all_validation_examples') != evaluation['validation_examples'] or
                result.get('independent_test_used_for_training_or_selection') is not False or
                result.get('raw_answers_repaired') is not False or
                set(result.get('raw_validation_by_memory', {})) != {'normal', 'zero', 'shuffled'}):
            raise ValueError('Complete paired validation of the exact clean SFT must precede matches')
        stage = 'checking_selected_clean_sft'
        source, parent = checked_sft(pipeline, evaluation, recipe)
        if capability['inputs'].get(str(source)) != parent['outputs'][str(source)]:
            raise ValueError('The match student differs from the selected model actually evaluated')
        bound.bind(source, parent['outputs'][str(source)])
        bound.bind(Path(pipeline) / 'manifest.json')
        stage = 'running_clean_sft_full_matches'
        atomic_json(root / 'state.json', {'status': stage, 'selected_checkpoint': str(source),
            'paired_validation_completed': True, 'planned_games': len(openings) * 2 * len(config['nodes']),
            'cuda_visible_devices': config['cuda_visible_devices']})
        destination = root / 'matches'
        arguments = {'checkpoint': str(source), 'nodes': config['nodes'], 'max_plies': config['max_plies'],
            'action_mode': config['action_mode'], 'move_beams': config['move_beams'],
            'opening_suite': config['opening_suite'], 'expert_weights': config['expert_weights'],
            'executable': config['executable'], 'weights': config['weights'], 'output': str(destination)}
        command = [sys.executable, '-u', '-m', 'xqgeneral.evaluate_games']
        for key, value in arguments.items():
            command += ['--' + key.replace('_', '-')]
            command += [str(v) for v in value] if isinstance(value, list) else [str(value)]
        if destination.exists():command.append('--resume')
        with (root / 'commands.jsonl').open('a') as handle:handle.write(json.dumps(command) + '\n')
        unchanged()
        execute_matches(command, root / 'log.txt', config['cuda_visible_devices'])
        unchanged()
        child, games = checked_matches(destination, arguments, openings, bound, contract['code'])
        final = {'status': 'complete', 'evidence_state': 'reconstructed_baseline', 'split': 'validation',
            'clean_sft_and_paired_validation_completed_before_matches': True,
            'actual_selected_checkpoint': str(source), 'actual_selected_step': parent['verification']['selected_sft_step'],
            'opening_count': len(openings), 'games': len(games), 'node_budgets': config['nodes'],
            'maximum_new_plies': config['max_plies'], 'raw_match_metrics': child['verification'],
            'model_turns': sum('raw' in t for g in games for t in g['turns']),
            'full_explanation_contract_valid_turns': sum(t.get('verification', {}).get('valid') is True
                                                       for g in games for t in g['turns']),
            'raw_model_answers_repaired': False, 'rule_legal_action_constraints_used': False,
            'independent_final_test_used': False, 'strong_student_or_reliable_coach_proven': False}
        atomic_json(root / 'verification.json', final)
        completed = manifest('clean_initial_sft_full_match_baseline', contract, (),
                             [root / 'verification.json', root / 'commands.jsonl', destination / 'manifest.json'],
                             final, code=contract['code'])
        completed['inputs'].update(bound.artifacts)
        completed['outputs'].update(child['outputs'])
        unchanged()
        atomic_json(root / 'manifest.json', completed)
        atomic_json(root / 'state.json', final)
        return final
    except Exception as error:
        atomic_json(root / 'state.json', {'status': 'failed', 'stage': stage, 'error_type': type(error).__name__,
            'partial_match_records_preserved': True, 'baseline_completion_proven': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--producer-session', required=True)
    parser.add_argument('--config', default='configs/matches-clean-sft-baseline-v1.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(run_baseline(args.pipeline, args.validation, args.producer_session, args.config, args.output,
                                 resume=args.resume, poll_seconds=args.poll_seconds)), flush=True)


if __name__ == '__main__':main()
