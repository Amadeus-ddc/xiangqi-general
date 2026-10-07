"""Sequential foundation training governed by raw, task-balanced validation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import torch

from .curriculum_data import STAGES
from .evaluate_qa import balanced_rows, normalized, qa_summary
from .evidence import atomic_json, digest, load_jsonl, manifest
from .select_checkpoint import snapshot_checkpoint
from .selfplay_grounding import question_variant


TASKS = {
    'static_current': ('piece', 'count', 'locate', 'empty', 'material', 'rank'),
    'dynamic_current': ('legal', 'illegal', 'moves', 'captures', 'checks'),
    'static_future': ('piece', 'count', 'locate', 'empty', 'material', 'rank'),
    'dynamic_future': ('legal', 'illegal', 'moves', 'captures', 'checks'),
}
CONTROL_KEYS = {'course_mixtures', 'course_budgets', 'raw_qa_gates', 'ddp_world_size', 'output', 'purpose'}


def validate_recipe(recipe):
    if len(recipe['course_mixtures']) != len(STAGES) or len(recipe['course_budgets']) != len(STAGES):
        raise ValueError('Exactly four ordered foundation courses are required')
    world = recipe.get('ddp_world_size', 1)
    if type(recipe.get('ddp_find_unused_parameters', False)) is not bool:
        raise ValueError('ddp_find_unused_parameters must be a boolean')
    if type(recipe.get('feature_cache_mmap', False)) is not bool:
        raise ValueError('feature_cache_mmap must be a boolean')
    if (type(world) is not int or world < 1 or type(recipe['batch_size']) is not int or
            recipe['batch_size'] < 1 or recipe['batch_size'] % world):
        raise ValueError('Global batch size must be divisible by the positive DDP world size')
    if recipe.get('mode', 'bridge') != 'bridge' or recipe.get('decoder_training', 'frozen') != 'frozen':
        raise ValueError('Foundation courses train the bridge with a frozen decoder')
    gates = recipe['raw_qa_gates']
    if type(gates['per_task']) is not int or gates['per_task'] < 1:
        raise ValueError('A positive per-task raw validation sample is required')
    if gates['question_formats'] not in (1, 3):
        raise ValueError('Raw QA must use the supported question formats')
    for index, stage in enumerate(STAGES):
        mixture, budget = recipe['course_mixtures'][index], recipe['course_budgets'][index]
        if (stage not in mixture or set(mixture) - set(STAGES[:index + 1]) or
                any(weight <= 0 for weight in mixture.values()) or abs(sum(mixture.values()) - 1) > 1e-9):
            raise ValueError('Course replay must include the new course and only introduced courses')
        if (any(type(budget[k]) is not int or budget[k] < 1 for k in ['steps', 'minimum_steps', 'qa_every']) or
                budget['minimum_steps'] > budget['steps'] or budget['learning_rate'] <= 0):
            raise ValueError('Invalid foundation course budget')
        target = gates['targets'][stage]
        if any(not 0 < target[k] <= 1 for k in ['accuracy', 'minimum_task_accuracy']):
            raise ValueError('Raw QA targets must be between zero and one')


def raw_qa_gate(records, stages, targets, per_task):
    """No aggregate score can hide a missing task or a broken capture course."""
    expected = {f'{stage}/{task}' for stage in stages for task in TASKS[stage]}
    groups = Counter(f"{r['stage']}/{r['task_type']}" for r in records)
    if set(groups) != expected or any(n != per_task for n in groups.values()):
        raise ValueError('Raw validation must cover every introduced task with the declared count')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Raw validation contains duplicate example identities')
    if any(r['correct'] != (normalized(r['generated']) == normalized(r['expected'])) for r in records):
        raise ValueError('Stored raw correctness differs from the actual generated answer')
    summary, courses = qa_summary(records), {}
    for stage in stages:
        selected = [r for r in records if r['stage'] == stage]
        accuracy = sum(r['correct'] for r in selected) / len(selected)
        minimum = min(summary['by_task'][f'{stage}/{task}']['accuracy'] for task in TASKS[stage])
        target = targets[stage]
        courses[stage] = {'examples': len(selected), 'accuracy': accuracy, 'minimum_task_accuracy': minimum,
                          'passed': accuracy >= target['accuracy'] and minimum >= target['minimum_task_accuracy']}
    return dict(summary, courses=courses, passed=all(c['passed'] for c in courses.values()),
                validation_only=True, raw_generation=True, oracle_used=False)


def read_gate(output, checkpoint, stages, gates, data_path=None, feature_path=None):
    output = Path(output)
    proof = json.loads((output / 'manifest.json').read_text())
    config, metrics = proof['config'], json.loads((output / 'metrics.json').read_text())
    if (proof['status'] != 'complete' or config['checkpoint'] != str(checkpoint) or
            config['split'] != 'validation' or config['memory'] != 'normal' or config['stages'] != list(stages) or
            config['per_task'] != gates['per_task'] or config['question_formats'] != gates['question_formats'] or
            config['seed'] != gates['seed'] or str(checkpoint) not in proof['inputs'] or
            not metrics.get('raw_generation') or metrics.get('oracle_used') is not False):
        raise ValueError('A gate requires unchanged, normal-memory, raw validation evidence')
    if ((data_path is not None and config['data'] != data_path) or
            (feature_path is not None and config['features'] != feature_path)):
        raise ValueError('Raw validation must use the configured dataset and feature cache')
    for path, item in {**proof['inputs'], **proof['outputs']}.items():
        if digest(path) != item['sha256']:
            raise ValueError(f'Changed raw validation evidence: {path}')
    records = load_jsonl(output / 'predictions.jsonl')
    if data_path is not None:
        expected = balanced_rows(load_jsonl(Path(data_path) / 'validation.jsonl'),
                                 gates['per_task'], gates['seed'], stages)
        counts = Counter()
        if len(expected) != len(records):
            raise ValueError('Raw validation predictions differ from the declared sample')
        for row, prediction in zip(expected, records, strict=True):
            key = row['stage'], row['task_type']
            variant = counts[key] % 3 if gates['question_formats'] == 3 else row.get('question_variant', 0)
            counts[key] += 1
            question = question_variant(row, variant) if gates['question_formats'] == 3 else row['question']
            if (any(row[k] != prediction[k] for k in ['id', 'game_id', 'stage', 'task_type']) or
                    row['answer'] != prediction['expected'] or question != prediction['question'] or
                    variant != prediction['question_variant']):
                raise ValueError('Raw validation questions or native expected answers changed')
    result = raw_qa_gate(records, stages, gates['targets'], gates['per_task'])
    if metrics['by_task'] != result['by_task'] or metrics['accuracy'] != result['accuracy']:
        raise ValueError('Raw predictions disagree with the evaluation summary')
    return result


def training_config(recipe, index, root, parent):
    budget, stage = recipe['course_budgets'][index], STAGES[index]
    config = {k: v for k, v in recipe.items() if k not in CONTROL_KEYS}
    config.update(stages=[stage], mixture=recipe['course_mixtures'][index],
                  validation_stages=list(STAGES[:index + 1]), steps=budget['steps'],
                  min_steps=budget['steps'], patience=budget['steps'] + 1,
                  learning_rate=budget['learning_rate'], token_learning_rate=budget['learning_rate'] / 10,
                  output=str(root / stage / 'training'))
    if parent:
        config['init_from'] = str(parent)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    recipe = json.loads(Path(args.config).read_text())
    validate_recipe(recipe)
    root, gates = Path(recipe['output']), recipe['raw_qa_gates']
    parent, stages_done = recipe.get('init_from'), []
    for index, stage in enumerate(STAGES):
        stage_root = root / stage
        config = training_config(recipe, index, root, parent)
        config_path = stage_root / 'training.config.json'
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Changed course configuration requires a fresh output directory')
        else:
            atomic_json(config_path, config)
        contract_path = stage_root / 'manifest.json'
        if contract_path.exists():
            proof = json.loads(contract_path.read_text())
            for path, item in {**proof['inputs'], **proof['outputs']}.items():
                if digest(path) != item['sha256']:
                    raise ValueError(f'Changed completed course evidence: {path}')
            result = proof['verification']
            selected = Path(result['selected_checkpoint'])
            verified = read_gate(selected.parent / 'qa', selected, STAGES[:index + 1], gates,
                                 recipe['data_path'], recipe['feature_path'])
            if verified != result['gate']:
                raise ValueError('Completed course gate differs from its raw validation artifacts')
            if not result['gate']['passed']:
                raise RuntimeError('Course budget exhausted without raw QA mastery; later courses remain unstarted')
            parent = result['selected_checkpoint']
            stages_done.append(result)
            continue
        budget = recipe['course_budgets'][index]
        latest = Path(config['output']) / 'latest.pt'
        while True:
            step = 0
            if latest.exists():
                saved = torch.load(latest, map_location='cpu', weights_only=True, mmap=True)
                if saved['config'] != config or saved['world_size'] != recipe.get('ddp_world_size', 1):
                    raise ValueError('Course resume checkpoint differs from its frozen configuration')
                step = saved['step']
                del saved
            if step:
                candidate, actual_step = snapshot_checkpoint(latest, stage_root / 'candidates')
                if actual_step != step:
                    raise ValueError('Raw QA must assess the actual latest update')
                checkpoint, evaluation = candidate / 'adapter.pt', candidate / 'qa'
                introduced = STAGES[:index + 1]
                if not (evaluation / 'manifest.json').exists():
                    command = [sys.executable, '-u', '-m', 'xqgeneral.evaluate_qa', '--checkpoint', str(checkpoint),
                        '--data', recipe['data_path'], '--features', recipe['feature_path'], '--split', 'validation',
                        '--memory', 'normal', '--per-task', str(gates['per_task']),
                        '--batch-size', str(gates.get('batch_size', 8)),
                        '--max-new-tokens', str(gates.get('max_new_tokens', 384)),
                        '--question-formats', str(gates['question_formats']), '--seed', str(gates['seed']),
                        '--stages', *introduced, '--output', str(evaluation)]
                    with (stage_root / 'qa.log').open('a') as handle:
                        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
                gate = read_gate(evaluation, checkpoint, introduced, gates,
                                 recipe['data_path'], recipe['feature_path'])
                atomic_json(candidate / 'gate.json', gate)
                print(json.dumps({'event': 'raw_course_gate', 'stage': stage, 'step': step, **gate}), flush=True)
                if (gate['passed'] and step >= budget['minimum_steps']) or step >= budget['steps']:
                    result = {'stage': stage, 'actual_steps': step, 'global_batch_size': config['batch_size'],
                        'sample_presentations': step * config['batch_size'], 'gate': gate,
                        'selected_checkpoint': str(checkpoint), 'selected_checkpoint_sha256': digest(checkpoint),
                        'next_course_permitted': gate['passed'], 'optimizer_reset_between_courses': True,
                        'independent_test_used': False}
                    atomic_json(contract_path, manifest('raw_qa_gated_foundation_course', config,
                        [args.config, config_path, candidate / 'snapshot.json', evaluation / 'manifest.json'],
                        [checkpoint, candidate / 'gate.json'], result))
                    if not gate['passed']:
                        raise RuntimeError('Course budget exhausted without raw QA mastery; later courses remain unstarted')
                    parent = checkpoint
                    stages_done.append(result)
                    break
            command = [sys.executable, '-u']
            world = recipe.get('ddp_world_size', 1)
            if world > 1:
                command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                            f'--nproc_per_node={world}', '--module', 'xqgeneral.train']
            else:
                command += ['-m', 'xqgeneral.train']
            command += ['--config', str(config_path), '--stop-after', str(min(step + budget['qa_every'], budget['steps']))]
            if latest.exists():
                command += ['--resume', str(latest)]
            with (stage_root / 'training.log').open('a') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    atomic_json(root / 'manifest.json', manifest('four_course_raw_qa_gated_curriculum', recipe,
        [args.config, *[root / stage / 'manifest.json' for stage in STAGES]], [parent],
        {'courses': stages_done, 'all_raw_qa_gates_passed': True, 'independent_test_used': False}))


if __name__ == '__main__':
    main()
