"""Train four courses sequentially, inheriting each course's best raw-validation weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from .curriculum_data import STAGES
from .curriculum_selection import (COURSE_KIND, CURRICULUM_KIND, course_contract, read_history,
                                   selection_decision, validate_course_selection, validate_policy)
from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest
from .foundation_handoff import Artifacts
from .foundation_preflight import clean_recipe
from .gated_curriculum import raw_gate_from_artifacts
from .modeling import foundation_state_summary
from .select_checkpoint import snapshot_checkpoint


def verify_pending(artifacts):
    """Freshly hash new immutable files once; guard all previously hashed file identities."""
    artifacts.unchanged()
    for name, expected in artifacts.expected.items():
        if name in artifacts.stats:
            continue
        before = artifacts.signature(name)
        if before[2] != expected['bytes'] or digest(name) != expected['sha256']:
            raise ValueError(f'Changed selection evidence: {name}')
        if artifacts.signature(name) != before:
            raise ValueError('Selection evidence changed while hashing')
        artifacts.stats[name] = before


def evaluate_candidate(candidate, checkpoint, stages, recipe):
    evaluation = candidate / 'qa'
    gates = recipe['raw_qa_gates']
    if not (evaluation / 'manifest.json').exists():
        command = [sys.executable, '-u', '-m', 'xqgeneral.evaluate_qa', '--checkpoint', str(checkpoint),
            '--data', recipe['data_path'], '--features', recipe['feature_path'], '--split', 'validation',
            '--memory', 'normal', '--per-task', str(gates['per_task']),
            '--batch-size', str(gates.get('batch_size', 8)), '--max-new-tokens', str(gates.get('max_new_tokens', 384)),
            '--question-formats', str(gates['question_formats']), '--seed', str(gates['seed']),
            '--stages', *stages, '--output', str(evaluation)]
        with (candidate.parent.parent / 'qa.log').open('a') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    proof = json.loads((evaluation / 'manifest.json').read_text())
    gate = raw_gate_from_artifacts(proof, json.loads((evaluation / 'metrics.json').read_text()),
        load_jsonl(evaluation / 'predictions.jsonl'), checkpoint, stages, gates,
        recipe['data_path'], recipe['feature_path'])
    path = candidate / 'gate.json'
    if path.exists() and json.loads(path.read_text()) != gate:
        raise ValueError('Preserve a changed raw gate; do not overwrite it')
    if not path.exists():
        atomic_json(path, gate)


def run_curriculum(recipe_path, *, resume=False, prepare_only=False):
    recipe_path = Path(recipe_path)
    recipe = json.loads(recipe_path.read_text())
    clean_recipe(recipe)
    validate_policy(recipe['validation_selection'])
    root = Path(recipe['output'])
    code = code_identity()
    contract = {'recipe': str(recipe_path), 'recipe_sha256': digest(recipe_path), 'code': code}
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Resume only the exact preserved selection contract; otherwise use a fresh output')
    else:
        root.mkdir(parents=True)
        atomic_json(root / 'contract.json', contract)
    artifacts = Artifacts()
    artifacts.json(recipe_path)
    parent, completed = None, []
    for index, stage in enumerate(STAGES):
        stage_root = root / stage
        expected, candidates, training_code = course_contract(recipe, index, root, parent, artifacts, code)
        config_path = stage_root / 'training.config.json'
        if config_path.exists() and artifacts.json(config_path) != expected:
            raise ValueError('Changed course configuration requires a fresh output directory')
        if not config_path.exists():
            atomic_json(config_path, expected)
            artifacts.json(config_path)
        stage_manifest = stage_root / 'manifest.json'
        previous = artifacts.proof(stage_manifest, COURSE_KIND) if stage_manifest.exists() else None
        imported = recipe.get('initial_course_import') if index == 0 else None
        if previous:
            step = previous['verification']['actual_steps']
        elif imported:
            step = imported['through_step']
        else:
            latest = Path(expected['output']) / 'latest.pt'
            step = 0
            if latest.exists():
                saved = torch.load(latest, map_location='cpu', weights_only=True, mmap=True)
                if (saved['config'] != expected or saved['world_size'] != recipe['ddp_world_size'] or
                        saved['code'] != code):
                    raise ValueError('Course resume differs from its frozen configuration, source or world size')
                step = saved['step']
                del saved
        while True:
            if step:
                if not previous and not imported:
                    candidate, actual = snapshot_checkpoint(latest, candidates)
                    if actual != step:
                        raise ValueError('Evaluate the actual latest update')
                    evaluate_candidate(candidate, candidate / 'adapter.pt', STAGES[:index + 1], recipe)
                history = read_history(candidates, step, expected, training_code, STAGES[:index + 1], recipe, artifacts)
                decision = selection_decision(history, recipe['validation_selection'], recipe['course_budgets'][index])
                verify_pending(artifacts)
                atomic_json(stage_root / 'selection-progress.json', {'stage': stage, 'decision': decision,
                    'reference_targets_are_required': False, 'history': history})
                print(json.dumps({'event': 'raw_validation_selection', 'stage': stage, **decision}), flush=True)
                if previous or imported or decision['stop']:
                    if not decision['stop']:
                        raise ValueError('Imported or completed course has not met its declared stopping policy')
                    chosen = next(row for row in history if row['step'] == decision['selected_step'])
                    checkpoint = Path(chosen['checkpoint'])
                    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
                    state = foundation_state_summary(saved, artifacts.json(Path(recipe['model_path']) / 'config.json'))
                    del saved
                    verify_pending(artifacts)
                    result = {'stage': stage, 'actual_steps': step, 'selected_step': chosen['step'],
                        'global_batch_size': recipe['batch_size'], 'sample_presentations': step * recipe['batch_size'],
                        'gate': chosen['gate'], 'reference_targets_passed': chosen['gate']['passed'],
                        'selected_checkpoint': str(checkpoint), 'selected_checkpoint_sha256': chosen['checkpoint_sha256'],
                        'selection': {'policy': recipe['validation_selection'], 'budget': recipe['course_budgets'][index],
                                      'decision': decision, 'history': history},
                        'imported_original_clean_first_course': bool(imported),
                        'candidate_root': str(candidates), 'training_code': training_code,
                        'selected_checkpoint_state_checks': state, 'next_course_permitted': True,
                        'optimizer_reset_between_courses': True, 'independent_test_used': False}
                    validate_course_selection(result)
                    if previous:
                        if previous['config'] != expected or previous['code'] != code or previous['verification'] != result:
                            raise ValueError('Completed course differs from its recomputed best-validation selection')
                    else:
                        proof = manifest(COURSE_KIND, expected, outputs=[checkpoint], verification=result, code=code)
                        proof['inputs'] = dict(artifacts.expected)
                        atomic_json(stage_manifest, proof)
                        artifacts.proof(stage_manifest, COURSE_KIND)
                        verify_pending(artifacts)
                    parent = checkpoint
                    completed.append(result)
                    atomic_json(root / 'state.json', {'status': 'course_completed', 'courses_completed': len(completed),
                        'stage': stage, 'actual_steps': step, 'selected_step': chosen['step'],
                        'selected_accuracy': chosen['gate']['accuracy'],
                        'reference_targets_passed': chosen['gate']['passed'], 'next_course_permitted': True,
                        'gpu_sft_started': False, 'independent_test_used': False})
                    break
            if imported or previous:
                raise ValueError('The declared imported/completed course is missing actual raw validation')
            if prepare_only:
                atomic_json(root / 'state.json', {'status': 'prepared_for_next_course', 'courses_completed': len(completed),
                    'next_stage': stage, 'selected_parent': str(parent) if parent else None,
                    'new_course_gpu_training_started': False, 'gpu_sft_started': False})
                return completed
            verify_pending(artifacts)
            atomic_json(root / 'state.json', {'status': 'training_course', 'stage': stage,
                'courses_completed': len(completed), 'start_step': step, 'selected_parent': str(parent) if parent else None,
                'gpu_sft_started': False})
            command = [sys.executable, '-u']
            world = recipe['ddp_world_size']
            if world > 1:
                command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                            f'--nproc_per_node={world}', '--module', 'xqgeneral.train']
            else:
                command += ['-m', 'xqgeneral.train']
            command += ['--config', str(config_path), '--stop-after',
                        str(min(step + recipe['course_budgets'][index]['qa_every'], recipe['course_budgets'][index]['steps']))]
            latest = Path(expected['output']) / 'latest.pt'
            if latest.exists():
                command += ['--resume', str(latest)]
            with (stage_root / 'training.log').open('a') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            saved = torch.load(latest, map_location='cpu', weights_only=True, mmap=True)
            if saved['config'] != expected or saved['code'] != code or saved['world_size'] != world:
                raise ValueError('Actual course training differs from its frozen contract')
            step = saved['step']
            del saved
    verification = {'courses': completed, 'all_courses_completed_by_declared_validation_policy': True,
        'all_raw_qa_gates_passed': all(course['gate']['passed'] for course in completed),
        'reference_targets_required_for_progression': False, 'independent_test_used': False}
    atomic_json(root / 'manifest.json', manifest(CURRICULUM_KIND, recipe,
        [recipe_path, *[root / stage / 'manifest.json' for stage in STAGES]], [parent], verification, code=code))
    atomic_json(root / 'state.json', dict(verification, status='four_courses_completed', gpu_sft_started=False))
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Verify inherited courses and prepare the next config without GPU training')
    args = parser.parse_args()
    run_curriculum(args.config, resume=args.resume, prepare_only=args.prepare_only)


if __name__ == '__main__':
    main()
