"""Recheck all four validation courses after completed initial or distilled SFT."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .clean_sft_evaluation import checked_sft, validation_spec
from .course_tasks import PAPER_PROFILE, row_profile, task_groups
from .curriculum_data import STAGES
from .curriculum_selection import CURRICULUM_KIND, USER_CURRICULUM_KIND
from .distillation_preflight import ROUND_KIND, TOKEN_KIND
from .evaluate_qa import balanced_rows, prediction_record, qa_summary
from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest
from .feature_store import feature_paths
from .foundation_handoff import Artifacts
from .selfplay_grounding import question_variant
from .verified_inputs import input_session_from_environment


KIND = 'completed_sft_four_course_validation'
MEMORIES = ['normal', 'zero', 'shuffled']
QUESTION_FIELDS = ['id', 'game_id', 'stage', 'task_type', 'question', 'question_variant', 'expected']


def selected_course_rows(config):
    """Use the existing evaluator's sample and question rendering, on validation only."""
    if config['split'] != 'validation' or config['stages'] != list(STAGES) or config['memories'] != MEMORIES:
        raise ValueError('Course replay requires all four validation courses and three paired memories')
    for key in ['per_task', 'batch_size', 'max_new_tokens']:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Course validation budgets must be positive integers')
    if (type(config['seed']) is not int or type(config['question_formats']) is not int or
            config['question_formats'] not in [1, 3] or config['batch_size'] < 2 or
            not isinstance(config['cuda_visible_devices'], str) or not config['cuda_visible_devices']):
        raise ValueError('Use explicit paired question, device, seed and batch settings')
    profile = config['task_profile'];groups = task_groups(profile)
    rows = load_jsonl(Path(config['data']) / 'validation.jsonl')
    if (not rows or len({r['id'] for r in rows}) != len(rows) or
            any(r['split'] != 'validation' or row_profile(r) != profile or
                r['stage'] not in groups or r['task_type'] not in groups[r['stage']] for r in rows)):
        raise ValueError('Course validation must retain unique, declared validation identities and task types')
    selected = balanced_rows(rows, config['per_task'], config['seed'], config['stages'])
    expected = {(stage, task): config['per_task'] for stage in STAGES for task in groups[stage]}
    if Counter((r['stage'], r['task_type']) for r in selected) != expected:
        raise ValueError('Every declared course task needs its full requested validation coverage')
    if len(selected) % config['batch_size'] == 1:
        raise ValueError('Every shuffled validation batch needs at least two histories')
    if config['question_formats'] == 3:
        counts = Counter();varied = []
        for row in selected:
            key = row['stage'], row['task_type'];variant = counts[key] % 3;counts[key] += 1
            varied.append(dict(row, question=question_variant(row, variant), question_variant=variant))
        selected = varied
    return selected


def bound_json(core, path, expected=None):
    value = core.json(path)
    if expected is not None:core.add(path, expected)
    return value


def course_reference(parent, config, core):
    """Follow completed round headers to the actual initial SFT's four-course data."""
    current = parent;seen = set()
    while current.get('kind') == ROUND_KIND:
        token_path = current['config']['token_preflight']
        if token_path in seen:raise ValueError('Cyclic distillation completion lineage')
        seen.add(token_path)
        token = bound_json(core, token_path, current['inputs'][token_path])
        if token.get('status') != 'complete' or token.get('kind') != TOKEN_KIND:
            raise ValueError('Distilled replay requires its actual completed data preflight')
        previous_path = token['config']['parent_completion']
        previous = bound_json(core, previous_path, current['inputs'][previous_path])
        source = current['config']['source']
        if (previous.get('status') != 'complete' or
                previous.get('kind') not in [ROUND_KIND, 'clean_curriculum_to_initial_sft_pipeline'] or
                previous['verification'].get('selected_checkpoint') != source or
                previous['outputs'].get(source) != current['inputs'].get(source) or
                previous['verification'].get('trained_search_distillation_rounds', 0) + 1 !=
                current['verification']['trained_search_distillation_rounds']):
            raise ValueError('Distilled replay must retain its exact completed parent lineage')
        current = previous
    if (current.get('status') != 'complete' or current.get('kind') != 'clean_curriculum_to_initial_sft_pipeline' or
            current['verification'].get('four_courses_completed_before_any_gpu_sft_work') is not True):
        raise ValueError('Replay requires actual completed initial SFT after all four courses')
    curriculum = Path(current['config']['curriculum']) / 'manifest.json'
    whole = bound_json(core, curriculum, current['inputs'][str(curriculum)])
    policy = ('all_raw_qa_gates_passed' if whole.get('kind') == 'four_course_raw_qa_gated_curriculum' else
              'all_courses_completed_by_declared_or_user_authorized_policy' if whole.get('kind') == USER_CURRICULUM_KIND else
              'all_courses_completed_by_declared_validation_policy')
    if (whole.get('status') != 'complete' or
            whole.get('kind') not in ['four_course_raw_qa_gated_curriculum', CURRICULUM_KIND, USER_CURRICULUM_KIND] or
            [r['stage'] for r in whole['verification']['courses']] != list(STAGES) or
            whole['verification'].get(policy) is not True or
            whole['verification'].get('independent_test_used') is not False or
            Path(whole['config']['data_path']).resolve() != Path(config['data']).resolve() or
            row_profile(whole['config']) != config['task_profile']):
        raise ValueError('Replay data must be the actual completed foundation validation source')
    stage = Path(whole['config']['output']) / STAGES[-1] / 'manifest.json'
    fourth = bound_json(core, stage, whole['inputs'][str(stage)])
    selected = fourth['verification']['selected_checkpoint']
    if (fourth.get('status') != 'complete' or fourth['verification'].get('stage') != STAGES[-1] or
            fourth['verification'].get('independent_test_used') is not False or
            set(whole['outputs']) != {selected} or whole['outputs'][selected] != fourth['outputs'][selected]):
        raise ValueError('Replay reference must retain the actual selected fourth-course model')
    path = Path(selected).parent / 'qa/manifest.json'
    reference = bound_json(core, path, fourth['inputs'][str(path)])
    if (reference.get('status') != 'complete' or reference.get('kind') != 'balanced_board_qa' or
            reference['config'].get('memory') != 'normal' or reference['config'].get('split') != 'validation' or
            reference['config'].get('checkpoint') != selected or reference['code'] != fourth['code']):
        raise ValueError('Require the selected fourth course\'s original raw validation')
    for section in ['inputs', 'outputs']:
        for name, binding in reference[section].items():core.add(name, binding)
    original_validation = str(Path(whole['config']['data_path']) / 'validation.jsonl')
    core.add(Path(config['data']) / 'validation.jsonl', reference['inputs'][original_validation])
    return path, reference


def question_identity(record):
    return tuple(record.get(k, 0 if k == 'question_variant' else None) for k in QUESTION_FIELDS)


def grade_records(records, rows):
    if len(records) != len(rows) or any(record != prediction_record(row, record['generated'])
                                      for record, row in zip(records, rows, strict=True)):
        raise ValueError('Raw course answers, ordered questions or recorded grades changed')


def check_metrics(metrics, records):
    if (any(metrics.get(k) != v for k, v in qa_summary(records).items()) or
            metrics.get('raw_generation') is not True or metrics.get('oracle_used') is not False or
            metrics.get('split') != 'validation' or metrics.get('oracle_repairs', 0) != 0 or
            metrics.get('legality_is_imposed_by_decoding', False) is not False):
        raise ValueError('Course aggregates must match unmodified raw validation answers')


def checked_qa(root, rows, args, inputs, code):
    proof = json.loads((root / 'manifest.json').read_text())
    expected = dict(args)
    if row_profile(rows[0]) == PAPER_PROFILE:expected['task_profile'] = PAPER_PROFILE
    if (proof.get('status') != 'complete' or proof.get('kind') != 'balanced_board_qa' or
            proof['config'] != expected or proof['inputs'] != inputs or proof['code'] != code):
        raise ValueError('Course evaluation differs from its actual model, inputs or paired budget')
    files = [root / 'predictions.jsonl', root / 'metrics.json']
    if set(proof['outputs']) != {str(p) for p in files}:
        raise ValueError('Course evaluation must retain its complete raw predictions and metrics')
    outputs = Artifacts()
    for p in files:outputs.add(p, proof['outputs'][str(p)])
    outputs.verify()
    records = load_jsonl(files[0]);metrics = json.loads(files[1].read_text())
    grade_records(records, rows);check_metrics(metrics, records)
    if (proof['verification'] != metrics or type(metrics.get('seconds')) not in [int, float] or
            not math.isfinite(metrics['seconds']) or metrics['seconds'] < 0):
        raise ValueError('Course completion must match its raw metrics and measured execution')
    outputs.unchanged()
    return metrics, records, proof


def by_course(records):
    grouped = defaultdict(list)
    for record in records:grouped[record['stage']].append(record)
    result = {}
    for stage in STAGES:
        rows = grouped[stage];summary = qa_summary(rows)
        result[stage] = dict(examples=len(rows), correct=sum(r['correct'] for r in rows),
            accuracy=summary['accuracy'], minimum_task_accuracy=min(v['accuracy'] for v in summary['by_task'].values()),
            answer_format_errors=(sum(not r['answer_format_valid'] for r in rows)
                                  if row_profile(rows[0]) == PAPER_PROFILE else None),
            distinct_game_ids=len({r['game_id'] for r in rows}), by_task=summary['by_task'])
    return result


def reference_comparison(reference, current, rows):
    same = [question_identity(r) for r in reference] == [question_identity(r) for r in current]
    result = dict(same_ordered_questions_and_gold=same, paired_changes=None)
    if same:
        grade_records(reference, rows)
        result['paired_changes'] = {stage: dict(
            examples=sum(r['stage'] == stage for r in rows),
            wrong_to_correct=sum(not a['correct'] and b['correct'] for a, b in zip(reference, current, strict=True) if a['stage'] == stage),
            correct_to_wrong=sum(a['correct'] and not b['correct'] for a, b in zip(reference, current, strict=True) if a['stage'] == stage))
            for stage in STAGES}
        result['reference_by_course'] = by_course(reference)
    return result


def run_validation(pipeline, model_config_path, config_path, output, *, resume=False):
    pipeline, root = Path(pipeline), Path(output)
    if not (pipeline / 'manifest.json').exists():
        raise ValueError('A completed selected full-decoder SFT model is required before replay')
    core = Artifacts();model_config = core.json(model_config_path);config = core.json(config_path)
    rows = selected_course_rows(config)
    _, token, recipe = validation_spec(model_config)
    source, parent = checked_sft(pipeline, model_config, recipe)
    core.add(pipeline / 'manifest.json', {'sha256': digest(pipeline / 'manifest.json'),
                                       'bytes': (pipeline / 'manifest.json').stat().st_size})
    saved = core.json(source.parent / 'config.json')
    if row_profile(saved) != config['task_profile']:
        raise ValueError('Replay task profile differs from the actual trained model')
    reference_path, reference = course_reference(parent, config, core)
    core.add(source, parent['outputs'][str(source)])
    for p in [model_config['sft_recipe'], model_config['token_preflight']]:
        core.add(p, {'sha256': digest(p), 'bytes': Path(p).stat().st_size})
    validation = Path(config['data']) / 'validation.jsonl'
    qa_paths = [source, validation, *feature_paths(model_config['features'])]
    for p in qa_paths:core.add(p, {'sha256': digest(p), 'bytes': Path(p).stat().st_size})
    core.verify()
    inputs = {str(p): core.expected[str(p)] for p in qa_paths}
    contract = dict(pipeline=str(pipeline), model_config=str(model_config_path), config=str(config_path),
        code=code_identity(), source=str(source), source_binding=parent['outputs'][str(source)],
        prerequisite_inputs=dict(core.expected))
    complete = (root / 'manifest.json').exists()
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Resume only the exact preserved course replay; otherwise use a fresh output')
    else:
        root.mkdir(parents=True);atomic_json(root / 'contract.json', contract)

    def unchanged():
        core.unchanged()
        if code_identity() != contract['code']:
            raise ValueError('Preserved course replay execution source changed')

    metrics, records_by_memory, proofs = {}, {}, []
    try:
        for memory in MEMORIES:
            unchanged();dest = root / memory
            args = dict(checkpoint=str(source), data=config['data'], features=model_config['features'],
                split='validation', memory=memory, stages=list(STAGES), per_task=config['per_task'],
                seed=config['seed'], question_formats=config['question_formats'], batch_size=config['batch_size'],
                max_new_tokens=config['max_new_tokens'], output=str(dest))
            if not (dest / 'manifest.json').exists():
                if complete or dest.exists():
                    raise ValueError('Preserve incomplete raw course evaluation; use a fresh output')
                atomic_json(root / 'state.json', dict(status='evaluating_' + memory, independent_test_used=False))
                command = [sys.executable, '-u', '-m', 'xqgeneral.evaluate_qa']
                for key, value in args.items():
                    command += ['--' + key.replace('_', '-'), *([str(v) for v in value] if isinstance(value, list) else [str(value)])]
                with (root / 'commands.jsonl').open('a') as handle:handle.write(json.dumps(command) + '\n')
                with (root / 'log.txt').open('a') as handle:
                    subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=config['cuda_visible_devices']))
            metrics[memory], records_by_memory[memory], proof = checked_qa(dest, rows, args, inputs, contract['code'])
            proofs.append(dest / 'manifest.json');unchanged()
        original = load_jsonl(reference_path.parent / 'predictions.jsonl')
        original_metrics = json.loads((reference_path.parent / 'metrics.json').read_text())
        if reference['verification'] != original_metrics:
            raise ValueError('Original fourth-course metrics differ from their completed raw proof')
        check_metrics(original_metrics, original)
        courses = {memory: by_course(records_by_memory[memory]) for memory in MEMORIES}
        result = dict(status='complete', split='validation', actual_completed_full_decoder_model=True,
            selected_checkpoint=str(source), selected_step=parent['verification'].get('selected_sft_step', parent['verification'].get('selected_step')),
            trained_search_distillation_rounds=parent['verification'].get('trained_search_distillation_rounds', 0),
            task_profile=config['task_profile'], examples_per_memory=len(rows), task_types=len(metrics['normal']['by_task']),
            distinct_game_ids=len({r['game_id'] for r in rows}), raw_validation_by_memory=metrics, by_course=courses,
            normal_minus_ablation_accuracy={memory: dict(overall=metrics['normal']['accuracy'] - metrics[memory]['accuracy'],
                by_course={stage: courses['normal'][stage]['accuracy'] - courses[memory][stage]['accuracy'] for stage in STAGES})
                for memory in ['zero', 'shuffled']},
            original_fourth_course_reference=str(reference_path),
            original_foundation_comparison=reference_comparison(original, records_by_memory['normal'], rows),
            raw_answers_repaired=False, independent_test_used_for_training_or_selection=False,
            automatic_acceptance_threshold_applied=False, strong_play_or_reliable_coaching_proven=False)
        unchanged()
        if complete:
            saved_proof = json.loads((root / 'manifest.json').read_text())
            expected_inputs = {str(p): {'sha256': digest(p), 'bytes': Path(p).stat().st_size}
                               for p in [model_config_path, config_path, *proofs]}
            expected_inputs.update(core.expected)
            expected_outputs = {str(root / 'verification.json'): {
                'sha256': digest(root / 'verification.json'), 'bytes': (root / 'verification.json').stat().st_size}}
            for p in proofs:expected_outputs.update(json.loads(p.read_text())['outputs'])
            outputs = Artifacts()
            for name, binding in saved_proof['outputs'].items():outputs.add(name, binding)
            outputs.verify()
            if (saved_proof.get('kind') != KIND or saved_proof.get('status') != 'complete' or
                    saved_proof['config'] != contract or saved_proof['code'] != contract['code'] or
                    saved_proof['inputs'] != expected_inputs or saved_proof['outputs'] != expected_outputs or
                    saved_proof['verification'] != result or json.loads((root / 'verification.json').read_text()) != result):
                raise ValueError('Completed course replay differs from its actual raw report')
            return result
        atomic_json(root / 'verification.json', result)
        completed = manifest(KIND, contract, [model_config_path, config_path, *proofs], [root / 'verification.json'], result)
        completed['inputs'].update(core.expected)
        for p in proofs:completed['outputs'].update(json.loads(p.read_text())['outputs'])
        atomic_json(root / 'manifest.json', completed);atomic_json(root / 'state.json', result)
        return result
    except Exception as error:
        if not complete:
            atomic_json(root / 'state.json', dict(status='failed', error_type=type(error).__name__,
                preserve_partial_outputs=True, course_validation_completion_proven=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', required=True)
    parser.add_argument('--model-config', required=True, help='The actual completed model\'s explanation-validation configuration')
    parser.add_argument('--config', default='configs/curriculum-replay-validation-v1.json')
    parser.add_argument('--output', required=True);parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with input_session_from_environment():
        print(json.dumps(run_validation(args.pipeline, args.model_config, args.config, args.output, resume=args.resume)), flush=True)


if __name__ == '__main__':main()
