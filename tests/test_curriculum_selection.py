"""Controlled tensor/QA contracts; these fixtures do not establish trained model ability."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from test_foundation_handoff import completed_curriculum
from xqgeneral import validation_curriculum
from xqgeneral.curriculum_data import STAGES
from xqgeneral.curriculum_selection import HANDOFF_KIND, selection_decision, validate_course_selection
from xqgeneral.evaluate_qa import balanced_rows, qa_summary
from xqgeneral.evidence import atomic_json, code_identity, digest, load_jsonl, manifest, write_jsonl
from xqgeneral.foundation_handoff import export_curriculum
from xqgeneral.gated_curriculum import raw_qa_gate
from xqgeneral.sft import checked_parent
from xqgeneral.selfplay_grounding import question_variant


POLICY = {'metric': 'balanced_raw_qa_accuracy', 'tie_breaker': 'minimum_task_accuracy_then_earliest', 'patience': 3}


def gate(accuracy, minimum, passed=False):
    return {'accuracy': accuracy, 'by_task': {'a': {'accuracy': accuracy}, 'b': {'accuracy': minimum}},
            'validation_only': True, 'raw_generation': True, 'oracle_used': False, 'passed': passed}


def test_plateau_selects_earlier_best_and_retains_failed_reference_targets():
    scores = [(i * 512, .5 + i * .04, .2 + i * .06) for i in range(1, 11)] + [
        (5632, 752 / 768, 121 / 128), (6144, 752 / 768, 119 / 128),
        (6656, 743 / 768, 117 / 128), (7168, 746 / 768, 120 / 128)]
    history = [{'step': step, 'gate': gate(accuracy, minimum)} for step, accuracy, minimum in scores]
    budget = {'qa_every': 512, 'steps': 50000, 'minimum_steps': 2048}
    decision = selection_decision(history, POLICY, budget)
    assert decision['stop'] and decision['stopping_reason'] == 'validation_plateau'
    assert decision['selected_step'] == 5632 and decision['steps_completed'] == 7168
    assert decision['checks_without_improvement'] == 3
    assert not history[10]['gate']['passed']
    improved = deepcopy(history);improved[-1]['gate'] = gate(.985, .95)
    assert not selection_decision(improved, POLICY, budget)['stop']


def test_warmup_minimum_and_maximum_budget_do_not_choose_latest_or_wait_forever():
    budget = {'qa_every': 8, 'steps': 26, 'minimum_steps': 16}
    history = [{'step': s, 'gate': gate(a, .8)} for s, a in [(8, .99), (16, .95), (24, .9), (26, .91)]]
    decision = selection_decision(history, POLICY, budget)
    assert decision['selected_step'] == 16 and decision['stopping_reason'] == 'maximum_budget'
    assert selection_decision(history[:1], POLICY, budget)['selected_step'] is None
    history[1]['gate']['passed'] = True
    assert selection_decision(history[:2], POLICY, budget)['stopping_reason'] == 'reference_targets_met'


@pytest.mark.parametrize('problem', ['missing_check', 'duplicate_step', 'test', 'oracle', 'patience_bool', 'metric'])
def test_invalid_selection_evidence_and_policy_rejected(problem):
    history = [{'step': s, 'gate': gate(.9, .8)} for s in [8, 16, 24, 32]]
    policy = dict(POLICY)
    if problem == 'missing_check':history.pop(1)
    elif problem == 'duplicate_step':history[1]['step'] = 8
    elif problem == 'test':history[0]['gate']['validation_only'] = False
    elif problem == 'oracle':history[0]['gate']['oracle_used'] = True
    elif problem == 'patience_bool':policy['patience'] = True
    elif problem == 'metric':policy['metric'] = 'training_loss'
    with pytest.raises(ValueError):selection_decision(history, policy, {'qa_every': 8, 'steps': 40, 'minimum_steps': 8})


def raw_candidate(candidate, saved, step, recipe, stages, wrong=0):
    candidate.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate / 'adapter.pt'
    torch.save(dict(saved, selected_step=step), checkpoint)
    source = candidate / 'source-checkpoint.pt'
    torch.save(dict(saved, step=step, optimizer={}), source)
    atomic_json(candidate / 'snapshot.json', {'actual_step': step, 'atomic_source_inode_pinned': True,
        'source_sha256': digest(source), 'adapter_sha256': digest(checkpoint), 'optimizer_state_in_adapter': False,
        'training_config': saved['config'], 'training_input_hashes': saved['input_hashes'], 'training_code': saved['code']})
    rows = balanced_rows(load_jsonl(Path(recipe['data_path']) / 'validation.jsonl'), 3, 13, stages)
    records, counts = [], {}
    for row in rows:
        key = row['stage'], row['task_type'];variant = counts.get(key, 0) % 3;counts[key] = counts.get(key, 0) + 1
        records.append({k: row[k] for k in ['id', 'game_id', 'stage', 'task_type']} | {
            'question': question_variant(row, variant), 'question_variant': variant,
            'expected': row['answer'], 'generated': row['answer'], 'correct': True})
    for record in records[:wrong]:record.update(generated='wrong raw answer', correct=False)
    qa = candidate / 'qa';write_jsonl(qa / 'predictions.jsonl', records)
    metrics = dict(qa_summary(records), raw_generation=True, oracle_used=False, split='validation')
    atomic_json(qa / 'metrics.json', metrics)
    arguments = {'checkpoint': str(checkpoint), 'data': recipe['data_path'], 'features': recipe['feature_path'],
        'split': 'validation', 'memory': 'normal', 'stages': list(stages), 'per_task': 3, 'seed': 13, 'question_formats': 3}
    atomic_json(qa / 'manifest.json', manifest('balanced_board_qa', arguments,
        [checkpoint, Path(recipe['data_path']) / 'validation.jsonl', recipe['feature_path']],
        [qa / 'predictions.jsonl', qa / 'metrics.json'], metrics, code=saved['code']))
    atomic_json(candidate / 'gate.json', raw_qa_gate(records, stages, recipe['raw_qa_gates']['targets'], 3))


def imported_recipe(tmp_path, original=None):
    if original is None:
        original, _ = completed_curriculum(tmp_path)
    recipe_path = tmp_path / 'recipe.json';recipe = json.loads(recipe_path.read_text())
    source = tmp_path / 'original-source/source/xqgeneral';source.mkdir(parents=True)
    for path in Path('src/xqgeneral').glob('*.py'):shutil.copy2(path, source / path.name)
    origin_code = dict(code_identity(), revision='controlled-original-source')
    source_manifest = source.parent.parent / 'source-manifest.json'
    atomic_json(source_manifest, {'revision': origin_code['revision'],
        'source_sha256': {Path(n).name: sha for n, sha in origin_code['source_sha256'].items()}})
    candidates = original / 'static_current/candidates'
    saved = torch.load(candidates / 'step-8/adapter.pt', weights_only=True)
    saved['code'] = origin_code
    raw_candidate(candidates / 'step-8', saved, 8, recipe, STAGES[:1], wrong=1)
    saved['trainable'] = {n: t + .1 for n, t in saved['trainable'].items()}
    raw_candidate(candidates / 'step-16', saved, 16, recipe, STAGES[:1], wrong=3)
    recipe.update(output=str(tmp_path / 'new-curriculum'),
        validation_selection=dict(POLICY, patience=1),
        initial_course_import={'recipe': str(recipe_path), 'through_step': 16, 'source_manifest': str(source_manifest)})
    path = tmp_path / 'new-recipe.json';atomic_json(path, recipe)
    return path, candidates / 'step-8/adapter.pt'


def test_actual_earlier_import_preserves_original_source_and_only_prepares_second_course(tmp_path, monkeypatch):
    path, selected = imported_recipe(tmp_path)
    def forbidden(*args, **kwargs):raise AssertionError('Preparing the import must not load a model or launch GPU training')
    monkeypatch.setattr(validation_curriculum.subprocess, 'run', forbidden)
    # code_identity itself invokes subprocess.run; preserve the actual identity without recursion.
    monkeypatch.setattr(validation_curriculum, 'code_identity', lambda: {'revision': 'controlled-new-source',
        'source_sha256': {f'src/xqgeneral/{p.name}': digest(p) for p in Path('src/xqgeneral').glob('*.py')}})
    result = validation_curriculum.run_curriculum(path, prepare_only=True)
    assert len(result) == 1 and result[0]['selected_checkpoint'] == str(selected)
    assert result[0]['actual_steps'] == 16 and result[0]['selected_step'] == 8
    assert not result[0]['gate']['passed'] and result[0]['next_course_permitted']
    root = Path(json.loads(path.read_text())['output'])
    second = json.loads((root / 'dynamic_current/training.config.json').read_text())
    assert second['init_from'] == str(selected) and second['mixture']['static_current'] == .1
    assert not (root / 'dynamic_current/training').exists() and not (root / 'manifest.json').exists()
    assert result == validation_curriculum.run_curriculum(path, prepare_only=True, resume=True)


def install_curriculum_runner(monkeypatch, recipe):
    original_run = subprocess.run;calls = []
    def run(command, **kwargs):
        if command[0] == 'git':return original_run(command, **kwargs)
        assert 'xqgeneral.train' in command
        config = json.loads(Path(command[command.index('--config') + 1]).read_text())
        root = Path(config['output']);root.mkdir(parents=True, exist_ok=True)
        saved = torch.load(config['init_from'], weights_only=True)
        saved.update(config=config, code=code_identity(), world_size=1,
            step=int(command[command.index('--stop-after') + 1]),
            trainable={n: t + .01 for n, t in saved['trainable'].items()},
            input_hashes={str(p): digest(p) for p in [config['feature_path'],
                Path(config['data_path']) / 'train.jsonl', Path(config['data_path']) / 'validation.jsonl', config['init_from']]})
        torch.save(saved, root / 'latest.pt');calls.append(command)
        return SimpleNamespace(returncode=0)
    def evaluate(candidate, checkpoint, stages, actual_recipe):
        saved = torch.load(checkpoint, weights_only=True)
        raw_candidate(candidate, saved, saved['selected_step'], actual_recipe, stages)
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(validation_curriculum, 'evaluate_candidate', evaluate)
    return calls


def test_complete_import_then_three_courses_handoff_and_sft_with_honest_failed_targets(tmp_path, monkeypatch):
    path, selected = imported_recipe(tmp_path);recipe = json.loads(path.read_text())
    calls = install_curriculum_runner(monkeypatch, recipe)
    result = validation_curriculum.run_curriculum(path)
    assert len(calls) == 3 and [r['stage'] for r in result] == list(STAGES)
    configs = [json.loads(Path(c[c.index('--config') + 1]).read_text()) for c in calls]
    assert configs[0]['init_from'] == str(selected)
    assert configs[1]['init_from'] == result[1]['selected_checkpoint']
    assert configs[2]['init_from'] == result[2]['selected_checkpoint']
    root = Path(recipe['output']);whole = json.loads((root / 'manifest.json').read_text())
    assert whole['verification']['all_courses_completed_by_declared_validation_policy']
    assert whole['verification']['all_raw_qa_gates_passed'] is False
    assert result == validation_curriculum.run_curriculum(path, resume=True)
    assert len(calls) == 3
    output = tmp_path / 'handoff';exported = export_curriculum(root, output)
    assert exported['all_four_validation_selection_rules_recomputed']
    assert exported['all_reference_targets_passed'] is False
    assert json.loads((output / 'manifest.json').read_text())['kind'] == HANDOFF_KIND
    assert checked_parent(output / 'adapter.pt', True) == configs[-1]
    assert digest(output / 'adapter.pt') == digest(result[-1]['selected_checkpoint'])


@pytest.mark.parametrize('problem', ['prediction', 'origin_source', 'origin_config', 'origin_snapshot', 'recipe_change', 'selected_latest'])
def test_changed_import_or_forged_completion_cannot_advance(tmp_path, problem):
    path, selected = imported_recipe(tmp_path);recipe = json.loads(path.read_text())
    if problem == 'selected_latest':
        result = validation_curriculum.run_curriculum(path, prepare_only=True)[0]
        result['selected_step'] = 16
        with pytest.raises(ValueError):validate_course_selection(result)
        return
    if problem == 'recipe_change':
        recipe['course_budgets'][0]['learning_rate'] *= 2;atomic_json(path, recipe)
    else:
        targets = {'prediction': selected.parent / 'qa/predictions.jsonl',
            'origin_source': Path(recipe['initial_course_import']['source_manifest']).parent / 'source/xqgeneral/train.py',
            'origin_config': selected.parent.parent.parent / 'training.config.json',
            'origin_snapshot': selected.parent / 'snapshot.json'}
        with targets[problem].open('ab') as handle:handle.write(b'changed')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        validation_curriculum.run_curriculum(path, prepare_only=True)
    assert not (Path(recipe['output']) / 'static_current/manifest.json').exists()


def test_unchanged_big_inputs_hashed_once_and_mutation_detected(tmp_path, monkeypatch):
    from collections import Counter
    path, selected = imported_recipe(tmp_path);recipe = json.loads(path.read_text());counts = Counter()
    original = validation_curriculum.digest
    def counted(name):counts[str(name)] += 1;return original(name)
    monkeypatch.setattr(validation_curriculum, 'digest', counted)
    install_curriculum_runner(monkeypatch, recipe)
    validation_curriculum.run_curriculum(path)
    assert counts[recipe['feature_path']] == 1
    assert counts[str(Path(recipe['data_path']) / 'train.jsonl')] == 1


def test_validation_selected_four_courses_feed_the_actual_sequential_sft_contract(tmp_path, monkeypatch):
    from test_sft_pipeline import setup_data, install_runner
    from xqgeneral import sft_pipeline
    original, sft_recipe, token = setup_data(tmp_path)
    path, _ = imported_recipe(tmp_path, original=original)
    recipe = json.loads(path.read_text())
    install_curriculum_runner(monkeypatch, recipe)
    validation_curriculum.run_curriculum(path)
    runner = install_runner(monkeypatch)
    result = sft_pipeline.run_pipeline(recipe['output'], 'controlled-owned-session', sft_recipe, token, tmp_path / 'pipeline')
    assert result['four_courses_completed_before_any_gpu_sft_work']
    assert result['initial_explanation_sft_completed'] and len(runner.jobs) == 4
    assert result['trained_search_distillation_rounds'] == 0


def test_changed_evidence_during_selected_tensor_readback_cannot_complete_course(tmp_path, monkeypatch):
    path, selected = imported_recipe(tmp_path)
    original = validation_curriculum.foundation_state_summary
    def altered(saved, config):
        with (selected.parent / 'qa/predictions.jsonl').open('ab') as handle:handle.write(b'changed-during-check')
        return original(saved, config)
    monkeypatch.setattr(validation_curriculum, 'foundation_state_summary', altered)
    with pytest.raises(ValueError, match='changed during verification'):
        validation_curriculum.run_curriculum(path, prepare_only=True)
    root = Path(json.loads(path.read_text())['output'])
    assert not (root / 'static_current/manifest.json').exists()
