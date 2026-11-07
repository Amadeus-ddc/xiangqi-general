"""Full original validation coverage; controlled models do not prove GPU quality."""
from collections import Counter
import json
from pathlib import Path

import pytest

from test_distillation_joint_replay import controlled_joint_round, joint_budget
from xqgeneral import distillation_training
from xqgeneral.curriculum_data import STAGES
from xqgeneral.evidence import atomic_json, digest
from xqgeneral.training import select_validation_rows
from xqgeneral.training_index import build_index, IndexedTrainingRows


PROFILE = 'joint_explanation_and_course_replay_v1'


def joint_validation(recipe):
    return dict(recipe, validation_profile=PROFILE, validation_stages=['explanation', *STAGES])


def test_explicit_joint_validation_compiles_all_original_explanations_and_four_courses(tmp_path):
    template, token = joint_budget()
    recipe, budget = distillation_training.compile_recipe(joint_validation(template), token, tmp_path)
    assert recipe['validation_examples'] == 384 + 4 * 32
    assert recipe['validation_rows_by_stage'] == {'explanation': 384, **dict.fromkeys(STAGES, 32)}
    assert budget['validation_rows'] == recipe['validation_examples']
    assert budget['validation_profile'] == PROFILE
    assert budget['validation_rows_by_stage'] == recipe['validation_rows_by_stage']
    assert budget['validation_uses_all_original_rows'] is True
    assert budget['validation_metric'] == 'token_nll_over_complete_original_validation'
    assert budget['accepted_training_rows'] == 1000


@pytest.mark.parametrize('problem', ['missing_course', 'reordered', 'unknown_profile', 'none_profile',
                                    'no_replay', 'forged_counts'])
def test_joint_validation_cannot_silently_drop_courses_or_trust_a_template_count(tmp_path, problem):
    template, token = joint_budget()
    template = joint_validation(template)
    if problem == 'missing_course': template['validation_stages'].pop()
    elif problem == 'reordered': template['validation_stages'].reverse()
    elif problem == 'unknown_profile': template['validation_profile'] = 'future-profile'
    elif problem == 'none_profile': template['validation_profile'] = None
    elif problem == 'no_replay':
        token['config']['replay_data_paths'] = []
        template.update(stages=['explanation'], mixture={'explanation': 1.0}, replay_data_paths=[])
    else: template['validation_rows_by_stage'] = {'explanation': 384, **dict.fromkeys(STAGES, 1)}
    with pytest.raises(ValueError):
        distillation_training.compile_recipe(template, token, tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


def test_original_explanation_only_contract_remains_identical_without_opt_in(tmp_path):
    template, token = joint_budget()
    recipe, budget = distillation_training.compile_recipe(template, token, tmp_path)
    assert recipe['validation_stages'] == ['explanation'] and recipe['validation_examples'] == 384
    assert 'validation_profile' not in recipe and 'validation_rows_by_stage' not in recipe
    assert budget['validation_rows'] == 384 and 'validation_profile' not in budget


@pytest.mark.parametrize('field', ['validation_profile', 'validation_rows_by_stage'])
def test_joint_validation_configuration_is_part_of_exact_round_resume(tmp_path, monkeypatch, field):
    source, recipe_path, token, runner, _ = controlled_joint_round(tmp_path, monkeypatch)
    recipe = joint_validation(json.loads(recipe_path.read_text()))
    atomic_json(recipe_path, recipe)
    install_joint_validation_runner(monkeypatch, runner)
    root = tmp_path / 'completed-joint-round'
    distillation_training.run_round(source, recipe_path, token, root)
    if field == 'validation_profile': recipe[field] = 'explanation_only_v1'
    else: recipe[field] = {'explanation': 16, **dict.fromkeys(STAGES, 1)}
    atomic_json(recipe_path, recipe)
    with pytest.raises(ValueError):
        distillation_training.run_round(source, recipe_path, token, root, resume=True)
    assert len(runner.jobs) == 4


def test_actual_source_count_weights_reach_sft_without_losing_full_replay_coverage(tmp_path):
    from xqgeneral.evidence import write_jsonl
    from xqgeneral.sft import prepare_config
    template, token = joint_budget()
    data, replay = tmp_path / 'lectures', tmp_path / 'courses'
    token['config'].update(data=str(data), replay_data_paths=[str(replay)])
    token['verification']['course_replay']['data_paths'] = [str(replay)]
    template.update(replay_data_paths=[str(replay)],
                    mixture={'explanation': 1000, **dict.fromkeys(STAGES, 128)})
    for directory, stages in [(data, ['explanation']), (replay, STAGES)]:
        write_jsonl(directory / 'train.jsonl', [
            {'id': f'controlled-{stage}-{i}', 'stage': stage, 'split': 'train'}
            for stage in stages for i in range(template['mixture'][stage])])
    compiled, budget = distillation_training.compile_recipe(joint_validation(template), token, tmp_path / 'round')
    config = prepare_config({'mode': 'lora'}, compiled, tmp_path / 'unloaded-parent.pt', tmp_path / 'planning')
    assert 0 < config['mixture']['explanation'] <= 1
    assert budget['finite_mixture']['selected_rows_by_stage'] == template['mixture']
    assert budget['rows_per_complete_pass'] == 1512
    assert config['steps'] == budget['max_updates']


def test_real_index_selection_covers_all_sources_once_and_never_opens_test(tmp_path):
    template, token = joint_budget()
    template = joint_validation(template)
    token['verification']['by_split']['validation'] = 3
    recipe, budget = distillation_training.compile_recipe(template, token, tmp_path / 'round')
    files = []
    from xqgeneral.evidence import write_jsonl
    for source, stages in [('lectures', ['explanation']), ('replay', STAGES)]:
        path = tmp_path / source / 'validation.jsonl'
        rows = [{'id': f'{stage}-{i}', 'feature_key': f'feature-{stage}-{i}', 'stage': stage,
                 'split': 'validation', 'answer': 'controlled validation answer'}
                for stage in stages for i in range(recipe['validation_rows_by_stage'][stage])]
        write_jsonl(path, rows)
        files.append(path)
    index = tmp_path / 'validation-index'
    build_index(files, index, split='validation')
    indexed = IndexedTrainingRows(index, files, recipe['mixture'], split='validation')
    try:
        selected = select_validation_rows(indexed, recipe['validation_stages'], recipe['seed'],
                                          recipe['validation_examples'])
        from xqgeneral.distillation_validation import joint_validation_coverage
        coverage = joint_validation_coverage(indexed, [r['id'] for r in selected],
                                              budget['validation_rows_by_stage'])
        assert Counter(r['stage'] for r in selected) == budget['validation_rows_by_stage']
        assert coverage['all_original_validation_rows_covered_once']
        assert coverage['independent_test_used'] is False
    finally:
        indexed.close()


@pytest.mark.parametrize('problem', ['drop', 'duplicate', 'foreign', 'wrong_stage', 'wrong_count'])
def test_recorded_validation_identity_cannot_replace_missing_or_repeated_examples(problem):
    from xqgeneral.distillation_validation import joint_validation_coverage
    rows = [{'id': f'row-{stage}', 'stage': stage} for stage in ['explanation', *STAGES]]
    selected = [r['id'] for r in rows]
    counts = dict.fromkeys(['explanation', *STAGES], 1)
    if problem == 'drop': selected.pop()
    elif problem == 'duplicate': selected[-1] = selected[0]
    elif problem == 'foreign': selected[-1] = 'not-an-original-validation-id'
    elif problem == 'wrong_stage': rows[-1]['stage'] = 'unverified-course'
    else: counts[STAGES[0]] = 2
    with pytest.raises(ValueError):
        joint_validation_coverage(rows, selected, counts)


def install_joint_validation_runner(monkeypatch, runner, problem=None):
    def run(command, **kwargs):
        result = runner(command, **kwargs)
        if 'xqgeneral.sft' not in command:
            return result
        root = Path(command[command.index('--output') + 1])
        config = json.loads((root / 'config.json').read_text())
        sources = [Path(p) / 'validation.jsonl' for p in
                   [config['data_path'], *config['replay_data_paths']]]
        indexed = IndexedTrainingRows(config['row_index_paths']['validation'], sources,
                                      config['mixture'], split='validation')
        try:
            selected = select_validation_rows(indexed, config['validation_stages'], config['seed'],
                                              config['validation_examples'])
            ids = [r['id'] for r in selected]
            from xqgeneral.distillation_validation import joint_validation_coverage
            coverage = joint_validation_coverage(indexed, ids, config['validation_rows_by_stage'])
        finally:
            indexed.close()
        if problem == 'duplicate': ids[-1] = ids[0]
        elif problem == 'missing': ids.pop()
        elif problem == 'order': ids.reverse()
        path = root / 'validation_ids.json'
        atomic_json(path, ids)
        if problem == 'missing_ledger': path.unlink()
        return result
    monkeypatch.setattr(distillation_training.subprocess, 'run', run)


def test_joint_round_reopens_native_validation_ids_and_preserves_completed_resume(tmp_path, monkeypatch):
    source, recipe_path, token, runner, _ = controlled_joint_round(tmp_path, monkeypatch)
    atomic_json(recipe_path, joint_validation(json.loads(recipe_path.read_text())))
    install_joint_validation_runner(monkeypatch, runner)
    root = tmp_path / 'joint-validation-round'
    result = distillation_training.run_round(source, recipe_path, token, root)
    assert result['joint_validation_coverage']['validation_rows_by_stage'] == {
        'explanation': 16, **dict.fromkeys(STAGES, 2)}
    assert result['joint_validation_coverage']['validation_rows'] == 24
    assert result['joint_validation_coverage']['all_original_validation_rows_covered_once']
    path = root / 'sft/validation_ids.json'
    proof = json.loads((root / 'manifest.json').read_text())
    assert proof['outputs'][str(path)]['sha256'] == digest(path)
    before = {p: (digest(p), Path(p).stat().st_mtime_ns) for p in proof['outputs']}
    assert distillation_training.run_round(source, recipe_path, token, root, resume=True) == result
    assert len(runner.jobs) == 4
    assert before == {p: (digest(p), Path(p).stat().st_mtime_ns) for p in proof['outputs']}


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'missing_ledger', 'order'])
def test_round_rejects_a_forged_complete_validation_claim(tmp_path, monkeypatch, problem):
    source, recipe_path, token, runner, _ = controlled_joint_round(tmp_path, monkeypatch)
    atomic_json(recipe_path, joint_validation(json.loads(recipe_path.read_text())))
    install_joint_validation_runner(monkeypatch, runner, problem)
    root = tmp_path / 'must-not-qualify'
    with pytest.raises(ValueError, match='validation'):
        distillation_training.run_round(source, recipe_path, token, root)
    assert not (root / 'manifest.json').exists()
