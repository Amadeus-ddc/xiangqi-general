"""Complete finite-source coverage; controlled runners do not execute CUDA."""
import json
from pathlib import Path

import pytest

from test_distillation_joint_replay import (controlled_joint_round, joint_budget,
                                          replay_contract, rewrite_rows)
from test_joint_distillation_validation import install_joint_validation_runner
from xqgeneral import distillation_training
from xqgeneral.curriculum_data import STAGES
from xqgeneral.evidence import atomic_json, digest, write_jsonl
from xqgeneral.finite_training import FINITE_PROFILE, prepare_finite_epochs
from xqgeneral.sft import prepare_config
from xqgeneral.training_index import build_index, IndexedTrainingRows

PROFILE = 'all_actual_training_rows_v1'


def full_recipe():
    template, token = joint_budget(train=32, old=64)
    counts = {'explanation': 32, **dict.fromkeys(STAGES, 64)}
    template.update(source_coverage_profile=PROFILE, mixture=counts,
        validation_profile='joint_explanation_and_course_replay_v1',
        validation_stages=['explanation', *STAGES], max_finite_passes=4)
    return template, token, counts


def files_for_config(tmp_path):
    template, token, counts = full_recipe()
    data, replay = tmp_path / 'lectures', tmp_path / 'replay'
    for directory, stages in [(data, ['explanation']), (replay, STAGES)]:
        write_jsonl(directory / 'train.jsonl', [
            {'id': f'{stage}-{i}', 'stage': stage, 'split': 'train',
             'feature_key': f'feature-{stage}-{i}', 'answer': 'controlled'}
            for stage in stages for i in range(counts[stage])])
    template['replay_data_paths'] = [str(replay)]
    token['config'].update(data=str(data), replay_data_paths=[str(replay)])
    token['verification']['course_replay']['data_paths'] = [str(replay)]
    return template, token, counts, data, replay


def test_all_sources_compile_when_old_courses_are_larger_than_explanations(tmp_path):
    template, token, counts = full_recipe()
    recipe, budget = distillation_training.compile_recipe(template, token, tmp_path)
    assert budget['finite_mixture']['anchor_stage'] == 'static_current'
    assert budget['finite_mixture']['selected_rows_by_stage'] == counts
    assert budget['source_coverage_profile'] == PROFILE
    assert budget['all_actual_training_rows_selected_each_epoch'] is True
    assert budget['rows_per_complete_pass'] == 288
    assert budget['updates_per_complete_pass'] == 9 and budget['max_updates'] == 36
    assert recipe['min_steps'] == recipe['eval_every'] == 9
    assert recipe['validation_examples'] == 512
    assert budget['accepted_training_rows'] == 32
    assert not budget['paper_first_round_150000_label_scale_reached']


def test_compiled_recipe_native_sft_and_real_index_agree_on_all_original_rows(tmp_path):
    template, token, counts, data, replay = files_for_config(tmp_path)
    recipe, budget = distillation_training.compile_recipe(template, token, tmp_path / 'round')
    config = prepare_config({'mode': 'lora'}, recipe, 'unopened.pt', tmp_path / 'planning')
    assert config['steps'] == budget['max_updates'] == 36
    sources = [data / 'train.jsonl', replay / 'train.jsonl']
    index = tmp_path / 'train-index'
    build_index(sources, index, split='train')
    rows = IndexedTrainingRows(index, sources, recipe['mixture'])
    try:
        finite = prepare_finite_epochs(rows, recipe['mixture'], config, world=4)
        assert finite.mix['selected_rows_by_stage'] == counts
        assert set(map(int, finite.selected)) == set(range(288))
        assert finite.contract()['source_coverage_profile'] == PROFILE
        assert finite.contract()['all_actual_training_rows_selected_each_epoch'] is True
    finally:
        rows.close()


@pytest.mark.parametrize('problem', ['subset', 'unknown', 'none', 'boolean', 'missing_source', 'cap'])
def test_full_coverage_claim_cannot_hide_subsets_or_invalid_contracts(tmp_path, problem):
    template, token, _ = full_recipe()
    if problem == 'subset': template['mixture'] = {'explanation': .8, **dict.fromkeys(STAGES, .05)}
    elif problem == 'unknown': template['source_coverage_profile'] = 'unknown'
    elif problem == 'none': template['source_coverage_profile'] = None
    elif problem == 'boolean': template['source_coverage_profile'] = True
    elif problem == 'missing_source': template['mixture'].pop(STAGES[-1])
    else: template['max_steps'] = 8
    with pytest.raises(ValueError):
        distillation_training.compile_recipe(template, token, tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


def test_default_recipe_still_requires_explanation_anchor(tmp_path):
    template, token, _ = full_recipe()
    template.pop('source_coverage_profile')
    with pytest.raises(ValueError, match='every accepted explanation'):
        distillation_training.compile_recipe(template, token, tmp_path)


def test_full_scope_checks_actual_sft_files_for_undeclared_rows(tmp_path):
    template, token, _, _, replay = files_for_config(tmp_path)
    recipe, _ = distillation_training.compile_recipe(template, token, tmp_path / 'round')
    with (replay / 'train.jsonl').open('a') as handle:
        handle.write(json.dumps({'id': 'unverified', 'stage': 'foreign', 'split': 'train'}) + '\n')
    with pytest.raises(ValueError, match='source|coverage'):
        prepare_config({'mode': 'lora'}, recipe, 'unopened.pt', tmp_path / 'planning')


def sampler_rows():
    return [{'id': f'{stage}-{i}', 'stage': stage, 'split': 'train'}
            for stage, count in [('explanation', 3), ('old', 5)] for i in range(count)]


def sampler_config():
    return {'source_coverage_profile': PROFILE, 'training_data_profile': FINITE_PROFILE,
            'batch_size': 4, 'micro_batch_size': 1, 'seed': 17}


@pytest.mark.parametrize('problem', ['subset', 'unknown_stage', 'legacy_profile', 'unknown_profile'])
def test_native_epoch_sampler_also_rejects_invalid_full_source_claims(problem):
    rows, config = sampler_rows(), sampler_config()
    mixture = {'explanation': 3, 'old': 5}
    if problem == 'subset': mixture = {'explanation': 1, 'old': 1}
    elif problem == 'unknown_stage': rows.append({'id': 'outside', 'stage': 'unselected', 'split': 'train'})
    elif problem == 'legacy_profile': config.pop('training_data_profile')
    else: config['source_coverage_profile'] = 'unknown'
    with pytest.raises(ValueError):
        prepare_finite_epochs(rows, mixture, config, world=4)


def test_full_source_scope_is_bound_to_optimizer_resume_and_padding_is_explicit():
    rows, config = sampler_rows(), sampler_config()
    rows.append({'id': 'old-5', 'stage': 'old', 'split': 'train'})
    mixture = {'explanation': 3, 'old': 6}
    complete = prepare_finite_epochs(rows, mixture, config, world=4)
    assert len(complete.selected) == 9 and complete.contract()['padded_rows_per_epoch'] == 3
    assert complete.state_at(complete.steps_per_epoch)['unique_rows_consumed'] == 9
    assert complete.state_at(complete.steps_per_epoch)['padding_rows_consumed'] == 3
    resumed = prepare_finite_epochs(rows, mixture, config, world=4)
    resumed.check_resume(complete.state_at(2), 2)
    legacy = prepare_finite_epochs(rows, mixture,
                                  {k: v for k, v in config.items() if k != 'source_coverage_profile'}, world=4)
    with pytest.raises(ValueError, match='Resume finite'):
        legacy.check_resume(complete.state_at(2), 2)


def test_controlled_full_round_keeps_capacity_probe_bounded_and_formal_sources_complete(tmp_path, monkeypatch):
    source, recipe_path, token_path, runner, replay = controlled_joint_round(tmp_path, monkeypatch)
    rows = [{'id': f'controlled-{stage}-train-{i}', 'feature_key': f'old-{stage}-train-{i}',
             'stage': stage, 'split': 'train', 'answer': 'preserved controlled old answer'}
            for stage in STAGES for i in range(64)]
    rewrite_rows(replay, 'train', rows)
    template = json.loads(recipe_path.read_text())
    template.update(source_coverage_profile=PROFILE,
        mixture={'explanation': 32, **dict.fromkeys(STAGES, 64)}, max_steps=36,
        max_finite_passes=4, eval_every=9,
        validation_profile='joint_explanation_and_course_replay_v1',
        validation_stages=['explanation', *STAGES])
    token = json.loads(token_path.read_text())
    token['verification']['course_replay'] = replay_contract([str(replay)], train=64, validation=2)
    for name in ['manifest.json', 'train.jsonl']:
        path = replay / name
        token['inputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(recipe_path, template)
    atomic_json(token_path, token)
    install_joint_validation_runner(monkeypatch, runner)
    root = tmp_path / 'full-source-round'
    result = distillation_training.run_round(source, recipe_path, token_path, root)
    assert result['actual_steps'] == 36 and result['source_coverage_profile'] == PROFILE
    assert result['all_actual_training_rows_selected_each_epoch'] is True
    assert result['training_data_state']['epoch'] == 4
    assert result['training_data_state']['contract']['selected_rows_by_stage'] == {
        'explanation': 32, **dict.fromkeys(STAGES, 64)}
    probe_config = json.loads((root / 'preflight/continuous.config.json').read_text())
    assert 'source_coverage_profile' not in probe_config
    assert probe_config['mixture'] == {'explanation': .8, **dict.fromkeys(STAGES, .05)}
    assert probe_config['validation_stages'] == ['explanation']
    probe_result = json.loads((root / 'preflight/verification.json').read_text())
    assert probe_result['formal_source_coverage_profile'] == PROFILE
    assert probe_result['bounded_probe_is_full_training_coverage'] is False
    assert result['joint_validation_coverage']['validation_rows'] == 24
    assert distillation_training.run_round(source, recipe_path, token_path, root, resume=True) == result
    assert len(runner.jobs) == 4
