"""Finite budgets and actual config/index handoffs; GPU workers are controlled."""
import json
from pathlib import Path

import pytest

from test_distillation_joint_replay import joint_budget
from test_distillation_preflight import cpu_flow
from test_distillation_training import budget_token, round_setup
from xqgeneral import distillation_training
from xqgeneral.evidence import atomic_json
from xqgeneral.sft import prepare_config
from xqgeneral.training import training_memory_settings


def profile():
    return json.loads(Path('configs/search-distillation-sft-fsdp2-v1.json').read_text())


def test_actual_rounded_joint_pass_limits_updates_warmup_and_validation(tmp_path):
    template, token = joint_budget()
    template.update(max_steps=30000, max_finite_passes=4, eval_every=250)
    recipe, budget = distillation_training.compile_recipe(template, token, tmp_path)
    assert budget['rows_per_complete_pass'] == 1248
    assert budget['updates_per_complete_pass'] == 39
    assert recipe['max_steps'] == budget['max_updates'] == 156
    assert recipe['epochs'] == 4 and recipe['min_steps'] == 39 and recipe['eval_every'] == 39
    assert budget['configured_max_updates'] == 30000 and budget['finite_pass_limit_reduced_max_updates']
    assert not budget['last_allocated_pass_may_be_partial_at_update_cap']
    assert recipe['warmup_fraction'] == .05 and recipe['learning_rate'] == template['learning_rate']
    assert budget['accepted_training_rows'] == 1000 and not budget['paper_first_round_150000_label_scale_reached']
    assert not tmp_path.joinpath('train-index').exists()


def test_explicit_update_ceiling_still_limits_a_partial_last_pass(tmp_path):
    template, token = joint_budget()
    template.update(max_steps=100, max_finite_passes=4, eval_every=250)
    recipe, budget = distillation_training.compile_recipe(template, token, tmp_path)
    assert recipe['max_steps'] == 100 and recipe['epochs'] == 3
    assert budget['updates_for_requested_complete_passes'] == 156
    assert budget['configured_max_updates'] == 100 and not budget['finite_pass_limit_reduced_max_updates']
    assert budget['last_allocated_pass_may_be_partial_at_update_cap']
    assert recipe['min_steps'] == recipe['eval_every'] == 39


@pytest.mark.parametrize('value', [None, False, True, 0, -1, 1.5, '4', float('inf')])
def test_invalid_finite_pass_ceiling_never_creates_a_training_job(tmp_path, value):
    template = profile(); template['max_finite_passes'] = value
    with pytest.raises(ValueError, match='positive integer'):
        distillation_training.compile_recipe(template, budget_token(64), tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('value', [None, False, 0, -1])
def test_invalid_validation_interval_cannot_hide_all_intermediate_checks(tmp_path, value):
    template = profile(); template['eval_every'] = value
    with pytest.raises(ValueError, match='validation interval'):
        distillation_training.compile_recipe(template, budget_token(64), tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


def test_memory_profile_reaches_real_native_row_config_compiler(tmp_path, monkeypatch):
    source, _, data, cache, _, _, _ = cpu_flow(tmp_path, monkeypatch)
    token = budget_token(32, 2)
    token['config']['data'] = str(data)
    token['cache_header_only_input']['path'] = str(cache)
    recipe, budget = distillation_training.compile_recipe(profile(), token, tmp_path / 'unused')
    saved = {'model_path': str(tmp_path / 'model'), 'feature_path': str(cache), 'expert_feature_depths': [0, 3]}
    compiled = prepare_config(saved, recipe, source, tmp_path / 'unused/sft')
    settings = training_memory_settings(compiled)
    assert compiled['distributed_strategy'] == 'fsdp2'
    assert compiled['fsdp_shard_paths'] == ['bridges.*', 'base.model.layers.*', 'base']
    assert compiled['decoder_gradient_checkpointing'] is True
    assert compiled['autocast_cache_enabled'] is False and compiled['optimizer_state_sharding'] == 'none'
    assert compiled['decoder_training'] == 'full' and compiled['trainable_parameter_dtype'] == 'float32'
    assert compiled['batch_size'] == 32 and compiled['micro_batch_size'] == 1 and compiled['ddp_world_size'] == 4
    assert compiled['steps'] == recipe['max_steps'] == budget['max_updates'] == 4
    assert compiled['learning_rate'] == 1e-4 and compiled['decoder_learning_rate'] == 1e-5
    assert settings['distributed_strategy'] == 'fsdp2'
    assert not (tmp_path / 'unused').exists()


def test_round_indexes_and_final_actual_finite_state_use_compiled_cap(tmp_path, monkeypatch):
    source, recipe_path, token, runner, _ = round_setup(tmp_path, monkeypatch)
    template = json.loads(recipe_path.read_text())
    template.update(max_steps=30000, max_finite_passes=4, eval_every=250)
    atomic_json(recipe_path, template)
    output = tmp_path / 'round'
    result = distillation_training.run_round(source, recipe_path, token, output)
    assert result['actual_steps'] == 4 and result['training_data_state']['epoch'] == 4
    assert result['training_data_state']['unique_rows_consumed'] == 128
    compiled = json.loads((output / 'recipe.json').read_text())
    assert compiled['max_steps'] == 4 and compiled['eval_every'] == 1
    assert len(runner.jobs) == 4
    assert distillation_training.run_round(source, recipe_path, token, output, resume=True) == result
    assert len(runner.jobs) == 4
