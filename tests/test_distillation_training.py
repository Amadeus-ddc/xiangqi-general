"""Controlled orchestration fixtures; no real GPU model or mined labels."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from test_sft_pipeline import ControlledFullModel, ControlledRunner, install_runner, setup_data
from test_search_collection import query, response, TEACHER
from xqgeneral import distillation_preflight, distillation_training, modeling, sft_pipeline, sft_preflight
from xqgeneral.collect_search import consolidated_label
from xqgeneral.evidence import atomic_json, digest, history_key, iter_jsonl, manifest, write_jsonl
from xqgeneral.training import training_input_paths
from xqgeneral.finite_training import prepare_finite_epochs
from xqgeneral.training_index import IndexedTrainingRows


def template():
    return json.loads(Path('configs/search-distillation-sft-paper-v1.json').read_text())


def budget_token(train=150000, validation=384):
    return {'status': 'complete', 'kind': distillation_preflight.TOKEN_KIND,
        'config': {'data': 'actual-label-path'}, 'cache_header_only_input': {'path': 'actual-cache-path'},
        'verification': {'by_split': {'train': train, 'validation': validation, 'test': 512},
            'token_limit': 2048, 'all_train_validation_supervision_fits_without_truncation': True,
            'independent_test_used_for_training_or_selection': False}}


def test_paper_recipe_compiles_one_pass_floor_and_explicit_partial_cap(tmp_path):
    recipe, budget = distillation_training.compile_recipe(template(), budget_token(), tmp_path)
    assert recipe['batch_size'] == 32 and recipe['micro_batch_size'] == 1
    assert recipe['learning_rate'] == 1e-4 and recipe['decoder_learning_rate'] == recipe['token_learning_rate'] == 1e-5
    assert recipe['max_tokens'] == 2048 and recipe['max_steps'] == 30000
    assert recipe['lr_schedule'] == 'constant' and recipe['warmup_fraction'] == .05
    assert recipe['epochs'] == 7 and recipe['min_steps'] == 4688
    assert budget['updates_per_complete_pass'] == 4688
    assert budget['last_allocated_pass_may_be_partial_at_update_cap']
    assert budget['paper_first_round_150000_label_scale_reached']
    assert recipe['validation_examples'] == 384
    assert set(recipe['row_index_paths']) == {'train', 'validation'}
    assert 'test-index' not in json.dumps(recipe)


@pytest.mark.parametrize('problem', ['kind', 'status', 'fit', 'test_used', 'small', 'no_validation',
    'cap', 'batch', 'dtype', 'profile', 'token_limit', 'warmup', 'boolean_steps'])
def test_invalid_or_empty_contract_cannot_compile_a_training_job(tmp_path, problem):
    r, t = template(), budget_token()
    if problem == 'kind': t['kind'] = 'prepared_search_training_roots'
    elif problem == 'status': t['status'] = 'pending'
    elif problem == 'fit': t['verification']['all_train_validation_supervision_fits_without_truncation'] = False
    elif problem == 'test_used': t['verification']['independent_test_used_for_training_or_selection'] = True
    elif problem == 'small': t['verification']['by_split']['train'] = 31
    elif problem == 'no_validation': t['verification']['by_split']['validation'] = 0
    elif problem == 'cap': r['max_steps'] = 100
    elif problem == 'batch': r['batch_size'] = 16
    elif problem == 'dtype': r['trainable_parameter_dtype'] = 'bfloat16'
    elif problem == 'profile': r['training_data_profile'] = 'replacement-sampling'
    elif problem == 'token_limit': r['max_tokens'] = 1024
    elif problem == 'warmup': r['warmup_fraction'] = float('nan')
    else: r['max_steps'] = True
    with pytest.raises(ValueError):
        distillation_training.compile_recipe(r, t, tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


def test_small_control_budget_is_reported_below_paper_round_scale(tmp_path):
    r = template(); r['max_steps'] = 8
    recipe, budget = distillation_training.compile_recipe(r, budget_token(64, 16), tmp_path)
    assert recipe['epochs'] == 4 and recipe['min_steps'] == 2
    assert not budget['paper_first_round_150000_label_scale_reached']


class RoundRunner(ControlledRunner):
    def __call__(self, command, **kwargs):
        if 'xqgeneral.sft' in command and self.failure == 'formal_interrupt':
            self.jobs.append(command)
            raise subprocess.CalledProcessError(1, command)
        result = super().__call__(command, **kwargs)
        if 'xqgeneral.sft_probe_worker' in command:
            config = json.loads(Path(command[command.index('--config') + 1]).read_text())
            p = Path(config['output']) / 'latest.pt'
        elif 'xqgeneral.sft' in command:
            root = Path(command[command.index('--output') + 1])
            config = json.loads((root / 'config.json').read_text())
            p = root / 'adapter.pt'
            metrics = json.loads((root / 'metrics.json').read_text())
            metrics['global_batch_size'] = config['batch_size']
            atomic_json(root / 'metrics.json', metrics)
            proof = json.loads((root / 'manifest.json').read_text())
            proof['verification'] = metrics
        else:
            return result
        saved = torch.load(p, weights_only=True)
        saved['input_hashes'] = {str(q): digest(q) for q in training_input_paths(config)}
        indexed = IndexedTrainingRows(config['row_index_paths']['train'],
            [Path(config['data_path']) / 'train.jsonl'], config['mixture'])
        try:
            epochs = prepare_finite_epochs(indexed, config['mixture'], config, 4)
            saved['data_state'] = epochs.state_at(saved.get('step', config['steps']))
        finally:
            indexed.close()
        torch.save(saved, p)
        if 'xqgeneral.sft' in command:
            metrics['training_data_state'] = saved['data_state']
            if self.failure == 'finite_state': metrics['training_data_state']['epoch'] = 99
            atomic_json(root / 'metrics.json', metrics)
            proof['verification'] = metrics
            proof['inputs'] = {str(q): {'sha256': digest(q), 'bytes': Path(q).stat().st_size}
                               for q in training_input_paths(config)}
            proof['outputs'][str(p)] = {'sha256': digest(p), 'bytes': p.stat().st_size}
            atomic_json(root / 'manifest.json', proof)
        return result


def round_setup(tmp_path, monkeypatch):
    curriculum, old_recipe, old_token = setup_data(tmp_path)
    original_token = json.loads(old_token.read_text())
    for split in ['train', 'validation']:
        path = Path(original_token['config']['data']) / (split + '.jsonl')
        rows = list(iter_jsonl(path))
        for row in rows:
            row['feature_key'] = 'controlled-history-' + row['id']
        write_jsonl(path, rows)
        original_token['inputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(old_token, original_token)
    original = install_runner(monkeypatch)
    initial = tmp_path / 'initial'
    sft_pipeline.run_pipeline(curriculum, 'owned', old_recipe, old_token, initial)
    source = initial / 'sft/adapter.pt'
    token = json.loads(old_token.read_text())
    new_cache = tmp_path / 'controlled-extended-cache.pt'
    shutil.copyfile(token['cache_header_only_input']['path'], new_cache)
    token['kind'] = distillation_preflight.TOKEN_KIND
    token['config'].update(source=str(source), parent_completion=str(initial / 'manifest.json'))
    token['cache_header_only_input'] = {'path': str(new_cache), 'sha256': digest(new_cache), 'bytes': new_cache.stat().st_size}
    token['inputs'][str(new_cache)] = {k: token['cache_header_only_input'][k] for k in ['sha256', 'bytes']}
    token['verification'].update(token_limit=2048, by_split={'train': 32, 'validation': 16, 'test': 0},
        independent_test_used_for_training_or_selection=False,
        all_full_native_histories_facts_branches_and_explicit_prose_moves_checked=True,
        protected_validation_and_test_bytes_unchanged=True, all_label_feature_contexts_present=True,
        original_selected_branch_evaluation_filter_rechecked=True,
        parent_full_decoder_completion_verified=True, all_original_feature_keys_and_tensor_references_preserved=True)
    token_path = tmp_path / 'controlled-token.json'; atomic_json(token_path, token)
    recipe = template(); recipe.update(max_steps=8, eval_every=2)
    recipe_path = tmp_path / 'round-template.json'; atomic_json(recipe_path, recipe)
    runner = RoundRunner(original.original)
    monkeypatch.setattr(distillation_training.subprocess, 'run', runner)
    def load_selected_parent(config, **kwargs):
        state = torch.load(config['init_from'], weights_only=True)['trainable']
        model = ControlledFullModel(state)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(state[name])
        return model, None
    monkeypatch.setattr(modeling, 'load_model', load_selected_parent)
    return source, recipe_path, token_path, runner, initial


def test_complete_round_controls_cover_full_batch_indexes_and_selected_parent(tmp_path, monkeypatch):
    source, recipe, token, runner, _ = round_setup(tmp_path, monkeypatch)
    root = tmp_path / 'round'
    result = distillation_training.run_round(source, recipe, token, root)
    assert result['actual_distillation_sft_completed'] and result['actual_steps'] == 8
    assert result['global_batch_size'] == 32 and result['trained_search_distillation_rounds'] == 1
    assert not result['paper_first_round_150000_label_scale_reached']
    assert not result['strength_and_coaching_gain_evaluated'] and not result['strong_play_or_reliable_coaching_proven']
    probe = json.loads((root / 'preflight/verification.json').read_text())
    assert len(probe['longest_actual_train_validation_probe']['train']['ids']) == 32
    assert len(probe['longest_actual_train_validation_probe']['validation']['ids']) == 12
    assert probe['full_decoder_initialization_preserved_all_parent_tensors']
    assert not probe['full_decoder_initialization_preserved_all_foundation_tensors']
    assert json.loads((root / 'preflight/manifest.json').read_text())['kind'] == 'four_gpu_search_distillation_sft_preflight'
    assert len(runner.jobs) == 4
    assert distillation_training.run_round(source, recipe, token, root, resume=True) == result
    assert len(runner.jobs) == 4
    _, completed = distillation_preflight.completed_parent(root / 'sft/adapter.pt', root / 'manifest.json')
    assert completed['verification']['trained_search_distillation_rounds'] == 1


@pytest.mark.parametrize('problem', ['wrong_source', 'no_native', 'test_used', 'changed_parent', 'partial_parent'])
def test_distillation_rejects_missing_or_changed_parent_before_gpu(tmp_path, monkeypatch, problem):
    source, recipe, token_path, runner, initial = round_setup(tmp_path, monkeypatch)
    token = json.loads(token_path.read_text())
    if problem == 'wrong_source': token['config']['source'] = 'foreign-model.pt'
    elif problem == 'no_native': token['verification']['all_full_native_histories_facts_branches_and_explicit_prose_moves_checked'] = False
    elif problem == 'test_used': token['verification']['independent_test_used_for_training_or_selection'] = True
    elif problem == 'changed_parent': source.write_bytes(source.read_bytes() + b'changed')
    else:
        p = initial / 'manifest.json'; d = json.loads(p.read_text()); d['verification']['initial_explanation_sft_completed'] = False; atomic_json(p, d)
    atomic_json(token_path, token)
    with pytest.raises(ValueError):
        distillation_training.run_round(source, recipe, token_path, tmp_path / 'round')
    assert not runner.jobs


def test_failed_real_preflight_never_starts_formal_round_or_publishes_completion(tmp_path, monkeypatch):
    source, recipe, token, runner, _ = round_setup(tmp_path, monkeypatch)
    runner.failure = 'preflight'
    root = tmp_path / 'round'
    with pytest.raises(Exception):
        distillation_training.run_round(source, recipe, token, root)
    assert not any('xqgeneral.sft' in c for c in runner.jobs)
    assert not (root / 'manifest.json').exists()
    assert json.loads((root / 'state.json').read_text())['stage'] == 'four_gpu_full_decoder_distillation_preflight'


def test_round_resume_reuses_completed_probe_and_preserves_index_and_recipe_bytes(tmp_path, monkeypatch):
    source, recipe, token, runner, _ = round_setup(tmp_path, monkeypatch)
    runner.failure = 'formal_interrupt'; root = tmp_path / 'round'
    with pytest.raises(subprocess.CalledProcessError):
        distillation_training.run_round(source, recipe, token, root)
    assert len(runner.jobs) == 4 and not (root / 'manifest.json').exists()
    preserved = [root / 'recipe.json', root / 'budget.json', root / 'preflight/manifest.json']
    preserved += [p for p in (root / 'train-index').iterdir() if p.is_file()]
    before = {p: (digest(p), p.stat().st_mtime_ns) for p in preserved}
    runner.failure = None
    result = distillation_training.run_round(source, recipe, token, root, resume=True)
    assert result['actual_distillation_sft_completed'] and len(runner.jobs) == 5
    assert all((digest(p), p.stat().st_mtime_ns) == value for p, value in before.items())


@pytest.mark.parametrize('problem', ['selected_state', 'changed_formal_labels', 'finite_state'])
def test_false_formal_completion_is_not_published(tmp_path, monkeypatch, problem):
    source, recipe, token, runner, _ = round_setup(tmp_path, monkeypatch)
    runner.failure = problem; root = tmp_path / 'round'
    with pytest.raises(ValueError):
        distillation_training.run_round(source, recipe, token, root)
    assert not (root / 'manifest.json').exists()
    assert json.loads((root / 'state.json').read_text())['formal_distillation_completion_proven'] is False


def test_native_label_rechecks_real_rules_and_refuses_changed_history_identity():
    row = consolidated_label(query(), response(), TEACHER)
    row['history'] = [row['fen']]; row['feature_key'] = history_key(row['history'])
    assert distillation_preflight.native_label(row) == row['feature_key']
    row['feature_key'] = 'wrong-full-history'
    with pytest.raises(ValueError):
        distillation_preflight.native_label(row)


@pytest.mark.parametrize('problem', ['missing', 'numeric_claim', 'threshold'])
def test_native_label_refuses_changed_pre_rewrite_filter(problem):
    row = consolidated_label(query(), response(), TEACHER)
    row['history'] = [row['fen']]; row['feature_key'] = history_key(row['history'])
    if problem == 'missing': row.pop('search_evaluation_filter')
    elif problem == 'numeric_claim': row['search_evaluation_filter']['absolute_winrate_error'] = .5
    else: row['search_evaluation_filter']['max_absolute_winrate_error_exclusive'] = .1
    with pytest.raises(ValueError, match='original student evaluation'):
        distillation_preflight.native_label(row)
