"""Two-worker native orchestration controls; GPU producers remain CPU fixtures."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

import test_auxiliary_policy_evaluation as evaluation_control
import test_auxiliary_policy_preflight as inputs_control
import test_auxiliary_policy_training as training_control
import test_auxiliary_search_replay as replay_control
import test_auxiliary_three_workers as probe_control
from xqgeneral import auxiliary_policy_preflight as auxiliary
from xqgeneral import auxiliary_policy_training as training, clean_sft_evaluation as evaluation
from xqgeneral import distillation_training, sft_preflight
from xqgeneral.distillation_preflight import completed_parent
from xqgeneral.evidence import atomic_json, load_jsonl
from xqgeneral.finite_training import prepare_finite_epochs
from xqgeneral.full_decoder_topology import auxiliary_topology, search_topology
from xqgeneral.training_index import IndexedTrainingRows, build_index


def template(kind='auxiliary'):
    filename = ('auxiliary-policy-sft-two-gpu-v1.json' if kind == 'auxiliary' else
                'search-distillation-sft-two-gpu-with-auxiliary-v1.json')
    return json.loads((Path('configs') / filename).read_text())


@pytest.mark.parametrize('kind,topology', [('auxiliary', auxiliary_topology), ('search', search_topology)])
def test_two_workers_require_their_explicit_full_parameter_profile(kind, topology):
    actual = topology(template(kind))
    assert actual['world_size'] == 2 and actual['global_batch_size'] == 30
    assert actual['preflight_kind'] == 'two_gpu_' + ('auxiliary_policy' if kind == 'auxiliary' else 'search_distillation') + '_sft_preflight'
    assert actual['completion_flag'] == 'actual_two_gpu_full_decoder_preflight_passed'


@pytest.mark.parametrize('kind,topology', [('auxiliary', auxiliary_topology), ('search', search_topology)])
@pytest.mark.parametrize('change', [dict(ddp_world_size=3), dict(ddp_world_size=4), dict(ddp_world_size=2.0),
    dict(batch_size=32), dict(micro_batch_size=2), dict(distributed_strategy='ddp')])
def test_two_worker_profile_cannot_allocate_a_third_rank_or_change_batch_geometry(kind, topology, change):
    with pytest.raises(ValueError):
        topology(dict(template(kind), **change))


def test_auxiliary_two_rank_finite_budget_visits_every_actual_source_row(tmp_path, monkeypatch):
    args, _, _ = inputs_control.fixture(tmp_path, monkeypatch)
    auxiliary.prepare(**args)
    token = json.loads((args['output'] / 'manifest.json').read_text())
    recipe, config, budget = auxiliary.compile_recipe(token, template(), tmp_path / 'round')
    assert config['steps'] == 48 and config['min_steps'] == 12
    assert budget['distributed_tail_repeated_rows_per_pass'] == 0
    build_index([args['data'] / 'train.jsonl'], recipe['row_index_paths']['train'])
    rows = IndexedTrainingRows(recipe['row_index_paths']['train'], [args['data'] / 'train.jsonl'], recipe['mixture'])
    try:
        finite = prepare_finite_epochs(rows, recipe['mixture'], config, 2)
        observed = Counter()
        for step in range(12):
            for rank in range(2):
                indices, report = finite.batch_indices(step, rank)
                observed.update(rows.row_metadata(i)['id'] for i in indices)
        assert set(observed) == {row['id'] for row in load_jsonl(args['data'] / 'train.jsonl')}
        assert len(observed) == sum(observed.values()) == 350
        assert report['global_valid_rows'] == 20 and report['global_padding_rows'] == 0
    finally:
        rows.close()
    assert recipe['validation_examples'] == 7


def two_worker_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_control, 'template', template)
    return probe_control.three_worker_probe(tmp_path, monkeypatch)


def test_native_two_rank_probe_checks_actual_rank_resources_and_exact_resume(tmp_path, monkeypatch):
    root, result, source, recipe, token, runner = two_worker_probe(tmp_path, monkeypatch)
    assert result['world_size'] == 2 and result['exact_resume']['world_sizes'] == [2, 2]
    assert result['exact_resume']['mismatched_fields'] == [] and result['actual_finite_epoch_state_verified']
    assert len(result['actual_gpu_worker_resources']) == 6
    assert {item['rank'] for item in result['actual_gpu_worker_resources']} == {0, 1}
    assert all('--nproc_per_node=2' in command for command in runner.jobs)
    assert json.loads((root / 'manifest.json').read_text())['kind'] == 'two_gpu_auxiliary_policy_sft_preflight'
    assert sft_preflight.run_preflight(source, recipe, token, root, resume=True) == result
    assert len(runner.jobs) == 3


@pytest.mark.parametrize('damage', ['missing_rank', 'world', 'moments', 'epoch', 'rng'])
def test_two_rank_native_probe_rejects_missing_resources_and_invalid_full_state(tmp_path, monkeypatch, damage):
    root, _, _, _, _, _ = two_worker_probe(tmp_path, monkeypatch)
    expected = torch.load(root / 'continuous/latest.pt', weights_only=True)
    if damage == 'missing_rank':
        (root / 'continuous/resources-step-4-rank-1.json').unlink()
    else:
        for name in ['continuous', 'resumed']:
            path = root / name / 'latest.pt'; saved = torch.load(path, weights_only=True)
            if damage == 'world': saved['world_size'] = 3
            elif damage == 'moments': saved['optimizer']['state'][0]['exp_avg'].fill_(float('inf'))
            elif damage == 'epoch': saved['data_state']['epoch'] += 1
            else: saved['cuda_rng_states'].pop()
            torch.save(saved, path)
    with pytest.raises((ValueError, FileNotFoundError)):
        sft_preflight.verify_updated_probe(root, root / 'zero-init.pt', expected['input_hashes'],
                                          expected['config'], expected['code'])


def test_completed_two_worker_auxiliary_model_reopens_and_preserves_round_counts(tmp_path, monkeypatch):
    root, config, recipe, result = evaluation_control.native_completed_auxiliary(tmp_path, monkeypatch, world=2)
    assert result['actual_two_gpu_full_decoder_preflight_passed'] is True
    assert result['actual_four_gpu_full_decoder_preflight_passed'] is False
    assert result['world_size'] == 2 and result['global_batch_size'] == 30
    checkpoint, proof = evaluation.checked_sft(root, config, recipe)
    assert checkpoint == Path(result['selected_checkpoint']) and proof['verification'] == result
    assert result['trained_search_distillation_rounds'] == 2 and result['trained_auxiliary_policy_rounds'] == 1


@pytest.mark.parametrize('damage', ['wrong_probe_world', 'wrong_probe_kind'])
def test_other_worker_capacity_proofs_cannot_authorize_two_worker_training(tmp_path, monkeypatch, damage):
    args, calls = training_control.setup(tmp_path, monkeypatch, damage, world=2)
    with pytest.raises(ValueError): training.run_round(**args)
    assert calls == ['GPU_preflight'] and not (args['output'] / 'manifest.json').exists()


@pytest.mark.parametrize('change', [dict(world_size=3), dict(actual_two_gpu_full_decoder_preflight_passed=False),
    dict(actual_four_gpu_full_decoder_preflight_passed=True), dict(actual_three_gpu_full_decoder_preflight_passed=True),
    dict(auxiliary_topology_profile='full_decoder_auxiliary_three_workers_v1')])
def test_completed_two_worker_parent_rejects_other_worker_claims(tmp_path, monkeypatch, change):
    root, _, _, result = evaluation_control.native_completed_auxiliary(tmp_path, monkeypatch, world=2)
    path = root / 'manifest.json'; proof = json.loads(path.read_text())
    proof['verification'].update(deepcopy(change)); atomic_json(path, proof)
    with pytest.raises(ValueError): completed_parent(Path(result['selected_checkpoint']), path)


def test_two_worker_search_keeps_all_six_replay_tasks_and_original_seven_stage_validation(tmp_path):
    _, token = replay_control.budget()
    value = template('search'); value['replay_data_paths'] = token['config']['replay_data_paths']
    recipe, budget = distillation_training.compile_recipe(value, token, tmp_path / 'round')
    assert budget['world_size'] == 2 and budget['global_batch_size'] == 30
    assert budget['rows_per_complete_pass'] == 271 and budget['distributed_tail_repeated_rows_per_pass'] == 1
    assert budget['finite_mixture']['selected_rows_by_stage'] == {'explanation': 31, **dict.fromkeys(replay_control.replay.STAGES, 40)}
    assert set(recipe['validation_stages']) == set(replay_control.ALL_STAGES)
    assert recipe['loss_normalization'] == 'token' and budget['auxiliary_policy_replay_included']
