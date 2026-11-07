"""Real finite indexes and native gates with controlled, CPU-only worker output."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
import torch

import test_auxiliary_policy_evaluation as evaluation_control
import test_auxiliary_policy_preflight as inputs_control
import test_auxiliary_policy_training as training_control
from test_sft_pipeline import install_runner
from test_sft_replay_preflight import IndexedProbeRunner
from xqgeneral import auxiliary_policy_preflight as auxiliary
from xqgeneral import auxiliary_policy_training as training, sft_preflight
from xqgeneral import clean_sft_evaluation as evaluation
from xqgeneral.distillation_preflight import completed_parent
from xqgeneral.evidence import atomic_json, load_jsonl
from xqgeneral.finite_training import prepare_finite_epochs
from xqgeneral.full_decoder_topology import THREE_WORKERS, auxiliary_topology
from xqgeneral.training_index import IndexedTrainingRows, build_index


def template():
    return json.loads(Path('configs/auxiliary-policy-sft-three-gpu-v1.json').read_text())


@pytest.mark.parametrize('change', [
    dict(batch_size=32), dict(ddp_world_size=4), dict(micro_batch_size=2),
    dict(distributed_strategy='ddp'), dict(auxiliary_topology_profile='unknown'),
    dict(auxiliary_topology_profile=None), dict(ddp_world_size=3.0),
])
def test_three_worker_profile_rejects_incompatible_geometry_before_launch(change):
    with pytest.raises(ValueError): auxiliary_topology(dict(template(), **change))


def test_three_workers_need_explicit_profile_and_keep_legacy_four_worker_contract():
    old = inputs_control.template()
    assert auxiliary_topology(old)['world_size'] == 4
    with pytest.raises(ValueError): auxiliary_topology(dict(old, ddp_world_size=3, batch_size=30))


def test_compiled_budget_matches_actual_three_rank_rows_and_partial_tail(tmp_path, monkeypatch):
    args, _, _ = inputs_control.fixture(tmp_path, monkeypatch)
    auxiliary.prepare(**args)
    token = json.loads((args['output'] / 'manifest.json').read_text())
    recipe, config, budget = auxiliary.compile_recipe(token, template(), tmp_path / 'round')
    assert config['steps'] == 48 and config['min_steps'] == budget['steps_per_pass'] == 12
    assert budget['distributed_tail_repeated_rows_per_pass'] == 1
    build_index([args['data'] / 'train.jsonl'], recipe['row_index_paths']['train'])
    rows = IndexedTrainingRows(recipe['row_index_paths']['train'], [args['data'] / 'train.jsonl'], recipe['mixture'])
    try:
        finite = prepare_finite_epochs(rows, recipe['mixture'], config, 3)
        observed = Counter()
        for step in range(12):
            for rank in range(3):
                indices, report = finite.batch_indices(step, rank)
                observed.update(rows.row_metadata(i)['id'] for i in indices)
        assert len(observed) == 350 and sum(observed.values()) == 351
        assert report['global_valid_rows'] == 20 and report['global_padding_rows'] == 1
        assert finite.contract()['padded_rows_per_epoch'] == budget['distributed_tail_repeated_rows_per_pass']
        assert set(observed) == {r['id'] for r in load_jsonl(args['data'] / 'train.jsonl')}
    finally: rows.close()
    assert sum(recipe['validation_rows_by_stage'].values()) == recipe['validation_examples'] == 7


def three_worker_probe(tmp_path, monkeypatch):
    args, saved, _ = inputs_control.fixture(tmp_path, monkeypatch)
    saved['decoder_training'] = 'full'
    state = {'bridges.0.weight': torch.ones(2, 3), 'board_weight': torch.ones(2, 3),
             'base.decoder.weight': torch.ones(2, 3)}
    torch.save(dict(trainable=state, config=saved), args['source'])
    atomic_json(args['source'].parent / 'config.json', saved)
    atomic_json(args['parent_completion'], dict(status='complete', controlled=True))
    data = args['data']; rows = load_jsonl(data / 'validation.jsonl')
    base = next(row for row in rows if row['stage'] == 'explanation')
    rows += [dict(base, id=base['id'] + '-probe-only-' + str(i)) for i in range(5)]
    inputs_control.corpus_control.rebind(data, 'validation.jsonl', rows)
    proof_path = data / 'manifest.json'; proof = json.loads(proof_path.read_text())
    proof['verification']['validation_rows_by_stage']['explanation'] += 5
    proof['verification']['validation_question_rows'] += 5
    atomic_json(proof_path, proof)
    auxiliary.prepare(**args)
    token_path = args['output'] / 'manifest.json'; token = json.loads(token_path.read_text())
    recipe, _, _ = auxiliary.compile_recipe(token, template(), tmp_path / 'formal')
    recipe_path = tmp_path / 'recipe.json'; atomic_json(recipe_path, recipe)
    monkeypatch.setattr(sft_preflight, 'pinned_decoder_assets', auxiliary.pinned_decoder_assets)
    original = install_runner(monkeypatch)
    runner = IndexedProbeRunner(original.original, [])
    monkeypatch.setattr(subprocess, 'run', runner)
    root = tmp_path / 'probe'
    result = sft_preflight.run_preflight(args['source'], recipe_path, token_path, root)
    return root, result, args['source'], recipe_path, token_path, runner


def test_native_three_rank_probe_checks_all_workers_finite_state_and_exact_resume(tmp_path, monkeypatch):
    root, result, source, recipe, token, runner = three_worker_probe(tmp_path, monkeypatch)
    assert result['world_size'] == 3 and result['exact_resume']['world_sizes'] == [3, 3]
    assert result['exact_resume']['mismatched_fields'] == []
    assert result['actual_finite_epoch_state_verified']
    assert result['longest_actual_train_validation_probe']['train']['rows_by_stage'].keys() == set(auxiliary.ALL_STAGES)
    assert len(result['longest_actual_train_validation_probe']['train']['ids']) == 30
    assert len(result['actual_gpu_worker_resources']) == 9
    assert set(r['rank'] for r in result['actual_gpu_worker_resources']) == {0, 1, 2}
    assert all('--nproc_per_node=3' in command for command in runner.jobs)
    proof = json.loads((root / 'manifest.json').read_text())
    assert proof['kind'] == 'three_gpu_auxiliary_policy_sft_preflight'
    assert sft_preflight.run_preflight(source, recipe, token, root, resume=True) == result
    assert len(runner.jobs) == 3


@pytest.mark.parametrize('damage', ['missing_rank', 'world', 'moments', 'epoch', 'rng'])
def test_self_consistent_three_rank_probes_cannot_hide_invalid_native_state(tmp_path, monkeypatch, damage):
    root, _, _, _, _, _ = three_worker_probe(tmp_path, monkeypatch)
    expected = torch.load(root / 'continuous/latest.pt', weights_only=True)
    if damage == 'missing_rank': (root / 'continuous/resources-step-4-rank-2.json').unlink()
    else:
        for run in ['continuous', 'resumed']:
            path = root / run / 'latest.pt'; saved = torch.load(path, weights_only=True)
            if damage == 'world': saved['world_size'] = 4
            elif damage == 'moments': saved['optimizer']['state'][0]['exp_avg'].fill_(float('inf'))
            elif damage == 'epoch': saved['data_state']['epoch'] += 1
            else: saved['cuda_rng_states'].pop()
            torch.save(saved, path)
    with pytest.raises((ValueError, FileNotFoundError)):
        sft_preflight.verify_updated_probe(root, root / 'zero-init.pt', expected['input_hashes'],
                                           expected['config'], expected['code'])


def test_three_worker_completion_and_evaluation_preserve_separate_round_counts(tmp_path, monkeypatch):
    root, config, recipe, result = evaluation_control.native_completed_auxiliary(tmp_path, monkeypatch, world=3)
    assert result['actual_three_gpu_full_decoder_preflight_passed'] is True
    assert result['actual_four_gpu_full_decoder_preflight_passed'] is False
    assert result['world_size'] == 3 and result['global_batch_size'] == 30 and result['actual_steps'] == 48
    checkpoint, proof = evaluation.checked_sft(root, config, recipe)
    assert checkpoint == Path(result['selected_checkpoint']) and proof['verification'] == result
    assert result['trained_search_distillation_rounds'] == 2 and result['trained_auxiliary_policy_rounds'] == 1


@pytest.mark.parametrize('damage', ['wrong_probe_world', 'wrong_probe_kind'])
def test_four_worker_probe_cannot_authorize_three_worker_formal_training(tmp_path, monkeypatch, damage):
    args, calls = training_control.setup(tmp_path, monkeypatch, damage, world=3)
    with pytest.raises(ValueError): training.run_round(**args)
    assert calls == ['GPU_preflight'] and not (args['output'] / 'manifest.json').exists()


@pytest.mark.parametrize('change', [
    dict(world_size=4), dict(actual_three_gpu_full_decoder_preflight_passed=False),
    dict(actual_four_gpu_full_decoder_preflight_passed=True),
    dict(auxiliary_topology_profile='unknown'), dict(global_batch_size=32),
])
def test_three_worker_parent_rejects_relabelled_geometry_and_capacity(tmp_path, monkeypatch, change):
    root, _, _, result = evaluation_control.native_completed_auxiliary(tmp_path, monkeypatch, world=3)
    path = root / 'manifest.json'; proof = json.loads(path.read_text())
    proof['verification'].update(deepcopy(change)); atomic_json(path, proof)
    with pytest.raises(ValueError): completed_parent(Path(result['selected_checkpoint']), path)
