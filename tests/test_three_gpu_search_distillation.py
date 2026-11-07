"""Native CPU contracts with controlled GPU producers; no real training/quality claim."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

import test_distillation_training as training_control
import test_distillation_joint_replay as replay_control
import test_post_distillation_evaluation as evaluation_control
from test_clean_sft_evaluation import ControlledEvaluation
from xqgeneral import clean_sft_evaluation as evaluation, distillation_training as training, sft_preflight
from xqgeneral.distillation_preflight import completed_parent, AUXILIARY_ROUND_KIND
from xqgeneral.evidence import atomic_json, load_jsonl, write_jsonl
from xqgeneral.finite_training import prepare_finite_epochs
from xqgeneral.full_decoder_topology import THREE_SEARCH_WORKERS, search_topology
from xqgeneral.training_index import build_index, IndexedTrainingRows


def template():
    value = json.loads(Path('configs/search-distillation-sft-paper-v1.json').read_text())
    value.update(search_topology_profile=THREE_SEARCH_WORKERS, ddp_world_size=3, batch_size=30,
                 distributed_strategy='fsdp2', fsdp_shard_paths=['bridges.*', 'base'])
    return value


@pytest.mark.parametrize('change', [dict(ddp_world_size=4), dict(ddp_world_size=3.0), dict(batch_size=32),
    dict(micro_batch_size=2), dict(distributed_strategy='ddp'), dict(search_topology_profile=None),
    dict(search_topology_profile='unknown'), dict(search_topology_profile='full_decoder_auxiliary_three_workers_v1')])
def test_search_geometry_requires_its_explicit_three_worker_profile(change):
    with pytest.raises(ValueError): search_topology(dict(template(), **change))


def test_existing_four_worker_profile_does_not_implicitly_switch_to_three():
    old = training_control.template(); assert search_topology(old)['world_size'] == 4
    with pytest.raises(ValueError): search_topology(dict(old, ddp_world_size=3, batch_size=30))


def test_three_worker_paper_budget_uses_actual_global_batch_and_one_pass_floor(tmp_path):
    recipe, budget = training.compile_recipe(template(), training_control.budget_token(), tmp_path)
    assert recipe['epochs'] == 6 and recipe['min_steps'] == 5000 and recipe['max_steps'] == 30000
    assert budget['global_batch_size'] == 30 and budget['world_size'] == 3
    assert budget['updates_per_complete_pass'] == 5000
    assert budget['distributed_tail_repeated_rows_per_pass'] == 0
    assert not budget['mixture_derived_from_actual_token_verified_counts']


def test_full_source_three_worker_template_derives_all_actual_rows_and_token_loss(tmp_path):
    recipe = json.loads(Path('configs/search-distillation-sft-three-gpu-v1.json').read_text())
    _, token = replay_control.joint_budget(train=31, old=40)
    recipe['replay_data_paths'] = token['config']['replay_data_paths']
    compiled, budget = training.compile_recipe(recipe, token, tmp_path / 'round')
    expected = {'explanation': 31, **dict.fromkeys(replay_control.STAGES, 40)}
    assert budget['finite_mixture']['selected_rows_by_stage'] == expected
    assert budget['rows_per_complete_pass'] == 191 and budget['updates_per_complete_pass'] == 7
    assert budget['max_updates'] == 28 and budget['minimum_updates'] == 7
    assert budget['distributed_tail_repeated_rows_per_pass'] == 1
    assert budget['mixture_derived_from_actual_token_verified_counts']
    assert compiled['loss_normalization'] == budget['loss_normalization'] == 'token'
    assert compiled['validation_examples'] == 512
    data = tmp_path / 'actual-row-index'; rows = [dict(id=stage + '-' + str(i), stage=stage,
        feature_key=stage + '-' + str(i), split='train', answer='controlled preserved answer')
        for stage, count in expected.items() for i in range(count)]
    raw = tmp_path / 'actual-train.jsonl'; write_jsonl(raw, rows); build_index([raw], data)
    indexed = IndexedTrainingRows(data, [raw], compiled['mixture'])
    try:
        finite = prepare_finite_epochs(indexed, compiled['mixture'], dict(compiled, steps=28), 3)
        observed = Counter()
        for step in range(7):
            for rank in range(3):
                indices, report = finite.batch_indices(step, rank)
                observed.update(indexed.row_metadata(i)['id'] for i in indices)
        assert len(observed) == 191 and sum(observed.values()) == 192
        assert set(observed) == {row['id'] for row in rows}
        assert report['global_padding_rows'] == finite.contract()['padded_rows_per_epoch'] == 1
    finally: indexed.close()


@pytest.fixture
def three_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(training_control, 'template', template)
    return evaluation_control.completed_round.__wrapped__(tmp_path, monkeypatch)


def test_native_three_worker_round_and_completed_parent_reopen_exact_actual_state(three_completed):
    value = three_completed; root = value['root']; proof = json.loads((root / 'manifest.json').read_text())
    result = proof['verification']
    assert result['world_size'] == 3 and result['global_batch_size'] == 30
    assert result['actual_three_gpu_full_decoder_preflight_passed']
    assert result['actual_four_gpu_full_decoder_preflight_passed'] is False
    assert result['trained_search_distillation_rounds'] == 1 and result['trained_auxiliary_policy_rounds'] == 0
    saved, completed = completed_parent(value['source'], root / 'manifest.json')
    assert completed == proof and saved['search_topology_profile'] == THREE_SEARCH_WORKERS
    preflight = json.loads((root / 'preflight/manifest.json').read_text())
    assert preflight['kind'] == 'three_gpu_search_distillation_sft_preflight'
    assert preflight['verification']['exact_resume']['world_sizes'] == [3, 3]
    assert len(preflight['verification']['longest_actual_train_validation_probe']['train']['ids']) == 30
    assert {p.name for p in (root / 'preflight/continuous').glob('resources-step-4-rank-*.json')} == {
        'resources-step-4-rank-' + str(i) + '.json' for i in range(3)}
    source, _ = evaluation.checked_sft(root, value['values'], value['recipe'])
    assert source == value['source']


def test_three_worker_search_model_reaches_full_paired_raw_review_without_repair(three_completed, monkeypatch):
    value = three_completed; controlled = ControlledEvaluation(value['original'], monkeypatch)
    result = evaluation_control.execute(value)
    assert controlled.loads == 3 and len(controlled.commands) == 6
    assert result['trained_auxiliary_policy_rounds'] == 0 and result['trained_search_distillation_rounds'] == 1
    assert result['actual_full_decoder_distillation_round_completed_before_evaluation']
    assert result['all_validation_examples'] == 16 and result['blinded_prose_review']['rejected'] == 1
    assert all(m['parse_valid_rate'] == 0 for m in result['raw_validation_by_memory'].values())
    assert not result['raw_answers_repaired'] and not result['strong_play_or_reliable_coaching_proven']


@pytest.mark.parametrize('damage', ['missing_profile', 'wrong_profile', 'four_flag', 'missing_three_flag',
                                  'false_world', 'false_batch', 'false_auxiliary', 'false_search'])
def test_declared_three_worker_completion_cannot_relabel_topology_or_ancestry(three_completed, damage):
    value = three_completed; path = value['root'] / 'manifest.json'; proof = json.loads(path.read_text())
    result = proof['verification']
    if damage == 'missing_profile': proof['config']['compiled_recipe'].pop('search_topology_profile')
    elif damage == 'wrong_profile': result['search_topology_profile'] = 'full_decoder_auxiliary_three_workers_v1'
    elif damage == 'four_flag': result['actual_four_gpu_full_decoder_preflight_passed'] = True
    elif damage == 'missing_three_flag': result.pop('actual_three_gpu_full_decoder_preflight_passed')
    elif damage == 'false_world': result['world_size'] = 4
    elif damage == 'false_batch': result['global_batch_size'] = 32
    elif damage == 'false_auxiliary': result['trained_auxiliary_policy_rounds'] = 1
    else: result['trained_search_distillation_rounds'] = 2
    atomic_json(path, proof)
    with pytest.raises(ValueError): evaluation.checked_sft(value['root'], value['values'], value['recipe'])


@pytest.mark.parametrize('damage', ['missing_rank', 'wrong_rank'])
def test_three_worker_probe_requires_actual_last_rank_evidence(three_completed, damage):
    root = three_completed['root'] / 'preflight'; path = root / 'continuous/resources-step-4-rank-2.json'
    if damage == 'missing_rank': path.unlink()
    else:
        value = json.loads(path.read_text()); value['world_size'] = 4; atomic_json(path, value)
    latest = torch.load(root / 'continuous/latest.pt', weights_only=True)
    config = json.loads((root / 'continuous.config.json').read_text())
    with pytest.raises((ValueError, FileNotFoundError)):
        sft_preflight.verify_updated_probe(root, root / 'zero-init.pt', latest['input_hashes'], config, latest['code'])


def test_controlled_auxiliary_parent_round_count_is_preserved_by_real_round_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(training_control, 'template', template)
    source, recipe, token, runner, _ = training_control.round_setup(tmp_path, monkeypatch)
    original = training.completed_parent
    def controlled_auxiliary(*args):
        # Upstream completed-parent identity is a controlled seam here, not a
        # new claim that the fixture initial model underwent auxiliary SGD.
        saved, proof = original(*args); proof = deepcopy(proof); proof['kind'] = AUXILIARY_ROUND_KIND
        proof['verification'].update(trained_auxiliary_policy_rounds=2, trained_search_distillation_rounds=3)
        return saved, proof
    monkeypatch.setattr(training, 'completed_parent', controlled_auxiliary)
    result = training.run_round(source, recipe, token, tmp_path / 'round')
    assert result['trained_auxiliary_policy_rounds'] == 2 and result['trained_search_distillation_rounds'] == 4
    assert len(runner.jobs) == 4


@pytest.mark.parametrize('damage', ['wrong_kind', 'false_body'])
def test_mismatched_returned_capacity_or_native_manifest_never_starts_formal_SGD(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(training_control, 'template', template)
    source, recipe, token, runner, _ = training_control.round_setup(tmp_path, monkeypatch)
    native = training.run_preflight
    def changed(*args, **kwargs):
        result = native(*args, **kwargs); path = Path(args[3]) / 'manifest.json'
        proof = json.loads(path.read_text())
        if damage == 'wrong_kind': proof['kind'] = 'four_gpu_search_distillation_sft_preflight'
        else: proof['verification']['world_size'] = 4
        atomic_json(path, proof); return result
    monkeypatch.setattr(training, 'run_preflight', changed)
    output = tmp_path / 'round'
    with pytest.raises(ValueError): training.run_round(source, recipe, token, output)
    assert len(runner.jobs) == 3 and all('xqgeneral.sft_probe_worker' in c for c in runner.jobs)
    assert not (output / 'sft').exists() and not (output / 'manifest.json').exists()


@pytest.mark.parametrize('key,value', [('trained_auxiliary_policy_rounds', True), ('trained_search_distillation_rounds', -1)])
def test_invalid_parent_round_counts_stop_before_any_probe_or_formal_update(tmp_path, monkeypatch, key, value):
    monkeypatch.setattr(training_control, 'template', template)
    source, recipe, token, runner, _ = training_control.round_setup(tmp_path, monkeypatch)
    original = training.completed_parent
    def changed(*args):
        saved, proof = original(*args); proof['verification'][key] = value; return saved, proof
    monkeypatch.setattr(training, 'completed_parent', changed)
    output = tmp_path / 'round'
    with pytest.raises(ValueError): training.run_round(source, recipe, token, output)
    assert not runner.jobs and not output.exists()
