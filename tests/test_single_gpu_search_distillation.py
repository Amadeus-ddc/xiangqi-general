"""CPU protocol/optimizer controls; these fixtures do not execute real GPU SGD."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import test_auxiliary_search_replay as replay_control
import test_distillation_joint_replay as joint_control
import test_distillation_training as training_control
import test_post_distillation_evaluation as evaluation_control
from test_clean_sft_evaluation import ControlledEvaluation, setup_evaluation
from xqgeneral import distillation_training, training, optimizer_offload
from xqgeneral import clean_sft_evaluation as evaluation
from xqgeneral.distillation_preflight import completed_parent
from xqgeneral.evidence import atomic_json, digest, load_jsonl, write_jsonl
from xqgeneral.finite_training import prepare_finite_epochs
from xqgeneral.full_decoder_topology import search_topology, completed_search_topology
from xqgeneral.training_index import build_index, IndexedTrainingRows
from xqgeneral.rules import START_FEN


def template():
    return json.loads(Path('configs/search-distillation-sft-one-gpu-cpu-offload-with-auxiliary-v1.json').read_text())


def test_single_worker_native_recipe_keeps_all_six_replay_tasks_and_complete_validation(tmp_path):
    _, token = replay_control.budget()
    value = template(); value['replay_data_paths'] = token['config']['replay_data_paths']
    recipe, budget = distillation_training.compile_recipe(value, token, tmp_path / 'round')
    assert budget['world_size'] == 1 and budget['global_batch_size'] == 30
    assert budget['rows_per_complete_pass'] == 271 and budget['updates_per_complete_pass'] == 10
    expected = dict(explanation=31, **dict.fromkeys(replay_control.replay.STAGES, 40))
    assert budget['finite_mixture']['selected_rows_by_stage'] == expected
    assert budget['distributed_tail_repeated_rows_per_pass'] == 0
    assert budget['mixture_derived_from_actual_token_verified_counts']
    assert recipe['validation_examples'] == 523 and recipe['loss_normalization'] == 'token'
    assert recipe['optimizer_state_offload'] == 'cpu'
    assert budget['auxiliary_policy_replay_included'] and not budget['paper_first_round_150000_label_scale_reached']
    rows = [dict(id=stage + str(i), feature_key=stage + str(i), stage=stage, split='train', answer='controlled')
            for stage, count in expected.items() for i in range(count)]
    raw = tmp_path / 'rows.jsonl'; write_jsonl(raw, rows); index = tmp_path / 'index';build_index([raw], index)
    indexed = IndexedTrainingRows(index, [raw], recipe['mixture'])
    try:
        finite = prepare_finite_epochs(indexed, recipe['mixture'], dict(recipe, steps=budget['max_updates']), 1)
        seen = Counter()
        for step in range(10):
            indices, report = finite.batch_indices(step, 0)
            seen.update(indexed.row_metadata(i)['id'] for i in indices)
        assert set(seen) == {row['id'] for row in rows} and sum(seen.values()) == 271
        assert report['global_padding_rows'] == 0
    finally:
        indexed.close()
    optimizer = training.training_optimizer(nn.Linear(2, 2), recipe)
    assert isinstance(optimizer, optimizer_offload.CPUOffloadedAdamW)


@pytest.mark.parametrize('change', [dict(ddp_world_size=2), dict(ddp_world_size=True),
    dict(batch_size=32), dict(micro_batch_size=2), dict(optimizer_state_offload='none'),
    dict(optimizer_foreach=True), dict(optimizer_state_sharding='zero1'),
    dict(distributed_strategy='fsdp2'), dict(trainable_parameter_dtype='bfloat16'),
    dict(decoder_training='lora')])
def test_one_worker_profile_cannot_silently_change_geometry_precision_or_state_placement(change):
    with pytest.raises(ValueError): search_topology(dict(template(), **change))


def test_one_worker_requires_explicit_profile_and_distinct_actual_capacity_flag():
    value = template(); actual = search_topology(value)
    assert actual['preflight_kind'] == 'one_gpu_search_distillation_sft_preflight'
    assert actual['completion_flag'] == 'actual_one_gpu_full_decoder_preflight_passed'
    value.pop('search_topology_profile')
    with pytest.raises(ValueError): search_topology(value)


def completed():
    return dict(config=dict(compiled_recipe=template()), verification=dict(world_size=1,
        search_topology_profile=template()['search_topology_profile'],
        actual_one_gpu_full_decoder_preflight_passed=True,
        actual_four_gpu_full_decoder_preflight_passed=False,
        actual_cpu_optimizer_state_placement_verified=True))


@pytest.mark.parametrize('damage', ['missing_one_flag','two_flag','three_flag','four_flag','world','profile','placement'])
def test_one_worker_completion_needs_its_own_capacity_and_runtime_moment_evidence(damage):
    proof = completed(); result = proof['verification']
    if damage == 'missing_one_flag': result.pop('actual_one_gpu_full_decoder_preflight_passed')
    elif damage in {'two_flag','three_flag','four_flag'}:
        result['actual_' + damage.split('_')[0] + '_gpu_full_decoder_preflight_passed'] = True
    elif damage == 'world': result['world_size'] = 2
    elif damage == 'profile': result['search_topology_profile'] = 'full_decoder_search_two_workers_v1'
    else: result['actual_cpu_optimizer_state_placement_verified'] = False
    with pytest.raises(ValueError): completed_search_topology(proof)


def placement():
    return dict(parameters_with_state=3,state_tensors=9,parameter_device='cuda:0',
        all_states_on_CPU=True,all_moments_FP32=True,total_state_bytes=252,peak_active_moment_bytes=120)


@pytest.mark.parametrize('change', [dict(parameters_with_state=2),dict(state_tensors=8),
    dict(parameters_with_state=3.0),dict(state_tensors=9.0),
    dict(parameter_device='cpu'),dict(all_states_on_CPU=False),dict(all_moments_FP32=False),
    dict(total_state_bytes=240),dict(peak_active_moment_bytes=0)])
def test_partial_or_nonoffloaded_runtime_state_cannot_authorize_single_gpu_training(change):
    with pytest.raises(ValueError):
        optimizer_offload.validate_cpu_moment_placement(dict(placement(),**change),3,30)


def test_complete_cpu_state_placement_is_checked_without_changing_values():
    value = placement(); before=deepcopy(value)
    optimizer_offload.validate_cpu_moment_placement(value,3,30)
    assert value == before


def controlled_placement(state):
    """Synthetic producer metadata for CPU protocol controls, never GPU evidence."""
    return dict(parameters_with_state=len(state), state_tensors=len(state)*3,
        parameter_device='cuda:0', all_states_on_CPU=True, all_moments_FP32=True,
        total_state_bytes=sum(t.numel()*8+4 for t in state.values()),
        peak_active_moment_bytes=max(t.numel()*8 for t in state.values()),
        constructed_CPU_fixture_no_actual_GPU=True)


class OneWorkerRunner(training_control.RoundRunner):
    """Keep real orchestration and checkers; replace only GPU artifact producers."""
    def __call__(self, command, **kwargs):
        result = super().__call__(command, **kwargs)
        if 'xqgeneral.sft_probe_worker' in command:
            config = json.loads(Path(command[command.index('--config')+1]).read_text())
            if config.get('optimizer_state_offload') != 'cpu': return result
            root = Path(config['output']); saved = torch.load(root/'latest.pt', weights_only=True)
            for step in ([2,4] if root.name == 'continuous' else [saved['step']]):
                atomic_json(root/f'optimizer-placement-step-{step}.json',
                    dict(controlled_placement(saved['trainable']), actual_step=step))
            if self.failure == 'missing_placement':
                (root/f'optimizer-placement-step-{saved["step"]}.json').unlink()
        elif 'xqgeneral.sft' in command:
            root = Path(command[command.index('--output')+1])
            config = json.loads((root/'config.json').read_text())
            if config.get('optimizer_state_offload') != 'cpu': return result
            saved = torch.load(root/'adapter.pt', weights_only=True)
            metrics = json.loads((root/'metrics.json').read_text())
            metrics['optimizer_runtime_state_placement'] = controlled_placement(saved['trainable'])
            if self.failure == 'nonoffloaded_formal':
                metrics['optimizer_runtime_state_placement']['all_states_on_CPU'] = False
            atomic_json(root/'metrics.json', metrics)
            proof = json.loads((root/'manifest.json').read_text())
            proof['verification'] = metrics; atomic_json(root/'manifest.json', proof)
            indexed = IndexedTrainingRows(config['row_index_paths']['validation'],
                [Path(p)/'validation.jsonl' for p in [config['data_path'],*config['replay_data_paths']]],
                config['mixture'],split='validation')
            try:
                rows = training.select_validation_rows(indexed, config['validation_stages'],
                    config['seed'],config['validation_examples'])
                atomic_json(root/'validation_ids.json',[row['id'] for row in rows])
            finally: indexed.close()
        return result


@pytest.fixture
def one_setup(tmp_path, monkeypatch):
    original_setup=training_control.setup_data
    def native_rows(path):
        curriculum,recipe,token_path=original_setup(path)
        token=json.loads(token_path.read_text())
        for split in ['train','validation']:
            p=Path(token['config']['data'])/(split+'.jsonl');rows=load_jsonl(p)
            for row in rows: row.update(fen=START_FEN,initial_fen=START_FEN,moves=[],question='preserved native question')
            write_jsonl(p,rows)
            token['inputs'][str(p)]=dict(sha256=digest(p),bytes=p.stat().st_size)
        atomic_json(token_path,token)
        return curriculum,recipe,token_path
    monkeypatch.setattr(training_control,'setup_data',native_rows)
    monkeypatch.setattr(training_control,'template',template)
    source, recipe_path, token_path, old_runner, raw = joint_control.controlled_joint_round(tmp_path/'assets',monkeypatch)
    token = json.loads(token_path.read_text())
    for split,count in [('train',4),('validation',2)]:
        path = raw/(split+'.jsonl'); rows=load_jsonl(path)
        rows += [dict(id=f'{stage}-{split}-{i}',feature_key=f'policy-{stage}-{split}-{i}',
            stage=stage,split=split,answer='controlled fixture, no neural quality claim')
            for stage in ['engine_numeric','move_quality'] for i in range(count)]
        write_jsonl(path,rows)
        for stage in ['engine_numeric','move_quality']:
            token['verification']['course_replay']['by_split_and_stage'][split][stage]=count
        token['inputs'][str(path)] = dict(sha256=digest(path),bytes=path.stat().st_size)
    token['verification']['course_replay']['train_validation_rows_tokenized']+=12
    token['config']['replay_profile']=replay_control.replay.PROFILE
    recipe=dict(template(),replay_data_paths=[str(raw)],max_steps=12,max_finite_passes=2,eval_every=2)
    atomic_json(recipe_path,recipe);atomic_json(token_path,token)
    runner=OneWorkerRunner(old_runner.original)
    monkeypatch.setattr(distillation_training.subprocess,'run',runner)
    return dict(source=source,recipe_path=recipe_path,token=token_path,runner=runner,root=tmp_path/'round')


@pytest.fixture
def one_completed(one_setup):
    s=one_setup
    s['result']=distillation_training.run_round(s['source'],s['recipe_path'],s['token'],s['root'])
    return s


def test_native_one_worker_preflight_seven_stage_round_reopen_and_resume(one_completed):
    s=one_completed;root=s['root'];result=s['result']
    assert result['world_size']==1 and result['global_batch_size']==30
    assert result['actual_one_gpu_full_decoder_preflight_passed']
    assert result['actual_cpu_optimizer_state_placement_verified']
    assert set(result['selected_training_rows_by_stage'])==set(replay_control.ALL_STAGES)
    assert result['joint_validation_coverage']['validation_rows']==28
    preflight=json.loads((root/'preflight/manifest.json').read_text())
    assert preflight['kind']=='one_gpu_search_distillation_sft_preflight'
    assert preflight['verification']['exact_resume']['world_sizes']==[1,1]
    assert len(preflight['verification']['actual_optimizer_state_placement'])==4
    assert len([p for p in preflight['outputs'] if 'optimizer-placement-step-' in p])==4
    completed_parent(root/'sft/adapter.pt',root/'manifest.json')
    assert distillation_training.run_round(s['source'],s['recipe_path'],s['token'],root,resume=True)==result
    assert len(s['runner'].jobs)==4


@pytest.mark.parametrize('failure',['missing_placement','nonoffloaded_formal'])
def test_missing_or_incompatible_residency_never_publishes_a_completed_round(one_setup,failure):
    s=one_setup;s['runner'].failure=failure
    with pytest.raises((ValueError,FileNotFoundError)):
        distillation_training.run_round(s['source'],s['recipe_path'],s['token'],s['root'])
    assert not (s['root']/'manifest.json').exists()
    if failure=='missing_placement':
        assert not any('xqgeneral.sft' in command for command in s['runner'].jobs)


def test_rehashed_completed_round_cannot_hide_incompatible_runtime_residency(one_completed):
    root=one_completed['root'];p=root/'sft/manifest.json';proof=json.loads(p.read_text())
    proof['verification']['optimizer_runtime_state_placement']['all_states_on_CPU']=False
    atomic_json(p,proof)
    outer=root/'manifest.json';proof=json.loads(outer.read_text())
    proof['inputs'][str(p)]=dict(sha256=digest(p),bytes=p.stat().st_size);atomic_json(outer,proof)
    with pytest.raises(ValueError,match='CPU moment residency'):
        completed_parent(root/'sft/adapter.pt',outer)


def test_single_worker_completed_model_enters_original_three_memory_raw_review(one_completed,tmp_path,monkeypatch):
    s=one_completed;root=s['root']
    _,config,_=setup_evaluation(tmp_path/'evaluation-assets',monkeypatch,complete=False)
    values=json.loads(config.read_text());recipe=json.loads((root/'recipe.json').read_text())
    values.update(sft_recipe=str(root/'recipe.json'),token_preflight=str(s['token']),
        data=recipe['data_path'],features=recipe['feature_path'])
    atomic_json(config,values)
    value=dict(root=root,config=config,output=tmp_path/'capability')
    controlled=ControlledEvaluation(s['runner'].original,monkeypatch)
    result=evaluation_control.execute(value)
    assert controlled.loads==3 and len(controlled.commands)==6
    assert result['trained_search_distillation_rounds']==1
    assert result['all_validation_examples']==16
    assert not result['raw_answers_repaired'] and not result['strong_play_or_reliable_coaching_proven']
    assert all(v['oracle_repairs']==0 for v in result['raw_validation_by_memory'].values())
