"""Fake GPU/SFT producers exercise on-disk gates; no CUDA or real training."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import test_auxiliary_policy_preflight as inputs_control
from xqgeneral import auxiliary_policy_preflight as auxiliary,auxiliary_policy_training as training
from xqgeneral.evidence import atomic_json,code_identity,digest,manifest
from xqgeneral.finite_training import prepare_finite_epochs
from xqgeneral.training_index import IndexedTrainingRows


def setup(tmp_path,monkeypatch,damage=None,*,world=4,validation_histories=1,parent_rounds=2):
    args,saved,_=inputs_control.fixture(tmp_path,monkeypatch,validation_histories=validation_histories);auxiliary.prepare(**args)
    token=args['output']/'manifest.json';template=tmp_path/'recipe-template.json';recipe=inputs_control.template()
    if world in {2,3}:
        profile={2:'full_decoder_auxiliary_two_workers_v1',3:'full_decoder_auxiliary_three_workers_v1'}[world]
        recipe.update(auxiliary_topology_profile=profile,ddp_world_size=world,batch_size=30,
            distributed_strategy='fsdp2',fsdp_shard_paths=['bridges.*','base'])
    atomic_json(template,recipe)
    monkeypatch.setattr(training,'completed_parent',lambda source,completion,bound:(
        auxiliary.completed_parent(source,completion,bound)[0],dict(verification=dict(trained_search_distillation_rounds=parent_rounds))))
    calls=[]
    def gpu(source,recipe_path,token_path,root,resume=False):
        calls.append('GPU_preflight');root=Path(root);root.mkdir()
        initial=root/'zero-init.pt';torch.save(dict(trainable=dict(bridge=torch.zeros(4))),initial)
        result=dict(status='complete',world_size=world,actual_continuous_four_and_two_plus_two_updates_compared=True,
            full_decoder_initialization_preserved_all_parent_tensors=True,trainable_tensors=1)
        if damage=='failed_GPU':result['status']='failed'
        kind={2:'two',3:'three',4:'four'}[world]+'_gpu_auxiliary_policy_sft_preflight'
        if damage=='wrong_probe_kind':kind='four_gpu_auxiliary_policy_sft_preflight'
        if damage=='wrong_probe_world':result['world_size']=4
        proof=manifest(kind,{},[],[initial],result)
        proof['verified_feature_cache_input']=json.loads(token_path.read_text())['cache_header_only_input']
        atomic_json(root/'manifest.json',proof);return result
    monkeypatch.setattr(training,'run_preflight',gpu)
    def sft(command,**kwargs):
        calls.append('formal_SFT');root=Path(command[command.index('--output')+1]);root.mkdir()
        recipe=json.loads(Path(command[command.index('--recipe')+1]).read_text())
        source=Path(command[command.index('--init')+1]);config=training.prepare_config(saved,recipe,source,root)
        indexed=IndexedTrainingRows(recipe['row_index_paths']['train'],[Path(recipe['data_path'])/'train.jsonl'],recipe['mixture'])
        try:data_state=prepare_finite_epochs(indexed,recipe['mixture'],config,world).state_at(config['steps'])
        finally:indexed.close()
        selected_step=0 if damage=='step_zero' else config['steps']
        metrics=dict(world_size=world,decoder_training='full',global_batch_size=recipe['batch_size'],selected_step=selected_step,
            steps_completed=config['steps'],training_data_state=data_state,trainable_parameters=4)
        if damage=='wrong_epoch':metrics['training_data_state']['optimizer_batch_in_epoch']=999
        ids=training._validation(recipe,recipe['validation_rows_by_stage'])['selected_ids']
        if damage=='missing_validation':ids=ids[:-1]
        atomic_json(root/'validation_ids.json',ids);atomic_json(root/'metrics.json',metrics);atomic_json(root/'config.json',config)
        inputs=[Path(recipe['data_path'])/'train.jsonl',Path(recipe['data_path'])/'validation.jsonl',Path(recipe['feature_path'])]
        values={str(p):digest(p) for p in inputs}
        state=dict(config=config,selected_step=selected_step,code=code_identity(),input_hashes=values,
            trainable=dict(bridge=torch.ones(4)))
        if damage=='bad_selected_binding':state['input_hashes'][str(inputs[0])]='0'*64
        torch.save(state,root/'adapter.pt')
        proof=manifest('model_training',config,inputs,[root/p for p in ['adapter.pt','config.json','metrics.json','validation_ids.json']],metrics)
        atomic_json(root/'manifest.json',proof)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(training,'subprocess',SimpleNamespace(run=sft,STDOUT=training.subprocess.STDOUT))
    return dict(source=args['source'],recipe_template=template,token_manifest=token,output=tmp_path/'round'),calls


def test_full_auxiliary_protocol_preserves_search_round_count_and_reopens_without_GPU_or_SGD(tmp_path,monkeypatch):
    args,calls=setup(tmp_path,monkeypatch);result=training.run_round(**args)
    assert calls==['GPU_preflight','formal_SFT'] and result['actual_auxiliary_policy_sft_completed']
    assert result['trained_search_distillation_rounds']==2 and result['trained_auxiliary_policy_rounds']==1
    assert not result['auxiliary_supervision_is_new_on_policy_search_round'] and result['actual_steps']==44
    assert result['validation_coverage']['validation_rows']==7 and result['loss_normalization']=='token'
    before=inputs_control.corpus_control.snapshot(args['output']);assert training.run_round(**args,resume=True)==result
    assert calls==['GPU_preflight','formal_SFT'] and inputs_control.corpus_control.snapshot(args['output'])==before
    # The new type can serve as a completed full-decoder parent for later stages.
    from xqgeneral.distillation_preflight import completed_parent
    saved,proof=completed_parent(Path(result['selected_checkpoint']),args['output']/'manifest.json')
    assert proof['kind']==training.KIND and saved['loss_normalization']=='token'


@pytest.mark.parametrize('damage',['failed_GPU','step_zero','wrong_epoch','missing_validation','bad_selected_binding'])
def test_failed_capacity_and_unqualified_native_training_are_never_promoted(tmp_path,monkeypatch,damage):
    args,calls=setup(tmp_path,monkeypatch,damage)
    with pytest.raises(ValueError):training.run_round(**args)
    assert not (args['output']/'manifest.json').exists()
    if damage=='failed_GPU':assert calls==['GPU_preflight']
    else:assert calls==['GPU_preflight','formal_SFT']
    state=json.loads((args['output']/'state.json').read_text());assert state['formal_auxiliary_completion_proven'] is False


def test_all_seven_validation_stages_need_original_unique_ids(tmp_path,monkeypatch):
    args,_=setup(tmp_path,monkeypatch);token=json.loads(args['token_manifest'].read_text());counts=token['verification']['rows_by_split_and_stage']['validation']
    rows=inputs_control.corpus_control.load_jsonl(Path(token['config']['logical']['data'])/'validation.jsonl');ids=[r['id'] for r in rows]
    assert auxiliary.validation_coverage(rows,ids,counts)['every_original_validation_id_covered_once']
    with pytest.raises(ValueError):auxiliary.validation_coverage(rows,ids[:-1],counts)
    changed=deepcopy(rows);changed[0]['split']='test'
    with pytest.raises(ValueError):auxiliary.validation_coverage(changed,ids,counts)
