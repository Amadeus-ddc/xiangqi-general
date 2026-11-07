"""Controlled parent/tokenizer fixtures check CPU contracts, not model quality."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

import test_oracle_policy_corpus as corpus_control
from xqgeneral import auxiliary_policy_preflight as auxiliary,oracle_policy_corpus as corpus
from xqgeneral.evidence import atomic_json,digest,load_jsonl
from xqgeneral.feature_store import FeatureStoreWriter
from xqgeneral.recorded_search_inputs import BoundInputs


def features(keys):
    return dict(keys=keys,depths=[0,3],features=[torch.zeros(len(keys),90,512,dtype=torch.float16) for _ in range(2)],
        wdl=torch.zeros(len(keys),3,dtype=torch.float32))


def cache_chain(root,keys):
    paths=[]
    for index,part in enumerate([keys[:40],keys[40:45],keys[45:]]):
        writer=FeatureStoreWriter(root/('cache'+str(index)),[0,3],'a'*64)
        if paths:writer.add_existing(paths[-1],paths[-1])
        writer.append(features(part));writer.finish();paths.append(writer.output/'manifest.json')
    return paths


def fixture(tmp_path,monkeypatch,*,validation_histories=1):
    args=corpus_control.fixture(tmp_path,validation_histories=validation_histories);corpus.prepare(**args);data=args['output']
    keys=sorted({r['feature_key'] for s in ['train','validation'] for r in load_jsonl(data/(s+'.jsonl'))})
    paths=cache_chain(tmp_path,keys)
    parent=tmp_path/'parent';parent.mkdir();source=parent/'adapter.pt';source.write_bytes(b'controlled state; no weights loaded')
    saved=dict(mode='bridge',model_path=str(tmp_path/'model'),model_revision='fixture',feature_path=str(paths[0]),
        board_tokens=True,expert_feature_depths=[0,3],task_profile=corpus.PAPER_PROFILE)
    atomic_json(parent/'config.json',saved);atomic_json(parent/'manifest.json',dict(controlled=True))
    completion=tmp_path/'parent-complete.json';atomic_json(completion,dict(controlled=True))
    pinned=tmp_path/'pinned-decoder.json';atomic_json(pinned,dict(controlled=True))
    def completed(path,proof,bound):
        assert Path(path)==source and Path(proof)==completion
        for p in [source,completion,parent/'config.json',parent/'manifest.json']:bound.bind(p)
        return saved,dict(controlled=True)
    def decoder(formal,recipe,foundation,bound):
        reference=recipe['pinned_base_model_manifest'];assert reference['path']==str(pinned)
        bound.bind(pinned);return {},reference
    calls=[]
    def initialize(path,limit):calls.append((path,limit))
    def tokens(rows):
        return [dict(id=r['id'],split=r['split'],tokens=150+(10 if r['stage']=='explanation' else 0),
            supervised_tokens=50+(10 if r['stage']=='explanation' else 0),prompt_tokens=100) for r in rows]
    monkeypatch.setattr(auxiliary,'completed_parent',completed)
    monkeypatch.setattr(auxiliary,'pinned_decoder_assets',decoder)
    monkeypatch.setattr(auxiliary,'checked_parent',lambda path,clean:saved)
    monkeypatch.setattr(auxiliary,'token_initializer',initialize);monkeypatch.setattr(auxiliary,'token_chunk',tokens)
    return dict(data=data,source=source,parent_completion=completion,features=paths[-1],
        pinned_base_model_manifest=pinned,output=tmp_path/'token-preflight',workers=1),saved,calls


def template():
    return dict(decoder_training='full',trainable_parameter_dtype='float32',ddp_world_size=4,batch_size=32,
        micro_batch_size=1,loss_normalization='token',training_data_profile=auxiliary.FINITE_PROFILE,
        source_coverage_profile=auxiliary.FULL_SOURCE_PROFILE,max_finite_passes=4,learning_rate=1e-5,
        decoder_learning_rate=1e-6,eval_every=250,patience=4,seed=20261107,feature_cache_mmap=True)


def test_full_actual_row_coverage_and_multilevel_cache_prefix_compile_native_budget(tmp_path,monkeypatch):
    args,saved,calls=fixture(tmp_path,monkeypatch);counts=auxiliary.prepare(**args)
    assert counts['rows_by_split']==dict(train=350,validation=7)
    assert len(counts['feature_ancestry'])==2 and counts['training_unique_history_contexts']==50
    assert counts['independent_test_rows_parsed_or_tokenized']==counts['new_SGD_updates']==0
    proof=json.loads((args['output']/'manifest.json').read_text())
    recipe,config,budget=auxiliary.compile_recipe(proof,template(),tmp_path/'round')
    assert recipe['validation_examples']==7 and set(recipe['stages'])==set(corpus.ALL_STAGES)
    assert config['steps']==44 and config['min_steps']==11
    assert budget['distributed_tail_repeated_rows_per_pass']==2 and budget['max_steps']==44
    assert not budget['auxiliary_supervision_is_new_on_policy_search_round']
    assert config['loss_normalization']=='token' and config['feature_path']==str(args['features'])
    original=corpus_control.snapshot(args['output']);again=auxiliary.prepare(**args,resume=True)
    assert again==counts and corpus_control.snapshot(args['output'])==original and len(calls)==2
    receipt=json.loads((args['output']/'verified-inputs.json').read_text())
    assert str(args['source']) in receipt['inputs'] and str(args['features']) in receipt['inputs']


@pytest.mark.parametrize('damage',['missing_feature','same_cache','ancestor_contract','encoder_contract','token_overflow','zero_supervision','duplicate_id'])
def test_changed_histories_cache_ancestry_or_native_token_contract_cannot_pass(tmp_path,monkeypatch,damage):
    args,saved,_=fixture(tmp_path,monkeypatch)
    if damage=='same_cache':args['features']=Path(saved['feature_path'])
    elif damage=='ancestor_contract':
        path=args['features'];proof=json.loads(path.read_text());proof['config']['base_proof']='other';atomic_json(path,proof)
    elif damage=='encoder_contract':
        path=args['features'];proof=json.loads(path.read_text());proof['code']['source_sha256']['src/xqgeneral/expert.py']='0'*64;atomic_json(path,proof)
    elif damage in ['missing_feature','duplicate_id']:
        path=args['data']/'train.jsonl';rows=load_jsonl(path)
        if damage=='missing_feature':
            row=corpus_control.explanation(corpus_control.context(['h0g2','h9g7']));rows[0]=row
        else:rows[1]=deepcopy(rows[0])
        corpus_control.rebind(args['data'],'train.jsonl',rows)
    else:
        original=auxiliary.token_chunk
        def bad(rows):
            values=original(rows);values[0].update(tokens=2049,prompt_tokens=1999) if damage=='token_overflow' else values[0].update(supervised_tokens=0,prompt_tokens=values[0]['tokens'])
            return values
        monkeypatch.setattr(auxiliary,'token_chunk',bad)
    with pytest.raises(ValueError):auxiliary.prepare(**args)
    assert not (args['output']/'manifest.json').exists()


@pytest.mark.parametrize('change',[dict(loss_normalization='example'),dict(source_coverage_profile=None),dict(max_finite_passes=0),dict(ddp_world_size=2)])
def test_recipe_changes_cannot_silently_drop_full_coverage_or_author_token_loss(tmp_path,monkeypatch,change):
    args,_,_=fixture(tmp_path,monkeypatch);auxiliary.prepare(**args)
    proof=json.loads((args['output']/'manifest.json').read_text())
    with pytest.raises(ValueError):auxiliary.compile_recipe(proof,dict(template(),**change),tmp_path/'round')


@pytest.mark.parametrize('damage',['source','output','code_contract'])
def test_completed_preflight_reopen_preserves_outputs_on_changed_contracts(tmp_path,monkeypatch,damage):
    args,_,_=fixture(tmp_path,monkeypatch);auxiliary.prepare(**args)
    if damage=='source':args['source'].write_bytes(b'changed immutable parent')
    elif damage=='output':(args['output']/'token-lengths.jsonl').write_bytes(b'changed\n')
    else:args['max_tokens']=4096
    before=corpus_control.snapshot(args['output'])
    with pytest.raises(ValueError):auxiliary.prepare(**args,resume=True)
    assert corpus_control.snapshot(args['output'])==before


def test_auxiliary_gpu_capacity_probe_keeps_every_stage_longest_and_original_bytes(tmp_path,monkeypatch):
    args,saved,_=fixture(tmp_path,monkeypatch);data=args['data'];rows=load_jsonl(data/'validation.jsonl')
    base=next(row for row in rows if row['stage']=='explanation')
    rows += [dict(base,id=base['id']+'-controlled-variant-'+str(i)) for i in range(5)]
    corpus_control.rebind(data,'validation.jsonl',rows)
    path=data/'manifest.json';proof=json.loads(path.read_text());proof['verification']['validation_rows_by_stage']['explanation']+=5
    proof['verification']['validation_question_rows']+=5;atomic_json(path,proof)
    auxiliary.prepare(**args);token=json.loads((args['output']/'manifest.json').read_text())
    recipe,_,_=auxiliary.compile_recipe(token,template(),tmp_path/'round')
    from xqgeneral.sft_preflight import probe_data
    result=probe_data(recipe,token,tmp_path/'probe',train_examples=32)
    assert len(result['train']['ids'])==32 and len(result['validation']['ids'])==12
    for split in ['train','validation']:
        assert set(result[split]['rows_by_stage'])==set(corpus.ALL_STAGES)
        assert result[split]['longest_row_from_every_stage_retained']
        old={row['id']:raw for pointer,raw,row in corpus._lines(data/(split+'.jsonl'))}
        for pointer,raw,row in corpus._lines(tmp_path/'probe'/(split+'.jsonl')):assert raw==old[row['id']]
    assert auxiliary.auxiliary_parent(args['source'],token,recipe,BoundInputs())==saved


@pytest.mark.parametrize('damage',['parent','stage','loss','coverage'])
def test_auxiliary_parent_gate_rejects_wrong_recipe_before_any_gpu_work(tmp_path,monkeypatch,damage):
    args,_,_=fixture(tmp_path,monkeypatch);auxiliary.prepare(**args)
    token=json.loads((args['output']/'manifest.json').read_text());recipe,_,_=auxiliary.compile_recipe(token,template(),tmp_path/'round')
    if damage=='parent':token['config']['logical']['source']='other'
    elif damage=='stage':recipe['stages']=recipe['stages'][:-1]
    elif damage=='loss':recipe['loss_normalization']='example'
    else:recipe['validation_examples']-=1
    with pytest.raises(ValueError):auxiliary.auxiliary_parent(args['source'],token,recipe,BoundInputs())
