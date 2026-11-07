"""Controlled receipt fixtures exercise admission; they execute no neural student."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_recorded_search_inputs import fixture as recorded_fixture, identities
from test_training_game_scoring import Oracles
from xqgeneral import student_game_inputs as pools
from xqgeneral.evidence import atomic_json, code_identity, digest, file_signature, load_jsonl, manifest, position_key, write_jsonl
from xqgeneral.model_inputs import checkpoint_dependencies
from xqgeneral.neural_training_games import verified_neural_games
from xqgeneral.recorded_search_inputs import BoundInputs, prepare as recorded_prepare
from xqgeneral.rules import legal_moves, replay
from xqgeneral.training_game_scoring import score_games
from xqgeneral.training_play import fingerprint, run_games
from xqgeneral.verified_inputs import KIND as RECEIPT_KIND


def fixture(tmp_path):
    model=tmp_path/'model';model.mkdir()
    for name in ['config.json','tokenizer.json','tokenizer_config.json']:
        atomic_json(model/name,{})
    (model/'model.safetensors').write_bytes(b'controlled CPU asset, no decoder is constructed')
    pinned=tmp_path/'decoder-pin.json'
    atomic_json(pinned,manifest('pinned_local_decoder_assets',
        {'model_path':str(model),'model_revision':'controlled-model-revision'},sorted(model.iterdir()),[],{}))
    assets=json.loads(pinned.read_text())['inputs']
    checkpoint=tmp_path/'selected.pt'
    torch.save({'config':{'model_path':str(model),'model_revision':'controlled-model-revision',
        'pinned_base_model_manifest':{'path':str(pinned),'sha256':digest(pinned)}},
        'trainable':{'controlled':torch.zeros(1)}},checkpoint)
    expert=tmp_path/'expert';expert.write_bytes(b'controlled expert bytes')
    engine=tmp_path/'engine';engine.write_bytes(b'controlled executable identity')
    weights=tmp_path/'nnue';weights.write_bytes(b'controlled oracle identity')
    policy=tmp_path/'gameplay.json'
    atomic_json(policy,{'run_id':'controlled-neural-receipt-fixture','games':2,'game_start':0,'seed':17,
        'concurrent_games':2,'batch_size':2,'max_plies':12,'opponent_nodes_min':100,'opponent_nodes_max':100,
        'generation':{'do_sample':True,'temperature':.6,'top_k':20,'top_p':.95,'max_new_tokens':4096}})
    receipt=tmp_path/'input-digests.json'
    atomic_json(receipt,{'kind':RECEIPT_KIND,'status':'complete','inputs':{
        str(Path(p).resolve()):dict(binding,signature=list(file_signature(p))) for p,binding in assets.items()}})
    source=tmp_path/'games'
    config={k:str(v) for k,v in dict(checkpoint=checkpoint,expert_weights=expert,policy=policy,
        executable=engine,weights=weights,output=source).items()}
    config['student_autocast_cache']='on'
    bound=BoundInputs()
    for p in [policy,checkpoint,expert,engine,weights]:bound.bind(p)
    actor=fingerprint({str(p):bound.artifacts[str(p)] for p in [checkpoint,expert]})
    code=code_identity()
    class Student:
        def generate_batch(self,records,**kwargs):return ['controlled invalid raw response' for _ in records]
    class Engine:
        def choose_move(self,fen,nodes,*args):
            move=sorted(legal_moves(fen))[-1]
            return {'best_move':move,'requested_nodes':nodes,'score_consumed':False,'raw_output':['bestmove '+move]}
        def close(self):pass
    summary=run_games(source,json.loads(policy.read_text()),actor,Student,Engine,
        code={'identity':code,'config':config},inputs=bound.artifacts)
    outputs=[source/'contract.json',source/'games.jsonl',source/'summary.json',source/'student-batches.partial.jsonl',
        *sorted((source/'games').glob('game-*.json'))]
    native=manifest('student_engine_training_games',config,(),outputs,summary,code=code)
    native['inputs']=bound.artifacts
    native['runtime_execution']={'input_digest_verification':{'profile':RECEIPT_KIND,
        'receipt_sha256':digest(receipt),'all_receipt_file_signatures_checked':True,'receipt_bound_files':len(assets)}}
    atomic_json(source/'manifest.json',native)
    execution=tmp_path/'neural-execution.json'
    atomic_json(execution,{'status':'complete_actual_neural_student_training_games',
        'native_manifest_sha256':digest(source/'manifest.json'),'summary':summary,'physical_GPU':0,
        'qualified_source_commit':code['revision'],
        'source_sha256':{k.removeprefix('src/xqgeneral/'):v for k,v in code['source_sha256'].items()},
        'checkpoint_sha256':digest(checkpoint),'preallocated_game_ids_match':True,
        'all_native_completed_outputs_reopened_without_change':True,'student_SGD_updates':0,'new_teacher_queries':0})
    scores=tmp_path/'scores';score_bound=BoundInputs();score_bound.bind(engine);score_bound.bind(weights)
    score_games(source,scores,Oracles().factory,bound=score_bound)
    reader=dict(checkpoint=checkpoint,expert_weights=expert,gameplay_policy=policy,executable=engine,weights=weights,
        decoder_model=model,decoder_manifest=pinned,input_digest_receipt=receipt)
    return dict(games=source,scores=scores,execution_proof=execution,**reader)


def reader_args(args):
    return {k:v for k,v in args.items() if k not in {'games','scores','execution_proof'}}


def authority(tmp_path):
    args,games=recorded_fixture(tmp_path)
    out=tmp_path/'authority'
    recorded_prepare(**args,output=out,min_ply=1,per_game=3)
    return out,args


def test_real_receipt_contract_checks_all_pinned_dependencies_without_loading_models(tmp_path):
    args=fixture(tmp_path)
    games,info=verified_neural_games(args['games'],args['execution_proof'],**reader_args(args))
    assert len(games)==2
    assert info['source_neural_execution_verified'] is True and info['source_model_assets_verified'] is True
    dependencies=info['decoder_dependency_verification']
    assert len(dependencies['assets'])==4
    assert dependencies['model_or_expert_constructed'] is False
    assert dependencies['actual_neural_decoder_dependency_file_signatures_preserved'] is True
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',[
    'CPU_control','no_reopen','GPU_boolean','changed_manifest','changed_checkpoint_receipt',
    'changed_model_asset','wrong_explicit_model','wrong_explicit_expert','changed_source_code',
    'no_dependency_receipt','changed_receipt_identity','missing_native_digest_session','extra_native_asset'])
def test_incompatible_neural_receipts_and_assets_cannot_enter_root_admission(tmp_path,damage):
    args=fixture(tmp_path);execution=args['execution_proof'];proof=json.loads(execution.read_text())
    native=args['games']/'manifest.json';n=json.loads(native.read_text())
    if damage=='CPU_control':proof['status']='complete_scripted_student_real_CPU_engine_training_control'
    elif damage=='no_reopen':proof['all_native_completed_outputs_reopened_without_change']=False
    elif damage=='GPU_boolean':proof['physical_GPU']=True
    elif damage=='changed_manifest':proof['native_manifest_sha256']='0'*64
    elif damage=='changed_checkpoint_receipt':proof['checkpoint_sha256']='0'*64
    elif damage=='changed_model_asset':(args['decoder_model']/'tokenizer.json').write_text('{"changed":true}')
    elif damage=='wrong_explicit_model':args['decoder_model']=tmp_path/'different-model'
    elif damage=='wrong_explicit_expert':args['expert_weights']=tmp_path/'different-expert'
    elif damage=='changed_source_code':proof['source_sha256']['inference.py']='0'*64
    elif damage=='no_dependency_receipt':args['input_digest_receipt']=None
    elif damage=='changed_receipt_identity':
        p=args['input_digest_receipt'];a=json.loads(p.read_text());a['inputs'][str(args['decoder_model']/'config.json')]['signature'][0]+=1;atomic_json(p,a)
    else:
        if damage=='missing_native_digest_session':n.pop('runtime_execution')
        else:n['inputs'][str(tmp_path/'unowned') ]={'sha256':'0'*64,'bytes':0}
        atomic_json(native,n);proof['native_manifest_sha256']=digest(native)
    atomic_json(execution,proof)
    with pytest.raises((ValueError,FileNotFoundError)):
        verified_neural_games(args['games'],execution,**reader_args(args))


def test_missing_dependency_receipt_and_wrong_pinned_reference_fail_without_decoder_loading(tmp_path):
    args=fixture(tmp_path);bound=BoundInputs()
    with pytest.raises(ValueError,match='explicit decoder'):
        checkpoint_dependencies(args['checkpoint'],bound)
    with pytest.raises(ValueError,match='checkpoint metadata'):
        checkpoint_dependencies(args['checkpoint'],BoundInputs(),decoder_model=args['decoder_model'],
            decoder_manifest=tmp_path/'different.json',input_digest_receipt=args['input_digest_receipt'])


def test_actual_student_roots_keep_their_own_type_and_reuse_the_existing_authority(tmp_path):
    args=fixture(tmp_path);pool,base=authority(tmp_path)
    out=tmp_path/'student-roots'
    result=pools.prepare(**args,authority=pool,data=base['data'],heldout_data=base['heldout_data'],output=out)
    rows=load_jsonl(out/'roots.jsonl');proof=json.loads((out/'manifest.json').read_text())
    reserved=set(json.loads((pool/'heldout-positions.json').read_text()))
    assert rows and proof['kind']=='prepared_student_training_game_roots'
    assert result['candidate_training_histories']==len(rows)
    assert result['complete_actor_assets_verified'] is True
    assert result['new_distillation_targets_generated'] is False
    assert result['all_future_mined_branches_isolated'] is False
    assert not (out/'heldout-positions.json').exists()  # Reuse the exact authority; do not make a divergent copy.
    games={g['game_id']:g for g in load_jsonl(args['games']/'games.jsonl')}
    for row in rows:
        assert row['source_kind']=='student_engine_training_game' and row['split']=='train'
        assert row['student_game_continuation_is_best_move_label'] is False
        assert row['provenance']=='student_engine_training_game_search_input'
        assert 'answer' not in row and not any(k.startswith('recorded_') for k in row)
        g=games[row['game_id']];p=row['ply']
        assert row['future_moves']==g['moves'][p:p+8]
        assert row['student_game_source']['played_move']==g['moves'][p]
        states=replay(row['fen'],row['future_moves'])
        assert not {position_key(s) for s in states}&reserved
    selected,actual=pools.prepared_student_roots(out,base['data'],1,20261013,identities(base))
    assert selected==rows[:1] and actual==reserved
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}
    pools.prepared_student_roots(out,base['data'],1,20261013,identities(base))
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}


@pytest.mark.parametrize('damage',['forged_root','lost_root','forged_count','different_authority','range_overrun'])
def test_complete_root_pool_is_reconstructed_even_if_tampered_outputs_are_rehashed(tmp_path,damage):
    args=fixture(tmp_path);pool,base=authority(tmp_path);out=tmp_path/'student-roots'
    pools.prepare(**args,authority=pool,data=base['data'],heldout_data=base['heldout_data'],output=out)
    rows=load_jsonl(out/'roots.jsonl');p=out/'manifest.json';proof=json.loads(p.read_text())
    offset=0
    if damage=='forged_root':
        rows[0]['student_game_source']['played_move']='a0a9';write_jsonl(out/'roots.jsonl',rows)
    elif damage=='lost_root':write_jsonl(out/'roots.jsonl',rows[1:])
    elif damage=='forged_count':
        a=json.loads((out/'counts.json').read_text());a['candidate_training_histories']+=1;atomic_json(out/'counts.json',a);proof['verification']=a
    elif damage=='different_authority':proof['config']['shared_isolation_authority']['reserved_positions_binding']['sha256']='0'*64
    else:offset=len(rows)
    for name in ['roots.jsonl','counts.json']:
        proof['outputs'][str(out/name)]={'sha256':digest(out/name),'bytes':(out/name).stat().st_size}
    atomic_json(p,proof)
    with pytest.raises(ValueError):pools.prepared_student_roots(out,base['data'],1,20261013,identities(base),offset=offset)


def test_protected_original_or_color_future_excludes_the_entire_selected_occurrence(tmp_path,monkeypatch):
    args=fixture(tmp_path);pool,base=authority(tmp_path);out=tmp_path/'student-roots'
    pools.prepare(**args,authority=pool,data=base['data'],heldout_data=base['heldout_data'],output=out)
    first=load_jsonl(out/'roots.jsonl')[0]
    from xqgeneral.symmetry import mirror_fen
    protected=position_key(mirror_fen(replay(first['fen'],first['future_moves'])[-1]))
    original=pools.shared_reservations
    def with_extra(*args,**kwargs):
        reserved,proof=original(*args,**kwargs)
        return reserved|{protected},dict(proof,reserved_positions=len(reserved|{protected}))
    monkeypatch.setattr(pools,'shared_reservations',with_extra)
    later=tmp_path/'with-extra-protection'
    result=pools.prepare(**args,authority=pool,data=base['data'],heldout_data=base['heldout_data'],output=later)
    assert first['id'] not in {r['id'] for r in load_jsonl(later/'roots.jsonl')}
    assert result['excluded']['reserved_root_or_actual_continuation']>=1


def test_CPU_control_receipt_cannot_create_a_student_training_pool(tmp_path):
    args=fixture(tmp_path);proof=json.loads(args['execution_proof'].read_text())
    proof['status']='complete_scripted_student_real_CPU_engine_training_control'
    atomic_json(args['execution_proof'],proof)
    with pytest.raises(ValueError,match='actual neural gameplay'):
        pools.prepare(**args,authority=tmp_path/'unused',data=tmp_path/'unused-data',
            heldout_data=tmp_path/'unused-heldout',output=tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()


def test_equivalent_relative_base_paths_reopen_but_conflicting_aliases_do_not(tmp_path,monkeypatch):
    args=fixture(tmp_path);pool,base=authority(tmp_path);out=tmp_path/'student-roots'
    pools.prepare(**args,authority=pool,data=base['data'],heldout_data=base['heldout_data'],output=out)
    expected=pools.prepared_student_roots(out,base['data'],1,20261013,identities(base))
    relative={str(Path(name).relative_to(tmp_path)):binding for name,binding in identities(base).items()}
    monkeypatch.chdir(tmp_path)
    assert pools.prepared_student_roots(out,Path('data'),1,20261013,relative)==expected
    alternate={'data/../'+name:binding for name,binding in relative.items()}
    assert pools.prepared_student_roots(out,Path('data/../data'),1,20261013,alternate)==expected
    conflicting=dict(relative,**{str(base['data']/'train.jsonl'):{'sha256':'0'*64,'bytes':0}})
    with pytest.raises(ValueError,match='duplicate path aliases'):
        pools.prepared_student_roots(out,Path('data'),1,20261013,conflicting)
