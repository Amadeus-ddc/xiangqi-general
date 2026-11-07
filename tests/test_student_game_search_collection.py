"""Real native source reconstruction with controlled execution/teacher fixtures."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from test_merge_search import setup_merge, rebind
from test_search_collection import response
from test_search_pilot import setup_pilot,ControlledChain
from test_student_game_inputs import fixture as neural_fixture, authority
from xqgeneral import collect_search, inference, merge_search, search_distillation as search, search_pilot, student_game_inputs as pools
from xqgeneral.evidence import atomic_json,digest,load_jsonl,manifest,position_key,write_jsonl
from xqgeneral.explanations import move_facts
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.rules import legal_moves,play


def setup_student_mining(tmp_path,monkeypatch):
    args=neural_fixture(tmp_path);pool,base=authority(tmp_path);owned=tmp_path/'student-pool'
    pools.prepare(**args,authority=pool,data=base['data'],heldout_data=base['heldout_data'],output=owned)
    rows=load_jsonl(owned/'roots.jsonl');assert len(rows)>2
    features=tmp_path/'features';features.write_bytes(b'controlled feature identity')
    loaded=[]
    monkeypatch.setattr(search,'feature_paths',lambda path:[Path(path)])
    class Student:
        def __init__(self,checkpoint,**kwargs):loaded.append((checkpoint,kwargs))
    class Oracle:
        def __init__(self,*args,**kwargs):pass
        def close(self):pass
    class Miner:
        def __init__(self,*args,**kwargs):self.counts=Counter()
        def mine(self,row):
            return {'id':row['id']+'-search','source_root_id':row['id'],'depth':0,'record':row},[], 'accepted_for_consolidation'
    monkeypatch.setattr(inference,'Predictor',Student);monkeypatch.setattr(search,'Pikafish',Oracle)
    monkeypatch.setattr(search,'SearchMiner',Miner)
    out=tmp_path/'mining'
    command=['search','--checkpoint',str(args['checkpoint']),'--data',str(base['data']),
        '--features',str(features),'--student-game-inputs',str(owned),'--output',str(out),
        '--executable',str(args['executable']),'--weights',str(args['weights']),
        '--limit','2','--root-offset','1','--max-depth','0']
    monkeypatch.setattr(sys,'argv',command);search.main()
    return args,base,owned,rows,out,loaded,command


def test_actual_source_pool_range_and_expert_are_forwarded_to_native_mining(tmp_path,monkeypatch):
    args,base,pool,rows,out,loaded,command=setup_student_mining(tmp_path,monkeypatch)
    proof=json.loads((out/'manifest.json').read_text());queries=load_jsonl(out/'queries.jsonl')
    assert [q['record'] for q in queries]==rows[1:3]
    assert proof['verification']['student_game_training_inputs'] is True
    assert proof['verification']['student_game_root_offset']==1
    assert 'unused_recorded_training_inputs' not in proof['verification']
    assert 'selected_student_game_root_ids_sha256' in proof['verification']
    assert 'selected_recorded_root_ids_sha256' not in proof['verification']
    assert loaded==[(str(args['checkpoint']),{'feature_cache':command[command.index('--features')+1],
        'weights':str(args['expert_weights'])})]
    reserved,info=collect_search.mining_reservations(out/'queries.jsonl',queries,base['data'],base['heldout_data'],
        BoundInputs(),student_game_inputs=pool,expected_attempted_root_ids=[r['id'] for r in rows[1:3]])
    assert reserved and info['student_game_mining_isolation_rechecked_during_collection'] is True
    assert 'recorded_mining_isolation_rechecked_during_collection' not in info


@pytest.mark.parametrize('damage',['raw_origin','actor','source_type','declared_pool','missing_manifest'])
def test_new_source_ownership_cannot_be_changed_or_fall_back_to_legacy_collection(tmp_path,monkeypatch,damage):
    args,base,pool,rows,out,loaded,command=setup_student_mining(tmp_path,monkeypatch)
    qp=out/'queries.jsonl';queries=load_jsonl(qp);mp=out/'manifest.json';proof=json.loads(mp.read_text())
    if damage=='raw_origin':queries[0]['record']['student_game_source']['played_move']='a0a9'
    elif damage=='actor':queries[0]['record']['student_game_source']['actor_identity']='0'*64
    elif damage=='source_type':proof['verification']['unused_recorded_training_inputs']=True
    elif damage=='declared_pool':proof['config'].pop('student_game_inputs')
    else:mp.unlink()
    write_jsonl(qp,queries)
    if damage!='missing_manifest':
        proof['outputs'][str(qp)]={'sha256':digest(qp),'bytes':qp.stat().st_size};atomic_json(mp,proof)
    with pytest.raises(ValueError):
        collect_search.mining_reservations(qp,queries,base['data'],base['heldout_data'],BoundInputs())


def test_new_source_cannot_mine_with_a_different_parent_checkpoint(tmp_path,monkeypatch):
    args,base,pool,rows,out,loaded,command=setup_student_mining(tmp_path,monkeypatch)
    other=tmp_path/'other.pt';other.write_bytes(b'other actor')
    command[command.index('--checkpoint')+1]=str(other)
    command[command.index('--output')+1]=str(tmp_path/'wrong-parent')
    monkeypatch.setattr(sys,'argv',command)
    with pytest.raises(ValueError,match='same original search actor'):search.main()
    assert len(loaded)==1 and not (tmp_path/'wrong-parent').exists()


def setup_mixed(tmp_path,monkeypatch):
    old=tmp_path/'recorded';old.mkdir()
    previous,leaves,_=setup_merge(old,monkeypatch)
    own=tmp_path/'neural';own.mkdir();args=neural_fixture(own);source=args['checkpoint']
    for leaf in leaves:
        mp=leaf.parent/'mining/manifest.json';proof=json.loads(mp.read_text())
        proof['config']['checkpoint']=str(source);proof['inputs'].pop(str(previous))
        proof['inputs'][str(source)]={'sha256':digest(source),'bytes':source.stat().st_size};atomic_json(mp,proof)
        rebind(leaf/'manifest.json','inputs',mp)
    base=old/'data';labels=old/'labels';authority=old/'merge-pool';pool=own/'student-pool'
    pools.prepare(**args,authority=authority,data=base,heldout_data=labels,output=pool)
    roots=load_jsonl(pool/'roots.jsonl');forbidden=set(json.loads((authority/'heldout-positions.json').read_text()))
    queries=[]
    for root in roots:
        choices=[m for m in sorted(legal_moves(root['fen'])) if position_key(play(root['fen'],m)) not in forbidden]
        if len(choices)<2:continue
        score={'type':'cp','value':0,'perspective':'side_to_move'}
        queries.append({'id':root['id']+'-search','source_root_id':root['id'],'depth':0,'record':root,
            'target_fields':{'move':choices[0],'pv':[choices[0]],'candidates':choices[:2],
                'facts':move_facts(root['fen'],choices[0]),'evaluation':score,
                'branches':[{'move':m,'pv':[m],'evaluation':score} for m in choices[:2]]}})
    assert queries
    part=own/'contribution';part.mkdir();mining=part/'mining';teacher=part/'teacher';mining.mkdir();teacher.mkdir()
    qp=mining/'queries.jsonl';rp=teacher/'responses.jsonl';write_jsonl(qp,queries)
    teachers=old/'complete-teachers.json';weights=old/'teacher-model/weights.manifest.json'
    identity=json.loads((leaves[0].parent/'teacher/deployment.json').read_text())
    answers=[]
    for query in queries:
        raw=response();raw.update(id=query['id'],teacher=identity);answers.append(raw)
    write_jsonl(rp,answers);atomic_json(teacher/'deployment.json',identity)
    atomic_json(teacher/'manifest.json',manifest('local_teacher_inference',
        {'input':str(qp),'config':str(teachers),'backend':'vllm'},[qp,teachers,weights],
        [rp,teacher/'deployment.json'],{'examples':len(answers),'neural_teacher_inference_executed':True,
            'teacher':identity,'controlled_CPU_fixture_not_real_teacher_execution':True}))
    original=json.loads((leaves[0].parent/'mining/manifest.json').read_text())
    config=dict(original['config'],recorded_inputs=None,student_game_inputs=str(pool),limit=len(roots))
    verification=dict(original['verification'],processed=len(roots),accepted_for_consolidation=len(queries),
        student_game_training_inputs=True,student_game_continuation_is_best_move_label=False,
        additional_reserved_canonical_and_explanation_positions=len(forbidden))
    verification.pop('unused_recorded_training_inputs')
    by_id={q['source_root_id']:q for q in queries}
    attempts=[{'source_root_id':r['id'],'trace':[],'query':by_id.get(r['id']),
        'reason':'accepted_for_consolidation' if r['id'] in by_id else 'recursion_limit'} for r in roots]
    verification['reasons']=dict(Counter(a['reason'] for a in attempts))
    write_jsonl(mining/'results.partial.jsonl',attempts)
    inputs=[source,*[base/f'{s}.jsonl' for s in ['train','validation','test']],*[pool/name for name in pools.ARTIFACTS]]
    atomic_json(mining/'manifest.json',manifest('search_distillation_mining',config,inputs,
        [qp,mining/'results.partial.jsonl'],verification))
    output=part/'labels'
    monkeypatch.setattr(sys,'argv',['collect_search','--queries',str(qp),'--responses',str(rp),
        '--config',str(teachers),'--heldout-data',str(base),'--validation-data',str(labels),
        '--student-game-inputs',str(pool),'--output',str(output)])
    collect_search.main()
    assert load_jsonl(output/'train.jsonl')
    return source,leaves+[output],pool


def test_mixed_sources_require_explicit_profile_and_reopen_all_native_provenance(tmp_path,monkeypatch):
    source,leaves,pool=setup_mixed(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='explicit shared canonical'):
        merge_search.merge_collections(leaves,source,tmp_path/'not-opted-in')
    assert not (tmp_path/'not-opted-in').exists()
    out=tmp_path/'merged'
    summary=merge_search.merge_collections(leaves,source,out,source_profile=merge_search.SHARED_SOURCE_PROFILE)
    assert summary['accepted_training_roots']>=35
    assert summary['source_kind_counts']=={'original_recorded_game_or_tactical_source':2,'student_engine_training_game':1}
    assert summary['all_source_mining_isolation_rechecked_during_collection'] is True
    assert 'recorded_mining_isolation_rechecked_during_collection' not in summary
    assert summary['shared_isolation_authority']==json.loads((pool/'manifest.json').read_text())['config']['shared_isolation_authority']
    assert merge_search.collected_dataset(out,source,BoundInputs())['verification']==summary


def test_neural_leaf_cannot_claim_recorded_isolation_to_bypass_source_profile(tmp_path,monkeypatch):
    source,leaves,pool=setup_mixed(tmp_path,monkeypatch)
    p=leaves[-1]/'manifest.json';proof=json.loads(p.read_text())
    proof['verification']['recorded_mining_isolation_rechecked_during_collection']=True;atomic_json(p,proof)
    with pytest.raises(ValueError,match='disguised'):
        merge_search.merge_collections(leaves,source,tmp_path/'bad',source_profile=merge_search.SHARED_SOURCE_PROFILE)


def test_student_game_pilot_uses_distinct_kind_and_complete_native_stage_chain(tmp_path,monkeypatch):
    old=tmp_path/'pipeline-fixture';old.mkdir()
    pipeline,validation,config,_,capability=setup_pilot(old,monkeypatch)
    own=tmp_path/'neural-fixture';own.mkdir();args=neural_fixture(own)
    settings=json.loads(config.read_text());authority=Path(settings.pop('recorded_inputs'))
    authority_config=json.loads((authority/'manifest.json').read_text())['config']
    pool=own/'student-pool'
    pools.prepare(**args,authority=authority,data=settings['data'],heldout_data=authority_config['heldout_data'],output=pool)
    settings['student_game_inputs']=str(pool);atomic_json(config,settings)
    parent=json.loads((pipeline/'manifest.json').read_text())
    parent['outputs']={str(args['checkpoint']):{'sha256':digest(args['checkpoint']),'bytes':args['checkpoint'].stat().st_size}}
    atomic_json(pipeline/'manifest.json',parent)
    capability['inputs']=parent['outputs'];atomic_json(validation/'manifest.json',capability)
    monkeypatch.setattr(search_pilot,'checked_sft',lambda *a:(args['checkpoint'],parent))
    root=load_jsonl(pool/'roots.jsonl')[0]
    forbidden=set(json.loads((authority/'heldout-positions.json').read_text()))
    choices=[m for m in sorted(legal_moves(root['fen'])) if position_key(play(root['fen'],m)) not in forbidden][:2]
    assert len(choices)==2
    score={'type':'cp','value':0,'perspective':'side_to_move'}
    query={'id':root['id']+'-search','source_root_id':root['id'],'depth':0,'record':root,
        'messages':[{'role':'user','content':'控制协议测试，未执行真实教师。'}],
        'target_fields':{'move':choices[0],'pv':[choices[0]],'candidates':choices,
            'facts':move_facts(root['fen'],choices[0]),'evaluation':score,
            'branches':[{'move':m,'pv':[m],'evaluation':score} for m in choices]}}
    chain=ControlledChain(monkeypatch,query);out=tmp_path/'pilot'
    result=search_pilot.run_pilot(pipeline,validation,'controlled-validation-session',config,out)
    assert result['accepted_structured_training_labels']==1
    assert json.loads((out/'manifest.json').read_text())['kind']=='clean_student_game_search_yield_pilot'
    assert len(chain.commands)==3 and chain.student_loads==chain.teacher_loads==1
    for module in ['xqgeneral.search_distillation','xqgeneral.collect_search']:
        command=next(c for c in chain.commands if module in c)
        assert command[command.index('--student-game-inputs')+1]==str(pool)
        assert '--recorded-inputs' not in command
    assert search_pilot.run_pilot(pipeline,validation,'controlled-validation-session',config,out,resume=True)==result
    assert len(chain.commands)==3
