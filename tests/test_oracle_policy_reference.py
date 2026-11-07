"""Controlled raw logs and fake assets test contracts, not teacher qualification."""
from copy import deepcopy
import json

import pytest

from xqgeneral import oracle_policy_reference as reference
from xqgeneral.evidence import atomic_json,code_identity,digest,history_key,load_jsonl,manifest,write_jsonl
from xqgeneral.oracle import parse_analysis
from xqgeneral.reviewed_explanations import ENGINE_QUALIFIED_KIND
from xqgeneral.rules import START_FEN,legal_moves,play,replay


def sample(moves=('b0c2',)):
    history=replay(START_FEN,list(moves));fen=history[-1];key=history_key(history)
    row=dict(id='validation-'+key,feature_key=key,game_id='heldout-'+key,split='validation',
        initial_fen=START_FEN,moves=list(moves),history=history,fen=fen,stage='explanation',
        answer='controlled current reviewed answer')
    lines=[]
    for rank,move in enumerate(sorted(legal_moves(fen))[:3],1):
        pv=[move];position=play(fen,move)
        for _ in range(6):
            choices=sorted(legal_moves(position))
            if not choices:break
            move=choices[0];pv.append(move);position=play(position,move)
        lines.append(f'info depth 18 multipv {rank} score cp {21-rank} nodes 1000000 pv '+' '.join(pv))
    lines.append('bestmove '+sorted(legal_moves(fen))[0])
    return row,parse_analysis(fen,lines,1000000)


def setup(tmp_path,monkeypatch,moves=(('b0c2',),('h0g2',))):
    data=tmp_path/'data';data.mkdir()
    records=[sample(m)[0] for m in moves]
    write_jsonl(data/'validation.jsonl',records)
    proof=manifest(ENGINE_QUALIFIED_KIND,{},(),[data/'validation.jsonl'],
        dict(heldout_files_byte_identical=True,by_split={'validation':len(records)}))
    atomic_json(data/'manifest.json',proof)
    executable=tmp_path/'engine';weights=tmp_path/'weights'
    executable.write_bytes(b'unit fake engine asset, never executed');weights.write_bytes(b'unit fake weights')
    calls=[]
    class Fake:
        def __init__(self,e,w,threads,multipv):
            assert (e,w,threads,multipv)==(executable,weights,1,3)
        def analyze(self,fen,nodes,initial,moves):
            assert nodes==1000000 and initial==START_FEN
            calls.append(list(moves));return sample(moves)[1]
        def close(self):pass
    monkeypatch.setattr(reference,'Pikafish',Fake)
    args=dict(data=data,executable=executable,weights=weights,workers=2)
    return args,calls,records


def legacy(tmp_path,args,records,*,difference=False,same_parser=False):
    directory=tmp_path/'legacy';directory.mkdir()
    originals=[dict(row,answer='earlier teacher answer') for row in records]
    labels=directory/'labels.jsonl';write_jsonl(labels,originals)
    rows=[]
    for index,row in enumerate(originals):
        response=sample(row['moves'])[1]
        if difference and index==0:
            response=deepcopy(response);response['candidates'][0]['score']+=1
        rows.append(dict(id=row['id'],record=row,raw=row['answer'],judgment=dict(
            first_move_legal=True,first_move_oracle=response,
            engine_judgments=[dict(fen=row['fen'],oracle=response)])))
    path=directory/'judged-teacher-labels.jsonl';write_jsonl(path,rows)
    config=dict(source='existing_teacher_labels',labels=str(labels),nodes=1000000)
    verified=dict(examples=len(records),split='validation',student_model_loaded_or_evaluated=False,
        independent_test_answers_loaded=False,teacher_labels_changed=False)
    proof=manifest('existing_teacher_labels_independent_engine_reference',config,
        [labels,args['executable'],args['weights']],[path],verified)
    if difference and not same_parser:
        proof['code']['source_sha256']['src/xqgeneral/oracle.py']='0'*64
    atomic_json(directory/'manifest.json',proof)
    return directory


def snapshot(directory):
    return {p.name:(digest(p),p.stat().st_mtime_ns) for p in directory.iterdir() if p.is_file()}


def test_fresh_references_preserve_validation_and_completed_reopen_never_searches(tmp_path,monkeypatch):
    args,calls,records=setup(tmp_path,monkeypatch);out=tmp_path/'references'
    before=(args['data']/'validation.jsonl').read_bytes()
    counts=reference.prepare(**args,output=out)
    assert counts['root_queries_generated']==counts['validation_history_contexts']==2
    assert counts['root_queries_reused']==0 and len(calls)==2
    assert [x['record'] for x in load_jsonl(out/'roots.jsonl')]==[reference.validation_record(x) for x in records]
    assert (args['data']/'validation.jsonl').read_bytes()==before
    saved=snapshot(out)
    assert reference.prepare(**dict(args,workers=1),output=out,resume=True)==counts
    assert len(calls)==2 and snapshot(out)==saved


def test_compatible_historical_queries_are_reused_despite_changed_reviewed_prose(tmp_path,monkeypatch):
    args,calls,records=setup(tmp_path,monkeypatch);old=legacy(tmp_path,args,records);out=tmp_path/'out'
    counts=reference.prepare(**args,output=out,reuse_reference=old)
    assert counts['root_queries_reused']==2 and counts['root_queries_generated']==0 and calls==[]
    assert not load_jsonl(out/'rejected-reuse.jsonl')
    assert {x['origin']['kind'] for x in load_jsonl(out/'roots.jsonl')}=={'reused_validation_root_query'}
    assert reference.prepare(**args,output=out,reuse_reference=old,resume=True)==counts and calls==[]


def test_legacy_parser_difference_searches_only_the_incompatible_root(tmp_path,monkeypatch):
    args,calls,records=setup(tmp_path,monkeypatch);old=legacy(tmp_path,args,records,difference=True)
    before=(old/'judged-teacher-labels.jsonl').read_bytes();out=tmp_path/'out'
    counts=reference.prepare(**args,output=out,reuse_reference=old)
    assert counts['root_queries_generated']==counts['root_queries_reused']==1
    assert calls==[records[0]['moves']]
    assert load_jsonl(out/'rejected-reuse.jsonl')[0]['reason']=='legacy_parser_interpretation_differs'
    assert (old/'judged-teacher-labels.jsonl').read_bytes()==before
    before=snapshot(out)
    reference.prepare(**args,output=out,reuse_reference=old,resume=True)
    assert snapshot(out)==before and len(calls)==1


def test_same_parser_response_contradiction_is_source_corruption_not_a_new_search(tmp_path,monkeypatch):
    args,calls,records=setup(tmp_path,monkeypatch);old=legacy(tmp_path,args,records,difference=True,same_parser=True)
    with pytest.raises(ValueError,match='same native parser'):
        reference.prepare(**args,output=tmp_path/'out',reuse_reference=old)
    assert calls==[] and not (tmp_path/'out').exists()


@pytest.mark.parametrize('damage',['training','test','history','feature','terminal','duplicate'])
def test_validation_ownership_and_full_context_are_mandatory(tmp_path,monkeypatch,damage):
    args,calls,records=setup(tmp_path,monkeypatch)
    if damage in {'training','test'}:records[0]['split']='train' if damage=='training' else 'test'
    elif damage=='history':records[0]['history']=records[0]['history'][:-1]
    elif damage=='feature':records[0]['feature_key']='0'*64
    elif damage=='terminal':records[0]['initial_fen']='9/9/9/9/9/9/9/9/9/9 w - - 0 1'
    else:records[1]=deepcopy(records[0])
    path=args['data']/'validation.jsonl';write_jsonl(path,records)
    proof=json.loads((args['data']/'manifest.json').read_text())
    proof['outputs'][str(path)]={'sha256':digest(path),'bytes':path.stat().st_size}
    atomic_json(args['data']/'manifest.json',proof)
    with pytest.raises(ValueError):reference.prepare(**args,output=tmp_path/'out')
    assert calls==[]


@pytest.mark.parametrize('damage',['asset','budget','context','coverage'])
def test_legacy_reuse_requires_exact_assets_budget_context_and_coverage(tmp_path,monkeypatch,damage):
    args,calls,records=setup(tmp_path,monkeypatch);old=legacy(tmp_path,args,records)
    proof=json.loads((old/'manifest.json').read_text())
    if damage=='asset':args['weights'].write_bytes(b'different fake asset')
    elif damage=='budget':proof['config']['nodes']=100000
    else:
        path=old/'judged-teacher-labels.jsonl';rows=load_jsonl(path)
        if damage=='context':rows[0]['record']['game_id']='wrong-game'
        else:rows.pop()
        write_jsonl(path,rows);proof['outputs'][str(path)]={'sha256':digest(path),'bytes':path.stat().st_size}
    atomic_json(old/'manifest.json',proof)
    with pytest.raises(ValueError):reference.prepare(**args,output=tmp_path/'out',reuse_reference=old)
    assert calls==[]


def test_completed_output_is_natively_reparsed_even_after_tampering_is_resealed(tmp_path,monkeypatch):
    args,calls,_=setup(tmp_path,monkeypatch);out=tmp_path/'out';reference.prepare(**args,output=out)
    path=out/'roots.jsonl';rows=load_jsonl(path);rows[0]['response']['candidates'][0]['score']+=1;write_jsonl(path,rows)
    proof=json.loads((out/'manifest.json').read_text());proof['outputs'][str(path)]={'sha256':digest(path),'bytes':path.stat().st_size}
    atomic_json(out/'manifest.json',proof);before=snapshot(out)
    with pytest.raises(ValueError,match='native raw-log'):
        reference.prepare(**args,output=out,resume=True)
    assert snapshot(out)==before and len(calls)==2


def test_partial_output_and_changed_completed_budget_are_preserved(tmp_path,monkeypatch):
    args,calls,_=setup(tmp_path,monkeypatch);partial=tmp_path/'partial';partial.mkdir();(partial/'evidence').write_text('preserve')
    with pytest.raises(FileExistsError):reference.prepare(**args,output=partial,resume=True)
    assert (partial/'evidence').read_text()=='preserve' and calls==[]
    out=tmp_path/'out';reference.prepare(**args,output=out);before=snapshot(out)
    with pytest.raises(ValueError,match='configuration'):
        reference.prepare(**dict(args,nodes=2000000),output=out,resume=True)
    assert snapshot(out)==before and len(calls)==2


@pytest.mark.parametrize('options',[{'nodes':100000},{'workers':0},{'workers':True}])
def test_reference_search_budgets_are_explicit_and_bounded(tmp_path,monkeypatch,options):
    args,calls,_=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError):reference.prepare(**dict(args,**options),output=tmp_path/'out')
    assert calls==[]
