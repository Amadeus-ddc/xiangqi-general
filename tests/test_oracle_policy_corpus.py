"""Controlled typed sources test corpus isolation, not upstream scientific quality."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from xqgeneral import oracle_policy_corpus as corpus,oracle_policy_data as policy
from xqgeneral.evidence import atomic_json,digest,history_key,load_jsonl,manifest,position_key,write_jsonl
from xqgeneral.explanations import EXPLANATION_QUESTION,move_facts
from xqgeneral.oracle import parse_analysis
from xqgeneral.oracle_policy_reference import KIND as REFERENCE_KIND,PROFILE as REFERENCE_PROFILE,RECORD_FIELDS
from xqgeneral.reviewed_explanations import ENGINE_QUALIFIED_KIND
from xqgeneral.rules import START_FEN,legal_moves,play,replay


def binding(path):return dict(sha256=digest(path),bytes=path.stat().st_size)


def context(moves,split='train'):
    history=replay(START_FEN,moves);key=history_key(history)
    return dict(id=split+'-'+key,feature_key=key,game_id=split+'-game-'+key,split=split,
        initial_fen=START_FEN,moves=moves,history=history,fen=history[-1])


def query(record,nodes=100000):
    lines=[]
    for rank,first in enumerate(sorted(legal_moves(record['fen']))[:3],1):
        pv=[first];fen=play(record['fen'],first)
        while len(pv)<7:
            choices=sorted(legal_moves(fen))
            if not choices:break
            move=choices[(rank+len(pv))%len(choices)];pv.append(move);fen=play(fen,move)
        lines.append(f'info depth 18 multipv {rank} score cp {25-rank} nodes {nodes} pv '+' '.join(pv))
    lines.append('bestmove '+sorted(legal_moves(record['fen']))[0])
    return dict(record=record,searchmoves=[],requested_nodes=nodes,
        response=parse_analysis(record['fen'],lines,nodes))


def explanation(record):
    q=query(record);response=q['response'];best=response['candidates'][0]
    target=dict(move=best['move'],pv=best['pv'],candidates=[c['move'] for c in response['candidates']],
        branches=[dict(move=c['move'],pv=c['pv'],evaluation=dict(type='cp',value=c['score'],perspective='side_to_move'))
                  for c in response['candidates']],
        evaluation=dict(type='cp',value=best['score'],perspective='side_to_move'),
        facts=move_facts(record['fen'],best['move']),explanation='这是用于合同测试的合法变化。')
    return dict(record,id='explanation-'+record['id'],stage='explanation',question=EXPLANATION_QUESTION,
        answer=json.dumps(target,ensure_ascii=False),future_moves=target['pv'])


def qa(record,stage):
    task=corpus.PAPER_TASKS[stage][0]
    return dict(record,id='qa-'+stage+'-'+record['id'],stage=stage,task_type=task,
        task_profile=corpus.PAPER_PROFILE,query={},question='原生来源合同测试',answer='测试',future_moves=[])


def dataset(path,kind,train,validation,verification,inputs=()):
    path.mkdir()
    write_jsonl(path/'train.jsonl',train);write_jsonl(path/'validation.jsonl',validation)
    # The corpus may hash this protected artifact but must never parse it.
    (path/'test.jsonl').write_bytes(b'independent test content is deliberately not JSON\n')
    proof=manifest(kind,{},inputs,[path/(s+'.jsonl') for s in ['train','validation','test']],verification)
    atomic_json(path/'manifest.json',proof)
    return proof


def rebind(directory,name,rows):
    path=directory/name;write_jsonl(path,rows)
    proof=json.loads((directory/'manifest.json').read_text());proof['outputs'][str(path)]=binding(path)
    atomic_json(directory/'manifest.json',proof)


def fixture(tmp_path, *, validation_histories=1):
    reds=sorted(legal_moves(START_FEN));records=[]
    for red in reds[:10]:
        black=sorted(legal_moves(play(START_FEN,red)))
        records.extend(context([red,move]) for move in black[:5])
    heldouts=[context([reds[-1],sorted(legal_moves(play(START_FEN,reds[-1])))[-1-i]],'validation')
              for i in range(validation_histories)]
    explanations=tmp_path/'explanations';base=tmp_path/'original-courses';rehearsal=tmp_path/'replay'
    policies=tmp_path/'policies';references=tmp_path/'references';authority=tmp_path/'authority'
    vals=[explanation(row) for row in heldouts];val=vals[0];train=[explanation(row) for row in records]
    dataset(explanations,ENGINE_QUALIFIED_KIND,train,vals,dict(
        completed_parent_native_and_isolation_proofs_reused=True,by_split=dict(train=len(train),validation=len(vals),test=1)))
    qa_train=[qa(row,stage) for row in records for stage in corpus.STAGES]
    qa_val=[qa(row,stage) for row in heldouts for stage in corpus.STAGES]
    dataset(base,'controlled_original_qa',qa_train,qa_val,dict(by_split=dict(train=len(qa_train),validation=len(qa_val),test=1)))
    required=['all_selected_train_and_validation_answers_natively_regraded',
        'all_selected_full_histories_and_futures_natively_checked',
        'original_completed_train_validation_test_split_proof_reused',
        'training_rows_drawn_only_from_original_train_split','original_training_rows_preserved_byte_for_byte',
        'validation_bytes_unchanged']
    proof=dataset(rehearsal,'latent_only_course_replay_data_preflight',qa_train,qa_val,
        dict(by_split=dict(train=len(qa_train),validation=len(qa_val),test=0),**dict.fromkeys(required,True)),[base/'train.jsonl'])
    proof['config']['source_data']=str(base);atomic_json(rehearsal/'manifest.json',proof)
    executable,weights=tmp_path/'engine',tmp_path/'nnue'
    executable.write_bytes(b'controlled engine identity; never run');weights.write_bytes(b'controlled weights')
    authority.mkdir();atomic_json(authority/'manifest.json',dict(status='controlled_fixture'))
    reserved=set().union(*(corpus._expanded(corpus._positions(row)) for row in vals))
    atomic_json(authority/'heldout-positions.json',sorted(reserved))
    shared=dict(profile='shared_canonical_heldout_authority_v1',authority=str(authority),
        authority_manifest_binding=binding(authority/'manifest.json'),reserved_positions_binding=binding(authority/'heldout-positions.json'),
        base_data_identities={str(base/(s+'.jsonl')):binding(base/(s+'.jsonl')) for s in ['train','validation','test']},
        heldout_label_identities={s:binding(explanations/(s+'.jsonl')) for s in ['validation','test']},reserved_positions=len(reserved))
    mining=tmp_path/'mining';mining.mkdir();ledger=mining/'oracle-queries.partial.jsonl'
    queries=[query(row) for row in records];write_jsonl(ledger,queries)
    rows=[];offset=0
    for number,(raw,item) in enumerate(zip(ledger.read_bytes().splitlines(keepends=True),queries),1):
        target,terminal=policy._numeric_after_isolation(item);key=policy._context_key(item['record'])
        pointer=dict(ledger=str(ledger),line=number,byte_offset=offset,raw_line_sha256=hashlib.sha256(raw).hexdigest())
        decision=dict(key=key,record=item['record'],query=pointer,target=target,termination=terminal,
            checked_positions=policy._isolated_positions(item,set()),signature=[target['move'],'cp',24])
        rows.extend(policy._views(decision));offset+=len(raw)
    proof=dataset(policies,policy.KIND,rows,[],dict(root_and_all_recorded_candidate_PVs_and_color_mirrors_isolated=True,
        question_views_counted_by_unique_history_union=True,question_rows=len(rows),validation_rows=0,test_rows=0),[ledger,executable,weights])
    logical=dict(profile=policy.PROFILE,data=str(base),heldout_data=str(explanations),authority=str(authority),
        executable=str(executable),weights=str(weights))
    atomic_json(policies/'contract.json',dict(logical=logical,shared_isolation_authority=shared))
    proof['config']['logical']=logical
    proof['outputs'][str(policies/'contract.json')]=binding(policies/'contract.json');atomic_json(policies/'manifest.json',proof)
    references.mkdir();root_rows=[]
    for row in vals:
        bare={k:row[k] for k in RECORD_FIELDS};q=query(bare,1000000)
        root_rows.append(dict(record=bare,response=q['response']))
    write_jsonl(references/'roots.jsonl',root_rows)
    logical=dict(profile=REFERENCE_PROFILE,data=str(explanations),executable=str(executable),weights=str(weights),nodes=1000000)
    proof=manifest(REFERENCE_KIND,dict(logical=logical),[executable,weights,explanations/'validation.jsonl'],
        [references/'roots.jsonl'],dict(validation_only=True))
    atomic_json(references/'manifest.json',proof)
    return dict(explanations=[explanations],replay=[rehearsal],policies=policies,references=references,
        output=tmp_path/'out',workers=2)


def snapshot(directory):return {p.name:(digest(p),p.stat().st_mtime_ns) for p in directory.iterdir() if p.is_file()}


def test_seven_stages_preserve_source_bytes_and_reopen_all_rows_without_queries(tmp_path):
    args=fixture(tmp_path);counts=corpus.prepare(**args)
    assert set(counts['train_rows_by_stage'])==set(counts['validation_rows_by_stage'])==set(corpus.ALL_STAGES)
    assert counts['training_unique_history_contexts']>=32 and counts['new_independent_validation_history_contexts']==0
    assert counts['validation_question_rows']==7 and counts['new_SGD_updates']==counts['source_oracle_queries']==0
    original=(args['explanations'][0]/'validation.jsonl').read_bytes()+(args['replay'][0]/'validation.jsonl').read_bytes()
    assert (args['output']/'validation.jsonl').read_bytes().startswith(original)
    assert (args['output']/'test.jsonl').read_bytes()==b'' and counts['final_independent_test_rows_parsed_or_tokenized']==0
    before=snapshot(args['output']);assert corpus.prepare(**dict(args,workers=1),resume=True)==counts
    assert snapshot(args['output'])==before


def test_cross_source_identical_rows_coalesce_but_do_not_inflate_histories(tmp_path):
    args=fixture(tmp_path);old=args['replay'][0];new=tmp_path/'second-replay';new.mkdir()
    for path in old.iterdir():
        if path.name!='manifest.json':(new/path.name).write_bytes(path.read_bytes())
    proof=json.loads((old/'manifest.json').read_text())
    proof['outputs']={str(new/Path(name).name):identity for name,identity in proof['outputs'].items()}
    atomic_json(new/'manifest.json',proof);args['replay'].append(new)
    counts=corpus.prepare(**args)
    assert counts['identical_duplicate_rows_by_split']==dict(validation=4,train=200)
    assert counts['training_unique_history_contexts']==50
    assert sum(counts['source_training_rows'].values())==counts['training_question_rows']+sum(counts['rejected_training_by_stage_reason'].values())+200


@pytest.mark.parametrize('damage',['validation_duplicate','training_duplicate','changed_history','wrong_course_profile','incomplete_validation','wrong_source_train_hash','wrong_reference_context','wrong_reference_parser','wrong_asset','wrong_heldout_source'])
def test_changed_source_contracts_and_incomplete_coverage_are_rejected(tmp_path,damage):
    args=fixture(tmp_path);directory=args['explanations'][0]
    if damage in {'validation_duplicate','incomplete_validation'}:
        rows=load_jsonl(directory/'validation.jsonl');rebind(directory,'validation.jsonl',rows*2 if damage=='validation_duplicate' else [])
    elif damage in {'training_duplicate','changed_history'}:
        rows=load_jsonl(directory/'train.jsonl')
        if damage=='training_duplicate':rows.append(deepcopy(rows[0]))
        else:rows[0]['history'][0]=None
        rebind(directory,'train.jsonl',rows)
    elif damage in {'wrong_course_profile','wrong_source_train_hash'}:
        directory=args['replay'][0]
        if damage=='wrong_course_profile':
            rows=load_jsonl(directory/'train.jsonl');rows[0]['task_profile']='legacy';rebind(directory,'train.jsonl',rows)
        else:
            path=directory/'manifest.json';proof=json.loads(path.read_text());next(iter(proof['inputs'].values()))['sha256']='0'*64;atomic_json(path,proof)
    elif damage=='wrong_heldout_source':
        directory=args['policies'];path=directory/'manifest.json';proof=json.loads(path.read_text())
        contract=json.loads((directory/'contract.json').read_text());contract['logical']['heldout_data']=str(tmp_path/'other')
        proof['config']['logical']=contract['logical'];atomic_json(directory/'contract.json',contract)
        proof['outputs'][str(directory/'contract.json')]=binding(directory/'contract.json');atomic_json(path,proof)
    else:
        directory=args['references'];path=directory/'manifest.json';proof=json.loads(path.read_text())
        if damage=='wrong_reference_context':
            rows=load_jsonl(directory/'roots.jsonl');rows[0]['record']['id']='other';rebind(directory,'roots.jsonl',rows)
        else:
            if damage=='wrong_reference_parser':proof['code']['source_sha256']['src/xqgeneral/oracle.py']='0'*64
            else:proof['inputs'][str(tmp_path/'engine')]['sha256']='0'*64
            atomic_json(path,proof)
    with pytest.raises((ValueError,KeyError)):corpus.prepare(**args)


def test_extended_validation_candidate_pv_and_mirror_remove_training_rows(tmp_path):
    args=fixture(tmp_path);directory=args['explanations'][0];rows=load_jsonl(directory/'train.jsonl')
    reference=load_jsonl(args['references']/'roots.jsonl')[0]
    response=reference['response'];pv=response['candidates'][1]['pv']
    end=corpus.future_fens(reference['record']['fen'],tuple(pv))[-1]
    pv=[*pv,sorted(legal_moves(end))[-1]]
    raw=list(response['raw_output']);raw[1]=raw[1].split(' pv ')[0]+' pv '+' '.join(pv)
    reference['response']=parse_analysis(reference['record']['fen'],raw,1000000)
    rebind(args['references'],'roots.jsonl',[reference])
    # This root is outside the old authority but enters a saved validation PV.
    moves=reference['record']['moves']+pv
    extra=explanation(context(moves));rows.append(extra);rebind(directory,'train.jsonl',rows)
    proof=json.loads((directory/'manifest.json').read_text());proof['verification']['by_split']['train']+=1;atomic_json(directory/'manifest.json',proof)
    counts=corpus.prepare(**args)
    rejected=load_jsonl(args['output']/'rejected-train.jsonl')
    assert any(row['id']==extra['id'] and row['reason']=='shared_or_extended_heldout_root_future_or_mirror' for row in rejected)
    assert counts['extended_protected_positions']>counts['shared_protected_positions']


def test_new_validation_variant_cannot_enter_other_protected_positions(tmp_path):
    args=fixture(tmp_path);authority=tmp_path/'authority'
    reference=load_jsonl(args['references']/'roots.jsonl')[0]
    # A short original E line never exposed the second ply of the new numeric target.
    source=args['explanations'][0];rows=load_jsonl(source/'validation.jsonl');value=json.loads(rows[0]['answer'])
    value['pv']=value['pv'][:1]
    for branch in value['branches']:branch['pv']=branch['pv'][:1]
    rows[0]['future_moves']=value['pv'];rows[0]['answer']=json.dumps(value,ensure_ascii=False);rebind(source,'validation.jsonl',rows)
    positions=sorted(corpus._expanded(corpus._positions(rows[0])))
    extra=next(iter(corpus._line_positions(reference['record']['fen'],tuple(reference['response']['candidates'][0]['pv'][:2]))-set(positions)))
    contract=json.loads((args['policies']/'contract.json').read_text());shared=contract['shared_isolation_authority']
    shared['heldout_label_identities']['validation']=binding(source/'validation.jsonl')
    if extra not in positions:positions.append(extra)
    atomic_json(authority/'heldout-positions.json',positions);shared['reserved_positions']=len(positions)
    shared['reserved_positions_binding']=binding(authority/'heldout-positions.json')
    atomic_json(args['policies']/'contract.json',contract)
    path=args['policies']/'manifest.json';proof=json.loads(path.read_text());proof['outputs'][str(args['policies']/'contract.json')]=binding(args['policies']/'contract.json');atomic_json(path,proof)
    path=args['references']/'manifest.json';proof=json.loads(path.read_text());proof['inputs'][str(source/'validation.jsonl')]=binding(source/'validation.jsonl');atomic_json(path,proof)
    # Move-only validation survives; numeric variant is excluded, so the seven-stage gate rejects this tiny fixture.
    with pytest.raises(ValueError,match='All seven stages'):corpus.prepare(**args)
    assert any(row['stage']=='engine_numeric' and row['reason']=='new_target_enters_other_protected_positions'
        for row in load_jsonl(args['output']/'rejected-new-validation.jsonl'))


@pytest.mark.parametrize('damage',['input','output','configuration','partial'])
def test_completed_reopen_never_overwrites_changed_or_partial_attempts(tmp_path,damage):
    args=fixture(tmp_path);corpus.prepare(**args)
    if damage=='input':(args['explanations'][0]/'train.jsonl').write_bytes(b'changed\n')
    elif damage=='output':(args['output']/'train.jsonl').write_bytes(b'changed\n')
    elif damage=='configuration':args['replay']=[]
    else:(args['output']/'manifest.json').unlink()
    before=snapshot(args['output'])
    with pytest.raises((ValueError,FileNotFoundError)):corpus.prepare(**args,resume=True)
    assert snapshot(args['output'])==before
