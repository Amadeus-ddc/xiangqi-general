"""Combine typed explanation/rehearsal/policy sources with isolated new validation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading

from .curriculum_data import STAGES
from .course_tasks import PAPER_PROFILE,PAPER_TASKS
from .evidence import atomic_json,code_identity,digest,history_key,manifest,position_key
from .explanations import parse_explanation
from .foundation_preflight import bounded_results
from .oracle import parse_analysis
from .oracle_policy_data import (KIND as POLICY_KIND,NUMERIC_STAGE,PROFILE as POLICY_PROFILE,
    _binding,_context_key,_isolated_positions,_numeric_after_isolation,Excluded)
from .oracle_policy_reference import KIND as REFERENCE_KIND,PROFILE as REFERENCE_PROFILE,RECORD_FIELDS
from .policy_data import QUESTION as MOVE_QUESTION,move_label
from .recorded_search_inputs import BoundInputs
from .reviewed_explanations import ENGINE_QUALIFIED_KIND
from .rules import future_fens
from .search_distillation import NUMERIC_CHILD_QUESTION
from .symmetry import mirror_fen
from .verified_inputs import input_session_from_environment

KIND='auxiliary_oracle_policy_corpus'
PROFILE='seven_stage_explanation_policy_replay_v1'
ALL_STAGES=('explanation',*STAGES,NUMERIC_STAGE,'move_quality')
FILES=('train.jsonl','validation.jsonl','test.jsonl','rejected-train.jsonl',
       'rejected-new-validation.jsonl','duplicates.jsonl','row-sources.jsonl','counts.json','contract.json')
CLOSURE=('oracle_policy_corpus.py','oracle_policy_reference.py','oracle_policy_data.py','oracle.py',
         'rules.py','evidence.py','explanations.py','policy_data.py','symmetry.py','course_tasks.py',
         'curriculum_data.py','recorded_search_inputs.py','foundation_preflight.py')


def _json(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def _closure():
    identities=code_identity()['source_sha256']
    return {name:identities['src/xqgeneral/'+name] for name in CLOSURE}


def _lines(path):
    offset=0
    with Path(path).open('rb') as stream:
        for number,raw in enumerate(stream,1):
            if not raw.endswith(b'\n') or not raw.strip():
                raise ValueError('Corpus sources need complete nonempty JSONL records')
            yield dict(path=str(path),line=number,byte_offset=offset,bytes=len(raw),
                raw_line_sha256=hashlib.sha256(raw).hexdigest()),raw,json.loads(raw)
            offset+=len(raw)


def _basic(row,split,stages):
    if (not isinstance(row,dict) or row.get('split')!=split or row.get('stage') not in stages or
            any(not isinstance(row.get(k),str) or not row[k] for k in
                ['id','game_id','feature_key','initial_fen','fen','question','answer']) or
            not isinstance(row.get('moves'),list) or not isinstance(row.get('history'),list) or
            any(not isinstance(value,str) or not value for key in ['moves','history'] for value in row[key]) or
            len(row['history'])!=len(row['moves'])+1 or not row['history'] or
            row['history'][0]!=row['initial_fen'] or row['history'][-1]!=row['fen'] or
            history_key(row['history'])!=row['feature_key']):
        raise ValueError('Source ownership, stage or immutable native-history identity differs')
    if row['stage'] in STAGES and (row.get('task_profile')!=PAPER_PROFILE or
            row.get('task_type') not in PAPER_TASKS[row['stage']]):
        raise ValueError('Course replay lost its original paper task profile')
    if row['stage']=='move_quality' and row['question']!=MOVE_QUESTION:
        raise ValueError('The native single-move question changed')
    if row['stage']==NUMERIC_STAGE and row['question']!=NUMERIC_CHILD_QUESTION:
        raise ValueError('The native numeric question changed')


@lru_cache(maxsize=32768)
def _line_positions(fen,line):
    return frozenset([position_key(fen),*[position_key(x) for x in future_fens(fen,line)]])


@lru_cache(maxsize=65536)
def _mirror(key):
    return position_key(mirror_fen(key+' - - 0 1'))


def _expanded(positions):
    return set(positions)|{_mirror(key) for key in positions}


def _positions(row):
    lines=[row.get('future_moves',[]),*row.get('future_branches',[])]
    if row['stage']=='explanation':
        value=parse_explanation(row['answer'])
        lines += [value['pv'],*[x['pv'] for x in value.get('branches',[])]]
    elif row['stage']==NUMERIC_STAGE:
        value=json.loads(row['answer'])
        lines += [value['pv'],*[[move] for move in value['candidates']]]
    result={position_key(row['fen'])}
    for line in lines:
        if not isinstance(line,list) or any(not isinstance(move,str) for move in line):
            raise ValueError('Supervised future lines must preserve their native move lists')
        result.update(_line_positions(row['fen'],tuple(line)))
    return result


def _dataset(directory,kind,stages,bound):
    proof=bound.json(directory/'manifest.json')
    if proof.get('status')!='complete' or proof.get('kind') not in kind:
        raise ValueError('Use completed typed corpus sources')
    for split in ['train','validation','test']:
        path=directory/(split+'.jsonl');bound.bind(path,_binding(proof.get('outputs'),path))
    return dict(path=directory,proof=proof,stages=tuple(stages))


def _source_count(source,split):
    proof=source['proof'];verified=proof['verification']
    if proof['kind']==POLICY_KIND:
        value=verified.get('question_rows' if split=='train' else split+'_rows')
    else:
        value=verified.get('roots' if proof['kind']=='merged_search_consolidated_dataset' else 'by_split',{}).get(split)
    if type(value) is not int or value<0:
        raise ValueError('A typed source lost its actual split counts')
    return value


def _sources(explanations,replay,policies,references,bound):
    primary=_dataset(explanations[0],{ENGINE_QUALIFIED_KIND},['explanation'],bound)
    if primary['proof']['verification'].get('completed_parent_native_and_isolation_proofs_reused') is not True:
        raise ValueError('Original explanations need completed native and isolation proofs')
    teachers=[primary]
    for path in explanations[1:]:
        source=_dataset(path,{'merged_search_consolidated_dataset'},['explanation'],bound)
        verified=source['proof'].get('verification',{})
        if any(verified.get(key) is not True for key in [
                'all_source_collections_require_immediate_mate_consistency',
                'all_leaf_accepted_and_rejected_labels_reconstructed',
                'all_actual_teacher_deployments_and_responses_reopened']):
            raise ValueError('Strict explanation replay needs complete native collection reconstruction')
        for split in ['validation','test']:
            if bound.artifacts[str(path/(split+'.jsonl'))]!=bound.artifacts[str(primary['path']/(split+'.jsonl'))]:
                raise ValueError('Strict explanations changed the original protected split bytes')
        teachers.append(source)
    policy=_dataset(policies,{POLICY_KIND},[NUMERIC_STAGE,'move_quality'],bound)
    p=policy['proof'];contract=bound.json(policies/'contract.json',_binding(p['outputs'],policies/'contract.json'))
    if (contract.get('logical',{}).get('profile')!=POLICY_PROFILE or
            p.get('config',{}).get('logical')!=contract['logical'] or
            p.get('verification',{}).get('root_and_all_recorded_candidate_PVs_and_color_mirrors_isolated') is not True or
            p['verification'].get('question_views_counted_by_unique_history_union') is not True):
        raise ValueError('Oracle supervision lost its native profile or shared isolation proof')
    current=code_identity()['source_sha256']
    if any(p.get('code',{}).get('source_sha256',{}).get('src/xqgeneral/'+name)!=current['src/xqgeneral/'+name]
           for name in ['oracle.py','rules.py','oracle_policy_data.py']):
        raise ValueError('Oracle corpus parser, rules or data contract differ')
    shared=contract['shared_isolation_authority']
    if (shared.get('profile')!='shared_canonical_heldout_authority_v1' or
            Path(contract['logical'].get('heldout_data','')).resolve()!=primary['path'] or
            Path(contract['logical'].get('authority','')).resolve()!=Path(shared.get('authority','')).resolve()):
        raise ValueError('Require the same original explanation heldout authority')
    for split in ['validation','test']:
        if bound.artifacts[str(primary['path']/(split+'.jsonl'))]!=shared['heldout_label_identities'][split]:
            raise ValueError('Protected original explanation identity differs')
    authority=Path(shared['authority'])
    bound.bind(authority/'manifest.json',shared['authority_manifest_binding'])
    positions=bound.json(authority/'heldout-positions.json',shared['reserved_positions_binding'])
    if (not isinstance(positions,list) or any(not isinstance(key,str) for key in positions) or
            len(set(positions))!=len(positions) or len(positions)!=shared['reserved_positions']):
        raise ValueError('Shared protected-position coverage differs')
    base=Path(contract['logical']['data']).resolve()
    for split in ['train','validation','test']:
        path=base/(split+'.jsonl')
        bound.bind(path,shared['base_data_identities'][str(path)])
    rehearsals=[]
    for path in replay:
        source=_dataset(path,{'latent_only_course_replay_data_preflight'},STAGES,bound)
        proof=source['proof'];verified=proof.get('verification',{})
        if (Path(proof['config']['source_data']).resolve()!=base or
                _binding(proof.get('inputs'),base/'train.jsonl')!=shared['base_data_identities'][str(base/'train.jsonl')] or
                any(verified.get(key) is not True for key in [
                    'all_selected_train_and_validation_answers_natively_regraded',
                    'all_selected_full_histories_and_futures_natively_checked',
                    'original_completed_train_validation_test_split_proof_reused',
                    'training_rows_drawn_only_from_original_train_split',
                    'original_training_rows_preserved_byte_for_byte','validation_bytes_unchanged'])):
            raise ValueError('Course replay must preserve the actual original native source and split proof')
        source_validation=base/'validation.jsonl'
        if bound.artifacts[str(path/'validation.jsonl')]!=shared['base_data_identities'][str(source_validation)]:
            raise ValueError('Original course validation bytes changed')
        rehearsals.append(source)
    reference=bound.json(references/'manifest.json')
    cfg=reference.get('config',{}).get('logical',{})
    if (reference.get('status')!='complete' or reference.get('kind')!=REFERENCE_KIND or
            cfg.get('profile')!=REFERENCE_PROFILE or
            Path(cfg.get('data','')).resolve()!=primary['path'] or
            reference.get('verification',{}).get('validation_only') is not True or cfg.get('nodes',0)<1000000):
        raise ValueError('New task validation needs its completed exact original-context root references')
    for name in ['executable','weights']:
        left=Path(contract['logical'][name]);right=Path(cfg[name])
        if left!=right or _binding(p['inputs'],left)!=_binding(reference['inputs'],right):
            raise ValueError('Training and validation oracle assets differ')
        bound.bind(left,_binding(p['inputs'],left))
    for name in ['oracle.py','rules.py','oracle_policy_reference.py']:
        if reference.get('code',{}).get('source_sha256',{}).get('src/xqgeneral/'+name)!=current['src/xqgeneral/'+name]:
            raise ValueError('Validation reference parser, rules or source contract differ')
    if _binding(reference.get('inputs'),primary['path']/'validation.jsonl')!=bound.artifacts[str(primary['path']/'validation.jsonl')]:
        raise ValueError('Reference source differs from the original validation bytes')
    bound.bind(references/'roots.jsonl',_binding(reference['outputs'],references/'roots.jsonl'))
    return teachers,rehearsals,policy,reference,set(positions),shared


def _reference_views(references,proof,original,known_validation,reserved):
    numeric,move,rejected,extra=[],[],[],set()
    cfg=proof['config']['logical'];seen=set();expected_order=list(original)
    for pointer,_,item in _lines(references/'roots.jsonl'):
        record=item.get('record',{});identity=record.get('id')
        if (len(seen)>=len(expected_order) or identity!=expected_order[len(seen)] or
                identity in seen or {k:original[identity].get(k) for k in RECORD_FIELDS}!=record):
            raise ValueError('New validation references changed original explanation context/order coverage')
        seen.add(identity);response=item['response']
        if parse_analysis(record['fen'],response['raw_output'],cfg['nodes'])!=response:
            raise ValueError('New validation reference differs from raw engine reparse')
        positions={position_key(record['fen'])}
        for candidate in response['candidates']:
            positions.update(_line_positions(record['fen'],tuple(candidate['pv'])))
        extra.update(_expanded(positions))
        base=dict(record,provenance='pinned_validation_only_engine_root_reference')
        metadata=dict(source_reference=pointer,source_record_id=identity,
            full_teacher_explanation=False,history_before_initial_FEN_reconstructed=False,
            teacher=dict(provider='pinned_Pikafish',neural_prose_generated=False,requested_nodes=cfg['nodes']))
        one=move_label(dict(id='oracle-validation-'+identity,record=base,oracle={'best_move':response['best_move']}))
        one.update(metadata)
        choices=[one]
        try:
            target,terminal=_numeric_after_isolation(dict(record=record,response=response))
        except Excluded as error:
            rejected.append(dict(id=identity,stage=NUMERIC_STAGE,reason=str(error)))
        else:
            choices.append(dict(base,id='oracle-validation-numeric-'+identity,stage=NUMERIC_STAGE,
                task_type='move_PV_candidates_evaluation_v1',question=NUMERIC_CHILD_QUESTION,
                answer=_json(target),future_moves=target['pv'],future_branches=[[m] for m in target['candidates']],
                target_line_termination=terminal,**metadata))
        for row in choices:
            target_positions=_expanded(_positions(row))
            if any(key in reserved and key not in known_validation for key in target_positions):
                rejected.append(dict(id=identity,stage=row['stage'],reason='new_target_enters_other_protected_positions'))
            else:
                (numeric if row['stage']==NUMERIC_STAGE else move).append(row)
    if seen!=set(original):raise ValueError('New validation references do not cover all original explanations')
    return [*numeric,*move],rejected,extra


def prepare(explanations,replay,policies,references,output,*,workers=4,resume=False):
    if (not isinstance(explanations,(list,tuple)) or not explanations or
            not isinstance(replay,(list,tuple)) or not replay or type(workers) is not int or not 1<=workers<=16):
        raise ValueError('Require original and optional strict explanations, course replay and bounded workers')
    explanations=[Path(p).resolve() for p in explanations];replay=[Path(p).resolve() for p in replay]
    policies,references,output=map(lambda p:Path(p).resolve(),[policies,references,output])
    if len(set([*explanations,*replay,policies,references]))!=len(explanations)+len(replay)+2:
        raise ValueError('Corpus sources must be distinct declared directories')
    if any(output==source or output in source.parents for source in [*explanations,*replay,policies,references]):
        raise ValueError('A corpus output cannot replace an input directory or its parent')
    if output.exists():
        if not resume:raise FileExistsError('Preserve completed/partial corpus; use a fresh output or completed --resume')
        return _reopen(explanations,replay,policies,references,output,workers=workers)
    if resume:raise FileNotFoundError('Only a completed corpus can be reopened; preserve partial attempts')
    return _build(explanations,replay,policies,references,output,workers=workers)


def _reopen(explanations,replay,policies,references,output,*,workers):
    bound=BoundInputs();proof=bound.json(output/'manifest.json')
    if (proof.get('status')!='complete' or proof.get('kind')!=KIND or
            proof.get('config',{}).get('profile')!=PROFILE):
        raise ValueError('Only a completed corpus with the same profile can be reopened')
    contract=bound.json(output/'contract.json',_binding(proof.get('outputs'),output/'contract.json'))
    expected=dict(explanations=[str(p) for p in explanations],replay=[str(p) for p in replay],
        policies=str(policies),references=str(references),closure=_closure())
    if any(contract.get(key)!=value for key,value in expected.items()):
        raise ValueError('Completed corpus sources or code changed; use a fresh attempt')
    for name,identity in proof['inputs'].items():bound.bind(name,identity)
    for name in FILES:bound.bind(output/name,_binding(proof['outputs'],output/name))
    with TemporaryDirectory(prefix='oracle-policy-reopen-',dir=output.parent) as work:
        rebuilt=Path(work)/'dataset'
        counts=_build(explanations,replay,policies,references,rebuilt,workers=workers)
        if counts!=proof['verification'] or any(
                dict(sha256=digest(rebuilt/name),bytes=(rebuilt/name).stat().st_size)!=bound.artifacts[str(output/name)]
                for name in FILES):
            raise ValueError('Completed corpus differs from full native source replay')
    bound.unchanged()
    return counts


def _build(explanations,replay,policies,references,output,*,workers):
    bound=BoundInputs()
    closure=_closure()
    teachers,rehearsals,policy,reference,reserved,shared=_sources(explanations,replay,policies,references,bound)
    sources=[*teachers,*rehearsals,policy]
    validation=[];original={};seen={};duplicates=[];known=set();validation_games=set()
    for source in sources:
        count=0
        for pointer,raw,row in _lines(source['path']/'validation.jsonl'):
            count+=1
            _basic(row,'validation',source['stages']);identity=row['id']
            if identity in seen:
                if (seen[identity]['path']==pointer['path'] or
                        seen[identity]['raw_line_sha256']!=pointer['raw_line_sha256'] or seen[identity]['bytes']!=len(raw)):
                    raise ValueError('Duplicate original validation IDs have conflicting bytes')
                duplicates.append(dict(split='validation',id=identity,first=seen[identity],duplicate=pointer));continue
            seen[identity]=pointer;validation.append((raw,row,pointer))
            known.update(_expanded(_positions(row)));validation_games.add(row['game_id'])
            if source is teachers[0]:original[identity]=row
        expected=_source_count(source,'validation')
        if count!=expected:raise ValueError('Original validation-row coverage differs')
    additions,rejected_validation,extra=_reference_views(references,reference,original,known,reserved)
    original_validation_counts=dict(Counter(row['stage'] for _,row,_ in validation))
    for row in additions:
        if row['id'] in seen:raise ValueError('Derived validation IDs collide with original rows')
        raw=(_json(row)+'\n').encode();pointer=dict(path=str(references/'roots.jsonl'),source_reference=row['source_reference'])
        seen[row['id']]=pointer;validation.append((raw,row,pointer))
    protected_before=len(reserved);reserved.update(extra)
    output.mkdir(parents=True)
    atomic_json(output/'contract.json',dict(profile=PROFILE,explanations=[str(p) for p in explanations],
        replay=[str(p) for p in replay],policies=str(policies),references=str(references),
        closure=closure,
        original_shared_isolation_authority=shared,source_native_history_proofs_reused=True,
        new_validation_questions_do_not_add_independent_positions=True))
    local=threading.local();handles=[];lock=threading.Lock()
    def oracle(group):
        row=group[0][2];query=row['source_query'];ledger=Path(query['ledger'])
        if not hasattr(local,'files'):local.files={}
        stream=local.files.get(str(ledger))
        if stream is None:
            bound_identity=_binding(policy['proof']['inputs'],ledger)
            # Bind outside workers before dispatch, then use independent readers per thread.
            if str(ledger) not in ledger_bindings or ledger_bindings[str(ledger)]!=bound_identity:
                raise ValueError('Oracle pointer has no exact completed source binding')
            stream=ledger.open('rb');local.files[str(ledger)]=stream
            with lock:handles.append(stream)
        stream.seek(query['byte_offset']);raw=stream.readline()
        if hashlib.sha256(raw).hexdigest()!=query['raw_line_sha256']:
            raise ValueError('Original oracle query pointer changed')
        item=json.loads(raw)
        expected={k:(row['source_record_id'] if k=='id' else row[k]) for k in RECORD_FIELDS}
        if any(paired['source_query']!=query or paired['source_context_id']!=row['source_context_id'] or
                {k:(paired['source_record_id'] if k=='id' else paired[k]) for k in RECORD_FIELDS}!=expected
                for _,_,paired in group):
            raise ValueError('Paired oracle questions changed their shared query context')
        if item.get('record')!=expected or _context_key(item['record'])!=row['source_context_id'] or item.get('searchmoves'):
            raise ValueError('Oracle question differs from its original unrestricted full-history query')
        try:_isolated_positions(item,reserved)
        except Excluded as error:return str(error)
        return None
    ledger_bindings={}
    for name,identity in policy['proof']['inputs'].items():
        if Path(name).name=='oracle-queries.partial.jsonl':ledger_bindings[str(Path(name).resolve())]=bound.bind(name,identity)
    def decide(group):
        if any(row['game_id'] in validation_games for _,_,row in group):return group,'validation_game_owner'
        if group[0][2]['stage'] in {NUMERIC_STAGE,'move_quality'}:return group,oracle(group)
        for _,_,row in group:
            if _expanded(_positions(row))&reserved:return group,'shared_or_extended_heldout_root_future_or_mirror'
        return group,None
    source_counts={};contexts=set();train_counts=Counter();rejected=Counter()
    def groups():
        for source in sources:
            count=0;pending=[];key=None;visited=set()
            for pointer,raw,row in _lines(source['path']/'train.jsonl'):
                _basic(row,'train',source['stages']);count+=1
                if source is policy:
                    current=row['source_context_id']
                    if pending and current!=key:
                        visited.add(key);yield pending;pending=[]
                    if current in visited:raise ValueError('Oracle context grouping changed')
                    key=current;pending.append((pointer,raw,row))
                else:yield [(pointer,raw,row)]
            if pending:yield pending
            expected=_source_count(source,'train')
            if expected!=count:raise ValueError('Source training-row coverage differs')
            source_counts[str(source['path'])]=count
    streams={name:(output/name).open('xb') for name in FILES[:7]}
    try:
        for raw,row,pointer in validation:
            streams['validation.jsonl'].write(raw)
            streams['row-sources.jsonl'].write((_json(dict(id=row['id'],split='validation',stage=row['stage'],source=pointer))+'\n').encode())
        for item in rejected_validation:streams['rejected-new-validation.jsonl'].write((_json(item)+'\n').encode())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk in bounded_results(pool,lambda chunk:[decide(group) for group in chunk],groups(),workers,32):
                for group,reason in chunk:
                    for pointer,raw,row in group:
                        identity=row['id']
                        if identity in seen:
                            previous=seen[identity]
                            if ('raw_line_sha256' not in previous or previous.get('split','validation')!='train' or
                                    previous['path']==pointer['path'] or previous['raw_line_sha256']!=pointer['raw_line_sha256'] or
                                    previous['bytes']!=len(raw)):
                                raise ValueError('Source IDs conflict across splits, sources or bytes')
                            duplicates.append(dict(split='train',id=identity,first=previous,duplicate=pointer));continue
                        seen[identity]=dict(pointer,split='train')
                        if reason:
                            rejected[row['stage']+'/'+reason]+=1
                            streams['rejected-train.jsonl'].write((_json(dict(id=identity,stage=row['stage'],source=pointer,reason=reason))+'\n').encode());continue
                        streams['train.jsonl'].write(raw);train_counts[row['stage']]+=1;contexts.add(row['feature_key'])
                        streams['row-sources.jsonl'].write((_json(dict(id=identity,split='train',stage=row['stage'],source=pointer))+'\n').encode())
        for item in duplicates:streams['duplicates.jsonl'].write((_json(item)+'\n').encode())
    finally:
        for stream in streams.values():stream.close()
        for stream in handles:stream.close()
    validation_counts=Counter(row['stage'] for _,row,_ in validation)
    duplicate_counts=Counter(item['split'] for item in duplicates)
    if sum(source_counts.values())!=sum(train_counts.values())+sum(rejected.values())+duplicate_counts['train']:
        raise ValueError('Corpus training source accounting differs')
    if set(train_counts)!=set(ALL_STAGES) or set(validation_counts)!=set(ALL_STAGES) or len(contexts)<32:
        raise ValueError('All seven stages need actual train/validation coverage and32 training histories')
    counts=dict(train_rows_by_stage=dict(train_counts),validation_rows_by_stage=dict(validation_counts),
        original_validation_rows_by_stage=original_validation_counts,training_question_rows=sum(train_counts.values()),
        training_unique_history_contexts=len(contexts),validation_question_rows=len(validation),
        validation_unique_history_contexts=len({row['feature_key'] for _,row,_ in validation}),
        new_independent_validation_history_contexts=0,source_training_rows=source_counts,
        rejected_training_by_stage_reason=dict(rejected),rejected_new_validation_questions=len(rejected_validation),
        identical_duplicate_rows_coalesced=len(duplicates),test_rows=0,
        identical_duplicate_rows_by_split=dict(duplicate_counts),
        shared_protected_positions=protected_before,extended_protected_positions=len(reserved),
        unchanged_source_row_bytes_retained=True,original_validation_row_bytes_preserved=True,
        upstream_complete_native_history_proofs_reused=True,all_supervised_positions_and_mirrors_isolated=True,
        no_new_policy_validation_target_enters_other_protected_positions=True,
        source_oracle_queries=0,student_or_teacher_LLM_queries=0,new_SGD_updates=0,
        final_independent_test_rows_parsed_or_tokenized=0,formal_training_recipe_compiled=False)
    atomic_json(output/'counts.json',counts);bound.unchanged()
    if _closure()!=closure:raise ValueError('Corpus source code changed during preparation')
    proof=manifest(KIND,dict(profile=PROFILE,workers=workers,contract=str(output/'contract.json')),
        (),[output/name for name in FILES],counts);proof['inputs']=bound.artifacts
    bound.unchanged();atomic_json(output/'manifest.json',proof)
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--explanations',nargs='+',required=True);parser.add_argument('--replay',nargs='+',required=True)
    for name in ['policies','references','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    with input_session_from_environment():print(json.dumps(prepare(**vars(parser.parse_args())),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
