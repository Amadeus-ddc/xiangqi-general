"""Pinned validation-only engine references, with checked historical-query reuse."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading

from .evidence import atomic_json, code_identity, digest, history_key, manifest
from .oracle import Pikafish, parse_analysis
from .oracle_policy_data import _binding
from .recorded_search_inputs import BoundInputs
from .reviewed_explanations import ENGINE_QUALIFIED_KIND
from .rules import adjudicate, replay, validate_position
from .verified_inputs import input_session_from_environment

KIND = 'heldout_oracle_policy_references'
PROFILE = 'validation_only_exact_root_search_v1'
RECORD_FIELDS = ('id','feature_key','game_id','split','initial_fen','moves','history','fen')
FILES = ('roots.jsonl','rejected-reuse.jsonl','counts.json','contract.json')
CLOSURE = ('oracle_policy_reference.py','oracle.py','rules.py','evidence.py',
           'oracle_policy_data.py','recorded_search_inputs.py')


def _json(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def _lines(path):
    with Path(path).open('rb') as stream:
        for number, raw in enumerate(stream,1):
            if not raw.endswith(b'\n') or not raw.strip():
                raise ValueError('A validation source needs complete nonempty JSONL records')
            yield number, raw, json.loads(raw)


def validation_record(row):
    if (not isinstance(row,dict) or row.get('split')!='validation' or
            any(not isinstance(row.get(k),str) or not row[k]
                for k in ['id','feature_key','game_id','initial_fen','fen']) or
            any(not isinstance(row.get(k),list) or any(not isinstance(v,str) for v in row[k])
                for k in ['moves','history'])):
        raise ValueError('References require validation-owned full-history records')
    validate_position(row['initial_fen'])
    history=replay(row['initial_fen'],row['moves'])
    if (history!=row['history'] or history[-1]!=row['fen'] or
            history_key(history)!=row['feature_key'] or adjudicate(row['initial_fen'],row['moves'])['ended']):
        raise ValueError('Validation history, feature identity or terminal status differs')
    return {k:row[k] for k in RECORD_FIELDS}


def _source(data,bound):
    proof=bound.json(data/'manifest.json')
    if (proof.get('status')!='complete' or proof.get('kind')!=ENGINE_QUALIFIED_KIND or
            proof.get('verification',{}).get('heldout_files_byte_identical') is not True):
        raise ValueError('Use a complete engine-qualified explanation source with preserved heldout files')
    path=data/'validation.jsonl';bound.bind(path,_binding(proof.get('outputs'),path))
    rows=[];ids=set();features=set()
    for _,_,row in _lines(path):
        record=validation_record(row)
        if row.get('stage')!='explanation' or record['id'] in ids or record['feature_key'] in features:
            raise ValueError('Original explanation validation needs unique IDs and histories')
        ids.add(record['id']);features.add(record['feature_key']);rows.append(record)
    if not rows or proof['verification'].get('by_split',{}).get('validation')!=len(rows):
        raise ValueError('Original validation coverage differs from its completed source')
    return rows


def _response(record,response,nodes):
    if (not isinstance(response,dict) or response.get('requested_nodes')!=nodes or
            not isinstance(response.get('raw_output'),list) or not response['raw_output'] or
            any(not isinstance(line,str) for line in response['raw_output'])):
        raise ValueError('The pinned root budget and complete original engine text are required')
    return parse_analysis(record['fen'],response['raw_output'],nodes)


def _reuse(directory,records,executable,weights,nodes,bound):
    if directory is None:
        return {},[],dict(legacy_reference_supplied=False)
    proof=bound.json(directory/'manifest.json');config=proof.get('config',{});verified=proof.get('verification',{})
    if (proof.get('status')!='complete' or proof.get('kind')!='existing_teacher_labels_independent_engine_reference' or
            config.get('source')!='existing_teacher_labels' or config.get('nodes')!=nodes or
            verified.get('split')!='validation' or verified.get('student_model_loaded_or_evaluated') is not False or
            verified.get('independent_test_answers_loaded') is not False or verified.get('teacher_labels_changed') is not False):
        raise ValueError('Only completed teacher-validation root references may be reused')
    for path in [executable,weights]:
        bound.bind(path,_binding(proof.get('inputs'),path))
    original=Path(config['labels']).resolve()
    bound.bind(original,_binding(proof.get('inputs'),original))
    original_rows=[row for _,_,row in _lines(original)]
    originals={row['id']:row for row in original_rows}
    if len(originals)!=len(original_rows):
        raise ValueError('Historical validation IDs must be unique')
    source=directory/'judged-teacher-labels.jsonl';bound.bind(source,_binding(proof.get('outputs'),source))
    expected={row['id']:row for row in records}
    if set(originals)!=set(expected) or verified.get('examples')!=len(records):
        raise ValueError('Historical and current validation must cover the same original IDs')
    current_parser=code_identity()['source_sha256']['src/xqgeneral/oracle.py']
    old_parser=proof.get('code',{}).get('source_sha256',{}).get('src/xqgeneral/oracle.py')
    if not isinstance(old_parser,str) or len(old_parser)!=64:
        raise ValueError('Historical reference lost its parser identity')
    cached,rejected,seen={},[],set()
    for number,raw,item in _lines(source):
        identity=item.get('id')
        if identity not in expected or identity in seen or item.get('record')!=originals[identity] or item.get('raw')!=originals[identity]['answer']:
            raise ValueError('Historical reference rows differ from their hash-bound original labels')
        seen.add(identity);record=expected[identity]
        if {k:item['record'].get(k) for k in RECORD_FIELDS}!=record:
            raise ValueError('Historical root differs from the current validation context')
        judge=item.get('judgment',{});response=judge.get('first_move_oracle')
        judged=judge.get('engine_judgments',[])
        if (not judged or judged[0].get('fen')!=record['fen'] or judged[0].get('oracle')!=response or
                judge.get('first_move_legal') is not True):
            raise ValueError('Unrestricted root query is missing or differs from the original judgment')
        parsed=_response(record,response,nodes)
        origin=dict(kind='reused_validation_root_query',reference=str(source),line=number,
            source_line_sha256=hashlib.sha256(raw).hexdigest(),reference_manifest=str(directory/'manifest.json'))
        if parsed!=response:
            if old_parser==current_parser:
                raise ValueError('Stored response contradicts the same native parser')
            rejected.append(dict(id=identity,reason='legacy_parser_interpretation_differs',
                original_parser_sha256=old_parser,current_parser_sha256=current_parser,source=origin))
        else:
            cached[identity]=dict(response=response,origin=origin)
    if seen!=set(expected):
        raise ValueError('Historical reference coverage is incomplete')
    return cached,rejected,dict(legacy_reference_supplied=True,legacy_parser_sha256=old_parser,
        legacy_validation_rows=len(seen),reused_rows=len(cached),fresh_rows_required=len(records)-len(cached))


def _checked_rows(path,records,nodes,cached):
    result=[]
    for _,_,item in _lines(path):
        index=len(result)
        if index>=len(records) or item.get('record')!=records[index]:
            raise ValueError('Published references changed original validation order or context')
        record=records[index]
        if _response(record,item.get('response'),nodes)!=item['response']:
            raise ValueError('Published reference differs from its native raw-log reparse')
        if record['id'] in cached:
            if {k:item.get(k) for k in ['response','origin']}!=cached[record['id']]:
                raise ValueError('Published reused query differs from its original source')
        elif item.get('origin')!={'kind':'fresh_validation_root_query','requested_nodes':nodes}:
            raise ValueError('Fresh validation query provenance differs')
        result.append(item)
    if len(result)!=len(records):raise ValueError('Published validation root coverage differs')
    return result


def prepare(data,output,*,executable,weights,nodes=1000000,workers=4,reuse_reference=None,resume=False):
    if (type(nodes) is not int or nodes<1000000 or type(workers) is not int or not 1<=workers<=16 or
            type(resume) is not bool):
        raise ValueError('Require at least one million nodes and bounded positive CPU workers')
    data,output,executable,weights=map(lambda p:Path(p).resolve(),[data,output,executable,weights])
    reuse_reference=None if reuse_reference is None else Path(reuse_reference).resolve()
    bound=BoundInputs();records=_source(data,bound)
    for path in [executable,weights]:bound.bind(path)
    cached,rejected,reuse=_reuse(reuse_reference,records,executable,weights,nodes,bound)
    code=code_identity()['source_sha256']
    logical=dict(profile=PROFILE,data=str(data),output=str(output),executable=str(executable),weights=str(weights),
        nodes=nodes,reuse_reference=None if reuse_reference is None else str(reuse_reference),
        closure={name:code['src/xqgeneral/'+name] for name in CLOSURE})
    contract=dict(logical=logical,inputs=bound.artifacts)
    counts=dict(validation_history_contexts=len(records),root_queries_reused=len(cached),
        root_queries_generated=len(records)-len(cached),rejected_reuse_queries=len(rejected),
        validation_only=True,training_rows_created=0,independent_test_rows_read=0,
        original_validation_records_modified=False,student_or_teacher_LLM_queries=0,**reuse)
    if output.exists():
        if not resume or not (output/'manifest.json').exists():
            raise FileExistsError('Preserve partial reference output; use a fresh directory or completed --resume')
        outputs=BoundInputs();saved=outputs.json(output/'manifest.json')
        if (saved.get('kind')!=KIND or saved.get('status')!='complete' or saved.get('config',{}).get('logical')!=logical or
                saved.get('inputs')!=bound.artifacts or saved.get('verification')!=counts or
                json.loads((output/'contract.json').read_text())!=contract):
            raise ValueError('Completed reference source, parser or configuration changed')
        if {Path(name).resolve() for name in saved.get('outputs',{})}!={output/name for name in FILES}:
            raise ValueError('Reference output coverage changed')
        for name in FILES:outputs.bind(output/name,_binding(saved['outputs'],output/name))
        _checked_rows(output/'roots.jsonl',records,nodes,cached)
        if json.loads((output/'counts.json').read_text())!=counts:
            raise ValueError('Published reference counts differ')
        if [x for _,_,x in _lines(output/'rejected-reuse.jsonl')]!=rejected:
            raise ValueError('Rejected reuse provenance changed')
        bound.unchanged();outputs.unchanged();return counts
    if resume:raise FileNotFoundError('Completed reference output is required for --resume')
    output.mkdir(parents=True);atomic_json(output/'contract.json',contract)
    local,engines,lock=threading.local(),[],threading.Lock()
    def one(record):
        if record['id'] in cached:return dict(record=record,**cached[record['id']])
        if not hasattr(local,'engine'):
            local.engine=Pikafish(executable,weights,threads=1,multipv=3)
            with lock:engines.append(local.engine)
        response=local.engine.analyze(record['fen'],nodes,record['initial_fen'],record['moves'])
        if _response(record,response,nodes)!=response:
            raise ValueError('New root search differs from its native parser')
        return dict(record=record,response=response,origin=dict(kind='fresh_validation_root_query',requested_nodes=nodes))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool,(output/'roots.jsonl').open('x') as stream:
            for item in pool.map(one,records):stream.write(_json(item)+'\n')
    finally:
        for engine in engines:engine.close()
    with (output/'rejected-reuse.jsonl').open('x') as stream:
        for item in rejected:stream.write(_json(item)+'\n')
    _checked_rows(output/'roots.jsonl',records,nodes,cached)
    atomic_json(output/'counts.json',counts);bound.unchanged()
    result=manifest(KIND,dict(logical=logical,execution=dict(workers=workers,engine_threads=1,multipv=3)),
        (),[output/name for name in FILES],counts)
    result['inputs']=bound.artifacts;bound.unchanged();atomic_json(output/'manifest.json',result)
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['data','output','executable','weights']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--reuse-reference');parser.add_argument('--nodes',type=int,default=1000000)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--resume',action='store_true')
    args=vars(parser.parse_args())
    with input_session_from_environment():
        counts=prepare(**args)
        print(json.dumps(dict(counts=counts,actual_new_queries_this_invocation=0 if args['resume'] else counts['root_queries_generated']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
