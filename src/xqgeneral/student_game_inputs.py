"""Isolate real student training-game contexts under an existing held-out authority."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .evidence import atomic_json, code_identity, history_key, load_jsonl, manifest, position_key, write_jsonl
from .explanations import EXPLANATION_QUESTION
from .neural_training_games import verified_neural_games
from .recorded_search_inputs import (ARTIFACTS as RECORDED_ARTIFACTS, BoundInputs,
                                     checked_root_offset, prepared_roots)
from .rules import replay
from .sampling import checked_seed
from .symmetry import mirror_fen
from .training_game_scoring import PUBLICATIONS, score_games
from .verified_inputs import input_session_from_environment


KIND = 'prepared_student_training_game_roots'
ARTIFACTS = ('manifest.json','roots.jsonl','counts.json')
SCORING_CLOSURE = ('training_game_scoring.py','training_play.py','rules.py','explanations.py',
                   'sampling.py','evidence.py','oracle.py','calibration.py')


def _binding(proof,section,path):
    matches = [v for k,v in proof.get(section,{}).items() if Path(k).resolve()==Path(path).resolve()]
    if len(matches)!=1:
        raise ValueError('A source must bind exactly one identity for each required artifact')
    return matches[0]


def _canonical_inputs(bound):
    result = {}
    for name,value in bound.artifacts.items():
        path = str(Path(name).resolve())
        if path in result and result[path]!=value:
            raise ValueError('Input aliases disagree about actual source bytes')
        result[path] = value
    return result


def shared_reservations(authority,data,heldout_data,seed,bound):
    authority,data,heldout_data = map(lambda p:Path(p).resolve(),[authority,data,heldout_data])
    proof = bound.json(authority/'manifest.json')
    config = proof.get('config',{})
    if (proof.get('kind')!='prepared_unused_recorded_search_roots' or proof.get('status')!='complete' or
            Path(config.get('data','')).resolve()!=data or
            Path(config.get('heldout_data','')).resolve()!=heldout_data or config.get('seed')!=seed):
        raise ValueError('Student roots require the exact existing canonical isolation authority')
    inherited = config.get('base_data_identities_reused_from_completed_footprints')
    if not isinstance(inherited,dict) or not inherited:
        raise ValueError('The isolation authority lost its completed base-data identities')
    identities = {str(data/f'{split}.jsonl'):bound.bind(data/f'{split}.jsonl',
        _binding({'inputs':inherited},'inputs',data/f'{split}.jsonl')) for split in ['train','validation','test']}
    for name in RECORDED_ARTIFACTS:
        path = authority/name
        if name!='manifest.json':
            bound.bind(path,_binding(proof,'outputs',path))
    # This existing reader revalidates original ownership, complete root coverage,
    # protected-position counts, all canonical inputs and the exact base identities.
    _,reserved = prepared_roots(authority,data,1,seed,identities)
    for name,binding in proof['inputs'].items():
        bound.bind(name,binding)
    labels = {split:bound.bind(heldout_data/f'{split}.jsonl',
        _binding(proof,'inputs',heldout_data/f'{split}.jsonl')) for split in ['validation','test']}
    bound.unchanged()
    return reserved,shared_isolation_identity(authority,data,heldout_data,bound,len(reserved))


def shared_isolation_identity(authority,data,heldout_data,bound,reserved_count):
    """Describe an authority after its actual source-specific replay has succeeded."""
    authority,data,heldout_data = map(lambda p:Path(p).resolve(),[authority,data,heldout_data])
    proof = bound.json(authority/'manifest.json');config = proof.get('config',{})
    if (proof.get('kind')!='prepared_unused_recorded_search_roots' or proof.get('status')!='complete' or
            Path(config.get('data','')).resolve()!=data or Path(config.get('heldout_data','')).resolve()!=heldout_data or
            proof.get('verification',{}).get('reserved_root_and_future_positions')!=reserved_count):
        raise ValueError('Shared source isolation must use the exact completed canonical authority')
    identities = {str(data/f'{split}.jsonl'):bound.bind(data/f'{split}.jsonl',_binding(
        {'inputs':config['base_data_identities_reused_from_completed_footprints']},'inputs',data/f'{split}.jsonl'))
        for split in ['train','validation','test']}
    labels = {split:bound.bind(heldout_data/f'{split}.jsonl',_binding(proof,'inputs',heldout_data/f'{split}.jsonl'))
        for split in ['validation','test']}
    bound.bind(authority/'heldout-positions.json',_binding(proof,'outputs',authority/'heldout-positions.json'))
    return {'profile':'shared_canonical_heldout_authority_v1','authority':str(authority),
        'authority_manifest_binding':bound.artifacts[str(authority/'manifest.json')],
        'reserved_positions_binding':bound.artifacts[str(authority/'heldout-positions.json')],
        'base_data_identities':identities,'heldout_label_identities':labels,
        'reserved_positions':reserved_count}


def _reopen_scores(source,scores,executable,weights,scoring_policy,bound):
    source,scores = Path(source).resolve(),Path(scores).resolve()
    proof = bound.json(scores/'manifest.json')
    contract = proof.get('config',{})
    if (proof.get('kind')!='strong_scored_student_training_contexts' or proof.get('status')!='complete' or
            contract.get('source')!=str(source) or contract.get('code')!=proof.get('code')):
        raise ValueError('Only the exact completed strong scoring publication can supply student roots')
    current = code_identity()['source_sha256']
    for name in SCORING_CLOSURE:
        key = 'src/xqgeneral/'+name
        if proof.get('code',{}).get('source_sha256',{}).get(key)!=current[key]:
            raise ValueError('The scoring replay implementation differs from its original publication')
    outputs = [scores/'contract.json',scores/'oracle-queries.partial.jsonl',scores/'positions.partial.jsonl',
               *[scores/name for name in PUBLICATIONS]]
    if len(proof.get('outputs',{}))!=len(outputs) or {
            Path(p).resolve() for p in proof['outputs']}!=set(outputs):
        raise ValueError('Strong scoring changed its completed output coverage')
    for path in outputs:
        bound.bind(path,_binding(proof,'outputs',path))
    score_bound = BoundInputs()
    for path in [executable,weights,*([scoring_policy] if scoring_policy is not None else [])]:
        matches = [name for name in proof['inputs'] if Path(name).resolve()==Path(path).resolve()]
        if len(matches)!=1:
            raise ValueError('Strong scoring lost its explicitly supplied engine or policy bytes')
        score_bound.bind(matches[0],proof['inputs'][matches[0]])
    def forbidden_engine(role):
        raise ValueError('A completed scoring source cannot require new oracle queries during admission')
    rebuilt = score_games(source,scores,forbidden_engine,policy=contract['policy'],resume=True,
                          code=proof['code'],bound=score_bound)
    if rebuilt!=proof['verification']:
        raise ValueError('Strong scoring verification differs from its original raw-query replay')
    for path,binding in score_bound.artifacts.items():
        bound.bind(path,binding)
    return load_jsonl(scores/'selected-positions.jsonl')


def _plan(config,bound):
    seed = checked_seed(config['seed'])
    future_plies = config['future_plies']
    if type(future_plies) is not int or future_plies<1:
        raise ValueError('Student source continuation coverage must be positive')
    games,source_info = verified_neural_games(config['games'],config['execution_proof'],
        checkpoint=config['checkpoint'],expert_weights=config['expert_weights'],
        gameplay_policy=config['gameplay_policy'],executable=config['executable'],weights=config['weights'],
        decoder_model=config['decoder_model'],decoder_manifest=config['decoder_manifest'],
        input_digest_receipt=config['input_digest_receipt'],bound=bound)
    if source_info.get('source_model_assets_verified') is not True:
        raise ValueError('Student roots cannot admit unverified decoder or expert assets')
    selected = _reopen_scores(config['games'],config['scores'],config['executable'],config['weights'],
                              config['scoring_policy'],bound)
    reserved,authority = shared_reservations(config['authority'],config['data'],config['heldout_data'],seed,bound)
    by_index = {game['index']:game for game in games}
    roots,keys,excluded = [],set(),Counter()
    for position in selected:
        game = by_index.get(position.get('game_index'))
        ply = position.get('ply')
        if game is None or type(ply) is not int or not 0<=ply<len(game['turns']):
            raise ValueError('Selected student root is outside its original owned neural episode')
        turn = game['turns'][ply]
        record = turn['record']
        if (position.get('id')!=record['id'] or position.get('record')!=record or
                position.get('played_move')!=turn['played_move'] or
                record['history']!=replay(record['initial_fen'],record['moves']) or
                record.get('split')!='train' or record['game_id']!=game['game_id']):
            raise ValueError('Selected student root changed its actual full-history occurrence')
        future = game['moves'][ply:ply+future_plies]
        future_fens = game['history'][ply:ply+len(future)+1]
        footprint = {position_key(f) for f in future_fens}
        footprint.update(position_key(mirror_fen(f)) for f in future_fens)
        if footprint & reserved:
            excluded['reserved_root_or_actual_continuation'] += 1
            continue
        key = history_key(record['history'])
        if key in keys:
            excluded['duplicate_full_history'] += 1
            continue
        keys.add(key)
        provenance = {'game_index':game['index'],'ply':ply,'actor_identity':source_info['actor_identity'],
            'model_color':game['model_color'],'opponent_nodes':game['opponent_nodes'],
            'played_move':turn['played_move'],'move_source':position['move_source'],
            'cached_root':position['cached_root'],'source_position_id':position['id'],
            'native_gameplay_manifest_binding':source_info['native_manifest_binding'],
            'neural_execution_proof_binding':source_info['execution_proof_binding'],
            'strong_scoring_manifest_binding':bound.artifacts[str(Path(config['scores'])/'manifest.json')]}
        roots.append(dict(deepcopy(record),id='student-game-search-'+key,feature_key=key,ply=ply,
            question=EXPLANATION_QUESTION,provenance='student_engine_training_game_search_input',
            future_moves=future,future_branches=[],
            student_game_continuation_is_best_move_label=False,student_game_source=provenance))
    roots.sort(key=lambda row:(hashlib.sha256(f"{seed}/{row['feature_key']}".encode()).hexdigest(),row['feature_key']))
    counts = {'candidate_training_histories':len(roots),'distinct_training_episodes':len({r['game_id'] for r in roots}),
        'selected_scored_occurrences':len(selected),'excluded':dict(excluded),
        'candidate_counts_by_mover':dict(Counter('red' if r['fen'].split()[1]=='w' else 'black' for r in roots)),
        'neural_source_execution_receipt_verified':True,'complete_actor_assets_verified':True,
        'original_train_ownership_precedes_generation':True,'full_scoring_raw_query_replay_verified':True,
        'shared_heldout_authority_verified':True,'reserved_root_and_actual_future_positions_excluded':True,
        'additional_reserved_canonical_and_explanation_positions':len(reserved),
        'new_color_derived_candidates':0,'new_distillation_targets_generated':False,
        'new_student_teacher_or_oracle_queries':0,'student_SGD_updates':0,
        'all_future_mined_branches_isolated':False,'student_benefit_proven':False}
    bound.unchanged()
    return roots,counts,authority,reserved


def prepare(games,scores,execution_proof,authority,data,heldout_data,output,*,checkpoint,expert_weights,
            gameplay_policy,executable,weights,scoring_policy=None,decoder_model=None,decoder_manifest=None,
            input_digest_receipt=None,seed=20261013,future_plies=8):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve previous student-root pools; prepare into a fresh output directory')
    paths = dict(games=games,scores=scores,execution_proof=execution_proof,authority=authority,data=data,
        heldout_data=heldout_data,checkpoint=checkpoint,expert_weights=expert_weights,
        gameplay_policy=gameplay_policy,executable=executable,weights=weights,scoring_policy=scoring_policy,
        decoder_model=decoder_model,decoder_manifest=decoder_manifest,input_digest_receipt=input_digest_receipt)
    config = {name:str(Path(value).resolve()) if value is not None else None for name,value in paths.items()}
    config.update(seed=seed,future_plies=future_plies)
    bound = BoundInputs()
    roots,counts,common,_ = _plan(config,bound)
    if not roots:
        raise ValueError('No actually scored neural training roots survive the shared isolation authority')
    config['shared_isolation_authority'] = common
    output.mkdir(parents=True)
    write_jsonl(output/'roots.jsonl',roots);atomic_json(output/'counts.json',counts)
    saved = manifest(KIND,config,(),[output/'roots.jsonl',output/'counts.json'],counts)
    saved['inputs'] = _canonical_inputs(bound)
    bound.unchanged();atomic_json(output/'manifest.json',saved)
    return counts


def prepared_student_roots(pool,data,limit,seed,base_identities,*,offset=0):
    pool,data = Path(pool).resolve(),Path(data).resolve()
    offset = checked_root_offset(offset)
    if type(limit) is not int or limit<1:
        raise ValueError('Student root ranges require a positive limit')
    canonical_base = {str(Path(name).resolve()):binding for name,binding in base_identities.items()}
    if len(canonical_base)!=len(base_identities):
        raise ValueError('Base data identities must not contain duplicate path aliases')
    bound = BoundInputs();saved = bound.json(pool/'manifest.json')
    config = saved.get('config',{})
    if (saved.get('kind')!=KIND or saved.get('status')!='complete' or config.get('data')!=str(data) or
            config.get('seed')!=seed or
            config.get('shared_isolation_authority',{}).get('base_data_identities')!=canonical_base):
        raise ValueError('Student root pool differs from the actual base data, seed or completion contract')
    if {Path(p).resolve() for p in saved.get('outputs',{})}!={pool/'roots.jsonl',pool/'counts.json'}:
        raise ValueError('Student root pool must bind exactly its completed roots and counts')
    for path,binding in saved['inputs'].items():
        bound.bind(path,binding)
    for name in ARTIFACTS[1:]:
        bound.bind(pool/name,_binding(saved,'outputs',pool/name))
    current = code_identity()['source_sha256']
    for name in ['student_game_inputs.py','neural_training_games.py','model_inputs.py',*SCORING_CLOSURE]:
        key = 'src/xqgeneral/'+name
        if saved.get('code',{}).get('source_sha256',{}).get(key)!=current[key]:
            raise ValueError('Student source-pool replay implementation differs from its completed publication')
    roots,counts,common,reserved = _plan(config,bound)
    if (roots!=load_jsonl(pool/'roots.jsonl') or counts!=bound.json(pool/'counts.json') or
            counts!=saved['verification'] or common!=config['shared_isolation_authority'] or
            offset+limit>len(roots) or not roots):
        raise ValueError('Student root coverage, isolation, source ownership or requested range changed')
    bound.unchanged()
    return roots[offset:offset+limit],reserved


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['games','scores','execution-proof','authority','data','heldout-data','output',
                 'checkpoint','expert-weights','gameplay-policy','executable','weights']:
        parser.add_argument('--'+name,required=True)
    for name in ['scoring-policy','decoder-model','decoder-manifest','input-digest-receipt']:
        parser.add_argument('--'+name)
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--future-plies',type=int,default=8)
    args = vars(parser.parse_args())
    print(json.dumps(prepare(**args),ensure_ascii=False),flush=True)


def main():
    with input_session_from_environment():
        return _main()


if __name__=='__main__':
    main()
