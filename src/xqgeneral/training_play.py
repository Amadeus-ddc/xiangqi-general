"""Training-only student/engine games with explicit, durable legal fallbacks.

The raw model action is never replaced in the ledger. These games generate
candidate training contexts and are not playing-strength evaluation matches.
"""
import argparse
from contextlib import ExitStack, closing
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import re

from .evidence import atomic_json, code_identity, history_key, manifest, write_jsonl
from .explanations import EXPLANATION_QUESTION, parse_explanation
from .recorded_search_inputs import BoundInputs
from .rules import START_FEN, adjudicate, legal_moves, play
from .sampling import checked_seed, search_generation_policy
from .verified_inputs import input_session_from_environment

POLICY_KEYS = {'run_id','games','game_start','seed','concurrent_games','batch_size','max_plies',
               'opponent_nodes_min','opponent_nodes_max','generation'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def checked_policy(value):
    if not isinstance(value,dict) or set(value)!=POLICY_KEYS:
        raise ValueError('Training gameplay requires its complete explicit policy')
    positive = ['games','concurrent_games','batch_size','max_plies','opponent_nodes_min','opponent_nodes_max']
    if (any(type(value[k]) is not int or value[k]<=0 for k in positive) or
            type(value['game_start']) is not int or value['game_start']<0 or
            not value['batch_size']<=value['concurrent_games']<=value['games'] or
            value['opponent_nodes_max']<value['opponent_nodes_min'] or
            not isinstance(value['run_id'],str) or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',value['run_id'])):
        raise ValueError('Invalid training gameplay ownership or positive budgets')
    checked_seed(value['seed'])
    return dict(value,generation=search_generation_policy(value['generation']))


def turn_seed(seed,index,ply):
    # The released author gameplay uses this stable game/ply derivation.
    return int.from_bytes(hashlib.blake2b(f'{seed}|{index}|{ply}'.encode(),digest_size=8).digest(),'big')%(2**31-1)


def planned_game(policy,index,actor):
    policy = checked_policy(policy)
    if (not isinstance(actor,str) or not re.fullmatch('[0-9a-f]{64}',actor) or
            type(index) is not int or not policy['game_start']<=index<policy['game_start']+policy['games']):
        raise ValueError('A training game needs its original actor and owned index')
    rng = random.Random(turn_seed(policy['seed'],index,-1))
    nodes = round(math.exp(rng.uniform(math.log(policy['opponent_nodes_min']),math.log(policy['opponent_nodes_max']))))
    identity = fingerprint({'policy':policy,'actor':actor,'index':index})
    return _state({'game_id':'student-train-'+identity,'index':index,'split':'train',
        'source_kind':'student_engine_training_game','actor_identity':actor,
        'initial_fen':START_FEN,'model_color':'red' if index%2==0 else 'black',
        'opponent_nodes':nodes,'moves':[],'history':[START_FEN],'turns':[]},policy)


def choose_training_move(fen,raw,seed):
    """Select only root fields; a deterministic fallback is labeled, never hidden."""
    legal = legal_moves(fen)
    if not legal:
        raise ValueError('Cannot select a training move from a terminal board')
    try:
        value = parse_explanation(raw)
        parseable = True
    except (ValueError,TypeError,AttributeError):
        parseable = False
        prefix = re.sub(r'^```(?:json)?\s*','',raw.strip()) if isinstance(raw,str) else ''
        first = re.match(r'^\{\s*"move"\s*:\s*"([a-i][0-9][a-i][0-9])"(?=\s*(?:,|}|$))',prefix)
        value = {'move':first[1]} if first else {}
    move = value.get('move')
    declared = move if isinstance(move,str) else None
    pv = value.get('pv')
    options = [(move,'best_move' if parseable else 'best_move_prefix')]
    if isinstance(pv,list) and pv:
        options.append((pv[0],'first_pv_move'))
    candidates = value.get('candidates')
    if isinstance(candidates,list):
        options.extend((candidate,'candidate_move') for candidate in candidates)
    for proposed,source in options:
        if isinstance(proposed,str) and proposed in legal:
            return {'played_move':proposed,'move_source':source,'raw_declared_move':declared,
                    'raw_declared_move_legal':isinstance(move,str) and move in legal,
                    'full_raw_output_parseable':parseable}
    return {'played_move':random.Random(seed).choice(sorted(legal)),
            'move_source':'random_legal_fallback','raw_declared_move':declared,
            'raw_declared_move_legal':False,'full_raw_output_parseable':parseable}


def _state(game,policy):
    outcome = adjudicate(game['initial_fen'],game['moves'])
    if outcome['ended']:
        return dict(game,status='completed',winner=outcome['winner'],reason=outcome['reason'])
    if len(game['moves'])>=policy['max_plies']:
        return dict(game,status='censored',winner=None,reason='ply_limit')
    return dict(game,status='playing',winner=None,reason=None)


def _record(game):
    return {'id':game['game_id']+'-ply-'+str(len(game['moves'])),'game_id':game['game_id'],
        'split':'train','source_kind':'student_engine_training_game',
        'initial_fen':game['initial_fen'],'moves':list(game['moves']),'history':list(game['history']),
        'fen':game['history'][-1],'feature_key':history_key(game['history']),
        'question':EXPLANATION_QUESTION}


def _model_turn(game,raw,policy):
    seed = turn_seed(policy['seed'],game['index'],len(game['moves']))
    return {'ply':len(game['moves']),'mover':game['model_color'],'record':_record(game),
        'raw':raw,'seed':str(seed),'model_oracle_used':False,
        **choose_training_move(game['history'][-1],raw,seed)}


def _advance(game,turn,policy):
    updated = deepcopy(game)
    updated['turns'].append(deepcopy(turn))
    updated['moves'].append(turn['played_move'])
    updated['history'].append(play(game['history'][-1],turn['played_move']))
    return _state(updated,policy)


def _load_batches(path,policy,owners):
    cache = {}
    ordinal = -1
    if not path.exists():
        return cache,0
    for ordinal,line in enumerate(path.read_text().splitlines()):
        try:
            batch = json.loads(line)
            requests = batch['requests']
            if (batch['ordinal']!=ordinal or batch['generation_policy']!=policy['generation'] or
                    not isinstance(requests,list) or not 0<len(requests)<=policy['batch_size']):
                raise ValueError('Changed training decode batch contract')
            for item in requests:
                record,raw = item['record'],item['raw']
                owner = owners[record['game_id']]
                if (record['split']!='train' or not isinstance(raw,str) or record['id'] in cache or
                        item['seed']!=str(turn_seed(policy['seed'],owner['index'],len(record['moves'])))):
                    raise ValueError('Changed training generation slot or raw response')
                cache[record['id']] = item
        except (KeyError,TypeError,json.JSONDecodeError) as error:
            raise ValueError('Malformed preserved training decode ledger') from error
    return cache,ordinal+1


def replay_training_game(saved,policy,index,actor,cache):
    game = planned_game(policy,index,actor)
    try:
        for turn in saved['turns']:
            if game['status']!='playing':
                raise ValueError('Training turns continue after a terminal or censored game')
            fen = game['history'][-1]
            mover = 'red' if fen.split()[1]=='w' else 'black'
            if mover==game['model_color']:
                item = cache[_record(game)['id']]
                if item['record']!=_record(game) or turn!=_model_turn(game,item['raw'],policy):
                    raise ValueError('Training model turn differs from its preserved raw output')
            else:
                oracle = turn['oracle']
                expected = {'ply':len(game['moves']),'mover':mover,'record':_record(game),
                    'played_move':oracle['best_move'],'oracle':oracle}
                if (turn!=expected or oracle.get('requested_nodes')!=game['opponent_nodes'] or
                        turn['played_move'] not in legal_moves(fen)):
                    raise ValueError('Training opponent turn differs from its owned history or node budget')
            game = _advance(game,turn,policy)
        if game!=saved:
            raise ValueError('Saved training game ownership, history, result or raw turns changed')
    except (KeyError,TypeError,IndexError) as error:
        raise ValueError('Malformed saved training game') from error
    return game


def run_games(output,policy,actor,make_predictor,make_oracle,*,resume=False,code=None,inputs=None,
              unchanged=lambda:None,on_progress=None):
    policy = checked_policy(policy)
    owners = [planned_game(policy,i,actor) for i in range(policy['game_start'],policy['game_start']+policy['games'])]
    output = Path(output)
    contract = {'schema_version':1,'kind':'training_only_student_engine_gameplay',
        'policy':policy,'actor_identity':actor,'code':code,'inputs':inputs}
    if output.exists():
        if not resume or not (output/'contract.json').is_file():
            raise ValueError('Preserve existing gameplay; resume requires the original contract')
        if json.loads((output/'contract.json').read_text())!=contract:
            raise ValueError('Training gameplay source, actor, inputs or policy changed')
    else:
        if resume:
            raise ValueError('Resume requires an existing training gameplay directory')
        output.mkdir(parents=True)
        (output/'games').mkdir()
        atomic_json(output/'contract.json',contract)
        (output/'student-batches.partial.jsonl').touch(exist_ok=False)
    owned_names = {f"game-{g['index']:08d}.json" for g in owners}
    actual_names = {p.name for p in (output/'games').glob('game-*.json')}
    published = any((output/name).exists() for name in ['games.jsonl','summary.json','manifest.json'])
    if actual_names-owned_names or published and actual_names!=owned_names:
        raise ValueError('Training game file coverage differs from the originally owned games')
    cache,ordinal = _load_batches(output/'student-batches.partial.jsonl',policy,{g['game_id']:g for g in owners})
    games = {}
    for owner in owners:
        path = output/'games'/f"game-{owner['index']:08d}.json"
        games[owner['index']] = (replay_training_game(json.loads(path.read_text()),policy,owner['index'],actor,cache)
            if path.exists() else owner)
    if published and any(game['status']=='playing' for game in games.values()):
        raise ValueError('Completed training game coverage contains an unfinished owned game')
    # Every uncommitted raw response must be for exactly its current owned root.
    consumed = {turn['record']['id'] for game in games.values() for turn in game['turns']
                if turn['mover']==game['model_color']}
    current = {_record(g)['id']:_record(g) for g in games.values() if g['status']=='playing' and
               ('red' if g['history'][-1].split()[1]=='w' else 'black')==g['model_color']}
    for key,item in cache.items():
        if key not in consumed and current.get(key)!=item['record']:
            raise ValueError('Uncommitted training response differs from its current game history')
    def event(phase,**values):
        unchanged()
        if on_progress is not None:
            on_progress({'phase':phase,**values})
    def save_game(game):
        games[game['index']] = game
        atomic_json(output/'games'/f"game-{game['index']:08d}.json",game)
        event('turn_saved',index=game['index'],plies=len(game['moves']))
    predictor,oracle = None,None
    unchanged()
    with ExitStack() as stack:
        while any(game['status']=='playing' for game in games.values()):
            active = [g for g in games.values() if g['status']=='playing'][:policy['concurrent_games']]
            for game in active:
                mover = 'red' if game['history'][-1].split()[1]=='w' else 'black'
                if mover==game['model_color']:
                    continue
                if oracle is None:
                    unchanged()
                    oracle = stack.enter_context(closing(make_oracle()))
                unchanged()
                result = oracle.choose_move(game['history'][-1],game['opponent_nodes'],game['initial_fen'],game['moves'])
                move = result.get('best_move')
                if result.get('requested_nodes')!=game['opponent_nodes'] or move not in legal_moves(game['history'][-1]):
                    raise ValueError('Training opponent returned an illegal move or changed its budget')
                turn = {'ply':len(game['moves']),'mover':mover,'record':_record(game),'played_move':move,'oracle':result}
                save_game(_advance(game,turn,policy))
            active = [games[g['index']] for g in active if games[g['index']]['status']=='playing']
            for start in range(0,len(active),policy['batch_size']):
                chunk = active[start:start+policy['batch_size']]
                records = [_record(g) for g in chunk]
                missing = [(g,row) for g,row in zip(chunk,records) if row['id'] not in cache]
                if missing:
                    if predictor is None:
                        unchanged()
                        predictor = make_predictor()
                    rows = [row for game,row in missing]
                    seeds = [turn_seed(policy['seed'],game['index'],len(game['moves'])) for game,row in missing]
                    generation = policy['generation']
                    unchanged()
                    answers = predictor.generate_batch(rows,max_new_tokens=generation['max_new_tokens'],
                        sampling={k:v for k,v in generation.items() if k!='max_new_tokens'},seeds=seeds)
                    if not isinstance(answers,list) or len(answers)!=len(rows) or any(not isinstance(a,str) for a in answers):
                        raise ValueError('Training student batch coverage differs from requested game turns')
                    requests = [{'record':row,'raw':raw,'seed':str(seed)} for row,raw,seed in zip(rows,answers,seeds)]
                    with (output/'student-batches.partial.jsonl').open('a') as stream:
                        stream.write(json.dumps({'ordinal':ordinal,'generation_policy':generation,'requests':requests},ensure_ascii=False)+'\n')
                        stream.flush()
                    ordinal += 1
                    for item in requests:
                        cache[item['record']['id']] = item
                    event('student_batch_saved',requests=len(requests),ordinal=ordinal-1)
                for game,row in zip(chunk,records):
                    item = cache[row['id']]
                    if item['record']!=row:
                        raise ValueError('Preserved training generation differs from its complete current history')
                    save_game(_advance(game,_model_turn(game,item['raw'],policy),policy))
    ordered = [games[owner['index']] for owner in owners]
    summary = {'games':len(ordered),'model_turns':sum(t['mover']==g['model_color'] for g in ordered for t in g['turns']),
        'raw_illegal_or_missing_declared_moves':sum(t.get('raw_declared_move_legal') is False
            for g in ordered for t in g['turns'] if t['mover']==g['model_color']),
        'random_legal_fallbacks':sum(t.get('move_source')=='random_legal_fallback' for g in ordered for t in g['turns']),
        'completed_games':sum(g['status']=='completed' for g in ordered),
        'censored_games':sum(g['status']=='censored' for g in ordered),
        'all_games_training_owned_before_generation':True,'raw_model_outputs_preserved':True,
        'full_histories_preserved':True,'training_labels_generated':False,'evaluation_matches_generated':False,
        'heldout_position_isolation_not_yet_applied':True,'playing_strength_improvement_proven':False}
    for name,value in [('games.jsonl',ordered),('summary.json',summary)]:
        path = output/name
        if path.exists():
            actual = ([json.loads(line) for line in path.read_text().splitlines()] if name.endswith('jsonl') else json.loads(path.read_text()))
            if actual!=value:
                raise ValueError('Completed training gameplay publication differs from its raw replay')
        elif name.endswith('jsonl'):
            write_jsonl(path,value)
        else:
            atomic_json(path,value)
    unchanged()
    return summary


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--policy',required=True,help='Complete training-only gameplay JSON policy')
    parser.add_argument('--output',required=True)
    parser.add_argument('--expert-weights')
    parser.add_argument('--executable',default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights',default='vendor/pikafish/src/pikafish.nnue')
    parser.add_argument('--student-autocast-cache',choices=['on','off'],default='on')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    policy = checked_policy(json.loads(Path(args.policy).read_text()))
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/('.'+output.name+'.training-play.lock')).open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('This exact training game output already has a live producer') from error
        return _execute(args,policy)


def _execute(args,policy):
    from .model_bundle import checkpoint_expert_path, checkpoint_input_paths
    from .oracle import Pikafish
    core = BoundInputs()
    args.expert_weights = args.expert_weights or str(checkpoint_expert_path(args.checkpoint))
    checkpoint_paths = checkpoint_input_paths(args.checkpoint)
    for path in [args.policy,*checkpoint_paths,args.expert_weights,args.executable,args.weights]:
        core.bind(path)
    actor = fingerprint({str(p):core.artifacts[str(p)] for p in [*checkpoint_paths,args.expert_weights]})
    code = code_identity()
    def predictor():
        from .inference import Predictor
        return Predictor(args.checkpoint,weights=args.expert_weights,autocast_cache_enabled=args.student_autocast_cache=='on')
    config = {k:v for k,v in vars(args).items() if k!='resume'}
    output = Path(args.output)
    summary = run_games(output,policy,actor,predictor,lambda:Pikafish(args.executable,args.weights,threads=2),
        resume=args.resume,code={'identity':code,'config':config},inputs=core.artifacts,unchanged=core.unchanged,
        on_progress=lambda event:print(json.dumps(event),flush=True))
    outputs = [output/'contract.json',output/'games.jsonl',output/'summary.json',
        output/'student-batches.partial.jsonl',*sorted((output/'games').glob('game-*.json'))]
    if (output/'manifest.json').exists():
        original_inputs = dict(core.artifacts)
        saved = core.json(output/'manifest.json')
        if (saved['kind']!='student_engine_training_games' or saved['status']!='complete' or
                saved['code']!=code or saved['config']!=config or saved['verification']!=summary or
                saved['inputs']!=original_inputs or set(saved['outputs'])!={str(p) for p in outputs}):
            raise ValueError('Completed training game manifest differs from its actual replay')
        for path in outputs:
            core.bind(path,saved['outputs'][str(path)])
    else:
        saved = manifest('student_engine_training_games',config,(),outputs,summary,code=code)
        saved['inputs'] = dict(core.artifacts)
        core.unchanged();atomic_json(output/'manifest.json',saved)
    print(json.dumps(summary),flush=True)


def main():
    with input_session_from_environment():
        return _main()


if __name__ == '__main__':
    main()
