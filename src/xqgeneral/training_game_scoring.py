"""CPU strong-oracle audits of training games; selected contexts are not labels."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import random

from .calibration import candidate_probability
from .evidence import atomic_json, code_identity, load_jsonl, manifest, position_key, write_jsonl
from .oracle import parse_analysis
from .recorded_search_inputs import BoundInputs
from .rules import legal_moves, replay
from .sampling import checked_seed
from .training_play import checked_policy, fingerprint, planned_game, replay_training_game, _load_batches
from .verified_inputs import input_session_from_environment


DEFAULT_POLICY = {'oracle_nodes':100000,'force_keep_winrate_drop':0.05,
    'high_eval_cp':250,'high_eval_downsample':4,'ordinary_ply_divisor':10,'seed':20261013}
PUBLICATIONS = ('scored-positions.jsonl','selected-positions.jsonl','summary.json')


def checked_scoring_policy(policy):
    if not isinstance(policy,dict) or set(policy)!=set(DEFAULT_POLICY):
        raise ValueError('Scoring requires its complete explicit policy')
    for name in ['oracle_nodes','high_eval_cp','high_eval_downsample','ordinary_ply_divisor']:
        if type(policy[name]) is not int or policy[name]<=0:
            raise ValueError('Scoring budgets and selection divisors must be positive integers')
    threshold = policy['force_keep_winrate_drop']
    if type(threshold) not in {int,float} or not math.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('The forced-mistake threshold must be a finite probability')
    checked_seed(policy['seed'])
    return dict(policy)


def _summary(games):
    model_turns = [t for g in games for t in g['turns'] if t['mover']==g['model_color']]
    return {'games':len(games),'model_turns':len(model_turns),
        'raw_illegal_or_missing_declared_moves':sum(t['raw_declared_move_legal'] is False for t in model_turns),
        'random_legal_fallbacks':sum(t['move_source']=='random_legal_fallback' for t in model_turns),
        'completed_games':sum(g['status']=='completed' for g in games),
        'censored_games':sum(g['status']=='censored' for g in games),
        'all_games_training_owned_before_generation':True,'raw_model_outputs_preserved':True,
        'full_histories_preserved':True,'training_labels_generated':False,'evaluation_matches_generated':False,
        'heldout_position_isolation_not_yet_applied':True,'playing_strength_improvement_proven':False}


def load_training_games(source,bound):
    """Replay known local rules, without executing source code or loading a model.

    A native manifest is checked when present. This mechanical reader does not
    attest that a neural student was executed or authenticate its model assets.
    The later training-pool admission must establish that separately.
    """
    source = Path(source).resolve()
    contract = bound.json(source/'contract.json')
    if contract.get('schema_version')!=1 or contract.get('kind')!='training_only_student_engine_gameplay':
        raise ValueError('Only explicitly training-owned gameplay can be scored')
    policy = checked_policy(contract['policy'])
    actor = contract['actor_identity']
    indices = range(policy['game_start'],policy['game_start']+policy['games'])
    owners = [planned_game(policy,i,actor) for i in indices]
    paths = [source/'games'/f'game-{i:08d}.json' for i in indices]
    if set((source/'games').glob('game-*.json'))!=set(paths):
        raise ValueError('Completed gameplay must contain exactly its owned game files')
    bound.bind(source/'student-batches.partial.jsonl')
    cache,_ = _load_batches(source/'student-batches.partial.jsonl',policy,{g['game_id']:g for g in owners})
    games = [replay_training_game(bound.json(path),policy,g['index'],actor,cache)
             for path,g in zip(paths,owners)]
    if any(g['status'] not in {'completed','censored'} for g in games):
        raise ValueError('Unfinished training games cannot be published for scoring')
    consumed = {t['record']['id'] for g in games for t in g['turns'] if t['mover']==g['model_color']}
    if set(cache)!=consumed:
        raise ValueError('The completed gameplay ledger has missing or unowned student responses')
    bound.bind(source/'games.jsonl')
    if load_jsonl(source/'games.jsonl')!=games or bound.json(source/'summary.json')!=_summary(games):
        raise ValueError('Gameplay publication differs from its raw history replay')
    native = None
    if (source/'manifest.json').exists():
        native = bound.json(source/'manifest.json')
        outputs = [source/'contract.json',source/'games.jsonl',source/'summary.json',
                   source/'student-batches.partial.jsonl',*paths]
        declared = {str(Path(k).resolve()):v for k,v in native.get('outputs',{}).items()}
        if (native.get('kind')!='student_engine_training_games' or native.get('status')!='complete' or
                native.get('verification')!=_summary(games) or set(declared)!={str(p) for p in outputs} or
                contract.get('code')!={'identity':native.get('code'),'config':native.get('config')} or
                contract.get('inputs')!=native.get('inputs')):
            raise ValueError('Native gameplay manifest differs from its complete raw publication')
        # Only the replay closure needs to match this consumer, not unrelated new modules.
        current = code_identity()['source_sha256']
        producer = native.get('code',{}).get('source_sha256',{})
        for name in ['training_play.py','rules.py','explanations.py','sampling.py','evidence.py']:
            key = 'src/xqgeneral/'+name
            if producer.get(key)!=current[key]:
                raise ValueError('The native gameplay replay implementation changed')
        for path in outputs:
            bound.bind(path,declared[str(path)])
    bound.unchanged()
    return games,{'actor_identity':actor,'gameplay_policy':policy,
        'native_gameplay_manifest_present':native is not None,
        'source_neural_execution_verified':False,'source_model_assets_verified':False}


def _request(record,role,nodes,searchmoves):
    if (role not in {'root','single','pair'} or role=='root' and searchmoves is not None or
            type(nodes) is not int or nodes<=0 or
            replay(record['initial_fen'],record['moves'])!=record['history'] or
            record['history'][-1]!=record['fen']):
        raise ValueError('Oracle scoring requires its complete original legal history')
    if searchmoves is not None and (not searchmoves or len(set(searchmoves))!=len(searchmoves) or
            any(m not in legal_moves(record['fen']) for m in searchmoves)):
        raise ValueError('Oracle scoring restrictions must contain distinct legal root moves')
    return {'fen':record['fen'],'initial_fen':record['initial_fen'],'moves':list(record['moves']),
            'role':role,'nodes':nodes,'searchmoves':list(searchmoves) if searchmoves is not None else None}


def _parsed(request,raw):
    if not isinstance(raw,list) or not raw or any(not isinstance(line,str) for line in raw):
        raise ValueError('Oracle scoring must preserve its actual raw UCI response')
    try:
        result = parse_analysis(request['fen'],raw,request['nodes'])
    except RuntimeError as error:
        # A node stop may choose a newly discovered move with only a bound.
        # It is valid as the move to compare, never as an exact evaluation.
        if request['role']!='root' or str(error)!='Engine completed without a scored best-move candidate':
            raise
        result = {'best_move':raw[-1].split()[1],'candidates':[],
            'requested_nodes':request['nodes'],'score_perspective':'side_to_move','raw_output':raw,
            'best_move_score_missing':True}
    if request['searchmoves'] is not None and any(c['move'] not in request['searchmoves'] for c in result['candidates']):
        raise ValueError('An exact oracle candidate escaped its requested root restriction')
    return result


class QueryLedger:
    """Save before returning; successful queries replay without constructing an engine."""
    def __init__(self,path,make_oracle,stack,unchanged):
        self.path,self.make_oracle,self.stack,self.unchanged = Path(path),make_oracle,stack,unchanged
        self.cache,self.engines,self.used,self.requests = {},{},set(),{}
        self.ordinal = 0
        self.physical_queries = 0
        for row in load_jsonl(self.path):
            request = row['request']
            request_id = fingerprint(request)
            record = dict(request,history=replay(request['initial_fen'],request['moves']))
            if (_request(record,request['role'],request['nodes'],request['searchmoves'])!=request or
                    row.get('ordinal')!=self.ordinal or row.get('id')!=request_id or
                    request_id in self.cache or row.get('status') not in {'success','failed'}):
                raise ValueError('Preserved scoring query ownership or ordinal changed')
            if row['status']=='success':
                self.cache[request_id] = _parsed(request,row['raw_output'])
            self.requests[request_id] = request
            self.ordinal += 1

    def query(self,record,role,nodes,searchmoves=None):
        request = _request(record,role,nodes,searchmoves)
        request_id = fingerprint(request)
        self.used.add(request_id)
        if request_id not in self.cache:
            self.unchanged()
            engine_role = 'pair' if role=='pair' else 'single'
            if engine_role not in self.engines:
                self.engines[engine_role] = self.stack.enter_context(closing(self.make_oracle(engine_role)))
            engine = self.engines[engine_role]
            engine.last_output = None
            self.physical_queries += 1
            raw = None
            try:
                raw = engine.search(request['fen'],nodes,request['initial_fen'],request['moves'],searchmoves)
                result = _parsed(request,raw)
            except Exception as error:
                self._save(request_id,request,'failed',raw if raw is not None else getattr(engine,'last_output',None),type(error).__name__)
                raise
            self._save(request_id,request,'success',raw)
            self.cache[request_id] = result
            self.unchanged()
        return request_id,deepcopy(self.cache[request_id])

    def _save(self,request_id,request,status,raw,error=None):
        row = {'ordinal':self.ordinal,'id':request_id,'request':request,'status':status,'raw_output':raw}
        if error is not None:
            row['error_type'] = error
        with self.path.open('a') as handle:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            handle.flush();os.fsync(handle.fileno())
        self.ordinal += 1
        self.requests[request_id] = request


def audit_played_move(record,played_move,queries,nodes):
    """Compare in one restricted MultiPV search; retry missing exact scores alone."""
    if played_move not in legal_moves(record['fen']):
        raise ValueError('Only an actually played legal training move can be audited')
    query_id,root = queries.query(record,'root',nodes)
    ids = [query_id]
    best = root['best_move']
    primary = next((c for c in root['candidates'] if c['move']==best),None)
    if played_move==best:
        if primary is None:
            query_id,single = queries.query(record,'single',nodes,[best])
            ids.append(query_id)
            best_score = next(c for c in single['candidates'] if c['move']==best)
        else:
            best_score = primary
        played_score = best_score
    else:
        query_id,pair = queries.query(record,'pair',2*nodes,[best,played_move])
        ids.append(query_id)
        candidates = {c['move']:c for c in pair['candidates']}
        for move in [best,played_move]:
            if move not in candidates:
                query_id,single = queries.query(record,'single',nodes,[move])
                ids.append(query_id)
                candidates[move] = next(c for c in single['candidates'] if c['move']==move)
        best_score,played_score = candidates[best],candidates[played_move]
    best_probability = candidate_probability(record['fen'],best_score)
    played_probability = candidate_probability(record['fen'],played_score)
    return {'oracle_move':best,'primary_oracle_candidate':primary,'primary_best_exact_score_missing':primary is None,
        'oracle_candidate':best_score,'played_candidate':played_score,
        'oracle_expected_score':best_probability,'played_expected_score':played_probability,
        'winrate_drop':max(0.0,best_probability-played_probability),'query_ids':ids,
        'probability_model':'pinned_Pikafish_material_WDL_not_empirical_human_winrate'}


def select_positions(positions,policy):
    policy = checked_scoring_policy(policy)
    played_plies,by_position = defaultdict(set),{}
    for row in positions:
        drop = row['audit']['winrate_drop']
        if type(drop) not in {int,float} or not math.isfinite(drop) or not 0<=drop<=1:
            raise ValueError('A selection error must be a finite probability loss')
        played_plies[row['game_index']].add(row['ply'])
        key = position_key(row['record']['fen'])
        rank = lambda item:(item['audit']['winrate_drop'],item['game_index'],item['ply'],item['mover'])
        if key not in by_position or rank(row)>rank(by_position[key]):
            by_position[key] = row
    quotas = {i:math.ceil(len(plies)/policy['ordinary_ply_divisor']) for i,plies in played_plies.items()}
    unique = list(by_position.values())
    forced = [r for r in unique if r['audit']['winrate_drop']>=policy['force_keep_winrate_drop']]
    ordinary = [r for r in unique if r['audit']['winrate_drop']<policy['force_keep_winrate_drop']]
    def high(row):
        candidate = row['audit']['oracle_candidate']
        return candidate['score_type']=='mate' or abs(candidate['score'])>=policy['high_eval_cp']
    upper = [r for r in ordinary if high(r)]
    lower = [r for r in ordinary if not high(r)]
    rng = random.Random(policy['seed'])
    rng.shuffle(upper)
    eligible_upper = upper[:math.ceil(len(upper)/policy['high_eval_downsample'])]
    eligible = defaultdict(list)
    for row in lower+eligible_upper:
        eligible[row['game_index']].append(row)
    chosen,selected_per_game = [],{}
    for index in sorted(quotas):
        rng.shuffle(eligible[index])
        current = eligible[index][:quotas[index]]
        chosen.extend(current)
        selected_per_game[str(index)] = len(current)
    selected = forced+chosen
    rng.shuffle(selected)
    counts = {'raw_positions':len(positions),'unique_positions':len(unique),
        'duplicates_removed':len(positions)-len(unique),'forced_positions':len(forced),
        'ordinary_target_positions':sum(quotas.values()),'ordinary_selected_positions':len(chosen),
        'ordinary_shortfall':sum(quotas.values())-len(chosen),
        'ordinary_quota_by_game':{str(i):q for i,q in sorted(quotas.items())},
        'ordinary_selected_by_game':selected_per_game,'ordinary_high_eval_positions':len(upper),
        'ordinary_high_eval_eligible':len(eligible_upper),'selected_positions':len(selected),
        'selected_movers':dict(Counter(r['mover'] for r in selected)),
        'ordinary_policy':'ceil_played_plies_div_10_random_per_game' if policy['ordinary_ply_divisor']==10
                          else 'explicit_per_game_ply_divisor'}
    return selected,counts


def score_games(source,output,make_oracle,*,policy=None,resume=False,bound=None,code=None,on_progress=None):
    policy = checked_scoring_policy(DEFAULT_POLICY if policy is None else policy)
    bound = bound or BoundInputs()
    games,source_info = load_training_games(source,bound)
    output = Path(output)
    contract = {'schema_version':1,'kind':'CPU_strong_training_game_scoring',
        'source':str(Path(source).resolve()),'source_info':source_info,'policy':policy,
        'inputs':dict(bound.artifacts),'code':code or code_identity()}
    if output.exists():
        if not resume or not (output/'contract.json').exists() or json.loads((output/'contract.json').read_text())!=contract:
            raise ValueError('Scoring resume requires the exact original completed source, inputs and configuration')
    else:
        if resume:
            raise ValueError('Resume requires an existing scoring output')
        output.mkdir(parents=True)
        atomic_json(output/'contract.json',contract)
        (output/'oracle-queries.partial.jsonl').touch(exist_ok=False)
        (output/'positions.partial.jsonl').touch(exist_ok=False)
    saved_manifest = None
    if (output/'manifest.json').exists():
        saved_manifest = json.loads((output/'manifest.json').read_text())
        expected = [output/'contract.json',output/'oracle-queries.partial.jsonl',output/'positions.partial.jsonl',
                    *[output/name for name in PUBLICATIONS]]
        if (saved_manifest.get('kind')!='strong_scored_student_training_contexts' or
                saved_manifest.get('status')!='complete' or saved_manifest.get('config')!=contract or
                saved_manifest.get('inputs')!=bound.artifacts or saved_manifest.get('code')!=contract['code'] or
                set(saved_manifest.get('outputs',{}))!={str(p) for p in expected}):
            raise ValueError('Completed scoring publication changed its contract or file coverage')
        # Verify all final publications before any engine can be constructed.
        check = BoundInputs()
        for path in expected:
            check.bind(path,saved_manifest['outputs'][str(path)])
    completed = load_jsonl(output/'positions.partial.jsonl')
    positions = []
    with ExitStack() as stack:
        ledger = QueryLedger(output/'oracle-queries.partial.jsonl',make_oracle,stack,bound.unchanged)
        for game in games:
            for turn in game['turns']:
                student = turn['mover']==game['model_color']
                row = {'id':turn['record']['id'],'game_index':game['index'],'ply':turn['ply'],
                    'mover':turn['mover'],'model_color':game['model_color'],'opponent_nodes':game['opponent_nodes'],
                    'record':deepcopy(turn['record']),'played_move':turn['played_move'],
                    'move_source':turn['move_source'] if student else 'engine',
                    'cached_root':{'raw':turn['raw'],'seed':turn['seed'],
                        'generation_policy':source_info['gameplay_policy']['generation']} if student else None,
                    'audit':audit_played_move(turn['record'],turn['played_move'],ledger,policy['oracle_nodes'])}
                if len(positions)<len(completed):
                    if row!=completed[len(positions)]:
                        raise ValueError('Saved strong scoring row differs from its actual raw-query replay')
                else:
                    with (output/'positions.partial.jsonl').open('a') as handle:
                        handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        handle.flush();os.fsync(handle.fileno())
                positions.append(row)
                bound.unchanged()
                if on_progress is not None:
                    on_progress({'scored_positions':len(positions),'physical_queries_this_invocation':ledger.physical_queries})
        if len(completed)>len(positions) or set(ledger.requests)!=ledger.used:
            raise ValueError('Scoring contains queries or positions outside its actual owned game coverage')
    selected,counts = select_positions(positions,policy)
    summary = dict(counts,games=len(games),successful_distinct_oracle_queries=len(ledger.cache),
        preserved_failed_query_attempts=ledger.ordinal-len(ledger.cache),
        source_neural_execution_verified=False,source_model_assets_verified=False,
        heldout_position_isolation_applied=False,training_labels_generated=False,
        new_student_queries=0,new_teacher_queries=0,student_SGD_updates=0,playing_strength_improvement_proven=False)
    for name,value in zip(PUBLICATIONS,[positions,selected,summary]):
        path = output/name
        if path.exists():
            actual = load_jsonl(path) if name.endswith('jsonl') else json.loads(path.read_text())
            if actual!=value:
                raise ValueError('Published scoring differs from its exact raw-query replay')
        elif name.endswith('jsonl'):
            write_jsonl(path,value)
        else:
            atomic_json(path,value)
    bound.unchanged()
    if saved_manifest is not None:
        if saved_manifest['verification']!=summary:
            raise ValueError('Completed scoring verification changed')
        check.unchanged()
    else:
        outputs = [output/'contract.json',output/'oracle-queries.partial.jsonl',output/'positions.partial.jsonl',
                   *[output/name for name in PUBLICATIONS]]
        proof = manifest('strong_scored_student_training_contexts',contract,(),outputs,summary,code=contract['code'])
        proof['inputs'] = dict(bound.artifacts)
        bound.unchanged();atomic_json(output/'manifest.json',proof)
    return summary


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games',required=True,help='Completed training_play output, never evaluation matches')
    parser.add_argument('--output',required=True)
    parser.add_argument('--policy',help='Complete explicit scoring policy; defaults to released selection with Xiangqi WDL')
    parser.add_argument('--executable',default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights',default='vendor/pikafish/src/pikafish.nnue')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    policy = checked_scoring_policy(json.loads(Path(args.policy).read_text()) if args.policy else DEFAULT_POLICY)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/('.'+output.name+'.training-score.lock')).open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('This exact scoring output already has a live producer') from error
        bound = BoundInputs()
        for path in [args.executable,args.weights,*([args.policy] if args.policy else [])]:
            bound.bind(path)
        from .oracle import Pikafish
        summary = score_games(args.games,output,lambda role:Pikafish(args.executable,args.weights,threads=2,
            multipv=2 if role=='pair' else 1),policy=policy,resume=args.resume,bound=bound,
            on_progress=lambda event:print(json.dumps(event),flush=True))
        print(json.dumps(summary),flush=True)


def main():
    with input_session_from_environment():
        return _main()


if __name__=='__main__':
    main()
