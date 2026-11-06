"""Scoring preserves source ownership, true played moves and raw oracle evidence."""
from collections import Counter
from copy import deepcopy
import json
import random

import pytest

from xqgeneral import training_game_scoring as scoring
from xqgeneral.calibration import candidate_probability
from xqgeneral.evidence import atomic_json, code_identity, load_jsonl, manifest
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.rules import START_FEN, legal_moves, play
from xqgeneral.training_play import run_games


POLICY = {'run_id':'scoring-owned-CPU-fixture','games':2,'game_start':2,'seed':17,
    'concurrent_games':2,'batch_size':2,'max_plies':4,'opponent_nodes_min':100,'opponent_nodes_max':100,
    'generation':{'do_sample':True,'temperature':.6,'top_k':20,'top_p':.95,'max_new_tokens':4096}}


def gameplay(path,*,native=False):
    class Student:
        def generate_batch(self,records,**kwargs):
            return ['invalid raw student response' for _ in records]
    class Engine:
        def choose_move(self,fen,nodes,*args):
            move=sorted(legal_moves(fen))[-1]
            return {'best_move':move,'requested_nodes':nodes,'raw_output':['bestmove '+move],'score_consumed':False}
        def close(self):pass
    code = {'identity':code_identity(),'config':{'CPU_fixture':True}} if native else {'CPU_fixture':True}
    summary = run_games(path,POLICY,'a'*64,Student,Engine,code=code,inputs={})
    if native:
        files=[path/'contract.json',path/'games.jsonl',path/'summary.json',path/'student-batches.partial.jsonl',
               *sorted((path/'games').glob('game-*.json'))]
        proof=manifest('student_engine_training_games',code['config'],(),files,summary,code=code['identity'])
        atomic_json(path/'manifest.json',proof)
    return path


class Oracles:
    def __init__(self,*,fail_once=False,missing_played=False,cp_only=False):
        self.counts=Counter();self.requests=[]
        self.fail_once,self.missing_played,self.cp_only=fail_once,missing_played,cp_only

    def factory(self,role):
        self.counts['load:'+role]+=1
        owner=self
        class Engine:
            def search(self,fen,nodes,initial_fen,moves,searchmoves):
                owner.counts['search']+=1
                owner.requests.append((fen,nodes,initial_fen,list(moves),searchmoves,role))
                if owner.fail_once and owner.counts['search']==2:
                    self.last_output=['info string preserved incomplete CPU attempt']
                    raise RuntimeError('controlled scoring interruption')
                allowed=searchmoves or sorted(legal_moves(fen))
                selected=allowed[:1] if role=='single' or owner.missing_played else allowed
                lines=[]
                for rank,move in enumerate(selected,1):
                    score=120 if rank==1 else -120
                    wdl='' if owner.cp_only else (' wdl 800 100 100' if rank==1 else ' wdl 100 100 800')
                    lines.append(f'info depth 12 multipv {rank} score cp {score}{wdl} pv {move}')
                lines.append('bestmove '+selected[0]);self.last_output=lines
                return lines
            def close(self):owner.counts['close:'+role]+=1
        return Engine()


def score(source,out,oracles,**kwargs):
    return scoring.score_games(source,out,oracles.factory,code={'CPU_consumer':'v1'},**kwargs)


@pytest.mark.parametrize('field,value',[
    ('oracle_nodes',True),('oracle_nodes',0),('force_keep_winrate_drop',float('nan')),
    ('force_keep_winrate_drop',float('inf')),('force_keep_winrate_drop',-1),
    ('force_keep_winrate_drop',True),('high_eval_cp',0),('high_eval_downsample',0),
    ('ordinary_ply_divisor',1.5),('seed',False),('unknown',1)])
def test_invalid_scoring_policy_fails_before_output_or_loading_players(tmp_path,field,value):
    players=Oracles()
    with pytest.raises(ValueError):
        scoring.score_games(tmp_path/'missing',tmp_path/'out',players.factory,
            policy=dict(scoring.DEFAULT_POLICY,**{field:value}))
    assert not (tmp_path/'out').exists() and not players.counts


def test_actual_played_move_is_scored_with_complete_history_and_raw_provenance(tmp_path):
    source=gameplay(tmp_path/'source');out=tmp_path/'scores';players=Oracles()
    result=score(source,out,players)
    rows=load_jsonl(out/'scored-positions.jsonl')
    games=load_jsonl(source/'games.jsonl')
    expected=[(g,t) for g in games for t in g['turns']]
    assert result['raw_positions']==8 and result['games']==2
    for row,(game,turn) in zip(rows,expected):
        assert row['record']==turn['record'] and row['played_move']==turn['played_move']
        assert row['game_index']==game['index'] and row['ply']==turn['ply']
        assert row['audit']['winrate_drop'] in {0.,.7}
        assert row['audit']['played_candidate']['move']==turn['played_move']
        if turn['mover']==game['model_color']:
            assert row['move_source']=='random_legal_fallback'
            assert row['cached_root']['raw']=='invalid raw student response'
            assert row['cached_root']['seed']==turn['seed']
        else:
            assert row['cached_root'] is None and row['move_source']=='engine'
    assert all(request[1]==(200000 if request[-1]=='pair' else 100000) for request in players.requests)
    assert any(request[3] for request in players.requests)
    assert result['training_labels_generated'] is False and result['source_neural_execution_verified'] is False
    assert result['heldout_position_isolation_applied'] is False
    assert players.counts['load:single']==players.counts['close:single']==1
    assert players.counts['load:pair']==players.counts['close:pair']==1
    for item in load_jsonl(out/'oracle-queries.partial.jsonl'):
        assert item['status']=='success' and item['raw_output'][-1].startswith('bestmove ')


def record():
    return {'initial_fen':START_FEN,'moves':[],'history':[START_FEN],'fen':START_FEN}


def test_best_move_reuses_one_root_score_and_non_best_missing_exact_score_is_retried(tmp_path):
    from contextlib import ExitStack
    (tmp_path/'queries').touch();players=Oracles(missing_played=True)
    with ExitStack() as stack:
        ledger=scoring.QueryLedger(tmp_path/'queries',players.factory,stack,lambda:None)
        best=sorted(legal_moves(START_FEN))[0]
        same=scoring.audit_played_move(record(),best,ledger,100000)
        assert same['winrate_drop']==0 and players.counts['search']==1
        other=sorted(legal_moves(START_FEN))[-1]
        scored=scoring.audit_played_move(record(),other,ledger,100000)
    assert len(scored['query_ids'])==3
    assert players.counts['search']==3  # The existing full-history root is reused.
    assert players.requests[-1][4]==[other] and players.requests[-1][1]==100000
    assert scored['played_candidate']['move']==other


def test_cp_comparison_uses_pinned_material_model(tmp_path):
    from contextlib import ExitStack
    (tmp_path/'queries').touch();players=Oracles(cp_only=True)
    with ExitStack() as stack:
        ledger=scoring.QueryLedger(tmp_path/'queries',players.factory,stack,lambda:None)
        scored=scoring.audit_played_move(record(),sorted(legal_moves(START_FEN))[-1],ledger,100000)
    expected=candidate_probability(START_FEN,scored['oracle_candidate'])-candidate_probability(START_FEN,scored['played_candidate'])
    assert scored['winrate_drop']==expected>0
    assert scored['oracle_candidate']['score_type']=='cp'


def test_mate_scores_stay_typed_and_do_not_become_arbitrary_centipawns(tmp_path):
    class Queries:
        def query(self,record,role,nodes,searchmoves=None):
            first='info depth 8 score mate 3 pv b0c2'
            pair=['info depth 8 multipv 1 score mate 3 pv b0c2',
                  'info depth 8 multipv 2 score cp 0 pv h0g2']
            return role,scoring._parsed(scoring._request(record,role,nodes,searchmoves),
                ([first] if role=='root' else pair)+['bestmove b0c2'])
    result=scoring.audit_played_move(record(),'h0g2',Queries(),100000)
    assert result['oracle_candidate']['score_type']=='mate'
    assert result['oracle_candidate']['score']==3
    assert result['winrate_drop']==.5


@pytest.mark.parametrize('played',['b0c2','h0g2'])
def test_newly_chosen_bounded_root_move_is_rescored_and_never_used_as_an_exact_score(tmp_path,played):
    from contextlib import ExitStack
    calls=[]
    class Engine:
        def search(self,fen,nodes,initial_fen,moves,searchmoves):
            calls.append((nodes,searchmoves))
            if searchmoves is None:
                return ['info depth 8 score cp 20 pv b0c2',
                    'info depth 9 score cp 400 lowerbound pv h0g2','bestmove h0g2']
            return [f'info depth 8 multipv {i+1} score cp 0 pv {m}' for i,m in enumerate(searchmoves)]+['bestmove '+searchmoves[0]]
        def close(self):pass
    (tmp_path/'queries').touch()
    with ExitStack() as stack:
        ledger=scoring.QueryLedger(tmp_path/'queries',lambda role:Engine(),stack,lambda:None)
        result=scoring.audit_played_move(record(),played,ledger,100000)
    assert result['oracle_move']=='h0g2' and result['primary_oracle_candidate'] is None
    assert result['primary_best_exact_score_missing'] is True
    assert result['oracle_candidate']['score']==0 and result['played_candidate']['score']==0
    assert result['winrate_drop']==0 and len(calls)==2
    assert calls[-1]==((100000,['h0g2']) if played=='h0g2' else (200000,['h0g2','b0c2']))
    saved=load_jsonl(tmp_path/'queries')
    assert len(saved)==2 and saved[0]['status']=='success'
    assert 'lowerbound' in saved[0]['raw_output'][1]


def test_failed_actual_raw_query_is_preserved_then_resume_reuses_successful_prefix(tmp_path):
    source=gameplay(tmp_path/'source');out=tmp_path/'scores';first=Oracles(fail_once=True)
    with pytest.raises(RuntimeError,match='controlled'):
        score(source,out,first)
    prefix=(out/'oracle-queries.partial.jsonl').read_bytes()
    rows=load_jsonl(out/'oracle-queries.partial.jsonl')
    assert len(rows)==2 and rows[-1]['status']=='failed'
    assert rows[-1]['raw_output']==['info string preserved incomplete CPU attempt']
    assert not (out/'manifest.json').exists()
    resumed=Oracles();summary=score(source,out,resumed,resume=True)
    assert (out/'oracle-queries.partial.jsonl').read_bytes().startswith(prefix)
    assert summary['preserved_failed_query_attempts']==1
    assert resumed.requests[0][-1]=='pair'  # The successful root is never rerun.
    assert summary['successful_distinct_oracle_queries']+1==len(load_jsonl(out/'oracle-queries.partial.jsonl'))


def test_completed_scoring_reopen_constructs_no_engines_and_changes_no_artifacts(tmp_path):
    source=gameplay(tmp_path/'source');out=tmp_path/'scores'
    summary=score(source,out,Oracles())
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}
    players=Oracles();assert score(source,out,players,resume=True)==summary
    assert not players.counts
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}


@pytest.mark.parametrize('damage',['game_missing','game_extra','unfinished','summary','raw','unowned_raw'])
def test_changed_or_incomplete_gameplay_rejects_before_any_oracle_load(tmp_path,damage):
    source=gameplay(tmp_path/'source');players=Oracles()
    if damage=='game_missing':next((source/'games').glob('*.json')).unlink()
    elif damage=='game_extra':(source/'games/game-99999999.json').write_text('{}')
    elif damage=='unfinished':
        path=next((source/'games').glob('*.json'));a=json.loads(path.read_text());a['status']='playing';atomic_json(path,a)
    elif damage=='summary':
        path=source/'summary.json';a=json.loads(path.read_text());a['games']+=1;atomic_json(path,a)
    else:
        path=source/'student-batches.partial.jsonl';a=load_jsonl(path)
        if damage=='raw':a[0]['requests'][0]['raw']='different still-invalid raw answer'
        else:
            a.append(deepcopy(a[-1]));a[-1]['ordinal']+=1;a[-1]['requests'][0]['record']['id']='unowned-slot'
        path.write_text(''.join(json.dumps(row)+'\n' for row in a))
    with pytest.raises((ValueError,KeyError)):
        score(source,tmp_path/'scores',players)
    assert not players.counts and not (tmp_path/'scores').exists()


@pytest.mark.parametrize('damage',['source_changed','config_changed','scores_changed','final_missing','ledger_changed'])
def test_completed_scoring_changes_are_not_repaired_by_new_engine_queries(tmp_path,damage):
    source=gameplay(tmp_path/'source');out=tmp_path/'scores';score(source,out,Oracles());players=Oracles()
    policy=dict(scoring.DEFAULT_POLICY)
    if damage=='source_changed':(source/'summary.json').write_text('{}')
    elif damage=='config_changed':policy['oracle_nodes']+=1
    elif damage=='scores_changed':(out/'selected-positions.jsonl').write_text('[]\n')
    elif damage=='final_missing':(out/'scored-positions.jsonl').unlink()
    else:(out/'oracle-queries.partial.jsonl').write_text('')
    with pytest.raises((ValueError,FileNotFoundError)):
        score(source,out,players,resume=True,policy=policy)
    assert not players.counts


def test_native_manifest_checks_bytes_but_does_not_attest_neural_execution(tmp_path):
    source=gameplay(tmp_path/'source',native=True)
    games,info=scoring.load_training_games(source,BoundInputs())
    assert len(games)==2 and info['native_gameplay_manifest_present'] is True
    assert info['source_neural_execution_verified'] is False and info['source_model_assets_verified'] is False
    p=source/'manifest.json';v=json.loads(p.read_text());v['outputs'][str(source/'summary.json')]['sha256']='0'*64
    atomic_json(p,v)
    with pytest.raises(ValueError):scoring.load_training_games(source,BoundInputs())


def test_history_mismatch_and_restricted_candidate_escape_fail_with_raw_evidence(tmp_path):
    from contextlib import ExitStack
    (tmp_path/'queries').touch();players=Oracles()
    with ExitStack() as stack:
        ledger=scoring.QueryLedger(tmp_path/'queries',players.factory,stack,lambda:None)
        broken=dict(record(),moves=['b0c2'])
        with pytest.raises(ValueError,match='history'):ledger.query(broken,'single',100)
    assert not players.counts
    request=scoring._request(record(),'single',100,['b0c2'])
    with pytest.raises(ValueError,match='restriction'):
        scoring._parsed(request,['info depth 8 score cp 20 pv h0g2','bestmove h0g2'])


def selection_rows(count=45):
    rng=random.Random(37);fen=START_FEN;history=[fen];moves=[];rows=[]
    for ply in range(count):
        rows.append({'id':str(ply),'game_index':0,'ply':ply,'mover':'red' if ply%2==0 else 'black',
            'record':{'fen':fen,'initial_fen':START_FEN,'moves':list(moves),'history':list(history)},
            'audit':{'winrate_drop':0.,'oracle_candidate':{'score_type':'cp','score':0}}})
        move=rng.choice(sorted(legal_moves(fen)));fen=play(fen,move);history.append(fen);moves.append(move)
    return rows


def test_selection_forces_mistakes_and_uses_whole_game_quota_before_deduplication():
    rows=selection_rows(21)
    rows[0]['audit']['winrate_drop']=.05
    duplicate=deepcopy(rows[0]);duplicate['game_index']=1;duplicate['id']='winning-duplicate';duplicate['audit']['winrate_drop']=.1
    rows.append(duplicate)
    selected,stats=scoring.select_positions(rows,scoring.DEFAULT_POLICY)
    assert stats['raw_positions']==22 and stats['duplicates_removed']==1
    assert stats['ordinary_quota_by_game']=={'0':3,'1':1}
    assert stats['forced_positions']==1 and stats['ordinary_selected_positions']==3
    assert stats['ordinary_shortfall']==1
    assert any(row['id']=='winning-duplicate' for row in selected)
    assert all(row['id']!='0' for row in selected)
    assert (selected,stats)==scoring.select_positions(rows,scoring.DEFAULT_POLICY)


def test_high_evaluation_ordinary_positions_are_downsampled_but_mistakes_survive():
    rows=selection_rows(40)
    for row in rows:row['audit']['oracle_candidate']['score']=300
    rows[0]['audit']['winrate_drop']=.7
    rows[0]['audit']['oracle_candidate']={'score_type':'mate','score':2}
    selected,stats=scoring.select_positions(rows,scoring.DEFAULT_POLICY)
    assert stats['forced_positions']==1 and rows[0] in selected
    assert stats['ordinary_high_eval_positions']==39 and stats['ordinary_high_eval_eligible']==10
    assert stats['ordinary_selected_positions']==4 and len(selected)==5


def test_dedup_ties_keep_later_owned_game_and_never_rewrite_its_history():
    row=selection_rows(1)[0];row['audit']['winrate_drop']=.1
    later=deepcopy(row);later.update(game_index=9,ply=42,id='later-owned-occurrence')
    selected,stats=scoring.select_positions([row,later],scoring.DEFAULT_POLICY)
    assert selected==[later] and selected[0]['record']==later['record']
    assert stats['duplicates_removed']==1
