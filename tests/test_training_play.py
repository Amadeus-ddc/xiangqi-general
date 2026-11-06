"""Training games preserve raw actions and ownership; CPU players do not prove chess strength."""
from collections import Counter
import json

import pytest

from xqgeneral import training_play
from xqgeneral.rules import START_FEN, legal_moves, replay


ACTOR = 'a'*64
POLICY = {'run_id':'controlled-training-games-v1','games':4,'game_start':0,'seed':20261013,
    'concurrent_games':4,'batch_size':2,'max_plies':4,'opponent_nodes_min':100,
    'opponent_nodes_max':100000,'generation':{'do_sample':True,'temperature':.6,
        'top_k':20,'top_p':.95,'max_new_tokens':4096}}


class Players:
    def __init__(self, raw='invalid generated answer'):
        self.counts = Counter()
        self.raw = raw

    def predictor(self):
        self.counts['student_loads'] += 1
        owner = self
        class Student:
            def generate_batch(self, rows, max_new_tokens, *, sampling, seeds):
                owner.counts['student_calls'] += 1
                owner.counts['student_rows'] += len(rows)
                assert max_new_tokens == 4096 and sampling == {k:v for k,v in POLICY['generation'].items() if k!='max_new_tokens'}
                assert len(rows)==len(seeds) and len(set(seeds))==len(seeds)
                return [owner.raw for row in rows]
        return Student()

    def oracle(self):
        self.counts['engine_loads'] += 1
        owner = self
        class Engine:
            def choose_move(self, fen, nodes, initial_fen, moves):
                assert replay(initial_fen,moves)[-1]==fen
                owner.counts['engine_calls'] += 1
                move=legal_moves(fen)[0]
                return {'best_move':move,'requested_nodes':nodes,
                    'raw_output':['bestmove '+move],'score_consumed':False}
            def close(self):
                owner.counts['engine_closed'] += 1
        return Engine()


def run(path, players, *, resume=False, on_progress=None, policy=POLICY):
    return training_play.run_games(path,policy,ACTOR,players.predictor,players.oracle,
        resume=resume,on_progress=on_progress,code={'revision':'controlled_CPU'},inputs={})


@pytest.mark.parametrize('field,value',[
    ('games',True),('games',0),('game_start',-1),('max_plies',0),('batch_size',0),
    ('concurrent_games',0),('opponent_nodes_min',0),('opponent_nodes_max',99),('run_id','../other'),
    ('seed',False),('generation',{}),('unexpected',1)])
def test_invalid_policy_fails_before_output_or_player_loading(tmp_path,field,value):
    policy=dict(POLICY,**{field:value});players=Players();out=tmp_path/'games'
    with pytest.raises(ValueError):run(out,players,policy=policy)
    assert not out.exists() and not players.counts


def test_training_game_ids_and_balanced_colors_exist_before_generating_any_question():
    rows=[training_play.planned_game(POLICY,i,ACTOR) for i in range(4)]
    assert len({g['game_id'] for g in rows})==4
    assert [g['model_color'] for g in rows]==['red','black','red','black']
    assert all(g['split']=='train' and g['source_kind']=='student_engine_training_game' for g in rows)
    assert all(100<=g['opponent_nodes']<=100000 and g['moves']==[] and g['turns']==[] for g in rows)
    assert rows==[training_play.planned_game(POLICY,i,ACTOR) for i in range(4)]
    assert rows[0]['game_id']!=training_play.planned_game(POLICY,0,'b'*64)['game_id']


@pytest.mark.parametrize('value,source',[
    ({'move':'b0c2'},'best_move'),
    ({'move':'a0a9','pv':['h0g2','b9c7']},'first_pv_move'),
    ({'move':'a0a9','pv':['a0a9','h0g2'],'candidates':['b0c2']},'candidate_move'),
    ({'move':'a0a9','pv':['a0a9','h0g2']},'random_legal_fallback')])
def test_only_legal_root_fields_are_played_and_later_PV_moves_are_not_salvaged(value,source):
    raw=json.dumps(value)
    result=training_play.choose_training_move(START_FEN,raw,17)
    assert result['move_source']==source and result['played_move'] in legal_moves(START_FEN)
    assert result['raw_declared_move']==value['move']
    assert training_play.choose_training_move(START_FEN,raw,17)==result


def test_training_rollout_preserves_raw_fallbacks_and_never_scores_a_ply_cap_as_a_draw(tmp_path):
    players=Players();out=tmp_path/'games';result=run(out,players)
    assert result['games']==4 and result['model_turns']==8 and result['random_legal_fallbacks']==8
    assert result['censored_games']==4 and result['raw_model_outputs_preserved'] is True
    games=[json.loads(line) for line in (out/'games.jsonl').read_text().splitlines()]
    for game in games:
        assert game['split']=='train' and game['status']=='censored' and game['winner'] is None
        assert game['reason']=='ply_limit' and len(game['moves'])==4
        assert replay(game['initial_fen'],game['moves'])[-1]==game['history'][-1]
        for turn in game['turns']:
            if turn['mover']==game['model_color']:
                assert turn['raw']=='invalid generated answer'
                assert turn['move_source']=='random_legal_fallback'
                assert turn['model_oracle_used'] is False
    assert players.counts['engine_closed']==1
    assert result['training_labels_generated'] is False and result['evaluation_matches_generated'] is False


@pytest.mark.parametrize('phase',['student_batch_saved','turn_saved'])
def test_resume_reuses_every_saved_raw_generation_and_finished_game(tmp_path,phase):
    original=Players();expected=run(tmp_path/'whole',original)
    interrupted=Players();stopped=[]
    def stop(event):
        if event['phase']==phase and not stopped:
            stopped.append(event);raise InterruptedError('controlled interruption')
    out=tmp_path/'resumed'
    with pytest.raises(InterruptedError):run(out,interrupted,on_progress=stop)
    resumed=Players();actual=run(out,resumed,resume=True)
    assert (out/'games.jsonl').read_bytes()==(tmp_path/'whole/games.jsonl').read_bytes()
    assert actual==expected
    assert interrupted.counts['student_rows']+resumed.counts['student_rows']==original.counts['student_rows']
    before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    finished=Players();assert run(out,finished,resume=True)==expected
    assert not finished.counts and before=={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change',['policy','raw','played_move','ownership'])
def test_changed_recipe_or_saved_turn_is_rejected_before_loading_players(tmp_path,change):
    out=tmp_path/'games';run(out,Players());policy=dict(POLICY)
    if change=='policy':policy['seed']+=1
    else:
        f=sorted((out/'games').glob('game-*.json'))[0];game=json.loads(f.read_text())
        if change=='raw':game['turns'][0]['raw']='different raw generation'
        elif change=='played_move':game['turns'][0]['played_move']='a0a9'
        else:game['split']='validation'
        f.write_text(json.dumps(game))
    players=Players()
    with pytest.raises(ValueError):run(out,players,resume=True,policy=policy)
    assert not players.counts


def test_nonfinite_raw_field_is_preserved_without_poisoning_saved_game_JSON(tmp_path):
    players=Players('{"move":NaN}')
    out=tmp_path/'games'
    assert run(out,players)['raw_illegal_or_missing_declared_moves']==8
    games=[json.loads(line) for line in (out/'games.jsonl').read_text().splitlines()]
    for game in games:
        for turn in game['turns']:
            if turn['mover']==game['model_color']:
                assert turn['raw']=='{"move":NaN}' and turn['raw_declared_move'] is None
    finished=Players();run(out,finished,resume=True);assert not finished.counts


def test_empty_decode_ledger_can_resume_before_any_student_call(tmp_path):
    out=tmp_path/'games';players=Players()
    def stop(event):
        raise InterruptedError('before model')
    with pytest.raises(InterruptedError):run(out,players,on_progress=stop)
    (out/'student-batches.partial.jsonl').touch()
    assert run(out,Players(),resume=True)['games']==4


def setup_cli(tmp_path,monkeypatch):
    import sys
    from xqgeneral import inference,model_bundle,oracle
    paths={name:tmp_path/name for name in ['checkpoint.pt','expert.pb','engine','engine.nnue','policy.json']}
    for name,path in paths.items():path.write_text(json.dumps(POLICY) if name=='policy.json' else name)
    players=Players()
    monkeypatch.setattr(model_bundle,'checkpoint_input_paths',lambda _: [paths['checkpoint.pt']])
    monkeypatch.setattr(model_bundle,'checkpoint_expert_path',lambda _: paths['expert.pb'])
    monkeypatch.setattr(inference,'Predictor',lambda *a,**k:players.predictor())
    monkeypatch.setattr(oracle,'Pikafish',lambda *a,**k:players.oracle())
    argv=['training_play','--checkpoint',str(paths['checkpoint.pt']),'--policy',str(paths['policy.json']),
        '--executable',str(paths['engine']),'--weights',str(paths['engine.nnue']),'--output',str(tmp_path/'output')]
    monkeypatch.setattr(sys,'argv',argv)
    return players,argv,paths


def test_native_cli_emits_training_only_manifest_then_reopens_without_loading_players(tmp_path,monkeypatch):
    import sys
    players,argv,paths=setup_cli(tmp_path,monkeypatch)
    training_play.main()
    out=tmp_path/'output';proof=json.loads((out/'manifest.json').read_text())
    assert proof['kind']=='student_engine_training_games' and proof['verification']['training_labels_generated'] is False
    assert proof['verification']['heldout_position_isolation_not_yet_applied'] is True
    counts=dict(players.counts);before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    monkeypatch.setattr(sys,'argv',argv+['--resume']);training_play.main()
    assert counts==dict(players.counts) and before=={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}


def test_completed_manifest_cannot_authorize_reading_an_unrelated_file(tmp_path,monkeypatch):
    import sys
    from xqgeneral.recorded_search_inputs import BoundInputs
    players,argv,paths=setup_cli(tmp_path,monkeypatch);training_play.main()
    unrelated=tmp_path/'unrelated';unrelated.write_text('private canary')
    f=tmp_path/'output/manifest.json';proof=json.loads(f.read_text());proof['outputs'][str(unrelated)]={'sha256':'0'*64,'bytes':14};f.write_text(json.dumps(proof))
    original=BoundInputs.bind
    def guarded(self,path,*a,**k):
        assert str(path)!=str(unrelated),'A manifest is evidence, not authority to read unrelated files'
        return original(self,path,*a,**k)
    monkeypatch.setattr(BoundInputs,'bind',guarded);monkeypatch.setattr(sys,'argv',argv+['--resume'])
    with pytest.raises(ValueError,match='manifest'):training_play.main()


def test_second_native_producer_is_rejected_before_input_or_player_loading(tmp_path,monkeypatch):
    import fcntl
    players,argv,paths=setup_cli(tmp_path,monkeypatch)
    with (tmp_path/'.output.training-play.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='live producer'):training_play.main()
    assert not players.counts


def test_eight_training_games_use_four_request_batches_without_losing_owned_turns(tmp_path):
    policy=dict(POLICY,games=8,concurrent_games=8,batch_size=4,max_plies=6)
    players=Players();out=tmp_path/'games';summary=run(out,players,policy=policy)
    assert summary['games']==8 and summary['model_turns']==24
    batches=[json.loads(line) for line in (out/'student-batches.partial.jsonl').read_text().splitlines()]
    assert all(0<len(b['requests'])<=4 for b in batches) and any(len(b['requests'])==4 for b in batches)
    ids=[r['record']['id'] for b in batches for r in b['requests']]
    assert len(ids)==len(set(ids))==24


def test_bad_student_batch_preserves_played_prefix_and_resume_regenerates_only_unrecorded_slots(tmp_path):
    players=Players();out=tmp_path/'games'
    class Bad:
        def generate_batch(self,*a,**k):return ['one response for two requests']
    with pytest.raises(ValueError,match='coverage'):
        training_play.run_games(out,POLICY,ACTOR,Bad,players.oracle,code={'revision':'controlled_CPU'},inputs={})
    prefix={f.name:json.loads(f.read_text())['turns'] for f in (out/'games').glob('game-*.json')}
    assert not (out/'games.jsonl').exists() and (out/'student-batches.partial.jsonl').read_bytes()==b''
    assert run(out,Players(),resume=True)['games']==4
    for name,turns in prefix.items():assert json.loads((out/'games'/name).read_text())['turns'][:len(turns)]==turns


def test_unfinished_JSON_can_preserve_only_its_explicit_legal_root_move_prefix():
    choice=training_play.choose_training_move(START_FEN,'{"move":"b0c2","pv":[',17)
    assert choice['played_move']=='b0c2' and choice['move_source']=='best_move_prefix'
    assert choice['full_raw_output_parseable'] is False
    choice=training_play.choose_training_move(START_FEN,'{"pv":["b0c2"],"branches":[{"move":"h0g2"',17)
    assert choice['move_source']=='random_legal_fallback'


@pytest.mark.parametrize('problem',['missing','extra'])
def test_completed_game_file_coverage_must_fail_before_reconstructing_any_player(tmp_path,problem):
    policy=dict(POLICY,games=1,game_start=1,concurrent_games=1,batch_size=1,max_plies=1)
    out=tmp_path/'games';run(out,Players(),policy=policy)
    f=next((out/'games').glob('game-*.json'))
    if problem=='missing':f.unlink()
    else:(out/'games/game-99999999.json').write_bytes(f.read_bytes())
    players=Players()
    with pytest.raises(ValueError,match='coverage|owned'):
        run(out,players,policy=policy,resume=True)
    assert not players.counts


def test_zero_student_turn_native_game_has_an_empty_preserved_decode_ledger(tmp_path,monkeypatch):
    import sys
    players,argv,paths=setup_cli(tmp_path,monkeypatch)
    policy=dict(POLICY,games=1,game_start=1,concurrent_games=1,batch_size=1,max_plies=1)
    paths['policy.json'].write_text(json.dumps(policy));training_play.main()
    out=tmp_path/'output'
    assert (out/'student-batches.partial.jsonl').read_bytes()==b'' and players.counts['student_loads']==0
    before=dict(players.counts);monkeypatch.setattr(sys,'argv',argv+['--resume']);training_play.main()
    assert dict(players.counts)==before
