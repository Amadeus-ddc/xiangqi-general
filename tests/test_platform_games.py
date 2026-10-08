from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from xqgeneral.evidence import atomic_json, digest, manifest
from xqgeneral.platform_games import import_exports, native_export

CUTOFF = int(datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp() * 1000)
CREATED = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def record(moves='h3e3 h10g8', p1_bot=False, p2_bot=True):
    players = {}
    headers = {'Variant':'Xiangqi','Date':'2026.01.01','UTCDate':'2026.01.01',
               'P1':'Alpha','P2':'Beta','Result':'1-0'}
    for slot, bot in [('p1',p1_bot),('p2',p2_bot)]:
        user = {'id':headers[slot.upper()].lower(),'name':headers[slot.upper()]}
        if bot: user['title'] = 'BOT'; headers[slot.upper()+'Title'] = 'BOT'
        players[slot] = {'user':user}
    return {'id':'AbCd1234','variant':'xiangqi','status':'resign','createdAt':CREATED,
        'lastMoveAt':CREATED+1000,'winner':'p1','moves':moves,'players':players,
        'pgn':'\n'.join(f'[{k} {json.dumps(v)}]' for k,v in headers.items())+'\n\n'+moves+' 1-0'}


def acquisition(tmp_path, records, name='source'):
    path = tmp_path/(name+'.jsonl')
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))
    capture = {'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size,
        'records':len(records),'captured_utc':'2026-10-07T00:00:00+00:00',
        'url':'https://playstrategy.org/api/games/user/alpha?pgnInJson=true&moves=true'}
    target = tmp_path/(name+'-manifest.json')
    atomic_json(target,manifest('public_platform_recorded_game_acquisition',{},[path],[path],{'captures':[capture]}))
    return target


def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.mark.parametrize('red_bot,black_bot,kind',[
    (False,False,'recorded_human_match'),(False,True,'recorded_human_computer_match'),
    (True,True,'recorded_computer_match')])
def test_native_export_preserves_digit_headers_rank_ten_and_source_kind(red_bot,black_bot,kind):
    value = record(p1_bot=red_bot,p2_bot=black_bot)
    parsed = native_export(value,CUTOFF,minimum_plies=2)
    assert parsed['moves'] == ['h2e2','h9g7']
    assert len(parsed['history']) == 3 and parsed['source_kind'] == kind
    assert parsed['original_platform_headers']['P1'] == 'Alpha'
    assert parsed['headers']['Red'] == 'Alpha'
    assert parsed['real_world_player_identity_or_unassisted_play_authenticated'] is False
    assert parsed['recorded_move_optimality_proven'] is False
    with pytest.raises(ValueError,match='minimum-ply'): native_export(value,CUTOFF)


@pytest.mark.parametrize('change',[
    lambda r:r.update(status=[]),
    lambda r:r.update(winner='unexpected'),
    lambda r:r.update(lastMoveAt=CUTOFF+1),
    lambda r:r.update(variant='minixiangqi'),
    lambda r:r.update(moves='h3c3 h10g8'),
    lambda r:r.update(pgn=r['pgn'].replace('2026.01.01','2029.01.01')),
    lambda r:r.update(pgn=r['pgn'].replace('P2Title "BOT"','P2Title "GM"')),
    lambda r:r.update(pgn=r['pgn']+'\n[P1 "Alpha"]'),
    lambda r:r['players']['p1']['user'].update(id=['unexpected']),
])
def test_conflicting_or_malformed_original_export_is_rejected(change):
    value = record();change(value)
    with pytest.raises(ValueError): native_export(value,CUTOFF,minimum_plies=2)


def test_native_full_history_terminal_cannot_be_bypassed_by_platform_result():
    repeated = ('b1c3 b10c8 c3b1 c8b10 '*3)+'h1g3'
    with pytest.raises(ValueError,match='full-history terminal'):
        native_export(record(repeated),CUTOFF,minimum_plies=2)


def test_serial_parallel_imports_match_and_quarantine_bad_metadata(tmp_path):
    first = record();second = record('h3c3 h10g8');bad = deepcopy(first);bad['status'] = []
    captured = acquisition(tmp_path,[first,second,first,bad])
    a = import_exports(captured,tmp_path/'serial',workers=1,minimum_plies=2)
    b = import_exports(captured,tmp_path/'parallel',workers=2,minimum_plies=2)
    assert a == b
    assert a['new_unique_recorded_games'] == 2 and a['duplicate_records'] == 1 and a['quarantined_records'] == 1
    for name in ['games','quarantine','attribution-differences']:
        assert (tmp_path/'serial'/(name+'.jsonl')).read_bytes() == (tmp_path/'parallel'/(name+'.jsonl')).read_bytes()
    x,y = rows(tmp_path/'serial/duplicates.jsonl'),rows(tmp_path/'parallel/duplicates.jsonl')
    x[0]['first'].pop('existing_collection');y[0]['first'].pop('existing_collection');assert x == y


def test_previous_native_collection_dedup_preserves_split_and_attribution(tmp_path):
    original = record();capture = acquisition(tmp_path,[original]);prior = tmp_path/'prior'
    import_exports(capture,prior,workers=1,minimum_plies=2)
    changed = record();changed['players']['p1']['user'] = {'id':'gamma','name':'Gamma'}
    changed['pgn'] = changed['pgn'].replace('P1 "Alpha"','P1 "Gamma"')
    next_capture = acquisition(tmp_path,[changed],'next');output = tmp_path/'next-import'
    summary = import_exports(next_capture,output,[prior],workers=1,minimum_plies=2)
    assert summary['prior_unique_recorded_games'] == summary['combined_unique_recorded_games'] == 1
    assert summary['new_unique_recorded_games'] == 0 and summary['duplicate_records_with_differing_attribution'] == 1
    assert rows(output/'duplicates.jsonl')[0]['first']['split'] == rows(prior/'games.jsonl')[0]['split']
    assert rows(output/'attribution-differences.jsonl')[0]['attributions_authenticated'] is False


def test_changed_capture_or_disallowed_source_contract_fails_before_output_creation(tmp_path):
    capture = acquisition(tmp_path,[record()]);proof = json.loads(capture.read_text())
    raw = Path(proof['verification']['captures'][0]['path']);raw.write_bytes(raw.read_bytes()+b' ')
    with pytest.raises(ValueError,match='changed'): import_exports(capture,tmp_path/'changed',workers=1,minimum_plies=2)
    assert not (tmp_path/'changed').exists()
    capture = acquisition(tmp_path,[record()],'other');proof = json.loads(capture.read_text())
    proof['verification']['captures'][0]['url'] += '&players=unverified-name-override'
    atomic_json(capture,proof)
    with pytest.raises(ValueError,match='unrecognized'): import_exports(capture,tmp_path/'bad-query',workers=1,minimum_plies=2)
    assert not (tmp_path/'bad-query').exists()


def test_incorrect_record_count_and_overwrite_cannot_produce_completed_proof(tmp_path):
    capture = acquisition(tmp_path,[record()]);proof = json.loads(capture.read_text())
    proof['verification']['captures'][0]['records'] = 2;atomic_json(capture,proof)
    out = tmp_path/'out'
    with pytest.raises(ValueError,match='record count'): import_exports(capture,out,workers=1,minimum_plies=2)
    assert not (out/'manifest.json').exists()
    with pytest.raises(FileExistsError): import_exports(capture,out,workers=1,minimum_plies=2)
