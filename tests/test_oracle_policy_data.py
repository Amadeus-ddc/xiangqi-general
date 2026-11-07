"""Scripted raw-log fixtures exercise data contracts; no engine or model runs."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from xqgeneral import oracle_policy_data as policy
from xqgeneral.evidence import atomic_json, code_identity, digest, history_key, load_jsonl, manifest, position_key, write_jsonl
from xqgeneral.oracle import parse_analysis
from xqgeneral.policy_data import QUESTION as MOVE_QUESTION
from xqgeneral.rules import START_FEN, adjudicate, legal_moves, play, replay
from xqgeneral.search_distillation import NUMERIC_CHILD_QUESTION
from xqgeneral.symmetry import mirror_fen


def query(moves=('b0c2',), score=20, plies=8, record=None):
    history = replay(START_FEN, list(moves)) if record is None else record['history']
    if record is not None:
        moves = record['moves']
    root = history[-1]
    lines = []
    for rank, first in enumerate(sorted(legal_moves(root))[:3], 1):
        pv, fen = [first], play(root, first)
        while len(pv) < plies and not adjudicate(START_FEN, [*moves, *pv])['ended']:
            options = sorted(legal_moves(fen))
            move = options[(len(pv)+rank) % len(options)]
            pv.append(move); fen = play(fen, move)
        lines.append(f'info depth 12 multipv {rank} score cp {score-rank+1} nodes 100000 pv '+ ' '.join(pv))
    lines.append('bestmove '+sorted(legal_moves(root))[0])
    row = dict(id='recorded-search-'+history_key(history), feature_key=history_key(history),
        game_id='train-'+history_key(history), split='train', initial_fen=START_FEN,
        moves=list(moves), history=history, fen=root)
    if record is not None:
        row = {key:record[key] for key in row}
    return dict(record=row, searchmoves=[], requested_nodes=100000,
                response=parse_analysis(root, lines, 100000))


def fixture(tmp_path, monkeypatch, queries=None, reserved=(), roots=None):
    root = Path(tmp_path)
    authority, data, heldout = [root/name for name in ['authority', 'data', 'heldout']]
    authority.mkdir(); data.mkdir(); heldout.mkdir()
    rows = queries if queries is not None else [query()]
    roots = roots if roots is not None else list({q['record']['id']:q['record'] for q in rows}.values())
    roots = [dict(row,recorded_source_kind='recorded_human_match') for row in roots]
    atomic_json(authority/'positions.json', sorted(reserved))
    atomic_json(authority/'counts.json',{'candidate_training_histories':len(roots)})
    write_jsonl(authority/'roots.jsonl',roots)
    for directory in [data, heldout]:
        for split in ['train', 'validation', 'test']:
            write_jsonl(directory/(split+'.jsonl'), [])
    def reservations(a, d, h, seed, bound):
        assert (a, d, h, seed) == (authority, data, heldout, 20261013)
        for directory in [data, heldout]:
            for split in ['train', 'validation', 'test']:
                bound.bind(directory/(split+'.jsonl'))
        positions = bound.json(authority/'positions.json')
        bound.bind(authority/'counts.json');bound.bind(authority/'roots.jsonl')
        return set(positions), dict(profile='controlled_authority_boundary', reserved_positions=len(positions))
    monkeypatch.setattr(policy, 'shared_reservations', reservations)
    executable, weights = root/'engine', root/'nnue'
    executable.write_bytes(b'controlled executable identity, not run')
    weights.write_bytes(b'controlled oracle identity, not run')
    mining = root/'mining'; mining.mkdir()
    ledger = mining/'oracle-queries.partial.jsonl'
    write_jsonl(ledger, rows)
    config = dict(output=str(mining), recorded_inputs=str(authority), data=str(data),
        executable=str(executable), weights=str(weights), nodes=100000, seed=20261013,
        root_offset=0,limit=len(roots))
    proof = manifest('search_distillation_mining', config, [executable, weights], [ledger],
        dict(training_roots_only=True, all_successful_oracle_queries_preserved=True,
             restricted_move_queries_in_oracle_ledger=True, processed=len(roots), oracle_query_records=len(rows)))
    atomic_json(mining/'manifest.json', proof)
    return dict(mining=[mining], authority=authority, data=data, heldout_data=heldout,
                executable=executable, weights=weights, workers=1, chunk_rows=2)


def snapshot(path):
    return {p.name:(digest(p),p.stat().st_mtime_ns) for p in Path(path).iterdir() if p.is_file()}


def test_both_views_share_history_capacity_and_completed_reopen_changes_no_outputs(tmp_path, monkeypatch):
    first, second = query(), query(('h0g2',))
    restricted = deepcopy(first); restricted['searchmoves'] = [restricted['response']['best_move']]
    args = fixture(tmp_path, monkeypatch, [first, deepcopy(first), restricted, second])
    out = tmp_path/'views'
    counts = policy.prepare(**args, output=out)
    assert counts['raw_queries'] == 4 and counts['excluded_query_restricted_query'] == 1
    assert counts['native_checked_potential_records'] == 3 and counts['duplicate_query_records'] == 1
    assert counts['accepted_history_contexts'] == 2 and counts['question_rows'] == 4
    assert counts['rows_by_stage'] == {'engine_numeric':2, 'move_quality':2}
    assert counts['full_teacher_explanations_created'] == 0 and counts['new_engine_queries'] == 0
    assert counts['validation_rows'] == counts['test_rows'] == counts['new_color_derived_rows'] == 0
    rows = load_jsonl(out/'train.jsonl')
    assert len({r['id'] for r in rows}) == 4 and len({r['feature_key'] for r in rows}) == 2
    for numeric, move in zip(rows[::2],rows[1::2]):
        assert numeric['stage'] == 'engine_numeric' and numeric['question'] == NUMERIC_CHILD_QUESTION
        assert move['stage'] == 'move_quality' and move['question'] == MOVE_QUESTION
        assert numeric['source_context_id'] == move['source_context_id']
        assert numeric['history'] == move['history']
        assert json.loads(numeric['answer'])['move'] == move['answer']
        assert move['future_moves'] == [move['answer']]
        assert 'derivation' not in move['teacher']
        assert not numeric['full_teacher_explanation'] and not move['teacher']['neural_prose_generated']
        assert numeric['source_query']['raw_line_sha256']
    before = snapshot(out)
    assert policy.prepare(**dict(args, workers=2, chunk_rows=1), output=out, resume=True) == counts
    assert snapshot(out) == before
    assert not (out/'validation.jsonl').read_bytes() and not (out/'test.jsonl').read_bytes()


def test_serial_and_parallel_publications_have_identical_data_and_counts(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch, [query(),query(('h0g2',))])
    first, second = tmp_path/'one', tmp_path/'two'
    a = policy.prepare(**args, output=first)
    b = policy.prepare(**dict(args, workers=2, chunk_rows=1), output=second)
    assert a == b
    for name in policy.FILES[:7]:
        assert (first/name).read_bytes() == (second/name).read_bytes()


@pytest.mark.parametrize('change', ['score', 'best_move'])
def test_duplicate_conflicts_follow_each_views_target_contract(tmp_path, monkeypatch, change):
    first, changed = query(), query(score=21)
    if change == 'best_move':
        changed = deepcopy(first)
        raw = changed['response']['raw_output']
        raw[-1] = 'bestmove '+changed['response']['candidates'][1]['move']
        changed['response'] = parse_analysis(changed['record']['fen'],raw,100000)
    args = fixture(tmp_path, monkeypatch, [first,changed,query(('h0g2',))])
    out = tmp_path/'views'; counts = policy.prepare(**args, output=out)
    assert counts['accepted_history_contexts'] == (2 if change == 'score' else 1)
    assert counts['duplicate_query_records'] == 1
    assert counts['conflicting_best_move_or_score_contexts'] == 1
    assert counts['conflicting_best_move_contexts'] == int(change == 'best_move')
    assert counts['context_rejections_by_stage']['engine_numeric'] == {'repeated_context_has_conflicting_best_move_or_score':1}
    assert counts['rows_by_stage'] == {'engine_numeric':1,'move_quality':2 if change == 'score' else 1}
    assert all(r['source_context_id'] != policy._context_key(first['record'])
               for r in load_jsonl(out/'train.jsonl') if r['stage'] == 'engine_numeric')


def test_actual_descendant_context_keeps_the_selected_recorded_root_family(tmp_path,monkeypatch):
    from xqgeneral.search_distillation import descend
    base = query()
    child = query(record=descend(base['record'],base['response']['best_move']))
    args = fixture(tmp_path,monkeypatch,[base,child],roots=[base['record']]);out=tmp_path/'views'
    counts = policy.prepare(**args,output=out)
    assert counts['original_search_roots_processed'] == 1 and counts['accepted_history_contexts'] == 2
    assert counts['contexts_by_origin'] == {'recorded_root':1,'searched_descendant':1}
    rows = load_jsonl(out/'train.jsonl')
    assert {row['source_query']['source_root_id'] for row in rows} == {base['record']['id']}
    assert {row['source_query']['extra_plies_after_recorded_root'] for row in rows} == {0,1}


@pytest.mark.parametrize('damage',['outside_range','wrong_game','wrong_suffix'])
def test_queries_cannot_forge_a_selected_recorded_root_family(tmp_path,monkeypatch,damage):
    base=query();changed=deepcopy(base)
    if damage == 'outside_range':changed['record']['id']='recorded-search-'+'0'*64
    elif damage == 'wrong_game':changed['record']['game_id']='unrelated-game'
    else:changed['record']['id']+='-a0a1'
    args=fixture(tmp_path,monkeypatch,[changed],roots=[base['record']])
    with pytest.raises(ValueError,match='root|range'):
        policy.prepare(**args,output=tmp_path/'views')


def test_first_query_precedence_preserves_valid_move_but_not_short_numeric_PV(tmp_path, monkeypatch):
    short, later = query(plies=2), query()
    args = fixture(tmp_path,monkeypatch,[short,later,query(('h0g2',))])
    counts = policy.prepare(**args,output=tmp_path/'views')
    assert counts['accepted_history_contexts'] == 2 and counts['duplicate_query_records'] == 1
    assert counts['rows_by_stage'] == {'engine_numeric':1,'move_quality':2}
    assert counts['context_rejections_by_stage']['engine_numeric'] == {'short_nonterminal_best_PV':1}
    assert counts['context_rejections_by_stage']['move_quality'] == {}


def test_missing_extra_root_candidates_does_not_discard_a_valid_single_move(tmp_path,monkeypatch):
    item = query()
    raw = item['response']['raw_output']
    raw.pop(2)
    item['response'] = parse_analysis(item['record']['fen'],raw,100000)
    args = fixture(tmp_path,monkeypatch,[item]);out=tmp_path/'views'
    counts = policy.prepare(**args,output=out)
    assert counts['accepted_history_contexts'] == 1
    assert counts['rows_by_stage'] == {'engine_numeric':0,'move_quality':1}
    assert counts['context_rejections_by_stage']['engine_numeric'] == {'missing_exact_root_candidates':1}
    assert load_jsonl(out/'train.jsonl')[0]['answer'] == item['response']['best_move']


@pytest.mark.parametrize('reserved', ['root','candidate_PV','color_mirror'])
def test_root_all_candidate_PVs_and_color_mirrors_are_excluded(reserved):
    item = query(); root = item['record']['fen']
    position = root if reserved == 'root' else play(root,item['response']['candidates'][2]['pv'][0])
    if reserved == 'color_mirror':
        position = mirror_fen(position)
    with pytest.raises(policy.Excluded,match='reserved'):
        policy.numeric_target(item,{position_key(position)})


def test_raw_bounds_cannot_replace_the_last_exact_score():
    item = query(); raw = item['response']['raw_output']; best = item['response']['best_move']
    raw.insert(-1,'info depth 30 multipv 1 score cp 9999 lowerbound pv '+best)
    item['response'] = parse_analysis(item['record']['fen'],raw,100000)
    _,best = policy.checked_query(item)
    assert best['score'] == 20
    target,_,_ = policy.numeric_target(item,set())
    assert target['evaluation']['value'] == 20


@pytest.mark.parametrize('damage', ['heldout','history','fen','stored_score','bound_only','invalid_initial'])
def test_contradictory_successful_source_records_fail_clearly(damage):
    item = query()
    if damage == 'heldout':item['record']['split'] = 'test'
    elif damage == 'history':item['record']['history'] = item['record']['history'][:-1]
    elif damage == 'fen':item['record']['fen'] = START_FEN
    elif damage == 'stored_score':item['response']['candidates'][0]['score'] += 1
    elif damage == 'bound_only':
        best = item['response']['best_move']
        item['response']['raw_output'] = ['info depth 1 score cp 100 lowerbound pv '+best,'bestmove '+best]
    else:item['record']['initial_fen'] = '9/9/9/9/9/9/9/9/9/9 w - - 0 1'
    with pytest.raises(ValueError):policy.checked_query(item)


@pytest.mark.parametrize('damage',['incomplete','source_code','ownership','different_engine','ledger_bytes','seed'])
def test_source_admission_checks_completed_provenance_and_exact_assets(tmp_path,monkeypatch,damage):
    args = fixture(tmp_path,monkeypatch)
    path = args['mining'][0]/'manifest.json'; proof = json.loads(path.read_text())
    if damage == 'incomplete':proof['status'] = 'partial'
    elif damage == 'source_code':proof['code']['source_sha256']['src/xqgeneral/oracle.py'] = '0'*64
    elif damage == 'ownership':proof['verification']['training_roots_only'] = False
    elif damage == 'different_engine':proof['config']['executable'] = str(tmp_path/'other')
    elif damage == 'seed':proof['config']['seed'] += 1
    else:(args['mining'][0]/'oracle-queries.partial.jsonl').write_bytes(b'changed\n')
    atomic_json(path,proof)
    with pytest.raises(ValueError):policy.prepare(**args,output=tmp_path/'views')


def test_completed_reopen_recomputes_content_even_if_a_changed_output_is_resealed(tmp_path,monkeypatch):
    args = fixture(tmp_path,monkeypatch); out = tmp_path/'views'
    policy.prepare(**args,output=out)
    rows = load_jsonl(out/'train.jsonl'); rows[0]['answer'] = 'a0a1'; write_jsonl(out/'train.jsonl',rows)
    proof = json.loads((out/'manifest.json').read_text()); path = out/'train.jsonl'
    proof['outputs'][str(path)] = {'sha256':digest(path),'bytes':path.stat().st_size}
    atomic_json(out/'manifest.json',proof); before = snapshot(out)
    with pytest.raises(ValueError,match='replay differs'):
        policy.prepare(**args,output=out,resume=True)
    assert snapshot(out) == before


def test_partial_outputs_and_changed_source_contracts_are_preserved(tmp_path,monkeypatch):
    args = fixture(tmp_path,monkeypatch); out = tmp_path/'views'; out.mkdir()
    (out/'partial').write_text('preserve')
    with pytest.raises(FileExistsError):policy.prepare(**args,output=out,resume=True)
    assert (out/'partial').read_text() == 'preserve'
    complete = tmp_path/'complete'; policy.prepare(**args,output=complete)
    with pytest.raises(ValueError,match='contract changed'):
        policy.prepare(**dict(args,minimum_nodes=1000000),output=complete,resume=True)


def test_native_finite_SFT_accepts_both_explicit_stages_without_explanation_relabeling(tmp_path,monkeypatch):
    from xqgeneral.finite_training import FINITE_PROFILE, FULL_SOURCE_PROFILE
    from xqgeneral.sft import prepare_config
    args = fixture(tmp_path,monkeypatch); out = tmp_path/'views'
    policy.prepare(**args,output=out)
    recipe = dict(data_path=str(out),stages=['move_quality','engine_numeric'],
        mixture={'move_quality':.5,'engine_numeric':.5},batch_size=2,micro_batch_size=1,
        epochs=1,max_steps=10,min_steps=1,training_data_profile=FINITE_PROFILE,
        source_coverage_profile=FULL_SOURCE_PROFILE)
    config = prepare_config({'mode':'text_lora'},recipe,tmp_path/'parent.pt',tmp_path/'training')
    assert config['steps'] == 1 and config['mixture'] == recipe['mixture']
    assert 'explanation' not in config['stages']
