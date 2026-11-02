from copy import deepcopy
import json

import pytest

from xqgeneral.course_tasks import grade_answer, native_tag, paper_answer, with_rationale_profile
from xqgeneral.evidence import atomic_json, load_jsonl
from xqgeneral.forward_rationale import (FORWARD_MOVES_PROFILE, FORWARD_PARRY_PROFILE,
                                        PARRY_REASONING_PROFILE, forward_move_narration)
from xqgeneral.foundation_preflight import native_answer
from xqgeneral.paper_curriculum import main as curriculum_main, readback
from xqgeneral.parry_rationale import parry_rationale
from xqgeneral.rules import check_evasion_facts, king_checkers, legal_moves, validate_position

from test_paper_course_tasks import CHECK, MATE, STALEMATE, row
from test_paper_curriculum import build, source_fixture


def position(**pieces):
    """Construct explicit mechanism probes; these are not recorded games."""
    ranks = []
    for rank in range(9, -1, -1):
        text, empty = '', 0
        for file in 'abcdefghi':
            piece = pieces.get(f'{file}{rank}')
            if piece is None:
                empty += 1
            else:
                text += (str(empty) if empty else '') + piece
                empty = 0
        ranks.append(text + (str(empty) if empty else ''))
    return validate_position('/'.join(ranks) + ' w - - 0 1')


@pytest.mark.parametrize('pieces,move,checker,mechanism,text', [
    ({'e0':'K','e9':'k','e5':'r','d5':'R'}, 'd5e5', 'e5', 'capture_checker', '吃掉将军的棋子'),
    ({'e0':'K','e9':'k','e5':'r','d3':'R'}, 'd3e3', 'e5', 'block_rook_line', '挡住车的将军线路'),
    ({'e0':'K','e9':'k','e5':'c','e2':'R'}, 'e2d2', 'e5', 'remove_cannon_screen', '撤走炮架'),
    ({'e0':'K','e9':'k','e5':'c','e3':'p','d2':'R'}, 'd2e2', 'e5', 'add_cannon_screen', '添子阻隔炮线'),
    ({'e0':'K','e9':'k','e6':'p','d2':'h','c1':'R'}, 'c1d1', 'd2', 'block_horse_leg', '塞住马腿'),
    ({'e0':'K','e9':'k','e6':'p','e1':'p','d1':'R'}, 'd1e1', 'e1', 'capture_checker', '吃掉将军的棋子'),
    ({'e0':'K','e9':'k','e6':'p','e1':'p','d1':'R'}, 'e0f0', 'e1', 'king_move', '移动将帅避将'),
])
def test_checker_and_evasion_mechanisms_cover_xiangqi_specific_rules(pieces, move, checker, mechanism, text):
    fen = position(**pieces)
    assert king_checkers(fen) == ((checker, pieces[checker]),)
    facts = check_evasion_facts(fen)
    assert [fact.move for fact in facts] == sorted(legal_moves(fen))
    fact = next(fact for fact in facts if fact.move == move)
    assert fact.checker_mechanisms == ((checker, mechanism),)
    assert fact.piece == pieces[move[:2]] and fact.captured == pieces.get(move[2:])
    prose = parry_rationale(fen, ' '.join(sorted(legal_moves(fen))))
    assert text in prose and '红帅在e0' in prose and checker in prose


def test_one_move_can_resolve_simultaneous_cannon_and_horse_checks():
    fen = position(e0='K', e9='k', e5='c', d2='h', e1='R')
    assert king_checkers(fen) == (('d2', 'h'), ('e5', 'c'))
    fact = next(fact for fact in check_evasion_facts(fen) if fact.move == 'e1d1')
    assert fact.checker_mechanisms == (('d2','block_horse_leg'), ('e5','remove_cannon_screen'))
    text = parry_rationale(fen, native_tag(row('parries', fen=fen)))
    assert '同时将军' in text and '撤走炮架并塞住马腿' in text


def test_countercheck_capture_and_checkmate_are_preserved_as_separate_facts():
    fen = position(e0='K', e9='k', e5='r', d5='R')
    fact = next(fact for fact in check_evasion_facts(fen) if fact.move == 'd5e5')
    assert fact.captured == 'r' and fact.check and not fact.checkmate
    assert '红车从d5到e5吃黑车并将军' in parry_rationale(fen, native_tag(row('parries', fen=fen)))
    assert check_evasion_facts(MATE) == ()
    assert '黑将在e9' in parry_rationale(MATE, '无')
    assert '已被将死' in parry_rationale(MATE, '无')
    with pytest.raises(ValueError, match='canonical native move tag'):
        parry_rationale(fen, '无')


@pytest.mark.parametrize('fen', [STALEMATE, '4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1'])
def test_no_native_check_is_not_relabelled_as_a_check_evasion_question(fen):
    with pytest.raises(ValueError, match='native in-check'):
        check_evasion_facts(fen)
    with pytest.raises(ValueError, match='General color'):
        king_checkers(fen, 'red')


@pytest.mark.parametrize('profile', [PARRY_REASONING_PROFILE, FORWARD_PARRY_PROFILE])
def test_current_parry_annotation_keeps_canonical_gold_and_all_expert_inputs(profile):
    question = row('parries', fen=CHECK)
    question.update(id='current-parry', answer=paper_answer(question))
    original = deepcopy(question)
    new = with_rationale_profile(question, profile)
    assert new['id'] != question['id'] and new['rationale_source_question_id'] == question['id']
    assert new['answer'].startswith('红帅在e0，正被e5的黑车将军。')
    assert grade_answer(question['answer'], new['answer'], 'parries')['correct']
    assert native_answer(new) == new['answer']
    assert with_rationale_profile(new, profile) == new and question == original
    for key in ('fen','question','history','initial_fen','moves','future_moves','feature_key','query'):
        assert new.get(key) == question.get(key)


def test_combined_future_profile_narrates_supplied_line_then_explains_final_check():
    fen = position(e0='K', e9='k', d5='r')
    question = row('parries', fen=fen, future=['e0f0','d5f5'])
    question.update(id='future-parry', answer=paper_answer(question))
    parry = with_rationale_profile(question, PARRY_REASONING_PROFILE)
    combined = with_rationale_profile(question, FORWARD_PARRY_PROFILE)
    assert parry['answer'].startswith('红帅在f0，正被f5的黑车将军。')
    assert combined['answer'] == forward_move_narration(question) + parry['answer']
    assert parry['id'] != combined['id']
    assert grade_answer(question['answer'], combined['answer'], 'parries')['correct']


def test_profiles_apply_only_to_declared_tasks_and_previous_forward_profile_is_preserved():
    current = row('counts', id='current-nonparry')
    future = row('piece', {'square':'b9'}, future=['b2b9','a9b9'], id='future-nonparry')
    for question in (current, future):
        question['answer'] = paper_answer(question)
    assert with_rationale_profile(current, FORWARD_PARRY_PROFILE) == current
    assert with_rationale_profile(future, PARRY_REASONING_PROFILE) == future
    forward = with_rationale_profile(future, FORWARD_MOVES_PROFILE)
    combined = with_rationale_profile(future, FORWARD_PARRY_PROFILE)
    assert forward['answer'] == combined['answer'] == forward_move_narration(future) + future['answer']
    with pytest.raises(ValueError, match='does not apply'):
        paper_answer(dict(current, rationale_profile=PARRY_REASONING_PROFILE))
    with pytest.raises(ValueError, match='Unknown curriculum rationale'):
        with_rationale_profile(future, [])


@pytest.mark.parametrize('profile', [PARRY_REASONING_PROFILE, FORWARD_PARRY_PROFILE])
def test_profiled_production_and_complete_readback_bind_new_identities(tmp_path, profile):
    roots, proof, _ = source_fixture(tmp_path)
    baseline, profiled = tmp_path/'baseline', tmp_path/'profiled'
    build([roots], [proof], baseline, seed=7)
    result = build([roots], [proof], profiled, seed=7, rationale_profile=profile)
    assert result['questions'] == 300 and result['rationale_profile'] == profile
    changed = 0
    for split in ('train','validation','test'):
        for old,new in zip(load_jsonl(baseline/f'{split}.jsonl'),load_jsonl(profiled/f'{split}.jsonl'),strict=True):
            applies = old['task_type']=='parries' or (profile==FORWARD_PARRY_PROFILE and old['stage'].endswith('_future'))
            if not applies:
                assert new == old
                continue
            changed += 1
            assert new['rationale_source_question_id']==old['id'] and new['id']!=old['id']
            assert grade_answer(old['answer'],new['answer'],old['task_type'])['correct']
            assert native_answer(new) == new['answer']
            ignored = {'id','answer','rationale_profile','rationale_source_question_id'}
            assert {k:v for k,v in old.items() if k not in ignored} == {k:v for k,v in new.items() if k not in ignored}
    assert changed > 0
    checked = readback(profiled,tmp_path/'readback')
    assert checked['questions_regenerated_and_compared']==300 and checked['rationale_profile']==profile
    proof_path = profiled/'manifest.json'
    value = json.loads(proof_path.read_text())
    value['config']['rationale_profile'] = FORWARD_MOVES_PROFILE
    atomic_json(proof_path,value)
    with pytest.raises(ValueError,match='Stored rationale profile'):
        readback(profiled,tmp_path/'rejected')


@pytest.mark.parametrize('profile', [PARRY_REASONING_PROFILE, FORWARD_PARRY_PROFILE])
def test_cli_binds_each_explicit_new_profile(tmp_path, monkeypatch, profile):
    roots, proof, _ = source_fixture(tmp_path)
    output = tmp_path/'cli-data'
    monkeypatch.setattr('sys.argv',['paper_curriculum','--roots',str(roots),'--source-manifests',str(proof),
        '--reference-data',str(roots.parent/'reference'),'--rationale-profile',profile,'--output',str(output)])
    curriculum_main()
    assert json.loads((output/'manifest.json').read_text())['config']['rationale_profile']==profile
