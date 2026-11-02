from copy import deepcopy
import json

import pytest

from xqgeneral.course_tasks import grade_answer, paper_answer, with_rationale_profile
from xqgeneral.evidence import atomic_json, load_jsonl
from xqgeneral.forward_rationale import FORWARD_MOVES_PROFILE, forward_move_facts, forward_move_narration
from xqgeneral.foundation_preflight import native_answer
from xqgeneral.paper_curriculum import readback
from xqgeneral.paper_curriculum import main as curriculum_main
from xqgeneral.rules import in_check, legal_moves, replay

from test_paper_course_tasks import BEFORE_MATE, row
from test_paper_curriculum import build, source_fixture


def test_capture_narration_uses_each_pre_move_board_in_chronological_order():
    question = row('piece', {'square': 'b9'}, future=['b2b9', 'a9b9'])
    facts = forward_move_facts(question)
    assert [(fact.piece, fact.captured) for fact in facts] == [('C', 'h'), ('r', 'C')]
    text = forward_move_narration(question)
    assert text == '首先，红炮从b2走到b9，吃掉黑马。最后，黑车从a9走到b9，吃掉红炮。'
    annotated = dict(question, rationale_profile=FORWARD_MOVES_PROFILE)
    assert paper_answer(annotated).endswith('答案：黑车')
    assert grade_answer(paper_answer(question), paper_answer(annotated), 'piece')['correct']


@pytest.mark.parametrize('variant,first', [(0, '首先，'), (1, '第1着，')])
def test_narration_is_deterministic_and_does_not_mutate_the_supplied_question(variant, first):
    question = row('counts', future=['b0c2', 'b9c7'], rationale_variant=variant)
    original = deepcopy(question)
    assert forward_move_narration(question) == forward_move_narration(question)
    assert forward_move_narration(question).startswith(first + '红马从b0走到c2。')
    assert question == original


@pytest.mark.parametrize('fen,move,check,mate,stalemate,text', [
    ('4k4/9/9/9/3R5/9/9/9/9/5K3 w - - 0 1', 'd5e5', True, False, False, '形成将军'),
    (BEFORE_MATE, 'f7e7', True, True, False, '将死对方'),
    (BEFORE_MATE, 'f0f1', False, False, True, '形成困毙'),
])
def test_check_checkmate_and_xiangqi_stalemate_are_distinct_native_facts(fen, move, check, mate, stalemate, text):
    question = row('mate', fen=fen, future=[move])
    target = replay(fen, [move])[-1]
    assert in_check(target) is check
    assert (not legal_moves(target)) is (mate or stalemate)
    fact, = forward_move_facts(question)
    assert (fact.check, fact.checkmate, fact.stalemate) == (check, mate, stalemate)
    assert text in forward_move_narration(question)


@pytest.mark.parametrize('future', [[], ['a0a9'], ['bad'], ['b0c2'] * 9])
def test_invalid_supplied_lines_fail_without_a_move_substitution(future):
    with pytest.raises(ValueError):
        forward_move_facts(row('counts', future=future))


def test_annotation_preserves_inputs_and_tag_with_distinct_reproducible_identity():
    question = row('counts', future=['b0c2', 'b9c7'])
    question['id'] = 'source-native-question'
    question['answer'] = paper_answer(question)
    original = deepcopy(question)
    annotated = with_rationale_profile(question, FORWARD_MOVES_PROFILE)
    assert annotated['id'] != question['id']
    assert annotated['rationale_source_question_id'] == question['id']
    assert with_rationale_profile(annotated, FORWARD_MOVES_PROFILE) == annotated
    assert annotated['answer'].split('\n\n答案：')[1] == question['answer'].split('\n\n答案：')[1]
    assert question == original
    for key in ['fen', 'history', 'initial_fen', 'moves', 'future_moves', 'feature_key', 'query']:
        assert annotated[key] == question[key]
    with pytest.raises(ValueError, match='profiles differ'):
        with_rationale_profile(annotated)


def test_current_questions_keep_their_original_identity_and_answer():
    question = row('counts')
    question.update(id='current-question', answer=paper_answer(question))
    assert with_rationale_profile(question, FORWARD_MOVES_PROFILE) == question
    assert with_rationale_profile(question) == question
    with pytest.raises(ValueError, match='future-stage'):
        paper_answer(dict(question, rationale_profile=FORWARD_MOVES_PROFILE))
    with pytest.raises(ValueError, match='Unknown curriculum rationale'):
        paper_answer(dict(question, rationale_profile='unknown'))


def test_profiled_corpus_is_reconstructed_and_foundation_checks_accept_unchanged_tags(tmp_path):
    roots, proof, _ = source_fixture(tmp_path)
    baseline, profiled = tmp_path / 'baseline', tmp_path / 'profiled'
    build([roots], [proof], baseline, seed=7)
    result = build([roots], [proof], profiled, seed=7, rationale_profile=FORWARD_MOVES_PROFILE)
    assert result['rationale_profile'] == FORWARD_MOVES_PROFILE
    assert result['questions'] == 300
    changed = 0
    for split in ('train', 'validation', 'test'):
        for old, new in zip(load_jsonl(baseline / f'{split}.jsonl'), load_jsonl(profiled / f'{split}.jsonl'), strict=True):
            if old['stage'].endswith('_current'):
                assert new == old
            else:
                changed += 1
                assert new['rationale_source_question_id'] == old['id'] and new['id'] != old['id']
                assert grade_answer(old['answer'], new['answer'], old['task_type'])['correct']
                assert native_answer(new) == new['answer']
                assert {k:v for k,v in new.items() if k not in ['id','answer','rationale_profile','rationale_source_question_id']} == {
                    k:v for k,v in old.items() if k not in ['id','answer']}
    assert changed > 0
    checked = readback(profiled, tmp_path / 'readback')
    assert checked['questions_regenerated_and_compared'] == 300
    assert checked['rationale_profile'] == FORWARD_MOVES_PROFILE


def test_profile_declaration_cannot_be_removed_or_replaced_after_preparation(tmp_path):
    roots, proof, _ = source_fixture(tmp_path)
    data = tmp_path / 'profiled'
    build([roots], [proof], data, rationale_profile=FORWARD_MOVES_PROFILE)
    path = data / 'manifest.json'
    value = json.loads(path.read_text())
    value['config'].pop('rationale_profile')
    atomic_json(path, value)
    with pytest.raises(ValueError, match='Stored rationale profile'):
        readback(data, tmp_path / 'rejected-readback')
    with pytest.raises(ValueError, match='Unknown curriculum rationale'):
        build([roots], [proof], tmp_path / 'unknown', rationale_profile='unknown')
    assert not (tmp_path / 'unknown').exists()


def test_cli_prepares_and_reads_the_explicit_profile_without_readback_overrides(tmp_path, monkeypatch):
    roots, proof, _ = source_fixture(tmp_path)
    data, checked = tmp_path / 'cli-data', tmp_path / 'cli-readback'
    monkeypatch.setattr('sys.argv', ['paper_curriculum', '--roots', str(roots),
        '--source-manifests', str(proof), '--reference-data', str(roots.parent / 'reference'),
        '--rationale-profile', FORWARD_MOVES_PROFILE, '--output', str(data)])
    curriculum_main()
    assert json.loads((data / 'manifest.json').read_text())['config']['rationale_profile'] == FORWARD_MOVES_PROFILE
    monkeypatch.setattr('sys.argv', ['paper_curriculum', '--readback', str(data), '--output', str(checked)])
    curriculum_main()
    assert json.loads((checked / 'verification.json').read_text())['questions_regenerated_and_compared'] == 300
    monkeypatch.setattr('sys.argv', ['paper_curriculum', '--readback', str(data),
        '--rationale-profile', FORWARD_MOVES_PROFILE, '--output', str(tmp_path / 'override')])
    with pytest.raises(SystemExit) as error:
        curriculum_main()
    assert error.value.code == 2
    assert not (tmp_path / 'override').exists()
