import copy
import json

import pytest

from xqgeneral.explanations import move_facts
from xqgeneral.rules import START_FEN, replay
from xqgeneral.revise_prose import annotation_hash, read_review_bundle, revise_row
from xqgeneral.evidence import atomic_json, digest, manifest, write_jsonl


def fixture_row():
    score = {'type': 'cp', 'value': 12, 'perspective': 'side_to_move'}
    value = {'move': 'b0c2', 'pv': ['b0c2', 'b9c7'], 'candidates': ['b0c2'],
             'branches': [{'move': 'b0c2', 'pv': ['b0c2', 'b9c7'], 'evaluation': score}],
             'facts': move_facts(START_FEN, 'b0c2'), 'evaluation': score,
             'explanation': '红马从b0发展至c2，随后黑马从b9至c7。' * 5}
    return {'id': 'train-explanation', 'split': 'train', 'stage': 'explanation',
            'initial_fen': START_FEN, 'moves': [], 'fen': START_FEN, 'history': [START_FEN],
            'question': '分析局面', 'answer': json.dumps(value, ensure_ascii=False),
            'feature_key': 'immutable', 'teacher_query_id': 'original-query',
            'future_moves': ['b0c2', 'b9c7'], 'future_branches': [['b0c2', 'b9c7']],
            'teacher': {'teacher_model': 'gpt-6-astra', 'reasoning_effort': 'low', 'backend': 'codex_subagent'}}


def annotation(row, prose=None):
    return {'id': row['id'], 'explanation': prose or json.loads(row['answer'])['explanation'], **row['teacher']}


def test_reviewed_revision_preserves_moves_scores_histories_and_original():
    row = fixture_row(); before = copy.deepcopy(row); original = annotation(row)
    corrected = dict(original, explanation='推荐红马从b0至c2，示例黑方以b9c7发展马。两步都没有吃子或将军，不能据此声称强制获胜。' * 2)
    result = revise_row(row, corrected, annotation_hash(original), annotation_hash(corrected), 'review-manifest')
    assert row == before
    old, new = json.loads(row['answer']), json.loads(result['answer'])
    assert {k:v for k,v in new.items() if k != 'explanation'} == {k:v for k,v in old.items() if k != 'explanation'}
    assert result['prose_review']['changed'] is True
    assert result['prose_review']['human_rating'] is False
    for key in row:
        if key != 'answer': assert result[key] == row[key]
    unchanged = revise_row(row, original, annotation_hash(original), annotation_hash(original), 'review-manifest')
    assert unchanged['answer'] == row['answer']


def test_revision_rejects_heldout_foreign_teacher_and_changed_hash_bindings():
    row = fixture_row(); original = annotation(row); sha = annotation_hash(original)
    for bad_row, bad_annotation, source_hash, final_hash in [
        (dict(row, split='test'), original, sha, sha),
        (row, dict(original, id='foreign'), sha, annotation_hash(dict(original, id='foreign'))),
        (row, dict(original, teacher_model='other'), sha, annotation_hash(dict(original, teacher_model='other'))),
        (row, original, 'changed-source', sha),
        (row, original, sha, 'changed-review'),
    ]:
        with pytest.raises(ValueError): revise_row(bad_row, bad_annotation, source_hash, final_hash, 'review')


def test_revision_rejects_continuation_after_complete_history_terminal():
    row = fixture_row(); moves = (['b0c2','b9c7','c2b0','c7b9']*2)[:-1]
    history = replay(START_FEN,moves); score = {'type':'cp','value':0,'perspective':'side_to_move'}
    value = {'move':'c7b9','pv':['c7b9','b0c2'],'candidates':['c7b9'],
             'branches':[{'move':'c7b9','pv':['c7b9'],'evaluation':score}],
             'evaluation':score,'facts':move_facts(history[-1],'c7b9'),
             'explanation':'红黑马的变化必须遵守完整历史，不能越过重复终局继续。'*4}
    row.update(moves=moves,history=history,fen=history[-1],answer=json.dumps(value,ensure_ascii=False))
    old = annotation(row); sha = annotation_hash(old)
    with pytest.raises(ValueError): revise_row(row,old,sha,sha,'review')


def test_review_bundle_requires_exact_rejected_parent_and_accepted_final(tmp_path):
    original = annotation(fixture_row()); source_hash = annotation_hash(original)
    packets = tmp_path/'packets.jsonl'; write_jsonl(packets,[{'teacher_annotation':original}])
    rejection = tmp_path/'rejection.json'
    atomic_json(rejection,{'results':[{'id':original['id'],'verdict':'reject','reviewed_annotation_sha256':source_hash}]})
    correction = dict(original,explanation='红马发展后，黑马也展开；示例的两步都没有吃子或将军，不能把有限变化当作强制胜法。'*3,
                      corrected_from_annotation_sha256=source_hash,correction_review_sha256=digest(rejection))
    resolved = tmp_path/'annotations.jsonl';write_jsonl(resolved,[correction])
    acceptance = tmp_path/'acceptance.json'
    def save_bundle():
        atomic_json(acceptance,{'results':[{'id':original['id'],'verdict':'accept',
                                         'reviewed_annotation_sha256':annotation_hash(correction)}]})
        atomic_json(tmp_path/'manifest.json',manifest('review_fixture',{},[packets,rejection,acceptance],[resolved],{'human_rating':False}))
    save_bundle()
    assert read_review_bundle(tmp_path)[original['id']][1:3] == (source_hash,annotation_hash(correction))
    correction['corrected_from_annotation_sha256']='unknown-parent';write_jsonl(resolved,[correction]);save_bundle()
    with pytest.raises(ValueError,match='exact rejected parent'):read_review_bundle(tmp_path)
    correction['corrected_from_annotation_sha256']=source_hash;write_jsonl(resolved,[correction]);save_bundle()
    resolved.write_text('[]\n')
    with pytest.raises(ValueError,match='input or output changed'):read_review_bundle(tmp_path)
