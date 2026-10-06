import json
import pytest
from xqgeneral.collect_teacher import assemble_label, mirror_label
from xqgeneral.evidence import history_key
from xqgeneral.rules import START_FEN, replay


def sample():
    history = replay(START_FEN, ['b0c2'])
    row = {'id': 'q', 'fen': history[-1], 'initial_fen': START_FEN, 'moves': ['b0c2'],
           'history': history, 'feature_key': history_key(history), 'game_id': 'g', 'split': 'train',
           'query': {}, 'provenance': 'test', 'future_moves': []}
    query = {'id': 'q-explanation', 'record': row,
             'oracle': {'best_move': 'b9c7', 'candidates': [
                 {'move': 'b9c7', 'pv': ['b9c7'], 'score_type': 'cp', 'score': 25}]}}
    annotation = {'id': query['id'], 'teacher_model': 'gpt-6-astra',
                  'reasoning_effort': 'low', 'backend': 'codex_subagent',
                  'explanation': '黑方可走b9c7，把马从底线调出来。这一步没有吃子，也没有将军。'
                                 '评分为黑方视角的正25厘兵，仍应留意红方后续出子和炮的调动，'
                                 '这个数字不能证明已经获胜，更不能当作强制杀棋。'}
    return query, annotation


def test_teacher_contract_and_color_derived_facts():
    query, annotation = sample()
    original = assemble_label(query, annotation)
    derived = mirror_label(original)
    value = json.loads(derived['answer'])
    assert value['move'] == 'b0c2'
    assert value['facts']['piece'] == '红马'
    assert value['evaluation']['value'] == 25
    assert derived['split'] == original['split']
    assert derived['teacher']['independent_teacher_call'] is False
    with pytest.raises(ValueError, match='identity'):
        assemble_label(query, dict(annotation, teacher_model='different-teacher'))


def make_batch(root, records):
    from xqgeneral.evidence import atomic_json, write_jsonl
    annotations = []
    for name, split, moves in records:
        query, annotation = sample()
        history = replay(START_FEN, moves)
        query['record'].update(id=name, split=split, game_id=name, moves=moves,
                               history=history, fen=history[-1], feature_key=history_key(history))
        query.update(id=name+'-explanation', feature_key=history_key(history))
        annotation['id'] = query['id']
        annotations.append((query, annotation))
    for split in ('train', 'validation', 'test'):
        write_jsonl(root/f'{split}.queries.jsonl', [q for q, _ in annotations if q['record']['split'] == split])
    write_jsonl(root/'shards/annotations-0.jsonl', [a for _, a in annotations])
    atomic_json(root/'queries.manifest.json', {'status': 'complete'})


def test_merge_teacher_uses_preparation_manifest_and_preserves_heldout(tmp_path, monkeypatch):
    from xqgeneral.merge_teacher import main
    from xqgeneral.evidence import load_jsonl
    base, extra, dest = [tmp_path/n for n in ('base', 'extra', 'combined')]
    make_batch(base, [('a','train',['b0c2']), ('v','validation',['h0g2']), ('t','test',['h0i2'])])
    make_batch(extra, [('b','train',['b0a2'])])
    monkeypatch.setattr('sys.argv', ['merge_teacher','--inputs',str(base),str(extra),'--output',str(dest)])
    main()
    for split in ('validation', 'test'):
        assert (dest/f'{split}.queries.jsonl').read_bytes() == (base/f'{split}.queries.jsonl').read_bytes()
    assert len(load_jsonl(dest/'train.queries.jsonl')) == 2
    proof = json.loads((dest/'manifest.json').read_text())
    assert str(base/'queries.manifest.json') in proof['inputs']
    assert proof['verification']['held_out_sets_unchanged']


def test_merge_teacher_rejects_added_heldout_before_writing(tmp_path, monkeypatch):
    from xqgeneral.merge_teacher import main
    base, extra, dest = [tmp_path/n for n in ('base', 'extra', 'combined')]
    make_batch(base, [('a','train',['b0c2'])])
    make_batch(extra, [('v','validation',['h0g2'])])
    monkeypatch.setattr('sys.argv', ['merge_teacher','--inputs',str(base),str(extra),'--output',str(dest)])
    with pytest.raises(ValueError, match='held-out'):
        main()
    assert not dest.exists()
