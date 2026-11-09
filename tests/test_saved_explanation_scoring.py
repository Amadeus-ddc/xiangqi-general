import json
from pathlib import Path
import random

import pytest

from test_explanation_score_coverage import BoundOracle, raw_answer
from xqgeneral import score_saved_explanations as scoring
from xqgeneral.evidence import atomic_json, digest, load_jsonl, write_jsonl
from xqgeneral.explanations import EXPLANATION_QUESTION
from xqgeneral.rules import START_FEN


def binding(path):
    return {'sha256': digest(path), 'bytes': path.stat().st_size}


def preserved_generation(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    validation = data / 'validation.jsonl'
    rows = [dict(id=str(i), split='validation', initial_fen=START_FEN, fen=START_FEN,
                 moves=[], question='original archived question') for i in range(2)]
    write_jsonl(validation, rows)
    random.Random(17).shuffle(rows)
    raw = tmp_path / 'preserved-raw.jsonl'
    write_jsonl(raw, [dict(id=r['id'], record=dict(r, question=EXPLANATION_QUESTION),
                           raw=raw_answer() if i == 0 else 'unmodified invalid JSON')
                      for i, r in enumerate(rows)])
    origin = tmp_path / 'origin.json'
    atomic_json(origin, {'generation_finished': True, 'judging_finished': False})
    engine, weights = tmp_path / 'engine', tmp_path / 'engine.nnue'
    engine.write_bytes(b'controlled engine'); weights.write_bytes(b'controlled weights')
    receipt = tmp_path / 'receipt.json'
    atomic_json(receipt, {'kind': scoring.KIND, 'status': 'complete_raw_answers_only',
        'generation_config': {'data': str(data), 'checkpoint': 'archived-not-loaded', 'features': 'not-loaded',
            'split': 'validation', 'limit': 2, 'batch_size': 2, 'max_new_tokens': 768,
            'memory': 'normal', 'seed': 17, 'nodes': 1000000, 'workers': 2,
            'executable': str(engine), 'weights': str(weights)},
        'origin_artifacts': {str(origin): binding(origin)},
        'generation_code': {'revision': 'controlled-archived-generation', 'source_sha256': {}},
        'validation': dict(path=str(validation), **binding(validation)),
        'raw_predictions': dict(path=str(raw), **binding(raw)),
        'engine_artifacts': {str(engine): binding(engine), str(weights): binding(weights)}})
    return receipt, raw


def test_real_saved_scoring_entry_preserves_every_raw_error_and_unscored_question(tmp_path, monkeypatch):
    receipt, raw = preserved_generation(tmp_path)
    old_raw = raw.read_bytes()
    output = tmp_path / 'scores'
    seen = []
    class Oracle(BoundOracle):
        def __init__(self, *args, **kwargs):
            super().__init__(1)
            seen.append(self)
        def close(self):pass
    monkeypatch.setattr(scoring, 'Pikafish', Oracle)
    result = scoring.run_scoring(receipt, output)
    assert result['examples'] == 2 and result['parse_valid_rate'] == .5
    assert result['first_move_legal_rate'] == .5 and result['oracle_repairs'] == 0
    assert result['oracle_scoring_coverage']['questions_with_unscored_queries'] == 1
    assert result['new_model_or_teacher_queries'] == 0
    assert (output / 'raw-predictions.jsonl').read_bytes() == old_raw == raw.read_bytes()
    judged = load_jsonl(output / 'judged-predictions.jsonl')
    assert [row['id'] for row in judged] == ['0', '1']
    assert sorted(row['raw'] for row in judged) == sorted(row['raw'] for row in load_jsonl(raw))
    assert sum(len(oracle.calls) for oracle in seen) == 2
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['kind'] == 'recovered_independent_explanation_evaluation'
    assert proof['config']['generation_config']['nodes'] == 1000000
    assert proof['verification'] == result
    assert proof['inputs'][str(raw)] == binding(raw)
    with pytest.raises(FileExistsError):scoring.run_scoring(receipt, output)


@pytest.mark.parametrize('change', ['raw_hash', 'origin_hash', 'engine_hash', 'foreign_question', 'duplicate_id',
                                   'reordered', 'test_split', 'bool_budget'])
def test_saved_scoring_refuses_changed_inputs_before_engine_load(tmp_path, monkeypatch, change):
    receipt, raw = preserved_generation(tmp_path)
    proof = json.loads(receipt.read_text())
    if change == 'raw_hash':raw.write_text(raw.read_text()+'\n')
    elif change == 'origin_hash':Path(next(iter(proof['origin_artifacts']))).write_text('{}')
    elif change == 'engine_hash':Path(next(iter(proof['engine_artifacts']))).write_bytes(b'changed')
    elif change in {'foreign_question', 'duplicate_id', 'reordered'}:
        rows = load_jsonl(raw)
        if change == 'foreign_question':rows[0]['record']['question'] = 'different'
        elif change == 'duplicate_id':rows[1]['id'] = rows[0]['id']
        else:rows.reverse()
        write_jsonl(raw, rows);proof['raw_predictions'].update(binding(raw))
    elif change == 'test_split':proof['generation_config']['split'] = 'test'
    else:proof['generation_config']['nodes'] = True
    atomic_json(receipt, proof)
    def forbidden(*args, **kwargs):raise AssertionError('Invalid inputs must fail before engine load')
    monkeypatch.setattr(scoring, 'Pikafish', forbidden)
    with pytest.raises(ValueError):scoring.run_scoring(receipt, tmp_path / 'scores')
    assert not (tmp_path / 'scores/manifest.json').exists()
