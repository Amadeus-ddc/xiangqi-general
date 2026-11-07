"""Native grading/data/resume with a controlled Predictor; no neural-score claims."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import test_auxiliary_policy_preflight as input_control
import test_oracle_policy_corpus as corpus_control
from xqgeneral import inference, oracle_policy_evaluation as evaluation, oracle_policy_corpus as corpus
from xqgeneral.evidence import atomic_json, digest, load_jsonl, write_jsonl
from xqgeneral.feature_store import FeatureStoreWriter
from xqgeneral.oracle_policy_data import _numeric_after_isolation
from xqgeneral.policy_data import QUESTION
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.search_distillation import NUMERIC_CHILD_QUESTION


def artifact(path): return dict(sha256=digest(path), bytes=Path(path).stat().st_size)


def example_rows():
    context = corpus_control.context(['h0g2', 'h9g7'], 'validation')
    target, _ = _numeric_after_isolation(corpus_control.query(context, 1000000))
    return [dict(context, id='numeric', stage='engine_numeric', task_type='move_PV_candidates_evaluation_v1',
                 question=NUMERIC_CHILD_QUESTION, answer=json.dumps(target)),
            dict(context, id='move', stage='move_quality', task_type='best_move', question=QUESTION, answer=target['move'])]


def test_native_reference_answers_pass_without_changing_the_raw_text():
    for row in example_rows():
        result = evaluation.prediction_record(row, row['answer'])
        assert result['generated'] == row['answer'] and result['grade']['reference_exact']
        assert result['grade']['contract_valid'] and result['grade']['raw_exact']
    numeric, move = example_rows()
    value = json.loads(numeric['answer']); value['explanation'] = '额外字段'
    grade = evaluation.grade(numeric, json.dumps(value))
    assert grade['first_move_legal'] and not grade['answer_format_valid'] and not grade['contract_valid']
    assert not evaluation.grade(move, '推荐走法是' + move['answer'])['first_move_legal']


def test_removed_supervision_does_not_change_the_actual_messages():
    from xqgeneral.training import messages
    for row in example_rows():
        row.update(future_moves=['a0a9'], future_branches=[['a0a9']], source_reference={'controlled': True})
        for board_text in [None, 'fen', 'dictionary']:
            assert messages(row, board_text=board_text) == messages(evaluation.prompt_record(row), board_text=board_text)


@pytest.mark.parametrize('damage', ['invalid_json', 'short_pv', 'illegal_pv', 'duplicate_candidates',
                                  'wrong_perspective', 'boolean_score', 'wrong_move', 'long_pv'])
def test_numeric_diagnostics_separate_format_legality_history_and_score(damage):
    row = example_rows()[0]; value = json.loads(row['answer'])
    if damage == 'invalid_json': raw = 'raw invalid answer'
    else:
        if damage == 'short_pv': value['pv'] = value['pv'][:1]
        elif damage == 'illegal_pv': value['pv'][1] = 'a0a9'
        elif damage == 'duplicate_candidates': value['candidates'][1] = value['candidates'][0]
        elif damage == 'wrong_perspective': value['evaluation']['perspective'] = 'red'
        elif damage == 'boolean_score': value['evaluation']['value'] = True
        elif damage == 'wrong_move': value['move'] = 'a0a9'
        else: value['pv'].append(value['pv'][0])
        raw = json.dumps(value)
    result = evaluation.prediction_record(row, raw)
    assert result['generated'] == raw and not result['grade']['contract_valid']
    if damage == 'short_pv': assert result['grade']['pv_history_legal'] and not result['grade']['six_ply_or_terminal']


def test_summary_counts_all_answers_and_marks_reference_error_as_conditional():
    numeric, move = example_rows()
    records = [evaluation.prediction_record(row, row['answer']) for row in [numeric, move]]
    bad = dict(numeric, id='bad'); records.append(evaluation.prediction_record(bad, 'invalid'))
    result = evaluation.summary(records)
    assert result['examples'] == 3 and result['unique_history_contexts'] == 1
    assert result['by_stage']['engine_numeric']['contract_valid_rate'] == .5
    assert result['by_stage']['engine_numeric']['scored_evaluation_responses'] == 1
    assert result['by_stage']['engine_numeric']['mean_absolute_reference_expected_score_error_conditional'] == 0
    assert not result['reference_is_measured_human_strength'] and not result['raw_answers_repaired']


def test_prepare_reconstructs_all_original_views_without_opening_train_or_test(tmp_path, monkeypatch):
    args = corpus_control.fixture(tmp_path, validation_histories=2); corpus.prepare(**args)
    original = Path.open
    def protect(path, *more, **kwargs):
        if path.name in {'train.jsonl', 'test.jsonl'}: pytest.fail('Diagnostic selection must not open train/test answers')
        return original(path, *more, **kwargs)
    monkeypatch.setattr(Path, 'open', protect)
    result = evaluation.prepare(args['output'], tmp_path / 'selected')
    assert result['examples'] == 4 and result['unique_history_contexts'] == 2
    assert len(evaluation.checked_inputs(tmp_path / 'selected', BoundInputs())) == 4
    expected = b''.join(raw for _, raw, row in corpus._lines(args['output'] / 'validation.jsonl') if row['stage'] in evaluation.STAGES)
    assert (tmp_path / 'selected/validation.jsonl').read_bytes() == expected


@pytest.mark.parametrize('damage', ['question', 'answer', 'order', 'omitted'])
def test_self_rehashed_prepared_rows_cannot_replace_original_validation(tmp_path, damage):
    args = corpus_control.fixture(tmp_path, validation_histories=2); corpus.prepare(**args)
    output = tmp_path / 'selected'; evaluation.prepare(args['output'], output)
    path = output / 'validation.jsonl'; rows = load_jsonl(path)
    if damage == 'question': rows[0]['question'] += ' changed'
    elif damage == 'answer': rows[0]['answer'] = 'changed'
    elif damage == 'order': rows.reverse()
    else: rows.pop()
    write_jsonl(path, rows); proof_path = output / 'manifest.json'; proof = json.loads(proof_path.read_text())
    proof['outputs'][str(path)] = artifact(path); atomic_json(proof_path, proof)
    with pytest.raises(ValueError): evaluation.checked_inputs(output, BoundInputs())


def controlled_case(tmp_path, monkeypatch):
    expert = tmp_path / 'expert.pb'; expert.write_bytes(b'controlled expert identity; never loaded')
    def cache_chain(root, keys):
        paths = []
        for number, part in enumerate([keys[:40], keys[40:45], keys[45:]]):
            writer = FeatureStoreWriter(root / ('cache' + str(number)), [0, 3], digest(expert))
            if paths: writer.add_existing(paths[-1], paths[-1])
            writer.append(input_control.features(part)); writer.finish(); paths.append(writer.output / 'manifest.json')
        return paths
    monkeypatch.setattr(input_control, 'cache_chain', cache_chain)
    args, saved, _ = input_control.fixture(tmp_path, monkeypatch, validation_histories=2)
    model = Path(saved['model_path']); model.mkdir()
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors']: (model / name).write_text('{}')
    selected = tmp_path / 'selected'; evaluation.prepare(args['data'], selected)
    rows = load_jsonl(selected / 'validation.jsonl'); answers = {row['id']: row['answer'] for row in rows}
    def completed(source, path, bound):
        assert source == args['source'] and path == args['parent_completion']
        bound.bind(source); bound.bind(path)
        return saved, dict(verification=dict(trained_auxiliary_policy_rounds=0, trained_search_distillation_rounds=0))
    monkeypatch.setattr(evaluation, 'completed_parent', completed)
    class Producer:
        loads = 0
        calls = []
        fail_after = None
        short = False
        def __init__(self, *more, **kwargs): type(self).loads += 1; self.config = saved
        def generate_batch(self, batch, budget, memory):
            assert all('answer' not in row and 'future_moves' not in row and 'source_reference' not in row for row in batch)
            owner = type(self); owner.calls.append(([row['id'] for row in batch], budget, memory))
            if owner.fail_after == len(owner.calls): raise RuntimeError('controlled interrupted generation')
            return [] if owner.short else [answers[row['id']] for row in batch]
    monkeypatch.setattr(inference, 'Predictor', Producer)
    return dict(checkpoint=args['source'], completion=args['parent_completion'], features=args['features'],
        inputs=selected, output=tmp_path / 'diagnostics', expert_weights=expert, batch_size=2), Producer


def test_raw_model_protocol_strips_labels_binds_cache_and_resumes_without_generation(tmp_path, monkeypatch):
    args, producer = controlled_case(tmp_path, monkeypatch)
    result = evaluation.run_evaluation(**args)
    assert result['examples'] == 4 and result['unique_history_contexts'] == 2
    assert [call[1] for call in producer.calls] == [768, 64]
    assert not result['strong_chess_or_reliable_teaching_proven'] and not result['generation_end_reason_exposed']
    assert evaluation.run_evaluation(**args, resume=True) == result and producer.loads == 1
    assert len(producer.calls) == 2


def test_interrupted_batches_keep_original_raw_prefix_and_complete_on_resume(tmp_path, monkeypatch):
    args, producer = controlled_case(tmp_path, monkeypatch); producer.fail_after = 2; args['batch_size'] = 1
    with pytest.raises(RuntimeError): evaluation.run_evaluation(**args)
    partial = args['output'] / 'predictions.partial.jsonl'; before = partial.read_bytes()
    assert len(load_jsonl(partial)) == 1 and not (args['output'] / 'manifest.json').exists()
    producer.fail_after = None; result = evaluation.run_evaluation(**args, resume=True)
    assert result['examples'] == 4 and partial.read_bytes().startswith(before)
    assert len({r['id'] for r in load_jsonl(partial)}) == 4


def test_short_batch_is_a_failed_attempt_not_a_smaller_success(tmp_path, monkeypatch):
    args, producer = controlled_case(tmp_path, monkeypatch); producer.short = True
    with pytest.raises(ValueError): evaluation.run_evaluation(**args)
    assert not (args['output'] / 'manifest.json').exists()


def test_wrong_expert_is_rejected_before_predictor_load(tmp_path, monkeypatch):
    args, producer = controlled_case(tmp_path, monkeypatch); bad = tmp_path / 'wrong-expert'; bad.write_bytes(b'wrong')
    args['expert_weights'] = bad
    with pytest.raises(ValueError): evaluation.run_evaluation(**args)
    assert producer.loads == 0


def test_completed_self_rehashed_repair_cannot_disagree_with_original_journal(tmp_path, monkeypatch):
    args, _ = controlled_case(tmp_path, monkeypatch); evaluation.run_evaluation(**args)
    path = args['output'] / 'predictions.jsonl'; rows = load_jsonl(path); rows[0] = evaluation.prediction_record(rows[0]['record'], 'changed raw')
    write_jsonl(path, rows); proof_path = args['output'] / 'manifest.json'; proof = json.loads(proof_path.read_text())
    proof['outputs'][str(path)] = artifact(path); proof['verification'].update(evaluation.summary(rows)); atomic_json(proof_path, proof)
    with pytest.raises(ValueError): evaluation.run_evaluation(**args, resume=True)
