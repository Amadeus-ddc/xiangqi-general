from copy import deepcopy

import pytest

from xqgeneral.curriculum_data import STAGES
from xqgeneral.evaluate_qa import balanced_rows, qa_summary
from xqgeneral.evidence import atomic_json, manifest, write_jsonl
from xqgeneral.gated_curriculum import TASKS, raw_qa_gate, read_gate, training_config, validate_recipe
from xqgeneral.selfplay_grounding import question_variant


def recipe():
    return {'mode': 'bridge', 'decoder_training': 'frozen', 'batch_size': 256,
            'micro_batch_size': 4, 'ddp_world_size': 2, 'output': 'run',
            'course_mixtures': [{'static_current': 1},
                {'static_current': .1, 'dynamic_current': .9},
                {'static_current': .06, 'dynamic_current': .08, 'static_future': .86},
                {'static_current': .06, 'dynamic_current': .06, 'static_future': .08, 'dynamic_future': .8}],
            'course_budgets': [dict(steps=n, minimum_steps=500, qa_every=500, learning_rate=rate)
                for n, rate in [(50000, 1e-4), (60000, 8e-5), (30000, 7.5e-5), (100000, 2e-4)]],
            'raw_qa_gates': {'per_task': 10, 'question_formats': 3, 'seed': 47,
                'targets': {stage: {'accuracy': .9, 'minimum_task_accuracy': .9} for stage in STAGES}}}


def predictions(stages):
    return [{'id': f'{stage}/{task}/{i}', 'stage': stage, 'task_type': task,
             'expected': '无', 'generated': '无', 'correct': True}
            for stage in stages for task in TASKS[stage] for i in range(10)]


def test_capture_failure_cannot_be_hidden_by_other_correct_answers():
    config = recipe()
    records = predictions(STAGES[:2])
    for r in records:
        if r['stage'] == 'dynamic_current' and r['task_type'] == 'captures':
            r.update(generated='a0a1', correct=False)
    result = raw_qa_gate(records, STAGES[:2], config['raw_qa_gates']['targets'], 10)
    assert result['accuracy'] > .9
    assert result['courses']['static_current']['passed']
    assert not result['passed']
    assert result['by_task']['dynamic_current/captures']['accuracy'] == 0
    missing = [r for r in records if r['task_type'] != 'captures']
    with pytest.raises(ValueError, match='every introduced task'):
        raw_qa_gate(missing, STAGES[:2], config['raw_qa_gates']['targets'], 10)


def test_prior_course_must_remain_mastered_and_raw_errors_stay_errors():
    config, records = recipe(), predictions(STAGES[:3])
    for r in records:
        if r['stage'] == 'static_current':
            r.update(generated='建议回答无', correct=False)
    result = raw_qa_gate(records, STAGES[:3], config['raw_qa_gates']['targets'], 10)
    assert result['courses']['static_future']['passed'] and not result['passed']
    records[0]['correct'] = True
    with pytest.raises(ValueError, match='actual generated answer'):
        raw_qa_gate(records, STAGES[:3], config['raw_qa_gates']['targets'], 10)


def test_course_budget_and_inheritance_do_not_use_nll_to_advance(tmp_path):
    config = recipe()
    validate_recipe(config)
    stage = training_config(config, 1, tmp_path, 'raw-selected.pt')
    assert stage['batch_size'] == 256 and stage['micro_batch_size'] == 4
    assert stage['steps'] == stage['min_steps'] == 60000
    assert stage['learning_rate'] == 8e-5 and stage['token_learning_rate'] == pytest.approx(8e-6)
    assert stage['init_from'] == 'raw-selected.pt'
    assert stage['validation_stages'] == list(STAGES[:2])
    assert 'raw_qa_gates' not in stage and 'course_budgets' not in stage
    bad = deepcopy(config)
    bad['course_mixtures'][0] = {'static_current': .9, 'dynamic_future': .1}
    with pytest.raises(ValueError, match='introduced courses'):
        validate_recipe(bad)


def test_balanced_stage_selection_excludes_unintroduced_courses():
    selected = balanced_rows(predictions(STAGES), 3, 11, STAGES[:1])
    assert len(selected) == len(TASKS['static_current']) * 3
    assert {r['stage'] for r in selected} == {'static_current'}


@pytest.mark.parametrize('alteration', ['test', 'zero', 'changed_prediction'])
def test_gate_readback_rejects_test_ablation_and_changed_raw_evidence(tmp_path, alteration):
    config, records = recipe(), predictions(STAGES[:1])
    output, checkpoint, inputs = tmp_path / 'qa', tmp_path / 'adapter.pt', tmp_path / 'validation.jsonl'
    checkpoint.write_bytes(b'actual-checkpoint-fixture')
    inputs.write_bytes(b'actual-data-fixture')
    write_jsonl(output / 'predictions.jsonl', records)
    metrics = dict(qa_summary(records), raw_generation=True, oracle_used=False)
    atomic_json(output / 'metrics.json', metrics)
    arguments = {'checkpoint': str(checkpoint), 'split': 'validation', 'memory': 'normal',
                 'stages': list(STAGES[:1]), 'per_task': 10, 'question_formats': 3, 'seed': 47}
    proof = manifest('balanced_board_qa', arguments, [checkpoint, inputs],
                     [output / 'predictions.jsonl', output / 'metrics.json'], metrics)
    atomic_json(output / 'manifest.json', proof)
    assert read_gate(output, checkpoint, STAGES[:1], config['raw_qa_gates'])['passed']
    if alteration == 'changed_prediction':
        with (output / 'predictions.jsonl').open('a') as handle:
            handle.write('{}\n')
    else:
        arguments['split' if alteration == 'test' else 'memory'] = alteration
        proof['config'] = arguments
        atomic_json(output / 'manifest.json', proof)
    with pytest.raises(ValueError):
        read_gate(output, checkpoint, STAGES[:1], config['raw_qa_gates'])


def test_readback_checks_actual_sample_questions_and_expected_targets(tmp_path):
    config, rows = recipe(), []
    for prediction in predictions(STAGES[:1]):
        rows.append(dict(prediction, game_id='validation-game', answer=prediction['expected'],
                         question='原始提问', future_moves=[], query={
                             'square': 'a0', 'symbol': 'R', 'red': True, 'rank': 0}))
    data, output = tmp_path / 'data', tmp_path / 'qa'
    checkpoint, features = tmp_path / 'adapter.pt', tmp_path / 'features.pt'
    checkpoint.write_bytes(b'checkpoint-fixture')
    features.write_bytes(b'feature-fixture')
    write_jsonl(data / 'validation.jsonl', rows)
    selected = balanced_rows(rows, 10, 47, STAGES[:1])
    counts, records = {}, []
    for row in selected:
        task = row['task_type']
        variant = counts.get(task, 0) % 3
        counts[task] = counts.get(task, 0) + 1
        records.append(dict(row, question=question_variant(row, variant), question_variant=variant))
    arguments = {'checkpoint': str(checkpoint), 'split': 'validation', 'memory': 'normal',
                 'stages': list(STAGES[:1]), 'per_task': 10, 'question_formats': 3, 'seed': 47,
                 'data': str(data), 'features': str(features)}

    def save():
        write_jsonl(output / 'predictions.jsonl', records)
        metrics = dict(qa_summary(records), raw_generation=True, oracle_used=False)
        atomic_json(output / 'metrics.json', metrics)
        atomic_json(output / 'manifest.json', manifest('balanced_board_qa', arguments,
            [checkpoint, features, data / 'validation.jsonl'],
            [output / 'predictions.jsonl', output / 'metrics.json'], metrics))

    save()
    assert read_gate(output, checkpoint, STAGES[:1], config['raw_qa_gates'], str(data), str(features))['passed']
    records[0].update(expected='different-target', generated='different-target')
    save()
    with pytest.raises(ValueError, match='expected answers changed'):
        read_gate(output, checkpoint, STAGES[:1], config['raw_qa_gates'], str(data), str(features))
