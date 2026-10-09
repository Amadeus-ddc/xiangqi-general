import json
import pytest

from xqgeneral.diagnose_moves import diagnose_predictions, main
from xqgeneral.evaluate_qa import normalized
from xqgeneral.evidence import history_key, write_jsonl
from xqgeneral.foundation_preflight import native_answer, native_context
from xqgeneral.rules import START_FEN, replay
from xqgeneral.selfplay_grounding import question_variant


def record(identifier='rook', *, future=False, source='a0'):
    history = replay(START_FEN, [])
    row = {'id': identifier, 'game_id': 'validation-game', 'split': 'validation',
           'stage': 'dynamic_future' if future else 'dynamic_current', 'task_type': 'moves',
           'initial_fen': START_FEN, 'moves': [], 'history': history, 'fen': history[-1],
           'feature_key': history_key(history), 'future_moves': ['h2e2'] if future else [],
           'query': {'source': source}, 'question': f'列出 {source} 上棋子的所有合法走法。'}
    row['answer'] = native_answer(row)
    return row


def prediction(row, generated=None, variant=1):
    generated = row['answer'] if generated is None else generated
    return {key: row[key] for key in ['id', 'game_id', 'stage', 'task_type']} | {
        'question': question_variant(row, variant), 'question_variant': variant,
        'expected': row['answer'], 'generated': generated,
        'correct': normalized(generated) == normalized(row['answer'])}


def inputs(tmp_path, rows, candidates):
    validation = tmp_path / 'validation.jsonl'
    write_jsonl(validation, rows)
    paths = []
    for index, values in enumerate(candidates):
        path = tmp_path / f'predictions-{index}.jsonl'
        write_jsonl(path, values)
        paths.append(path)
    return validation, paths


@pytest.mark.parametrize('case,category', [
    ('whitespace', 'raw_correct'), ('missing', 'missing_only'),
    ('extra', 'extra_only'), ('both', 'missing_and_extra'),
    ('duplicates', 'duplicates_only'), ('order', 'order_only'),
    ('order_duplicates', 'order_and_duplicates'), ('malformed', 'malformed_output'),
    ('none', 'missing_only'), ('wrong_source', 'extra_only'),
])
def test_native_diagnosis_distinguishes_real_errors_without_repairing_raw_answers(tmp_path, case, category):
    row = record()
    gold = row['answer'].split()
    assert len(gold) >= 2
    outputs = {'whitespace': '\n ' + '  '.join(gold) + '\t',
               'missing': ' '.join(gold[:-1]), 'extra': row['answer'] + ' a0b0',
               'both': ' '.join(gold[:-1] + ['a0b0']),
               'duplicates': row['answer'] + ' ' + gold[0],
               'order': ' '.join(reversed(gold)),
               'order_duplicates': ' '.join(list(reversed(gold)) + [gold[-1]]),
               'malformed': '建议走 ' + row['answer'], 'none': '无',
               'wrong_source': row['answer'] + ' i0i1'}
    raw = prediction(row, outputs[case])
    validation, paths = inputs(tmp_path, [row], [[raw]])
    output = tmp_path / 'diagnosis'
    report = diagnose_predictions(validation, paths, output)
    item = json.loads((output / 'item-diagnostics.jsonl').read_text())
    summary = report['candidates'][0]['summary']
    assert item['category'] == category
    assert item['generated'] == raw['generated'] and item['raw_correct'] == raw['correct']
    assert summary['raw_correct'] == (case == 'whitespace')
    if case in ['duplicates', 'order', 'order_duplicates']:
        assert summary['exact_move_set_answers'] == 1 and summary['raw_correct'] == 0
    if case == 'wrong_source':
        assert item['legal_wrong_source'] == 1 and item['illegal_extra_moves'] == 0
    if case == 'extra':
        assert item['illegal_extra_moves'] == 1 and item['legal_wrong_source'] == 0
    if case == 'malformed':
        assert summary['parseable_answers'] == 0
        assert summary['diagnostic_unique_move_precision'] is None
        assert summary['diagnostic_unique_move_recall'] is None
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['status'] == 'complete' and not proof['verification']['raw_predictions_repaired']
    assert set(proof['inputs']) == {str(validation), str(paths[0])}


def test_pairs_match_ids_despite_file_reordering_and_report_corrected_and_new_errors(tmp_path):
    rook, cannon = record(), record('cannon', source='b2')
    first = [prediction(rook), prediction(cannon, '无')]
    second = [prediction(cannon), prediction(rook, '无')]
    third = [prediction(rook), prediction(cannon)]
    validation, paths = inputs(tmp_path, [rook, cannon], [first, second, third])
    report = diagnose_predictions(validation, paths, tmp_path / 'diagnosis')
    assert len(report['paired_same_wording_diagnostics']) == 3
    first_pair = report['paired_same_wording_diagnostics'][0]
    assert first_pair['summary']['corrected'] == 1
    assert first_pair['summary']['newly_wrong'] == 1
    assert first_pair['summary']['net_correct_change'] == 0
    assert first_pair['groups']['piece_kind']['r']['net_correct_change'] == -1
    assert first_pair['groups']['piece_kind']['c']['net_correct_change'] == 1
    assert report['paired_same_wording_diagnostics'][-1]['summary']['net_correct_change'] == 1


def test_future_diagnosis_uses_actual_target_board_and_current_empty_move_list(tmp_path):
    future = record(future=True, source='b9')
    current = record('empty', source='b9')
    assert future['answer'] != '无' and current['answer'] == '无'
    validation, paths = inputs(tmp_path, [future], [[prediction(future)]])
    report = diagnose_predictions(validation, paths, tmp_path / 'future', stage='dynamic_future')
    assert report['native_future_sequences_replayed'] == 1
    assert report['candidates'][0]['groups']['piece_kind']['h']['raw_correct'] == 1
    validation, paths = inputs(tmp_path, [current], [[prediction(current)]])
    report = diagnose_predictions(validation, paths, tmp_path / 'current')
    assert report['candidates'][0]['groups']['gold_move_count_bucket']['0']['raw_correct'] == 1


@pytest.mark.parametrize('changed', ['ids', 'question', 'question_variant', 'expected'])
def test_pairing_rejects_changed_questions_and_gold(tmp_path, changed):
    row = record()
    raw = prediction(row)
    altered = dict(raw)
    if changed == 'ids':
        altered['id'] = 'different'
    elif changed == 'question':
        altered['question'] += ' changed'
    elif changed == 'question_variant':
        altered = prediction(row, variant=2)
    else:
        altered['expected'] = altered['generated'] = '无'
    validation, paths = inputs(tmp_path, [row], [[raw], [altered]])
    with pytest.raises(ValueError, match='identical move questions'):
        diagnose_predictions(validation, paths, tmp_path / 'diagnosis')
    assert not (tmp_path / 'diagnosis').exists()


@pytest.mark.parametrize('corruption,error', [
    ('history', 'full-history'), ('gold', 'native rules'), ('test', 'validation data only'),
    ('correct', 'strict raw exact match'), ('boolean', 'raw boolean'),
    ('duplicate_prediction', 'Duplicate move prediction'),
    ('duplicate_validation', 'Duplicate selected validation'), ('missing_validation', 'missing'),
])
def test_invalid_inputs_fail_before_creating_a_completed_output(tmp_path, corruption, error):
    row = record()
    raw = prediction(row)
    rows, candidates = [row], [[raw]]
    if corruption == 'history':
        row['feature_key'] = 'wrong-history'
    elif corruption == 'gold':
        row['answer'] = '无'
    elif corruption == 'test':
        row['split'] = 'test'
    elif corruption == 'correct':
        raw['correct'] = False
    elif corruption == 'boolean':
        raw['correct'] = 1
    elif corruption == 'duplicate_prediction':
        candidates[0].append(dict(raw))
    elif corruption == 'duplicate_validation':
        rows.append(dict(row))
    else:
        rows.clear()
    validation, paths = inputs(tmp_path, rows, candidates)
    with pytest.raises(ValueError, match=error):
        diagnose_predictions(validation, paths, tmp_path / 'diagnosis')
    assert not (tmp_path / 'diagnosis').exists()


def test_cli_filters_other_tasks_and_refuses_overwriting_or_duplicate_inputs(tmp_path, monkeypatch, capsys):
    row = record()
    other = dict(prediction(row), id='other', task_type='legal')
    validation, paths = inputs(tmp_path, [row], [[other, prediction(row)]])
    output = tmp_path / 'diagnosis'
    monkeypatch.setattr('sys.argv', ['diagnose_moves', '--validation', str(validation),
                                   '--predictions', str(paths[0]), '--output', str(output)])
    main()
    assert json.loads(capsys.readouterr().out)['native_move_lists'] == 1
    preserved = (output / 'analysis.json').read_bytes()
    with pytest.raises(FileExistsError, match='fresh'):
        diagnose_predictions(validation, paths, output)
    assert (output / 'analysis.json').read_bytes() == preserved
    with pytest.raises(ValueError, match='Duplicate prediction file'):
        diagnose_predictions(validation, [paths[0], paths[0]], tmp_path / 'duplicate')
    with pytest.raises(ValueError, match='at least one'):
        diagnose_predictions(validation, [], tmp_path / 'empty')


@pytest.mark.parametrize('field,value', [('game_id', 'other-game'), ('question', 'different request'),
                                       ('expected', '无')])
def test_even_a_single_candidate_must_match_the_native_validation_query(tmp_path, field, value):
    row = record()
    raw = prediction(row)
    raw[field] = value
    raw['correct'] = normalized(raw['generated']) == normalized(raw['expected'])
    validation, paths = inputs(tmp_path, [row], [[raw]])
    with pytest.raises(ValueError, match='native validation row'):
        diagnose_predictions(validation, paths, tmp_path / 'diagnosis')


def test_input_changed_during_native_readback_cannot_emit_a_completed_manifest(tmp_path, monkeypatch):
    row = record()
    validation, paths = inputs(tmp_path, [row], [[prediction(row)]])

    def changed_input(value):
        result = native_context(value)
        paths[0].write_bytes(paths[0].read_bytes() + b'\n')
        return result

    monkeypatch.setattr('xqgeneral.diagnose_moves.native_context', changed_input)
    with pytest.raises(ValueError, match='input changed'):
        diagnose_predictions(validation, paths, tmp_path / 'diagnosis')
    assert not (tmp_path / 'diagnosis').exists()
