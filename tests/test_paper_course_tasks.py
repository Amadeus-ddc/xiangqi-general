from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random

import pytest

from xqgeneral.course_tasks import (
    DIAGONALS, PAPER_PROFILE, PAPER_TASKS, grade_answer, native_tag, paper_answer,
    paper_question, paper_records, recipe_profile, task_groups, validate_query,
)
from xqgeneral.curriculum_data import STAGES
from xqgeneral.course_sampling import CHANGED, EMPTY, SAME, FrequencySampler, TRAIN_QUERIES
from xqgeneral.evaluate_qa import balanced_rows, prediction_record, qa_summary
from xqgeneral.evidence import atomic_json, history_key, manifest, write_jsonl
from xqgeneral.foundation_preflight import native_answer, native_context, validate_task
from xqgeneral.gated_curriculum import raw_qa_gate, read_gate, validate_recipe
from xqgeneral.rules import START_FEN, in_check, legal_moves, piece_map, play, replay, square_controllers
from xqgeneral.selfplay_grounding import question_variant
from xqgeneral.symmetry import mirrored_qa
from xqgeneral.training import compatible_resume


MATE = '4k4/3RR4/9/9/9/9/9/9/9/5K3 b - - 0 1'
STALEMATE = '4k4/3R5/5R3/9/9/9/9/9/9/5K3 b - - 0 1'
CHECK = '4k4/9/9/9/4r4/9/9/9/9/4K4 w - - 0 1'
BEFORE_MATE = '4k4/3R5/5R3/9/9/9/9/9/9/5K3 w - - 0 1'


def root(fen=START_FEN, moves=(), future=(), split='train', name='controlled-native-probe'):
    history = replay(fen, list(moves))
    return {'game_id': name, 'split': split, 'initial_fen': fen, 'moves': list(moves),
            'history': history, 'fen': history[-1], 'feature_key': history_key(history),
            'future_moves': list(future), 'provenance': 'constructed_test_probe;not_recorded_human_game'}


def row(task, query=None, fen=START_FEN, future=(), **extra):
    kind = 'static' if task in PAPER_TASKS['static_current'] else 'dynamic'
    return dict(root(fen, future=future), task_profile=PAPER_PROFILE, task_type=task,
                stage=kind + ('_future' if future else '_current'), query=query or {}, **extra)


def covering_rows(split='validation'):
    records = []
    for source in [root(future=['b0c2', 'b9c7'], split=split),
                   root(CHECK, future=['e0d0'], split=split, name='in-check-probe'),
                   root(BEFORE_MATE, future=['f7e7'], split=split, name='mate-in-future-probe')]:
        records.extend(paper_records(source, random.Random(7)))
    selected = {}
    for record in records:
        selected.setdefault((record['stage'], record['task_type']), record)
    return list(selected.values())


def test_declared_profile_covers_the_native_versions_of_all_thirteen_paper_tasks():
    assert len(PAPER_TASKS['static_current']) == 7
    assert len(PAPER_TASKS['dynamic_current']) == 6
    records = covering_rows()
    assert len(records) == 26
    assert {(r['stage'], r['task_type']) for r in records} == {
        (stage, task) for stage in STAGES for task in PAPER_TASKS[stage]}
    assert len({r['id'] for r in records}) == len(records)
    for record in records:
        validate_task(record, PAPER_PROFILE)
        assert native_context(record) == record['feature_key']
        assert native_answer(record) == record['answer']
        assert record['teacher']['neural_prose_generated'] is False
        assert record['split'] == 'validation'


@pytest.mark.parametrize('task,query,tag', [
    ('piece', {'square': 'b0'}, '红马'),
    ('piece', {'square': 'e5'}, '空'),
    ('file', {'file': 'a'}, '红车=1 红兵=1 黑车=1 黑卒=1'),
    ('rank', {'rank': 0}, '红车=2 红仕=2 红马=2 红相=2 红帅=1'),
    ('diagonal', {'start': 'a0', 'end': 'i8'}, '红车=1 黑炮=1 黑卒=1'),
    ('materials', {}, '红=47 黑=47'),
])
def test_native_static_labels_count_types_and_both_material_sides(task, query, tag):
    record = row(task, query)
    assert native_tag(record) == tag
    assert paper_answer(record).endswith('答案：' + tag)
    assert '答案：' in paper_question(record)


def test_all_counts_are_both_sides_and_detailed_material_uses_xiangqi_values():
    counts = native_tag(row('counts'))
    assert len(counts.split()) == 14
    assert '红兵=5' in counts and '黑卒=5' in counts
    answer = paper_answer(row('materials', rationale_variant=1))
    assert answer.count('=47') == 4  # two worked totals and the two final tagged values
    assert '2×9' in answer and '5×1' in answer
    assert len(DIAGONALS) == 32
    assert all(len(d) >= 2 for d in DIAGONALS)


@pytest.mark.parametrize('fen,mate,check', [(MATE, '是', True), (STALEMATE, '否', False), (CHECK, '否', True)])
def test_mate_is_current_position_checkmate_and_stalemate_is_a_different_loss(fen, mate, check):
    assert in_check(fen) is check
    record = row('mate', fen=fen)
    assert native_tag(record) == mate
    if fen == STALEMATE:
        assert not legal_moves(fen)
        assert '困毙' in paper_answer(record) and '仍判负' in paper_answer(record)
    if fen == MATE:
        assert native_tag(row('parries', fen=fen)) == '无'
    if fen == CHECK:
        assert native_tag(row('parries', fen=fen)) == ' '.join(sorted(legal_moves(fen)))


def test_future_move_query_tracks_the_initial_piece_and_allows_immobile_pieces():
    record = row('moves', {'source': 'b0'}, future=['b0c2', 'b9c7'])
    target = replay(START_FEN, record['future_moves'])[-1]
    assert native_tag(record) == ' '.join(sorted(m for m in legal_moves(target) if m[:2] == 'c2'))
    assert '最初位于 b0' in paper_question(record)
    with pytest.raises(ValueError, match='absent initial'):
        native_tag(dict(record, query={'source': 'c2'}))
    immobile = '4k4/9/9/9/4r4/9/9/9/9/H3K4 w - - 0 1'
    assert native_tag(row('moves', {'source': 'a0'}, fen=immobile)) == '无'
    queries = [paper_records(root(immobile), random.Random(seed)) for seed in range(10)]
    assert any(r['task_type'] == 'moves' and r['query']['source'] == 'a0' for records in queries for r in records)
    with pytest.raises(ValueError, match='captured piece'):
        native_tag(row('moves', {'source': 'b9'}, future=['b2b9', 'a9b9']))


def test_current_mate_and_mate_after_actual_future_are_allowed_but_history_endings_are_not():
    terminal = root(MATE)
    generated = paper_records(terminal, random.Random(2))
    assert {r['stage'] for r in generated} == {'static_current', 'dynamic_current'}
    assert native_context(dict(terminal, task_profile=PAPER_PROFILE,
                               recorded_source_kind='native_terminal_probe')) == terminal['feature_key']
    before = root(BEFORE_MATE, future=['f7e7'])
    generated = paper_records(before, random.Random(2))
    mate = next(r for r in generated if r['stage'] == 'dynamic_future' and r['task_type'] == 'mate')
    assert native_tag(mate) == '是'
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 2
    with pytest.raises(ValueError, match='history rules'):
        paper_records(root(moves=cycle), random.Random(2))
    with pytest.raises(ValueError, match='terminal prefix'):
        paper_records(root(moves=cycle + ['b0c2']), random.Random(2))
    with pytest.raises(ValueError, match='history rules'):
        paper_records(root(moves=cycle[:-1], future=cycle[-1:]), random.Random(2))


@pytest.mark.parametrize('task,query', [('file', {'file': 'ab'}), ('file', {'file': None}),
    ('rank', {'rank': True}), ('diagonal', {'start': 'a0', 'end': 'c2'}),
    ('piece', {'square': 'j0'}), ('controllers', {'square': 'a10'})])
def test_invalid_native_query_schema_fails_before_training(task, query):
    with pytest.raises(ValueError):
        validate_query(row(task, query))


def test_parries_require_actual_check_and_current_future_course_metadata_is_strict():
    with pytest.raises(ValueError, match='in-check'):
        paper_answer(row('parries'))
    with pytest.raises(ValueError, match='declared course'):
        paper_answer(dict(row('piece', {'square': 'a0'}), future_moves=['b0c2']))
    with pytest.raises(ValueError, match='declared course'):
        paper_answer(dict(row('piece', {'square': 'a0'}), stage='static_future'))
    with pytest.raises(ValueError, match='recipe task profile'):
        validate_task(row('piece', {'square': 'a0'}), 'legacy')


def test_color_mirror_regenerates_native_tags_and_preserves_game_and_initial_piece_identity():
    records = covering_rows()
    assert records == covering_rows()
    for record in records:
        mirrored = mirrored_qa(record)
        assert mirrored['split'] == record['split'] and mirrored['game_id'] == record['game_id']
        assert mirrored['augmentation_parent'] == record['id']
        assert mirrored['answer'] == native_answer(mirrored)
        restored = mirrored_qa(mirrored)
        assert restored['query'] == record['query']
        assert restored['answer'] == record['answer']
        for variant in (0, 1, 2):
            assert question_variant(mirrored, variant) == paper_question(mirrored, variant)


def test_native_pikafish_fixture_controllers_match_all_ninety_squares_in_each_probe():
    path = Path(__file__).parents[1] / 'evidence/paper-course-controller-fixtures-v1.json'
    fixture = json.loads(path.read_text())
    assert fixture['source'] == 'unmodified_pinned_Pikafish_Position_attackers_to'
    assert fixture['weights_loaded'] is False
    for position in fixture['positions']:
        fen = position['fen']
        assert len(position['controllers_by_square']) == 90
        board = piece_map(fen)
        for square, native in position['controllers_by_square'].items():
            attackers, defenders = square_controllers(fen, square)
            assert sorted(s for s, _ in [*attackers, *defenders]) == native
            assert sorted(s for s, _ in defenders) == sorted(s for s in native
                if square in board and board[s].isupper() == board[square].isupper())


def test_geometric_controllers_include_pinned_rook_and_empty_targets_have_no_defenders():
    fen = '4k4/9/4r4/9/9/9/9/9/4R4/4K4 w - - 0 1'
    assert 'e1d1' not in legal_moves(fen)
    attackers, defenders = square_controllers(fen, 'd1')
    assert ('e1', 'R') in attackers and defenders == ()
    assert 'e1:红车' in native_tag(row('controllers', {'square': 'd1'}, fen=fen))


@pytest.mark.parametrize('task,expected,generated', [
    ('moves', 'a0a1 b0c2', 'b0c2 a0a1'),
    ('locate', 'a0 i0', 'i0 a0'),
    ('counts', '红车=2 黑卒=5', '黑卒=5 红车=2'),
    ('materials', '红=47 黑=47', '黑=47 红=47'),
    ('controllers', '攻击=a0:红车,b0:红马;保护=无', '保护=无;攻击=b0:红马,a0:红车'),
])
def test_content_grading_accepts_reordered_tags_and_never_claims_to_grade_prose(task, expected, generated):
    scores = grade_answer('标签依据\n答案：' + expected, '另一种解释\n答案：' + generated, task)
    assert scores == {'correct': True, 'answer_format_valid': True, 'canonical_tag_correct': False,
                      'raw_exact_correct': False, 'prose_graded': False}


@pytest.mark.parametrize('task,expected,generated', [
    ('moves', '无', ''), ('moves', '无', '无 无'), ('moves', 'a0a1', 'a0a1 a0a1'),
    ('moves', 'a0a1', '建议走a0a1'), ('locate', 'a0', 'a10'),
    ('counts', '红车=2', '红车=2 红车=2'), ('counts', '无', '红车=0'),
    ('materials', '红=47 黑=47', '红=47'),
    ('controllers', '攻击=无;保护=无', '攻击=无;攻击=无'),
    ('piece', '红马', '红马\n答案：红马'), ('mate', '否', '否。'),
])
def test_malformed_duplicate_missing_or_extra_tag_answers_remain_errors(task, expected, generated):
    assert not grade_answer('答案：' + expected, '答案：' + generated, task)['correct']
    assert not grade_answer('答案：' + expected, generated.replace('答案：', ''), task)['correct']


def test_gold_is_not_repaired_and_raw_predicted_text_is_preserved():
    with pytest.raises(ValueError):
        grade_answer('malformed stored supervision', '答案：无', 'moves')
    source = covering_rows()[0]
    answer = '这段解释未经事实评分。\n答案：' + native_tag(source)
    measured = prediction_record(source, answer)
    assert measured['generated'] == answer and measured['expected'] == source['answer']
    assert measured['correct'] and not measured['prose_graded'] and not measured['raw_exact_correct']


def test_new_gate_requires_all_twenty_six_groups_recomputed_tag_metrics_and_explicit_profile():
    rows = covering_rows()
    records = [prediction_record(r, '不同措辞\n答案：' + native_tag(r)) for r in rows]
    targets = {stage: {'accuracy': 1, 'minimum_task_accuracy': 1} for stage in STAGES}
    gate = raw_qa_gate(records, STAGES, targets, 1, PAPER_PROFILE)
    assert gate['passed'] and gate['accuracy'] == 1 and gate['raw_exact_correct_rate'] == 0
    assert gate['prose_graded'] is False
    with pytest.raises(ValueError, match='every introduced task'):
        raw_qa_gate([r for r in records if r['task_type'] != 'parries'], STAGES, targets, 1, PAPER_PROFILE)
    with pytest.raises(ValueError, match='task profile'):
        raw_qa_gate(records, STAGES, targets, 1)
    corrupted = deepcopy(records)
    corrupted[0]['raw_exact_correct'] = True
    with pytest.raises(ValueError, match='actual generated'):
        raw_qa_gate(corrupted, STAGES, targets, 1, PAPER_PROFILE)
    with pytest.raises(ValueError, match='mix task profiles'):
        qa_summary([records[0], dict(records[1], task_profile='legacy')])


def test_artifact_gate_recomputes_selected_questions_and_refuses_tag_metric_tampering(tmp_path):
    data, output = tmp_path / 'data', tmp_path / 'qa'
    checkpoint, features = tmp_path / 'checkpoint.pt', tmp_path / 'features.pt'
    checkpoint.write_bytes(b'controlled-checkpoint-placeholder')
    features.write_bytes(b'controlled-cache-placeholder')
    rows = covering_rows()
    write_jsonl(data / 'validation.jsonl', rows)
    selected = balanced_rows(rows, 1, 7, STAGES)
    records = [prediction_record(dict(r, question=question_variant(r, 0), question_variant=0),
                                '不同措辞\n答案：' + native_tag(r)) for r in selected]
    write_jsonl(output / 'predictions.jsonl', records)
    metrics = dict(qa_summary(records), raw_generation=True, oracle_used=False)
    config = {'checkpoint': str(checkpoint), 'split': 'validation', 'memory': 'normal',
              'data': str(data), 'features': str(features), 'stages': list(STAGES),
              'per_task': 1, 'question_formats': 3, 'seed': 7, 'task_profile': PAPER_PROFILE}
    gates = {'per_task': 1, 'question_formats': 3, 'seed': 7, 'task_profile': PAPER_PROFILE,
             'targets': {stage: {'accuracy': 1, 'minimum_task_accuracy': 1} for stage in STAGES}}

    def save():
        atomic_json(output / 'metrics.json', metrics)
        atomic_json(output / 'manifest.json', manifest('balanced_board_qa', config,
            [checkpoint, features, data / 'validation.jsonl'],
            [output / 'predictions.jsonl', output / 'metrics.json'], metrics))

    save()
    assert read_gate(output, checkpoint, STAGES, gates, str(data), str(features))['passed']
    metrics['raw_exact_correct_rate'] = 1
    save()
    with pytest.raises(ValueError, match='tag/prose metrics'):
        read_gate(output, checkpoint, STAGES, gates, str(data), str(features))


def test_recipe_and_resume_cannot_silently_switch_the_task_contract():
    path = Path(__file__).parents[1] / 'configs/foundation-human-engine-clean-v3.json'
    recipe = json.loads(path.read_text())
    assert recipe_profile(recipe) == 'legacy'
    new = dict(recipe, task_profile=PAPER_PROFILE,
               raw_qa_gates=dict(recipe['raw_qa_gates'], task_profile=PAPER_PROFILE))
    validate_recipe(new)
    with pytest.raises(ValueError, match='same task profile'):
        validate_recipe(dict(new, raw_qa_gates=recipe['raw_qa_gates']))
    with pytest.raises(ValueError, match='task profile differs'):
        compatible_resume({}, {'task_profile': PAPER_PROFILE})
    with pytest.raises(ValueError, match='Unknown foundation'):
        task_groups('guessed_profile')


def test_supplied_paper_positions_are_validated_before_native_move_queries():
    with pytest.raises(ValueError, match='both generals'):
        paper_answer(row('mate', fen='9/9/9/9/9/9/9/9/9/4K4 w - - 0 1'))
    with pytest.raises(ValueError, match='complete native'):
        paper_answer(row('mate', fen='9/9/9/9/9/9/9/9/9/4K4'))


def test_sft_keeps_the_declared_foundation_profile_and_accepts_the_legacy_default_alias(tmp_path):
    from xqgeneral.sft import prepare_config
    data = tmp_path / 'explanations'
    write_jsonl(data / 'train.jsonl', [{'stage': 'explanation'}])
    recipe = {'data_path': str(data), 'stages': ['explanation'], 'mixture': {'explanation': 1},
              'epochs': 1, 'max_steps': 1, 'min_steps': 1, 'batch_size': 1}
    saved = {'mode': 'bridge', 'task_profile': PAPER_PROFILE}
    assert prepare_config(saved, recipe, 'parent.pt', 'output')['task_profile'] == PAPER_PROFILE
    with pytest.raises(ValueError, match='foundation task profile'):
        prepare_config(saved, dict(recipe, task_profile='legacy'), 'parent.pt', 'output')
    assert prepare_config({'mode': 'bridge'}, dict(recipe, task_profile='legacy'),
                          'parent.pt', 'output')['task_profile'] == 'legacy'


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
def test_author_queries_are_distinct_and_heldout_uses_one_question_per_task(split):
    sampler = FrequencySampler()
    source = root(split=split)
    groups = sampler.query_groups(source, source['fen'], random.Random(7), False)
    for task, requested in TRAIN_QUERIES.items():
        if task == 'parries':  # The initial board is not in check.
            continue
        queries = [q for group in groups for t, q in group if t == task]
        assert len(queries) == (requested if split == 'train' else 1)
        assert len({json.dumps(q, sort_keys=True) for q in queries}) == len(queries)


def test_native_answer_classes_use_final_board_and_counters_are_split_and_task_local():
    sampler = FrequencySampler()
    examples = [row('piece', {'square': 'b0'}, future=['b0c2', 'b9c7']),
                row('locate', {'symbol': 'H'}, future=['b0c2', 'b9c7']),
                row('moves', {'source': 'b0'}, future=['b0c2', 'b9c7']),
                row('controllers', {'square': 'b0'}, future=['b0c2', 'b9c7'])]
    sampler.observe(examples)
    counts = sampler.verification()['answer_class_frequency_by_split_stage_task']
    assert counts['train/static_future/piece'] == {CHANGED: 1, EMPTY: 1, 'square:b0': 1}
    assert counts['train/static_future/locate'] == {'square:c2': 1, 'square:h0': 1}
    assert counts['train/dynamic_future/moves'] == {'piece:H': 1}
    assert counts['train/dynamic_future/controllers'] == {EMPTY: 1}
    assert not any(k.startswith(('validation/', 'test/')) or '/static_current/' in k for k in counts)


def test_observed_training_answers_cannot_change_heldout_question_sampling():
    sampler = FrequencySampler()
    heldout = root(split='validation')
    before = sampler.query_groups(heldout, heldout['fen'], random.Random(17), False)
    sampler.observe([row('piece', {'square': 'a3'}), row('moves', {'source': 'a3'})] * 1000)
    after = sampler.query_groups(heldout, heldout['fen'], random.Random(17), False)
    assert before == after


def test_cumulative_frequency_downweights_common_species_and_future_unchanged_answers():
    source = root(future=['b0c2', 'b9c7'])
    target = replay(source['fen'], source['future_moves'])[-1]
    board = piece_map(source['fen'])
    sampler = FrequencySampler()
    sampler.frequencies[('train', 'static_current', 'piece')]['piece:P'] = 100000
    sampler.frequencies[('train', 'dynamic_current', 'moves')]['piece:P'] = 100000
    sampler.frequencies[('train', 'static_future', 'piece')][SAME] = 100000
    pawns, changed = Counter(), 0
    for seed in range(100):
        static, dynamic = sampler.query_groups(source, source['fen'], random.Random(seed), False)
        for task, query in static + dynamic:
            if task in ('piece', 'moves'):
                square = query.get('square', query.get('source'))
                pawns[task] += board.get(square) == 'P'
        static, _ = sampler.query_groups(source, target, random.Random(seed), True)
        changed += sum(board.get(q['square']) != piece_map(target).get(q['square'])
                       for task, q in static if task == 'piece')
    assert pawns['piece'] < 3 and pawns['moves'] < 3
    assert changed > 350  # Four changed squares are available, queried without replacement.


def test_terminal_move_queries_are_capped_without_dropping_immobile_general():
    sampler = FrequencySampler()
    records = paper_records(root(MATE), random.Random(7), sampler)
    moves = [r for r in records if r['task_type'] == 'moves']
    assert len(moves) == 1 and moves[0]['query'] == {'source': 'e9'}
    assert native_tag(moves[0]) == '无'
    sampler.observe(records)
    assert sampler.verification()['answer_class_frequency_by_split_stage_task'][
        'train/dynamic_current/moves'] == {'piece:k': 1}


def test_sampling_verification_does_not_expose_mutable_training_policy():
    proof = FrequencySampler().verification()
    proof['training_queries_requested_per_root_by_task']['piece'] = 90
    assert TRAIN_QUERIES['piece'] == 4
