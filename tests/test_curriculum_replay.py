"""Controlled CPU SFT/QA integration; these scores do not measure a trained model."""
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_clean_sft_evaluation import artifact, setup_evaluation
from test_foundation_handoff import fixture_rows
from test_post_distillation_evaluation import completed_round
from xqgeneral import curriculum_replay as replay, evaluate_qa, inference
from xqgeneral.course_tasks import PAPER_TASKS, paper_records
from xqgeneral.evidence import atomic_json, history_key, load_jsonl, write_jsonl
from xqgeneral.rules import in_check, replay as native_replay


@pytest.fixture
def initial(tmp_path, monkeypatch):
    pipeline, model_config, runner = setup_evaluation(tmp_path / 'assets', monkeypatch)
    parent = json.loads((pipeline / 'manifest.json').read_text())
    whole = json.loads((Path(parent['config']['curriculum']) / 'manifest.json').read_text())
    config = dict(data=whole['config']['data_path'], task_profile='legacy', split='validation',
        stages=list(replay.STAGES), memories=list(replay.MEMORIES), per_task=3, seed=13,
        question_formats=3, batch_size=4, max_new_tokens=32, cuda_visible_devices='3')
    config_path = tmp_path / 'course-validation.json';atomic_json(config_path, config)
    return dict(pipeline=pipeline, model_config=model_config, config=config, config_path=config_path,
                original=runner.original, output=tmp_path / 'replay')


class ControlledQA:
    def __init__(self, original, monkeypatch):
        self.original = original;self.patch = monkeypatch;self.commands = [];self.loads = 0
        self.batches = [];self.failure = None;owner = self
        class Predictor:
            def __init__(self, checkpoint, **kwargs):
                owner.loads += 1
                self.config = json.loads((Path(checkpoint).parent / 'config.json').read_text())
                self.offset = 0
            def generate_batch(self, rows, max_new_tokens, memory):
                owner.batches.append((memory, [r['id'] for r in rows], max_new_tokens))
                result = []
                for i, row in enumerate(rows):
                    good = (memory == 'normal' and self.offset + i > 0 or
                            memory == 'shuffled' and (self.offset + i) % 2 == 0)
                    result.append(row['answer'] if good else 'unmodified invalid student answer')
                self.offset += len(rows)
                return result[:-1] if owner.failure == 'short_batch' else result
        monkeypatch.setattr(inference, 'Predictor', Predictor)
        monkeypatch.setattr(subprocess, 'run', self)

    def __call__(self, command, **kwargs):
        if command[0] in ['git', 'tmux']:return self.original(command, **kwargs)
        assert 'xqgeneral.evaluate_qa' in command
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '3'
        self.commands.append(command)
        module = command.index('xqgeneral.evaluate_qa')
        with self.patch.context() as patch:
            patch.setattr(sys, 'argv', ['evaluate_qa', *command[module + 1:]])
            evaluate_qa.main()
        dest = Path(command[command.index('--output') + 1])
        proof_path = dest / 'manifest.json';proof = json.loads(proof_path.read_text())
        if self.failure in ['false_grade', 'foreign_question']:
            path = dest / 'predictions.jsonl';rows = load_jsonl(path)
            if self.failure == 'false_grade':rows[0]['correct'] = not rows[0]['correct']
            else:rows[0]['question'] += ' changed'
            write_jsonl(path, rows);proof['outputs'][str(path)] = artifact(path)
        elif self.failure == 'foreign_memory':proof['config']['memory'] = 'foreign'
        elif self.failure == 'changed_input':
            path = Path(proof['config']['data']) / 'validation.jsonl';path.write_bytes(path.read_bytes() + b' ')
        atomic_json(proof_path, proof)
        return SimpleNamespace(returncode=0)


def execute(s, **kwargs):
    return replay.run_validation(s['pipeline'], s['model_config'], s['config_path'], s['output'], **kwargs)


def test_completed_initial_model_runs_all_four_courses_paired_raw_and_idempotent_resume(initial, monkeypatch):
    s = initial;controlled = ControlledQA(s['original'], monkeypatch)
    result = execute(s)
    assert result['examples_per_memory'] == 66 and result['task_types'] == 22
    assert result['trained_search_distillation_rounds'] == 0
    assert controlled.loads == len(controlled.commands) == 3
    assert result['raw_validation_by_memory']['normal']['accuracy'] == 65 / 66
    assert result['raw_validation_by_memory']['zero']['accuracy'] == 0
    assert result['distinct_game_ids'] == 1
    assert list(result['by_course']['normal']) == list(replay.STAGES)
    assert sum(v['examples'] for v in result['by_course']['normal'].values()) == 66
    comparison = result['original_foundation_comparison']
    assert comparison['same_ordered_questions_and_gold']
    assert sum(v['correct_to_wrong'] for v in comparison['paired_changes'].values()) == 1
    assert sum(v['wrong_to_correct'] for v in comparison['paired_changes'].values()) == 0
    identities = [[ids for memory, ids, _ in controlled.batches if memory == value] for value in replay.MEMORIES]
    assert identities[0] == identities[1] == identities[2]
    raw = load_jsonl(s['output'] / 'normal/predictions.jsonl')
    assert raw[0]['generated'] == 'unmodified invalid student answer' and raw[0]['correct'] is False
    assert result['raw_answers_repaired'] is result['independent_test_used_for_training_or_selection'] is False
    assert result['automatic_acceptance_threshold_applied'] is result['strong_play_or_reliable_coaching_proven'] is False
    original = (s['output'] / 'manifest.json').read_bytes()
    assert execute(s, resume=True) == result
    assert controlled.loads == 3 and (s['output'] / 'manifest.json').read_bytes() == original


@pytest.mark.parametrize('problem', ['test_split', 'one_memory', 'missing_course', 'bool_budget', 'singleton', 'wrong_profile'])
def test_invalid_validation_contract_never_loads_a_model(initial, monkeypatch, problem):
    s = initial;controlled = ControlledQA(s['original'], monkeypatch)
    if problem == 'test_split':s['config']['split'] = 'test'
    elif problem == 'one_memory':s['config']['memories'] = ['normal']
    elif problem == 'missing_course':s['config']['stages'].pop()
    elif problem == 'bool_budget':s['config']['per_task'] = True
    elif problem == 'singleton':s['config']['batch_size'] = 65
    else:s['config']['task_profile'] = 'paper_xiangqi_v1'
    atomic_json(s['config_path'], s['config'])
    with pytest.raises(ValueError):execute(s)
    assert not controlled.loads and not controlled.commands and not s['output'].exists()


def test_incomplete_model_cannot_start_course_validation(initial, monkeypatch):
    s = initial;controlled = ControlledQA(s['original'], monkeypatch)
    (s['pipeline'] / 'manifest.json').rename(s['pipeline'] / 'preserved-completion.json')
    with pytest.raises(ValueError, match='completed selected full-decoder'):execute(s)
    assert not controlled.loads and not s['output'].exists()


@pytest.mark.parametrize('problem', ['false_grade', 'foreign_question', 'foreign_memory', 'changed_input', 'short_batch'])
def test_failed_raw_stages_preserve_evidence_and_never_claim_completion(initial, monkeypatch, problem):
    s = initial;controlled = ControlledQA(s['original'], monkeypatch);controlled.failure = problem
    with pytest.raises(ValueError):execute(s)
    assert not (s['output'] / 'manifest.json').exists()
    assert json.loads((s['output'] / 'state.json').read_text())['preserve_partial_outputs']
    assert controlled.loads == 1


def test_changed_question_selection_does_not_claim_a_paired_forgetting_measure(initial, monkeypatch):
    s = initial;controlled = ControlledQA(s['original'], monkeypatch)
    s['config']['seed'] = 14;atomic_json(s['config_path'], s['config'])
    result = execute(s)
    assert result['original_foundation_comparison'] == dict(same_ordered_questions_and_gold=False, paired_changes=None)
    assert controlled.loads == 3


def test_completed_resume_regrades_raw_answers_and_refuses_missing_stages(initial, monkeypatch):
    s = initial;controlled = ControlledQA(s['original'], monkeypatch);execute(s)
    proof_path = s['output'] / 'normal/manifest.json';proof = json.loads(proof_path.read_text())
    raw_path = s['output'] / 'normal/predictions.jsonl';raw = load_jsonl(raw_path);raw[0]['correct'] = True
    write_jsonl(raw_path, raw);proof['outputs'][str(raw_path)] = artifact(raw_path);atomic_json(proof_path, proof)
    with pytest.raises(ValueError, match='recorded grades changed'):execute(s, resume=True)
    assert controlled.loads == 3
    proof_path.rename(s['output'] / 'normal/preserved-manifest.json')
    with pytest.raises(ValueError, match='Preserve incomplete'):execute(s, resume=True)
    assert controlled.loads == 3


def test_completed_distillation_round_uses_its_original_foundation_and_actual_selected_model(completed_round, tmp_path, monkeypatch):
    s = completed_round;controlled = ControlledQA(s['original'], monkeypatch)
    token = json.loads(s['token'].read_text());initial = json.loads(Path(token['config']['parent_completion']).read_text())
    whole = json.loads((Path(initial['config']['curriculum']) / 'manifest.json').read_text())
    config = dict(data=whole['config']['data_path'], task_profile='legacy', split='validation', stages=list(replay.STAGES),
        memories=list(replay.MEMORIES), per_task=3, seed=13, question_formats=3, batch_size=4,
        max_new_tokens=32, cuda_visible_devices='3')
    path = tmp_path / 'round-course-config.json';atomic_json(path, config)
    result = replay.run_validation(s['root'], s['config'], path, tmp_path / 'round-course-result')
    assert result['trained_search_distillation_rounds'] == 1
    assert result['selected_checkpoint'] == str(s['source'])
    assert result['original_foundation_comparison']['same_ordered_questions_and_gold']
    assert controlled.loads == 3 and result['examples_per_memory'] == 66


def test_all26_paper_tasks_keep_native_questions_raw_grades_and_format_errors(tmp_path):
    rows = []
    checked = '4k4/9/9/9/9/9/9/9/3Rr4/4K4 w - - 0 1'
    history = native_replay(checked, [])
    assert in_check(checked) and in_check(native_replay(checked, ['d1e1'])[-1])
    for i in range(3):
        root = dict(fixture_rows()[0], id='paper-root-' + str(i), future_moves=['d1e1'],
                    initial_fen=checked, fen=checked, moves=[], history=history, feature_key=history_key(history),
                    provenance='controlled native validation fixture')
        rows += [dict(row, id=f'controlled-paper/{i}/{j}')
                 for j, row in enumerate(paper_records(root, random.Random(i)))]
    data = tmp_path / 'paper';write_jsonl(data / 'validation.jsonl', rows)
    config = dict(data=str(data), task_profile='paper_xiangqi_v1', split='validation', stages=list(replay.STAGES),
        memories=list(replay.MEMORIES), per_task=3, seed=13, question_formats=3, batch_size=4,
        max_new_tokens=32, cuda_visible_devices='3')
    selected = replay.selected_course_rows(config)
    assert len(selected) == 78
    assert {(r['stage'], r['task_type']) for r in selected} == {(s, t) for s in replay.STAGES for t in PAPER_TASKS[s]}
    records = [evaluate_qa.prediction_record(row, row['answer']) for row in selected]
    replay.grade_records(records, selected)
    records[0] = evaluate_qa.prediction_record(selected[0], 'preserved malformed raw answer')
    courses = replay.by_course(records)
    assert sum(v['correct'] for v in courses.values()) == 77
    assert sum(v['answer_format_errors'] for v in courses.values()) == 1
    assert records[0]['generated'] == 'preserved malformed raw answer'
    removed = [r for r in rows if (r['stage'], r['task_type']) != ('dynamic_future', 'mate')]
    write_jsonl(data / 'validation.jsonl', removed)
    with pytest.raises(ValueError, match='full requested validation coverage'):replay.selected_course_rows(config)


@pytest.mark.parametrize('problem', ['incomplete_curriculum', 'test_used', 'foreign_data'])
def test_replay_requires_the_actual_completed_curriculum_validation_source(initial, monkeypatch, tmp_path, problem):
    s = initial;controlled = ControlledQA(s['original'], monkeypatch)
    parent_path = s['pipeline'] / 'manifest.json';parent = json.loads(parent_path.read_text())
    whole_path = Path(parent['config']['curriculum']) / 'manifest.json';whole = json.loads(whole_path.read_text())
    if problem == 'foreign_data':
        other = tmp_path / 'renamed-data';other.mkdir()
        (other / 'validation.jsonl').write_bytes((Path(s['config']['data']) / 'validation.jsonl').read_bytes())
        s['config']['data'] = str(other);atomic_json(s['config_path'], s['config'])
    else:
        if problem == 'incomplete_curriculum':whole['verification']['all_raw_qa_gates_passed'] = False
        else:whole['verification']['independent_test_used'] = True
        atomic_json(whole_path, whole);parent['inputs'][str(whole_path)] = artifact(whole_path);atomic_json(parent_path, parent)
    with pytest.raises(ValueError):execute(s)
    assert controlled.loads == 0 and not controlled.commands


@pytest.mark.parametrize('problem', ['summary', 'input_binding', 'raw_output_binding'])
def test_completed_report_rejects_rebound_summaries_and_omitted_artifact_bindings(initial, monkeypatch, problem):
    s = initial;controlled = ControlledQA(s['original'], monkeypatch);execute(s)
    path = s['output'] / 'manifest.json';proof = json.loads(path.read_text())
    if problem == 'summary':
        report = s['output'] / 'verification.json';value = json.loads(report.read_text());value['examples_per_memory'] = 1
        atomic_json(report, value);proof['verification'] = value;proof['outputs'][str(report)] = artifact(report)
    elif problem == 'input_binding':proof['inputs'].pop(str(s['output'] / 'normal/manifest.json'))
    else:proof['outputs'].pop(str(s['output'] / 'normal/predictions.jsonl'))
    atomic_json(path, proof)
    with pytest.raises(ValueError, match='actual raw report'):execute(s, resume=True)
    assert controlled.loads == 3
