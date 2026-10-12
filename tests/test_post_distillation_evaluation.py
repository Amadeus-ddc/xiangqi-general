"""Completed distilled-model evaluation uses controlled CPU producers, never real training."""
import json
from pathlib import Path

import pytest
import torch

import test_distillation_training as round_tests
import test_clean_sft_matches as match_tests
from test_clean_sft_evaluation import ControlledEvaluation, artifact, setup_evaluation
from xqgeneral import clean_sft_evaluation as evaluation, clean_sft_matches as matches, distillation_training
from xqgeneral.evidence import atomic_json, load_jsonl, write_jsonl
from xqgeneral.rules import START_FEN


@pytest.fixture
def completed_round(tmp_path, monkeypatch):
    original_setup = round_tests.setup_data
    def native_data(path):
        curriculum, recipe, token = original_setup(path)
        proof = json.loads(token.read_text())
        for split in ['train', 'validation']:
            file = Path(proof['config']['data']) / (split + '.jsonl')
            rows = load_jsonl(file)
            for row in rows:
                row.update(fen=START_FEN, initial_fen=START_FEN, moves=[], question='preserved native question')
            write_jsonl(file, rows);proof['inputs'][str(file)] = artifact(file)
        atomic_json(token, proof)
        return curriculum, recipe, token
    monkeypatch.setattr(round_tests, 'setup_data', native_data)
    source, template, token, runner, initial = round_tests.round_setup(tmp_path / 'training', monkeypatch)
    root = tmp_path / 'round'
    distillation_training.run_round(source, template, token, root)
    _, config, _ = setup_evaluation(tmp_path / 'evaluation-assets', monkeypatch, complete=False)
    values = json.loads(config.read_text());recipe = json.loads((root / 'recipe.json').read_text())
    values.update(sft_recipe=str(root / 'recipe.json'), token_preflight=str(token), data=recipe['data_path'],
                  features=recipe['feature_path'])
    atomic_json(config, values)
    return {'root': root, 'config': config, 'values': values, 'token': token, 'recipe': recipe,
            'source': root / 'sft/adapter.pt', 'initial': initial, 'original': runner.original,
            'output': tmp_path / 'capability'}


def execute(s):
    return evaluation.run_evaluation(s['root'], 'owned-round', s['config'], s['output'])


def test_completed_round_has_full_paired_raw_review_and_resume_without_new_queries(completed_round, monkeypatch):
    s = completed_round;controlled = ControlledEvaluation(s['original'], monkeypatch)
    result = execute(s)
    assert result['trained_search_distillation_rounds'] == 1
    assert result['actual_full_decoder_distillation_round_completed_before_evaluation'] is True
    assert result['actual_clean_initial_sft_completed_before_model_or_reviewer_load'] is False
    assert result['actual_selected_full_decoder_training_completed_before_model_or_reviewer_load'] is True
    assert result['all_validation_examples'] == 16 and result['split'] == 'validation'
    assert controlled.loads == 3 and len(controlled.commands) == 6
    path = s['output'] / 'manifest.json';original = path.read_bytes()
    for problem in ['kind', 'contract', 'summary']:
        changed = json.loads(original)
        if problem == 'kind':changed['kind'] = 'clean_initial_sft_capability_validation'
        elif problem == 'contract':changed['config']['pipeline'] = 'foreign-training'
        else:changed['verification']['trained_search_distillation_rounds'] = 2
        atomic_json(path, changed)
        with pytest.raises(ValueError, match='Completed paired evaluation'):
            evaluation.run_evaluation(s['root'], 'owned-round', s['config'], s['output'], resume=True)
        path.write_bytes(original)
    assert controlled.loads == 3 and len(controlled.commands) == 6
    assert all(v['parse_valid_rate'] == 0 and v['oracle_repairs'] == 0 and v['examples'] == 16
               for v in result['raw_validation_by_memory'].values())
    ids = [[ids for memory, ids, _ in controlled.batches if memory == value]
           for value in ['normal', 'zero', 'shuffled']]
    assert ids[0] == ids[1] == ids[2]
    assert result['blinded_prose_review']['valid_ratings'] == 15
    assert result['blinded_prose_review']['rejected'] == 1
    assert result['raw_answers_repaired'] is result['independent_test_used_for_training_or_selection'] is False
    assert result['strong_play_or_reliable_coaching_proven'] is False
    proof = json.loads((s['output'] / 'manifest.json').read_text())
    assert proof['kind'] == 'search_distillation_capability_validation'
    assert proof['inputs'][str(s['source'])] == artifact(s['source'])
    assert evaluation.run_evaluation(s['root'], 'owned-round', s['config'], s['output'], resume=True) == result
    assert controlled.loads == 3 and len(controlled.commands) == 6


@pytest.mark.parametrize('problem', ['unfinished_round', 'foreign_recipe', 'wrong_token', 'nonfinite_selected',
                                    'wrong_input_hashes', 'wrong_selected_code', 'partial_tensors'])
def test_incomplete_or_inconsistent_round_never_loads_student_or_reviewer(completed_round, monkeypatch, problem):
    s = completed_round;controlled = ControlledEvaluation(s['original'], monkeypatch)
    proof_path = s['root'] / 'manifest.json';proof = json.loads(proof_path.read_text())
    if problem == 'unfinished_round':proof['verification']['actual_distillation_sft_completed'] = False
    elif problem == 'foreign_recipe':proof['config']['compiled_recipe']['batch_size'] = 16
    elif problem == 'wrong_token':proof['config']['token_preflight_sha256'] = '0' * 64
    else:
        checkpoint = torch.load(s['source'], weights_only=True)
        if problem == 'nonfinite_selected':next(iter(checkpoint['trainable'].values())).view(-1)[0] = float('nan')
        elif problem == 'wrong_input_hashes':checkpoint['input_hashes'] = {}
        elif problem == 'wrong_selected_code':checkpoint['code'] = {'revision': 'foreign'}
        else:checkpoint['trainable'].pop(next(iter(checkpoint['trainable'])))
        torch.save(checkpoint, s['source'])
        proof['outputs'][str(s['source'])] = artifact(s['source'])
        trained_path = s['source'].parent / 'manifest.json';trained = json.loads(trained_path.read_text())
        trained['outputs'][str(s['source'])] = artifact(s['source']);atomic_json(trained_path, trained)
        proof['inputs'][str(trained_path)] = artifact(trained_path)
    atomic_json(proof_path, proof)
    with pytest.raises(ValueError):execute(s)
    assert controlled.loads == 0 and not controlled.commands and not (s['output'] / 'manifest.json').exists()


@pytest.mark.parametrize('problem', ['test_split', 'changed_reference', 'foreign_cache', 'test_used'])
def test_round_validation_retains_the_complete_protected_reference_contract(completed_round, monkeypatch, problem):
    s = completed_round;controlled = ControlledEvaluation(s['original'], monkeypatch)
    if problem == 'test_split':s['values']['split'] = 'test';atomic_json(s['config'], s['values'])
    elif problem == 'foreign_cache':s['values']['features'] = str(s['initial'] / 'foreign-cache');atomic_json(s['config'], s['values'])
    elif problem == 'changed_reference':
        path = Path(s['values']['data']) / 'validation.jsonl';path.write_bytes(path.read_bytes() + b'changed')
    else:
        token = json.loads(s['token'].read_text());token['verification']['independent_test_used_for_training_or_selection'] = True
        atomic_json(s['token'], token)
    with pytest.raises(ValueError):execute(s)
    assert not controlled.commands and controlled.loads == 0 and not s['output'].exists()


def test_actual_distilled_checkpoint_reaches_full_match_matrix_after_paired_validation(completed_round, tmp_path, monkeypatch):
    s = completed_round;controlled = ControlledEvaluation(s['original'], monkeypatch);execute(s)
    context = match_tests.setup.__wrapped__(tmp_path / 'match-assets', monkeypatch)
    monkeypatch.setattr(matches, 'checked_sft', evaluation.checked_sft)
    monkeypatch.setattr(matches, 'validation_spec', evaluation.validation_spec)
    context['config']['evaluation_config'] = str(s['config']);atomic_json(context['config_path'], context['config'])
    match_tests.execute_cpu(monkeypatch)
    result = matches.run_baseline(s['root'], s['output'], 'owned-evaluation', context['config_path'], context['root'])
    assert result['actual_distillation_round_completed_before_matches'] is True
    assert result['trained_search_distillation_rounds'] == 1
    assert result['actual_selected_checkpoint'] == str(s['source'])
    assert result['actual_selected_step'] == json.loads((s['root'] / 'manifest.json').read_text())['verification']['selected_step']
    assert result['games'] == 12 and result['node_budgets'] == [100, 1000, 10000]
    assert result['raw_match_metrics']['censored'] == 12 and result['raw_match_metrics']['draws'] == 0
    assert result['model_turns'] == 24 and result['full_explanation_contract_valid_turns'] == 0
    assert result['raw_model_answers_repaired'] is result['independent_final_test_used'] is False
    assert json.loads((context['root'] / 'manifest.json').read_text())['kind'] == 'search_distillation_full_match_baseline'
    monkeypatch.setattr(matches, 'execute_matches', lambda *args: pytest.fail('completed matches must not generate again'))
    assert matches.run_baseline(s['root'], s['output'], 'owned-evaluation', context['config_path'], context['root'], resume=True) == result
    path = context['root'] / 'manifest.json';proof = json.loads(path.read_text())
    proof['verification']['trained_search_distillation_rounds'] = 2;atomic_json(path, proof)
    with pytest.raises(ValueError, match='preserved raw report'):
        matches.run_baseline(s['root'], s['output'], 'owned-evaluation', context['config_path'], context['root'], resume=True)
