"""Controlled native completion, paired raw outputs and match gates; no real GPU."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import test_auxiliary_policy_training as training_control
import test_clean_sft_evaluation as paired_control
import test_clean_sft_matches as match_control
from xqgeneral import auxiliary_policy_preflight as auxiliary
from xqgeneral import auxiliary_policy_training as training
from xqgeneral import clean_sft_evaluation as evaluation, clean_sft_matches as matches
from xqgeneral.evidence import atomic_json, digest, load_jsonl, write_jsonl


def artifact(path):
    return dict(sha256=digest(path), bytes=Path(path).stat().st_size)


def auxiliary_validation(config):
    """Use real raw files and hashes with controlled tokenizer completion metadata."""
    config = dict(config)
    recipe_path = Path(config['sft_recipe'])
    token_path = Path(config['token_preflight'])
    recipe = json.loads(recipe_path.read_text())
    token = json.loads(token_path.read_text())
    data = recipe_path.parent / 'auxiliary-corpus'
    data.mkdir()
    original = Path(config['data']) / 'validation.jsonl'
    rows = load_jsonl(original)
    extras = [dict(rows[0], id='auxiliary-' + stage, stage=stage)
              for stage in auxiliary.ALL_STAGES if stage != 'explanation']
    # Retain original row bytes and order, rather than reserializing them.
    extra = data / 'extra.jsonl'
    write_jsonl(extra, extras)
    (data / 'validation.jsonl').write_bytes(original.read_bytes() + extra.read_bytes())
    counts = {stage: len(rows) if stage == 'explanation' else 1 for stage in auxiliary.ALL_STAGES}
    recipe.update(data_path=str(data), require_clean_foundation_handoff=False, loss_normalization='token',
        validation_profile=auxiliary.VALIDATION_PROFILE, validation_rows_by_stage=counts)
    atomic_json(recipe_path, recipe)
    token.update(kind=auxiliary.KIND, config=dict(logical=dict(profile=auxiliary.PROFILE,
        data=str(data), features=config['features'], source=str(recipe_path.parent / 'controlled-parent.pt'))))
    token['inputs'][str(data / 'validation.jsonl')] = artifact(data / 'validation.jsonl')
    token['verification'].update(rows_by_split_and_stage=dict(validation=counts),
        all_actual_corpus_rows_tokenized=True, all_train_validation_supervision_fits_without_truncation=True,
        all_corpus_feature_keys_present=True, full_decoder_parent_completion_verified=True,
        original_parent_feature_keys_shards_and_expert_preserved=True,
        independent_test_rows_parsed_or_tokenized=0, independent_test_used_for_training_or_selection=False)
    atomic_json(token_path, token)
    return config, token, recipe, data


def validation_case(tmp_path, monkeypatch):
    pipeline, path, original = paired_control.setup_evaluation(tmp_path, monkeypatch)
    config, token, recipe, data = auxiliary_validation(json.loads(path.read_text()))
    atomic_json(path, config)
    return pipeline, path, original, config, token, recipe, data


def test_all_original_explanations_are_selected_without_reading_final_test(tmp_path, monkeypatch):
    _, _, _, config, token, _, data = validation_case(tmp_path, monkeypatch)
    protected = Path(config['data']) / 'test.jsonl'
    original = Path.open
    def no_test(path, *args, **kwargs):
        if path == protected:
            pytest.fail('Independent final-test answers must not be opened')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_test)
    rows, actual, _ = evaluation.validation_spec(config)
    assert len(rows) == 16 and actual == token
    assert all(row['stage'] == 'explanation' for row in rows)
    assert evaluation.capability_kind(token) == 'auxiliary_policy_capability_validation'
    assert len(load_jsonl(data / 'validation.jsonl')) == 22


@pytest.mark.parametrize('damage', ['omitted', 'reordered', 'changed_answer', 'false_parent', 'wrong_cache', 'test_access'])
def test_auxiliary_evaluation_rejects_changed_original_rows_or_token_contract(tmp_path, monkeypatch, damage):
    _, path, _, config, token, _, _ = validation_case(tmp_path, monkeypatch)
    original = Path(config['data']) / 'validation.jsonl'
    if damage in ['omitted', 'reordered', 'changed_answer']:
        rows = load_jsonl(original)
        if damage == 'omitted': rows.pop()
        elif damage == 'reordered': rows.reverse()
        else: rows[0]['answer'] += ' changed'
        write_jsonl(original, rows)
        # Update metadata so this specifically tests the source-row seam.
        token['inputs'][str(original)] = artifact(original)
    elif damage == 'false_parent': token['verification']['full_decoder_parent_completion_verified'] = False
    elif damage == 'wrong_cache': token['config']['logical']['features'] = 'other'
    else: token['verification']['independent_test_rows_parsed_or_tokenized'] = 1
    atomic_json(Path(config['token_preflight']), token)
    with pytest.raises(ValueError): evaluation.validation_spec(config)


def native_completed_auxiliary(tmp_path, monkeypatch):
    args, calls = training_control.setup(tmp_path, monkeypatch)
    native_gpu = training.run_preflight
    def bounded_gpu(source, recipe, token, output, resume=False):
        result = native_gpu(source, recipe, token, output, resume=resume)
        path = Path(output) / 'manifest.json'
        proof = json.loads(path.read_text())
        proof['config'] = dict(source=str(source), recipe=str(recipe), token_preflight=str(token))
        atomic_json(path, proof)
        return result
    monkeypatch.setattr(training, 'run_preflight', bounded_gpu)
    result = training.run_round(**args)
    root = args['output']
    config = dict(sft_recipe=str(root / 'recipe.json'), token_preflight=str(args['token_manifest']),
                  features=json.loads(args['token_manifest'].read_text())['config']['logical']['features'])
    recipe = json.loads((root / 'recipe.json').read_text())
    saved = auxiliary.checked_parent(args['source'], False)
    original = evaluation.completed_parent
    def parent(source, completion, *more, **kwargs):
        if Path(source) == args['source']:
            return saved, dict(verification=dict(trained_search_distillation_rounds=2))
        return original(source, completion, *more, **kwargs)
    monkeypatch.setattr(evaluation, 'completed_parent', parent)
    monkeypatch.setattr(evaluation, 'checked_parent', lambda source, clean: saved)
    return root, config, recipe, result


def test_native_completed_auxiliary_checkpoint_is_admitted_with_separate_round_counts(tmp_path, monkeypatch):
    root, config, recipe, result = native_completed_auxiliary(tmp_path, monkeypatch)
    checkpoint, proof = evaluation.checked_sft(root, config, recipe)
    assert checkpoint == root / 'sft/adapter.pt'
    assert proof['verification'] == result
    assert result['trained_search_distillation_rounds'] == 2
    assert result['trained_auxiliary_policy_rounds'] == 1


@pytest.mark.parametrize('damage', ['auxiliary_count', 'search_count', 'step_zero'])
def test_native_consumer_rejects_relabelled_rounds_and_unselected_updates(tmp_path, monkeypatch, damage):
    root, config, recipe, _ = native_completed_auxiliary(tmp_path, monkeypatch)
    path = root / 'manifest.json';proof = json.loads(path.read_text())
    key, value = dict(auxiliary_count=('trained_auxiliary_policy_rounds', 9),
        search_count=('trained_search_distillation_rounds', 3), step_zero=('selected_step', 0))[damage]
    proof['verification'][key] = value;atomic_json(path, proof)
    with pytest.raises(ValueError): evaluation.checked_sft(root, config, recipe)


def test_paired_raw_outputs_remain_auxiliary_and_completed_reopen_runs_no_inference(tmp_path, monkeypatch):
    pipeline, path, original, _, _, _, _ = validation_case(tmp_path, monkeypatch)
    source = pipeline / 'sft/adapter.pt';parent = json.loads((pipeline / 'manifest.json').read_text())
    parent['kind'] = evaluation.AUXILIARY_ROUND_KIND
    parent['verification'].update(trained_auxiliary_policy_rounds=1, trained_search_distillation_rounds=0)
    monkeypatch.setattr(evaluation, 'checked_sft', lambda *args: (source, parent))
    controlled = paired_control.ControlledEvaluation(original, monkeypatch)
    root = tmp_path / 'auxiliary-evaluation'
    result = evaluation.run_evaluation(pipeline, 'controlled-producer', path, root)
    assert controlled.loads == 3
    assert result['actual_full_decoder_auxiliary_round_completed_before_evaluation'] is True
    assert result['actual_full_decoder_distillation_round_completed_before_evaluation'] is False
    assert result['actual_clean_initial_sft_completed_before_model_or_reviewer_load'] is False
    assert result['trained_auxiliary_policy_rounds'] == 1 and result['trained_search_distillation_rounds'] == 0
    assert result['raw_answers_repaired'] is False and result['all_validation_examples'] == 16
    assert json.loads((root / 'manifest.json').read_text())['kind'] == 'auxiliary_policy_capability_validation'
    before = controlled.loads
    assert evaluation.run_evaluation(pipeline, 'controlled-producer', path, root, resume=True) == result
    assert controlled.loads == before


@pytest.mark.parametrize('damage', [None, 'round_count', 'wrong_stage'])
def test_auxiliary_full_matches_preserve_stage_and_counts(tmp_path, monkeypatch, damage):
    state = match_control.setup.__wrapped__(tmp_path, monkeypatch)
    token_path = Path(json.loads(Path(state['config']['evaluation_config']).read_text())['token_preflight'])
    atomic_json(token_path, dict(kind=auxiliary.KIND))
    state['parent'].update(kind=evaluation.AUXILIARY_ROUND_KIND)
    state['parent']['verification'].update(trained_auxiliary_policy_rounds=1, trained_search_distillation_rounds=2)
    capability = deepcopy(state['capability']);capability['kind'] = 'auxiliary_policy_capability_validation'
    capability['verification'].update(actual_clean_initial_sft_completed_before_model_or_reviewer_load=False,
        actual_full_decoder_auxiliary_round_completed_before_evaluation=True,
        trained_auxiliary_policy_rounds=9 if damage == 'round_count' else 1, trained_search_distillation_rounds=2)
    if damage == 'wrong_stage': state['parent']['kind'] = evaluation.ROUND_KIND
    atomic_json(state['validation'] / 'manifest.json', capability)
    match_control.execute_cpu(monkeypatch)
    if damage:
        with pytest.raises(ValueError): match_control.run(state)
        assert match_control.Student.calls == 0
    else:
        result = match_control.run(state)
        assert result['actual_auxiliary_round_completed_before_matches'] is True
        assert result['actual_distillation_round_completed_before_matches'] is False
        assert result['trained_auxiliary_policy_rounds'] == 1 and result['trained_search_distillation_rounds'] == 2
        assert result['games'] == 12 and result['raw_model_answers_repaired'] is False
        proof = json.loads((state['root'] / 'manifest.json').read_text())
        assert proof['kind'] == 'auxiliary_policy_full_match_baseline'
