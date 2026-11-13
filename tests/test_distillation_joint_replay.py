"""Native replay checks and real finite indexes; GPU/model results are controlled."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random

import pytest

from test_distillation_preflight import cpu_flow
from test_distillation_training import budget_token, round_setup, template
from test_feature_store import block
from test_paper_curriculum import source_fixture
from xqgeneral import distillation_preflight, distillation_training
from xqgeneral.course_tasks import paper_records
from xqgeneral.curriculum_data import STAGES
from xqgeneral.evidence import atomic_json, digest, iter_jsonl, manifest, write_jsonl
from xqgeneral.feature_store import FeatureStoreWriter


def replay_contract(paths, train=128, validation=32):
    return {'data_paths': paths, 'by_split_and_stage': {
        'train': {stage: train for stage in STAGES},
        'validation': {stage: validation for stage in STAGES}},
        'all_raw_train_validation_answers_natively_regraded': True,
        'all_features_present': True, 'combined_split_isolation_verified': True,
        'train_validation_rows_tokenized': 4 * (train + validation), 'independent_test_rows_tokenized': 0}


def joint_budget(train=1000, old=128):
    recipe, token = template(), budget_token(train)
    recipe.update(stages=['explanation', *STAGES], mixture={'explanation': .8, **dict.fromkeys(STAGES, .05)},
                  loss_normalization='example', replay_data_paths=['actual-course-replay'], max_steps=100)
    token['config']['replay_data_paths'] = recipe['replay_data_paths']
    token['verification']['course_replay'] = replay_contract(recipe['replay_data_paths'], train=old)
    return recipe, token


def test_joint_budget_uses_rounded_finite_quotas_and_retains_lecture_validation(tmp_path):
    recipe, token = joint_budget()
    compiled, budget = distillation_training.compile_recipe(recipe, token, tmp_path)
    assert budget['finite_mixture']['selected_rows_by_stage'] == {'explanation': 1000, **dict.fromkeys(STAGES, 62)}
    assert budget['rows_per_complete_pass'] == 1248
    assert budget['updates_per_complete_pass'] == 39
    assert compiled['epochs'] == 3 and compiled['min_steps'] == 39
    assert compiled['validation_examples'] == 384 and compiled['validation_stages'] == ['explanation']
    assert compiled['replay_data_paths'] == ['actual-course-replay']
    assert not budget['paper_first_round_150000_label_scale_reached']
    assert not budget['course_replay_counts_toward_paper_label_scale']


@pytest.mark.parametrize('problem', [
    'unaudited', 'wrong_paths', 'duplicate_paths', 'missing_course', 'token_loss', 'missing_mixture',
    'insufficient_old_rows', 'false_native', 'false_isolation', 'missing_features', 'test_tokens',
    'wrong_token_count', 'unknown_stage', 'boolean_count', 'nonlecture_anchor', 'zero_selected_course', 'cap'])
def test_unverified_or_unrealizable_replay_is_rejected_before_creating_a_job(tmp_path, problem):
    recipe, token = joint_budget()
    replay = token['verification']['course_replay']
    if problem == 'unaudited': token['verification'].pop('course_replay')
    elif problem == 'wrong_paths': recipe['replay_data_paths'] = ['other-source']
    elif problem == 'duplicate_paths':
        recipe['replay_data_paths'] = ['actual-course-replay', './actual-course-replay']
        token['config']['replay_data_paths'] = recipe['replay_data_paths']
        replay['data_paths'] = recipe['replay_data_paths']
    elif problem == 'missing_course': recipe['stages'].pop()
    elif problem == 'token_loss': recipe['loss_normalization'] = 'token'
    elif problem == 'missing_mixture': recipe['mixture'].pop(STAGES[-1])
    elif problem == 'insufficient_old_rows':
        token['verification']['course_replay'] = replay_contract(recipe['replay_data_paths'], train=61)
    elif problem == 'false_native': replay['all_raw_train_validation_answers_natively_regraded'] = False
    elif problem == 'false_isolation': replay['combined_split_isolation_verified'] = False
    elif problem == 'missing_features': replay['all_features_present'] = False
    elif problem == 'test_tokens': replay['independent_test_rows_tokenized'] = 1
    elif problem == 'wrong_token_count': replay['train_validation_rows_tokenized'] += 1
    elif problem == 'unknown_stage': replay['by_split_and_stage']['train']['foreign'] = 1
    elif problem == 'boolean_count': replay['by_split_and_stage']['train'][STAGES[0]] = True
    elif problem == 'nonlecture_anchor': recipe['mixture'][STAGES[0]] = 1
    elif problem == 'zero_selected_course': recipe['mixture'][STAGES[0]] = 1e-8
    else: recipe['max_steps'] = 38
    output = tmp_path / 'unused'
    with pytest.raises(ValueError):
        distillation_training.compile_recipe(recipe, token, output)
    assert not output.exists()


def test_old_course_volume_cannot_be_reported_as_new_paper_distillation_labels(tmp_path):
    recipe, token = joint_budget(old=100000)
    _, budget = distillation_training.compile_recipe(recipe, token, tmp_path)
    assert budget['accepted_training_rows'] == 1000
    assert sum(budget['course_replay_source_rows_by_stage'].values()) == 400000
    assert not budget['paper_first_round_150000_label_scale_reached']


def native_replay(tmp_path, monkeypatch):
    args = cpu_flow(tmp_path, monkeypatch)
    source, parent, data, cache, foundation, original_rows, tokenized = args
    _, _, roots = source_fixture(tmp_path / 'qa-sources')
    rows = []
    for root in roots:
        if root['split'] != 'test':
            rows.extend(paper_records(root, random.Random(71)))
    replay = tmp_path / 'native-replay'
    for split in ['train', 'validation', 'test']:
        write_jsonl(replay / (split + '.jsonl'), [row for row in rows if row['split'] == split])
    proof = manifest('latent_only_course_replay_data_preflight', {'controlled_producer': True}, [],
        [replay / (split + '.jsonl') for split in ['train', 'validation', 'test']], {
            'all_selected_train_and_validation_answers_natively_regraded': True,
            'all_selected_full_histories_and_futures_natively_checked': True,
            'training_rows_drawn_only_from_original_train_split': True,
            'original_training_rows_preserved_byte_for_byte': True,
            'independent_test_used_for_training_or_selection': False,
            'by_split': {**Counter(row['split'] for row in rows), 'test': 0}})
    atomic_json(replay / 'manifest.json', proof)
    previous = Path(json.loads(Path(cache).read_text())['config']['base_cache'])
    writer = FeatureStoreWriter(tmp_path / 'with-replay-features', [0, 3], 'a' * 64,
                               config={'base_cache': str(previous)})
    writer.add_existing(previous, previous)
    writer.append(block(sorted({row['feature_key'] for row in rows}), width=512)); writer.finish()
    return (source, parent, data, writer.output / 'manifest.json', foundation), replay, rows, original_rows, tokenized


def rewrite_rows(directory, split, rows):
    path = directory / (split + '.jsonl'); write_jsonl(path, rows)
    proof = json.loads((directory / 'manifest.json').read_text())
    proof['outputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(directory / 'manifest.json', proof)


def test_native_preflight_regrades_replay_and_tokenizes_all_actual_sources_except_test(tmp_path, monkeypatch):
    args, replay, rows, originals, tokenized = native_replay(tmp_path, monkeypatch)
    output = tmp_path / 'joint-token-proof'
    result = distillation_preflight.preflight(*args, output, workers=1, replay_data_paths=[replay])
    assert result['by_split'] == {'train': 32, 'validation': 2, 'test': 1}
    assert result['course_replay']['train_validation_rows_tokenized'] == len(rows)
    assert result['course_replay']['combined_split_isolation_verified']
    assert set(tokenized) == {row['id'] for row in [*originals, *rows] if row['split'] != 'test'}
    assert len(tokenized) == 34 + len(rows)
    assert not result['model_weights_loaded_or_training_started']
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['config']['replay_data_paths'] == [str(replay)]
    assert all(str(replay / name) in proof['inputs'] for name in ['manifest.json', 'train.jsonl', 'validation.jsonl'])


def test_answer_profile_reaches_token_initializer_and_completed_native_proof(tmp_path, monkeypatch):
    from xqgeneral.explanation_answer_format import EXPLANATION_FIRST_PROFILE, checked_token_answer_kind
    args, replay, rows, originals, tokenized = native_replay(tmp_path, monkeypatch)
    initializer = distillation_preflight.token_initializer
    seen = []

    def observe(model_path, limit, profile):
        seen.append(profile)
        return initializer(model_path, limit)

    monkeypatch.setattr(distillation_preflight, 'token_initializer', observe)
    output = tmp_path / 'ordered-token-proof'
    before = {str(path): path.read_bytes() for path in Path(args[2]).glob('*.jsonl')}
    result = distillation_preflight.preflight(*args, output, workers=1, replay_data_paths=[replay],
        explanation_answer_profile=EXPLANATION_FIRST_PROFILE)
    proof = json.loads((output / 'manifest.json').read_text())
    assert seen == [EXPLANATION_FIRST_PROFILE]
    assert result['explanation_answer_profile'] == proof['config']['explanation_answer_profile'] == EXPLANATION_FIRST_PROFILE
    assert proof['kind'] != distillation_preflight.TOKEN_KIND
    assert checked_token_answer_kind(proof) == distillation_preflight.TOKEN_KIND
    assert Counter(tokenized) == Counter(row['id'] for row in [*originals, *rows] if row['split'] != 'test')
    assert all(Path(path).read_bytes() == raw for path, raw in before.items())
    assert result['independent_test_rows_tokenized'] == 0


@pytest.mark.parametrize('problem', ['answer', 'heldout_history', 'duplicate_ID', 'duplicate_directory', 'uncertified'])
def test_replay_claims_do_not_replace_native_answers_history_isolation_or_unique_sources(tmp_path, monkeypatch, problem):
    args, replay, rows, originals, tokenized = native_replay(tmp_path, monkeypatch)
    selected = [deepcopy(row) for row in rows if row['split'] == 'train']
    paths = [replay]
    if problem == 'answer': selected[0]['answer'] = 'incorrect native answer'
    elif problem == 'heldout_history':
        protected = next(row for row in originals if row['split'] == 'validation')
        selected[0]['game_id'] = protected['game_id']
    elif problem == 'duplicate_ID': selected[0]['id'] = originals[0]['id']
    elif problem == 'duplicate_directory': paths.append(replay)
    else:
        proof = json.loads((replay / 'manifest.json').read_text());proof['kind'] = 'uncertified'
        atomic_json(replay / 'manifest.json', proof)
    if problem in ['answer', 'heldout_history', 'duplicate_ID']:
        rewrite_rows(replay, 'train', selected)
    output = tmp_path / 'must-not-complete'
    with pytest.raises(ValueError):
        distillation_preflight.preflight(*args, output, workers=1, replay_data_paths=paths)
    assert not tokenized and not (output / 'manifest.json').exists()


def test_token_observations_cannot_silently_replace_a_replay_row_with_a_duplicate(tmp_path, monkeypatch):
    args, replay, _, _, _ = native_replay(tmp_path, monkeypatch)
    original = distillation_preflight.token_chunk
    def duplicated(batch):
        result = original(batch)
        result[-1] = result[0].copy()
        return result
    monkeypatch.setattr(distillation_preflight, 'token_chunk', duplicated)
    output = tmp_path / 'rejected-token-coverage'
    with pytest.raises(ValueError, match='distinct actual supervision'):
        distillation_preflight.preflight(*args, output, workers=1, replay_data_paths=[replay])
    assert not (output / 'manifest.json').exists()


def controlled_joint_round(tmp_path, monkeypatch):
    source, recipe_path, token_path, runner, initial = round_setup(tmp_path, monkeypatch)
    recipe = json.loads(recipe_path.read_text());token = json.loads(token_path.read_text())
    replay = tmp_path / 'controlled-course-replay'
    for split, count in [('train', 4), ('validation', 2)]:
        write_jsonl(replay / (split + '.jsonl'), [
            {'id': f'controlled-{stage}-{split}-{index}', 'feature_key': f'old-{stage}-{split}-{index}',
             'stage': stage, 'split': split, 'answer': 'preserved controlled old answer'}
            for stage in STAGES for index in range(count)])
    atomic_json(replay / 'manifest.json', manifest('controlled_replay_fixture', {'no_real_training': True}, [],
                [replay / (split + '.jsonl') for split in ['train', 'validation']]))
    recipe.update(stages=['explanation', *STAGES], mixture={'explanation': .8, **dict.fromkeys(STAGES, .05)},
                  loss_normalization='example', replay_data_paths=[str(replay)])
    token['config']['replay_data_paths'] = [str(replay)]
    token['verification']['course_replay'] = replay_contract([str(replay)], train=4, validation=2)
    for name in ['manifest.json', 'train.jsonl', 'validation.jsonl']:
        path = replay / name;token['inputs'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic_json(recipe_path, recipe);atomic_json(token_path, token)
    return source, recipe_path, token_path, runner, replay


def test_round_uses_all_sources_in_native_probe_formal_indexes_and_actual_finite_state(tmp_path, monkeypatch):
    source, recipe, token, runner, replay = controlled_joint_round(tmp_path, monkeypatch)
    root = tmp_path / 'joint-round'
    result = distillation_training.run_round(source, recipe, token, root)
    assert result['actual_steps'] == 8 and result['rows_per_complete_pass'] == 40
    assert result['selected_training_rows_by_stage'] == {'explanation': 32, **dict.fromkeys(STAGES, 2)}
    state = result['training_data_state']
    assert state['epoch'] == 4 and state['unique_rows_consumed'] == 160
    assert state['contract']['source_rows_by_stage'] == {'explanation': 32, **dict.fromkeys(STAGES, 4)}
    assert len(runner.jobs) == 4
    proof = json.loads((root / 'manifest.json').read_text())
    assert all(str(replay / name) in proof['inputs'] for name in ['manifest.json', 'train.jsonl', 'validation.jsonl'])
    assert distillation_training.run_round(source, recipe, token, root, resume=True) == result


def test_uncertified_raw_replay_cannot_reach_the_gpu_preflight(tmp_path, monkeypatch):
    source, recipe, token_path, runner, replay = controlled_joint_round(tmp_path, monkeypatch)
    token = json.loads(token_path.read_text());token['inputs'].pop(str(replay / 'validation.jsonl'))
    atomic_json(token_path, token)
    with pytest.raises(ValueError, match='bound by its completed token preflight'):
        distillation_training.run_round(source, recipe, token_path, tmp_path / 'unused')
    assert not runner.jobs
