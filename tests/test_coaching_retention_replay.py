"""Exercise original-coaching retention without claiming a model-quality gain."""
from collections import Counter
import json
from pathlib import Path

import pytest

import test_auxiliary_search_replay as auxiliary_control
import test_engine_qualified_explanations as qualified_control
from xqgeneral import auxiliary_search_replay as replay
from xqgeneral import distillation_preflight, distillation_training
from xqgeneral.engine_qualified_explanations import qualify
from xqgeneral.evidence import atomic_json, digest, load_jsonl, write_jsonl
from xqgeneral.finite_training import prepare_finite_epochs
from xqgeneral.recorded_search_inputs import BoundInputs
from xqgeneral.training_index import IndexedTrainingRows, build_index


def coaching_source(root):
    root.mkdir()
    data, reference = qualified_control.fixture(root, specs=[(0., [0., 0.])] * 3,
                                                opaque_heldout=True)
    output = root / 'qualified'
    qualify(data, reference, output, artifact_root=root)
    return output


def prepare(tmp_path, *, quota=2):
    data = auxiliary_control.dataset(tmp_path)
    coaching = coaching_source(tmp_path / 'original-coaching')
    output = tmp_path / 'retention'
    result = replay.prepare(data, output, train_per_stage=2, coaching_data=coaching,
                            coaching_train_rows=quota)
    return data, coaching, output, result


def rebind_output(output, split, rows):
    path = output / (split + '.jsonl')
    write_jsonl(path, rows)
    proof = json.loads((output / 'manifest.json').read_text())
    proof['outputs'][str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
    atomic_json(output / 'manifest.json', proof)


def test_native_retention_selects_original_train_bytes_without_copying_heldout(tmp_path):
    data, coaching, output, result = prepare(tmp_path)
    old_only = tmp_path / 'six-task-replay'
    replay.prepare(data, old_only, train_per_stage=2)
    raw = (output / 'train.jsonl').read_bytes()
    assert raw.startswith((old_only / 'train.jsonl').read_bytes())
    originals = {json.loads(line)['id']: line
                 for line in (coaching / 'train.jsonl').read_bytes().splitlines(keepends=True)}
    selected = [line for line in raw.splitlines(keepends=True)
                if json.loads(line)['stage'] == 'explanation']
    assert len(selected) == 2
    assert all(originals[json.loads(line)['id']] == line for line in selected)
    assert (output / 'validation.jsonl').read_bytes() == (old_only / 'validation.jsonl').read_bytes()
    assert (output / 'test.jsonl').read_bytes() == b''
    assert result['by_split_and_stage']['train']['explanation'] == 2
    assert 'explanation' not in result['by_split_and_stage']['validation']
    assert result['original_coaching_replay']['source_training_rows'] == 3
    assert result['original_coaching_replay']['selected_training_rows'] == 2
    assert result['original_coaching_replay']['heldout_answers_parsed_or_tokenized'] == 0
    rows, checked = distillation_preflight.course_replay_sources(
        [output], BoundInputs(), profile=replay.COACHING_PROFILE)
    assert len(rows) == sum(result['by_split'].values())
    assert checked['original_coaching_replay'] == result['original_coaching_replay']
    with pytest.raises(ValueError):
        distillation_preflight.course_replay_sources([output], BoundInputs(), profile=replay.PROFILE)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert replay.prepare(data, output, train_per_stage=2, coaching_data=coaching,
                          coaching_train_rows=2, resume=True) == result
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize('damage', ['omit_old', 'duplicate_old', 'change_old_answer', 'copy_old_validation'])
def test_rehashed_retention_must_reconstruct_every_original_byte(tmp_path, damage):
    _, coaching, output, _ = prepare(tmp_path)
    rows = load_jsonl(output / 'train.jsonl')
    old = next(row for row in rows if row['stage'] == 'explanation')
    split = 'train'
    if damage == 'omit_old': rows.remove(old)
    elif damage == 'duplicate_old': rows.append(old)
    elif damage == 'change_old_answer': old['answer'] += ' changed'
    else:
        split = 'validation'
        rows = load_jsonl(output / 'validation.jsonl') + [dict(old, split='validation')]
    rebind_output(output, split, rows)
    with pytest.raises(ValueError):
        replay.checked_subset(output, BoundInputs(), profile=replay.COACHING_PROFILE)


@pytest.mark.parametrize('quota', [0, -1, True, 1.5])
def test_explicit_original_coaching_source_requires_positive_integer_quota(tmp_path, quota):
    with pytest.raises(ValueError):
        replay.prepare(tmp_path / 'source', tmp_path / 'unused',
                       coaching_data=tmp_path / 'old', coaching_train_rows=quota)
    assert not (tmp_path / 'unused').exists()


def test_original_coaching_cannot_be_requested_without_its_source(tmp_path):
    with pytest.raises(ValueError):
        replay.prepare(tmp_path / 'source', tmp_path / 'unused', coaching_train_rows=2)
    assert not (tmp_path / 'unused').exists()


def coaching_budget():
    template, token = auxiliary_control.budget()
    template['replay_profile'] = token['config']['replay_profile'] = replay.COACHING_PROFILE
    source = token['verification']['course_replay']
    source['by_split_and_stage']['train']['explanation'] = 17
    source['train_validation_rows_tokenized'] += 17
    source['original_coaching_replay'] = dict(source_training_rows=9823, selected_training_rows=17,
                                            heldout_answers_parsed_or_tokenized=0)
    return template, token


def test_explanation_first_profile_preserves_the_full_coaching_replay_budget(tmp_path):
    from xqgeneral.explanation_answer_format import EXPLANATION_FIRST_PROFILE
    template, token = coaching_budget()
    _, before = distillation_training.compile_recipe(template, token, tmp_path / 'legacy')
    template['explanation_answer_profile'] = EXPLANATION_FIRST_PROFILE
    token['config']['explanation_answer_profile'] = EXPLANATION_FIRST_PROFILE
    token['verification']['explanation_answer_profile'] = EXPLANATION_FIRST_PROFILE
    from xqgeneral.explanation_answer_format import profiled_token_kind
    token['kind'] = profiled_token_kind(distillation_preflight.TOKEN_KIND, EXPLANATION_FIRST_PROFILE)
    recipe, after = distillation_training.compile_recipe(template, token, tmp_path / 'active')
    assert recipe['explanation_answer_profile'] == EXPLANATION_FIRST_PROFILE
    assert before == after


@pytest.mark.parametrize('change', ['recipe_only', 'token_only', 'unknown'])
def test_explanation_profile_mismatch_cannot_reuse_a_previous_token_proof(tmp_path, change):
    from xqgeneral.explanation_answer_format import EXPLANATION_FIRST_PROFILE
    template, token = coaching_budget()
    if change != 'token_only':
        template['explanation_answer_profile'] = EXPLANATION_FIRST_PROFILE
    if change != 'recipe_only':
        token['config']['explanation_answer_profile'] = EXPLANATION_FIRST_PROFILE
    if change == 'unknown':
        template['explanation_answer_profile'] = token['config']['explanation_answer_profile'] = 'unknown'
    with pytest.raises(ValueError, match='answer profile'):
        distillation_training.compile_recipe(template, token, tmp_path / 'unused')


def test_compiler_sums_old_and_new_explanations_but_keeps_new_label_budget_and_validation(tmp_path):
    template, token = coaching_budget()
    recipe, budget = distillation_training.compile_recipe(template, token, tmp_path)
    assert budget['accepted_training_rows'] == 31
    assert budget['original_coaching_replay_training_rows'] == 17
    assert budget['all_explanation_training_rows'] == 48
    assert budget['rows_per_complete_pass'] == 288
    assert budget['finite_mixture']['source_rows_by_stage']['explanation'] == 48
    assert recipe['validation_examples'] == 523
    assert recipe['validation_rows_by_stage']['explanation'] == 384
    assert not budget['auxiliary_replay_counts_toward_new_search_label_scale']
    rows = [dict(id=f'{origin}-{i}', stage='explanation', split='train', feature_key=f'{origin}-{i}')
            for origin, count in [('new', 31), ('old', 17)] for i in range(count)]
    rows += [dict(id=f'{stage}-{i}', stage=stage, split='train', feature_key=f'{stage}-{i}')
             for stage in replay.STAGES for i in range(40)]
    path = tmp_path / 'rows.jsonl'
    write_jsonl(path, rows)
    build_index([path], tmp_path / 'index')
    indexed = IndexedTrainingRows(tmp_path / 'index', [path], recipe['mixture'])
    try:
        finite = prepare_finite_epochs(indexed, recipe['mixture'], dict(recipe, steps=budget['max_updates']), 3)
        seen = Counter()
        for step in range(budget['updates_per_complete_pass']):
            for rank in range(3):
                indices, _ = finite.batch_indices(step, rank)
                seen.update(indexed.row_metadata(i)['id'] for i in indices)
        assert set(seen) == {row['id'] for row in rows}
        assert sum(seen.values()) == 288
    finally:
        indexed.close()


@pytest.mark.parametrize('damage', ['missing_old', 'old_validation', 'missing_proof', 'wrong_count', 'legacy_template'])
def test_compiler_rejects_incomplete_or_duplicated_coaching_coverage(tmp_path, damage):
    template, token = coaching_budget()
    source = token['verification']['course_replay']
    if damage == 'missing_old': source['by_split_and_stage']['train'].pop('explanation')
    elif damage == 'old_validation': source['by_split_and_stage']['validation']['explanation'] = 384
    elif damage == 'missing_proof': source.pop('original_coaching_replay')
    elif damage == 'wrong_count': source['original_coaching_replay']['selected_training_rows'] -= 1
    else: template['replay_profile'] = replay.PROFILE
    with pytest.raises(ValueError):
        distillation_training.compile_recipe(template, token, tmp_path / 'unused')


@pytest.mark.parametrize('missing_old_feature', [False, True])
def test_native_token_gate_covers_old_coaching_and_requires_its_exact_features(tmp_path, monkeypatch,
                                                                           missing_old_feature):
    from test_feature_store import block
    from xqgeneral.feature_store import FeatureStoreWriter
    args, _, _, originals, tokenized = auxiliary_control.joint_control.native_replay(tmp_path, monkeypatch)
    data = auxiliary_control.dataset(tmp_path / 'auxiliary-source')
    coaching = coaching_source(tmp_path / 'original-coaching')
    output = tmp_path / 'retention'
    replay.prepare(data, output, train_per_stage=2, coaching_data=coaching, coaching_train_rows=2)
    rows = load_jsonl(output / 'train.jsonl') + load_jsonl(output / 'validation.jsonl')
    cache = Path(args[3])
    base = Path(json.loads(cache.read_text())['config']['base_cache'])
    def keys(path):
        return set(json.loads(Path(json.loads(path.read_text())['catalog']['keys']).read_text()))
    old_keys = {row['feature_key'] for row in rows if row['stage'] == 'explanation'}
    wanted = {row['feature_key'] for row in rows} | keys(cache)
    if missing_old_feature:
        missing = next(iter(old_keys - keys(base) - keys(cache)))
        wanted.remove(missing)
    writer = FeatureStoreWriter(tmp_path / 'retention-features', [0, 3], 'a' * 64,
                                config=dict(base_cache=str(base), base_proof=str(base)))
    writer.add_existing(base, base)
    writer.append(block(sorted(wanted - keys(base)), width=512))
    writer.finish()
    call = lambda: distillation_preflight.preflight(
        *[*args[:3], writer.output / 'manifest.json', args[4]], tmp_path / 'token-gate', workers=1,
        replay_data_paths=[output], replay_profile=replay.COACHING_PROFILE)
    if missing_old_feature:
        with pytest.raises(ValueError, match='missing its expert history features'):
            call()
        assert not (tmp_path / 'token-gate').exists()
    else:
        result = call()
        assert result['course_replay']['original_coaching_replay']['selected_training_rows'] == 2
        assert Counter(tokenized) == Counter(row['id'] for row in originals + rows if row['split'] != 'test')
        assert result['course_replay']['combined_split_isolation_verified']
        assert result['independent_test_rows_tokenized'] == 0


def test_old_and_new_coaching_cannot_repeat_a_history_under_different_ids(tmp_path, monkeypatch):
    args, _, _, originals, _ = auxiliary_control.joint_control.native_replay(tmp_path, monkeypatch)
    old = dict(next(row for row in originals if row['split'] == 'train'), id='different-old-coaching-id')
    monkeypatch.setattr(distillation_preflight, 'course_replay_sources',
                        lambda *args, **kwargs: ([old], dict(data_paths=['controlled-source'])))
    with pytest.raises(ValueError, match='distinct full histories'):
        distillation_preflight.preflight(*args, tmp_path / 'unused', workers=1,
                                        replay_data_paths=['controlled-source'], replay_profile=replay.COACHING_PROFILE)
    assert not (tmp_path / 'unused').exists()
