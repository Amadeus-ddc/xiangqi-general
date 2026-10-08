import copy
import json
from pathlib import Path

import pytest

from xqgeneral.collect_teacher import mirror_label
from xqgeneral.evidence import atomic_json, history_key, manifest
from xqgeneral.explanations import EXPLANATION_QUESTION, move_facts
from xqgeneral.reviewed_explanations import merge_reviewed
from xqgeneral.revise_prose import TEACHER
from xqgeneral.rules import START_FEN, legal_moves, replay


def label(identity, split, moves):
    history = replay(START_FEN, moves);fen = history[-1];move = legal_moves(fen)[0]
    score = {'type': 'cp', 'value': 0, 'perspective': 'side_to_move'}
    value = {'move': move, 'pv': [move], 'candidates': [move], 'evaluation': score,
             'facts': move_facts(fen, move), 'explanation': '受控数据合同测试，不表示真实教师质量。',
             'branches': [{'move': move, 'pv': [move], 'evaluation': score}]}
    return {'id': identity, 'game_id': 'game-' + identity, 'split': split, 'stage': 'explanation',
            'task_type': 'explanation', 'query': {}, 'question': EXPLANATION_QUESTION,
            'initial_fen': START_FEN, 'moves': moves, 'history': history, 'fen': fen,
            'feature_key': history_key(history), 'future_moves': [], 'provenance': 'controlled_contract_fixture',
            'answer': json.dumps(value, ensure_ascii=False), 'teacher': dict(TEACHER),
            'prose_review': {'source_annotation_sha256': 'a' * 64,
                             'reviewed_annotation_sha256': 'b' * 64,
                             'review_manifest_sha256': 'c' * 64, 'changed': True}}


def source(root, rows, kind='reviewed_recorded_coach_initial_teacher_labels'):
    root.mkdir(parents=True, exist_ok=True)
    paths, counts = [], {}
    for split in ['train', 'validation', 'test']:
        path = root / f'{split}.jsonl';selected = [r for r in rows if r['split'] == split]
        # Deliberate noncanonical JSON whitespace proves preservation of actual bytes.
        path.write_bytes(b''.join(('  ' + json.dumps(r, ensure_ascii=False) + '  \n').encode() for r in selected))
        paths.append(path);counts[split] = len(selected)
    proof = {'by_split': counts, 'original_teacher_annotations': len(rows),
             'independently_accepted_original_annotations': len(rows),
             'all2048_exact_independent_neural_acceptance_chains_verified': True,
             'all6662_prior_train_records_and_byte_prefix_preserved': True,
             'heldout_files_byte_identical': True}
    atomic_json(root / 'manifest.json', manifest(kind, {}, [], paths, proof))
    return root


def pair(tmp_path):
    first = [label('a', 'train', ['b0c2']), label('b', 'validation', ['h0g2']),
             label('c', 'test', ['b2e2'])]
    second = [label('d', 'train', ['b0a2']), label('e', 'validation', ['h0i2']),
              label('f', 'test', ['h2e2'])]
    return source(tmp_path / 'first', first), source(tmp_path / 'second', second)


def test_merge_preserves_every_source_byte_and_appends_heldouts_in_their_original_splits(tmp_path):
    roots = pair(tmp_path);output = tmp_path / 'combined'
    proof = merge_reviewed(roots, output, 1)
    for split in ['train', 'validation', 'test']:
        assert (output / f'{split}.jsonl').read_bytes() == b''.join((p / f'{split}.jsonl').read_bytes() for p in roots)
    assert proof['by_split'] == {'train': 2, 'validation': 2, 'test': 2}
    assert proof['complete_native_histories_facts_and_all_continuations_revalidated']
    assert proof['derived_teacher_records_generated'] == 0 and not proof['student_training_started']
    assert proof['independent_test_used_for_isolation_only'] and proof['four_new_courses_required_before_sft']


def test_serial_and_spawned_native_checks_preserve_identical_label_files(tmp_path):
    roots = pair(tmp_path);serial = tmp_path / 'serial';parallel = tmp_path / 'parallel'
    merge_reviewed(roots, serial, 1);merge_reviewed(roots, parallel, 2)
    for name in ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'verification.json']:
        assert (serial / name).read_bytes() == (parallel / name).read_bytes()


@pytest.mark.parametrize('problem', ['changed_bytes', 'incomplete_manifest', 'unsupported_kind', 'wrong_count'])
def test_changed_or_unqualified_sources_refuse_before_creating_output(tmp_path, problem):
    roots = pair(tmp_path);path = roots[0] / 'manifest.json';proof = json.loads(path.read_text())
    if problem == 'changed_bytes':
        p = roots[0] / 'train.jsonl';p.write_bytes(p.read_bytes() + b'\n')
    else:
        if problem == 'incomplete_manifest':proof['status'] = 'running'
        if problem == 'unsupported_kind':proof['kind'] = 'unreviewed_teacher_predictions'
        if problem == 'wrong_count':proof['verification']['by_split']['train'] += 1
        atomic_json(path, proof)
    with pytest.raises(ValueError):merge_reviewed(roots, tmp_path / 'not-created', 1)
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('problem', ['duplicate_id', 'duplicate_history', 'same_game'])
def test_identity_collisions_and_games_across_splits_cannot_be_merged(tmp_path, problem):
    a = label('a', 'train', ['b0c2']);b = label('b', 'test', ['h0g2'])
    if problem == 'duplicate_id':b['id'] = a['id']
    if problem == 'duplicate_history':b = dict(copy.deepcopy(a), id='b', game_id='game-b', split='test')
    if problem == 'same_game':b['game_id'] = a['game_id']
    roots = [source(tmp_path / 'a', [a]), source(tmp_path / 'b', [b])]
    with pytest.raises(ValueError, match='overlap'):merge_reviewed(roots, tmp_path / 'not-created', 1)
    assert not (tmp_path / 'not-created').exists()


def test_legal_answer_continuation_cannot_become_a_heldout_root(tmp_path):
    a = label('a', 'train', ['b0c2']);continuation = json.loads(a['answer'])['pv']
    b = label('b', 'test', [*a['moves'], *continuation])
    roots = [source(tmp_path / 'a', [a]), source(tmp_path / 'b', [b])]
    with pytest.raises(ValueError, match='continuation or color'):merge_reviewed(roots, tmp_path / 'not-created', 1)
    assert not (tmp_path / 'not-created').exists()


def test_color_counterpart_is_reserved_even_when_not_used_as_a_teacher_label(tmp_path):
    a = label('a', 'train', ['b0c2']);b = mirror_label(a)
    b.update(split='test', game_id='distinct-game')
    roots = [source(tmp_path / 'a', [a]), source(tmp_path / 'b', [b])]
    with pytest.raises(ValueError, match='continuation or color'):merge_reviewed(roots, tmp_path / 'not-created', 1)
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('problem', ['history', 'recorded_future', 'native_facts', 'teacher', 'review_hash'])
def test_hash_bound_but_native_inconsistent_labels_are_rejected(tmp_path, problem):
    row = label('a', 'train', ['b0c2'])
    if problem == 'history':row['history'] = [START_FEN]
    if problem == 'recorded_future':row['future_moves'] = ['b0b9']
    if problem == 'native_facts':
        value = json.loads(row['answer']);value['facts']['piece'] = '红帅';row['answer'] = json.dumps(value)
    if problem == 'teacher':row['teacher']['reasoning_effort'] = 'high'
    if problem == 'review_hash':row['prose_review']['reviewed_annotation_sha256'] = 'unbound'
    root = source(tmp_path / 'source', [row])
    with pytest.raises(ValueError):merge_reviewed([root], tmp_path / 'not-created', 1)
    assert not (tmp_path / 'not-created').exists()


def test_finished_repetition_context_is_not_an_explanation_root(tmp_path):
    row = label('a', 'train', ['b0c2', 'b9c7', 'c2b0', 'c7b9'] * 3)
    root = source(tmp_path / 'source', [row])
    with pytest.raises(ValueError, match='terminal'):merge_reviewed([root], tmp_path / 'not-created', 1)


def test_new_recorded_heldout_labels_require_exact_inline_review_metadata(tmp_path):
    rows = [label('a', 'train', ['b0c2']), label('b', 'test', ['h0g2'])];rows[1].pop('prose_review')
    root = source(tmp_path / 'source', rows)
    with pytest.raises(ValueError, match='inline acceptance'):merge_reviewed([root], tmp_path / 'not-created', 1)
    assert not (tmp_path / 'not-created').exists()


def legacy_pool(root, row):
    root.mkdir();paths = []
    for split in ['train', 'validation', 'test']:
        path = root / f'{split}.jsonl'
        path.write_text(json.dumps(dict(row, stage='selfplay_explanation')) + '\n' if split == 'train' else '')
        paths.append(path)
    atomic_json(root / 'manifest.json', manifest('reviewed_selfplay_explanation_replay_pool', {}, [], paths,
        {'train': 1, 'validation': 0, 'test': 0, 'all_384_annotations_cross_reviewed': True,
         'all_39_revisions_independently_accepted': True, 'changed_field_only': 'stage',
         'prompts_and_answers_byte_values_preserved': True}))
    return root


def test_legacy_training_requires_exact_supplemental_acceptance_and_preserves_unrated_heldouts(tmp_path):
    a = label('a', 'train', ['b0c2']);b = label('b', 'test', ['h0g2'])
    a.pop('prose_review');b.pop('prose_review')
    root = source(tmp_path / 'source', [a, b], 'fully_reviewed_selfplay_teacher_append')
    with pytest.raises(ValueError, match='supplemental acceptance'):merge_reviewed([root], tmp_path / 'not-created', 1)
    pool = legacy_pool(tmp_path / 'legacy', a);output = tmp_path / 'combined'
    proof = merge_reviewed([root], output, 1, [pool])
    assert proof['legacy_training_rows_bound_to_exact_supplemental_acceptance'] == 1
    assert proof['legacy_rows_without_inline_review_metadata_by_split'] == {'train': 1, 'test': 1}
    assert proof['legacy_heldout_semantic_acceptance_not_newly_established']
    assert (output / 'test.jsonl').read_bytes() == (root / 'test.jsonl').read_bytes()


def test_legacy_acceptance_with_the_same_id_cannot_authorize_changed_prose(tmp_path):
    row = label('a', 'train', ['b0c2']);row.pop('prose_review')
    root = source(tmp_path / 'source', [row], 'fully_reviewed_selfplay_teacher_append')
    other = copy.deepcopy(row);value = json.loads(other['answer']);value['explanation'] += '另一个正文。'
    other['answer'] = json.dumps(value);pool = legacy_pool(tmp_path / 'legacy', other)
    with pytest.raises(ValueError, match='supplemental acceptance'):merge_reviewed([root], tmp_path / 'not-created', 1, [pool])
    assert not (tmp_path / 'not-created').exists()


def test_existing_output_and_repeated_source_are_preserved(tmp_path):
    roots = pair(tmp_path);output = tmp_path / 'existing';output.mkdir();(output / 'keep').write_text('preserved')
    with pytest.raises(FileExistsError):merge_reviewed(roots, output, 1)
    assert (output / 'keep').read_text() == 'preserved'
    with pytest.raises(ValueError, match='supplied twice'):merge_reviewed([roots[0], roots[0]], tmp_path / 'not-created', 1)


class TokenizerFixture:
    eos_token = '<EOS>'

    def add_tokens(self, tokens, special_tokens=False):
        return len(tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return '\n'.join(m['content'] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def preflight_fixture(tmp_path, monkeypatch, *, literal_prompt=False, long_test=False,
                      missing_feature=False, wrong_precision=False):
    import torch
    from xqgeneral.evidence import digest
    from transformers import AutoTokenizer
    rows = [label('a', 'train', ['b0c2']), label('b', 'validation', ['h0g2']),
            label('c', 'test', ['b2e2'])]
    if literal_prompt:rows[0]['question'] += rows[0]['fen']
    if long_test:
        value = json.loads(rows[2]['answer']);value['explanation'] *= 1000
        rows[2]['answer'] = json.dumps(value, ensure_ascii=False)
    upstream = source(tmp_path / 'source', rows);data = tmp_path / 'combined'
    merge_reviewed([upstream], data, 1)
    recipe = json.loads(Path('configs/foundation-human-engine-clean-v2.json').read_text())
    cache_path = tmp_path / 'features.pt';keys = [r['feature_key'] for r in rows]
    if missing_feature:keys[-1] = 'd' * 64
    dtype = torch.float32 if wrong_precision else torch.float16
    cache = {'keys': keys, 'depths': recipe['expert_feature_depths'],
             'features': [torch.zeros((len(keys), 90, 512), dtype=dtype)
                          for _ in recipe['expert_feature_depths']],
             'wdl': torch.zeros((len(keys), 3), dtype=torch.float32)}
    torch.save(cache, cache_path);cache_manifest = cache_path.with_suffix('.manifest.json')
    atomic_json(cache_manifest, manifest('controlled_cache_fixture', {}, [], [cache_path]))
    model = tmp_path / 'tokenizer';model.mkdir()
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']:(model / name).write_text('{}')
    recipe.update(model_path=str(model), feature_path=str(cache_path), data_path=str(data))
    config = tmp_path / 'foundation.json';atomic_json(config, recipe)
    completed = tmp_path / 'full-preflight.json'
    atomic_json(completed, manifest('recorded_engine_clean_latent_foundation_preflight', {},
        [config, cache_manifest, cache_path], [],
        {'cache_sha256': digest(cache_path), 'source_recipe_sha256': digest(config)}))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: TokenizerFixture())
    return data, config, completed, cache_path


def test_token_preflight_covers_all_train_validation_and_features_but_never_encodes_test_answers(tmp_path, monkeypatch):
    from xqgeneral.explanation_preflight import preflight
    data, config, completed, cache = preflight_fixture(tmp_path, monkeypatch, long_test=True)
    root = tmp_path / 'preflight';proof = preflight(data, config, completed, root, max_tokens=4096, workers=1)
    observations = [json.loads(x) for x in (root / 'token-lengths.jsonl').read_text().splitlines()]
    assert {x['split'] for x in observations} == {'train', 'validation'}
    assert proof['feature_contexts_checked'] == 3 and proof['feature_contexts_missing'] == 0
    assert proof['train_validation_rows_tokenized'] == 2 and proof['independent_test_rows_tokenized'] == 0
    assert proof['all_train_validation_supervision_fits_without_truncation']
    assert not proof['full_feature_cache_rehashed_again_by_this_preflight']
    assert not proof['ready_to_start_sft'] and not proof['model_weights_loaded_or_training_started']
    result = json.loads((root / 'manifest.json').read_text())
    assert str(cache) not in result['inputs'] and result['cache_header_only_input']['path'] == str(cache)


@pytest.mark.parametrize('problem', ['incomplete_cache', 'incomplete_preflight', 'different_cache_identity'])
def test_token_preflight_requires_the_exact_completed_foundation_and_cache_proofs(tmp_path, monkeypatch, problem):
    from xqgeneral.explanation_preflight import preflight
    data, config, completed, cache = preflight_fixture(tmp_path, monkeypatch)
    path = cache.with_suffix('.manifest.json') if problem == 'incomplete_cache' else completed
    doc = json.loads(path.read_text())
    if problem == 'different_cache_identity':doc['verification']['cache_sha256'] = 'e' * 64
    else:doc['status'] = 'running'
    atomic_json(path, doc)
    with pytest.raises(ValueError):preflight(data, config, completed, tmp_path / 'not-created', workers=1)
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('problem', ['missing_feature', 'wrong_precision'])
def test_token_preflight_checks_actual_cache_header_coverage_and_precision(tmp_path, monkeypatch, problem):
    from xqgeneral.explanation_preflight import preflight
    args = preflight_fixture(tmp_path, monkeypatch, **{problem: True})
    with pytest.raises(ValueError):preflight(*args[:3], tmp_path / 'not-created', workers=1)
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('problem', ['literal_prompt', 'token_overflow'])
def test_failed_token_preflight_preserves_failure_and_never_marks_data_ready(tmp_path, monkeypatch, problem):
    from xqgeneral.explanation_preflight import preflight
    args = preflight_fixture(tmp_path, monkeypatch, literal_prompt=problem == 'literal_prompt')
    output = tmp_path / 'incomplete'
    with pytest.raises(ValueError):preflight(*args[:3], output, workers=1, max_tokens=1 if problem == 'token_overflow' else 4096)
    assert not (output / 'manifest.json').exists()
    with pytest.raises(FileExistsError):preflight(*args[:3], output, workers=1)
