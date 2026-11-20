"""Check target ordering across encoding, validation, preflight and resume."""
from contextlib import nullcontext
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from xqgeneral import explanation_preflight, sft_preflight, training
from xqgeneral.explanation_answer_format import (
    EXPLANATION_FIRST_PROFILE as PROFILE, answer_profile_kwargs, checked_answer_profile,
    checked_token_answer_kind, explanation_first_json, profiled_token_kind,
    require_matching_answer_profiles, serialized_answer,
)


def test_explanation_profile_preserves_admission_of_the_protected_native_course_corpus():
    from xqgeneral import oracle_policy_corpus, search_protection

    current = oracle_policy_corpus._closure()
    producer = dict(current)
    # The real protected corpus binds this original foundation module. Its course
    # tokenizer does not consume explanation targets and must retain its identity.
    producer['foundation_preflight.py'] = '30452d75da538b0383cefb92e1c90546828cb307ec37cea55b8a2044ceefdf03'
    producer['oracle.py'] = search_protection._LEGACY_ORACLE
    result = search_protection._decoder_compatibility(producer, current)
    assert result['profile'] == 'exact_successful_decoder_typed_unscored_error_upgrade_v2'
    assert result['all_other_closure_modules_byte_identical'] is True


def payload():
    return dict(move='a0a1', pv=['a0a1', 'a9a8'],
                facts=dict(piece='红车', check=False, integer=1, decimal=1.0),
                explanation='比较 "引号", {括号} 和换行\n第二行，再给出走法。',
                branches=[], extra=dict(values=[True, 1, 1.0, None]))


def record():
    return dict(id='one', game_id='game', stage='explanation', split='validation', feature_key='one',
                fen='rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1',
                question='请分析当前局面，以JSON回答。', answer=json.dumps(payload(), ensure_ascii=False))


@pytest.mark.parametrize('options', [
    dict(ensure_ascii=False), dict(ensure_ascii=True),
    dict(ensure_ascii=False, separators=(',', ':')), dict(ensure_ascii=False, indent=2),
])
def test_reordering_keeps_nested_values_escapes_whitespace_and_member_content(options):
    original = '\n ' + json.dumps(payload(), **options) + '\t\r\n'
    result = explanation_first_json(original)
    assert json.dumps(json.loads(original), sort_keys=True) == json.dumps(json.loads(result), sort_keys=True)
    assert list(json.loads(result)) == ['explanation', 'move', 'pv', 'facts', 'branches', 'extra']
    assert result.startswith('\n ') and result.endswith('\t\r\n')
    assert explanation_first_json(result) == result


def test_raw_numeric_spellings_and_escaped_member_text_are_not_reserialized():
    raw = r'{ "move" : "a0a1", "value":1e+02, "minus":-0.0, "explanation" : "\u4e2d\n\"x\"", "other":{"b":true} }'
    result = explanation_first_json(raw)
    for member in [r'"explanation" : "\u4e2d\n\"x\""', '"value":1e+02', '"minus":-0.0', '"other":{"b":true}']:
        assert member in result
    assert list(json.loads(result))[0] == 'explanation'


@pytest.mark.parametrize('value', [
    None, [], '{}', '[]', '{"move":"a0a1"}',
    '{"move":1,"explanation":"text"}', '{"move":"a0a1","explanation":[]}',
    '{"move":"a0a1","explanation":"a","explanation":"b"}',
    '{"move":"a0a1","explanation":"a","nested":{"x":1,"x":2}}',
    '{"move":"a0a1","explanation":"a","score":NaN}',
])
def test_ambiguous_or_invalid_active_targets_fail_instead_of_being_repaired(value):
    with pytest.raises(ValueError):
        explanation_first_json(value)


def test_default_does_not_parse_or_rewrite_existing_answers():
    row = record()
    row['answer'] = '  legacy unstructured target\n'
    before = copy.deepcopy(row)
    assert serialized_answer(row) == before['answer'] and row == before
    assert answer_profile_kwargs({}) == {}
    assert answer_profile_kwargs({'explanation_answer_profile': None}) == {}


@pytest.mark.parametrize('stage', ['static_current', 'dynamic_current', 'static_future',
                                  'dynamic_future', 'move_quality', 'engine_numeric'])
def test_opt_in_keeps_every_other_task_target_exact(stage):
    row = record()
    row.update(stage=stage, answer='  原生非讲解答案 a0a1\n')
    assert serialized_answer(row, PROFILE) == row['answer']


@pytest.mark.parametrize('profile', [False, True, 0, '', 'unknown', {}, []])
def test_unknown_or_boolean_profiles_are_rejected(profile):
    with pytest.raises(ValueError):
        checked_answer_profile(profile)


class TinyInputs(dict):
    def to(self, _):
        return self


class Tokenizer:
    eos_token = '~'
    pad_token_id = 0
    merged_header = '{"explanation":'
    merged_id = 0x110000

    def apply_chat_template(self, messages, **_):
        return '\n'.join(m['role'] + ':' + m['content'] for m in messages) + '\nassistant:'

    def encode(self, text, **_):
        if text.startswith(self.merged_header):
            return [self.merged_id] + [ord(c) for c in text[len(self.merged_header):]]
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.merged_header if int(i) == self.merged_id else chr(int(i))
                       for i in ids if not skip_special_tokens or int(i) not in (0, ord(self.eos_token)))

    def __call__(self, text, **_):
        return TinyInputs(input_ids=torch.tensor([self.encode(text)]))


def decoded_target(tokenizer, labels):
    return tokenizer.decode([int(i) for i in labels if int(i) != -100])


@pytest.mark.parametrize('board_tokens', [False, True])
def test_encoder_changes_only_supervised_target_and_keeps_prompt_and_source_record(board_tokens):
    from xqgeneral.board_tokens import encode_board_text
    tok, row = Tokenizer(), record()
    before = copy.deepcopy(row)
    old_ids, old_labels = training.encode_record(tok, row, 5000, board_tokens=board_tokens)
    ids, labels = training.encode_record(tok, row, 5000, board_tokens=board_tokens,
                                        explanation_answer_profile=PROFILE)
    assert [i for i, label in zip(old_ids, old_labels) if label == -100] == [
        i for i, label in zip(ids, labels) if label == -100]
    expected = dict(explanation=payload()['explanation'], **{k: v for k, v in payload().items() if k != 'explanation'})
    expected = json.dumps(expected, ensure_ascii=False)
    if board_tokens:
        expected = encode_board_text(expected)
    assert decoded_target(tok, labels) == expected + tok.eos_token
    assert row == before


def test_cpu_token_worker_and_training_batch_encode_the_same_active_target(monkeypatch):
    tok, row = Tokenizer(), record()
    monkeypatch.setattr(explanation_preflight, '_TOKENIZER', tok)
    monkeypatch.setattr(explanation_preflight, '_TOKEN_LIMIT', 5000)
    monkeypatch.setattr(explanation_preflight, '_ANSWER_PROFILE', PROFILE)
    observed = explanation_preflight.token_chunk([row])[0]
    batch = training.batch_inputs(tok, [row], 5000, 'cpu', board_tokens=True,
                                  explanation_answer_profile=PROFILE)
    assert observed['tokens'] == batch['input_ids'].shape[1]
    assert observed['supervised_tokens'] == int((batch['labels'] != -100).sum())
    assert explanation_preflight.encoded_lengths(tok, row, 5000)['tokens'] != observed['tokens']


@pytest.mark.parametrize('zero,rotate', [(False, False), (True, False), (False, True)])
def test_all_validation_memory_modes_use_the_declared_target_profile(monkeypatch, zero, rotate):
    tok, first = Tokenizer(), record()
    second = dict(first, id='two', feature_key='two')
    captured = []

    class Model:
        def eval(self):
            pass

        def board_context(self, _):
            return nullcontext()

        def __call__(self, **inputs):
            captured.append(inputs)
            return SimpleNamespace(loss=torch.tensor(1.))

    monkeypatch.setattr(training, 'select_features', lambda *a, **k: [])
    config = dict(batch_size=2, max_tokens=5000, explanation_answer_profile=PROFILE)
    assert training.evaluate(Model(), tok, [first, second], {}, {}, config, 'cpu', zero=zero, rotate=rotate) == 1.
    for labels in captured[0]['labels']:
        assert list(json.loads(decoded_target(tok, labels).removesuffix(tok.eos_token)))[0] == 'explanation'


def test_generated_examples_compare_against_the_declared_target(monkeypatch):
    tok, row = Tokenizer(), record()
    expected = serialized_answer(row, PROFILE)

    def generate(**inputs):
        return torch.cat([inputs['input_ids'], torch.tensor([tok.encode(expected + tok.eos_token)])], dim=1)

    model = SimpleNamespace(eval=lambda: None, board_context=lambda _: nullcontext(),
                            base=SimpleNamespace(generate=generate))
    monkeypatch.setattr(training, 'select_features', lambda *a, **k: [])
    result = training.generate_examples(model, tok, [row], {}, {}, 'cpu', explanation_answer_profile=PROFILE)[0]
    assert result['expected'] == expected and result['exact_match'] and row['answer'] != expected


def test_exact_resume_rejects_switching_in_either_direction():
    base = dict(mode='text_lora', batch_size=2, micro_batch_size=1)
    active = dict(base, explanation_answer_profile=PROFILE)
    training.compatible_resume(active, active)
    for left, right in [(base, active), (active, base)]:
        with pytest.raises(ValueError, match='answer profile'):
            training.compatible_resume(left, right)


@pytest.mark.parametrize('left,right', [(None, PROFILE), (PROFILE, None)])
def test_token_profile_cannot_be_changed_by_a_training_recipe(left, right):
    with pytest.raises(ValueError, match='answer profile'):
        require_matching_answer_profiles({'explanation_answer_profile': left},
                                         {'explanation_answer_profile': right})


def test_gpu_preflight_rejects_profile_mismatch_before_parent_or_asset_loading(monkeypatch, tmp_path):
    class Guard:
        def json(self, path):
            return dict(explanation_answer_profile=PROFILE) if str(path).endswith('recipe.json') else dict(status='complete')

        def bind(self, *args):
            pytest.fail('Profile mismatch must fail before reading weight/cache inputs')

    monkeypatch.setattr(sft_preflight, 'BoundInputs', Guard)
    with pytest.raises(ValueError, match='answer profile'):
        sft_preflight.run_preflight('unused.pt', tmp_path / 'recipe.json', tmp_path / 'token.json', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('base', ['latent_only_search_distillation_data_preflight',
                                 'latent_only_reviewed_explanation_data_preflight'])
def test_active_token_proof_has_a_distinct_raw_kind_and_explicit_matching_declarations(base):
    token = dict(kind=profiled_token_kind(base, PROFILE),
                 config=dict(explanation_answer_profile=PROFILE),
                 verification=dict(explanation_answer_profile=PROFILE))
    assert token['kind'] != base
    assert checked_token_answer_kind(token) == base
    assert profiled_token_kind(base) == base
    assert checked_token_answer_kind(dict(kind=base)) == base


@pytest.mark.parametrize('damage', ['legacy_kind', 'missing_profile', 'missing_verification',
                                   'wrong_verification', 'unknown_profile'])
def test_format_metadata_alone_cannot_turn_a_legacy_token_proof_into_the_new_kind(damage):
    base = 'latent_only_search_distillation_data_preflight'
    token = dict(kind=profiled_token_kind(base, PROFILE),
                 config=dict(explanation_answer_profile=PROFILE),
                 verification=dict(explanation_answer_profile=PROFILE))
    if damage == 'legacy_kind':
        token['kind'] = base
    elif damage == 'missing_profile':
        token['config'] = {}
    elif damage == 'missing_verification':
        token['verification'] = {}
    elif damage == 'wrong_verification':
        token['verification']['explanation_answer_profile'] = 'unknown'
    else:
        token['config']['explanation_answer_profile'] = 'unknown'
    with pytest.raises(ValueError, match='answer profile'):
        checked_token_answer_kind(token)


def test_example_changes_only_answer_order_and_description():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'configs'
    before = json.loads((root / 'search-distillation-sft-one-gpu-cpu-offload-with-coaching-v1.json').read_text())
    after = json.loads((root / 'search-distillation-sft-one-gpu-cpu-offload-with-coaching-explanation-first-v1.json').read_text())
    assert after.pop('explanation_answer_profile') == PROFILE
    before.pop('purpose')
    after.pop('purpose')
    assert before == after


@pytest.mark.parametrize('base', ['latent_only_search_distillation_data_preflight',
                                 'latent_only_reviewed_explanation_data_preflight'])
def test_typed_format_keeps_its_search_parent_and_capability_evaluation_family(base):
    from xqgeneral.search_parent import family
    from xqgeneral.clean_sft_evaluation import capability_kind
    legacy = dict(kind=base, status='complete')
    active = dict(kind=profiled_token_kind(base, PROFILE), status='complete',
                  config=dict(explanation_answer_profile=PROFILE),
                  verification=dict(explanation_answer_profile=PROFILE))
    assert family(active) == family(legacy)
    assert capability_kind(active) == capability_kind(legacy)
