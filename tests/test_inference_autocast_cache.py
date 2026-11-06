"""CPU framework-context and relocatable-asset contracts, not strength tests."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from test_model_bundle import training
from xqgeneral.evidence import atomic_json, digest
from xqgeneral.inference import Predictor
from xqgeneral.model_bundle import export_model, read_bundle
from xqgeneral.rules import START_FEN


MISSING = object()


def records():
    return [{'id': 1, 'fen': START_FEN, 'question': 'one'},
            {'id': 2, 'fen': START_FEN, 'question': 'two'}]


def controlled_predictor(setting):
    seen = []

    class Tokens(dict):
        def to(self, device):
            assert device == 'cpu'
            return self

    class Tokenizer:
        padding_side = 'right'
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]['content']

        def __call__(self, texts, **kwargs):
            return Tokens(input_ids=torch.tensor([[1, 2]] * len(texts)),
                          attention_mask=torch.ones((len(texts), 2), dtype=torch.long))

        def decode(self, tokens, **kwargs):
            assert tokens.tolist() == [3]
            return 'raw model output'

    class Decoder:
        def generate(self, **kwargs):
            seen.append(torch.is_autocast_cache_enabled())
            assert kwargs['max_new_tokens'] == 1
            return torch.cat([kwargs['input_ids'], torch.full((2, 1), 3)], dim=1)

    predictor = Predictor.__new__(Predictor)
    predictor.device = 'cpu'
    predictor.config = {} if setting is MISSING else {'autocast_cache_enabled': setting}
    predictor.tokenizer = Tokenizer()
    predictor.model = SimpleNamespace(base=Decoder(), board_context=lambda features: nullcontext())
    predictor.features = lambda record: [torch.full((1, 90, 4), float(record['id']))]
    return predictor, seen


@pytest.mark.parametrize('setting,expected', [(False, False), (True, True), (MISSING, True)])
def test_generation_uses_the_declared_real_torch_cache_context_and_restores_caller(setting, expected):
    predictor, seen = controlled_predictor(setting)
    with torch.autocast('cuda', enabled=False, cache_enabled=False):
        answers = predictor.generate_batch(records(), max_new_tokens=1)
        assert torch.is_autocast_cache_enabled() is False
    assert seen == [expected]
    assert answers == ['raw model output', 'raw model output']
    assert predictor.tokenizer.padding_side == 'right'


@pytest.mark.parametrize('memory', ['zero', 'shuffled'])
def test_declared_disabled_cache_also_applies_to_memory_controls(memory):
    predictor, seen = controlled_predictor(False)
    answers = predictor.generate_batch(records(), max_new_tokens=1, memory=memory)
    assert seen == [False] and answers == ['raw model output', 'raw model output']


def test_invalid_cache_policy_never_reaches_generation():
    predictor, seen = controlled_predictor('false')
    with pytest.raises(ValueError, match='autocast_cache_enabled must be a boolean'):
        predictor.generate_batch(records(), max_new_tokens=1)
    assert seen == []


@pytest.mark.parametrize('enabled', [False, True])
def test_explicit_inference_cache_override_reaches_torch_without_mutating_checkpoint(monkeypatch, enabled):
    from xqgeneral import inference
    controlled, seen = controlled_predictor(not enabled)
    saved = dict(controlled.config, mode='text_lora')
    monkeypatch.setattr(inference, 'load_checkpoint',
                        lambda *args: (controlled.model, controlled.tokenizer, saved))
    predictor = Predictor('controlled-no-download', device='cpu', autocast_cache_enabled=enabled)
    predictor.features = controlled.features
    assert predictor.generate_batch(records(), max_new_tokens=1) == ['raw model output'] * 2
    assert seen == [enabled]
    assert saved['autocast_cache_enabled'] is (not enabled)


@pytest.mark.parametrize('value', [0, 1, 'false'])
def test_invalid_inference_override_fails_before_loading_model(monkeypatch, value):
    from xqgeneral import inference
    def forbidden(*args):
        raise AssertionError('Invalid inference option reached model loading')
    monkeypatch.setattr(inference, 'load_checkpoint', forbidden)
    with pytest.raises(ValueError, match='autocast_cache_enabled'):
        Predictor('absent.pt', device='cpu', autocast_cache_enabled=value)


def set_training_policy(fixture, setting):
    run, expert, _, _, _ = fixture
    selected = torch.load(run / 'adapter.pt', weights_only=True)
    if setting is not MISSING:
        selected['config']['autocast_cache_enabled'] = setting
    torch.save(selected, run / 'adapter.pt')
    proof = json.loads((run / 'manifest.json').read_text())
    proof['config'] = selected['config']
    proof['outputs'][str(run / 'adapter.pt')] = {'sha256': digest(run / 'adapter.pt'),
                                               'bytes': (run / 'adapter.pt').stat().st_size}
    atomic_json(run / 'manifest.json', proof)
    return run, expert


@pytest.mark.parametrize('setting', [False, True, MISSING])
def test_export_and_relocation_preserve_explicit_policy_and_legacy_absence(training, tmp_path, setting):
    run, expert = set_training_policy(training, setting)
    original = tmp_path / 'bundle'
    export_model(run, expert, original)
    relocated = tmp_path / 'relocated'
    original.rename(relocated)
    loaded, proof = read_bundle(relocated)
    if setting is MISSING:
        assert 'autocast_cache_enabled' not in loaded['config']
    else:
        assert loaded['config']['autocast_cache_enabled'] is setting
        assert proof['config']['autocast_cache_enabled'] is setting


def test_invalid_declared_policy_rejected_before_export(training, tmp_path):
    run, expert = set_training_policy(training, 'false')
    output = tmp_path / 'invalid-bundle'
    with pytest.raises(ValueError, match='autocast_cache_enabled must be a boolean'):
        export_model(run, expert, output)
    assert not output.exists()


def test_rebound_bundle_with_invalid_policy_still_rejected(training, tmp_path):
    run, expert = set_training_policy(training, False)
    output = tmp_path / 'bundle'
    export_model(run, expert, output)
    proof = json.loads((output / 'manifest.json').read_text())
    config = json.loads((output / 'config.json').read_text())
    selected = torch.load(output / 'adapter.pt', weights_only=True)
    config['autocast_cache_enabled'] = 'false'
    selected['config'] = config
    torch.save(selected, output / 'adapter.pt')
    atomic_json(output / 'config.json', config)
    proof['config'] = config
    for name in ['config.json', 'adapter.pt']:
        proof['files'][name] = {'sha256': digest(output / name), 'bytes': (output / name).stat().st_size}
    atomic_json(output / 'manifest.json', proof)
    with pytest.raises(ValueError, match='autocast_cache_enabled must be a boolean'):
        read_bundle(output)
