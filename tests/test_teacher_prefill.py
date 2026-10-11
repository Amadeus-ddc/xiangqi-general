"""Teacher calls must bound prefill tensors without dropping prompt tokens."""
from types import SimpleNamespace
import json
import sys

import pytest
import torch

from xqgeneral.local_teacher import LocalConsolidator
from xqgeneral.teacher_prefill import checked_prefill_chunk_size, generate_prefilled


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class Tokenizer:
    eos_token_id = 999999

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def __call__(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        sizes = [len(t) for t in texts]
        width = max(sizes)
        ids = torch.zeros(len(sizes), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, size in enumerate(sizes):
            ids[i, -size:] = torch.arange(1, size + 1)
            mask[i, -size:] = 1
        return Batch(input_ids=ids, attention_mask=mask)

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids.tolist()))


class BoundedModel:
    device = torch.device('cpu')
    generation_config = SimpleNamespace(eos_token_id=999999)

    def __init__(self, width=64):
        self.width, self.prefills, self.generations = width, [], []

    def __call__(self, input_ids, attention_mask, past_key_values=None, **kwargs):
        assert input_ids.shape[1] <= self.width
        prior = torch.empty(input_ids.shape[0], 0, dtype=torch.long) if past_key_values is None else past_key_values.ids
        merged = torch.cat([prior, input_ids], dim=1)
        assert attention_mask.shape == merged.shape
        assert kwargs['use_cache'] is True and kwargs['logits_to_keep'] == 1
        positions = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
        assert torch.equal(kwargs['position_ids'], positions[:, -input_ids.shape[1]:])
        self.prefills.append({'ids': input_ids.clone(), 'mask': attention_mask.clone()})
        return SimpleNamespace(past_key_values=SimpleNamespace(ids=merged, get_seq_length=lambda: merged.shape[1]))

    def generate(self, input_ids, **kwargs):
        cache = kwargs.get('past_key_values')
        if cache is None and input_ids.shape[1] > self.width:
            raise RuntimeError('prefill tensor exceeds the available memory')
        if cache is not None:
            size = cache.get_seq_length()
            assert torch.equal(cache.ids, input_ids[:, :size]), 'Prompt prefix was changed or dropped'
            assert 0 < input_ids.shape[1] - size <= self.width
            positions = (kwargs['attention_mask'].long().cumsum(-1) - 1).masked_fill(kwargs['attention_mask'] == 0, 0)
            assert torch.equal(kwargs['position_ids'], positions), 'Full text positions were changed'
        self.generations.append({'ids': input_ids.clone(), **kwargs})
        return torch.cat([input_ids, torch.tensor([[7, 8]]).expand(input_ids.shape[0], -1)], dim=1)


def teacher(chunk=64):
    item = LocalConsolidator.__new__(LocalConsolidator)
    item.tokenizer, item.model = Tokenizer(), BoundedModel()
    item.max_context_tokens, item.generation_profile = 512, None
    item.prefill_chunk_size = chunk
    return item


def messages(size):
    return [{'role': 'user', 'content': 'x' * size}]


def test_single_teacher_bounds_prefill_and_keeps_complete_generation_input():
    item = teacher()
    answer = item.generate(messages(200), 32)
    assert answer['prompt_tokens'] == 200 and answer['generated_tokens'] == 2
    assert [x['ids'].shape[1] for x in item.model.prefills] == [64, 64, 64]
    assert item.model.generations[0]['ids'].shape == (1, 200)
    assert item.model.generations[0]['max_new_tokens'] == 32


def test_batch_teacher_bounds_padding_before_cached_linear_attention():
    item = teacher()
    answers = item.generate_batch([messages(200), messages(100), messages(190)], 32)
    assert [a['prompt_tokens'] for a in answers] == [200, 100, 190]
    assert [g['ids'].shape for g in item.model.generations] == [(2, 200), (1, 100)]
    assert all(int((g['attention_mask'] == 0).sum(1).max()) < 64 for g in item.model.generations)
    assert [a['text'] for a in answers] == ['7 8'] * 3


@pytest.mark.parametrize('size', [0, -1, True, False, 1.5, '64', [], {}])
def test_invalid_prefill_setting_rejected_without_loading_models(size):
    with pytest.raises(ValueError, match='prefill chunk size'):
        LocalConsolidator({'prefill_chunk_size': size})


def test_disabled_or_short_prefill_keeps_original_generation_call():
    for size in [None, 64]:
        model = BoundedModel()
        inputs = Tokenizer()('x' * 32)
        result = generate_prefilled(model, inputs, size, do_sample=False, max_new_tokens=2)
        assert result.shape == (1, 34) and not model.prefills
        assert 'past_key_values' not in model.generations[0]


@pytest.mark.parametrize('mask', [[0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 2]])
def test_invalid_padding_refused_before_neural_calls(mask):
    model = BoundedModel()
    inputs = {'input_ids': torch.ones(1, 4, dtype=torch.long), 'attention_mask': torch.tensor([mask])}
    with pytest.raises(ValueError, match='contiguous left padding'):
        generate_prefilled(model, inputs, 8, max_new_tokens=2)
    assert not model.prefills and not model.generations


def test_cached_chunk_cannot_start_with_a_fully_padded_row():
    model = BoundedModel()
    inputs = Tokenizer()(['x' * 200, 'x' * 100])
    with pytest.raises(ValueError, match='grouped'):
        generate_prefilled(model, inputs, 64, max_new_tokens=2)
    assert not model.prefills and not model.generations


@pytest.mark.parametrize('defect', ['missing', 'short', 'duplicated'])
def test_wrong_cache_length_refused_before_generation(defect):
    model = BoundedModel()
    base = model.__class__
    class Defective(base):
        def __call__(self, *args, **kwargs):
            output = super().__call__(*args, **kwargs)
            if defect == 'missing':
                return SimpleNamespace(past_key_values=None)
            delta = -1 if defect == 'short' else 1
            output.past_key_values.get_seq_length = lambda: output.past_key_values.ids.shape[1] + delta
            return output
    model = Defective()
    with pytest.raises(RuntimeError, match='exact complete input prefix'):
        generate_prefilled(model, Tokenizer()('x' * 200), 64, max_new_tokens=2)
    assert not model.generations


@pytest.mark.parametrize('option', [{'use_cache': False}, {'past_key_values': object()}])
def test_prefill_requires_its_own_enabled_cache(option):
    model = BoundedModel()
    with pytest.raises(ValueError, match='fresh enabled cache'):
        generate_prefilled(model, Tokenizer()('x' * 200), 64, max_new_tokens=2, **option)
    assert not model.prefills and not model.generations


def test_context_overflow_is_rejected_before_any_prefill():
    item = teacher()
    with pytest.raises(ValueError, match='no output room'):
        item.generate(messages(512), 32)
    assert not item.model.prefills and not item.model.generations


def test_prefill_keeps_the_remaining_context_output_cap():
    item = teacher()
    result = item.generate(messages(510), 32)
    assert result['effective_max_new_tokens'] == 2 and result['context_limited_output'] is True
    assert item.model.generations[0]['max_new_tokens'] == 2
    prefix = torch.cat([p['ids'] for p in item.model.prefills], dim=1)
    assert torch.equal(prefix, item.model.generations[0]['ids'][:, :prefix.shape[1]])
    assert 0 < 510 - prefix.shape[1] <= 64


def cli_fixture(tmp_path, monkeypatch, chunk=64):
    from xqgeneral import local_teacher
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'weights.manifest.json').write_text('{}')
    config = tmp_path / 'teachers.json'
    config.write_text(json.dumps({'search_consolidator': {'model_path': str(model)}}))
    queries = tmp_path / 'queries.jsonl'
    queries.write_text(json.dumps({'id': 'fixture', 'messages': messages(4)}) + '\n')
    output = tmp_path / 'output'
    loaded = []
    class Stub:
        def __init__(self, config):
            loaded.append(config)
            self.identity = {} if 'prefill_chunk_size' not in config else {'prefill_chunk_size': config['prefill_chunk_size']}
        def generate(self, messages, budget):
            return {'text': 'fixture output', 'prompt_tokens': 4, 'generated_tokens': 2, 'hit_generation_limit': False}
    monkeypatch.setattr(local_teacher, 'LocalConsolidator', Stub)
    monkeypatch.setattr(local_teacher, 'generation_software', lambda: {'controlled_fixture': 'not_neural'})
    argv = ['local_teacher', '--config', str(config), '--input', str(queries), '--output', str(output), '--max-new-tokens', '8']
    if chunk is not None:
        argv += ['--prefill-chunk-size', str(chunk)]
    monkeypatch.setattr(sys, 'argv', argv)
    local_teacher.main()
    return local_teacher, output, loaded, argv


def test_cli_binds_prefill_configuration_and_model_identity(tmp_path, monkeypatch):
    _, output, loaded, _ = cli_fixture(tmp_path, monkeypatch)
    contract = json.loads((output / 'contract.json').read_text())
    proof = json.loads((output / 'manifest.json').read_text())
    assert loaded[0]['prefill_chunk_size'] == contract['teacher']['prefill_chunk_size'] == 64
    assert 'code' in contract and 'generation_software' in contract
    assert proof['config']['prefill_chunk_size'] == proof['verification']['prefill_chunk_size'] == 64
    assert json.loads((output / 'responses.jsonl').read_text())['teacher']['prefill_chunk_size'] == 64


@pytest.mark.parametrize('change', ['argument', 'cached_identity', 'boolean_identity'])
def test_prefill_resume_tampering_refused_before_model_load(tmp_path, monkeypatch, change):
    module, output, loaded, argv = cli_fixture(tmp_path, monkeypatch)
    (output / 'manifest.json').unlink()
    if change == 'argument':
        argv[-1] = '32'
    else:
        response = json.loads((output / 'responses.partial.jsonl').read_text())
        response['teacher']['prefill_chunk_size'] = True if change == 'boolean_identity' else 32
        (output / 'responses.partial.jsonl').write_text(json.dumps(response) + '\n')
    with pytest.raises(ValueError, match='continuation'):
        module.main()
    assert len(loaded) == 1


def test_absent_prefill_cli_preserves_legacy_contract(tmp_path, monkeypatch):
    _, output, loaded, _ = cli_fixture(tmp_path, monkeypatch, None)
    contract = json.loads((output / 'contract.json').read_text())
    proof = json.loads((output / 'manifest.json').read_text())
    assert 'prefill_chunk_size' not in loaded[0]
    assert 'prefill_chunk_size' not in proof['config']
    assert 'prefill_chunk_size' not in proof['verification']
    assert 'code' not in contract and 'generation_software' not in contract
