import json
from types import SimpleNamespace

import pytest
import torch

from xqgeneral import local_teacher
from xqgeneral.evidence import digest, load_jsonl, write_jsonl


class TensorInputs(dict):
    def to(self, device):
        return TensorInputs({key: tensor.to(device) for key, tensor in self.items()})


class Tokenizer:
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return messages[0]['content']

    def __call__(self, texts, **kwargs):
        assert kwargs == dict(add_special_tokens=False, padding=True, padding_side='left',
                              return_attention_mask=True, return_tensors='pt')
        rows = [[int(x) for x in text.split()] for text in texts]
        width = max(map(len, rows))
        return TensorInputs({'input_ids': torch.tensor([[0] * (width - len(r)) + r for r in rows]),
            'attention_mask': torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])})

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return ' '.join(str(int(x)) for x in tokens if int(x) not in (8, 9))


class Model:
    device = torch.device('cpu')

    def __init__(self, generated, eos):
        self.generated = torch.tensor(generated)
        self.generation_config = SimpleNamespace(eos_token_id=eos)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.cat((kwargs['input_ids'], self.generated), dim=1)


def consolidator(generated, eos=9):
    teacher = local_teacher.LocalConsolidator.__new__(local_teacher.LocalConsolidator)
    teacher.tokenizer = Tokenizer()
    teacher.model = Model(generated, eos)
    return teacher


@pytest.mark.parametrize('eos,first_stop', [(9, 9), ([9, 8], 8), (None, 9)])
def test_mixed_length_prompts_and_early_eos_never_decode_padding(eos, first_stop):
    teacher = consolidator([[31, first_stop, 9, 9], [32, 33, 34, 9]], eos)
    result = teacher.generate_batch([[{'content': '10 11'}], [{'content': '12 13 14 15'}]], 4)
    assert result == [dict(text='31', prompt_tokens=2, generated_tokens=2, hit_generation_limit=False),
                      dict(text='32 33 34', prompt_tokens=4, generated_tokens=4, hit_generation_limit=True)]
    call = teacher.model.calls[0]
    assert call['input_ids'].tolist() == [[0, 0, 10, 11], [12, 13, 14, 15]]
    assert call['attention_mask'].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert call['do_sample'] is False and call['return_dict_in_generate'] is False


def test_generation_limit_without_eos_and_empty_batch():
    teacher = consolidator([[31, 32, 33]])
    assert teacher.generate_batch([], 3) == []
    assert not teacher.model.calls
    assert teacher.generate_batch([[{'content': '10 11'}]], 3) == [
        dict(text='31 32 33', prompt_tokens=2, generated_tokens=3, hit_generation_limit=True)]


@pytest.mark.parametrize('budget', [0, -1, True, 1.5])
def test_invalid_budget_before_model_call(budget):
    teacher = consolidator([[31]])
    with pytest.raises(ValueError, match='positive integer'):
        teacher.generate_batch([[{'content': '10'}]], budget)
    assert not teacher.model.calls


def cli_fixture(tmp_path, monkeypatch, count=5):
    config = tmp_path / 'config.json'
    pinned = {'repository': 'fixture-only', 'revision': 'cpu', 'quantization': 'none',
              'inference_dtype': 'bfloat16', 'model_path': str(tmp_path / 'fixture-model')}
    model_root = tmp_path / 'fixture-model'
    model_root.mkdir()
    (model_root / 'weights.manifest.json').write_text('{"fixture_only": true}')
    config.write_text(json.dumps({'search_consolidator': pinned}))
    queries = tmp_path / 'queries.jsonl'
    write_jsonl(queries, [{'id': f'q{i}', 'messages': [{'role': 'user', 'content': f'm{i}'}]}
                         for i in range(count)])
    output = tmp_path / 'out'
    calls, loads = [], []

    class Teacher:
        identity = {'fixture': 'CPU protocol control, no neural model'}
        interrupt = False
        incomplete = False

        def __init__(self, cfg):
            assert cfg == pinned
            loads.append(cfg)

        def generate(self, messages, budget):
            calls.append(('single', [messages[0]['content']]))
            return self.response(messages, budget)

        def generate_batch(self, messages, budget):
            calls.append(('batch', [m[0]['content'] for m in messages]))
            if self.interrupt and messages[0][0]['content'] == 'm2':
                raise RuntimeError('planned interruption')
            results = [self.response(m, budget) for m in messages]
            return results[:-1] if self.incomplete else results

        def response(self, messages, budget):
            return dict(text=messages[0]['content'], prompt_tokens=3, generated_tokens=2,
                        hit_generation_limit=False)

    monkeypatch.setattr(local_teacher, 'LocalConsolidator', Teacher)

    def run(batch=None, budget=4):
        args = ['local_teacher', '--config', str(config), '--input', str(queries),
                '--output', str(output), '--max-new-tokens', str(budget)]
        if batch is not None:
            args += ['--batch-size', str(batch)]
        monkeypatch.setattr('sys.argv', args)
        local_teacher.main()

    return SimpleNamespace(run=run, calls=calls, loads=loads, teacher=Teacher,
        config=config, pinned=pinned, queries=queries, output=output)


def test_batched_interruption_reuses_exact_prefix_and_covers_tail(tmp_path, monkeypatch):
    f = cli_fixture(tmp_path, monkeypatch)
    f.teacher.interrupt = True
    with pytest.raises(RuntimeError, match='planned interruption'):
        f.run(2)
    partial = f.output / 'responses.partial.jsonl'
    prefix = partial.read_bytes()
    assert [r['id'] for r in load_jsonl(partial)] == ['q0', 'q1']
    assert not (f.output / 'manifest.json').exists()
    f.teacher.interrupt = False
    f.calls.clear()
    f.run(2)
    assert f.calls == [('batch', ['m2', 'm3']), ('batch', ['m4'])]
    assert partial.read_bytes().startswith(prefix)
    results = load_jsonl(f.output / 'responses.jsonl')
    assert [(r['id'], r['text']) for r in results] == [(f'q{i}', f'm{i}') for i in range(5)]
    proof = json.loads((f.output / 'manifest.json').read_text())
    assert proof['config']['batch_size'] == 2
    assert proof['verification']['responses_reused'] == 2
    assert proof['verification']['new_teacher_queries'] == 3
    assert proof['verification']['generation_batches_this_invocation'] == 2
    assert proof['inputs'][str(f.queries)]['sha256'] == digest(f.queries)
    assert proof['outputs'][str(f.output / 'responses.jsonl')]['sha256'] == digest(f.output / 'responses.jsonl')


@pytest.mark.parametrize('batch', [None, 1])
def test_default_and_explicit_single_keep_existing_contract(tmp_path, monkeypatch, batch):
    f = cli_fixture(tmp_path, monkeypatch, 2)
    f.run(batch)
    assert f.calls == [('single', ['m0']), ('single', ['m1'])]
    contract = json.loads((f.output / 'contract.json').read_text())
    assert contract == dict(input_sha256=digest(f.queries), teacher=f.pinned, max_new_tokens=4)
    proof = json.loads((f.output / 'manifest.json').read_text())
    assert 'batch_size' not in proof['config']
    assert proof['verification']['configured_batch_size'] == 1


@pytest.mark.parametrize('batch,budget', [(0, 4), (-1, 4), (2, 0), (2, -1)])
def test_invalid_cli_budget_never_loads_teacher(tmp_path, monkeypatch, batch, budget):
    f = cli_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        f.run(batch, budget)
    assert not f.loads and not f.output.exists()


@pytest.mark.parametrize('changed', ['batch', 'input'])
def test_resume_rejects_changed_generation_contract_before_model(tmp_path, monkeypatch, changed):
    f = cli_fixture(tmp_path, monkeypatch)
    f.teacher.interrupt = True
    with pytest.raises(RuntimeError):
        f.run(2)
    loads = len(f.loads)
    if changed == 'input':
        with f.queries.open('a') as handle:
            handle.write(json.dumps({'id': 'extra', 'messages': []}) + '\n')
    with pytest.raises(ValueError, match='inputs changed'):
        f.run(3 if changed == 'batch' else 2)
    assert len(f.loads) == loads


def test_incomplete_batch_rejected_without_persisting_misaligned_ids(tmp_path, monkeypatch):
    f = cli_fixture(tmp_path, monkeypatch)
    f.teacher.incomplete = True
    with pytest.raises(RuntimeError, match='coverage'):
        f.run(2)
    assert not load_jsonl(f.output / 'responses.partial.jsonl')
    assert not (f.output / 'manifest.json').exists()
