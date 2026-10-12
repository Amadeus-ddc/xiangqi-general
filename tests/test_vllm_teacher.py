"""CPU protocol controls; actual neural measurements are separate evidence."""
import json
import sys
from types import SimpleNamespace

import pytest

from xqgeneral import local_teacher, vllm_teacher
from xqgeneral.evidence import digest, load_jsonl, write_jsonl
from xqgeneral.teacher_generation import checked_response


@pytest.fixture
def backend(tmp_path, monkeypatch):
    root = tmp_path / 'model'
    root.mkdir()
    for name, content in [('config.json', json.dumps({'text_config': {'max_position_embeddings': 64},
                                                     'vision_config': {}})),
                          ('tokenizer.json', '{}'), ('tokenizer_config.json', '{}'),
                          ('model-1.safetensors', 'CPU fixture, not neural weights')]:
        (root / name).write_text(content)
    shard = root / 'model-1.safetensors'
    proof = {'repository': 'fixture-only', 'revision': 'cpu', 'quantization': 'none',
        'all_official_lfs_sha256_matched': True, 'stored_tensor_dtypes': {'BF16': 1},
        'total_weight_bytes': shard.stat().st_size,
        'shards': {shard.name: {'bytes': shard.stat().st_size, 'sha256': digest(shard)}}}
    (root / 'weights.manifest.json').write_text(json.dumps(proof))
    profile = dict(do_sample=True, temperature=1.0, top_k=0, top_p=0.95,
                   max_context_tokens=12, max_new_tokens=8, reasoning_effort='low')
    config = dict(repository='fixture-only', revision='cpu', model_path=str(root),
                  quantization='none', inference_dtype='bfloat16', max_context_tokens=12,
                  generation_profile=profile)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, reasoning_effort='low')
            return messages[0]['content']

        def __call__(self, text, **kwargs):
            assert kwargs == dict(add_special_tokens=False)
            return {'input_ids': [int(token) for token in text.split()]}

        def decode(self, tokens, skip_special_tokens):
            assert skip_special_tokens
            return ''.join({31: '<think>analysis', 32: '</think> raw answer'}[token] for token in tokens)

    class Engine:
        instances = []
        interrupt_on_generation = None
        malformed = None
        quantized = False

        def __init__(self, **kwargs):
            self.options = kwargs
            self.calls = []
            self.closed = 0
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=self.shutdown),
                vllm_config=SimpleNamespace(model_config=SimpleNamespace(quantization=None)))
            self.instances.append(self)

        def shutdown(self, timeout):
            assert timeout == 20
            self.closed += 1

        def collective_rpc(self, name):
            assert name == 'xqgeneral_teacher_identity'
            return [{'parameter_elements_by_dtype': {'torch.bfloat16': 32, 'torch.float32': 2},
                'quant_methods': ['QuantizedLinear' if self.quantized else 'UnquantizedLinearMethod'],
                'model_quant_config_is_none': True, 'model_class': 'CPUProtocolFixture',
                'vision_tower_present': True}]

        def generate(self, prompts, parameters, use_tqdm):
            assert use_tqdm is False
            self.calls.append((prompts, parameters))
            if len(self.calls) == self.interrupt_on_generation:
                raise RuntimeError('planned interruption')
            outputs = []
            for prompt, parameter in zip(prompts, parameters, strict=True):
                tokens = [31, 32][:parameter.max_tokens]
                outputs.append(SimpleNamespace(prompt_token_ids=list(prompt['prompt_token_ids']),
                    outputs=[SimpleNamespace(token_ids=tokens,
                        finish_reason='length' if len(tokens) == parameter.max_tokens else 'stop')]))
            if self.malformed == 'prefix':
                outputs[0].prompt_token_ids = [999]
            elif self.malformed == 'coverage':
                outputs.pop()
            elif self.malformed == 'overflow':
                outputs[0].outputs[0].token_ids = [31] * (parameters[0].max_tokens + 1)
            return outputs

    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())))
    monkeypatch.setattr(vllm_teacher, 'version', lambda name: 'cpu-fixture-only')
    monkeypatch.setattr(local_teacher, 'version', lambda name: 'cpu-fixture-only')
    return SimpleNamespace(config=config, profile=profile, root=root, engine=Engine)


def test_uneven_context_budgets_preserve_every_input_and_raw_reasoning(backend):
    with local_teacher.teacher_session(backend.config, 'vllm') as teacher:
        rows = teacher.generate_batch([[{'content': '1 2'}], [{'content': ' '.join(map(str, range(10)))}]],
                                      8, sampling_seed=123)
        assert [row['prompt_tokens'] for row in rows] == [2, 10]
        assert [row['effective_max_new_tokens'] for row in rows] == [8, 2]
        assert [row['context_limited_output'] for row in rows] == [False, True]
        assert [row['hit_generation_limit'] for row in rows] == [False, True]
        assert all(row['raw_text'] == '<think>analysis</think> raw answer' and
                   row['text'] == 'raw answer' for row in rows)
        prompts, parameters = teacher.model.calls[0]
        assert [prompt['prompt_token_ids'] for prompt in prompts] == [[1, 2], list(range(10))]
        assert [parameter.max_tokens for parameter in parameters] == [8, 2]
        assert all(parameter.top_k == -1 for parameter in parameters)
        assert [row['request_sampling_seed'] for row in rows] == [
            vllm_teacher.request_seed(123, 0), vllm_teacher.request_seed(123, 1)]
        assert len({row['request_sampling_seed'] for row in rows}) == 2
        for row in rows:
            checked_response(row, 8, 12, backend.profile)
    assert backend.engine.instances[0].closed == 1


@pytest.mark.parametrize('case', ['context', 'batch', 'seed'])
def test_invalid_request_never_generates_or_silently_truncates(backend, case):
    with local_teacher.teacher_session(backend.config, 'vllm') as teacher:
        messages = [[{'content': '1 2'}]]
        seed = 123
        if case == 'context':
            messages = [[{'content': ' '.join(map(str, range(12)))}]]
        elif case == 'batch':
            messages *= 17
        else:
            seed = None
        with pytest.raises(ValueError):
            teacher.generate_batch(messages, 8, sampling_seed=seed)
        assert not teacher.model.calls
    assert backend.engine.instances[0].closed == 1


@pytest.mark.parametrize('case', ['prefix', 'coverage', 'overflow'])
def test_invalid_engine_outputs_fail_and_release_owned_engine(backend, case):
    backend.engine.malformed = case
    with pytest.raises(RuntimeError):
        with local_teacher.teacher_session(backend.config, 'vllm') as teacher:
            teacher.generate_batch([[{'content': '1 2'}]], 8, sampling_seed=123)
    assert backend.engine.instances[0].closed == 1


def test_modified_weight_bytes_fail_before_engine_loading(backend):
    (backend.root / 'model-1.safetensors').write_text('changed')
    with pytest.raises(ValueError, match='Changed handoff evidence'):
        vllm_teacher.VllmConsolidator(backend.config)
    assert not backend.engine.instances


def test_loaded_quantization_is_rejected_and_cleaned_up(backend):
    backend.engine.quantized = True
    with pytest.raises(ValueError, match='unquantized BF16 contract'):
        vllm_teacher.VllmConsolidator(backend.config)
    assert backend.engine.instances[0].closed == 1


def test_cli_interruption_resume_and_backend_seed_tampering(backend, tmp_path, monkeypatch):
    config, generation, queries, output = [tmp_path / name for name in
                                         ['teachers.json', 'generation.json', 'queries.jsonl', 'out']]
    teacher_config = {key: value for key, value in backend.config.items()
                      if key not in {'generation_profile', 'max_context_tokens'}}
    config.write_text(json.dumps({'search_consolidator': teacher_config}))
    generation.write_text(json.dumps(backend.profile))
    write_jsonl(queries, [{'id': f'q{i}', 'messages': [{'role': 'user', 'content': '1 2'}]}
                         for i in range(3)])
    def run(which='vllm'):
        monkeypatch.setattr(sys, 'argv', ['teacher', '--config', str(config), '--input', str(queries),
            '--output', str(output), '--max-new-tokens', '8', '--batch-size', '2',
            '--generation-config', str(generation), '--seed', '123', '--backend', which])
        local_teacher.main()
    backend.engine.interrupt_on_generation = 2
    with pytest.raises(RuntimeError, match='planned interruption'):
        run()
    partial = output / 'responses.partial.jsonl'
    prefix = partial.read_bytes()
    assert len(load_jsonl(partial)) == 2 and backend.engine.instances[0].closed == 1
    assert not (output / 'manifest.json').exists()
    with pytest.raises(ValueError, match='continuation inputs changed'):
        run('transformers')
    assert len(backend.engine.instances) == 1
    rows = load_jsonl(partial)
    rows[0]['request_sampling_seed'] += 1
    write_jsonl(partial, rows)
    with pytest.raises(ValueError, match='vLLM request identity changed'):
        run()
    assert len(backend.engine.instances) == 1
    partial.write_bytes(prefix)
    backend.engine.interrupt_on_generation = None
    run()
    assert partial.read_bytes().startswith(prefix)
    assert [row['id'] for row in load_jsonl(output / 'responses.jsonl')] == ['q0', 'q1', 'q2']
    assert len(backend.engine.instances[1].calls) == 1 and backend.engine.instances[1].closed == 1
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['config']['backend'] == 'vllm'
    assert proof['verification']['teacher_engine_closed'] is True
    assert proof['verification']['responses_reused'] == 2
    assert proof['verification']['new_teacher_queries'] == 1
    assert proof['verification']['request_seed_profile'] == vllm_teacher.REQUEST_SEED_PROFILE
