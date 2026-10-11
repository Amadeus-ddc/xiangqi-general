"""Context limits, reasoning boundaries and immutable consolidation decoding."""
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from test_local_teacher import Model, TensorInputs, cli_fixture
from test_search_pilot import ControlledChain, setup_pilot
from test_search_collection import query, response, recorded_collection, collect_args
from xqgeneral import collect_search, local_teacher, search_pilot
from xqgeneral.evidence import atomic_json, load_jsonl
from xqgeneral.teacher_generation import generation_profile


PROFILE = dict(do_sample=True, temperature=1.0, top_k=20, top_p=0.95,
               max_context_tokens=16, max_new_tokens=4, reasoning_effort='low')


class Tokenizer:
    eos_token_id = 9
    raw = None

    def __init__(self, thinking=False):
        self.thinking = thinking

    def apply_chat_template(self, messages, **kwargs):
        expected = {'reasoning_effort': 'low'} if self.thinking else {'enable_thinking': False}
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, **expected)
        return messages[0]['content']

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            assert kwargs == dict(add_special_tokens=False, return_tensors='pt')
            texts = [texts]
        else:
            assert kwargs == dict(add_special_tokens=False, padding=True, padding_side='left',
                                  return_attention_mask=True, return_tensors='pt')
        rows = [[int(x) for x in text.split()] for text in texts]
        width = max(map(len, rows))
        return TensorInputs(input_ids=torch.tensor([[0] * (width - len(r)) + r for r in rows]),
                            attention_mask=torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows]))

    def decode(self, tokens, skip_special_tokens):
        return self.raw if self.raw is not None else ' '.join(str(int(x)) for x in tokens if int(x) != 9)


def decoder(context, *, profile=None, generated=None):
    teacher = local_teacher.LocalConsolidator.__new__(local_teacher.LocalConsolidator)
    teacher.max_context_tokens, teacher.generation_profile = context, profile
    teacher.tokenizer = Tokenizer(thinking=profile is not None)
    class AdaptiveModel(Model):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            indices = [min(0 if int(row[row != 0][0]) == 10 else 1, len(self.generated) - 1)
                       for row in kwargs['input_ids']]
            return torch.cat((kwargs['input_ids'], self.generated[indices]), dim=1)
    teacher.model = AdaptiveModel(generated or [[31, 9]], 9)
    return teacher


@pytest.mark.parametrize('batched', [False, True])
def test_each_prompt_keeps_its_own_context_budget_and_overlong_prompts_never_reach_model(batched):
    prompts = [[{'content': '10 11'}], [{'content': '12 13 14 15'}]]
    teacher = decoder(8, generated=[[31, 9], [32, 9]] if batched else [[31, 9]])
    call = (lambda: teacher.generate_batch(prompts, 4)) if batched else (
           lambda: teacher.generate(prompts[1], 4))
    result = call()
    rows = result if batched else [result]
    assert [r['prompt_tokens'] for r in rows] == ([2, 4] if batched else [4])
    assert all(r['padded_prompt_tokens'] == 4 for r in rows)
    teacher.max_context_tokens = 7
    teacher.model.calls.clear()
    result = call()
    rows = result if batched else [result]
    assert [r['effective_max_new_tokens'] for r in rows] == ([4, 3] if batched else [3])
    assert [r['padded_prompt_tokens'] for r in rows] == ([2, 4] if batched else [4])
    assert [c['max_new_tokens'] for c in teacher.model.calls] == ([4, 3] if batched else [3])
    assert [r['text'] for r in rows] == (['31', '32'] if batched else ['31'])
    teacher.model.calls.clear()
    teacher.max_context_tokens = 5
    prompts[1][0]['content'] = '12 13 14 15 16'
    with pytest.raises(ValueError, match='no prompt was truncated'):
        call()
    assert teacher.model.calls == []


@pytest.mark.parametrize('key,value', [('max_context_tokens', True), ('max_context_tokens', 4),
    ('max_new_tokens', 0), ('temperature', float('nan')), ('top_k', True),
    ('top_p', 0), ('reasoning_effort', []), ('reasoning_effort', 'unknown')])
def test_malformed_profile_rejected_before_any_model_load(key, value):
    profile = dict(PROFILE, **{key: value})
    with pytest.raises(ValueError):
        local_teacher.LocalConsolidator({'generation_profile': profile})


def test_configured_context_exceeding_model_capacity_rejected_before_weights(tmp_path, monkeypatch):
    class ForbiddenLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            pytest.fail('An incompatible context reached model or tokenizer loading')
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForMultimodalLM=ForbiddenLoader, AutoTokenizer=ForbiddenLoader))
    (tmp_path / 'config.json').write_text('{"text_config": {"max_position_embeddings": 8}}')
    with pytest.raises(ValueError, match='pinned model capacity'):
        local_teacher.LocalConsolidator(dict(model_path=str(tmp_path), max_context_tokens=16,
            quantization='none', inference_dtype='bfloat16'))


@pytest.mark.parametrize('completed', [False, True])
def test_low_reasoning_and_sampling_keep_raw_text_and_only_strip_finished_thought(completed):
    teacher = decoder(16, profile=PROFILE)
    raw = ' 思考中的错误 JSON {"explanation":"错"} '
    if completed:
        raw += '\n</think>\n {"explanation":"最后的讲解"}\n '
    teacher.tokenizer.raw = raw
    value = teacher.generate_batch([[{'content': '10 11'}]], 4)[0]
    assert value['raw_text'] == raw and value['reasoning_completed'] is completed
    assert value['text'] == ('{"explanation":"最后的讲解"}' if completed else raw.strip())
    call = teacher.model.calls[0]
    assert {key: call[key] for key in ['do_sample', 'temperature', 'top_k', 'top_p']} == dict(
        do_sample=True, temperature=1.0, top_k=20, top_p=0.95)
    with pytest.raises(ValueError, match='generation profile'):
        teacher.generate_batch([[{'content': '10'}]], 5)
    assert len(teacher.model.calls) == 1


def profile_cli(tmp_path, monkeypatch):
    f = cli_fixture(tmp_path, monkeypatch)
    config = tmp_path / 'generation.json'
    atomic_json(config, PROFILE)
    f.pinned.update(max_context_tokens=16, generation_profile=PROFILE)
    f.teacher.model = SimpleNamespace(device=torch.device('cpu'))

    def generated(self, messages, budget):
        raw = f'{torch.rand(1).item()}\n</think>\n{messages[0]["content"]}'
        return dict(text=messages[0]['content'], raw_text=raw, reasoning_completed=True,
                    prompt_tokens=3, padded_prompt_tokens=3, generated_tokens=2, hit_generation_limit=False,
                    effective_max_new_tokens=budget, context_limited_output=False)
    f.teacher.response = generated

    def run(**overrides):
        values = dict(config=f.config, input=f.queries, output=f.output, max_new_tokens=4,
                      batch_size=2, generation_config=config, seed=72)
        values.update(overrides)
        monkeypatch.setattr('sys.argv', ['local_teacher', *[part for key, value in values.items()
            if value is not None for part in ('--' + key.replace('_', '-'), str(value))]])
        local_teacher.main()
    f.profile_path, f.run_profile = config, run
    return f


def test_sampled_continuation_keeps_prefix_and_restores_callers_rng(tmp_path, monkeypatch):
    f = profile_cli(tmp_path, monkeypatch)
    original_rng = torch.get_rng_state().clone()
    f.teacher.interrupt = True
    with pytest.raises(RuntimeError, match='interruption'):
        f.run_profile()
    partial = f.output / 'responses.partial.jsonl'
    prefix = partial.read_bytes()
    assert torch.equal(torch.get_rng_state(), original_rng)
    f.teacher.interrupt = False
    f.run_profile()
    assert partial.read_bytes().startswith(prefix) and torch.equal(torch.get_rng_state(), original_rng)
    rows = load_jsonl(f.output / 'responses.jsonl')
    assert [r['id'] for r in rows] == [f'q{i}' for i in range(5)]
    assert rows[0]['sampling_seed'] == rows[1]['sampling_seed'] != rows[2]['sampling_seed']
    assert rows[4]['generation_batch_ids'] == ['q4']
    proof = json.loads((f.output / 'manifest.json').read_text())
    assert proof['verification']['responses_reused'] == 2
    assert proof['verification']['max_context_tokens'] == 16
    assert proof['verification']['generation_profile'] == PROFILE
    assert str(f.profile_path) in proof['inputs']


@pytest.mark.parametrize('change', ['profile', 'seed', 'raw_text', 'tokens'])
def test_changed_profile_or_cached_response_refused_before_reload(tmp_path, monkeypatch, change):
    f = profile_cli(tmp_path, monkeypatch)
    f.teacher.interrupt = True
    with pytest.raises(RuntimeError):
        f.run_profile()
    loads = len(f.loads)
    overrides = {}
    if change == 'profile':
        atomic_json(f.profile_path, dict(PROFILE, temperature=0.7))
    elif change == 'seed':
        overrides['seed'] = 73
    else:
        partial = f.output / 'responses.partial.jsonl'
        rows = load_jsonl(partial)
        rows[0]['text' if change == 'raw_text' else 'padded_prompt_tokens'] = 'changed' if change == 'raw_text' else 15
        partial.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        f.run_profile(**overrides)
    assert len(f.loads) == loads


@pytest.mark.parametrize('override', [{'seed': None}, {'max_new_tokens': 3}, {'max_context_tokens': 17}])
def test_cli_inconsistent_generation_settings_never_load_teacher(tmp_path, monkeypatch, override):
    f = profile_cli(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        f.run_profile(**override)
    assert not f.loads and not f.output.exists()


def test_incomplete_reasoning_rejected_without_changing_raw_teacher_answer():
    r = response()
    r['teacher'] = dict(r['teacher'], generation_profile=PROFILE)
    r['reasoning_completed'], r['raw_text'] = False, r['text']
    before = json.dumps(r)
    with pytest.raises(ValueError, match='reasoning did not finish'):
        collect_search.consolidated_label(query(), r, r['teacher'])
    assert json.dumps(r) == before


@pytest.mark.parametrize('wrong_proof', [False, True])
def test_recorded_paper_acceptance_collection_uses_the_actual_policy(tmp_path, monkeypatch, wrong_proof):
    from xqgeneral.search_critical import acceptance_evidence
    args, pool, miner, q = recorded_collection(tmp_path)
    path = miner / 'manifest.json'
    proof = json.loads(path.read_text())
    proof['config']['acceptance_policy'] = 'paper_recursive'
    proof['verification'].update(acceptance_policy='paper_recursive', **acceptance_evidence('paper_recursive'))
    if wrong_proof:
        proof['verification']['strict_pv_improvement_required'] = True
    atomic_json(path, proof)
    monkeypatch.setattr('sys.argv', collect_args(tmp_path, args, miner))
    if wrong_proof:
        with pytest.raises(ValueError, match='critical-line acceptance'):
            collect_search.main()
        assert not (tmp_path / 'accepted').exists()
    else:
        collect_search.main()
        assert len(load_jsonl(tmp_path / 'accepted/train.jsonl')) == 1


@pytest.mark.parametrize('context', [True, 8192, 8193])
def test_pilot_profile_mismatch_rejected_before_search_or_teacher(tmp_path, monkeypatch, context):
    pipeline, validation, config, q, _ = setup_pilot(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    path = tmp_path / 'profile.json'
    atomic_json(path, dict(PROFILE, max_context_tokens=16384, max_new_tokens=8192))
    settings.update(teacher_generation_config=str(path), teacher_seed=2,
                    teacher_max_new_tokens=8192, teacher_max_context_tokens=context)
    atomic_json(config, settings)
    chain = ControlledChain(monkeypatch, q)
    with pytest.raises(ValueError):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert not chain.commands and chain.student_loads == chain.teacher_loads == 0


@pytest.mark.parametrize('false_legacy_claim', [False, True])
def test_actual_pilot_cli_routes_paper_teacher_and_honest_acceptance_proof(tmp_path, monkeypatch, false_legacy_claim):
    pipeline, validation, config, q, _ = setup_pilot(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    profile = tmp_path / 'teacher-generation.json'
    student = tmp_path / 'student-generation.json'
    atomic_json(profile, dict(PROFILE, max_context_tokens=16384, max_new_tokens=8192))
    atomic_json(student, dict(do_sample=True, temperature=0.6, top_k=20, top_p=0.95, max_new_tokens=4096))
    settings.update(teacher_generation_config=str(profile), teacher_seed=32,
        teacher_max_new_tokens=8192, teacher_max_context_tokens=16384, teacher_batch_size=2,
        student_generation_config=str(student), acceptance_policy='paper_recursive',
        root_candidate_policy='paper_fill', max_root_retries=3, reuse_child_analysis=True, min_clean_ply=1)
    atomic_json(config, settings)

    class Chain(ControlledChain):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if false_legacy_claim and 'xqgeneral.search_distillation' in command:
                path = Path(command[command.index('--output') + 1]) / 'manifest.json'
                saved = json.loads(path.read_text())
                saved['verification']['strict_pv_improvement_required'] = True
                atomic_json(path, saved)
            return result

    from pathlib import Path
    chain = Chain(monkeypatch, q)
    original = local_teacher.LocalConsolidator

    class Teacher(original):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.config = cfg
            self.model = SimpleNamespace(device=torch.device('cpu'))
            self.identity = dict(self.identity, generation_profile=cfg['generation_profile'])

        def generate(self, *args):
            value = super().generate(*args)
            value.update(raw_text='受控思考\n</think>\n' + value['text'], reasoning_completed=True,
                         prompt_tokens=3, padded_prompt_tokens=3, generated_tokens=2,
                         effective_max_new_tokens=args[1], context_limited_output=False)
            return value

    monkeypatch.setattr(local_teacher, 'LocalConsolidator', Teacher)
    output = tmp_path / 'pilot'
    if false_legacy_claim:
        with pytest.raises(ValueError, match='critical-line acceptance'):
            search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
        assert len(chain.commands) == 1 and chain.teacher_loads == 0
    else:
        result = search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
        assert result['accepted_structured_training_labels'] == 1
        command = next(c for c in chain.commands if 'xqgeneral.local_teacher' in c)
        assert command[command.index('--max-new-tokens') + 1] == '8192'
        assert command[command.index('--max-context-tokens') + 1] == '16384'
        assert command[command.index('--generation-config') + 1] == str(profile)
        assert command[command.index('--seed') + 1] == '32'
        mining = json.loads((output / 'mining/manifest.json').read_text())['verification']
        assert mining['strict_pv_improvement_required'] is False
        assert mining['paper_critical_line_acceptance_required'] is True
        assert result == search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output, resume=True)
        assert len(chain.commands) == 3
