"""Exercise deployment identity and the actual pilot/teacher/collection CLI seam."""
import copy
import json
from pathlib import Path

import pytest

from test_search_collection import response
from test_search_pilot import ControlledChain, setup_pilot
from xqgeneral import local_teacher, search_pilot, vllm_teacher
from xqgeneral.evidence import atomic_json, digest


def setup_deployment(tmp_path, monkeypatch):
    pipeline, validation, config, query, capability = setup_pilot(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    evaluation = json.loads(Path(settings['evaluation_config']).read_text())
    original = evaluation['teacher_config']
    reviewed = json.loads(Path(original).read_text())
    reviewed['search_consolidator'].update(tensor_parallel_size=2, gpu_memory_utilization=.45,
                                          max_context_tokens=16384)
    atomic_json(original, reviewed)
    deployed = copy.deepcopy(reviewed)
    deployed['search_consolidator'].update(tensor_parallel_size=1, gpu_memory_utilization=.70)
    replacement = tmp_path / 'teacher-tp1.json'
    atomic_json(replacement, deployed)
    settings.update(teacher_deployment_config=str(replacement), teacher_backend='vllm',
                    teacher_max_context_tokens=16384)
    atomic_json(config, settings)
    return pipeline, validation, config, query, original, replacement


def controlled_vllm(monkeypatch, chain):
    # This is an in-process CPU double; installed optional runtimes are irrelevant.
    monkeypatch.setattr(local_teacher, 'generation_software', lambda: {'runtime': 'controlled_cpu_fixture'})
    monkeypatch.setattr(local_teacher, 'version', lambda name: 'controlled_cpu_fixture')
    closed = []
    class Teacher:
        def __init__(self, config):
            chain.teacher_loads += 1
            self.identity = dict(response()['teacher'], backend='vllm',
                request_seed_profile=vllm_teacher.REQUEST_SEED_PROFILE,
                **vllm_teacher.deployment_settings(config))

        def generate_batch(self, messages, budget, *, sampling_seed=None):
            base = {key: value for key, value in response().items() if key not in ['id', 'teacher']}
            base.update(prompt_tokens=1, generated_tokens=1, padded_prompt_tokens=1,
                        effective_max_new_tokens=budget, context_limited_output=False,
                        generated_token_ids=[1])
            return [copy.deepcopy(base) for _ in messages]

        def close(self):
            closed.append(True)
    monkeypatch.setattr(vllm_teacher, 'VllmConsolidator', Teacher)
    return closed


@pytest.mark.parametrize('empty', [False, True])
def test_tp1_reaches_real_cli_and_collection_without_changing_review_inputs(tmp_path, monkeypatch, empty):
    pipeline, validation, config, query, original, replacement = setup_deployment(tmp_path, monkeypatch)
    original_sha, capability_sha = digest(original), digest(validation / 'manifest.json')
    chain = ControlledChain(monkeypatch, query, empty=empty)
    closed = controlled_vllm(monkeypatch, chain)
    output = tmp_path / 'pilot'
    result = search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output)
    note = result['consolidation_teacher_deployment']
    assert note['validation_teacher_config'] == original
    assert note['consolidation_teacher_config'] == str(replacement)
    assert note['configured_placement'] == dict(tensor_parallel_size=1, gpu_memory_utilization=.70)
    assert note['actual_neural_consolidation_executed'] is (not empty)
    assert note['numeric_equivalence_to_validation_deployment_proven'] is False
    assert digest(original) == original_sha and digest(validation / 'manifest.json') == capability_sha
    proof = json.loads((output / 'manifest.json').read_text())
    assert proof['inputs'][original]['sha256'] == original_sha
    assert proof['inputs'][str(replacement)]['sha256'] == digest(replacement)
    assert chain.teacher_loads == int(not empty) and closed == ([] if empty else [True])
    if not empty:
        for module in ['xqgeneral.local_teacher', 'xqgeneral.collect_search']:
            command = next(c for c in chain.commands if module in c)
            assert command[command.index('--config') + 1] == str(replacement)
        actual = json.loads((output / 'consolidation/deployment.json').read_text())
        assert actual['tensor_parallel_size'] == 1 and actual['gpu_memory_utilization'] == .70
        assert result['accepted_structured_training_labels'] == 1
    calls = len(chain.commands)
    assert result == search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output, resume=True)
    assert len(chain.commands) == calls
    changed = json.loads(replacement.read_text())
    changed['search_consolidator']['gpu_memory_utilization'] = .65
    atomic_json(replacement, changed)
    with pytest.raises(ValueError, match='Resume only the exact preserved pilot contract'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, output, resume=True)
    assert len(chain.commands) == calls


@pytest.mark.parametrize('field,value', [
    ('repository', 'different/model'), ('revision', 'different'), ('model_path', 'different'),
    ('quantization', 'Q4_K_M'), ('inference_dtype', 'float16'), ('max_context_tokens', 8192),
    ('prefill_chunk_size', 512), ('purpose', 'replace teacher instructions'),
    ('generation_profile', {'do_sample': True}),
])
def test_deployment_refuses_any_teacher_semantic_or_asset_change_before_models(tmp_path, monkeypatch, field, value):
    pipeline, validation, config, query, _, replacement = setup_deployment(tmp_path, monkeypatch)
    changed = json.loads(replacement.read_text())
    changed['search_consolidator'][field] = value
    atomic_json(replacement, changed)
    chain = ControlledChain(monkeypatch, query)
    with pytest.raises(ValueError, match='may change only tensor parallelism'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert not chain.commands and chain.student_loads == chain.teacher_loads == 0


@pytest.mark.parametrize('field,value', [
    ('tensor_parallel_size', 0), ('tensor_parallel_size', True),
    ('gpu_memory_utilization', 0), ('gpu_memory_utilization', 1.1),
    ('gpu_memory_utilization', float('nan')),
])
def test_invalid_placement_refused_before_models(tmp_path, monkeypatch, field, value):
    pipeline, validation, config, query, _, replacement = setup_deployment(tmp_path, monkeypatch)
    changed = json.loads(replacement.read_text())
    changed['search_consolidator'][field] = value
    atomic_json(replacement, changed)
    chain = ControlledChain(monkeypatch, query)
    with pytest.raises(ValueError, match='Teacher (tensor_parallel_size|gpu_memory_utilization)'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert not chain.commands and chain.student_loads == chain.teacher_loads == 0


@pytest.mark.parametrize('path', ['', None, True])
def test_deployment_path_must_be_explicit(tmp_path, monkeypatch, path):
    pipeline, validation, config, query, _, _ = setup_deployment(tmp_path, monkeypatch)
    settings = json.loads(config.read_text()); settings['teacher_deployment_config'] = path
    atomic_json(config, settings)
    chain = ControlledChain(monkeypatch, query)
    with pytest.raises(ValueError, match='explicit configuration file path'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert not chain.commands


def test_deployment_override_requires_declared_vllm_backend(tmp_path, monkeypatch):
    pipeline, validation, config, query, _, _ = setup_deployment(tmp_path, monkeypatch)
    settings = json.loads(config.read_text()); settings.pop('teacher_backend')
    atomic_json(config, settings)
    chain = ControlledChain(monkeypatch, query)
    with pytest.raises(ValueError, match='requires the vLLM backend'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert not chain.commands


@pytest.mark.parametrize('document', [{}, {'search_consolidator': None}, {'search_consolidator': []}, []])
def test_deployment_requires_a_consolidator_object_before_models(tmp_path, monkeypatch, document):
    pipeline, validation, config, query, _, replacement = setup_deployment(tmp_path, monkeypatch)
    atomic_json(replacement, document)
    chain = ControlledChain(monkeypatch, query)
    with pytest.raises(ValueError, match='original search_consolidator identity'):
        search_pilot.run_pilot(pipeline, validation, 'owned-validation', config, tmp_path / 'pilot')
    assert not chain.commands and chain.student_loads == chain.teacher_loads == 0
