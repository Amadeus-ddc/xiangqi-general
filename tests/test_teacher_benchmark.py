"""Admission and interrupted comparison tests with a controlled CPU teacher."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from xqgeneral import teacher_benchmark as worker
from xqgeneral.clean_sft_evaluation import teacher_artifacts
from xqgeneral.evidence import atomic_json, code_identity, digest, load_jsonl, manifest, write_jsonl


@pytest.fixture
def setup(tmp_path, monkeypatch):
    teacher_root = tmp_path / 'controlled-teacher';teacher_root.mkdir()
    shard = teacher_root / 'model.safetensors';shard.write_bytes(b'CPU protocol fixture, not model weights')
    teacher = {'model_path': str(teacher_root), 'repository': 'controlled/teacher', 'revision': 'controlled-revision',
        'quantization': 'none', 'inference_dtype': 'bfloat16'}
    atomic_json(teacher_root / 'weights.manifest.json', {'repository': teacher['repository'],
        'revision': teacher['revision'], 'quantization': 'none', 'all_official_lfs_sha256_matched': True,
        'stored_tensor_dtypes': {'BF16': 1}, 'total_weight_bytes': shard.stat().st_size,
        'shards': {shard.name: {'sha256': digest(shard), 'bytes': shard.stat().st_size}}})
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        (teacher_root / name).write_text('{}')
    teacher_config = tmp_path / 'teachers.json';atomic_json(teacher_config, {'search_consolidator': teacher})
    evaluation = tmp_path / 'evaluation.json'
    atomic_json(evaluation, {'teacher_config': str(teacher_config), 'validation_examples': 4})
    pilot_config = tmp_path / 'pilot.json'
    settings = {'evaluation_config': str(evaluation), 'data': 'controlled-course-data',
        'recorded_inputs': 'controlled-isolated-pool', 'limit': 5, 'nodes': 100000,
        'max_depth': 5, 'child_contract': 'move_eval', 'seed': 20261013, 'teacher_max_new_tokens': 1536}
    atomic_json(pilot_config, settings)
    pipeline = tmp_path / 'pipeline';source = pipeline / 'sft/adapter.pt'
    source.parent.mkdir(parents=True);source.write_bytes(b'CPU parent fixture, not a trained model')
    parent = manifest('clean_curriculum_to_initial_sft_pipeline', {}, (), [source], {
        'four_courses_completed_before_any_gpu_sft_work': True, 'initial_explanation_sft_completed': True,
        'independent_test_used_for_training_or_selection': False, 'selected_checkpoint': str(source)})
    atomic_json(pipeline / 'manifest.json', parent)
    validation = tmp_path / 'validation'
    capability = manifest('clean_initial_sft_capability_validation', {'pipeline': str(pipeline),
        'config': str(evaluation), 'config_sha256': digest(evaluation)}, [source], (), {
        'actual_clean_initial_sft_completed_before_model_or_reviewer_load': True, 'split': 'validation',
        'all_validation_examples': 4, 'independent_test_used_for_training_or_selection': False,
        'raw_answers_repaired': False, 'raw_validation_by_memory': {'normal': {}, 'zero': {}, 'shuffled': {}}})
    atomic_json(validation / 'manifest.json', capability)
    pilot = tmp_path / 'pilot';pilot.mkdir()
    _, guard = teacher_artifacts(teacher_config)
    contract = {'pipeline': str(pipeline), 'validation': str(validation), 'config': str(pilot_config),
        'config_sha256': digest(pilot_config), 'code': code_identity(), 'teacher_assets': guard.expected,
        'prerequisite_inputs': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size}
            for p in [pilot_config, evaluation, teacher_config]}}
    atomic_json(pilot / 'contract.json', contract)
    queries = [{'id': f'q-{i}', 'messages': [{'role': 'user', 'content': '测试查询' * (i + 1)}]}
               for i in range(5)]
    mining = pilot / 'mining';write_jsonl(mining / 'queries.jsonl', queries)
    attempts = [{'source_root_id': f'root-{i}', 'query': q, 'reason': 'accepted'} for i, q in enumerate(queries)]
    write_jsonl(mining / 'results.partial.jsonl', attempts)
    proof = manifest('search_distillation_mining', {**settings, 'checkpoint': str(source)}, [source],
        [mining / 'queries.jsonl'], {'processed': 5, 'accepted_for_consolidation': 5, 'reasons': {'accepted': 5},
        **{key: True for key in ['unused_recorded_training_inputs', 'training_roots_only',
            'heldout_positions_excluded', 'all_child_branch_positions_isolated',
            'raw_child_reachable_prefixes_isolated', 'all_inferred_target_lines_history_validated',
            'strict_pv_improvement_required', 'all_successful_oracle_queries_preserved']}}, code=contract['code'])
    atomic_json(mining / 'manifest.json', proof)
    config = {'pilot_config': str(pilot_config), 'teacher_config': str(teacher_config), 'sample_limit': 4,
        'batch_sizes': [1, 2, 4], 'max_new_tokens': 1536, 'warmup_new_tokens': 32,
        'cuda_visible_devices': '3', 'minimum_free_gib': 70}
    config_path = tmp_path / 'benchmark.json';atomic_json(config_path, config)

    class Teacher:
        loads = 0
        calls = []
        interrupt = False
        batch_calls = 0
        wrong_counts = False
        def __init__(self, settings):
            Teacher.loads += 1
            self.identity = {**teacher, 'visible_cuda_devices': 1,
                'parameter_elements_by_dtype': {'torch.bfloat16': 100},
                'weight_manifest_sha256': digest(teacher_root / 'weights.manifest.json')}
        def response(self, messages, *, changed=False):
            content = messages[0]['content']
            return {'text': ('批量差异：' if changed else '') + content,
                'prompt_tokens': len(content) + 3 + int(Teacher.wrong_counts and changed),
                'generated_tokens': 3 if changed else 2, 'hit_generation_limit': False}
        def generate(self, messages, budget):
            Teacher.calls.append((1, [messages[0]['content']], budget))
            return self.response(messages)
        def generate_batch(self, messages, budget):
            Teacher.batch_calls += 1
            Teacher.calls.append((len(messages), [m[0]['content'] for m in messages], budget))
            if Teacher.interrupt and Teacher.batch_calls == 2:
                Teacher.interrupt = False
                raise RuntimeError('controlled interruption during an uncompleted batch')
            return [self.response(m, changed=len(messages) == 2 and i == 0) for i, m in enumerate(messages)]

    monkeypatch.setattr(worker, 'LocalConsolidator', Teacher)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    monkeypatch.setattr(worker.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(worker.torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(worker.torch.cuda, 'mem_get_info', lambda _: (80 * 2**30, 96 * 2**30))
    monkeypatch.setattr(worker.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(worker.torch.cuda, 'reset_peak_memory_stats', lambda _: None)
    monkeypatch.setattr(worker.torch.cuda, 'max_memory_allocated', lambda _: 3 * 2**30)
    return {'pilot': pilot, 'config': config, 'config_path': config_path, 'root': tmp_path / 'benchmark',
        'Teacher': Teacher, 'queries': queries, 'mining': proof, 'capability': capability,
        'pipeline': parent, 'validation': validation, 'source': source, 'shard': shard}


def run(s, **kwargs):
    return worker.run_benchmark(s['pilot'], 'owned-clean-search', s['config_path'], s['root'], **kwargs)


@pytest.mark.parametrize('key,value', [
    ('sample_limit', 0), ('sample_limit', True), ('max_new_tokens', 0), ('warmup_new_tokens', 2048),
    ('batch_sizes', [2, 4]), ('batch_sizes', [1, True]), ('batch_sizes', [1, 2, 2]),
    ('batch_sizes', [1, 4, 2]), ('batch_sizes', [1, 8]), ('cuda_visible_devices', '0,3'),
    ('minimum_free_gib', float('inf')), ('minimum_free_gib', False),
])
def test_invalid_resource_or_generation_budgets_never_load_teacher(setup, key, value):
    s = setup;s['config'][key] = value;atomic_json(s['config_path'], s['config'])
    with pytest.raises(ValueError):
        run(s)
    assert s['Teacher'].loads == 0 and not s['root'].exists()


@pytest.mark.parametrize('change', ['incomplete_sft', 'test_used', 'repaired', 'missing_memory',
    'wrong_parent', 'wrong_mining_code', 'lost_isolation', 'false_yield', 'changed_queries'])
def test_unfinished_or_mismatched_clean_mining_refuses_measurement(setup, change):
    s = setup
    if change in {'incomplete_sft', 'test_used', 'repaired', 'missing_memory'}:
        p = deepcopy(s['capability']);v = p['verification']
        if change == 'incomplete_sft':v['actual_clean_initial_sft_completed_before_model_or_reviewer_load'] = False
        elif change == 'test_used':v['independent_test_used_for_training_or_selection'] = True
        elif change == 'repaired':v['raw_answers_repaired'] = True
        else:del v['raw_validation_by_memory']['shuffled']
        atomic_json(s['validation'] / 'manifest.json', p)
    else:
        p = deepcopy(s['mining'])
        if change == 'wrong_parent':p['config']['checkpoint'] = 'different-parent.pt'
        elif change == 'wrong_mining_code':p['code'] = {'revision': 'other'}
        elif change == 'lost_isolation':p['verification']['all_child_branch_positions_isolated'] = False
        elif change == 'false_yield':p['verification']['accepted_for_consolidation'] += 1
        else:(s['pilot'] / 'mining/queries.jsonl').write_text('changed raw queries')
        atomic_json(s['pilot'] / 'mining/manifest.json', p)
    with pytest.raises(ValueError):
        run(s)
    assert s['Teacher'].loads == 0 and not (s['root'] / 'manifest.json').exists()


def test_actual_raw_comparisons_keep_differences_and_complete_resume_has_no_calls(setup, monkeypatch):
    s = setup;result = run(s)
    assert s['Teacher'].loads == 1
    assert [row['id'] for row in load_jsonl(s['root'] / 'selected-queries.jsonl')] == ['q-0', 'q-1', 'q-3', 'q-4']
    metrics = result['generation_by_batch_size']
    assert [metrics[str(size)]['generation_calls'] for size in [1, 2, 4]] == [4, 2, 1]
    assert metrics['2']['raw_text_matches_serial'] == 2 and metrics['4']['raw_text_matches_serial'] == 4
    assert all(metrics[str(size)]['queries'] == 4 for size in [1, 2, 4])
    assert metrics['1']['generated_tokens'] == 8 and metrics['2']['generated_tokens'] == 10
    assert not result['strategic_prose_quality_measured'] and result['new_training_labels'] == 0
    assert len(s['Teacher'].calls) == 8  # one warmup plus seven comparison groups
    before = {str(p): p.read_bytes() for p in s['root'].iterdir() if p.is_file()}
    monkeypatch.setattr(worker, 'wait_for_capacity', lambda *a: pytest.fail('completed result admitted a GPU'))
    assert run(s, resume=True) == result and s['Teacher'].loads == 1
    assert before == {str(p): p.read_bytes() for p in s['root'].iterdir() if p.is_file()}


def test_interrupted_batch_resume_reuses_every_completed_group(setup):
    s = setup;s['Teacher'].interrupt = True
    with pytest.raises(RuntimeError, match='controlled interruption'):
        run(s)
    groups = json.loads((s['root'] / 'groups.json').read_text())
    assert len(groups) == 5 and not (s['root'] / 'manifest.json').exists()
    before = len(s['Teacher'].calls)
    result = run(s, resume=True)
    assert s['Teacher'].loads == 2 and len(s['Teacher'].calls) - before == 3
    assert json.loads((s['root'] / 'groups.json').read_text())[:5] == groups
    assert result['teacher_model_loads_recorded'] == 2 and result['distinct_queries_measured'] == 4


@pytest.mark.parametrize('change', ['config', 'weights', 'foreign_group'])
def test_partial_continuation_refuses_changed_contract_or_coverage_before_reload(setup, change):
    s = setup;s['Teacher'].interrupt = True
    with pytest.raises(RuntimeError):
        run(s)
    if change == 'config':
        s['config']['batch_sizes'] = [1, 4];atomic_json(s['config_path'], s['config'])
    elif change == 'weights':s['shard'].write_bytes(b'changed controlled shard')
    else:
        groups = json.loads((s['root'] / 'groups.json').read_text());groups[0]['ids'][0] = 'foreign'
        atomic_json(s['root'] / 'groups.json', groups)
    with pytest.raises(ValueError):
        run(s, resume=True)
    assert s['Teacher'].loads == 1


@pytest.mark.parametrize('name', ['selected-queries.jsonl', 'groups.json', 'metrics.json'])
def test_completed_missing_or_corrupt_outputs_are_not_recreated(setup, name):
    s = setup;run(s)
    target = s['root'] / name
    if name == 'selected-queries.jsonl':target.unlink()
    else:target.write_text('corrupted output preserved')
    with pytest.raises(ValueError, match='changed or is missing'):
        run(s, resume=True)
    assert s['Teacher'].loads == 1
    assert not target.exists() if name == 'selected-queries.jsonl' else target.read_text() == 'corrupted output preserved'


def test_zero_mining_yield_finishes_without_cuda_or_teacher(setup, monkeypatch):
    s = setup;mining = s['pilot'] / 'mining'
    write_jsonl(mining / 'queries.jsonl', [])
    write_jsonl(mining / 'results.partial.jsonl', [{'source_root_id': f'root-{i}', 'query': None,
        'reason': 'no_improvement'} for i in range(5)])
    p = deepcopy(s['mining']);p['outputs'][str(mining / 'queries.jsonl')] = {'sha256': digest(mining / 'queries.jsonl'), 'bytes': 0}
    p['verification'].update(accepted_for_consolidation=0, reasons={'no_improvement': 5})
    atomic_json(mining / 'manifest.json', p)
    monkeypatch.setattr(worker, 'wait_for_capacity', lambda *a: pytest.fail('empty yield used CUDA'))
    result = run(s)
    assert not result['neural_teacher_inference_executed'] and s['Teacher'].loads == 0
    assert result['distinct_queries_measured'] == 0 and result['teacher'] is None
    assert all(m['queries_per_second'] is None for m in result['generation_by_batch_size'].values())


def test_live_mining_wait_does_not_admit_teacher_before_manifest(setup, monkeypatch):
    s = setup;path = s['pilot'] / 'mining/manifest.json';path.unlink()
    original = worker.subprocess.run
    monkeypatch.setattr(worker.subprocess, 'run', lambda command, **kw:
        SimpleNamespace(returncode=0) if command[0] == 'tmux' else original(command, **kw))
    def release(seconds):
        state = json.loads((s['root'] / 'state.json').read_text())
        assert state['status'] == 'waiting_for_completed_clean_search_mining' and state['teacher_loaded'] is False
        assert s['Teacher'].loads == 0 and seconds == 1
        atomic_json(path, s['mining'])
    monkeypatch.setattr(worker.time, 'sleep', release)
    assert run(s, poll_seconds=1)['distinct_queries_measured'] == 4


def test_stopped_owned_miner_never_loads_teacher(setup, monkeypatch):
    s = setup;(s['pilot'] / 'mining/manifest.json').unlink()
    original = worker.subprocess.run
    def stopped(command, **kwargs):
        if command[0] != 'tmux':
            return original(command, **kwargs)
        assert command == ['tmux', 'has-session', '-t', '=owned-clean-search']
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(worker.subprocess, 'run', stopped)
    with pytest.raises(RuntimeError, match='stopped before'):
        run(s)
    assert s['Teacher'].loads == 0


def test_capacity_wait_and_wrong_visible_gpu_cannot_load_prematurely(setup, monkeypatch):
    s = setup;free = [60 * 2**30]
    monkeypatch.setattr(worker.torch.cuda, 'mem_get_info', lambda _: (free[0], 96 * 2**30))
    def release(seconds):
        assert json.loads((s['root'] / 'state.json').read_text())['status'] == 'waiting_for_benchmark_gpu_capacity'
        assert s['Teacher'].loads == 0
        free[0] = 80 * 2**30
    monkeypatch.setattr(worker.time, 'sleep', release)
    assert run(s)['distinct_queries_measured'] == 4
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    other = dict(s, root=s['root'].parent / 'other-benchmark')
    with pytest.raises(ValueError, match='declared visible CUDA device'):
        run(other)
    assert s['Teacher'].loads == 1


def test_padding_token_count_mismatch_is_preserved_and_not_promoted(setup):
    s = setup;s['Teacher'].wrong_counts = True
    with pytest.raises(ValueError, match='prompt token counts differ'):
        run(s)
    assert len(json.loads((s['root'] / 'groups.json').read_text())) == 7
    assert not (s['root'] / 'manifest.json').exists()


def test_impossible_gpu_capacity_refuses_instead_of_waiting_forever(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(worker.torch.cuda, 'mem_get_info', lambda _: (24 * 2**30, 24 * 2**30))
    monkeypatch.setattr(worker.time, 'sleep', lambda *a: pytest.fail('impossible capacity waited'))
    with pytest.raises(ValueError, match='exceeds the GPU total'):
        run(s)
    assert s['Teacher'].loads == 0
