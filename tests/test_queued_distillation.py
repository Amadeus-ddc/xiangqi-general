"""Producer identity, exact data, actual OS leases and controlled round handoffs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_distillation_joint_replay import joint_budget
from xqgeneral.curriculum_data import STAGES
from xqgeneral.evidence import atomic_json, code_identity, digest, file_signature
from xqgeneral.verified_inputs import write_verified_input_receipt

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/queued_distillation.py'


@pytest.fixture
def queue():
    spec = importlib.util.spec_from_file_location('controlled_queued_distillation', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed(queue, root, value):
    atomic_json(root / 'verification.json', value)
    atomic_json(root / 'exit.json', {'returncode': 0, 'verification_sha256': queue.sha(root / 'verification.json')})


@pytest.fixture
def prepared(tmp_path, monkeypatch, queue):
    monkeypatch.setenv('XQGENERAL_CODE_REVISION', 'controlled-queued-round-no-model-inference')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    source = Path(__import__('xqgeneral').__file__).resolve().parent.parent
    identity = code_identity()
    data, replay, parent = [tmp_path / name for name in ['labels', 'replay', 'parent']]
    for directory in [data, replay, parent]:
        directory.mkdir()
    (data / 'train.jsonl').write_text((json.dumps({'stage': 'explanation'}) + '\n') * 1000)
    (replay / 'train.jsonl').write_text(''.join((json.dumps({'stage': stage}) + '\n') * 128 for stage in STAGES))
    feature = tmp_path / 'features.json'; atomic_json(feature, {'controlled': True})
    weight = parent / 'adapter.pt';weight.write_bytes(b'controlled weight identity, not a model')
    saved = parent / 'config.json';atomic_json(saved, {'model_path': str(tmp_path / 'model'),
        'feature_path': str(feature), 'expert_feature_depths': [0, 3]})
    receipt = tmp_path / 'receipt.json'
    write_verified_input_receipt(receipt, {str(weight): {'sha256': digest(weight), 'bytes': weight.stat().st_size,
        'signature': list(file_signature(weight))}}, origin={'controlled_CPU_fixture': True})
    template, token = joint_budget()
    profile = json.loads((SCRIPT.parents[1] / 'configs/search-distillation-sft-fsdp2-v1.json').read_text())
    template.update({key: profile[key] for key in ['distributed_strategy', 'fsdp_shard_paths',
        'decoder_gradient_checkpointing', 'autocast_cache_enabled', 'optimizer_state_sharding', 'max_finite_passes']})
    template.update(max_steps=30000, eval_every=250, replay_data_paths=[str(replay)])
    token['config'].update(data=str(data), source=str(weight), parent_completion=str(tmp_path / 'parent-completion.json'),
                           replay_data_paths=[str(replay)])
    token['cache_header_only_input']['path'] = str(feature)
    token['verification'].update(by_split={'train': 1000, 'validation': 384, 'test': 512}, independent_test_rows_tokenized=0)
    token['verification']['course_replay']['data_paths'] = [str(replay)]
    token_path = tmp_path / 'token.json';atomic_json(token_path, token)
    template_path = tmp_path / 'template.json';atomic_json(template_path, template)
    ready = {'cpu_validation': {'status': 'complete', 'tests_passed': 1,
        'checked_files': {'scripts/queued_distillation.py': queue.sha(SCRIPT)},
        'source_revision': identity['revision'], 'source_sha256': {key.rsplit('/', 1)[-1]: value for key, value in identity['source_sha256'].items()}},
        'data_preparation': {'status': 'complete_actual_scaled_joint_distillation_data_preparation',
            'accepted_merged_training_histories': 1000, 'actual_search_and_selected_parent': str(weight),
            'token_preflight_manifest_sha256': queue.sha(token_path), 'feature_manifest_sha256': queue.sha(feature)},
        'development_evaluation': {'status': 'complete_joint_replay_SFT_retention_and_paired_development_evaluation'}}
    dependencies = {}
    for role, value in ready.items():
        directory = tmp_path / role;completed(queue, directory, value)
        dependencies[role] = {'run': str(directory), 'status': value['status']}
    plan = {'project_root': str(tmp_path), 'source': str(source),
        'source_sha256': ready['cpu_validation']['source_sha256'], 'source_revision': identity['revision'],
        'entry_sha256': queue.sha(SCRIPT), 'fixed_sha256': {str(path): queue.sha(path) for path in [saved, template_path, receipt]},
        'expected_CPU_tests': 1, 'dependencies': dependencies, 'parent': str(weight),
        'parent_completion': token['config']['parent_completion'], 'recipe_template': str(template_path),
        'round_output': str(tmp_path / 'runs/round'), 'token_preflight': str(token_path),
        'feature_manifest': str(feature), 'minimum_accepted_training_rows': 512,
        'minimum_free_MiB_each_gpu': 35840, 'receipt': str(receipt), 'teacher_lock_path': str(tmp_path / 'teacher.lock'),
        'expected_memory_controls': {key: profile[key] for key in ['distributed_strategy', 'fsdp_shard_paths',
            'decoder_gradient_checkpointing', 'autocast_cache_enabled', 'optimizer_state_sharding']}}
    plan_path = tmp_path / 'plan.json';atomic_json(plan_path, plan)
    monkeypatch.chdir(tmp_path)
    return plan_path, plan, ready


def test_pending_dependency_requires_exact_current_process_identity(tmp_path, queue):
    root = tmp_path / 'producer';root.mkdir()
    identity = queue.process_identity(os.getpid())
    atomic_json(root / 'owner.json', {'identity': identity})
    item = {'run': str(root), 'status': 'complete'}
    assert queue.completed_dependencies({'data': item}) == ({}, ['data'])
    identity['start_time'] += 1
    atomic_json(root / 'owner.json', {'identity': identity})
    with pytest.raises(RuntimeError, match='identity changed'):
        queue.completed_dependencies({'data': item})


@pytest.mark.parametrize('problem', ['failed', 'changed_proof', 'wrong_status'])
def test_failed_dependency_is_not_hidden_behind_another_live_wait(tmp_path, queue, problem):
    first, second = tmp_path / 'pending', tmp_path / 'terminal'
    atomic_json(first / 'owner.json', {'identity': queue.process_identity(os.getpid())})
    completed(queue, second, {'status': 'complete'})
    if problem == 'failed':atomic_json(second / 'exit.json', {'returncode': 7})
    elif problem == 'changed_proof':atomic_json(second / 'verification.json', {'status': 'changed'})
    else:completed(queue, second, {'status': 'wrong'})
    with pytest.raises(ValueError):
        queue.completed_dependencies({'pending': {'run': str(first), 'status': 'complete'},
                                      'terminal': {'run': str(second), 'status': 'complete'}})


def test_prepared_data_compiles_exact_finite_joint_budget_without_creating_round(prepared, queue):
    _, plan, ready = prepared
    recipe, budget, final = queue.prepared_recipe(plan, ready)
    assert recipe['max_steps'] == final['steps'] == budget['max_updates'] == 156
    assert budget['rows_per_complete_pass'] == 1248 and recipe['epochs'] == 4
    assert all(final[key] == value for key, value in plan['expected_memory_controls'].items())
    assert not Path(plan['round_output']).exists()


@pytest.mark.parametrize('problem', ['other_parent', 'pilot_only', 'changed_tokens', 'lost_memory', 'test_tokens', 'wrong_cpu'])
def test_unqualified_data_cannot_reach_native_round(prepared, queue, problem):
    _, plan, ready = prepared
    if problem == 'other_parent':ready['data_preparation']['actual_search_and_selected_parent'] = 'other-model'
    elif problem == 'pilot_only':ready['data_preparation']['accepted_merged_training_histories'] = 70
    elif problem == 'changed_tokens':
        path = Path(plan['token_preflight']);path.write_text(path.read_text() + '\n')
    elif problem == 'lost_memory':
        template = queue.read(plan['recipe_template']);template.pop('distributed_strategy');atomic_json(plan['recipe_template'], template)
    elif problem == 'test_tokens':
        token = queue.read(plan['token_preflight']);token['verification']['independent_test_rows_tokenized'] = 1
        atomic_json(plan['token_preflight'], token);ready['data_preparation']['token_preflight_manifest_sha256'] = queue.sha(plan['token_preflight'])
    else:ready['cpu_validation']['source_revision'] = 'unvalidated-source'
    with pytest.raises(ValueError):queue.prepared_recipe(plan, ready)
    assert not Path(plan['round_output']).exists()


LOCK_CHECK = "import fcntl,sys\nwith open(sys.argv[1],'a+') as f:\n try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n except BlockingIOError:sys.exit(0 if sys.argv[2]=='held' else 17)\n else:sys.exit(0 if sys.argv[2]=='free' else 18)\n"


@pytest.mark.parametrize('case', ['complete', 'native_failure', 'changed_while_waiting'])
def test_real_OS_lease_and_receipt_enclose_controlled_round_and_release_on_failure(prepared, queue, monkeypatch, case):
    plan_path, plan, _ = prepared
    original_run = subprocess.run
    gpu_queries, round_calls = [], []
    def driver(command, **kwargs):
        if command[0] != 'nvidia-smi':return original_run(command, **kwargs)
        gpu_queries.append(command)
        if len(gpu_queries) == 1:
            if case == 'changed_while_waiting':
                path = Path(plan['token_preflight']);path.write_text(path.read_text() + '\n')
            return subprocess.CompletedProcess(command, 0, stdout='30000\n' * 4)
        return subprocess.CompletedProcess(command, 0, stdout='50000\n' * 4)
    monkeypatch.setattr(subprocess, 'run', driver)
    monkeypatch.setenv('XQGENERAL_VERIFIED_INPUT_RECEIPT', 'prior-environment-value')
    def controlled_round(parent, template, token, output, *, resume):
        round_calls.append((parent, template, token, output, resume))
        assert original_run([sys.executable, '-c', LOCK_CHECK, plan['teacher_lock_path'], 'held']).returncode == 0
        assert digest(parent) == queue.sha(parent)
        Path(output).mkdir(parents=True);(Path(output) / 'controlled-partial.txt').write_text('not a model or SGD result')
        if case == 'native_failure':raise RuntimeError('controlled native preflight failure')
        return {'status': 'complete', 'actual_distillation_sft_completed': True, 'controlled_result_not_model_evidence': True}
    monkeypatch.setattr(queue, 'run_round', controlled_round)
    output = Path(plan['project_root']) / 'runs/supervisor'
    if case == 'complete':
        result = queue.run(plan_path, queue.sha(plan_path), output, resume_round=True, poll_seconds=.001)
        assert result['input_digest_verification']['reused_digest_calls'] > 0
        assert queue.read(output / 'exit.json')['returncode'] == 0
        assert round_calls == [(plan['parent'], plan['recipe_template'], plan['token_preflight'], plan['round_output'], True)]
    else:
        with pytest.raises((RuntimeError, ValueError)):
            queue.run(plan_path, queue.sha(plan_path), output, poll_seconds=.001)
        assert queue.read(output / 'exit.json')['returncode'] == 1
        assert bool(round_calls) == (case == 'native_failure')
    assert len(gpu_queries) == 2
    assert original_run([sys.executable, '-c', LOCK_CHECK, plan['teacher_lock_path'], 'free']).returncode == 0
    assert os.environ['XQGENERAL_VERIFIED_INPUT_RECEIPT'] == 'prior-environment-value'
    assert subprocess.run is driver
    if round_calls:assert (Path(plan['round_output']) / 'controlled-partial.txt').exists()


@pytest.mark.parametrize('problem', ['plan', 'entry', 'source', 'fixed_input'])
def test_changed_pinned_plan_source_or_inputs_rejected(prepared, queue, problem):
    plan_path, plan, _ = prepared
    expected = queue.sha(plan_path)
    queue.verify_pins(plan, plan_path, expected)
    if problem == 'plan':plan_path.write_text(plan_path.read_text() + '\n')
    elif problem == 'entry':plan['entry_sha256'] = '0' * 64
    elif problem == 'source':plan['source_sha256']['__init__.py'] = '0' * 64
    else:
        path = Path(plan['recipe_template']);path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError):queue.verify_pins(plan, plan_path, expected)
