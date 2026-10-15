"""Real Linux ownership/CLI controls with an explicit non-training stand-in.

These tests exercise actual processes, signals, exec and receipt consumption.
The stand-in curriculum and capacity gate do not claim GPU/model validation.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

TEMPLATE = Path('scripts/templates/verified_input_continuation.py')
pytestmark = pytest.mark.skipif(not hasattr(os, 'pidfd_open'), reason='Linux pidfds required')


def load(path, name='continuation_control'):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {'sha256': digest(path), 'bytes': path.stat().st_size}


def await_value(callback, timeout=12):
    deadline = time.monotonic() + timeout
    while True:
        result = callback()
        if result:
            return result
        if time.monotonic() > deadline:
            raise AssertionError('Owned process condition did not arrive')
        time.sleep(.025)


PROCESS_HELPER = '''import hashlib, os, signal
from pathlib import Path
def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=int(fields[19]), ppid=int(fields[1]), state=fields[0],
                    cmdline_sha256=hashlib.sha256((proc / 'cmdline').read_bytes()).hexdigest())
    except (FileNotFoundError, ProcessLookupError):
        return None
def same(expected, parent=True):
    actual = identity(expected['pid'])
    keys = ['pid', 'start_ticks', 'cmdline_sha256'] + (['ppid'] if parent else [])
    return actual if actual and all(actual[k] == expected[k] for k in keys) else None
def send(expected, kind):
    try:
        descriptor = os.pidfd_open(expected['pid'])
    except ProcessLookupError:
        return False
    try:
        if same(expected, parent=False) is None:
            return False
        signal.pidfd_send_signal(descriptor, kind)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(descriptor)
def tree(expected):
    leader = same(expected)
    if leader is None:
        return []
    result, pending = [leader], [leader]
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent['pid']}/task/{parent['pid']}/children").read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        for pid in children:
            child = identity(int(pid))
            if child and child['ppid'] == parent['pid']:
                result.append(child)
                pending.append(child)
    return result
'''

STAND_IN_CURRICULUM = '''"""CLI/process routing stand-in; no neural model or GPU work."""
import json, os, sys, time
from pathlib import Path
from xqgeneral.verified_inputs import verified_input_session, digest_with_verified_inputs
config = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text())
root = Path(config['output']); root.mkdir(exist_ok=True)
prepare = '--prepare-only' in sys.argv
phase = 'prepare' if prepare else 'train'
with verified_input_session(os.environ['XQGENERAL_VERIFIED_INPUT_RECEIPT']) as session:
    actual = digest_with_verified_inputs(config['input'])
    proof = session.metadata()
    (root / (phase + '.json')).write_text(json.dumps(dict(phase=phase, input_sha256=actual,
        receipt=proof, revision=os.environ['XQGENERAL_CODE_REVISION'], argv=sys.argv)))
if config.get('hold') == phase:
    while True: time.sleep(.1)
sys.exit(config.get(phase + '_exit', 0))
'''


@pytest.fixture
def case(tmp_path, request):
    project = tmp_path / 'project'
    run = project / 'runs' / 'continuation'
    run.mkdir(parents=True)
    entry = run / 'entry.py'
    entry.write_bytes(TEMPLATE.read_bytes())
    module = load(entry)
    helper = run / 'processes.py'
    helper.write_text(PROCESS_HELPER)
    processes = load(helper, 'actual_owned_process_fixture')
    children = []

    def cleanup():
        lease = run / 'lease.json'
        if lease.exists():
            value = json.loads(lease.read_text())
            for process in [value.get('watchdog'), *(value.get('worker') or {}).values()]:
                if isinstance(process, dict) and 'pid' in process:
                    processes.send(process, signal.SIGKILL)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
    request.addfinalizer(cleanup)

    def sleeper():
        command = [sys.executable, '-c', 'import time; time.sleep(600)']
        child = subprocess.Popen(command)
        children.append(child)
        return module.spawned_identity(processes, child, command)

    original, reader = sleeper(), sleeper()
    # The fixed reader has completed; its PID/start/cmdline proof is retained.
    processes.send(reader, signal.SIGTERM)
    children[-1].wait(timeout=5)
    source = run / 'cpu' / 'xqgeneral'
    gpu = run / 'gpu' / 'xqgeneral'
    for folder in [source, gpu]:
        folder.mkdir(parents=True)
        (folder / '__init__.py').write_text('')
        (folder / 'verified_inputs.py').write_bytes(Path('src/xqgeneral/verified_inputs.py').read_bytes())
    (gpu / 'validation_curriculum.py').write_text(STAND_IN_CURRICULUM)
    data = project / 'data' / 'immutable.bin'
    data.parent.mkdir()
    data.write_bytes(b'fixed-small-byte-hash-control')
    stat = data.stat()
    signature = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    preflight = run / 'preflight.json'
    save(preflight, {'status': 'complete', 'inputs': {str(data): binding(data)}})
    before, after = run / 'before.json', run / 'after.json'
    for path in [before, after]:
        save(path, {str(data): signature})
    candidate = run / 'candidate.json'
    save(candidate, {'kind': 'unpublished_verified_input_seed_candidate',
        'status': 'pending_complete_source_cache_base_readback', 'consumer_receipt_published': False,
        'bindings': {str(data): {**binding(data), 'signature': signature}},
        'preflight_manifest_sha256': digest(preflight),
        'preflight_original_signatures_before_sha256': digest(before),
        'preflight_original_signatures_after_sha256': digest(after)})
    output = project / 'runs' / 'paper-four'
    recipe = run / 'recipe.json'
    save(recipe, {'output': str(output), 'input': str(data)})
    verification = run / 'original-verification.json'
    original_entry = run / 'original.py'
    original_entry.write_text('''import json, os, subprocess, sys
from pathlib import Path
def state(*args, **kwargs): pass
def wait_gpu(*args): raise AssertionError('Formal capacity gate was not installed')
def main():
    config = Path(__file__).parent / 'recipe.json'
    command = [sys.executable, '-u', '-m', 'xqgeneral.validation_curriculum', '--config', str(config)]
    subprocess.run(command + ['--prepare-only'], env=dict(os.environ), check=True)
    wait_gpu([0, 1, 2, 3], 32768, 'waiting_for_formal_four_gpu_foundation_capacity')
    subprocess.run(command + ['--resume'], env=dict(os.environ), check=True)
    (Path(__file__).parent / 'original-verification.json').write_text(json.dumps({'four_courses_completed': True}))
''')
    capacity = run / 'capacity.py'
    capacity.write_text('''import json
from pathlib import Path
def await_capacity(indices, threshold, state, phase, observations):
    assert indices == [0, 1, 2, 3] and threshold == 74884
    Path(observations).write_text(json.dumps({'stand_in_capacity_gate_called': True}))
''')
    token = run / 'token.py'
    token.write_text('''def certify(*args):
    return dict(total_rule_records=1, all_observed_inputs_covered_by_capacity_control=True,
                conditional_required_free_mib_per_gpu=74884)
''')
    token_plan = run / 'token-plan.json'
    save(token_plan, {k: 'explicit-non-gpu-control' for k in ['recipe', 'preflight_manifest',
        'capacity_manifest', 'capacity_readback_manifest', 'corpus_manifest_sha256']})
    ownership = run / 'ownership.json'
    save(ownership, {'status': 'complete-owned-fixture', 'continuation': original, 'source_cache_base_reader': reader})
    readback = run / 'readback.json'
    proof_output = run / 'readback-verification.json'
    save(proof_output, {'source_cache_base_read_back': True})
    readback_value = {'status': 'complete', 'kind': 'recorded_foundation_source_and_cache_readback',
        'code': {'fixture': 'source-cache-base'},
        'inputs': {str(preflight): binding(preflight)}, 'outputs': {str(proof_output): binding(proof_output)},
        'verification': {'total_rule_records': 1, 'student_training_performed_by_readback': False}}
    save(readback, readback_value)
    control = run / 'control.json'
    save(control, {'status': 'complete', 'all_controls_passed': True, 'entry_sha256': digest(entry),
                   'scope': 'test fixture permits its own isolated processes only'})
    plan = dict(project=str(project), run=str(run), entry=str(entry), entry_sha256=digest(entry),
        process_helper=str(helper), original_parent=original, original_reader=reader,
        ownership_proof=str(ownership), ownership_verification={'status': 'complete-owned-fixture'},
        sources={name: {'path': str(folder), 'sha256': {p.name: digest(p) for p in folder.glob('*.py')}}
                 for name, folder in [('cpu', source), ('gpu', gpu)]},
        cpu_source=str(source), gpu_source=str(gpu), cpu_revision='fixture-cpu', gpu_revision='fixture-gpu',
        original_entry=str(original_entry), capacity_entry=str(capacity), token_helper=str(token), token_plan=str(token_plan),
        question_rows=1, preparation_verification=str(verification), control_proof=str(control),
        seed_candidate=str(candidate), preflight_manifest=str(preflight), signatures_before=str(before), signatures_after=str(after),
        readback_manifest=str(readback), readback_code=readback_value['code'],
        readback_input_bindings=dict(readback_value['inputs']), required_readback_verification=dict(readback_value['verification']),
        paper_output=str(output), receipt=str(run / 'receipt.json'), plan_file=str(run / 'plan.json'))
    bound = [helper, ownership, candidate, preflight, before, after, original_entry, capacity, token, token_plan, control]

    def refresh():
        plan['bound_files'] = {str(p): digest(p) for p in bound}
        save(run / 'plan.json', plan)

    def start(*, resume=False):
        refresh()
        log = (run / 'coordinator.log').open('a')
        command = [sys.executable, str(entry), '--plan', plan['plan_file']]
        if resume:
            command.append('--resume-committed')
        child = subprocess.Popen(command, stdout=log, stderr=log)
        log.close()
        children.append(child)
        return child

    refresh()
    # Let the real inode leave the receipt module's timestamp collision window.
    time.sleep(2.03)
    yield dict(plan=plan, module=module, processes=processes, run=run, data=data, output=output,
               original=original, reader=reader, readback=readback_value, proof_output=proof_output,
               refresh=refresh, start=start, sleeper=sleeper, children=children, recipe=recipe)


def read_lease(case):
    path = case['run'] / 'lease.json'
    return json.loads(path.read_text()) if path.exists() else None


def test_real_cli_routes_prepare_and_training_and_preserves_unrelated_process(case):
    sentinel = case['sleeper']()
    child = case['start']()
    assert child.wait(timeout=15) == 0, (case['run'] / 'coordinator.log').read_text()
    lease = read_lease(case)
    assert lease['committed'] and lease['retirement_complete'] and lease['released']
    assert case['module'].terminal(case['processes'], case['original'])
    assert case['processes'].same(sentinel)['state'] not in {'T', 'Z', 'X'}
    results = [json.loads((case['output'] / (phase + '.json')).read_text()) for phase in ['prepare', 'train']]
    assert all(v['revision'] == 'fixture-gpu' and v['input_sha256'] == digest(case['data']) for v in results)
    assert all(v['receipt']['reused_digest_calls'] == 1 and v['receipt']['fresh_digest_calls'] == 0 for v in results)
    observations = list(case['run'].glob('formal-capacity-observations-*.jsonl'))
    assert len(observations) == 1 and json.loads(observations[0].read_text())['stand_in_capacity_gate_called']


def test_incomplete_readback_waits_without_stopping_original_or_publishing_receipt(case):
    Path(case['plan']['readback_manifest']).unlink()
    child = case['start']()
    await_value(lambda: (case['run'] / 'state.json').exists())
    assert child.poll() is None
    assert case['processes'].same(case['original'])['state'] not in {'T', 'Z', 'X'}
    assert read_lease(case)['holds'] == []
    assert not Path(case['plan']['receipt']).exists() and not case['output'].exists()
    child.kill()
    child.wait(timeout=5)
    await_value(lambda: read_lease(case)['released'])


@pytest.mark.parametrize('problem', ['partial', 'wrong-code', 'changed-output', 'changed-prerequisite',
                                   'missing-verification', 'existing-curriculum', 'wrong-owner'])
def test_invalid_completed_binding_never_pauses_or_trains(case, problem):
    proof = case['readback']
    if problem == 'partial': proof['status'] = 'running'
    elif problem == 'wrong-code': proof['code'] = {'different': 'source'}
    elif problem == 'changed-output': case['proof_output'].write_text('different-output')
    elif problem == 'changed-prerequisite': Path(case['plan']['preflight_manifest']).write_text('changed')
    elif problem == 'missing-verification': proof['verification'].pop('total_rule_records')
    elif problem == 'existing-curriculum': case['output'].mkdir()
    elif problem == 'wrong-owner': case['plan']['original_parent']['cmdline_sha256'] = '0' * 64
    save(case['plan']['readback_manifest'], proof)
    case['refresh']()
    with pytest.raises((ValueError, KeyError)):
        case['module'].pinned(case['plan'])
        case['module'].readback_ready(case['plan'], case['processes'])
    actual = case['processes'].identity(case['original']['pid'])
    assert actual['state'] not in {'T', 'Z', 'X'}
    assert not Path(case['plan']['receipt']).exists()


def test_failed_real_prepare_rolls_back_original_and_preserves_aborted_output(case):
    recipe = json.loads(case['recipe'].read_text())
    save(case['recipe'], dict(recipe, prepare_exit=17))
    child = case['start']()
    assert child.wait(timeout=15) == 1
    lease = read_lease(case)
    assert lease['released'] and not lease['committed']
    assert lease['resumed_owned_parents'] == [case['original']['pid']]
    assert case['processes'].same(case['original'])['state'] not in {'T', 'Z', 'X'}
    archived = Path(lease['preserved_aborted_preparation'])
    assert (archived / 'prepare.json').exists() and not case['output'].exists()
    assert 'returned non-zero exit status 17' in (case['run'] / 'coordinator.log').read_text()


def test_failed_real_training_keeps_committed_output_and_never_resumes_old_parent(case):
    recipe = json.loads(case['recipe'].read_text())
    save(case['recipe'], dict(recipe, train_exit=23))
    child = case['start']()
    assert child.wait(timeout=15) == 1
    lease = read_lease(case)
    assert lease['committed'] and lease['released'] and lease['retirement_complete']
    assert 'resumed_owned_parents' not in lease and case['output'].exists()
    assert case['module'].terminal(case['processes'], case['original'])
    assert 'returned non-zero exit status 23' in (case['run'] / 'coordinator.log').read_text()


def test_live_stopped_controller_is_not_replaced_then_dead_controller_is_recovered(case):
    recipe = json.loads(case['recipe'].read_text())
    save(case['recipe'], dict(recipe, hold='prepare'))
    child = case['start']()
    await_value(lambda: (case['output'] / 'prepare.json').exists())
    owner = read_lease(case)['controller']
    case['processes'].send(owner, signal.SIGSTOP)
    await_value(lambda: case['processes'].same(owner)['state'] == 'T')
    case['module'].change(case['plan'], lambda value: value.update(heartbeat=0))
    time.sleep(2.2)
    assert not read_lease(case)['released']
    assert case['processes'].same(case['original'])['state'] == 'T'
    child.kill()
    child.wait(timeout=5)
    await_value(lambda: read_lease(case)['released'])
    lease = read_lease(case)
    assert lease['release_reason'] == 'controller_exact_identity_terminal'
    assert case['processes'].same(case['original'])['state'] not in {'T', 'Z', 'X'}
    assert all(case['module'].terminal(case['processes'], lease['worker'][k]) for k in ['bootstrap', 'execution'])
    assert Path(lease['preserved_aborted_preparation']).exists()


def test_watchdog_preserves_real_registered_training_after_committed_owner_death(case):
    recipe = json.loads(case['recipe'].read_text())
    save(case['recipe'], dict(recipe, hold='train'))
    child = case['start']()
    await_value(lambda: (case['output'] / 'train.json').exists())
    child.kill()
    child.wait(timeout=5)
    await_value(lambda: read_lease(case)['released'])
    lease = read_lease(case)
    assert lease['committed'] and lease['retirement_complete']
    assert case['processes'].same(lease['worker']['execution'], parent=False)['state'] not in {'T', 'Z', 'X'}
    assert case['module'].terminal(case['processes'], case['original'])
    assert case['output'].exists() and 'resumed_owned_parents' not in lease
    before = (case['run'] / 'lease.json').read_bytes()
    duplicate = case['start'](resume=True)
    assert duplicate.wait(timeout=10) == 1
    assert (case['run'] / 'lease.json').read_bytes() == before
    assert 'Preserve the still-live registered training worker' in (case['run'] / 'coordinator.log').read_text()


def test_changed_readback_after_owned_pause_is_revalidated_and_original_restored(case, monkeypatch):
    original_acquire = case['module'].acquire_parent
    def acquire_then_change(plan, processes):
        original_acquire(plan, processes)
        case['proof_output'].write_text('changed after owned parent pause')
    monkeypatch.setattr(case['module'], 'acquire_parent', acquire_then_change)
    with pytest.raises(ValueError, match='output changed'):
        case['module'].coordinate(case['plan'])
    assert read_lease(case)['released'] and not read_lease(case)['committed']
    assert case['processes'].same(case['original'])['state'] not in {'T', 'Z', 'X'}
    assert not Path(case['plan']['receipt']).exists()


def test_worker_request_cannot_disguise_training_as_prepare(case):
    owner = case['processes'].identity(os.getpid())
    save(case['run'] / 'lease.json', {'controller': owner, 'pending_worker': 'stand-in',
         'pending_worker_sha256': '', 'committed': False, 'released': False})
    request = case['run'] / 'worker-stand-in.json'
    save(request, {'id': 'stand-in', 'prepare_only': True,
        'args': [sys.executable, '-u', '-m', 'xqgeneral.validation_curriculum', '--resume']})
    with pytest.raises(ValueError, match='Unexpected owned worker command'):
        case['module'].worker(case['plan'], request)


def test_foreign_stop_is_never_adopted_or_resumed(case):
    case['processes'].send(case['original'], signal.SIGSTOP)
    await_value(lambda: case['processes'].same(case['original'])['state'] == 'T')
    with pytest.raises(ValueError, match='already stopped'):
        case['module'].readback_ready(case['plan'], case['processes'])
    assert not (case['run'] / 'lease.json').exists()
    assert case['processes'].same(case['original'])['state'] == 'T'


def test_pinned_execution_rejects_source_change_and_outside_project(case):
    case['module'].pinned(case['plan'])
    path = Path(case['plan']['gpu_source']) / 'validation_curriculum.py'
    path.write_text(path.read_text() + '\n# changed source\n')
    with pytest.raises(ValueError, match='source differs'):
        case['module'].pinned(case['plan'])
    case['plan']['paper_output'] = str(Path(case['plan']['project']).parent / 'other-project')
    with pytest.raises(ValueError, match='outside its project'):
        case['module'].pinned(case['plan'])


def test_explicit_committed_resume_archives_prior_lease_and_rechecks_capacity(case):
    first = case['start']()
    assert first.wait(timeout=15) == 0, (case['run'] / 'coordinator.log').read_text()
    old = read_lease(case)
    second = case['start'](resume=True)
    assert second.wait(timeout=15) == 0, (case['run'] / 'coordinator.log').read_text()
    new = read_lease(case)
    assert new['resume_committed'] and new['committed'] and new['released']
    assert new['attempt_id'] != old['attempt_id']
    archived = list(case['run'].glob('prior-committed-lease-*.json'))
    assert len(archived) == 1 and json.loads(archived[0].read_text()) == old
    assert len(list(case['run'].glob('formal-capacity-observations-*.jsonl'))) == 2
    assert case['module'].terminal(case['processes'], case['original'])


def test_changed_original_hash_seed_cannot_publish_a_consumable_receipt(case):
    candidate_path = Path(case['plan']['seed_candidate'])
    candidate = json.loads(candidate_path.read_text())
    candidate['bindings'][str(case['data'])]['sha256'] = '0' * 64
    save(candidate_path, candidate)
    child = case['start']()
    assert child.wait(timeout=15) == 1
    assert not Path(case['plan']['receipt']).exists()
    assert 'original successful full byte-hash binding' in (case['run'] / 'coordinator.log').read_text()
    assert read_lease(case)['resumed_owned_parents'] == [case['original']['pid']]


def test_new_child_empty_exec_cmdline_is_waited_for_without_accepting_it(case):
    command = [sys.executable, '-c', 'import time; time.sleep(600)']
    child = subprocess.Popen(command)
    case['children'].append(child)
    original_identity = case['processes'].identity
    observations = []
    def transient(pid):
        result = original_identity(pid)
        if result is not None and not observations:
            observations.append('transient-empty-cmdline')
            return dict(result, cmdline_sha256=hashlib.sha256(b'').hexdigest())
        return result
    class DuringExec:
        identity = staticmethod(transient)
    confirmed = case['module'].spawned_identity(DuringExec, child, command)
    assert observations == ['transient-empty-cmdline']
    assert confirmed['cmdline_sha256'] == hashlib.sha256(b'\0'.join(os.fsencode(x) for x in command) + b'\0').hexdigest()
