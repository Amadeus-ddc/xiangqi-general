"""Resume a frozen preparation with qualified input-hash reuse after full readback."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def change(plan, callback):
    run = Path(plan['run'])
    with (run / 'lease.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        value = read(run / 'lease.json')
        callback(value)
        write(run / 'lease.json', value)
        return value


def helper(plan):
    return load_module('owned_input_continuation_processes', plan['process_helper'])


def terminal(processes, expected):
    actual = processes.identity(expected['pid'])
    return actual is None or actual['start_ticks'] != expected['start_ticks'] or actual['state'] in {'Z', 'X'}


def spawned_identity(processes, child, command, *, timeout=5):
    """An exec can briefly expose an empty cmdline after Popen returns."""
    expected_hash = hashlib.sha256(b'\0'.join(os.fsencode(x) for x in command) + b'\0').hexdigest()
    start, deadline = None, time.monotonic() + timeout
    while child.poll() is None:
        actual = processes.identity(child.pid)
        if actual is not None:
            start = actual['start_ticks'] if start is None else start
            if actual['start_ticks'] != start or actual['ppid'] != os.getpid():
                raise ValueError('New child identity was replaced before exec confirmation')
            if actual['cmdline_sha256'] == expected_hash:
                return actual
        if time.monotonic() >= deadline:
            raise TimeoutError('New child did not confirm its expected command')
        time.sleep(.01)
    raise ValueError('New child ended before exec confirmation')


def pinned(plan):
    project = Path(plan['project']).resolve()
    if Path(plan['entry']).resolve() != Path(__file__).resolve():
        raise ValueError('Continuation entry location changed')
    for key in ['run', 'paper_output', 'readback_manifest', 'original_entry', 'process_helper',
                'control_proof', 'ownership_proof', 'seed_candidate', 'preflight_manifest',
                'signatures_before', 'signatures_after', 'receipt']:
        if not Path(plan[key]).resolve().is_relative_to(project):
            raise ValueError('Continuation path is outside its project: ' + key)
    if not Path(plan['run']).resolve().is_relative_to(project / 'runs'):
        raise ValueError('Continuation state must belong to a project run')
    required = [plan[k] for k in ['process_helper', 'control_proof', 'ownership_proof', 'seed_candidate',
                                'preflight_manifest', 'signatures_before', 'signatures_after', 'original_entry']]
    if not all(path in plan['bound_files'] for path in required):
        raise ValueError('Continuation prerequisites must be explicitly byte-bound')
    if sha(__file__) != plan['entry_sha256']:
        raise ValueError('Continuation execution source changed')
    for name, expected in plan['bound_files'].items():
        if sha(name) != expected:
            raise ValueError('Continuation prerequisite changed: ' + name)
    for source in plan['sources'].values():
        if not Path(source['path']).resolve().is_relative_to(project):
            raise ValueError('Qualified execution source is outside its project')
        actual = {p.name: sha(p) for p in Path(source['path']).glob('*.py')}
        if actual != source['sha256']:
            raise ValueError('Qualified execution source differs')
    control = read(plan['control_proof'])
    if (control.get('status') != 'complete' or control.get('all_controls_passed') is not True or
            control.get('entry_sha256') != plan['entry_sha256']):
        raise ValueError('Continuation controls are incomplete or apply to another source')
    ownership = read(plan['ownership_proof'])
    for key, expected in plan['ownership_verification'].items():
        if ownership.get(key) != expected:
            raise ValueError('Original continuation ownership proof differs: ' + key)
    for name, key in [('original_parent', 'continuation'), ('original_reader', 'source_cache_base_reader')]:
        if any(ownership[key].get(field) != plan[name].get(field)
               for field in ['pid', 'start_ticks', 'ppid', 'cmdline_sha256']):
            raise ValueError('Original process is not bound by its completed ownership proof')


def validate_readback(plan):
    path = Path(plan['readback_manifest'])
    if not path.exists():
        return None
    proof = read(path)
    if proof.get('status') != 'complete' or proof.get('kind') != 'recorded_foundation_source_and_cache_readback':
        raise ValueError('Source/cache/base readback is not complete')
    if proof.get('code') != plan['readback_code']:
        raise ValueError('Readback execution source differs')
    for key, expected in plan['required_readback_verification'].items():
        if proof['verification'].get(key) != expected:
            raise ValueError('Readback verification differs: ' + key)
    if proof['verification'].get('student_training_performed_by_readback') is not False:
        raise ValueError('Readback is not an untouched foundation preparation')
    for name, expected in proof['outputs'].items():
        if not Path(name).resolve().is_relative_to(Path(plan['project']).resolve()):
            raise ValueError('Readback output is outside its project')
        if Path(name).stat().st_size != expected['bytes'] or sha(name) != expected['sha256']:
            raise ValueError('Readback output changed')
    for name, expected in plan['readback_input_bindings'].items():
        if (proof['inputs'].get(name) != expected or Path(name).stat().st_size != expected['bytes'] or
                sha(name) != expected['sha256']):
            raise ValueError('Readback prerequisite changed')
    if not proof['outputs'] or not plan['readback_input_bindings']:
        raise ValueError('Readback must bind its verification and original prerequisites')
    return proof


def readback_ready(plan, processes):
    if validate_readback(plan) is None:
        return False
    if not terminal(processes, plan['original_reader']):
        return False
    parent = processes.same(plan['original_parent'])
    if parent is None or parent['state'] in {'T', 't', 'Z', 'X'}:
        raise ValueError('Original parent is missing, terminal, or already stopped')
    if len(processes.tree(plan['original_parent'])) != 1:
        return False
    output = Path(plan['paper_output'])
    if output.exists():
        raise ValueError('Preserve an existing curriculum; fresh takeover requires no output')
    return True


def acquire_parent(plan, processes):
    parent = processes.same(plan['original_parent'])
    if parent is None or parent['state'] not in {'S', 'R', 'D'}:
        raise ValueError('Cannot adopt a missing or foreign-stopped parent')
    if len(processes.tree(plan['original_parent'])) != 1:
        raise ValueError('Original parent still owns a child')
    def record(value):
        if value['released'] or value['holds']:
            raise ValueError('Continuation pause was already acquired or released')
        value['holds'] = [parent]
    change(plan, record)
    if not processes.send(parent, signal.SIGSTOP):
        raise ValueError('Original parent changed before its owned pause')
    deadline = time.monotonic() + 10
    while True:
        actual = processes.same(parent)
        if actual is None or actual['state'] in {'Z', 'X'}:
            raise ValueError('Original parent ended before pause confirmation')
        if actual['state'] == 'T':
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Owned parent pause was not confirmed')
        time.sleep(.02)
    if len(processes.tree(parent)) != 1:
        raise ValueError('A child appeared during parent pause acquisition')


def retire(processes, expected):
    if terminal(processes, expected):
        return
    if not processes.same(expected, parent=False) or not processes.send(expected, signal.SIGKILL):
        raise ValueError('Owned retirement identity changed')
    deadline = time.monotonic() + 10
    while not terminal(processes, expected):
        if time.monotonic() >= deadline:
            raise TimeoutError('Owned process did not retire')
        time.sleep(.02)


def recover(plan, reason):
    processes = helper(plan)
    def restore(value):
        if value['released']:
            return
        if value['committed']:
            for expected in value.get('retire_intents', []):
                retire(processes, expected)
            value.update(released=True, retirement_complete=True, release_reason=reason)
            return
        worker = value.get('worker')
        if worker:
            if worker['prepare_only'] is not True:
                raise ValueError('Uncommitted continuation cannot own a training worker')
            for key in ['execution', 'bootstrap']:
                actual = processes.same(worker[key], parent=False)
                if actual:
                    for expected in reversed(processes.tree(actual)):
                        retire(processes, expected)
                    break
        output = Path(plan['paper_output'])
        if value.get('preparation_output_owned') and output.exists():
            if list(output.rglob('*.pt')) or list(output.rglob('*.safetensors')):
                raise ValueError('Uncommitted preparation unexpectedly contains model weights; preserve both owners stopped')
            destination = Path(plan['run']) / ('aborted-prepare-' + uuid.uuid4().hex)
            output.replace(destination)
            value['preserved_aborted_preparation'] = str(destination)
        resumed = []
        for expected in reversed(value['holds']):
            if processes.send(expected, signal.SIGCONT):
                resumed.append(expected['pid'])
        value.update(released=True, release_reason=reason, resumed_owned_parents=resumed)
    return change(plan, restore)


def watchdog(plan):
    processes = helper(plan)
    state_path = Path(plan['run']) / 'watchdog-state.json'
    while True:
        value = read(Path(plan['run']) / 'lease.json')
        if value['released']:
            write(state_path, {'status': 'released', 'utc': now()})
            return
        owner = processes.same(value['controller'], parent=False)
        if owner is None or owner['state'] in {'Z', 'X'}:
            recovered = recover(plan, 'controller_exact_identity_terminal')
            write(state_path, {'status': 'recovered', 'committed': recovered['committed'], 'utc': now()})
            return
        # Live D/S/R/T owners are not replaced merely for a stale observation.
        write(state_path, {'status': 'armed', 'controller': value['controller'], 'utc': now()})
        time.sleep(2)


class OriginalCalls:
    """Override only the frozen entry's subprocess reference, never the global module."""
    def __init__(self, plan):
        self.plan = plan

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, command, **kwargs):
        if '-m' not in command or command[command.index('-m') + 1] != 'xqgeneral.validation_curriculum':
            return subprocess.run(command, **kwargs)
        plan = self.plan
        lease = read(Path(plan['run']) / 'lease.json')
        prepare = '--prepare-only' in command
        if lease['released'] or (not prepare and not lease['committed']):
            raise ValueError('Training cannot start before committed capacity-qualified takeover')
        environment = dict(kwargs.pop('env'))
        environment.update(PYTHONPATH=str(Path(plan['gpu_source']).parent),
                           XQGENERAL_CODE_REVISION=plan['gpu_revision'],
                           XQGENERAL_VERIFIED_INPUT_RECEIPT=plan['receipt'])
        request_id = uuid.uuid4().hex
        request = Path(plan['run']) / ('worker-' + request_id + '.json')
        write(request, {'id': request_id, 'args': command, 'prepare_only': prepare})
        def pending(value):
            if value['released'] or (not prepare and not value['committed']):
                raise ValueError('Worker launch was released before registration')
            if prepare and not value['committed']:
                processes = helper(plan)
                if (not value['holds'] or any(not processes.same(p) or processes.same(p)['state'] != 'T'
                                             for p in value['holds']) or Path(plan['paper_output']).exists()):
                    raise ValueError('Fresh preparation output requires the confirmed owned parent pause')
                value['preparation_output_owned'] = True
            value.update(pending_worker=request_id, pending_worker_sha256=sha(request), worker=None)
        change(plan, pending)
        check = kwargs.pop('check', False)
        child = subprocess.Popen([sys.executable, plan['entry'], '--plan', plan['plan_file'],
                                  '--worker', str(request)], env=environment, start_new_session=True, **kwargs)
        code = child.wait()
        def completed(value):
            if value.get('pending_worker') == request_id:
                value.update(worker_returncode=code)
        change(plan, completed)
        if check and code:
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)


def worker(plan, request_path):
    processes = helper(plan)
    request = read(request_path)
    if (Path(request_path).resolve().parent != Path(plan['run']).resolve() or
            Path(request_path).name != 'worker-' + request['id'] + '.json'):
        raise ValueError('Worker request is outside its owned run')
    command = request['args']
    if (not isinstance(command, list) or not all(isinstance(x, str) for x in command) or
            command[:4] != [sys.executable, '-u', '-m', 'xqgeneral.validation_curriculum'] or
            request['prepare_only'] is not ('--prepare-only' in command)):
        raise ValueError('Unexpected owned worker command')
    current = processes.identity(os.getpid())
    execution = dict(current, cmdline_sha256=hashlib.sha256(b'\0'.join(os.fsencode(x) for x in command) + b'\0').hexdigest())
    def register(value):
        if (value['released'] or value['pending_worker'] != request['id'] or
                value['pending_worker_sha256'] != sha(request_path) or
                not processes.same(value['controller']) or current['ppid'] != value['controller']['pid'] or
                (not request['prepare_only'] and not value['committed'])):
            raise ValueError('Worker owner ended or launch was superseded')
        value['worker'] = {'bootstrap': current, 'execution': execution, 'prepare_only': request['prepare_only']}
    change(plan, register)
    lease = read(Path(plan['run']) / 'lease.json')
    if lease['released'] or not processes.same(lease['controller']):
        raise ValueError('Worker owner ended before execution')
    os.execvpe(command[0], command, os.environ)


def commit_capacity(plan, processes):
    def commit(value):
        if value['released'] or value['committed'] or not value['holds']:
            raise ValueError('Takeover cannot commit without its confirmed owned parent')
        if any(not processes.same(p) or processes.same(p)['state'] != 'T' for p in value['holds']):
            raise ValueError('Owned parent pause changed')
        value.update(committed=True, retire_intents=list(value['holds']))
    lease = change(plan, commit)
    for expected in lease['retire_intents']:
        retire(processes, expected)
    change(plan, lambda value: value.update(retirement_complete=True, holds=[]))


def execute_original(plan, processes):
    sys.path.insert(0, str(Path(plan['cpu_source']).parent))
    os.environ.update(PYTHONPATH=str(Path(plan['cpu_source']).parent),
                      XQGENERAL_CODE_REVISION=plan['cpu_revision'],
                      XQGENERAL_VERIFIED_INPUT_RECEIPT=plan['receipt'])
    from xqgeneral.verified_inputs import verified_input_session, write_verified_input_receipt
    candidate = read(plan['seed_candidate'])
    if (candidate.get('kind') != 'unpublished_verified_input_seed_candidate' or
            candidate['status'] != 'pending_complete_source_cache_base_readback' or candidate['consumer_receipt_published']):
        raise ValueError('Unexpected source-input seed candidate')
    before, after = read(plan['signatures_before']), read(plan['signatures_after'])
    preflight = read(plan['preflight_manifest'])
    allowed = [Path(plan['project']).resolve() / name for name in ['data', 'models']]
    expected = {name: {**binding, 'signature': before[name]} for name, binding in preflight['inputs'].items()
                if any(Path(name).resolve().is_relative_to(prefix) for prefix in allowed)}
    if (preflight.get('status') != 'complete' or before != after or candidate['bindings'] != expected or
            candidate['preflight_manifest_sha256'] != sha(plan['preflight_manifest']) or
            candidate['preflight_original_signatures_before_sha256'] != sha(plan['signatures_before']) or
            candidate['preflight_original_signatures_after_sha256'] != sha(plan['signatures_after'])):
        raise ValueError('Input seed is not the original successful full byte-hash binding')
    if validate_readback(plan) is None:
        raise ValueError('Complete source/cache/base readback disappeared')
    write_verified_input_receipt(plan['receipt'], candidate['bindings'], origin={
        'complete_source_cache_base_readback_sha256': sha(plan['readback_manifest']),
        'original_preflight_input_signatures_used': True})
    original = load_module('original_frozen_preparation_with_verified_inputs', plan['original_entry'])
    original.subprocess = OriginalCalls(plan)
    previous_wait = original.wait_gpu
    capacity = load_module('original_verified_capacity_gate', plan['capacity_entry'])
    token = load_module('original_full_length_capacity_binding', plan['token_helper'])
    token_plan = read(plan['token_plan'])

    def wait_gpu(indices, threshold, phase):
        if phase != 'waiting_for_formal_four_gpu_foundation_capacity':
            return previous_wait(indices, threshold, phase)
        if indices != [0, 1, 2, 3] or threshold != 32768:
            raise ValueError('Original four-card capacity contract changed')
        certificate = token.certify(token_plan['recipe'], token_plan['preflight_manifest'],
            token_plan['capacity_manifest'], token_plan['capacity_readback_manifest'], token_plan['corpus_manifest_sha256'])
        if certificate['total_rule_records'] != plan['question_rows']:
            raise ValueError('Capacity control covers another corpus')
        required = 90601
        if certificate['all_observed_inputs_covered_by_capacity_control']:
            required = certificate['conditional_required_free_mib_per_gpu']
            if required != 74884:
                raise ValueError('Conditional capacity threshold changed')
        write(Path(plan['run']) / 'formal-capacity-certificate.json', certificate)
        attempt = read(Path(plan['run']) / 'lease.json')['attempt_id']
        capacity.await_capacity(indices, required, original.state, phase,
                                Path(plan['run']) / ('formal-capacity-observations-' + attempt + '.jsonl'))
        pinned(plan)
        lease = read(Path(plan['run']) / 'lease.json')
        if lease['committed']:
            if not lease.get('retirement_complete') or not terminal(processes, plan['original_parent']):
                raise ValueError('Committed resume requires the original parent to remain terminal')
        else:
            commit_capacity(plan, processes)

    original.wait_gpu = wait_gpu
    with verified_input_session(plan['receipt']):
        original.main()
    if read(plan['preparation_verification']).get('four_courses_completed') is not True:
        raise ValueError('Original preparation did not finish all four courses')


def coordinate(plan, *, resume_committed=False):
    run = Path(plan['run'])
    processes = helper(plan)
    lease_path = run / 'lease.json'
    if resume_committed:
        previous = read(lease_path)
        if (not previous['committed'] or not previous['released'] or not terminal(processes, previous['controller']) or
                not terminal(processes, plan['original_parent']) or not Path(plan['paper_output']).exists()):
            raise ValueError('Resume requires a released committed owner, terminal original parent and preserved curriculum')
        worker_state = previous.get('worker')
        if worker_state and any(not terminal(processes, worker_state[key]) for key in ['bootstrap', 'execution']):
            raise ValueError('Preserve the still-live registered training worker; do not duplicate it')
        old_watch = previous.get('watchdog')
        deadline = time.monotonic() + 5
        while old_watch and not terminal(processes, old_watch):
            if time.monotonic() >= deadline:
                raise TimeoutError('Prior recovery process has not acknowledged its released lease')
            time.sleep(.05)
        if validate_readback(plan) is None:
            raise ValueError('Resume requires the complete original source/cache/base readback')
        pinned(plan)
        write(run / ('prior-committed-lease-' + uuid.uuid4().hex + '.json'), previous)
    elif lease_path.exists():
        raise ValueError('Preserve an existing continuation; use --resume-committed only after committed-owner recovery')
    controller = processes.identity(os.getpid())
    write(lease_path, {'controller': controller, 'holds': [], 'worker': None,
        'committed': resume_committed, 'retirement_complete': resume_committed, 'released': False,
        'output_initially_absent': not Path(plan['paper_output']).exists(), 'resume_committed': resume_committed,
        'attempt_id': uuid.uuid4().hex})
    try:
        with (run / 'watchdog.log').open('a') as log:
            watch_command = [sys.executable, plan['entry'], '--plan', plan['plan_file'], '--watchdog']
            watchdog_process = subprocess.Popen(watch_command,
                                                start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        watch = spawned_identity(processes, watchdog_process, watch_command)
        change(plan, lambda value: value.update(watchdog=watch))
        while True:
            if not processes.same(watch):
                raise ValueError('Independent recovery process ended')
            state_path = run / 'watchdog-state.json'
            recovery_state = read(state_path) if state_path.exists() else {}
            armed = recovery_state.get('status') == 'armed' and recovery_state.get('controller') == controller
            ready = validate_readback(plan) is not None if resume_committed else readback_ready(plan, processes)
            if armed and ready:
                break
            if not resume_committed and terminal(processes, plan['original_parent']):
                raise ValueError('Original preparation parent ended before takeover')
            write(run / 'state.json', {'status': 'waiting_for_complete_source_cache_base_readback', 'utc': now()})
            time.sleep(2)
        pinned(plan)
        if not resume_committed:
            acquire_parent(plan, processes)
        # Revalidate the completed proof after owning the parent, without
        # treating its intentional stop as a new foreign pause.
        accepted = validate_readback(plan)
        if accepted is None or not terminal(processes, plan['original_reader']):
            raise ValueError('Completed readback changed after pause acquisition')
        write(run / 'accepted-readback.json', {'sha256': sha(plan['readback_manifest']),
            'verification': accepted['verification'], 'utc': now()})
        write(run / 'state.json', {'status': 'running_qualified_input_reuse_preparation', 'utc': now()})
        execute_original(plan, processes)
        change(plan, lambda value: value.update(released=True, release_reason='four_courses_complete'))
        write(run / 'verification.json', {'status': 'complete', 'four_courses_completed': True,
            'original_frozen_source_files_modified': False, 'derived_execution_declared': True,
            'overall_goal_complete': False, 'utc': now()})
        write(run / 'exit.json', {'returncode': 0, 'utc': now()})
    except BaseException as exc:
        recover(plan, 'continuation_exception_' + type(exc).__name__)
        write(run / 'exit.json', {'returncode': 1, 'error_type': type(exc).__name__, 'utc': now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--resume-committed', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    plan['plan_file'] = str(Path(args.plan).resolve())
    pinned(plan)
    if args.watchdog:
        watchdog(plan)
    elif args.worker:
        worker(plan, args.worker)
    else:
        coordinate(plan, resume_committed=args.resume_committed)


if __name__ == '__main__':
    main()
