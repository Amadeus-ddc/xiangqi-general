"""Complete a verified native-reader handoff, then run the frozen four courses."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def write(path, value):
    atomic_bytes(Path(path), json_bytes(value))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, raw, *, exclusive=False):
    """Flush a complete file before atomically publishing its name."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                     suffix='.partial', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pinned(plan):
    assert sha(__file__) == plan['entry_sha256']
    for path, expected in plan['bound_files'].items():
        assert sha(path) == expected, 'Frozen continuation prerequisite changed'
    for field in ['cpu', 'gpu', 'reuse']:
        actual = {p.name: sha(p) for p in Path(plan[field + '_source']).glob('*.py')}
        assert actual == plan[field + '_source_sha256']
    control = read(plan['control_proof'])
    assert control['status'] == 'complete' and control['all_controls_passed'] is True
    assert control['entry_sha256'] == plan['entry_sha256']


def process_helper(plan):
    return load_file('owned_native_process_helpers', plan['process_helper'])


def terminal(helper, expected):
    actual = helper.identity(expected['pid'])
    return actual is None or actual['start_ticks'] != expected['start_ticks'] or actual['state'] in {'Z', 'X'}


def retire(helper, expected, timeout=10):
    # Only journalled, identity-bound handles may reach this function.
    if not terminal(helper, expected):
        assert helper.same(expected, parent=False), 'Retirement command identity changed'
        helper.send(expected, signal.SIGKILL)
    deadline = time.monotonic() + timeout
    while not terminal(helper, expected):
        assert time.monotonic() < deadline, 'Owned retired process has not ended'
        time.sleep(.02)


def pause_one(helper, lease_path, expected):
    actual = helper.same(expected)
    assert actual and actual['state'] in {'S', 'R'}, 'A foreign parent hold cannot be acquired'
    helper.change_lease(lease_path, lambda value: value['holds'].append(expected))
    assert helper.send(expected, signal.SIGSTOP)
    deadline = time.monotonic() + 5
    while (actual := helper.same(expected)) and actual['state'] != 'T':
        assert time.monotonic() < deadline
        time.sleep(.02)
    assert actual and actual['state'] == 'T'


def publish_gold(staged, target, published):
    """Copy the small leaf; keep the original completed proof and its inode intact."""
    staged, target, published = map(Path, [staged, target, published])
    proof = read(staged / 'manifest.json')
    leaf = staged / 'verification.json'
    assert proof['status'] == 'complete'
    assert proof['kind'] == 'paper_curriculum_complete_source_readback'
    assert proof['config']['output'] == str(staged)
    assert set(proof['outputs']) == {str(leaf)}
    raw = leaf.read_bytes()
    binding = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    assert proof['outputs'][str(leaf)] == binding
    assert json.loads(raw) == proof['verification']
    relocated = deepcopy(proof)
    relocated['config']['output'] = str(target)
    relocated['outputs'] = {str(target / leaf.name): binding}
    manifest_raw = json_bytes(relocated)
    preparing = published.with_name(published.name + '.preparing')
    journal = published.with_name(published.name + '.publication-intent.json')
    intent = {'kind': 'owned_native_gold_atomic_publication', 'schema_version': 1,
              'staged': str(staged), 'target': str(target), 'published': str(published),
              'staged_manifest_sha256': sha(staged / 'manifest.json'),
              'verification': binding,
              'published_manifest_sha256': hashlib.sha256(manifest_raw).hexdigest()}
    if os.path.lexists(target):
        assert target.is_symlink() and target.resolve() == published.resolve()
    if journal.exists():
        assert not journal.is_symlink() and read(journal) == intent, 'Publication intent changed'
    else:
        assert not os.path.lexists(published) and not os.path.lexists(preparing), 'Unknown publication directory'
        atomic_bytes(journal, json_bytes(intent), exclusive=True)
    if os.path.lexists(published):
        assert published.is_dir() and not published.is_symlink()
        assert (published / leaf.name).read_bytes() == raw
        assert (published / 'manifest.json').read_bytes() == manifest_raw
        assert set(p.name for p in published.iterdir()) == {leaf.name, 'manifest.json'}
    else:
        if os.path.lexists(preparing):
            assert preparing.is_dir() and not preparing.is_symlink()
            for path in preparing.iterdir():
                owned_partial = any(path.name.startswith('.' + name + '.') and path.name.endswith('.partial')
                                    for name in [leaf.name, 'manifest.json'])
                assert path.is_file() and not path.is_symlink()
                assert path.name in {leaf.name, 'manifest.json'} or owned_partial, 'Unknown staged publication file'
                if owned_partial:
                    path.unlink()
        else:
            preparing.mkdir()
            sync_directory(preparing.parent)
        # Retry only the directory bound by the durable intent. The completed
        # source is preserved, and the final directory can never be half written.
        atomic_bytes(preparing / leaf.name, raw)
        atomic_bytes(preparing / 'manifest.json', manifest_raw)
        sync_directory(preparing)
        os.rename(preparing, published)
        sync_directory(published.parent)
    if os.path.lexists(target):
        assert target.is_symlink() and target.resolve() == published.resolve()
    else:
        os.symlink(published, target)
        sync_directory(target.parent)
    assert sha(target / leaf.name) == binding['sha256']
    return {'staged_manifest_sha256': sha(staged / 'manifest.json'),
            'published_manifest_sha256': sha(target / 'manifest.json'),
            'original_completed_staging_preserved': True, 'output_paths_only_rebound': True}


def check_signatures(proof):
    for path, expected in proof['signatures'].items():
        assert signature(path) == expected, 'Independently verified gold artifact changed'


def recovery_live(plan, helper):
    path = RUN / 'watchdog-launch.json'
    if not path.exists() or not (RUN / 'watchdog-state.json').exists():
        return False
    value = read(path)
    actual = helper.same(value['identity'])
    if not actual or actual['state'] in {'Z', 'X'}:
        return False
    pane = subprocess.check_output(['tmux', 'list-panes', '-t', plan['watchdog_session'],
                                    '-F', '#{pane_pid} #{pane_dead}'], text=True).strip()
    state = read(RUN / 'watchdog-state.json')
    return (pane == value['pane'] and state['status'] == 'armed' and
            time.time() - datetime.fromisoformat(state['utc']).timestamp() < 20)


def ready(plan, helper, *, owned_handoff=False):
    prep, grouped = Path(plan['preparation']), Path(plan['grouped_run'])
    if (prep / 'question-readback/manifest.json').exists():
        return 'original_proof_already_published'
    wanted = [grouped / 'reader-exit.json', grouped / 'reader-verification.json',
              Path(plan['cache_manifest']), prep / 'preflight/manifest.json']
    if not all(p.exists() for p in wanted):
        return False
    assert read(grouped / 'reader-exit.json')['returncode'] == 0
    result = read(grouped / 'reader-verification.json')
    assert result['status'] == 'complete' and result['questions'] == plan['questions']
    assert result['staged_manifest_sha256'] == sha(grouped / 'staged-question-readback/manifest.json')
    for run in plan['completed_auxiliaries']:
        run = Path(run)
        if not read(run / 'lease.json')['released']:
            return False
        exit_name = plan['auxiliary_exit_receipts'][str(run)]
        assert exit_name in {'cache-exit.json', 'exit.json'}, 'Unknown auxiliary completion contract'
        assert read(run / exit_name)['returncode'] == 0
        assert read(run / 'verification.json')['status'] == 'complete'
    reuse = read(Path(plan['reuse_run']) / 'lease.json')
    assert not reuse['pause_requested'] and reuse['worker'] is None
    for run in plan['capacity_guards']:
        value = read(Path(run) / 'lease.json')
        if value['hold_requested']:
            return False
    if any((actual := helper.same(p, parent=False)) and actual['state'] in {'T', 't'}
           for p in plan['waiting_actors']):
        return False
    assert read(prep / 'state.json')['status'] == 'regenerating_all_paper_question_gold_and_order'
    assert helper.same(plan['original_parent']) and helper.same(plan['original_gold'])
    if plan.get('gold_guard_run'):
        guard = Path(plan['gold_guard_run'])
        if not (guard / 'held-ready.json').exists():
            return False
        previous = read(guard / 'lease.json')
        if owned_handoff:
            assert previous['released'], 'Previous gold owner still owns its pause'
            held = read(RUN / 'lease.json')
            assert not held['released'] and not held['committed']
            expected = [plan['original_parent'], *plan['original_gold_tree']]
        else:
            held = previous
            if held['released']:
                return False
            expected = plan['original_gold_tree']
        assert held['reader'] is None and helper.same(held['controller'])
        keys = ('pid', 'start_ticks', 'ppid', 'cmdline_sha256')
        expected_keys = {tuple(p[k] for k in keys) for p in expected}
        actual_keys = {tuple(p[k] for k in keys) for p in held['holds']}
        assert len(held['holds']) == len(expected) and actual_keys == expected_keys
        assert all((actual := helper.same(p)) and actual['state'] == 'T' for p in held['holds'])
        ready = read(guard / 'held-ready.json')
        assert ready['staged_manifest_sha256'] == sha(grouped / 'staged-question-readback/manifest.json')
    return True


def verify_gold(plan):
    pinned(plan)
    grouped = Path(plan['grouped_run'])
    path = grouped / 'staged-question-readback/manifest.json'
    receipt = read(grouped / 'reader-verification.json')
    assert read(grouped / 'reader-exit.json')['returncode'] == 0
    assert receipt['questions'] == plan['questions'] and receipt['staged_manifest_sha256'] == sha(path)
    proof = read(path)
    assert proof['code'] == read(plan['corpus_manifest'])['code']
    runtime = proof['runtime_execution']
    assert runtime['math_module_code'] == proof['code']
    assert runtime['every_original_question_byte_regenerated_and_compared'] is True
    assert runtime['every_selected_history_checked_by_original_native_worker'] is True
    assert runtime['original_full_split_verification_executed'] is True
    for name, embedded in runtime['embedded_execution_artifacts'].items():
        raw = (grouped / name).read_bytes()
        assert embedded == {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                            'utf8_source_or_plan': raw.decode('utf-8')}
    sys.path.insert(0, str(Path(plan['reuse_source']).parent))
    from xqgeneral.paper_readback_reuse import verify_completed_readback
    verified = verify_completed_readback(Path(plan['corpus_manifest']).parent, path)
    assert verified.verification['questions_regenerated_and_compared'] == plan['questions']
    verified.check_unchanged()
    pinned(plan)
    write(RUN / 'gold-verification.json', {'status': 'complete', 'questions': plan['questions'],
        'staged_manifest_sha256': verified.manifest_sha256,
        'all_completed_gold_source_and_output_bindings_freshly_rehashed': True,
        'signatures': {str(path): list(value) for path, value in verified.signatures.items()},
        'verifier_code_revision': plan['reuse_revision'], 'utc': now()})


def finish_commit(plan, helper):
    """Recoverable commit point: retirement was fully authorized before this call."""
    with (RUN / 'commit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = read(RUN / 'lease.json')
        if value['released']:
            return
        assert value['committed']
        assert sha(RUN / 'gold-verification.json') == value['gold_verification_sha256']
        check_signatures(read(RUN / 'gold-verification.json'))
        # The parent is first, so killing its child cannot run the old exception path.
        for expected in value['retire_intents']:
            retire(helper, expected)
        if 'reuse_run' in plan:
            old_reuse = Path(plan['reuse_run'])
            old = read(old_reuse / 'lease.json')
            assert old['worker'] is None and not old['pause_requested']
            old.update(released=True, readback_accepted=False, released_utc=now(),
                       release_reason='waiting_queue_replaced_by_verified_training_continuation')
            write(old_reuse / 'lease.json', old)
            write(old_reuse / 'retirement.json', {'status': 'waiting_queue_retired',
                'replacement': str(RUN), 'worker_was_not_started': True, 'utc': now()})
            for run in plan['capacity_guards']:
                directory = Path(run)
                old = read(directory / 'lease.json')
                assert not old['hold_requested']
                old.update(complete=True, superseded_by=str(RUN),
                           terminal_status='synchronous_capacity_gate_in_continuation', heartbeat=time.time())
                write(directory / 'lease.json', old)
        prep = Path(plan['preparation'])
        target = prep / 'question-readback'
        archived = RUN / 'retired-original-question-readback'
        if not archived.exists():
            assert target.is_dir() and not target.is_symlink() and not list(target.iterdir())
            assert signature(target) == value['original_empty_target_signature']
            target.rename(archived)
        publication = publish_gold(Path(plan['grouped_run']) / 'staged-question-readback',
                                   target, RUN / 'published-question-readback')
        assert all(terminal(helper, p) for p in value['retire_intents'])
        helper.change_lease(RUN / 'lease.json', lambda v: v.update(
            released=True, release_reason='verified_original_processes_retired',
            retirement_complete=True, publication=publication, released_utc=now()))
        write(RUN / 'continuation-permit.json', {'status': 'permitted', 'plan_sha256': sha(RUN / 'plan.json'),
            'old_processes_confirmed_terminal': True, 'publication': publication, 'utc': now()})
        write(RUN / 'handoff-verification.json', {'status': 'complete', 'publication': publication,
            'original_parent_and_gold_workers_confirmed_terminal': True,
            'old_exception_or_zero_exit_not_forged': True, 'formal_new_curriculum_updates': 0,
            'overall_goal_complete': False, 'utc': now()})


def recover(plan, helper, reason):
    value = read(RUN / 'lease.json')
    if value['committed']:
        finish_commit(plan, helper)
        return
    helper.release(RUN / 'lease.json', reason, stop_candidate=True)
    write(RUN / 'continuation-abort.json', {'reason': reason, 'utc': now()})


def watchdog(plan, helper):
    while not (RUN / 'lease.json').exists():
        time.sleep(1)
    while True:
        value = read(RUN / 'lease.json')
        if value['released']:
            if value['committed'] and not (RUN / 'continuation-permit.json').exists():
                # finish_commit writes the permit before its coordinator exits;
                # a crash between the release and permit must also be recoverable.
                helper.change_lease(RUN / 'lease.json', lambda v: v.update(released=False))
                finish_commit(plan, helper)
            write(RUN / 'watchdog-state.json', {'status': 'complete_owned_handoff_released', 'utc': now()})
            return
        actor = helper.same(value['controller'])
        if not actor or actor['state'] in {'Z', 'X'} or time.time() - value['heartbeat'] > 90:
            recover(plan, helper, 'coordinator_missing_or_stale')
            write(RUN / 'watchdog-state.json', {'status': 'recovered_owned_handoff', 'committed': value['committed'], 'utc': now()})
            return
        write(RUN / 'watchdog-state.json', {'status': 'armed', 'utc': now(), 'committed': value['committed']})
        time.sleep(5)


def continuation(plan):
    pinned(plan)
    write(RUN / 'continuation-ready.json', {'pid': os.getpid(), 'plan_sha256': sha(RUN / 'plan.json'), 'utc': now()})
    while not (RUN / 'continuation-permit.json').exists():
        if (RUN / 'continuation-abort.json').exists():
            raise RuntimeError('Pre-commit handoff aborted; original pipeline retained')
        time.sleep(1)
    permit = read(RUN / 'continuation-permit.json')
    assert permit['plan_sha256'] == sha(RUN / 'plan.json')
    assert sha(Path(plan['preparation']) / 'question-readback/manifest.json') == permit['publication']['published_manifest_sha256']
    # The same optional 100-module readback runs in its own fresh interpreter.
    prep, staging = Path(plan['preparation']), RUN / 'staged-foundation-readback'
    environment = dict(os.environ, PYTHONPATH=str(Path(plan['reuse_source']).parent),
                       XQGENERAL_CODE_REVISION=plan['reuse_revision'], CUDA_VISIBLE_DEVICES='')
    args = [sys.executable, '-u', '-m', 'xqgeneral.foundation_readback', '--config', str(prep / 'foundation.config.json'),
        '--preflight', str(prep / 'preflight'), '--base-proof', plan['base_proof'], '--output', str(staging),
        '--cpu-threads', '4', '--public-evidence', str(RUN / 'foundation-readback-summary.json'),
        '--completed-question-readback', str(prep / 'question-readback/manifest.json')]
    write(RUN / 'continuation-state.json', {'status': 'full_source_cache_and_base_readback', 'utc': now()})
    with (RUN / 'foundation-readback.log').open('a') as log:
        subprocess.run(args, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = read(staging / 'manifest.json')
    assert receipt['status'] == 'complete'
    assert receipt['code'] == {'revision': plan['reuse_revision'], 'source_sha256':
        {f'src/xqgeneral/{n}': h for n, h in plan['reuse_source_sha256'].items()}}
    v = receipt['verification']
    assert v['paper_questions_regenerated'] == plan['questions'] and v['paper_question_readback_reused'] is True
    assert v['reused_native_readback_artifacts_freshly_rehashed'] is True
    assert v['student_training_performed_by_readback'] is False
    publisher = load_file('original_reuse_output_publisher', plan['reuse_entry'])
    publisher.relocate_complete(staging, prep / 'readback', receipt)
    write(prep / 'readback-summary.json', dict(v, manifest_sha256=sha(prep / 'readback/manifest.json')))
    pinned(plan)
    sys.path.insert(0, str(Path(plan['cpu_source']).parent))
    os.environ.update(PYTHONPATH=str(Path(plan['cpu_source']).parent), XQGENERAL_CODE_REVISION=plan['cpu_revision'])
    original = load_file('original_frozen_paper_preparation', prep / 'entry.py')
    old_wait = original.wait_gpu

    def wait_gpu(indices, threshold, phase):
        if phase != 'waiting_for_formal_four_gpu_foundation_capacity':
            return old_wait(indices, threshold, phase)
        assert indices == [0, 1, 2, 3] and threshold == 32768
        token_plan = read(plan['token_plan'])
        token = load_file('same_completed_full_length_capacity_binding', plan['token_helper'])
        certificate = token.certify(token_plan['recipe'], token_plan['preflight_manifest'],
            token_plan['capacity_manifest'], token_plan['capacity_readback_manifest'], token_plan['corpus_manifest_sha256'])
        assert certificate['total_rule_records'] == plan['questions']
        actual = 90601
        if certificate['all_observed_inputs_covered_by_capacity_control']:
            actual = certificate['conditional_required_free_mib_per_gpu']
            assert actual == 74884
        write(RUN / 'formal-capacity-certificate.json', certificate)
        await_capacity(indices, actual, original.state, phase, RUN / 'formal-capacity-observations.jsonl')

    original.wait_gpu = wait_gpu
    write(RUN / 'continuation-state.json', {'status': 'original_frozen_preparation_resumed_with_synchronous_capacity_gate',
        'source_module_files_or_recipe_modified': False, 'original_entry_sha256': sha(prep / 'entry.py'), 'utc': now()})
    original.main()
    assert read(prep / 'verification.json')['four_courses_completed'] is True
    write(RUN / 'continuation-verification.json', {'status': 'complete', 'four_courses_completed': True,
        'original_frozen_entry_used': True, 'additional_runtime_adapter_declared': True,
        'synchronous_four_gpu_capacity_gate_executed': True, 'overall_goal_complete': False, 'utc': now()})


def await_capacity(indices, threshold, state, phase, observations, *, stable_seconds=30, poll_seconds=5, sampler=None):
    def actual_sample():
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=10)
        return {int(line.split(',')[0]): int(line.split(',')[1]) for line in raw.splitlines()}
    sample = sampler or actual_sample
    since = None
    with Path(observations).open('x') as log:
        while True:
            values = sample()
            available = all(values.get(index, 0) >= threshold for index in indices)
            since = (since or time.monotonic()) if available else None
            log.write(json.dumps({'utc': now(), 'free_mib': values, 'required_mib': threshold,
                                  'all_four_qualified': available}) + '\n')
            log.flush()
            state(phase, gpu_indices=indices, required_free_mib_per_gpu=threshold,
                  continuous_qualified_seconds=0 if since is None else time.monotonic() - since)
            if since is not None and time.monotonic() - since >= stable_seconds:
                return
            time.sleep(poll_seconds)


def coordinate(plan, helper):
    path = RUN / 'lease.json'
    assert not path.exists()
    write(path, {'controller': helper.identity(os.getpid()), 'reader': None, 'holds': [],
                 'released': False, 'committed': False, 'heartbeat': time.time(), 'created_utc': now()})

    def beat(value=None):
        helper.change_lease(path, lambda lease: lease.update(heartbeat=time.time()))
        lease = read(path)
        if lease['holds'] and not lease['committed']:
            assert recovery_live(plan, helper), 'Independent recovery disappeared during owned handoff'

    try:
        while True:
            beat()
            result = ready(plan, helper)
            if result == 'original_proof_already_published':
                helper.release(path, 'original_complete_proof_retained', stop_candidate=False)
                return
            write(RUN / 'state.json', {'status': 'waiting_for_full_grouped_gold_cache_and_preflight',
                'utc': now(), 'formal_new_curriculum_updates': 0,
                'original_parent_live': helper.same(plan['original_parent']) is not None,
                'original_gold_live': helper.same(plan['original_gold']) is not None})
            if result:
                break
            time.sleep(5)
        pinned(plan)
        with (RUN / 'gold-verifier.log').open('x') as log:
            verifier = subprocess.Popen([sys.executable, '-u', __file__, '--role', 'verify-gold'], cwd=ROOT,
                                        stdout=log, stderr=subprocess.STDOUT)
            helper.change_lease(path, lambda value: value.update(reader=helper.identity(verifier.pid)))
            while verifier.poll() is None:
                beat()
                time.sleep(5)
            assert verifier.returncode == 0
        helper.change_lease(path, lambda value: value.update(reader=None))
        assert ready(plan, helper) is True
        check_signatures(read(RUN / 'gold-verification.json'))
        # Retire only the completed candidate's waiting guard. Its original
        # recovery releases its own gold stop before this queue acquires one.
        grouped = Path(plan.get('gold_guard_run', plan['grouped_run']))
        old_lease = read(grouped / 'lease.json')
        if not old_lease['released']:
            controller = old_lease['controller']
            actual = helper.same(controller)
            assert actual and actual['state'] not in {'T', 't', 'Z', 'X'}
            retire(helper, controller)
            helper.release(grouped / 'lease.json', 'completed_gold_transferred_to_continuation', stop_candidate=False)
        # Original cache/preflight are complete; the optional old readback and
        # capacity waiters have no worker or pause to preserve.
        for expected in plan['waiting_actors']:
            actual = helper.same(expected, parent=False)
            assert not actual or actual['state'] not in {'T', 't'}
        assert recovery_live(plan, helper), 'Independent recovery must be live before acquiring any pause'
        pause_one(helper, path, plan['original_parent'])
        helper.pause_tree(path, plan['original_gold'], plan['original_gold_tree'])
        target = Path(plan['preparation']) / 'question-readback'
        assert target.is_dir() and not target.is_symlink() and not list(target.iterdir())
        command = 'exec ' + shlex.join([sys.executable, '-u', __file__, '--role', 'continue'])
        command += ' >> ' + shlex.quote(str(RUN / 'continuation.log')) + ' 2>&1'
        pane_id = subprocess.check_output(['tmux', 'new-window', '-d', '-t', plan['original_session'],
            '-n', plan['continuation_window'], '-P', '-F', '#{pane_id}', '-c', str(ROOT), command], text=True).strip()
        pane = subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane_id,
            '#{pane_pid} #{pane_dead}'], text=True).strip()
        assert pane.endswith(' 0')
        child = helper.identity(int(pane.split()[0]))
        # Wait through tmux's shell exec before binding the final Python argv.
        deadline = time.monotonic() + 5
        while str(Path(__file__).resolve()).encode() not in Path(f"/proc/{child['pid']}/cmdline").read_bytes().split(b'\0'):
            assert time.monotonic() < deadline
            time.sleep(.02)
        child = helper.identity(child['pid'])
        deadline = time.monotonic() + 30
        while not (RUN / 'continuation-ready.json').exists():
            beat()
            assert helper.same(child) and time.monotonic() < deadline
            time.sleep(.1)
        prepared = read(RUN / 'continuation-ready.json')
        assert prepared['pid'] == child['pid'] and prepared['plan_sha256'] == sha(RUN / 'plan.json')
        check_signatures(read(RUN / 'gold-verification.json'))
        assert ready(plan, helper, owned_handoff=True) is True
        intents = [plan['original_parent'], *reversed(plan['original_gold_tree']), *plan['waiting_actors']]
        helper.change_lease(path, lambda value: value.update(committed=True, retire_intents=intents,
            continuation_identity=child, continuation_pane=pane_id,
            gold_verification_sha256=sha(RUN / 'gold-verification.json'),
            original_empty_target_signature=signature(target), heartbeat=time.time()))
        finish_commit(plan, helper)
        write(RUN / 'state.json', {'status': 'complete_verified_handoff_continuation_running', 'utc': now(),
            'continuation_identity': child, 'formal_new_curriculum_updates': 0, 'overall_goal_complete': False})
    except BaseException:
        recover(plan, helper, 'handoff_failed_before_or_during_commit')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['coordinate', 'watchdog', 'verify-gold', 'continue'], required=True)
    role = parser.parse_args().role
    plan = read(RUN / 'plan.json')
    pinned(plan)
    helper = process_helper(plan)
    if role == 'watchdog':
        watchdog(plan, helper)
    elif role == 'verify-gold':
        verify_gold(plan)
    elif role == 'continue':
        try:
            continuation(plan)
        except BaseException as error:
            write(RUN / 'continuation-exit.json', {'returncode': 1, 'error_type': type(error).__name__, 'utc': now()})
            if (RUN / 'continuation-permit.json').exists():
                write(Path(plan['preparation']) / 'exit.json', {'returncode': 1, 'error_type': type(error).__name__,
                    'owned_continuation': str(RUN), 'utc': now()})
            raise
        write(RUN / 'continuation-exit.json', {'returncode': 0, 'utc': now()})
    else:
        coordinate(plan, helper)
        write(RUN / 'coordinator-exit.json', {'returncode': 0, 'utc': now()})


if __name__ == '__main__':
    main()
