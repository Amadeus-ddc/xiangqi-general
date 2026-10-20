import copy
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import pytest

from test_engine_qualified_explanations import label
from test_reviewed_explanations import source
from xqgeneral import training_label_reference as reference
from xqgeneral.engine_qualified_explanations import qualify
from xqgeneral.evidence import atomic_json, digest
from xqgeneral.reference_oracle import ReferencePikafish
from xqgeneral.rules import START_FEN, legal_moves, play, replay
from xqgeneral.training_label_reference_readback import verify_row


@pytest.fixture
def case(tmp_path, monkeypatch):
    rows = [label('a', 'train', ['b0c2']), label('b', 'train', ['h0g2']),
            label('c', 'train', ['b2e2'])]
    data = source(tmp_path / 'data', [*rows, label('v', 'validation', ['h2e2']),
                                    label('t', 'test', ['a0a1'])])
    executable, weights = tmp_path / 'fixture-engine', tmp_path / 'fixture-weights'
    executable.write_bytes(b'controlled engine fixture, no subprocess executable')
    weights.write_bytes(b'controlled weight identity fixture')
    book = {}
    for row in rows:
        position = row['fen']
        for move in json.loads(row['answer'])['pv']:
            book[position] = move
            position = play(position, move)
    bad = play(rows[1]['fen'], json.loads(rows[1]['answer'])['pv'][0])
    control = dict(fail_row=None, unscored=False, constructors=0, closed=0, calls=[])

    class Engine(ReferencePikafish):
        def __init__(self, executable, weights, threads=1):
            self.begin_row(None)
            self.returned_unscored = False
            control['constructors'] += 1

        def search(self, fen, nodes, initial_fen, moves, searchmoves=None):
            assert replay(initial_fen, moves)[-1] == fen
            control['calls'].append(dict(row=self.row_index, fen=fen, nodes=nodes,
                                         initial_fen=initial_fen, moves=list(moves), searchmoves=searchmoves))
            if control['fail_row'] == self.row_index:
                raise RuntimeError('controlled original engine failure')
            move = searchmoves[0] if searchmoves else book[fen]
            score = -1000 if searchmoves and fen == bad else 0
            if not searchmoves and fen == bad:
                move = next(value for value in legal_moves(fen) if value != book[fen])
            if control['unscored'] and fen == rows[0]['fen'] and not searchmoves and not self.returned_unscored:
                self.returned_unscored = True
                return ['bestmove ' + move]
            return [f'info depth 8 multipv 1 score cp {score} pv {move}', 'bestmove ' + move]

        def close(self):
            control['closed'] += 1

    monkeypatch.setattr(reference, 'ReferencePikafish', Engine)
    args = dict(data=data, output=tmp_path / 'reference', executable=executable,
                weights=weights, workers=1, nodes=1000000, artifact_root=tmp_path)
    return args, rows, control


def test_full_reference_runs_a_real_reader_process_then_qualifies_exact_source_bytes(case, monkeypatch):
    args, rows, control = case
    data = args['data']
    before = {split: (data / f'{split}.jsonl').read_bytes() for split in ['train', 'validation', 'test']}
    real_open = Path.open
    real_popen = subprocess.Popen
    inherited_locks = []

    def start_reader(*a, **kw):
        if '--reference' in a[0]:
            descriptors = kw.get('pass_fds', ())
            assert len(descriptors) == 1
            inherited_locks.append(os.fstat(descriptors[0]).st_ino)
        return real_popen(*a, **kw)

    def train_only(path, *a, **kw):
        if path in {data / 'validation.jsonl', data / 'test.jsonl'}:
            raise AssertionError('The train-only reference must not open held-out files')
        return real_open(path, *a, **kw)

    with monkeypatch.context() as guard:
        guard.setattr(Path, 'open', train_only)
        guard.setattr(subprocess, 'Popen', start_reader)
        metrics = reference.run_reference(**args)
    output = args['output']
    reader = reference.read(output / 'full/independent-readback.json')
    assert metrics['examples'] == 3 and metrics['full_pv_no_mistake_count'] == 2
    assert reader['actual_retained_metric_counts']['first_move_no_mistake'] == 3
    assert reader['actual_retained_metric_counts']['full_pv_no_mistake'] == 2
    assert reader['actual_engine_calls_in_this_reader'] == 0
    assert reader['all_ordered_engine_requests_bound_to_original_full_histories']
    worker = reference.read(output / 'attempts/000001/readback-worker.json')
    exit_proof = reference.read(output / 'full-readback-exit.json')
    assert worker['pid'] == exit_proof['actual_readback_pid'] and exit_proof['returncode'] == 0
    assert exit_proof['verification_sha256'] == digest(output / 'full/independent-readback.json')
    assert control['constructors'] == control['closed'] == 1
    assert inherited_locks == [(output / '.run.lock').stat().st_ino]
    selected = output.parent / 'qualified'
    result = qualify(data, output, selected, artifact_root=output.parent, workers=1)
    assert result['by_split'] == {'train': 2, 'validation': 1, 'test': 1}
    assert (selected / 'train.jsonl').read_bytes() == b''.join(
        line for line in before['train'].splitlines(keepends=True) if json.loads(line)['id'] in {'a', 'c'})
    for split in ['validation', 'test']:
        assert (selected / f'{split}.jsonl').read_bytes() == before[split]


def test_completed_resume_rebinds_receipts_without_running_another_engine_or_rewriting_outputs(case):
    args, _, control = case
    first = reference.run_reference(**args)
    output = args['output']
    hashes = {str(path): digest(path) for path in output.rglob('*') if path.is_file()}
    constructors = control['constructors']
    assert reference.run_reference(**args, resume=True) == first
    assert control['constructors'] == constructors
    assert {name: digest(name) for name in hashes} == hashes


def test_partial_resume_preserves_completed_rows_and_the_original_failure(case):
    args, _, control = case
    control['fail_row'] = 1
    with pytest.raises(RuntimeError, match='controlled original'):
        reference.run_reference(**args)
    output = args['output']
    row = output / 'full/rows/000000.json'
    original = row.read_bytes()
    failure = output / 'attempts/000001/failed-row-000001.json'
    failure_bytes = failure.read_bytes()
    assert reference.read(failure)['last_engine_request']['raw_output'] is None
    assert not (output / 'full-pipeline-exit.json').exists()
    control['fail_row'] = None
    control['calls'].clear()
    reference.run_reference(**args, resume=True)
    assert row.read_bytes() == original and failure.read_bytes() == failure_bytes
    assert {value['row'] for value in control['calls']} == {1, 2}
    assert reference.read(output / 'full-pipeline-exit.json')['independent_readback_complete']


@pytest.mark.parametrize('problem', ['loss', 'history', 'unused_request'])
def test_independent_reader_rejects_changed_math_history_or_unaccounted_queries(case, problem):
    args, rows, _ = case
    reference.run_reference(**args)
    output = args['output']
    value = reference.read(output / 'full/rows/000000.json')
    if problem == 'loss':
        value['judgment']['pv_move_losses'][0] = .2
    if problem == 'history':
        value['engine_requests'][0]['moves'] = []
    if problem == 'unused_request':
        value['engine_requests'].append(copy.deepcopy(value['engine_requests'][0]))
    with pytest.raises(ValueError):
        verify_row(value, rows[0], 0, reference.row_binding(output, reference.read(output / 'plan.json')))


def test_capability_scope_cannot_masquerade_as_a_complete_qualification_reference(case):
    args, _, _ = case
    result = reference.run_reference(**args, scope='capability', indices=[0, 2])
    output = args['output']
    assert result['examples'] == 2 and not result['all_source_training_labels_scored']
    assert not (output / 'full').exists()
    with pytest.raises(FileNotFoundError):
        qualify(args['data'], output, output.parent / 'not-qualified', artifact_root=output.parent)
    assert not (output.parent / 'not-qualified').exists()


def test_same_best_score_recovery_keeps_both_actual_requests_and_is_independently_reparsed(case):
    args, _, control = case
    control['unscored'] = True
    metrics = reference.run_reference(**args)
    reader = reference.read(args['output'] / 'full/independent-readback.json')
    row = reference.read(args['output'] / 'full/rows/000000.json')
    assert metrics['additional_best_score_requests'] == reader['explicitly_verified_additional_engine_requests'] == 1
    assert row['engine_requests'][0]['searchmoves'] is None
    best = row['engine_requests'][0]['raw_output'][-1].split()[1]
    assert row['engine_requests'][1]['searchmoves'] == [best]
    assert row['oracle_exact_score_recoveries'][0]['strict_result']['best_move'] == best


@pytest.mark.parametrize('problem', ['node_budget', 'weights', 'row_bytes', 'reader_exit', 'reader_launcher', 'source_modules'])
def test_changed_completed_reference_or_resume_configuration_refuses_before_new_engine(case, problem, monkeypatch):
    args, _, control = case
    reference.run_reference(**args)
    output = args['output']
    before = control['constructors']
    if problem == 'node_budget':
        args = dict(args, nodes=2000000)
    if problem == 'weights':
        args['weights'].write_bytes(b'changed engine weight identity')
    if problem == 'row_bytes':
        path = output / 'full/rows/000000.json'
        path.write_bytes(path.read_bytes() + b' ')
    if problem == 'reader_exit':
        path = output / 'full-readback-exit.json'
        value = reference.read(path);value['returncode'] = 1;atomic_json(path, value)
    if problem == 'reader_launcher':
        (output / 'readback.py').write_text('# changed reader launcher\n')
    if problem == 'source_modules':
        previous = reference.source_hashes()
        monkeypatch.setattr(reference, 'source_hashes', lambda: dict(previous, changed_module='a' * 64))
    with pytest.raises(ValueError):
        reference.run_reference(**args, resume=True)
    assert control['constructors'] == before


@pytest.mark.parametrize('interruption', ['manifest.json', 'exit.json', 'full-pipeline-exit.json'])
def test_resume_finishes_interrupted_publication_without_rescoring_or_replacing_completed_bytes(
        case, monkeypatch, interruption):
    args, _, control = case
    output = args['output']
    target = output / interruption if interruption.startswith('full-') else output / 'full' / interruption
    original = reference.write_once

    def interrupted(path, value):
        if Path(path) == target:
            raise OSError('controlled interruption before publishing completion')
        return original(path, value)

    with monkeypatch.context() as interrupt:
        interrupt.setattr(reference, 'write_once', interrupted)
        with pytest.raises(OSError, match='controlled interruption'):
            reference.run_reference(**args)
    finished = {str(path): digest(path) for path in (output / 'full').rglob('*.json')}
    before = control['constructors']
    reference.run_reference(**args, resume=True)
    assert control['constructors'] == before
    assert all(digest(path) == expected for path, expected in finished.items())
    assert reference.read(output / 'full-pipeline-exit.json')['returncode'] == 0


def test_tampered_unfinished_row_is_recomputed_from_raw_requests_before_any_new_search(case):
    args, _, control = case
    control['fail_row'] = 1
    with pytest.raises(RuntimeError):
        reference.run_reference(**args)
    row = args['output'] / 'full/rows/000000.json'
    value = reference.read(row)
    value['judgment']['first_move_expected_score_loss'] = .5
    atomic_json(row, value)
    before = control['constructors']
    control['fail_row'] = None
    with pytest.raises(ValueError, match='first-move loss'):
        reference.run_reference(**args, resume=True)
    assert control['constructors'] == before


def test_an_added_heldout_input_is_refused_before_opening_it(case, monkeypatch):
    args, _, control = case
    reference.run_reference(**args)
    output = args['output']
    heldout = args['data'] / 'test.jsonl'
    path = output / 'plan.json'
    value = reference.read(path)
    value['inputs'][str(heldout)] = reference.bound(heldout)
    atomic_json(path, value)
    real_open = Path.open

    def guarded(path, *a, **kw):
        if path == heldout:
            raise AssertionError('A changed reference plan must not admit held-out input')
        return real_open(path, *a, **kw)

    before = control['constructors']
    monkeypatch.setattr(Path, 'open', guarded)
    with pytest.raises(ValueError, match='input schema'):
        reference.run_reference(**args, resume=True)
    assert control['constructors'] == before


def test_existing_outputs_and_live_owner_are_preserved(case):
    args, _, control = case
    reference.run_reference(**args)
    output = args['output']
    before = control['constructors']
    with pytest.raises(FileExistsError):
        reference.run_reference(**args)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='owner is still live'):
            reference.run_reference(**args, resume=True)
    assert control['constructors'] == before


@pytest.mark.parametrize('problem', ['changed_best', 'timeout', 'illegal_best'])
def test_extra_score_policy_refuses_other_protocol_errors_or_a_different_best_move(problem):
    engine = ReferencePikafish.__new__(ReferencePikafish)
    engine.begin_row(0)
    first, second = legal_moves(START_FEN)[:2]
    calls = []

    def search(fen, nodes, initial_fen, moves, searchmoves=None):
        calls.append(searchmoves)
        if problem == 'timeout':
            raise TimeoutError('controlled timeout, no retry')
        if problem == 'illegal_best':
            return ['bestmove a0a9']
        if searchmoves is None:
            return ['bestmove ' + first]
        return [f'info depth 8 score cp 0 pv {second}', 'bestmove ' + second]

    engine.search = search
    with pytest.raises((RuntimeError, TimeoutError)):
        engine.analyze(START_FEN, 1000000, START_FEN, [])
    assert not engine.recoveries
    assert len(calls) == (2 if problem == 'changed_best' else 1)


@pytest.mark.skipif(not hasattr(os, 'pidfd_open'), reason='Linux process descriptors required')
def test_orphan_reader_keeps_the_lock_after_its_original_parent_exits(tmp_path):
    parent = '''from pathlib import Path
import hashlib,json,os,subprocess,sys
from xqgeneral.training_label_reference import owned_run
root=Path(sys.argv[1])
with owned_run(root/'reference',False) as descriptor:
 (root/'reference/plan.json').write_text('{"purpose":"controlled_lock_probe"}')
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],pass_fds=(descriptor,),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 proc=Path('/proc')/str(child.pid);stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
 (root/'child.json').write_text(json.dumps(dict(pid=child.pid,start_time=int(stat[19]),command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())))
 os._exit(0)
'''
    env = dict(os.environ, PYTHONPATH=str(Path(reference.__file__).resolve().parent.parent))
    subprocess.run([sys.executable, '-P', '-c', parent, str(tmp_path)], env=env, check=True)
    child = reference.read(tmp_path / 'child.json')
    import hashlib
    proc = Path('/proc') / str(child['pid'])
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    assert int(stat[19]) == child['start_time'] and stat[0] not in {'Z', 'X'}
    assert hashlib.sha256((proc / 'cmdline').read_bytes()).hexdigest() == child['command_sha256']
    descriptor = os.pidfd_open(child['pid'])
    try:
        with pytest.raises(RuntimeError, match='owner is still live'):
            with reference.owned_run(tmp_path / 'reference', True):
                raise AssertionError('An orphan reader must retain the original lock')
    finally:
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        poll = select.poll()
        poll.register(descriptor, select.POLLIN)
        assert poll.poll(5000)
        os.close(descriptor)
    with reference.owned_run(tmp_path / 'reference', True):
        pass


def test_caught_parent_interruption_waits_for_the_actual_owned_child_to_exit(monkeypatch):
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    original_wait, interrupted = child.wait, False

    def interrupted_wait(*a, **kw):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt('controlled parent interruption')
        return original_wait(*a, **kw)

    monkeypatch.setattr(child, 'wait', interrupted_wait)
    try:
        with pytest.raises(KeyboardInterrupt, match='controlled parent'):
            reference.wait_readback(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            original_wait(timeout=10)
