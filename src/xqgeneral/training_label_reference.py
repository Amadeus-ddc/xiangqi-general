"""Generate a train-only retained-engine reference and run its independent reader."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .evidence import atomic_json, code_identity, digest
from .evaluate_explanations import explanation_summary, judgment
from .explanations import parse_explanation
from .reference_oracle import RECOVERY_POLICY, ReferencePikafish
from .reviewed_explanations import (REFERENCE_CORRECTION_KIND, SOURCE_KINDS,
                                    checked_label, source_contract)


KIND = 'existing_training_teacher_labels_deep_engine_reference'
READER = '''"""Independently read the retained engine reference; do not run searches."""
from xqgeneral.training_label_reference_readback import main

if __name__ == '__main__':
    main()
'''


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    path = Path(path)
    return dict(sha256=digest(path), bytes=path.stat().st_size)


def source_hashes():
    return {path.name: digest(path) for path in sorted(Path(__file__).parent.glob('*.py'))}


def write_once(path, value):
    """Publish one complete artifact without replacing a previous attempt's bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def check_bindings(bindings):
    for name, expected in bindings.items():
        if bound(name) != expected:
            raise ValueError(f'Reference input or output bytes changed: {name}')


def load_training(data, artifact_root):
    """Use the accepted manifest and exact training bytes; never open held-out files."""
    data, artifact_root = Path(data).resolve(), Path(artifact_root).resolve()
    proof = read(data / 'manifest.json')
    counts = proof.get('verification', {}).get('by_split', {})
    if (proof.get('status') != 'complete' or
            proof.get('kind') not in SOURCE_KINDS | {REFERENCE_CORRECTION_KIND} or
            set(counts) != {'train', 'validation', 'test'} or
            any(type(n) is not int or n < 0 for n in counts.values()) or not counts['train']):
        raise ValueError('A completed original reviewed dataset with declared splits is required')
    source_contract(proof, counts)
    path = data / 'train.jsonl'
    keys = [str(path)]
    if path.is_relative_to(artifact_root):
        keys.append(str(path.relative_to(artifact_root)))
    matches = [proof.get('outputs', {})[key] for key in keys if key in proof.get('outputs', {})]
    if len(matches) != 1 or matches[0] != bound(path):
        raise ValueError('Training bytes differ from their unique accepted source binding')
    raw = path.read_bytes()
    if not raw.endswith(b'\n') or any(not line.strip() for line in raw.splitlines()):
        raise ValueError('Training files require complete nonempty JSONL records')
    rows = [json.loads(line) for line in raw.splitlines()]
    if (len(rows) != counts['train'] or any(not isinstance(row, dict) or
            row.get('split') != 'train' or row.get('stage') != 'explanation' or
            any(not isinstance(row.get(k), str) or not row[k] for k in ['id', 'feature_key'])
            for row in rows) or len({row['id'] for row in rows}) != len(rows) or
            len({row['feature_key'] for row in rows}) != len(rows)):
        raise ValueError('Original training identities, counts or split membership differ')
    return rows


def validate_plan(reference):
    reference = Path(reference).resolve()
    plan = read(reference / 'plan.json')
    if (plan.get('kind') != 'native_training_label_reference_plan' or
            plan.get('source_sha256') != source_hashes() or
            plan.get('reader_launcher_sha256') != digest(reference / 'readback.py')):
        raise ValueError('The pinned reference source or reader launcher changed')
    config = plan['config']
    expected = {str(Path(config['data']) / name) for name in ['manifest.json', 'train.jsonl']}
    expected.update([config['executable'], config['weights']])
    if (set(plan['inputs']) != expected or config['output'] != str(reference) or
            config['scope'] not in {'full', 'capability'} or
            any(type(config[key]) is not int or config[key] < 1 for key in ['nodes', 'workers', 'threads']) or
            config['indices'] != sorted(set(config['indices'])) or not config['indices'] or
            any(type(index) is not int or not 0 <= index < config['source_training_rows']
                for index in config['indices']) or
            (config['scope'] == 'full' and config['indices'] != list(range(config['source_training_rows'])))):
        raise ValueError('The pinned train-only reference coverage or input schema differs')
    check_bindings(plan['inputs'])
    return plan


def row_binding(reference, plan):
    config = plan['config']
    return dict(plan_sha256=digest(Path(reference) / 'plan.json'),
                nodes_per_position=config['nodes'], engine_threads=config['threads'],
                executable_sha256=plan['inputs'][config['executable']]['sha256'],
                weights_sha256=plan['inputs'][config['weights']]['sha256'])


@contextmanager
def owned_run(output, resume):
    if output.exists() and not resume:
        raise FileExistsError('Preserve existing references; use a fresh output or explicit --resume')
    if resume and not (output / 'plan.json').is_file():
        raise ValueError('Resume requires an existing pinned reference plan')
    output.mkdir(parents=True, exist_ok=resume)
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('The original reference owner is still live') from error
        # Closing our descriptor must not unlock an inherited descriptor while
        # the independent reader is still alive after an owner interruption.
        yield lock.fileno()


def wait_readback(child):
    """Finish the owned reader before unwinding a caught parent interruption."""
    try:
        return child.wait()
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        raise


def complete_pv(value):
    verdict = value['judgment']
    pv = parse_explanation(value['raw'])['pv']
    losses = verdict.get('pv_move_losses', [])
    return bool(verdict.get('pv_legal') and pv and len(pv) == len(losses) and all(v < .10 for v in losses))


def produce(reference, plan, rows, attempt):
    config = plan['config']
    destination = reference / config['scope']
    indices = config['indices']
    actual_config = dict(config, plan_sha256=digest(reference / 'plan.json'))
    manifest_path = destination / 'manifest.json'
    if manifest_path.is_file():
        previous = read(manifest_path)
        if previous.get('status') != 'complete' or previous.get('config') != actual_config:
            raise ValueError('The completed producer belongs to another pinned reference')
        check_bindings(previous['inputs'])
        check_bindings(previous['outputs'])
        if not (destination / 'exit.json').exists():
            write_once(destination / 'exit.json', dict(returncode=0, manifest_sha256=digest(manifest_path),
                       completed_producer_artifacts_verified_on_resume=True))
        exit_proof = read(destination / 'exit.json')
        if exit_proof.get('returncode') != 0 or exit_proof.get('manifest_sha256') != digest(manifest_path):
            raise ValueError('The completed producer exit receipt differs')
        return previous['metrics']
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / 'config.json').exists():
        write_once(destination / 'config.json', actual_config)
    elif read(destination / 'config.json') != actual_config:
        raise ValueError('The unfinished producer configuration changed')
    binding = row_binding(reference, plan)
    done = {}
    from .training_label_reference_readback import verify_row
    for path in sorted((destination / 'rows').glob('*.json')):
        index = int(path.stem)
        if index not in indices or path.name != f'{index:06}.json':
            raise ValueError('The unfinished reference contains an unexpected training row')
        value = read(path)
        verify_row(value, rows[index], index, binding)
        done[index] = value
    lock, stop = threading.Lock(), threading.Event()
    started = time.monotonic()

    def state(status):
        atomic_json(reference / 'state.json', dict(status=status, completed=len(done),
            requested=len(indices), scope=config['scope'], owner_pid=os.getpid(),
            observed_at_utc=datetime.now(timezone.utc).isoformat(),
            teacher_labels_changed=False, model_or_gpu_started=False,
            validation_or_independent_test_records_parsed=False))

    def shard(rank):
        pending = [index for index in indices[rank::config['workers']] if index not in done]
        if not pending:
            return
        oracle, index = None, pending[0]
        try:
            oracle = ReferencePikafish(config['executable'], config['weights'], threads=config['threads'])
            for index in pending:
                if stop.is_set():
                    break
                oracle.begin_row(index)
                row = rows[index]
                checked_label(row)
                verdict = judgment(oracle, row, row['answer'], config['nodes'])
                value = dict(row_index=index, id=row['id'], record=row, raw=row['answer'],
                    judgment=verdict, audit_binding=binding,
                    engine_requests=oracle.requests,
                    oracle_exact_score_recoveries=oracle.recoveries)
                write_once(destination / 'rows' / f'{index:06}.json', value)
                with lock:
                    done[index] = value
                    state('judging_original_training_labels')
        except Exception as error:
            stop.set()
            write_once(attempt / f'failed-row-{index:06}.json', dict(row_index=index,
                error_type=type(error).__name__, message=str(error),
                last_engine_request=getattr(oracle, 'last_request', None),
                exact_score_recoveries=getattr(oracle, 'recoveries', [])))
            raise
        finally:
            if oracle is not None:
                oracle.close()

    state('judging_original_training_labels')
    pool = ThreadPoolExecutor(max_workers=config['workers'])
    try:
        futures = [pool.submit(shard, rank) for rank in range(config['workers'])]
        for future in futures:
            future.result()
    except BaseException:
        stop.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if set(done) != set(indices):
        raise RuntimeError('The reference producer did not finish its declared training coverage')
    results = [done[index] for index in indices]
    groups = {}
    for name, augmented in [('with_augmentation_parent', True), ('without_augmentation_parent', False)]:
        selected = [value for value in results if bool(value['record'].get('augmentation_parent')) == augmented]
        groups[name] = dict(examples=len(selected),
            first_move_no_mistake=sum(value['judgment']['first_move_no_mistake'] for value in selected),
            full_pv_no_mistake=sum(complete_pv(value) for value in selected))
    metrics = dict(explanation_summary(results), raw_generation=False,
        source='existing_teacher_training_labels', split='train', scope=config['scope'],
        full_pv_no_mistake_count=sum(complete_pv(value) for value in results),
        full_pv_no_mistake_rate=sum(complete_pv(value) for value in results) / len(results),
        nodes_per_engine_request=config['nodes'], engine_threads=config['threads'],
        additional_best_score_requests=sum(len(value['oracle_exact_score_recoveries']) for value in results),
        exact_score_recovery_policy=RECOVERY_POLICY, source_training_rows=len(rows),
        all_source_training_labels_scored=indices == list(range(len(rows))),
        by_color_derivation=groups, teacher_labels_changed=False,
        student_model_loaded_or_evaluated=False, gpu_used=False,
        validation_answers_loaded=False, independent_test_answers_loaded=False,
        seconds=time.monotonic() - started, independent_readback_complete=False,
        student_strength_or_coaching_benefit_proven=False)
    validate_plan(reference)
    metrics_path = destination / 'metrics.json'
    if metrics_path.exists():
        previous = read(metrics_path)
        if (set(previous) != set(metrics) or
                any(previous[key] != value for key, value in metrics.items() if key != 'seconds') or
                type(previous.get('seconds')) not in {int, float} or
                not math.isfinite(previous['seconds']) or previous['seconds'] < 0):
            raise ValueError('Interrupted producer finalization has different retained metrics')
        metrics = previous
    else:
        write_once(metrics_path, metrics)
    paths = [destination / 'rows' / f'{index:06}.json' for index in indices] + [destination / 'metrics.json']
    proof = dict(status='complete', kind=KIND, evidence_state='reconstructed_baseline',
        config=actual_config, inputs=plan['inputs'], outputs={str(path): bound(path) for path in paths},
        metrics=metrics, code=dict(source_sha256=plan['source_sha256']), oracle_repairs=0,
        loss_cutoff='expected_score_loss_strictly_less_than_0.10', independent_readback_complete=False)
    write_once(manifest_path, proof)
    write_once(destination / 'exit.json', dict(returncode=0, manifest_sha256=digest(manifest_path)))
    state('complete_producer_pending_independent_readback')
    return metrics


def run_reference(data, output, executable, weights, *, nodes=1000000, workers=2, threads=1,
                  scope='full', indices=None, artifact_root=None, resume=False):
    """Run/resume immutable engine scoring, then a separate process with no searches."""
    if any(type(value) is not int or value < 1 for value in [nodes, workers, threads]):
        raise ValueError('Node, worker and engine thread budgets must be positive integers')
    data, output, executable, weights = [Path(value).resolve() for value in [data, output, executable, weights]]
    artifact_root = Path(artifact_root or Path.cwd()).resolve()
    rows = load_training(data, artifact_root)
    if scope not in {'full', 'capability'}:
        raise ValueError('Reference scope must be full or capability')
    if scope == 'full':
        if indices is not None:
            raise ValueError('Full references must score every original training row')
        indices = list(range(len(rows)))
    elif (not isinstance(indices, list) or not indices or indices != sorted(set(indices)) or
          any(type(index) is not int or not 0 <= index < len(rows) for index in indices)):
        raise ValueError('Capability scope requires sorted distinct original training indices')
    config = dict(data=str(data), output=str(output), executable=str(executable), weights=str(weights),
        artifact_root=str(artifact_root), nodes=nodes, workers=workers, threads=threads,
        scope=scope, indices=indices, source_training_rows=len(rows))
    inputs = {str(path): bound(path) for path in [data / 'manifest.json', data / 'train.jsonl', executable, weights]}
    with owned_run(output, resume) as lock_fd:
        if resume:
            plan = validate_plan(output)
            if plan['config'] != config or plan['inputs'] != inputs:
                raise ValueError('Resume requires the exact original inputs and search configuration')
        else:
            with (output / 'readback.py').open('x') as stream:
                stream.write(READER)
            plan = dict(kind='native_training_label_reference_plan', config=config, inputs=inputs,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=source_hashes(), code_revision=code_identity()['revision'],
                reader_launcher_sha256=digest(output / 'readback.py'), recovery_policy=RECOVERY_POLICY,
                teacher_labels_will_be_rewritten=False, heldout_files_will_be_opened=False)
            write_once(output / 'plan.json', plan)
        from .training_label_reference_readback import completed_readback
        if (output / f'{scope}-pipeline-exit.json').exists():
            completed_readback(output, plan)
            pipeline = read(output / f'{scope}-pipeline-exit.json')
            reader_exit = read(output / f'{scope}-readback-exit.json')
            readback_sha = digest(output / scope / 'independent-readback.json')
            if (pipeline.get('returncode') != 0 or pipeline.get('independent_readback_complete') is not True or
                    pipeline.get('stages') != [dict(stage='producer', returncode=0), dict(stage='readback', returncode=0)] or
                    reader_exit.get('returncode') != 0 or reader_exit.get('verification_sha256') != readback_sha or
                    pipeline.get('readback_verification_sha256') != readback_sha):
                raise ValueError('The completed reference pipeline receipt differs')
            return read(output / scope / 'metrics.json')
        number = 1
        while (output / 'attempts' / f'{number:06}').exists():
            number += 1
        attempt = output / 'attempts' / f'{number:06}'
        attempt.mkdir(parents=True)
        write_once(attempt / 'execution.json', dict(owner_pid=os.getpid(), config=config,
            observed_at_utc=datetime.now(timezone.utc).isoformat()))
        try:
            metrics = produce(output, plan, rows, attempt)
            write_once(attempt / 'producer-exit.json', dict(returncode=0))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            package_root = str(Path(__file__).resolve().parent.parent)
            env['PYTHONPATH'] = package_root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            command = [sys.executable, '-P', '-u', str(output / 'readback.py'), '--reference', str(output)]
            with (attempt / 'readback.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(lock_fd,))
                write_once(attempt / 'readback-worker.json', dict(pid=child.pid, command=command))
                code = wait_readback(child)
            receipt = dict(returncode=code, observed_at_utc=datetime.now(timezone.utc).isoformat(),
                           actual_readback_pid=child.pid)
            if code == 0:
                receipt['verification_sha256'] = digest(output / scope / 'independent-readback.json')
            write_once(attempt / 'readback-exit.json', receipt)
            if code:
                raise RuntimeError(f'Independent reference readback failed; inspect {attempt / "readback.log"}')
            completed_readback(output, plan)
            exit_path = output / f'{scope}-readback-exit.json'
            if not exit_path.exists():
                write_once(exit_path, receipt)
            elif (read(exit_path).get('returncode') != 0 or
                  read(exit_path).get('verification_sha256') != receipt['verification_sha256']):
                raise ValueError('The prior completed readback exit receipt differs')
            write_once(output / f'{scope}-pipeline-exit.json', dict(returncode=0,
                stages=[dict(stage='producer', returncode=0), dict(stage='readback', returncode=0)],
                readback_verification_sha256=receipt['verification_sha256'],
                independent_readback_complete=True, model_or_gpu_started=False,
                validation_or_independent_test_answers_read=False))
            atomic_json(output / 'state.json', dict(status='complete_reference_and_independent_readback',
                scope=scope, completed=len(indices), requested=len(indices),
                model_or_gpu_started=False, teacher_labels_changed=False))
            return metrics
        except BaseException as error:
            write_once(attempt / 'failure.json', dict(error_type=type(error).__name__, message=str(error)))
            atomic_json(output / 'state.json', dict(status='failed_attempt_requires_inspection',
                scope=scope, attempt=number, model_or_gpu_started=False, teacher_labels_changed=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--executable', default='vendor/pikafish/src/pikafish')
    parser.add_argument('--weights', default='vendor/pikafish/src/pikafish.nnue')
    parser.add_argument('--nodes', type=int, default=1000000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--scope', choices=['full', 'capability'], default='full')
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--artifact-root')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.nice(10)
    metrics = run_reference(**vars(args))
    print(json.dumps(dict(status='complete_reference_and_independent_readback',
        scope=args.scope, examples=metrics['examples'], model_or_gpu_started=False)))


if __name__ == '__main__':
    main()
