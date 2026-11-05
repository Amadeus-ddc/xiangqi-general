"""Wait for qualified producers, then run the existing native-gated distillation."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from xqgeneral.distillation_training import compile_recipe, run_round
from xqgeneral.evidence import atomic_json, code_identity
from xqgeneral.sft import prepare_config
from xqgeneral.verified_inputs import verified_input_session


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().split(') ')[1].split()
    if fields[0] in {'Z', 'X'}:
        raise RuntimeError('Owned producer has terminated without completion evidence')
    return {'pid': pid, 'start_time': int(fields[19]),
            'command_sha256': sha(proc / 'cmdline')}


def completed_dependencies(dependencies):
    """Check every failure immediately; a pending state alone is never liveness."""
    ready, pending = {}, []
    for role, item in dependencies.items():
        root = Path(item['run'])
        exit_path = root / item.get('exit', 'exit.json')
        if not exit_path.exists():
            expected = read(root / item.get('owner', 'owner.json'))['identity']
            try:
                actual = process_identity(expected['pid'])
                if any(actual[key] != expected[key] for key in actual):
                    raise RuntimeError('Owned producer identity changed: ' + role)
            except (FileNotFoundError, RuntimeError):
                if not exit_path.exists():
                    raise
            if not exit_path.exists():
                pending.append(role)
                continue
        end = read(exit_path)
        proof_path = root / item.get('proof', 'verification.json')
        if end.get('returncode') != 0 or end.get('verification_sha256') != sha(proof_path):
            raise ValueError('Producer failed or completion proof changed: ' + role)
        proof = read(proof_path)
        if proof.get('status') != item['status']:
            raise ValueError('Producer completion status differs: ' + role)
        ready[role] = proof
    return ready, pending


def verify_pins(plan, plan_path, plan_sha256):
    if sha(plan_path) != plan_sha256 or sha(__file__) != plan['entry_sha256']:
        raise ValueError('Queued plan or entry changed')
    source = Path(plan['source']) / 'xqgeneral'
    if ({p.name: sha(p) for p in source.glob('*.py')} != plan['source_sha256'] or
            code_identity() != {'revision': plan['source_revision'],
                'source_sha256': {'src/xqgeneral/' + key: value for key, value in plan['source_sha256'].items()}}):
        raise ValueError('Actual training source differs from the frozen plan')
    if any(sha(path) != expected for path, expected in plan['fixed_sha256'].items()):
        raise ValueError('Queued fixed prerequisite changed')


def prepared_recipe(plan, ready):
    """Compile actual data before acquiring GPUs; never substitute another parent."""
    checks = ready['cpu_validation']
    if (checks.get('tests_passed') != plan['expected_CPU_tests'] or
            checks.get('source_sha256') != plan['source_sha256'] or
            checks.get('source_revision') != plan['source_revision'] or
            checks.get('checked_files', {}).get('scripts/queued_distillation.py') != plan['entry_sha256']):
        raise ValueError('Require matching complete CPU validation')
    prepared = ready['data_preparation']
    count = prepared.get('accepted_merged_training_histories')
    if type(count) is not int or count < plan['minimum_accepted_training_rows']:
        raise ValueError('Expanded collection needs more accepted training histories')
    if prepared.get('actual_search_and_selected_parent') != plan['parent']:
        raise ValueError('Preparation belongs to a different search parent')
    token_path = Path(plan['token_preflight'])
    token = read(token_path)
    if (sha(token_path) != prepared['token_preflight_manifest_sha256'] or
            sha(plan['feature_manifest']) != prepared['feature_manifest_sha256'] or
            token['config']['source'] != plan['parent'] or
            token['config']['parent_completion'] != plan['parent_completion'] or
            token['cache_header_only_input']['path'] != plan['feature_manifest'] or
            token['verification']['by_split'] != {'train': count, 'validation': 384, 'test': 512} or
            token['verification']['independent_test_rows_tokenized'] != 0):
        raise ValueError('Require the exact completed expanded-data token contract')
    recipe, budget = compile_recipe(read(plan['recipe_template']), token, Path(plan['round_output']))
    final = prepare_config(read(Path(plan['parent']).parent / 'config.json'), recipe,
                           Path(plan['parent']), Path(plan['round_output']) / 'sft')
    if (final['steps'] != recipe['max_steps'] or
            any(final.get(key) != value for key, value in plan['expected_memory_controls'].items())):
        raise ValueError('Explicit full-model memory settings were lost')
    return recipe, budget, final


def redacted_training_run(original, command, **kwargs):
    target = kwargs.get('stdout')
    own = any(isinstance(arg, str) and (arg.startswith('xqgeneral.') or arg == 'torch.distributed.run')
              for arg in command)
    if not own or not hasattr(target, 'write') or kwargs.get('stderr') != subprocess.STDOUT:
        return original(command, **kwargs)
    options = dict(kwargs)
    check = options.pop('check', False)
    options.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    child = subprocess.Popen(command, **options)
    for line in child.stdout:
        line = re.sub(r'https?://[^\s)\]]+', '[REDACTED_URL]', line)
        target.write(re.sub(r'(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?', '[REDACTED_ADDRESS]', line))
        target.flush()
    result = child.wait()
    if check and result:
        raise subprocess.CalledProcessError(result, command)
    return subprocess.CompletedProcess(command, result)


def run(plan_path, plan_sha256, output, *, resume_round=False, poll_seconds=30):
    plan_path, output = Path(plan_path), Path(output)
    plan = read(plan_path)
    if (type(plan['minimum_accepted_training_rows']) is not int or plan['minimum_accepted_training_rows'] < 32 or
            type(plan['minimum_free_MiB_each_gpu']) is not int or plan['minimum_free_MiB_each_gpu'] < 1):
        raise ValueError('Positive resource and real-batch data thresholds are required')
    root = Path(plan['project_root']).resolve()
    if (Path.cwd() != root or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3' or
            not output.resolve().is_relative_to(root / 'runs') or
            not Path(plan['round_output']).resolve().is_relative_to(root / 'runs')):
        raise ValueError('Use the declared project, four GPUs and isolated run outputs')
    verify_pins(plan, plan_path, plan_sha256)
    output.mkdir(parents=True, exist_ok=False)

    def state(status, **fields):
        atomic_json(output / 'state.json', {'status': status, 'utc': datetime.now(timezone.utc).isoformat(),
                    'full_goal_complete': False, **fields})

    original_run = subprocess.run
    previous_receipt = os.environ.get('XQGENERAL_VERIFIED_INPUT_RECEIPT')
    try:
        atomic_json(output / 'owner.json', {'identity': process_identity(os.getpid())})
        atomic_json(output / 'contract.json', {'plan': str(plan_path), 'plan_sha256': plan_sha256,
            'code': code_identity(), 'entry_sha256': sha(__file__), 'resume_round': resume_round})
        while True:
            verify_pins(plan, plan_path, plan_sha256)
            ready, pending = completed_dependencies(plan['dependencies'])
            if not pending:
                break
            state('waiting_for_exact_qualified_producers', pending=pending, model_loaded=False)
            time.sleep(poll_seconds)
        recipe, budget, final = prepared_recipe(plan, ready)
        bound = {str(plan_path): plan_sha256}
        for item in plan['dependencies'].values():
            for name in [item.get('exit', 'exit.json'), item.get('proof', 'verification.json')]:
                path = str(Path(item['run']) / name)
                bound[path] = sha(path)
        for path in [plan['token_preflight'], plan['feature_manifest']]:
            bound[path] = sha(path)
        for name, value in [('compiled-recipe-control.json', recipe), ('compiled-budget.json', budget),
                            ('prepared-training-config.json', final)]:
            atomic_json(output / name, value)
        with Path(plan['teacher_lock_path']).open('a+') as lease:
            while True:
                verify_pins(plan, plan_path, plan_sha256)
                if any(sha(path) != expected for path, expected in bound.items()):
                    raise ValueError('Completed producer or prepared inputs changed while queued')
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    state('waiting_for_project_training_resource_lease', model_loaded=False)
                    time.sleep(poll_seconds)
            while True:
                free = [int(value) for value in original_run(['nvidia-smi', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout.splitlines()]
                if len(free) != 4:
                    raise ValueError('Require four actual device memory readings')
                if min(free) >= plan['minimum_free_MiB_each_gpu']:
                    break
                state('waiting_for_full_four_GPU_headroom', free_MiB=free, model_loaded=False)
                time.sleep(poll_seconds)
                verify_pins(plan, plan_path, plan_sha256)
            if any(sha(path) != expected for path, expected in bound.items()):
                raise ValueError('Prepared inputs changed before the native preflight')
            os.environ['XQGENERAL_VERIFIED_INPUT_RECEIPT'] = plan['receipt']
            with verified_input_session(plan['receipt']) as session:
                state('actual_native_preflight_then_search_distillation', compiled_max_updates=budget['max_updates'])
                subprocess.run = lambda command, **kwargs: redacted_training_run(original_run, command, **kwargs)
                try:
                    result = run_round(plan['parent'], plan['recipe_template'], plan['token_preflight'],
                                       plan['round_output'], resume=resume_round)
                finally:
                    subprocess.run = original_run
                verify_pins(plan, plan_path, plan_sha256)
                if any(sha(path) != expected for path, expected in bound.items()):
                    raise ValueError('Qualified producer or prepared inputs changed during the round')
                metadata = session.metadata()
            fcntl.flock(lease, fcntl.LOCK_UN)
        if result.get('actual_distillation_sft_completed') is not True or result.get('status') != 'complete':
            raise ValueError('The native-gated round did not prove actual SGD completion')
        proof = {'status': 'complete_qualified_expanded_distillation_development_round',
            'dependencies_sha256': bound, 'budget': budget, 'round': result,
            'input_digest_verification': metadata, 'new_joint_model_substituted_as_search_parent': False,
            'coaching_or_strength_improvement_proven': False, 'independent_test_used': False,
            'full_goal_complete': False}
        atomic_json(output / 'verification.json', proof)
        state(proof['status'])
        atomic_json(output / 'exit.json', {'returncode': 0, 'verification_sha256': sha(output / 'verification.json')})
        return proof
    except BaseException as error:
        state('failed', error_type=type(error).__name__, preserve_partial_outputs=True)
        atomic_json(output / 'exit.json', {'returncode': 1, 'error_type': type(error).__name__})
        raise
    finally:
        subprocess.run = original_run
        if previous_receipt is None:
            os.environ.pop('XQGENERAL_VERIFIED_INPUT_RECEIPT', None)
        else:
            os.environ['XQGENERAL_VERIFIED_INPUT_RECEIPT'] = previous_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'plan-sha256', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--resume-round', action='store_true', help='Use a fresh supervisor directory and the same round contract')
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.plan_sha256, args.output, resume_round=args.resume_round)), flush=True)


if __name__ == '__main__':
    main()
