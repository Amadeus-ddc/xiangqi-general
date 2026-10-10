"""Compare BF16 teacher batches on real clean-model mining queries."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from .clean_sft_evaluation import teacher_artifacts
from .evidence import atomic_json, code_identity, digest, file_signature, load_jsonl, manifest, write_jsonl
from .local_teacher import LocalConsolidator
from .recorded_search_inputs import BoundInputs


KIND = 'clean_search_teacher_batch_benchmark'


def benchmark_spec(config):
    for key in ['sample_limit', 'max_new_tokens', 'warmup_new_tokens']:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError('Benchmark token and sample budgets must be positive integers')
    sizes = config['batch_sizes']
    if (not isinstance(sizes, list) or not sizes or sizes[0] != 1 or
            any(type(size) is not int or size <= 0 for size in sizes) or
            sizes != sorted(set(sizes)) or max(sizes) > config['sample_limit'] or
            config['warmup_new_tokens'] > config['max_new_tokens'] or
            not isinstance(config['cuda_visible_devices'], str) or
            not config['cuda_visible_devices'].isdigit() or
            type(config['minimum_free_gib']) not in (int, float) or
            not math.isfinite(config['minimum_free_gib']) or config['minimum_free_gib'] <= 0):
        raise ValueError('Benchmark requires serial-first distinct batch sizes and one declared GPU')


def wait_for_mining(pilot, producer_session, root, poll_seconds):
    if not 0 < poll_seconds <= 60:
        raise ValueError('Use a positive polling interval of at most 60 seconds')
    path = Path(pilot) / 'mining/manifest.json'
    while not path.exists():
        if subprocess.run(['tmux', 'has-session', '-t', '=' + producer_session], capture_output=True).returncode:
            if path.exists():
                break
            raise RuntimeError('The owned clean search stopped before its completed mining queries')
        atomic_json(root / 'state.json', {'status': 'waiting_for_completed_clean_search_mining',
            'pilot': str(pilot), 'producer_session': producer_session,
            'teacher_loaded': False, 'new_training_labels': 0})
        time.sleep(poll_seconds)
    return path


def admitted_queries(pilot, pilot_contract, pilot_config, evaluation, bound):
    """Read actual miner outputs and recorded parent lineage, without rehashing its feature cache."""
    pilot = Path(pilot)
    capability = bound.json(Path(pilot_contract['validation']) / 'manifest.json')
    result = capability['verification']
    if (capability.get('status') != 'complete' or
            capability.get('kind') != 'clean_initial_sft_capability_validation' or
            capability['config']['pipeline'] != pilot_contract['pipeline'] or
            capability['config']['config'] != pilot_config['evaluation_config'] or
            capability['config']['config_sha256'] != bound.artifacts[pilot_config['evaluation_config']]['sha256'] or
            result.get('actual_clean_initial_sft_completed_before_model_or_reviewer_load') is not True or
            result.get('split') != 'validation' or
            result.get('all_validation_examples') != evaluation['validation_examples'] or
            result.get('independent_test_used_for_training_or_selection') is not False or
            result.get('raw_answers_repaired') is not False or
            set(result.get('raw_validation_by_memory', {})) != {'normal', 'zero', 'shuffled'}):
        raise ValueError('Actual clean SFT and its complete paired validation must precede a benchmark')
    pipeline = bound.json(Path(pilot_contract['pipeline']) / 'manifest.json')
    source = str(Path(pilot_contract['pipeline']) / 'sft/adapter.pt')
    parent = pipeline['verification']
    if (pipeline.get('status') != 'complete' or
            pipeline.get('kind') != 'clean_curriculum_to_initial_sft_pipeline' or
            parent.get('four_courses_completed_before_any_gpu_sft_work') is not True or
            parent.get('initial_explanation_sft_completed') is not True or
            parent.get('independent_test_used_for_training_or_selection') is not False or
            parent.get('selected_checkpoint') != source or
            pipeline['outputs'].get(source) != capability['inputs'].get(source)):
        raise ValueError('Benchmark mining must descend from the exact completed clean SFT parent')
    mining = bound.json(pilot / 'mining/manifest.json')
    if (mining.get('status') != 'complete' or mining.get('kind') != 'search_distillation_mining' or
            mining['code'] != pilot_contract['code'] or mining['config']['checkpoint'] != source or
            mining['inputs'].get(source) != pipeline['outputs'][source] or
            any(mining['config'].get(key) != pilot_config[key]
                for key in ['data', 'recorded_inputs', 'limit', 'nodes', 'max_depth', 'child_contract', 'seed'])):
        raise ValueError('Completed mining source, parent or search budgets differ from the queued pilot')
    path = pilot / 'mining/queries.jsonl'
    bound.bind(path, mining['outputs'][str(path)])
    attempts_path = pilot / 'mining/results.partial.jsonl'
    bound.bind(attempts_path)
    queries, attempts = load_jsonl(path), load_jsonl(attempts_path)
    verified = mining['verification']
    if (len(attempts) != pilot_config['limit'] or
            len({row['source_root_id'] for row in attempts}) != len(attempts) or
            [row['query'] for row in attempts if row['query'] is not None] != queries or
            verified.get('processed') != len(attempts) or
            verified.get('accepted_for_consolidation') != len(queries) or
            verified.get('reasons') != dict(Counter(row['reason'] for row in attempts)) or
            any(verified.get(key) is not True for key in [
                'unused_recorded_training_inputs', 'training_roots_only', 'heldout_positions_excluded',
                'all_child_branch_positions_isolated', 'raw_child_reachable_prefixes_isolated',
                'all_inferred_target_lines_history_validated', 'strict_pv_improvement_required',
                'all_successful_oracle_queries_preserved'])):
        raise ValueError('Actual training-query coverage, yield or heldout isolation changed')
    if (len({row['id'] for row in queries}) != len(queries) or
            any(not isinstance(row['id'], str) or not row['id'] or
                not isinstance(row['messages'], list) or not row['messages'] or
                any(not isinstance(message, dict) or message.get('role') not in {'system', 'user', 'assistant'} or
                    not isinstance(message.get('content'), str) for message in row['messages']) for row in queries)):
        raise ValueError('Real benchmark query IDs and text prompts must be distinct and complete')
    return queries


def selected_queries(queries, limit):
    ordered = sorted(queries, key=lambda row: (len(json.dumps(row['messages'], ensure_ascii=False)), row['id']))
    if len(ordered) <= limit:
        return ordered
    indices = [round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [len(ordered) - 1]
    return [ordered[i] for i in indices]


def group_plan(rows, sizes):
    return [(size, rows[start:start + size]) for size in sizes for start in range(0, len(rows), size)]


def checked_response(response, budget):
    if (not isinstance(response, dict) or set(response) != {
            'text', 'prompt_tokens', 'generated_tokens', 'hit_generation_limit'} or
            not isinstance(response['text'], str) or type(response['prompt_tokens']) is not int or
            response['prompt_tokens'] <= 0 or type(response['generated_tokens']) is not int or
            not 0 < response['generated_tokens'] <= budget or
            response['hit_generation_limit'] is not (response['generated_tokens'] >= budget)):
        raise ValueError('Each raw teacher response must retain its bounded actual token counts')


def checked_groups(groups, plan, budget):
    if len(groups) > len(plan):
        raise ValueError('Benchmark continuation has foreign groups')
    for item, (size, rows) in zip(groups, plan):
        if (item['batch_size'] != size or item['ids'] != [row['id'] for row in rows] or
                len(item['responses']) != len(rows) or type(item['seconds']) not in (int, float) or
                not math.isfinite(item['seconds']) or item['seconds'] <= 0 or
                type(item['peak_allocated_gib']) not in (int, float) or
                not math.isfinite(item['peak_allocated_gib']) or item['peak_allocated_gib'] < 0):
            raise ValueError('Benchmark continuation differs from the declared ordered query groups')
        for response in item['responses']:
            checked_response(response, budget)


def comparison_metrics(groups, rows, sizes):
    by_size, comparisons = {}, []
    for size in sizes:
        items = [item for item in groups if item['batch_size'] == size]
        responses = {identifier: response for item in items
                     for identifier, response in zip(item['ids'], item['responses'], strict=True)}
        seconds = sum(item['seconds'] for item in items)
        tokens = sum(response['generated_tokens'] for response in responses.values())
        by_size[str(size)] = {'queries': len(responses), 'generation_calls': len(items),
            'seconds': seconds, 'generated_tokens': tokens,
            'queries_per_second': len(responses) / seconds if seconds else None,
            'generated_tokens_per_second': tokens / seconds if seconds else None,
            'actual_largest_group': max((len(item['ids']) for item in items), default=0),
            'peak_allocated_gib': max((item['peak_allocated_gib'] for item in items), default=0),
            'generation_limits_hit': sum(response['hit_generation_limit'] for response in responses.values())}
        if size == 1:
            serial = responses
        else:
            for row in rows:
                identifier = row['id'];current, reference = responses[identifier], serial[identifier]
                if current['prompt_tokens'] != reference['prompt_tokens']:
                    raise ValueError('Padded and serial prompt token counts differ')
                comparisons.append({'id': identifier, 'batch_size': size,
                    'same_raw_text_as_serial': current['text'] == reference['text'],
                    'serial_generated_tokens': reference['generated_tokens'],
                    'batch_generated_tokens': current['generated_tokens'],
                    'serial_hit_generation_limit': reference['hit_generation_limit'],
                    'batch_hit_generation_limit': current['hit_generation_limit']})
            by_size[str(size)]['raw_text_matches_serial'] = sum(
                item['same_raw_text_as_serial'] for item in comparisons if item['batch_size'] == size)
        by_size[str(size)]['query_throughput_ratio_to_serial'] = (
            by_size['1']['seconds'] / seconds if seconds else None)
    return by_size, comparisons


def wait_for_capacity(config, root, poll_seconds):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config['cuda_visible_devices'] or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Run the benchmark with exactly its declared visible CUDA device')
    while True:
        free, total = torch.cuda.mem_get_info(0)
        if total / 2**30 < config['minimum_free_gib']:
            raise ValueError('Declared benchmark capacity exceeds the GPU total memory')
        if free / 2**30 >= config['minimum_free_gib']:
            return
        atomic_json(root / 'state.json', {'status': 'waiting_for_benchmark_gpu_capacity',
            'teacher_loaded': False, 'required_free_gib': config['minimum_free_gib']})
        time.sleep(poll_seconds)


def benchmark_result(groups, rows, queries, config, identity, loads):
    metrics, comparisons = comparison_metrics(groups, rows, config['batch_sizes'])
    return {'actual_mined_queries': len(queries), 'distinct_queries_measured': len(rows),
        'configured_batch_sizes': config['batch_sizes'], 'max_new_tokens': config['max_new_tokens'],
        'teacher': identity, 'teacher_model_loads_recorded': len(loads),
        'neural_teacher_inference_executed': bool(groups), 'generation_by_batch_size': metrics,
        'generation_only_timing_excludes_load_and_warmup': True,
        'query_sample_spans_sorted_prompt_character_lengths': True,
        'fixed_mode_order_single_pass': True, 'strategic_prose_quality_measured': False,
        'large_upstream_cache_or_student_weights_rehashed': False,
        'new_training_labels': 0, 'trained_distillation_rounds': 0,
        'existing_search_queue_modified': False, 'independent_test_used': False}, comparisons


def run_benchmark(pilot, producer_session, config_path, output, *, resume=False, poll_seconds=30):
    root, config_path = Path(output), Path(config_path)
    bound = BoundInputs();config = bound.json(config_path);benchmark_spec(config)
    pilot_contract = bound.json(Path(pilot) / 'contract.json')
    pilot_config = bound.json(config['pilot_config'])
    evaluation = bound.json(pilot_config['evaluation_config'])
    teacher, guard = teacher_artifacts(config['teacher_config'])
    bound.bind(config['teacher_config'])
    if (pilot_contract['config'] != config['pilot_config'] or
            pilot_contract['config_sha256'] != bound.artifacts[config['pilot_config']]['sha256'] or
            evaluation['teacher_config'] != config['teacher_config'] or
            config['max_new_tokens'] != pilot_config['teacher_max_new_tokens'] or
            guard.expected != pilot_contract['teacher_assets']):
        raise ValueError('Benchmark must use the exact queued search teacher and generation budget')
    for path in [config['pilot_config'], pilot_config['evaluation_config'], config['teacher_config']]:
        if bound.artifacts[path] != pilot_contract['prerequisite_inputs'].get(path):
            raise ValueError('The queued search prerequisite changed')
    signatures = {path: list(file_signature(path)) for path in guard.expected}
    contract = {'pilot': str(pilot), 'producer_session': producer_session,
        'config': str(config_path), 'config_sha256': bound.artifacts[str(config_path)]['sha256'],
        'code': code_identity(), 'prerequisite_inputs': dict(bound.artifacts),
        'teacher_assets': guard.expected, 'teacher_asset_signatures': signatures}
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Resume only the exact benchmark contract; use a fresh output for changes')
    else:
        root.mkdir(parents=True);atomic_json(root / 'contract.json', contract)

    def unchanged():
        bound.unchanged();guard.unchanged()
        if (code_identity() != contract['code'] or
                any(list(file_signature(path)) != sig for path, sig in signatures.items())):
            raise ValueError('Preserved benchmark source or teacher assets changed')

    stage = 'waiting_for_completed_clean_search_mining'
    try:
        wait_for_mining(pilot, producer_session, root, poll_seconds);unchanged()
        queries = admitted_queries(pilot, pilot_contract, pilot_config, evaluation, bound)
        rows = selected_queries(queries, config['sample_limit']);plan = group_plan(rows, config['batch_sizes'])
        completed = root / 'manifest.json'
        old = json.loads(completed.read_text()) if completed.exists() else None
        if old is not None:
            required = {str(root / name) for name in ['selected-queries.jsonl', 'groups.json',
                'loads.json', 'comparisons.jsonl', 'metrics.json']}
            if old.get('verification', {}).get('teacher') is not None:
                required.add(str(root / 'deployment.json'))
            if (old.get('status') != 'complete' or old.get('kind') != KIND or old.get('code') != contract['code'] or
                    old.get('config') != contract or old.get('inputs') != {**bound.artifacts, **guard.expected} or
                    set(old.get('outputs', {})) != required):
                raise ValueError('Completed benchmark differs from its declared inputs or output coverage')
            for path, expected in old['outputs'].items():
                if (not Path(path).is_file() or Path(path).stat().st_size != expected['bytes'] or
                        digest(path) != expected['sha256']):
                    raise ValueError('Completed benchmark raw output changed or is missing')
        selected = root / 'selected-queries.jsonl'
        if selected.exists():
            if load_jsonl(selected) != rows:
                raise ValueError('Preserved benchmark query sample changed')
        else:
            write_jsonl(selected, rows)
        groups_path, loads_path = root / 'groups.json', root / 'loads.json'
        groups = json.loads(groups_path.read_text()) if groups_path.exists() else []
        loads = json.loads(loads_path.read_text()) if loads_path.exists() else []
        checked_groups(groups, plan, config['max_new_tokens'])
        identity_path = root / 'deployment.json'
        identity = json.loads(identity_path.read_text()) if identity_path.exists() else None
        if groups and (not loads or identity is None):
            raise ValueError('Partial comparison groups lack their recorded deployment')
        if old is not None:
            if len(groups) != len(plan):
                raise ValueError('Completed benchmark differs from its comparison coverage')
            expected, comparisons = benchmark_result(groups, rows, queries, config, identity, loads)
            if (old['verification'] != expected or json.loads((root / 'metrics.json').read_text()) != expected or
                    load_jsonl(root / 'comparisons.jsonl') != comparisons):
                raise ValueError('Completed benchmark differs from its recorded raw comparisons')
            unchanged()
            return expected
        if len(groups) < len(plan):
            stage = 'waiting_for_benchmark_gpu_capacity';wait_for_capacity(config, root, poll_seconds)
            unchanged();guard.verify();unchanged()
            stage = 'loading_full_bf16_teacher';atomic_json(root / 'state.json', {'status': stage})
            started = time.monotonic();model = LocalConsolidator(teacher);load_seconds = time.monotonic() - started
            deployed = model.identity
            fields = ['repository', 'revision', 'quantization', 'inference_dtype']
            if (any(deployed.get(key) != teacher[key] for key in fields) or
                    set(deployed['parameter_elements_by_dtype']) != {'torch.bfloat16'} or
                    deployed['parameter_elements_by_dtype']['torch.bfloat16'] <= 0 or
                    deployed.get('visible_cuda_devices') != 1):
                raise ValueError('Benchmark teacher deployment differs from the full BF16 identity')
            if identity is not None and any(deployed.get(key) != identity.get(key)
                    for key in fields + ['parameter_elements_by_dtype', 'weight_manifest_sha256']):
                raise ValueError('Resumed benchmark deployment changed')
            if identity is None:
                identity = deployed;atomic_json(identity_path, identity)
            torch.cuda.synchronize();started = time.monotonic()
            warmup = model.generate(rows[-1]['messages'], config['warmup_new_tokens']);torch.cuda.synchronize()
            checked_response(warmup, config['warmup_new_tokens'])
            loads.append({'identity': deployed, 'load_seconds': load_seconds,
                'warmup_seconds': time.monotonic() - started, 'warmup_id': rows[-1]['id'], 'warmup_response': warmup})
            atomic_json(loads_path, loads)
            for size, group in plan[len(groups):]:
                unchanged();stage = 'measuring_teacher_batch'
                atomic_json(root / 'state.json', {'status': stage, 'batch_size': size,
                    'completed_groups': len(groups), 'requested_groups': len(plan)})
                torch.cuda.reset_peak_memory_stats(0);torch.cuda.synchronize();started = time.monotonic()
                responses = ([model.generate(group[0]['messages'], config['max_new_tokens'])] if size == 1 else
                    model.generate_batch([row['messages'] for row in group], config['max_new_tokens']))
                torch.cuda.synchronize();seconds = time.monotonic() - started
                item = {'batch_size': size, 'ids': [row['id'] for row in group], 'responses': responses,
                    'seconds': seconds, 'peak_allocated_gib': torch.cuda.max_memory_allocated(0) / 2**30}
                checked_groups([item], [(size, group)], config['max_new_tokens'])
                groups.append(item);atomic_json(groups_path, groups)
        else:
            atomic_json(groups_path, groups);atomic_json(loads_path, loads)
        unchanged();result, comparisons = benchmark_result(groups, rows, queries, config, identity, loads)
        compared, scored = root / 'comparisons.jsonl', root / 'metrics.json'
        write_jsonl(compared, comparisons);atomic_json(scored, result)
        outputs = [selected, groups_path, loads_path, compared, scored]
        if identity is not None:
            outputs.append(identity_path)
        proof = manifest(KIND, contract, [], outputs, result, code=contract['code'])
        proof['inputs'] = {**bound.artifacts, **guard.expected}
        atomic_json(root / 'manifest.json', proof);atomic_json(root / 'state.json', {'status': 'complete', **result})
        return result
    except Exception as error:
        atomic_json(root / 'state.json', {'status': 'failed', 'stage': stage,
            'error_type': type(error).__name__, 'benchmark_completion_proven': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--producer-session', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text());benchmark_spec(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_visible_devices']
    print(json.dumps(run_benchmark(args.pilot, args.producer_session, args.config, args.output,
        resume=args.resume, poll_seconds=args.poll_seconds), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
