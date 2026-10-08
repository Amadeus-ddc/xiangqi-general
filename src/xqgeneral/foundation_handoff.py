"""Export an unchanged, four-course raw-gated foundation checkpoint for explanation SFT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import torch

from .curriculum_data import STAGES
from .evidence import atomic_json, digest, load_jsonl, manifest
from .explanations import continuation_positions
from .foundation_preflight import clean_recipe, native_answer, native_context, validate_task
from .gated_curriculum import raw_gate_from_artifacts, training_config
from .modeling import foundation_state_summary


class Artifacts:
    """Hash each distinct declared input once, then guard its file identity through export."""
    def __init__(self):
        self.expected = {}
        self.stats = {}

    def add(self, path, expected):
        name = str(Path(path))
        if (set(expected) != {'sha256', 'bytes'} or not isinstance(expected['sha256'], str) or
                re.fullmatch('[0-9a-f]{64}', expected['sha256']) is None or
                type(expected['bytes']) is not int or expected['bytes'] < 0):
            raise ValueError('Malformed declared handoff artifact')
        if name in self.expected and self.expected[name] != expected:
            raise ValueError(f'Conflicting handoff artifact identities: {name}')
        self.expected[name] = expected

    def json(self, path):
        raw = Path(path).read_bytes()
        self.add(path, {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
        return json.loads(raw)

    def proof(self, path, kind):
        value = self.json(path)
        if value.get('status') != 'complete' or value.get('kind') != kind:
            raise ValueError(f'A completed {kind} proof is required')
        for section in ['inputs', 'outputs']:
            for name, expected in value[section].items():
                self.add(name, expected)
        return value

    @staticmethod
    def signature(path):
        stat = Path(path).stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def verify(self):
        for name, expected in self.expected.items():
            before = self.signature(name)
            if before[2] != expected['bytes'] or digest(name) != expected['sha256']:
                raise ValueError(f'Changed handoff evidence: {name}')
            if self.signature(name) != before:
                raise ValueError(f'Handoff input changed while hashing: {name}')
            self.stats[name] = before

    def unchanged(self, linked_checkpoint=None):
        for name, signature in self.stats.items():
            actual = self.signature(name)
            # Creating the export hard link changes only the selected inode's ctime.
            stable = actual[:-1] == signature[:-1] if name == linked_checkpoint else actual == signature
            if not stable:
                raise ValueError('Handoff evidence changed during verification')


def export_curriculum(curriculum, output):
    """Prepare a selected checkpoint, never start training or consult independent test answers."""
    root, output = Path(curriculum), Path(output)
    if output.exists():
        raise FileExistsError('Preserve completed and interrupted handoffs; use a fresh output directory')
    source_manifest = root / 'manifest.json'
    if not source_manifest.exists():
        raise ValueError('Four completed raw QA courses are required before handoff')
    artifacts = Artifacts()
    whole = artifacts.proof(source_manifest, 'four_course_raw_qa_gated_curriculum')
    recipe, overall = whole['config'], whole['verification']
    clean_recipe(recipe)
    if (Path(recipe['output']).resolve() != root.resolve() or
            overall.get('all_raw_qa_gates_passed') is not True or
            overall.get('independent_test_used') is not False or
            len(overall.get('courses', [])) != len(STAGES)):
        raise ValueError('Handoff requires all four new courses without independent-test selection')
    # Stored paths are relative to the producer workspace, even when the caller uses an absolute root.
    root = Path(recipe['output'])
    stage_paths = {str(root / stage / 'manifest.json') for stage in STAGES}
    recipe_paths = set(whole['inputs']) - stage_paths
    if len(recipe_paths) != 1 or artifacts.json(next(iter(recipe_paths))) != recipe:
        raise ValueError('Completed curriculum does not bind its unchanged source recipe')
    recipe_path = next(iter(recipe_paths))
    courses, parent = [], None
    for index, stage in enumerate(STAGES):
        stage_root = root / stage
        stage_path = stage_root / 'manifest.json'
        if str(stage_path) not in whole['inputs']:
            raise ValueError('Completed curriculum is missing an ordered course proof')
        proof = artifacts.proof(stage_path, 'raw_qa_gated_foundation_course')
        result = proof['verification']
        expected_config = training_config(recipe, index, root, parent)
        if (result != overall['courses'][index] or result['stage'] != stage or
                proof['config'] != expected_config or
                result.get('next_course_permitted') is not True or
                result.get('optimizer_reset_between_courses') is not True or
                result.get('independent_test_used') is not False or not result['gate']['passed']):
            raise ValueError('Course order, raw gate or clean initialization lineage differs')
        step = result['actual_steps'];budget = recipe['course_budgets'][index]
        if (type(step) is not int or not budget['minimum_steps'] <= step <= budget['steps'] or
                result['global_batch_size'] != recipe['batch_size'] or
                result['sample_presentations'] != step * recipe['batch_size']):
            raise ValueError('Course actual update count differs from its declared budget')
        candidate = stage_root / 'candidates' / f'step-{step}'
        checkpoint = candidate / 'adapter.pt'
        if (result['selected_checkpoint'] != str(checkpoint) or
                proof['outputs'].get(str(checkpoint), {}).get('sha256') != result['selected_checkpoint_sha256']):
            raise ValueError('Course selected weight differs from its actual gated candidate')
        config_path, snapshot_path, qa_path = stage_root / 'training.config.json', candidate / 'snapshot.json', candidate / 'qa' / 'manifest.json'
        if any(str(p) not in proof['inputs'] for p in [recipe_path, config_path, snapshot_path, qa_path]):
            raise ValueError('Course proof does not bind its configuration, snapshot and raw evaluation')
        if str(candidate / 'gate.json') not in proof['outputs']:
            raise ValueError('Course proof does not bind its actual raw gate')
        if artifacts.json(config_path) != expected_config:
            raise ValueError('Stored course configuration differs')
        snapshot = artifacts.json(snapshot_path)
        if (snapshot['actual_step'] != step or snapshot['training_config'] != expected_config or
                snapshot['adapter_sha256'] != result['selected_checkpoint_sha256'] or
                snapshot.get('atomic_source_inode_pinned') is not True or
                snapshot.get('optimizer_state_in_adapter') is not False):
            raise ValueError('Course snapshot differs from the selected actual update')
        for name, sha in snapshot['training_input_hashes'].items():
            artifacts.add(name, {'sha256': sha, 'bytes': Path(name).stat().st_size})
        qa = artifacts.proof(qa_path, 'balanced_board_qa')
        if proof['code'] != whole['code'] or qa['code'] != whole['code']:
            raise ValueError('All courses and raw evaluations must use the same preserved execution source')
        for path in [checkpoint, Path(recipe['data_path']) / 'validation.jsonl', Path(recipe['feature_path'])]:
            if str(path) not in qa['inputs']:
                raise ValueError('Raw course evaluation lacks a required input binding')
        for path in [candidate / 'qa' / 'predictions.jsonl', candidate / 'qa' / 'metrics.json']:
            if str(path) not in qa['outputs']:
                raise ValueError('Raw course evaluation lacks actual predictions or metrics')
        courses.append((checkpoint, snapshot, proof, qa, result));parent = checkpoint
    if set(whole['outputs']) != {str(parent)}:
        raise ValueError('Completed curriculum must select the actual final-course candidate')
    model_config_path = Path(recipe['model_path']) / 'config.json'
    base_config = artifacts.json(model_config_path)
    artifacts.verify()
    validation = load_jsonl(Path(recipe['data_path']) / 'validation.jsonl')
    validation_by_id = {row['id']: row for row in validation}
    if len(validation_by_id) != len(validation) or any(row['split'] != 'validation' for row in validation):
        raise ValueError('Foundation validation has duplicate identities or an independent-test row')
    summaries, natively_checked, continuations_checked = [], set(), set()
    torch.set_num_threads(1)
    for index, (checkpoint, snapshot, proof, qa, result) in enumerate(courses):
        directory = checkpoint.parent / 'qa'
        metrics = json.loads((directory / 'metrics.json').read_text())
        if (metrics.get('oracle_repairs', 0) != 0 or metrics.get('legality_is_imposed_by_decoding', False) or
                metrics.get('rule_legal_constraints', False)):
            raise ValueError('Foundation handoff requires unmodified raw model answers')
        records = load_jsonl(directory / 'predictions.jsonl')
        recomputed = raw_gate_from_artifacts(qa, metrics,
            records, checkpoint, STAGES[:index + 1], recipe['raw_qa_gates'],
            recipe['data_path'], recipe['feature_path'], validation)
        if recomputed != result['gate'] or artifacts.json(checkpoint.parent / 'gate.json') != recomputed:
            raise ValueError('Stored course pass differs from the actual raw validation answers')
        for record in records:
            if record['id'] not in natively_checked:
                row = validation_by_id[record['id']]
                native_context(row);validate_task(row)
                context = row['feature_key'], tuple(row.get('future_moves', []))
                if context not in continuations_checked:
                    continuation_positions(row, {'pv': list(context[1])})
                    continuations_checked.add(context)
                if native_answer(row) != row['answer']:
                    raise ValueError('Raw foundation validation target differs from native rules')
                natively_checked.add(record['id'])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if (saved['config'] != proof['config'] or saved['selected_step'] != result['actual_steps'] or
                saved['code'] != snapshot['training_code'] or saved['code'] != whole['code'] or
                saved['input_hashes'] != snapshot['training_input_hashes']):
            raise ValueError('Selected checkpoint differs from its frozen training contract')
        inputs = [recipe['feature_path'], *[str(Path(recipe['data_path']) / f'{s}.jsonl') for s in ['train', 'validation']]]
        if index:inputs.append(str(courses[index - 1][0]))
        if set(saved['input_hashes']) != set(inputs):
            raise ValueError('Selected checkpoint has an unexpected training input lineage')
        for path, sha in saved['input_hashes'].items():
            if path not in artifacts.expected or artifacts.expected[path]['sha256'] != sha:
                raise ValueError('Checkpoint training inputs differ from hash-verified course evidence')
        summaries.append(foundation_state_summary(saved, base_config))
        del saved
    artifacts.unchanged()
    output.mkdir(parents=True)
    destination = output / 'adapter.pt'
    os.link(parent, destination)
    config_path = output / 'config.json'
    atomic_json(config_path, courses[-1][2]['config'])
    verification = {'all_four_ordered_raw_qa_gates_recomputed': True,
        'all_training_lineage_and_snapshot_contracts_checked': True,
        'courses': overall['courses'], 'checkpoint_state_checks': summaries,
        'freshly_hashed_source_artifacts': len(artifacts.expected),
        'sampled_native_histories_and_gold_answers_recomputed': len(natively_checked),
        'sampled_full_history_continuations_recomputed': len(continuations_checked),
        'source_checkpoint': str(parent), 'source_checkpoint_sha256': artifacts.expected[str(parent)]['sha256'],
        'exported_weights_unchanged': True, 'independent_test_answers_used': False,
        'model_weights_loaded_on_gpu_or_sft_started': False, 'four_course_handoff_export_completed': True}
    proof = manifest('raw_qa_gated_foundation_handoff', courses[-1][2]['config'],
        outputs=[destination, config_path], verification=verification)
    if proof['outputs'][str(destination)] != artifacts.expected[str(parent)]:
        raise ValueError('Exported foundation weight bytes changed')
    artifacts.unchanged(linked_checkpoint=str(parent))
    proof['inputs'] = artifacts.expected
    atomic_json(output / 'manifest.json', proof)
    return verification


def validate_sft_handoff(proof, saved):
    """Require the export's ordered-course contract before SFT loads any model."""
    result = proof['verification']
    if (proof['config'] != saved or
            result.get('all_four_ordered_raw_qa_gates_recomputed') is not True or
            result.get('all_training_lineage_and_snapshot_contracts_checked') is not True or
            result.get('exported_weights_unchanged') is not True or
            result.get('independent_test_answers_used') is not False or
            result.get('four_course_handoff_export_completed') is not True or
            [c.get('stage') for c in result.get('courses', [])] != list(STAGES) or
            any(c.get('independent_test_used') is not False or c.get('next_course_permitted') is not True or
                c['gate'].get('passed') is not True or c['gate'].get('validation_only') is not True or
                c['gate'].get('raw_generation') is not True or c['gate'].get('oracle_used') is not False
                for c in result['courses'])):
        raise ValueError('SFT requires an unchanged export of all four ordered raw foundation courses')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export_curriculum(args.curriculum, args.output)), flush=True)


if __name__ == '__main__':
    main()
