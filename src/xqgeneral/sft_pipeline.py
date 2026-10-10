"""Wait for four courses, verify full-decoder execution, then train initial SFT."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .evidence import atomic_json, code_identity, digest, manifest
from .curriculum_selection import CURRICULUM_KIND
from .foundation_handoff import Artifacts, export_curriculum
from .sft import checked_parent, prepare_config
from .sft_preflight import checked_artifacts, full_parameter_summary, run_preflight
from .feature_store import feature_paths


def wait_for_curriculum(curriculum, producer_session, output, poll_seconds=30):
    if not 0 < poll_seconds <= 60:raise ValueError('Use a positive polling interval of at most 60 seconds')
    path = Path(curriculum) / 'manifest.json'
    while not path.exists():
        if subprocess.run(['tmux', 'has-session', '-t', producer_session], capture_output=True).returncode:
            if path.exists():break
            raise RuntimeError('The owned curriculum producer is no longer live and four courses are incomplete')
        atomic_json(Path(output) / 'state.json', {'status': 'waiting_for_four_courses',
            'curriculum': str(curriculum), 'producer_session': producer_session, 'gpu_model_loaded_or_sft_started': False})
        time.sleep(poll_seconds)
    proof = json.loads(path.read_text())
    if proof.get('status') != 'complete' or proof.get('kind') not in ('four_course_raw_qa_gated_curriculum', CURRICULUM_KIND):
        raise ValueError('A completed four-course curriculum is required')
    return path


def run_pipeline(curriculum, producer_session, recipe_path, token_manifest, output, *, resume=False, poll_seconds=30):
    root = Path(output);recipe_path = Path(recipe_path);token_manifest = Path(token_manifest)
    recipe = json.loads(recipe_path.read_text())
    if recipe.get('require_clean_foundation_handoff') is not True:
        raise ValueError('The sequential pipeline requires the clean four-course SFT recipe')
    token = json.loads(token_manifest.read_text())
    checked_artifacts(token)
    if token.get('kind') != 'latent_only_reviewed_explanation_data_preflight' or token['config']['data'] != recipe['data_path']:
        raise ValueError('The sequential pipeline needs the exact completed reviewed-data token preflight')
    contract = {'curriculum': str(curriculum), 'producer_session': producer_session,
        'recipe': str(recipe_path), 'recipe_sha256': digest(recipe_path), 'token_preflight': str(token_manifest),
        'token_preflight_sha256': digest(token_manifest), 'code': code_identity()}
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Resume only this exact preserved pipeline contract; otherwise use a fresh output')
        if (root / 'manifest.json').exists():
            proof = json.loads((root / 'manifest.json').read_text());checked_artifacts(proof);return proof['verification']
    else:root.mkdir(parents=True);atomic_json(root / 'contract.json', contract)
    stage = 'waiting_for_four_courses'
    def unchanged():
        if (digest(recipe_path) != contract['recipe_sha256'] or
                digest(token_manifest) != contract['token_preflight_sha256'] or code_identity() != contract['code']):
            raise ValueError('Pipeline recipe, prerequisite or preserved execution source changed')
    try:
        curriculum_manifest = wait_for_curriculum(curriculum, producer_session, root, poll_seconds)
        unchanged();checked_artifacts(token)
        stage = 'foundation_handoff';atomic_json(root / 'state.json', {'status': stage})
        handoff = root / 'handoff';source = handoff / 'adapter.pt'
        if handoff.exists():
            handoff_proof = json.loads((handoff / 'manifest.json').read_text());checked_artifacts(handoff_proof)
            if handoff_proof['inputs'].get(str(curriculum_manifest), {}).get('sha256') != digest(curriculum_manifest):
                raise ValueError('Preserved handoff belongs to another curriculum')
            checked_parent(source, True)
        else:export_curriculum(curriculum, handoff)
        cache_signatures = {str(p): Artifacts.signature(p) for p in feature_paths(recipe['feature_path'])}
        def unchanged_cache():
            if any(Artifacts.signature(path) != signature for path, signature in cache_signatures.items()):
                raise ValueError('The verified feature cache changed during sequential SFT execution')
        stage = 'four_gpu_full_decoder_preflight';atomic_json(root / 'state.json', {'status': stage})
        preflight = root / 'preflight'
        result = run_preflight(source, recipe_path, token_manifest, preflight, resume=preflight.exists())
        if result.get('status') != 'complete' or result.get('actual_continuous_four_and_two_plus_two_updates_compared') is not True:
            raise ValueError('Actual full-decoder GPU preflight must pass before formal SFT')
        unchanged();checked_artifacts(token);unchanged_cache()
        stage = 'initial_explanation_sft';atomic_json(root / 'state.json', {'status': stage})
        trained = root / 'sft'
        command = [sys.executable, '-u', '-m', 'xqgeneral.sft', '--recipe', str(recipe_path),
                   '--init', str(source), '--output', str(trained)]
        with (root / 'sft-controller.log').open('a') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        unchanged();checked_artifacts(token);unchanged_cache()
        proof = json.loads((trained / 'manifest.json').read_text());checked_artifacts(proof)
        metrics = json.loads((trained / 'metrics.json').read_text())
        expected = prepare_config(checked_parent(source, True), recipe, source, trained)
        if (proof.get('kind') != 'model_training' or proof['config'] != expected or
                proof['verification'] != metrics or metrics.get('world_size') != 4 or
                metrics.get('decoder_training') != 'full' or metrics.get('selected_step', 0) <= 0 or
                metrics['steps_completed'] < expected['min_steps'] or metrics['steps_completed'] > expected['steps']):
            raise ValueError('Formal SFT completion must match its actual full-decoder configuration and selected updates')
        selected = torch.load(trained / 'adapter.pt', map_location='cpu', weights_only=True, mmap=True)
        reference = torch.load(preflight / 'zero-init.pt', map_location='cpu', weights_only=True, mmap=True)['trainable']
        actual_inputs = {path: item['sha256'] for path, item in proof['inputs'].items()}
        actual_inputs[expected['feature_path']] = json.loads((preflight / 'manifest.json').read_text())['fresh_feature_cache_input']['sha256']
        if (selected['config'] != expected or selected['selected_step'] != metrics['selected_step'] or
                selected['code'] != contract['code'] or selected['input_hashes'] != actual_inputs):
            raise ValueError('Actual selected SFT checkpoint differs from completed execution and inputs')
        summary = full_parameter_summary(selected['trainable'], reference)
        if summary['trainable_parameters'] != metrics['trainable_parameters'] or summary['trainable_tensors'] != result['trainable_tensors']:
            raise ValueError('Actual full-decoder selected parameter counts differ from preflight')
        del selected, reference
        final = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
            'four_courses_completed_before_any_gpu_sft_work': True,
            'clean_handoff_verified': True, 'actual_four_gpu_full_decoder_preflight_passed': True,
            'initial_explanation_sft_completed': True, 'actual_sft_steps': metrics['steps_completed'],
            'selected_sft_step': metrics['selected_step'], 'global_batch_size': metrics['global_batch_size'],
            'world_size': metrics['world_size'], 'selected_checkpoint': str(trained / 'adapter.pt'), **summary,
            'independent_test_used_for_training_or_selection': False,
            'strong_play_or_reliable_coaching_proven': False, 'trained_search_distillation_rounds': 0}
        atomic_json(root / 'verification.json', final)
        atomic_json(root / 'manifest.json', manifest('clean_curriculum_to_initial_sft_pipeline', contract,
            [recipe_path, token_manifest, curriculum_manifest, handoff / 'manifest.json', preflight / 'manifest.json',
             trained / 'manifest.json', trained / 'config.json'],
            [root / 'verification.json', trained / 'adapter.pt', trained / 'metrics.json'], final))
        atomic_json(root / 'state.json', final)
        return final
    except Exception as error:
        atomic_json(root / 'state.json', {'status': 'failed', 'stage': stage, 'error_type': type(error).__name__,
            'preserve_partial_outputs': True, 'formal_sft_completion_proven': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum', required=True);parser.add_argument('--producer-session', required=True)
    parser.add_argument('--recipe', required=True);parser.add_argument('--token-preflight', required=True)
    parser.add_argument('--output', required=True);parser.add_argument('--resume', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30);args = parser.parse_args()
    torch.set_num_threads(8)
    print(json.dumps(run_pipeline(args.curriculum, args.producer_session, args.recipe, args.token_preflight,
        args.output, resume=args.resume, poll_seconds=args.poll_seconds)), flush=True)


if __name__ == '__main__':main()
