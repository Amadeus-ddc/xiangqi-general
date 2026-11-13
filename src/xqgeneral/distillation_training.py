"""Train one real search-distillation round after its data and GPU preflights."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from .distillation_preflight import ROUND_KIND, TOKEN_KIND, completed_parent
from .curriculum_data import STAGES
from .evidence import atomic_json, code_identity, digest, manifest
from .finite_training import (FINITE_PROFILE, FULL_SOURCE_PROFILE, check_full_source_selection,
                              mixture_counts, prepare_finite_epochs)
from .recorded_search_inputs import BoundInputs
from .sft import prepare_config
from .sft_preflight import checked_artifacts, full_parameter_summary, run_preflight
from .training_index import build_index, index_paths, IndexedTrainingRows
from .full_decoder_topology import search_topology
from .distillation_validation import (JOINT_VALIDATION_PROFILE, LEGACY_VALIDATION_PROFILE,
    AUXILIARY_VALIDATION_PROFILE, joint_validation_coverage, validation_stages as stage_names)
from .auxiliary_search_replay import (PROFILE as AUXILIARY_REPLAY_PROFILE,
    COACHING_PROFILE as COACHING_REPLAY_PROFILE,
    LEGACY_PROFILE as LEGACY_REPLAY_PROFILE, STAGES as AUXILIARY_REPLAY_STAGES)


def compile_recipe(template, token, root):
    """Cover all accepted labels at least once within the explicit update cap."""
    if (token.get('status') != 'complete' or token.get('kind') != TOKEN_KIND or
            token['verification'].get('all_train_validation_supervision_fits_without_truncation') is not True or
            token['verification'].get('independent_test_used_for_training_or_selection') is not False):
        raise ValueError('Require the exact completed search-label token preflight')
    validation_profile = template.get('validation_profile', LEGACY_VALIDATION_PROFILE)
    validation_stages = stage_names(validation_profile)
    joint_validation = validation_profile in (JOINT_VALIDATION_PROFILE, AUXILIARY_VALIDATION_PROFILE)
    auxiliary_replay = validation_profile == AUXILIARY_VALIDATION_PROFILE
    replay_stages = AUXILIARY_REPLAY_STAGES if auxiliary_replay else STAGES
    topology = search_topology(template)
    replay_profile = template.get('replay_profile', LEGACY_REPLAY_PROFILE)
    allowed_profiles = (AUXILIARY_REPLAY_PROFILE, COACHING_REPLAY_PROFILE) if auxiliary_replay else (LEGACY_REPLAY_PROFILE,)
    coaching_replay = replay_profile == COACHING_REPLAY_PROFILE
    if (replay_profile not in allowed_profiles or
            token['config'].get('replay_profile', LEGACY_REPLAY_PROFILE) != replay_profile or
            auxiliary_replay and (topology['world_size'] not in {1, 2, 3} or template.get('loss_normalization') != 'token' or
                template.get('source_coverage_profile') != FULL_SOURCE_PROFILE)):
        raise ValueError('Six-stage replay requires matching declared-worker token-loss and full-subset profiles')
    if (template.get('require_clean_foundation_handoff') is not False or
            template.get('decoder_training') != 'full' or template.get('trainable_parameter_dtype') != 'float32' or
            template.get('training_data_profile') != FINITE_PROFILE or
            template.get('validation_stages') != validation_stages or
            template.get('max_tokens') != token['verification']['token_limit'] or
            type(template.get('max_steps')) is not int or template['max_steps'] < 1):
        raise ValueError('Use the finite declared-worker full-decoder distillation recipe and exact length limit')
    counts = token['verification']['by_split']
    if (type(counts.get('train')) is not int or counts['train'] < topology['global_batch_size'] or
            type(counts.get('validation')) is not int or counts['validation'] < 1):
        raise ValueError('A real complete training batch and protected validation labels are required')
    base_reference = token['config'].get('pinned_base_model_manifest')
    if template.get('pinned_base_model_manifest') not in (None, base_reference):
        raise ValueError('The recipe cannot replace its token-verified pinned decoder manifest')
    if base_reference is not None:
        if (not isinstance(base_reference, dict) or set(base_reference) != {'path', 'sha256'} or
                not isinstance(base_reference['path'], str) or not base_reference['path'] or
                not isinstance(base_reference['sha256'], str) or len(base_reference['sha256']) != 64 or
                any(character not in '0123456789abcdef' for character in base_reference['sha256']) or
                token.get('inputs', {}).get(base_reference['path'], {}).get('sha256') != base_reference['sha256']):
            raise ValueError('The pinned decoder manifest must be bound by the completed token preflight')
    replay_paths = token['config'].get('replay_data_paths', [])
    if (not isinstance(replay_paths, list) or any(not isinstance(path, str) or not path for path in replay_paths) or
            len({str(Path(path).resolve()) for path in replay_paths}) != len(replay_paths) or
            template.get('replay_data_paths', []) != replay_paths):
        raise ValueError('Training must use the exact token-verified course replay sources')
    source_counts = {'explanation': counts['train']}
    if replay_paths:
        replay = token['verification'].get('course_replay', {})
        if (template.get('stages') != ['explanation', *replay_stages] or
                not isinstance(template.get('mixture'), dict) or set(template['mixture']) != {'explanation', *replay_stages} or
                template.get('loss_normalization') not in ({'example', 'token'} if topology['world_size'] in {1, 2, 3} else {'example'}) or
                replay.get('data_paths') != replay_paths or
                any(replay.get(key) is not True for key in ['all_raw_train_validation_answers_natively_regraded',
                    'all_features_present', 'combined_split_isolation_verified']) or
                replay.get('independent_test_rows_tokenized') != 0):
            raise ValueError('Joint distillation requires all four certified courses and its declared loss reduction')
        available = replay.get('by_split_and_stage', {})
        expected_stages = dict(train=set(replay_stages), validation=set(replay_stages))
        if coaching_replay:
            expected_stages['train'].add('explanation')
        if (not isinstance(available, dict) or set(available) != {'train', 'validation'} or
                any(not isinstance(available[split], dict) or set(available[split]) != expected_stages[split]
                    for split in available) or
                any(type(available[split][stage]) is not int or available[split][stage] < 1
                    for split in available for stage in expected_stages[split])):
            raise ValueError('Course replay needs actual positive train and validation coverage for every course')
        if replay.get('train_validation_rows_tokenized') != sum(sum(available[split].values()) for split in available):
            raise ValueError('Course replay token coverage differs from its raw source counts')
        if coaching_replay:
            retained = replay.get('original_coaching_replay', {})
            if (type(retained.get('selected_training_rows')) is not int or
                    retained['selected_training_rows'] != available['train']['explanation'] or
                    type(retained.get('source_training_rows')) is not int or
                    retained['source_training_rows'] < retained['selected_training_rows'] or
                    retained.get('heldout_answers_parsed_or_tokenized') != 0):
                raise ValueError('Original coaching retention needs exact train-only source coverage')
        for stage, count in available['train'].items():
            source_counts[stage] = source_counts.get(stage, 0) + count
    elif template.get('stages') != ['explanation'] or template.get('mixture') != {'explanation': 1.0}:
        raise ValueError('Explanation-only distillation must retain its original mixture')
    validation_counts = {'explanation': counts['validation']}
    if joint_validation:
        if not replay_paths:
            raise ValueError('Joint validation requires the four token-verified course replay sources')
        validation_counts.update(available['validation'])
    if ('validation_rows_by_stage' in template and
            (not joint_validation or template['validation_rows_by_stage'] != validation_counts)):
        raise ValueError('Declared validation counts differ from actual token-verified original sources')
    weights = template['mixture']
    derived_full_source = topology['world_size'] in {1, 2, 3} and template.get('source_coverage_profile') == FULL_SOURCE_PROFILE
    if derived_full_source:
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError('Declared full-source mixture weights must be positive finite values')
        weights = source_counts
    mix = mixture_counts(source_counts, weights)
    full_source = check_full_source_selection(template, mix, source_counts)
    if (not full_source and mix['anchor_stage'] != 'explanation' or
            any(count < 1 for count in mix['selected_rows_by_stage'].values())):
        raise ValueError('Keep every accepted explanation and at least one actual row from each declared course')
    per_pass = math.ceil(mix['rows_per_epoch'] / template['batch_size'])
    if per_pass > template['max_steps']:
        raise ValueError('The update cap cannot cover even one complete pass through accepted labels')
    warmup = template.get('warmup_fraction', 0)
    if type(warmup) not in (int, float) or not math.isfinite(warmup) or not 0 <= warmup <= 1:
        raise ValueError('A finite warmup fraction is required')
    max_updates = template['max_steps']
    passes = template.get('max_finite_passes')
    if 'max_finite_passes' in template:
        if type(passes) is not int or passes < 1:
            raise ValueError('Maximum finite passes must be a positive integer')
        if type(template.get('eval_every')) is not int or template['eval_every'] < 1:
            raise ValueError('Finite-pass budgets require a positive integer validation interval')
        max_updates = min(max_updates, passes * per_pass)
    root = Path(root)
    recipe = dict(template, data_path=token['config']['data'], feature_path=token['cache_header_only_input']['path'],
        validation_examples=sum(validation_counts.values()), max_steps=max_updates, epochs=math.ceil(max_updates / per_pass),
        min_steps=min(max_updates, max(per_pass, math.ceil(max_updates * warmup))),
        row_index_paths={split: str(root / (split + '-index')) for split in ['train', 'validation']})
    if 'max_finite_passes' in template:
        recipe['eval_every'] = min(template['eval_every'], per_pass)
    if base_reference is not None:
        recipe['pinned_base_model_manifest'] = base_reference
    if any(weight > 1 for weight in weights.values()):
        recipe['mixture'] = mix['requested_weights']
    if joint_validation:
        recipe['validation_rows_by_stage'] = validation_counts
    budget = {'accepted_training_rows': counts['train'], 'validation_rows': sum(validation_counts.values()),
        'global_batch_size': template['batch_size'], 'updates_per_complete_pass': per_pass,
        'max_updates': max_updates, 'minimum_updates': recipe['min_steps'],
        'finite_passes_allocated_before_update_cap': recipe['epochs'],
        'last_allocated_pass_may_be_partial_at_update_cap': max_updates % per_pass != 0,
        'at_least_one_complete_pass_required_before_early_stop': True,
        'paper_first_round_150000_label_scale_reached': counts['train'] >= 150000}
    if 'max_finite_passes' in template:
        budget.update(configured_max_updates=template['max_steps'], max_finite_passes=passes,
            updates_for_requested_complete_passes=passes * per_pass,
            finite_pass_limit_reduced_max_updates=max_updates < template['max_steps'],
            validation_interval=recipe['eval_every'])
    if replay_paths:
        budget.update(course_replay_source_rows_by_stage={stage: source_counts[stage] for stage in replay_stages},
            finite_mixture=mix, rows_per_complete_pass=mix['rows_per_epoch'],
            course_replay_counts_toward_paper_label_scale=False)
    if joint_validation:
        budget.update(validation_profile=validation_profile, validation_rows_by_stage=validation_counts,
            validation_uses_all_original_rows=True, validation_metric='token_nll_over_complete_original_validation')
    if auxiliary_replay:
        budget.update(replay_profile=replay_profile, auxiliary_policy_replay_included=True,
                      replay_is_explicit_original_training_subset=True,
                      auxiliary_replay_counts_toward_new_search_label_scale=False)
    if coaching_replay:
        budget.update(original_coaching_replay_training_rows=available['train']['explanation'],
                      all_explanation_training_rows=source_counts['explanation'],
                      original_coaching_replay_additional_validation_rows=0,
                      original_coaching_replay_counts_toward_new_search_label_scale=False)
    if full_source:
        budget.update(source_coverage_profile=template['source_coverage_profile'],
            all_actual_training_rows_selected_each_epoch=True, finite_anchor_stage=mix['anchor_stage'])
    if topology['world_size'] in {1, 2, 3}:
        budget.update(search_topology_profile=topology['profile'], world_size=topology['world_size'],
            loss_normalization=recipe['loss_normalization'],
            mixture_derived_from_actual_token_verified_counts=derived_full_source,
            distributed_tail_repeated_rows_per_pass=(-mix['rows_per_epoch']) % topology['world_size'])
    return recipe, budget


def run_round(source, recipe_template, token_manifest, output, *, resume=False):
    root, source = Path(output), Path(source)
    recipe_template, token_manifest = Path(recipe_template), Path(token_manifest)
    bound = BoundInputs()
    template = bound.json(recipe_template)
    token = bound.json(token_manifest)
    if token.get('status') != 'complete':
        raise ValueError('Completed distillation token evidence is required')
    for section in ['inputs', 'outputs']:
        for path, expected in token[section].items():
            bound.bind(path, expected)
    recipe, budget = compile_recipe(template, token, root)
    topology = search_topology(recipe)
    world, batch = topology['world_size'], topology['global_batch_size']
    for directory in recipe.get('replay_data_paths', []):
        for filename in ['manifest.json', 'train.jsonl', 'validation.jsonl']:
            path = str(Path(directory) / filename)
            if path not in token['inputs']:
                raise ValueError('Course replay source must be bound by its completed token preflight')
            bound.bind(path, token['inputs'][path])
    if token['config']['source'] != str(source):
        raise ValueError('The actual search and selected training parent must be identical')
    saved, parent = completed_parent(source, token['config']['parent_completion'], bound)
    previous_rounds = parent['verification'].get('trained_search_distillation_rounds', 0)
    previous_auxiliary = parent['verification'].get('trained_auxiliary_policy_rounds', 0)
    if any(type(value) is not int or value < 0 for value in [previous_rounds, previous_auxiliary]):
        raise ValueError('The actual completed parent must declare valid cumulative training round counts')
    contract = {'source': str(source), 'source_sha256': digest(source),
        'recipe_template': str(recipe_template), 'recipe_template_sha256': digest(recipe_template),
        'token_preflight': str(token_manifest), 'token_preflight_sha256': digest(token_manifest),
        'compiled_recipe': recipe, 'code': code_identity()}
    if root.exists():
        if not resume or json.loads((root / 'contract.json').read_text()) != contract:
            raise ValueError('Resume only the exact preserved distillation round; otherwise use a fresh output')
        if (root / 'manifest.json').exists():
            proof = json.loads((root / 'manifest.json').read_text()); checked_artifacts(proof)
            if proof.get('kind') != ROUND_KIND or proof.get('config') != contract:
                raise ValueError('Completed distillation round differs from its actual contract')
            return proof['verification']
    else:
        root.mkdir(parents=True); atomic_json(root / 'contract.json', contract)
    recipe_path = root / 'recipe.json'
    for path, value in [(recipe_path, recipe), (root / 'budget.json', budget)]:
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError('Preserved compiled recipe or budget differs')
        else:
            atomic_json(path, value)
        bound.bind(path)
    stage = 'training_row_indexes'

    def unchanged():
        bound.unchanged()
        if (json.loads(recipe_path.read_text()) != recipe or code_identity() != contract['code']):
            raise ValueError('Actual distillation parent, recipe, token contract or execution source changed')

    try:
        atomic_json(root / 'state.json', {'status': stage, 'formal_distillation_started': False})
        for split, directory in recipe['row_index_paths'].items():
            paths = [Path(directory) / (split + '.jsonl') for directory in
                     [recipe['data_path'], *recipe.get('replay_data_paths', [])]]
            if Path(directory).exists():
                IndexedTrainingRows(directory, paths, recipe['mixture'], split=split).close()
            else:
                build_index(paths, directory, split=split)
        validation_plan = None
        if recipe.get('validation_profile') in (JOINT_VALIDATION_PROFILE, AUXILIARY_VALIDATION_PROFILE):
            from .training import select_validation_rows
            indexed = IndexedTrainingRows(recipe['row_index_paths']['validation'],
                [Path(directory) / 'validation.jsonl' for directory in
                 [recipe['data_path'], *recipe['replay_data_paths']]], recipe['mixture'], split='validation')
            try:
                selected_ids = [row['id'] for row in select_validation_rows(indexed,
                    recipe['validation_stages'], recipe['seed'], recipe['validation_examples'])]
                validation_plan = {'selected_ids': selected_ids, 'coverage': joint_validation_coverage(
                    indexed, selected_ids, budget['validation_rows_by_stage'], profile=recipe['validation_profile'])}
            finally:
                indexed.close()
            path = root / 'joint-validation-plan.json'
            if path.exists():
                if json.loads(path.read_text()) != validation_plan:
                    raise ValueError('Preserved joint validation selection changed')
            else:
                atomic_json(path, validation_plan)
            bound.bind(path)
        unchanged()
        stage = topology['preflight_kind'].replace('_search_distillation_sft_preflight', '_full_decoder_distillation_preflight')
        atomic_json(root / 'state.json', {'status': stage, 'formal_distillation_started': False})
        preflight_root = root / 'preflight'
        probe = run_preflight(source, recipe_path, token_manifest, preflight_root, resume=preflight_root.exists())
        if (probe.get('status') != 'complete' or
                probe.get('world_size') != world or
                probe.get('actual_continuous_four_and_two_plus_two_updates_compared') is not True or
                probe.get('full_decoder_initialization_preserved_all_parent_tensors') is not True):
            raise ValueError('The actual full-decoder distillation GPU preflight must pass first')
        if world == 1 and probe.get('actual_cpu_optimizer_state_placement_verified') is not True:
            raise ValueError('One-worker distillation requires actual CPU moment residency during its GPU probe')
        probe_proof = json.loads((preflight_root / 'manifest.json').read_text())
        if (probe_proof.get('status') != 'complete' or probe_proof.get('kind') != topology['preflight_kind'] or
                probe_proof.get('verification') != probe or probe_proof.get('code') != contract['code'] or
                probe_proof.get('config') != {'recipe': str(recipe_path), 'source': str(source),
                                            'token_preflight': str(token_manifest)}):
            raise ValueError('Actual search preflight must identify the declared worker topology')
        unchanged()
        stage = 'actual_search_distillation_sft'
        atomic_json(root / 'state.json', {'status': stage, 'formal_distillation_started': True})
        trained = root / 'sft'
        command = [sys.executable, '-u', '-m', 'xqgeneral.sft', '--recipe', str(recipe_path),
                   '--init', str(source), '--output', str(trained)]
        with (root / 'sft-controller.log').open('a') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        unchanged()
        proof = json.loads((trained / 'manifest.json').read_text()); checked_artifacts(proof)
        config = json.loads((trained / 'config.json').read_text())
        expected = prepare_config(saved, recipe, source, trained)
        metrics = json.loads((trained / 'metrics.json').read_text())
        if (proof.get('kind') != 'model_training' or config != expected or proof.get('config') != expected or
                proof.get('verification') != metrics or proof.get('code') != contract['code'] or
                metrics.get('world_size') != world or metrics.get('decoder_training') != 'full' or
                metrics.get('global_batch_size') != batch or metrics.get('selected_step', 0) <= 0 or
                not expected['min_steps'] <= metrics.get('steps_completed', 0) <= expected['steps']):
            raise ValueError('Actual distillation completion differs from its full-decoder recipe and budget')
        indexed = IndexedTrainingRows(recipe['row_index_paths']['train'],
            [Path(directory) / 'train.jsonl' for directory in
             [recipe['data_path'], *recipe.get('replay_data_paths', [])]], recipe['mixture'])
        try:
            prepare_finite_epochs(indexed, recipe['mixture'], expected, world).check_resume(
                metrics.get('training_data_state'), metrics['steps_completed'])
        finally:
            indexed.close()
        validation_coverage = None
        if recipe.get('validation_profile') in (JOINT_VALIDATION_PROFILE, AUXILIARY_VALIDATION_PROFILE):
            validation_ids_path = trained / 'validation_ids.json'
            if not validation_ids_path.is_file():
                raise ValueError('Joint validation requires the actual native saved validation ID ledger')
            bound.bind(validation_ids_path)
            indexed = IndexedTrainingRows(recipe['row_index_paths']['validation'],
                [Path(directory) / 'validation.jsonl' for directory in
                 [recipe['data_path'], *recipe.get('replay_data_paths', [])]], recipe['mixture'], split='validation')
            try:
                actual_ids = json.loads(validation_ids_path.read_text())
                validation_coverage = joint_validation_coverage(indexed, actual_ids,
                    budget['validation_rows_by_stage'], profile=recipe['validation_profile'])
            finally:
                indexed.close()
            if (actual_ids != validation_plan['selected_ids'] or
                    validation_coverage != validation_plan['coverage']):
                raise ValueError('Actual native joint validation differs from its complete pretraining selection')
        selected_path = trained / 'adapter.pt'
        selected = torch.load(selected_path, weights_only=True, map_location='cpu', mmap=True)
        reference = torch.load(preflight_root / 'zero-init.pt', weights_only=True, map_location='cpu', mmap=True)['trainable']
        actual_inputs = {p: b['sha256'] for p, b in proof['inputs'].items()}
        actual_inputs[expected['feature_path']] = json.loads((preflight_root / 'manifest.json').read_text())['fresh_feature_cache_input']['sha256']
        if (selected['config'] != expected or selected['selected_step'] != metrics['selected_step'] or
                selected['code'] != contract['code'] or selected['input_hashes'] != actual_inputs):
            raise ValueError('Actual selected distillation weights differ from the completed training contract')
        summary = full_parameter_summary(selected['trainable'], reference)
        if summary['trainable_parameters'] != metrics['trainable_parameters'] or summary['trainable_tensors'] != probe['trainable_tensors']:
            raise ValueError('Distillation did not retain the actual full-decoder parameter set')
        if world == 1:
            from .optimizer_offload import validate_cpu_moment_placement
            validate_cpu_moment_placement(metrics.get('optimizer_runtime_state_placement'),
                                          summary['trainable_tensors'], summary['trainable_parameters'])
        del selected, reference
        result = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
            'initial_explanation_sft_completed_before_round': True,
            'actual_four_gpu_full_decoder_preflight_passed': world == 4,
            'actual_distillation_sft_completed': True,
            'trained_search_distillation_rounds': previous_rounds + 1, 'selected_checkpoint': str(selected_path),
            'trained_auxiliary_policy_rounds': previous_auxiliary,
            'selected_step': metrics['selected_step'], 'actual_steps': metrics['steps_completed'],
            'world_size': world, 'global_batch_size': batch, 'accepted_training_rows': budget['accepted_training_rows'],
            'actual_finite_epoch_state_verified': True, 'training_data_state': metrics['training_data_state'],
            'paper_first_round_150000_label_scale_reached': budget['paper_first_round_150000_label_scale_reached'],
            'validation_selects_checkpoint': True, 'independent_test_used_for_training_or_selection': False,
            'raw_answers_repaired': False, 'strength_and_coaching_gain_evaluated': False,
            'strong_play_or_reliable_coaching_proven': False, **summary}
        if world in {1, 2, 3}:
            result.update({topology['completion_flag']: True},
                          search_topology_profile=topology['profile'])
        if world == 1:
            result['actual_cpu_optimizer_state_placement_verified'] = True
        if recipe.get('replay_data_paths'):
            result.update(course_replay_sources=recipe['replay_data_paths'],
                selected_training_rows_by_stage=budget['finite_mixture']['selected_rows_by_stage'],
                rows_per_complete_pass=budget['rows_per_complete_pass'], loss_normalization=recipe['loss_normalization'],
                course_replay_counts_toward_paper_label_scale=False)
        if validation_coverage is not None:
            result['joint_validation_coverage'] = validation_coverage
        if 'source_coverage_profile' in budget:
            result.update(source_coverage_profile=budget['source_coverage_profile'],
                          all_actual_training_rows_selected_each_epoch=True)
        atomic_json(root / 'verification.json', result)
        inputs = [source, source.parent / 'manifest.json', source.parent / 'config.json',
            recipe_template, recipe_path, token_manifest, Path(token['config']['parent_completion']),
            preflight_root / 'manifest.json', trained / 'manifest.json', trained / 'config.json']
        inputs += [p for directory in recipe['row_index_paths'].values() for p in index_paths(directory)]
        inputs += [Path(directory) / filename for directory in recipe.get('replay_data_paths', [])
                   for filename in ['manifest.json', 'train.jsonl', 'validation.jsonl']]
        atomic_json(root / 'manifest.json', manifest(ROUND_KIND, contract, inputs,
            [root / 'verification.json', root / 'budget.json', selected_path, trained / 'metrics.json',
             *([root / 'joint-validation-plan.json', trained / 'validation_ids.json']
               if validation_coverage is not None else [])], result))
        atomic_json(root / 'state.json', result)
        return result
    except Exception as error:
        atomic_json(root / 'state.json', {'status': 'failed', 'stage': stage, 'error_type': type(error).__name__,
            'preserve_partial_outputs': True, 'formal_distillation_completion_proven': False})
        raise


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['init', 'recipe', 'token-preflight', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    print(json.dumps(run_round(args.init, args.recipe, args.token_preflight, args.output, resume=args.resume)), flush=True)


def main():
    from .verified_inputs import input_session_from_environment
    with input_session_from_environment():
        _main()


if __name__ == '__main__':
    main()
