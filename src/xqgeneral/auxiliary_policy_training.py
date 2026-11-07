"""Train an auxiliary correction round after its declared-worker GPU checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from .auxiliary_policy_preflight import KIND as TOKEN_KIND,compile_recipe,validation_coverage
from .distillation_preflight import AUXILIARY_ROUND_KIND as KIND,completed_parent
from .evidence import atomic_json,code_identity,digest,manifest
from .finite_training import prepare_finite_epochs
from .full_decoder_topology import auxiliary_topology
from .recorded_search_inputs import BoundInputs
from .sft import prepare_config
from .sft_preflight import checked_artifacts,full_parameter_summary,run_preflight
from .training import select_validation_rows
from .training_index import IndexedTrainingRows,build_index,index_paths
from .verified_inputs import input_session_from_environment


def _validation(recipe,counts,actual_ids=None):
    with_rows=IndexedTrainingRows(recipe['row_index_paths']['validation'],
        [Path(recipe['data_path'])/'validation.jsonl'],recipe['mixture'],split='validation')
    try:
        planned=[r['id'] for r in select_validation_rows(with_rows,recipe['validation_stages'],recipe['seed'],recipe['validation_examples'])]
        if actual_ids is not None and actual_ids!=planned:
            raise ValueError('Actual native validation IDs differ from the complete frozen selection')
        return dict(selected_ids=planned,coverage=validation_coverage(with_rows,planned,counts))
    finally:with_rows.close()


def run_round(source,recipe_template,token_manifest,output,*,resume=False,resource_source=None,retained_ranks=None,prepare_only=False):
    source,recipe_template,token_manifest,root=map(lambda p:Path(p).resolve(),[source,recipe_template,token_manifest,output])
    bound=BoundInputs();template=bound.json(recipe_template);token=bound.json(token_manifest)
    topology=auxiliary_topology(template);world=topology['world_size'];batch=topology['global_batch_size']
    if token.get('kind')!=TOKEN_KIND or token.get('status')!='complete':
        raise ValueError('Require the actual auxiliary corpus token/cache/parent preflight')
    for section in ['inputs','outputs']:
        for name,expected in token[section].items():bound.bind(name,expected)
    if token['config']['logical']['source']!=str(source):raise ValueError('Auxiliary training must retain its exact selected parent')
    saved,parent=completed_parent(source,token['config']['logical']['parent_completion'],bound)
    recipe,expected,budget=compile_recipe(token,template,root)
    contract=dict(source=str(source),source_sha256=digest(source),recipe_template=str(recipe_template),
        recipe_template_sha256=digest(recipe_template),token_preflight=str(token_manifest),
        token_preflight_sha256=digest(token_manifest),compiled_recipe=recipe,code=code_identity())
    transition_reference=None
    if resource_source is not None:
        from .auxiliary_resource_transition import source_spec
        if world!=2:raise ValueError('Released three-worker state can continue only under an explicit two-worker profile')
        transition_reference=source_spec(resource_source,source,token_manifest,retained_ranks,bound)
        contract['resource_source']=transition_reference
    elif retained_ranks is not None:
        raise ValueError('Retained RNG ranks require an explicit released resource source')
    if root.exists():
        if not resume or json.loads((root/'contract.json').read_text())!=contract:
            raise ValueError('Preserve the existing auxiliary round; resume only its exact contract')
        if (root/'manifest.json').exists():
            proof=bound.json(root/'manifest.json');checked_artifacts(proof)
            if proof.get('kind')!=KIND or proof.get('config')!=contract:
                raise ValueError('Completed auxiliary round lost its original source contract')
            return proof['verification']
    else:root.mkdir(parents=True);atomic_json(root/'contract.json',contract)
    recipe_path=root/'recipe.json'
    for path,value in [(recipe_path,recipe),(root/'budget.json',budget)]:
        if path.exists():
            if json.loads(path.read_text())!=value:raise ValueError('Preserved auxiliary recipe or budget changed')
        else:atomic_json(path,value)
        bound.bind(path)
    def unchanged():
        bound.unchanged()
        if json.loads(recipe_path.read_text())!=recipe or code_identity()!=contract['code']:
            raise ValueError('Actual auxiliary recipe, immutable inputs or execution code changed')
    stage='native_seven_stage_row_indexes'
    try:
        atomic_json(root/'state.json',dict(status=stage,formal_auxiliary_SGD_started=False))
        for split,directory in recipe['row_index_paths'].items():
            files=[Path(recipe['data_path'])/(split+'.jsonl')]
            if Path(directory).exists():IndexedTrainingRows(directory,files,recipe['mixture'],split=split).close()
            else:build_index(files,directory,split=split)
        plan=_validation(recipe,budget['validation_rows_by_stage']);path=root/'validation-plan.json'
        if path.exists():
            if json.loads(path.read_text())!=plan:raise ValueError('Frozen complete validation plan changed')
        else:atomic_json(path,plan)
        bound.bind(path);unchanged()
        trained=root/'sft'
        if transition_reference is not None:
            from .auxiliary_resource_transition import prepare_transition,install_transition
            stage='CPU_full_optimizer_resource_transition'
            atomic_json(root/'state.json',dict(status=stage,formal_auxiliary_SGD_started=False))
            transition=root/'resource-transition'
            prepare_transition(transition_reference['path'],expected,transition,
                retained_ranks=transition_reference['retained_ranks'],resume=transition.exists())
            install_transition(transition,trained)
            bound.bind(transition/'manifest.json')
            unchanged()
        if prepare_only:
            prepared=dict(status='prepared_auxiliary_round_without_GPU_or_SGD',world_size=world,
                global_batch_size=batch,formal_auxiliary_SGD_started=False,
                resource_transition_source_step=transition_reference['saved_step'] if transition_reference else None)
            atomic_json(root/'state.json',prepared)
            return prepared
        stage=f'actual_{world}_worker_auxiliary_capacity_and_exact_resume_preflight'
        atomic_json(root/'state.json',dict(status=stage,formal_auxiliary_SGD_started=False))
        preflight=root/'preflight';probe=run_preflight(source,recipe_path,token_manifest,preflight,resume=preflight.exists())
        if (probe.get('status')!='complete' or
                probe.get('world_size')!=world or
                probe.get('actual_continuous_four_and_two_plus_two_updates_compared') is not True or
                probe.get('full_decoder_initialization_preserved_all_parent_tensors') is not True):
            raise ValueError('Actual declared-worker parent-preserving updates and exact resume must pass first')
        gpu=bound.json(preflight/'manifest.json')
        if gpu.get('kind')!=topology['preflight_kind']:
            raise ValueError('Auxiliary round cannot substitute a different GPU probe type')
        for section in ['inputs','outputs']:
            for name,identity in gpu[section].items():bound.bind(name,identity)
        unchanged()
        stage='actual_full_decoder_auxiliary_SGD'
        atomic_json(root/'state.json',dict(status=stage,formal_auxiliary_SGD_started=True,
            **({'resuming_preserved_global_step':transition_reference['saved_step']} if transition_reference else {})))
        command=[sys.executable,'-u','-m','xqgeneral.sft','--recipe',str(recipe_path),'--init',str(source),'--output',str(trained)]
        with (root/'sft-controller.log').open('a') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        unchanged();training=bound.json(trained/'manifest.json');checked_artifacts(training)
        metrics=bound.json(trained/'metrics.json');config=bound.json(trained/'config.json')
        if (training.get('kind')!='model_training' or config!=expected or training.get('config')!=expected or
                training.get('code')!=contract['code'] or training.get('verification')!=metrics or
                metrics.get('world_size')!=world or metrics.get('decoder_training')!='full' or metrics.get('global_batch_size')!=batch or
                metrics.get('selected_step',0)<=0 or not expected['min_steps']<=metrics.get('steps_completed',0)<=expected['steps']):
            raise ValueError('Actual auxiliary completion differs from its full-decoder source, budget or selected updates')
        indexed=IndexedTrainingRows(recipe['row_index_paths']['train'],[Path(recipe['data_path'])/'train.jsonl'],recipe['mixture'])
        try:prepare_finite_epochs(indexed,recipe['mixture'],expected,world).check_resume(metrics.get('training_data_state'),metrics['steps_completed'])
        finally:indexed.close()
        actual_ids=bound.json(trained/'validation_ids.json')
        if _validation(recipe,budget['validation_rows_by_stage'],actual_ids)!=plan:
            raise ValueError('Completed auxiliary validation differs from every original source ID')
        selected_path=trained/'adapter.pt';selected=torch.load(selected_path,weights_only=True,map_location='cpu',mmap=True)
        reference=torch.load(preflight/'zero-init.pt',weights_only=True,map_location='cpu',mmap=True)['trainable']
        inputs={name:value['sha256'] for name,value in training['inputs'].items()}
        inputs[expected['feature_path']]=gpu['verified_feature_cache_input']['sha256']
        if (selected['config']!=expected or selected['selected_step']!=metrics['selected_step'] or
                selected['code']!=contract['code'] or selected['input_hashes']!=inputs):
            raise ValueError('Selected auxiliary weights differ from actual native input/code/config bindings')
        summary=full_parameter_summary(selected['trainable'],reference)
        if (summary['trainable_parameters']!=metrics['trainable_parameters'] or summary['trainable_tensors']!=probe['trainable_tensors']):
            raise ValueError('Auxiliary round did not preserve its actual complete decoder/bridge parameter set')
        del selected,reference
        result=dict(status='complete',evidence_state='reconstructed_baseline',
            initial_explanation_sft_completed_before_round=True,actual_four_gpu_full_decoder_preflight_passed=(world==4),
            actual_auxiliary_policy_sft_completed=True,trained_auxiliary_policy_rounds=parent['verification'].get('trained_auxiliary_policy_rounds',0)+1,
            trained_search_distillation_rounds=parent['verification'].get('trained_search_distillation_rounds',0),
            auxiliary_supervision_is_new_on_policy_search_round=False,selected_checkpoint=str(selected_path),
            selected_step=metrics['selected_step'],actual_steps=metrics['steps_completed'],world_size=world,global_batch_size=batch,
            training_question_rows=budget['training_question_rows'],training_unique_history_contexts=budget['training_unique_history_contexts'],
            rows_per_complete_pass=budget['training_question_rows'],selected_checkpoint_complete_passes=metrics['selected_step']//budget['steps_per_pass'],
            actual_finite_epoch_state_verified=True,training_data_state=metrics['training_data_state'],
            full_source_sampling_profile_verified=True,loss_normalization='token',validation_coverage=plan['coverage'],
            validation_selects_checkpoint=True,independent_test_used_for_training_or_selection=False,raw_answers_repaired=False,
            strong_play_or_reliable_coaching_proven=False,**summary)
        if world in {2,3}:result.update({topology['completion_flag']:True},auxiliary_topology_profile=topology['profile'])
        if transition_reference is not None:
            from .auxiliary_resource_transition import transition_ledger
            result['training_resource_transition']=transition_ledger(root/'resource-transition',transition_reference,
                trained,expected,metrics,contract['code'],bound)
        unchanged();atomic_json(root/'verification.json',result)
        inputs=[source,source.parent/'manifest.json',source.parent/'config.json',recipe_template,recipe_path,token_manifest,
            Path(token['config']['logical']['parent_completion']),preflight/'manifest.json',trained/'manifest.json',trained/'config.json']
        inputs += [path for directory in recipe['row_index_paths'].values() for path in index_paths(directory)]
        if transition_reference is not None:
            inputs += [Path(transition_reference['path']),root/'resource-transition/manifest.json',
                       trained/'execution.json',trained/'resource-transition-installed.json']
        outputs=[root/'verification.json',root/'budget.json',root/'validation-plan.json',selected_path,trained/'metrics.json',trained/'validation_ids.json']
        atomic_json(root/'manifest.json',manifest(KIND,contract,inputs,outputs,result));atomic_json(root/'state.json',result)
        return result
    except Exception as error:
        atomic_json(root/'state.json',dict(status='failed',stage=stage,error_type=type(error).__name__,preserve_partial_outputs=True,
            formal_auxiliary_completion_proven=False));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['init','recipe','token-preflight','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--resource-source')
    parser.add_argument('--retained-ranks',nargs=2,type=int)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    with input_session_from_environment():
        print(json.dumps(run_round(args.init,args.recipe,args.token_preflight,args.output,resume=args.resume,
            resource_source=args.resource_source,retained_ranks=args.retained_ranks,prepare_only=args.prepare_only)),flush=True)


if __name__=='__main__':main()
