"""Verify completed neural training-game receipts and recorded actor assets.

This reader does not load a student, create a training label, or establish
held-out isolation. The explicit model and engine paths are supplied by the
caller, never adopted blindly from a game or a log.
"""
from pathlib import Path
import re

from .evidence import code_identity
from .model_bundle import checkpoint_input_paths
from .recorded_search_inputs import BoundInputs
from .training_game_scoring import load_training_games
from .training_play import checked_policy, fingerprint


EXECUTION_STATUSES = {'complete_actual_32_neural_student_training_games',
                      'complete_actual_neural_student_training_games'}
NEURAL_CLOSURE = ('training_play.py','inference.py','modeling.py','model_bundle.py',
    'expert.py','bridge.py','board_tokens.py','training.py','rules.py','net_pb2.py',
    'sampling.py','explanations.py','oracle.py','verified_inputs.py','evidence.py')


def verified_neural_games(source,execution_proof,*,checkpoint,expert_weights,gameplay_policy,
                          executable,weights,decoder_model=None,decoder_manifest=None,
                          input_digest_receipt=None,bound=None):
    bound = bound or BoundInputs()
    proof = bound.json(execution_proof)
    if (proof.get('status') not in EXECUTION_STATUSES or
            not isinstance(proof.get('qualified_source_commit'),str) or
            not re.fullmatch('[0-9a-f]{40}',proof['qualified_source_commit']) or
            proof.get('preallocated_game_ids_match') is not True or
            proof.get('all_native_completed_outputs_reopened_without_change') is not True or
            type(proof.get('physical_GPU')) is not int or proof['physical_GPU']<0 or
            type(proof.get('student_SGD_updates')) is not int or proof['student_SGD_updates']!=0 or
            type(proof.get('new_teacher_queries')) is not int or proof['new_teacher_queries']!=0):
        raise ValueError('Generated-root admission requires actual neural gameplay execution, not a CPU control')
    source = Path(source).resolve()
    native = bound.json(source/'manifest.json')
    if bound.artifacts[str(source/'manifest.json')]['sha256']!=proof.get('native_manifest_sha256'):
        raise ValueError('Neural execution receipt belongs to a different native gameplay manifest')
    games,info = load_training_games(source,bound)
    if (proof.get('summary')!=native.get('verification') or len(games)!=info['gameplay_policy']['games'] or
            proof['summary'].get('model_turns',0)<1 or
            proof['status']=='complete_actual_32_neural_student_training_games' and len(games)!=32):
        raise ValueError('Neural execution receipt and actual completed gameplay coverage differ')
    declared_code = {'revision':proof.get('qualified_source_commit'),
        'source_sha256':{'src/xqgeneral/'+k:v for k,v in proof.get('source_sha256',{}).items()}}
    if native.get('code')!=declared_code:
        raise ValueError('Neural execution receipt changed its actual producer source')
    current = code_identity()['source_sha256']
    for name in NEURAL_CLOSURE:
        key = 'src/xqgeneral/'+name
        if declared_code['source_sha256'].get(key)!=current[key]:
            raise ValueError('The executed neural gameplay implementation differs from the installed reader')
    config = native['config']
    expected_paths = {'checkpoint':checkpoint,'expert_weights':expert_weights,'policy':gameplay_policy,
                      'executable':executable,'weights':weights,'output':source}
    for field,path in expected_paths.items():
        if not isinstance(config.get(field),str) or Path(config[field]).resolve()!=Path(path).resolve():
            raise ValueError('Explicit neural actor, policy, engine or output path differs from its actual producer')
    if config.get('student_autocast_cache') not in {'on','off'}:
        raise ValueError('Missing actual neural inference precision/cache configuration')
    checkpoint_paths = checkpoint_input_paths(config['checkpoint'])
    actor_paths = [*checkpoint_paths,config['expert_weights']]
    assets = [config['policy'],*actor_paths,config['executable'],config['weights']]
    declared = native.get('inputs',{})
    if set(declared)!={str(p) for p in assets}:
        raise ValueError('Native neural gameplay did not bind its exact model, expert, policy and engine assets')
    for path in assets:
        bound.bind(path,declared[str(path)])
    actual_policy = checked_policy(bound.json(gameplay_policy,declared[config['policy']]))
    if actual_policy!=info['gameplay_policy']:
        raise ValueError('Actual gameplay policy bytes differ from the owned neural episodes')
    actor = fingerprint({str(p):declared[str(p)] for p in actor_paths})
    if actor!=info['actor_identity']:
        raise ValueError('Neural training episodes belong to different complete actor assets')
    checkpoint_bindings = {str(p):declared[str(p)] for p in checkpoint_paths}
    if Path(checkpoint).is_file():
        if proof.get('checkpoint_sha256')!=next(iter(checkpoint_bindings.values()))['sha256']:
            raise ValueError('Neural execution receipt names different selected checkpoint bytes')
    elif proof.get('checkpoint_inputs')!=checkpoint_bindings:
        raise ValueError('Portable neural execution must bind all original checkpoint inputs explicitly')
    from .model_inputs import checkpoint_dependencies
    dependencies = checkpoint_dependencies(checkpoint,bound,decoder_model=decoder_model,
        decoder_manifest=decoder_manifest,input_digest_receipt=input_digest_receipt,
        execution=native.get('runtime_execution'))
    bound.unchanged()
    return games,dict(info,source_neural_execution_verified=True,
        source_recorded_actor_assets_verified=True,
        source_model_assets_verified=dependencies['complete_model_assets_verified'],
        decoder_dependency_verification=dependencies,
        execution_proof=str(Path(execution_proof).resolve()),
        execution_proof_binding=bound.artifacts[str(execution_proof)],
        native_manifest_binding=bound.artifacts[str(source/'manifest.json')],
        complete_actor_bindings={str(p):declared[str(p)] for p in actor_paths},
        original_gameplay_source=declared_code,heldout_position_isolation_applied=False)
