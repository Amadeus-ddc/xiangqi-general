"""Declared auxiliary-training geometry, independent of physical GPU placement."""

THREE_WORKERS = 'full_decoder_auxiliary_three_workers_v1'
FOUR_WORKERS = 'full_decoder_auxiliary_four_workers_v1'


def auxiliary_topology(config):
    if not isinstance(config, dict):
        raise ValueError('Auxiliary topology requires its complete recipe or training configuration')
    profile = config.get('auxiliary_topology_profile', FOUR_WORKERS)
    if profile not in {THREE_WORKERS, FOUR_WORKERS}:
        raise ValueError('Unknown auxiliary full-decoder topology profile')
    world, batch, word = (3, 30, 'three') if profile == THREE_WORKERS else (4, 32, 'four')
    for key, expected in [('ddp_world_size', world), ('batch_size', batch), ('micro_batch_size', 1)]:
        if type(config.get(key)) is not int or config[key] != expected:
            raise ValueError('Auxiliary topology differs from its declared worker count and divisible global batch')
    if profile == THREE_WORKERS and config.get('distributed_strategy') != 'fsdp2':
        raise ValueError('The three-worker auxiliary profile requires full parameter and gradient sharding')
    return dict(profile=profile, world_size=world, global_batch_size=batch,
        preflight_kind=word + '_gpu_auxiliary_policy_sft_preflight',
        completion_flag='actual_' + word + '_gpu_full_decoder_preflight_passed')


def completed_auxiliary_topology(proof):
    topology = auxiliary_topology(proof.get('config', {}).get('compiled_recipe'))
    result = proof.get('verification', {})
    if (result.get('world_size') != topology['world_size'] or
            result.get(topology['completion_flag']) is not True or
            topology['world_size'] == 3 and (
                result.get('actual_four_gpu_full_decoder_preflight_passed') is not False or
                result.get('auxiliary_topology_profile') != topology['profile'])):
        raise ValueError('Completed auxiliary model must identify its actual worker count and capacity proof')
    return topology
