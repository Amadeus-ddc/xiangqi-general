"""Declared full-decoder geometry, independent of physical GPU placement."""

THREE_WORKERS = 'full_decoder_auxiliary_three_workers_v1'
FOUR_WORKERS = 'full_decoder_auxiliary_four_workers_v1'
THREE_SEARCH_WORKERS = 'full_decoder_search_three_workers_v1'
FOUR_SEARCH_WORKERS = 'full_decoder_search_four_workers_v1'


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


def search_topology(config):
    if not isinstance(config, dict):
        raise ValueError('Search topology requires its complete recipe or training configuration')
    profile = config.get('search_topology_profile', FOUR_SEARCH_WORKERS)
    if profile not in {THREE_SEARCH_WORKERS, FOUR_SEARCH_WORKERS}:
        raise ValueError('Unknown search full-decoder topology profile')
    world, batch, word = (3, 30, 'three') if profile == THREE_SEARCH_WORKERS else (4, 32, 'four')
    for key, expected in [('ddp_world_size', world), ('batch_size', batch), ('micro_batch_size', 1)]:
        if type(config.get(key)) is not int or config[key] != expected:
            raise ValueError('Search topology differs from its declared worker count and divisible global batch')
    if world == 3 and config.get('distributed_strategy') != 'fsdp2':
        raise ValueError('The three-worker search profile requires full parameter and gradient sharding')
    return dict(profile=profile, world_size=world, global_batch_size=batch,
        preflight_kind=word + '_gpu_search_distillation_sft_preflight',
        completion_flag='actual_' + word + '_gpu_full_decoder_preflight_passed')


def completed_search_topology(proof):
    recipe = proof.get('config', {}).get('compiled_recipe')
    # Preserve the original four-worker completion contract. Explicit new
    # profiles also validate the full recipe; a three-worker completion cannot
    # fall back to a legacy declaration or claim a four-worker capacity check.
    if isinstance(recipe, dict) and 'search_topology_profile' in recipe:
        topology = search_topology(recipe)
    else:
        topology = search_topology(dict(ddp_world_size=4, batch_size=32, micro_batch_size=1))
    result = proof.get('verification', {})
    if (result.get('world_size') != topology['world_size'] or
            result.get(topology['completion_flag']) is not True or
            topology['world_size'] == 3 and (
                result.get('actual_four_gpu_full_decoder_preflight_passed') is not False or
                result.get('search_topology_profile') != topology['profile'])):
        raise ValueError('Completed search model must identify its actual worker count and capacity proof')
    return topology
