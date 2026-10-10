from collections import Counter
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from xqgeneral.evidence import write_jsonl
from xqgeneral.finite_training import (
    FINITE_PROFILE, LEGACY_PROFILE, FiniteMixtureEpochs, data_profile, mixture_counts, prepare_finite_epochs,
)
from xqgeneral.sft import prepare_config
from xqgeneral.training import compatible_resume


def rows(sizes):
    return [{'id': f'{stage}/{i}', 'stage': stage, 'split': 'train',
             'task_type': 'rare' if i == 0 else 'common', 'x': i + 1}
            for stage, count in sizes.items() for i in range(count)]


def plan(records, mixture=None, **kwargs):
    settings = dict(seed=41, global_batch_size=8, micro_batch_size=2)
    settings.update(kwargs)
    return FiniteMixtureEpochs(records, mixture or {'new': 1}, **settings)


def test_paper_mixture_uses_whole_anchor_and_fixed_uniform_replay_subset():
    records = rows({'old': 90, 'new': 90})
    sample = plan(records, {'old': 1, 'new': 9}, mixture_seed=17)
    assert sample.mix['selected_rows_by_stage'] == {'old': 10, 'new': 90}
    expected_old = np.random.default_rng(17).permutation(90)[:10]
    assert sample.selected[:10].tolist() == expected_old.tolist()
    assert sample.selected[10:].tolist() == list(range(90, 180))
    selected = {records[i]['id'] for i in sample.selected}
    for epoch in range(3):
        actual = []
        for step in range(sample.steps_per_epoch):
            batch, report = sample.batch(epoch * sample.steps_per_epoch + step)
            actual.extend(row['id'] for row in batch)
            assert report['global_padding_rows'] == 0
        assert len(actual) == len(set(actual)) == 100
        assert set(actual) == selected
    assert sample.contract()['old_course_subsets_reselected_each_epoch'] is False


def test_fixed_epoch_retains_natural_task_counts_and_changes_only_order():
    records = rows({'new': 19})
    sample = plan(records)
    epochs = []
    for epoch in range(2):
        batches = [sample.batch(epoch * sample.steps_per_epoch + i)[0] for i in range(sample.steps_per_epoch)]
        combined = [row for batch in batches for row in batch]
        assert Counter(row['task_type'] for row in combined) == {'common': 18, 'rare': 1}
        assert sorted(row['id'] for row in combined) == sorted(row['id'] for row in records)
        epochs.append([row['id'] for row in combined])
    assert epochs[0] != epochs[1]
    assert sample.state_at(6)['unique_rows_consumed'] == 38
    assert sample.state_at(6)['padding_rows_consumed'] == 0


@pytest.mark.parametrize('size', [1, 3, 7, 8, 9, 17, 31])
def test_distributed_microbatches_visit_complete_epoch_with_only_declared_tail_padding(size):
    sample = plan(rows({'new': size}), global_batch_size=16, world=4, micro_batch_size=2)
    generator = torch.Generator().manual_seed(41)
    order = torch.randperm(size, generator=generator).tolist()
    expected = order + [order[i % size] for i in range((-size) % 8)]
    reconstructed = []
    for step in range(sample.steps_per_epoch):
        chunks = [sample.batch_indices(step, rank)[0] for rank in range(4)]
        reports = [sample.batch_indices(step, rank)[1] for rank in range(4)]
        assert all(report == reports[0] for report in reports)
        for start in range(0, len(chunks[0]), 2):
            for rank in range(4):
                reconstructed.extend(chunks[rank][start:start + 2])
    assert reconstructed == expected
    state = sample.state_at(sample.steps_per_epoch)
    assert state['unique_rows_consumed'] == size
    assert state['padding_rows_consumed'] == len(expected) - size
    assert state['global_rows_consumed'] == len(expected)


def test_resume_recovers_each_rank_order_across_epoch_and_tail_without_saved_rng():
    records = rows({'old': 63, 'new': 37})
    mixture = {'old': 1, 'new': 3}
    original = plan(records, mixture, global_batch_size=16, world=4)
    state = json.loads(json.dumps(original.state_at(2)))
    restored = plan(records, mixture, global_batch_size=16, world=4)
    restored.check_resume(state, 2)
    for step in range(2, original.steps_per_epoch * 4):
        for rank in range(4):
            assert restored.batch(step, rank) == original.batch(step, rank)


@pytest.mark.parametrize('change', ['position', 'seed', 'mixture_seed', 'row_order', 'row_id', 'world'])
def test_resume_refuses_changed_position_selection_or_distributed_geometry(change):
    records = rows({'old': 36, 'new': 17})
    mixture = {'old': 1, 'new': 3}
    original = plan(records, mixture)
    state = original.state_at(2)
    settings = {}
    if change == 'position': state['epoch'] += 1
    elif change == 'row_order': records = list(reversed(records))
    elif change == 'row_id':
        records = deepcopy(records); records[int(original.selected[0])]['id'] += '/changed'
    elif change == 'world': settings['world'] = 2
    else: settings[change] = 52
    changed = plan(records, mixture, **settings)
    with pytest.raises(ValueError, match='Resume finite'):
        changed.check_resume(state, 2)


def test_checkpoint_restores_identical_cpu_optimizer_weights_and_dropout_after_epoch_boundary():
    records = rows({'new': 19})
    def setup():
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.25), torch.nn.Linear(3, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        return model, optimizer
    def update(model, optimizer, sample, begin, end):
        exposure = []
        for step in range(begin, end):
            batch, _ = sample.batch(step)
            x = torch.tensor([[row['x'] / 20, 1.] for row in batch])
            y = torch.tensor([[row['x'] / 40] for row in batch])
            optimizer.zero_grad(); loss = (model(x) - y).square().mean(); loss.backward(); optimizer.step()
            exposure.append([row['id'] for row in batch])
        return exposure
    torch.manual_seed(17); whole, whole_optimizer = setup()
    expected = update(whole, whole_optimizer, plan(records), 0, 8)
    torch.manual_seed(17); prefix, prefix_optimizer = setup(); sample = plan(records)
    exposure = update(prefix, prefix_optimizer, sample, 0, 4)
    checkpoint = deepcopy({'weights': prefix.state_dict(), 'optimizer': prefix_optimizer.state_dict(),
                           'rng': torch.get_rng_state(), 'data_state': sample.state_at(4)})
    resumed, resumed_optimizer = setup()
    resumed.load_state_dict(checkpoint['weights']); resumed_optimizer.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['rng'])
    restored = plan(records); restored.check_resume(checkpoint['data_state'], 4)
    exposure += update(resumed, resumed_optimizer, restored, 4, 8)
    assert exposure == expected
    for key, value in whole.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
    for index, values in whole_optimizer.state_dict()['state'].items():
        for key, value in values.items():
            assert torch.equal(value, resumed_optimizer.state_dict()['state'][index][key])


def test_largest_weight_tie_and_bankers_rounding_match_author_mixer():
    assert mixture_counts({'first': 5, 'second': 9}, {'first': 1, 'second': 1})['anchor_stage'] == 'first'
    assert mixture_counts({'new': 5, 'old': 8}, {'new': 2, 'old': 1})['selected_rows_by_stage'] == {'new': 5, 'old': 2}
    with pytest.raises(ValueError, match='needs 5 rows'):
        mixture_counts({'new': 5, 'old': 4}, {'new': 1, 'old': 1})


@pytest.mark.parametrize('mixture', [{}, {'new': 0}, {'new': -1}, {'new': True}, {'new': float('nan')}, {'new': float('inf')}])
def test_invalid_mixture_does_not_create_an_epoch(mixture):
    with pytest.raises(ValueError, match='Finite'):
        FiniteMixtureEpochs(rows({'new': 8}), mixture, seed=0, global_batch_size=4)


@pytest.mark.parametrize('kwargs', [{'global_batch_size': 0}, {'global_batch_size': '8'}, {'world': 0},
                                  {'world': True}, {'world': 4, 'micro_batch_size': 3},
                                  {'seed': True}, {'seed': -1}, {'mixture_seed': .2}])
def test_invalid_epoch_batch_or_seed_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError, match='Finite'):
        plan(rows({'new': 8}), **kwargs)


@pytest.mark.parametrize('bad_row', [{'id': ''}, {'id': None}, {'split': 'validation'}, {'split': 'test'}])
def test_heldout_or_missing_identity_cannot_enter_training_epoch(bad_row):
    records = rows({'new': 8}); records[0].update(bad_row)
    with pytest.raises(ValueError, match='unique IDs and the training split'):
        plan(records)
    records = rows({'new': 8}); records[1]['id'] = records[0]['id']
    with pytest.raises(ValueError, match='unique IDs'):
        plan(records)


def test_explicit_profile_preserves_legacy_default_and_guards_exact_resume():
    assert data_profile({}) == LEGACY_PROFILE
    assert prepare_finite_epochs([], {}, {}, 1) is None
    config = {'training_data_profile': FINITE_PROFILE, 'mixture_seed': 0}
    compatible_resume(config, deepcopy(config))
    with pytest.raises(ValueError, match='traversal'):
        compatible_resume({}, config)
    with pytest.raises(ValueError, match='mixture seed'):
        compatible_resume(config, dict(config, mixture_seed=12))
    with pytest.raises(ValueError, match='requires the finite'):
        data_profile({'mixture_seed': 0})


def sft_recipe(tmp_path, count=18, **changes):
    data = tmp_path / 'data'
    write_jsonl(data / 'train.jsonl', rows({'explanation': count}))
    for split in ('validation', 'test'):
        (data / f'{split}.jsonl').write_text('Held-out contents must not be read for the epoch budget')
    recipe = {'data_path': str(data), 'stages': ['explanation'], 'mixture': {'explanation': 1.},
              'epochs': 4, 'max_steps': 64, 'min_steps': 64, 'batch_size': 8,
              'micro_batch_size': 1, 'ddp_world_size': 4, 'training_data_profile': FINITE_PROFILE}
    recipe.update(changes)
    return recipe


def test_sft_budget_finishes_four_whole_epochs_including_short_last_updates(tmp_path):
    recipe = sft_recipe(tmp_path)
    compiled = prepare_config({}, recipe, tmp_path/'parent.pt', tmp_path/'output')
    assert compiled['steps'] == compiled['min_steps'] == 12
    recipe.pop('training_data_profile')
    assert prepare_config({}, recipe, tmp_path/'parent.pt', tmp_path/'legacy')['steps'] == 9


def test_sft_replay_selection_counts_all_training_sources_and_honors_cap(tmp_path):
    recipe = sft_recipe(tmp_path, count=19)
    replay = tmp_path / 'replay';write_jsonl(replay / 'train.jsonl', rows({'replay': 11}))
    recipe.update(replay_data_paths=[str(replay)], mixture={'explanation': .75, 'replay': .25}, max_steps=13)
    # 19 primary + round(19 / 3) replay = 25; four epochs need sixteen updates.
    assert prepare_config({}, recipe, tmp_path/'parent.pt', tmp_path/'output')['steps'] == 13
    recipe['max_steps'] = 64
    assert prepare_config({}, recipe, tmp_path/'parent.pt', tmp_path/'output2')['steps'] == 16
    write_jsonl(replay / 'train.jsonl', rows({'replay': 5}))
    with pytest.raises(ValueError, match='only 5 available'):
        prepare_config({}, recipe, tmp_path/'parent.pt', tmp_path/'bad')


@pytest.mark.parametrize('change', [{'epochs': 1.5}, {'micro_batch_size': 3}])
def test_finite_sft_rejects_ambiguous_epoch_or_microbatch_budget(tmp_path, change):
    recipe = sft_recipe(tmp_path, **change)
    with pytest.raises(ValueError, match='Finite SFT'):
        prepare_config({}, recipe, tmp_path/'parent.pt', tmp_path/'bad')
