from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import random

import pytest
import torch

from xqgeneral import training


def feature_fixture(keys):
    unique = list(dict.fromkeys(keys))
    indices = {key: i for i, key in enumerate(unique)}
    values = torch.arange(1, len(unique) + 1, dtype=torch.float32).view(-1, 1)
    return {'features': [values, values + 10]}, indices


def inputs(tokenizer, records, max_tokens, device, *args, **kwargs):
    return {'input_ids': torch.tensor([[row['value']] for row in records]),
            'labels': torch.tensor([[0, 1] for _ in records])}


class RecordingModel:
    def __init__(self):
        self.pairs = []

    def eval(self):
        pass

    @contextmanager
    def board_context(self, features):
        self.features = features[0].float()[:, 0]
        yield

    def __call__(self, input_ids, **kwargs):
        expected = input_ids[:, 0].float()
        self.pairs.extend(zip(expected.tolist(), self.features.tolist()))
        return SimpleNamespace(loss=(expected - self.features).abs().mean())


@pytest.mark.parametrize('batch_size', [1, 2, 3, 4])
def test_shuffled_validation_changes_every_history_even_in_singleton_batches(monkeypatch, batch_size):
    keys = ['a', 'b', 'c', 'd', 'e']
    cache, indices = feature_fixture(keys)
    rows = [{'id': key, 'feature_key': key, 'value': i + 1} for i, key in enumerate(keys)]
    original = deepcopy(rows)
    monkeypatch.setattr(training, 'batch_inputs', inputs)
    model = RecordingModel()
    config = {'batch_size': 16, 'eval_batch_size': batch_size, 'max_tokens': 10}
    loss = training.evaluate(model, None, rows, cache, indices, config, 'cpu', rotate=True)
    assert loss > 0
    assert len(model.pairs) == len(rows)
    assert all(target != feature for target, feature in model.pairs)
    assert Counter(feature for _, feature in model.pairs) == Counter(range(1, 6))
    assert [target for target, _ in model.pairs] == list(range(1, 6))
    assert rows == original


@pytest.mark.parametrize('keys', [
    ['a', 'a', 'b', 'b'],
    ['a', 'a', 'a', 'b', 'b', 'c'],
    ['a', 'a', 'b', 'b', 'c', 'c'],
    ['a', 'b', 'c', 'd', 'e'],
])
def test_feature_shuffle_preserves_each_layer_multiset_and_excludes_matching_histories(keys):
    cache, indices = feature_fixture(keys)
    rows = [{'feature_key': key} for key in keys]
    normal = training.select_features(cache, indices, rows, 'cpu')
    shuffled = training.select_features(cache, indices, rows, 'cpu', rotate=True)
    for before, after in zip(normal, shuffled):
        assert torch.all(before != after)
        assert Counter(before.flatten().tolist()) == Counter(after.flatten().tolist())


@pytest.mark.parametrize('keys', [[], ['a'], ['a', 'a'], ['a', 'a', 'a', 'b']])
def test_impossible_mismatch_is_rejected_instead_of_reported_as_shuffled(keys):
    cache, indices = feature_fixture(keys)
    rows = [{'feature_key': key} for key in keys]
    with pytest.raises(ValueError, match='different histories'):
        training.select_features(cache, indices, rows, 'cpu', rotate=True)


def test_normal_and_zero_validation_keep_original_questions_and_pairing(monkeypatch):
    keys = ['a', 'b', 'c', 'd', 'e']
    cache, indices = feature_fixture(keys)
    rows = [{'feature_key': key, 'value': i + 1} for i, key in enumerate(keys)]
    config = {'batch_size': 16, 'eval_batch_size': 1, 'max_tokens': 10}
    monkeypatch.setattr(training, 'batch_inputs', inputs)
    normal, zero = RecordingModel(), RecordingModel()
    assert training.evaluate(normal, None, rows, cache, indices, config, 'cpu') == 0
    assert training.evaluate(zero, None, rows, cache, indices, config, 'cpu', zero=True) == 3
    assert normal.pairs == [(float(i), float(i)) for i in range(1, 6)]
    assert zero.pairs == [(float(i), 0.0) for i in range(1, 6)]


def test_shuffled_validation_does_not_advance_training_rng(monkeypatch):
    cache, indices = feature_fixture(['a', 'b', 'c'])
    rows = [{'feature_key': key, 'value': i + 1} for i, key in enumerate(['a', 'b', 'c'])]
    config = {'batch_size': 16, 'eval_batch_size': 1, 'max_tokens': 10}
    monkeypatch.setattr(training, 'batch_inputs', inputs)
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    first, second = RecordingModel(), RecordingModel()
    training.evaluate(first, None, rows, cache, indices, config, 'cpu', rotate=True)
    training.evaluate(second, None, rows, cache, indices, config, 'cpu', rotate=True)
    assert first.pairs == second.pairs
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
