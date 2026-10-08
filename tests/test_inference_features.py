import pytest
import torch

from xqgeneral.evidence import history_key
from xqgeneral.expert import encode_history
from xqgeneral.inference import Predictor
from xqgeneral.rules import START_FEN, replay
from xqgeneral.training import select_features


@pytest.mark.parametrize('include_history', [True, False])
@pytest.mark.parametrize('cache_miss', ['missing_key', 'no_cache'])
def test_uncached_expert_memory_matches_training_and_cached_inference(include_history, cache_miss):
    history = replay(START_FEN, ['b0c2'])
    key = history_key(history)
    row = {'initial_fen': START_FEN, 'moves': ['b0c2'], 'fen': history[-1], 'feature_key': key}
    if include_history:
        row['history'] = history
    first = torch.tensor([1.004, -1.004, 0.502, 2.008]).repeat(90 * 512 // 4).reshape(1, 90, 512)
    features = [first, first * 0.5]
    cache = {'features': [torch.cat([torch.zeros_like(f.half()), f.half()]) for f in features]}
    indices = {'unrelated-history': 0, key: 1}
    expected = select_features(cache, indices, [row], 'cpu')
    assert any(not torch.equal(f.bfloat16(), e) for f, e in zip(features, expected, strict=True))

    class Expert:
        calls = 0

        def __call__(self, planes, *, depths):
            self.calls += 1
            assert depths == [2, 7]
            assert torch.equal(planes, encode_history(history).unsqueeze(0))
            return features, torch.zeros((1, 3))

    predictor = Predictor.__new__(Predictor)
    predictor.device = 'cpu'
    predictor.config = {'expert_feature_depths': [2, 7]}
    predictor.cache = cache
    predictor.indices = indices
    predictor.expert = Expert()
    cached = predictor.features(row)
    assert predictor.expert.calls == 0
    assert all(torch.equal(f, e) for f, e in zip(cached, expected, strict=True))
    predictor.indices = {}
    if cache_miss == 'no_cache':
        predictor.cache = None
    fresh = predictor.features(row)
    assert predictor.expert.calls == 1
    assert len(fresh) == len(expected)
    assert all(f.dtype == torch.bfloat16 and torch.equal(f, e)
               for f, e in zip(fresh, expected, strict=True))
