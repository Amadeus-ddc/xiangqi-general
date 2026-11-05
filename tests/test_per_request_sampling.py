"""Independent request seeds through the real CPU decoder generation interface."""
import pytest
import torch

from test_stochastic_search_generation import POLICY, neural_predictor, reference_tokens, row


OPTIONS = {key: value for key, value in POLICY.items() if key != 'max_new_tokens'}


def test_batch_matches_independent_neural_request_decoding(neural_predictor):
    seeds = [0, 19, 73]
    before = torch.get_rng_state().clone()
    answers = neural_predictor.generate_batch([row(str(i)) for i in range(3)],
        max_new_tokens=8, sampling=OPTIONS, seeds=seeds)
    assert answers == [reference_tokens(neural_predictor.model.base, seed) for seed in seeds]
    assert len(set(answers)) > 1
    assert torch.equal(before, torch.get_rng_state())


def test_reordering_and_resuming_other_requests_keeps_own_stream(neural_predictor):
    requests = [('one', 35), ('two', 72), ('three', 129)]

    def run(items):
        values = neural_predictor.generate_batch([row(identity) for identity, _ in items],
            max_new_tokens=8, sampling=OPTIONS, seeds=[seed for _, seed in items])
        return dict(zip([identity for identity, _ in items], values))

    full = run(requests)
    assert full == run(list(reversed(requests)))
    assert full == {**run(requests[:1]), **run(requests[1:])}
    assert full == run(requests)


@pytest.mark.parametrize('seeds', [[1], [1, 2, 3], [1, True], [1, -1], [1, 2**63],
                                  [1, 3.5], '12'])
def test_invalid_request_seeds_fail_before_features(neural_predictor, seeds):
    def unexpected(*args, **kwargs):
        raise AssertionError('Invalid requests must fail before model inputs')

    neural_predictor.features = unexpected
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        neural_predictor.generate_batch([row('one'), row('two')], sampling=OPTIONS, seeds=seeds)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('extra', [{'seed': 7}, {'sampling': None}, {'_beams': 2},
                                  {'_move_constraints': [object(), object()]}])
def test_unsupported_request_seed_combinations_fail_before_features(neural_predictor, extra):
    def unexpected(*args, **kwargs):
        raise AssertionError('Unsupported requests must fail before model inputs')

    neural_predictor.features = unexpected
    options = dict(sampling=OPTIONS, seeds=[10, 11], **{})
    options.update(extra)
    with pytest.raises(ValueError):
        neural_predictor.generate_batch([row('one'), row('two')], **options)


def test_empty_seeded_batch_is_empty_without_advancing_rng(neural_predictor):
    before = torch.get_rng_state().clone()
    assert neural_predictor.generate_batch([], sampling=OPTIONS, seeds=[]) == []
    assert torch.equal(before, torch.get_rng_state())
