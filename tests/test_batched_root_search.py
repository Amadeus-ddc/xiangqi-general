"""Root-state isolation, native recursive retries and bounded decoder dispatch."""
from contextlib import contextmanager
import threading

import pytest
import torch

from test_stochastic_search_generation import POLICY, neural_predictor, reference_tokens, row
from xqgeneral.batched_search import batched_roots
from xqgeneral.sampling import generation_seed
from xqgeneral.search_distillation import SearchMiner


OPTIONS = {k: v for k, v in POLICY.items() if k != 'max_new_tokens'}


def test_real_cpu_decoder_batches_requests_on_the_calling_thread(neural_predictor):
    owner = threading.get_ident()
    calls, closed = [], []
    original = neural_predictor.generate_batch
    def observed(records, **kwargs):
        assert threading.get_ident() == owner
        calls.append(len(records))
        return original(records, **kwargs)
    neural_predictor.generate_batch = observed
    barrier = threading.Barrier(4)
    @contextmanager
    def worker(predictor):
        def mine(record):
            barrier.wait(timeout=5)
            return [predictor.generate(record, record['question'], 8, sampling=OPTIONS,
                     seed=generation_seed(17, record, i)) for i in range(2)]
        try: yield mine
        finally: closed.append(threading.get_ident())
    records = [row(str(i)) for i in range(4)]
    before = torch.get_rng_state().clone()
    stats = {}
    answers = list(batched_roots(records, neural_predictor, worker, POLICY,
                   batch_size=4, workers=4, wait_ms=30, statistics=stats))
    assert [r['id'] for r, _ in answers] == [r['id'] for r in records]
    for record, values in answers:
        assert values == [reference_tokens(neural_predictor.model.base,
                         generation_seed(17, record, i)) for i in range(2)]
    assert calls == [4, 4]
    assert len(closed) == len(set(closed)) == 4 and owner not in closed
    assert stats['generated_requests'] == 8 and stats['decoder_batches'] == 2
    assert torch.equal(before, torch.get_rng_state())


def test_native_miner_retries_keep_each_original_root_ordinal_on_resume(neural_predictor):
    class Oracle:
        def analyze(self, *args, **kwargs):
            raise AssertionError('Malformed raw model output must not be repaired')
    @contextmanager
    def worker(predictor):
        miner = SearchMiner(predictor, Oracle(), set(), generation_policy=POLICY,
                            generation_seed=41, max_root_retries=1)
        yield miner.mine
    records = [row(str(i)) for i in range(7)]
    def run(values):
        return {r['id']: value for r, value in batched_roots(values,
            neural_predictor, worker, POLICY, batch_size=3, workers=3, wait_ms=20)}
    full = run(records)
    assert full == {**run(records[:2]), **run(records[2:])}
    assert full == run(list(reversed(records)))
    for record in records:
        query, trace, reason = full[record['id']]
        assert query is None and reason == 'invalid_root_evaluation'
        attempts = trace[0]['root_generation_attempts']
        assert len(attempts) == 2
        for i, attempt in enumerate(attempts):
            assert attempt['generation']['call_index'] == i
            assert attempt['generation']['request_sampling_profile'] == 'independent_request_multinomial_v1'
            assert attempt['generation']['seed'] == str(generation_seed(41, record, i))
            assert attempt['raw'] == reference_tokens(neural_predictor.model.base,
                                    generation_seed(41, record, i))


def test_decoder_failure_unblocks_all_workers_and_closes_their_contexts(neural_predictor):
    closed = []
    def fail(*args, **kwargs): raise ValueError('controlled decoder failure')
    neural_predictor.generate_batch = fail
    barrier = threading.Barrier(3)
    @contextmanager
    def worker(predictor):
        def mine(record):
            barrier.wait(timeout=5)
            return predictor.generate(record, record['question'], 8, sampling=OPTIONS, seed=7)
        try: yield mine
        finally: closed.append(threading.get_ident())
    with pytest.raises(ValueError, match='controlled decoder failure'):
        list(batched_roots([row(str(i)) for i in range(3)], neural_predictor,
             worker, POLICY, batch_size=3, workers=3, wait_ms=20))
    assert len(closed) == 3
    assert not any(t.name.startswith('xq-root-batch-') for t in threading.enumerate())


def test_worker_failure_cancels_other_roots_waiting_for_neural_generation(neural_predictor):
    closed = []
    barrier = threading.Barrier(3)
    @contextmanager
    def worker(predictor):
        def mine(record):
            barrier.wait(timeout=5)
            if record['id'] == '1': raise RuntimeError('controlled root failure')
            return predictor.generate(record, record['question'], 8, sampling=OPTIONS, seed=7)
        try: yield mine
        finally: closed.append(threading.get_ident())
    with pytest.raises(RuntimeError, match='controlled root failure'):
        list(batched_roots([row(str(i)) for i in range(3)], neural_predictor,
             worker, POLICY, batch_size=3, workers=3, wait_ms=20))
    assert len(closed) == 3
    assert not any(t.name.startswith('xq-root-batch-') for t in threading.enumerate())


def test_empty_roots_open_no_worker_or_decoder(neural_predictor):
    def forbidden(*args, **kwargs): raise AssertionError('Empty work must stay empty')
    neural_predictor.generate_batch = forbidden
    assert list(batched_roots([], neural_predictor, forbidden, POLICY,
                             batch_size=4, workers=4)) == []


@pytest.mark.parametrize('options', [{'batch_size': 0}, {'batch_size': True},
    {'workers': 0}, {'workers': True}, {'batch_size': 5, 'workers': 4},
    {'wait_ms': -1}, {'wait_ms': float('nan')}, {'wait_ms': True}])
def test_invalid_parallelism_is_rejected_before_worker_start(neural_predictor, options):
    def forbidden(*args, **kwargs): raise AssertionError('Invalid work must not open workers')
    values = dict(batch_size=4, workers=4, wait_ms=10)
    values.update(options)
    with pytest.raises(ValueError):
        list(batched_roots([row()], neural_predictor, forbidden, POLICY, **values))


def test_early_consumer_close_releases_requests_and_private_workers(neural_predictor):
    closed = []
    @contextmanager
    def worker(predictor):
        def mine(record):
            return predictor.generate(record, record['question'], 8, sampling=OPTIONS, seed=7)
        try: yield mine
        finally: closed.append(threading.get_ident())
    stats = {}
    values = batched_roots([row(str(i)) for i in range(12)], neural_predictor,
        worker, POLICY, batch_size=2, workers=2, wait_ms=10, statistics=stats)
    next(values)
    values.close()
    assert len(closed) == 2 and stats['completed'] is False
    assert not any(t.name.startswith('xq-root-batch-') for t in threading.enumerate())


def test_context_close_failure_is_not_reported_as_completed(neural_predictor):
    @contextmanager
    def worker(predictor):
        yield lambda record: 'root completed'
        raise ValueError('controlled close failure')
    stats = {}
    with pytest.raises(ValueError, match='controlled close failure'):
        list(batched_roots([row()], neural_predictor, worker, POLICY,
             batch_size=1, workers=1, statistics=stats))
    assert stats['completed'] is False


@pytest.mark.parametrize('answers', [[], ['unpaired'], [9, 10]])
def test_unpaired_or_nontext_decoder_output_is_rejected(neural_predictor, answers):
    barrier = threading.Barrier(2)
    @contextmanager
    def worker(predictor):
        def mine(record):
            barrier.wait(timeout=5)
            return predictor.generate(record, record['question'], 8, sampling=OPTIONS, seed=7)
        yield mine
    neural_predictor.generate_batch = lambda *args, **kwargs: answers
    with pytest.raises(RuntimeError, match='unpaired or nontext'):
        list(batched_roots([row('one'), row('two')], neural_predictor, worker,
             POLICY, batch_size=2, workers=2, wait_ms=30))


def test_raw_batch_ledger_failure_stops_before_workers_receive_outputs(neural_predictor):
    completed, closed = [], []
    barrier = threading.Barrier(2)
    @contextmanager
    def worker(predictor):
        def mine(record):
            barrier.wait(timeout=5)
            value = predictor.generate(record, record['question'], 8, sampling=OPTIONS, seed=7)
            completed.append(record['id'])
            return value
        try: yield mine
        finally: closed.append(threading.get_ident())
    def broken_ledger(records, seeds, answers):
        assert len(records) == len(seeds) == len(answers) == 2
        assert not completed
        raise OSError('controlled ledger write failure')
    with pytest.raises(OSError, match='controlled ledger write failure'):
        list(batched_roots([row('one'), row('two')], neural_predictor, worker,
             POLICY, batch_size=2, workers=2, wait_ms=30, record_batch=broken_ledger))
    assert not completed and len(closed) == 2
