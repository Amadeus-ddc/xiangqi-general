"""Bounded independent root workers with one caller-owned neural decoder."""
from concurrent.futures import Future
import math
import queue
import threading
import time

from .sampling import REQUEST_SAMPLING_PROFILE, checked_seed, search_generation_policy


ROOT_BATCH_PROFILE = 'bounded_independent_root_workers_v1'


def checked_batching(batch_size, workers, wait_ms):
    if (type(batch_size) is not int or batch_size <= 0 or
            type(workers) is not int or workers <= 0 or batch_size > workers or
            type(wait_ms) not in (int, float) or not math.isfinite(wait_ms) or wait_ms < 0):
        raise ValueError('Root batching needs positive workers/batch size and a finite nonnegative wait')
    return {'profile': ROOT_BATCH_PROFILE, 'request_sampling_profile': REQUEST_SAMPLING_PROFILE,
            'batch_size': batch_size, 'workers': workers, 'wait_ms': float(wait_ms),
            'maximum_pending_roots': 2 * workers}


class _Requests:
    """Workers may enqueue only; the owning caller executes all model calls."""
    request_sampling_profile = REQUEST_SAMPLING_PROFILE

    def __init__(self, policy, workers):
        self.policy = search_generation_policy(policy)
        self.sampling = {k: v for k, v in self.policy.items() if k != 'max_new_tokens'}
        self.queue = queue.Queue(maxsize=workers)
        self.lock, self.pending, self.error = threading.Lock(), set(), None

    def generate(self, record, question=None, max_new_tokens=768, *, sampling=None, seed=None):
        if (sampling != self.sampling or max_new_tokens != self.policy['max_new_tokens']):
            raise ValueError('Every batched request must retain the declared sampling and token budget')
        checked_seed(seed)
        future = Future()
        with self.lock:
            if self.error is not None:
                raise self.error
            self.pending.add(future)
        item = (dict(record, question=question or record['question']), seed, future)
        try:
            while not future.done():
                try:
                    self.queue.put(item, timeout=.05)
                    break
                except queue.Full:
                    pass
            return future.result()
        finally:
            with self.lock:
                self.pending.discard(future)

    def take(self, batch_size, wait_ms):
        wait = wait_ms / 1000
        try:
            first = self.queue.get(timeout=max(.001, wait))
        except queue.Empty:
            return []
        result = [first] if not first[2].done() else []
        deadline = time.monotonic() + wait
        while len(result) < batch_size:
            if time.monotonic() >= deadline:
                break
            try:
                item = self.queue.get(timeout=max(.001, deadline - time.monotonic()))
            except queue.Empty:
                break
            if not item[2].done():
                result.append(item)
            if time.monotonic() >= deadline:
                break
        return result

    def abort(self, error):
        with self.lock:
            if self.error is None:
                self.error = error
            for future in self.pending:
                if not future.done():
                    future.set_exception(self.error)


def batched_roots(records, predictor, worker_context, generation_policy, *,
                  batch_size=4, workers=4, wait_ms=10, statistics=None, record_batch=None):
    """Yield ordered ``(record, result)`` pairs without sharing mutable miners.

    ``worker_context(request_predictor)`` opens one private worker context and
    yields its root function. Its enter, root calls and exit stay on that worker
    thread. Only this generator's caller invokes ``predictor.generate_batch``.
    At most twice the worker count of roots may be pending, including buffered
    out-of-order results. Callers that stop early must close the generator.
    """
    profile = checked_batching(batch_size, workers, wait_ms)
    requests = _Requests(generation_policy, workers)
    iterator = iter(records)
    stop, errors = threading.Event(), queue.Queue()
    tasks = queue.Queue(maxsize=profile['maximum_pending_roots'])
    pending, next_input, next_output, exhausted = {}, 0, 0, False
    counters = {'decoder_batches': 0, 'generated_requests': 0, 'batch_sizes': {}}
    complete = False

    def fill():
        nonlocal next_input, exhausted
        while not exhausted and len(pending) < profile['maximum_pending_roots']:
            try:
                record = next(iterator)
            except StopIteration:
                exhausted = True
                break
            future = Future()
            pending[next_input] = (record, future)
            tasks.put_nowait((record, future))
            next_input += 1

    def work():
        try:
            with worker_context(requests) as mine:
                while not stop.is_set():
                    try:
                        record, future = tasks.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if not future.set_running_or_notify_cancel():
                        continue
                    try:
                        future.set_result(mine(record))
                    except BaseException as error:
                        future.set_exception(error)
                        errors.put(error)
                        return
        except BaseException as error:
            errors.put(error)

    threads = []
    try:
        fill()
        if not pending:
            complete = True
            return
        for index in range(workers):
            thread = threading.Thread(target=work, name=f'xq-root-batch-{index}', daemon=False)
            thread.start()
            threads.append(thread)
        while pending:
            if not errors.empty():
                raise errors.get_nowait()
            while next_output in pending and pending[next_output][1].done():
                record, future = pending.pop(next_output)
                value = future.result()
                next_output += 1
                yield record, value
                fill()
            if not pending:
                break
            batch = requests.take(batch_size, wait_ms)
            if not batch:
                continue
            answers = predictor.generate_batch([item[0] for item in batch],
                max_new_tokens=requests.policy['max_new_tokens'], sampling=requests.sampling,
                seeds=[item[1] for item in batch])
            if (not isinstance(answers, list) or len(answers) != len(batch) or
                    any(not isinstance(answer, str) for answer in answers)):
                raise RuntimeError('Batched decoder returned unpaired or nontext answers')
            if record_batch is not None:
                record_batch([item[0] for item in batch], [item[1] for item in batch], answers)
            counters['decoder_batches'] += 1
            counters['generated_requests'] += len(batch)
            counters['batch_sizes'][str(len(batch))] = counters['batch_sizes'].get(str(len(batch)), 0) + 1
            for item, answer in zip(batch, answers):
                item[2].set_result(answer)
        complete = True
    finally:
        stop.set()
        requests.abort(RuntimeError('Root batch runner closed'))
        for _, future in pending.values():
            future.cancel()
        for thread in threads:
            thread.join()
        if statistics is not None:
            statistics.update(profile, **counters, completed=complete and errors.empty())
    if not errors.empty():
        raise errors.get_nowait()
