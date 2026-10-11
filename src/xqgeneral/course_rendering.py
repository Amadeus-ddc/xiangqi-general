"""Ordered native question workers with sequential adaptive query selection."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import islice
import multiprocessing
import random

from .course_tasks import (_complete_paper_answers, _paper_record_queries,
                           paper_records, validate_context)


def checked_question_rendering(workers, batch_size, position_profile, color_mirror):
    if (type(workers) is not int or workers < 1 or
            type(batch_size) is not int or batch_size < 1 or
            workers == 1 and batch_size != 1 or
            workers != 1 and (position_profile is None or color_mirror)):
        raise ValueError('Positive question workers require position sampling without color mirrors; '
                         'question batching requires multiple workers')


def _native_record_batch(groups):
    for root, rows in groups:
        validate_context(root, check_future=True)
        _complete_paper_answers(rows)
    return groups


def render_positions(selections, seed, sampler, workers=1, batch_size=1):
    """Emit complete groups in source order; never emit a draft answer."""
    checked_question_rendering(workers, batch_size, True, False)
    if workers == 1:
        for root, stage, task in selections:
            rng = random.Random(int(hashlib.sha256(
                f'{seed}/{stage}/{task}/{root["feature_key"]}'.encode()).hexdigest(), 16))
            rows = paper_records(root, rng, sampler, only_stage=stage, only_task=task)
            sampler.observe(rows)
            yield root, rows
        return

    def drafts():
        for root, stage, task in selections:
            rng = random.Random(int(hashlib.sha256(
                f'{seed}/{stage}/{task}/{root["feature_key"]}'.encode()).hexdigest(), 16))
            rows = _paper_record_queries(root, rng, sampler, only_stage=stage, only_task=task)
            # Frequencies use board/query identities, never answer text. Updating
            # here preserves each next root's adaptive choices while workers run.
            sampler.observe(rows)
            yield root, rows

    source = iter(drafts())
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = deque()
        while batch := list(islice(source, batch_size)):
            pending.append(pool.submit(_native_record_batch, batch))
            if len(pending) >= workers * 2:
                yield from pending.popleft().result()
        for job in pending:
            yield from job.result()
