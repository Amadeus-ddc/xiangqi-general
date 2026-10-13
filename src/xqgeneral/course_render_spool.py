"""Disk-backed native answer workers grouped by game, emitted in source order."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import islice
import json
import multiprocessing
from pathlib import Path
import random
import sqlite3

from xqgeneral.course_rendering import _native_record_batch
from xqgeneral.course_tasks import _paper_record_queries


def render_grouped_positions(selections, seed, sampler, scratch, *, workers=1,
                             max_job_groups=256, progress=None):
    """Keep adaptive queries sequential and run the original native checks by game.

    No answer or history check is skipped. Grouping related histories gives the
    existing pure rule caches a chance to reuse common prefixes. Drafts and
    completed answers live in separate SQLite files; only completed native
    results are emitted, in exactly the original selection order.
    """
    if (type(workers) is not int or workers < 1 or
            type(max_job_groups) is not int or max_job_groups < 1):
        raise ValueError('Positive native spool workers and job group limit required')
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=False)
    drafts, answers = None, None

    def report(phase, **values):
        if progress is not None:
            progress({'phase': phase, **values})

    try:
        drafts = sqlite3.connect(scratch / 'drafts.sqlite')
        answers = sqlite3.connect(scratch / 'answers.sqlite')
        for connection in (drafts, answers):
            connection.execute('PRAGMA journal_mode=OFF')
            connection.execute('PRAGMA synchronous=OFF')
        drafts.execute('CREATE TABLE drafts (ordinal INTEGER PRIMARY KEY, game TEXT NOT NULL, '
                       'plies INTEGER NOT NULL, payload TEXT NOT NULL)')
        answers.execute('CREATE TABLE answers (ordinal INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
        groups, questions = 0, 0
        for root, stage, task in selections:
            game = root.get('game_id')
            if not isinstance(game, str) or not game:
                raise ValueError('Native game grouping requires the source game identity')
            rng = random.Random(int(hashlib.sha256(
                f'{seed}/{stage}/{task}/{root["feature_key"]}'.encode()).hexdigest(), 16))
            rows = _paper_record_queries(root, rng, sampler, only_stage=stage, only_task=task)
            sampler.observe(rows)
            drafts.execute('INSERT INTO drafts VALUES (?, ?, ?, ?)',
                (groups, game, len(root['moves']), json.dumps([root, rows], ensure_ascii=False)))
            groups += 1
            questions += len(rows)
            if groups % 1024 == 0:
                drafts.commit()
                report('drafting_adaptive_queries_in_original_order', groups=groups, questions=questions)
        drafts.commit()
        drafts.execute('CREATE INDEX game_order ON drafts(game, plies DESC, ordinal)')
        drafts.commit()
        report('all_ordered_drafts_complete', groups=groups, questions=questions)

        def jobs():
            for (game,) in drafts.execute('SELECT DISTINCT game FROM drafts ORDER BY game'):
                source = iter(drafts.execute('SELECT ordinal, payload FROM drafts WHERE game=? '
                                            'ORDER BY plies DESC, ordinal', (game,)))
                while batch := list(islice(source, max_job_groups)):
                    yield [row[0] for row in batch], [json.loads(row[1]) for row in batch]

        completed_groups, completed_questions = 0, 0

        def collect(ordinals, results):
            nonlocal completed_groups, completed_questions
            if len(results) != len(ordinals):
                raise ValueError('Native worker omitted a selected source group')
            answers.executemany('INSERT INTO answers VALUES (?, ?)',
                ((ordinal, json.dumps(result[1], ensure_ascii=False))
                 for ordinal, result in zip(ordinals, results, strict=True)))
            completed_groups += len(results)
            completed_questions += sum(len(result[1]) for result in results)
            answers.commit()
            report('checking_every_native_history_and_answer_by_game',
                   groups=completed_groups, questions=completed_questions,
                   expected_groups=groups, expected_questions=questions)

        if workers == 1:
            for ordinals, batch in jobs():
                collect(ordinals, _native_record_batch(batch))
        else:
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                pending = deque()
                for ordinals, batch in jobs():
                    pending.append((ordinals, pool.submit(_native_record_batch, batch)))
                    if len(pending) >= workers * 2:
                        first, future = pending.popleft()
                        collect(first, future.result())
                for ordinals, future in pending:
                    collect(ordinals, future.result())
        if completed_groups != groups or completed_questions != questions:
            raise ValueError('Native spool did not check every selected history and question')
        completed = iter(answers.execute('SELECT ordinal, payload FROM answers ORDER BY ordinal'))
        emitted_groups, emitted_questions = 0, 0
        for ordinal, payload in drafts.execute('SELECT ordinal, payload FROM drafts ORDER BY ordinal'):
            answer_ordinal, answer_payload = next(completed)
            if answer_ordinal != ordinal:
                raise ValueError('Completed native answer order differs from its source selections')
            root, _ = json.loads(payload)
            rows = json.loads(answer_payload)
            yield root, rows
            emitted_groups += 1
            emitted_questions += len(rows)
            if emitted_groups % 1024 == 0:
                report('emitting_checked_native_answers_in_original_order',
                       groups=emitted_groups, questions=emitted_questions,
                       expected_groups=groups, expected_questions=questions)
        if next(completed, None) is not None or emitted_questions != questions:
            raise ValueError('Native spool contains extra or missing answers')
        report('complete', groups=emitted_groups, questions=emitted_questions)
    finally:
        for connection in (drafts, answers):
            if connection is not None:
                connection.close()
