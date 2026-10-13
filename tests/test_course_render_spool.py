import json
from pathlib import Path

import pytest

from test_course_rendering import options
from test_native_catalog_courses import prepared
from test_paper_curriculum import source_fixture
from xqgeneral.course_render_spool import render_grouped_positions
from xqgeneral.course_sampling import SAMPLING_PROFILE, make_sampler
from xqgeneral.evidence import digest
from xqgeneral.paper_curriculum import build, readback


def test_grouped_native_workers_keep_all_queries_answers_order_and_complete_readback(tmp_path, monkeypatch):
    import xqgeneral.paper_curriculum as module
    db, roots, proof, catalog_proof, reference = prepared(tmp_path)
    source_before = digest(db), digest(roots), digest(proof)
    original, grouped = tmp_path / 'original', tmp_path / 'grouped'
    settings = dict(reference_data=[reference], catalog_proof=catalog_proof,
                    question_workers=2, question_batch_size=2, **options())
    expected = build([roots], [proof], original, **settings)
    progress, spools = [], []

    def renderer(selections, seed, sampler, workers, batch_size):
        scratch = tmp_path / f'spool-{len(spools)}'
        spools.append(scratch)
        return render_grouped_positions(selections, seed, sampler, scratch,
                                        workers=2, max_job_groups=3, progress=progress.append)

    monkeypatch.setattr(module, 'render_positions', renderer)
    actual = build([roots], [proof], grouped, **settings)
    assert actual == expected
    for split in ('train', 'validation', 'test'):
        assert (grouped / f'{split}.jsonl').read_bytes() == (original / f'{split}.jsonl').read_bytes()
    verification = readback(grouped, tmp_path / 'readback')
    assert verification['questions_regenerated_and_compared'] == expected['questions']
    assert verification['all_source_and_question_bytes_checked'] is True
    assert source_before == (digest(db), digest(roots), digest(proof))
    assert len(spools) == 2 and all((p / 'drafts.sqlite').exists() for p in spools)
    assert progress[-1]['phase'] == 'complete'
    assert progress[-1]['questions'] == expected['questions']


@pytest.mark.parametrize('workers', [1, 2])
def test_grouped_native_checks_reject_forged_history_before_emitting_any_answers(tmp_path, workers):
    _, _, originals = source_fixture(tmp_path)
    root = dict(originals[0], feature_key='forged-unvalidated-history')
    generator = render_grouped_positions([(root, 'static_current', 'piece')], 37,
        make_sampler(SAMPLING_PROFILE), tmp_path / 'spool', workers=workers)
    with pytest.raises(ValueError, match='native full-history'):
        next(generator)


def test_grouped_native_checks_reject_illegal_future_even_for_current_questions(tmp_path):
    _, _, originals = source_fixture(tmp_path)
    root = dict(originals[0], future_moves=['a0a0'])
    with pytest.raises(ValueError, match='Illegal move'):
        list(render_grouped_positions([(root, 'static_current', 'piece')], 37,
            make_sampler(SAMPLING_PROFILE), tmp_path / 'spool', workers=2))


def test_existing_spool_is_preserved_before_source_is_consumed(tmp_path):
    scratch = tmp_path / 'spool'
    scratch.mkdir()
    marker = scratch / 'preserved.txt'
    marker.write_text('existing work')

    def selections():
        raise AssertionError('Existing spool must be rejected before reading source inputs')
        yield

    with pytest.raises(FileExistsError):
        list(render_grouped_positions(selections(), 37, make_sampler(SAMPLING_PROFILE), scratch))
    assert marker.read_text() == 'existing work'


@pytest.mark.parametrize('field,value', [('workers', 0), ('workers', True), ('workers', 1.5),
    ('max_job_groups', 0), ('max_job_groups', False), ('max_job_groups', 1.5)])
def test_invalid_spool_settings_fail_before_outputs(tmp_path, field, value):
    scratch = tmp_path / 'spool'
    with pytest.raises(ValueError, match='Positive native spool'):
        list(render_grouped_positions([], 37, make_sampler(SAMPLING_PROFILE), scratch, **{field: value}))
    assert not scratch.exists()
