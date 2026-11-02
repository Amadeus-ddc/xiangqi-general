import json
import sys

import pytest

from test_paper_curriculum import source_fixture
from xqgeneral import paper_curriculum
from xqgeneral.evidence import digest


def test_native_split_workers_preserve_corpus_and_readback(tmp_path, monkeypatch):
    roots, proof, _ = source_fixture(tmp_path)
    original = paper_curriculum.verify_splits
    calls = []

    def observe(rows, workers=1):
        calls.append(workers)
        return original(rows, workers=workers)

    monkeypatch.setattr(paper_curriculum, 'verify_splits', observe)
    common = dict(roots=[roots], source_manifests=[proof],
                  reference_data=[roots.parent / 'reference'], seed=7)
    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    first = paper_curriculum.build(output=serial, **common)
    second = paper_curriculum.build(output=parallel, split_workers=2, **common)
    assert first == second
    assert 'split_workers' not in json.loads((serial / 'manifest.json').read_text())['config']
    assert json.loads((parallel / 'manifest.json').read_text())['config']['split_workers'] == 2
    for split in ('train', 'validation', 'test'):
        assert (serial / f'{split}.jsonl').read_bytes() == (parallel / f'{split}.jsonl').read_bytes()
    before = {str(p): digest(p) for p in parallel.iterdir() if p.is_file()}

    saved = paper_curriculum.readback(parallel, tmp_path / 'saved')
    override = paper_curriculum.readback(parallel, tmp_path / 'override', split_workers=1)
    legacy = paper_curriculum.readback(serial, tmp_path / 'legacy', split_workers=2)
    assert saved == override == legacy
    assert saved['questions_regenerated_and_compared'] == 300
    assert calls == [1, 2, 2, 1, 2]
    assert {str(p): digest(p) for p in parallel.iterdir() if p.is_file()} == before
    for name, workers in [('saved', 2), ('override', 1), ('legacy', 2)]:
        assert json.loads((tmp_path / name / 'manifest.json').read_text())['config']['split_workers'] == workers


@pytest.mark.parametrize('workers', [0, -1, True, 1.5])
@pytest.mark.parametrize('operation', ['build', 'readback'])
def test_invalid_split_workers_rejected_before_io(tmp_path, workers, operation):
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='positive integer'):
        if operation == 'build':
            paper_curriculum.build([], [], output, split_workers=workers)
        else:
            paper_curriculum.readback(tmp_path / 'missing-input', output, split_workers=workers)
    assert not output.exists()


def test_readback_override_does_not_hide_invalid_saved_workers(tmp_path):
    roots, proof, _ = source_fixture(tmp_path)
    data = tmp_path / 'data'
    paper_curriculum.build([roots], [proof], data, reference_data=[roots.parent / 'reference'])
    path = data / 'manifest.json'
    completed = json.loads(path.read_text())
    completed['config']['split_workers'] = 0
    path.write_text(json.dumps(completed))
    output = tmp_path / 'reject'
    with pytest.raises(ValueError, match='positive integer'):
        paper_curriculum.readback(data, output, split_workers=2)
    assert not output.exists()


def test_cli_accepts_split_workers_for_build_and_readback(tmp_path, monkeypatch, capsys):
    roots, proof, _ = source_fixture(tmp_path)
    data, output = tmp_path / 'data', tmp_path / 'readback'
    monkeypatch.setattr(sys, 'argv', ['paper_curriculum', '--roots', str(roots),
        '--source-manifests', str(proof), '--reference-data', str(roots.parent / 'reference'),
        '--output', str(data), '--split-workers', '2'])
    paper_curriculum.main()
    assert json.loads((data / 'manifest.json').read_text())['config']['split_workers'] == 2
    monkeypatch.setattr(sys, 'argv', ['paper_curriculum', '--readback', str(data),
        '--output', str(output), '--split-workers', '1'])
    paper_curriculum.main()
    assert json.loads((output / 'manifest.json').read_text())['config']['split_workers'] == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])['status'] == 'complete'

