"""Legacy import manifests use project-relative output keys; queues pass absolute paths."""
import copy
import json

import pytest

from xqgeneral.evidence import atomic_json, load_jsonl
from xqgeneral.recorded_search_inputs import prepare, prepared_roots
from test_recorded_search_inputs import fixture, identities


def test_absolute_search_sources_accept_original_relative_output_declarations(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    original = tmp_path / 'original-pool'
    before = prepare(**args, output=original, min_ply=1, per_game=3)
    source = args['games'][0]
    manifest = source / 'manifest.json'
    proof = json.loads(manifest.read_text())
    absolute = str(source / 'games.jsonl')
    relative = str((source / 'games.jsonl').relative_to(tmp_path))
    proof['outputs'][relative] = proof['outputs'].pop(absolute)
    atomic_json(manifest, proof)
    saved_manifest = manifest.read_bytes()
    source_bytes = (source / 'games.jsonl').read_bytes()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'relative-declared-pool'
    result = prepare(**args, output=output, min_ply=1, per_game=3)
    assert result == before
    assert (output / 'roots.jsonl').read_bytes() == (original / 'roots.jsonl').read_bytes()
    assert manifest.read_bytes() == saved_manifest and (source / 'games.jsonl').read_bytes() == source_bytes
    rows, _ = prepared_roots(output, args['data'], 2, 20261013, identities(args))
    assert rows == load_jsonl(original / 'roots.jsonl')[:2]


@pytest.mark.parametrize('problem', ['wrong-directory', 'conflicting-alias', 'changed-canonical-file'])
def test_relative_output_resolution_keeps_exact_path_and_byte_checks(tmp_path, monkeypatch, problem):
    args, _ = fixture(tmp_path)
    source = args['games'][0]
    manifest = source / 'manifest.json'
    proof = json.loads(manifest.read_text())
    absolute = str(source / 'games.jsonl')
    relative = str((source / 'games.jsonl').relative_to(tmp_path))
    value = proof['outputs'].pop(absolute)
    proof['outputs'][relative] = value
    if problem == 'wrong-directory':
        proof['outputs']['other/games.jsonl'] = proof['outputs'].pop(relative)
    elif problem == 'conflicting-alias':
        proof['outputs'][absolute] = dict(copy.deepcopy(value), sha256='0' * 64)
    else:
        with (source / 'games.jsonl').open('ab') as handle:
            handle.write(b'changed source bytes')
    atomic_json(manifest, proof)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'must-not-publish'
    with pytest.raises(ValueError, match='canonical import|search source changed'):
        prepare(**args, output=output, min_ply=1, per_game=3)
    assert not (output / 'manifest.json').exists()
