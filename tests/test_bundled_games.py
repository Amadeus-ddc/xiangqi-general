import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from xqgeneral.bundled_games import import_bundles, native_record, pgn_records
from xqgeneral.evidence import atomic_json, digest, manifest
from xqgeneral.human_games import assigned_split, parse_game


def pgn(moves='1. H2-E2 H9-G7', red='红方', result='1-0'):
    return (f'[Game "Chinese Chess"]\n[Event "Cup {{literal}}"]\n[Date "2019-01-01"]\n'
            f'[Red "{red}"]\n[Black "黑方"]\n[Result "{result}"]\n\n{moves} {result}\n\n').encode()


def acquisition(tmp_path, records, extra_member=None):
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('nested/games.pgns', b''.join(records))
        if extra_member:
            z.writestr(extra_member, 'unexecuted archived content')
    raw = archive.read_bytes()
    item = {'path': str(archive), 'bytes': len(raw), 'sha256': digest(archive),
            'git_blob_sha1': hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()}
    source = {'repository': 'source/collection', 'revision': 'a' * 40,
              'upstream_collection': 'https://github.com/source/collection'}
    path = tmp_path / 'acquisition.json'
    atomic_json(path, manifest('test_pinned_acquisition', source, [archive], [], {'actual_downloads': [item]}))
    return path


def previous(tmp_path, raw):
    root = tmp_path / 'previous'
    root.mkdir()
    game = parse_game(raw)
    game.update(split=assigned_split(game['game_id'], 20261051), source={'repository': 'old/source'})
    path = root / 'games.jsonl'
    path.write_text(json.dumps(game) + '\n')
    atomic_json(root / 'manifest.json', manifest('test_previous_native_import', {'seed': 20261051}, [], [path],
        {'all_retained_moves_native_legal': True, 'full_history_terminal_checks_passed': True,
         'game_split_assigned_before_questions': True}))
    return root, game


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_streaming_preserves_bom_crlf_bytes_and_record_boundaries():
    first = b'\xef\xbb\xbf' + pgn().replace(b'\n', b'\r\n')
    second = pgn('1. H2-C2')
    assert list(pgn_records(io.BytesIO(first + second))) == [first, second]
    assert parse_game(first)['moves'] == ['h2e2', 'h9g7']


def test_game_headers_in_comments_or_variations_do_not_become_new_records():
    commented = pgn().replace(b'1. H2-E2 H9-G7',
        b'1. H2-E2 {\n[Game "Chinese Chess"]\n} H9-G7 ; [Game "Chinese Chess"]')
    varied = pgn().replace(b'1. H2-E2 H9-G7',
        b'1. H2-E2 (\n[Game "Chinese Chess"]\n) H9-G7')
    actual = list(pgn_records(io.BytesIO(commented + varied + pgn())))
    assert actual == [commented, varied, pgn()]


@pytest.mark.parametrize('raw, limit', [(b'orphan movetext\n' + pgn(), 10000), (pgn(), 10)])
def test_structural_or_byte_limit_failures_are_not_silently_skipped(raw, limit):
    with pytest.raises(ValueError):
        list(pgn_records(io.BytesIO(raw), limit))


def test_native_parser_quarantines_terminal_followup_without_repair():
    raw = pgn('1. B0-C2 B9-C7 2. C2-B0 C7-B9 3. B0-C2 B9-C7 '
              '4. C2-B0 C7-B9 5. B0-C2 B9-C7 6. C2-B0 C7-B9 7. H0-G2', result='1/2-1/2')
    game, rejected = native_record((raw, {'record_ordinal': 7}))
    assert game is None and rejected['record_ordinal'] == 7
    assert 'full-history terminal' in rejected['error']


def test_import_deduplicates_canonical_games_and_preserves_source_and_attribution(tmp_path):
    prior, old = previous(tmp_path, pgn(red='旧姓名'))
    known = pgn('1. 炮二平五 馬８進７', red='新姓名')
    fresh = pgn('1. H2-C2', red='独立姓名')
    acquired = acquisition(tmp_path, [known, fresh, fresh, pgn('1. H2-G3')])
    output = tmp_path / 'import'
    summary = import_bundles(acquired, output, [prior], workers=1)
    assert summary['candidate_records_natively_examined'] == 4
    assert summary['new_unique_recorded_games'] == 1 and summary['combined_unique_recorded_games'] == 2
    assert summary['duplicate_records'] == 2 and summary['quarantined_records'] == 1
    assert summary['duplicate_records_with_differing_attribution'] == 1
    game = rows(output / 'games.jsonl')[0]
    assert game['source']['repository'] == 'source/collection'
    assert game['source']['declared_license'] == 'unspecified'
    assert game['source']['record_ordinal'] == 2
    assert game['source']['archive_member'] == 'nested/games.pgns'
    assert game['source']['content_sha256'] == hashlib.sha256(fresh).hexdigest()
    assert game['source_kind'] == 'published_recorded_match'
    assert game['split'] == assigned_split(game['game_id'], 20261051)
    difference = rows(output / 'attribution-differences.jsonl')[0]
    assert difference['game_id'] == old['game_id']
    assert difference['first']['headers']['Red'] == '旧姓名'
    assert difference['duplicate_headers']['Red'] == '新姓名'
    assert difference['attributions_authenticated'] is False


def test_serial_and_bounded_parallel_imports_have_identical_record_outputs(tmp_path):
    acquired = acquisition(tmp_path, [pgn(), pgn('1. H2-C2'), pgn(), pgn('1. H2-G3')])
    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    a = import_bundles(acquired, serial, workers=1)
    b = import_bundles(acquired, parallel, workers=2)
    assert {k: v for k, v in a.items() if k not in ['archives_processed']} == {
        k: v for k, v in b.items() if k not in ['archives_processed']}
    for name in ['games', 'quarantine', 'attribution-differences']:
        assert (serial / (name + '.jsonl')).read_bytes() == (parallel / (name + '.jsonl')).read_bytes()
    # Duplicate provenance names the current output collection; all source facts match.
    x, y = rows(serial / 'duplicates.jsonl'), rows(parallel / 'duplicates.jsonl')
    x[0]['first'].pop('existing_collection'); y[0]['first'].pop('existing_collection')
    assert x == y


def test_changed_archive_and_previous_split_contract_fail_before_output_creation(tmp_path):
    acquired = acquisition(tmp_path, [pgn()])
    archive = tmp_path / 'source.zip'
    archive.write_bytes(archive.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed'):
        import_bundles(acquired, tmp_path / 'out', workers=1)
    assert not (tmp_path / 'out').exists()
    acquired = acquisition(tmp_path, [pgn()])
    prior, _ = previous(tmp_path, pgn())
    with pytest.raises(ValueError, match='split'):
        import_bundles(acquired, tmp_path / 'other', [prior], workers=1, seed=1)
    assert not (tmp_path / 'other').exists()


def test_archive_code_members_are_rejected_without_extracting_or_executing(tmp_path):
    acquired = acquisition(tmp_path, [pgn()], extra_member='../untrusted.py')
    with pytest.raises(ValueError, match='Only PGN data'):
        import_bundles(acquired, tmp_path / 'output', workers=1)
    assert not (tmp_path / 'untrusted.py').exists()


def test_optional_fixture_budget_does_not_claim_full_import_or_overwrite(tmp_path):
    acquired = acquisition(tmp_path, [pgn(), pgn('1. H2-C2')])
    output = tmp_path / 'limited'
    summary = import_bundles(acquired, output, workers=1, limit_records=1)
    assert summary['record_budget_limited'] is True and summary['candidate_records_natively_examined'] == 1
    with pytest.raises(FileExistsError):
        import_bundles(acquired, output, workers=1)
