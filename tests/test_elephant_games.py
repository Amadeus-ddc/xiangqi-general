import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile

import pytest

from xqgeneral.elephant_games import COLUMNS, csv_groups, import_csv, native_game
from xqgeneral.evidence import digest


CUTOFF = datetime(2026, 10, 17, tzinfo=timezone.utc)


def game_rows(identity='game1', moves='h2e2 h9g7', red='anon-red'):
    rows = []
    for index, move in enumerate(moves.split()):
        row = dict.fromkeys(COLUMNS, '')
        row.update(timestamp=f'2026-01-01T00:00:{index:02d}Z', move_index=str(index),
                   move=move, game_id=identity, red_player=red, black_player='anon-black',
                   game_status='RESIGNED', outcome='RED_WINS', analysis='untrusted source analysis')
        rows.append(row)
    return rows


def csv_bytes(rows, columns=COLUMNS):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode()


def acquisition(tmp_path, groups, name='capture', members=None):
    source = tmp_path / name
    source.mkdir()
    archive = source / 'pvp_game_moves_xiangqi_2026-06.zip'
    with zipfile.ZipFile(archive, 'w') as packed:
        if members is None:
            members = {'pvp_game_moves_xiangqi_001.csv': csv_bytes([r for g in groups for r in g])}
        for member, raw in members.items():
            packed.writestr(member, raw)
    page = source / 'datasets.html'
    page.write_text('Explicit published anonymous dataset source; GPL-3.0 declaration.')
    url = 'https://cdn.elephantchess.io/data/' + archive.name
    proof = {'status': 'complete_acquisition_and_structural_inventory_only',
             'kind': 'published_anonymous_xiangqi_pvp_export',
             'dataset_page': 'https://elephantchess.io/about/datasets',
             'dataset_page_sha256': digest(page), 'archive_url': url,
             'archive_path': str(archive), 'archive_sha256': digest(archive),
             'archive_bytes': archive.stat().st_size, 'observed_at_utc': CUTOFF.isoformat(),
             'distinct_archive_urls': 1, 'all_listed_zip_links': [url, url]}
    path = source / 'manifest.json'
    path.write_text(json.dumps(proof))
    return path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_native_rows_preserve_zero_based_moves_and_full_history_without_using_analysis():
    value = game_rows()
    parsed = native_game(value, CUTOFF, minimum_plies=2)
    assert parsed['moves'] == ['h2e2', 'h9g7'] and len(parsed['history']) == 3
    assert parsed['source_format'] == 'ElephantchessAnonymousXiangqiCSV'
    assert parsed['declared_nonterminal_result_reason'] == 'RESIGNED'
    assert parsed['real_world_player_identity_or_unassisted_play_authenticated'] is False
    assert parsed['csv_analysis_or_cpl_used_for_labels'] is False
    assert 'analysis' not in parsed['original_csv_game_metadata']
    with pytest.raises(ValueError, match='minimum-ply'):
        native_game(value, CUTOFF)


@pytest.mark.parametrize('change', [
    lambda rows: rows[0].update(move_index='1'),
    lambda rows: rows[1].update(move_index='0'),
    lambda rows: rows[1].update(move_index='+1'),
    lambda rows: rows[1].update(game_id='other'),
    lambda rows: rows[1].update(red_player='changed'),
    lambda rows: rows[0].update(black_player=''),
    lambda rows: rows[0].update(move='h3e3'),
    lambda rows: rows[0].update(move='h3-e3'),
    lambda rows: rows[1].update(timestamp='2025-01-01T00:00:00Z'),
    lambda rows: rows[1].update(timestamp='2030-01-01T00:00:00Z'),
    lambda rows: rows[1].update(timestamp='2026-01-01T00:00:01'),
    lambda rows: rows[1].update(timestamp='invalid'),
    lambda rows: rows[0].update(game_status='JOINED', outcome=''),
    lambda rows: rows[0].update(game_status='DRAW_ACCEPTED'),
    lambda rows: rows[0].update(outcome='DRAW'),
    lambda rows: rows[0].update(extra='unexpected'),
])
def test_invalid_metadata_indices_timestamps_or_coordinates_cannot_be_repaired(change):
    rows = game_rows()
    change(rows)
    with pytest.raises(ValueError):
        native_game(rows, CUTOFF, minimum_plies=2)


def test_claimed_mate_and_full_history_repetition_are_natively_checked():
    rows = game_rows()
    for row in rows:
        row['game_status'] = 'CHECKMATED'
    with pytest.raises(ValueError, match='claimed board-rule terminal'):
        native_game(rows, CUTOFF, minimum_plies=2)
    repeated = ('b0c2 b9c7 c2b0 c7b9 ' * 3) + 'h0g2'
    with pytest.raises(ValueError, match='full-history terminal'):
        native_game(game_rows(moves=repeated), CUTOFF, minimum_plies=2)


def test_group_can_span_adjacent_csv_members_but_never_noncontiguous_game_fragments(tmp_path):
    rows = game_rows()
    path = acquisition(tmp_path, [], members={
        'pvp_game_moves_xiangqi_001.csv': csv_bytes(rows[:1]),
        'pvp_game_moves_xiangqi_002.csv': csv_bytes(rows[1:]),
    })
    archive = json.loads(path.read_text())['archive_path']
    groups = list(csv_groups(archive))
    assert len(groups) == 1 and groups[0][0] == rows
    assert [span['member'] for span in groups[0][1]] == [
        'pvp_game_moves_xiangqi_001.csv', 'pvp_game_moves_xiangqi_002.csv']
    path = acquisition(tmp_path, [rows, game_rows('other'), rows], 'reappeared')
    with pytest.raises(ValueError, match='reappeared'):
        list(csv_groups(json.loads(path.read_text())['archive_path']))


def test_serial_and_parallel_native_imports_deduplicate_canonical_games(tmp_path):
    groups = [game_rows('first'), game_rows('second', 'h2c2 h9g7'), game_rows('duplicate'),
              game_rows('bad')]
    for row in groups[-1]:
        row['game_status'], row['outcome'] = 'JOINED', ''
    capture = acquisition(tmp_path, groups)
    a = import_csv(capture, tmp_path / 'serial', workers=1, minimum_plies=2)
    b = import_csv(capture, tmp_path / 'parallel', workers=2, minimum_plies=2)
    assert a == b
    assert a['candidate_records_natively_examined'] == 4
    assert a['new_unique_recorded_games'] == 2 and a['duplicate_records'] == 1
    assert a['quarantined_records'] == 1
    assert a['heldout_position_isolation_complete'] is False and a['accepted_search_roots'] == 0
    for name in ['games', 'quarantine', 'attribution-differences']:
        assert (tmp_path / 'serial' / (name + '.jsonl')).read_bytes() == (
            tmp_path / 'parallel' / (name + '.jsonl')).read_bytes()
    x = read_rows(tmp_path / 'serial/duplicates.jsonl')
    y = read_rows(tmp_path / 'parallel/duplicates.jsonl')
    x[0]['first'].pop('existing_collection')
    y[0]['first'].pop('existing_collection')
    assert x == y


def test_previous_collection_reuse_preserves_split_and_differing_attribution(tmp_path):
    capture = acquisition(tmp_path, [game_rows()])
    previous = tmp_path / 'previous'
    import_csv(capture, previous, workers=1, minimum_plies=2)
    other = acquisition(tmp_path, [game_rows('another-source-id', red='another-anon')], 'other')
    summary = import_csv(other, tmp_path / 'next', [previous], workers=1, minimum_plies=2)
    assert summary['new_unique_recorded_games'] == 0
    assert summary['duplicate_records_with_differing_attribution'] == 1
    first = read_rows(previous / 'games.jsonl')[0]
    duplicate = read_rows(tmp_path / 'next/duplicates.jsonl')[0]
    assert duplicate['first']['split'] == first['split']
    assert read_rows(tmp_path / 'next/attribution-differences.jsonl')[0]['attributions_authenticated'] is False


def test_pinned_archive_change_is_rejected_before_output_creation(tmp_path):
    capture = acquisition(tmp_path, [game_rows()])
    archive = Path(json.loads(capture.read_text())['archive_path'])
    archive.write_bytes(archive.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed'):
        import_csv(capture, tmp_path / 'bad', workers=1, minimum_plies=2)
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('members', [
    {'payload.py': b'raise RuntimeError("must never run")'},
    {'pvp_game_moves_xiangqi_001.csv': csv_bytes([], columns=COLUMNS[:-1])},
    {'pvp_game_moves_xiangqi_001.csv': csv_bytes(game_rows()) + b'too,few,columns\n'},
])
def test_unexpected_archive_schema_cannot_produce_a_completed_manifest(tmp_path, members):
    capture = acquisition(tmp_path, [], members=members)
    output = tmp_path / 'failed'
    with pytest.raises(ValueError):
        import_csv(capture, output, workers=1, minimum_plies=2)
    assert not (output / 'manifest.json').exists()
    with pytest.raises(FileExistsError):
        import_csv(capture, output, workers=1, minimum_plies=2)


def test_bounded_game_size_fails_without_silently_truncating_the_record(tmp_path):
    capture = acquisition(tmp_path, [game_rows()])
    with pytest.raises(ValueError, match='byte budget'):
        import_csv(capture, tmp_path / 'large', workers=1, minimum_plies=2, max_record_bytes=1)
    assert not (tmp_path / 'large/manifest.json').exists()
