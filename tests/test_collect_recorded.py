import json
import sys

import pytest

from xqgeneral import collect_recorded
from xqgeneral.evidence import atomic_json, digest, load_jsonl, manifest
from xqgeneral.human_games import assigned_split, parse_dhtml_game


def source_page(date, moves):
    fields = {
        'binit': '8979695949392919097717866646260600102030405060708012720323436383',
        'firstnum': 0, 'length': 2, 'type': '全局', 'event': '2026年测试赛事',
        'date': date, 'red': '棋手甲', 'black': '棋手乙', 'result': '红胜',
        'movelist': moves,
    }
    return ('[DhtmlXQ]' + ''.join(f'[DhtmlXQ_{k}]{v}[/DhtmlXQ_{k}]'
                                 for k, v in fields.items()) + '[/DhtmlXQ]').encode()


def completed_cache(root):
    root.mkdir()
    visited = {}
    for number, date, moves in [(1, '2026-09-18', '77477062'),
                                (2, '2029-09-18', '77277062'),
                                (3, '2023-09-18', '77577062'),
                                (4, '2029-09-18', '77277062')]:
        page = root / f'View-{number}.html'
        page.write_bytes(source_page(date, moves))
        url = f'{collect_recorded.ORIGIN}/Category/{page.name}'
        visited[url] = {'path': str(page), 'sha256': digest(page), 'bytes': page.stat().st_size}
    atomic_json(root / 'visited-sources.json', visited)
    atomic_json(root / 'manifest.json', manifest('pinned_public_source_test_fixture', {},
                outputs=[root / 'visited-sources.json', *(row['path'] for row in visited.values())]))
    return visited


def test_cached_collection_quarantines_future_source_date_without_replacing_it_with_event_year(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    visited = completed_cache(cache)

    def no_network(*args, **kwargs):
        raise AssertionError('An offline reparse must not make network requests')

    monkeypatch.setattr(collect_recorded, 'urlopen', no_network)

    def collect(name, extra=()):
        output = tmp_path / name
        monkeypatch.setattr(sys, 'argv', ['collect-recorded', '--cached-data', str(cache),
            '--min-year', '2024', '--seed', '17', '--output', str(output),
            '--public-evidence', str(tmp_path / f'{name}-proof.json'), *extra])
        collect_recorded.main()
        return output

    # Historical unbounded runs retain the exact source date for reproducibility.
    original = collect('unbounded')
    original_games = load_jsonl(original / 'games.jsonl')
    assert len(original_games) == 2
    future = next(game for game in original_games if game['headers']['Date'].startswith('2029'))
    assert future['headers']['Event'] == '2026年测试赛事'
    assert future['headers']['Date'] == '2029-09-18'
    assert parse_dhtml_game((cache / 'View-2.html').read_bytes())['history'] == future['history']
    assert {row['url']: row['error'] for row in load_jsonl(original / 'quarantine.jsonl')} == {
        f'{collect_recorded.ORIGIN}/Category/View-3.html':
            'Actual game date is outside the requested recent-game range',
        f'{collect_recorded.ORIGIN}/Category/View-4.html': 'Duplicate recorded main line',
    }

    bounded = collect('bounded', ['--max-year', '2026'])
    games = load_jsonl(bounded / 'games.jsonl')
    assert games == [next(game for game in original_games if game['headers']['Date'].startswith('2026'))]
    assert games[0]['split'] == assigned_split(games[0]['game_id'], 17)
    rejected = load_jsonl(bounded / 'quarantine.jsonl')
    assert {row['url'] for row in rejected} == {
        f'{collect_recorded.ORIGIN}/Category/View-2.html',
        f'{collect_recorded.ORIGIN}/Category/View-3.html',
        f'{collect_recorded.ORIGIN}/Category/View-4.html',
    }
    assert all(row['error'] == 'Actual game date is outside the requested recent-game range'
               for row in rejected)
    proof = json.loads((tmp_path / 'bounded-proof.json').read_text())
    assert proof['games_by_year'] == {'2026': 1}
    assert proof['requested_source_year_minimum'] == 2024
    assert proof['requested_source_year_maximum'] == 2026
    assert proof['network_requests_performed'] is False
    assert proof['download_failures'] == 0
    assert json.loads((original / 'manifest.json').read_text())['config']['max_year'] is None
    assert json.loads((bounded / 'visited-sources.json').read_text()) == visited
    assert all(digest(row['path']) == row['sha256'] for row in visited.values())


def test_invalid_source_year_range_fails_before_output_or_network(tmp_path, monkeypatch):
    output = tmp_path / 'invalid-range'
    proof = tmp_path / 'proof.json'

    def no_network(*args, **kwargs):
        raise AssertionError('Invalid configuration must fail before any request')

    monkeypatch.setattr(collect_recorded, 'urlopen', no_network)
    monkeypatch.setattr(sys, 'argv', ['collect-recorded', '--game',
        f'{collect_recorded.ORIGIN}/Category/View-1.html', '--min-year', '2024',
        '--max-year', '2023', '--output', str(output), '--public-evidence', str(proof)])
    with pytest.raises(ValueError, match='Maximum source year'):
        collect_recorded.main()
    assert not output.exists()
    assert not proof.exists()
