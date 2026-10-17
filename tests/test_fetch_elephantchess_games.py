import csv
import io
import json
from pathlib import Path
import runpy
import urllib.request
import zipfile

import pytest

from xqgeneral.elephant_games import COLUMNS, import_csv


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/fetch_elephantchess_games.py'
PAGE = 'https://elephantchess.io/about/datasets'
ZIP = 'https://cdn.elephantchess.io/data/pvp_game_moves_xiangqi_2026-06.zip'


def archive_bytes():
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=COLUMNS)
    writer.writeheader()
    for index, move in enumerate(['h2e2', 'h9g7']):
        row = dict.fromkeys(COLUMNS, '')
        row.update(game_id='anonymous-source', move_index=str(index), move=move,
                   timestamp=f'2020-01-01T00:00:0{index}Z', red_player='anon-red',
                   black_player='anon-black', game_status='RESIGNED', outcome='RED_WINS')
        writer.writerow(row)
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, 'w') as packed:
        packed.writestr('pvp_game_moves_xiangqi_001.csv', text.getvalue().encode())
    return raw.getvalue()


def install_source(monkeypatch, html=None, length=None, redirect=False):
    body = archive_bytes()
    html = html or ('<p>Games played by users are public and data is available under GPL-3.0 license.</p>'
                    f'<a href="{ZIP}">June</a><a href="{ZIP}">July</a>')
    calls = []

    class Response(io.BytesIO):
        def __init__(self, raw, url):
            super().__init__(raw)
            self.url = url
            self.headers = {'Content-Length': str(length if url == ZIP and length is not None else len(raw))}

        def geturl(self):
            return 'https://example.invalid/unexpected' if redirect else self.url

    def open_url(request, timeout):
        url = request.full_url
        calls.append(url)
        assert url in {PAGE, ZIP} and timeout == 30
        return Response(html.encode() if url == PAGE else body, url)

    monkeypatch.setattr(urllib.request, 'urlopen', open_url)
    return calls


def test_one_linked_archive_is_captured_once_and_the_receipt_imports_natively(tmp_path, monkeypatch):
    calls = install_source(monkeypatch)
    acquire = runpy.run_path(str(SCRIPT))['acquire']
    output = tmp_path / 'capture'
    proof = acquire(output)
    assert calls == [PAGE, ZIP] and proof['distinct_archive_urls'] == 1
    assert len(proof['all_listed_zip_links']) == 2
    assert proof['native_game_validation_complete'] is False
    summary = import_csv(output / 'manifest.json', tmp_path / 'native', workers=1, minimum_plies=2)
    assert summary['new_unique_recorded_games'] == 1 and summary['new_recorded_plies'] == 2
    assert json.loads((output / 'manifest.json').read_text()) == proof
    with pytest.raises(FileExistsError):
        acquire(output)
    assert calls == [PAGE, ZIP]


@pytest.mark.parametrize('html', [
    '<p>Data license unavailable.</p>',
    '<p>data is available under GPL-3.0 license</p>'
    f'<a href="{ZIP}">June</a><a href="{ZIP.replace("06.zip", "07.zip")}">July</a>',
    '<p>data is available under GPL-3.0 license</p><a href="https://example.invalid/archive.zip">ZIP</a>',
])
def test_changed_license_or_ambiguous_source_is_not_downloaded(tmp_path, monkeypatch, html):
    calls = install_source(monkeypatch, html=html)
    acquire = runpy.run_path(str(SCRIPT))['acquire']
    with pytest.raises(ValueError):
        acquire(tmp_path / 'failed')
    assert calls == [PAGE] and not (tmp_path / 'failed/manifest.json').exists()


def test_redirect_is_not_followed_as_an_approved_new_source(tmp_path, monkeypatch):
    calls = install_source(monkeypatch, redirect=True)
    with pytest.raises(ValueError, match='redirect'):
        runpy.run_path(str(SCRIPT))['acquire'](tmp_path / 'redirect')
    assert calls == [PAGE] and not (tmp_path / 'redirect/manifest.json').exists()


def test_wrong_http_length_preserves_incomplete_attempt_without_a_success_receipt(tmp_path, monkeypatch):
    install_source(monkeypatch, length=1)
    with pytest.raises(ValueError, match='Content-Length'):
        runpy.run_path(str(SCRIPT))['acquire'](tmp_path / 'truncated')
    assert not (tmp_path / 'truncated/manifest.json').exists()
    assert (tmp_path / 'truncated/pvp_game_moves_xiangqi_2026-06.zip.partial').exists()
