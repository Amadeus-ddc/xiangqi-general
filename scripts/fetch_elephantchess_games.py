"""Download the single anonymous Xiangqi export explicitly linked by its publisher."""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile


PAGE = 'https://elephantchess.io/about/datasets'


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.text = [], []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get('href', '')
        if tag == 'a' and '.zip' in href:
            self.links.append(urllib.parse.urljoin(PAGE, href))

    def handle_data(self, value):
        self.text.append(value)


def request(url):
    response = urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'Xiangqi-General/0.1 (published anonymous export research)'}), timeout=30)
    if response.geturl() != url:
        response.close()
        raise ValueError('Review an unexpected source redirect before downloading')
    return response


def acquire(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve completed or interrupted acquisitions; use a fresh directory')
    output.mkdir(parents=True)
    with request(PAGE) as response:
        raw = response.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError('Dataset page exceeds the bounded source-page size')
    (output / 'datasets.html').write_bytes(raw)
    page = Page()
    page.feed(raw.decode('utf-8'))
    if 'data is available under GPL-3.0 license' not in ' '.join(' '.join(page.text).split()):
        raise ValueError('Review the changed dataset license declaration before acquisition')
    urls = list(dict.fromkeys(page.links))
    if len(urls) != 1:
        raise ValueError('Review changed export links; month labels are not disjoint datasets')
    url = urls[0]
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.netloc != 'cdn.elephantchess.io'
            or not re.fullmatch(r'/data/pvp_game_moves_xiangqi_\d{4}-\d{2}\.zip', parsed.path)
            or parsed.query or parsed.fragment):
        raise ValueError('Only an explicitly published anonymous Xiangqi ZIP is supported')
    archive = output / Path(parsed.path).name
    partial = archive.with_suffix('.zip.partial')
    digest, size = hashlib.sha256(), 0
    with request(url) as response, partial.open('xb') as target:
        headers = {key: response.headers.get(key) for key in
                   ['Content-Type', 'Content-Length', 'ETag', 'Last-Modified']}
        while chunk := response.read(1_048_576):
            size += len(chunk)
            if size > 268_435_456:
                raise ValueError('Archive exceeds the bounded download size')
            target.write(chunk)
            digest.update(chunk)
    if headers['Content-Length'] is not None and size != int(headers['Content-Length']):
        raise ValueError('Archive size differs from HTTP Content-Length')
    partial.rename(archive)
    with zipfile.ZipFile(archive) as packed:
        members = [item for item in packed.infolist() if not item.is_dir()]
        if (not members or len(members) > 100000
                or sum(item.file_size for item in members) > 4_294_967_296
                or len({item.filename for item in members}) != len(members)
                or any(not re.fullmatch(r'pvp_game_moves_xiangqi_\d+\.csv', item.filename) for item in members)):
            raise ValueError('Unsupported anonymous CSV archive inventory')
    proof = {'status': 'complete_acquisition_and_structural_inventory_only',
             'observed_at_utc': datetime.now(timezone.utc).isoformat(),
             'kind': 'published_anonymous_xiangqi_pvp_export', 'dataset_page': PAGE,
             'dataset_page_sha256': hashlib.sha256(raw).hexdigest(),
             'source_license_declaration': 'GPL-3.0 stated specifically for user-game data on the dataset page',
             'all_listed_zip_links': page.links, 'distinct_archive_urls': 1,
             'month_labels_do_not_prove_disjoint_archives': True,
             'archive_url': url, 'archive_path': str(archive),
             'archive_sha256': digest.hexdigest(), 'archive_bytes': size,
             'http_headers': headers, 'csv_files': len(members),
             'uncompressed_bytes': sum(item.file_size for item in members),
             'native_game_validation_complete': False, 'cross_source_deduplication_complete': False,
             'heldout_isolation_complete': False, 'accepted_search_roots': 0,
             'neural_labels_generated': 0, 'public_data_redistribution_performed': False}
    with (output / 'manifest.json').open('x') as handle:
        json.dump(proof, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    print(json.dumps(acquire(parser.parse_args().output), ensure_ascii=False))
