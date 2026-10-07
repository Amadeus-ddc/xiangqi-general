"""Collect bounded public tournament pages and import recorded main-line facts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .evidence import atomic_json, digest, manifest, write_jsonl
from .human_games import assigned_split, parse_dhtml_game, recorded_year
from .recorded_sources import participant_label


ORIGIN = 'https://www.xiangqiqipu.com'


def public_url(value):
    parsed = urlparse(value)
    if (parsed.scheme != 'https' or parsed.hostname != 'www.xiangqiqipu.com'
            or parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment):
        raise ValueError('Only credential-free public source URLs are supported')
    if not re.fullmatch(r'/Category/(?:View-\d+|List-\d+-\d+)\.html', parsed.path):
        raise ValueError('Unsupported public game or catalogue URL')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', nargs='*', default=[])
    parser.add_argument('--game', nargs='*', default=[])
    parser.add_argument('--cached-data', help='Completed collection to reparse without any new requests')
    parser.add_argument('--max-games', type=int, default=256)
    parser.add_argument('--min-year', type=int, default=2014)
    parser.add_argument('--delay-seconds', type=float, default=1)
    parser.add_argument('--seed', type=int, default=20261052)
    parser.add_argument('--output', required=True)
    parser.add_argument('--public-evidence', required=True)
    args = parser.parse_args()
    if args.max_games < 1 or args.delay_seconds < 1 or not (args.index or args.game or args.cached_data):
        raise ValueError('Provide source URLs, a positive game budget and at least one second between requests')
    urls = [public_url(value) for value in [*args.index, *args.game]]
    cached, pinned_inputs = {}, []
    if args.cached_data:
        cached_root = Path(args.cached_data)
        cached_manifest = cached_root / 'manifest.json'
        proof = json.loads(cached_manifest.read_text())
        if proof['status'] != 'complete':
            raise ValueError('Cached source collection is incomplete')
        for path, metadata in proof['outputs'].items():
            if digest(path) != metadata['sha256']:
                raise ValueError('Cached source collection output changed')
        cached = json.loads((cached_root / 'visited-sources.json').read_text())
        for url, metadata in cached.items():
            public_url(url)
            if digest(metadata['path']) != metadata['sha256']:
                raise ValueError('Cached public source page changed')
        pinned_inputs = [cached_manifest, cached_root / 'visited-sources.json']
    root = Path(args.output)
    if root.exists():
        raise FileExistsError('Preserve existing public-source collection; use a fresh directory')
    root.mkdir(parents=True)
    sources, failures, visited, game_urls = [], [], {}, dict.fromkeys(args.game)
    if cached:
        game_urls.update(dict.fromkeys(u for u in cached if re.search(r'/View-\d+\.html$', u)))
    last_request = 0

    def fetch(url):
        nonlocal last_request
        public_url(url)
        if args.cached_data:
            if url not in cached:
                raise ValueError('Requested source URL was not in the pinned cache')
            metadata = cached[url]
            path = Path(metadata['path'])
            sources.append(path)
            visited[url] = metadata
            return path.read_bytes()
        time.sleep(max(0, args.delay_seconds - (time.monotonic() - last_request)))
        last_request = time.monotonic()
        request = Request(url, headers={'User-Agent': 'Xiangqi-General/0.1 (public recorded-game research)'})
        with urlopen(request, timeout=30) as response:
            public_url(response.geturl())
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError('Public source page exceeds the bounded download size')
        path = root / 'sources' / (urlparse(url).path.rsplit('/', 1)[-1])
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        sources.append(path)
        visited[url] = {'path': str(path), 'sha256': digest(path), 'bytes': len(raw)}
        return raw

    for url in args.index:
        try:
            raw = fetch(url)
            # A link's placement is only discovery provenance. Actual year/event must
            # come from the game record, never the index or page publication date.
            for identity in re.findall(rb'/Category/View-(\d+)\.html', raw):
                game_urls.setdefault(ORIGIN + '/Category/View-' + identity.decode() + '.html', None)
        except Exception as error:
            failures.append({'url': url, 'error_type': type(error).__name__})
    accepted, rejected, seen = [], [], set()
    for index, url in enumerate(list(game_urls)[:args.max_games]):
        try:
            raw = fetch(url)
            game = parse_dhtml_game(raw)
            year = recorded_year(game['headers'])
            if year is None or year < args.min_year:
                raise ValueError('Actual game date is outside the requested recent-game range')
            if game['game_id'] in seen:
                raise ValueError('Duplicate recorded main line')
            seen.add(game['game_id'])
            # This is a published recorded match, without an independently proven
            # human/computer identity. Named-player provenance is kept in the headers.
            game.update(players=[participant_label(game['headers'][key]) for key in ['Red', 'Black']],
                        source_kind='published_recorded_match',
                        split=assigned_split(game['game_id'], args.seed),
                        provenance='public_recorded_mainline;native_full_history_verified',
                        source={'url': url, 'content_sha256': visited[url]['sha256'],
                                'declared_license': 'unspecified',
                                'public_redistribution_permission_established': False,
                                'source_authenticity_independently_proven': False})
            accepted.append(game)
        except ValueError as error:
            rejected.append({'url': url, 'error': str(error)})
        except Exception as error:
            failures.append({'url': url, 'error_type': type(error).__name__})
        if (index + 1) % 16 == 0:
            print(json.dumps({'pages_examined': index + 1, 'retained_recent_games': len(accepted),
                              'quarantined': len(rejected), 'download_failures': len(failures)}), flush=True)
    if not accepted:
        raise ValueError('No recent recorded games passed the complete native import')
    write_jsonl(root / 'games.jsonl', accepted)
    write_jsonl(root / 'quarantine.jsonl', rejected)
    atomic_json(root / 'download-failures.json', failures)
    atomic_json(root / 'visited-sources.json', visited)
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
               'discovered_game_urls': len(game_urls), 'candidate_game_pages_examined': min(len(game_urls), args.max_games),
               'retained_recent_recorded_games': len(accepted), 'quarantined_game_pages': len(rejected),
               'download_failures': len(failures), 'recorded_plies': sum(len(g['moves']) for g in accepted),
               'games_by_year': dict(Counter(str(recorded_year(g['headers'])) for g in accepted)),
               'games_by_split': dict(Counter(g['split'] for g in accepted)),
               'distinct_normalized_participant_labels': len({p for g in accepted for p in g['players']}),
               'participant_label_top_20': Counter(p for g in accepted for p in g['players']).most_common(20),
               'all_retained_moves_native_legal': True, 'full_history_terminal_checks_passed': True,
               'game_split_assigned_before_questions': True, 'site_prose_and_variations_used_for_labels': False,
               'declared_source_license': 'unspecified', 'raw_records_or_pages_bundled_with_code': False,
               'public_redistribution_permission_established': False,
               'source_authenticity_or_participant_identity_independently_proven': False,
               'recorded_move_optimality_proven': False, 'training_data_or_neural_labels_generated': False}
    summary['network_requests_performed'] = not bool(args.cached_data)
    atomic_json(root / 'manifest.json', manifest('public_recent_recorded_games_native_import', vars(args),
        [*pinned_inputs, *sources], [root / 'games.jsonl', root / 'quarantine.jsonl', root / 'download-failures.json',
                  root / 'visited-sources.json'], summary))
    atomic_json(args.public_evidence, dict(summary, manifest_sha256=digest(root / 'manifest.json')))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
