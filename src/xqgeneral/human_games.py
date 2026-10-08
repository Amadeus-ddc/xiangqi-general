"""Import recorded games through native rules, without assuming move optimality."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
import json
import multiprocessing
from pathlib import Path
import random
import re
import subprocess
import unicodedata

import pyffish
from .evidence import atomic_json, digest, manifest, write_jsonl
from .rules import START_FEN, VARIANT, adjudicate, legal_moves, move_notation, piece_map, piece_name, play


RESULTS = {'1-0', '0-1', '1/2-1/2'}
TRANSLATION = str.maketrans({'車': '车', '俥': '车', '馬': '马', '傌': '马', '帥': '将',
    '帅': '将', '將': '将', '仕': '士', '相': '象', '砲': '炮', '進': '进', '後': '后',
    **dict(zip('一二三四五六七八九', '123456789'))})
TAG = re.compile(r'^\[([A-Za-z]+)\s+"((?:\\.|[^"\\])*)"\]\s*$', re.M)


def pgn_headers(raw):
    for encoding in ['utf-8-sig', 'big5', 'cp950']:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('Unsupported PGN encoding')
    entries = TAG.findall(text)
    headers = {key: re.sub(r'\\(["\\])', r'\1', value) for key, value in entries}
    if len(headers) != len(entries):
        raise ValueError('Duplicate PGN headers or multiple games in one source')
    if any(line.lstrip().startswith('[') and not TAG.fullmatch(line) for line in text.splitlines()):
        raise ValueError('Malformed PGN header')
    return text, headers, encoding


def recorded_year(headers):
    """Recognize a supplied year; never substitute a file or page publication date."""
    dates = re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', headers.get('Date', ''))
    if len(set(dates)) == 1:
        return int(dates[0])
    return None


def normalized_notation(value):
    return unicodedata.normalize('NFKC', value).translate(TRANSLATION).rstrip('!?+#')


def mainline_text(value):
    """Remove PGN comments and variations while rejecting unbalanced delimiters."""
    output, brace, variation, line_comment = [], 0, 0, False
    for char in value:
        if line_comment:
            if char in '\r\n':
                line_comment = False
                output.append(' ')
            continue
        if brace:
            if char == '{':
                brace += 1
            elif char == '}':
                brace -= 1
                if not brace:
                    output.append(' ')
            continue
        if char == '{':
            brace = 1
        elif char == '}':
            raise ValueError('Unbalanced PGN comment')
        elif char == ';':
            line_comment = True
        elif char == '(':
            variation += 1
            output.append(' ')
        elif char == ')':
            if not variation:
                raise ValueError('Unbalanced PGN variation')
            variation -= 1
            output.append(' ')
        elif not variation:
            output.append(char)
    if brace or variation:
        raise ValueError('Unclosed PGN comment or variation')
    return re.sub(r'\$\d+|\d+\.(?:\.\.)?', ' ', ''.join(output))


def resolve_move(fen, notation):
    moves = legal_moves(fen)
    if re.fullmatch(r'[A-Ia-i][0-9]-?[A-Ia-i][0-9]', notation):
        move = notation.replace('-', '').lower()
        if move not in moves:
            raise ValueError(f'Illegal recorded ICCS move: {notation}')
        return move
    wanted = normalized_notation(notation)
    names = {'r': '车', 'h': '马', 'c': '炮', 'p': '兵卒', 'a': '士', 'e': '象', 'k': '将'}
    pieces, matches = piece_map(fen), []
    piece_char = wanted[1] if wanted and wanted[0] in '前后中' and len(wanted) > 1 else wanted[:1]
    for move in moves:
        if piece_char in '车马炮兵卒士象将' and piece_char not in names[pieces[move[:2]].lower()]:
            continue
        notation_forms = move_notation(fen, move)
        symbol = pieces[move[:2]]
        red = symbol.isupper()
        file_number = lambda square: 9 - (ord(square[0]) - 97) if red else ord(square[0]) - 96
        delta = int(move[3]) - int(move[1])
        direction = '平' if not delta else '进' if (delta > 0) == red else '退'
        destination = file_number(move[2:]) if not delta or symbol.lower() in 'hea' else abs(delta)
        plain_file = f'{piece_name(symbol)[1:]}{file_number(move[:2])}{direction}{destination}'
        if any(wanted == normalized_notation(form) for form in
               [notation_forms['chinese'], notation_forms['wxf'], plain_file] if form):
            matches.append(move)
    if len(matches) != 1:
        raise ValueError(f'Recorded notation has {len(matches)} legal interpretations: {notation}')
    return matches[0]


def parse_game(raw, *, require_players=True):
    if type(require_players) is not bool:
        raise ValueError('Recorded participant requirement must be a boolean')
    text, headers, encoding = pgn_headers(raw)
    if headers.get('Game', 'Chinese Chess') != 'Chinese Chess':
        raise ValueError('Source is not a Chinese Chess game')
    if (require_players and (not headers.get('Red') or not headers.get('Black')) or
            headers.get('Result') not in RESULTS):
        raise ValueError('A recorded full game needs both players and a declared result')
    tokens = mainline_text(TAG.sub('', text)).split()
    if not tokens or tokens[-1] != headers['Result']:
        raise ValueError('Movetext result differs from the declared result')
    if any(token in RESULTS or token == '*' for token in tokens[:-1]):
        raise ValueError('Unexpected result within the recorded main line')
    source_fen = headers.get('FEN', START_FEN)
    pieces = piece_map(source_fen)
    if sum(p == 'K' for p in pieces.values()) != 1 or sum(p == 'k' for p in pieces.values()) != 1:
        raise ValueError('Recorded game needs one general per side')
    initial = pyffish.get_fen(VARIANT, source_fen, [])
    history, moves = [initial], []
    for index, token in enumerate(tokens[:-1]):
        outcome = adjudicate(initial, moves)
        if outcome['ended']:
            raise ValueError(f'Recorded move follows a native full-history terminal at ply {index}')
        move = resolve_move(history[-1], token)
        history.append(play(history[-1], move))
        moves.append(move)
    if not moves:
        raise ValueError('Recorded game has no moves')
    outcome = adjudicate(initial, moves)
    expected = {'red': '1-0', 'black': '0-1', None: '1/2-1/2'}[outcome['winner']]
    if outcome['ended'] and expected != headers['Result']:
        raise ValueError('Declared result contradicts the native terminal result')
    identity = hashlib.sha256(json.dumps([initial, moves], separators=(',', ':')).encode()).hexdigest()
    return {'game_id': 'recorded-' + identity, 'initial_fen': initial, 'source_initial_fen': source_fen,
            'moves': moves, 'history': history, 'headers': headers, 'source_encoding': encoding,
            'declared_result': headers['Result'], 'native_terminal': outcome,
            'declared_nonterminal_result_reason': 'unspecified' if not outcome['ended'] else None,
            'human_move_optimality_proven': False, 'recorded_move_optimality_proven': False,
            'neural_explanations_generated': False}


def parse_dhtml_game(raw):
    """Extract recorded main-line facts; exclude site annotations and analysis branches."""
    text = html.unescape(raw.decode('utf-8-sig'))
    blocks = re.findall(r'\[DhtmlXQ\](.*?)\[/DhtmlXQ\]', text, re.S) or [text]
    parsed = []
    standard = '8979695949392919097717866646260600102030405060708012720323436383'
    for block in blocks:
        fields = dict(re.findall(r'\[DhtmlXQ_([a-z]+)\](.*?)\[/DhtmlXQ_\1\]', block, re.S))
        if not fields.get('movelist'):
            continue
        if fields.get('binit') != standard or fields.get('firstnum', '0').strip() != '0':
            raise ValueError('DhtmlXQ source is a partial or nonstandard starting position')
        coordinates = re.sub(r'\s+', '', fields['movelist'])
        if not re.fullmatch(r'[0-9]+', coordinates) or len(coordinates) % 4:
            raise ValueError('Invalid DhtmlXQ main-line coordinates')
        if 'length' in fields and int(fields['length']) != len(coordinates) // 4:
            raise ValueError('DhtmlXQ declared length differs from its actual main line')
        if fields.get('type', '全局').strip() != '全局':
            raise ValueError('DhtmlXQ source does not declare a full game')
        moves = []
        for index in range(0, len(coordinates), 4):
            a, y, b, z = map(int, coordinates[index:index + 4])
            if a > 8 or b > 8:
                raise ValueError('DhtmlXQ file is outside the board')
            moves.append(f'{chr(97 + a)}{9-y}{chr(97 + b)}{9-z}')
        results = {'红胜': '1-0', '先胜': '1-0', '黑胜': '0-1', '先负': '0-1',
                   '和棋': '1/2-1/2', '和': '1/2-1/2', '先和': '1/2-1/2'}
        result = results.get(fields.get('result', '').strip())
        if not result:
            raise ValueError('DhtmlXQ source lacks a declared result')
        event = fields.get('event', '').strip()
        date = fields.get('date', '').strip()
        precision = 'source_date'
        if not date:
            year = re.match(r'(20\d{2})年', event)
            if not year:
                raise ValueError('DhtmlXQ source lacks a usable game date or event year')
            date, precision = year[1] + '-??-??', 'event_year_only'
        headers = {'Game': 'Chinese Chess', 'Event': event, 'Date': date,
                   'Round': fields.get('round', '').strip(), 'Red': fields.get('red', fields.get('redname', '')).strip(),
                   'Black': fields.get('black', fields.get('blackname', '')).strip(), 'Result': result}
        # JSON quoting preserves factual header text using PGN-compatible quote escapes.
        body = '\n'.join(f'[{key} {json.dumps(value, ensure_ascii=False)}]' for key, value in headers.items())
        game = parse_game((body + '\n\n' + ' '.join(moves) + ' ' + result).encode())
        parsed.append(dict(game, source_format='DhtmlXQ', date_precision=precision,
                           source_declared_result=fields['result'].strip(),
                           source_comments_or_analysis_branches_used=False))
    if not parsed:
        raise ValueError('No DhtmlXQ main line found')
    if len({g['game_id'] for g in parsed}) != 1:
        raise ValueError('Page contains distinct recorded games')
    return parsed[0]


def assigned_split(game_id, seed):
    bucket = int(hashlib.sha256(f'{seed}/{game_id}'.encode()).hexdigest(), 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def import_one(item):
    candidate, raw = item
    if hashlib.sha256(raw).hexdigest() != candidate['content_sha256']:
        raise ValueError('Candidate source bytes changed')
    try:
        game = parse_game(raw)
    except ValueError as error:
        return None, {'blob_sha1': candidate['blob_sha1'], 'source_paths': candidate['source_paths'],
                      'error': str(error)}
    if game['headers'] != candidate['headers']:
        raise ValueError('Parsed headers differ from pinned source discovery')
    if ' '.join(game['initial_fen'].split()[:2]) != ' '.join(START_FEN.split()[:2]):
        return None, {'blob_sha1': candidate['blob_sha1'], 'source_paths': candidate['source_paths'],
                      'error': 'Recorded full-match category has a nonstandard starting position'}
    kind = candidate.get('source_kind', 'recorded_human_match')
    return dict(game, players=candidate['players'], source_kind=kind,
        source={'repository': 'Yvonne761/Chinese-Chess-Practical-Dataset',
        'revision': candidate['source_revision'], 'paths': candidate['source_paths'],
        'blob_sha1': candidate['blob_sha1'], 'content_sha256': candidate['content_sha256'],
        'declared_license': 'CC-BY-4.0', 'category': kind,
        'source_authenticity_independently_proven': False}), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--max-games', type=int, default=512)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261050)
    parser.add_argument('--output', required=True)
    parser.add_argument('--public-evidence', default='evidence/human-master-games-native-import-v1.json')
    args = parser.parse_args()
    if args.max_games < 1 or args.workers < 1:
        raise ValueError('Positive game and worker budgets are required')
    root = Path(args.output)
    if root.exists():
        raise FileExistsError('Preserve prior human-game output; use a fresh directory')
    revision = subprocess.check_output(['git', '-C', args.source, 'rev-parse', 'HEAD'], text=True).strip()
    if revision != args.revision:
        raise ValueError('Game source differs from the pinned revision')
    candidates = json.loads(Path(args.candidates).read_text())
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    candidates.sort(key=lambda c: -(c.get('year') or 0))
    root.mkdir(parents=True)
    process = subprocess.Popen(['git', '-C', args.source, 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    jobs = []
    for candidate in candidates:
        process.stdin.write((candidate['blob_sha1'] + '\n').encode())
        process.stdin.flush()
        descriptor = process.stdout.readline().decode().split()
        if descriptor[:2] != [candidate['blob_sha1'], 'blob']:
            raise ValueError('Pinned source blob was not returned')
        raw = process.stdout.read(int(descriptor[2]))
        assert process.stdout.read(1) == b'\n'
        source_path = root / 'sources' / (candidate['blob_sha1'] + '.pgn')
        source_path.parent.mkdir(exist_ok=True)
        source_path.write_bytes(raw)
        jobs.append((dict(candidate, source_revision=revision), raw))
    process.stdin.close()
    assert process.wait(timeout=30) == 0
    accepted, rejected, duplicates, budget_omissions = [], [], [], 0
    seen = set()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for index, (game, error) in enumerate(pool.map(import_one, jobs, chunksize=8)):
            if error:
                rejected.append(error)
            elif game['game_id'] in seen:
                duplicates.append(game['source'])
            else:
                seen.add(game['game_id'])
                if len(accepted) < args.max_games:
                    accepted.append(dict(game, split=assigned_split(game['game_id'], args.seed),
                                         provenance=game['source_kind'] + ';native_full_history_verified'))
                else:
                    budget_omissions += 1
            if (index + 1) % 128 == 0:
                print(json.dumps({'native_candidates_processed': index + 1, 'retained_games': len(accepted),
                                  'quarantined': len(rejected), 'duplicates': len(duplicates)}), flush=True)
    if not accepted:
        raise ValueError('No recorded games passed native import')
    write_jsonl(root / 'games.jsonl', accepted)
    write_jsonl(root / 'quarantine.jsonl', rejected)
    atomic_json(root / 'duplicates.json', duplicates)
    license_path = root / 'SOURCE_LICENSE'
    license_path.write_bytes(subprocess.check_output(['git', '-C', args.source, 'show', 'HEAD:LICENSE']))
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
        'source_revision': revision, 'declared_source_license': 'CC-BY-4.0',
        'candidate_games_natively_examined': len(jobs), 'retained_unique_recorded_games': len(accepted),
        'quarantined_games': len(rejected), 'duplicate_games': len(duplicates),
        'valid_games_omitted_for_retained_budget': budget_omissions,
        'games_by_split': dict(Counter(g['split'] for g in accepted)),
        'games_by_normalized_participant_label': dict(Counter(p for g in accepted for p in g['players'])),
        'distinct_normalized_participant_labels': len({p for g in accepted for p in g['players']}),
        'participant_identity_independently_verified': False,
        'games_by_source_kind': dict(Counter(g['source_kind'] for g in accepted)),
        'games_by_year': dict(Counter(str(recorded_year(g['headers'])) for g in accepted)),
        'recorded_plies': sum(len(g['moves']) for g in accepted),
        'all_retained_moves_native_legal': True, 'full_history_terminal_checks_passed': True,
        'game_split_assigned_before_questions': True, 'human_move_optimality_proven': False,
        'source_authenticity_independently_proven': False, 'training_data_or_neural_labels_generated': False}
    atomic_json(root / 'manifest.json', manifest('recorded_games_native_import', vars(args),
        [args.candidates, license_path, *[root / 'sources' / (c['blob_sha1'] + '.pgn') for c in candidates]],
        [root / 'games.jsonl', root / 'quarantine.jsonl', root / 'duplicates.json'], summary))
    public = {key: value for key, value in summary.items()
              if key != 'games_by_normalized_participant_label'}
    public['participant_label_top_20'] = Counter(summary['games_by_normalized_participant_label']).most_common(20)
    atomic_json(args.public_evidence, dict(public, manifest_sha256=digest(root / 'manifest.json')))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
