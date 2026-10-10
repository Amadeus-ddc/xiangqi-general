"""Declared foundation QA profiles, native labels and raw answer grading."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import re

from .curriculum_data import MATERIAL, SQUARES, STAGES, SYMBOLS
from .evidence import history_key
from .rules import (
    adjudicate, gives_check, in_check, legal_moves, piece_map, piece_name, replay,
    square_controllers, validate_position,
)

PAPER_PROFILE = 'paper_xiangqi_v1'
LEGACY_TASKS = {
    stage: ('piece', 'count', 'locate', 'empty', 'material', 'rank') if stage.startswith('static') else
           ('legal', 'illegal', 'moves', 'captures', 'checks') for stage in STAGES
}
PAPER_TASKS = {
    stage: ('piece', 'locate', 'file', 'rank', 'diagonal', 'counts', 'materials') if stage.startswith('static') else
           ('moves', 'controllers', 'captures', 'checks', 'parries', 'mate') for stage in STAGES
}
PROFILES = {'legacy': LEGACY_TASKS, PAPER_PROFILE: PAPER_TASKS}
DIAGONALS = tuple(tuple(s for s in SQUARES if (ord(s[0]) - 97) + sign * int(s[1]) == index)
                  for sign in (1, -1) for index in range(-9, 18)
                  if sum((ord(s[0]) - 97) + sign * int(s[1]) == index for s in SQUARES) >= 2)
NAMES = {piece_name(symbol): symbol for symbol in SYMBOLS}


def task_groups(profile='legacy'):
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError('Unknown foundation task profile')
    return PROFILES[profile]


def row_profile(row):
    profile = row.get('task_profile', 'legacy')
    task_groups(profile)
    return profile


def recipe_profile(recipe):
    profile = row_profile(recipe)
    if row_profile(recipe.get('raw_qa_gates', {})) != profile:
        raise ValueError('Recipe and raw QA gates must declare the same task profile')
    return profile


def validate_context(root, check_future=False):
    """Validate full native history; terminal boards may end, never extend, a line."""
    validate_position(root['initial_fen'])
    history = replay(root['initial_fen'], root['moves'])
    if (history != root['history'] or history[-1] != root['fen'] or
            history_key(history) != root['feature_key']):
        raise ValueError('Paper course root differs from its native full-history feature key')
    board = piece_map(root['fen'])
    if sum(p == 'K' for p in board.values()) != 1 or sum(p == 'k' for p in board.values()) != 1:
        raise ValueError('Paper course requires both native generals')
    # Optional AXF endings can still leave legal board moves. They are not
    # checkmate/stalemate examples, and no supplied history may cross one.
    for index in range(len(root['moves'])):
        if adjudicate(root['initial_fen'], root['moves'][:index])['ended']:
            raise ValueError('Paper course history continues after a terminal prefix')
    if adjudicate(root['initial_fen'], root['moves'])['ended'] and legal_moves(root['fen']):
        raise ValueError('Paper course root ended by history rules, not a terminal board')
    if check_future:
        line = root.get('future_moves', [])
        if not isinstance(line, list) or len(line) > 8:
            raise ValueError('Paper future must have zero to eight actual legal moves')
        _validate_future(root['initial_fen'], tuple(root['moves']), tuple(line))
    return root['feature_key']


@lru_cache(maxsize=16384)
def _validate_future(initial_fen, moves, line):
    validate_position(initial_fen)
    for index in range(len(line)):
        if adjudicate(initial_fen, [*moves, *line[:index]])['ended']:
            raise ValueError('Paper future continues after a terminal prefix')
    history = replay(initial_fen, [*moves, *line])
    if line and adjudicate(initial_fen, [*moves, *line])['ended'] and legal_moves(history[-1]):
        raise ValueError('Paper future ended by history rules, not a terminal board')


def target_fen(row):
    return replay(row['fen'], row.get('future_moves', []))[-1]


def traced_square(row):
    """Identify a surviving queried piece by its square before the future line."""
    source = row['query']['source']
    if source not in piece_map(row['fen']):
        raise ValueError('Future move query identifies an absent initial piece')
    for move in row.get('future_moves', []):
        if move[2:] == source and move[:2] != source:
            raise ValueError('Future move query identifies a captured piece')
        if move[:2] == source:
            source = move[2:]
    return source


def _line_squares(query, task):
    if task == 'file':
        return tuple(s for s in SQUARES if s[0] == query['file'])
    if task == 'rank':
        return tuple(s for s in SQUARES if int(s[1]) == query['rank'])
    start, end = query['start'], query['end']
    return next(d for d in DIAGONALS if {d[0], d[-1]} == {start, end})


def validate_query(row):
    """Check profile, query schema and future-piece identity before GPU work."""
    profile = row_profile(row)
    if profile != PAPER_PROFILE:
        return
    validate_position(row['fen'])
    task, query = row['task_type'], row['query']
    if row['stage'] not in PAPER_TASKS or task not in PAPER_TASKS[row['stage']]:
        raise ValueError('Task differs from the declared paper foundation profile')
    keys = {'piece': {'square'}, 'locate': {'symbol'}, 'file': {'file'}, 'rank': {'rank'},
            'diagonal': {'start', 'end'}, 'moves': {'source'}, 'controllers': {'square'}}.get(task, set())
    if not isinstance(query, dict) or set(query) != keys:
        raise ValueError('Paper task query fields differ from the declared schema')
    if any(not isinstance(query[k], str) or query[k] not in SQUARES
           for k in keys & {'square', 'source', 'start', 'end'}):
        raise ValueError('Paper task query contains an invalid square')
    if task == 'locate' and (not isinstance(query['symbol'], str) or query['symbol'] not in SYMBOLS):
        raise ValueError('Paper task query contains an invalid piece')
    if task == 'file' and (not isinstance(query['file'], str) or len(query['file']) != 1 or
                           query['file'] not in 'abcdefghi'):
        raise ValueError('Paper task query contains an invalid file')
    if task == 'rank' and (type(query['rank']) is not int or not 0 <= query['rank'] < 10):
        raise ValueError('Paper task query contains an invalid rank')
    if task == 'diagonal' and not any({d[0], d[-1]} == {query['start'], query['end']} for d in DIAGONALS):
        raise ValueError('Paper task diagonal must be a complete board diagonal of length at least two')
    if type(row.get('rationale_variant', 0)) is not int or row.get('rationale_variant', 0) not in (0, 1):
        raise ValueError('Paper rationale variant must be zero or one')
    line = row.get('future_moves', [])
    if (not isinstance(line, list) or len(line) > 8 or
            (row['stage'].endswith('current') and line) or
            (row['stage'].endswith('future') and not line)):
        raise ValueError('Paper question continuation differs from its declared course')
    if line and 'initial_fen' in row and 'moves' in row:
        _validate_future(row['initial_fen'], tuple(row['moves']), tuple(line))
    validate_position(target_fen(row))
    if task == 'moves':
        source = traced_square(row)
        board, fen = piece_map(target_fen(row)), target_fen(row)
        if source not in board or board[source].isupper() != (fen.split()[1] == 'w'):
            raise ValueError('Move query must identify a surviving side-to-move piece')
    if task == 'parries' and not in_check(target_fen(row)):
        raise ValueError('Paper check-parry questions require an in-check target')


def _counts(items):
    counts = Counter(items)
    return ' '.join(f'{piece_name(s)}={counts[s]}' for s in SYMBOLS if counts[s]) or '无'


def _pairs(items):
    return ','.join(f'{square}:{piece_name(piece)}' for square, piece in sorted(items)) or '无'


def native_tag(row):
    validate_query(row)
    task, query, fen = row['task_type'], row['query'], target_fen(row)
    board = piece_map(fen)
    if task == 'piece':
        return piece_name(board.get(query['square']))
    if task == 'locate':
        return ' '.join(sorted(s for s, p in board.items() if p == query['symbol'])) or '无'
    if task in ('file', 'rank', 'diagonal'):
        return _counts(board[s] for s in _line_squares(query, task) if s in board)
    if task == 'counts':
        return _counts(board.values())
    if task == 'materials':
        totals = [sum(MATERIAL[p.lower()] for p in board.values() if p.isupper() == red) for red in (True, False)]
        return f'红={totals[0]} 黑={totals[1]}'
    if task == 'controllers':
        attackers, defenders = square_controllers(fen, query['square'])
        return f'攻击={_pairs(attackers)};保护={_pairs(defenders)}'
    moves = legal_moves(fen)
    if task == 'mate':
        return '是' if in_check(fen) and not moves else '否'
    if task == 'moves':
        moves = [m for m in moves if m[:2] == traced_square(row)]
    elif task == 'captures':
        moves = [m for m in moves if m[2:] in board]
    elif task == 'checks':
        moves = [m for m in moves if gives_check(fen, m)]
    elif task != 'parries':
        raise ValueError('Unsupported paper task')
    return ' '.join(sorted(moves)) or '无'


def paper_answer(row):
    """Native short rationale plus a separately graded structured answer tag."""
    tag = native_tag(row)
    task, query, fen = row['task_type'], row['query'], target_fen(row)
    board = piece_map(fen)
    detailed = row.get('rationale_variant', 0) == 1
    if task == 'piece':
        rationale = f"{query['square']} 格是{tag}。"
    elif task == 'locate':
        rationale = f"{piece_name(query['symbol'])}的位置为{tag}。"
    elif task in ('file', 'rank', 'diagonal', 'counts'):
        squares = sorted(board) if task == 'counts' else _line_squares(query, task)
        walk = '，'.join(f'{s}为{piece_name(board.get(s))}' for s in squares)
        rationale = (walk + '。统计这些棋子得到以下数量。') if detailed else f'按棋子类型统计，结果为{tag}。'
    elif task == 'materials':
        values = []
        for red in (True, False):
            counts = Counter(p.lower() for p in board.values() if p.isupper() == red)
            terms = '+'.join(f'{n}×{MATERIAL[p]}' for p, n in sorted(counts.items()) if MATERIAL[p]) or '0'
            total = sum(n * MATERIAL[p] for p, n in counts.items())
            values.append(f"{'红' if red else '黑'}方{terms}={total}")
        rationale = '；'.join(values) + '。' if detailed else f'按指定分值计算双方子力，{tag}。'
    elif task == 'controllers':
        rationale = f"按吃子几何与阻挡关系检查 {query['square']}，不排除受牵制棋子；{tag}。"
    elif task == 'mate':
        if in_check(fen):
            moves = legal_moves(fen)
            rationale = f'正在被将军，但仍可用 {sorted(moves)[0]} 解将，因此未被将死。' if moves else '正在被将军且没有合法解将，行棋方已被将死。'
        else:
            rationale = '当前未被将军，未被将死。' if legal_moves(fen) else '当前未被将军但没有合法着法，属于困毙；象棋中行棋方仍判负。'
    else:
        if tag == '无':
            rationale = '按当前局面的合法走法检查，没有符合要求的着法。'
        else:
            details = []
            for move in tag.split():
                text = f'{piece_name(board[move[:2]])}从{move[:2]}到{move[2:]}'
                if move[2:] in board:
                    text += f'吃{piece_name(board[move[2:]])}'
                if gives_check(fen, move):
                    text += '并将军'
                details.append(text)
            rationale = '合法着法为：' + '；'.join(details) + '。'
    return rationale + '\n\n答案：' + tag


def paper_question(row, variant=0):
    if variant not in (0, 1, 2):
        raise ValueError('Question variant must be zero, one or two')
    task, query = row['task_type'], row['query']
    prefix = f"依次走 {' '.join(row.get('future_moves', []))} 后，" if row.get('future_moves') else ''
    if task == 'piece':
        request = f"{query['square']} 格上是什么棋子？空格回答空。"
        form = '棋子名称'
    elif task == 'locate':
        request = f"找出所有{piece_name(query['symbol'])}的位置。"
        form = '格子坐标，用空格分隔'
    elif task in ('file', 'rank', 'diagonal'):
        place = f"{query['file']} 列" if task == 'file' else (f"第{query['rank']}行" if task == 'rank' else f"从{query['start']}到{query['end']}的对角线")
        request = f'统计{place}的棋子类型与数量。'
        form = '棋子名=非零数量，用空格分隔'
    elif task == 'counts':
        request = '统计红黑双方全部棋子，分别按类型给出数量。'
        form = '棋子名=非零数量，用空格分隔'
    elif task == 'materials':
        request = '按车9马4炮4仕士2相象2兵卒1帅将0计，求红黑双方各自的子力总值。'
        form = '红=整数 黑=整数'
    elif task == 'controllers':
        request = f"哪些棋子攻击或保护 {query['square']} 格？按吃子几何判断，忽略牵制；空格的控制者都列为攻击者。"
        form = '攻击=坐标:棋子名,...;保护=坐标:棋子名,...'
    elif task == 'moves':
        origin = '最初位于' if row.get('future_moves') else '位于'
        request = f"枚举{origin} {query['source']} 的那枚棋子此时的所有合法着法。"
        form = '四字符着法，用空格分隔'
    elif task in ('captures', 'checks', 'parries'):
        request = {'captures': '枚举行棋方所有合法吃子着法。', 'checks': '枚举行棋方所有合法将军着法。',
                   'parries': '行棋方正在被将军，枚举所有合法解将着法。'}[task]
        form = '四字符着法，用空格分隔'
    elif task == 'mate':
        request = '行棋方是否已经被将死？将死指正在被将军且没有合法解将，未被将军的困毙另作区分。'
        form = '是或否'
    else:
        raise ValueError('Unsupported paper task')
    lead = ('', '请根据局面核对：', '请分析这个象棋局面：')[variant]
    return prefix + lead + request + f'给出简短依据，最后以“答案：”给出{form}，空列表用无。'


def _occupied_square(board, squares, rng):
    symbols = sorted({board[s] for s in squares})
    symbol = rng.choice(symbols)
    return rng.choice([s for s in squares if board[s] == symbol])


def paper_queries(fen, rng, initial_board=None):
    """One native query per available task; terminal and immobile pieces remain eligible."""
    board = piece_map(fen)
    occupied, empty = sorted(board), sorted(set(SQUARES) - board.keys())
    # Per-board multiplicity correction: empty is one class, like each piece
    # type, rather than receiving the mass of every unoccupied square.
    choices = list(SQUARES)
    if initial_board is not None:
        changed = [s for s in choices if initial_board.get(s) != board.get(s)]
        same = [s for s in choices if initial_board.get(s) == board.get(s)]
        choices = rng.choice([changed, same]) if changed and same else changed or same
    counts = Counter(board.get(s) for s in choices)
    piece_square = rng.choices(choices, weights=[1 / counts[board.get(s)] for s in choices], k=1)[0]
    diagonal = rng.choices(DIAGONALS, weights=[len(d) for d in DIAGONALS], k=1)[0]
    static = [('piece', {'square': piece_square}), ('locate', {'symbol': rng.choice(SYMBOLS)}),
              ('file', {'file': rng.choice('abcdefghi')}), ('rank', {'rank': rng.randrange(10)}),
              ('diagonal', {'start': diagonal[0], 'end': diagonal[-1]}), ('counts', {}), ('materials', {})]
    sources = [s for s, p in board.items() if p.isupper() == (fen.split()[1] == 'w')]
    species = sorted({board[s] for s in sources})
    symbol = rng.choice(species)
    source = rng.choice(sorted(s for s in sources if board[s] == symbol))
    square = rng.choice(empty) if rng.random() < .5 and empty else _occupied_square(board, occupied, rng)
    dynamic = [('moves', {'source': source}), ('controllers', {'square': square}),
               ('captures', {}), ('checks', {}), ('mate', {})]
    if in_check(fen):
        dynamic.append(('parries', {}))
    return static, dynamic


def paper_records(root, rng, sampler=None):
    """Use an already split, complete native root and its actual future line."""
    if root['split'] not in ('train', 'validation', 'test') or not isinstance(root['game_id'], str):
        raise ValueError('Paper questions require a preassigned native game split')
    validate_context(root, check_future=True)
    rows = []
    for future in (False, True):
        line = root.get('future_moves', []) if future else []
        if future and not line:
            continue
        fen = replay(root['fen'], line)[-1]
        origins = {s: s for s in piece_map(root['fen'])}
        for move in line:
            origins.pop(move[2:], None)
            origins[move[2:]] = origins.pop(move[:2])
        static, dynamic = (paper_queries(fen, rng, piece_map(root['fen']) if future else None)
                           if sampler is None else sampler.query_groups(root, fen, rng, future))
        for kind, tasks in [('static', static), ('dynamic', dynamic)]:
            stage = f"{kind}_{'future' if future else 'current'}"
            for task, query in tasks:
                if future and task == 'moves':
                    query = dict(query, source=origins[query['source']])
                row = dict(root, stage=stage, task_type=task, task_profile=PAPER_PROFILE, query=query,
                           future_moves=list(line), future_branches=[], rationale_variant=rng.randrange(2),
                           question_variant=rng.randrange(3),
                           teacher={'provider': 'pinned_native_xiangqi_rules', 'neural_prose_generated': False},
                           provenance=root['provenance'] + ';paper_xiangqi_rule_grounding')
                if 'id' in root:
                    row['foundation_source_root_id'] = root['id']
                if 'augmentation_parent' in root:
                    row['foundation_source_augmentation_parent'] = row.pop('augmentation_parent')
                row['question'] = paper_question(row, row['question_variant'])
                row['answer'] = paper_answer(row)
                identity = {k: row[k] for k in ('game_id', 'feature_key', 'stage', 'task_type', 'query',
                            'future_moves', 'rationale_variant', 'question_variant', 'task_profile')}
                row['id'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
                rows.append(row)
    return rows


def mirror_query(query):
    query = dict(query)
    for key in ('square', 'source', 'start', 'end'):
        if key in query:
            query[key] = query[key][0] + str(9 - int(query[key][1]))
    if 'symbol' in query:
        query['symbol'] = query['symbol'].swapcase()
    if 'rank' in query:
        query['rank'] = 9 - query['rank']
    return query


def _tag(text):
    if not isinstance(text, str) or text.count('答案：') != 1:
        raise ValueError('One explicit answer tag is required')
    tag = text.split('答案：', 1)[1].strip()
    if not tag:
        raise ValueError('An empty tag is not the explicit empty-list answer')
    return tag


def _parse(tag, task):
    if task not in set(PAPER_TASKS['static_current']) | set(PAPER_TASKS['dynamic_current']):
        raise ValueError('Unknown paper answer task')
    if task in ('piece', 'mate'):
        if tag not in (set(NAMES) | {'空'} if task == 'piece' else {'是', '否'}):
            raise ValueError('Invalid scalar answer')
        return tag
    if task in ('counts', 'file', 'rank', 'diagonal', 'materials'):
        fields = tag.split()
        if tag == '无' and task != 'materials':
            return {}
        pairs = [f.split('=') for f in fields]
        allowed = {'红', '黑'} if task == 'materials' else set(NAMES)
        if (any(len(p) != 2 or p[0] not in allowed or not re.fullmatch(r'0|[1-9][0-9]*', p[1]) for p in pairs) or
                len({p[0] for p in pairs}) != len(pairs) or not pairs):
            raise ValueError('Invalid count answer')
        result = {name: int(n) for name, n in pairs}
        if task == 'materials' and set(result) != allowed:
            raise ValueError('Both material sides are required')
        if task != 'materials' and any(n < 1 for n in result.values()):
            raise ValueError('Counts list only nonzero piece types')
        return result
    if task == 'controllers':
        groups = tag.split(';')
        if len(groups) != 2:
            raise ValueError('Both attack and defense lists are required')
        result = {}
        for group in groups:
            key, value = group.split('=', 1)
            if key not in ('攻击', '保护') or key in result:
                raise ValueError('Invalid control group')
            values = [] if value == '无' else [item.split(':') for item in value.split(',')]
            if any(len(p) != 2 or p[0] not in SQUARES or p[1] not in NAMES for p in values):
                raise ValueError('Invalid controller pair')
            result[key] = Counter(tuple(p) for p in values)
        return result
    values = [] if tag == '无' else tag.split()
    pattern = r'[a-i][0-9]' if task == 'locate' else r'[a-i][0-9][a-i][0-9]'
    if any(not re.fullmatch(pattern, value) for value in values):
        raise ValueError('Invalid coordinate or move answer')
    return Counter(values)


def grade_answer(expected, generated, task):
    """Compare parsed native tags while retaining the entire raw model response."""
    expected_tag = _tag(expected)
    expected_value = _parse(expected_tag, task)
    try:
        generated_tag = _tag(generated)
        content = expected_value == _parse(generated_tag, task)
        formatted = ' '.join(expected_tag.split()) == ' '.join(generated_tag.split())
        valid = True
    except (ValueError, TypeError):
        content = formatted = valid = False
    return {'correct': content, 'answer_format_valid': valid, 'canonical_tag_correct': formatted,
            'raw_exact_correct': isinstance(generated, str) and expected.strip() == generated.strip(),
            'prose_graded': False}
