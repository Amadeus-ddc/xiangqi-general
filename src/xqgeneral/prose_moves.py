"""Check explicit prose moves against native notation in the supplied variations.

This checks coordinates and move names, not arbitrary strategic or factual prose.
"""
from collections import defaultdict
import json
import re

from .rules import move_notation, play


NUMBERS = '一二三四五六七八九零〇十0123456789０１２３４５６７８９'
PIECES = '车車马馬炮砲相象仕士帅帥将將兵卒'
CHINESE_MOVE = re.compile(fr'(?:[{PIECES}][{NUMBERS}]+|[前后後中][{PIECES}])'
                          fr'[进進退平][{NUMBERS}]+')
UCCI_MOVE = re.compile(r'(?<![a-z0-9])[a-z][0-9]{1,2}[a-z][0-9]{1,2}(?![a-z0-9])', re.I)
TRANSLATION = str.maketrans(dict(zip('一二三四五六七八九零〇', '12345678900')) |
    dict(zip('０１２３４５６７８９', '0123456789')) |
    {'十':'10', '車':'车', '馬':'马', '砲':'炮', '帥':'帅', '將':'将', '進':'进', '後':'后'})


def notation_lines(fen, target):
    lines, seen = [], set()
    for line in [target['pv'], *[branch['pv'] for branch in target['branches']]]:
        key = tuple(line)
        if key in seen:
            continue
        seen.add(key)
        current, plies = fen, []
        for ply, move in enumerate(line, 1):
            plies.append(dict(move_notation(current, move), ply=ply))
            current = play(current, move)
        lines.append({'move':line[0], 'plies':plies})
    return lines


def validate_prose_moves(fen, target, prose):
    allowed, names = set(), defaultdict(set)
    for line in notation_lines(fen, target):
        for item in line['plies']:
            allowed.add(item['move'])
            if item['chinese']:
                names[item['chinese'].translate(TRANSLATION)].add(item['move'])
    coordinates = list(UCCI_MOVE.finditer(prose))
    chinese = list(CHINESE_MOVE.finditer(prose))
    errors = [f'unverified_coordinate:{match.group()}' for match in coordinates
              if match.group().lower() not in allowed]
    for name in chinese:
        normal = name.group().translate(TRANSLATION)
        if normal not in names:
            errors.append(f'unsupported_chinese_notation:{name.group()}')
            continue
        paired = [m.group().lower() for m in coordinates if
                  (m.end() <= name.start() and re.fullmatch(r'\s*[（(\[]?\s*[红黑]?\s*',
                                                           prose[m.end():name.start()])) or
                  (name.end() <= m.start() and re.fullmatch(r'\s*[（(\[]?\s*',
                                                           prose[name.end():m.start()]))]
        if not paired:
            errors.append(f'unpaired_chinese_notation:{name.group()}')
        elif not all(move in names[normal] for move in paired):
            errors.append(f'coordinate_notation_mismatch:{name.group()}:{paired}')
    return {'valid':not errors, 'errors':errors,
            'coordinate_mentions':[m.group() for m in coordinates],
            'chinese_notation_mentions':[m.group() for m in chinese]}


def consolidation_messages(packet, fen, target):
    content = dict(packet, verified_move_notation=notation_lines(fen, target),
                   unverified_child_prose_omitted=True,
                   score_sources={'target_fields.evaluation':'独立引擎根评分，根行棋方视角',
                                  'target_fields.branches':'子分析评分反号后的分支估计，不是独立根评分'})
    if 'child_analyses' in content:
        content['child_analyses'] = [{k:v for k,v in child.items() if k != 'unverified_explanation'}
                                    for child in content['child_analyses']]
    if 'verified_child_analyses' in content:
        content['verified_child_analyses'] = [dict(child, analysis={k:v for k,v in
            (child.get('analysis') or {}).items() if k != 'explanation'})
            for child in content['verified_child_analyses']]
    system = ('你是中国象棋讲解汇总教师。只依据根棋盘、target_fields、逐步事实和实际提供的变化，'
              '解释推荐着法、备选差异、收益与风险。学生正文未经语义核验，已省略。根评分来自根行棋方，'
              '子评分来自对手，不得混淆。不同分支不能混接，禁止补充未给出的着法、吃子、将军或强制结果。'
              '根评分只能引用target_fields.evaluation；分支评分如需引用，必须明确叫作子分析估计。'
              '主线按双方实际先后顺序说明，不能省略关键吃子代价；比较备选时指出给定变化中的具体风险。'
              '具体着法必须用UCCI坐标；如附中文记谱，只能引用verified_move_notation中的chinese，'
              '紧邻对应坐标，写成坐标（中文记谱）。chinese为空时只用坐标。'
              '不要把撤去炮架解将误写为挡将，不要把有限变化与评分写成强制获胜。'
              '只输出JSON对象，唯一字段explanation，为120到240字中文。')
    source_context = content.get('recorded_source_context')
    if isinstance(source_context, dict) and source_context.get('pre_fragment_game_history_available') is False:
        system += ('来源只提供一个对局片段，起点之前的历史未提供。禁止杜撰此前的走子或布局过程；'
                   '棋手字段只作来源署名，不据此推断身份、棋风或赛事背景。')
    return [{'role':'system', 'content':system},
            {'role':'user', 'content':json.dumps(content, ensure_ascii=False)}]
