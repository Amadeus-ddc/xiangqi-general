"""Catalogue source-declared human, computer and human-computer game records."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

from .evidence import atomic_json, digest, manifest
from .human_games import pgn_headers, recorded_year


NAME_CHARS = str.maketrans(dict(zip(
    '許銀鄭趙呂欽胡榮華蔣孫劉陳吳貴臨徐紅於漢馬來張鳳國義歐陽謝馮駿廣東龍楊羅黃軍偉譚鋒趙興',
    '许银郑赵吕钦胡荣华蒋孙刘陈吴贵临徐红于汉马来张凤国义欧阳谢冯骏广东龙杨罗黄军伟谭锋赵兴')))
KNOWN_NAMES = set('王天一 许银川 郑惟桐 赵鑫鑫 蒋川 吕钦 洪智 胡荣华 孙勇征 柳大华 '
                  '赵国荣 李来群 徐天红 陶汉明 于幼华 杨官璘 刘殿中 卜凤波 徐超 谢靖 '
                  '王斌 汪洋 孟辰 李少庚 郑一泓 陈泓盛 李翰林 陈寒峰 赵剑 景学义 '
                  '唐丹 王琳娜 陈丽淳 左文静 陈幸琳 梁妍婷 刘欢 赵冠芳 党国蕾 '
                  '吴可欣 欧阳琦琳 张婷婷 韩冰 胡明 金海英 伍霞'.split())
TEAM_PREFIXES = set('中国 中国台北 香港 台湾 台北 澳门 广东 上海 江苏 浙江 福建 安徽 山东 '
                    '河北 河南 湖北 湖南 江西 广西 海南 四川 重庆 云南 贵州 辽宁 吉林 '
                    '黑龙江 北京 天津 山西 陕西 甘肃 内蒙古 宁夏 新疆 青海 西藏 火车头 '
                    '煤矿 冶金 新加坡 越南 马来西亚 美国 加拿大'.split())


def participant_label(value):
    """Normalize labels conservatively; this does not establish a person's identity."""
    value = unicodedata.normalize('NFKC', value).translate(NAME_CHARS).strip()
    # Explicit whitespace separates team names in many PGNs. Preserve joined labels.
    parts = value.split()
    if len(parts) > 1 and re.fullmatch(r'[\u4e00-\u9fff]{2,4}', parts[-1]):
        value = parts[-1]
    # Only declared aliases with an explicit known team prefix are collapsed.
    # Unknown joined labels remain intact instead of guessing a person identity.
    for name in sorted(KNOWN_NAMES, key=lambda n: (-len(n), n)):
        if value.endswith(name):
            prefix = value[:-len(name)].replace('蘇', '苏').replace('臺', '台').replace('灣', '湾')
            if prefix in TEAM_PREFIXES or prefix.removesuffix('省') in TEAM_PREFIXES:
                return name
    return value


def source_kind(path):
    if path.startswith('Dataset/對局/大師對局/'):
        return 'recorded_human_match'
    if path.startswith('Dataset/對局/電腦對局/人機賽/'):
        return 'recorded_human_computer_match'
    if path.startswith('Dataset/對局/電腦對局/電腦對局競賽/'):
        return 'recorded_computer_match'
    return None


def select_diverse_games(games, budget, participant_cap, event_cap, seed):
    """Choose unique games with explicit caps on both participants and exact events."""
    if min(budget, participant_cap, event_cap) < 1:
        raise ValueError('Positive diversity budgets are required')
    # Interleave source kind and decade before filling; neither famous names nor wins
    # receive privileged slots. The stable hash resolves ties without input-order bias.
    pools = defaultdict(list)
    for game in games:
        year = recorded_year(game['headers'])
        group = (game['source_kind'], (year // 10 * 10) if year else 0,
                 game['declared_result'])
        pools[group].append(game)
    for pool in pools.values():
        pool.sort(key=lambda g: hashlib.sha256(f"{seed}/{g['game_id']}".encode()).hexdigest())
    participants, events, seen, accepted, rejected = Counter(), Counter(), set(), [], Counter()
    offsets = Counter()
    while len(accepted) < budget:
        progressed = False
        for group in sorted(pools):
            pool = pools[group]
            while offsets[group] < len(pool):
                progressed = True
                game = pool[offsets[group]]
                offsets[group] += 1
                identity = game['game_id']
                people = set(game['players'])
                event = unicodedata.normalize('NFKC', game['headers'].get('Event', '')).translate(NAME_CHARS).strip() or '(unspecified)'
                if identity in seen:
                    rejected['duplicate_game'] += 1
                    continue
                seen.add(identity)
                if any(participants[p] >= participant_cap for p in people):
                    rejected['participant_cap'] += 1
                    continue
                if events[event] >= event_cap:
                    rejected['event_cap'] += 1
                    continue
                accepted.append(game)
                participants.update(people)
                events.update([event])
                break
            if len(accepted) == budget:
                break
        if not progressed:
            break
    return accepted, dict(rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise FileExistsError('Use a fresh source catalogue output')
    revision = subprocess.check_output(['git', '-C', args.source, 'rev-parse', 'HEAD'], text=True).strip()
    if revision != args.revision:
        raise ValueError('Recorded source differs from the pinned revision')
    root.mkdir(parents=True)
    tree = subprocess.check_output(['git', '-C', args.source, 'ls-tree', '-r', '-z', revision])
    by_blob, counts = {}, Counter()
    for item in tree.split(b'\0'):
        if not item:
            continue
        descriptor, raw_path = item.split(b'\t', 1)
        path = raw_path.decode('utf-8')
        kind = source_kind(path)
        if kind and path.endswith('.pgn'):
            blob = descriptor.split()[2].decode('ascii')
            by_blob.setdefault(blob, []).append(path)
            counts.update([kind])
    paths_path, objects_path = root / 'source-paths-by-blob.json', root / 'objects.bin'
    atomic_json(paths_path, by_blob)
    with objects_path.open('wb') as output:
        result = subprocess.run(['git', '-C', args.source, 'cat-file', '--batch'],
            input=('\n'.join(by_blob) + '\n').encode('ascii'), stdout=output,
            stderr=subprocess.PIPE, timeout=600)
    if result.returncode:
        raise RuntimeError(f'Public source object extraction failed: {result.returncode}')
    candidates, quarantine, participants, years, kinds = [], [], Counter(), Counter(), Counter()
    with objects_path.open('rb') as handle:
        for index, (expected, paths) in enumerate(by_blob.items()):
            descriptor = handle.readline().decode('ascii').split()
            if descriptor[:2] != [expected, 'blob'] or not 0 < int(descriptor[2]) < 2_000_000:
                raise ValueError('Unexpected recorded source blob')
            raw = handle.read(int(descriptor[2]))
            if handle.read(1) != b'\n' or hashlib.sha1(
                b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != expected:
                raise ValueError('Source Git object hash differs')
            try:
                _, headers, encoding = pgn_headers(raw)
                if not headers.get('Red') or not headers.get('Black'):
                    raise ValueError('Both participant labels are required')
            except ValueError as error:
                quarantine.append({'blob_sha1': expected, 'error': str(error)})
                continue
            declared_kinds = {source_kind(p) for p in paths}
            kind = next(iter(declared_kinds)) if len(declared_kinds) == 1 else 'source_category_conflict'
            people = [participant_label(headers[p]) for p in ['Red', 'Black']]
            year = recorded_year(headers)
            candidate = {'blob_sha1': expected, 'source_paths': paths,
                         'source_revision': revision, 'source_kind': kind,
                         'source_encoding': encoding, 'headers': headers, 'players': people,
                         'content_sha256': hashlib.sha256(raw).hexdigest(), 'year': year,
                         'full_move_legality_or_authenticity_verified': False}
            candidates.append(candidate)
            participants.update(people)
            years.update([str(year)])
            kinds.update([kind])
            if (index + 1) % 5000 == 0:
                print(json.dumps({'source_blobs_examined': index + 1,
                                  'candidate_full_games': len(candidates)}), flush=True)
        if handle.read(1):
            raise ValueError('Unexpected trailing source objects')
    candidate_path = root / 'candidates.json'
    atomic_json(candidate_path, candidates)
    atomic_json(root / 'header-quarantine.json', quarantine)
    summary = {'status': 'complete', 'source_revision': revision,
               'category_file_counts_before_deduplication': dict(counts),
               'unique_source_blobs': len(by_blob), 'candidate_full_games': len(candidates),
               'malformed_header_blobs_quarantined': len(quarantine),
               'candidate_counts_by_source_kind': dict(kinds), 'candidate_counts_by_year': dict(years),
               'distinct_normalized_participant_labels': len(participants),
               'participant_label_top_20': participants.most_common(20),
               'participant_identity_independently_verified': False,
               'source_category_is_not_independent_authenticity_proof': True,
               'native_legality_not_yet_examined': True}
    atomic_json(root / 'manifest.json', manifest('recorded_games_source_catalogue', vars(args),
        [], [paths_path, objects_path, candidate_path, root / 'header-quarantine.json'], summary))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
