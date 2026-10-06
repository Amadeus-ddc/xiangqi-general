"""Terminal chess study with complete history and unmodified model advice."""
import argparse
import json
from pathlib import Path

from .evidence import atomic_json
from .explanations import EXPLANATION_QUESTION, parse_explanation, validate_explanation
from .rules import START_FEN, adjudicate, piece_map, piece_name, replay, side


def board_text(fen):
    board = piece_map(fen)
    rows = [f'{rank}  ' + ' '.join(piece_name(board[f'{file}{rank}']) if f'{file}{rank}' in board else '空格'
                                for file in 'abcdefghi') for rank in range(9, -1, -1)]
    return '\n'.join([*rows, '   ' + '    '.join('abcdefghi'), f'轮到{side(fen)}'])


def advise(predictor, initial_fen, moves):
    history = replay(initial_fen, moves)
    record = {'initial_fen': initial_fen, 'moves': list(moves), 'history': history,
              'fen': history[-1], 'question': EXPLANATION_QUESTION}
    raw = predictor.generate(record, max_new_tokens=768)
    try:
        value = parse_explanation(raw)
        proof = validate_explanation(record['fen'], value, require_branches=True)
    except (ValueError, TypeError, KeyError):
        value, proof = None, {'valid': False, 'errors': ['invalid_json']}
    return {'record': record, 'raw': raw, 'analysis': value, 'verification': proof,
            'external_oracle_used': False, 'answer_repaired': False}


def print_advice(result):
    value = result['analysis']
    if value is None:
        print('这次回答格式有误，请查看原始回答：\n' + result['raw'])
        return
    print(f"建议走法：{value.get('move', '缺失')}")
    for key, label in [('pv', '主要变化'), ('candidates', '候选走法')]:
        items = value.get(key)
        print(label + '：' + (' '.join(str(m) for m in items) if isinstance(items, list) else '缺失或格式错误'))
    print(str(value.get('explanation', '')))
    if not result['verification']['valid']:
        print('本次回答的走法、变化或棋盘事实未通过规则核验，详情已保存在分析记录中。')


def main():
    parser = argparse.ArgumentParser(description='中国象棋学习：显示棋盘、分析局面并保存完整历史')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', default='models/px0-latest.pb.gz')
    parser.add_argument('--initial-fen', default=START_FEN)
    parser.add_argument('--moves', nargs='*', default=[])
    parser.add_argument('--resume', help='之前保存的学习记录 JSON')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--output', help='保存本次分析或交互学习记录')
    args = parser.parse_args()
    initial, moves, analyses = args.initial_fen, list(args.moves), []
    if args.resume:
        saved = json.loads(Path(args.resume).read_text())
        initial, moves, analyses = saved['initial_fen'], saved['moves'], saved.get('analyses', [])
    history = replay(initial, moves)
    print(board_text(history[-1]))
    from .inference import Predictor
    predictor = Predictor(args.checkpoint, args.weights)

    def save():
        if args.output:
            atomic_json(args.output, {'initial_fen': initial, 'moves': moves, 'analyses': analyses,
                                      'checkpoint': args.checkpoint})

    def hint():
        outcome = adjudicate(initial, moves)
        if outcome['ended']:
            print('棋局已结束：' + str(outcome['winner'] or '和棋'))
            return
        result = advise(predictor, initial, moves)
        result['checkpoint'] = args.checkpoint
        analyses.append(result)
        print_advice(result)
        save()

    hint()
    if not args.interactive:
        save()
        return
    print('输入 UCCI 走法（如 b0c2）；hint 分析，undo 悔棋，board 显示，quit 退出。')
    while True:
        try:
            command = input('象棋> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if command == 'quit':
            break
        if command == 'hint':
            hint()
        elif command == 'undo':
            if moves:
                moves.pop()
            print(board_text(replay(initial, moves)[-1]))
            save()
        elif command == 'board':
            print(board_text(replay(initial, moves)[-1]))
        else:
            try:
                if adjudicate(initial, moves)['ended']:
                    print('棋局已结束；可用 undo 返回上一手。')
                    continue
                candidate = replay(initial, [*moves, command])
            except ValueError:
                print('这个走法不合法，请使用 a0 至 i9 坐标。')
                continue
            moves.append(command)
            print(board_text(candidate[-1]))
            save()
    save()


if __name__ == '__main__':
    main()
