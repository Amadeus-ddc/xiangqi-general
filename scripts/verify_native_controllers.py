"""Compare all squares with pinned Pikafish; retain inputs and unmodified output."""
import argparse
import json
from pathlib import Path
import random
import subprocess

from xqgeneral.evidence import atomic_json, digest, manifest
from xqgeneral.rules import START_FEN, legal_moves, piece_map, play, square_controllers


PROBES = [
    START_FEN,
    '4k4/9/4r4/9/9/9/9/9/4R4/4K4 w - - 0 1',
    '4k4/9/9/9/9/4P4/2P6/2H6/9/4K4 w - - 0 1',
    '4k4/9/9/9/9/4P4/9/9/3H5/2E1K4 w - - 0 1',
    '4k4/1h7/9/9/9/1H2P4/9/1C7/9/4K4 w - - 0 1',
    '4k4/9/9/9/2P6/P3P4/9/9/9/4K4 b - - 0 1',
    '4k4/3RR4/9/9/9/9/9/9/9/5K3 b - - 0 1',
    '4k4/3R5/5R3/9/9/9/9/9/9/5K3 b - - 0 1',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--public-evidence', type=Path)
    parser.add_argument('--fixture-output', type=Path)
    parser.add_argument('--seed', type=int, default=20261071)
    parser.add_argument('--games', type=int, default=16)
    parser.add_argument('--plies', type=int, default=48)
    args = parser.parse_args()
    if args.output.exists() or min(args.games, args.plies) < 1:
        raise ValueError('Use a fresh reference output and positive board-walk budgets')
    proof_path = args.reference.parent / 'manifest.json'
    proof = json.loads(proof_path.read_text())
    root = Path(__file__).resolve().parents[1]
    pinned = next(s['revision'] for s in json.loads((root / 'configs/sources.json').read_text())
                  if s['name'] == 'pikafish')
    if proof['status'] != 'complete' or proof['vendor_revision'] != pinned or proof['weights_loaded'] is not False:
        raise ValueError('Reference executable has no completed pinned-source build')
    if any(digest(path) != expected for path, expected in proof['source_sha256'].items()):
        raise ValueError('A pinned controller reference source changed after build')
    if digest(args.reference) != proof['executable_sha256']:
        raise ValueError('The completed native reference executable changed')
    fens = list(PROBES)
    for game in range(args.games):
        rng, fen = random.Random(args.seed + game), START_FEN
        for _ in range(args.plies):
            moves = legal_moves(fen)
            if not moves:
                break
            fen = play(fen, rng.choice(moves))
            fens.append(fen)
    fens = list(dict.fromkeys(fens))
    args.output.mkdir(parents=True)
    inputs = args.output / 'input-fens.json'
    atomic_json(inputs, fens)
    result = subprocess.run([str(args.reference)], input='\n'.join(fens) + '\n',
                             capture_output=True, text=True, timeout=60)
    raw = args.output / 'native-output.txt'
    raw.write_text(result.stdout)
    (args.output / 'native-stderr.txt').write_text(result.stderr)
    atomic_json(args.output / 'native-exit.json', {'returncode': result.returncode})
    if result.returncode:
        raise ValueError('Native controller reference rejected an input; inspect preserved stderr')
    lines = result.stdout.splitlines()
    if len(lines) != 90 * len(fens):
        raise ValueError('Native reference did not return every square of every input board')
    fixtures, mismatches = [], []
    for index, fen in enumerate(fens):
        native = {}
        for entry in lines[index * 90:(index + 1) * 90]:
            square, sources = entry.split(':', 1)
            native[square] = sorted(sources.split(',')) if sources else []
        board = piece_map(fen)
        for square, sources in native.items():
            attackers, defenders = square_controllers(fen, square)
            union = sorted(s for s, _ in [*attackers, *defenders])
            expected_defenders = sorted(s for s in sources
                if square in board and board[s].isupper() == board[square].isupper())
            if union != sources or sorted(s for s, _ in defenders) != expected_defenders:
                mismatches.append({'fen': fen, 'square': square, 'native': sources,
                                   'python_attackers': attackers, 'python_defenders': defenders})
        if fen in PROBES:
            fixtures.append({'fen': fen, 'controllers_by_square': native})
    comparison = args.output / 'comparison.json'
    atomic_json(comparison, {'mismatches': mismatches})
    if mismatches:
        raise ValueError(f'Native controller comparison found {len(mismatches)} mismatches')
    fixture = {'schema_version': 1, 'source': 'unmodified_pinned_Pikafish_Position_attackers_to',
               'vendor_revision': pinned, 'positions': fixtures, 'weights_loaded': False}
    fixture_path = args.output / 'fixtures.json'
    atomic_json(fixture_path, fixture)
    summary = {'status': 'complete', 'evidence_state': 'reconstructed_baseline',
               'native_vendor_revision': pinned, 'positions': len(fens),
               'squares_compared': len(lines), 'mismatches': 0, 'controlled_positions': len(fixtures),
               'input_fens_sha256': digest(inputs), 'native_output_sha256': digest(raw),
               'fixture_sha256': digest(fixture_path), 'controller_geometry_ignores_pins': True,
               'native_weights_loaded': False, 'student_model_loaded': False,
               'training_executed': False, 'student_strength_measured': False}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    atomic_json(args.output / 'manifest.json', manifest('native_course_controller_verification', config,
        [inputs, args.reference, proof_path, *proof['source_sha256']], [raw, comparison, fixture_path], summary))
    if args.fixture_output:
        args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.fixture_output.write_bytes(fixture_path.read_bytes())
    if args.public_evidence:
        atomic_json(args.public_evidence, summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
