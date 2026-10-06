"""Select real checkpoints by raw move and planning validation, outside the trainer."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .evidence import atomic_json, digest, manifest


def functional_score(move, plan):
    for result in [move, plan]:
        if (result.get('split') != 'validation' or result.get('raw_generation') is not True or
                result.get('oracle_repairs') != 0 or result.get('examples', 0) < 1):
            raise ValueError('Selection requires nonempty raw validation without repairs')
        if result.get('legality_is_imposed_by_decoding', False) or result.get('rule_legal_constraints', False):
            raise ValueError('Rule-constrained legality cannot select a raw model')
    values = [move['no_mistake_rate'], plan['contract_valid_rate']]
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid functional quality metric')
    return .6 * values[0] + .4 * values[1]


def snapshot_checkpoint(source, output):
    """Pin the inode before inspecting its step; optimizer state stays in the source archive."""
    import torch
    from .training import atomic_checkpoint
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    incoming = output / ('incoming-' + uuid.uuid4().hex + '.pt')
    os.link(source, incoming)
    saved = torch.load(incoming, map_location='cpu', weights_only=True, mmap=True)
    step = saved.get('step', saved.get('selected_step'))
    if type(step) is not int or step < 0:
        raise ValueError('Checkpoint has no valid actual step')
    root = output / f'step-{step}'
    if root.exists():
        previous = torch.load(root / 'adapter.pt', map_location='cpu', weights_only=True, mmap=True)
        if (any(previous[k] != saved[k] for k in ['config', 'input_hashes', 'code']) or
                set(previous['trainable']) != set(saved['trainable']) or
                any(not torch.equal(previous['trainable'][k], v) for k, v in saved['trainable'].items())):
            raise ValueError('An existing candidate at this step belongs to different source weights or inputs')
        incoming.unlink()
        return root, step
    root.mkdir()
    incoming.rename(root / 'source-checkpoint.pt')
    checkpoint = {'trainable': saved['trainable'], 'config': saved['config'], 'selected_step': step,
                  'input_hashes': saved['input_hashes'], 'code': saved['code']}
    atomic_checkpoint(root / 'adapter.pt', checkpoint)
    atomic_json(root / 'snapshot.json', {'actual_step': step, 'atomic_source_inode_pinned': True,
                'source_sha256': digest(root / 'source-checkpoint.pt'),
                'adapter_sha256': digest(root / 'adapter.pt'), 'optimizer_state_in_adapter': False,
                'training_config': saved['config'], 'training_input_hashes': saved['input_hashes'],
                'training_code': saved['code']})
    return root, step


def read_validation(root, checkpoint):
    root = Path(root)
    proof = json.loads((root / 'manifest.json').read_text())
    if proof['status'] != 'complete' or proof['verification']['split'] != 'validation':
        raise ValueError('Incomplete or non-validation evaluation cannot select checkpoints')
    if proof['inputs'][str(checkpoint)]['sha256'] != digest(checkpoint):
        raise ValueError('Evaluation checkpoint identity differs')
    for path, artifact in proof['outputs'].items():
        if digest(path) != artifact['sha256']:
            raise ValueError('Evaluation output changed')
    if json.loads((root / 'metrics.json').read_text()) != proof['verification']:
        raise ValueError('Evaluation metrics differ from the manifest')
    return proof['verification']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', required=True)
    parser.add_argument('--training-session', required=True)
    parser.add_argument('--data', default='data/move-planning-v1')
    parser.add_argument('--features', default='data/move-quality-selfplay-v2/features-16.pt')
    parser.add_argument('--every', type=int, default=2000)
    parser.add_argument('--nodes', type=int, default=1000000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout-hours', type=float, default=12)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.every, args.nodes, args.workers, args.timeout_hours) <= 0:
        raise ValueError('Selection budgets must be positive')
    training, output = Path(args.training), Path(args.output)
    if (output / 'manifest.json').exists():
        raise FileExistsError('Completed selection exists; use a fresh output')
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_hours * 3600
    results = json.loads((output / 'progress.json').read_text()) if (output / 'progress.json').exists() else []
    for row in results:
        checkpoint = Path(row['checkpoint'])
        root = checkpoint.parent
        if (digest(checkpoint) != row['checkpoint_sha256'] or
                digest(root / 'moves/manifest.json') != row['move_manifest_sha256'] or
                digest(root / 'plans/manifest.json') != row['plan_manifest_sha256']):
            raise ValueError('Saved candidate evidence changed')
        if functional_score(read_validation(root / 'moves', checkpoint),
                            read_validation(root / 'plans', checkpoint)) != row['functional_score']:
            raise ValueError('Saved functional score differs from verified evidence')
    seen = {r['step'] for r in results}
    best = max(results, key=lambda r: r['functional_score']) if results else None

    def evaluate(source):
        nonlocal best
        root, step = snapshot_checkpoint(source, output / 'candidates')
        snapshot = json.loads((root / 'snapshot.json').read_text())
        contract = {'arguments': vars(args), **{k: snapshot[k] for k in
                    ['training_config', 'training_input_hashes', 'training_code']}}
        contract_path = output / 'contract.json'
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError('Selection continuation configuration, training inputs or source changed')
        atomic_json(contract_path, contract)
        if step in seen:
            return
        checkpoint = root / 'adapter.pt'
        for module, name, limit in [('evaluate_moves', 'moves', 192), ('evaluate_plans', 'plans', 96)]:
            dest = root / name
            if not (dest / 'manifest.json').exists():
                subprocess.run([sys.executable, '-u', '-m', 'xqgeneral.' + module, '--checkpoint', str(checkpoint),
                    '--data', args.data, '--features', args.features, '--split', 'validation', '--limit', str(limit),
                    '--nodes', str(args.nodes), '--workers', str(args.workers), '--output', str(dest)], check=True)
        move, plan = [read_validation(root / name, checkpoint) for name in ['moves', 'plans']]
        row = {'step': step, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
               'functional_score': functional_score(move, plan),
               'raw_move_no_mistake_rate': move['no_mistake_rate'],
               'raw_move_legal_rate': move['legal_rate'], 'raw_plan_contract_valid_rate': plan['contract_valid_rate'],
               'move_manifest_sha256': digest(root / 'moves/manifest.json'),
               'plan_manifest_sha256': digest(root / 'plans/manifest.json')}
        results.append(row); seen.add(step)
        if best is None or row['functional_score'] > best['functional_score']:
            best = row
            pointer = output / ('selected-' + uuid.uuid4().hex + '.pt.partial')
            os.link(checkpoint, pointer); pointer.replace(output / 'selected.pt')
            atomic_json(output / 'selected.json', best)
        atomic_json(output / 'progress.json', results)
        print(json.dumps({'event': 'functional_validation', **row}), flush=True)

    next_step = (max(seen, default=0) // args.every + 1) * args.every
    while True:
        complete = (training / 'metrics.json').exists()
        latest = training / 'latest.pt'
        if latest.exists():
            import torch
            saved = torch.load(latest, map_location='cpu', weights_only=True, mmap=True)
            current_step = saved['step']
            del saved
            if current_step >= next_step or complete:
                evaluate(latest)
                next_step = (current_step // args.every + 1) * args.every
        if complete:
            evaluate(training / 'adapter.pt')
            break
        alive = subprocess.run(['tmux', 'has-session', '-t', '=' + args.training_session],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not alive or time.monotonic() > deadline:
            raise RuntimeError('Training stopped or selection timed out before a completed run')
        time.sleep(30)
    if best is None:
        raise RuntimeError('No candidate passed actual functional validation')
    summary = {'candidates': results, 'selected': best, 'selection_split': 'validation',
               'selection_rule': '0.6 * raw_move_no_mistake_rate + 0.4 * raw_plan_contract_valid_rate',
               'NLL_training_selection_preserved': True, 'independent_test_used': False,
               'strong_play_or_explanation_quality_established': False}
    atomic_json(output / 'manifest.json', manifest('functional_checkpoint_selection', vars(args),
        [training / 'manifest.json', Path(args.data) / 'validation.jsonl', args.features],
        [output / 'selected.pt', output / 'selected.json', output / 'progress.json'], summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
