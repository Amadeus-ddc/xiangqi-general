"""Portable inference artifacts bound to completed training and the trained expert."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil

import torch

from .evidence import atomic_json, code_identity, digest, file_signature
from .feature_store import KIND as FEATURE_KIND


KIND = 'portable_xiangqi_inference_model'
RUNTIME_FIELDS = ('mode', 'decoder_training', 'decoder_bridge_positions', 'bridge_width',
                  'bridge_architecture', 'bridge_heads', 'board_tokens', 'board_text',
                  'trainable_parameter_dtype', 'expert_feature_depths', 'model_revision',
                  'mixture', 'feature_cache_mmap')
RUNTIME_MODULES = ('expert.py', 'bridge.py', 'board_tokens.py', 'modeling.py', 'inference.py',
                   'training.py', 'rules.py', 'net_pb2.py', 'sampling.py')


def _files(root):
    result = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part.startswith('.') for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError('Inference assets must be self-contained regular files')
        if path.is_file():
            result[relative.as_posix()] = path
    return result


def _decoder_assets(root):
    available = _files(root)
    names = {'config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
             'tokenizer.model', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
             'sentencepiece.bpe.model', 'spiece.model', 'chat_template.jinja', 'README.md',
             'LICENSE', 'LICENSE.md', 'LICENSE.txt', 'NOTICE', 'NOTICE.md',
             'model.safetensors', 'model.safetensors.index.json', 'pytorch_model.bin', 'pytorch_model.bin.index.json'}
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if name in available:
            names.update(json.loads(available[name].read_text()).get('weight_map', {}).values())
    files = {name: path for name, path in available.items()
             if name in names or name.startswith('chat_templates/')}
    if ('config.json' not in files or 'tokenizer_config.json' not in files or
            not any(name in files for name in ('tokenizer.json', 'tokenizer.model')) or
            not any(name in files for name in ('model.safetensors', 'model.safetensors.index.json',
                                              'pytorch_model.bin', 'pytorch_model.bin.index.json'))):
        raise ValueError('The decoder must include local config, tokenizer and weights')
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if name in files:
            index = json.loads(files[name].read_text())
            names = set(index.get('weight_map', {}).values())
            if not names or not names.issubset(files):
                raise ValueError('Decoder weight index has missing or external shards')
    return files


def _parameters(state):
    if (not isinstance(state, dict) or not state or
            any(not isinstance(name, str) or not isinstance(t, torch.Tensor) or
                not t.is_floating_point() or t.device.type != 'cpu' or not torch.isfinite(t).all()
                for name, t in state.items())):
        raise ValueError('The complete trained tensor mapping must be finite and on CPU')
    return {'trainable_tensors': len(state), 'trainable_parameters': sum(t.numel() for t in state.values()),
            'trainable_dtypes': sorted({str(t.dtype) for t in state.values()})}


def _binding(path):
    return {'sha256': digest(path), 'bytes': path.stat().st_size}


def export_model(training, weights, output):
    """Copy inference dependencies and exact trained values; leave all source artifacts intact."""
    training, weights, output = Path(training).resolve(), Path(weights).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve previous bundles; choose a new output directory')
    proof_path, checkpoint_path = training / 'manifest.json', training / 'adapter.pt'
    signatures = {p: file_signature(p) for p in (proof_path, checkpoint_path, weights)}
    proof = json.loads(proof_path.read_text())
    binding = _binding(checkpoint_path)
    outputs = {str(Path(p).resolve()): v for p, v in proof.get('outputs', {}).items()}
    if (proof.get('status') != 'complete' or proof.get('kind') != 'model_training' or
            outputs.get(str(checkpoint_path)) != binding):
        raise ValueError('A completed training manifest must bind the actual selected checkpoint')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
    config, metrics = saved['config'], proof['verification']
    selected = saved.get('selected_step')
    if (config != proof['config'] or saved.get('code') != proof.get('code') or
            type(selected) is not int or selected < 1 or metrics.get('selected_step') != selected or
            type(metrics.get('steps_completed')) is not int or metrics['steps_completed'] < selected or
            config.get('mode', 'bridge') != 'bridge' or
            config.get('decoder_training', 'frozen') not in {'frozen', 'full'}):
        raise ValueError('Selected trained updates, architecture and source must match completed execution')
    summary = _parameters(saved['trainable'])
    if summary['trainable_parameters'] != metrics.get('trainable_parameters'):
        raise ValueError('Selected parameter count differs from completed training')
    cache = Path(config['feature_path']).resolve()
    cache_proof = cache / 'manifest.json' if cache.is_dir() else cache
    signatures[cache_proof] = file_signature(cache_proof)
    feature = json.loads(cache_proof.read_text())
    inputs = {str(Path(p).resolve()): v for p, v in proof.get('inputs', {}).items()}
    expert_binding = _binding(weights)
    if (inputs.get(str(cache_proof)) != _binding(cache_proof) or
            feature.get('status') != 'complete' or feature.get('kind') != FEATURE_KIND or
            feature['verification'].get('expert_sha256') != expert_binding['sha256'] or
            feature['verification'].get('depths') != config.get('expert_feature_depths') or
            feature.get('code', {}).get('source_sha256', {}).get('src/xqgeneral/expert.py') !=
            digest(Path(__file__).with_name('expert.py'))):
        raise ValueError('Exported expert, depths and encoder must match the actual training cache proof')
    decoder = Path(config['model_path']).resolve()
    if output.is_relative_to(training) or output.is_relative_to(decoder):
        raise ValueError('Export outside the preserved training and decoder directories')
    assets = _decoder_assets(decoder)
    signatures.update({p: file_signature(p) for p in assets.values()})
    runtime = {k: config[k] for k in RUNTIME_FIELDS if k in config}
    runtime['model_path'] = 'decoder'
    output.mkdir(parents=True, exist_ok=False)
    copied = {}
    for relative, source in {**{f'decoder/{k}': v for k, v in assets.items()},
                             'expert/px0.pb.gz': weights}.items():
        before = _binding(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _binding(target) != before or file_signature(source) != signatures[source]:
            raise ValueError('An inference dependency changed while being copied')
        copied[relative] = before
    atomic_json(output / 'config.json', runtime)
    torch.save({'trainable': saved['trainable'], 'config': runtime, 'selected_step': selected,
                'code': saved['code']}, output / 'adapter.pt')
    exported = torch.load(output / 'adapter.pt', map_location='cpu', weights_only=True, mmap=True)
    if (exported['config'] != runtime or set(exported['trainable']) != set(saved['trainable']) or
            any(t.dtype != exported['trainable'][n].dtype or not torch.equal(t, exported['trainable'][n])
                for n, t in saved['trainable'].items())):
        raise ValueError('Export must preserve every trained value and dtype exactly')
    if any(file_signature(p) != signature for p, signature in signatures.items()):
        raise ValueError('Source training or dependencies changed during export')
    copied.update({name: _binding(output / name) for name in ('config.json', 'adapter.pt')})
    result = {'schema_version': 1, 'kind': KIND, 'status': 'complete',
              'evidence_state': 'reconstructed_baseline', 'code': code_identity(),
              'config': runtime, 'files': copied,
              'source_training': {'manifest_sha256': digest(proof_path), 'checkpoint': binding,
                                  'code': proof['code'], 'selected_step': selected,
                                  'steps_completed': metrics['steps_completed']},
              'verification': {**summary, 'exact_trained_values_and_dtypes_preserved': True,
                               'expert_matches_training_cache': True,
                               'optimizer_or_rng_state_included': False,
                               'datasets_or_feature_cache_included': False,
                               'quality_assessed_by_export': False}}
    atomic_json(output / 'manifest.json', result)
    return result


def read_bundle(directory):
    """Freshly check a relocatable bundle before loading its local model or expert."""
    root = Path(directory).resolve()
    proof = json.loads((root / 'manifest.json').read_text())
    if (proof.get('schema_version') != 1 or proof.get('kind') != KIND or
            proof.get('status') != 'complete' or not isinstance(proof.get('files'), dict)):
        raise ValueError('A completed portable inference manifest is required')
    files = proof['files']
    required = {'config.json', 'adapter.pt', 'expert/px0.pb.gz'}
    if not required.issubset(files):
        raise ValueError('Inference bundle is missing required dependencies')
    for relative, binding in files.items():
        path = PurePosixPath(relative)
        if (not isinstance(relative, str) or path.is_absolute() or '..' in path.parts or
                '\\' in relative or path.as_posix() != relative or not path.parts or
                any(part.startswith('.') for part in path.parts)):
            raise ValueError('Inference asset paths must stay inside the bundle')
        target = root / relative
        parent = root
        for component in path.parts:
            parent = parent / component
            if parent.is_symlink():
                raise ValueError('Inference dependencies must not use symbolic links')
        if not target.is_file() or _binding(target) != binding:
            raise ValueError(f'Inference asset differs from its bound identity: {relative}')
    if set(_files(root)) != set(files) | {'manifest.json'}:
        raise ValueError('Inference bundle contains unbound assets')
    source = Path(__file__).resolve().parent
    hashes = proof.get('code', {}).get('source_sha256', {})
    if any(hashes.get('src/xqgeneral/' + name) != digest(source / name) for name in RUNTIME_MODULES):
        raise ValueError('Installed inference and encoder source differs from the recorded bundle runtime')
    _decoder_assets(root / 'decoder')
    checkpoint = torch.load(root / 'adapter.pt', map_location='cpu', weights_only=True, mmap=True)
    config = json.loads((root / 'config.json').read_text())
    if (config != proof.get('config') or checkpoint['config'] != config or config.get('model_path') != 'decoder' or
            checkpoint.get('selected_step') != proof['source_training']['selected_step'] or
            _parameters(checkpoint['trainable']) != {k: proof['verification'][k]
                                                   for k in ('trainable_tensors', 'trainable_parameters', 'trainable_dtypes')}):
        raise ValueError('Inference parameters and runtime configuration differ from the bundle contract')
    config = dict(config, model_path=str(root / 'decoder'), inference_expert_path=str(root / 'expert/px0.pb.gz'),
                  inference_expert_sha256=files['expert/px0.pb.gz']['sha256'])
    checkpoint = dict(checkpoint, config=config)
    return checkpoint, proof


def main():
    parser = argparse.ArgumentParser(description='Export a completed model for local Xiangqi inference')
    parser.add_argument('--training', required=True, help='Completed model_training directory with adapter.pt')
    parser.add_argument('--weights', required=True, help='The Px0 weights used for the training cache')
    parser.add_argument('--output', required=True, help='New portable model directory')
    args = parser.parse_args()
    torch.set_num_threads(4)
    proof = export_model(args.training, args.weights, args.output)
    print(json.dumps({'status': proof['status'], 'output': args.output, **proof['verification']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
