"""Sequential/replay training with validation selection, resumable state, and DDP."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from .evidence import atomic_json, code_identity, digest, load_jsonl, manifest
from .modeling import initialize_trainable, load_model, load_trainable, trainable_state
from .rules import piece_map, piece_name, prompt
from .board_tokens import decode_board_text, encode_board_text

SYSTEM = "你是中国象棋助手。根据输入棋盘回答，只输出问题要求的内容。"


def model_autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda'))


def example_normalized_loss(logits, labels):
    """Give each supervised example equal weight despite different answer lengths."""
    targets = labels[:, 1:]
    counts = (targets != -100).sum(dim=1)
    if not torch.all(counts > 0):
        raise ValueError('Every training example must contain supervised target tokens')
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                             reduction='none', ignore_index=-100)
    return (losses.sum(dim=1) / counts).mean()


def configure_determinism(enabled):
    """Set the CUDA reproducibility contract before creating a CUDA context."""
    if not isinstance(enabled, bool):
        raise ValueError('deterministic_training must be a boolean')
    workspace = None
    if enabled:
        workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if workspace not in {':4096:8', ':16:8'}:
            raise ValueError('Deterministic training requires a supported CUBLAS workspace configuration')
    torch.use_deterministic_algorithms(enabled)
    return {'deterministic_training': enabled, 'cublas_workspace_config': workspace}


def atomic_checkpoint(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + '.partial')
    torch.save(value, partial)
    partial.replace(path)


def messages(record, mode='bridge', board_text=None):
    text = prompt(record)
    if mode == 'text_lora' or board_text is not None:
        representation = board_text or 'fen'
        if representation == 'fen':
            board = f"棋盘FEN：{record['fen']}。"
        elif representation == 'dictionary':
            occupied = piece_map(record['fen'])
            entries = [f'{f}{r}:{piece_name(occupied.get(f"{f}{r}"))}' for r in range(10) for f in 'abcdefghi']
            board = '棋盘各格：' + ' '.join(entries) + '。'
        else:
            raise ValueError('Board text must be fen or dictionary')
        text = board + text
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]


def encode_record(tokenizer, record, max_tokens, mode='bridge', board_tokens=False, board_text=None):
    text = tokenizer.apply_chat_template(messages(record, mode, board_text), tokenize=False, add_generation_prompt=True)
    answer = record['answer']
    if board_tokens:
        text, answer = encode_board_text(text), encode_board_text(answer)
    prefix = tokenizer.encode(text, add_special_tokens=False)
    target = tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False)
    if len(prefix) + len(target) > max_tokens:
        raise ValueError(f"Record {record['id']} exceeds max_tokens; supervision must not be silently truncated")
    return prefix + target, [-100] * len(prefix) + target


def batch_inputs(tokenizer, records, max_tokens, device, mode='bridge', board_tokens=False, board_text=None):
    pairs = [encode_record(tokenizer, r, max_tokens, mode, board_tokens, board_text) for r in records]
    length = max(len(p[0]) for p in pairs)
    ids, labels, mask = [], [], []
    for inputs, targets in pairs:
        padding = length - len(inputs)
        ids.append(inputs + [tokenizer.pad_token_id] * padding)
        labels.append(targets + [-100] * padding)
        mask.append([1] * len(inputs) + [0] * padding)
    return {'input_ids': torch.tensor(ids, device=device), 'labels': torch.tensor(labels, device=device),
            'attention_mask': torch.tensor(mask, device=device), 'use_cache': False}


def select_features(cache, indices, records, device, zero=False, rotate=False):
    keys = [indices[r['feature_key']] for r in records]
    if rotate:
        keys = keys[1:] + keys[:1]
    features = [level[keys].to(device=device, dtype=torch.bfloat16) for level in cache['features']]
    return [torch.zeros_like(f) for f in features] if zero else features


def sample_groups(rows):
    pools = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pools[row['stage']][row.get('task_type', 'default')].append(row)
    return dict(pools)


def sample_batch(rows, mixture, rng, global_batch_size, rank=0, world=1, pools=None):
    if global_batch_size % world:
        raise ValueError('Global batch size must be divisible by world size')
    pools = pools if pools is not None else sample_groups(rows)
    stages, weights = list(mixture), list(mixture.values())
    if not all(s in pools and mixture[s] > 0 for s in stages):
        raise ValueError('Mixture contains an empty stage or non-positive weight')
    batch = []
    for _ in range(global_batch_size):
        stage = rng.choices(stages, weights=weights, k=1)[0]
        task = rng.choice(sorted(pools[stage]))
        batch.append(rng.choice(pools[stage][task]))
    local = global_batch_size // world
    return batch[rank * local:(rank + 1) * local]


@torch.no_grad()
def evaluate(model, tokenizer, rows, cache, indices, config, device, zero=False, rotate=False):
    model.eval()
    weighted, count = 0.0, 0
    size = config.get('eval_batch_size', config['batch_size'])
    for offset in range(0, len(rows), size):
        batch = rows[offset:offset + size]
        inputs = batch_inputs(tokenizer, batch, config['max_tokens'], device, config.get('mode', 'bridge'),
                              config.get('board_tokens', False), config.get('board_text'))
        tokens = int((inputs['labels'][:, 1:] != -100).sum())
        with model.board_context(select_features(cache, indices, batch, device, zero, rotate)), model_autocast(device):
            loss = model(**inputs).loss
        weighted += float(loss) * tokens
        count += tokens
    if not count:
        raise ValueError('Evaluation has no supervised tokens')
    return weighted / count


@torch.no_grad()
def generate_examples(model, tokenizer, rows, cache, indices, device, mode='bridge', max_new_tokens=64, board_tokens=False, board_text=None):
    model.eval()
    results = []
    for record in rows:
        text = tokenizer.apply_chat_template(messages(record, mode, board_text), tokenize=False, add_generation_prompt=True)
        if board_tokens:
            text = encode_board_text(text)
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(device)
        with model.board_context(select_features(cache, indices, [record], device)), model_autocast(device):
            tokens = model.base.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                         pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if board_tokens:
            answer = decode_board_text(answer)
        results.append({'id': record['id'], 'game_id': record['game_id'], 'stage': record['stage'],
                        'question': record['question'], 'expected': record['answer'], 'generated': answer,
                        'exact_match': answer == record['answer']})
    return results


def compatible_resume(saved, requested):
    # Optimizer/data/sampling contracts cannot change in an exact continuation.
    if saved.get('loss_normalization', 'token') != requested.get('loss_normalization', 'token'):
        raise ValueError('Resume loss normalization differs; initialize a new experiment instead')
    if saved.get('deterministic_training', False) != requested.get('deterministic_training', False):
        raise ValueError('Resume determinism differs; initialize a new experiment instead')
    if saved.get('trainable_parameter_dtype', 'base') != requested.get('trainable_parameter_dtype', 'base'):
        raise ValueError('Resume trainable precision differs; initialize a new experiment instead')
    keys = ['model_path', 'model_revision', 'feature_path', 'data_path', 'mode', 'decoder_bridge_positions',
            'bridge_width', 'stages', 'mixture', 'seed', 'batch_size', 'learning_rate', 'max_tokens', 'steps',
            'board_tokens', 'expert_feature_depths', 'decoder_training', 'decoder_learning_rate',
            'token_learning_rate', 'replay_data_paths', 'lr_schedule', 'weight_decay', 'micro_batch_size',
            'validation_stages', 'validation_examples', 'eval_every', 'warmup_fraction', 'min_steps', 'patience', 'min_delta',
            'board_text']
    if any(saved.get(k) != requested.get(k) for k in keys):
        raise ValueError('Resume configuration differs; initialize a new experiment instead')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int, help='Save an incomplete checkpoint at this absolute step for controlled resume verification')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    normalization = config.get('loss_normalization', 'token')
    if normalization not in {'token', 'example'}:
        raise ValueError('Loss normalization must be token or example')
    compute_contract = configure_determinism(config.get('deterministic_training', False))
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    if world > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda'
    dest = Path(config['output'])
    if (dest / 'metrics.json').exists():
        raise FileExistsError('Completed run exists; use a new output directory')
    if dest.exists() and (dest / 'config.json').exists() and not args.resume:
        raise FileExistsError('Incomplete run exists; use --resume or a new output')
    torch.set_num_threads(config.get('cpu_threads', 4))
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = random.Random(config['seed'])
    started = time.monotonic()
    execution_code = code_identity()
    cache = torch.load(config['feature_path'], map_location='cpu', weights_only=True)
    if len(cache['features']) != len(config['decoder_bridge_positions']):
        raise ValueError('Cached expert levels do not match bridge positions')
    if config.get('expert_feature_depths') and list(cache['depths']) != config['expert_feature_depths']:
        raise ValueError('Expert layer contract differs from the cached features')
    indices = {key: i for i, key in enumerate(cache['keys'])}
    data_path = Path(config.get('data_path', 'data'))
    data_roots = [data_path, *[Path(p) for p in config.get('replay_data_paths', [])]]
    input_paths = [config['feature_path'], *[p / f'{s}.jsonl' for p in data_roots for s in ['train', 'validation']]]
    if config.get('init_from'):
        input_paths.append(config['init_from'])
    input_hashes = {str(p): digest(p) for p in input_paths} if rank == 0 else None
    if world > 1:
        objects = [input_hashes]
        dist.broadcast_object_list(objects, src=0)
        input_hashes = objects[0]
    mixture = config.get('mixture', {s: 1.0 for s in config['stages']})
    rows = {s: [r for p in data_roots for r in load_jsonl(p / f'{s}.jsonl') if r['stage'] in mixture]
            for s in ['train', 'validation']}
    if not rows['train'] or not rows['validation']:
        raise ValueError('Training and validation sets must both be nonempty')
    for records in rows.values():
        if any(r['feature_key'] not in indices for r in records):
            raise ValueError('Dataset contains a missing expert history context')
    validation_pool = [r for r in rows['validation'] if r['stage'] in config.get('validation_stages', mixture)]
    if not validation_pool:
        raise ValueError('Selected validation stages have no examples')
    validation = random.Random(config['seed']).sample(validation_pool,
                   min(config['validation_examples'], len(validation_pool)))
    model, tokenizer = load_model(config, cache['features'][0].shape[-1], device)
    pools = sample_groups(rows['train'])
    parameters = [p for p in model.parameters() if p.requires_grad]
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = 'board' if name.endswith('board_weight') else ('bridge' if name.startswith('bridges.') else 'decoder')
            groups[group].append(parameter)
    rates = {'bridge': config['learning_rate'], 'decoder': config.get('decoder_learning_rate', config['learning_rate']),
             'board': config.get('token_learning_rate', config['learning_rate'])}
    optimizer = torch.optim.AdamW([{'params': values, 'lr': rates[name], 'base_lr': rates[name], 'name': name}
                                   for name, values in groups.items()], weight_decay=config.get('weight_decay', 0.01))
    begin, best_loss, best_step, tokens_seen = 0, float('inf'), 0, 0
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location='cpu', weights_only=True)
        compatible_resume(resume['config'], config)
        if compute_contract['deterministic_training'] and resume.get('compute_contract') != compute_contract:
            raise ValueError('Resume deterministic compute contract differs')
        if resume.get('input_hashes') != input_hashes:
            raise ValueError('Resume input hashes are missing or changed; initialize a new experiment')
        if resume.get('code') != execution_code:
            raise ValueError('Exact resume requires the same preserved execution source')
        if resume['world_size'] != world:
            raise ValueError('Exact optimizer resume requires the same DDP world size')
        load_trainable(model, resume)
        optimizer.load_state_dict(resume['optimizer'])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value) and name != 'step':
                    state[name] = value.to(device)
        begin, best_loss, best_step, tokens_seen = (resume[k] for k in ['step', 'best_loss', 'best_step', 'tokens_seen'])
        rng.setstate(resume['random_state'])
        torch.set_rng_state(resume['torch_rng_state'])
        torch.cuda.set_rng_state(resume['cuda_rng_states'][rank], device=device)
    elif config.get('init_from'):
        source = torch.load(config['init_from'], map_location='cpu', weights_only=True)
        if source['config'].get('mode', 'bridge') != config.get('mode', 'bridge'):
            raise ValueError('Cannot initialize a different model mode')
        initialize_trainable(model, source, config)
    frozen = [(p, p.detach().flatten()[:1024].clone()) for p in model.parameters() if not p.requires_grad]
    decoder_probe = next((p for n, p in model.named_parameters()
                          if n.endswith(('q_proj.weight', 'q_proj.base_layer.weight')) and p.requires_grad), None)
    decoder_before = decoder_probe.detach().flatten()[:1024].clone() if decoder_probe is not None else None
    wrapped = DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])],
                                       broadcast_buffers=False) if world > 1 else model
    if rank == 0:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_json(dest / 'config.json', config)
        atomic_json(dest / 'validation_ids.json', [r['id'] for r in validation])
        if not resume:
            best_loss = evaluate(model, tokenizer, validation, cache, indices, config, device)
            atomic_checkpoint(dest / 'adapter.pt', {'trainable': trainable_state(model), 'config': config,
                              'selected_step': 0, 'input_hashes': input_hashes, 'code': execution_code})
        initial_loss = resume['initial_loss'] if resume else best_loss
        atomic_json(dest / 'execution.json', {'code': execution_code, 'input_hashes': input_hashes,
                                            'resumed_from_step': begin, 'initial_loss': initial_loss,
                                            'compute_contract': compute_contract})
        handle = (dest / 'training.jsonl').open('a' if resume else 'w')
    if world > 1:
        state = [best_loss, best_step]
        dist.broadcast_object_list(state, src=0)
        best_loss, best_step = state
        dist.barrier()
    completed = begin
    incomplete = False
    losses, norms = [], []
    patience_count = resume.get('patience_count', 0) if resume else 0
    for step in range(begin, config['steps']):
        wrapped.train()
        records = sample_batch(rows['train'], mixture, rng, config['batch_size'], rank, world, pools)
        inputs = batch_inputs(tokenizer, records, config['max_tokens'], device, config.get('mode', 'bridge'),
                              config.get('board_tokens', False), config.get('board_text'))
        local_tokens = (inputs['labels'][:, 1:] != -100).sum().to(dtype=torch.float32)
        total_tokens = local_tokens.clone()
        if world > 1:
            dist.all_reduce(total_tokens)
        total_units = total_tokens
        if normalization == 'example':
            total_units = torch.tensor(len(records), dtype=torch.float32, device=device)
            if world > 1:
                dist.all_reduce(total_units)
        warmup = max(1, int(config['steps'] * config.get('warmup_fraction', 0.05)))
        schedule = config.get('lr_schedule', 'cosine')
        if schedule not in {'cosine', 'constant'}:
            raise ValueError('Unknown learning-rate schedule')
        factor = (step + 1) / warmup if step < warmup else (1.0 if schedule == 'constant' else
                 0.05 + 0.95 * (1 + math.cos(math.pi * (step - warmup) / max(1, config['steps'] - warmup))) / 2)
        for group in optimizer.param_groups:
            group['lr'] = group.get('base_lr', config['learning_rate']) * factor
        optimizer.zero_grad(set_to_none=True)
        micro_size = config.get('micro_batch_size', len(records))
        if micro_size <= 0:
            raise ValueError('Micro batch size must be positive')
        weighted = torch.zeros((), device=device)
        for offset in range(0, len(records), micro_size):
            batch = records[offset:offset + micro_size]
            micro_inputs = {k: v[offset:offset + micro_size] if torch.is_tensor(v) else v for k, v in inputs.items()}
            micro_tokens = (micro_inputs['labels'][:, 1:] != -100).sum()
            micro_units = len(batch) if normalization == 'example' else micro_tokens
            synchronize = offset + micro_size >= len(records)
            context = wrapped.no_sync() if world > 1 and not synchronize else nullcontext()
            with context, model.board_context(select_features(cache, indices, batch, device)), model_autocast(device):
                if normalization == 'example':
                    forward_inputs = {k: v for k, v in micro_inputs.items() if k != 'labels'}
                    raw_loss = example_normalized_loss(wrapped(**forward_inputs).logits, micro_inputs['labels'])
                else:
                    raw_loss = wrapped(**micro_inputs).loss
                loss = raw_loss * micro_units * world / total_units
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
            weighted += raw_loss.detach() * micro_units
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite gradients')
        optimizer.step()
        if world > 1:
            dist.all_reduce(weighted)
        value = float(weighted / total_units)
        tokens_seen += int(total_tokens)
        losses.append(value)
        norms.append(float(norm))
        completed = step + 1
        if rank == 0:
            item = {'step': completed, 'loss': value, 'gradient_norm': float(norm), 'supervised_tokens': tokens_seen,
                    'lr': optimizer.param_groups[0]['lr'], 'seconds': time.monotonic() - started}
            handle.write(json.dumps(item) + '\n')
            handle.flush()
            if completed % 20 == 0 or completed == 1:
                print(json.dumps(item), flush=True)
        check = completed % config.get('eval_every', 100) == 0 or completed == config['steps'] or completed == args.stop_after
        stop = False
        if check:
            if world > 1:
                dist.barrier()
            if rank == 0:
                current = evaluate(model, tokenizer, validation, cache, indices, config, device)
                if current < best_loss - config.get('min_delta', 0.001):
                    best_loss, best_step, patience_count = current, completed, 0
                    atomic_checkpoint(dest / 'adapter.pt', {'trainable': trainable_state(model), 'config': config,
                                      'selected_step': best_step, 'input_hashes': input_hashes, 'code': execution_code})
                else:
                    patience_count += 1
                stop = patience_count >= config.get('patience', 5) and completed >= config.get('min_steps', config['steps'])
                print(json.dumps({'event': 'validation', 'step': completed, 'nll': current, 'best_step': best_step}), flush=True)
            if world > 1:
                state = [best_loss, best_step, stop]
                dist.broadcast_object_list(state, src=0)
                best_loss, best_step, stop = state
            cuda_states = [None] * world
            if world > 1:
                dist.all_gather_object(cuda_states, torch.cuda.get_rng_state(device))
            else:
                cuda_states = [torch.cuda.get_rng_state(device)]
            if rank == 0:
                atomic_checkpoint(dest / 'latest.pt', {'trainable': trainable_state(model), 'optimizer': optimizer.state_dict(), 'step': completed,
                            'config': config, 'best_loss': best_loss, 'best_step': best_step, 'tokens_seen': tokens_seen,
                            'random_state': rng.getstate(), 'torch_rng_state': torch.get_rng_state(),
                            'cuda_rng_states': cuda_states, 'world_size': world, 'input_hashes': input_hashes,
                            'initial_loss': initial_loss, 'patience_count': patience_count, 'code': execution_code,
                            'compute_contract': compute_contract})
        if args.stop_after and completed >= args.stop_after and completed < config['steps']:
            incomplete = True
            break
        if stop:
            break
    if world > 1:
        dist.barrier()
    if rank == 0:
        handle.close()
    if incomplete:
        if rank == 0:
            print(json.dumps({'status': 'incomplete', 'step': completed, 'resume_from': str(dest / 'latest.pt')}), flush=True)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    if rank == 0:
        assert all(torch.equal(saved, parameter.detach().flatten()[:1024]) for parameter, saved in frozen)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        load_trainable(model, torch.load(dest / 'adapter.pt', map_location='cpu', weights_only=True))
        after = evaluate(model, tokenizer, validation, cache, indices, config, device)
        zero = evaluate(model, tokenizer, validation, cache, indices, config, device, zero=True)
        shuffled = evaluate(model, tokenizer, validation, cache, indices, config, device, rotate=True)
        examples = generate_examples(model, tokenizer, validation[:config.get('generation_examples', 12)], cache,
                                     indices, device, config.get('mode', 'bridge'), config.get('max_new_tokens', 64),
                                     config.get('board_tokens', False), config.get('board_text'))
        atomic_json(dest / 'examples.json', examples)
        result = {'mode': config.get('mode', 'bridge'), 'courses': list(mixture), 'mixture': mixture,
                  'steps_completed': completed, 'selected_step': best_step, 'world_size': world,
                  'global_batch_size': config['batch_size'], 'supervised_tokens_seen': tokens_seen,
                  'training_loss_normalization': normalization, 'validation_loss_normalization': 'token',
                  **compute_contract,
                  'initial_validation_nll': initial_loss, 'final_validation_nll': after,
                  'zero_memory_validation_nll': zero, 'shuffled_memory_validation_nll': shuffled,
                  'trainable_parameters': sum(p.numel() for p in parameters),
                  'trainable_parameter_dtypes': {str(dtype): sum(p.numel() for p in parameters if p.dtype == dtype)
                                               for dtype in {p.dtype for p in parameters}},
                  'frozen_base_verified': config.get('decoder_training', 'frozen') == 'frozen',
                  'decoder_training': config.get('decoder_training', 'frozen'),
                  'decoder_parameter_dtype': str(model.base.get_input_embeddings().weight.dtype),
                  'frozen_parameter_dtypes': {str(dtype): sum(p.numel() for p, _ in frozen if p.dtype == dtype)
                                             for dtype in {p.dtype for p, _ in frozen}},
                  'compute_dtype': 'torch.bfloat16',
                  'selected_decoder_probe_max_change': (float((decoder_probe.detach().flatten()[:1024] - decoder_before).abs().max())
                                                        if decoder_probe is not None else None),
                  'frozen_parameter_gradients_verified': True,
                  'nonzero_gradients': any(n > 0 for n in norms), 'generation_examples': len(examples),
                  'generation_exact_matches': sum(e['exact_match'] for e in examples),
                  'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                  'seconds': time.monotonic() - started, 'purpose': config['purpose']}
        inputs = [p / f'{s}.jsonl' for p in data_roots for s in ['train', 'validation']]
        if config.get('init_from'):
            inputs.append(config['init_from'])
        result['feature_cache_sha256'] = input_hashes[config['feature_path']]
        atomic_json(dest / 'manifest.json', manifest('model_training', config, inputs,
                    [dest / 'adapter.pt', dest / 'examples.json', dest / 'training.jsonl'], result, code=execution_code))
        atomic_json(dest / 'metrics.json', result)
        print(json.dumps(result), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
