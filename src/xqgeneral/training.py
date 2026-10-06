"""Sequential/replay training with validation selection, resumable state, and DDP."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from .evidence import atomic_json, digest, load_jsonl, manifest
from .modeling import load_model, load_trainable, trainable_state
from .rules import prompt

SYSTEM = "你是中国象棋助手。根据输入棋盘回答，只输出问题要求的内容。"


def messages(record, mode='bridge'):
    text = prompt(record)
    if mode == 'text_lora':
        text = f"棋盘FEN：{record['fen']}。" + text
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]


def encode_record(tokenizer, record, max_tokens, mode='bridge'):
    text = tokenizer.apply_chat_template(messages(record, mode), tokenize=False, add_generation_prompt=True)
    prefix = tokenizer.encode(text, add_special_tokens=False)
    target = tokenizer.encode(record['answer'] + tokenizer.eos_token, add_special_tokens=False)
    if len(prefix) + len(target) > max_tokens:
        raise ValueError(f"Record {record['id']} exceeds max_tokens; supervision must not be silently truncated")
    return prefix + target, [-100] * len(prefix) + target


def batch_inputs(tokenizer, records, max_tokens, device, mode='bridge'):
    pairs = [encode_record(tokenizer, r, max_tokens, mode) for r in records]
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
        inputs = batch_inputs(tokenizer, batch, config['max_tokens'], device, config.get('mode', 'bridge'))
        tokens = int((inputs['labels'][:, 1:] != -100).sum())
        with model.board_context(select_features(cache, indices, batch, device, zero, rotate)):
            loss = model(**inputs).loss
        weighted += float(loss) * tokens
        count += tokens
    if not count:
        raise ValueError('Evaluation has no supervised tokens')
    return weighted / count


@torch.no_grad()
def generate_examples(model, tokenizer, rows, cache, indices, device, mode='bridge', max_new_tokens=64):
    model.eval()
    results = []
    for record in rows:
        text = tokenizer.apply_chat_template(messages(record, mode), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(device)
        with model.board_context(select_features(cache, indices, [record], device)):
            tokens = model.base.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                         pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        results.append({'id': record['id'], 'game_id': record['game_id'], 'stage': record['stage'],
                        'question': record['question'], 'expected': record['answer'], 'generated': answer,
                        'exact_match': answer == record['answer']})
    return results


def compatible_resume(saved, requested):
    # Optimizer/data/sampling contracts cannot change in an exact continuation.
    keys = ['model_path', 'model_revision', 'feature_path', 'data_path', 'mode', 'decoder_bridge_positions',
            'bridge_width', 'stages', 'mixture', 'seed', 'batch_size', 'learning_rate', 'max_tokens', 'steps']
    if any(saved.get(k) != requested.get(k) for k in keys):
        raise ValueError('Resume configuration differs; initialize a new experiment instead')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int, help='Save an incomplete checkpoint at this absolute step for controlled resume verification')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
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
    cache = torch.load(config['feature_path'], map_location='cpu', weights_only=True)
    indices = {key: i for i, key in enumerate(cache['keys'])}
    data_path = Path(config.get('data_path', 'data'))
    mixture = config.get('mixture', {s: 1.0 for s in config['stages']})
    rows = {s: [r for r in load_jsonl(data_path / f'{s}.jsonl') if r['stage'] in mixture]
            for s in ['train', 'validation']}
    if not rows['train'] or not rows['validation']:
        raise ValueError('Training and validation sets must both be nonempty')
    for records in rows.values():
        if any(r['feature_key'] not in indices for r in records):
            raise ValueError('Dataset contains a missing expert history context')
    validation = random.Random(config['seed']).sample(rows['validation'],
                   min(config['validation_examples'], len(rows['validation'])))
    model, tokenizer = load_model(config, cache['features'][0].shape[-1], device)
    pools = sample_groups(rows['train'])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'])
    begin, best_loss, best_step, tokens_seen = 0, float('inf'), 0, 0
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location='cpu', weights_only=True)
        compatible_resume(resume['config'], config)
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
        load_trainable(model, source)
    frozen = next(p for p in model.parameters() if not p.requires_grad)
    frozen_sample = frozen.detach().flatten()[:1024].clone()
    wrapped = DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])],
                                       broadcast_buffers=False) if world > 1 else model
    if rank == 0:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_json(dest / 'config.json', config)
        atomic_json(dest / 'validation_ids.json', [r['id'] for r in validation])
        if not resume:
            best_loss = evaluate(model, tokenizer, validation, cache, indices, config, device)
            torch.save({'trainable': trainable_state(model), 'config': config, 'selected_step': 0}, dest / 'adapter.pt')
        initial_loss = best_loss
        handle = (dest / 'training.jsonl').open('a' if resume else 'w')
    if world > 1:
        state = [best_loss, best_step]
        dist.broadcast_object_list(state, src=0)
        best_loss, best_step = state
        dist.barrier()
    completed = begin
    incomplete = False
    losses, norms = [], []
    patience_count = 0
    for step in range(begin, config['steps']):
        wrapped.train()
        records = sample_batch(rows['train'], mixture, rng, config['batch_size'], rank, world, pools)
        inputs = batch_inputs(tokenizer, records, config['max_tokens'], device, config.get('mode', 'bridge'))
        local_tokens = (inputs['labels'][:, 1:] != -100).sum().to(dtype=torch.float32)
        total_tokens = local_tokens.clone()
        if world > 1:
            dist.all_reduce(total_tokens)
        warmup = max(1, int(config['steps'] * config.get('warmup_fraction', 0.05)))
        factor = (step + 1) / warmup if step < warmup else 0.05 + 0.95 * (1 + math.cos(
            math.pi * (step - warmup) / max(1, config['steps'] - warmup))) / 2
        for group in optimizer.param_groups:
            group['lr'] = config['learning_rate'] * factor
        optimizer.zero_grad(set_to_none=True)
        with model.board_context(select_features(cache, indices, records, device)):
            raw_loss = wrapped(**inputs).loss
            loss = raw_loss * local_tokens * world / total_tokens
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite gradients')
        optimizer.step()
        weighted = raw_loss.detach() * local_tokens
        if world > 1:
            dist.all_reduce(weighted)
        value = float(weighted / total_tokens)
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
                    torch.save({'trainable': trainable_state(model), 'config': config, 'selected_step': best_step},
                               dest / 'adapter.pt')
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
                torch.save({'trainable': trainable_state(model), 'optimizer': optimizer.state_dict(), 'step': completed,
                            'config': config, 'best_loss': best_loss, 'best_step': best_step, 'tokens_seen': tokens_seen,
                            'random_state': rng.getstate(), 'torch_rng_state': torch.get_rng_state(),
                            'cuda_rng_states': cuda_states, 'world_size': world}, dest / 'latest.pt')
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
        assert torch.equal(frozen_sample, frozen.detach().flatten()[:1024])
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        load_trainable(model, torch.load(dest / 'adapter.pt', map_location='cpu', weights_only=True))
        after = evaluate(model, tokenizer, validation, cache, indices, config, device)
        zero = evaluate(model, tokenizer, validation, cache, indices, config, device, zero=True)
        shuffled = evaluate(model, tokenizer, validation, cache, indices, config, device, rotate=True)
        examples = generate_examples(model, tokenizer, validation[:config.get('generation_examples', 12)], cache,
                                     indices, device, config.get('mode', 'bridge'), config.get('max_new_tokens', 64))
        atomic_json(dest / 'examples.json', examples)
        result = {'mode': config.get('mode', 'bridge'), 'courses': list(mixture), 'mixture': mixture,
                  'steps_completed': completed, 'selected_step': best_step, 'world_size': world,
                  'global_batch_size': config['batch_size'], 'supervised_tokens_seen': tokens_seen,
                  'initial_validation_nll': initial_loss, 'final_validation_nll': after,
                  'zero_memory_validation_nll': zero, 'shuffled_memory_validation_nll': shuffled,
                  'trainable_parameters': sum(p.numel() for p in parameters), 'frozen_base_verified': True,
                  'nonzero_gradients': any(n > 0 for n in norms), 'generation_examples': len(examples),
                  'generation_exact_matches': sum(e['exact_match'] for e in examples),
                  'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                  'seconds': time.monotonic() - started, 'purpose': config['purpose']}
        inputs = [data_path / f'{s}.jsonl' for s in ['train', 'validation']]
        if config.get('init_from'):
            inputs.append(config['init_from'])
        result['feature_cache_sha256'] = digest(config['feature_path'])
        atomic_json(dest / 'manifest.json', manifest('model_training', config, inputs,
                    [dest / 'adapter.pt', dest / 'examples.json', dest / 'training.jsonl'], result))
        atomic_json(dest / 'metrics.json', result)
        print(json.dumps(result), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
