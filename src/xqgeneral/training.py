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
from .evidence import atomic_json, code_identity, digest, manifest
from .modeling import decoder_checkpointing, initialize_trainable, load_model, load_trainable, trainable_state
from .rules import piece_map, piece_name, prompt
from .board_tokens import decode_board_text, encode_board_text
from .finite_training import data_profile, prepare_finite_epochs
from .training_index import IndexedTrainingRows, index_paths, training_row_indexes
from .feature_store import FeatureStore, feature_paths, gather_features, is_feature_store, open_feature_cache
from .verified_inputs import input_digest_execution, input_session_from_environment
from .explanation_answer_format import (answer_profile_kwargs, checked_answer_profile, serialized_answer)
from .training_parallel import (parallel_settings, shard_training_model, parameter_prefix,
                                gather_sharded_state, restore_sharded_optimizer, load_sharded_trainable)

SYSTEM = "你是中国象棋助手。根据输入棋盘回答，只输出问题要求的内容。"


def model_autocast(device, cache_enabled=True):
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda'),
                          cache_enabled=cache_enabled)


def autocast_cache_setting(config):
    enabled = config.get('autocast_cache_enabled', True)
    if type(enabled) is not bool:
        raise ValueError('autocast_cache_enabled must be a boolean')
    return enabled


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


def training_memory_settings(config):
    """Validate explicit memory choices while retaining the legacy defaults."""
    checked_answer_profile(config.get('explanation_answer_profile'))
    foreach = config.get('optimizer_foreach')
    bucket_view = config.get('ddp_gradient_as_bucket_view', False)
    if foreach is not None and type(foreach) is not bool:
        raise ValueError('optimizer_foreach must be a boolean or null')
    if type(bucket_view) is not bool:
        raise ValueError('ddp_gradient_as_bucket_view must be a boolean')
    sharding = config.get('optimizer_state_sharding', 'none')
    if not isinstance(sharding, str) or sharding not in {'none', 'zero1'}:
        raise ValueError('optimizer_state_sharding must be none or zero1')
    offload = config.get('optimizer_state_offload', 'none')
    if not isinstance(offload, str) or offload not in {'none', 'cpu'}:
        raise ValueError('optimizer_state_offload must be none or cpu')
    if offload == 'cpu' and (foreach is not False or sharding != 'none' or
            config.get('ddp_world_size', 1) != 1 or config.get('distributed_strategy', 'ddp') != 'ddp'):
        raise ValueError('CPU optimizer state offload requires single-rank, unsharded, foreach=false training')
    settings = {'optimizer_foreach': foreach, 'ddp_gradient_as_bucket_view': bucket_view, **parallel_settings(config)}
    if decoder_checkpointing(config):
        settings['decoder_gradient_checkpointing'] = True
    if not autocast_cache_setting(config):
        settings['autocast_cache_enabled'] = False
    if sharding != 'none':
        settings['optimizer_state_sharding'] = sharding
        settings['optimizer_checkpoint_transport'] = 'gloo_cpu'
    if offload != 'none':
        settings['optimizer_state_offload'] = offload
    return settings


def microbatch_padding_policy(config):
    policy = config.get('microbatch_padding', 'rank')
    if not isinstance(policy, str) or policy not in ('rank', 'micro'):
        raise ValueError('microbatch_padding must be rank or micro')
    return policy


def training_microbatches(inputs, rows, micro_size, padding='rank'):
    """Keep row order and optionally drop only ignored, right-padded columns."""
    padding = microbatch_padding_policy({'microbatch_padding': padding})
    if type(rows) is not int or rows < 1 or type(micro_size) is not int or micro_size < 1:
        raise ValueError('Training rows and micro batch size must be positive integers')
    lengths = None
    token_keys = ('input_ids', 'labels', 'attention_mask')
    if padding == 'micro':
        tensors = [inputs.get(key) for key in token_keys]
        if (any(not torch.is_tensor(value) or value.ndim != 2 for value in tensors) or
                tensors[0].shape[0] != rows or tensors[0].shape[1] < 1 or
                any(value.shape != tensors[0].shape or value.device != tensors[0].device for value in tensors)):
            raise ValueError('Micro padding requires matching two-dimensional token, label and attention tensors')
        if any(torch.is_tensor(value) for key, value in inputs.items() if key not in token_keys):
            raise ValueError('Micro padding supports the standard token tensors and scalar forward options')
        host_mask = inputs['attention_mask'].detach().cpu()
        if not torch.all((host_mask == 0) | (host_mask == 1)):
            raise ValueError('Micro padding requires a binary right-padded attention mask')
        lengths = host_mask.sum(dim=1).to(dtype=torch.int64).tolist()
        expected = torch.arange(host_mask.shape[1])[None, :] < torch.tensor(lengths)[:, None]
        if min(lengths) < 1 or not torch.equal(host_mask.bool(), expected):
            raise ValueError('Micro padding requires nonempty, right-padded rows')
        ignored = inputs['labels'].masked_select(inputs['attention_mask'] == 0)
        if not torch.all(ignored == -100):
            raise ValueError('Micro padding cannot remove supervised target tokens')
    for offset in range(0, rows, micro_size):
        end = min(offset + micro_size, rows)
        batch = {key: value[offset:end] if torch.is_tensor(value) else value
                 for key, value in inputs.items()}
        if lengths is not None:
            width = max(lengths[offset:end])
            for key in token_keys:
                batch[key] = batch[key][:, :width]
        yield offset, batch


def training_optimizer(model, config):
    settings = training_memory_settings(config)
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = 'board' if name.endswith('board_weight') else ('bridge' if name.startswith('bridges.') else 'decoder')
            groups[group].append(parameter)
    rates = {'bridge': config['learning_rate'], 'decoder': config.get('decoder_learning_rate', config['learning_rate']),
             'board': config.get('token_learning_rate', config['learning_rate'])}
    parameters = [{'params': values, 'lr': rates[name], 'base_lr': rates[name], 'name': name}
                  for name, values in groups.items()]
    options = dict(weight_decay=config.get('weight_decay', 0.01), foreach=settings['optimizer_foreach'])
    if settings.get('optimizer_state_offload') == 'cpu':
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError('CPU optimizer state offload requires one actual rank')
        from .optimizer_offload import CPUOffloadedAdamW
        return CPUOffloadedAdamW(parameters, **options)
    if settings.get('optimizer_state_sharding') == 'zero1':
        if not dist.is_initialized() or dist.get_world_size() < 2:
            raise ValueError('zero1 requires an initialized multi-rank process group')
        from torch.distributed.optim import ZeroRedundancyOptimizer
        class CPUCheckpointedZero(ZeroRedundancyOptimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._checkpoint_group = dist.new_group(backend='gloo')

            def load_state_dict(self, state_dict):
                super().load_state_dict(state_dict)
                # Native loading also casts the wrapper's exposed state to the
                # parameter device. Expose the actual local Adam state instead
                # of retaining a second GPU copy of every owned moment.
                self.state = self.optim.state

            def consolidate_state_dict(self, to=0):
                # Keep the native global parameter/state mapping, while moving
                # large serialized checkpoint traffic off the CUDA process group.
                self._check_overlap_initialized()
                self._sync_param_groups(self.param_groups, self.optim.param_groups)
                local = optimizer_state_on_cpu(self.optim.state_dict())
                gathered = [None] * self.world_size if self.rank == to else None
                destination = dist.get_global_rank(self.process_group, to)
                dist.gather_object(local, gathered, dst=destination, group=self._checkpoint_group)
                self._all_state_dicts = gathered if gathered is not None else []

        return CPUCheckpointedZero(parameters, optimizer_class=torch.optim.AdamW,
                                   overlap_with_ddp=False, parameters_as_bucket_view=False, **options)
    return torch.optim.AdamW(parameters, **options)


def optimizer_state_on_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: optimizer_state_on_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [optimizer_state_on_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(optimizer_state_on_cpu(item) for item in value)
    return value


def consolidate_optimizer_state(optimizer, config):
    """Every rank participates before rank zero saves the complete Adam state."""
    if training_memory_settings(config).get('optimizer_state_sharding') == 'zero1':
        optimizer.consolidate_state_dict(to=0)


def restore_optimizer_state(optimizer, saved, config, device):
    optimizer.load_state_dict(saved)
    if training_memory_settings(config).get('optimizer_state_offload') == 'cpu':
        if not optimizer.placement_summary()['all_states_on_CPU']:
            raise ValueError('Restored offloaded optimizer state must remain on CPU')
        return
    local = optimizer.optim if training_memory_settings(config).get('optimizer_state_sharding') == 'zero1' else optimizer
    for state in local.state.values():
        for name, value in state.items():
            if torch.is_tensor(value) and name != 'step':
                state[name] = value.to(device)


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


def encode_record(tokenizer, record, max_tokens, mode='bridge', board_tokens=False, board_text=None, *,
                  explanation_answer_profile=None):
    text = tokenizer.apply_chat_template(messages(record, mode, board_text), tokenize=False, add_generation_prompt=True)
    answer = serialized_answer(record, explanation_answer_profile)
    if board_tokens:
        text, answer = encode_board_text(text), encode_board_text(answer)
    prefix = tokenizer.encode(text, add_special_tokens=False)
    target = tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False)
    if len(prefix) + len(target) > max_tokens:
        raise ValueError(f"Record {record['id']} exceeds max_tokens; supervision must not be silently truncated")
    return prefix + target, [-100] * len(prefix) + target


def batch_inputs(tokenizer, records, max_tokens, device, mode='bridge', board_tokens=False, board_text=None, *,
                 explanation_answer_profile=None):
    options = answer_profile_kwargs({'explanation_answer_profile': explanation_answer_profile})
    pairs = [encode_record(tokenizer, r, max_tokens, mode, board_tokens, board_text, **options) for r in records]
    length = max(len(p[0]) for p in pairs)
    ids, labels, mask = [], [], []
    for inputs, targets in pairs:
        padding = length - len(inputs)
        ids.append(inputs + [tokenizer.pad_token_id] * padding)
        labels.append(targets + [-100] * padding)
        mask.append([1] * len(inputs) + [0] * padding)
    return {'input_ids': torch.tensor(ids, device=device), 'labels': torch.tensor(labels, device=device),
            'attention_mask': torch.tensor(mask, device=device), 'use_cache': False}


def shuffled_feature_rows(records):
    """Permute the full feature multiset without retaining any matching history."""
    groups = defaultdict(list)
    for position, record in enumerate(records):
        groups[record['feature_key']].append(position)
    ordered_groups = sorted(groups.values(), key=len, reverse=True)
    largest = len(ordered_groups[0]) if ordered_groups else 0
    if not largest or 2 * largest > len(records):
        raise ValueError('Feature shuffling requires enough different histories to mismatch every row')
    order = [position for group in ordered_groups for position in group]
    # A shift by the largest group excludes every original group, including
    # repeated histories, while preserving each feature's original multiplicity.
    donors = order[largest:] + order[:largest]
    result = [None] * len(records)
    for position, donor in zip(order, donors):
        result[position] = records[donor]
    return result


def select_features(cache, indices, records, device, zero=False, rotate=False):
    feature_rows = shuffled_feature_rows(records) if rotate else records
    keys = [indices[r['feature_key']] for r in feature_rows]
    features = gather_features(cache, keys, device)
    return [torch.zeros_like(f) for f in features] if zero else features


def sample_groups(rows):
    pools = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pools[row['stage']][row.get('task_type', 'default')].append(row)
    return dict(pools)


def load_training_rows(paths, stages):
    """Keep selected records in file order without materializing whole JSONL text."""
    rows = []
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row['stage'] in stages:
                        rows.append(row)
    return rows


def training_metadata(rows):
    return rows.iter_metadata() if isinstance(rows, IndexedTrainingRows) else iter(rows)


def select_validation_rows(rows, stages, seed, maximum):
    rng = random.Random(seed)
    if isinstance(rows, IndexedTrainingRows):
        pool = [i for i, row in enumerate(rows.iter_metadata()) if row['stage'] in stages]
        chosen = rng.sample(pool, min(maximum, len(pool)))
        return rows.get_batch(chosen)
    pool = [row for row in rows if row['stage'] in stages]
    return rng.sample(pool, min(maximum, len(pool)))


def training_input_paths(config):
    roots = [Path(config.get('data_path', 'data')), *[Path(p) for p in config.get('replay_data_paths', [])]]
    paths = [*feature_paths(config['feature_path']),
             *[p / f'{split}.jsonl' for p in roots for split in ['train', 'validation']],
             *[p for directory in training_row_indexes(config).values() for p in index_paths(directory)]]
    if config.get('init_from'):
        paths.append(config['init_from'])
    return paths


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
    feature_rows = shuffled_feature_rows(rows) if rotate else rows
    model.eval()
    weighted, count = 0.0, 0
    size = config.get('eval_batch_size', config['batch_size'])
    for offset in range(0, len(rows), size):
        batch = rows[offset:offset + size]
        inputs = batch_inputs(tokenizer, batch, config['max_tokens'], device, config.get('mode', 'bridge'),
                              config.get('board_tokens', False), config.get('board_text'), **answer_profile_kwargs(config))
        tokens = int((inputs['labels'][:, 1:] != -100).sum())
        features = select_features(cache, indices, feature_rows[offset:offset + size], device, zero=zero)
        with model.board_context(features), model_autocast(device, autocast_cache_setting(config)):
            loss = model(**inputs).loss
        weighted += float(loss) * tokens
        count += tokens
    if not count:
        raise ValueError('Evaluation has no supervised tokens')
    return weighted / count


@torch.no_grad()
def generate_examples(model, tokenizer, rows, cache, indices, device, mode='bridge', max_new_tokens=64, board_tokens=False, board_text=None, autocast_cache_enabled=True, synced_gpus=False, *,
                      explanation_answer_profile=None):
    checked_answer_profile(explanation_answer_profile)
    model.eval()
    results = []
    for record in rows:
        text = tokenizer.apply_chat_template(messages(record, mode, board_text), tokenize=False, add_generation_prompt=True)
        if board_tokens:
            text = encode_board_text(text)
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(device)
        with model.board_context(select_features(cache, indices, [record], device)), model_autocast(device, autocast_cache_enabled):
            options = {'synced_gpus': True} if synced_gpus else {}
            tokens = model.base.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                         pad_token_id=tokenizer.pad_token_id, **options)
        answer = tokenizer.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if board_tokens:
            answer = decode_board_text(answer)
        expected = serialized_answer(record, explanation_answer_profile)
        results.append({'id': record['id'], 'game_id': record['game_id'], 'stage': record['stage'],
                        'question': record['question'], 'expected': expected, 'generated': answer,
                        'exact_match': answer == expected})
    return results


def compatible_resume(saved, requested):
    # Optimizer/data/sampling contracts cannot change in an exact continuation.
    from .course_tasks import row_profile
    if checked_answer_profile(saved.get('explanation_answer_profile')) != checked_answer_profile(requested.get('explanation_answer_profile')):
        raise ValueError('Resume explanation answer profile differs; initialize a new experiment instead')
    if training_memory_settings(saved) != training_memory_settings(requested):
        raise ValueError('Resume optimizer/DDP memory settings differ; initialize a new experiment instead')
    if microbatch_padding_policy(saved) != microbatch_padding_policy(requested):
        raise ValueError('Resume microbatch padding differs; initialize a new experiment instead')
    if saved.get('row_index_paths', {}) != requested.get('row_index_paths', {}):
        raise ValueError('Resume training row index paths differ; initialize a new experiment')
    if data_profile(saved) != data_profile(requested):
        raise ValueError('Resume training data traversal differs; initialize a new experiment instead')
    if saved.get('mixture_seed', 0) != requested.get('mixture_seed', 0):
        raise ValueError('Resume finite mixture seed differs; initialize a new experiment instead')
    if row_profile(saved) != row_profile(requested):
        raise ValueError('Resume task profile differs; initialize a new experiment instead')
    if saved.get('loss_normalization', 'token') != requested.get('loss_normalization', 'token'):
        raise ValueError('Resume loss normalization differs; initialize a new experiment instead')
    if saved.get('deterministic_training', False) != requested.get('deterministic_training', False):
        raise ValueError('Resume determinism differs; initialize a new experiment instead')
    if saved.get('trainable_parameter_dtype', 'base') != requested.get('trainable_parameter_dtype', 'base'):
        raise ValueError('Resume trainable precision differs; initialize a new experiment instead')
    if saved.get('ddp_find_unused_parameters', False) != requested.get('ddp_find_unused_parameters', False):
        raise ValueError('Resume distributed reduction contract differs')
    if saved.get('feature_cache_mmap', False) != requested.get('feature_cache_mmap', False):
        raise ValueError('Resume feature-cache loading contract differs')
    keys = ['model_path', 'model_revision', 'feature_path', 'data_path', 'mode', 'decoder_bridge_positions',
            'bridge_width', 'stages', 'mixture', 'seed', 'batch_size', 'learning_rate', 'max_tokens', 'steps',
            'board_tokens', 'expert_feature_depths', 'decoder_training', 'decoder_learning_rate',
            'token_learning_rate', 'replay_data_paths', 'lr_schedule', 'weight_decay', 'micro_batch_size',
            'validation_stages', 'validation_examples', 'eval_every', 'warmup_fraction', 'min_steps', 'patience', 'min_delta',
            'board_text']
    if any(saved.get(k) != requested.get(k) for k in keys):
        raise ValueError('Resume configuration differs; initialize a new experiment instead')
    if saved.get('mode', 'bridge') == 'bridge':
        from .bridge import bridge_settings
        if bridge_settings(saved) != bridge_settings(requested):
            raise ValueError('Resume bridge architecture or attention heads differ; initialize a new experiment instead')


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int, help='Save an incomplete checkpoint at this absolute step for controlled resume verification')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    data_profile(config)
    index_roots = training_row_indexes(config)
    normalization = config.get('loss_normalization', 'token')
    if normalization not in {'token', 'example'}:
        raise ValueError('Loss normalization must be token or example')
    memory_settings = training_memory_settings(config)
    sharded = bool(parallel_settings(config))
    padding_policy = microbatch_padding_policy(config)
    compute_contract = {**configure_determinism(config.get('deterministic_training', False)), **memory_settings}
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    if memory_settings.get('optimizer_state_offload') == 'cpu' and world != 1:
        raise ValueError('CPU optimizer state offload requires one actual rank before CUDA initialization')
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
    cache_mmap = config.get('feature_cache_mmap', False)
    if type(cache_mmap) is not bool:
        raise ValueError('feature_cache_mmap must be a boolean')
    data_path = Path(config.get('data_path', 'data'))
    data_roots = [data_path, *[Path(p) for p in config.get('replay_data_paths', [])]]
    input_paths = training_input_paths(config)
    input_hashes = {str(p): digest(p) for p in input_paths} if rank == 0 else None
    if world > 1:
        objects = [input_hashes]
        dist.broadcast_object_list(objects, src=0)
        input_hashes = objects[0]
    cache = open_feature_cache(config['feature_path'], mmap=cache_mmap, verified_hashes=input_hashes)
    if len(cache['features']) != len(config['decoder_bridge_positions']):
        raise ValueError('Cached expert levels do not match bridge positions')
    if config.get('expert_feature_depths') and list(cache['depths']) != config['expert_feature_depths']:
        raise ValueError('Expert layer contract differs from the cached features')
    indices = {key: i for i, key in enumerate(cache['keys'])}
    mixture = config.get('mixture', {s: 1.0 for s in config['stages']})
    rows = {s: (IndexedTrainingRows(index_roots[s], [p / f'{s}.jsonl' for p in data_roots], mixture,
                                   split=s, verified_hashes=input_hashes) if s in index_roots else
                load_training_rows([p / f'{s}.jsonl' for p in data_roots], mixture))
            for s in ['train', 'validation']}
    if not rows['train'] or not rows['validation']:
        raise ValueError('Training and validation sets must both be nonempty')
    if config.get('task_profile') == 'paper_xiangqi_v1':
        from .course_tasks import row_profile
        from .curriculum_data import STAGES
        if any(row_profile(row) != config['task_profile'] for records in rows.values() for row in training_metadata(records)
               if row['stage'] in STAGES):
            raise ValueError('Foundation training data differs from its declared task profile')
    for records in rows.values():
        if any(r['feature_key'] not in indices for r in training_metadata(records)):
            raise ValueError('Dataset contains a missing expert history context')
    validation = select_validation_rows(rows['validation'], config.get('validation_stages', mixture),
                                        config['seed'], config['validation_examples'])
    if not validation:
        raise ValueError('Selected validation stages have no examples')
    finite_epochs = prepare_finite_epochs(rows['train'], mixture, config, world)
    model, tokenizer = load_model(config, cache['features'][0].shape[-1], device)
    pools = None if finite_epochs is not None else sample_groups(rows['train'])
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
        if finite_epochs is not None:
            finite_epochs.check_resume(resume.get('data_state'), resume['step'])
        load_trainable(model, resume)
        begin, best_loss, best_step, tokens_seen = (resume[k] for k in ['step', 'best_loss', 'best_step', 'tokens_seen'])
    elif config.get('init_from'):
        source = torch.load(config['init_from'], map_location='cpu', weights_only=True)
        if source['config'].get('mode', 'bridge') != config.get('mode', 'bridge'):
            raise ValueError('Cannot initialize a different model mode')
        initialize_trainable(model, source, config)
        del source
    model = shard_training_model(model, config)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = training_optimizer(model, config)
    if resume:
        if sharded:
            restore_sharded_optimizer(model, optimizer, resume['optimizer'])
        else:
            restore_optimizer_state(optimizer, resume['optimizer'], config, device)
        if any(group.get('foreach') is not memory_settings['optimizer_foreach'] for group in optimizer.param_groups):
            raise ValueError('Resume optimizer state differs from its foreach configuration')
        rng.setstate(resume['random_state'])
        torch.set_rng_state(resume['torch_rng_state'])
        torch.cuda.set_rng_state(resume['cuda_rng_states'][rank], device=device)
    frozen = [(p, parameter_prefix(p)) for p in model.parameters() if not p.requires_grad]
    decoder_probe = next((p for n, p in model.named_parameters()
                          if n.endswith(('q_proj.weight', 'q_proj.base_layer.weight')) and p.requires_grad), None)
    decoder_before = parameter_prefix(decoder_probe) if decoder_probe is not None else None
    find_unused = config.get('ddp_find_unused_parameters', False)
    if type(find_unused) is not bool:
        raise ValueError('ddp_find_unused_parameters must be a boolean')
    wrapped = DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])],
                                       broadcast_buffers=False, find_unused_parameters=find_unused,
                                       gradient_as_bucket_view=memory_settings['ddp_gradient_as_bucket_view']) if world > 1 and not sharded else model
    if not resume:
        if rank == 0 or sharded:
            best_loss = evaluate(model, tokenizer, validation, cache, indices, config, device)
        initial_state = gather_sharded_state(model)[0] if sharded else (trainable_state(model) if rank == 0 else None)
    initial_loss = resume['initial_loss'] if resume else best_loss
    if rank == 0:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_json(dest / 'config.json', config)
        atomic_json(dest / 'validation_ids.json', [r['id'] for r in validation])
        if not resume:
            atomic_checkpoint(dest / 'adapter.pt', {'trainable': initial_state, 'config': config,
                              'selected_step': 0, 'input_hashes': input_hashes, 'code': execution_code})
        atomic_json(dest / 'execution.json', {'code': execution_code, 'input_hashes': input_hashes,
                                            'resumed_from_step': begin, 'initial_loss': initial_loss,
                                            'compute_contract': compute_contract, **input_digest_execution()})
        if finite_epochs is not None:
            atomic_json(dest / 'training_data.json', finite_epochs.contract())
        handle = (dest / 'training.jsonl').open('a' if resume else 'w')
    if not resume:
        del initial_state
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
        if finite_epochs is None:
            records = sample_batch(rows['train'], mixture, rng, config['batch_size'], rank, world, pools)
            data_batch = None
        else:
            records, data_batch = finite_epochs.batch(step, rank)
        inputs = batch_inputs(tokenizer, records, config['max_tokens'], device, config.get('mode', 'bridge'),
                              config.get('board_tokens', False), config.get('board_text'), **answer_profile_kwargs(config))
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
        for offset, micro_inputs in training_microbatches(inputs, len(records), micro_size, padding_policy):
            batch = records[offset:offset + micro_size]
            micro_tokens = (micro_inputs['labels'][:, 1:] != -100).sum()
            micro_units = len(batch) if normalization == 'example' else micro_tokens
            synchronize = offset + micro_size >= len(records)
            # FSDP reduces every microbatch so gradients remain sharded while
            # accumulating; no_sync would retain full gradient replicas.
            context = wrapped.no_sync() if world > 1 and not synchronize and not sharded else nullcontext()
            with context, model.board_context(select_features(cache, indices, batch, device)), model_autocast(device, autocast_cache_setting(config)):
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
        if sharded:
            # Resolve the distributed scalar on every rank before rank-zero
            # logging; converting a Partial DTensor only there would deadlock.
            norm = norm.full_tensor()
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
            if data_batch is not None:
                item['data_batch'] = data_batch
            handle.write(json.dumps(item) + '\n')
            handle.flush()
            if completed % 20 == 0 or completed == 1:
                print(json.dumps(item), flush=True)
        check = completed % config.get('eval_every', 100) == 0 or completed == config['steps'] or completed == args.stop_after
        stop = False
        if check:
            if isinstance(cache, FeatureStore):
                cache.check_unchanged()
            if world > 1:
                dist.barrier()
            if rank == 0 or sharded:
                current = evaluate(model, tokenizer, validation, cache, indices, config, device)
            candidate_state = gather_sharded_state(model)[0] if sharded else None
            if rank == 0:
                if current < best_loss - config.get('min_delta', 0.001):
                    best_loss, best_step, patience_count = current, completed, 0
                    atomic_checkpoint(dest / 'adapter.pt', {'trainable': candidate_state if sharded else trainable_state(model), 'config': config,
                                      'selected_step': best_step, 'input_hashes': input_hashes, 'code': execution_code})
                else:
                    patience_count += 1
                stop = patience_count >= config.get('patience', 5) and completed >= config.get('min_steps', config['steps'])
                print(json.dumps({'event': 'validation', 'step': completed, 'nll': current, 'best_step': best_step}), flush=True)
            if world > 1:
                state = [best_loss, best_step, stop]
                dist.broadcast_object_list(state, src=0)
                best_loss, best_step, stop = state
            if sharded:
                latest_state, latest_optimizer = gather_sharded_state(model, optimizer)
            else:
                consolidate_optimizer_state(optimizer, config)
            cuda_states = [None] * world
            if world > 1:
                dist.all_gather_object(cuda_states, torch.cuda.get_rng_state(device))
            else:
                cuda_states = [torch.cuda.get_rng_state(device)]
            if rank == 0:
                if memory_settings.get('optimizer_state_offload') == 'cpu':
                    atomic_json(dest / f'optimizer-placement-step-{completed}.json',
                                dict(actual_step=completed, **optimizer.placement_summary()))
                atomic_checkpoint(dest / 'latest.pt', {'trainable': latest_state if sharded else trainable_state(model),
                            'optimizer': latest_optimizer if sharded else optimizer.state_dict(), 'step': completed,
                            'config': config, 'best_loss': best_loss, 'best_step': best_step, 'tokens_seen': tokens_seen,
                            'random_state': rng.getstate(), 'torch_rng_state': torch.get_rng_state(),
                            'cuda_rng_states': cuda_states, 'world_size': world, 'input_hashes': input_hashes,
                            'initial_loss': initial_loss, 'patience_count': patience_count, 'code': execution_code,
                            'compute_contract': compute_contract,
                            **({'data_state': finite_epochs.state_at(completed)} if finite_epochs is not None else {})})
            if sharded:
                del candidate_state, latest_state, latest_optimizer
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
    if rank == 0 or sharded:
        assert all(torch.equal(saved, parameter_prefix(parameter)) for parameter, saved in frozen)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        selected = torch.load(dest / 'adapter.pt', map_location='cpu', weights_only=True)
        if sharded:
            load_sharded_trainable(model, selected)
        else:
            load_trainable(model, selected)
        del selected
        after = evaluate(model, tokenizer, validation, cache, indices, config, device)
        zero = evaluate(model, tokenizer, validation, cache, indices, config, device, zero=True)
        shuffled = evaluate(model, tokenizer, validation, cache, indices, config, device, rotate=True)
        examples = generate_examples(model, tokenizer, validation[:config.get('generation_examples', 12)], cache,
                                     indices, device, config.get('mode', 'bridge'), config.get('max_new_tokens', 64),
                                     config.get('board_tokens', False), config.get('board_text'),
                                     autocast_cache_enabled=autocast_cache_setting(config), synced_gpus=sharded,
                                     **answer_profile_kwargs(config))
    if rank == 0:
        atomic_json(dest / 'examples.json', examples)
        result = {'mode': config.get('mode', 'bridge'), 'courses': list(mixture), 'mixture': mixture,
                  'steps_completed': completed, 'selected_step': best_step, 'world_size': world,
                  'global_batch_size': config['batch_size'], 'supervised_tokens_seen': tokens_seen,
                  'training_loss_normalization': normalization, 'validation_loss_normalization': 'token',
                  **compute_contract,
                  'optimizer_foreach_by_group': {g['name']: g['foreach'] for g in optimizer.param_groups},
                  **({'optimizer_runtime_state_placement': optimizer.placement_summary()}
                     if memory_settings.get('optimizer_state_offload') == 'cpu' else {}),
                  'gradient_parameters': sum(p.numel() for p in parameters if p.grad is not None),
                  'gradient_view_parameters': sum(p.numel() for p in parameters
                                                  if p.grad is not None and p.grad._is_view()),
                  'initial_validation_nll': initial_loss, 'final_validation_nll': after,
                  'zero_memory_validation_nll': zero, 'shuffled_memory_validation_nll': shuffled,
                  'shuffled_memory_validation_pairing': {
                      'method': 'global_feature_multiset_derangement_v1',
                      'examples': len(validation), 'different_history_pairs': len(validation),
                      'independent_of_eval_batch_size': True},
                  'trainable_parameters': sum(p.numel() for p in parameters),
                  'trainable_parameter_dtypes': {str(dtype): sum(p.numel() for p in parameters if p.dtype == dtype)
                                               for dtype in {p.dtype for p in parameters}},
                  'frozen_base_verified': config.get('decoder_training', 'frozen') == 'frozen',
                  'decoder_training': config.get('decoder_training', 'frozen'),
                  'decoder_parameter_dtype': str(model.base.get_input_embeddings().weight.dtype),
                  'frozen_parameter_dtypes': {str(dtype): sum(p.numel() for p, _ in frozen if p.dtype == dtype)
                                             for dtype in {p.dtype for p, _ in frozen}},
                  'compute_dtype': 'torch.bfloat16',
                  'selected_decoder_probe_max_change': (float((parameter_prefix(decoder_probe) - decoder_before).abs().max())
                                                        if decoder_probe is not None else None),
                  'frozen_parameter_gradients_verified': True,
                  'nonzero_gradients': any(n > 0 for n in norms), 'generation_examples': len(examples),
                  'generation_exact_matches': sum(e['exact_match'] for e in examples),
                  'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                  'seconds': time.monotonic() - started, 'purpose': config['purpose']}
        if sharded:
            result['decoder_probe_scope'] = 'rank_zero_local_parameter_shard'
        result.update(answer_profile_kwargs(config))
        inputs = [p / f'{s}.jsonl' for p in data_roots for s in ['train', 'validation']]
        if config.get('init_from'):
            inputs.append(config['init_from'])
        inputs += [p for directory in index_roots.values() for p in index_paths(directory)]
        if is_feature_store(config['feature_path']):
            inputs += feature_paths(config['feature_path'])
        result['feature_cache_sha256'] = input_hashes[config['feature_path']]
        if finite_epochs is not None:
            result['training_data_state'] = finite_epochs.state_at(completed)
        atomic_json(dest / 'manifest.json', manifest('model_training', config, inputs,
                    [dest / 'adapter.pt', dest / 'examples.json', dest / 'training.jsonl',
                     *([dest / 'training_data.json'] if finite_epochs is not None else [])], result, code=execution_code))
        atomic_json(dest / 'metrics.json', result)
        print(json.dumps(result), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    with input_session_from_environment():
        return _main()


if __name__ == '__main__':
    main()
