"""Finite author-style course mixtures and reproducible distributed epoch order."""
import hashlib
import json
import math

import numpy as np
import torch

LEGACY_PROFILE = 'replacement_task_balanced_v1'
FINITE_PROFILE = 'author_finite_mixture_epochs_xiangqi_v1'


def data_profile(config):
    profile = config.get('training_data_profile', LEGACY_PROFILE)
    if profile not in (LEGACY_PROFILE, FINITE_PROFILE):
        raise ValueError('Unknown training data traversal profile')
    if 'mixture_seed' in config and profile != FINITE_PROFILE:
        raise ValueError('A mixture seed requires the finite training data profile')
    return profile


def mixture_counts(counts, mixture):
    """Keep the largest-weight source whole, rounding the other source targets."""
    if (not isinstance(mixture, dict) or not mixture or
            any(not isinstance(s, str) or not s or type(w) not in (int, float) or
                not math.isfinite(w) or w <= 0 for s, w in mixture.items())):
        raise ValueError('Finite mixture stages and weights must be positive and finite')
    if any(type(counts.get(s, 0)) is not int or counts.get(s, 0) < 0 for s in mixture):
        raise ValueError('Finite source row counts must be nonnegative integers')
    total_weight = sum(mixture.values())
    if not math.isfinite(total_weight):
        raise ValueError('Finite mixture weight sum must remain finite')
    stages = list(mixture)
    weights = {s: mixture[s] / total_weight for s in stages}
    anchor = max(stages, key=weights.get)
    size = counts.get(anchor, 0)
    if not size:
        raise ValueError('Finite mixture anchor source cannot be empty')
    selected = {s: size if s == anchor else round(size * weights[s] / weights[anchor]) for s in stages}
    for stage, target in selected.items():
        if target > counts.get(stage, 0):
            raise ValueError(f'Finite mixture needs {target} rows from {stage}, only {counts.get(stage, 0)} available')
    total = sum(selected.values())
    return {'anchor_stage': anchor, 'source_rows_by_stage': {s: counts.get(s, 0) for s in stages},
            'selected_rows_by_stage': selected, 'requested_weights': weights,
            'realized_weights': {s: n / total for s, n in selected.items()}, 'rows_per_epoch': total}


class FiniteMixtureEpochs:
    """Select old-course subsets once, then address every optimizer batch by step."""
    def __init__(self, rows, mixture, *, seed, global_batch_size, world=1, micro_batch_size=None,
                 mixture_seed=0):
        if any(type(v) is not int or v < 1 for v in (global_batch_size, world)):
            raise ValueError('Finite distributed batch geometry is invalid')
        if not isinstance(mixture, dict) or not mixture:
            raise ValueError('Finite mixture must declare its stage weights')
        micro = global_batch_size // world
        micro = micro if micro_batch_size is None else micro_batch_size
        if (any(type(v) is not int or v < 1 for v in (global_batch_size, world, micro)) or
                global_batch_size % (world * micro) or
                any(type(v) is not int or v < 0 for v in (seed, mixture_seed))):
            raise ValueError('Finite seeds, distributed batch and microbatch geometry are invalid')
        self.rows, self.seed, self.mixture_seed = rows, seed, mixture_seed
        self.batch_size, self.world, self.micro = global_batch_size, world, micro
        self.global_micro = world * micro
        groups = {stage: [] for stage in mixture}
        seen = set()
        metadata = rows.iter_metadata() if hasattr(rows, 'iter_metadata') else rows
        for index, row in enumerate(metadata):
            if row['stage'] not in groups:
                continue
            identity = row.get('id')
            if not isinstance(identity, str) or not identity or identity in seen or row.get('split', 'train') != 'train':
                raise ValueError('Finite training rows require unique IDs and the training split')
            seen.add(identity)
            groups[row['stage']].append(index)
        self.mix = mixture_counts({s: len(v) for s, v in groups.items()}, mixture)
        pieces = []
        for stage, source in groups.items():
            indices = np.asarray(source, dtype=np.int64)
            if stage != self.mix['anchor_stage']:
                order = np.random.default_rng(mixture_seed).permutation(len(source))
                indices = indices[order[:self.mix['selected_rows_by_stage'][stage]]]
            pieces.append(indices)
        self.selected = np.concatenate(pieces)
        self.size = len(self.selected)
        self.steps_per_epoch = math.ceil(self.size / global_batch_size)
        self.padded_per_epoch = (-self.size) % self.global_micro if world > 1 else 0
        identity = hashlib.sha256()
        for index in self.selected:
            row = rows.row_metadata(int(index)) if hasattr(rows, 'row_metadata') else rows[int(index)]
            identity.update(json.dumps([int(index), row['id']],
                                       ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
        self.selection_sha256 = identity.hexdigest()
        self._epoch, self._order = None, None

    def contract(self):
        return {'profile': FINITE_PROFILE, **self.mix, 'mixture_seed': self.mixture_seed,
                'epoch_seed': self.seed, 'selected_row_id_and_index_sha256': self.selection_sha256,
                'global_batch_size': self.batch_size, 'world_size': self.world,
                'micro_batch_size': self.micro, 'optimizer_steps_per_epoch': self.steps_per_epoch,
                'padded_rows_per_epoch': self.padded_per_epoch,
                'tail_policy': 'pad_distributed_microbatch_from_epoch_start' if self.world > 1 else 'retain_partial_microbatch',
                'old_course_subsets_reselected_each_epoch': False,
                'task_types_resampled_uniformly': False, 'independent_test_used': False}

    def state_at(self, completed_steps):
        if type(completed_steps) is not int or completed_steps < 0:
            raise ValueError('Finite completed steps must be a nonnegative integer')
        epoch, offset = divmod(completed_steps, self.steps_per_epoch)
        return {'contract': self.contract(), 'epoch': epoch, 'optimizer_batch_in_epoch': offset,
                'unique_rows_consumed': epoch * self.size + offset * self.batch_size,
                'padding_rows_consumed': epoch * self.padded_per_epoch,
                'global_rows_consumed': epoch * (self.size + self.padded_per_epoch) + offset * self.batch_size}

    def check_resume(self, state, completed_steps):
        if state != self.state_at(completed_steps):
            raise ValueError('Resume finite data selection, geometry or epoch position differs')

    def batch_indices(self, step, rank=0):
        if type(step) is not int or step < 0 or type(rank) is not int or not 0 <= rank < self.world:
            raise ValueError('Finite batch step or distributed rank is invalid')
        epoch, batch = divmod(step, self.steps_per_epoch)
        if self._epoch != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._order = torch.randperm(self.size, generator=generator).numpy()
            self._epoch = epoch
        start = batch * self.batch_size
        order = self._order[start:start + self.batch_size]
        valid = len(order)
        padding = (-valid) % self.global_micro if self.world > 1 else 0
        if padding:
            order = np.concatenate([order, np.resize(self._order, padding)])
        indices = self.selected[order]
        if self.world > 1:
            indices = indices.reshape(-1, self.world, self.micro)[:, rank, :].reshape(-1)
        return indices.tolist(), {'epoch': epoch, 'optimizer_batch_in_epoch': batch,
                                  'global_valid_rows': valid, 'global_padding_rows': padding,
                                  'global_rows': len(order)}

    def batch(self, step, rank=0):
        indices, report = self.batch_indices(step, rank)
        records = self.rows.get_batch(indices) if hasattr(self.rows, 'get_batch') else [self.rows[index] for index in indices]
        return records, report


def prepare_finite_epochs(rows, mixture, config, world):
    if data_profile(config) == LEGACY_PROFILE:
        return None
    return FiniteMixtureEpochs(rows, mixture, seed=config['seed'], mixture_seed=config.get('mixture_seed', 0),
                               global_batch_size=config['batch_size'], world=world,
                               micro_batch_size=config.get('micro_batch_size'))
