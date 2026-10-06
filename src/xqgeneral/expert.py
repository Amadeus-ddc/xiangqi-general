"""Frozen Px0 attention-body weights, following the official Px0 backend formulas.

The pilot deliberately accepts only the verified classical-input 20x512 asset.
Newer AB-legacy / dense-position-embedding formats require a separate adapter.
"""
import collections
import gzip
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import net_pb2
from .rules import START_FEN, piece_map

PIECE_ORDER = "racphek"
FEATURE_DEPTHS = (4, 9, 14, 19)


def decode(layer):
    encoding = layer.encoding or 1
    if encoding == 1:
        values = np.frombuffer(layer.params, dtype="<u2").astype(np.float32)
        theta = values / np.float32(65535)
        values = np.float32(layer.min_val) * (1 - theta) + np.float32(layer.max_val) * theta
    elif encoding == 2:
        values = np.frombuffer(layer.params, dtype="<f2").astype(np.float32)
    elif encoding == 4:
        values = np.frombuffer(layer.params, dtype="<f4").copy()
    else:
        raise ValueError(f"Unsupported Px0 layer encoding: {encoding}")
    return torch.from_numpy(values)


def encode_history(history):
    """Px0 INPUT_CLASSICAL_112_PLANE actually has 124 planes on a 10x9 board."""
    if not history:
        raise ValueError("A root position is required")
    planes = torch.zeros(124, 10, 9)
    black = history[-1].split()[1] == "b"
    repeats = collections.Counter()
    repetition_counts = []
    for fen in history:
        key = " ".join(fen.split()[:2])
        repetition_counts.append(repeats[key])
        repeats[key] += 1
    starts_at_initial = history[0].split()[0] == START_FEN.split()[0] and history[0].split()[1] == "w"
    for step in range(8):
        index = len(history) - 1 - step
        if index < 0 and starts_at_initial:
            break
        index = max(index, 0)
        board = piece_map(history[index])
        for square, piece in board.items():
            ours = piece.islower() if black else piece.isupper()
            channel = step * 15 + (0 if ours else 7) + PIECE_ORDER.index(piece.lower())
            rank = int(square[1])
            if black:
                rank = 9 - rank
            planes[channel, rank, ord(square[0]) - 97] = 1.0
        if repetition_counts[index] >= 1:
            planes[step * 15 + 14].fill_(1.0)
    planes[120].fill_(float(black))
    planes[121].fill_(float(history[-1].split()[4]))
    planes[123].fill_(1.0)
    return planes


class Buffers(nn.Module):
    def add(self, name, layer, shape=None):
        value = decode(layer)
        if shape is not None:
            value = value.reshape(shape)
        self.register_buffer(name, value)


class Px0Layer(Buffers):
    def __init__(self, proto, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim = dim
        for stem in ["q", "k", "v", "dense"]:
            self.add(stem + "_w", getattr(proto.mha, stem + "_w"), (dim, dim))
            self.add(stem + "_b", getattr(proto.mha, stem + "_b"))
        for field in ["ln1_gammas", "ln1_betas", "ln2_gammas", "ln2_betas"]:
            self.add(field, getattr(proto, field))
        ff_dim = decode(proto.ffn.dense1_b).numel()
        self.add("ff1_w", proto.ffn.dense1_w, (ff_dim, dim))
        self.add("ff1_b", proto.ffn.dense1_b)
        self.add("ff2_w", proto.ffn.dense2_w, (dim, ff_dim))
        self.add("ff2_b", proto.ffn.dense2_b)
        if any(getattr(proto.mha, f).params for f in ["rpe_q", "rpe_k", "rpe_v"]):
            raise ValueError("Relative-position-weight networks need a separate adapter")
        sm = proto.mha.smolgen
        hidden = decode(sm.dense1_b).numel()
        generated = decode(sm.dense2_b).numel()
        compressed = decode(sm.compress).numel() // dim
        self.add("compress", sm.compress, (compressed, dim))
        self.add("sm1_w", sm.dense1_w, (hidden, 90 * compressed))
        self.add("sm1_b", sm.dense1_b)
        self.add("sm2_w", sm.dense2_w, (generated, hidden))
        self.add("sm2_b", sm.dense2_b)
        for source, target in [("ln1_gammas", "sm1_g"), ("ln1_betas", "sm1_betas"),
                               ("ln2_gammas", "sm2_g"), ("ln2_betas", "sm2_betas")]:
            self.add(target, getattr(sm, source))

    def forward(self, x, global_weights, alpha):
        b = x.shape[0]
        q, k, v = [F.linear(x, getattr(self, f"{s}_w"), getattr(self, f"{s}_b"))
                   .view(b, 90, self.heads, self.dim // self.heads).transpose(1, 2)
                   for s in ["q", "k", "v"]]
        compressed = F.linear(x, self.compress).flatten(1)
        sm = F.silu(F.linear(compressed, self.sm1_w, self.sm1_b))
        sm = F.layer_norm(sm, (sm.shape[-1],), self.sm1_g, self.sm1_betas, eps=1e-3)
        sm = F.silu(F.linear(sm, self.sm2_w, self.sm2_b))
        sm = F.layer_norm(sm, (sm.shape[-1],), self.sm2_g, self.sm2_betas, eps=1e-3)
        bias = F.linear(sm.reshape(b, self.heads, -1), global_weights).reshape(b, self.heads, 90, 90)
        attention = (q @ k.transpose(-1, -2)) * (self.dim // self.heads) ** -0.5 + bias
        attended = (attention.softmax(-1) @ v).transpose(1, 2).reshape(b, 90, self.dim)
        out = F.linear(attended, self.dense_w, self.dense_b)
        x = F.layer_norm(x + alpha * out, (self.dim,), self.ln1_gammas, self.ln1_betas, eps=1e-6)
        ff = F.linear(F.relu(F.linear(x, self.ff1_w, self.ff1_b)), self.ff2_w, self.ff2_b)
        return F.layer_norm(x + alpha * ff, (self.dim,), self.ln2_gammas, self.ln2_betas, eps=1e-6)


class FrozenPx0(Buffers):
    def __init__(self, path):
        super().__init__()
        raw = Path(path).read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        net = net_pb2.Net()
        net.ParseFromString(raw)
        fmt = net.format.network_format
        weights = net.weights
        if (fmt.network, fmt.input, fmt.default_activation, fmt.input_embedding) != (6, 1, 0, 0):
            raise ValueError(f"Unsupported network format; use the pinned 20x512 asset: {fmt}")
        if weights.residual or weights.ip_emb_preproc_w.params or len(weights.encoder) != 20:
            raise ValueError("This adapter supports the pinned pure attention network only")
        if fmt.ffn_activation not in (0, 2) or fmt.smolgen_activation != 7:
            raise ValueError("Unexpected FFN or Smolgen activation")
        self.dim = decode(weights.ip_emb_b).numel()
        self.depth = len(weights.encoder)
        self.heads = weights.headcount
        self.add("embedding_w", weights.ip_emb_w, (self.dim, 124))
        self.add("embedding_b", weights.ip_emb_b)
        self.add("mult_gate", weights.ip_mult_gate, (self.dim, 90))
        self.add("add_gate", weights.ip_add_gate, (self.dim, 90))
        self.add("global_weights", weights.smolgen_w, (90 * 90, -1))
        self.layers = nn.ModuleList([Px0Layer(p, self.dim, self.heads) for p in weights.encoder])
        val_dim = decode(weights.ip_val_b).numel()
        val_hidden = decode(weights.ip1_val_b).numel()
        self.add("val_w", weights.ip_val_w, (val_dim, self.dim))
        self.add("val_b", weights.ip_val_b)
        self.add("val1_w", weights.ip1_val_w, (val_hidden, 90 * val_dim))
        self.add("val1_b", weights.ip1_val_b)
        self.add("val2_w", weights.ip2_val_w, (3, val_hidden))
        self.add("val2_b", weights.ip2_val_b)
        if weights.pol_encoder or weights.ip4_pol_w.params:
            raise ValueError("This asset must use the verified Xiangqi attention policy head")
        pol_dim = decode(weights.ip_pol_b).numel()
        pol_model = decode(weights.ip2_pol_b).numel()
        self.add("pol_w", weights.ip_pol_w, (pol_dim, self.dim))
        self.add("pol_b", weights.ip_pol_b)
        for index, stem in [(2, "pol_q"), (3, "pol_k")]:
            self.add(stem + "_w", getattr(weights, f"ip{index}_pol_w"), (pol_model, pol_dim))
            self.add(stem + "_b", getattr(weights, f"ip{index}_pol_b"))
        mov_dim = decode(weights.ip_mov_b).numel()
        mov_hidden = decode(weights.ip1_mov_b).numel()
        self.add("mov_w", weights.ip_mov_w, (mov_dim, self.dim))
        self.add("mov_b", weights.ip_mov_b)
        self.add("mov1_w", weights.ip1_mov_w, (mov_hidden, 90 * mov_dim))
        self.add("mov1_b", weights.ip1_mov_b)
        self.add("mov2_w", weights.ip2_mov_w, (1, mov_hidden))
        self.add("mov2_b", weights.ip2_mov_b)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, planes, depths=FEATURE_DEPTHS, all_outputs=False):
        if planes.shape[1:] != (124, 10, 9):
            raise ValueError(f"Expected Bx124x10x9 planes, got {planes.shape}")
        x = planes.flatten(2).transpose(1, 2)
        x = F.relu(F.linear(x, self.embedding_w, self.embedding_b))
        x = x * self.mult_gate.T + self.add_gate.T
        features = []
        alpha = (2.0 * self.depth) ** -0.25
        for i, layer in enumerate(self.layers):
            x = layer(x, self.global_weights, alpha)
            if i in depths:
                # Present all memory tokens to the LM in absolute a0..i9 order.
                absolute = x.reshape(-1, 10, 9, self.dim)
                black = planes[:, 120, 0, 0].bool()
                absolute = torch.where(black[:, None, None, None], absolute.flip(1), absolute)
                features.append(absolute.reshape(-1, 90, self.dim).clone())
        val = F.relu(F.linear(x, self.val_w, self.val_b)).flatten(1)
        val = F.relu(F.linear(val, self.val1_w, self.val1_b))
        wdl = F.linear(val, self.val2_w, self.val2_b).softmax(-1)
        if all_outputs:
            policy = F.relu(F.linear(x, self.pol_w, self.pol_b))
            q = F.linear(policy, self.pol_q_w, self.pol_q_b)
            k = F.linear(policy, self.pol_k_w, self.pol_k_b)
            policy = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
            mov = F.relu(F.linear(x, self.mov_w, self.mov_b)).flatten(1)
            mov = F.relu(F.linear(mov, self.mov1_w, self.mov1_b))
            moves_left = F.relu(F.linear(mov, self.mov2_w, self.mov2_b))
            return features, {"wdl": wdl, "policy_matrix": policy, "moves_left": moves_left}
        return features, wdl
