from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


def bridge_settings(config):
    """Canonical architecture identity, including defaults in historical configs."""
    architecture = config.get('bridge_architecture', 'bottleneck')
    if architecture not in {'bottleneck', 'flamingo_dense'}:
        raise ValueError('Unknown bridge architecture')
    heads = config.get('bridge_heads', 6 if architecture == 'bottleneck' else 16)
    if type(heads) is not int or heads < 1:
        raise ValueError('Bridge head count must be a positive integer')
    return {'architecture': architecture, 'heads': heads}


class SquaredReLU(nn.Module):
    def forward(self, hidden):
        return F.relu(hidden).square()


class DenseBridge(nn.Module):
    """Paper Flamingo block: decoder-width attention and a 2x ReLU-squared FFN.

    Both gates start at zero and output projections retain random initialization.
    This chooses the paper initialization, rather than the released code's
    alternative nonzero gate and zero attention-output projection.
    """
    def __init__(self, decoder_dim, expert_dim, heads=16):
        super().__init__()
        if (any(type(n) is not int or n < 1 for n in (decoder_dim, expert_dim, heads)) or
                decoder_dim % heads):
            raise ValueError('Dense bridge dimensions must be positive and decoder width divisible by heads')
        self.width, self.expert_dim, self.heads = decoder_dim, expert_dim, heads
        self.query_norm = nn.LayerNorm(decoder_dim)
        self.memory_norm = nn.LayerNorm(expert_dim)
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.k = nn.Linear(expert_dim, decoder_dim, bias=False)
        self.v = nn.Linear(expert_dim, decoder_dim, bias=False)
        self.out = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.ff_norm = nn.LayerNorm(decoder_dim)
        self.ff = nn.Sequential(nn.Linear(decoder_dim, 2 * decoder_dim, bias=False),
                                SquaredReLU(), nn.Linear(2 * decoder_dim, decoder_dim, bias=False))
        self.alpha_attention = nn.Parameter(torch.zeros(1))
        self.alpha_ff = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, memory):
        if hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError('Dense bridge expects batch x text x decoder-width hidden states')
        batch, text, _ = hidden.shape
        if memory.shape != (batch, 90, self.expert_dim):
            raise ValueError('Dense bridge requires one 90-square expert context per text example')
        query = self.q(self.query_norm(hidden)).view(batch, text, self.heads, -1).transpose(1, 2)
        normalized = self.memory_norm(memory)
        key = self.k(normalized).view(batch, 90, self.heads, -1).transpose(1, 2)
        value = self.v(normalized).view(batch, 90, self.heads, -1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value).transpose(1, 2).reshape(batch, text, self.width)
        hidden = hidden + self.alpha_attention.tanh().to(hidden.dtype) * self.out(attended)
        return hidden + self.alpha_ff.tanh().to(hidden.dtype) * self.ff(self.ff_norm(hidden))


def make_bridge(decoder_dim, expert_dim, width, architecture='bottleneck', heads=None):
    config = {'bridge_architecture': architecture}
    if heads is not None:
        config['bridge_heads'] = heads
    settings = bridge_settings(config)
    if architecture == 'flamingo_dense':
        if width != decoder_dim:
            raise ValueError('Paper dense attention width must equal the decoder width')
        return DenseBridge(decoder_dim, expert_dim, settings['heads'])
    return GatedBridge(decoder_dim, expert_dim, width, settings['heads'])


class GatedBridge(nn.Module):
    """A smaller pilot of Queen's gated cross-attention, keeping 90 memory tokens."""
    def __init__(self, decoder_dim, expert_dim, width=384, heads=6):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("Bridge width must be divisible by the attention head count")
        self.heads = heads
        self.width = width
        self.query_norm = nn.LayerNorm(decoder_dim)
        self.memory_norm = nn.LayerNorm(expert_dim)
        self.square_embedding = nn.Parameter(torch.zeros(90, expert_dim))
        self.q = nn.Linear(decoder_dim, width, bias=False)
        self.kv = nn.Linear(expert_dim, width * 2, bias=False)
        self.out = nn.Linear(width, decoder_dim, bias=False)
        self.ff_norm = nn.LayerNorm(decoder_dim)
        self.ff = nn.Sequential(nn.Linear(decoder_dim, width), nn.GELU(), nn.Linear(width, decoder_dim))
        self.alpha_attention = nn.Parameter(torch.ones(()))
        self.alpha_ff = nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, hidden, memory):
        b, t, _ = hidden.shape
        if memory.shape[:2] != (b, 90):
            raise ValueError(f"Expected {b} x 90 board memory, got {memory.shape}")
        q = self.q(self.query_norm(hidden)).view(b, t, self.heads, -1).transpose(1, 2)
        k, v = self.kv(self.memory_norm(memory + self.square_embedding)).chunk(2, dim=-1)
        k = k.view(b, 90, self.heads, -1).transpose(1, 2)
        v = v.view(b, 90, self.heads, -1).transpose(1, 2)
        value = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, self.width)
        hidden = hidden + self.alpha_attention.tanh() * self.out(value)
        return hidden + self.alpha_ff.tanh() * self.ff(self.ff_norm(hidden))


class BoardLanguageModel(nn.Module):
    def __init__(self, base, expert_dim, positions=(3, 11, 19, 27), width=384, freeze_decoder=True,
                 bridge_architecture='bottleneck', bridge_heads=None):
        super().__init__()
        self.base = base
        self.freeze_decoder = freeze_decoder
        if freeze_decoder:
            self.base.requires_grad_(False)
        self.positions = tuple(positions)
        self.bridges = nn.ModuleList([make_bridge(base.config.hidden_size, expert_dim, width,
                                                 bridge_architecture, bridge_heads) for _ in positions])
        self.memory = None
        self.handles = []
        layers = base.model.layers
        if len(set(positions)) != len(positions) or any(p < 0 or p >= len(layers) for p in positions):
            raise ValueError("Bridge positions must be distinct, valid decoder layer indices")
        for slot, position in enumerate(positions):
            def inject(module, args, kwargs, slot=slot):
                if self.memory is None:
                    raise RuntimeError("Set board context before calling the language model")
                if args:
                    updated = self.bridges[slot](args[0], self.memory[slot])
                    return (updated,) + args[1:], kwargs
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = self.bridges[slot](kwargs["hidden_states"], self.memory[slot])
                return args, kwargs
            self.handles.append(layers[position].register_forward_pre_hook(inject, with_kwargs=True))

    @contextmanager
    def board_context(self, features):
        if len(features) != len(self.bridges):
            raise ValueError("One expert feature level is required per bridge")
        previous = self.memory
        self.memory = features
        try:
            yield
        finally:
            self.memory = previous

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_decoder:
            self.base.eval()
        return self

    def forward(self, **kwargs):
        return self.base(**kwargs)
