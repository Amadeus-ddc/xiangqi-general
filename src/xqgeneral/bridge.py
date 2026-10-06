from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


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
    def __init__(self, base, expert_dim, positions=(3, 11, 19, 27), width=384):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.positions = tuple(positions)
        self.bridges = nn.ModuleList([GatedBridge(base.config.hidden_size, expert_dim, width) for _ in positions])
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
        self.base.eval()
        return self

    def forward(self, **kwargs):
        return self.base(**kwargs)
