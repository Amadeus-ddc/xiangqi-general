import pytest
import torch
from torch import nn
from xqgeneral.modeling import initialize_trainable


def test_full_sft_preserves_pretrained_base_and_requires_every_bridge():
    model = nn.Module()
    model.base = nn.Linear(3, 3)
    model.base.board_weight = nn.Parameter(torch.randn(2, 3))
    model.bridges = nn.ModuleList([nn.Linear(3, 3)])
    saved_base = model.base.weight.detach().clone()
    config = {'mode': 'bridge', 'decoder_bridge_positions': [0], 'bridge_width': 3,
              'board_tokens': True, 'model_revision': 'fixed', 'decoder_training': 'full'}
    state = {n: torch.ones_like(p) for n, p in model.named_parameters()
             if n.startswith('bridges.') or n.endswith('board_weight')}
    checkpoint = {'config': dict(config, decoder_training='frozen'), 'trainable': state}
    initialize_trainable(model, checkpoint, config)
    assert torch.equal(saved_base, model.base.weight)
    assert all(torch.equal(dict(model.named_parameters())[n], v) for n, v in state.items())
    with pytest.raises(ValueError, match='all bridges'):
        initialize_trainable(model, dict(checkpoint, trainable={k: v for k, v in state.items()
                                                              if k != 'bridges.0.bias'}), config)
