import pytest
import torch
from torch import nn
from xqgeneral import modeling
from xqgeneral.modeling import expanded_initialization_reference, initialize_trainable


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


def test_full_parent_reference_preserves_every_decoder_tensor_without_expansion(tmp_path, monkeypatch):
    model = nn.Linear(3, 2)
    config = {'mode': 'plain', 'decoder_training': 'full', 'trainable_parameter_dtype': 'float32'}
    state = {name: torch.full_like(p, 7) for name, p in model.named_parameters()}
    source = tmp_path / 'full-parent.pt'
    torch.save({'config': config, 'trainable': state}, source)
    monkeypatch.setattr(modeling, 'load_model', lambda *args, **kwargs: (model, None))
    result = expanded_initialization_reference(config, source)
    assert result.keys() == state.keys()
    assert all(torch.equal(value, state[name]) for name, value in result.items())


def test_full_parent_reference_refuses_nonfinite_decoder_weights(tmp_path, monkeypatch):
    model = nn.Linear(3, 2)
    config = {'mode': 'plain', 'decoder_training': 'full', 'trainable_parameter_dtype': 'float32'}
    state = {name: p.detach().clone() for name, p in model.named_parameters()}
    state['weight'][0, 0] = float('nan')
    source = tmp_path / 'full-parent.pt'
    torch.save({'config': config, 'trainable': state}, source)
    monkeypatch.setattr(modeling, 'load_model', lambda *args, **kwargs: (model, None))
    with pytest.raises(ValueError, match='every parent tensor'):
        expanded_initialization_reference(config, source)
