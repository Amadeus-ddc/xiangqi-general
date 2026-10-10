from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xqgeneral.board_tokens import PAIRS
from xqgeneral.bridge import BoardLanguageModel, DenseBridge, GatedBridge, bridge_settings, make_bridge
from xqgeneral.evidence import write_jsonl
from xqgeneral.modeling import foundation_state_summary, initialize_trainable, trainable_state
from xqgeneral.sft import prepare_config
from xqgeneral.training import compatible_resume


@pytest.fixture(scope='module', autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_zero_gates_open_without_zero_projection_deadlock():
    torch.manual_seed(2048)
    bridge = DenseBridge(48, 16)
    hidden, memory = torch.randn(2, 5, 48), torch.randn(2, 90, 16)
    assert torch.equal(bridge(hidden, memory), hidden)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=.01, weight_decay=0)
    gate_gradients = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = (bridge(hidden, memory) - .75).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        gate_gradients.append([float(p.grad.abs().sum()) for p in
                               (bridge.alpha_attention, bridge.alpha_ff)])
        optimizer.step()
    assert all(g > 0 for g in gate_gradients[0])
    assert bridge.k.weight.grad.abs().sum() > 0 and bridge.v.weight.grad.abs().sum() > 0
    assert not torch.allclose(bridge(hidden, memory), bridge(hidden, memory.flip(0)))


@pytest.mark.parametrize('decoder,expert,expected', [(2048, 1024, 469925920), (2560, 512, 671268896)])
def test_full_width_geometry_and_parameter_count_without_weight_allocation(decoder, expert, expected):
    with torch.device('meta'):
        bridge = make_bridge(decoder, expert, decoder, 'flamingo_dense')
    assert 16 * sum(p.numel() for p in bridge.parameters()) == expected
    assert bridge.heads == 16 and bridge.q.out_features == decoder
    assert bridge.ff[0].out_features == 2 * decoder
    assert all(p.device.type == 'meta' for p in bridge.parameters())


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=48)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(48, 48), nn.Linear(48, 48)])

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_dense_hook_training_preserves_frozen_decoder_and_restores_context():
    torch.manual_seed(90)
    base = Decoder()
    original = {n: p.detach().clone() for n, p in base.named_parameters()}
    model = BoardLanguageModel(base, 16, positions=[0, 1], width=48, bridge_architecture='flamingo_dense')
    hidden, memory = torch.randn(2, 5, 48), [torch.randn(2, 90, 16) for _ in range(2)]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model.train()
    assert not base.training
    for _ in range(3):
        optimizer.zero_grad()
        with model.board_context(memory):
            model(hidden_states=hidden).square().mean().backward()
        assert model.memory is None
        optimizer.step()
    assert all(p.grad is None and torch.equal(p, original[n]) for n, p in base.named_parameters())
    assert all(b.k.weight.grad.abs().sum() > 0 for b in model.bridges)
    with model.board_context(memory):
        normal = model(hidden_states=hidden)
    with model.board_context([m.flip(0) for m in memory]):
        shuffled = model(hidden_states=hidden)
    assert not torch.allclose(normal, shuffled)
    with pytest.raises(RuntimeError, match='board context'):
        model(hidden_states=hidden)


def test_fp32_dense_trainable_parameters_support_bfloat16_compute():
    torch.manual_seed(2560)
    bridge = DenseBridge(48, 16)
    hidden = torch.randn(2, 5, 48, dtype=torch.bfloat16)
    memory = torch.randn(2, 90, 16, dtype=torch.bfloat16)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = bridge(hidden, memory)
    assert output.dtype == torch.bfloat16 and torch.equal(output, hidden)
    output.float().square().mean().backward()
    assert all(p.dtype == torch.float32 for p in bridge.parameters())
    assert bridge.alpha_attention.grad.abs().sum() > 0


@pytest.mark.parametrize('dimensions', [(47, 16, 16), (48, 0, 16), (48, 16, 0), (True, 16, 1)])
def test_invalid_dense_dimensions_are_rejected(dimensions):
    with pytest.raises(ValueError, match='dimensions'):
        DenseBridge(*dimensions)


@pytest.mark.parametrize('shape', [(2, 64, 16), (1, 90, 16), (2, 90, 32)])
def test_incompatible_board_context_is_rejected(shape):
    with pytest.raises(ValueError, match='90-square'):
        DenseBridge(48, 16)(torch.randn(2, 5, 48), torch.randn(shape))


def test_historical_defaults_and_explicit_bottleneck_reproduce_identical_initial_state():
    assert bridge_settings({}) == bridge_settings({'bridge_architecture': 'bottleneck', 'bridge_heads': 6})
    torch.manual_seed(768)
    original = GatedBridge(48, 16, 24)
    torch.manual_seed(768)
    current = make_bridge(48, 16, 24)
    assert all(torch.equal(p, current.state_dict()[n]) for n, p in original.state_dict().items())
    compatible_resume({}, {'bridge_architecture': 'bottleneck', 'bridge_heads': 6})


@pytest.mark.parametrize('change', [{'bridge_architecture': 'flamingo_dense'}, {'bridge_heads': 8}])
def test_changed_architecture_or_same_shape_attention_heads_cannot_resume_or_initialize(change):
    model = nn.Linear(2, 2)
    checkpoint = {'config': {'mode': 'bridge'}, 'trainable': trainable_state(model)}
    with pytest.raises(ValueError, match='bridge architecture or attention heads'):
        compatible_resume(checkpoint['config'], dict(checkpoint['config'], **change))
    with pytest.raises(ValueError, match='bridge architecture or attention heads'):
        initialize_trainable(model, checkpoint, dict(checkpoint['config'], **change))


def dense_foundation():
    config = {'mode': 'bridge', 'decoder_training': 'frozen', 'board_tokens': True,
              'trainable_parameter_dtype': 'float32', 'decoder_bridge_positions': [0, 2],
              'expert_feature_depths': [0, 1], 'bridge_width': 48,
              'bridge_architecture': 'flamingo_dense', 'bridge_heads': 16}
    with torch.device('meta'):
        bridge = make_bridge(48, 512, 48, 'flamingo_dense', 16)
    state = {f'bridges.{i}.{n}': torch.zeros(p.shape) for i in range(2) for n, p in bridge.named_parameters()}
    state.update({n: torch.zeros(len(PAIRS), 48) for n in
                  ('base.model.embed_tokens.board_weight', 'base.lm_head.board_weight')})
    return {'config': config, 'trainable': state}


def test_complete_dense_foundation_cpu_handoff_checks_every_tensor():
    checkpoint = dense_foundation()
    result = foundation_state_summary(checkpoint, {'hidden_size': 48, 'num_hidden_layers': 4})
    assert result['trainable_tensors'] == 30 and not result['base_model_weights_loaded']
    assert result['all_bridge_and_board_shapes_checked'] and result['all_trainable_values_finite_fp32']
    assert result['trainable_parameters'] == sum(p.numel() for p in checkpoint['trainable'].values())


@pytest.mark.parametrize('corruption', ['missing', 'shape', 'precision', 'nonfinite', 'legacy_family'])
def test_corrupt_or_wrong_family_foundation_handoff_is_rejected(corruption):
    checkpoint = dense_foundation();state = checkpoint['trainable'];name = 'bridges.1.k.weight'
    if corruption == 'missing':
        state.pop(name)
    elif corruption == 'shape':
        state[name] = torch.zeros(1)
    elif corruption == 'precision':
        state[name] = state[name].bfloat16()
    elif corruption == 'nonfinite':
        state[name][0, 0] = float('nan')
    else:
        checkpoint['config'].update(bridge_architecture='bottleneck', bridge_heads=6)
    with pytest.raises(ValueError, match='Foundation checkpoint'):
        foundation_state_summary(checkpoint, {'hidden_size': 48, 'num_hidden_layers': 4})


def sft_inputs(tmp_path):
    write_jsonl(tmp_path / 'train.jsonl', [{'stage': 'explanation'} for _ in range(4)])
    recipe = {'data_path': str(tmp_path), 'stages': ['explanation'], 'mixture': {'explanation': 1},
              'epochs': 4, 'max_steps': 20, 'min_steps': 1, 'batch_size': 2, 'decoder_training': 'full'}
    return dense_foundation()['config'], recipe


def test_full_sft_configuration_keeps_dense_architecture_and_heads(tmp_path):
    saved, recipe = sft_inputs(tmp_path)
    result = prepare_config(saved, recipe, tmp_path / 'foundation.pt', tmp_path / 'sft')
    assert result['bridge_architecture'] == 'flamingo_dense' and result['bridge_heads'] == 16
    assert result['bridge_width'] == 48 and result['decoder_training'] == 'full'


@pytest.mark.parametrize('change', [{'bridge_architecture': 'bottleneck'}, {'bridge_heads': 8}])
def test_sft_cannot_silently_change_dense_architecture_or_heads(tmp_path, change):
    saved, recipe = sft_inputs(tmp_path)
    with pytest.raises(ValueError, match='bridge architecture and attention heads'):
        prepare_config(saved, dict(recipe, **change), tmp_path / 'foundation.pt', tmp_path / 'sft')


def test_full_decoder_initialization_preserves_dense_foundation_without_touching_base():
    torch.manual_seed(16)
    model = BoardLanguageModel(Decoder(), 16, positions=[0, 1], width=48,
                              freeze_decoder=False, bridge_architecture='flamingo_dense')
    model.base.board_weight = nn.Parameter(torch.randn(105, 48))
    base = {n: p.detach().clone() for n, p in model.base.named_parameters() if not n.endswith('board_weight')}
    config = {'mode': 'bridge', 'bridge_architecture': 'flamingo_dense', 'bridge_heads': 16,
              'bridge_width': 48, 'decoder_training': 'full'}
    state = {n: torch.full_like(p, .125) for n, p in model.named_parameters()
             if n.startswith('bridges.') or n.endswith('board_weight')}
    initialize_trainable(model, {'config': dict(config, decoder_training='frozen'), 'trainable': state}, config)
    assert all(torch.equal(p, dict(model.named_parameters())[n]) for n, p in state.items())
    assert all(torch.equal(p, base[n]) for n, p in model.base.named_parameters() if n in base)


@pytest.mark.parametrize('config', [{'bridge_architecture': 'unknown'}, {'bridge_heads': True}, {'bridge_heads': 0}])
def test_invalid_architecture_settings_are_rejected(config):
    with pytest.raises(ValueError, match='bridge|Bridge'):
        bridge_settings(config)


def test_dense_width_must_match_decoder_width():
    with pytest.raises(ValueError, match='equal the decoder width'):
        make_bridge(48, 16, 24, 'flamingo_dense')
