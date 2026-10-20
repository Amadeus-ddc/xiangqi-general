"""CPU supervision, causal-model and resume controls; no pretrained weights."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

from xqgeneral import training


def inputs(shift=0):
    lengths = [11, 4, 5, 8, 3]
    ids = (torch.arange(65).reshape(5, 13) + shift) % 29 + 1
    mask = torch.arange(13)[None, :] < torch.tensor(lengths)[:, None]
    ids = ids.masked_fill(~mask, 0)
    labels = ids.clone().masked_fill(~mask, -100)
    labels[:, :2] = -100
    return dict(input_ids=ids, attention_mask=mask.long(), labels=labels, use_cache=False)


def test_default_keeps_legacy_rank_padding_and_old_resume_defaults():
    value = inputs()
    assert training.microbatch_padding_policy({}) == 'rank'
    batches = list(training.training_microbatches(value, 5, 2))
    assert [offset for offset, _ in batches] == [0, 2, 4]
    for offset, batch in batches:
        assert batch['use_cache'] is False and batch['input_ids'].shape[1] == 13
        for key in ['input_ids', 'labels', 'attention_mask']:
            assert torch.equal(batch[key], value[key][offset:offset + 2])
    training.compatible_resume({}, {'microbatch_padding': 'rank'})


def test_supervised_eos_is_preserved_when_its_id_equals_the_padding_id():
    value = inputs()
    value['input_ids'][4, 2] = 0
    value['labels'][4, 2] = 0
    final = list(training.training_microbatches(value, 5, 2, 'micro'))[-1][1]
    assert final['input_ids'].shape == (1, 3)
    assert final['attention_mask'][0, 2] == 1 and final['labels'][0, 2] == 0


@pytest.mark.parametrize('micro', [1, 2, 8])
def test_cropping_preserves_order_valid_tokens_targets_and_scalar_options(micro):
    value = inputs();original = deepcopy(value)
    batches = list(training.training_microbatches(value, 5, micro, 'micro'))
    seen, targets = [], 0
    for offset, batch in batches:
        size = batch['input_ids'].shape[0]
        width = int(value['attention_mask'][offset:offset + size].sum(dim=1).max())
        assert batch['input_ids'].shape == (size, width)
        assert batch['use_cache'] is False
        for key in ['input_ids', 'labels', 'attention_mask']:
            assert torch.equal(batch[key], value[key][offset:offset + size, :width])
        seen.extend(batch['input_ids'][:, 0].tolist())
        targets += int((batch['labels'][:, 1:] != -100).sum())
    assert seen == value['input_ids'][:, 0].tolist()
    assert targets == int((value['labels'][:, 1:] != -100).sum())
    assert all(torch.equal(value[key], original[key]) for key in ['input_ids', 'labels', 'attention_mask'])


@pytest.mark.parametrize('problem', ['left_padding', 'hole', 'nonbinary', 'empty_row',
    'supervised_padding', 'different_shape', 'extra_tensor'])
def test_unsupported_masks_and_supervision_fail_without_truncation(problem):
    value = inputs()
    if problem == 'left_padding':value['attention_mask'][0] = torch.roll(value['attention_mask'][0], 2)
    elif problem == 'hole':value['attention_mask'][0, 3] = 0
    elif problem == 'nonbinary':value['attention_mask'][0, 3] = 2
    elif problem == 'empty_row':value['attention_mask'][0] = 0
    elif problem == 'supervised_padding':value['labels'][0, -1] = 9
    elif problem == 'different_shape':value['labels'] = value['labels'][:, :-1]
    else:value['position_ids'] = torch.arange(13).repeat(5, 1)
    with pytest.raises(ValueError):
        list(training.training_microbatches(value, 5, 2, 'micro'))


@pytest.mark.parametrize('policy', [None, True, 3, [], 'automatic'])
def test_invalid_policy_is_rejected_before_cuda_or_model_loading(tmp_path, monkeypatch, policy):
    path = tmp_path / 'config.json';path.write_text(json.dumps({'microbatch_padding': policy}))
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(path)])
    monkeypatch.setenv('WORLD_SIZE', '4');monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.delenv('XQGENERAL_VERIFIED_INPUT_RECEIPT', raising=False)
    def unexpected(*args, **kwargs):pytest.fail('Invalid padding policy reached CUDA/model loading')
    monkeypatch.setattr(torch.cuda, 'set_device', unexpected)
    monkeypatch.setattr(training, 'load_model', unexpected)
    with pytest.raises(ValueError, match='microbatch_padding'):
        training.main()


@pytest.mark.parametrize('saved,requested', [({}, {'microbatch_padding': 'micro'}),
    ({'microbatch_padding': 'micro'}, {}),
    ({'microbatch_padding': 'micro'}, {'microbatch_padding': 'rank'})])
def test_exact_resume_refuses_a_padding_policy_change(saved, requested):
    with pytest.raises(ValueError, match='Resume microbatch padding'):
        training.compatible_resume(saved, requested)


def tiny_model(frozen=True):
    transformers = pytest.importorskip('transformers')
    from xqgeneral.bridge import BoardLanguageModel
    torch.manual_seed(1701)
    config = transformers.Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64, attention_dropout=0.0)
    config._attn_implementation = 'sdpa'
    base = transformers.Qwen3ForCausalLM(config)
    model = BoardLanguageModel(base, 4, positions=[0, 1], width=16,
        freeze_decoder=frozen, bridge_architecture='flamingo_dense', bridge_heads=2).double()
    with torch.no_grad():
        for bridge in model.bridges:
            bridge.alpha_attention.fill_(0.2);bridge.alpha_ff.fill_(0.15)
    model.train()
    return model


def memory(step=0):
    generator = torch.Generator().manual_seed(71 + step)
    return [torch.randn(5, 90, 4, generator=generator, dtype=torch.float64) for _ in range(2)]


def objective(logits, labels, normalization):
    # Independent double-precision reference, including causal label shifting.
    labels = labels[:, 1:]
    loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels,
                           reduction='none', ignore_index=-100)
    if normalization == 'example':
        return (loss.sum(dim=1) / (labels != -100).sum(dim=1)).mean()
    return loss.sum() / (labels != -100).sum()


def gradients(model, value, features, normalization, micro=None, policy='rank'):
    model.zero_grad(set_to_none=True)
    total = value['input_ids'].shape[0] if normalization == 'example' else int((value['labels'][:, 1:] != -100).sum())
    batches = [(0, value)] if micro is None else training.training_microbatches(value, 5, micro, policy)
    weighted = torch.zeros((), dtype=torch.float64)
    outputs = []
    for offset, batch in batches:
        size = batch['input_ids'].shape[0]
        with model.board_context([item[offset:offset + size] for item in features]):
            logits = model(**{key: item for key, item in batch.items() if key != 'labels'}).logits
        units = size if normalization == 'example' else int((batch['labels'][:, 1:] != -100).sum())
        loss = objective(logits, batch['labels'], normalization) * units / total
        loss.backward();weighted += loss.detach()
        outputs.append((offset, batch['attention_mask'].bool(), logits.detach()))
    values = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
              if parameter.requires_grad}
    assert any(value.abs().sum() > 0 for name, value in values.items() if name.startswith('bridges.'))
    return weighted, values, outputs


@pytest.mark.parametrize('frozen', [True, False])
@pytest.mark.parametrize('normalization', ['token', 'example'])
@pytest.mark.parametrize('micro', [1, 2, 8])
def test_tiny_qwen_dense_bridges_preserve_valid_logits_loss_and_gradients(frozen, normalization, micro):
    reference = tiny_model(frozen)
    expected_loss, expected_gradients, full_outputs = gradients(reference, inputs(), memory(), normalization)
    full_logits = full_outputs[0][2]
    for policy in ['rank', 'micro']:
        candidate = tiny_model(frozen);candidate.load_state_dict(reference.state_dict())
        loss, actual_gradients, outputs = gradients(candidate, inputs(), memory(), normalization, micro, policy)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-9, atol=1e-10)
        assert actual_gradients.keys() == expected_gradients.keys()
        for name, value in actual_gradients.items():
            torch.testing.assert_close(value, expected_gradients[name], rtol=1e-8, atol=1e-10, msg=name)
        for offset, mask, logits in outputs:
            expected = full_logits[offset:offset + logits.shape[0], :logits.shape[1]]
            torch.testing.assert_close(logits[mask], expected[mask], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('frozen', [True, False])
@pytest.mark.parametrize('normalization', ['token', 'example'])
def test_cropped_tiny_qwen_adamw_restoration_matches_continuous_updates(frozen, normalization):
    continuous, partial = tiny_model(frozen), tiny_model(frozen)
    def optimizer(model):return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003, foreach=False)
    left, right = optimizer(continuous), optimizer(partial)
    def update(model, opt, step):
        gradients(model, inputs(step), memory(step), normalization, micro=2, policy='micro')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    for step in range(4):
        update(continuous, left, step)
        if step < 2:update(partial, right, step)
    checkpoint = dict(model=deepcopy(partial.state_dict()), optimizer=deepcopy(right.state_dict()),
                      config={'microbatch_padding': 'micro'}, rng=torch.get_rng_state())
    restored = tiny_model(frozen);restored.load_state_dict(checkpoint['model'])
    resumed = optimizer(restored);resumed.load_state_dict(checkpoint['optimizer'])
    training.compatible_resume(checkpoint['config'], {'microbatch_padding': 'micro'})
    torch.set_rng_state(checkpoint['rng'])
    for step in range(2, 4):update(restored, resumed, step)
    for name, value in continuous.state_dict().items():assert torch.equal(value, restored.state_dict()[name]), name
    a, b = left.state_dict(), resumed.state_dict()
    assert a['param_groups'] == b['param_groups']
    for key, state in a['state'].items():
        for name, value in state.items():assert torch.equal(value, b['state'][key][name])


def test_course_recipe_preserves_explicit_padding_policy(tmp_path):
    from xqgeneral.gated_curriculum import training_config, validate_recipe
    recipe = json.loads((Path(__file__).parents[1] / 'configs/foundation-paper-v1.json').read_text())
    recipe['microbatch_padding'] = 'micro'
    validate_recipe(recipe)
    assert training_config(recipe, 0, tmp_path, None)['microbatch_padding'] == 'micro'


def test_sft_recipe_preserves_explicit_padding_policy_and_validates_before_data(tmp_path):
    from xqgeneral.sft import prepare_config
    data = tmp_path / 'data';data.mkdir()
    (data / 'train.jsonl').write_text('{"stage":"explanation"}\n')
    recipe = dict(data_path=str(data), stages=['explanation'], mixture={'explanation': 1.0},
        epochs=1, max_steps=4, min_steps=1, batch_size=4, micro_batch_size=1, microbatch_padding='micro')
    saved = {'mode': 'bridge', 'bridge_width': 6, 'decoder_bridge_positions': [0]}
    assert prepare_config(saved, recipe, tmp_path / 'parent.pt', tmp_path / 'out')['microbatch_padding'] == 'micro'
    recipe.update(microbatch_padding='unknown', data_path=str(tmp_path / 'missing'))
    with pytest.raises(ValueError, match='microbatch_padding'):
        prepare_config(saved, recipe, tmp_path / 'parent.pt', tmp_path / 'out')
