import pytest
import torch
from torch import nn
from xqgeneral.board_tokens import BoardInputEmbedding


def test_float_master_board_embedding_preserves_frozen_decoder_input_dtype():
    frozen = nn.Embedding(4, 2, dtype=torch.bfloat16)
    embedding = BoardInputEmbedding(frozen, [3], torch.ones(1, 2, dtype=torch.bfloat16), 4)
    embedding.board_weight.data = embedding.board_weight.data.float()
    output = embedding(torch.tensor([[0, 3]]))
    assert output.dtype == torch.bfloat16
    assert torch.equal(output[0, 0], frozen.weight[0])
    optimizer = torch.optim.AdamW([embedding.board_weight], lr=1e-4, weight_decay=0)
    embedding.board_weight.grad = torch.ones_like(embedding.board_weight)
    optimizer.step()
    assert not torch.equal(embedding.board_weight, torch.ones_like(embedding.board_weight))


def test_float_master_updates_keep_frozen_weights_and_resume_contract():
    from xqgeneral.modeling import configure_trainable_precision
    from xqgeneral.training import compatible_resume
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).to(dtype=torch.bfloat16)
    model[0].requires_grad_(False)
    frozen = model[0].weight.detach().clone()
    model[1].weight.data.fill_(1)
    configure_trainable_precision(model, 'float32')
    assert model[0].weight.dtype == torch.bfloat16
    assert torch.equal(model[0].weight, frozen)
    assert model[1].weight.dtype == torch.float32
    optimizer = torch.optim.AdamW(model[1].parameters(), lr=1e-4, weight_decay=0)
    model[1].weight.grad = torch.ones_like(model[1].weight)
    optimizer.step()
    assert float(model[1].weight[0, 0].detach()) == pytest.approx(.9999)
    rounded = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    half_optimizer = torch.optim.AdamW([rounded], lr=1e-4, weight_decay=0)
    rounded.grad = torch.ones_like(rounded)
    half_optimizer.step()
    assert float(rounded.detach()) == 1
    compatible_resume({}, {'trainable_parameter_dtype': 'base'})
    with pytest.raises(ValueError, match='precision'):
        compatible_resume({}, {'trainable_parameter_dtype': 'float32'})
    with pytest.raises(ValueError, match='precision'):
        configure_trainable_precision(model, 'float16')
