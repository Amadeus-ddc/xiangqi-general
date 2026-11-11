"""Exercise the real dense bridge on CPU with training-rounded board memory."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xqgeneral.bridge import BoardLanguageModel
from xqgeneral.evidence import history_key
from xqgeneral.expert import encode_history
from xqgeneral.inference import Predictor
from xqgeneral.rules import START_FEN, replay


class Tokens(dict):
    def to(self, device):
        return Tokens({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    padding_side = 'right'
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]['content']

    def __call__(self, texts, **kwargs):
        return Tokens(input_ids=torch.tensor([[1, 2]] * len(texts)),
                      attention_mask=torch.ones((len(texts), 2), dtype=torch.long))

    def decode(self, tokens, **kwargs):
        return str(tokens.item())


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])
        self.embedding = nn.Embedding(5, 8)
        self.output = nn.Linear(8, 5)
        self.hidden = None

    def generate(self, input_ids, **kwargs):
        hidden = self.model.layers[0](self.embedding(input_ids))
        self.hidden = hidden.detach().clone()
        token = self.output(hidden[:, -1]).argmax(-1, keepdim=True)
        return torch.cat([input_ids, token], dim=1)


@pytest.mark.parametrize('device', ['cpu', torch.device('cpu')])
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('memory', ['normal', 'zero', 'shuffled'])
def test_cpu_generation_dense_bridge_preserves_rounded_features_without_autocast(device, cached, memory):
    torch.manual_seed(71)
    histories = [replay(START_FEN, [move]) for move in ['b0c2', 'h0g2']]
    records = [dict(initial_fen=START_FEN, moves=[move], history=history,
                    fen=history[-1], question='当前局面的特点是什么？')
               for move, history in zip(['b0c2', 'h0g2'], histories, strict=True)]
    raw = [torch.tensor([1.004, -1.004, 0.502, 2.008]).repeat(90).reshape(1, 90, 4),
           torch.tensor([-0.502, 2.008, 1.004, -1.004]).repeat(90).reshape(1, 90, 4)]
    rounded = torch.cat(raw).half().bfloat16()
    assert not torch.equal(torch.cat(raw).bfloat16(), rounded)

    class Expert:
        calls = 0

        def __call__(self, planes, *, depths):
            assert depths == [2]
            assert torch.equal(planes, encode_history(histories[self.calls]).unsqueeze(0))
            value = raw[self.calls]
            self.calls += 1
            return [value], torch.zeros((1, 3))

    predictor = Predictor.__new__(Predictor)
    predictor.device = device
    predictor.config = {'mode': 'bridge', 'expert_feature_depths': [2]}
    predictor.tokenizer = Tokenizer()
    predictor.expert = Expert()
    predictor.cache = {'features': [torch.cat(raw).half()]} if cached else None
    predictor.indices = {history_key(h): i for i, h in enumerate(histories)} if cached else {}
    base = Decoder()
    predictor.model = BoardLanguageModel(base, 4, positions=[0], width=8,
                                         freeze_decoder=False, bridge_architecture='flamingo_dense',
                                         bridge_heads=2).eval()
    with torch.no_grad():
        predictor.model.bridges[0].alpha_attention.fill_(0.6)
        predictor.model.bridges[0].alpha_ff.fill_(0.2)
    expected = rounded.float()
    if memory == 'zero':
        expected = torch.zeros_like(expected)
    elif memory == 'shuffled':
        expected = expected.roll(1, 0)
    with torch.no_grad(), predictor.model.board_context([expected]):
        reference = base.generate(torch.tensor([[1, 2], [1, 2]]))[:, -1]
        reference_hidden = base.hidden.clone()

    observed = []
    handle = predictor.model.bridges[0].register_forward_pre_hook(
        lambda module, args: observed.append(args[1].detach().clone()))
    try:
        assert not torch.is_autocast_enabled('cpu')
        answers = predictor.generate_batch(records, memory=memory, max_new_tokens=1)
        assert not torch.is_autocast_enabled('cpu')
    finally:
        handle.remove()
    assert answers == [str(token.item()) for token in reference]
    assert torch.equal(base.hidden, reference_hidden)
    assert len(observed) == 1 and observed[0].dtype == torch.float32
    assert torch.equal(observed[0], expected)
    assert predictor.expert.calls == (0 if cached else 2)
    assert predictor.model.memory is None
    assert predictor.tokenizer.padding_side == 'right'
