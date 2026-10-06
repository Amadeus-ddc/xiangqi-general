import torch
from torch import nn
from xqgeneral.board_tokens import BoardInputEmbedding, BoardOutputHead, decode_board_text, encode_board_text


def test_coordinate_piece_codec_round_trips_json_and_adjacent_ucci():
    text = '{"move":"b0c2","facts":{"piece":"红马","captured":"空"}}'
    assert '<XQ_b0><XQ_c2>' in encode_board_text(text)
    assert decode_board_text(encode_board_text(text)) == text


def test_added_embeddings_train_without_changing_pretrained_rows():
    torch.manual_seed(3)
    old_input, old_output = nn.Embedding(5, 4), nn.Linear(4, 5, bias=False)
    saved = old_input.weight.detach().clone(), old_output.weight.detach().clone()
    emb = BoardInputEmbedding(old_input, [3, 5], torch.randn(2, 4), 6)
    head = BoardOutputHead(old_output, [3, 5], torch.randn(2, 4), 6)
    x = emb(torch.tensor([[1, 3, 5]]))
    assert torch.equal(x[0, 0], saved[0][1])
    opt = torch.optim.SGD([p for m in [emb, head] for p in m.parameters() if p.requires_grad], lr=0.1)
    loss = head(x).sum(); loss.backward(); opt.step()
    assert emb.board_weight.grad.abs().sum() > 0
    assert head.board_weight.grad.abs().sum() > 0
    assert torch.equal(old_input.weight, saved[0]) and torch.equal(old_output.weight, saved[1])
    assert head(x).shape == (1, 3, 6)
