import torch
from xqgeneral.bridge import GatedBridge


def test_initial_identity_then_memory_changes_output_and_receives_gradients():
    torch.manual_seed(123)
    bridge = GatedBridge(48, 16, width=24, heads=6)
    x = torch.randn(2, 5, 48)
    m = torch.randn(2, 90, 16)
    assert torch.equal(bridge(x, m), x)
    opt = torch.optim.AdamW(bridge.parameters(), lr=0.01)
    for _ in range(3):
        opt.zero_grad()
        loss = (bridge(x, m) - 1).square().mean()
        loss.backward()
        opt.step()
    assert bridge.kv.weight.grad.abs().sum() > 0
    assert not torch.allclose(bridge(x, m), bridge(x, m.flip(0)))
