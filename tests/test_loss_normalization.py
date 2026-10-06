import math
import pytest
import torch
from xqgeneral.training import compatible_resume, configure_determinism, example_normalized_loss


def test_sample_weight_does_not_disappear_for_short_answers_and_accumulation_agrees():
    logits = torch.tensor([[[2.,0.]]*4]*2, requires_grad=True)
    labels = torch.tensor([[-100,0,-100,-100],[-100,1,1,1]])
    loss = example_normalized_loss(logits,labels)
    assert float(loss.detach()) == pytest.approx(math.log1p(math.exp(-2))+1)
    full_gradient = torch.autograd.grad(loss,logits)[0]
    micro_logits = logits.detach().clone().requires_grad_(True)
    for i in range(2):
        (example_normalized_loss(micro_logits[i:i+1],labels[i:i+1])/2).backward()
    torch.testing.assert_close(full_gradient,micro_logits.grad)
    with pytest.raises(ValueError,match='supervised target'):
        example_normalized_loss(logits,torch.full_like(labels,-100))


def test_resume_cannot_change_answer_length_weighting():
    compatible_resume({}, {'loss_normalization':'token'})
    with pytest.raises(ValueError,match='normalization'):
        compatible_resume({'loss_normalization':'token'}, {'loss_normalization':'example'})


def test_determinism_is_explicit_and_cannot_change_on_resume(monkeypatch):
    flags = []
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', flags.append)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    assert configure_determinism(False) == {'deterministic_training': False, 'cublas_workspace_config': None}
    assert configure_determinism(True) == {'deterministic_training': True, 'cublas_workspace_config': ':4096:8'}
    assert flags == [False, True]
    compatible_resume({}, {'deterministic_training': False})
    with pytest.raises(ValueError, match='determinism'):
        compatible_resume({}, {'deterministic_training': True})
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'invalid')
    with pytest.raises(ValueError, match='CUBLAS'):
        configure_determinism(True)
    with pytest.raises(ValueError, match='boolean'):
        configure_determinism('true')
