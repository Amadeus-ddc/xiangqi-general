"""Control cast-weight caching without changing FP32 trainables or inputs."""
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from test_decoder_checkpointing import loaded,batch
from xqgeneral import training


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(previous)


def test_weight_cache_is_explicit_and_cannot_change_during_exact_resume():
    baseline=training.training_memory_settings({})
    assert training.training_memory_settings({'autocast_cache_enabled':True})==baseline
    change={'autocast_cache_enabled':False}
    with pytest.raises(ValueError,match='memory settings'):training.compatible_resume({},change)
    with pytest.raises(ValueError,match='memory settings'):training.compatible_resume(change,{})
    training.compatible_resume(change,deepcopy(change))
    previous=torch.is_autocast_cache_enabled()
    with training.model_autocast('cpu',cache_enabled=False):assert not torch.is_autocast_cache_enabled()
    assert torch.is_autocast_cache_enabled()==previous


@pytest.mark.parametrize('value',[None,1,'false'])
def test_invalid_weight_cache_control_rejected_before_loading(value):
    with pytest.raises(ValueError,match='autocast_cache_enabled must be a boolean'):
        training.training_memory_settings({'autocast_cache_enabled':value})


@pytest.mark.parametrize('enabled',[False,True])
def test_validation_forward_receives_the_requested_weight_cache(monkeypatch,enabled):
    model=loaded(monkeypatch,True);values,features=batch()
    rows=[dict(feature_key='a'),dict(feature_key='b')]
    cache={'keys':['a','b'],'features':features}
    monkeypatch.setattr(training,'batch_inputs',lambda *args,**kwargs:values)
    # Feature gathering follows the GPU BF16 contract; use real CPU BF16 AMP
    # for this device-independent check of the evaluation call boundary.
    monkeypatch.setattr(training,'model_autocast',lambda device,cache_enabled=True:
        torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=cache_enabled))
    seen=[];hook=model.register_forward_pre_hook(lambda *args:seen.append(torch.is_autocast_cache_enabled()))
    config=dict(batch_size=2,eval_batch_size=2,max_tokens=32,autocast_cache_enabled=enabled)
    loss=training.evaluate(model,None,rows,cache,{'a':0,'b':1},config,'cpu')
    hook.remove();assert seen==[enabled] and torch.isfinite(torch.tensor(loss))


def test_real_BF16_recomputed_dense_decoder_preserves_loss_gradients_and_rng(monkeypatch):
    results=[]
    for enabled in [True,False]:
        model=loaded(monkeypatch,True,dropout=0.1);values,features=batch()
        with torch.random.fork_rng():
            torch.manual_seed(813)
            with model.board_context(features),torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=enabled):
                loss=model(**values).loss;loss.backward()
            rng=torch.get_rng_state().clone()
        gradients={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
        assert all(p.dtype==torch.float32 for p in model.parameters())
        results.append((loss.detach(),gradients,rng))
    left,right=results;torch.testing.assert_close(left[0],right[0],rtol=0,atol=0)
    assert left[1].keys()==right[1].keys()
    for name,value in left[1].items():torch.testing.assert_close(value,right[1][name],rtol=0,atol=0)
    assert torch.equal(left[2],right[2])


@pytest.mark.parametrize('enabled',[False,True])
def test_generation_receives_the_requested_weight_cache(monkeypatch,enabled):
    seen=[]
    def generate(**kwargs):
        seen.append(torch.is_autocast_cache_enabled())
        return torch.tensor([[1,2,3]])
    class Inputs(dict):
        def to(self,device):return self
    class Tokenizer:
        pad_token_id=0
        def apply_chat_template(self,*args,**kwargs):return 'prompt'
        def __call__(self,*args,**kwargs):return Inputs(input_ids=torch.tensor([[1,2]]))
        def decode(self,*args,**kwargs):return 'answer'
    model=SimpleNamespace(eval=lambda:None,board_context=lambda *args:nullcontext(),
                          base=SimpleNamespace(generate=generate))
    row=dict(id='a',game_id='g',stage='s',question='q',answer='answer')
    monkeypatch.setattr(training,'messages',lambda *args,**kwargs:[])
    monkeypatch.setattr(training,'select_features',lambda *args,**kwargs:None)
    result=training.generate_examples(model,Tokenizer(),[row],None,None,'cpu',
                                     autocast_cache_enabled=enabled)
    assert seen==[enabled] and result[0]['exact_match']
