"""Actual small Qwen decoder/latent bridges; no downloaded weights or GPU."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from xqgeneral import modeling,training


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(previous)


def loaded(monkeypatch,enabled,dropout=0.0):
    transformers=pytest.importorskip('transformers')
    def base(*args,**kwargs):
        config=transformers.Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
            head_dim=8,max_position_embeddings=64,attention_dropout=dropout)
        config._attn_implementation='sdpa'
        return transformers.Qwen3ForCausalLM(config)
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',base)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',
        lambda *args,**kwargs:SimpleNamespace(eos_token='EOS',pad_token=None))
    config=dict(model_path='controlled-local-base',mode='bridge',decoder_training='full',
        decoder_bridge_positions=[0,1],bridge_width=16,bridge_architecture='flamingo_dense',
        bridge_heads=2,board_tokens=False,trainable_parameter_dtype='float32',
        decoder_gradient_checkpointing=enabled)
    with torch.random.fork_rng():
        torch.manual_seed(1701);model,_=modeling.load_model(config,expert_dim=4,device='cpu')
    with torch.no_grad():
        for bridge in model.bridges:
            bridge.alpha_attention.fill_(0.2);bridge.alpha_ff.fill_(0.15)
    model.train()
    return model


def batch(step=0):
    generator=torch.Generator().manual_seed(91+step)
    ids=torch.randint(0,32,(2,11),generator=generator)
    labels=ids.clone();labels[:,:3]=-100
    features=[torch.randn(2,90,4,generator=generator) for _ in range(2)]
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False),features


@pytest.mark.parametrize('enabled',[False,True])
def test_loader_enables_explicit_nonreentrant_full_decoder_without_changing_parameters(monkeypatch,enabled):
    model=loaded(monkeypatch,enabled)
    assert model.base.is_gradient_checkpointing is enabled
    assert model.base.config.use_cache is True
    assert all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())
    reference=loaded(monkeypatch,False)
    assert model.state_dict().keys()==reference.state_dict().keys()
    assert all(torch.equal(value,reference.state_dict()[name]) for name,value in model.state_dict().items())


@pytest.mark.parametrize('setting',[None,1,'true'])
def test_invalid_decoder_checkpointing_fails_before_loading(monkeypatch,setting):
    with pytest.raises(ValueError,match='must be a boolean'):
        modeling.load_model(dict(model_path='never-loaded',decoder_training='full',decoder_gradient_checkpointing=setting),device='cpu')


def test_checkpointing_requires_an_explicit_full_decoder_before_loading(monkeypatch):
    with pytest.raises(ValueError,match='full decoder'):
        modeling.load_model(dict(model_path='never-loaded',decoder_gradient_checkpointing=True),device='cpu')


@pytest.mark.parametrize('normalization',['token','example'])
def test_real_recomputation_preserves_dense_bridge_loss_gradients_and_dropout_rng(monkeypatch,normalization):
    plain=loaded(monkeypatch,False,dropout=0.1);checkpointed=loaded(monkeypatch,True,dropout=0.1)
    assert checkpointed.base.is_gradient_checkpointing
    results=[]
    for model in [plain,checkpointed]:
        calls=[];handle=model.base.model.layers[0].register_forward_pre_hook(lambda *args:calls.append(1))
        values,features=batch()
        with torch.random.fork_rng():
            torch.manual_seed(441)
            with model.board_context(features):
                if normalization=='example':
                    output=model(**{k:v for k,v in values.items() if k!='labels'})
                    loss=training.example_normalized_loss(output.logits,values['labels'])
                else:loss=model(**values).loss
                loss.backward()
            rng=torch.get_rng_state().clone()
        handle.remove();assert model.memory is None
        gradients={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
        results.append((loss.detach(),gradients,rng,len(calls)))
    left,right=results
    assert right[3]>left[3]
    torch.testing.assert_close(left[0],right[0],rtol=0,atol=0)
    assert left[1].keys()==right[1].keys()
    for name,value in left[1].items():torch.testing.assert_close(value,right[1][name],rtol=0,atol=0)
    assert torch.equal(left[2],right[2])
    assert any(name.startswith('bridges.') and value.abs().sum()>0 for name,value in right[1].items())


def test_checkpointed_decoder_optimizer_and_rng_roundtrip_is_exact(monkeypatch):
    model=loaded(monkeypatch,True,dropout=0.1)
    assert model.base.is_gradient_checkpointing
    config=dict(learning_rate=.01,decoder_learning_rate=.001,token_learning_rate=.002,
        optimizer_foreach=False,decoder_training='full',decoder_gradient_checkpointing=True)
    optimizer=training.training_optimizer(model,config)
    def update(network,opt,step):
        values,features=batch(step);opt.zero_grad(set_to_none=True)
        with network.board_context(features):network(**values).loss.backward()
        opt.step()
    torch.manual_seed(531)
    for step in range(4):
        update(model,optimizer,step)
        if step==1:saved=dict(model=deepcopy(model.state_dict()),optimizer=deepcopy(optimizer.state_dict()),rng=torch.get_rng_state())
    expected_rng=torch.get_rng_state().clone()
    resumed=loaded(monkeypatch,True,dropout=0.1);resumed.load_state_dict(saved['model'])
    opt=training.training_optimizer(resumed,config);opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng'])
    for step in range(2,4):update(resumed,opt,step)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    for name,value in model.state_dict().items():assert torch.equal(value,resumed.state_dict()[name])
    left,right=optimizer.state_dict(),opt.state_dict();assert left['param_groups']==right['param_groups']
    for index,state in left['state'].items():
        for name,value in state.items():assert torch.equal(value,right['state'][index][name])


def test_exact_resume_cannot_change_decoder_recomputation():
    off=dict(decoder_training='full');on=dict(off,decoder_gradient_checkpointing=True)
    with pytest.raises(ValueError,match='memory settings'):training.compatible_resume(off,on)
    with pytest.raises(ValueError,match='memory settings'):training.compatible_resume(on,off)
    training.compatible_resume(on,deepcopy(on))
    training.compatible_resume(off,dict(off,decoder_gradient_checkpointing=False))
