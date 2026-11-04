"""Real four-rank AdamW partitioning and fixed-gradient checkpoint restoration."""
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from xqgeneral import training


def configuration(sharded):
    return dict(learning_rate=.003, decoder_learning_rate=.001, token_learning_rate=.002,
                optimizer_foreach=False, optimizer_state_sharding='zero1' if sharded else 'none')


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_weight=nn.Parameter(torch.randn(16,16))
        self.bridges=nn.ModuleList([nn.Linear(16,16),nn.LayerNorm(16)])
        self.decoder=nn.Sequential(nn.Linear(16,16),nn.Dropout(.1),nn.Linear(16,16))

    def forward(self,values):
        value=values @ self.board_weight
        for bridge in self.bridges:value=bridge(value)
        return self.decoder(value)


@pytest.mark.parametrize('value',[None,True,'zero2'])
def test_invalid_optimizer_partition_policy_is_rejected(value):
    with pytest.raises(ValueError,match='optimizer_state_sharding'):
        training.training_memory_settings({'optimizer_state_sharding':value})


def test_default_optimizer_contract_and_exact_resume_stay_explicit():
    legacy=training.training_memory_settings({})
    assert training.training_memory_settings({'optimizer_state_sharding':'none'})==legacy
    changed={'optimizer_state_sharding':'zero1'}
    assert training.training_memory_settings(changed)['optimizer_state_sharding']=='zero1'
    with pytest.raises(ValueError,match='memory settings'):training.compatible_resume({},changed)
    with pytest.raises(ValueError,match='memory settings'):training.compatible_resume(changed,{})
    training.compatible_resume(changed,deepcopy(changed))


def test_zero1_requires_an_initialized_multi_rank_group():
    assert not dist.is_initialized()
    with pytest.raises(ValueError,match='initialized.*multi'):
        training.training_optimizer(Network(),configuration(True))


def test_zero1_refuses_a_single_rank_group(tmp_path):
    dist.init_process_group('gloo',rank=0,world_size=1,
        init_method=(tmp_path/'single-group').as_uri(),timeout=timedelta(seconds=30))
    try:
        with pytest.raises(ValueError,match='initialized.*multi'):
            training.training_optimizer(Network(),configuration(True))
    finally:dist.destroy_process_group()


def worker(rank,home):
    from pathlib import Path
    from torch.distributed.optim import ZeroRedundancyOptimizer
    import torch.distributed.optim.zero_redundancy_optimizer as zero_module
    root=Path(home)
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=4,
        init_method=(root/'group').as_uri(),timeout=timedelta(seconds=90))
    original_broadcast=zero_module._broadcast_object
    def reject_legacy_gpu_checkpoint(*args,**kwargs):
        raise AssertionError('Optimizer checkpoints must use CPU collection, not the native GPU object broadcast')
    zero_module._broadcast_object=reject_legacy_gpu_checkpoint
    try:
        torch.manual_seed(217)
        plain=Network();sharded=deepcopy(plain)
        left=DistributedDataParallel(plain,gradient_as_bucket_view=True)
        right=DistributedDataParallel(sharded,gradient_as_bucket_view=True)
        dense=training.training_optimizer(plain,configuration(False))
        split=training.training_optimizer(sharded,configuration(True))
        assert isinstance(split,ZeroRedundancyOptimizer)
        assert split._overlap_with_ddp is False
        def update(network,optimizer,step):
            # Identical rank inputs isolate optimizer/checkpoint behavior from
            # Gloo reduction-order rounding when DDP rebuilds its buckets.
            # The production gate separately requires exact real-data CUDA
            # four-step versus two-plus-two continuation.
            generator=torch.Generator().manual_seed(513+step)
            value=torch.randn(3,16,generator=generator)
            target=torch.randn(3,16,generator=generator)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=group['base_lr']/(step+1)
            loss=(network(value)-target).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.0)
            optimizer.step()
            return loss.detach()
        torch.manual_seed(181)
        snapshots={}
        for step in range(4):
            before=torch.get_rng_state()
            dense_loss=update(left,dense,step)
            after=torch.get_rng_state()
            torch.set_rng_state(before)
            split_loss=update(right,split,step)
            assert torch.equal(after,torch.get_rng_state())
            assert torch.equal(dense_loss,split_loss)
            for name,value in plain.state_dict().items():assert torch.equal(value,sharded.state_dict()[name])
            snapshots[step]=(deepcopy(sharded.state_dict()),
                {name:p.grad.clone() for name,p in sharded.named_parameters()})
            if step==1:
                training.consolidate_optimizer_state(split,configuration(True))
                assert dist.get_backend(split._checkpoint_group)=='gloo'
                rngs=[None]*4;dist.all_gather_object(rngs,torch.get_rng_state())
                if rank==0:
                    torch.save(dict(model=sharded.state_dict(),optimizer=split.state_dict(),rngs=rngs),root/'checkpoint.pt')
                dist.barrier()
        expected_rng=torch.get_rng_state().clone()
        local=sum(parameter.numel() for parameter in split.optim.state)
        counts=[None]*4;dist.all_gather_object(counts,local)
        total=sum(p.numel() for p in sharded.parameters())
        assert sum(counts)==total and all(0<count<total for count in counts)
        training.consolidate_optimizer_state(split,configuration(True))
        if rank==0:
            actual,reference=split.state_dict(),dense.state_dict()
            assert actual['param_groups']==reference['param_groups']
            for index,state in reference['state'].items():
                for name,value in state.items():assert torch.equal(value,actual['state'][index][name])
        saved=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True)
        resumed=Network();resumed.load_state_dict(saved['model'])
        resumed=DistributedDataParallel(resumed,gradient_as_bucket_view=True)
        resumed_optimizer=training.training_optimizer(resumed.module,configuration(True))
        training.restore_optimizer_state(resumed_optimizer,saved['optimizer'],configuration(True),'cpu')
        assert resumed_optimizer.state is resumed_optimizer.optim.state
        torch.set_rng_state(saved['rngs'][rank])
        for step in range(2,4):
            update(resumed,resumed_optimizer,step)
            expected_weights,expected_gradients=snapshots[step]
            differences={name:float((value-resumed.module.state_dict()[name]).abs().max())
                for name,value in expected_weights.items() if not torch.equal(value,resumed.module.state_dict()[name])}
            gradient_differences={name:float((value-dict(resumed.module.named_parameters())[name].grad).abs().max())
                for name,value in expected_gradients.items() if not torch.equal(value,dict(resumed.module.named_parameters())[name].grad)}
            assert not differences,dict(step=step+1,weights=differences,gradients=gradient_differences)
        assert torch.equal(expected_rng,torch.get_rng_state())
        for name,value in sharded.state_dict().items():assert torch.equal(value,resumed.module.state_dict()[name])
        training.consolidate_optimizer_state(resumed_optimizer,configuration(True))
        if rank==0:
            expected,actual=split.state_dict(),resumed_optimizer.state_dict()
            assert expected['param_groups']==actual['param_groups']
            for index,state in expected['state'].items():
                for name,value in state.items():assert torch.equal(value,actual['state'][index][name])
        torch.save(dict(local_optimizer_parameters=local,all_rank_counts=counts,
            global_parameters=total,weights_and_global_adam_and_rng_exact=True),root/f'rank-{rank}.pt')
    finally:
        zero_module._broadcast_object=original_broadcast
        dist.destroy_process_group()


def test_actual_four_rank_adam_matches_dense_and_resumes_exactly(tmp_path):
    mp.spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    records=[torch.load(tmp_path/f'rank-{rank}.pt',weights_only=True) for rank in range(4)]
    assert all(row['weights_and_global_adam_and_rng_exact'] for row in records)
    assert sum(row['local_optimizer_parameters'] for row in records)==records[0]['global_parameters']
