# 完整参数和梯度的分片训练

四门课程已完成，真实全模型容量和精确续跑门禁已通过，正式讲解已开始。此前仅分片Adam时，完整模型连续4步的显存峰值约64.57GiB／卡；来源未知的占用增长后每卡约余36GiB，因此新增显式的FSDP2后端，继续使用同一完整模型和数据。

```json
{
  "distributed_strategy": "fsdp2",
  "fsdp_shard_paths": ["bridges.*", "base.model.layers.*", "base"],
  "optimizer_state_sharding": "none",
  "decoder_training": "full",
  "trainable_parameter_dtype": "float32",
  "decoder_gradient_checkpointing": true,
  "autocast_cache_enabled": false,
  "optimizer_foreach": false
}
```

默认仍为DDP；启用FSDP2须使用初始化后的多进程组，模块路径必须覆盖所有参数。星号只匹配一级子模块，子模块先分片，再分片父模块。模型布局由配置指定，调度代码不包含解码器类型分支。该后端使用PyTorch的[原位参数分片](https://docs.pytorch.org/docs/2.10/distributed.fsdp.fully_shard.html)，在优化器创建之前分片参数；FP32主参数、梯度和Adam状态按rank存放，前向／反向时收集当前模块，随后重新分片。BF16计算和原有激活重计算保持。

每个微批次均归约梯度，保持累积梯度处于分片状态。损失仍按全局监督词元加权，全局16、微批1、四轮2456次更新的正式数据预算不变。后端和模块路径纳入精确续跑合同，不能把DDP优化器续接到FSDP2；必须使用新实验目录。

验证、保存、选中模型重载及生成均由全部rank进入。每张卡运行相同的验证样本，只有rank 0写出结果；生成显式同步各卡。检查点通过[分布式状态字典接口](https://docs.pytorch.org/docs/2.10/distributed.checkpoint.html)收集到CPU，仍保存普通完整参数和整数索引的完整Adam字典，供现有推理、导出和核验工具读取。恢复时将CPU状态重新分片，避免完整CUDA动量副本；Adam超参数保留原检查点的类型和值。

诊断用的解码器变化量仅取rank 0的本地分片，结果中显式标注；完整参数是否更新与精确恢复，以完整CPU检查点逐值核验为准。

两进程CPU的实际分片测试检查完整权重、Adam、参数分组、随机状态和连续4步对2+2恢复。另有四卡小型Qwen与真实连接桥对照。上述软件与小模型结果不能替代完整4694274592参数模型的容量和精确恢复门禁，也不证明讲解质量或实战棋力。正式讲解、足量真实蒸馏、独立棋力与教学验收、模型交付仍待完成。

[完整模型开训读回](FULL_DECODER_SFT_START.md)记录624张量／4694274592参数的真实4对2+2逐值比较、12份工作进程资源记录及正式训练前三次更新。容量预检峰值24.81GiB指Torch张量分配，不能替代驱动总占用或正式全程峰值。
