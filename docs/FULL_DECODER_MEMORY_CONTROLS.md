# 全参数讲解训练的显存控制

四门课程和原生CPU导出已完成；正式讲解训练仍须通过真实四卡容量及精确续跑预检。当前完整解码器有4694274592个可训练参数，主参数保持FP32，计算使用BF16，全局批量16、微批量1，最大长度1024。

## 实际失败发生在哪里

未启用重计算和只启用重计算的两次原生预检，都已记录第1次更新，损失3.33524227142334，之后在下一次前向计算时显存不足。重计算试验的真实调用栈含`_gradient_checkpointing_func`和PyTorch checkpoint入口，说明开关已经生效。各卡当时约有21GiB来源未知的占用；失败时本任务占约73.6～73.9GiB。

不能因为`latest.pt`尚未保存，就推断没有更新：该文件在检查边界保存，逐步日志写在`optimizer.step()`与跨卡损失归约之后。此前零更新的判断不准确，已按原始逐步日志更正。两次失败的目录、配置、日志和初始产物均保留；这不是正式讲解完成，也不能据此评判讲解质量。

## 两个独立的可选设置

```json
{
  "decoder_training": "full",
  "trainable_parameter_dtype": "float32",
  "decoder_gradient_checkpointing": true,
  "autocast_cache_enabled": false
}
```

`decoder_gradient_checkpointing`默认关闭，仅允许显式的完整解码器训练。启用时使用非重入重计算并保存随机状态，不改变参数名称、数量或存储精度。训练已有的棋盘上下文覆盖前向和反向计算，重计算也在该上下文内执行；推理仍可使用原有KV缓存。

`autocast_cache_enabled`默认开启；关闭时不保留AMP上下文内的权重类型转换缓存，对应[PyTorch 2.10的`cache_enabled`](https://docs.pytorch.org/docs/2.10/amp.html)。训练、验证及训练后示例生成使用同一设置。它不表示关闭BF16计算，也不是量化。两项设置纳入精确续跑合同，改变设置必须新建实验输出，不能续接旧优化器合同。

CPU检查使用真实的小型Qwen解码器和连接桥，验证参数身份、实际重计算、损失／梯度／随机状态一致、Adam与随机状态连续4步对2+2恢复，以及验证和生成的缓存设置传递。未安装可选Transformers训练依赖时，开发测试仍可收集并运行独立检查；服务器完整训练环境执行全部模型检查。这些CPU结果不证明真实GPU显存足够。

## 当前服务器入口

当前主队列在`runs/pinned-base-postcourse-v4`，模型阶段在`runs/paper-sft-full-decoder-no-weight-cache-v1/pipeline`。相对只启用重计算的上一轮，唯一新增内存设置是`autocast_cache_enabled=false`。源代码、数据、输入凭据和配置分别冻结，先要求相同108模块消费者的完整CPU检查、构建与发布检查，再执行原生四卡连续4步对2+2恢复，全部通过后才进入2456次正式讲解更新。

可查看`queue-state.json`、`queue-plan.json`、`queue-handoff-verification.json`和`monitoring/hourly.jsonl`；原主队列终止后才能在新控制目录中恢复，并检查完整进程身份，避免重复启动。训练观测按小时取样，阶段变化与拥有者存活单独检查。未知GPU任务保持，不能用终止其他任务来代替本项目的容量验证。

完整讲解、足量实际搜索蒸馏、独立棋力与教学验收和可搬动模型交付尚未完成。

[软件核验记录](../evidence/full-decoder-memory-controls-v1.json)绑定111模块1866项和实际108模块1767项完整CPU检查及构建／发布检查；真实GPU容量及正式讲解完成仍单独验收。
