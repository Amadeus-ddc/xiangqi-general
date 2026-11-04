# 完整BF16教师的多卡推理

完整Qwen3.8-27B权重保持未量化BF16。vLLM部署可以显式设置张量并行数与每卡显存预算；默认仍为单卡、0.88。

```json
{
  "tensor_parallel_size": 4,
  "gpu_memory_utilization": 0.30,
  "max_context_tokens": 16384,
  "inference_dtype": "bfloat16",
  "quantization": "none"
}
```

设置放在教师配置的`search_consolidator`内。并行数必须为正整数，显存比例必须为大于0、至多1的有限数。部署时通过`CUDA_VISIBLE_DEVICES`选择相应数量的设备；vLLM按配置分片，不能将四个完整副本当作张量并行。

`gpu_memory_utilization`是每张GPU总显存的比例，覆盖模型权重、激活与KV缓存，见[vLLM 0.19.1接口说明](https://docs.vllm.ai/en/v0.19.1/api/vllm/entrypoints/llm/)。共享设备时应根据实际空闲显存制定新配置，并测量真实推理；示例不保证当前容量足够。

部署后通过命名RPC检查每个worker的参数精度、量化方法、完整视觉模块与模型量化配置。worker数量必须等于请求的并行数，任一worker不满足未量化合同即关闭引擎并拒收。多卡身份记录全部worker，参数元素合计包含复制的张量，不能用该和宣称模型的唯一参数总数。

配对开发评测可设置`review_teacher_backend: "vllm"`，使用相同完整教师进行正文盲评；默认继续使用Transformers。实际启动后端写入检查点之外的独立推理清单，并在结果读回时核对。学生评测与教师推理按顺序执行。搜索汇总原有的vLLM后端继续复用同一教师配置。

CPU检查使用受控引擎，验证设置传递、多worker覆盖、任一分片量化时拒收、异常关闭及后端清单绑定。完整教师的实际加载、最大上下文／批量推理与显存峰值必须另行记录；软件检查不证明教师容量或讲解质量。每次改变部署配置使用新的输出目录，保留已有结果。
